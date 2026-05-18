"""RaaS-level TCP weight receiver.

Runs inside the RaaS manager process (not as a subprocess inside SGLang).
Receives weights from the trainer's sender_agent via TCP, then writes them
as safetensors to /dev/shm so SGLang can load via its native
``/update_weights_from_disk`` API.

Architecture:
  Trainer sender_agent --TCP--> RaaSWeightReceiver (in RaaS manager)
                                    |-> safetensors.save_file() to /dev/shm
                                    |-> SGLang /update_weights_from_disk
"""

import logging
import os
import queue
import socket
import struct
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests as http_requests
import torch
import zmq

from astraflow.weight_manager.transfer.config import (
    TransferEngineConfig,
    TransferStatus,
)
from astraflow.weight_manager.transfer.receiver_agent import TransferBuffer
from astraflow.weight_manager.transfer.transfer_engine import TCPTransferEngine


logger = logging.getLogger(__name__)

# Default directory for safetensors files in shared memory
SHM_WEIGHTS_DIR = "/dev/shm/astraflow_weights"


class RaaSWeightReceiver:
    """TCP weight receiver that runs inside the RaaS manager process.

    Reuses the same TCP transfer engine and ZMQ protocol as the old
    receiver_agent, but lives in RaaS instead of inside SGLang.
    After receiving weights, it saves them as safetensors to /dev/shm
    for SGLang to load natively.

    In multi-model mode, create one receiver per model with a distinct
    ``shm_dir`` (e.g. ``/dev/shm/astraflow_weights/model0``) so that
    concurrent pulls don't overwrite each other's safetensors files.
    """

    DEFAULT_SHM_DIR = SHM_WEIGHTS_DIR

    def __init__(self, shm_dir: str = SHM_WEIGHTS_DIR):
        self.shm_dir = shm_dir
        self.buffer: Optional[TransferBuffer] = None
        self.transfer_engine: Optional[TCPTransferEngine] = None
        self.zmq_port: Optional[int] = None
        self.zmq_thread: Optional[threading.Thread] = None
        self._transfer_status_queue: queue.Queue = queue.Queue()
        self.sender_info: Dict[int, Dict] = {}
        self._started = False
        self._tensors_meta: Optional[List[Tuple[str, Tuple[List[int], str]]]] = None
        self._lora_config: Optional[Dict] = None

        # [Approach D] Persistent in-RAM copy of the weights data section.
        # Populated on each full pull, mutated in-place by delta apply.
        # Scatter happens here (private RAM, no shared mmap with SGLang or
        # anyone else), then we write the buffer to a fresh safetensors
        # file and atomic-rename — race-free by construction.
        self.weights_buffer: Optional[bytearray] = None
        # Cached header bytes (8-byte length + JSON header). Header doesn't
        # change between deltas (same tensor metadata), so we cache it after
        # the first full save.
        self._safetensors_header: Optional[bytes] = None

        os.makedirs(self.shm_dir, exist_ok=True)

    def start(
        self,
        sender_http_endpoint: str,
        handshake_port: int = 0,
        identity_port: int = 0,
    ) -> None:
        """Initialize receiver: query sender, allocate buffer, register.

        Parameters
        ----------
        sender_http_endpoint:
            ``"host:port"`` of the sender agent's HTTP server.
        handshake_port:
            Port for TCP handshake (0 = auto-assign).
        identity_port:
            Port used to build the unique ``instance_id`` that the sender
            uses to key this receiver (e.g. the RaaS HTTP service port).
            Must be unique per RaaS instance on the same host.
        """
        self._identity_port = identity_port
        if self._started:
            logger.info("[RaaSReceiver] Already started, skipping.")
            return

        host, port_str = sender_http_endpoint.rsplit(":", 1)
        port = int(port_str)
        sender_url = f"http://{host}:{port}"

        _t = lambda: time.monotonic()
        _t0 = _t()
        logger.info(
            "[RaaSReceiver][DEBUG] start: sender_endpoint=%s "
            "handshake_port=%d identity_port=%d",
            sender_http_endpoint, handshake_port, identity_port,
        )

        # 1. Query sender for buffer info (full model metadata)
        logger.info("[RaaSReceiver][DEBUG] step1: query sender buffer info ...")
        _step = _t()
        sender_buffer_info = self._query_sender_buffer_info(sender_url)
        self._tensors_meta = sender_buffer_info["tensors_meta"]
        self._lora_config = sender_buffer_info.get("lora_config")
        logger.info(
            "[RaaSReceiver][DEBUG] step1 done in %.2fs "
            "(num_tensors=%d, buffer_length=%d, lora=%s)",
            _t() - _step, len(self._tensors_meta),
            sender_buffer_info.get("single_buffer_length", -1),
            self._lora_config is not None,
        )

        # 2. Build meta tensors and allocate buffer
        logger.info("[RaaSReceiver][DEBUG] step2: build meta tensors + allocate buffer ...")
        _step = _t()
        meta_tensors = []
        for name, (shape, dtype_str) in self._tensors_meta:
            dt = getattr(torch, dtype_str)
            meta = torch.empty(shape, dtype=dt, device="meta")
            meta_tensors.append((name, meta))

        self.buffer = TransferBuffer(meta_tensors)
        logger.info(
            "[RaaSReceiver][DEBUG] step2 done in %.2fs (buffer.length=%d)",
            _t() - _step, self.buffer.length,
        )

        # 3. Initialize TCP engine and start listener
        logger.info("[RaaSReceiver][DEBUG] step3: init TCP engine + start listener ...")
        _step = _t()
        local_ip = self._get_local_ip()
        engine_config = TransferEngineConfig(
            local_hostname=local_ip,
            handshake_port=handshake_port,
        )
        self.transfer_engine = TCPTransferEngine(config=engine_config, num_threads=1)
        self.transfer_engine.is_receiver = True
        self.transfer_engine.register(self.buffer.ptr, self.buffer.length)
        self.transfer_engine.start_listener()
        logger.info(
            "[RaaSReceiver][DEBUG] step3 done in %.2fs "
            "(local_ip=%s session_id=%s rpc_port=%d)",
            _t() - _step, local_ip,
            self.transfer_engine.get_session_id(),
            self.transfer_engine.get_rpc_port(),
        )

        # 4. Start ZMQ listener
        logger.info("[RaaSReceiver][DEBUG] step4: start ZMQ listener ...")
        _step = _t()
        self._start_zmq_listener()
        logger.info(
            "[RaaSReceiver][DEBUG] step4 done in %.2fs (zmq_port=%s)",
            _t() - _step, self.zmq_port,
        )

        # 5. Register with sender
        logger.info("[RaaSReceiver][DEBUG] step5: register with sender ...")
        _step = _t()
        self._register_with_sender(sender_url)
        logger.info("[RaaSReceiver][DEBUG] step5 done in %.2fs", _t() - _step)

        self._started = True
        logger.info(
            "[RaaSReceiver] Started: buffer=%d bytes, tcp_session=%s, "
            "zmq_port=%d (total=%.2fs)",
            self.buffer.length,
            self.transfer_engine.get_session_id(),
            self.zmq_port,
            _t() - _t0,
        )

    def wait_for_transfer(self, timeout: float = 600) -> None:
        """Wait for TCP transfer completion via ZMQ signal."""
        remaining_senders = set(self.sender_info.keys())
        deadline = time.monotonic() + timeout

        while remaining_senders:
            wait_secs = max(0, deadline - time.monotonic())
            if wait_secs <= 0:
                logger.warning(
                    "[RaaSReceiver] Transfer timeout after %.0fs, proceeding.",
                    timeout,
                )
                return

            try:
                sender_rank, status = self._transfer_status_queue.get(
                    timeout=min(wait_secs, 10)
                )
            except queue.Empty:
                # ZMQ not yet received — keep waiting until deadline.
                continue

            if status == TransferStatus.SUCCESS:
                logger.info(
                    "[RaaSReceiver] ZMQ SUCCESS from sender_rank=%d",
                    sender_rank,
                )
                remaining_senders.discard(sender_rank)
            elif status == TransferStatus.FAILURE:
                raise RuntimeError(
                    f"[RaaSReceiver] ZMQ FAILURE from sender_rank={sender_rank}"
                )

    def apply_delta_and_save(self, delta_data: bytes,
                             target_version: int = -1) -> str:
        """Apply sparse delta in-process to ``self.weights_buffer``, then
        write to ``model.safetensors.new`` and atomic-rename.

        [Approach D] The scatter happens on the persistent in-RAM
        ``self.weights_buffer`` (a private bytearray), not on any mmap'd
        file. This is race-free by construction:

        - The buffer is private to this process (no MAP_SHARED hazard with
          SGLang or anyone else).
        - The output file is brand-new (no other process has it open).
        - Rename is atomic at the inode level.

        Numpy fancy-index assignment releases the GIL for the heavy memory
        copies, so the event loop stays responsive while this method runs
        in a worker thread (called via ``loop.run_in_executor``).

        Parameters
        ----------
        delta_data : bytes
            Sparse delta in the format ``[header 16 bytes][indices][values]``.
        target_version : int
            The version the buffer will hold after this apply (for logging).

        Returns the path to the safetensors directory.
        """
        if self.weights_buffer is None or self._safetensors_header is None:
            raise RuntimeError(
                "[RaaSReceiver] apply_delta_and_save called before any full "
                "pull populated weights_buffer; cannot apply delta"
            )

        t0 = time.monotonic()
        safetensors_path = os.path.join(self.shm_dir, "model.safetensors")

        # Parse delta header + indices + values
        num_nonzero, element_size, flags, _ = struct.unpack_from(
            "<QHHi", delta_data, 0,
        )
        use_uint64 = bool(flags & 1)
        idx_dtype = np.uint64 if use_uint64 else np.uint32
        idx_size = 8 if use_uint64 else 4
        header_size = 16
        indices_start = header_size
        indices_end = indices_start + num_nonzero * idx_size
        values_start = indices_end
        values_end = values_start + num_nonzero * element_size

        indices = np.frombuffer(
            delta_data[indices_start:indices_end], dtype=idx_dtype,
        )
        values = np.frombuffer(
            delta_data[values_start:values_end], dtype=np.uint8,
        ).reshape(num_nonzero, element_size)
        t_parse = time.monotonic()

        # In-place scatter on the persistent buffer.
        # Sort indices for sequential write order (better cache behavior).
        weight_view = np.frombuffer(self.weights_buffer, dtype=np.uint8)
        weight_2d = weight_view.reshape(-1, element_size)
        sort_order = np.argsort(indices)
        weight_2d[indices[sort_order]] = values[sort_order]
        t_scatter = time.monotonic()

        # Write [header || buffer] to a fresh file, then atomic-rename.
        new_path = safetensors_path + ".new"
        try:
            with open(new_path, "wb") as f:
                f.write(self._safetensors_header)
                f.write(self.weights_buffer)
            t_write = time.monotonic()
            os.rename(new_path, safetensors_path)
            t_rename = time.monotonic()
        except Exception:
            try:
                os.unlink(new_path)
            except OSError:
                pass
            raise

        logger.info(
            "[RaaSReceiver] Delta applied in-process v=%d: "
            "parse=%.2fs scatter=%.2fs write=%.2fs rename=%.2fs total=%.2fs",
            target_version,
            t_parse - t0, t_scatter - t_parse,
            t_write - t_scatter, t_rename - t_write,
            t_rename - t0,
        )
        return self.shm_dir

    def save_as_safetensors(self) -> dict:
        """Write the receive buffer as a safetensors file.

        Builds the safetensors header from metadata and writes it plus the
        raw buffer bytes directly — avoids per-tensor reconstruction and the
        ``safetensors.save_file()`` copy.

        For LoRA weights (when ``_lora_config`` is set), saves as
        ``adapter_model.safetensors`` + ``adapter_config.json`` in PEFT
        adapter format so SGLang/vLLM can load via ``/load_lora_adapter``.
        Also fuses separate ``gate_proj``/``up_proj`` LoRA weights into
        ``gate_up_proj`` to match SGLang's fused MLP parameter layout.

        Returns a dict with ``shm_path`` and ``use_lora`` keys.
        """
        import json
        import struct

        from safetensors.torch import save_file as st_save_file

        assert self.buffer is not None, "Buffer not allocated"
        assert self._tensors_meta is not None, "No tensors metadata"

        is_lora = self._lora_config is not None

        if is_lora:
            tensors = self._reconstruct_tensors()
            logger.info("[DEBUG-RECEIVER] LoRA tensors keys (%d): %s", len(tensors), list(tensors.keys())[:10])
            # Note: SGLang's normalize_gate_up_proj() handles
            # gate_proj/up_proj → gate_up_proj fusion internally,
            # so we don't need to fuse tensors here.
            filename = "adapter_model.safetensors"
            safetensors_path = os.path.join(self.shm_dir, filename)
            st_save_file(tensors, safetensors_path)

            # Write adapter_config.json for SGLang/vLLM PEFT loading.
            # Remap target_modules: replace separate gate_proj/up_proj with
            # fused gate_up_proj to match SGLang's parameter names.
            adapter_cfg = dict(self._lora_config)
            if "target_modules" in adapter_cfg:
                modules = adapter_cfg["target_modules"]
                has_gate = "gate_proj" in modules
                has_up = "up_proj" in modules
                if has_gate or has_up:
                    modules = [
                        m for m in modules if m not in ("gate_proj", "up_proj")
                    ]
                    if "gate_up_proj" not in modules:
                        modules.append("gate_up_proj")
                    adapter_cfg["target_modules"] = modules
            adapter_config_path = os.path.join(
                self.shm_dir, "adapter_config.json"
            )
            with open(adapter_config_path, "w") as f:
                json.dump(adapter_cfg, f)
            logger.info(
                "[RaaSReceiver] Wrote LoRA adapter (%d tensors, %.1f MB) "
                "to %s (adapter_config at %s)",
                len(tensors),
                sum(t.nbytes for t in tensors.values()) / (1024 * 1024),
                safetensors_path,
                adapter_config_path,
            )
        else:
            self._save_raw_safetensors()

        return {"shm_path": self.shm_dir, "use_lora": is_lora}

    def _reconstruct_tensors(self) -> Dict[str, torch.Tensor]:
        """Reconstruct individual tensors from the flat receive buffer."""
        tensors = {}
        offset = 0
        raw_np = self.buffer.buffer.numpy()
        for name, (shape, dtype_str) in self._tensors_meta:
            dtype = getattr(torch, dtype_str)
            numel = 1
            for d in shape:
                numel *= d
            elem_size = dtype.itemsize if hasattr(dtype, "itemsize") else (
                torch.finfo(dtype).bits // 8
            )
            size_bytes = numel * elem_size
            tensor_bytes = raw_np[offset : offset + size_bytes]
            t = torch.frombuffer(
                bytearray(tensor_bytes), dtype=dtype
            ).reshape(shape)
            tensors[name] = t
            offset += size_bytes
        return tensors

    def _save_raw_safetensors(self) -> None:
        """Save full-model weights as raw safetensors (fast zero-copy path)."""
        import json
        import struct

        _DTYPE_MAP = {
            "float16": "F16",
            "bfloat16": "BF16",
            "float32": "F32",
            "float64": "F64",
            "int8": "I8",
            "int16": "I16",
            "int32": "I32",
            "int64": "I64",
            "uint8": "U8",
            "bool": "BOOL",
        }

        header = {}
        offset = 0
        for name, (shape, dtype_str) in self._tensors_meta:
            dtype = getattr(torch, dtype_str)
            numel = 1
            for d in shape:
                numel *= d
            elem_size = dtype.itemsize if hasattr(dtype, "itemsize") else (
                torch.finfo(dtype).bits // 8
            )
            size_bytes = numel * elem_size
            sf_dtype = _DTYPE_MAP.get(dtype_str)
            if sf_dtype is None:
                raise ValueError(
                    f"Unsupported dtype for safetensors: {dtype_str}"
                )
            header[name] = {
                "dtype": sf_dtype,
                "shape": shape,
                "data_offsets": [offset, offset + size_bytes],
            }
            offset += size_bytes

        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        padding = (8 - len(header_bytes) % 8) % 8
        header_bytes += b" " * padding

        safetensors_path = os.path.join(self.shm_dir, "model.safetensors")
        raw = self.buffer.buffer.numpy().data
        with open(safetensors_path, "wb") as f:
            full_header = struct.pack("<Q", len(header_bytes)) + header_bytes
            f.write(full_header)
            f.write(raw)

        # [Approach D] Cache header + populate persistent weights_buffer.
        # On full pull, copy the just-received bytes into the persistent
        # buffer; subsequent delta applies mutate this buffer in-place.
        self._safetensors_header = full_header
        raw_bytes_len = len(raw)
        if (self.weights_buffer is None
                or len(self.weights_buffer) != raw_bytes_len):
            self.weights_buffer = bytearray(raw_bytes_len)
        # Copy raw bytes into the persistent buffer
        memoryview(self.weights_buffer)[:] = raw

        logger.info(
            "[RaaSReceiver] Saved %d tensors to %s (%.1f MB) "
            "+ populated persistent buffer",
            len(header),
            safetensors_path,
            offset / (1024 * 1024),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query_sender_buffer_info(
        self, sender_url: str, timeout: float = 600
    ) -> Dict:
        """Query sender for full-model buffer info (handles TP sharding)."""
        logger.info(
            "[RaaSReceiver][DEBUG] _query_sender_buffer_info: starting, "
            "sender_url=%s timeout=%.0fs",
            sender_url, timeout,
        )
        deadline = time.monotonic() + timeout
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            try:
                resp = http_requests.get(
                    f"{sender_url}/get_buffer_info", timeout=5
                )
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except Exception as je:
                        logger.warning(
                            "[RaaSReceiver][DEBUG] attempt=%d 200 but JSON "
                            "decode failed: %r (body=%r)",
                            attempt, je, resp.text[:300],
                        )
                        data = None
                    if data is not None and "tensors_meta" in data:
                        logger.info(
                            "[RaaSReceiver] Got sender buffer info: "
                            "buffer_length=%d, num_tensors=%d (attempt=%d)",
                            data["single_buffer_length"],
                            len(data["tensors_meta"]),
                            attempt,
                        )
                        return data
                    logger.warning(
                        "[RaaSReceiver][DEBUG] attempt=%d 200 but missing "
                        "'tensors_meta' (keys=%r)",
                        attempt,
                        sorted(data.keys()) if isinstance(data, dict) else None,
                    )
                else:
                    logger.warning(
                        "[RaaSReceiver][DEBUG] attempt=%d non-200: "
                        "status=%d body=%r",
                        attempt, resp.status_code, resp.text[:300],
                    )
            except Exception as exc:
                logger.warning(
                    "[RaaSReceiver][DEBUG] attempt=%d request failed: "
                    "%s: %s",
                    attempt, type(exc).__name__, exc,
                )
            time.sleep(5)
        raise RuntimeError(
            f"[RaaSReceiver] Failed to query sender buffer info "
            f"after {timeout}s"
        )

    def _start_zmq_listener(self) -> None:
        """Start ZMQ PULL socket in a background thread."""
        self.zmq_thread = threading.Thread(
            target=self._zmq_listener_thread, daemon=True
        )
        self.zmq_thread.start()
        time.sleep(0.5)
        deadline = time.monotonic() + 10
        while self.zmq_port is None and self.zmq_thread.is_alive():
            if time.monotonic() > deadline:
                raise RuntimeError("[RaaSReceiver] ZMQ listener failed to start")
            time.sleep(0.1)

    def _zmq_listener_thread(self) -> None:
        context = zmq.Context()
        sock = context.socket(zmq.PULL)
        port = self._find_free_port()
        bind_addr = f"tcp://0.0.0.0:{port}"

        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                sock.bind(bind_addr)
                self.zmq_port = port
                logger.info("[RaaSReceiver] ZMQ bound to %s", bind_addr)
                break
            except zmq.error.ZMQError:
                if attempt < max_attempts - 1:
                    port = self._find_free_port()
                    bind_addr = f"tcp://0.0.0.0:{port}"
                else:
                    raise

        while True:
            try:
                sender_rank_bytes, status_bytes = sock.recv_multipart()
                sender_rank = int(sender_rank_bytes.decode("ascii"))
                status = TransferStatus(int(status_bytes.decode("ascii")))
                self._transfer_status_queue.put((sender_rank, status))
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM:
                    break
                logger.error("[RaaSReceiver] ZMQ error: %s", e)
                break

    def _register_with_sender(
        self, sender_url: str, max_wait_secs: float = 600
    ) -> None:
        """Register this receiver with the sender agent."""
        local_ip = self._get_local_ip()
        session_ids = [self.transfer_engine.get_session_id()]
        handshake_ports = [self.transfer_engine.get_rpc_port()]

        # Use RaaS identity for registration (not an SGLang instance)
        payload = {
            "sglang_http_host": local_ip,
            "sglang_http_port": self._identity_port,  # unique per RaaS on same host
            "session_ids": session_ids,
            "buffer_ptr": self.buffer.ptr,
            "buffer_length": self.buffer.length,
            "zmq_endpoint": local_ip,
            "zmq_port": self.zmq_port,
            "handshake_ports": handshake_ports,
            "sender_group_index": 0,
        }

        elapsed = 0
        retry_interval = 5
        while elapsed < max_wait_secs:
            try:
                resp = http_requests.post(
                    f"{sender_url}/register_sglang_instance",
                    json=payload,
                    timeout=30,
                )
                resp.raise_for_status()
                response = resp.json()
                if isinstance(response, dict) and response.get("trainer_session_ids"):
                    sender_rank = response["trainer_global_rank"]
                    self.sender_info[sender_rank] = response
                    logger.info(
                        "[RaaSReceiver] Registered with sender: %s", response
                    )
                    return
            except Exception as e:
                logger.info(
                    "[RaaSReceiver] Sender not reachable: %s, retrying in %ds",
                    e, retry_interval,
                )
            time.sleep(retry_interval)
            elapsed += retry_interval

        raise RuntimeError(
            f"[RaaSReceiver] Failed to register with sender after {max_wait_secs}s"
        )

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    @staticmethod
    def _get_local_ip() -> str:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 1))
            ip = s.getsockname()[0]
        except Exception:
            ip = "127.0.0.1"
        finally:
            s.close()
        return ip
