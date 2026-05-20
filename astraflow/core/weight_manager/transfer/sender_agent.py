"""Sender-side TCP weight transfer agent (runs as a trainer subprocess).

Architecture (RaaS-initiated pull)
-----------------------------------
- **Trainer** copies weights into the shared-memory CPU buffer via
  ``WeightManager.offload()``, then signals this subprocess
  with ``"buffer_ready:<version>"``.  The event loop records the new version
  and immediately acknowledges back — the trainer never blocks.

- **RaaS** periodically checks its local version against the global version
  (broadcast by the trainer).  When behind by ≥2 versions, RaaS calls the
  sender's HTTP endpoint ``POST /request_transfer`` to pull weights.

- A ``_buffer_lock`` serialises buffer writes (trainer) and reads (transfer
  to RaaS) so they never overlap.
"""

import ctypes
import ctypes.util
import json
import logging
import mmap
import os
import queue
import struct
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any, Dict, List, Optional, Tuple

import torch.multiprocessing as mp
import zmq

from .config import ReceiverInfo, SenderAgentConfig, TransferStatus
from .transfer_engine import TCPTransferEngine

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_MAP_FLAGS = mmap.MAP_SHARED
try:
    _MAP_FLAGS |= mmap.MAP_POPULATE
except AttributeError:
    pass



def _configure_mmap(
    buffer_ptr: int, buffer_length: int, mmap_buffer: mmap.mmap
) -> None:
    """Apply best-effort performance hints to a shared-memory buffer."""
    try:
        mmap_buffer.madvise(mmap.MADV_DONTDUMP)
        mmap_buffer.madvise(mmap.MADV_DONTFORK)
    except Exception:
        pass

    libc = ctypes.CDLL(ctypes.util.find_library("c"))

    try:
        if (
            libc.mlock(
                ctypes.c_void_p(buffer_ptr), ctypes.c_size_t(buffer_length)
            )
            == 0
        ):
            logger.info("Locked %.1f MB in RAM", buffer_length / (1024 * 1024))
    except Exception as e:
        logger.debug("Could not mlock: %s", e)

    try:
        MADV_HUGEPAGE = 14
        if (
            libc.madvise(
                ctypes.c_void_p(buffer_ptr),
                ctypes.c_size_t(buffer_length),
                ctypes.c_int(MADV_HUGEPAGE),
            )
            == 0
        ):
            logger.info("Enabled transparent huge pages")
    except Exception as e:
        logger.debug("Could not enable transparent huge pages: %s", e)

    try:
        mmap_buffer.madvise(mmap.MADV_SEQUENTIAL)
        mmap_buffer.madvise(mmap.MADV_WILLNEED)
    except Exception:
        pass


def create_tensor_from_shared_memory(shm_path: str, buffer_length: int):
    """Create a torch tensor backed by the given shared memory file."""
    import torch
    import torch.cuda

    shm_fd = os.open(shm_path, os.O_RDWR)
    mmap_buffer = mmap.mmap(
        shm_fd,
        buffer_length,
        flags=_MAP_FLAGS,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    os.close(shm_fd)

    buffer_ptr = ctypes.addressof(ctypes.c_byte.from_buffer(mmap_buffer))
    _configure_mmap(buffer_ptr, buffer_length, mmap_buffer)

    torch.cuda.cudart().cudaHostRegister(buffer_ptr, buffer_length, 0)

    tensor = torch.frombuffer(mmap_buffer, dtype=torch.uint8)
    tensor._mmap_buffer = mmap_buffer
    return tensor


def _make_http_handler(transfer_agent: "TransferAgent"):
    """Create an HTTP request handler class bound to the given transfer agent.

    Provides HTTP endpoints:
    - ``GET  /get_version``: return current weight version.
    - ``GET  /get_buffer_info``: return buffer metadata.
    - ``GET  /get_capabilities``: return supported transfer strategies.
    - ``POST /register_sglang_instance``: receiver registers itself.
    - ``POST /request_transfer``: RaaS requests a weight transfer (pull model).
    """

    class TransferHTTPHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            logger.debug(format, *args)

        def _send_json(self, data: dict, status: int = 200):
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            if length == 0:
                return {}
            return json.loads(self.rfile.read(length))

        def do_GET(self):
            if self.path == "/get_version":
                self._send_json({"version": transfer_agent.weight_version})
            elif self.path == "/get_buffer_info":
                _sbl = transfer_agent._single_buffer_length
                _hfsl = getattr(transfer_agent, "_hf_serving_length", None)
                _hfmeta = getattr(transfer_agent, "_hf_tensors_meta", None)
                _tmeta = transfer_agent.tensors_meta
                logger.info(
                    "[Weight sender][DEBUG] /get_buffer_info called: "
                    "_single_buffer_length=%s _hf_serving_length=%s "
                    "tensors_meta=%s _hf_tensors_meta=%s",
                    _sbl, _hfsl,
                    f"list(len={len(_tmeta)})" if _tmeta is not None else None,
                    f"list(len={len(_hfmeta)})" if _hfmeta is not None else None,
                )
                if _sbl is not None:
                    # For Megatron: report HF serving buffer size/meta once available.
                    buf_len = _sbl
                    meta = _tmeta
                    if _hfsl is not None and _hfsl > 0:
                        buf_len = _hfsl
                    if _hfmeta is not None:
                        meta = _hfmeta
                    resp = {
                        "single_buffer_length": buf_len,
                        "tensors_meta": meta,
                    }
                    if transfer_agent.lora_config is not None:
                        resp["lora_config"] = transfer_agent.lora_config
                    if meta is None:
                        logger.warning(
                            "[Weight sender][DEBUG] /get_buffer_info: "
                            "buffer_length=%d but meta is None — "
                            "returning 503 instead of empty meta",
                            buf_len,
                        )
                        self._send_json(
                            {"error": "tensors_meta not populated yet"}, 503
                        )
                        return
                    self._send_json(resp)
                else:
                    self._send_json({"error": "buffer not allocated yet"}, 503)
            elif self.path == "/get_capabilities":
                self._send_json(transfer_agent.get_capabilities())
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path == "/register_sglang_instance":
                self._handle_register()
            elif self.path == "/request_transfer":
                self._handle_request_transfer()
            else:
                self._send_json({"error": "not found"}, 404)

        def _handle_register(self):
            try:
                payload = self._read_json()
                sglang_http_host = payload["sglang_http_host"]
                sglang_http_port = payload["sglang_http_port"]
                session_ids = payload["session_ids"]
                buffer_ptr = payload["buffer_ptr"]
                buffer_length = payload["buffer_length"]
                zmq_endpoint = payload["zmq_endpoint"]
                zmq_port = payload["zmq_port"]
                handshake_ports = payload["handshake_ports"]
                sender_group_index = payload.get("sender_group_index", 0)

                instance_id = f"{sglang_http_host}:{sglang_http_port}"
                if transfer_agent.use_tcp_engine:
                    assert len(session_ids) == 1
                elif len(session_ids) != transfer_agent.config.num_engines_per_group:
                    raise ValueError(
                        f"Expected {transfer_agent.config.num_engines_per_group} "
                        f"session IDs, got {len(session_ids)}"
                    )

                logger.info(
                    "[Weight sender] Registering sglang instance: %s, "
                    "remote_session_ids: %s",
                    instance_id,
                    session_ids,
                )

                transfer_agent.register_receiver_session(
                    instance_id,
                    session_ids,
                    buffer_ptr,
                    buffer_length,
                    zmq_endpoint,
                    zmq_port,
                    sglang_http_host,
                    sglang_http_port,
                    handshake_ports,
                    sender_group_index,
                )

                self._send_json({
                    "trainer_global_rank": transfer_agent.config.trainer_global_rank,
                    "trainer_world_size": transfer_agent.config.trainer_world_size,
                    "trainer_session_ids": transfer_agent.get_session_ids(),
                    "trainer_buffer_ptr": transfer_agent.buffer_ptr,
                    "trainer_buffer_length": transfer_agent.buffer_length,
                    "trainer_hostname": transfer_agent.get_hostname(),
                    "trainer_rpc_port": transfer_agent.get_rpc_port(),
                })
            except Exception as exc:
                logger.error(
                    "[Weight sender] register_sglang_instance failed: %s",
                    exc,
                    exc_info=True,
                )
                self._send_json({"error": str(exc)}, 500)

        def _handle_request_transfer(self):
            try:
                payload = self._read_json()
                instance_id = payload.get("instance_id", "")
                mode = payload.get("mode", "full")
                result = transfer_agent.handle_transfer_request(
                    instance_id, mode=mode,
                )
                self._send_json(result)
            except Exception as exc:
                logger.error(
                    "[Weight sender] request_transfer failed: %s",
                    exc,
                    exc_info=True,
                )
                self._send_json(
                    {"ok": False, "version": 0, "error": str(exc)}, 500
                )

    return TransferHTTPHandler


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTPServer that handles each request in a new thread.

    Essential when ``/request_transfer`` blocks for the duration of a TCP
    transfer — without threading, concurrent registration or version
    requests from other receivers would be starved.
    """

    daemon_threads = True


class TransferAgent:
    """Sender-side transfer agent with integrated version tracking.

    Replaces the old WeightController: version tracking, buffer locking,
    and transfer initiation are all handled here.  Transfers are initiated
    by RaaS (pull model) via HTTP, not by the data-acquisition layer.
    """

    def __init__(
        self,
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        config: SenderAgentConfig,
        strategies: Optional[List[str]] = None,
        delta_done_event=None,
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.config = config
        self.strategies = strategies or ["full"]
        self._delta_done_event = delta_done_event
        self.registered_receivers: Dict[str, ReceiverInfo] = {}
        self.buffer_length: Optional[int] = None
        self.buffer_ptr: Optional[int] = None
        self.transfer_engines: List[List] = []
        self.buffer_slices: List[List[Tuple[int, int, int]]] = []
        self.use_tcp_engine = (
            os.environ.get("TRANSFER_ENGINE_TYPE", "tcp").lower() == "tcp"
        )

        self.memfd: Optional[int] = None
        self.mmap_buffer: Optional[mmap.mmap] = None

        self.weight_version = 0
        self.tensors_meta = None
        self.lora_config: Optional[Dict] = None

        # Double-buffer state: the trainer always writes to the inactive half
        # while senders read from the active half.  Only Python int assignment
        # is needed for atomicity (GIL guarantees it).
        self._single_buffer_length: Optional[int] = None
        self._active_buf_idx: int = 0

        self._zmq_ctx: Optional[Any] = None
        self._zmq_push_sockets: Dict[str, Any] = {}

        # Serialises trainer buffer writes and RaaS-initiated transfers.
        self._buffer_lock = threading.Lock()

        # Delta transfer state — delta is computed by WeightManager on GPU
        # and written to a shared delta shm buffer.  The sender just serves it.
        self._delta_enabled = "delta" in self.strategies
        self._delta_shm_file = None
        self._delta_mmap: Optional[mmap.mmap] = None
        self._delta_size: int = 0  # actual used bytes in delta buffer
        self._delta_ready: bool = False
        self._delta_base_version: int = 0
        self._delta_version: int = 0
        self._delta_lock = threading.Lock()
        # Serializes writes (compute_delta) and reads (TCP send) of
        # _delta_mmap. compute_delta runs in the event_loop thread; TCP
        # send runs in the HTTP handler thread. Without this lock they
        # can race, yielding torn bytes that are inconsistent across
        # receivers' pulls.
        self._delta_mmap_lock = threading.Lock()
        # True once we have real weight data in the inactive half. On the
        # very first buffer_ready (including recovery push), the inactive
        # half is zero-filled, so computing a "delta" against it would
        # produce an ~100% dense result that is costly and useless.
        self._inactive_half_has_data: bool = False
        # Delta metrics (passed from WeightManager via buffer_ready message)
        self._delta_sparsity: float = 0.0
        self._delta_num_nonzero: int = 0
        self._delta_compute_time: float = 0.0

        self.initialize_transfer_engines()

    def initialize_transfer_engines(self):
        num_engines_per_group = self.config.num_engines_per_group
        for group_idx, engine_config in enumerate(self.config.engine_configs):
            engine = TCPTransferEngine(
                config=engine_config, num_threads=num_engines_per_group
            )
            self.transfer_engines.append([engine])
            logger.info(
                "Initialized TCP engine group %d with %d parallel streams",
                group_idx,
                engine.num_parallel_streams,
            )

    # ------------------------------------------------------------------
    # Megatron shard reassembly
    # ------------------------------------------------------------------

    def setup_megatron_reassembly(self, megatron_metadata: dict) -> None:
        """Configure CPU-side TP shard → HF param reassembly.

        Called at init time.  Stores metadata and prepares the HF
        conversion function that maps mcore params to HF params.
        """
        self._megatron_metadata = megatron_metadata
        self._megatron_tp_size = megatron_metadata["tp_size"]
        self._megatron_shard_specs = megatron_metadata["shard_specs"]
        self._megatron_conv_config = megatron_metadata["conversion_config"]

        # HF serving buffer is allocated after the main buffer in
        # allocate_transfer_buffer (once we know the buffer size).
        self._hf_serving_buffer: Optional[mmap.mmap] = None
        self._hf_serving_memfd: Optional[int] = None
        self._hf_serving_length: int = 0
        # HF tensors_meta is computed during first reassembly.
        self._hf_tensors_meta = None
        logger.info(
            "[Sender] Megatron reassembly configured: tp_size=%d, "
            "%d shard specs, model_type=%s",
            self._megatron_tp_size,
            len(self._megatron_shard_specs),
            self._megatron_conv_config.get("model_type", "?"),
        )

    def _allocate_hf_serving_buffer(self) -> None:
        """Allocate a separate shared-memory buffer for reassembled HF params.

        Called once after the main double-buffer is allocated.
        """
        if self._hf_serving_buffer is not None:
            return

        # HF buffer = same size as one half of the double buffer (full model).
        hf_length = self._single_buffer_length
        hf_file = tempfile.NamedTemporaryFile(
            dir="/dev/shm", delete=False, prefix="astraflow_hf_serving_"
        )
        os.ftruncate(hf_file.fileno(), hf_length)
        hf_mmap = mmap.mmap(
            hf_file.fileno(),
            hf_length,
            flags=_MAP_FLAGS,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )

        hf_ptr = ctypes.addressof(ctypes.c_byte.from_buffer(hf_mmap))
        _configure_mmap(hf_ptr, hf_length, hf_mmap)

        self._hf_serving_buffer = hf_mmap
        self._hf_serving_file = hf_file  # prevent GC from closing the fd
        self._hf_serving_memfd = hf_file.fileno()
        self._hf_serving_length = hf_length
        logger.info(
            "[Sender] HF serving buffer allocated: %s (%d bytes, %.1f MB)",
            hf_file.name, hf_length, hf_length / (1024 * 1024),
        )

    def _reassemble_megatron_to_hf(self, buf_idx: int) -> None:
        """Reassemble raw TP shards from the active buffer into HF format.

        Reads mcore-layout shards from buffer half ``buf_idx``, concatenates
        TP shards along partition_dim, applies GLU/fc2 fixes, converts
        mcore→HF format (QKV split, gate/up split, rename), and writes
        the result into the HF serving buffer.

        This runs on CPU in the sender subprocess — no GPU involved.
        """
        import numpy as np

        self._allocate_hf_serving_buffer()
        tp_size = self._megatron_tp_size
        conv_cfg = self._megatron_conv_config

        src_base = buf_idx * self._single_buffer_length
        src_view = memoryview(self.mmap_buffer)[src_base:src_base + self._single_buffer_length]
        dst_view = memoryview(self._hf_serving_buffer)

        _dtype_map = {
            "bfloat16": (np.uint16, 2),
            "float16": (np.float16, 2),
            "float32": (np.float32, 4),
        }

        head_dim = conv_cfg.get("kv_channels")
        if head_dim is None:
            head_dim = conv_cfg["hidden_size"] // conv_cfg["num_attention_heads"]
        num_heads = conv_cfg["num_attention_heads"]
        num_query_groups = conv_cfg["num_query_groups"]
        hidden_size = conv_cfg["hidden_size"]
        vocab_size = conv_cfg["vocab_size"]
        value_num_per_group = num_heads // num_query_groups

        src_offset = 0
        dst_offset = 0
        hf_tensors_meta = []

        for spec in self._megatron_shard_specs:
            np_dtype, elem_size = _dtype_map.get(
                spec["dtype"], (np.uint16, 2)
            )
            shard_shape = spec["shard_shape"]
            full_shape = spec["full_shape"]
            shard_numel = int(np.prod(shard_shape))
            full_numel = int(np.prod(full_shape))
            full_size = full_numel * elem_size
            shard_size = shard_numel * elem_size
            partition_dim = spec["partition_dim"]

            # Read shards from source buffer and assemble full param
            if spec["is_sharded"] and tp_size > 1:
                shards = []
                for tp_r in range(tp_size):
                    s_start = src_offset + tp_r * shard_size
                    shard_flat = np.frombuffer(
                        src_view[s_start:s_start + shard_size],
                        dtype=np_dtype,
                    ).copy()
                    shard_arr = shard_flat.reshape(shard_shape)
                    shards.append(shard_arr)

                # Handle GLU reorder: [gate0|up0, gate1|up1] → [gate0|gate1|up0|up1]
                if spec["is_glu"]:
                    split_shards = [np.split(s, 2, axis=0) for s in shards]
                    shards = [s[0] for s in split_shards] + [s[1] for s in split_shards]
                    full_param = np.concatenate(shards, axis=0)
                elif spec["is_fc2_bug"]:
                    # partition_dim is 0 but should be 1
                    full_param = np.concatenate(shards, axis=1)
                else:
                    full_param = np.concatenate(shards, axis=partition_dim)
            else:
                full_param = np.frombuffer(
                    src_view[src_offset:src_offset + full_size],
                    dtype=np_dtype,
                ).copy().reshape(full_shape)

            src_offset += full_size  # buffer always has full_size per param

            # Vocab unpadding
            if spec["needs_vocab_unpad"] and full_param.shape[0] > vocab_size:
                full_param = full_param[:vocab_size]

            # Convert mcore → HF: split QKV, split gate/up, rename
            hf_pairs = self._convert_mcore_to_hf_numpy(
                spec["mcore_name"], full_param, conv_cfg,
                head_dim, num_heads, num_query_groups, hidden_size,
                value_num_per_group,
            )

            for hf_name, hf_arr in hf_pairs:
                hf_bytes = hf_arr.tobytes()
                dst_view[dst_offset:dst_offset + len(hf_bytes)] = hf_bytes
                hf_shape = list(hf_arr.shape)
                hf_tensors_meta.append(
                    (hf_name, (hf_shape, spec["dtype"]))
                )
                dst_offset += len(hf_bytes)

        self._hf_tensors_meta = hf_tensors_meta

    @staticmethod
    def _convert_mcore_to_hf_numpy(
        mcore_name: str,
        param: "np.ndarray",
        conv_cfg: dict,
        head_dim: int,
        num_heads: int,
        num_query_groups: int,
        hidden_size: int,
        value_num_per_group: int,
    ) -> list:
        """Convert a single mcore param (numpy) to HF-format (name, array) pairs.

        Mirrors the logic in ``astraflow.train_worker.utils.megatron.convert_to_hf``
        but operates on numpy arrays (CPU-only, no torch dependency).
        """
        import re

        import numpy as np

        if mcore_name == "module.module.embedding.word_embeddings.weight":
            return [("model.embed_tokens.weight", param)]
        if mcore_name == "module.module.output_layer.weight":
            return [("lm_head.weight", param)]
        if mcore_name == "module.module.decoder.final_layernorm.weight":
            return [("model.norm.weight", param)]

        decoder_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
        match = re.match(decoder_pattern, mcore_name)
        if not match:
            # Fallback: keep mcore name for unknown params (e.g. MoE router)
            return [(mcore_name, param)]

        layer_idx, rest = match.groups()

        # Expert params
        expert_pattern = r"mlp\.experts\.(.+)\.weight(\d+)"
        ematch = re.match(expert_pattern, rest)
        if ematch:
            expert_rest, expert_idx = ematch.groups()
            if expert_rest == "linear_fc1":
                gate, up = np.split(param, 2, axis=0)
                return [
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight", gate),
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight", up),
                ]
            elif expert_rest == "linear_fc2":
                return [(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight", param)]
            else:
                return [(mcore_name, param)]

        # Shared expert params
        shared_expert_pattern = r"mlp\.shared_experts\.(.+)"
        smatch = re.match(shared_expert_pattern, rest)
        if smatch:
            srest = smatch.groups()[0]
            if srest == "linear_fc1.weight":
                gate, up = np.split(param, 2, axis=0)
                return [
                    (f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight", gate),
                    (f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight", up),
                ]
            elif srest == "linear_fc2.weight":
                return [(f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight", param)]
            else:
                return [(mcore_name, param)]

        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.weight":
            param = param.reshape(num_query_groups, -1, head_dim, hidden_size)
            q, k, v = np.split(
                param, [value_num_per_group, value_num_per_group + 1], axis=1
            )
            q = q.reshape(-1, hidden_size)
            k = k.reshape(-1, hidden_size)
            v = v.reshape(-1, hidden_size)
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.weight", q),
                (f"model.layers.{layer_idx}.self_attn.k_proj.weight", k),
                (f"model.layers.{layer_idx}.self_attn.v_proj.weight", v),
            ]
        elif rest == "self_attention.linear_qkv.bias":
            param = param.reshape(num_query_groups, -1)
            q_bias, k_bias, v_bias = np.split(
                param,
                [value_num_per_group * head_dim, value_num_per_group * head_dim + head_dim],
                axis=1,
            )
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.bias", q_bias.flatten()),
                (f"model.layers.{layer_idx}.self_attn.k_proj.bias", k_bias.flatten()),
                (f"model.layers.{layer_idx}.self_attn.v_proj.bias", v_bias.flatten()),
            ]
        elif rest == "mlp.linear_fc1.weight":
            gate, up = np.split(param, 2, axis=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        elif rest == "pre_mlp_layernorm.weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        elif rest == "mlp.router.weight":
            return [(f"model.layers.{layer_idx}.mlp.gate.weight", param)]
        elif rest == "mlp.router.expert_bias":
            return [(f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias", param)]
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.k_norm.weight", param)]
        else:
            logger.warning("[Sender] Unknown mcore param: %s", mcore_name)
            return [(mcore_name, param)]

    def allocate_transfer_buffer(self, params_size: List[Tuple[str, int]]):
        assert len(self.transfer_engines) > 0, "Transfer Engines not initialized"

        # Allocate TWO contiguous halves so that T1 (trainer write) and T2
        # (TCP read) always operate on different halves — no cross-process lock
        # required.  The trainer writes to the "inactive" half; after sending
        # buffer_ready the sender atomically swaps active_buf_idx so the next
        # T2 reads from the freshly-written half.
        self._single_buffer_length = sum(size for _, size in params_size)
        self.buffer_length = 2 * self._single_buffer_length

        self.shm_file = tempfile.NamedTemporaryFile(
            dir="/dev/shm", delete=False, prefix="astraflow_buffer_"
        )
        os.ftruncate(self.shm_file.fileno(), self.buffer_length)
        self.mmap_buffer = mmap.mmap(
            self.shm_file.fileno(),
            self.buffer_length,
            flags=_MAP_FLAGS,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )

        self.buffer_ptr = ctypes.addressof(
            ctypes.c_byte.from_buffer(self.mmap_buffer)
        )
        self.memfd = self.shm_file.fileno()
        _configure_mmap(self.buffer_ptr, self.buffer_length, self.mmap_buffer)

        logger.info(
            "[Sender] Created double shared memory buffer at %s "
            "(%d bytes total, %d bytes per half)",
            self.shm_file.name,
            self.buffer_length,
            self._single_buffer_length,
        )

        # Allocate delta shm buffer if delta is enabled.
        # Pre-allocate at full model size (worst case: all elements changed).
        delta_shm_path = None
        if self._delta_enabled:
            self._delta_shm_file = tempfile.NamedTemporaryFile(
                dir="/dev/shm", delete=False, prefix="astraflow_delta_"
            )
            # Size the delta buffer for a max-density cap of 10 %
            # (i.e. assume sparsity >= 90 %). Format is unchanged —
            # uint64 idx (8 B) + bf16 val (2 B) = 10 B per nonzero element,
            # times num_elements = single_buffer_length / 2, times density_cap:
            #   bytes = 10 * (single_buffer_length / 2) * 0.10
            #         = single_buffer_length / 2
            # Plus the 16 B header. Anything denser overflows and is caught by
            # the guard in _compute_delta which flips _delta_ready=False so
            # the next pull falls back to full.
            delta_buf_size = 16 + self._single_buffer_length // 2
            os.ftruncate(self._delta_shm_file.fileno(), delta_buf_size)
            self._delta_mmap = mmap.mmap(
                self._delta_shm_file.fileno(),
                delta_buf_size,
                flags=_MAP_FLAGS,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
            delta_shm_path = self._delta_shm_file.name
            logger.info(
                "[Sender] Created delta shm buffer at %s (%d bytes)",
                delta_shm_path, delta_buf_size,
            )

        # WeightManager maps the full double buffer; tell it the total size.
        # Also send delta shm path if delta is enabled.
        self.output_queue.put((self.shm_file.name, self.buffer_length, delta_shm_path))

        for group_idx, group_engines in enumerate(self.transfer_engines):
            assert (
                len(group_engines) == 1
            ), "TCP engine should have single instance per group"
            engine = group_engines[0]
            # Register the full double-buffer memfd; individual transfers
            # use the active-half offset (see submit_transfer_to_instance).
            engine.register_memfd(self.memfd, self.buffer_length)
            self.buffer_slices.append(
                [(self.buffer_ptr, 0, self.buffer_length)]
            )
            logger.info(
                "Registered shared memfd with TCP engine group %d", group_idx
            )

    def get_hostname(self):
        return (
            self.transfer_engines[0][0].get_hostname()
            if self.transfer_engines
            else None
        )

    def get_rpc_port(self):
        return (
            self.transfer_engines[0][0].get_rpc_port()
            if self.transfer_engines
            else None
        )

    def get_session_ids(self) -> List[List[str]]:
        session_ids = []
        for group_engines in self.transfer_engines:
            group_session_ids = [
                engine.get_session_id() for engine in group_engines
            ]
            session_ids.append(group_session_ids)
        return session_ids

    # ------------------------------------------------------------------
    # Event loop (trainer → sender communication via mp.Queue)
    # ------------------------------------------------------------------

    def event_loop(self):
        """Process messages from the trainer.

        On ``buffer_ready``:
        1. Swap active buffer index
        2. Ack the trainer immediately (trainer does NOT wait for delta)
        3. If delta enabled and version > 1: compute sparse delta
           asynchronously, then signal ``_delta_done_event``

        The trainer calls ``wait_delta_ready()`` before
        ``notify_version``, ensuring delta is ready when RaaS pulls.
        """
        try:
            while True:
                try:
                    request = self.input_queue.get(timeout=1.0)
                    if isinstance(request, str) and request.startswith(
                        "buffer_ready:"
                    ):
                        parts = request.split(":")
                        version = int(parts[1])
                        buf_idx = int(parts[2]) if len(parts) > 2 else 0

                        # Swap active buffer index
                        self._active_buf_idx = buf_idx
                        self.weight_version = version

                        with self._delta_lock:
                            self._delta_ready = False

                        # Megatron: reassemble TP shards → HF format
                        # in the HF serving buffer (CPU-only, before ack).
                        if hasattr(self, "_megatron_metadata") and self._megatron_metadata is not None:
                            t_reassemble_start = time.perf_counter()
                            self._reassemble_megatron_to_hf(buf_idx)
                            t_reassemble = time.perf_counter() - t_reassemble_start
                            # Register HF serving buffer as the memfd for
                            # TCP transfer (replaces the raw shard buffer).
                            for group_engines in self.transfer_engines:
                                for engine in group_engines:
                                    engine.register_memfd(
                                        self._hf_serving_memfd,
                                        self._hf_serving_length,
                                    )
                            # Update tensors_meta so RaaS sees HF param names.
                            if self._hf_tensors_meta is not None:
                                self.tensors_meta = self._hf_tensors_meta
                            print(
                                f"[Weight sender] Megatron reassembly: "
                                f"v={version}, {t_reassemble:.3f}s",
                                flush=True,
                            )

                        # Ack IMMEDIATELY — trainer unblocks without
                        # waiting for delta compute.
                        ack = {
                            "status": "ack",
                            "weight_version": version,
                        }
                        self.output_queue.put(ack)

                        # Compute delta AFTER ack (async w.r.t. trainer).
                        # After swap: active half = v=N (just written),
                        # inactive half = v=N-1 (written last step) — but
                        # only if we've completed at least one previous
                        # offload in this process. On the first
                        # buffer_ready (fresh start or recovery push) the
                        # inactive half is zero, so a "delta" would be
                        # ~100% dense, wasting 10s of seconds and huge
                        # memory for a useless result. Skip it.
                        if (
                            self._delta_enabled
                            and version > 1
                            and self._inactive_half_has_data
                        ):
                            self._compute_delta(version, buf_idx)
                            # Send delta metrics on output queue for the
                            # trainer to pick up in wait_delta_ready().
                            with self._delta_lock:
                                self.output_queue.put({
                                    "type": "delta_metrics",
                                    "delta_sparsity": self._delta_sparsity,
                                    "delta_size": self._delta_size,
                                    "delta_num_nonzero": self._delta_num_nonzero,
                                    "delta_compute_time": self._delta_compute_time,
                                })
                        else:
                            if self._delta_enabled and version > 1:
                                print(
                                    f"[Weight sender] Skipping delta v={version}: "
                                    f"inactive half has no prior data "
                                    f"(first offload in this process)",
                                    flush=True,
                                )
                            self.output_queue.put({"type": "no_delta"})

                        # The half we just wrote is now real data. After
                        # the trainer flips its inactive_buf_idx, the
                        # _next_ offload will make the CURRENT active half
                        # (which holds v=N) become the inactive source for
                        # the next delta — so flip the flag on.
                        self._inactive_half_has_data = True

                        # Signal: delta done (or no delta needed).
                        # Unblocks the guard in the next offload() and
                        # wait_delta_ready() before notify_version.
                        if self._delta_done_event is not None:
                            self._delta_done_event.set()

                except queue.Empty:
                    continue
        except Exception as e:
            logger.error(
                "Transfer agent event loop terminated: %s", e, exc_info=True
            )
            try:
                self.output_queue.put(f"error:{e}")
            except Exception:
                pass
            raise

    # Chunk granularity (in bf16 elements) for _compute_delta. 512 M × 2 B
    # = 1 GB per chunk. Peak transient: ~512 MB bool mask per chunk, far
    # below the ~7.6 GB peak of the whole-buffer approach. Sized to stay
    # well under INT_MAX so no numpy >2**31 code paths are exercised.
    _DELTA_CHUNK_ELEMS = 1 << 29

    def _compute_delta(self, version: int, buf_idx: int) -> None:
        """Compute sparse delta between active (new) and inactive (old) halves.

        Called in the event loop AFTER acking the trainer (async).
        The active half (buf_idx) holds v=N, the inactive half (1-buf_idx)
        holds v=N-1.  The inactive half is safe to read because the guard
        barrier in offload() prevents the next write until delta_done_event
        is set.

        Chunked: iterates the two halves in ~1 GB slices to cap the
        transient bool-mask allocation. Prevents the giant peak that
        coincided with concurrent FSDP checkpoint saves and caused the
        OOM-kill we saw in the delta experiment.
        """
        try:
            import numpy as np

            t0 = time.perf_counter()

            new_offset = buf_idx * self._single_buffer_length
            old_offset = (1 - buf_idx) * self._single_buffer_length
            buf_len = self._single_buffer_length

            element_size = 2  # bf16
            num_elements = buf_len // element_size

            # memoryview slices are views over self.mmap_buffer — cheap, no copy.
            new_view = memoryview(self.mmap_buffer)[new_offset:new_offset + buf_len]
            old_view = memoryview(self.mmap_buffer)[old_offset:old_offset + buf_len]
            new_full = np.frombuffer(new_view, dtype=np.uint16)
            old_full = np.frombuffer(old_view, dtype=np.uint16)

            # Header format: [num_nonzero u64][element_size u16][flags u16][reserved i32]
            header_size = 16
            use_uint64 = num_elements > (2**32 - 1)
            idx_dtype = np.uint64 if use_uint64 else np.uint32
            idx_size = 8 if use_uint64 else 4

            # Accumulate per-chunk (indices, values). Each chunk contributes a
            # sparse pair; final concatenation is ~the same size as the delta
            # body itself (typically ~300 MB for a 7 B model at 0.4 % density).
            idx_chunks: List[Any] = []
            val_chunks: List[Any] = []

            chunk_elems = self._DELTA_CHUNK_ELEMS
            for start in range(0, num_elements, chunk_elems):
                end = min(start + chunk_elems, num_elements)
                new_chunk = new_full[start:end]
                old_chunk = old_full[start:end]

                # Chunk-local bool mask: up to ~512 MB for a full-size chunk.
                diff_mask = new_chunk != old_chunk
                local_idx = np.where(diff_mask)[0]
                # Free the big mask before building the next chunk's.
                del diff_mask

                if local_idx.size == 0:
                    continue

                # local_idx is int64. Offset to global and cast to target dtype.
                global_idx = (local_idx + start).astype(idx_dtype)
                chunk_vals = new_chunk[local_idx]  # uint16

                idx_chunks.append(global_idx)
                val_chunks.append(chunk_vals)

            if idx_chunks:
                all_idx = np.concatenate(idx_chunks)
                all_vals = np.concatenate(val_chunks)
            else:
                all_idx = np.empty(0, dtype=idx_dtype)
                all_vals = np.empty(0, dtype=np.uint16)

            num_nonzero = int(all_idx.size)
            sparsity = (
                1.0 - (num_nonzero / num_elements) if num_elements > 0 else 1.0
            )

            indices_size = num_nonzero * idx_size
            values_size = num_nonzero * element_size
            total_size = header_size + indices_size + values_size

            # Overflow guard: pre-alloc now assumes <=10% density.
            # Raise so the existing except flips _delta_ready=False and
            # the next /request_transfer falls back to full.
            if total_size > len(self._delta_mmap):
                raise RuntimeError(
                    f"delta exceeds pre-allocated buffer "
                    f"({total_size} > {len(self._delta_mmap)} B); density cap"
                )

            flags = 1 if use_uint64 else 0
            # Acquire _delta_mmap_lock around the writes so that any
            # concurrent reader (TCP send in a different thread) sees a
            # fully-written delta or waits.
            with self._delta_mmap_lock:
                struct.pack_into(
                    "<QHHi", self._delta_mmap, 0,
                    num_nonzero, element_size, flags, 0,
                )
                if num_nonzero > 0:
                    self._delta_mmap[
                        header_size:header_size + indices_size
                    ] = all_idx.tobytes()
                    self._delta_mmap[
                        header_size + indices_size:
                        header_size + indices_size + values_size
                    ] = all_vals.view(np.uint8).tobytes()

            t1 = time.perf_counter()

            with self._delta_lock:
                self._delta_ready = True
                self._delta_size = total_size
                self._delta_base_version = version - 1
                self._delta_version = version
                self._delta_compute_time = t1 - t0
                self._delta_sparsity = sparsity
                self._delta_num_nonzero = num_nonzero

            print(
                f"[Weight sender] Delta computed: v={version} vs v={version-1}, "
                f"sparsity={sparsity:.4f}, nonzero={num_nonzero}/{num_elements}, "
                f"delta_size={total_size / (1024 * 1024):.1f} MB "
                f"(full={buf_len / (1024 * 1024):.1f} MB, "
                f"ratio={buf_len / total_size if total_size > 0 else 0:.1f}x), "
                f"compute_time={t1 - t0:.3f}s",
                flush=True,
            )

        except Exception as exc:
            logger.error(
                "[Weight sender] Delta computation failed: %s",
                exc, exc_info=True,
            )
            print(
                f"[Weight sender] Delta computation failed: {exc}",
                flush=True,
            )
            with self._delta_lock:
                self._delta_ready = False

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def get_capabilities(self) -> dict:
        """Return supported transfer strategies and delta state."""
        result = {
            "strategies": self.strategies,
            "version": self.weight_version,
        }
        if self._delta_enabled:
            with self._delta_lock:
                result["delta_ready"] = self._delta_ready
                result["delta_base_version"] = self._delta_base_version
                result["delta_version"] = self._delta_version
                result["delta_sparsity"] = self._delta_sparsity
                result["delta_num_nonzero"] = self._delta_num_nonzero
                result["delta_size_bytes"] = self._delta_size
        return result

    # ------------------------------------------------------------------
    # RaaS-initiated transfer (pull model)
    # ------------------------------------------------------------------

    def handle_transfer_request(self, instance_id: str, mode: str = "full") -> dict:
        """Handle a transfer request from a specific RaaS instance.

        Called via HTTP ``POST /request_transfer``.  Acquires the buffer
        lock so the transfer does not overlap with a trainer buffer write.

        Parameters
        ----------
        mode : str
            ``"full"`` for full weight transfer, ``"delta"`` for sparse delta.
        """
        if self.weight_version == 0:
            return {
                "ok": False,
                "version": 0,
                "error": "No weights available yet",
            }

        bare_id = instance_id.replace("http://", "")
        if bare_id not in self.registered_receivers:
            return {
                "ok": False,
                "version": self.weight_version,
                "error": f"Instance {instance_id} not registered",
            }

        # Delta mode: serve from delta buffer (no TCP engine, direct HTTP)
        if mode == "delta":
            return self._handle_delta_transfer(bare_id, instance_id)

        # Full mode: serve via TCP from the active half
        with self._buffer_lock:
            version = self.weight_version
            try:
                self._transfer_to_instance(bare_id)
                logger.info(
                    "[Weight sender] Pull transfer (full) to %s completed "
                    "(version=%d)",
                    instance_id,
                    version,
                )
                return {"ok": True, "version": version, "mode": "full"}
            except Exception as exc:
                logger.error(
                    "[Weight sender] Pull transfer to %s failed: %s",
                    instance_id,
                    exc,
                    exc_info=True,
                )
                return {
                    "ok": False,
                    "version": version,
                    "error": str(exc),
                }

    def _handle_delta_transfer(self, bare_id: str, instance_id: str) -> dict:
        """Transfer sparse delta via TCP using the delta shm buffer.

        Temporarily swaps the TCP engine's memfd to the delta shm file,
        transfers delta_size bytes, then restores the original memfd.
        Uses the same TCP sendfile() path as full transfers.
        """
        with self._delta_lock:
            if not self._delta_ready:
                logger.info(
                    "[Weight sender] Delta not ready for %s, suggesting full",
                    instance_id,
                )
                return {
                    "ok": False,
                    "version": self.weight_version,
                    "fallback": "full",
                    "reason": "delta_not_ready",
                }
            delta_size = self._delta_size
            delta_base = self._delta_base_version
            delta_version = self._delta_version
            sparsity = self._delta_sparsity
            num_nonzero = self._delta_num_nonzero
            compute_time = self._delta_compute_time

        with self._buffer_lock:
            receiver_info = self.registered_receivers[bare_id]
            group_idx = receiver_info.sender_group_index
            group_engines = self.transfer_engines[group_idx]
            assert len(group_engines) == 1

            engine = group_engines[0]
            target_session_id = receiver_info.session_ids[0]

            t_start = time.perf_counter()

            # Swap memfd to delta shm file
            delta_memfd = self._delta_shm_file.fileno()
            original_memfd = self.memfd
            engine.register_memfd(delta_memfd, self._single_buffer_length)
            t_swap = time.perf_counter()

            try:
                # Acquire _delta_mmap_lock around TCP send so concurrent
                # _compute_delta (in event_loop thread) cannot overwrite
                # the bytes mid-pull. This is the actual race-free fix.
                with self._delta_mmap_lock:
                    # Transfer delta_size bytes from offset 0 in delta buffer
                    batch_id = engine.transfer_submit_write(
                        target_session_id,
                        0,           # local offset (start of delta buffer)
                        0,           # remote offset on receiver
                        delta_size,  # only transfer delta_size, not full buffer
                    )
                    t_submit = time.perf_counter()

                    # Wait for completion
                    while True:
                        status = engine.transfer_check_status(batch_id)
                        if status == 1:
                            break
                        elif status < 0:
                            self.sync_status_to_receiver_endpoint(
                                bare_id, TransferStatus.FAILURE
                            )
                            raise RuntimeError(
                                f"Delta transfer to {instance_id} failed"
                            )
                        time.sleep(0.001)
                    t_tcp_done = time.perf_counter()

                    self.sync_status_to_receiver_endpoint(
                        bare_id, TransferStatus.SUCCESS
                    )
                    t_zmq = time.perf_counter()

                print(
                    f"[Weight sender] Delta TCP to {instance_id}: "
                    f"v={delta_version} (base={delta_base}), "
                    f"size={delta_size / (1024 * 1024):.1f} MB, "
                    f"swap_memfd={t_swap - t_start:.3f}s, "
                    f"submit={t_submit - t_swap:.3f}s, "
                    f"tcp_wait={t_tcp_done - t_submit:.3f}s, "
                    f"zmq={t_zmq - t_tcp_done:.3f}s, "
                    f"total={t_zmq - t_start:.3f}s "
                    f"({delta_size / (t_zmq - t_start) / (1024 * 1024):.0f} MB/s)",
                    flush=True,
                )

            finally:
                # Restore original memfd
                engine.register_memfd(original_memfd, self.buffer_length)

        return {
            "ok": True,
            "version": delta_version,
            "mode": "delta",
            "delta_base_version": delta_base,
            "delta_size": delta_size,
            "sparsity": sparsity,
            "num_nonzero": num_nonzero,
            "compute_time": compute_time,
        }

    def _transfer_to_instance(self, instance_id: str) -> None:
        """TCP-transfer the current buffer contents to a single instance."""
        batch_ids = self.submit_transfer_to_instance(instance_id)
        receiver_info = self.registered_receivers[instance_id]
        group_idx = receiver_info.sender_group_index
        group_engines = self.transfer_engines[group_idx]
        completed_flags = [False] * len(group_engines)

        start_time = time.perf_counter()
        while not all(completed_flags):
            time.sleep(0.001)
            for engine_idx, (engine, batch_id) in enumerate(
                zip(group_engines, batch_ids)
            ):
                if completed_flags[engine_idx]:
                    continue
                status = engine.transfer_check_status(batch_id)
                if status == 1:
                    completed_flags[engine_idx] = True
                elif status < 0:
                    self.sync_status_to_receiver_endpoint(
                        instance_id, TransferStatus.FAILURE
                    )
                    raise RuntimeError(
                        f"Transfer to {instance_id} engine {engine_idx} "
                        f"failed with status {status}"
                    )

        self.sync_status_to_receiver_endpoint(
            instance_id, TransferStatus.SUCCESS
        )
        elapsed = time.perf_counter() - start_time
        bw = self._single_buffer_length / elapsed / 1024 / 1024 if elapsed > 0 else 0
        logger.info(
            "[Weight sender] Transfer to %s done in %.2fs (%.1f MB/s)",
            instance_id,
            elapsed,
            bw,
        )

    # ------------------------------------------------------------------
    # Receiver registration
    # ------------------------------------------------------------------

    def wait_for_receiver_registration(
        self, instance_ids: List[str], timeout: float = 120.0
    ):
        deadline = time.monotonic() + timeout
        while True:
            missing = [
                iid
                for iid in instance_ids
                if iid not in self.registered_receivers
            ]
            if not missing:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"[Weight sender] Receiver registration timed out after "
                    f"{timeout}s. Still waiting for: {missing}"
                )
            time.sleep(1)

    def register_receiver_session(
        self,
        instance_id: str,
        session_ids: List[str],
        buffer_ptr: int,
        buffer_length: int,
        zmq_endpoint: str,
        zmq_port: int,
        sglang_http_host: str = None,
        sglang_http_port: int = None,
        handshake_ports: List[int] = None,
        sender_group_index: int = 0,
    ):
        if instance_id in self.registered_receivers:
            logger.info(
                "[Weight sender] Instance %s re-registering (e.g. RaaS rejoin)",
                instance_id,
            )
            # Close stale ZMQ socket — the rejoining receiver has a new
            # ZMQ port, so the cached socket points to a dead endpoint.
            old_sock = self._zmq_push_sockets.pop(instance_id, None)
            if old_sock is not None:
                try:
                    old_sock.close()
                except Exception:
                    pass
                logger.info(
                    "[Weight sender] Closed stale ZMQ socket for %s",
                    instance_id,
                )

        assert self._single_buffer_length is not None, "Transfer buffer not allocated"
        # Receiver allocates a buffer of single_buffer_length; sender holds a
        # double-sized buffer.  Compare against the per-half size.
        assert (
            self._single_buffer_length == buffer_length
        ), (
            f"Transfer buffer length mismatch: sender single_buffer="
            f"{self._single_buffer_length}, receiver buffer={buffer_length}"
        )
        assert sender_group_index < len(
            self.transfer_engines
        ), f"Invalid sender group index {sender_group_index}"

        self.registered_receivers[instance_id] = ReceiverInfo(
            session_ids=session_ids,
            buffer_ptr=buffer_ptr,
            buffer_length=buffer_length,
            zmq_endpoint=zmq_endpoint,
            zmq_port=zmq_port,
            sglang_http_host=sglang_http_host,
            sglang_http_port=sglang_http_port,
            handshake_ports=handshake_ports,
            sender_group_index=sender_group_index,
        )
        logger.info(
            "[Weight sender] Registered rollout instance %s with session "
            "IDs %s using group %d",
            instance_id,
            session_ids,
            sender_group_index,
        )
        return True

    # ------------------------------------------------------------------
    # Low-level transfer helpers
    # ------------------------------------------------------------------

    def submit_transfer_to_instance(self, instance_id: str) -> List[int]:
        assert (
            instance_id in self.registered_receivers
        ), f"Instance {instance_id} not registered"
        receiver_info = self.registered_receivers[instance_id]

        group_idx = receiver_info.sender_group_index
        group_engines = self.transfer_engines[group_idx]

        batch_ids = []
        assert (
            len(group_engines) == 1
        ), "TCP engine should have single instance per group"
        target_session_id = receiver_info.session_ids[0]

        # For Megatron: serve from single HF serving buffer (offset 0).
        # For FSDP: serve from the active half of the double buffer.
        if hasattr(self, "_hf_serving_buffer") and self._hf_serving_buffer is not None:
            active_offset = 0
            transfer_length = self._hf_serving_length
        else:
            active_offset = self._active_buf_idx * self._single_buffer_length
            transfer_length = self._single_buffer_length

        batch_id = group_engines[0].transfer_submit_write(
            target_session_id,
            active_offset,   # local memfd offset (active half)
            0,               # remote offset on receiver
            transfer_length,
        )
        batch_ids.append(batch_id)
        logger.debug(
            "Submitted TCP transfer to %s: batch_id=%d "
            "active_buf_idx=%d local_offset=%d length=%d",
            instance_id,
            batch_id,
            self._active_buf_idx,
            active_offset,
            transfer_length,
        )
        return batch_ids

    def _get_zmq_push_socket(self, instance_id: str):
        if self._zmq_ctx is None:
            self._zmq_ctx = zmq.Context()
        if instance_id not in self._zmq_push_sockets:
            receiver_info = self.registered_receivers[instance_id]
            zmq_addr = (
                f"tcp://{receiver_info.zmq_endpoint}:{receiver_info.zmq_port}"
            )
            sock = self._zmq_ctx.socket(zmq.PUSH)
            sock.connect(zmq_addr)
            self._zmq_push_sockets[instance_id] = sock
            logger.info(
                "[Weight sender] Created ZMQ PUSH socket for %s -> %s",
                instance_id,
                zmq_addr,
            )
        return self._zmq_push_sockets[instance_id]

    def sync_status_to_receiver_endpoint(
        self, instance_id: str, status: TransferStatus
    ):
        receiver_info = self.registered_receivers[instance_id]
        logger.info(
            "[Weight sender] Sending ZMQ status=%s to %s (zmq=%s:%d)",
            status.name,
            instance_id,
            receiver_info.zmq_endpoint,
            receiver_info.zmq_port,
        )
        sock = self._get_zmq_push_socket(instance_id)
        sock.send_multipart(
            [
                str(self.config.trainer_global_rank).encode("ascii"),
                str(int(status)).encode("ascii"),
            ]
        )
        logger.info(
            "[Weight sender] ZMQ status=%s sent successfully to %s",
            status.name,
            instance_id,
        )



def _init(
    config: SenderAgentConfig,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    event,
    delta_done_event=None,
):
    import traceback as _tb

    # Redirect sender stdout/stderr to a dedicated log file (if the trainer
    # set config.log_file) BEFORE any other work, so that crashes during
    # init still land on disk. C-level writes via fd 1/2 also go there —
    # important for faulthandler output and any libc panics that would
    # otherwise vanish into the torchrun/tee pipeline.
    log_path = getattr(config, "log_file", None)
    if log_path:
        try:
            log_dir = os.path.dirname(log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            _sender_log_f = open(log_path, "a", buffering=1)
            os.dup2(_sender_log_f.fileno(), 1)
            os.dup2(_sender_log_f.fileno(), 2)
            sys.stdout = _sender_log_f
            sys.stderr = _sender_log_f
        except Exception:
            pass

    # Dump Python + C traceback to stderr on fatal signals (SIGSEGV, SIGFPE,
    # SIGBUS, SIGABRT, SIGILL). Does NOT catch SIGKILL — OOM kills still
    # disappear silently, but for those we have /proc/vmstat:oom_kill.
    try:
        import faulthandler
        faulthandler.enable(all_threads=True)
    except Exception:
        pass

    try:
        weights_meta_size = input_queue.get()
        tensors_meta = input_queue.get()
        strategies = input_queue.get()
        megatron_metadata = input_queue.get()

        transfer_agent = TransferAgent(
            input_queue, output_queue, config, strategies=strategies,
            delta_done_event=delta_done_event,
        )
        transfer_agent.tensors_meta = tensors_meta

        if megatron_metadata is not None:
            transfer_agent.setup_megatron_reassembly(megatron_metadata)

        handler_class = _make_http_handler(transfer_agent)
        server = _ThreadingHTTPServer(
            ("0.0.0.0", config.http_bind_port), handler_class
        )
        logger.info(
            "Starting sender HTTP server on 0.0.0.0:%d (strategies=%s)",
            config.http_bind_port, strategies,
        )
        threading.Thread(target=server.serve_forever, daemon=True).start()

        lora_config = input_queue.get()
        transfer_agent.lora_config = lora_config
        transfer_agent.allocate_transfer_buffer(weights_meta_size)

        event.set()
    except BaseException as exc:
        err = (
            f"[sender_agent._init] {type(exc).__name__}: {exc}\n"
            f"{_tb.format_exc()}"
        )
        try:
            output_queue.put(("__sender_init_error__", err))
        except Exception:
            pass
        try:
            sys.stderr.write(err + "\n")
            sys.stderr.flush()
        except Exception:
            pass
        event.set()  # unblock parent so it can observe the error
        raise

    transfer_agent.event_loop()


def start_transfer_agent(
    config: SenderAgentConfig,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    ctx=None,
    delta_done_event=None,
) -> mp.Process:
    if ctx is None:
        ctx = mp.get_context("spawn")
    event = ctx.Event()
    proc = ctx.Process(
        target=_init,
        args=(
            config,
            input_queue,
            output_queue,
            event,
            delta_done_event,
        ),
    )
    proc.start()
    if not event.wait(timeout=120):
        if not proc.is_alive():
            err = _drain_sender_init_error(output_queue)
            raise RuntimeError(
                "Sender transfer agent died before signalling ready "
                f"(exitcode={proc.exitcode}). Child error:\n{err}"
            )
        raise RuntimeError(
            "Sender transfer agent did not signal ready within 120s "
            f"(pid={proc.pid} alive=True)."
        )
    if not proc.is_alive():
        err = _drain_sender_init_error(output_queue)
        raise RuntimeError(
            "Sender transfer agent died right after signalling ready "
            f"(exitcode={proc.exitcode}). Child error:\n{err}"
        )
    logger.info("Successfully started sender transfer agent process.")
    return proc


def _drain_sender_init_error(output_queue: mp.Queue) -> str:
    """Non-blocking drain of output_queue for a sender_agent init error."""
    try:
        while True:
            item = output_queue.get_nowait()
            if isinstance(item, tuple) and item and item[0] == "__sender_init_error__":
                return item[1]
    except Exception:
        pass
    return "<no error payload from child>"
