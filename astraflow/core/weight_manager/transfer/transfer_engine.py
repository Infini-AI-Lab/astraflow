import logging
import os
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

from .config import TransferEngineConfig

logger = logging.getLogger(__name__)


class TCPTransferEngine:
    def __init__(self, config: TransferEngineConfig, num_threads: int = 6):
        self.config = config

        self.buffer_ptr: Optional[int] = None
        self.buffer_length: Optional[int] = None
        self.buffer_memview: Optional[memoryview] = None
        self.listener_threads = []
        self.listener_sockets = []
        self.listener_ports = []

        if self.config.handshake_port:
            self.session_id = f"{self.config.local_hostname}:{self.config.handshake_port}"
        else:
            session_suffix = "_" + str(uuid.uuid4())
            self.session_id = self.config.local_hostname + session_suffix

        self.connections: Dict[str, socket.socket] = {}
        self.connection_lock = threading.Lock()
        self.num_parallel_streams = num_threads
        self.transfer_executor = ThreadPoolExecutor(max_workers=self.num_parallel_streams * 2)
        self.pending_transfers: Dict[int, Dict] = {}
        self.next_batch_id = 1
        self.batch_id_lock = threading.Lock()
        self.is_receiver = False
        # Optimize for large transfers
        self.rcvbuf_size = 16 * 1024 * 1024  # 16MB socket buffer
        self.sndbuf_size = 16 * 1024 * 1024  # 16MB socket buffer
        self.chunk_size = 64 * 1024 * 1024   # 64MB chunks

        # Enable SO_ZEROCOPY if available (Linux 4.14+)
        self.use_zerocopy = os.environ.get('TCP_ZEROCOPY', '0') == '1'

    def register(self, ptr: int, length: int):
        """Register buffer for receive operations (receiver side)."""
        self.buffer_ptr = ptr
        self.buffer_length = length
        import ctypes
        buf = (ctypes.c_byte * length).from_address(ptr)
        self.buffer_memview = memoryview(buf)
        logger.info(f"TCPTransferEngine registered buffer: ptr={ptr}, length={length}")

    def register_memfd(self, memfd: int, length: int):
        """Register memfd for sendfile operations (sender side)."""
        self.memfd = memfd
        self.buffer_length = length
        logger.info(f"TCPTransferEngine registered memfd: fd={memfd}, length={length}")

    def deregister(self, ptr: int):
        self.buffer_memview = None
        self.buffer_ptr = None
        self.buffer_length = None
        self.memfd = None

    def start_listener(self):
        if len(self.listener_threads) > 0:
            return

        for i in range(self.num_parallel_streams):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.rcvbuf_size)
            sock.bind(('', 0))
            port = sock.getsockname()[1]
            sock.listen(256)

            self.listener_sockets.append(sock)
            self.listener_ports.append(port)

            thread = threading.Thread(target=self._accept_connections, args=(sock, i), daemon=True)
            thread.start()
            self.listener_threads.append(thread)

        logger.info(f"TCPTransferEngine started {self.num_parallel_streams} listeners on ports {self.listener_ports}")

    def _accept_connections(self, sock: socket.socket, listener_idx: int):
        while True:
            try:
                conn, addr = sock.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.rcvbuf_size)
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.sndbuf_size)
                self.transfer_executor.submit(self._receive_data, conn, f"{addr[0]}:{addr[1]}-L{listener_idx}")
            except Exception as e:
                if sock:
                    logger.error(f"Accept error on listener {listener_idx}: {e}")
                break

    def _receive_data(self, conn: socket.socket, thread_id: str):
        try:
            if hasattr(socket, 'TCP_QUICKACK'):
                try:
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
                except Exception:
                    pass

            header = conn.recv(16)
            if len(header) < 16:
                logger.error(f"Invalid header from {thread_id}")
                return

            offset = int.from_bytes(header[:8], 'little')
            length = int.from_bytes(header[8:16], 'little')

            if offset + length > self.buffer_length:
                logger.error(f"Invalid offset/length from {thread_id}: {offset}/{length}")
                return

            view = self.buffer_memview[offset:offset + length]
            received = 0

            while received < length:
                chunk_size = min(self.chunk_size, length - received)
                n = conn.recv_into(view[received:received + chunk_size], chunk_size)
                if n == 0:
                    raise RuntimeError("Connection closed")
                received += n

            # Application-level ACK: tell sender we have received and
            # written all bytes. Sender waits for this before signaling
            # transfer-complete via ZMQ — otherwise ZMQ can arrive at
            # receiver before TCP fully delivers, and `wait_for_transfer`
            # returns prematurely with partial bytes in the buffer.
            conn.sendall(b"\x01")

            logger.info(f"Received {length} bytes at offset {offset} from {thread_id}")
        except Exception as e:
            logger.error(f"Receive error from {thread_id}: {e}")
        finally:
            conn.close()

    def _create_connection(self, target_host: str, target_port: int) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.sndbuf_size)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.rcvbuf_size)
        sock.connect((target_host, target_port))
        return sock

    def _send_data_chunk(self, sock: socket.socket, target_host: str, target_port: int,
                         local_offset: int, remote_offset: int, length: int):
        try:
            header = remote_offset.to_bytes(8, 'little') + length.to_bytes(8, 'little')
            sock.sendall(header)

            if not hasattr(self, 'memfd') or self.memfd is None:
                raise RuntimeError("memfd not registered - cannot send data")

            sent = 0
            while sent < length:
                n = os.sendfile(sock.fileno(), self.memfd, local_offset + sent, min(length - sent, 2147483647))
                if n == 0:
                    raise RuntimeError("sendfile returned 0 - connection may be broken")
                sent += n

            if sent != length:
                raise RuntimeError(f"sendfile incomplete: sent {sent}/{length} bytes")

            # Application-level ACK: wait for receiver to confirm it has
            # FULLY received and written all bytes. Without this,
            # `os.sendfile` returns when bytes are in the kernel SEND
            # buffer (not when they reach the receiver), and the
            # subsequent ZMQ "SUCCESS" message can race ahead of TCP
            # packets across the network — causing receivers (cross-node
            # only) to read their buffer before TCP recv has populated
            # it. Same-node loopback never hit this because TCP delivery
            # is essentially synchronous.
            ack = sock.recv(1)
            if not ack:
                raise RuntimeError("receiver closed before ACK")

            return True
        except Exception as e:
            logger.error(f"Send error to {target_host}:{target_port}: {e}")
            return False
        finally:
            sock.close()

    def transfer_sync(self, session_id: str, buffer: int, peer_buffer_address: int, length: int) -> int:
        batch_id = self.transfer_submit_write(session_id, buffer, peer_buffer_address, length)
        if batch_id < 0:
            return batch_id

        while True:
            status = self.transfer_check_status(batch_id)
            if status != 0:
                return 0 if status == 1 else -1
            time.sleep(0.001)

    def transfer_submit_write(self, session_id: str, buffer: int, peer_buffer_address: int, length: int) -> int:
        parts = session_id.split(':')
        if len(parts) < 2:
            logger.error(f"Invalid session_id format: {session_id}")
            return -1

        target_host = parts[0]

        if len(parts) == 2:
            target_ports = [int(parts[1])] * self.num_parallel_streams
        else:
            target_ports = [int(p) for p in parts[1:]]
            if len(target_ports) != self.num_parallel_streams:
                logger.error(f"Session ID has {len(target_ports)} ports, expected {self.num_parallel_streams}")
                return -1

        if self.buffer_ptr is not None:
            local_offset = buffer - self.buffer_ptr
        else:
            local_offset = buffer

        if local_offset < 0 or local_offset + length > self.buffer_length:
            logger.error(f"Invalid buffer range: offset={local_offset}, length={length}")
            return -1

        with self.batch_id_lock:
            batch_id = self.next_batch_id
            self.next_batch_id += 1
            self.pending_transfers[batch_id] = {
                'status': 0,
                'target': session_id,
                'length': length,
                'completed_chunks': 0,
                'total_chunks': self.num_parallel_streams,
            }

        chunk_size = length // self.num_parallel_streams
        futures = []

        for i in range(self.num_parallel_streams):
            chunk_offset = local_offset + i * chunk_size
            remote_chunk_offset = i * chunk_size

            if i == self.num_parallel_streams - 1:
                chunk_length = length - (i * chunk_size)
            else:
                chunk_length = chunk_size

            sock = self._create_connection(target_host, target_ports[i])
            future = self.transfer_executor.submit(
                self._send_data_chunk, sock, target_host, target_ports[i],
                chunk_offset, remote_chunk_offset, chunk_length
            )
            futures.append(future)

        def update_status():
            success_count = 0
            for future in futures:
                if future.result():
                    success_count += 1

            with self.batch_id_lock:
                if batch_id in self.pending_transfers:
                    if success_count == self.num_parallel_streams:
                        self.pending_transfers[batch_id]['status'] = 1
                    else:
                        self.pending_transfers[batch_id]['status'] = -1

        self.transfer_executor.submit(update_status)

        return batch_id

    def transfer_check_status(self, batch_id: int) -> int:
        with self.batch_id_lock:
            if batch_id not in self.pending_transfers:
                return -1
            return self.pending_transfers[batch_id]['status']

    def get_hostname(self):
        return self.config.local_hostname

    def get_session_id(self):
        if self.is_receiver and len(self.listener_ports) > 0:
            ports_str = ':'.join(str(p) for p in self.listener_ports)
            return f"{self.config.local_hostname}:{ports_str}"
        hostname, _ = self.session_id.split(":")
        if len(self.listener_ports) > 0:
            return f"{hostname}:{self.listener_ports[0]}"
        return f"{hostname}:{self.config.handshake_port}"

    def get_rpc_port(self):
        if self.is_receiver and len(self.listener_ports) > 0:
            return self.listener_ports[0]
        return self.config.handshake_port
