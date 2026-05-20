import ipaddress
import socket
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple


@dataclass
class TransferEngineConfig:
    local_hostname: str
    handshake_port: int


@dataclass
class SenderAgentConfig:
    trainer_global_rank: int
    trainer_world_size: int
    engine_configs: List[TransferEngineConfig]
    num_engines_per_group: int = 1
    http_bind_port: int = 18861
    # Absolute path for the sender subprocess to redirect its stdout/stderr.
    # If set, the spawned sender process will open this file (append mode),
    # dup2 fd 1/2 onto it, and enable faulthandler so fatal signals land a
    # traceback in the same file. Leave None to keep default behaviour
    # (inherited stdio piped through the parent).
    log_file: Optional[str] = None


@dataclass
class ReceiverAgentConfig:
    sglang_http_host: str
    sglang_http_port: int
    sender_http_endpoints: List[Tuple[str, int]]
    engine_config: TransferEngineConfig
    num_engines: int = 1
    zmq_bind_host: str = "0.0.0.0"


class TransferStatus(IntEnum):
    SUCCESS = 0
    FAILURE = 1


@dataclass
class ReceiverInfo:
    session_ids: List[str]
    buffer_ptr: int
    buffer_length: int
    zmq_endpoint: str
    zmq_port: int
    sglang_http_host: str
    sglang_http_port: int
    handshake_ports: List[int]
    sender_group_index: int


def get_node_ips() -> List[str]:
    try:
        import psutil
        all_interfaces = psutil.net_if_addrs()
        ips = []
        for interface, addrs in all_interfaces.items():
            if interface != 'lo':
                for addr in addrs:
                    if addr.family == socket.AF_INET:
                        ips.append(addr.address)
        return ips
    except Exception:
        try:
            hostname = socket.gethostname()
            return [socket.gethostbyname(hostname)]
        except Exception:
            return []


def filter_ips_by_config(all_ips: List[str], allowed_ips_config: str) -> List[str]:
    if allowed_ips_config == "0.0.0.0/0":
        return all_ips
    allowed_patterns = [s.strip() for s in allowed_ips_config.split(',')]
    filtered_ips = []
    for ip in all_ips:
        try:
            ip_obj = ipaddress.ip_address(ip)
            for pattern in allowed_patterns:
                if '/' in pattern:
                    if ip_obj in ipaddress.ip_network(pattern, strict=False):
                        filtered_ips.append(ip)
                        break
                elif ip == pattern:
                    filtered_ips.append(ip)
                    break
        except Exception:
            continue
    return filtered_ips
