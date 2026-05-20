from dataclasses import dataclass, field
from typing import List

from astraflow.core.weight_manager.transfer.config import SenderAgentConfig


@dataclass
class WeightManagerConfig:
    """Configuration for WeightManager.

    Parameters
    ----------
    sender_config : SenderAgentConfig
        Configuration for the sender agent subprocess.
    strategies : list[str]
        Supported transfer strategies. The sender prepares all listed
        modes; RaaS chooses which to pull. Options: ``"full"``, ``"delta"``.
    shm_prefix : str
        Prefix for shared memory file names in ``/dev/shm``.
    """

    sender_config: SenderAgentConfig
    strategies: List[str] = field(default_factory=lambda: ["full"])
    shm_prefix: str = "astraflow_buffer_"
