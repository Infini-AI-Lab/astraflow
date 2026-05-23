"""Transfer buffer for TCP weight receiving.

The TransferBuffer class is used by RaaSWeightReceiver (in
astraflow/raas/server/tcp_receiver.py) to hold received weight data
before saving as safetensors.
"""

import logging
from typing import List, Tuple

import torch

logger = logging.getLogger(__name__)


class TransferBuffer:
    """CPU memory buffer for receiving weights via TCP."""

    def __init__(self, params: List[Tuple[str, torch.Tensor]]):
        num_bytes = sum(p[1].numel() * p[1].element_size() for p in params)
        self.buffer = torch.zeros(num_bytes, dtype=torch.uint8, device="cpu")
        # Share memory so the buffer pointer stays valid when sent via mp.Queue
        self.buffer.share_memory_()
        self.ptr = self.buffer.data_ptr()
        self.length = self.buffer.numel()
