"""Shared transport layer for TCP-based weight transfer.

Used by both the trainer (sender side via WeightManager) and
RaaS (receiver side via RaaSWeightReceiver).
"""

from .config import (
    ReceiverInfo,
    SenderAgentConfig,
    TransferEngineConfig,
    TransferStatus,
)

__all__ = [
    "ReceiverInfo",
    "SenderAgentConfig",
    "TransferEngineConfig",
    "TransferStatus",
]
