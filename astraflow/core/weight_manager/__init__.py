"""WeightManager — independent weight transfer component.

Top-level component for managing weight transfer between trainer and RaaS.
Both trainer and RaaS import from this package; neither imports from the other.
"""

from astraflow.core.weight_manager.config import WeightManagerConfig
from astraflow.core.weight_manager.weight_manager import WeightManager

__all__ = [
    "WeightManager",
    "WeightManagerConfig",
]
