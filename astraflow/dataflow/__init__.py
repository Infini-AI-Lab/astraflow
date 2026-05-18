"""AstraFlow data management components.

This package provides an isolated rollout-data management path that can run
independently from the current trainer wiring.  It also includes an HTTP
service layer (``astraflow.service``) for decoupled trainer orchestration.
"""

from .astraflow import AstraFlow
from .buffer_filters import (
    FILTER_REGISTRY,
    BufferFilterFn,
    FilterZeroAdvFilter,
    KeepAllFilter,
    get_filter,
)
from .data_acquisition import AstraDataAcquisition, DataAcquisition
from .data_serving import AstraDataServing, DataServing, MultiModelDataServing
from .eval_manager import EvalManager
from .raas2_engine import RaaS2InferenceEngine
from .replay_selectors import (
    REPLAY_SELECTION_REGISTRY,
    ReplaySelectionFn,
    get_replay_selection,
    select_latest,
)
from .rollout_buffer import RolloutBuffer
from .service import AstraFlowService, create_app
from .service_config import AgentConfig, EvalConfig, ServiceConfig

__all__ = [
    "AgentConfig",
    "AstraDataAcquisition",
    "AstraDataServing",
    "AstraFlow",
    "AstraFlowService",
    "BufferFilterFn",
    "DataAcquisition",
    "DataServing",
    "MultiModelDataServing",
    "EvalConfig",
    "EvalManager",
    "FILTER_REGISTRY",
    "FilterZeroAdvFilter",
    "KeepAllFilter",
    "REPLAY_SELECTION_REGISTRY",
    "ReplaySelectionFn",
    "RolloutBuffer",
    "RaaS2InferenceEngine",
    "ServiceConfig",
    "create_app",
    "get_filter",
    "get_replay_selection",
    "select_latest",
]
