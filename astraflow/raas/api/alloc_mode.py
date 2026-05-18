import enum
from dataclasses import dataclass, field
from typing import Literal

from astraflow.raas.utils import logging

logger = logging.getLogger(__name__)


class AllocationType(enum.Enum):
    """Backward Compatible: Type of resource allocation strategy."""

    COLOCATE = 0  # Shared resources between training and inference (including SFT/training-only)
    DECOUPLED_TRAIN = 1  # Separate resources for training and inference
    LLM_SERVER_ONLY = 2  # Inference-only allocation
    DECOUPLED_EVAL = 3  # Separate resources for inference and evaluation


class AllocationValidationError(Exception):
    """Raised when allocation mode validation fails."""


class InvalidAllocationModeError(Exception):
    """Legacy exception for backward compatibility with existing code."""


@dataclass
class SchedulingStrategy:
    """Resource scheduling type for allocation components.

    Parameters
    ----------
    type : str
        "separation" for independent resources, "colocation" for shared resources
    target : str, optional
        For colocation, name of anchor component to colocate with, by default None
    """

    type: str  # "separation" or "colocation"
    target: str | None = None


@dataclass
class ParallelStrategy:
    """5D parallel strategy supporting tensor, pipeline, data, context, and expert parallelism.

    This class represents a comprehensive parallelization strategy for distributed ML workloads,
    particularly designed for large language models and mixture-of-experts architectures.

    The five dimensions of parallelism are:
    - Tensor parallelism: Splits individual operations (like matrix multiplications) across devices
    - Pipeline parallelism: Splits model layers across devices in a pipeline fashion
    - Data parallelism: Replicates the model and splits data across devices
    - Context parallelism: Splits sequence length across devices (attention-specific)
    - Expert parallelism: Splits experts in MoE models across devices

    For implementation details, refer to:
    https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe#moe-parallel-folding

    Args:
        tensor_parallel_size: Number of devices for tensor model parallelism (default: 1)
        pipeline_parallel_size: Number of pipeline parallel stages (default: 1)
        data_parallel_size: Number of data parallel replicas for ZeRO optimization (default: 1)
        context_parallel_size: Number of devices for context parallelism in attention modules (default: 1)
        expert_parallel_size: Number of devices for expert parallelism in MoE models (default: 1)
        expert_tensor_parallel_size: Tensor parallelism size specifically for expert modules (default: 1)

    Note:
        - Context parallelism is only effective for attention modules
        - Expert parallelism is only effective for MoE (Mixture of Experts) modules
    """

    tensor_parallel_size: int = field(
        default=1, metadata={"help": "Size of tensor-model parallelism"}
    )
    pipeline_parallel_size: int = field(
        default=1, metadata={"help": "Number of pipeline parallel stages"}
    )
    data_parallel_size: int = field(
        default=1, metadata={"help": "Data parallelism size for ZeRO optimization"}
    )
    context_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Context parallelism size for attention modules. "
            "Note that context parallelism is only effective for attention modules."
        },
    )
    expert_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Expert parallelism size for MoE models. "
            "Note that expert parallelism is only effective for expert modules."
        },
    )
    expert_tensor_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Tensor parallelism size for expert modules. "
            "By default, it is 1 which disables expert tensor parallelism."
        },
    )

    def __post_init__(self):
        """Initialize computed properties and validate configuration."""
        if self.expert_parallel_size > 1:
            # Calculate expert model parallel size for validation
            self.expert_model_parallel_size = (
                self.pipeline_parallel_size
                * self.expert_tensor_parallel_size
                * self.expert_parallel_size
            )

            # Validate that world size is divisible by expert model parallel size
            assert self.world_size % self.expert_model_parallel_size == 0, (
                f"Expert model parallel size {self.expert_model_parallel_size} "
                f"cannot divide world size {self.world_size}."
            )

    @property
    def expert_data_parallel_size(self) -> int:
        """Data parallelism size for expert modules in MoE models."""
        if not hasattr(self, "expert_model_parallel_size"):
            return self.data_parallel_size
        return self.world_size // self.expert_model_parallel_size

    # Abbreviated properties for convenience
    @property
    def tp_size(self) -> int:
        """Tensor parallelism size (abbreviated)."""
        return self.tensor_parallel_size

    @property
    def pp_size(self) -> int:
        """Pipeline parallelism size (abbreviated)."""
        return self.pipeline_parallel_size

    @property
    def dp_size(self) -> int:
        """Data parallelism size (abbreviated)."""
        return self.data_parallel_size

    @property
    def cp_size(self) -> int:
        """Context parallelism size (abbreviated)."""
        return self.context_parallel_size

    @property
    def ep_size(self) -> int:
        """Expert parallelism size (abbreviated)."""
        return self.expert_parallel_size

    @property
    def etp_size(self) -> int:
        """Expert tensor parallelism size (abbreviated)."""
        return self.expert_tensor_parallel_size

    @property
    def edp_size(self) -> int:
        """Expert data parallelism size (abbreviated)."""
        return self.expert_data_parallel_size

    @property
    def world_size(self) -> int:
        """Total number of devices required for this parallelization strategy."""
        return (
            self.data_parallel_size
            * self.context_parallel_size
            * self.tensor_parallel_size
            * self.pipeline_parallel_size
        )

    def __str__(self):
        """String representation showing all non-default parallelism dimensions."""
        parts = [
            f"tp={self.tensor_parallel_size}",
            f"pp={self.pipeline_parallel_size}",
            f"dp={self.data_parallel_size}",
        ]

        if self.context_parallel_size > 1:
            parts.append(f"cp={self.context_parallel_size}")
        if self.expert_parallel_size > 1:
            parts.append(f"ep={self.expert_parallel_size}")
            if self.expert_tensor_parallel_size != 1:
                parts.append(f"ep_tp={self.expert_tensor_parallel_size}")

        return f"Parallel({','.join(parts)})"

    @staticmethod
    def parallelism_eq(this, other):
        """Compare two parallelism configurations for equality."""
        return (
            (this.tensor_parallel_size == other.tensor_parallel_size)
            and (this.pipeline_parallel_size == other.pipeline_parallel_size)
            and (this.data_parallel_size == other.data_parallel_size)
            and (this.context_parallel_size == other.context_parallel_size)
            and (this.expert_parallel_size == other.expert_parallel_size)
            and (this.expert_tensor_parallel_size == other.expert_tensor_parallel_size)
        )


@dataclass
class FSDPParallelStrategy(ParallelStrategy):
    """FSDP parallel strategy."""

    @staticmethod
    def parallelism_eq(this, other):
        """Compare FSDP parallelism configurations."""
        return ParallelStrategy.parallelism_eq(this, other)


@dataclass
class MegatronParallelStrategy(ParallelStrategy):
    """Megatron parallel strategy with additional sequence parallelism and virtual pipeline parallelism."""

    virtual_pipeline_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Virtual pipeline parallelism size for megatron modules "
            "for interleaved pipeline schedule. Default value is 1 (disabled)."
        },
    )
    use_sequence_parallel: bool = field(
        default=False,
        metadata={
            "help": "Enable sequence parallelism. Only used with tensor-model parallelism in Megatron",
        },
    )

    def __post_init__(self):
        super().__post_init__()
        vpp = self.virtual_pipeline_parallel_size
        if vpp <= 1:
            self.virtual_pipeline_parallel_size = 1
        elif self.pipeline_parallel_size <= 1:
            raise AllocationValidationError(
                "Virtual pipeline parallelism requires pipeline_parallel_size > 1."
            )

    @staticmethod
    def parallelism_eq(this, other):
        """Compare Megatron parallelism configurations (excluding sequence parallelism)."""
        return ParallelStrategy.parallelism_eq(this, other) and (
            this.virtual_pipeline_parallel_size == other.virtual_pipeline_parallel_size
        )


@dataclass
class ModelAllocation:
    """Single model allocation with backend, name, parallel strategy, and scheduling."""

    backend: Literal["fsdp", "megatron", "vllm", "sglang", "cpu"]
    name: str | None
    parallel: ParallelStrategy | None
    scheduling_strategy: SchedulingStrategy
    _backend_explicit: bool = field(default=True, repr=False)

    def __post_init__(self):
        if self.backend is None:
            if (
                self.parallel.pipeline_parallel_size > 1
                or self.parallel.expert_parallel_size > 1
            ):
                logger.info(
                    "Auto-selecting megatron backend for pipeline/expert parallelism."
                )
                self.backend = "megatron"
            else:
                logger.info("Auto-selecting fsdp training backend.")
                self.backend = "fsdp"

        if self.backend == "fsdp":
            if (
                self.parallel.pipeline_parallel_size > 1
                or self.parallel.expert_parallel_size > 1
            ):
                raise AllocationValidationError(
                    f"FSDP backend only supports data/tensor/context parallelism. "
                    f"Got strategy: {self.parallel}"
                )

    @property
    def world_size(self):
        if self.scheduling_strategy.type == "colocation":
            return 0
        return self.parallel.world_size

    def __str__(self):
        dims = []
        if self.parallel.data_parallel_size != 1:
            dims.append(f"d{self.parallel.data_parallel_size}")
        if self.parallel.pipeline_parallel_size != 1:
            dims.append(f"p{self.parallel.pipeline_parallel_size}")
        if self.parallel.tensor_parallel_size != 1:
            dims.append(f"t{self.parallel.tensor_parallel_size}")
        if self.parallel.context_parallel_size != 1:
            dims.append(f"c{self.parallel.context_parallel_size}")
        if self.parallel.expert_parallel_size != 1:
            dims.append(f"e{self.parallel.expert_parallel_size}")

        if not dims:
            dims.append(f"d{self.parallel.data_parallel_size}")

        result = "".join(dims)
        if self._backend_explicit:
            if self.name:
                result = f"{self.backend}({self.name}):{result}"
            else:
                result = f"{self.backend}:{result}"
        elif self.name:
            result = f"({self.name}):{result}"
        return result


@dataclass
class AllocationMode:
    """Resource allocation configuration for distributed ML workloads.

    Manages allocation of GPUs across multiple models/components with support for
    named components, colocation, and flexible parallelization strategies.
    """

    allocations: list[ModelAllocation] = field(default_factory=list)

    @classmethod
    def from_engine_config(cls, engine: dict) -> "AllocationMode":
        """Build AllocationMode from a structured engine config dict.

        Supports two formats:

        Flat (single component)::

            {"backend": "fsdp", "data_parallel_size": 4}

        Per-model (multi-component)::

            {"model0": {"backend": "sglang", "data_parallel_size": 2},
             "model1": {"backend": "sglang", "data_parallel_size": 2}}
        """

        def _build_alloc(spec: dict, name: str | None = None) -> ModelAllocation:
            backend = spec.get("backend", "fsdp")
            parallel = ParallelStrategy(
                data_parallel_size=spec.get("data_parallel_size", 1),
                tensor_parallel_size=spec.get("tensor_parallel_size", 1),
                pipeline_parallel_size=spec.get("pipeline_parallel_size", 1),
                context_parallel_size=spec.get("context_parallel_size", 1),
                expert_parallel_size=spec.get("expert_parallel_size", 1),
            )
            return ModelAllocation(
                backend=backend,
                name=name,
                parallel=parallel,
                scheduling_strategy=SchedulingStrategy(type="separation"),
            )

        # Flat format: engine has "backend" key
        if "backend" in engine:
            return cls(allocations=[_build_alloc(engine)])

        # Per-model format: each key is a model_id
        allocations = [
            _build_alloc(spec, name=mid)
            for mid, spec in engine.items()
        ]
        return cls(allocations=allocations)

    @classmethod
    def resolve(cls, raw: "dict | AllocationMode") -> "AllocationMode":
        """Resolve an engine config dict or AllocationMode to an AllocationMode object."""
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, dict):
            return cls.from_engine_config(raw)
        raise TypeError(f"Expected dict or AllocationMode, got {type(raw)}")

    def __getitem__(self, name: str) -> ModelAllocation:
        """Get allocation by name."""
        for alloc in self.allocations:
            if alloc.name == name:
                return alloc
        raise KeyError(f"No allocation found with name: {name}")

    @property
    def world_size(self):
        return sum(alloc.world_size for alloc in self.allocations)

    def _get_inference_allocations(self) -> list[ModelAllocation]:
        """Get all inference allocations (sglang, vllm backends)."""
        return [a for a in self.allocations if a.backend in ("sglang", "vllm")]

    def _get_training_allocations(self) -> list[ModelAllocation]:
        """Get all training allocations (fsdp, megatron backends)."""
        return [a for a in self.allocations if a.backend in ("fsdp", "megatron")]

    ########### Legacy Attributes for Backward Compatiblity ###########
    @property
    def type_(self) -> AllocationType:
        """Backward compatible: Check if any allocation uses eval backend (cpu or eval)."""
        if len(self.allocations) not in [1, 2]:
            raise AttributeError(
                "Can only infer allocation type from 1 or 2 allocations."
            )

        if len(self.allocations) == 1:
            if self.allocations[0].backend in ("sglang", "vllm"):
                return AllocationType.LLM_SERVER_ONLY
            return AllocationType.COLOCATE

        for alloc in self.allocations:
            if alloc.backend == "cpu":
                return AllocationType.DECOUPLED_EVAL

        inf_alloc = self._get_inference_allocations()
        train_alloc = self._get_training_allocations()
        if not (len(inf_alloc) == 1 and len(train_alloc) == 1):
            raise AttributeError(
                "Ambiguous allocation type: expected one inference and one training allocation."
            )
        if (
            inf_alloc[0].scheduling_strategy.type == "separation"
            and train_alloc[0].scheduling_strategy.type == "separation"
        ):
            return AllocationType.DECOUPLED_TRAIN
        return AllocationType.COLOCATE

    @property
    def gen(self) -> ParallelStrategy:
        """Backward compatible: returns parallel strategy for single inference allocation."""
        inf_allocs = self._get_inference_allocations()
        if len(inf_allocs) == 0:
            return None
        if len(inf_allocs) > 1:
            raise AttributeError(
                f"Ambiguous 'gen' property: found {len(inf_allocs)} inference allocations. "
                f"Use allocation_mode[name] or allocation_mode.allocations instead."
            )
        return inf_allocs[0].parallel

    @property
    def train(self) -> ParallelStrategy | None:
        """Backward compatible: returns parallel strategy for single training allocation."""
        train_allocs = self._get_training_allocations()
        if len(train_allocs) == 0:
            return None
        if len(train_allocs) > 1:
            raise AttributeError(
                f"Ambiguous 'train' property: found {len(train_allocs)} training allocations. "
                f"Use allocation_mode[name] or allocation_mode.allocations instead."
            )
        return train_allocs[0].parallel

    @property
    def gen_backend(self) -> str | None:
        """Backward compatible: returns backend for single inference allocation."""
        inf_allocs = self._get_inference_allocations()
        if len(inf_allocs) == 0:
            return None
        if len(inf_allocs) > 1:
            raise AttributeError(
                f"Ambiguous 'gen_backend' property: found {len(inf_allocs)} inference allocations. "
                f"Use allocation_mode[name].backend or allocation_mode.allocations instead."
            )
        return inf_allocs[0].backend

    @property
    def train_backend(self) -> str | None:
        """Backward compatible: returns backend for single training allocation."""
        train_allocs = self._get_training_allocations()
        if len(train_allocs) == 0:
            return None
        if len(train_allocs) > 1:
            raise AttributeError(
                f"Ambiguous 'train_backend' property: found {len(train_allocs)} training allocations. "
                f"Use allocation_mode[name].backend or allocation_mode.allocations instead."
            )
        return train_allocs[0].backend

    @property
    def gen_instance_size(self) -> int:
        """Backward compatible: returns instance size for single inference allocation."""
        inf_allocs = self._get_inference_allocations()
        if len(inf_allocs) == 0:
            raise AttributeError("No inference allocations found")
        if len(inf_allocs) > 1:
            raise AttributeError(
                f"Ambiguous 'gen_instance_size' property: found {len(inf_allocs)} inference allocations. "
                f"Use allocation_mode[name].parallel.tp_size * pp_size instead."
            )
        return inf_allocs[0].parallel.tp_size * inf_allocs[0].parallel.pp_size
