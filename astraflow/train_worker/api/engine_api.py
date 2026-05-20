from __future__ import annotations

import abc
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from astraflow.train_worker.api.alloc_mode import ParallelStrategy
from astraflow.train_worker.api.io_struct import (
    LocalInfServerInfo,
    ModelRequest,
    ModelResponse,
    SaveLoadMeta,
    WeightUpdateMeta,
)

if TYPE_CHECKING:
    from astraflow.train_worker.utils.data import MicroBatchList
    from astraflow.core.workflow.api.workflow_api import RolloutWorkflow


class TrainEngine(abc.ABC):
    @abc.abstractmethod
    def create_process_group(self, parallel_strategy: ParallelStrategy | None = None):
        """Initialize PyTorch distributed communication groups.

        Parameters
        ----------
        parallel_strategy : ParallelStrategy, optional
            The parallel strategy configuration for distributed training, by default None
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def initialize(self, *args, **kwargs):
        """Initialize environments for distributed training and load models.

        This method should be called after `create_process_group`.

        Parameters
        ----------
        *args
            Variable length argument list
        **kwargs
            Arbitrary keyword arguments
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def data_parallel_group(self) -> dist.ProcessGroup:
        """Get the data parallel communication group of this engine.

        Returns
        -------
        dist.ProcessGroup
            The data parallel communication group
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def data_parallel_rank(self) -> int:
        """Get the rank of the current process in the data parallel group.

        Returns
        -------
        int
            The rank of the current process in the data parallel group
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def data_parallel_world_size(self) -> int:
        """Get the world size of the data parallel group.

        Returns
        -------
        int
            The world size of the data parallel group
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def current_data_parallel_head(self) -> int:
        """Get the current data parallel head rank.

        Returns
        -------
        int
            The rank of the current data parallel head
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def is_data_parallel_head(self) -> bool:
        """Check if the current rank is the data parallel head of the current engine.

        Returns
        -------
        bool
            True if the current rank is the data parallel head, False otherwise
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def context_and_model_parallel_group(self) -> dist.ProcessGroup:
        """Get the context and model parallel communication group of this engine.

        Returns
        -------
        dist.ProcessGroup
            The context and model parallel communication group
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def cpu_group(self) -> dist.ProcessGroup:
        """Get the CPU communication group of this engine.

        Returns
        -------
        dist.ProcessGroup
            The CPU communication group
        """
        raise NotImplementedError()

    def destroy(self):
        """Destroy the engine and release GPU memory of models."""

    @abc.abstractmethod
    def train(self, mode: bool = True):
        """Set the engine to training mode.

        Parameters
        ----------
        mode : bool, optional
            Whether to set the engine to training mode, by default True
        """
        raise NotImplementedError()

    def eval(self):
        """Set the engine to evaluation mode.

        This is a convenience method that calls `self.train(False)`.
        """
        return self.train(False)

    @abc.abstractmethod
    def update_weights(self, meta: WeightUpdateMeta):
        """Update weights to the inference engine in a blocking manner.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def connect_engine(self, engine: InferenceEngine, meta: WeightUpdateMeta):
        """Connect to an inference engine for online training.

        Parameters
        ----------
        engine : InferenceEngine
            The inference engine to connect to
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set_version(self, version: int):
        """Set the current weight version in the training engine.

        Parameters
        ----------
        version : int
            The weight version number to set
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_version(self) -> int:
        """Get the current weight version in the training engine.

        Returns
        -------
        int
            The current weight version number
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def save(self, meta: SaveLoadMeta):
        """Save model weights and optimizer states for later use.

        Parameters
        ----------
        meta : SaveLoadMeta
            Metadata containing information about where and how to save
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, meta: SaveLoadMeta):
        """Load model weights and optimizer states from a file.

        Parameters
        ----------
        meta : SaveLoadMeta
            Metadata containing information about where and how to load
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def optimizer_zero_grad(self):
        """Zero out all gradients in the optimizer."""
        raise NotImplementedError()

    @abc.abstractmethod
    def optimizer_step(self):
        """Perform a single optimization step.

        Returns
        -------
        dict[str, float]
            Training statistics containing ``update_successful``, ``grad_norm``, and ``lr``.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def lr_scheduler_step(self):
        """Advance the learning rate scheduler by one step."""
        raise NotImplementedError()

    def step_lr_scheduler(self):
        """This is an alias for `lr_scheduler_step()`."""
        return self.lr_scheduler_step()

    @abc.abstractmethod
    def forward_backward_batch(
        self,
        mb_list: MicroBatchList,
        process_output_fn: Callable[
            [torch.Tensor, dict[str, Any]], torch.Tensor | None
        ],
        forward_only: bool = False,
    ) -> None:
        """Process micro-batches through forward and optionally backward pass.

        Parameters
        ----------
        mb_list : MicroBatchList
            The micro-batch list, which is iterable and yields MicroBatchItem tuples.
        process_output_fn : Callable[[torch.Tensor, dict[str, Any]], torch.Tensor | None]
            A function that processes the model output (logits) and returns the loss tensor.
            If the returned loss is not None, backward() will be called on it.
            Results can be collected via closure if needed.
            Signature: ``(logits: Tensor, inputs: dict) -> loss | None``
        forward_only : bool, optional
            If True, skip backward pass. Default is False.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def train_batch(
        self,
        input_: dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> dict[str, float]:
        """Update the model with a batch of data and a loss function.

        Note
        ----
        The loss_fn should process packed 1D inputs, instead of 2D inputs.

        Parameters
        ----------
        input_ : dict[str, Any]
            The input data for model forward pass and the loss function.
            Redundant entries are allowed.
        loss_fn : Callable[..., torch.Tensor]
            The loss function. For actor (is_critic=False), it receives
            (logprobs, entropy, input_data). For critic (is_critic=True),
            it receives (values, input_data). Returns a scalar normalized loss.
        loss_weight_fn : Callable[[dict[str, Any]], torch.Tensor]
            A function used to calculate the weight of each micro-batch. Since
            loss_fn normalizes the loss for a micro-batch, we need a corresponding
            weight for each micro-batch to normalize the loss globally. The weight
            is usually the number of response tokens in the batch.

        Returns
        -------
        dict[str, float]
            Scalar statistics after training, e.g., the current learning rate,
            gradient norm, etc.
        """
        raise NotImplementedError()

    @torch.no_grad()
    @abc.abstractmethod
    def eval_batch(
        self,
        input_: dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> torch.Tensor | None:
        """Evaluate the model using the forward pass and loss function.

        Note
        ----
        The loss_fn should process packed 1D inputs, instead of 2D inputs.

        Parameters
        ----------
        input_ : dict[str, Any]
            The input data for model forward pass and the loss function.
            Redundant entries are allowed.
        loss_fn : Callable[..., torch.Tensor]
            The loss function. For actor (is_critic=False), it receives
            (logprobs, entropy, input_data). For critic (is_critic=True),
            it receives (values, input_data). Returns a scalar normalized loss.
        loss_weight_fn : Callable[[dict[str, Any]], torch.Tensor]
            A function used to calculate the weight of each micro-batch. Since
            loss_fn normalizes the loss for a micro-batch, we need a corresponding
            weight for each micro-batch to normalize the loss globally. The weight
            is usually the number of response tokens in the batch.

        Returns
        -------
        torch.Tensor or None
            A scalar loss or None. The evaluation statistics should be aggregated
            with `stats_tracker`.
        """
        raise NotImplementedError()

    @torch.no_grad()
    @abc.abstractmethod
    def forward_batch(
        self,
        input_: dict[str, Any],
        output_seqlens: list[int] | None = None,
        aggregate_fn: Callable[[list[Any]], Any] = torch.cat,
    ) -> torch.Tensor:
        """Run the forward pass or inference on the model.

        Note
        ----
        This operation is gradient-free.

        Parameters
        ----------
        input_ : dict[str, Any]
            The input data for model forward pass. Redundant entries are allowed.
        output_seqlens : list[int], optional
            The desired output sequence lengths. If None, assumes that the output
            has the same lengths as inputs, by default None.
        aggregate_fn : Callable[[list[Any]], Any], optional
            A function to aggregate micro-batched outputs, by default torch.cat.

        Returns
        -------
        Any
            For actor (is_critic=False): logprobs tensor aggregated by `aggregate_fn`.
            For critic (is_critic=True): values tensor aggregated by `aggregate_fn`.
        """
        raise NotImplementedError()

    @torch.no_grad()
    def forward(
        self,
        input_: dict[str, Any],
        output_seqlens: list[int] | None = None,
        aggregate_fn: Callable[[list[Any]], Any] = torch.cat,
    ) -> torch.Tensor:
        return self.forward_batch(input_, output_seqlens, aggregate_fn)

    @abc.abstractmethod
    def export_stats(self) -> dict[str, float]:
        """Export the statistics recorded in this engine process.

        Note
        ----
        Statistics will be all-reduced across the data parallel group
        and broadcasted from the last pipeline parallel stage.

        Returns
        -------
        dict[str, float]
            The exported scalar statistics.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def onload(self) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def offload(self) -> None:
        raise NotImplementedError()


class InferenceEngine(abc.ABC):
    def initialize(self, *args, **kwargs):
        """Initialize environments and launch the background thread for asynchronous distributed inference.

        For remote inference engines, this serves as a client and connects to the inference servers.
        For local inference engines, this creates an LLM engine on the local GPU.

        Parameters
        ----------
        *args
            Variable length argument list
        **kwargs
            Arbitrary keyword arguments
        """
        raise NotImplementedError()

    def destroy(self):
        """Destroy the engine and release GPU memory for the local inference engine."""
        raise NotImplementedError()

    def launch_server(self, server_args: dict[str, Any]) -> LocalInfServerInfo:
        """Launch a local inference server via subprocess and return its connection info.

        By default, an `InferenceEngine` instance acts as a client that connects to an existing
        remote inference server without occupying GPU resources. This is the typical usage in
        SPMD mode, where each training process has an attached inference client.

        This method enables launching a local inference server process, which is useful for:

        1. **Single-controller mode**: Launch a local server to serve the `InferenceEngine`
           instance with direct GPU worker control.

        2. **Standalone inference**: Use the inference engine in independent scripts or notebooks
           for running agentic workflows without managing separate server processes.

        Parameters
        ----------
        server_args : dict[str, Any]
            CLI arguments for the inference server (e.g., model path, GPU indices,
            port numbers, backend-specific settings)

        Returns
        -------
        LocalInfServerInfo
            Information about the launched server, including connection details and process metadata

        See Also
        --------
        teardown_server : Teardown the server launched by this method
        """
        raise NotImplementedError()

    def teardown_server(self):
        """Teardown the inference server launched by `launch_server`."""
        raise NotImplementedError()

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Asynchronously generate a response for the given request.

        Parameters
        ----------
        req : ModelRequest
            The model request containing input data and generation parameters

        Returns
        -------
        ModelResponse
            The generated response from the model
        """
        raise NotImplementedError()

    def set_version(self, version: int) -> None:
        """Set the current weight version in the inference engine.

        Parameters
        ----------
        version : int
            The weight version number to set
        """
        raise NotImplementedError()

    def get_version(self) -> int:
        """Get the current weight version in the inference engine.

        Returns
        -------
        int
            The current weight version number
        """
        raise NotImplementedError()

    def submit(
        self,
        data: dict[str, Any],
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str,
        should_accept_fn: Callable | None = None,
        workflow_kwargs: dict[str, Any] | None = None,
    ) -> int:
        """Submit a request to the inference engine and return immediately.

        Should be used together with subsequent `wait`.

        Parameters
        ----------
        data : dict[str, Any]
            The input data for rollout. Used by the user's customized workflow implementation.
        workflow : RolloutWorkflow | type[RolloutWorkflow] | str
            The workflow to use for rollout generation. Can be:

            - An instance of RolloutWorkflow (for sharing resources between rollouts)
            - A RolloutWorkflow class type (will be instantiated with workflow_kwargs)
            - A string module path like "astraflow.core.workflow.impl.rlvr.RLVRWorkflow" (will be imported
              and instantiated with workflow_kwargs)
        workflow_kwargs : dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor when workflow is a type or string.
            Required when workflow is a type or string, ignored when workflow is an instance.
            By default None.
        should_accept_fn : Callable, optional
            A function used to decide whether to accept a specific trajectory, i.e., dynamic filtering.
            It takes a complete trajectory output by the workflow, and returns a bool, by default None.

        Returns
        -------
        int
            The id assigned to this task
        """
        raise NotImplementedError()

    def wait(
        self, count: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> list[dict[str, Any] | None]:
        """Wait for a specified number of requests to complete, with a timeout.

        Should be used together with preceding `submit`.

        Parameters
        ----------
        count : int
            The number of accepted trajectories to wait for
        timeout : float, optional
            Timeout in seconds. Exceeding the timeout will raise a `TimeoutError`, by default None
        raise_timeout : bool, optional
            Whether to raise a `TimeoutError` when the timeout is exceeded,
            otherwise return an empty list, by default True

        Returns
        -------
        list[dict[str, Any] | None]
            A list of trajectory dictionaries. Each element may be None for rejected trajectories.

        Raises
        ------
        TimeoutError
            If the timeout is exceeded before enough trajectories are collected
        """
        raise NotImplementedError()

    def wait_for_task(
        self, task_id: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> dict[str, Any] | None:
        """Wait for a specific task to complete by task_id.

        Parameters
        ----------
        task_id : int
            The task ID returned by submit()
        timeout : float | None, optional
            Timeout in seconds, by default None
        raise_timeout : bool, optional
            Whether to raise TimeoutError on timeout, by default True

        Returns
        -------
        dict[str, Any] | None
            Trajectory dict, or None if rejected or timeout with raise_timeout=False

        Raises
        ------
        ValueError
            If task_id was never submitted or already consumed
        TimeoutError
            If timeout expires and raise_timeout=True
        """
        raise NotImplementedError()

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str,
        workflow_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit a batch of requests to the inference engine and wait for the results.

        This method does not support asynchronous rollout and should be used for offline
        data collection or debugging, not in production experiments.

        See `workflow_api.py` for concrete implementation.

        Parameters
        ----------
        data : list[dict[str, Any]]
            A list of input data dictionaries for rollout
        workflow : RolloutWorkflow | type[RolloutWorkflow] | str
            The workflow to use for rollout generation. Can be:

            - An instance of RolloutWorkflow (for sharing resources between rollouts)
            - A RolloutWorkflow class type (will be instantiated with workflow_kwargs)
            - A string module path like "astraflow.core.workflow.impl.rlvr.RLVRWorkflow" (will be imported
              and instantiated with workflow_kwargs)
        workflow_kwargs : dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor when workflow is a type or string.
            Required when workflow is a type or string, ignored when workflow is an instance.
            By default None.

        Returns
        -------
        dict[str, Any]
            A concatenated batch of trajectory results
        """
        raise NotImplementedError()

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable | None = None,
    ) -> dict[str, Any]:
        """Asynchronously submit and wait until a full batch is ready with controlled staleness.

        See `workflow_api.py` for concrete implementation.

        .. warning::

            This method caches an internal data generator on the first call.
            The ``dataloader``, ``workflow``, ``workflow_kwargs``, and
            ``should_accept_fn`` parameters are captured at the first invocation
            and reused in all subsequent calls. Passing different arguments in
            later calls will **not** take effect.

            If you need to switch configurations mid-training, consider:

            - Using a separate inference engine instance
            - Using the :meth:`submit` / :meth:`wait` pattern for finer control

        Parameters
        ----------
        dataloader : StatefulDataLoader
            The data loader to pull data from for batch preparation
        workflow : RolloutWorkflow | type[RolloutWorkflow] | str
            The workflow to use for rollout generation. Can be:

            - An instance of RolloutWorkflow (for sharing resources between rollouts)
            - A RolloutWorkflow class type (will be instantiated with workflow_kwargs)
            - A string module path like "astraflow.core.workflow.impl.rlvr.RLVRWorkflow" (will be imported
              and instantiated with workflow_kwargs)
        workflow_kwargs : dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor when workflow is a type or string.
            Required when workflow is a type or string, ignored when workflow is an instance.
            By default None.
        should_accept_fn : Callable, optional
            A function to decide whether to accept a trajectory, by default None

        Returns
        -------
        dict[str, Any]
            A full batch of trajectory results with controlled staleness
        """
        raise NotImplementedError()

    def pause_generation(self):
        """Pause the generation of inference engine.

        Used during updating weights from distributed or disk.
        """
        raise NotImplementedError()

    def continue_generation(self):
        """Continue the generation of inference engine."""
        raise NotImplementedError()

    def pause(self):
        """Pause request submission for async rollout.

        Used during evaluation to prevent data over-generation.
        """
        raise NotImplementedError()

    def resume(self):
        """Resume request submission for async rollout."""
        raise NotImplementedError()

    def is_paused(self) -> bool:
        """Check if the rollout engine is currently paused.

        Returns
        -------
        bool
            True if paused, False otherwise.
        """
        raise NotImplementedError()

    def offload(self):
        """Offload model from GPU to CPU for inference engine."""
        raise NotImplementedError()

    def onload(self, tags: list[str] | None = None):
        """Onload model from CPU to GPU for inference engine.

        Parameters
        ----------
        tags : list[str], optional
            Tags to onload specific components. If None, onloads all components.
        """
        raise NotImplementedError()

    def export_stats(self) -> dict[str, float]:
        """Export the statistics recorded during workflow execution in the process.

        Workflow should only record scalar metrics like "rewards".
        These metrics will be reduced in the controller side.

        Note
        ----
        This method should be only called by the controller.

        Returns
        -------
        dict[str, float]
            The recorded scalar statistics.
        """
        raise NotImplementedError()
