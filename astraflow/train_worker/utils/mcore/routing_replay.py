"""Rollout Routing Replay (R3) for Megatron-Core MoE routers.

The trainer re-runs forward passes over sequences the rollout engine (SGLang)
generated. Tiny numerical differences between the two stacks can flip top-k
routing decisions, so the training-time logprobs diverge from the rollout-time
ones. R3 forces every trainer router to the top-k the rollout recorded:
``probs = softmax(logits.masked_fill(~mask, -inf))`` in fp32, which equals the
post-softmax top-k renormalization (HF ``norm_topk_prob=True``) and stays
differentiable through the gate.

Records are installed per micro-batch and per model chunk right before that
chunk's forward. Every layer keeps two cursors: activation recompute re-forwards
a layer during its backward, and 1F1B interleaves later forwards with earlier
backwards, so a single FIFO read twice would serve the wrong micro-batch.
Expert ids are logical (pre-EPLB). The packed tensor is
``[tokens, num_hidden_layers * top_k]``, viewed to ``[tokens, L, K]`` at install.
"""

from collections.abc import Sequence
from typing import Any

import torch


class RoutingReplayError(RuntimeError):
    """Raised on any routing-replay bookkeeping violation (fail-fast)."""


class RoutingReplayContext:
    """Per-layer routing records for one forward/backward pass."""

    def __init__(
        self, chunk_layer_maps: Sequence[dict[int, int]], num_layers: int, top_k: int
    ):
        # one {megatron layer_number: HF decoder-layer index} per local model chunk
        self.chunk_layer_maps = [dict(m) for m in chunk_layer_maps]
        self.num_layers = num_layers
        self.top_k = top_k
        self._auto_backward = False
        self._records: dict[int, list[torch.Tensor]] = {}
        self._fwd: dict[int, int] = {}
        self._bwd: dict[int, int] = {}

    def begin_pass(self, forward_only: bool) -> None:
        """Make this the active context for one ``forward_backward_func`` call."""
        global _REPLAY_CONTEXT
        self._records, self._fwd, self._bwd = {}, {}, {}
        # forward-only passes issue no recompute, so a second read is an error
        self._auto_backward = not forward_only
        _REPLAY_CONTEXT = self

    def end_pass(self) -> None:
        global _REPLAY_CONTEXT
        _REPLAY_CONTEXT = None
        self._records = {}

    def install_packed(self, routed_experts: torch.Tensor, chunk_index: int) -> None:
        """Install one micro-batch ``[tokens, num_layers * top_k]`` for one model chunk."""
        # the view raises unless the column count is num_layers * top_k; RaaS and the
        # trainer must serve the same checkpoint (a different (L, K) of equal product passes)
        routed_experts = routed_experts.view(
            routed_experts.shape[0], self.num_layers, self.top_k
        )
        for layer_number, hf_index in self.chunk_layer_maps[chunk_index].items():
            # a view of the resident micro-batch tensor, no copy
            self._records.setdefault(layer_number, []).append(
                routed_experts[:, hf_index, :]
            )

    def fetch(self, layer_number: int, num_rows: int) -> torch.Tensor:
        """Serve the next record for ``layer_number`` as int64 ``[num_rows, top_k]``."""
        records = self._records.get(layer_number, [])
        cursors, cursor = self._fwd, self._fwd.get(layer_number, 0)
        if cursor < len(records):
            # lockstep: install happens right before the forward that consumes it
            if cursor != len(records) - 1:
                raise RoutingReplayError(
                    f"megatron layer {layer_number}: {len(records)} records installed "
                    f"but forward consumed only {cursor}"
                )
        else:
            if not self._auto_backward:
                raise RoutingReplayError(
                    f"megatron layer {layer_number}: no unconsumed routing record"
                )
            # forward queue exhausted => activation-recompute re-forward
            cursors, cursor = self._bwd, self._bwd.get(layer_number, 0)
            if cursor >= len(records):
                raise RoutingReplayError(
                    f"megatron layer {layer_number}: no routing record left "
                    f"({len(records)} installed, backward cursor {cursor})"
                )
        chunk = records[cursor]
        if chunk.shape[0] != num_rows:
            raise RoutingReplayError(
                f"megatron layer {layer_number}: recorded {chunk.shape[0]} rows, "
                f"router got {num_rows}"
            )
        cursors[layer_number] = cursor + 1
        # scatter_ needs int64; widen only the chunk being served
        return chunk.long()

    def assert_all_consumed(self) -> None:
        """Every mapped layer read all its records once, plus once more if recomputed."""
        # only PP=1 (a single chunk map) has been exercised on GPU
        for chunk_map in self.chunk_layer_maps:
            for layer_number in chunk_map:
                n = len(self._records.get(layer_number, ()))
                fwd = self._fwd.get(layer_number, 0)
                bwd = self._bwd.get(layer_number, 0)
                if n == 0 or fwd != n or bwd not in (0, n):
                    raise RoutingReplayError(
                        f"megatron layer {layer_number}: {n} records, forward consumed "
                        f"{fwd}, backward {bwd}"
                    )


# plain module global, NOT threading.local: recompute re-forwards run on the
# autograd worker thread
_REPLAY_CONTEXT: RoutingReplayContext | None = None


def get_replay_context() -> RoutingReplayContext | None:
    return _REPLAY_CONTEXT


# masked-softmax replay equals megatron's topk_softmax_with_capacity only here
_SUPPORTED_ROUTER = {
    "moe_router_score_function": "softmax",
    "moe_router_pre_softmax": False,
    "moe_router_topk_scaling_factor": None,
    "moe_expert_capacity_factor": None,
    "moe_router_group_topk": None,
    "moe_router_enable_expert_bias": False,
    "moe_router_load_balancing_type": "none",
    "moe_z_loss_coeff": None,
}


def assert_router_config_supported(config: Any) -> None:
    bad = {
        k: getattr(config, k) for k, ok in _SUPPORTED_ROUTER.items() if getattr(config, k) != ok
    }
    if bad:
        raise RoutingReplayError(f"moe_router_replay needs {_SUPPORTED_ROUTER}, got {bad}")


def install_topk_router_patch() -> None:
    """Idempotent. The replay branch mirrors the pre-routing() part of
    TopKRouter.forward in megatron-core 0.13.1 (pinned in pyproject): expert-bias
    upkeep, jitter, gating. With an active context the recorded top-k is forced
    and probs are the fp32 masked softmax; z/aux-loss and expert-bias bookkeeping
    are skipped (assert_router_config_supported rejects configs needing them).
    """
    from megatron.core.transformer.moe.router import TopKRouter

    if getattr(TopKRouter, "_astraflow_routing_replay_patched", False):
        return
    original_forward = TopKRouter.forward

    def forward(self, input: torch.Tensor):
        context = get_replay_context()
        if context is None:
            return original_forward(self, input)
        self._maintain_float32_expert_bias()
        logits = self.gating(self.apply_input_jitter(input))
        logits = logits.view(-1, self.config.num_moe_experts)
        topk_ids = context.fetch(self.layer_number, logits.shape[0])
        routing_map = torch.zeros_like(logits, dtype=torch.bool).scatter_(
            1, topk_ids, True
        )
        probs = torch.softmax(
            logits.float().masked_fill(~routing_map, float("-inf")), dim=-1
        ).to(logits.dtype)
        return probs, routing_map

    TopKRouter.forward = forward
    TopKRouter._astraflow_routing_replay_patched = True
