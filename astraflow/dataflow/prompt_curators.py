"""Prompt curators for selective rollout.

A ``PromptCurator`` decides per-prompt whether to roll it out
(``should_submit``) and learns from the outcome (``update``). It runs on
``DataAcquisition`` and is opt-in via ``AgentConfig.curator``.

Default behavior (no curator configured) is unchanged from before this
module existed — every prompt is submitted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RolloutOutcome:
    """Digested rollout result handed to ``PromptCurator.update``.

    All fields are computed inside ``_ingest_structured_result`` already;
    the curator only consumes scalars / a small reward tensor, never the
    full RaaS payload.
    """

    query_id: str
    rewards: torch.Tensor          # shape [n_trajs]
    g_mean: float
    g_std: float
    zero_adv: int                  # 1 if all rewards equal
    n_trajs: int
    version: int                   # weight version that produced this rollout
    source: str | None = None      # dataset tag


class PromptCurator(Protocol):
    """Per-prompt selective-rollout policy with closed-loop feedback."""

    def should_submit(self, data: dict[str, Any], *, version: int) -> bool:
        """Return True to submit this prompt, False to drop it."""

    def update(self, outcome: RolloutOutcome) -> None:
        """Learn from one completed rollout."""

    def on_version_changed(self, version: int) -> None:
        """Optional. Called when weights change. Default: no-op."""

    def notify_warmup_complete(self) -> None:
        """Optional. Orchestrator-driven signal: the warmup epoch is over.

        Called once by ``DataAcquisition`` after one full dataloader epoch
        of post-pre-fill samples has been emitted. Adaptive curators
        (e.g. ``GRESOCurator``) should start updating their tunable state
        here. Default: no-op. Must be idempotent — the dataflow may
        re-fire it after a checkpoint restore.
        """

    # state_dict / load_state_dict are optional. Implement them on
    # curators with non-trivial state so it survives buffer checkpoint
    # restores; framework checks hasattr before calling.


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CURATOR_REGISTRY: dict[str, type] = {}


def register_curator(name: str):
    """Class decorator that registers a curator implementation by name."""

    def _wrap(cls):
        if name in CURATOR_REGISTRY:
            raise ValueError(f"Curator already registered: {name!r}")
        CURATOR_REGISTRY[name] = cls
        return cls

    return _wrap


def get_curator(name: str, **kwargs) -> PromptCurator:
    """Construct a curator by registry name with the provided kwargs."""
    if name not in CURATOR_REGISTRY:
        raise ValueError(
            f"Unknown curator: {name!r}. "
            f"Registered: {sorted(CURATOR_REGISTRY)}"
        )
    return CURATOR_REGISTRY[name](**kwargs)


# ---------------------------------------------------------------------------
# Built-ins
# ---------------------------------------------------------------------------


@register_curator("accept_all")
class AcceptAllCurator:
    """Identity curator. Submits every prompt, learns nothing.

    Useful for testing the curator code path without changing behavior.
    """

    def should_submit(self, data, *, version):
        return True

    def update(self, outcome):
        pass

    def on_version_changed(self, version):
        pass


@register_curator("filter_solved")
class FilterSolvedCurator:
    """Stop submitting prompts once their EWMA success rate exceeds a threshold.

    Per-prompt success rate is the fraction of trajectories whose reward
    is positive. Updated as an exponential moving average on each
    rollout outcome.

    Parameters
    ----------
    threshold : float
        Submit only if EWMA success rate is below this. Default 0.95.
    decay : float
        EWMA weight on prior estimate. Default 0.9 (i.e. new sample has
        weight 0.1).
    prior : float
        Initial estimate for unseen prompts. Default 0.0 (assume
        unsolved → submit).
    """

    def __init__(
        self,
        threshold: float = 0.95,
        decay: float = 0.9,
        prior: float = 0.0,
    ):
        self._success_rate: dict[str, float] = {}
        self._threshold = float(threshold)
        self._decay = float(decay)
        self._prior = float(prior)

    def should_submit(self, data, *, version):
        qid = data.get("query_id")
        if qid is None:
            return True  # no id → can't track → submit
        rate = self._success_rate.get(qid, self._prior)
        return rate < self._threshold

    def update(self, outcome):
        qid = outcome.query_id
        if not qid:
            return
        success = float((outcome.rewards > 0).float().mean().item())
        prev = self._success_rate.get(qid, self._prior)
        self._success_rate[qid] = self._decay * prev + (1.0 - self._decay) * success

    def on_version_changed(self, version):
        # Success rate is a long-run statistic; keep across version bumps.
        pass

    def state_dict(self) -> dict:
        return {"success_rate": dict(self._success_rate)}

    def load_state_dict(self, state: dict) -> None:
        self._success_rate = dict(state.get("success_rate", {}))


@register_curator("greso")
class GRESOCurator:
    """GRESO: pre-rollout zero-variance filtering with self-adjusting exploration.

    Reference: Zheng et al., "Act Only When It Pays: Efficient Reinforcement
    Learning for LLM Reasoning via Selective Rollouts" (arXiv:2506.02177).

    Per prompt ``x_i`` tracks the most-recent consecutive zero-variance
    streak ``z_i``. Each prompt is submitted with probability
    ``p_e^{z_i}`` (equivalently filtered with prob ``1 - p_e^{z_i}``).
    Two base exploration probabilities are maintained, ``p_easy`` (for
    all-correct zero-variance) and ``p_hard`` (for all-wrong), and both
    self-adjust at each weight update toward target zero-variance
    ratios via ±``delta_p`` steps.

    Parameters
    ----------
    p_easy_init, p_hard_init : float
        Initial base exploration probabilities. Default 0.5 each.
    alpha_easy, alpha_hard : float
        Target zero-variance ratios per iteration. Defaults match the
        paper: 25% total split 1:2 → 8.3% easy, 16.7% hard.
    delta_p : float
        Step size for adaptive adjustment. Default 0.01 (paper).
    p_min, p_max : float
        Clamp range for ``p_easy`` / ``p_hard`` so neither degenerates.
    min_submit_easy, min_submit_hard : float
        Floor on the final submit probability per prompt: even at long
        streaks, a prompt is submitted with at least this probability.
        Easy streaks (all-correct, low priority) get a low floor; hard
        streaks (all-wrong, valuable to revisit) get a higher floor.
        Defaults: easy 0.10, hard 0.40.
    correct_threshold : float
        Mean reward strictly above this classifies a zero-variance group
        as ``easy`` (all-correct); else ``hard`` (all-wrong). Default
        0.5, suitable for binary RLVR rewards.

    Warmup
    ------
    ``p_easy`` / ``p_hard`` are held fixed at their inits until the
    orchestrator calls :meth:`notify_warmup_complete`. ``DataAcquisition``
    fires that signal after one full dataloader-epoch worth of
    post-pre-fill samples has been emitted (epoch-1 boundary). Filtering
    itself runs from step 1 — every prompt starts at ``z=0`` and is
    submitted unconditionally, so curated rejection only kicks in once
    streaks form.
    """

    def __init__(
        self,
        p_easy_init: float = 0.5,
        p_hard_init: float = 0.5,
        alpha_easy: float = 0.083,
        alpha_hard: float = 0.167,
        delta_p: float = 0.01,
        p_min: float = 0.01,
        p_max: float = 0.99,
        min_submit_easy: float = 0.10,
        min_submit_hard: float = 0.40,
        correct_threshold: float = 0.5,
    ):
        import random as _random

        self._streak: dict[str, int] = {}
        self._kind: dict[str, str] = {}  # "easy" | "hard", only set while streak > 0
        self._p_easy = float(p_easy_init)
        self._p_hard = float(p_hard_init)
        self._alpha_easy = float(alpha_easy)
        self._alpha_hard = float(alpha_hard)
        self._delta_p = float(delta_p)
        self._p_min = float(p_min)
        self._p_max = float(p_max)
        self._min_submit_easy = float(min_submit_easy)
        self._min_submit_hard = float(min_submit_hard)
        self._correct_threshold = float(correct_threshold)
        # Hold p_easy / p_hard fixed until the orchestrator calls
        # notify_warmup_complete().
        self._adjustment_armed: bool = False
        # Per-iteration tallies, reset on each on_version_changed call.
        self._n_easy = 0
        self._n_hard = 0
        self._n_total = 0
        # Snapshot of the last completed window's ratios — exposed via
        # get_telemetry so wandb sees the most recent observation between
        # version bumps. Zero until the first on_version_changed.
        self._last_easy_ratio = 0.0
        self._last_hard_ratio = 0.0
        self._rng = _random.Random()
        # Throttled-logging counters (free-debug instrumentation).
        # should_submit fires thousands of times per training step, so we
        # batch a one-line summary every SUBMIT_LOG_EVERY decisions.
        self._dbg_submit_calls = 0
        self._dbg_submit_easy_acc = 0
        self._dbg_submit_easy_rej = 0
        self._dbg_submit_hard_acc = 0
        self._dbg_submit_hard_rej = 0
        self._dbg_submit_z0_acc = 0
        self._dbg_submit_log_every = 500

    def should_submit(self, data, *, version):
        # Resolve qid via the shared helper so the streak table built by
        # update() (keyed on the workflow-stamped prompt_id) matches the
        # lookup here. Both sides MUST go through resolve_prompt_id.
        from astraflow.workflow.utils.data import resolve_prompt_id

        qid = resolve_prompt_id(data)
        if qid is None:
            return True
        z = self._streak.get(qid, 0)
        if z == 0:
            self._dbg_submit_z0_acc += 1
            self._dbg_submit_calls += 1
            self._maybe_log_submit_summary()
            return True  # p_f = 0; always submit
        kind = self._kind.get(qid, "easy")
        p_e = self._p_easy if kind == "easy" else self._p_hard
        # Submit probability = p_e^z, clamped to a per-kind floor so even
        # very long streaks keep some forced exploration.
        floor = self._min_submit_easy if kind == "easy" else self._min_submit_hard
        p_submit = max(p_e ** z, floor)
        accepted = self._rng.random() < p_submit
        if kind == "easy":
            if accepted:
                self._dbg_submit_easy_acc += 1
            else:
                self._dbg_submit_easy_rej += 1
        else:
            if accepted:
                self._dbg_submit_hard_acc += 1
            else:
                self._dbg_submit_hard_rej += 1
        self._dbg_submit_calls += 1
        self._maybe_log_submit_summary()
        return accepted

    def _maybe_log_submit_summary(self) -> None:
        if self._dbg_submit_calls < self._dbg_submit_log_every:
            return
        n = self._dbg_submit_calls
        ze = self._dbg_submit_z0_acc
        ea, er = self._dbg_submit_easy_acc, self._dbg_submit_easy_rej
        ha, hr = self._dbg_submit_hard_acc, self._dbg_submit_hard_rej
        n_streaked = len(self._streak)
        n_easy_streak = sum(1 for k in self._kind.values() if k == "easy")
        n_hard_streak = sum(1 for k in self._kind.values() if k == "hard")
        logger.info(
            "[GRESO submit %d] z=0:%d  easy(acc/rej)=%d/%d  hard(acc/rej)=%d/%d  "
            "p_e=%.3f p_h=%.3f floor_e=%.2f floor_h=%.2f  "
            "table: %d prompts (easy=%d hard=%d) armed=%s",
            n,
            ze,
            ea,
            er,
            ha,
            hr,
            self._p_easy,
            self._p_hard,
            self._min_submit_easy,
            self._min_submit_hard,
            n_streaked,
            n_easy_streak,
            n_hard_streak,
            self._adjustment_armed,
        )
        self._dbg_submit_calls = 0
        self._dbg_submit_easy_acc = 0
        self._dbg_submit_easy_rej = 0
        self._dbg_submit_hard_acc = 0
        self._dbg_submit_hard_rej = 0
        self._dbg_submit_z0_acc = 0

    def notify_warmup_complete(self) -> None:
        """Arm the adaptive controller. Idempotent.

        Called by ``DataAcquisition`` at the end of the warmup epoch.
        After this call, ``on_version_changed`` may adjust
        ``p_easy``/``p_hard``; before it, those stay at their inits.
        """
        if not self._adjustment_armed:
            self._adjustment_armed = True
            logger.info(
                "[GRESO] adjustment_armed=True (warmup epoch complete)"
            )

    def update(self, outcome):
        qid = outcome.query_id
        if not qid or outcome.rewards.numel() == 0:
            return
        self._n_total += 1
        prev_z = self._streak.get(qid, 0)
        prev_kind = self._kind.get(qid)
        if outcome.zero_adv == 1:
            mean_r = float(outcome.rewards.mean().item())
            kind = "easy" if mean_r > self._correct_threshold else "hard"
            prev = self._streak.get(qid, 0)
            # If streak kind flipped (rare), restart streak from 1.
            if self._kind.get(qid) != kind:
                prev = 0
            self._streak[qid] = prev + 1
            self._kind[qid] = kind
            if kind == "easy":
                self._n_easy += 1
            else:
                self._n_hard += 1
            logger.info(
                "[GRESO update] qid=%s zero_adv=1 kind=%s mean_r=%.3f "
                "streak: %d->%d (prev_kind=%s) reward=%s",
                qid,
                kind,
                mean_r,
                prev_z,
                self._streak[qid],
                prev_kind,
                outcome.rewards.tolist(),
            )
        else:
            # Effective rollout — variance recovered, reset streak.
            self._streak[qid] = 0
            self._kind.pop(qid, None)
            if prev_z > 0:
                logger.info(
                    "[GRESO update] qid=%s zero_adv=0 streak_reset prev_z=%d "
                    "prev_kind=%s mean_r=%.3f std=%.3f",
                    qid,
                    prev_z,
                    prev_kind,
                    outcome.g_mean,
                    outcome.g_std,
                )

    def on_version_changed(self, version):
        prev_p_easy = self._p_easy
        prev_p_hard = self._p_hard
        if self._n_total > 0:
            easy_ratio = self._n_easy / self._n_total
            hard_ratio = self._n_hard / self._n_total
            # Always snapshot observed ratios for telemetry visibility,
            # even during the warmup window when p_e is held fixed.
            self._last_easy_ratio = easy_ratio
            self._last_hard_ratio = hard_ratio
            if self._adjustment_armed:
                if easy_ratio >= self._alpha_easy:
                    self._p_easy = max(self._p_min, self._p_easy - self._delta_p)
                else:
                    self._p_easy = min(self._p_max, self._p_easy + self._delta_p)
                if hard_ratio >= self._alpha_hard:
                    self._p_hard = max(self._p_min, self._p_hard - self._delta_p)
                else:
                    self._p_hard = min(self._p_max, self._p_hard + self._delta_p)
            logger.info(
                "[GRESO on_version v=%s armed=%s] window: n_total=%d n_easy=%d n_hard=%d "
                "easy_r=%.3f (target=%.3f) hard_r=%.3f (target=%.3f)  "
                "p_easy: %.3f -> %.3f   p_hard: %.3f -> %.3f",
                version,
                self._adjustment_armed,
                self._n_total,
                self._n_easy,
                self._n_hard,
                easy_ratio,
                self._alpha_easy,
                hard_ratio,
                self._alpha_hard,
                prev_p_easy,
                self._p_easy,
                prev_p_hard,
                self._p_hard,
            )
        else:
            logger.info(
                "[GRESO on_version v=%s armed=%s] window: n_total=0 (no observations)",
                version,
                self._adjustment_armed,
            )
        self._n_easy = 0
        self._n_hard = 0
        self._n_total = 0

    def get_telemetry(self) -> dict[str, float]:
        """Expose adaptive state for wandb logging."""
        return {
            "p_easy": self._p_easy,
            "p_hard": self._p_hard,
            "observed_easy_ratio": self._last_easy_ratio,
            "observed_hard_ratio": self._last_hard_ratio,
            "adjustment_armed": 1.0 if self._adjustment_armed else 0.0,
        }

    def state_dict(self) -> dict:
        """Serialise per-prompt streaks + adaptive base + warmup flag.

        In-flight per-iteration tallies (n_easy/n_hard/n_total) are not
        saved — they're transient and reset at on_version_changed anyway.
        Warmup epoch tracking lives in ``DataAcquisition`` and is
        re-derived from the curator's ``adjustment_armed`` flag on resume.
        """
        return {
            "streak": dict(self._streak),
            "kind": dict(self._kind),
            "p_easy": float(self._p_easy),
            "p_hard": float(self._p_hard),
            "adjustment_armed": bool(self._adjustment_armed),
            "last_easy_ratio": float(self._last_easy_ratio),
            "last_hard_ratio": float(self._last_hard_ratio),
        }

    def load_state_dict(self, state: dict) -> None:
        self._streak = dict(state.get("streak", {}))
        self._kind = dict(state.get("kind", {}))
        if "p_easy" in state:
            self._p_easy = max(self._p_min, min(self._p_max, float(state["p_easy"])))
        if "p_hard" in state:
            self._p_hard = max(self._p_min, min(self._p_max, float(state["p_hard"])))
        self._adjustment_armed = bool(state.get("adjustment_armed", False))
        self._last_easy_ratio = float(state.get("last_easy_ratio", 0.0))
        self._last_hard_ratio = float(state.get("last_hard_ratio", 0.0))
        # Tallies always reset on load.
        self._n_easy = 0
        self._n_hard = 0
        self._n_total = 0


# ---------------------------------------------------------------------------
# Resolver helper used by AstraDataAcquisition.__init__.
# ---------------------------------------------------------------------------


def resolve_curator(
    curator: "PromptCurator | str | None",
    curator_args: dict | None = None,
) -> "PromptCurator | None":
    """Resolve a curator argument to an instance, or return ``None``.

    Accepts:
    - ``None`` → no curator (selective rollout disabled).
    - ``str`` → registry lookup, instantiated with ``curator_args``.
    - already-an-instance → returned as-is.
    """
    if curator is None:
        return None
    if isinstance(curator, str):
        return get_curator(curator, **(curator_args or {}))
    # Duck-typed: trust it implements the Protocol.
    if not all(hasattr(curator, m) for m in ("should_submit", "update")):
        raise TypeError(
            f"curator must be a registered name or implement PromptCurator; "
            f"got {type(curator).__name__}"
        )
    return curator


__all__ = [
    "PromptCurator",
    "RolloutOutcome",
    "CURATOR_REGISTRY",
    "register_curator",
    "get_curator",
    "resolve_curator",
    "AcceptAllCurator",
    "FilterSolvedCurator",
    "GRESOCurator",
]
