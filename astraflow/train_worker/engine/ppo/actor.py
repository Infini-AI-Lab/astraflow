import functools
from typing import Any

import torch

from astraflow.train_worker.api.cli_args import MicroBatchSpec, PPOActorConfig
from astraflow.train_worker.api.engine_api import TrainEngine
from astraflow.train_worker.engine.fsdp_engine import FSDPEngine
from astraflow.train_worker.engine.megatron_engine import MegatronEngine
from astraflow.train_worker.utils import logging, stats_tracker
from astraflow.train_worker.utils.data import (
    KLEstimator,
    Normalization,
    split_padded_tensor_dict_into_mb_list,
)
from astraflow.train_worker.utils.functional import (
    ppo_actor_loss_fn,
    reward_overlong_penalty,
    sapo_loss_fn,
)
from astraflow.train_worker.utils.perf_tracer import trace_perf

logger = logging.getLogger(__name__)


class PPOActor:
    def __init__(self, config: PPOActorConfig, engine: TrainEngine):
        self.config = config
        self.engine = engine

        self.reward_bias = config.reward_bias
        self.reward_scaling = config.reward_scaling
        self.reward_clip = config.reward_clip

        self.group_size = config.group_size

        self.kl_ctl = config.kl_ctl
        self.kl_penalty_coef = config.kl_penalty_coef
        self.kl_estimator = KLEstimator(config.kl_estimator)

        self.adv_norm = Normalization(config.adv_norm) if config.adv_norm else None
        self.reward_norm = (
            Normalization(config.reward_norm) if config.reward_norm else None
        )

        self.discount = config.discount
        self.gae_lambda = config.gae_lambda
        self.mask_no_eos_with_zero = config.mask_no_eos_with_zero

        self.temperature = config.temperature

        self.m2_threshold = config.m2_threshold

        # Log critical GSPO/GRPO configuration for reproducibility
        self._log_configuration()

    def _log_configuration(self):
        """Log PPO configuration including how proximal policy is computed."""
        config = self.config

        logger.info("=" * 70)
        logger.info("PPOActor Configuration")
        logger.info("=" * 70)

        # Log PPO mode
        if config.recompute_logprob:
            logger.info("  old_logp (π_old): RECOMPUTED from current policy")
        else:
            logger.info(
                "  old_logp (π_old): FROM INFERENCE (cached during rollout)"
            )

        # Log other critical config
        logger.info("=" * 70)
        logger.info("Training Parameters:")
        logger.info(
            f"  importance_sampling_level: {getattr(config, 'importance_sampling_level', 'token')}"
        )
        logger.info(
            f"  adv_norm: {config.adv_norm if config.adv_norm else 'DISABLED (None)'}"
        )
        logger.info(
            f"  reward_norm: {config.reward_norm if config.reward_norm else 'DISABLED (None)'}"
        )
        logger.info(f"  eps_clip: {config.eps_clip}")
        logger.info(f"  kl_penalty_coef: {config.kl_penalty_coef}")
        logger.info(f"  group_size: {config.group_size}")
        logger.info("=" * 70)

    @trace_perf("ppo_actor.compute_logp", category="compute")
    @torch.no_grad()
    def compute_logp(self, data: dict[str, Any]) -> torch.Tensor | None:
        self.engine.eval()
        return self.engine.forward(
            input_=data,
            aggregate_fn=lambda xs: torch.cat(xs, dim=-1),
        )

    def _can_use_stamped_group_stats(self, data: dict[str, Any]) -> bool:
        """Whether to use producer-stamped per-group stats over reward_norm()."""
        if not self.reward_norm:
            return False
        if "group_reward_mean" not in data or "group_reward_std" not in data:
            return False
        # Stamped stats are per-prompt mean/std. Only valid when both mean
        # and std are configured at group level, and not leave-one-out.
        if (
            self.reward_norm.mean_level != "group"
            or self.reward_norm.std_level != "group"
            or self.reward_norm.mean_leave1out
        ):
            return False
        return True

    def _apply_group_reward_norm(
        self, data: dict[str, Any], reward_score: torch.Tensor
    ) -> torch.Tensor:
        """Apply per-group reward normalization.

        Prefers producer-stamped per-group mean/std (computed on the full
        per-prompt × per-model group in AstraFlow before buffer scattering)
        over per-rank reward_norm(group_ids=...). The producer-stamped path
        is correct even when a group is split across DP ranks (multi-turn
        / variable-size workflows). Falls back to the per-rank path for
        backward compatibility with old buffers and for non-group configs.

        Assumes bias/scaling have already been applied to ``reward_score``.
        """
        if not self.reward_norm:
            return reward_score
        if self._can_use_stamped_group_stats(data):
            g_mean = data["group_reward_mean"]
            g_std = data["group_reward_std"]
            # Match group_id handling: tensors may be [bs] or [bs, seq_len]
            # (padded with other batch tensors). Take first column when 2D.
            if g_mean.ndim >= 2:
                g_mean = g_mean[:, 0]
            if g_std.ndim >= 2:
                g_std = g_std[:, 0]
            g_mean = g_mean.to(reward_score.device, reward_score.dtype)
            g_std = g_std.to(reward_score.device, reward_score.dtype)
            # Apply the same linear transform that was applied to per-sample
            # rewards (bias+scaling). std scales by |scaling|; clip is
            # non-linear so group stats are left unclipped — exact
            # equivalence to the legacy path holds for the common case
            # bias=0, scaling=1, clip=+inf.
            g_mean = (g_mean + self.reward_bias) * self.reward_scaling
            g_std = g_std * abs(self.reward_scaling)
            # Match Normalization: divide by (std + eps), not clamp.
            return (reward_score - g_mean) / (g_std + self.reward_norm.eps)
        # Legacy fallback: per-rank reward_norm; buggy on partial groups.
        group_ids = data.get("group_id")
        if group_ids is not None:
            group_ids = group_ids[:, 0] if group_ids.ndim >= 2 else group_ids
        return self.reward_norm(reward_score, group_ids=group_ids)

    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards using configured bias, scaling, clipping, and normalization.

        This function applies the same reward normalization logic used in
        compute_advantages(), but as a standalone function. It can be used to
        pre-normalize rewards before storing them in data["normalized_rewards"].

        Parameters
        ----------
        rewards : torch.Tensor
            Raw reward scores to normalize. Can be any shape (typically 1D for
            per-sequence rewards or 2D for per-token rewards).

        Returns
        -------
        torch.Tensor
            Normalized reward scores with the same shape as input, after applying:
            1. Bias: (rewards + reward_bias)
            2. Scaling: * reward_scaling
            3. Clipping: clipped to [-reward_clip, reward_clip]
            4. Normalization: if reward_norm is configured
        """
        reward_score = rewards
        reward_score = (reward_score + self.reward_bias) * self.reward_scaling
        reward_score = torch.clip(
            reward_score, max=self.reward_clip, min=-self.reward_clip
        )
        if self.reward_norm:
            reward_score = self.reward_norm(reward_score)
        return reward_score

    @trace_perf("ppo_actor.compute_advantages", category="compute")
    def compute_advantages(self, data: dict[str, Any]) -> dict[str, Any]:
        bs = data["input_ids"].shape[0]
        max_seqlen = data["input_ids"].shape[1]
        batch_indices = torch.arange(
            bs, device=data["input_ids"].device, dtype=torch.long
        )

        # Reward Penalty on length
        if self.config.overlong_reward_penalty:
            overlong_tokens = self.config.overlong_tokens
            overlong_penalty_factor = self.config.overlong_penalty_factor

            assert overlong_tokens is not None
            assert overlong_penalty_factor is not None
            data = reward_overlong_penalty(
                data,
                overlong_tokens=overlong_tokens,
                overlong_penalty_factor=overlong_penalty_factor,
                max_response_length=self.config.max_new_tokens,
            )

        # Reward Scaling
        reward_score = data["rewards"]
        reward_score = (reward_score + self.reward_bias) * self.reward_scaling
        reward_score = torch.clip(
            reward_score, max=self.reward_clip, min=-self.reward_clip
        )
        if self.reward_norm:
            reward_score = self._apply_group_reward_norm(data, reward_score)

        loss_mask = data["loss_mask"].float()
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)
        # Apply the mask to log probabilities.
        if self.config.recompute_logprob:
            # Overwrite logprobs produced by the inference engine
            prox_logp_value = data["prox_logp"]
            if prox_logp_value is None:
                raise ValueError(
                    "prox_logp is None but recompute_logprob=True. "
                    "This indicates compute_logp() was skipped incorrectly."
                )
            old_logp = data["logprobs"] = prox_logp_value
        else:
            old_logp = torch.roll(data["logprobs"], shifts=-1, dims=-1)
        ref_logp = data.get("ref_logp")
        if ref_logp is None:
            ref_logp = torch.zeros_like(old_logp)
        ref_logp *= loss_mask
        old_logp *= loss_mask

        # Compute KL-regularized rewards.
        attn_mask = data["attention_mask"]
        seqlens = attn_mask.sum(-1).long()
        seq_no_eos_mask = seqlens == attn_mask.shape[1]
        rewards = -self.kl_ctl * self.kl_estimator(old_logp, ref_logp)
        kl_rewards = rewards.clone()
        # KL rewards at the next token after eos is zero.
        rewards[batch_indices, seqlens - 1] = 0

        # Multi-model: place reward at the last token where loss_mask=1
        # (i.e. the last token of this trainer's model segment), instead
        # of seqlens-2 which may fall in another model's segment.
        if "model_ids" in data and loss_mask.sum() > 0:
            lm_positions = loss_mask * torch.arange(
                loss_mask.shape[1], device=loss_mask.device
            ).unsqueeze(0)
            indices = lm_positions.long().argmax(dim=1)
        else:
            indices = torch.clip(seqlens - 2, min=0)

        if self.mask_no_eos_with_zero:
            rewards[batch_indices, indices] += torch.where(
                seq_no_eos_mask, 0, reward_score
            )
        else:
            rewards[batch_indices, indices] += reward_score

        # Compute GAE.
        if "values" not in data:
            values = torch.zeros_like(rewards)
        else:
            values = data["values"]
        advantages_reversed = [
            torch.zeros(bs, dtype=torch.float32, device=values.device)
        ]
        lastgaelam = 0
        nextvalues = values[:, max_seqlen - 1] * seq_no_eos_mask
        for t in reversed(range(max_seqlen - 1)):
            delta = rewards[:, t] + self.discount * nextvalues - values[:, t]
            newgaelam = delta + self.discount * self.gae_lambda * lastgaelam

            # Skip tokens that do not contribute to the loss
            mask = loss_mask[:, t]
            nextvalues = nextvalues * (1 - mask) + values[:, t] * mask
            lastgaelam = lastgaelam * (1 - mask) + newgaelam * mask
            advantages_reversed.append(lastgaelam)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        data["returns"] = advantages + values

        # Optionally perform advantage normalization.
        if self.adv_norm is not None:
            advantages = self.adv_norm(advantages, loss_mask)

        # Store data in the dict.
        data["advantages"] = advantages
        data["kl_rewards"] = kl_rewards
        data["tot_rewards"] = rewards
        data["loss_mask"] = loss_mask
        # because we have rolled old_logp by -1
        data["logprobs"] = old_logp

        return data

    @trace_perf("ppo_actor.compute_advantages", category="compute")
    def compute_advantages_with_normalized_reward(self, data: dict[str, Any]) -> dict[str, Any]:
        """Compute advantages using pre-normalized rewards.

        This function assumes that rewards have already been normalized and are stored
        in data["normalized_rewards"]. It skips reward penalty, scaling, clipping, and
        normalization steps, using the normalized rewards directly.

        Parameters
        ----------
        data : dict[str, Any]
            Input data dictionary containing:
            - "normalized_rewards": Pre-normalized reward scores (tensor)
            - Other fields same as compute_advantages()

        Returns
        -------
        dict[str, Any]
            Data dictionary with computed advantages and related fields, same as
            compute_advantages().
        """
        bs = data["input_ids"].shape[0]
        max_seqlen = data["input_ids"].shape[1]
        batch_indices = torch.arange(
            bs, device=data["input_ids"].device, dtype=torch.long
        )

        # Use pre-normalized rewards from the buffer, or re-normalize using
        # producer-stamped group stats / group_id (AstraFlow v2 stamps both).
        group_ids = data.get("group_id")
        if group_ids is not None:
            group_ids = group_ids[:, 0] if group_ids.ndim >= 2 else group_ids
        used_stamped = self._can_use_stamped_group_stats(data)
        if self.reward_norm and (used_stamped or group_ids is not None):
            reward_score = data["rewards"]
            reward_score = (reward_score + self.reward_bias) * self.reward_scaling
            reward_score = torch.clip(
                reward_score, max=self.reward_clip, min=-self.reward_clip
            )
            reward_score = self._apply_group_reward_norm(data, reward_score)
        elif "normalized_rewards" in data:
            reward_score = data["normalized_rewards"]
        else:
            raise ValueError(
                "Neither group_id, group_reward_mean/std, nor "
                "normalized_rewards found in data. Cannot compute "
                "advantages without reward normalization."
            )

        # DEBUG zero_adv: After filter_zero_adv + correct group_size, every
        # sample's per-prompt z-score should be non-zero. Count zeros per rank.
        with torch.no_grad():
            _rs = reward_score.reshape(-1).detach()
            _n_zero = int((_rs.abs() < 1e-8).sum().item())
            _n = int(_rs.numel())
            try:
                import torch.distributed as _dist
                _rank = _dist.get_rank() if _dist.is_initialized() else 0
            except Exception:
                _rank = 0
            _uniq_groups = None
            if group_ids is not None:
                _uniq_groups = int(group_ids.unique().numel())
            import sys as _sys
            _path = "stamped" if used_stamped else "legacy"
            _sys.stdout.write(
                f"[DEBUG zero_adv] rank={_rank} n_zero_rs={_n_zero}/{_n} "
                f"rs_range=[{float(_rs.min().item()):.4f},{float(_rs.max().item()):.4f}] "
                f"n_unique_groups={_uniq_groups} path={_path}\n"
            )
            _sys.stdout.flush()

        loss_mask = data["loss_mask"].float()
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)
        # Apply the mask to log probabilities.
        if self.config.recompute_logprob:
            # Overwrite logprobs produced by the inference engine
            prox_logp_value = data["prox_logp"]
            if prox_logp_value is None:
                raise ValueError(
                    "prox_logp is None but recompute_logprob=True. "
                    "This indicates compute_logp() was skipped incorrectly."
                )
            old_logp = data["logprobs"] = prox_logp_value
        else:
            old_logp = torch.roll(data["logprobs"], shifts=-1, dims=-1)
        ref_logp = data.get("ref_logp")
        if ref_logp is None:
            ref_logp = torch.zeros_like(old_logp)
        ref_logp *= loss_mask
        old_logp *= loss_mask

        # Compute KL-regularized rewards.
        attn_mask = data["attention_mask"]
        seqlens = attn_mask.sum(-1).long()
        seq_no_eos_mask = seqlens == attn_mask.shape[1]
        rewards = -self.kl_ctl * self.kl_estimator(old_logp, ref_logp)
        kl_rewards = rewards.clone()
        # KL rewards at the next token after eos is zero.
        rewards[batch_indices, seqlens - 1] = 0

        # Multi-model: place reward at the last token where loss_mask=1
        # (i.e. the last token of this trainer's model segment), instead
        # of seqlens-2 which may fall in another model's segment.
        if "model_ids" in data and loss_mask.sum() > 0:
            lm_positions = loss_mask * torch.arange(
                loss_mask.shape[1], device=loss_mask.device
            ).unsqueeze(0)
            indices = lm_positions.long().argmax(dim=1)
        else:
            indices = torch.clip(seqlens - 2, min=0)

        if self.mask_no_eos_with_zero:
            rewards[batch_indices, indices] += torch.where(
                seq_no_eos_mask, 0, reward_score
            )
        else:
            rewards[batch_indices, indices] += reward_score

        # Compute GAE.
        if "values" not in data:
            values = torch.zeros_like(rewards)
        else:
            values = data["values"]
        advantages_reversed = [
            torch.zeros(bs, dtype=torch.float32, device=values.device)
        ]
        lastgaelam = 0
        nextvalues = values[:, max_seqlen - 1] * seq_no_eos_mask
        for t in reversed(range(max_seqlen - 1)):
            delta = rewards[:, t] + self.discount * nextvalues - values[:, t]
            newgaelam = delta + self.discount * self.gae_lambda * lastgaelam

            # Skip tokens that do not contribute to the loss
            mask = loss_mask[:, t]
            nextvalues = nextvalues * (1 - mask) + values[:, t] * mask
            lastgaelam = lastgaelam * (1 - mask) + newgaelam * mask
            advantages_reversed.append(lastgaelam)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        data["returns"] = advantages + values

        # Optionally perform advantage normalization.
        if self.adv_norm is not None:
            advantages = self.adv_norm(advantages, loss_mask)

        # Store data in the dict.
        data["advantages"] = advantages
        data["kl_rewards"] = kl_rewards
        data["tot_rewards"] = rewards
        data["loss_mask"] = loss_mask
        # because we have rolled old_logp by -1
        data["logprobs"] = old_logp

        return data

    @trace_perf("ppo_actor.ppo_update", category="compute")
    @stats_tracker.scope_func_wrapper("ppo_actor")
    def ppo_update(self, data: dict[str, Any]) -> None:
        attn_mask = data["attention_mask"]
        loss_mask = data["loss_mask"]
        reward_score = data["rewards"]
        seqlens = attn_mask.sum(-1)

        ########## Logging code starts ##########
        result_denominators = {
            "correct_n_seqs": (reward_score > 0).bool(),
            "incorrect_n_seqs": (reward_score <= 0).bool(),
        }
        if self.config.log_agent_stats:
            if "begin_of_trajectory" not in data:
                raise RuntimeError(
                    "'begin_of_trajectory' is expected to log agent statistics"
                )
            if len(self.config.log_agent_stats_keys) == 0:
                raise RuntimeError(
                    "`log_agent_stats_keys` should not be empty when log_agent_stats=True"
                )
            agent_denominator = (data["begin_of_trajectory"] > 0).bool()
            result_denominators["agent"] = agent_denominator
        global_denominators = dict(
            n_seqs=torch.ones_like(reward_score, dtype=torch.bool),
            n_tokens=torch.ones_like(loss_mask, dtype=torch.bool),
            n_valid_tokens=loss_mask.bool(),
            **result_denominators,
        )
        stats_tracker.denominator(**global_denominators)
        stats_tracker.stat(
            correct_seq_len=seqlens.float(), denominator="correct_n_seqs"
        )
        stats_tracker.stat(
            incorrect_seq_len=seqlens.float(), denominator="incorrect_n_seqs"
        )

        stats = dict(
            advantages=data["advantages"],
            kl_rewards=data["kl_rewards"],
            final_reward=data["tot_rewards"],
        )
        stats_tracker.stat(**stats, denominator="n_valid_tokens")

        prompt_lens = data["attention_mask"].sum(-1) - data["loss_mask"].sum(-1)
        seq_stats = dict(
            no_eos_ratios=(seqlens == attn_mask.shape[-1]).float(),
            task_reward=reward_score.float(),
            prompt_len=prompt_lens.float(),
            seq_len=seqlens.float(),
        )
        stats_tracker.stat(**seq_stats, denominator="n_seqs")
        scalars = dict(
            mask_no_eos_with_zero=self.config.mask_no_eos_with_zero,
            eps_clip=self.config.eps_clip,
        )
        if self.config.c_clip is not None:
            scalars["c_clip"] = self.config.c_clip
            scalars["use_dual_clip"] = 1
        else:
            scalars["use_dual_clip"] = 0
        stats_tracker.scalar(**scalars)

        if self.config.log_agent_stats:
            stats_tracker.stat(
                **{k: data[k].float() for k in self.config.log_agent_stats_keys},
                denominator="agent",
            )
        ########## Logging code ends ##########

        # Pop keys that are no longer needed after advantage computation
        # Note: "versions" is kept for staleness metrics in loss function
        for key in ["rewards", "tot_rewards", "kl_rewards"]:
            data.pop(key, None)
        # NOTE: calling engine.train() is critical to enabling gradient checkpointing
        self.engine.train()
        mb_inputs = split_padded_tensor_dict_into_mb_list(
            data,
            mb_spec=MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches),
        )

        with stats_tracker.scope("update"):
            # Get current version for proximal approximation metrics
            current_version = self.engine.get_version()

            for mb in mb_inputs.mbs:
                train_stat = self.engine.train_batch(
                    mb,
                    loss_fn=functools.partial(
                        grpo_loss_fn,
                        eps_clip=self.config.eps_clip,
                        eps_clip_higher=self.config.eps_clip_higher,
                        c_clip=self.config.c_clip,
                        kl_penalty_coef=self.config.kl_penalty_coef,
                        kl_estimator=self.config.kl_estimator,
                        m2_threshold=self.m2_threshold,
                        importance_sampling_level=self.config.importance_sampling_level,
                        current_version=current_version,
                        use_sapo_loss=self.config.use_sapo_loss,
                        sapo_tau_pos=self.config.sapo_tau_pos,
                        sapo_tau_neg=self.config.sapo_tau_neg,
                    ),
                    loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
                )
                stats_tracker.scalar(**train_stat)


class FSDPPPOActor(FSDPEngine):
    def __init__(self, config: PPOActorConfig):
        super().__init__(config)
        self.actor = PPOActor(config, self)

    @torch.no_grad()
    def compute_logp(self, *args, **kwargs) -> torch.Tensor | None:
        return self.actor.compute_logp(*args, **kwargs)

    @torch.no_grad()
    def compute_advantages(self, *args, **kwargs) -> dict[str, Any]:
        return self.actor.compute_advantages(*args, **kwargs)

    @torch.no_grad()
    def compute_advantages_with_normalized_reward(self, *args, **kwargs) -> dict[str, Any]:
        return self.actor.compute_advantages_with_normalized_reward(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> None:
        self.actor.ppo_update(*args, **kwargs)


class MegatronPPOActor(MegatronEngine):
    def __init__(self, config: PPOActorConfig):
        super().__init__(config)
        self.actor = PPOActor(config, self)

    @torch.no_grad()
    def compute_logp(self, *args, **kwargs) -> torch.Tensor | None:
        return self.actor.compute_logp(*args, **kwargs)

    @torch.no_grad()
    def compute_advantages(self, *args, **kwargs) -> dict[str, Any]:
        return self.actor.compute_advantages(*args, **kwargs)

    @torch.no_grad()
    def compute_advantages_with_normalized_reward(self, *args, **kwargs) -> dict[str, Any]:
        return self.actor.compute_advantages_with_normalized_reward(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> None:
        self.actor.ppo_update(*args, **kwargs)


def grpo_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict,
    eps_clip: float,
    eps_clip_higher: float | None,
    c_clip: float | None,
    kl_penalty_coef: float = 0.0,
    kl_estimator: str = "k1",
    m2_threshold: float | None = None,
    importance_sampling_level: str = "token",
    current_version: int | None = None,
    use_sapo_loss: bool = False,
    sapo_tau_pos: float = 1.0,
    sapo_tau_neg: float = 1.05,
    vocab_min_logits: torch.Tensor | None = None,
    vocab_max_logits: torch.Tensor | None = None,
):
    """Loss function for actor step, all inputs should be splitted into
    pipeline micro batches, returns loss and logging stats."""
    old_logp = input_data["logprobs"]
    advantages = input_data["advantages"]
    loss_mask = input_data["loss_mask"].bool()

    entropy = entropy.detach()

    # Apply M2PO clipping if threshold is set
    eps_clip_low = eps_clip
    eps_clip_high = eps_clip_higher
    if m2_threshold is not None:
        eps_clip_low, eps_clip_high, m2_mean = _calculate_m2po_clip_range(
            logprobs=logprobs,
            old_logp=old_logp,
            advantages=advantages,
            loss_mask=loss_mask,
            m2_threshold=m2_threshold,
        )
        stats_tracker.scalar(
            eps_clip_m2po_low=eps_clip_low,
            eps_clip_m2po_high=eps_clip_high,
            m2po_mean_m2=m2_mean,
        )

    # Use SAPO or PPO loss
    if use_sapo_loss:
        loss, stat = sapo_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logp,
            advantages=advantages,
            tau_pos=sapo_tau_pos,
            tau_neg=sapo_tau_neg,
            loss_mask=loss_mask,
            importance_sampling_level=importance_sampling_level,
            cu_seqlens=input_data.get("cu_seqlens"),
        )
    else:
        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logp,
            advantages=advantages,
            eps_clip=eps_clip_low,
            eps_clip_higher=eps_clip_high,
            loss_mask=loss_mask,
            c_clip=c_clip,
            importance_sampling_level=importance_sampling_level,
            cu_seqlens=input_data.get("cu_seqlens"),
        )

    kl_penalty = None
    kl_values = None
    if kl_penalty_coef > 0:
        ref_logp = input_data.get("ref_logp")
        if ref_logp is not None:
            kl_values = KLEstimator(kl_estimator)(logprobs, ref_logp)
            mask = loss_mask.float()
            kl_penalty = (kl_values * mask).sum() / mask.sum().clamp(min=1)
            loss = loss + kl_penalty_coef * kl_penalty

    # Log training statistics
    stats_tracker.denominator(
        # NOTE: n_tokens must have shape [batch, seq] to match vocab stats.
        # Using torch.ones_like(loss_mask) ensures correct shape when this function is called
        # standalone (e.g., by recipe/AEnt or tests), not just from ppo_update() which already
        # registers n_tokens.
        n_tokens=torch.ones_like(loss_mask, dtype=torch.bool, device=logprobs.device),
        n_valid_tokens=loss_mask.bool(),
        clipped_tokens=stat["clip_mask"],
        dual_clipped_tokens=stat["dual_clip_mask"],
    )
    if kl_penalty is not None:
        stats_tracker.stat(kl_penalty=kl_values, denominator="n_valid_tokens")
        stats_tracker.scalar(kl_penalty_loss=kl_penalty)

    importance_weight = stat["importance_weight"]
    importance_weight_abs_delta = torch.abs(importance_weight - 1.0)
    stats_tracker.denominator(
        importance_weight_gt1_tokens=(importance_weight > 1.0).logical_and(loss_mask),
        importance_weight_lt1_tokens=(importance_weight < 1.0).logical_and(loss_mask),
    )

    stats_tracker.stat(
        importance_weight=importance_weight,
        importance_weight_abs_delta=importance_weight_abs_delta,
        approx_kl=stat["approx_kl"],
        new_logp=logprobs.detach(),
        old_logp=old_logp,
        entropy=entropy.float(),
        actor_loss=stat["loss"],
        clip_ratio=stat["clip_mask"].float(),
        dual_clip_ratio=stat["dual_clip_mask"].float(),
        denominator="n_valid_tokens",
    )
    stats_tracker.stat(
        importance_weight_abs_delta_gt1=importance_weight_abs_delta,
        denominator="importance_weight_gt1_tokens",
    )
    stats_tracker.stat(
        importance_weight_abs_delta_lt1=importance_weight_abs_delta,
        denominator="importance_weight_lt1_tokens",
    )
    if vocab_min_logits is not None and vocab_max_logits is not None:
        stats_tracker.stat(
            vocab_min_logits=vocab_min_logits,
            vocab_max_logits=vocab_max_logits,
            denominator="n_tokens",
        )

    # Log SAPO-specific statistics
    if use_sapo_loss:
        stats_tracker.stat(
            sapo_soft_gate=stat["sapo_soft_gate"],
            sapo_scaled_gate_pos=stat["sapo_scaled_gate_pos"],
            sapo_scaled_gate_neg=stat["sapo_scaled_gate_neg"],
            denominator="n_valid_tokens",
        )
    else:
        # Log clipping statistics (PPO only)
        clip_mask = stat["clip_mask"]
        clipped_new_logp = torch.where(clip_mask, logprobs.detach(), 0.0)
        clipped_old_logp = torch.where(clip_mask, old_logp, 0.0)
        stats_tracker.stat(
            clipped_new_logp=clipped_new_logp,
            clipped_old_logp=clipped_old_logp,
            denominator="clipped_tokens",
        )

    # Log version staleness metrics
    if "versions" in input_data and current_version is not None:
        _log_version_staleness_stats(
            versions=input_data["versions"],
            current_version=current_version,
            version_metrics_mask=loss_mask,
        )

    return loss


# =============================================================================
# Core Functions
# =============================================================================


def _calculate_m2po_clip_range(
    logprobs: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    m2_threshold: float,
) -> tuple[float, float, float]:
    """
    Estimate eps-style clip bounds (eps_low, eps_high) from the M2 budget.

    Returns eps-style bounds for ratio clipping: clamp to [1 - eps_low, 1 + eps_high],
    plus the mean M2 value over trust-region violating tokens.
    When no clipping is needed, returns (0.95, 20.0, m2_mean).
    """
    ratio = torch.exp(logprobs - old_logp)
    trust_mask = ((advantages > 0) & (ratio > 1)) | (
        (advantages < 0) & (ratio < 1)
    )
    filtered_mask = loss_mask & trust_mask

    delta = old_logp - logprobs
    m2 = delta * delta
    mask_flat = filtered_mask.view(-1)
    m2_selected = m2.view(-1)[mask_flat]

    if m2_selected.numel() == 0:
        return 0.95, 20.0, 0.0

    m2_mean = float(m2_selected.mean().item())
    if m2_mean <= m2_threshold:
        return 0.95, 20.0, m2_mean

    sorted_m2, _ = torch.sort(m2_selected, descending=True)

    # Remove largest values until average of remaining drops below threshold.
    suffix_sums = sorted_m2.flip(0).cumsum(0).flip(0)
    counts = torch.arange(
        sorted_m2.numel(),
        0,
        -1,
        device=sorted_m2.device,
        dtype=sorted_m2.dtype,
    )
    avg_suffix = suffix_sums / counts
    below_threshold = torch.where(avg_suffix < m2_threshold)[0]

    if len(below_threshold) > 0:
        num_to_mask = below_threshold[0].item()
    else:
        num_to_mask = sorted_m2.numel() - 1

    cutoff_idx = min(num_to_mask, sorted_m2.numel() - 1)
    tau2 = float(sorted_m2[cutoff_idx].item())
    tau = float(torch.sqrt(torch.tensor(tau2)).item())
    ratio_low = float(torch.exp(torch.tensor(-tau)).item())
    ratio_high = float(torch.exp(torch.tensor(tau)).item())
    eps_low = 1.0 - ratio_low
    eps_high = ratio_high - 1.0
    if eps_low > 0.95:
        eps_low = 0.95
    if eps_high > 20.0:
        eps_high = 20.0
    return eps_low, eps_high, m2_mean


# =============================================================================
# Logging Helper Functions
# =============================================================================

def _tensor_scalar_stats(tensor: torch.Tensor) -> dict[str, float]:
    """Compute scalar statistics (avg, max, min) for a tensor."""
    t = tensor.float()
    return {
        "avg": t.mean().item(),
        "max": t.max().item(),
        "min": t.min().item(),
    }


def _log_version_staleness_stats(
    versions: torch.Tensor,
    current_version: int,
    version_metrics_mask: torch.Tensor,
) -> None:
    """
    Log sample staleness metrics based on policy versions.

    Args:
        versions: Per-token policy versions from rollout.
        current_version: Current training version.
        version_metrics_mask: Mask for valid tokens.
    """
    with stats_tracker.scope("version_stats"):
        stats_tracker.denominator(n_valid_tokens=version_metrics_mask.bool())

        v_proximal = current_version - 1
        v_theta = current_version
        v_behave = versions.float()

        # Filter to generated tokens only (version >= 0)
        valid_generated_mask = version_metrics_mask & (versions >= 0)

        if not valid_generated_mask.any():
            return

        # Compute staleness for valid tokens
        staleness_proximal = (v_proximal - v_behave)[valid_generated_mask]
        staleness_theta = (v_theta - v_behave)[valid_generated_mask]

        # Compute and log statistics
        proximal_stats = _tensor_scalar_stats(staleness_proximal)
        theta_stats = _tensor_scalar_stats(staleness_theta)

        stats_tracker.scalar(
            sample_staleness_proximal_avg=proximal_stats["avg"],
            sample_staleness_proximal_max=proximal_stats["max"],
            sample_staleness_proximal_min=proximal_stats["min"],
            sample_staleness_theta_avg=theta_stats["avg"],
            sample_staleness_theta_max=theta_stats["max"],
            sample_staleness_theta_min=theta_stats["min"],
            v_theta=v_theta,
            v_proximal=v_proximal,
            n_valid_generated_tokens=valid_generated_mask.sum().item(),
        )
