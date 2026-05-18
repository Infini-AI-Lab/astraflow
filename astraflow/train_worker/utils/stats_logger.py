import getpass
import os
import time
import uuid
from dataclasses import asdict

import swanlab
import torch.distributed as dist
from tensorboardX import SummaryWriter

import wandb

from astraflow.train_worker.api.cli_args import BaseExperimentConfig, StatsLoggerConfig
from astraflow.train_worker.api.io_struct import FinetuneSpec
from astraflow.train_worker.utils import logging
from astraflow.train_worker.utils.printing import tabulate_stats
from astraflow.train_worker.version import version_info

logger = logging.getLogger("StatsLogger", "system")


class StatsLogger:
    def __init__(self, config: BaseExperimentConfig, ft_spec: FinetuneSpec):
        if isinstance(config, StatsLoggerConfig):
            raise ValueError(
                "Passing config.stats_logger as the config is deprecated. "
                "Please pass the full config instead."
            )
        self.exp_config = config
        self.config = config.stats_logger
        self.ft_spec = ft_spec
        self.init()

        self._last_commit_step = 0

    def init(self):
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        if self.config.wandb.wandb_base_url:
            os.environ["WANDB_API_KEY"] = self.config.wandb.wandb_api_key
        if self.config.wandb.wandb_api_key:
            os.environ["WANDB_BASE_URL"] = self.config.wandb.wandb_base_url

        self.start_time = time.perf_counter()
        # wandb init, connect to remote wandb host
        if self.config.wandb.mode != "disabled":
            wandb.login()

        suffix = self.config.wandb.id_suffix
        if suffix == "timestamp":
            suffix = time.strftime("%Y_%m_%d_%H_%M_%S")
        elif suffix == "uid":
            suffix = uuid.uuid4().hex[:8]

        exp_config_dict = asdict(self.exp_config)
        exp_config_dict["version_info"] = {
            "commit_id": version_info.commit,
            "branch": version_info.branch,
            "is_dirty": version_info.is_dirty,
            "version": version_info.full_version_with_dirty_description,
        }

        wandb.init(
            mode=self.config.wandb.mode,
            entity=self.config.wandb.entity,
            project=self.config.wandb.project or self.config.experiment_name,
            name=self.config.wandb.name or self.config.trial_name,
            job_type=self.config.wandb.job_type,
            group=self.config.wandb.group
            or f"{self.config.experiment_name}_{self.config.trial_name}",
            notes=self.config.wandb.notes,
            tags=self.config.wandb.tags,
            config=exp_config_dict,  # save all experiment config to wandb
            dir=self.get_wandb_path(self.config),
            force=True,
            id=f"{self.config.trial_name}_{suffix}",
            resume="allow",
        )

        swanlab_config = self.config.swanlab
        if swanlab_config.mode != "disabled":
            if swanlab_config.api_key:
                swanlab.login(swanlab_config.api_key)
            else:
                swanlab.login()

        swanlab_config = self.config.swanlab
        swanlab.init(
            project=swanlab_config.project or self.config.experiment_name,
            experiment_name=swanlab_config.name or self.config.trial_name + "_train",
            # NOTE: change from swanlab_config.config to log all experiment config, to be tested
            config=exp_config_dict,
            logdir=self.get_log_path(self.config),
            mode=swanlab_config.mode,
        )
        # tensorboard logging
        self.summary_writer = None
        if self.config.tensorboard.path is not None:
            self.summary_writer = SummaryWriter(log_dir=self.config.tensorboard.path)

    def state_dict(self):
        return {
            "last_commit_step": self._last_commit_step,
        }

    def load_state_dict(self, state_dict):
        self._last_commit_step = state_dict["last_commit_step"]

    def close(self):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        logger.info(
            f"Training session closed. Total time elapsed {time.monotonic() - self.start_time:.2f}."
        )
        wandb.finish()
        swanlab.finish()
        if self.summary_writer is not None:
            self.summary_writer.close()

    def _rewrite_ppo_update_metrics(self, item: dict) -> dict:
        """
        Rewrite all 'ppo_actor/update/*' metrics into grouped namespaces, and
        remove the 'update/' segment in the returned keys.

        Example:
        ppo_actor/update/actor_loss/avg
            -> loss/actor_loss/avg

        ppo_actor/update/version_stats/v_theta
            -> staleness/version_stats/v_theta
        """
        UPDATE_MAPPING = {
            # Importance sampling / off-policy
            "importance_weight": "importance_sampling",

            # Clipping & trust region
            "clip_ratio": "clip",
            "clipped_tokens": "clip",
            "dual_clip_ratio": "clip",
            "dual_clipped_tokens": "clip",
            "eps_clip": "clip",
            "eps_clip_m2po_low": "clip",
            "eps_clip_m2po_high": "clip",
            "m2po_eps_clip_low": "clip",
            "m2po_eps_clip_high": "clip",
            "m2po_mean_m2": "clip",
            "use_dual_clip": "clip",

            # Entropy / exploration
            "entropy": "training",
            # Update success
            "update_successful": "training",
            # Loss
            "actor_loss": "training",
            # KL / distribution shift
            "approx_kl": "kl",
            "kl_penalty": "kl",
            "kl_penalty_loss": "kl",
            # Grad / optimizer
            "grad_norm": "training",
            "lr": "training",
            # Logits
            "vocab_max_logits": "training",
            "vocab_min_logits": "training",

            # Token counts
            "n_tokens": "tokens",
            "n_valid_tokens": "tokens",

            # Staleness / version stats
            "version_stats/n_valid_generated_tokens": "staleness",
            "version_stats/n_valid_tokens": "staleness",
            "version_stats/sample_staleness_proximal_avg": "staleness",
            "version_stats/sample_staleness_proximal_min": "staleness",
            "version_stats/sample_staleness_proximal_max": "staleness",
            "version_stats/sample_staleness_theta_avg": "staleness",
            "version_stats/sample_staleness_theta_min": "staleness",
            "version_stats/sample_staleness_theta_max": "staleness",
            "version_stats/v_proximal": "staleness",
            "version_stats/v_theta": "staleness",

            # fallback
            "*": "misc",
        }

        old_prefix = "ppo_actor/update/"
        new_item = {}

        for k, v in item.items():
            if k.startswith(old_prefix):
                # strip 'ppo_actor/update/' → get suffix
                suffix = k[len(old_prefix):]  # e.g., "actor_loss/avg"

                # find category by prefix matching on the suffix
                category = UPDATE_MAPPING["*"]
                for pattern, cat in UPDATE_MAPPING.items():
                    if pattern == "*":
                        continue
                    if suffix.startswith(pattern):
                        category = cat
                        break

                new_key = f"{category}/{suffix}"
                new_item[new_key] = v
            elif k.startswith('ppo_actor/advantages/'):
                new_key = k.replace('ppo_actor/advantages/', 'advantages/')
                new_item[new_key] = v
            elif k.startswith('ppo_actor/seq_len/'):
                new_key = k.replace('ppo_actor/seq_len/', 'length/seq_len/')
                new_item[new_key] = v
            elif k.startswith('ppo_actor/prompt_len/'):
                new_key = k.replace('ppo_actor/prompt_len/', 'length/prompt_len/')
                new_item[new_key] = v
            elif k.startswith('ppo_actor/incorrect_seq_len/'):
                new_key = k.replace('ppo_actor/incorrect_seq_len/', 'length/incorrect_seq_len/')
                new_item[new_key] = v
            elif k.startswith('ppo_actor/correct_seq_len/'):
                new_key = k.replace('ppo_actor/correct_seq_len/', 'length/correct_seq_len/')
                new_item[new_key] = v
            else:
                # keep all non-update metrics as-is
                new_item[k] = v

        return new_item


    def commit(self, epoch: int, step: int, global_step: int, data: dict | list[dict]):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        effective_total_steps = self.ft_spec.total_train_steps
        if self.exp_config.total_train_steps is not None:
            effective_total_steps = min(
                effective_total_steps, self.exp_config.total_train_steps
            )
        logger.info(
            f"Epoch {epoch + 1}/{self.ft_spec.total_train_epochs} "
            f"Step {step + 1}/{self.ft_spec.steps_per_epoch} "
            f"Train step {global_step + 1}/{effective_total_steps} done."
        )
        if isinstance(data, dict):
            data = [data]
        log_step = max(global_step, self._last_commit_step + 1)

        for i, item in enumerate(data):
            # Filter out counter keys for scalar variables
            item = {k: v for k, v in item.items() if not k.endswith("__count")}
            item = self._rewrite_ppo_update_metrics(item)
            # Hide selected noisy metrics from external loggers to reduce dashboard noise.
            hidden_exact_metrics = {
                "clip/dual_clipped_tokens",
                "clip/dual_clip_ratio",
                "importance_sampling/importance_weight_gt1_tokens",
                "importance_sampling/importance_weight_lt1_tokens",
                "importance_sampling/importance_weight_abs_delta/min",
                "importance_sampling/importance_weight_abs_delta/max",
                "importance_sampling/importance_weight_abs_delta_gt1/min",
                "importance_sampling/importance_weight_abs_delta_gt1/max",
                "importance_sampling/importance_weight_abs_delta_lt1/min",
                "importance_sampling/importance_weight_abs_delta_lt1/max",
            }
            hidden_prefix_metrics = (
                "clip/dual_clipped_tokens/",
                "clip/dual_clip_ratio/",
            )
            item = {
                k: v
                for k, v in item.items()
                if not (
                    k in hidden_exact_metrics
                    or any(k.startswith(prefix) for prefix in hidden_prefix_metrics)
                )
            }

            # # reorg keys to make it more readable in wandb
            # for key, value in item.items():
            #     if key.startswith('ppo_actor/advantages/'):
            #         new_key = key.replace('ppo_actor/advantages/', 'advantages/')
            #         item[new_key] = value
            #     else:
            #         item[key] = value

            logger.info(f"Stats ({i + 1}/{len(data)}):")
            self.print_stats(item)
            wandb.log(item, step=log_step + i)
            swanlab.log(item, step=log_step + i)
            if self.summary_writer is not None:
                for key, val in item.items():
                    self.summary_writer.add_scalar(f"{key}", val, log_step + i)
        self._last_commit_step = log_step + len(data) - 1

    def print_stats(self, stats: dict[str, float]):
        logger.info("\n" + tabulate_stats(stats))

    @staticmethod
    def get_wandb_path(config: StatsLoggerConfig):
        if config.wandb.wandb_dir:
            path = os.path.expanduser(config.wandb.wandb_dir)
            os.makedirs(path, exist_ok=True)
            return path
        return StatsLogger.get_log_path(config)

    @staticmethod
    def get_log_path(config: StatsLoggerConfig):
        path = f"{config.fileroot}/logs/{getpass.getuser()}/{config.experiment_name}/{config.trial_name}"
        os.makedirs(path, exist_ok=True)
        return path
