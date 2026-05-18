"""Buffered trainer entrypoint for ASearcher."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

from datasets import load_dataset
from omegaconf import OmegaConf

from areal.api.cli_args import (
    TrainDatasetConfig,
    parse_cli_args,
    save_config,
    to_structured_cfg,
)
from areal.experimental.trainer import BufferedPPOTrainer
from areal.utils import name_resolve
from areal.utils.stats_logger import StatsLogger

from astraEnv.ASearcher.ASearcher.train.asearcher import ASearcherWorkflow, AgentRLConfig

@dataclass
class AgentBufferedRLConfig(AgentRLConfig):
    # ASearcher configs omit train_dataset.type. Default to RL here.
    train_dataset: TrainDatasetConfig = field(
        default_factory=lambda: TrainDatasetConfig(path="", type="rl")
    )


def get_search_dataset(dataset_path: str):
    return load_dataset(path="json", split="train", data_files=dataset_path)


def load_agent_config(argv: list[str]) -> tuple[AgentBufferedRLConfig, str]:
    cfg, config_file = parse_cli_args(argv)

    cfg_dict = OmegaConf.to_container(cfg, resolve=False)
    assert isinstance(cfg_dict, dict)

    # Backward compatibility with legacy ASearcher YAML keys.
    cfg_dict.pop("async_training", None)
    if isinstance(cfg_dict.get("actor"), dict):
        actor_cfg = cfg_dict["actor"]
        actor_cfg.pop("backend", None)
        if "group_reward_norm" in actor_cfg:
            group_reward_norm = bool(actor_cfg.pop("group_reward_norm"))
            if group_reward_norm and "reward_norm" not in actor_cfg:
                gconfig_cfg = cfg_dict.get("gconfig", {})
                group_size = int(gconfig_cfg.get("n_samples", 1))
                actor_cfg["reward_norm"] = {
                    "mean_level": "group",
                    "std_level": "group",
                    "group_size": group_size,
                }
    if isinstance(cfg_dict.get("ref"), dict):
        cfg_dict["ref"].pop("backend", None)
    if isinstance(cfg_dict.get("train_dataset"), dict):
        cfg_dict["train_dataset"].setdefault("type", "rl")

    cfg = OmegaConf.create(cfg_dict)
    cfg = to_structured_cfg(cfg, AgentBufferedRLConfig)
    cfg = OmegaConf.to_object(cfg)
    assert isinstance(cfg, AgentBufferedRLConfig)

    name_resolve.reconfigure(cfg.cluster.name_resolve)
    if os.getenv("RANK", "0") == "0":
        save_config(cfg, StatsLogger.get_log_path(cfg.stats_logger))

    return cfg, str(config_file)


def main(args: list[str]) -> None:
    config, _ = load_agent_config(args)
    train_dataset = get_search_dataset(config.train_dataset.path)

    with BufferedPPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=None,
        buffer_size=65536,
        rollout_batch_size=1,
    ) as trainer:
        workflow = ASearcherWorkflow(
            gconfig=config.gconfig.new_with_stop_and_pad_token_ids(trainer.tokenizer),
            tokenizer=trainer.tokenizer,
            dump_dir=os.path.join(
                StatsLogger.get_log_path(config.stats_logger),
                "generated",
            ),
            dataset_path=config.train_dataset.path,
            max_turns=config.max_turns,
            n_trajs=config.n_trajs,
            search_client_type=config.search_client_type,
            reward_type=config.reward_type,
            topk=config.topk,
            valid_inst_ratio=config.valid_inst_ratio,
            max_tokens=config.actor.mb_spec.max_tokens_per_mb,
        )

        trainer.train(workflow)


if __name__ == "__main__":
    main(sys.argv[1:])
