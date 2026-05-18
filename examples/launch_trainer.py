"""Shared AstraFlow v2 GRPO training entrypoint for examples.

Uses AstraFlowPPOTrainer which communicates with the AstraFlow HTTP
service instead of embedding orchestration logic. The AstraFlow service
handles data acquisition, buffering, pause/resume, eval, and version
management.

Requires:
  - ASTRAFLOW_URL: AstraFlow HTTP service URL
  - ASTRAFLOW_RAAS_URL: RaaS inference service URL
"""

import sys

from datasets import Dataset

from astraflow.train_worker.api.cli_args import GRPOConfig, load_expr_config
from astraflow.train_worker.trainer.ppo_trainer import AstraFlowPPOTrainer


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)

    n_dummy = config.train_batch_size * min(config.total_train_epochs, 100)
    train_dataset = Dataset.from_dict({
        "messages": [[{"role": "user", "content": "dummy"}]] * n_dummy,
    })

    with AstraFlowPPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=None,
    ) as trainer:
        trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
