"""Eval orchestration for the AstraFlow service.

Runs eval through a
RaaS inference engine: eval_start → submit all items → collect results →
eval_end → compute per-dataset eval metrics.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any

import torch

logger = logging.getLogger(__name__)


class EvalManager:
    """Eval orchestration for the AstraFlow service.

    Manages eval datasets, workflow specs, and runs eval through a RaaS
    inference engine when triggered by a trainer's ``notify_version`` call.
    """

    def __init__(self, timeout: float | None = None):
        self.timeout = timeout
        # Per-agent eval configuration: {agent_name: [(name, k, dataset, wf_spec), ...]}
        self._agent_eval_runs: dict[str, list[tuple[str, int, Any, Any]]] = {}
        self._eval_running: bool = False
        self._agent_eval_ks: dict[str, dict[str, int]] = {}

    def configure_agent(
        self,
        agent_name: str,
        eval_datasets: dict[str, tuple[Any, int, Any]] | None = None,
    ) -> None:
        """Register eval datasets and workflow specs for an agent.

        Parameters
        ----------
        agent_name : str
            Agent identifier.
        eval_datasets : dict[str, tuple[Dataset, int, workflow_spec]] | None
            Mapping of dataset_name → (dataset, k, workflow_spec) where k is
            the number of eval runs per dataset used for repeated-sampling
            metrics, and workflow_spec is the resolved workflow configuration
            for this dataset.
        """
        if eval_datasets is None:
            self._agent_eval_runs[agent_name] = []
            self._agent_eval_ks[agent_name] = {}
            return

        runs: list[tuple[str, int, Any, Any]] = []
        ks: dict[str, int] = {}
        for name, (dataset, k, wf) in eval_datasets.items():
            ks[name] = k
            for run_idx in range(k):
                runs.append((name, run_idx, dataset, wf))

        self._agent_eval_runs[agent_name] = runs
        self._agent_eval_ks[agent_name] = ks

    def run_eval(
        self,
        agent_name: str,
        raas_engine: Any,
        create_dataloader_fn: Any | None = None,
    ) -> dict[str, Any]:
        """Run eval through RaaS. Blocking.

        Parameters
        ----------
        agent_name : str
            Agent identifier.
        raas_engine : RaaS2InferenceEngine
            The RaaS engine to submit eval tasks to.
        create_dataloader_fn : callable | None
            Function to create a dataloader from a dataset. If None, the
            dataset is iterated directly.

        Returns
        -------
        dict[str, Any]
            Eval results with per-dataset mean success rate and pass@k.
        """
        runs = self._agent_eval_runs.get(agent_name, [])
        if not runs:
            print(f"[EvalManager] no eval runs configured for agent {agent_name}", flush=True)
            return {}

        print(f"[EvalManager] eval_start for agent {agent_name}", flush=True)
        self._eval_running = True
        raas_engine.eval_start()

        try:
            return self._run_eval_inner(agent_name, raas_engine, runs, create_dataloader_fn)
        finally:
            raas_engine.eval_end()
            self._eval_running = False
            print(f"[EvalManager] eval_end for agent {agent_name}", flush=True)

    def _run_eval_inner(
        self,
        agent_name: str,
        raas_engine: Any,
        runs: list[tuple[str, int, Any, Any]],
        create_dataloader_fn: Any | None,
    ) -> dict[str, Any]:
        # Submit all items across all datasets
        task_id_to_run: dict[int, tuple[str, int, int]] = {}
        total_submitted = 0

        for name, run_idx, dataset, wf in runs:
            if create_dataloader_fn is not None:
                dl = create_dataloader_fn(dataset)
            else:
                dl = dataset

            sample_idx = 0
            for data in dl:
                if isinstance(data, list):
                    for item in data:
                        task_id = raas_engine.submit(item, wf)
                        task_id_to_run[task_id] = (name, run_idx, sample_idx)
                        total_submitted += 1
                        sample_idx += 1
                else:
                    task_id = raas_engine.submit(data, wf)
                    task_id_to_run[task_id] = (name, run_idx, sample_idx)
                    total_submitted += 1
                    sample_idx += 1

        print(
            f"[EvalManager] [{agent_name}] submitted {total_submitted} items "
            f"across {len(runs)} run(s), waiting...",
            flush=True,
        )

        # Collect all results
        results = raas_engine.wait(total_submitted, timeout=self.timeout)
        print(
            f"[EvalManager] [{agent_name}] collected {len(results)} results",
            flush=True,
        )

        return self._aggregate_results(agent_name, results, task_id_to_run)

    @staticmethod
    def _compute_pass_at_k(num_attempts: int, num_correct: int, k: int) -> float | None:
        """Return the standard pass@k estimator for one sample."""
        if num_attempts <= 0:
            return None
        effective_k = min(k, num_attempts)
        if effective_k <= 0 or num_correct <= 0:
            return 0.0
        if num_attempts - num_correct < effective_k:
            return 1.0
        total = math.comb(num_attempts, effective_k)
        misses = math.comb(num_attempts - num_correct, effective_k)
        return 1.0 - (misses / total)

    def _aggregate_results(
        self,
        agent_name: str,
        results: list[dict[str, Any] | None],
        task_id_to_run: dict[int, tuple[str, int, int]],
    ) -> dict[str, Any]:
        """Aggregate per-run rewards into per-dataset mean success rate and pass@k."""
        run_stats: dict[tuple[str, int], dict[str, Any]] = defaultdict(
            lambda: {"total": 0, "correct": 0, "reward_sum": 0.0}
        )
        sample_stats: dict[tuple[str, int, int], dict[str, int]] = defaultdict(
            lambda: {"attempts": 0, "correct": 0}
        )
        dataset_ks = self._agent_eval_ks.get(agent_name, {})

        n_skipped = 0
        for r in results:
            tid = r.get("task_id") if r is not None else None
            key = task_id_to_run.get(tid) if tid is not None else None
            if key is None:
                n_skipped += 1
                continue
            if r is None or not r.get("ok", False):
                n_skipped += 1
                continue
            result_data = r.get("result")
            if not isinstance(result_data, dict):
                n_skipped += 1
                continue
            rewards = result_data.get("rewards")
            if rewards is not None and torch.is_tensor(rewards):
                rewards_flat = rewards.flatten().float()
                # Use eval_correct (binary) for pass@k when available.
                # This ensures workflows with continuous rewards (e.g.
                # code_actor_and_verify where reward = pass_rate × 2.0)
                # are evaluated on the same "all tests pass" standard
                # as binary-reward workflows.
                eval_correct = result_data.get("eval_correct")
                if eval_correct is not None and torch.is_tensor(eval_correct):
                    correct_flat = eval_correct.flatten().float()
                else:
                    correct_flat = (rewards_flat > 0).float()
                name, run_idx, sample_idx = key
                s = run_stats[(name, run_idx)]
                s["total"] += rewards_flat.numel()
                s["correct"] += int(correct_flat.sum().item())
                s["reward_sum"] += float(rewards_flat.sum().item())
                for local_idx, (reward, correct) in enumerate(
                    zip(rewards_flat.tolist(), correct_flat.tolist())
                ):
                    sample_key = (name, sample_idx, local_idx)
                    sample_stat = sample_stats[sample_key]
                    sample_stat["attempts"] += 1
                    sample_stat["correct"] += int(correct > 0)
            else:
                n_skipped += 1
        if n_skipped:
            logger.warning(
                "EvalManager [%s]: %d/%d results skipped during aggregation",
                agent_name, n_skipped, len(results),
            )

        # Aggregate per-dataset
        dataset_stats: dict[str, dict[str, int | float]] = defaultdict(
            lambda: {
                "total": 0,
                "correct": 0,
                "reward_sum": 0.0,
                "sample_count": 0,
                "pass_at_k_sum": 0.0,
            }
        )
        for (name, run_idx), s in sorted(run_stats.items()):
            ds = dataset_stats[name]
            ds["total"] += s["total"]
            ds["correct"] += s["correct"]
            ds["reward_sum"] += s["reward_sum"]
        for (name, _sample_idx, _local_idx), s in sorted(sample_stats.items()):
            pass_at_k = self._compute_pass_at_k(
                num_attempts=s["attempts"],
                num_correct=s["correct"],
                k=dataset_ks.get(name, s["attempts"]),
            )
            if pass_at_k is None:
                continue
            ds = dataset_stats[name]
            ds["sample_count"] += 1
            ds["pass_at_k_sum"] += pass_at_k

        # Log per-dataset aggregate
        eval_results: dict[str, Any] = {"datasets": {}}
        for name, ds in sorted(dataset_stats.items()):
            if ds["total"] > 0:
                avg_at_k = ds["correct"] / ds["total"] * 100.0
                mean_r = ds["reward_sum"] / ds["total"]
                pass_k = dataset_ks.get(name, 1)
                pass_at_k = None
                if ds["sample_count"] > 0:
                    pass_at_k = ds["pass_at_k_sum"] / ds["sample_count"] * 100.0
                print(
                    f"[EvalManager] [{agent_name}/{name}] "
                    f"{ds['correct']}/{ds['total']} "
                    f"(avg@k={avg_at_k:.1f}%, "
                    f"pass@{pass_k}={pass_at_k:.1f}%, mean_reward={mean_r:.4f})",
                    flush=True,
                )
                eval_results["datasets"][name] = {
                    "avg@k": avg_at_k,
                    "mean_reward": mean_r,
                    "correct": ds["correct"],
                    "total": ds["total"],
                    "pass@k": pass_at_k,
                    "pass_k": pass_k,
                    "sample_count": ds["sample_count"],
                }

        # Overall average across datasets
        per_ds_avgs = []
        per_ds_passes = []
        for ds in dataset_stats.values():
            if ds["total"] > 0:
                per_ds_avgs.append(ds["correct"] / ds["total"] * 100.0)
            if ds["sample_count"] > 0:
                per_ds_passes.append(ds["pass_at_k_sum"] / ds["sample_count"] * 100.0)

        if per_ds_avgs:
            overall_avg_at_k = sum(per_ds_avgs) / len(per_ds_avgs)
            print(
                f"[EvalManager] [{agent_name}/overall] "
                f"avg across {len(per_ds_avgs)} datasets: avg@k={overall_avg_at_k:.1f}%",
                flush=True,
            )
            eval_results["overall_avg@k"] = overall_avg_at_k
        if per_ds_passes:
            overall_pass_at_k = sum(per_ds_passes) / len(per_ds_passes)
            print(
                f"[EvalManager] [{agent_name}/overall] "
                f"avg across {len(per_ds_passes)} datasets: pass@k={overall_pass_at_k:.1f}%",
                flush=True,
            )
            eval_results["overall_pass@k"] = overall_pass_at_k
        if not per_ds_avgs and not per_ds_passes:
            print(f"[EvalManager] [{agent_name}] no valid results", flush=True)

        return eval_results
