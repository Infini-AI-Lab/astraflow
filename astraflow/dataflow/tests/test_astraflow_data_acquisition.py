from __future__ import annotations

import time
from typing import Any

import pytest
import torch

from astraflow.dataflow.data_acquisition import AstraDataAcquisition


def _make_trajectory(v: int) -> dict[str, Any]:
    seq = {
        "input_ids": torch.tensor([[v, v + 1]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        "rewards": torch.tensor([1.0], dtype=torch.float32),
        "versions": torch.tensor([[1, 1]], dtype=torch.long),
    }
    return {
        "n_trajs": 1,
        "rewards": torch.tensor([1.0], dtype=torch.float32),
        "trajectories": [{"sequences": [seq]}],
    }


class _DummyLoader:
    sampler = None

    def __iter__(self):
        yield [{"x": 1}, {"x": 2}]


class _FakeRaaSRollout:
    def __init__(self):
        self._capacity = 8
        self._next_task_id = 1
        self._inflight = 0
        self._completed: list[dict[str, Any]] = []
        self.prepare_batch_called = False

    def get_raas_availability(self) -> dict[str, int]:
        return {
            "available": max(0, self._capacity - self._inflight),
            "inflight": self._inflight,
        }

    def submit_auto(self, data, workflow_spec=None, **kwargs):
        del kwargs
        task_id = self._next_task_id
        self._next_task_id += 1
        self._inflight += 1
        self._completed.append(
            {
                "task_id": task_id,
                "ok": True,
                "result": _make_trajectory(int(data["x"])),
                "error": None,
            }
        )
        return task_id

    def pull_completed(
        self, max_items: int = 256, timeout: float = 0.0
    ) -> list[dict[str, Any]]:
        del timeout
        n = min(max_items, len(self._completed))
        if n <= 0:
            return []
        results = self._completed[:n]
        del self._completed[:n]
        self._inflight = max(0, self._inflight - len(results))
        return results

    def prepare_batch(self, *args, **kwargs):
        del args, kwargs
        self.prepare_batch_called = True
        raise AssertionError("prepare_batch should not be called in RaaS service mode")


def test_astraflow_acquisition_uses_raas_service_mode():
    rollout = _FakeRaaSRollout()
    published: list[dict[str, Any]] = []

    def _publish(
        batch: dict[str, Any], metadata: dict[str, Any] | None, timeout: float | None
    ):
        del metadata, timeout
        published.append(batch)
        return True

    acquisition = AstraDataAcquisition(
        rollout=rollout,
        rollout_dataloader=_DummyLoader(),
        workflow_spec={"workflow_cls": "module.Workflow"},
        publish_fn=_publish,
    )
    acquisition.start()
    assert (
        acquisition._submit_thread is not None and acquisition._submit_thread.is_alive()
    )
    assert (
        acquisition._collect_thread is not None
        and acquisition._collect_thread.is_alive()
    )
    time.sleep(0.3)
    acquisition.stop(timeout=5.0)

    assert rollout.prepare_batch_called is False
    assert len(published) > 0
    stats = acquisition.get_stats(reset=False)
    assert stats["producer_batches"] > 0


def test_astraflow_reward_stats_pre_post_filter_and_reset():
    def _publish(
        batch: dict[str, Any], metadata: dict[str, Any] | None, timeout: float | None
    ):
        del metadata, timeout
        return True

    # Structured filter receives ({}, metadata) where metadata has "zero_adv".
    # zero_adv=0 means rewards vary (keep); zero_adv=1 means all same (filter).
    acquisition = AstraDataAcquisition(
        rollout=_FakeRaaSRollout(),
        rollout_dataloader=_DummyLoader(),
        workflow_spec={"workflow_cls": "module.Workflow"},
        filter_fn=lambda _example, metadata: metadata.get("zero_adv", 0) == 0,
        publish_fn=_publish,
    )
    seq1 = {
        "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
        "rewards": torch.tensor([0.1], dtype=torch.float32),
        "versions": torch.tensor([[1, 1]], dtype=torch.long),
    }
    seq2 = {
        "input_ids": torch.tensor([[3, 4]], dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
        "rewards": torch.tensor([0.9], dtype=torch.float32),
        "versions": torch.tensor([[1, 1]], dtype=torch.long),
    }
    # Two trajs with different rewards → zero_adv=0 → filter keeps them.
    batch = {
        "n_trajs": 2,
        "rewards": torch.tensor([0.1, 0.9], dtype=torch.float32),
        "trajectories": [
            {"sequences": [seq1]},
            {"sequences": [seq2]},
        ],
    }
    acquisition._ingest_one_result(batch)

    stats = acquisition.get_reward_stats(reset=False)
    assert int(stats["pre_filter_reward_count"]) == 2
    assert float(stats["pre_filter_reward_sum"]) == pytest.approx(1.0)
    assert float(stats["pre_filter_reward_min"]) == pytest.approx(0.1)
    assert float(stats["pre_filter_reward_max"]) == pytest.approx(0.9)
    assert int(stats["post_filter_reward_count"]) == 2
    assert float(stats["post_filter_reward_sum"]) == pytest.approx(1.0)
    assert float(stats["post_filter_reward_min"]) == pytest.approx(0.1)
    assert float(stats["post_filter_reward_max"]) == pytest.approx(0.9)

    acquisition.get_reward_stats(reset=True)
    reset_stats = acquisition.get_reward_stats(reset=False)
    assert int(reset_stats["pre_filter_reward_count"]) == 0
    assert int(reset_stats["post_filter_reward_count"]) == 0
