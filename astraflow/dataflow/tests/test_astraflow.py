import time

import torch

from astraflow import AstraFlow


class _DummyLoader:
    batch_size = 2


class _FakeRollout:
    def __init__(self):
        self._version = 1
        self._paused = False

    def prepare_batch(
        self,
        dataloader,
        workflow_spec=None,
        **kwargs,
    ):
        if self._paused:
            time.sleep(0.05)
        version = self._version
        self._version += 1
        return {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long),
            "attention_mask": torch.ones((2, 3), dtype=torch.long),
            "rewards": torch.tensor([1.0, 2.0], dtype=torch.float32),
            "versions": torch.tensor([[version, version, version]] * 2),
        }

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False


def test_astraflow_producer_and_buffer_path():
    rollout = _FakeRollout()
    flow = AstraFlow(
        rollout=rollout,
        rollout_dataloader=_DummyLoader(),
        workflow_spec={},
        buffer_size=32,
        replay_max_size=32,
    )
    try:
        flow.start()

        deadline = time.time() + 5.0
        while flow.size() < 4 and time.time() < deadline:
            time.sleep(0.05)
        assert flow.size() >= 4

        batch_result = flow.get_batch(batch_size=2, timeout=1.0, current_version=10)
        assert batch_result is not None
        batch, metadata = batch_result
        assert batch["input_ids"].shape[0] == 2
        assert len(metadata) == 2
        assert flow.replay_size() == 2

        replay_result = flow.get_replay_batch(batch_size=1, current_version=11)
        assert replay_result is not None
        replay_batch, replay_metadata = replay_result
        assert replay_batch["input_ids"].shape[0] == 1
        assert len(replay_metadata) == 1
        assert "train_versions" in replay_metadata[0]

        metrics = flow.get_metrics(reset=False)
        assert metrics["producer_batches"] > 0
        assert metrics["accepted"] > 0

        flow.pause()
        assert flow.get_metrics(reset=False)["is_paused"] is True
        flow.resume()
        assert flow.get_metrics(reset=False)["is_paused"] is False
    finally:
        flow.close()
        assert flow.is_closed()
