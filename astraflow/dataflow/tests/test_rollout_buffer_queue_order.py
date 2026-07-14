"""Tests for RolloutBuffer queue ordering (fifo vs edf) and staleness bias.

The edf order exists to remove a difficulty bias: long generations span more
weight versions, so their ``min_version`` is older on arrival and FIFO
(completion-order) consumption lets them expire in the queue far more often
than short generations. EDF consumes the most staleness-critical samples
first while preserving the exact ``max_staleness`` invariant.
"""

from __future__ import annotations

import torch

from astraflow.dataflow.rollout_buffer import RolloutBuffer


def _example(length: int, tag: int = 0) -> dict:
    return {
        "attention_mask": torch.ones(1, length, dtype=torch.long),
        "input_ids": torch.full((1, length), tag, dtype=torch.long),
        "tag": torch.tensor([tag], dtype=torch.long),
    }


def _put(buf: RolloutBuffer, length: int, min_version: int, tag: int = 0) -> None:
    buf.put(_example(length, tag=tag), metadata={"min_version": min_version})


class TestFifoParity:
    """queue_order='fifo' must reproduce the historical deque behavior."""

    def test_arrival_order(self):
        buf = RolloutBuffer(max_size=16, queue_order="fifo")
        for tag in range(5):
            _put(buf, length=4, min_version=tag, tag=tag)
        tags = [
            int(buf.get_with_metadata()[0]["tag"].item()) for _ in range(5)
        ]
        assert tags == [0, 1, 2, 3, 4]

    def test_stale_head_dropped(self):
        buf = RolloutBuffer(max_size=16, max_staleness=2, queue_order="fifo")
        _put(buf, length=4, min_version=0, tag=0)  # stale at version 5
        _put(buf, length=4, min_version=5, tag=1)
        example, _ = buf.get_with_metadata(current_version=5)
        assert int(example["tag"].item()) == 1
        stats = buf.get_and_reset_consume_stats()
        assert stats["skipped_stale"] == 1
        assert stats["consumed"] == 1

    def test_partial_batch_putback_preserves_order(self):
        # timeout=0 makes get_batch collect exactly one entry, hit the
        # deadline, and put it back (the blocking-on-empty quirk of
        # _pop_one predates this change, so a bigger shortfall would hang).
        buf = RolloutBuffer(max_size=16, queue_order="fifo")
        for tag in range(3):
            _put(buf, length=4, min_version=0, tag=tag)
        assert buf.get_batch(batch_size=5, timeout=0.0) is None
        tags = [
            int(buf.get_with_metadata()[0]["tag"].item()) for _ in range(3)
        ]
        assert tags == [0, 1, 2]

    def test_multi_entry_putback_restores_order(self):
        """White-box: pushing popped entries back restores exact order."""
        import heapq as _hq

        buf = RolloutBuffer(max_size=16, queue_order="fifo")
        for tag in range(4):
            _put(buf, length=4, min_version=0, tag=tag)
        entries = [buf._pop_one() for _ in range(3)]
        with buf._lock:
            for entry in entries:
                _hq.heappush(buf._heap, entry)
        tags = [
            int(buf.get_with_metadata()[0]["tag"].item()) for _ in range(4)
        ]
        assert tags == [0, 1, 2, 3]

    def test_eviction_drops_oldest_and_is_counted(self):
        buf = RolloutBuffer(max_size=2, queue_order="fifo")
        for tag in range(3):
            _put(buf, length=4, min_version=tag, tag=tag)
        stats = buf.get_and_reset_put_stats()
        assert stats["evicted"] == 1
        tags = [
            int(buf.get_with_metadata()[0]["tag"].item()) for _ in range(2)
        ]
        assert tags == [1, 2]


class TestEdfOrder:
    def test_consumes_oldest_min_version_first(self):
        buf = RolloutBuffer(max_size=16, queue_order="edf")
        _put(buf, length=4, min_version=7, tag=0)
        _put(buf, length=4, min_version=3, tag=1)
        _put(buf, length=4, min_version=5, tag=2)
        tags = [
            int(buf.get_with_metadata()[0]["tag"].item()) for _ in range(3)
        ]
        assert tags == [1, 2, 0]

    def test_ties_keep_arrival_order(self):
        """Samples of one group (same min_version) stay contiguous."""
        buf = RolloutBuffer(max_size=16, queue_order="edf")
        for tag in range(4):
            _put(buf, length=4, min_version=2, tag=tag)
        tags = [
            int(buf.get_with_metadata()[0]["tag"].item()) for _ in range(4)
        ]
        assert tags == [0, 1, 2, 3]

    def test_partial_batch_putback_preserves_priority(self):
        buf = RolloutBuffer(max_size=16, queue_order="edf")
        _put(buf, length=4, min_version=9, tag=0)
        _put(buf, length=4, min_version=1, tag=1)
        # timeout=0: collects the min_version=1 entry, then puts it back.
        assert buf.get_batch(batch_size=5, timeout=0.0) is None
        example, _ = buf.get_with_metadata()
        assert int(example["tag"].item()) == 1

    def test_unversioned_samples_consumed_eagerly(self):
        buf = RolloutBuffer(max_size=16, queue_order="edf")
        _put(buf, length=4, min_version=0, tag=0)
        buf.put(_example(4, tag=1), metadata={})  # no min_version
        example, _ = buf.get_with_metadata()
        assert int(example["tag"].item()) == 1

    def test_unknown_order_falls_back_to_edf(self):
        buf = RolloutBuffer(max_size=16, queue_order="lifo")
        assert buf.queue_order == "edf"

    def test_default_is_edf(self):
        assert RolloutBuffer(max_size=16).queue_order == "edf"


def _run_mixed_load(queue_order: str) -> dict[str, int]:
    """Deterministic mixed-length load with advancing weight versions.

    Per version tick v: 4 short samples arrive (generated within v,
    min_version=v) and 4 long samples arrive that STARTED 3 versions ago
    (min_version=v-3) — modelling generations that span several weight
    updates. The consumer drains 6 samples per tick, so a backlog builds and
    samples queue for multiple ticks. max_staleness=4 means shorts tolerate
    ~4 ticks of queueing while longs tolerate only ~1.
    """
    gen_span = 3
    max_staleness = 4
    buf = RolloutBuffer(
        max_size=4096, max_staleness=max_staleness, queue_order=queue_order
    )
    counts = {
        "consumed_long": 0,
        "consumed_short": 0,
        "dropped_long": 0,
        "dropped_short": 0,
        "violations": 0,
    }
    short_len, long_len = 8, 64

    for v in range(30):
        # Fresh shorts arrive first each tick, so a consuming pop always
        # finds a valid entry and never blocks on an all-stale queue.
        for _ in range(4):
            _put(buf, length=short_len, min_version=v)
        if v >= gen_span:
            for _ in range(4):
                _put(buf, length=long_len, min_version=v - gen_span)

        for _ in range(6):
            if buf.size() == 0:
                break
            result = buf.get_with_metadata(current_version=v)
            if result is None:
                break
            example, metadata = result
            length = int(example["attention_mask"].sum().item())
            is_long = length == long_len
            counts["consumed_long" if is_long else "consumed_short"] += 1
            if v - int(metadata["min_version"]) > max_staleness:
                counts["violations"] += 1
            stats = buf.get_and_reset_consume_stats()
            dropped = int(stats["skipped_stale"])
            if dropped:
                # Exact length attribution: with n dropped totalling len_sum
                # tokens, n = a + b and len_sum = long*a + short*b, so
                # a = (len_sum - short*n) / (long - short).
                dropped_len = int(stats["skipped_stale_len_sum"])
                n_long = (dropped_len - short_len * dropped) // (
                    long_len - short_len
                )
                counts["dropped_long"] += n_long
                counts["dropped_short"] += dropped - n_long
    return counts


class TestDifficultyBias:
    def test_fifo_expires_long_generations_more(self):
        fifo = _run_mixed_load("fifo")
        edf = _run_mixed_load("edf")

        # Invariant holds in both modes: nothing stale is ever consumed.
        assert fifo["violations"] == 0
        assert edf["violations"] == 0

        # The bias: FIFO drops long generations, EDF rescues them.
        assert fifo["dropped_long"] > edf["dropped_long"]
        assert edf["consumed_long"] > fifo["consumed_long"]

        # And EDF's rescue does not come from dropping shorts instead:
        # shorts have budget to wait, so total drops go down.
        total_fifo = fifo["dropped_long"] + fifo["dropped_short"]
        total_edf = edf["dropped_long"] + edf["dropped_short"]
        assert total_edf <= total_fifo


class TestStateDict:
    def test_roundtrip_preserves_order(self):
        for order in ("fifo", "edf"):
            buf = RolloutBuffer(max_size=16, queue_order=order)
            _put(buf, length=4, min_version=9, tag=0)
            _put(buf, length=4, min_version=1, tag=1)
            _put(buf, length=4, min_version=5, tag=2)
            state = buf.state_dict()

            restored = RolloutBuffer(max_size=16, queue_order=order)
            restored.load_state_dict(state)
            tags = [
                int(restored.get_with_metadata()[0]["tag"].item())
                for _ in range(3)
            ]
            expected = [0, 1, 2] if order == "fifo" else [1, 2, 0]
            assert tags == expected, (order, tags)

    def test_legacy_state_without_seq_nos(self):
        """Old deque-era checkpoints load in their stored (FIFO) order."""
        buf = RolloutBuffer(max_size=16, queue_order="fifo")
        legacy = {
            "buffer": [_example(4, tag=0), _example(4, tag=1)],
            "metadata": [{"min_version": 3}, {"min_version": 1}],
            "replay_buffer": [],
            "replay_metadata": [],
            "closed": False,
        }
        buf.load_state_dict(legacy)
        tags = [
            int(buf.get_with_metadata()[0]["tag"].item()) for _ in range(2)
        ]
        assert tags == [0, 1]
