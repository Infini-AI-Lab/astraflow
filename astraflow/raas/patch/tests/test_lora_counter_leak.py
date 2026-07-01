"""Regression tests for the LoRA usage-counter release helper.

These guard the deadlock fix in ``LoRACounterLeakPatch``: the per-adapter
``ConcurrentCounter`` in SGLang's ``LoRARegistry`` must be released exactly once
per request. A double-release is *fatal* — ``ConcurrentCounter.decrement`` has no
floor, so -1 makes ``wait_for_zero`` (hence ``wait_for_unload`` and the
``max_loaded_loras`` LRU eviction) hang forever.

The helper is dependency-free (no sglang import), so these tests exercise the
idempotency logic with fakes.
"""

import asyncio
from types import SimpleNamespace

from astraflow.raas.patch.sglang import release_lora_ref_once


class _RecordingRegistry:
    def __init__(self):
        self.released = []

    async def release(self, lora_id):
        self.released.append(lora_id)


def _make_tm(rid_to_state, enable_lora=True):
    reg = _RecordingRegistry()
    tm = SimpleNamespace(
        server_args=SimpleNamespace(enable_lora=enable_lora),
        rid_to_state=rid_to_state,
        lora_registry=reg,
    )
    return tm, reg


def test_release_once_then_idempotent():
    """First call releases; a second call (rid already popped) is a no-op."""

    async def run():
        rid = "rid-abc"
        tm, reg = _make_tm({rid: object()})
        sub = SimpleNamespace(lora_path="/adapter", rid=rid, lora_id="lid-1")

        await release_lora_ref_once(tm, sub)  # releases exactly once
        await release_lora_ref_once(tm, sub)  # no-op: rid no longer tracked

        assert reg.released == ["lid-1"], reg.released
        assert rid not in tm.rid_to_state

    asyncio.run(run())


def test_no_release_when_request_already_completed():
    """If normal/scheduler-abort already released (rid absent), never release."""

    async def run():
        tm, reg = _make_tm({})  # rid_to_state already cleaned by the native path
        sub = SimpleNamespace(lora_path="/adapter", rid="rid-xyz", lora_id="lid-1")

        await release_lora_ref_once(tm, sub)

        assert reg.released == []

    asyncio.run(run())


def test_no_release_when_lora_disabled_or_no_adapter():
    """No release for non-LoRA requests (enable_lora False or lora_path unset)."""

    async def run():
        tm, reg = _make_tm({"r": object()}, enable_lora=False)
        await release_lora_ref_once(
            tm, SimpleNamespace(lora_path="/adapter", rid="r", lora_id="lid")
        )
        assert reg.released == []

        tm2, reg2 = _make_tm({"r": object()}, enable_lora=True)
        await release_lora_ref_once(
            tm2, SimpleNamespace(lora_path=None, rid="r", lora_id="lid")
        )
        assert reg2.released == []

    asyncio.run(run())


def test_concurrent_teardown_releases_once():
    """Two coroutines tearing down the same request race to release only once."""

    async def run():
        rid = "rid-race"
        tm, reg = _make_tm({rid: object()})
        sub = SimpleNamespace(lora_path="/adapter", rid=rid, lora_id="lid-1")

        await asyncio.gather(
            release_lora_ref_once(tm, sub),
            release_lora_ref_once(tm, sub),
        )

        assert reg.released == ["lid-1"], reg.released

    asyncio.run(run())


if __name__ == "__main__":
    # Standalone runner (avoids third-party pytest plugins that may be missing
    # in the inference image, e.g. hypothesis -> pkg_resources).
    import sys

    failures = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {_name}: {type(exc).__name__}: {exc}")
    print(f"{'ALL PASSED' if failures == 0 else f'{failures} FAILED'}")
    sys.exit(1 if failures else 0)
