"""Simplified single-process stats tracker for workflow package.

No torch.distributed or platform dependencies.
"""

import time
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum, auto
from threading import Lock

import torch


class ReduceType(Enum):
    AVG_MIN_MAX = auto()
    AVG = auto()
    SUM = auto()
    MIN = auto()
    MAX = auto()
    SCALAR = auto()


class DistributedStatsTracker:
    def __init__(self, name: str = ""):
        self.lock = Lock()
        self.scope_stack = []
        if name:
            self.scope_stack.append(name.strip("/"))
        self.denominators = {}
        self.reduce_types = {}
        self.stats = defaultdict(list)

    def scope(self, name):
        with self.lock:
            return self.Scope(self, name)

    class Scope:
        def __init__(self, tracker, name):
            self.tracker = tracker
            self.name = name.strip("/")

        def __enter__(self):
            self.tracker.scope_stack.append(self.name)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.tracker.scope_stack.pop()

    def _get_full_key(self, key):
        if not self.scope_stack:
            return key
        return "/".join(self.scope_stack + [key])

    @contextmanager
    def record_timing(self, key):
        start_time = time.perf_counter()
        try:
            yield
        finally:
            with self.lock:
                full_key = f"timeperf/{key}"
                self._set_reduce_type(full_key, ReduceType.SCALAR)
                self.stats[full_key].append(time.perf_counter() - start_time)

    def denominator(self, **kwargs):
        with self.lock:
            for key, value in kwargs.items():
                if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
                    raise ValueError(
                        f"`{key}` must be a pytorch bool tensor: {value.dtype}"
                    )
                if value.numel() == 0:
                    raise ValueError(f"`{key}` must be non-empty")
                full_key = self._get_full_key(key)
                self._set_reduce_type(full_key, ReduceType.SUM)
                self.stats[full_key].append(value.detach().clone())

    def scalar(self, **kwargs):
        with self.lock:
            for key, value in kwargs.items():
                full_key = self._get_full_key(key)
                self._set_reduce_type(full_key, ReduceType.SCALAR)
                self.stats[full_key].append(float(value))

    def stat(
        self,
        denominator: str,
        reduce_type: ReduceType | None = None,
        **kwargs,
    ):
        with self.lock:
            for key, value in kwargs.items():
                if not isinstance(value, torch.Tensor) or value.dtype != torch.float:
                    raise ValueError(
                        f"`{key}` should be a pytorch float tensor: {value.dtype}"
                    )
                if value.numel() == 0:
                    raise ValueError(f"`{key}` should be non-empty")
                if reduce_type == ReduceType.SCALAR:
                    raise ValueError("Cannot use the scalar reduce type for a tensor")
                full_key = self._get_full_key(key)
                denorm = self._get_full_key(denominator)
                if denorm not in self.stats or not self.stats[denorm]:
                    raise ValueError(f"Denominator `{denorm}` does not exist")
                self.denominators[full_key] = denorm
                if reduce_type is None:
                    reduce_type = ReduceType.AVG_MIN_MAX
                self._set_reduce_type(full_key, reduce_type)
                self.stats[full_key].append(value.detach().clone())

    def _set_reduce_type(self, key, reduce_type):
        if not isinstance(reduce_type, ReduceType):
            raise ValueError("reduce_type must be a ReduceType enum")
        self.reduce_types[key] = reduce_type

    def export(self, key=None, reduce_group=None, reset=True) -> dict[str, float]:
        with self.lock:
            if key is not None:
                full_key = self._get_full_key(key)
                result = self._aggregate(full_key)
                if reset:
                    self.denominators.pop(full_key, None)
                    self.reduce_types.pop(full_key, None)
                    self.stats.pop(full_key, None)
                return result

            keys = list(self.stats.keys())
            results = {}
            for k in keys:
                results.update(self._aggregate(k))
            if reset:
                self.denominators = {}
                self.reduce_types = {}
                self.stats = defaultdict(list)
            results = {
                k: v.cpu().item() if torch.is_tensor(v) else v
                for k, v in results.items()
            }
            return results

    def _aggregate(self, key):
        reduce_type = self.reduce_types.get(key, ReduceType.SCALAR)
        result = {}

        if reduce_type == ReduceType.SCALAR:
            values = self.stats.get(key, [])
            if values:
                value = sum(values)
                cnt = len(values)
                result[key] = float(value / cnt)
                result[key + "__count"] = int(cnt)
        elif reduce_type == ReduceType.AVG_MIN_MAX:
            result[f"{key}/avg"] = self._avg_of(key)
            result[f"{key}/min"] = self._min_of(key)
            result[f"{key}/max"] = self._max_of(key)
        elif reduce_type == ReduceType.AVG:
            result[key] = self._avg_of(key)
        elif reduce_type == ReduceType.SUM:
            result[key] = self._sum_of(key)
        elif reduce_type == ReduceType.MIN:
            result[key] = self._min_of(key)
        elif reduce_type == ReduceType.MAX:
            result[key] = self._max_of(key)

        keys_to_pop = [k for k, v in result.items() if v is None]
        for k in keys_to_pop:
            result.pop(k)
        return result

    def _sum_of(self, key):
        values = self.stats.get(key, [])
        if key not in self.denominators:
            return float(sum(x.sum() for x in values))
        denominator = self.denominators[key]
        xs = []
        for v, d in zip(values, self.stats.get(denominator, [])):
            xs.append(torch.where(d, v, 0.0).sum())
        return float(sum(xs))

    def _avg_of(self, key):
        values = self.stats.get(key, [])
        denominator = self.denominators[key]
        xs, ds = [], []
        for v, d in zip(values, self.stats.get(denominator, [])):
            xs.append(torch.where(d, v, 0.0).sum())
            ds.append(d.sum())
        x = sum(xs)
        d = sum(ds)
        if d == 0:
            return None
        return float(x / d)

    def _min_of(self, key):
        values = self.stats.get(key, [])
        denominator = self.denominators[key]
        xs = []
        for v, d in zip(values, self.stats.get(denominator, [])):
            xs.append(torch.where(d, v, float("inf")).min())
        x = min(xs) if xs else float("inf")
        if torch.is_tensor(x) and torch.isinf(x):
            return None
        return float(x)

    def _max_of(self, key):
        values = self.stats.get(key, [])
        denominator = self.denominators[key]
        xs = []
        for v, d in zip(values, self.stats.get(denominator, [])):
            xs.append(torch.where(d, v, -float("inf")).max())
        x = max(xs) if xs else -float("inf")
        if torch.is_tensor(x) and torch.isinf(x):
            return None
        return float(x)


DEFAULT_TRACKER = DistributedStatsTracker()
stat = DEFAULT_TRACKER.stat
denominator = DEFAULT_TRACKER.denominator
export = DEFAULT_TRACKER.export
scope = DEFAULT_TRACKER.scope
scalar = DEFAULT_TRACKER.scalar
record_timing = DEFAULT_TRACKER.record_timing

TRACKERS = {"": DEFAULT_TRACKER}
LOCK = Lock()


def get(name: str = ""):
    global TRACKERS, LOCK
    with LOCK:
        if name not in TRACKERS:
            TRACKERS[name] = DistributedStatsTracker(name)
        return TRACKERS[name]


def export_all(reduce_group=None, reset=True) -> dict[str, float]:
    stat = {}
    for tracker_key in list(TRACKERS.keys()):
        tracker = get(tracker_key)
        x = tracker.export(reset=reset)
        stat.update(x)
    return stat
