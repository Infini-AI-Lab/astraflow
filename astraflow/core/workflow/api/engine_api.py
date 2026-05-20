"""Minimal InferenceEngine protocol for workflow package.

Only the methods needed by workflows: agenerate(), get_version(), managed_session().
Uses typing.Protocol for structural subtyping — any class with these methods
satisfies the interface without needing to inherit from this class.

Also provides ``EngineGroup`` for multi-model workflows: a dict-like container
of named engines that itself satisfies the ``InferenceEngine`` protocol by
delegating to a default engine.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Protocol, runtime_checkable

from astraflow.core.workflow.api.io_struct import ModelRequest, ModelResponse


@runtime_checkable
class InferenceEngine(Protocol):
    """Subset of the full InferenceEngine interface used by workflows.

    Implemented as a Protocol so that the train_worker and raas engine
    classes satisfy this structurally without inheritance.
    """

    async def agenerate(self, req: ModelRequest) -> ModelResponse: ...

    def get_version(self) -> int: ...

    def set_version(self, version: int) -> None: ...

    @asynccontextmanager
    async def managed_session(self):
        yield


class EngineGroup:
    """Named collection of inference engines for multi-model workflows.

    Implements the ``InferenceEngine`` protocol by delegating to a default
    engine, so existing single-model workflows work without modification.

    Multi-model workflows access specific engines via ``engine["model0"]``.

    Parameters
    ----------
    engines : dict[str, InferenceEngine]
        Mapping from model id (e.g. ``"model0"``) to engine instance.
    default_id : str | None
        Key of the default engine.  Falls back to the first key in *engines*.
    """

    def __init__(
        self,
        engines: dict[str, InferenceEngine],
        default_id: str | None = None,
    ):
        if not engines:
            raise ValueError("EngineGroup requires at least one engine")
        self._engines = engines
        self._default_id = default_id or next(iter(engines))
        if self._default_id not in self._engines:
            raise KeyError(
                f"default_id {self._default_id!r} not in engines: "
                f"{list(self._engines)}"
            )

    # -- dict-like access --------------------------------------------------

    def __getitem__(self, model_id: str) -> InferenceEngine:
        return self._engines[model_id]

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._engines

    def __len__(self) -> int:
        return len(self._engines)

    def keys(self):
        return self._engines.keys()

    @property
    def default(self) -> InferenceEngine:
        """Return the default engine."""
        return self._engines[self._default_id]

    # -- InferenceEngine protocol (delegate to default) --------------------

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        return await self.default.agenerate(req)

    def get_version(self) -> int:
        return self.default.get_version()

    def set_version(self, version: int) -> None:
        """Set version on **all** engines (global version sync)."""
        for engine in self._engines.values():
            engine.set_version(version)

    @asynccontextmanager
    async def managed_session(self):
        """No-op; individual engines manage their own sessions."""
        yield
