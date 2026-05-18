from __future__ import annotations

import pickle
from typing import Any

try:
    import cloudpickle  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    cloudpickle = None


def dumps_object(obj: Any) -> bytes:
    """Serialize a Python object for RaaS RPC transport."""
    if cloudpickle is not None:
        return cloudpickle.dumps(obj)
    return pickle.dumps(obj)


def loads_object(blob: bytes) -> Any:
    """Deserialize a Python object from RaaS RPC transport."""
    if cloudpickle is not None:
        return cloudpickle.loads(blob)
    return pickle.loads(blob)
