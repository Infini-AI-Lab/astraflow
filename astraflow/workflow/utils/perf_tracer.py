"""No-op stubs for perf_tracer used in workflow package."""

from contextlib import asynccontextmanager, contextmanager


@asynccontextmanager
async def atrace_session_phase(name: str):
    """Async no-op context manager for session phase tracing."""
    yield


def session_context():
    """No-op decorator that returns the function unchanged."""

    def decorator(fn):
        return fn

    return decorator


def trace_session(name: str):
    """No-op decorator that returns the function unchanged."""

    def decorator(fn):
        return fn

    return decorator
