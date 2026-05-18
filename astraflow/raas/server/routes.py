from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

import logging as _stdlib_logging

from astraflow.raas.utils import logging

from .manager import RaaS3Manager
from .protocol import dumps_object, loads_object

_base_logger = logging.getLogger(__name__)
logger = _base_logger  # replaced with LoggerAdapter after engine_id is known


def _decode_request(request_bytes: bytes) -> dict[str, Any]:
    """Decode and validate an RPC payload serialized as an object dict."""
    payload = loads_object(request_bytes)
    if not isinstance(payload, dict):
        raise ValueError("RaaS payload must be a dictionary.")
    return payload


def _encode_ok(result: Any) -> Response:
    """Encode a successful RPC response in binary payload format."""
    return Response(
        content=dumps_object({"ok": True, "result": result}),
        media_type="application/octet-stream",
    )


def _encode_error(exc: Exception) -> Response:
    """Encode an exception response in binary payload format."""
    return Response(
        content=dumps_object({"ok": False, "error": repr(exc)}),
        media_type="application/octet-stream",
        status_code=500,
    )


def _update_logger_with_engine_id(manager: RaaS3Manager) -> None:
    """Replace module-level logger with a LoggerAdapter that prefixes engine-id."""
    global logger
    eid = getattr(manager, "_engine_id", None)
    if eid and not isinstance(logger, _stdlib_logging.LoggerAdapter):
        adapter = _stdlib_logging.LoggerAdapter(_base_logger, {})
        adapter.process = lambda msg, kw, _eid=eid: (f"[{_eid}] {msg}", kw)
        logger = adapter


def build_app(manager: RaaS3Manager | None = None) -> FastAPI:
    """Build the FastAPI app exposing RaaS3 lifecycle endpoints."""
    app = FastAPI()
    app.state.manager = manager or RaaS3Manager()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @app.get("/status")
    async def status() -> dict[str, Any]:
        # Must be async so it runs directly on the event loop — never
        # competing for a thread in the default executor.  When weight
        # updates occupy executor threads (pull, pause, load, resume),
        # a sync handler would queue behind them, causing heartbeat
        # timeouts and eventual pool deregistration.
        return app.state.manager.get_status()

    @app.get("/availability")
    async def availability():
        result = await app.state.manager.get_availability()
        return result

    # ------------------------------------------------------------------
    # Workflow registration
    # ------------------------------------------------------------------

    @app.post("/register_workflow")
    async def register_workflow(request: Request):
        try:
            logger.info("RaaS3 route /register_workflow: request received, decoding...")
            payload = _decode_request(await request.body())
            logger.info(
                "RaaS3 route /register_workflow: decoded payload keys=%s, "
                "workflow_id=%r, workflow_cls=%r, reward_fn=%r",
                list(payload.keys()),
                payload.get("workflow_id"),
                payload.get("workflow_cls"),
                payload.get("reward_fn"),
            )
            result = app.state.manager.register_workflow(
                workflow_id=payload["workflow_id"],
                workflow_cls=payload["workflow_cls"],
                reward_fn=payload.get("reward_fn"),
                gconfig_overrides=payload.get("gconfig_overrides"),
                **payload.get("workflow_kwargs", {}),
            )
            logger.info("RaaS3 route /register_workflow: success, result=%s", result)
            return _encode_ok(result)
        except Exception as exc:
            logger.exception("RaaS3 register_workflow failed")
            return _encode_error(exc)

    # ------------------------------------------------------------------
    # Submit / Pull
    # ------------------------------------------------------------------

    @app.post("/submit")
    async def submit(request: Request):
        try:
            body = await request.body()
            payload = _decode_request(body)
            task_id = await app.state.manager.submit(
                data=payload["data"],
                workflow_id=payload.get("workflow_id", "default"),
            )
            return _encode_ok({"task_id": task_id})
        except Exception as exc:
            logger.exception("RaaS3 submit failed")
            return _encode_error(exc)

    @app.post("/pull")
    async def pull(request: Request):
        try:
            payload = _decode_request(await request.body())
            results = await app.state.manager.pull_completed(
                max_items=payload.get("max_items", 256),
                timeout=payload.get("timeout", 0.0),
            )
            return _encode_ok(results)
        except Exception as exc:
            logger.exception("RaaS3 pull failed")
            return _encode_error(exc)

    # ------------------------------------------------------------------
    # Training-engine reset (called before each eval window)
    # ------------------------------------------------------------------

    @app.post("/reset_training_engine")
    async def reset_training_engine(request: Request):
        logger.info(
            "RaaS3 route /reset_training_engine: request received"
        )
        try:
            payload = _decode_request(await request.body()) or {}
            timeout = float(payload.get("timeout", 5.0))
            result = await app.state.manager.reset_training_engine(
                timeout=timeout
            )
            logger.info(
                "RaaS3 route /reset_training_engine: completed, result=%s",
                result,
            )
            return _encode_ok(result)
        except Exception as exc:
            logger.exception("RaaS3 reset_training_engine failed")
            return _encode_error(exc)

    # ------------------------------------------------------------------
    # Eval Lifecycle
    # ------------------------------------------------------------------

    @app.post("/eval_start")
    async def eval_start():
        logger.info(
            "RaaS3 route /eval_start: request received, dispatching to manager..."
        )
        try:
            result = await app.state.manager.eval_start()
            logger.info("RaaS3 route /eval_start: completed, result=%s", result)
            return _encode_ok(result)
        except Exception as exc:
            logger.exception("RaaS3 eval_start failed")
            return _encode_error(exc)

    @app.post("/eval_end")
    async def eval_end():
        logger.info("RaaS3 route /eval_end: request received")
        try:
            result = await app.state.manager.eval_end()
            logger.info("RaaS3 route /eval_end: completed")
            return _encode_ok(result)
        except Exception as exc:
            logger.exception("RaaS3 eval_end failed")
            return _encode_error(exc)

    @app.post("/eval_submit")
    async def eval_submit(request: Request):
        try:
            payload = _decode_request(await request.body())
            task_id = await app.state.manager.eval_submit(
                data=payload["data"],
                workflow_id=payload.get("workflow_id", "default"),
            )
            return _encode_ok({"task_id": task_id})
        except Exception as exc:
            logger.exception("RaaS3 eval_submit failed")
            return _encode_error(exc)

    @app.post("/eval_pull")
    async def eval_pull(request: Request):
        try:
            payload = _decode_request(await request.body())
            results = await app.state.manager.eval_pull(
                max_items=payload.get("max_items", 256),
                timeout=payload.get("timeout", 0.0),
            )
            return _encode_ok(results)
        except Exception as exc:
            logger.exception("RaaS3 eval_pull failed")
            return _encode_error(exc)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Weight Updates (TCP path)
    # ------------------------------------------------------------------

    @app.post("/notify_version")
    async def notify_version(request: Request):
        """Per-model version notification from AstraFlow.

        Payload: {model_id: str, version: int, sender_endpoint: "host:port"}
        RaaS pulls weights for this single model and loads into its engine.
        """
        logger.info("RaaS3 route /notify_version: request received")
        try:
            payload = _decode_request(await request.body())
            result = await app.state.manager.notify_version(
                model_id=payload["model_id"],
                version=payload["version"],
                sender_endpoint=payload["sender_endpoint"],
            )
            logger.info("RaaS3 route /notify_version: completed (model=%s)", payload["model_id"])
            return _encode_ok(result)
        except Exception as exc:
            logger.exception("RaaS3 notify_version failed")
            return _encode_error(exc)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    @app.post("/shutdown")
    async def shutdown():
        """Gracefully shut down the RaaS server.

        Called by the trainer when training completes. Destroys all engines
        and terminates the uvicorn process.
        """
        import asyncio
        import os as _os

        logger.info("Shutdown requested — destroying all engines ...")
        print("=" * 60, flush=True)
        print("RaaS shutdown requested — destroying all engines ...", flush=True)
        print("=" * 60, flush=True)

        async def _destroy_and_exit():
            """Destroy engines then hard-exit, with a timeout safety net."""
            import signal
            import multiprocessing

            try:
                await asyncio.wait_for(
                    app.state.manager.destroy(), timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.warning("RaaS destroy timed out after 30s, forcing exit")
            except Exception:
                logger.exception("Error during RaaS shutdown destroy")
            # Kill any remaining child processes spawned by multiprocessing.
            for child in multiprocessing.active_children():
                try:
                    child.kill()
                except Exception:
                    pass
            _os._exit(0)

        asyncio.ensure_future(_destroy_and_exit())
        return _encode_ok("shutting down")

    @app.on_event("shutdown")
    async def shutdown_event():
        await app.state.manager.destroy()

    return app
