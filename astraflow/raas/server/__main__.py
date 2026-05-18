from __future__ import annotations

import argparse
import os
import socket
import uuid

import uvicorn

from .manager import RaaS3Manager
from .routes import build_app


def _generate_engine_id() -> str:
    """Generate a short random engine ID."""
    return f"raas-{uuid.uuid4().hex[:6]}"


def main() -> None:
    """CLI entrypoint for running the RaaS3 HTTP service."""
    parser = argparse.ArgumentParser(description="Rollout-as-a-Service v3 server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=19090)
    parser.add_argument("--log-level", type=str, default="info")
    parser.add_argument(
        "--config",
        type=str,
        action="append",
        default=None,
        help=(
            "YAML config path(s). Can be specified multiple times to merge "
            "configs in order (e.g. --config experiment.yaml --config raas.yaml). "
            "If set, RaaS3 bootstraps the engine and launches inference servers "
            "on startup."
        ),
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Optional Hydra-style overrides (repeatable), used with --config.",
    )
    parser.add_argument(
        "--engine-id",
        type=str,
        default=None,
        help=(
            "Engine ID for this instance. If omitted, auto-generated from "
            "hostname-port-uuid (e.g. 'myhost-19190-a1b2c3')."
        ),
    )
    parser.add_argument(
        "--astraflow-url",
        type=str,
        default=None,
        help=(
            "If set, self-register with AstraFlow service at this URL on startup "
            "(e.g. 'http://127.0.0.1:8000'). The RaaS will POST to "
            "/register_raas with its engine-id and URL."
        ),
    )
    args = parser.parse_args()

    # Auto-generate engine-id if not provided
    engine_id = args.engine_id or _generate_engine_id()
    print(f"[RaaS] engine-id: {engine_id}", flush=True)

    manager = RaaS3Manager(service_port=args.port, service_host=args.host)
    app = build_app(manager)

    # Bootstrap on uvicorn's event loop via startup event.
    if args.config:
        bootstrap_config = {
            "config_paths": args.config,
            "overrides": args.override,
            "engine_id": engine_id,
        }

        @app.on_event("startup")
        async def _bootstrap_on_startup():
            await manager.bootstrap_from_yaml(**bootstrap_config)

            # Self-register with AstraFlow if --astraflow-url is set
            if args.astraflow_url:
                import asyncio

                asyncio.ensure_future(
                    _self_register(args.astraflow_url, engine_id, args.host, args.port)
                )

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


async def _self_register(
    astraflow_url: str,
    engine_id: str,
    host: str,
    port: int,
    max_retries: int = 60,
    retry_interval: float = 5.0,
) -> None:
    """Register this RaaS instance with AstraFlow after bootstrap completes.

    Retries with fixed interval until AstraFlow is reachable, so that all
    components can be launched simultaneously without ordering constraints.
    """
    import asyncio

    import aiohttp

    # Determine the externally reachable URL for this RaaS
    if host == "0.0.0.0":
        raas_host = socket.gethostname()
    else:
        raas_host = host
    raas_url = f"http://{raas_host}:{port}"

    # Detect GPU count visible to this RaaS process.
    gpu_count = 1
    try:
        import torch as _torch

        if _torch.cuda.is_available():
            gpu_count = _torch.cuda.device_count()
    except Exception:
        pass

    # Verify local engines are healthy before registering with AstraFlow.
    # This prevents registering a RaaS whose SGLang engines died right
    # after bootstrap (e.g. GPU OOM shortly after /health passed).
    local_status_url = f"http://127.0.0.1:{port}/status"
    for check in range(1, 13):  # up to ~60s
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    local_status_url,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    data = await resp.json()
                    if data.get("status") == "ready":
                        break
                    if data.get("status") == "error":
                        print(
                            f"[RaaS] Local engine status=error, aborting registration: "
                            f"{data.get('message', '')}",
                            flush=True,
                        )
                        return
        except Exception as exc:
            print(
                f"[RaaS] Pre-registration health check {check}/12 failed: {exc}",
                flush=True,
            )
        await asyncio.sleep(5.0)
    else:
        print(
            "[RaaS] WARNING: Local engine never reached 'ready' after 60s, "
            "skipping registration.",
            flush=True,
        )
        return

    for attempt in range(1, max_retries + 1):
        print(
            f"[RaaS] Self-registering with AstraFlow at {astraflow_url} "
            f"(uid={engine_id}, url={raas_url}, gpu_count={gpu_count}, "
            f"attempt {attempt}/{max_retries}) ...",
            flush=True,
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{astraflow_url}/register_raas",
                    json={
                        "uid": engine_id,
                        "raas_url": raas_url,
                        "gpu_count": gpu_count,
                    },
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    result = await resp.json()
                    print(
                        f"[RaaS] Registered with AstraFlow: pool_size={result.get('pool_size')}",
                        flush=True,
                    )
                    return
        except Exception as exc:
            if attempt < max_retries:
                print(
                    f"[RaaS] Registration attempt {attempt} failed: {exc}. "
                    f"Retrying in {retry_interval}s ...",
                    flush=True,
                )
                await asyncio.sleep(retry_interval)
            else:
                print(
                    f"[RaaS] WARNING: Self-registration with AstraFlow failed after "
                    f"{max_retries} attempts: {exc}. "
                    f"You can manually register via: "
                    f'curl -X POST {astraflow_url}/register_raas '
                    f'-H "Content-Type: application/json" '
                    f'-d \'{{"uid": "{engine_id}", "raas_url": "{raas_url}", '
                    f'"gpu_count": {gpu_count}}}\'',
                    flush=True,
                )


if __name__ == "__main__":
    main()
