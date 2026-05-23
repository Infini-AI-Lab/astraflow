"""Entry point: ``python -m astraflow --config experiment.yaml``

Starts the AstraFlow HTTP service.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging

from astraflow.dataflow.service import AstraFlowService, create_app
from astraflow.dataflow.service_config import AgentConfig, EvalConfig, ServiceConfig

logger = logging.getLogger(__name__)


def _parse_config(config_path: str) -> ServiceConfig:
    """Parse an experiment YAML into a ServiceConfig."""
    from astraflow.core.config.loader import load_and_merge_configs, load_dataflow_config

    raw = load_and_merge_configs([config_path])
    af = load_dataflow_config(raw)

    known_fields = {f.name for f in dataclasses.fields(AgentConfig)}
    filtered = {k: v for k, v in af["agent"].items() if k in known_fields}
    agent = AgentConfig(**filtered)

    svc_kwargs: dict = {
        "host": af["host"],
        "port": af["port"],
        "agent": agent,
        "eval": EvalConfig(),
        "checkpoint_dir": af.get("checkpoint_dir"),
    }
    if "balance_report_freq" in af:
        svc_kwargs["balance_report_freq"] = af["balance_report_freq"]
    return ServiceConfig(**svc_kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="AstraFlow HTTP service for training data orchestration",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config file",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override port from config",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Override host from config",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = _parse_config(args.config)
    if args.host is not None:
        config.host = args.host
    if args.port is not None:
        config.port = args.port

    logger.info("Starting AstraFlow service on %s:%d", config.host, config.port)

    service = AstraFlowService(config)
    service.register_agent("default", config.agent)

    print("=" * 60, flush=True)
    print("RaaS ready. Waiting for trainer to connect ...", flush=True)
    print("=" * 60, flush=True)

    flask_app = create_app(service)
    flask_app.run(
        host=config.host,
        port=config.port,
        threaded=True,
    )


if __name__ == "__main__":
    main()
