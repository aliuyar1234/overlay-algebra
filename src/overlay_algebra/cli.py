from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .artifacts import build_run_paths
from .config import AppConfig, load_app_config


def _build_parser(command_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=command_name,
        description="M0 placeholder command for the Overlay Algebra scaffold.",
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a JSON summary instead of human-readable text.",
    )
    return parser


def _render_summary(command_name: str, config: AppConfig) -> dict[str, object]:
    run_paths = build_run_paths(config)
    return {
        "status": "placeholder-ready",
        "command": command_name,
        "config_command": config.command,
        "config_path": str(config.source_path),
        "experiment_name": config.experiment_name,
        "milestone": config.milestone,
        "seed": config.seed,
        "suggested_run_dir": str(run_paths.run_dir),
    }


def run_placeholder_command(command_name: str, argv: Sequence[str] | None = None) -> int:
    parser = _build_parser(command_name)
    args = parser.parse_args(argv)

    config = load_app_config(Path(args.config))
    if config.command != command_name:
        parser.error(
            f"Config command mismatch: expected {command_name!r}, found {config.command!r}."
        )

    summary = _render_summary(command_name, config)
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(
            f"[M0 placeholder] {command_name} loaded "
            f"{config.source_path} for experiment {config.experiment_name!r}."
        )
        print(f"Suggested run directory: {summary['suggested_run_dir']}")
    return 0
