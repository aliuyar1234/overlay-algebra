from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .config import AppConfig


_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(slots=True)
class RunPaths:
    run_dir: Path
    adapter_dir: Path
    checkpoints_dir: Path
    latest_checkpoint: Path
    heartbeat: Path
    control_dir: Path
    pause_request: Path
    resolved_config: Path
    run_manifest: Path
    metrics: Path
    predictions: Path
    stdout: Path
    environment: Path
    tuning_manifest: Path


def _slugify(value: str) -> str:
    slug = _SLUG_RE.sub("_", value.lower()).strip("_")
    return slug or "run"


def build_run_paths(config: AppConfig, run_id: str | None = None) -> RunPaths:
    if run_id is None:
        run_id = f"oa_{_slugify(config.milestone)}_{_slugify(config.experiment_name)}_{config.seed}"

    run_dir = Path(config.artifacts.root_dir) / run_id
    return RunPaths(
        run_dir=run_dir,
        adapter_dir=run_dir / "adapter",
        checkpoints_dir=run_dir / "checkpoints",
        latest_checkpoint=run_dir / "checkpoints" / "latest.json",
        heartbeat=run_dir / "heartbeat.json",
        control_dir=run_dir / "control",
        pause_request=run_dir / "control" / "pause.request",
        resolved_config=run_dir / config.artifacts.resolved_config_name,
        run_manifest=run_dir / "run_manifest.json",
        metrics=run_dir / config.artifacts.metrics_name,
        predictions=run_dir / config.artifacts.predictions_name,
        stdout=run_dir / config.artifacts.stdout_name,
        environment=run_dir / config.artifacts.environment_name,
        tuning_manifest=run_dir / config.artifacts.tuning_manifest_name,
    )
