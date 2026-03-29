from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(slots=True)
class ArtifactLayout:
    root_dir: str = "runs"
    resolved_config_name: str = "resolved_config.yaml"
    metrics_name: str = "metrics.json"
    predictions_name: str = "predictions.jsonl"
    stdout_name: str = "stdout.log"
    environment_name: str = "environment.txt"
    tuning_manifest_name: str = "tuning_manifest.json"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ArtifactLayout":
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise TypeError("artifacts must be a mapping when provided")
        return cls(
            root_dir=str(data.get("root_dir", "runs")),
            resolved_config_name=str(data.get("resolved_config_name", "resolved_config.yaml")),
            metrics_name=str(data.get("metrics_name", "metrics.json")),
            predictions_name=str(data.get("predictions_name", "predictions.jsonl")),
            stdout_name=str(data.get("stdout_name", "stdout.log")),
            environment_name=str(data.get("environment_name", "environment.txt")),
            tuning_manifest_name=str(data.get("tuning_manifest_name", "tuning_manifest.json")),
        )


@dataclass(slots=True)
class AppConfig:
    experiment_name: str
    milestone: str
    command: str
    seed: int = 17
    notes: str = ""
    paths: dict[str, str] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    artifacts: ArtifactLayout = field(default_factory=ArtifactLayout)
    source_path: Path = field(default=Path(), repr=False)


def _string_dict(value: Any, field_name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return {str(key): str(item) for key, item in value.items()}


def _object_dict(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return dict(value)


def load_app_config(path: str | Path) -> AppConfig:
    source_path = Path(path).resolve()
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))

    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError("Config root must be a mapping")

    for required_key in ("experiment_name", "milestone", "command"):
        if required_key not in raw:
            raise KeyError(f"Missing required config field: {required_key}")

    return AppConfig(
        experiment_name=str(raw["experiment_name"]),
        milestone=str(raw["milestone"]),
        command=str(raw["command"]),
        seed=int(raw.get("seed", 17)),
        notes=str(raw.get("notes", "")),
        paths=_string_dict(raw.get("paths"), "paths"),
        payload=_object_dict(raw.get("payload"), "payload"),
        artifacts=ArtifactLayout.from_mapping(raw.get("artifacts")),
        source_path=source_path,
    )
