from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from ..config import AppConfig, load_app_config


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True, slots=True)
class PaperArtifactExportConfig:
    experiment_name: str
    milestone: str
    output_dir: Path
    required_artifacts: dict[str, Path]
    optional_artifacts: dict[str, Path]
    source_config: Path

    @classmethod
    def from_app_config(cls, config: AppConfig) -> "PaperArtifactExportConfig":
        required = {str(key): Path(str(value)) for key, value in config.paths.items() if key != "output_dir"}
        optional_raw = config.payload.get("optional_artifacts", {})
        if optional_raw is None:
            optional_raw = {}
        if not isinstance(optional_raw, Mapping):
            raise TypeError("payload.optional_artifacts must be a mapping when provided")
        return cls(
            experiment_name=config.experiment_name,
            milestone=config.milestone,
            output_dir=Path(config.paths["output_dir"]),
            required_artifacts=required,
            optional_artifacts={str(key): Path(str(value)) for key, value in optional_raw.items()},
            source_config=config.source_path,
        )


def export_paper_artifacts(config: PaperArtifactExportConfig) -> dict[str, Any]:
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    missing_required: list[str] = []
    missing_optional: list[str] = []

    def copy_entry(label: str, source: Path) -> None:
        destination = output_dir / label
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        copied[label] = str(destination)

    for label, source in config.required_artifacts.items():
        if not source.exists():
            missing_required.append(label)
            continue
        copy_entry(label, source)
    for label, source in config.optional_artifacts.items():
        if not source.exists():
            missing_optional.append(label)
            continue
        copy_entry(label, source)

    index = {
        "experiment_name": config.experiment_name,
        "milestone": config.milestone,
        "copied_artifacts": copied,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "config_path": str(config.source_config),
    }
    _write_json(output_dir / "artifact_index.json", index)
    (output_dir / "resolved_config.yaml").write_text(config.source_config.read_text(encoding="utf-8"), encoding="utf-8")
    lines = ["# Paper Artifact Export", ""]
    lines.append("## Copied")
    if copied:
        for label, destination in sorted(copied.items()):
            lines.append(f"- {label}: {destination}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Missing Required")
    if missing_required:
        for label in missing_required:
            lines.append(f"- {label}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Missing Optional")
    if missing_optional:
        for label in missing_optional:
            lines.append(f"- {label}")
    else:
        lines.append("- none")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "artifact_index_path": str(output_dir / "artifact_index.json"),
        "report_path": str(output_dir / "report.md"),
        "missing_required": missing_required,
        "missing_optional": missing_optional,
    }


def load_paper_artifact_export_config(path: str | Path) -> PaperArtifactExportConfig:
    return PaperArtifactExportConfig.from_app_config(load_app_config(path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy saved reports/artifacts into a paper-facing export bundle.")
    parser.add_argument("--config", required=True, help="Path to the paper artifact export YAML config.")
    parser.add_argument("--json", action="store_true", help="Print a JSON summary.")
    args = parser.parse_args()

    config = load_paper_artifact_export_config(Path(args.config))
    summary = export_paper_artifacts(config)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        print(f"Wrote paper artifact bundle to {summary['output_dir']}.")
        print(f"Index: {summary['artifact_index_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
