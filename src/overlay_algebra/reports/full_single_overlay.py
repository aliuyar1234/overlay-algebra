from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from ..config import AppConfig, load_app_config


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"Expected numeric value or None, got {value!r}")


def _format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


@dataclass(frozen=True, slots=True)
class FullSingleOverlayStudyConfig:
    experiment_name: str
    milestone: str
    output_dir: Path
    overlay_report_paths: dict[str, Path]
    source_config: Path

    @classmethod
    def from_app_config(cls, config: AppConfig) -> "FullSingleOverlayStudyConfig":
        overlay_report_paths = {
            overlay_name: Path(config.paths[f"{overlay_name.lower()}_report"])
            for overlay_name in ("J", "C", "Q")
            if f"{overlay_name.lower()}_report" in config.paths
        }
        if not overlay_report_paths:
            raise ValueError("full single-overlay report config must define at least one *_report path")
        return cls(
            experiment_name=config.experiment_name,
            milestone=config.milestone,
            output_dir=Path(config.paths["output_dir"]),
            overlay_report_paths=overlay_report_paths,
            source_config=config.source_path,
        )


def _extract_overlay_row(report: Mapping[str, Any]) -> dict[str, Any]:
    systems = report["systems"]
    if not isinstance(systems, Mapping):
        raise TypeError("single-overlay report is missing a systems mapping")
    prompt_only = systems["prompt_only"]
    direct = systems["direct"]
    soar = systems["soar"]
    if not all(isinstance(item, Mapping) for item in (prompt_only, direct, soar)):
        raise TypeError("single-overlay systems entries must be mappings")

    derived = report.get("derived", {})
    if not isinstance(derived, Mapping):
        raise TypeError("single-overlay report derived block must be a mapping")

    primary_metric = str(report["primary_metric"])
    return {
        "primary_metric": primary_metric,
        "prompt_only_primary": float(prompt_only.get(primary_metric, 0.0)),
        "direct_primary": float(direct.get(primary_metric, 0.0)),
        "soar_primary": float(soar.get(primary_metric, 0.0)),
        "prompt_only_answer_f1": float(prompt_only.get("answer_f1", 0.0)),
        "direct_answer_f1": float(direct.get("answer_f1", 0.0)),
        "soar_answer_f1": float(soar.get("answer_f1", 0.0)),
        "recovery_ratio": _float_or_none(derived.get("recovery_ratio")),
        "soar_minus_prompt_only_primary": float(derived.get("soar_minus_prompt_only_primary", 0.0)),
        "semantic_gap_vs_prompt_only": float(derived.get("semantic_gap_vs_prompt_only", 0.0)),
    }


def build_full_single_overlay_report(config: FullSingleOverlayStudyConfig) -> dict[str, Any]:
    ordered_overlay_names = [
        overlay_name
        for overlay_name in ("J", "C", "Q")
        if overlay_name in config.overlay_report_paths
    ]
    overlay_rows = {
        overlay_name: _extract_overlay_row(_read_json(path))
        for overlay_name in ordered_overlay_names
        for path in [config.overlay_report_paths[overlay_name]]
    }
    recovery_ratios = [
        float(row["recovery_ratio"])
        for row in overlay_rows.values()
        if row["recovery_ratio"] is not None
    ]
    prompt_only_best_overlays = [
        overlay_name
        for overlay_name, row in overlay_rows.items()
        if float(row["prompt_only_primary"]) >= max(float(row["direct_primary"]), float(row["soar_primary"]))
    ]

    report = {
        "experiment_name": config.experiment_name,
        "milestone": config.milestone,
        "overlays": overlay_rows,
        "derived": {
            "mean_recovery_ratio": (sum(recovery_ratios) / len(recovery_ratios) if recovery_ratios else None),
            "prompt_only_best_overlays": prompt_only_best_overlays,
        },
        "sources": {
            "overlay_reports": {
                overlay_name: str(config.overlay_report_paths[overlay_name])
                for overlay_name in ordered_overlay_names
            },
            "config_path": str(config.source_config),
        },
    }
    for overlay_name in ordered_overlay_names:
        report["sources"][f"{overlay_name.lower()}_report"] = str(config.overlay_report_paths[overlay_name])

    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        config.source_config.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "report.md"
    _write_json(metrics_path, report)

    table_lines = [
        "# Full Single-Overlay Study Report",
        "",
        "| overlay | primary_metric | prompt_only | direct | soar | recovery_ratio | semantic_gap_vs_prompt_only |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for overlay_name in ordered_overlay_names:
        row = overlay_rows[overlay_name]
        table_lines.append(
            f"| {overlay_name} | {row['primary_metric']} | "
            f"{row['prompt_only_primary']:.4f} | {row['direct_primary']:.4f} | {row['soar_primary']:.4f} | "
            f"{_format_optional(row['recovery_ratio'])} | {row['semantic_gap_vs_prompt_only']:.4f} |"
        )

    mean_recovery_ratio = report["derived"]["mean_recovery_ratio"]
    prompt_only_best_text = ", ".join(prompt_only_best_overlays) if prompt_only_best_overlays else "none"
    table_lines.extend(
        [
            "",
            "Derived:",
            f"- mean_recovery_ratio: {_format_optional(_float_or_none(mean_recovery_ratio))}",
            f"- prompt_only_best_overlays: {prompt_only_best_text}",
        ]
    )
    _write_markdown(report_path, "\n".join(table_lines) + "\n")

    return {
        "output_dir": str(output_dir),
        "metrics_path": str(metrics_path),
        "report_path": str(report_path),
        "mean_recovery_ratio": mean_recovery_ratio,
    }


def load_full_single_overlay_study_config(path: str | Path) -> FullSingleOverlayStudyConfig:
    return FullSingleOverlayStudyConfig.from_app_config(load_app_config(path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the full single-overlay study report from saved overlay reports.")
    parser.add_argument("--config", required=True, help="Path to the full single-overlay report YAML config.")
    parser.add_argument("--json", action="store_true", help="Print a JSON summary.")
    args = parser.parse_args()

    config = load_full_single_overlay_study_config(Path(args.config))
    summary = build_full_single_overlay_report(config)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        print(f"Wrote full single-overlay study report to {summary['report_path']}.")
        print(f"Metrics: {summary['metrics_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
