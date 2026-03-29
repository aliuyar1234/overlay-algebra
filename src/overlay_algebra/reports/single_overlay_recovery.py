from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from ..config import AppConfig, load_app_config


_PRIMARY_METRIC_BY_OVERLAY = {
    "J": "json_strict_schema_valid",
    "C": "citation_exact",
    "Q": "quote_exact",
}


@dataclass(frozen=True, slots=True)
class SingleOverlayReportConfig:
    experiment_name: str
    milestone: str
    overlay_name: str
    output_dir: Path
    prompt_only_metrics_path: Path
    direct_metrics_path: Path
    soar_metrics_path: Path
    scaffold_plain_metrics_path: Path | None
    source_config: Path

    @classmethod
    def from_app_config(cls, config: AppConfig) -> "SingleOverlayReportConfig":
        overlay_name = str(config.payload["overlay_name"])
        if overlay_name not in _PRIMARY_METRIC_BY_OVERLAY:
            raise ValueError("overlay_name must be one of J, C, Q")
        return cls(
            experiment_name=config.experiment_name,
            milestone=config.milestone,
            overlay_name=overlay_name,
            output_dir=Path(config.paths["output_dir"]),
            prompt_only_metrics_path=Path(config.paths["prompt_only_metrics"]),
            direct_metrics_path=Path(config.paths["direct_metrics"]),
            soar_metrics_path=Path(config.paths["soar_metrics"]),
            scaffold_plain_metrics_path=Path(config.paths["scaffold_plain_metrics"])
            if "scaffold_plain_metrics" in config.paths
            else None,
            source_config=config.source_path,
        )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _extract_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    if "metrics" in payload and isinstance(payload["metrics"], Mapping):
        source = payload["metrics"]
    else:
        source = payload
    return {
        str(key): float(value)
        for key, value in source.items()
        if isinstance(value, (int, float))
    }


def _optional_float(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_single_overlay_recovery_report(config: SingleOverlayReportConfig) -> dict[str, Any]:
    primary_metric = _PRIMARY_METRIC_BY_OVERLAY[config.overlay_name]

    prompt_only_payload = _read_json(config.prompt_only_metrics_path)
    direct_payload = _read_json(config.direct_metrics_path)
    soar_payload = _read_json(config.soar_metrics_path)
    scaffold_plain_payload = (
        _read_json(config.scaffold_plain_metrics_path) if config.scaffold_plain_metrics_path is not None else None
    )

    prompt_only_metrics = _extract_metrics(prompt_only_payload)
    direct_metrics = _extract_metrics(direct_payload)
    soar_metrics = _extract_metrics(soar_payload)
    scaffold_plain_metrics = _extract_metrics(scaffold_plain_payload) if scaffold_plain_payload is not None else None

    report = {
        "experiment_name": config.experiment_name,
        "milestone": config.milestone,
        "overlay_name": config.overlay_name,
        "primary_metric": primary_metric,
        "systems": {
            "prompt_only": prompt_only_metrics,
            "direct": direct_metrics,
            "soar": soar_metrics,
        },
        "derived": {
            "recovery_ratio": _optional_float(
                float(soar_metrics.get(primary_metric, 0.0)),
                float(direct_metrics.get(primary_metric, 0.0)),
            ),
            "soar_minus_prompt_only_primary": float(soar_metrics.get(primary_metric, 0.0))
            - float(prompt_only_metrics.get(primary_metric, 0.0)),
            "semantic_gap_vs_prompt_only": float(soar_metrics.get("answer_f1", 0.0))
            - float(prompt_only_metrics.get("answer_f1", 0.0)),
        },
        "sources": {
            "prompt_only_metrics_path": str(config.prompt_only_metrics_path),
            "direct_metrics_path": str(config.direct_metrics_path),
            "soar_metrics_path": str(config.soar_metrics_path),
            "scaffold_plain_metrics_path": str(config.scaffold_plain_metrics_path)
            if config.scaffold_plain_metrics_path is not None
            else None,
            "config_path": str(config.source_config),
        },
    }
    if scaffold_plain_metrics is not None:
        report["systems"]["scaffold_plain"] = scaffold_plain_metrics
        report["derived"]["semantic_gap_vs_scaffold_plain"] = float(soar_metrics.get("answer_f1", 0.0)) - float(
            scaffold_plain_metrics.get("answer_f1", 0.0)
        )

    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil_path = output_dir / "resolved_config.yaml"
    shutil_path.write_text(config.source_config.read_text(encoding="utf-8"), encoding="utf-8")

    metrics_path = output_dir / "metrics.json"
    markdown_path = output_dir / "report.md"
    _write_json(metrics_path, report)

    header = f"# Single-Overlay Recovery Report ({config.overlay_name})\n\n"
    table = (
        "| system | answer_f1 | primary |\n"
        "|---|---:|---:|\n"
        f"| prompt_only | {_format_optional(prompt_only_metrics.get('answer_f1'))} | {_format_optional(prompt_only_metrics.get(primary_metric))} |\n"
        f"| direct | {_format_optional(direct_metrics.get('answer_f1'))} | {_format_optional(direct_metrics.get(primary_metric))} |\n"
        f"| soar | {_format_optional(soar_metrics.get('answer_f1'))} | {_format_optional(soar_metrics.get(primary_metric))} |\n"
    )
    if scaffold_plain_metrics is not None:
        table += (
            f"| scaffold_plain | {_format_optional(scaffold_plain_metrics.get('answer_f1'))} | n/a |\n"
        )

    derived_lines = (
        "\nDerived:\n"
        f"- recovery_ratio: {_format_optional(report['derived']['recovery_ratio'])}\n"
        f"- soar_minus_prompt_only_primary: {_format_optional(report['derived']['soar_minus_prompt_only_primary'])}\n"
        f"- semantic_gap_vs_prompt_only: {_format_optional(report['derived']['semantic_gap_vs_prompt_only'])}\n"
    )
    if "semantic_gap_vs_scaffold_plain" in report["derived"]:
        derived_lines += (
            f"- semantic_gap_vs_scaffold_plain: {_format_optional(report['derived']['semantic_gap_vs_scaffold_plain'])}\n"
        )

    _write_markdown(markdown_path, header + table + derived_lines)

    return {
        "output_dir": str(output_dir),
        "metrics_path": str(metrics_path),
        "report_path": str(markdown_path),
        "overlay_name": config.overlay_name,
        "primary_metric": primary_metric,
        "recovery_ratio": report["derived"]["recovery_ratio"],
    }


def load_single_overlay_report_config(path: str | Path) -> SingleOverlayReportConfig:
    app_config = load_app_config(path)
    return SingleOverlayReportConfig.from_app_config(app_config)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a saved-metrics single-overlay recovery report.")
    parser.add_argument("--config", required=True, help="Path to the single-overlay recovery report YAML config.")
    parser.add_argument("--json", action="store_true", help="Print a JSON summary.")
    args = parser.parse_args()

    config = load_single_overlay_report_config(Path(args.config))
    summary = build_single_overlay_recovery_report(config)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        print(
            f"Wrote single-overlay recovery report for {config.overlay_name} "
            f"to {summary['report_path']}."
        )
        print(f"Metrics: {summary['metrics_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
