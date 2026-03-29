from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import AppConfig, load_app_config


_PRIMARY_METRIC_BY_OVERLAY = {
    "J": "json_strict_schema_valid",
    "C": "citation_exact",
    "Q": "quote_exact",
}


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
class CompositionReportTask:
    condition: str
    active_overlays: tuple[str, ...]
    prompt_only_metrics_path: Path
    merge_metrics_path: Path
    soar_metrics_path: Path
    direct_metrics_path: Path

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CompositionReportTask":
        active_overlays = tuple(str(item) for item in payload.get("active_overlays", []))
        condition = str(payload["condition"])
        if not active_overlays:
            active_overlays = tuple(letter for letter in ("J", "C", "Q") if letter in condition)
        return cls(
            condition=condition,
            active_overlays=active_overlays,
            prompt_only_metrics_path=Path(str(payload["prompt_only_metrics"])),
            merge_metrics_path=Path(str(payload["merge_metrics"])),
            soar_metrics_path=Path(str(payload["soar_metrics"])),
            direct_metrics_path=Path(str(payload["direct_metrics"])),
        )


@dataclass(frozen=True, slots=True)
class CompositionStudyConfig:
    experiment_name: str
    milestone: str
    output_dir: Path
    tasks: tuple[CompositionReportTask, ...]
    semantic_guardrail_points: float
    c2_margin_points: float
    source_config: Path

    @classmethod
    def from_app_config(cls, config: AppConfig) -> "CompositionStudyConfig":
        raw_tasks = config.payload.get("tasks")
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise TypeError("payload.tasks must be a non-empty list")
        return cls(
            experiment_name=config.experiment_name,
            milestone=config.milestone,
            output_dir=Path(config.paths["output_dir"]),
            tasks=tuple(CompositionReportTask.from_mapping(item) for item in raw_tasks),
            semantic_guardrail_points=float(config.payload.get("semantic_guardrail_points", 3.0)),
            c2_margin_points=float(config.payload.get("c2_margin_points", 0.03)),
            source_config=config.source_path,
        )


def _active_primary_metrics(active_overlays: Sequence[str]) -> list[str]:
    return [_PRIMARY_METRIC_BY_OVERLAY[overlay] for overlay in active_overlays]


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


def _mean_active_overlay(metrics: Mapping[str, Any], active_overlays: Sequence[str]) -> float:
    values = [float(metrics.get(_PRIMARY_METRIC_BY_OVERLAY[overlay], 0.0)) for overlay in active_overlays]
    return sum(values) / len(values) if values else 0.0


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def build_composition_study_report(config: CompositionStudyConfig) -> dict[str, Any]:
    condition_rows: dict[str, Any] = {}
    overlay_gap_rows: dict[str, list[float]] = {"J": [], "C": [], "Q": []}
    overlay_semantic_rows: dict[str, list[float]] = {"J": [], "C": [], "Q": []}
    soar_pair_wins = []
    for task in config.tasks:
        prompt_only = _extract_metrics(_read_json(task.prompt_only_metrics_path))
        merge = _extract_metrics(_read_json(task.merge_metrics_path))
        soar = _extract_metrics(_read_json(task.soar_metrics_path))
        direct = _extract_metrics(_read_json(task.direct_metrics_path))
        mean_prompt_only = _mean_active_overlay(prompt_only, task.active_overlays)
        mean_merge = _mean_active_overlay(merge, task.active_overlays)
        mean_soar = _mean_active_overlay(soar, task.active_overlays)
        mean_direct = _mean_active_overlay(direct, task.active_overlays)
        semantic_gap = float(soar.get("answer_f1", 0.0)) - float(prompt_only.get("answer_f1", 0.0))
        guardrail_floor = float(prompt_only.get("answer_f1", 0.0)) - config.semantic_guardrail_points / 100.0
        guardrail_satisfied = float(soar.get("answer_f1", 0.0)) >= guardrail_floor
        for overlay in task.active_overlays:
            overlay_gap_rows[overlay].append(float(direct.get(_PRIMARY_METRIC_BY_OVERLAY[overlay], 0.0)) - float(soar.get(_PRIMARY_METRIC_BY_OVERLAY[overlay], 0.0)))
            overlay_semantic_rows[overlay].append(semantic_gap)
        if (
            mean_soar >= mean_prompt_only + config.c2_margin_points
            and mean_soar >= mean_merge + config.c2_margin_points
            and guardrail_satisfied
            and len(task.active_overlays) == 2
        ):
            soar_pair_wins.append(task.condition)

        condition_rows[task.condition] = {
            "active_overlays": list(task.active_overlays),
            "primary_metrics": _active_primary_metrics(task.active_overlays),
            "systems": {
                "prompt_only": prompt_only,
                "merge": merge,
                "soar": soar,
                "direct": direct,
            },
            "derived": {
                "mean_active_primary_prompt_only": mean_prompt_only,
                "mean_active_primary_merge": mean_merge,
                "mean_active_primary_soar": mean_soar,
                "mean_active_primary_direct": mean_direct,
                "soar_minus_prompt_only_mean_active": mean_soar - mean_prompt_only,
                "soar_minus_merge_mean_active": mean_soar - mean_merge,
                "composition_gap_vs_direct": mean_direct - mean_soar,
                "semantic_gap_vs_prompt_only": semantic_gap,
                "guardrail_floor_answer_f1": guardrail_floor,
                "guardrail_satisfied": guardrail_satisfied,
            },
        }

    overlay_summary = {
        overlay: {
            "median_composition_gap_vs_direct": _median(rows),
            "median_semantic_gap_vs_prompt_only": _median(overlay_semantic_rows[overlay]),
        }
        for overlay, rows in overlay_gap_rows.items()
    }
    report = {
        "experiment_name": config.experiment_name,
        "milestone": config.milestone,
        "conditions": condition_rows,
        "overlay_summary": overlay_summary,
        "derived": {
            "pair_conditions_beating_prompt_only_and_merge_under_guardrail": soar_pair_wins,
            "c2_margin_points": config.c2_margin_points,
        },
        "sources": {
            "config_path": str(config.source_config),
        },
    }

    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(config.source_config.read_text(encoding="utf-8"), encoding="utf-8")
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "report.md"
    _write_json(metrics_path, report)

    lines = [
        "# Composition Study Report",
        "",
        "| condition | overlays | prompt_only | merge | soar | direct | soar_minus_merge | gap_vs_direct | guardrail |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for task in config.tasks:
        row = condition_rows[task.condition]
        derived = row["derived"]
        lines.append(
            f"| {task.condition} | {'+'.join(task.active_overlays)} | "
            f"{derived['mean_active_primary_prompt_only']:.4f} | "
            f"{derived['mean_active_primary_merge']:.4f} | "
            f"{derived['mean_active_primary_soar']:.4f} | "
            f"{derived['mean_active_primary_direct']:.4f} | "
            f"{derived['soar_minus_merge_mean_active']:.4f} | "
            f"{derived['composition_gap_vs_direct']:.4f} | "
            f"{'yes' if derived['guardrail_satisfied'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Derived:",
            "- pair_conditions_beating_prompt_only_and_merge_under_guardrail: "
            + (", ".join(soar_pair_wins) if soar_pair_wins else "none"),
        ]
    )
    for overlay in ("J", "C", "Q"):
        summary = overlay_summary[overlay]
        lines.append(
            f"- {overlay}: median_gap_vs_direct={_format_optional(_float_or_none(summary['median_composition_gap_vs_direct']))}, "
            f"median_semantic_gap_vs_prompt_only={_format_optional(_float_or_none(summary['median_semantic_gap_vs_prompt_only']))}"
        )
    _write_markdown(report_path, "\n".join(lines) + "\n")

    return {
        "output_dir": str(output_dir),
        "metrics_path": str(metrics_path),
        "report_path": str(report_path),
        "pair_conditions_beating_prompt_only_and_merge_under_guardrail": soar_pair_wins,
    }


def load_composition_study_config(path: str | Path) -> CompositionStudyConfig:
    return CompositionStudyConfig.from_app_config(load_app_config(path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a pair/triple composition study report from saved metrics.")
    parser.add_argument("--config", required=True, help="Path to the composition study YAML config.")
    parser.add_argument("--json", action="store_true", help="Print a JSON summary.")
    args = parser.parse_args()

    config = load_composition_study_config(Path(args.config))
    summary = build_composition_study_report(config)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        print(f"Wrote composition study report to {summary['report_path']}.")
        print(f"Metrics: {summary['metrics_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
