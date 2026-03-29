from __future__ import annotations

import json
from pathlib import Path

from overlay_algebra.reports.composition_study import (
    CompositionStudyConfig,
    build_composition_study_report,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metrics_payload(**metrics: float) -> dict[str, object]:
    return {"metrics": metrics}


def test_composition_study_report_builds_pair_summary(tmp_path: Path) -> None:
    config_path = tmp_path / "composition.yaml"
    config_path.write_text("experiment_name: test\n", encoding="utf-8")

    prompt_only = tmp_path / "prompt.json"
    merge = tmp_path / "merge.json"
    soar = tmp_path / "soar.json"
    direct = tmp_path / "direct.json"
    _write_json(prompt_only, _metrics_payload(answer_f1=0.82, json_strict_schema_valid=0.60, citation_exact=0.55))
    _write_json(merge, _metrics_payload(answer_f1=0.80, json_strict_schema_valid=0.68, citation_exact=0.60))
    _write_json(soar, _metrics_payload(answer_f1=0.81, json_strict_schema_valid=0.75, citation_exact=0.70))
    _write_json(direct, _metrics_payload(answer_f1=0.83, json_strict_schema_valid=0.79, citation_exact=0.74))

    config = CompositionStudyConfig(
        experiment_name="pairs",
        milestone="M5",
        output_dir=tmp_path / "report_out",
        tasks=(
            type("Task", (), {
                "condition": "JC",
                "active_overlays": ("J", "C"),
                "prompt_only_metrics_path": prompt_only,
                "merge_metrics_path": merge,
                "soar_metrics_path": soar,
                "direct_metrics_path": direct,
            })(),
        ),
        semantic_guardrail_points=3.0,
        c2_margin_points=0.03,
        source_config=config_path,
    )

    summary = build_composition_study_report(config)

    assert summary["pair_conditions_beating_prompt_only_and_merge_under_guardrail"] == ["JC"]
    metrics = json.loads((tmp_path / "report_out" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["conditions"]["JC"]["derived"]["guardrail_satisfied"] is True
    assert metrics["derived"]["c2_margin_points"] == 0.03
    assert (tmp_path / "report_out" / "report.md").exists()


def test_composition_study_report_uses_configured_c2_margin(tmp_path: Path) -> None:
    config_path = tmp_path / "composition.yaml"
    config_path.write_text("experiment_name: test\n", encoding="utf-8")

    prompt_only = tmp_path / "prompt.json"
    merge = tmp_path / "merge.json"
    soar = tmp_path / "soar.json"
    direct = tmp_path / "direct.json"
    _write_json(prompt_only, _metrics_payload(answer_f1=0.82, json_strict_schema_valid=0.60, citation_exact=0.55))
    _write_json(merge, _metrics_payload(answer_f1=0.80, json_strict_schema_valid=0.68, citation_exact=0.60))
    _write_json(soar, _metrics_payload(answer_f1=0.81, json_strict_schema_valid=0.75, citation_exact=0.70))
    _write_json(direct, _metrics_payload(answer_f1=0.83, json_strict_schema_valid=0.79, citation_exact=0.74))

    config = CompositionStudyConfig(
        experiment_name="pairs",
        milestone="M5",
        output_dir=tmp_path / "report_out",
        tasks=(
            type("Task", (), {
                "condition": "JC",
                "active_overlays": ("J", "C"),
                "prompt_only_metrics_path": prompt_only,
                "merge_metrics_path": merge,
                "soar_metrics_path": soar,
                "direct_metrics_path": direct,
            })(),
        ),
        semantic_guardrail_points=3.0,
        c2_margin_points=0.20,
        source_config=config_path,
    )

    summary = build_composition_study_report(config)

    assert summary["pair_conditions_beating_prompt_only_and_merge_under_guardrail"] == []
