from __future__ import annotations

import json
from pathlib import Path

from overlay_algebra.reports.full_single_overlay import (
    FullSingleOverlayStudyConfig,
    build_full_single_overlay_report,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _single_overlay_payload(
    primary_metric: str,
    prompt_only_primary: float,
    direct_primary: float,
    soar_primary: float,
    prompt_only_answer_f1: float,
    direct_answer_f1: float,
    soar_answer_f1: float,
) -> dict[str, object]:
    return {
        "primary_metric": primary_metric,
        "systems": {
            "prompt_only": {"answer_f1": prompt_only_answer_f1, primary_metric: prompt_only_primary},
            "direct": {"answer_f1": direct_answer_f1, primary_metric: direct_primary},
            "soar": {"answer_f1": soar_answer_f1, primary_metric: soar_primary},
        },
        "derived": {
            "recovery_ratio": soar_primary / direct_primary,
            "soar_minus_prompt_only_primary": soar_primary - prompt_only_primary,
            "semantic_gap_vs_prompt_only": soar_answer_f1 - prompt_only_answer_f1,
        },
    }


def test_full_single_overlay_report_builds_combined_summary(tmp_path: Path) -> None:
    config_path = tmp_path / "report.yaml"
    config_path.write_text("experiment_name: test\n", encoding="utf-8")

    j_report = tmp_path / "j.json"
    c_report = tmp_path / "c.json"
    q_report = tmp_path / "q.json"
    _write_json(j_report, _single_overlay_payload("json_strict_schema_valid", 0.74, 0.99, 0.99, 0.56, 0.74, 0.75))
    _write_json(c_report, _single_overlay_payload("citation_exact", 0.93, 0.92, 0.93, 0.82, 0.81, 0.81))
    _write_json(q_report, _single_overlay_payload("quote_exact", 0.52, 0.87, 0.86, 0.51, 0.77, 0.76))

    config = FullSingleOverlayStudyConfig(
        experiment_name="full_single_overlay",
        milestone="M4",
        output_dir=tmp_path / "report_out",
        overlay_report_paths={"J": j_report, "C": c_report, "Q": q_report},
        source_config=config_path,
    )

    summary = build_full_single_overlay_report(config)

    assert summary["mean_recovery_ratio"] is not None
    metrics = json.loads((tmp_path / "report_out" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["overlays"]["J"]["primary_metric"] == "json_strict_schema_valid"
    assert metrics["derived"]["prompt_only_best_overlays"] == ["C"]
    assert (tmp_path / "report_out" / "report.md").exists()


def test_full_single_overlay_report_supports_reduced_scope_without_c(tmp_path: Path) -> None:
    config_path = tmp_path / "report.yaml"
    config_path.write_text("experiment_name: test\n", encoding="utf-8")

    j_report = tmp_path / "j.json"
    q_report = tmp_path / "q.json"
    _write_json(j_report, _single_overlay_payload("json_strict_schema_valid", 0.01, 0.99, 0.98, 0.61, 0.84, 0.83))
    _write_json(q_report, _single_overlay_payload("quote_exact", 0.52, 0.87, 0.86, 0.51, 0.77, 0.76))

    config = FullSingleOverlayStudyConfig(
        experiment_name="full_single_overlay_jq",
        milestone="M4",
        output_dir=tmp_path / "report_out",
        overlay_report_paths={"J": j_report, "Q": q_report},
        source_config=config_path,
    )

    summary = build_full_single_overlay_report(config)

    assert summary["mean_recovery_ratio"] is not None
    metrics = json.loads((tmp_path / "report_out" / "metrics.json").read_text(encoding="utf-8"))
    assert sorted(metrics["overlays"]) == ["J", "Q"]
    assert metrics["sources"]["overlay_reports"] == {"J": str(j_report), "Q": str(q_report)}
    report_text = (tmp_path / "report_out" / "report.md").read_text(encoding="utf-8")
    assert "| J |" in report_text
    assert "| Q |" in report_text
    assert "| C |" not in report_text
