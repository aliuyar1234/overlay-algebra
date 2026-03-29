from __future__ import annotations

import json
from pathlib import Path

from overlay_algebra.reports.single_overlay_recovery import (
    SingleOverlayReportConfig,
    build_single_overlay_recovery_report,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_single_overlay_recovery_report_builds_summary(tmp_path: Path) -> None:
    prompt_only = tmp_path / "prompt_only.json"
    direct = tmp_path / "direct.json"
    soar = tmp_path / "soar.json"
    config_path = tmp_path / "report.yaml"

    _write_json(prompt_only, {"metrics": {"answer_f1": 0.81, "json_strict_schema_valid": 0.40}})
    _write_json(direct, {"metrics": {"answer_f1": 0.79, "json_strict_schema_valid": 0.90}})
    _write_json(soar, {"metrics": {"answer_f1": 0.80, "json_strict_schema_valid": 0.72}})
    config_path.write_text("experiment_name: test\n", encoding="utf-8")

    config = SingleOverlayReportConfig(
        experiment_name="pilot_recovery_j",
        milestone="M3",
        overlay_name="J",
        output_dir=tmp_path / "report_out",
        prompt_only_metrics_path=prompt_only,
        direct_metrics_path=direct,
        soar_metrics_path=soar,
        scaffold_plain_metrics_path=None,
        source_config=config_path,
    )
    summary = build_single_overlay_recovery_report(config)

    assert summary["primary_metric"] == "json_strict_schema_valid"
    assert summary["recovery_ratio"] == 0.72 / 0.90
    assert (tmp_path / "report_out" / "metrics.json").exists()
    assert (tmp_path / "report_out" / "report.md").exists()
