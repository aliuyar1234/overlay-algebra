from __future__ import annotations

import json
from pathlib import Path

from overlay_algebra.reports.export_paper_artifacts import (
    PaperArtifactExportConfig,
    export_paper_artifacts,
)


def test_export_paper_artifacts_copies_required_and_reports_missing_optional(tmp_path: Path) -> None:
    config_path = tmp_path / "paper.yaml"
    config_path.write_text("experiment_name: test\n", encoding="utf-8")
    required_file = tmp_path / "full_single_overlay.json"
    required_file.write_text("{}", encoding="utf-8")

    config = PaperArtifactExportConfig(
        experiment_name="paper_bundle",
        milestone="M7",
        output_dir=tmp_path / "bundle",
        required_artifacts={"full_single_overlay": required_file},
        optional_artifacts={"transfer_slice": tmp_path / "missing_transfer"},
        source_config=config_path,
    )

    summary = export_paper_artifacts(config)

    assert summary["missing_required"] == []
    assert summary["missing_optional"] == ["transfer_slice"]
    index = json.loads((tmp_path / "bundle" / "artifact_index.json").read_text(encoding="utf-8"))
    assert "full_single_overlay" in index["copied_artifacts"]
