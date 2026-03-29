from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import overlay_algebra
from overlay_algebra import build_run_paths, load_app_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_package_imports_cleanly() -> None:
    assert overlay_algebra.__version__ == "0.1.0a0"
    assert overlay_algebra.PACKAGE_ROOT.name == "overlay_algebra"


def test_placeholder_config_loader_reads_yaml() -> None:
    config = load_app_config(REPO_ROOT / "configs" / "data" / "squad_jcq.yaml")
    run_paths = build_run_paths(config)

    assert config.command == "overlay_algebra.data.build_dataset"
    assert config.paths["output_dir"] == "data/processed/squad_jcq_v1_fixture"
    assert run_paths.run_dir.as_posix().startswith("runs/oa_m1_")


def test_python_can_import_package_from_repo_root() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import overlay_algebra; print(overlay_algebra.__version__)"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "0.1.0a0"
