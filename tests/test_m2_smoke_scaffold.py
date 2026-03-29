from __future__ import annotations

from pathlib import Path

from overlay_algebra.prompts import render_prompt
from overlay_algebra.smoke_pipeline import load_eval_smoke_config, load_train_smoke_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_prompt_renderer_matches_locked_logical_form() -> None:
    prompt = render_prompt(
        "Which planet is closest to the Sun?",
        "[S1] Mercury is the closest planet to the Sun.",
        "plain",
    )
    assert "Use only the provided context." in prompt
    assert "Required output format: plain answer text only" in prompt
    assert prompt.endswith("Response:\n")


def test_smoke_train_config_loads_without_runtime_imports() -> None:
    config = load_train_smoke_config(REPO_ROOT / "configs" / "train" / "smoke_scaffold.yaml")
    assert config.condition == "plain"
    assert config.base_model_name_or_path == "Qwen/Qwen2.5-7B-Instruct"
    assert config.target_modules == ["o_proj", "down_proj"]


def test_smoke_eval_config_points_to_expected_adapter_dir() -> None:
    config = load_eval_smoke_config(REPO_ROOT / "configs" / "eval" / "smoke_scaffold.yaml")
    assert config.adapter_dir.as_posix() == "runs/oa_m2_smoke_scaffold_17/adapter"
    assert config.condition == "plain"
