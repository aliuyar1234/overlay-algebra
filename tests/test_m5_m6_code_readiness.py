from __future__ import annotations

from pathlib import Path

from overlay_algebra.analysis.ablate import load_ablation_config
from overlay_algebra.analysis.compose import load_compose_config
from overlay_algebra.reports.composition_study import load_composition_study_config
from overlay_algebra.reports.export_paper_artifacts import load_paper_artifact_export_config
from overlay_algebra.smoke_pipeline import load_eval_smoke_config, load_train_smoke_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_m5_train_configs_exist_and_keep_runtime_controls() -> None:
    for name in ("full_JC.yaml", "full_JQ.yaml", "full_CQ.yaml", "full_JCQ.yaml"):
        config = load_train_smoke_config(REPO_ROOT / "configs" / "train" / name)
        assert config.batch_size == 8
        assert config.gradient_accumulation_steps == 4
        assert config.checkpoint_interval_optimizer_steps == 500


def test_m5_eval_configs_exist_for_pairs_and_triple() -> None:
    for name in (
        "full_prompt_only_JC.yaml",
        "full_merge_JC.yaml",
        "full_soar_JC.yaml",
        "full_direct_JC.yaml",
        "full_prompt_only_JQ.yaml",
        "full_merge_JQ.yaml",
        "full_soar_JQ.yaml",
        "full_direct_JQ.yaml",
        "full_prompt_only_CQ.yaml",
        "full_merge_CQ.yaml",
        "full_soar_CQ.yaml",
        "full_direct_CQ.yaml",
        "full_prompt_only_JCQ.yaml",
        "full_merge_JCQ.yaml",
        "full_soar_JCQ.yaml",
        "full_direct_JCQ.yaml",
    ):
        config = load_eval_smoke_config(REPO_ROOT / "configs" / "eval" / name)
        assert config.eval_batch_size == 4


def test_reduced_scope_jq_only_m5_configs_parse() -> None:
    pairs_jq_only = load_compose_config(REPO_ROOT / "configs" / "analysis" / "pairs_JQ_only.yaml")
    merge_jq_only = load_eval_smoke_config(REPO_ROOT / "configs" / "eval" / "full_merge_JQ_only.yaml")
    soar_jq_only = load_eval_smoke_config(REPO_ROOT / "configs" / "eval" / "full_soar_JQ_only.yaml")
    pair_report_jq_only = load_composition_study_config(
        REPO_ROOT / "configs" / "reports" / "full_pair_composition_JQ_only.yaml"
    )

    assert len(pairs_jq_only.compositions) == 1
    assert pairs_jq_only.compositions[0].condition == "JQ"
    assert pairs_jq_only.candidate_artifact_policy == "minimal"
    assert set(pairs_jq_only.residual_adapter_dirs.keys()) == {"J", "Q"}
    assert set(pairs_jq_only.direct_adapter_dirs.keys()) == {"J", "Q"}
    assert merge_jq_only.eval_batch_size == 4
    assert soar_jq_only.eval_batch_size == 4
    assert len(pair_report_jq_only.tasks) == 1
    assert pair_report_jq_only.tasks[0].condition == "JQ"


def test_m5_m6_m7_configs_parse() -> None:
    pairs = load_compose_config(REPO_ROOT / "configs" / "analysis" / "pairs.yaml")
    triple = load_compose_config(REPO_ROOT / "configs" / "analysis" / "triple.yaml")
    ablations = load_ablation_config(REPO_ROOT / "configs" / "analysis" / "ablations.yaml")
    pair_report = load_composition_study_config(REPO_ROOT / "configs" / "reports" / "full_pair_composition.yaml")
    paper_export = load_paper_artifact_export_config(REPO_ROOT / "configs" / "reports" / "paper.yaml")

    assert len(pairs.compositions) == 3
    assert pairs.checkpoint_interval_units == 5
    assert pairs.candidate_artifact_policy == "full"
    assert pairs.enforce_budget_parity is True
    assert pairs.merge_search_policy == "match_soar_budget"
    assert pairs.reuse_feature_cache is True
    assert pairs.sort_eval_by_length is True
    assert triple.compositions[0].condition == "JCQ"
    assert triple.checkpoint_interval_units == 5
    assert triple.candidate_artifact_policy == "full"
    assert triple.merge_search_policy == "match_soar_budget"
    assert len(ablations.tasks) >= 3
    assert len(pair_report.tasks) == 3
    assert pair_report.c2_margin_points == 0.03
    assert "full_single_overlay" in paper_export.required_artifacts
