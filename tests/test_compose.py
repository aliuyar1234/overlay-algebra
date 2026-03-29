from __future__ import annotations

from pathlib import Path

import pytest

from overlay_algebra.analysis.compose import (
    CompositionTask,
    CompositionCandidate,
    _assert_tuning_budget_parity,
    _merge_candidate_coefficients,
    _persist_full_candidate_artifacts,
    _soar_candidate_coefficients,
    choose_best_composition_candidate,
    load_compose_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _candidate(
    key: str,
    *,
    answer_f1: float,
    mean_primary_j: float,
    mean_primary_c: float,
    tie_distance: float,
) -> CompositionCandidate:
    return CompositionCandidate(
        system="soar",
        task_name="JC",
        key=key,
        coefficients={"J": 1.0, "C": 1.0},
        adapter_dir=Path(f"candidate_{key}"),
        metrics={
            "condition": "JC",
            "answer_f1": answer_f1,
            "json_strict_schema_valid": mean_primary_j,
            "citation_exact": mean_primary_c,
        },
        tie_distance=tie_distance,
    )


def test_choose_best_composition_candidate_prefers_guardrail_passing_score() -> None:
    chosen, selection = choose_best_composition_candidate(
        active_overlays=("J", "C"),
        baseline_answer_f1=0.90,
        guardrail_points=3.0,
        candidates=[
            _candidate("a", answer_f1=0.87, mean_primary_j=0.80, mean_primary_c=0.75, tie_distance=0.1),
            _candidate("b", answer_f1=0.88, mean_primary_j=0.84, mean_primary_c=0.79, tie_distance=0.2),
            _candidate("c", answer_f1=0.82, mean_primary_j=0.95, mean_primary_c=0.95, tie_distance=0.0),
        ],
    )

    assert chosen.key == "b"
    assert selection["guardrail_satisfied"] is True


def test_choose_best_composition_candidate_falls_back_to_semantic_f1() -> None:
    chosen, selection = choose_best_composition_candidate(
        active_overlays=("J", "Q"),
        baseline_answer_f1=0.95,
        guardrail_points=3.0,
        candidates=[
            CompositionCandidate(
                system="soar",
                task_name="JQ",
                key="a",
                coefficients={"J": 0.75, "Q": 1.0},
                adapter_dir=Path("a"),
                metrics={"condition": "JQ", "answer_f1": 0.70, "json_strict_schema_valid": 1.0, "quote_exact": 0.95},
                tie_distance=0.25,
            ),
            CompositionCandidate(
                system="soar",
                task_name="JQ",
                key="b",
                coefficients={"J": 1.0, "Q": 1.0},
                adapter_dir=Path("b"),
                metrics={"condition": "JQ", "answer_f1": 0.80, "json_strict_schema_valid": 0.85, "quote_exact": 0.82},
                tie_distance=0.0,
            ),
        ],
    )

    assert chosen.key == "b"
    assert selection["guardrail_satisfied"] is False


def test_pairs_config_parses_runtime_controls_and_tasks() -> None:
    config = load_compose_config(REPO_ROOT / "configs" / "analysis" / "pairs.yaml")

    assert config.checkpoint_interval_units == 5
    assert config.eval_batch_size == 4
    assert config.candidate_artifact_policy == "full"
    assert config.reuse_feature_cache is True
    assert config.sort_eval_by_length is True
    assert config.enforce_budget_parity is True
    assert config.merge_search_policy == "match_soar_budget"
    assert len(config.compositions) == 3
    assert config.compositions[1].condition == "JQ"


def test_match_soar_budget_policy_produces_equal_candidate_counts() -> None:
    task = CompositionTask(condition="JC", active_overlays=("J", "C"))
    soar_candidate_count = len(_soar_candidate_coefficients(task.active_overlays, (0.5, 0.75, 1.0, 1.25)))
    merge_candidate_count = len(
        _merge_candidate_coefficients(
            task.active_overlays,
            0.25,
            policy="match_soar_budget",
            target_budget=soar_candidate_count,
        )
    )

    assert merge_candidate_count == soar_candidate_count


def test_budget_parity_guard_raises_for_simplex_grid_step_policy() -> None:
    task = CompositionTask(condition="JC", active_overlays=("J", "C"))
    soar_candidate_count = len(_soar_candidate_coefficients(task.active_overlays, (0.5, 0.75, 1.0, 1.25)))
    merge_candidate_count = len(
        _merge_candidate_coefficients(
            task.active_overlays,
            0.25,
            policy="simplex_grid_step",
            target_budget=soar_candidate_count,
        )
    )

    with pytest.raises(ValueError, match="unequal tuning budgets"):
        _assert_tuning_budget_parity(
            task=task,
            soar_candidate_count=soar_candidate_count,
            merge_candidate_count=merge_candidate_count,
            tolerance=0,
        )


def test_candidate_artifact_policy_helper_supports_full_and_minimal() -> None:
    assert _persist_full_candidate_artifacts("full") is True
    assert _persist_full_candidate_artifacts("minimal") is False

    with pytest.raises(ValueError, match="candidate_artifact_policy"):
        _persist_full_candidate_artifacts("unexpected")
