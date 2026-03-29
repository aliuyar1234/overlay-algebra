from __future__ import annotations

from pathlib import Path

from overlay_algebra.analysis.residualize import BetaCandidate, choose_best_beta


def _candidate(beta: float, *, answer_f1: float, json_valid: float) -> BetaCandidate:
    return BetaCandidate(
        beta=beta,
        adapter_dir=Path(f"candidate_{beta}"),
        metrics={
            "condition": "J",
            "answer_f1": answer_f1,
            "json_strict_schema_valid": json_valid,
        },
        predictions=[],
    )


def test_choose_best_beta_prefers_guardrail_passing_overlay_score() -> None:
    chosen, selection = choose_best_beta(
        "J",
        baseline_answer_f1=0.90,
        guardrail_points=3.0,
        candidates=[
            _candidate(0.75, answer_f1=0.86, json_valid=0.70),
            _candidate(1.0, answer_f1=0.87, json_valid=0.80),
            _candidate(1.25, answer_f1=0.91, json_valid=0.75),
        ],
    )
    assert chosen.beta == 1.0
    assert selection["guardrail_satisfied"] is True


def test_choose_best_beta_falls_back_to_semantic_f1_when_guardrail_fails() -> None:
    chosen, selection = choose_best_beta(
        "J",
        baseline_answer_f1=0.95,
        guardrail_points=3.0,
        candidates=[
            _candidate(0.75, answer_f1=0.70, json_valid=1.0),
            _candidate(1.0, answer_f1=0.80, json_valid=0.60),
            _candidate(1.25, answer_f1=0.79, json_valid=0.95),
        ],
    )
    assert chosen.beta == 1.0
    assert selection["guardrail_satisfied"] is False


def test_choose_best_beta_tie_breaks_toward_one() -> None:
    chosen, _selection = choose_best_beta(
        "J",
        baseline_answer_f1=0.90,
        guardrail_points=3.0,
        candidates=[
            _candidate(0.75, answer_f1=0.88, json_valid=0.80),
            _candidate(1.25, answer_f1=0.89, json_valid=0.80),
        ],
    )
    assert chosen.beta == 0.75
