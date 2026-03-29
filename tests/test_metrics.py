from __future__ import annotations

from overlay_algebra.metrics import (
    answer_f1,
    normalize_answer,
    normalize_quote,
    quote_f1,
    score_prediction,
)


def test_answer_normalization_matches_squad_style() -> None:
    assert normalize_answer("The, Quick Brown Fox!") == "quick brown fox"
    assert answer_f1("Mercury", "the mercury") == 1.0


def test_json_metrics_do_not_salvage_invalid_json_by_default() -> None:
    metrics = score_prediction(
        "J",
        '{"answer":"Mercury"',
        gold_answer="Mercury",
    )
    assert metrics["json_valid"] == 0.0
    assert metrics["json_strict_schema_valid"] == 0.0
    assert metrics["answer_f1"] == 0.0


def test_citation_and_quote_metrics_are_strict() -> None:
    citation_metrics = score_prediction(
        "C",
        "ANSWER: Mercury\nSUPPORT: [S2]",
        gold_answer="Mercury",
        gold_support_idx=2,
    )
    quote_metrics = score_prediction(
        "Q",
        'ANSWER: Mercury\nQUOTE: "  Mercury is the closest planet to the Sun.  "',
        gold_answer="Mercury",
        gold_support_sentence="Mercury is the closest planet to the Sun.",
    )

    assert citation_metrics["citation_exact"] == 1.0
    assert quote_metrics["quote_exact"] == 1.0
    assert quote_metrics["quote_f1"] == 1.0


def test_quote_normalization_keeps_case_but_collapses_spacing() -> None:
    assert normalize_quote("  Mercury   is near  ") == "Mercury is near"
    assert quote_f1("Mercury is near", "Mercury is near") == 1.0


def test_parse_failures_count_as_zero_for_required_citation_and_quote_metrics() -> None:
    citation_metrics = score_prediction(
        "C",
        "Mercury",
        gold_answer="Mercury",
        gold_support_idx=2,
    )
    quote_metrics = score_prediction(
        "Q",
        "Mercury",
        gold_answer="Mercury",
        gold_support_sentence="Mercury is the closest planet to the Sun.",
    )
    jq_metrics = score_prediction(
        "JQ",
        '{"answer":"Mercury"}',
        gold_answer="Mercury",
        gold_support_sentence="Mercury is the closest planet to the Sun.",
    )

    assert citation_metrics["citation_exact"] == 0.0
    assert quote_metrics["quote_exact"] == 0.0
    assert quote_metrics["quote_f1"] == 0.0
    assert jq_metrics["json_valid"] == 1.0
    assert jq_metrics["json_strict_schema_valid"] == 0.0
    assert jq_metrics["quote_exact"] == 0.0
    assert jq_metrics["quote_f1"] == 0.0
