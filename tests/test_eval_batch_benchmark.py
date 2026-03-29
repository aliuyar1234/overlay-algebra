from __future__ import annotations

import json
from pathlib import Path

from overlay_algebra.analysis.eval_batch_benchmark import (
    _length_stratified_slice,
    _select_recommended_batch_size,
    _stress_slice,
    load_batch_benchmark_config,
    main,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_jq_eval_batch_benchmark_config_parses() -> None:
    config = load_batch_benchmark_config(REPO_ROOT / "configs" / "analysis" / "jq_eval_batch_benchmark.yaml")

    assert config.app.milestone == "M5"
    assert config.condition == "JQ"
    assert config.reference_batch_size == 4
    assert config.batch_sizes == (4, 8, 12, 16)
    assert config.stress_slice_size == 512
    assert config.representative_slice_size == 1024
    assert config.minimum_speedup_fraction == 0.15


def test_stress_slice_prefers_longest_features() -> None:
    features = [
        {"example_id": "a", "prompt_token_length": 20, "target_token_length": 5},
        {"example_id": "b", "prompt_token_length": 30, "target_token_length": 2},
        {"example_id": "c", "prompt_token_length": 30, "target_token_length": 9},
        {"example_id": "d", "prompt_token_length": 10, "target_token_length": 1},
    ]

    sliced = _stress_slice(features, 2)

    assert [row["example_id"] for row in sliced] == ["c", "b"]


def test_length_stratified_slice_is_deterministic() -> None:
    features = [
        {"example_id": f"ex{i}", "prompt_token_length": i, "target_token_length": i % 3}
        for i in range(10)
    ]

    sliced = _length_stratified_slice(features, 4)

    assert [row["example_id"] for row in sliced] == ["ex0", "ex3", "ex6", "ex9"]


def test_select_recommended_batch_size_uses_exactness_and_speed_threshold() -> None:
    results_by_batch = {
        4: {
            "status": "success",
            "slices": {
                "stress": {
                    "exact_prediction_match": True,
                    "exact_metric_match": True,
                },
                "representative": {
                    "exact_prediction_match": True,
                    "exact_metric_match": True,
                    "throughput": {"examples_per_second": 10.0},
                },
            },
        },
        8: {
            "status": "success",
            "slices": {
                "stress": {
                    "exact_prediction_match": True,
                    "exact_metric_match": True,
                },
                "representative": {
                    "exact_prediction_match": True,
                    "exact_metric_match": True,
                    "throughput": {"examples_per_second": 12.0},
                },
            },
        },
        12: {
            "status": "success",
            "slices": {
                "stress": {
                    "exact_prediction_match": True,
                    "exact_metric_match": True,
                },
                "representative": {
                    "exact_prediction_match": False,
                    "exact_metric_match": False,
                    "throughput": {"examples_per_second": 13.5},
                },
            },
        },
    }

    selection = _select_recommended_batch_size(
        results_by_batch=results_by_batch,
        reference_batch_size=4,
        minimum_speedup_fraction=0.15,
    )

    assert selection["accepted_batches"] == [4, 8]
    assert selection["recommended_batch_size"] == 8
    assert selection["rejection_reasons"]["12"] == "representative_predictions_changed"


def test_main_json_prints_full_summary(monkeypatch, capsys) -> None:
    def _fake_run(_config: object) -> dict[str, object]:
        return {
            "status": "completed",
            "selection": {"recommended_batch_size": 4},
            "run_dir": "runs/example",
        }

    monkeypatch.setattr("overlay_algebra.analysis.eval_batch_benchmark.run_eval_batch_benchmark", _fake_run)

    exit_code = main(["--config", str(REPO_ROOT / "configs" / "analysis" / "jq_eval_batch_benchmark.yaml"), "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["selection"]["recommended_batch_size"] == 4
