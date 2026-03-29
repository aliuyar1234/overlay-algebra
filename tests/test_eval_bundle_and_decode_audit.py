from __future__ import annotations

import json
from pathlib import Path

from overlay_algebra.analysis.adapter_eval import LoadedAdapterEvaluator
from overlay_algebra.analysis.adapter_eval import load_or_build_eval_features
from overlay_algebra.analysis.decode_audit import (
    _compare_prediction_rows,
    _parse_success_rate,
    load_decode_audit_config,
)
from overlay_algebra.eval.bundle import _collect_aggregate_predictions, load_bundle_eval_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_bundle_eval_configs_parse_with_runtime_controls() -> None:
    for name in ("full_bundle_J.yaml", "full_bundle_C.yaml", "full_bundle_Q.yaml"):
        config = load_bundle_eval_config(REPO_ROOT / "configs" / "eval" / name)
        assert len(config.tasks) == 3
        assert config.reuse_feature_cache is True
        assert config.sort_eval_by_length is True
        assert config.eval_batch_size == 4


def test_decode_audit_q_config_parses_expected_budget_grid() -> None:
    config = load_decode_audit_config(REPO_ROOT / "configs" / "analysis" / "decode_audit_q.yaml")

    assert len(config.tasks) == 3
    assert config.tasks[0].budgets == (64, 96, 128)
    assert config.reuse_feature_cache is True
    assert config.sort_eval_by_length is True


def test_decode_audit_q_pre_residualize_config_parses_expected_tasks() -> None:
    config = load_decode_audit_config(REPO_ROOT / "configs" / "analysis" / "decode_audit_q_pre_residualize.yaml")

    assert len(config.tasks) == 2
    assert {task.name for task in config.tasks} == {"prompt_only_q", "direct_q"}
    assert all(task.budgets == (64, 96, 128) for task in config.tasks)


def test_compare_prediction_rows_reports_changed_examples() -> None:
    reference_rows = [
        {"example_id": "ex-1", "prediction": "alpha"},
        {"example_id": "ex-2", "prediction": "beta"},
    ]
    candidate_rows = [
        {"example_id": "ex-1", "prediction": "alpha"},
        {"example_id": "ex-2", "prediction": "gamma"},
    ]

    comparison = _compare_prediction_rows(reference_rows=reference_rows, candidate_rows=candidate_rows)

    assert comparison["shared_example_count"] == 2
    assert comparison["changed_example_count"] == 1
    assert comparison["identical_prediction_rate"] == 0.5
    assert comparison["changed_example_ids_sample"] == ["ex-2"]


def test_parse_success_rate_uses_locked_parser() -> None:
    rows = [
        {"prediction": 'ANSWER: Paris\nQUOTE: "Paris is the capital of France."'},
        {"prediction": "not parseable"},
    ]

    assert _parse_success_rate("Q", rows) == 0.5


def test_collect_aggregate_predictions_reads_completed_task_outputs(tmp_path: Path) -> None:
    first_predictions = tmp_path / "first.jsonl"
    second_predictions = tmp_path / "second.jsonl"
    first_predictions.write_text(json.dumps({"example_id": "ex-1", "prediction": "alpha"}) + "\n", encoding="utf-8")
    second_predictions.write_text(json.dumps({"example_id": "ex-2", "prediction": "beta"}) + "\n", encoding="utf-8")

    state = {
        "tasks": {
            "first": {"status": "completed", "predictions_path": str(first_predictions)},
            "second": {"status": "completed", "predictions_path": str(second_predictions)},
            "pending": {"status": "pending", "predictions_path": str(tmp_path / "missing.jsonl")},
        }
    }

    rows = _collect_aggregate_predictions(state)

    assert rows == [
        {"example_id": "ex-1", "prediction": "alpha", "task_name": "first"},
        {"example_id": "ex-2", "prediction": "beta", "task_name": "second"},
    ]


def test_loaded_adapter_evaluator_releases_temporary_adapter_and_restores_previous() -> None:
    class DummyModel:
        def __init__(self) -> None:
            self.deleted: list[str] = []
            self.set_calls: list[str] = []

        def delete_adapter(self, name: str) -> None:
            self.deleted.append(name)

        def set_adapter(self, name: str) -> None:
            self.set_calls.append(name)

    evaluator = object.__new__(LoadedAdapterEvaluator)
    evaluator._model = DummyModel()
    evaluator._loaded_adapters = {"scaffold", "candidate"}
    evaluator._active_adapter_name = "candidate"

    evaluator._release_adapter("candidate", restore_adapter="scaffold")

    assert evaluator._loaded_adapters == {"scaffold"}
    assert evaluator._active_adapter_name == "scaffold"
    assert evaluator._model.deleted == ["candidate"]
    assert evaluator._model.set_calls == ["scaffold"]


def test_loaded_adapter_evaluator_uses_resolved_local_model_source(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class DummyTokenizer:
        def __init__(self) -> None:
            self.pad_token = None
            self.eos_token = "<eos>"

        @classmethod
        def from_pretrained(cls, name_or_path: str, *, local_files_only: bool):
            calls.append(("tokenizer", name_or_path))
            assert local_files_only is True
            return cls()

    class DummyModel:
        def to(self, _device: object) -> "DummyModel":
            return self

    class DummyAutoModel:
        @staticmethod
        def from_pretrained(name_or_path: str, *, local_files_only: bool, torch_dtype: object):
            calls.append(("model", name_or_path))
            assert local_files_only is True
            assert torch_dtype == "bf16"
            return DummyModel()

    class DummyTorch:
        bfloat16 = "bf16"
        float16 = "fp16"
        float32 = "fp32"

        @staticmethod
        def device(name: str) -> str:
            return name

    monkeypatch.setattr(
        "overlay_algebra.analysis.adapter_eval.resolve_pretrained_source",
        lambda name_or_path, *, local_files_only: "C:/cached/Qwen2.5-7B-Instruct",
    )
    monkeypatch.setattr(
        "overlay_algebra.analysis.adapter_eval._import_eval_stack",
        lambda: (DummyTorch, DummyAutoModel, DummyTokenizer, object),
    )

    evaluator = LoadedAdapterEvaluator(
        base_model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        local_files_only=True,
        device="cuda",
        torch_dtype="bf16",
    )

    assert evaluator.tokenizer.pad_token == "<eos>"
    assert calls == [
        ("tokenizer", "C:/cached/Qwen2.5-7B-Instruct"),
        ("model", "C:/cached/Qwen2.5-7B-Instruct"),
    ]


def test_load_or_build_eval_features_uses_logical_model_id_for_cache_descriptor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_path = tmp_path / "test.jsonl"
    dataset_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "overlay_algebra.analysis.adapter_eval.load_feature_cache_bundle",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "overlay_algebra.analysis.adapter_eval._build_eval_features",
        lambda **kwargs: [],
    )

    saved_descriptors: list[object] = []

    def _capture_save(*args, **kwargs) -> None:
        saved_descriptors.append(kwargs["descriptor"])

    monkeypatch.setattr(
        "overlay_algebra.analysis.adapter_eval.save_feature_cache_bundle",
        _capture_save,
    )

    class DummyTokenizer:
        name_or_path = "C:/cached/Qwen2.5-7B-Instruct"

    bundle = load_or_build_eval_features(
        dataset_path=dataset_path,
        condition="J",
        tokenizer=DummyTokenizer(),
        base_model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        max_length=16,
        max_examples=1,
        reuse_feature_cache=True,
    )

    assert bundle.descriptor.tokenizer_name_or_path == "Qwen/Qwen2.5-7B-Instruct"
    assert saved_descriptors[0].tokenizer_name_or_path == "Qwen/Qwen2.5-7B-Instruct"
