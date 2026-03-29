from __future__ import annotations

import json
from pathlib import Path

from overlay_algebra.analysis.residualize import load_residualize_config
from overlay_algebra.feature_cache import (
    build_feature_cache_descriptor,
    feature_cache_path,
    load_feature_cache,
    save_feature_cache,
)
from overlay_algebra.inference import batched_greedy_generate_records, batched_greedy_generate_texts
from overlay_algebra.prompts import prompt_protocol_fingerprint
from overlay_algebra.smoke_pipeline import load_eval_smoke_config, load_train_smoke_config


REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeTensor:
    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = [list(row) for row in rows]

    @property
    def shape(self) -> tuple[int, int]:
        width = len(self.rows[0]) if self.rows else 0
        return len(self.rows), width

    def to(self, _device: object) -> "_FakeTensor":
        return self

    def __getitem__(self, key: object) -> list[list[int]] | list[int]:
        if isinstance(key, tuple):
            row_selector, column_selector = key
            if isinstance(row_selector, slice):
                selected_rows = self.rows[row_selector]
            else:
                selected_rows = [self.rows[int(row_selector)]]
            return [row[column_selector] for row in selected_rows]
        return self.rows[int(key)]

    def __iter__(self):
        return iter(self.rows)


class _FakeTokenizer:
    def __init__(self) -> None:
        self.padding_side = "right"
        self.pad_token_id = 0
        self.eos_token_id = 99

    def __call__(
        self,
        prompts: list[str],
        *,
        return_tensors: str,
        padding: bool,
        truncation: bool,
        max_length: int,
    ) -> dict[str, _FakeTensor]:
        assert return_tensors == "pt"
        assert padding is True
        assert truncation is True
        token_rows = []
        mask_rows = []
        encoded_rows = [[len(token) for token in prompt.split()][:max_length] for prompt in prompts]
        width = max(len(row) for row in encoded_rows)
        for row in encoded_rows:
            pad_len = width - len(row)
            if self.padding_side == "left":
                token_rows.append([self.pad_token_id] * pad_len + row)
                mask_rows.append([0] * pad_len + [1] * len(row))
            else:
                token_rows.append(row + [self.pad_token_id] * pad_len)
                mask_rows.append([1] * len(row) + [0] * pad_len)
        return {
            "input_ids": _FakeTensor(token_rows),
            "attention_mask": _FakeTensor(mask_rows),
        }

    def decode(self, generated_tokens: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return "|".join(str(token) for token in generated_tokens)


class _FakeModel:
    def generate(
        self,
        *,
        input_ids: _FakeTensor,
        attention_mask: _FakeTensor,
        max_new_tokens: int,
        do_sample: bool,
        pad_token_id: int,
        eos_token_id: int,
    ) -> _FakeTensor:
        assert do_sample is False
        assert attention_mask.shape == input_ids.shape
        generated_rows = []
        for row in input_ids.rows:
            non_pad_tokens = [token for token in row if token != pad_token_id]
            prompt_signal = non_pad_tokens[-1] if non_pad_tokens else 0
            suffix = [prompt_signal, min(max_new_tokens, 7), eos_token_id]
            generated_rows.append(row + suffix)
        return _FakeTensor(generated_rows)


def test_batched_greedy_generate_texts_preserves_order_and_restores_padding_side() -> None:
    tokenizer = _FakeTokenizer()
    model = _FakeModel()

    predictions = batched_greedy_generate_texts(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        prompts=["alpha beta", "gamma", "delta epsilon zeta"],
        max_length=16,
        max_new_tokens=5,
        batch_size=2,
    )

    assert predictions == ["4|5|99", "5|5|99", "4|5|99"]
    assert tokenizer.padding_side == "right"


def test_batched_greedy_generate_records_sorting_preserves_original_example_order() -> None:
    tokenizer = _FakeTokenizer()
    model = _FakeModel()

    records = batched_greedy_generate_records(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        prompts=["alpha beta", "gamma", "delta epsilon zeta"],
        max_length=16,
        max_new_tokens=5,
        batch_size=2,
        sort_by_length=True,
        prompt_length_hints=[2, 1, 3],
        generation_length_hints=[10, 2, 50],
    )

    assert [record.text for record in records] == ["4|5|99", "5|5|99", "4|5|99"]
    assert [record.prompt_token_length for record in records] == [2, 1, 3]
    assert all(record.stopped_by_eos is True for record in records)
    assert all(record.hit_max_new_tokens is False for record in records)


def test_feature_cache_uses_dataset_manifest_identity(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "processed"
    dataset_dir.mkdir()
    dataset_path = dataset_dir / "train.jsonl"
    dataset_path.write_text("{}", encoding="utf-8")
    (dataset_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": "squad_jcq_v1",
                "manifest_sha256": "manifest-hash",
                "hashes": {"train_jsonl": "train-hash"},
            }
        ),
        encoding="utf-8",
    )

    descriptor = build_feature_cache_descriptor(
        dataset_path=dataset_path,
        feature_kind="train_supervised",
        base_model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        tokenizer_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        condition="J",
        prompt_protocol_sha256=prompt_protocol_fingerprint("J"),
        max_length=768,
        max_examples=78098,
    )

    assert descriptor.dataset_version == "squad_jcq_v1"
    assert descriptor.dataset_sha256 == "train-hash"
    assert descriptor.manifest_sha256 == "manifest-hash"


def test_feature_cache_round_trip_requires_matching_descriptor(tmp_path: Path) -> None:
    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text("{}", encoding="utf-8")
    descriptor = build_feature_cache_descriptor(
        dataset_path=dataset_path,
        feature_kind="train_supervised",
        base_model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        tokenizer_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        condition="plain",
        prompt_protocol_sha256=prompt_protocol_fingerprint("plain"),
        max_length=768,
        max_examples=78098,
    )
    cache_path = feature_cache_path(tmp_path / "cache", descriptor)
    save_feature_cache(
        cache_path,
        descriptor=descriptor,
        features=[{"example_id": "ex-1", "input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]}],
    )

    loaded = load_feature_cache(cache_path, descriptor=descriptor)
    mismatched = build_feature_cache_descriptor(
        dataset_path=dataset_path,
        feature_kind="train_supervised",
        base_model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        tokenizer_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        condition="J",
        prompt_protocol_sha256=prompt_protocol_fingerprint("J"),
        max_length=768,
        max_examples=78098,
    )

    assert loaded is not None
    assert loaded[0]["example_id"] == "ex-1"
    assert load_feature_cache(cache_path, descriptor=mismatched) is None


def test_full_m4_train_configs_use_safe_runtime_speedups() -> None:
    for config_name in ("scaffold.yaml", "full_J.yaml", "full_C.yaml", "full_Q.yaml"):
        config = load_train_smoke_config(REPO_ROOT / "configs" / "train" / config_name)
        assert config.batch_size == 8
        assert config.gradient_accumulation_steps == 4
        assert config.checkpoint_interval_optimizer_steps == 500
        assert config.reuse_feature_cache is True
        assert config.feature_cache_dir.as_posix().endswith("data/processed/squad_jcq_v1/_feature_cache")


def test_full_m4_eval_and_residualize_configs_use_batched_eval() -> None:
    for config_name in (
        "full_prompt_only_J.yaml",
        "full_direct_J.yaml",
        "full_soar_J.yaml",
        "full_prompt_only_C.yaml",
        "full_direct_C.yaml",
        "full_soar_C.yaml",
        "full_prompt_only_Q.yaml",
        "full_direct_Q.yaml",
        "full_soar_Q.yaml",
    ):
        config = load_eval_smoke_config(REPO_ROOT / "configs" / "eval" / config_name)
        assert config.eval_batch_size == 4
        assert config.reuse_feature_cache is True
        assert config.sort_eval_by_length is True

    for config_name in ("full_residuals_J.yaml", "full_residuals_C.yaml", "full_residuals_Q.yaml"):
        config = load_residualize_config(REPO_ROOT / "configs" / "analysis" / config_name)
        assert config.eval_batch_size == 4
        assert config.reuse_feature_cache is True
        assert config.sort_eval_by_length is True
        assert config.checkpoint_interval_units == 1
        assert config.feature_cache_dir.as_posix().endswith("data/processed/squad_jcq_v1/_feature_cache")
