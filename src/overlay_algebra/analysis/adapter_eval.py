from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import json
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

from ..data.processed import ProcessedExample, load_processed_examples
from ..feature_cache import (
    FeatureCacheDescriptor,
    build_feature_cache_descriptor,
    default_feature_cache_dir,
    feature_cache_path,
    load_feature_cache_bundle,
    save_feature_cache_bundle,
)
from ..hf_local import resolve_pretrained_source
from ..inference import GenerationRecord, batched_greedy_generate_records
from ..metrics import score_prediction
from ..prompts import prompt_protocol_fingerprint, render_prompt


def _import_eval_stack() -> tuple[Any, Any, Any, Any]:
    missing = [
        name
        for name in ("torch", "transformers", "peft")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeError(
            "Missing evaluation runtime dependencies: "
            + ", ".join(missing)
            + ". Use the CUDA-enabled runtime before running evaluation-heavy analysis jobs."
        )

    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    peft = importlib.import_module("peft")
    return torch, transformers.AutoModelForCausalLM, transformers.AutoTokenizer, peft.PeftModel


def resolve_dtype(torch_module: Any, dtype_name: str) -> Any:
    mapping = {
        "bf16": torch_module.bfloat16,
        "bfloat16": torch_module.bfloat16,
        "fp16": torch_module.float16,
        "float16": torch_module.float16,
        "fp32": torch_module.float32,
        "float32": torch_module.float32,
    }
    key = dtype_name.strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {dtype_name}")
    return mapping[key]


def _version_or_missing(name: str) -> str:
    try:
        return importlib.import_module(name).__version__
    except Exception:
        return "missing"


def _cuda_available_text() -> str:
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return "unknown"
    try:
        return str(bool(torch.cuda.is_available()))
    except Exception:
        return "unknown"


def write_environment_summary(path: Path) -> None:
    lines = [
        f"python={platform.python_version()}",
        f"platform={platform.platform()}",
        f"torch={_version_or_missing('torch')}",
        f"transformers={_version_or_missing('transformers')}",
        f"peft={_version_or_missing('peft')}",
        f"cuda_available={_cuda_available_text()}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True, slots=True)
class EvalFeatureBundle:
    features: list[dict[str, Any]]
    descriptor: FeatureCacheDescriptor
    cache_path: Path
    cache_hit: bool


@dataclass(frozen=True, slots=True)
class AdapterEvaluationResult:
    metrics: dict[str, float]
    predictions: list[dict[str, Any]]
    generation_summary: dict[str, Any]
    feature_cache_hit: bool
    feature_cache_path: str


@dataclass(frozen=True, slots=True)
class EvalTaskSpec:
    name: str
    adapter_name: str
    adapter_dir: Path
    condition: str
    max_length: int
    max_new_tokens: int
    batch_size: int
    adapter_mode: str = "persistent"


def mean_metric_rows(rows: Sequence[dict[str, float | None]]) -> dict[str, float]:
    metrics: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if value is None:
                continue
            metrics.setdefault(key, []).append(float(value))
    return {key: (sum(values) / len(values) if values else 0.0) for key, values in metrics.items()}


def load_examples(path: Path, *, limit: int) -> list[ProcessedExample]:
    examples = load_processed_examples(path, limit=limit)
    if not examples:
        raise ValueError(f"No processed examples found in {path}")
    return examples


def _build_eval_features_from_examples(
    *,
    examples: Sequence[ProcessedExample],
    condition: str,
    tokenizer: Any,
    max_length: int,
) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for example in examples:
        prompt = render_prompt(example.question, example.labeled_context, condition)
        prompt_tokens = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=max_length)
        target_tokens = tokenizer(example.targets[condition], add_special_tokens=False)
        features.append(
            {
                "example_id": example.example_id,
                "condition": condition,
                "prompt": prompt,
                "target": example.targets[condition],
                "gold_answer": example.answer_text,
                "gold_support_idx": example.support_sent_idx,
                "gold_support_sentence": example.support_sentence,
                "prompt_token_length": len(prompt_tokens["input_ids"]),
                "target_token_length": len(target_tokens["input_ids"]),
            }
        )
    return features


def _build_eval_features(
    *,
    dataset_path: Path,
    condition: str,
    tokenizer: Any,
    max_length: int,
    max_examples: int,
) -> list[dict[str, Any]]:
    return _build_eval_features_from_examples(
        examples=load_examples(dataset_path, limit=max_examples),
        condition=condition,
        tokenizer=tokenizer,
        max_length=max_length,
    )


def load_or_build_eval_features(
    *,
    dataset_path: Path,
    condition: str,
    tokenizer: Any,
    base_model_name_or_path: str,
    max_length: int,
    max_examples: int,
    cache_dir: Path | None = None,
    reuse_feature_cache: bool = True,
) -> EvalFeatureBundle:
    effective_cache_dir = cache_dir or default_feature_cache_dir(dataset_path)
    descriptor = build_feature_cache_descriptor(
        dataset_path=dataset_path,
        feature_kind="eval_prompt",
        base_model_name_or_path=base_model_name_or_path,
        tokenizer_name_or_path=base_model_name_or_path,
        condition=condition,
        prompt_protocol_sha256=prompt_protocol_fingerprint(condition),
        max_length=max_length,
        max_examples=max_examples,
    )
    cache_path = feature_cache_path(effective_cache_dir, descriptor)
    if reuse_feature_cache:
        cached = load_feature_cache_bundle(cache_path, descriptor=descriptor)
        if cached is not None:
            return EvalFeatureBundle(
                features=cached["features"],
                descriptor=descriptor,
                cache_path=cache_path,
                cache_hit=True,
            )

    features = _build_eval_features(
        dataset_path=dataset_path,
        condition=condition,
        tokenizer=tokenizer,
        max_length=max_length,
        max_examples=max_examples,
    )
    if reuse_feature_cache:
        save_feature_cache_bundle(
            cache_path,
            descriptor=descriptor,
            features=features,
            metadata={
                "feature_count": len(features),
                "sort_hints": {
                    "prompt_token_length": True,
                    "target_token_length": True,
                },
            },
        )
    return EvalFeatureBundle(
        features=features,
        descriptor=descriptor,
        cache_path=cache_path,
        cache_hit=False,
    )


class LoadedAdapterEvaluator:
    def __init__(
        self,
        *,
        base_model_name_or_path: str,
        local_files_only: bool,
        device: str,
        torch_dtype: str,
        reuse_feature_cache: bool = True,
    ) -> None:
        torch, AutoModelForCausalLM, AutoTokenizer, PeftModel = _import_eval_stack()
        self._torch = torch
        self._peft_model_cls = PeftModel
        self._device = torch.device(device)
        self._base_model_name_or_path = base_model_name_or_path
        self._local_files_only = local_files_only
        self._reuse_feature_cache = reuse_feature_cache
        resolved_model_source = resolve_pretrained_source(
            base_model_name_or_path,
            local_files_only=local_files_only,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            resolved_model_source,
            local_files_only=local_files_only,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            resolved_model_source,
            local_files_only=local_files_only,
            torch_dtype=resolve_dtype(torch, torch_dtype),
        )
        self._base_model = base_model.to(self._device)
        self._model: Any | None = None
        self._loaded_adapters: set[str] = set()
        self._active_adapter_name: str | None = None

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    def get_eval_features(
        self,
        *,
        dataset_path: Path,
        condition: str,
        max_length: int,
        max_examples: int,
        cache_dir: Path | None = None,
    ) -> EvalFeatureBundle:
        return load_or_build_eval_features(
            dataset_path=dataset_path,
            condition=condition,
            tokenizer=self._tokenizer,
            base_model_name_or_path=self._base_model_name_or_path,
            max_length=max_length,
            max_examples=max_examples,
            cache_dir=cache_dir,
            reuse_feature_cache=self._reuse_feature_cache,
        )

    def _ensure_adapter_loaded(self, *, adapter_name: str, adapter_dir: Path) -> None:
        adapter_path = adapter_dir.resolve()
        if self._model is None:
            self._model = self._peft_model_cls.from_pretrained(self._base_model, adapter_path, adapter_name=adapter_name)
            self._loaded_adapters.add(adapter_name)
        elif adapter_name not in self._loaded_adapters:
            self._model.load_adapter(adapter_path, adapter_name=adapter_name, is_trainable=False)
            self._loaded_adapters.add(adapter_name)
        self._model.to(self._device)
        self._model.eval()
        self._model.set_adapter(adapter_name)
        self._active_adapter_name = adapter_name

    def _release_adapter(self, adapter_name: str, *, restore_adapter: str | None = None) -> None:
        if self._model is None:
            return

        deleted = False
        if hasattr(self._model, "delete_adapter"):
            self._model.delete_adapter(adapter_name)
            deleted = True

        if not deleted:
            # Fallback for PEFT runtimes without adapter deletion support.
            self._model = None
            self._loaded_adapters.clear()
            self._active_adapter_name = None
            return

        self._loaded_adapters.discard(adapter_name)
        if restore_adapter is not None and restore_adapter in self._loaded_adapters:
            self._model.set_adapter(restore_adapter)
            self._active_adapter_name = restore_adapter
            return
        self._active_adapter_name = None

    def evaluate_adapter(
        self,
        *,
        adapter_name: str,
        adapter_dir: Path,
        eval_features: Sequence[Mapping[str, Any]] | None = None,
        examples: Sequence[ProcessedExample] | None = None,
        condition: str,
        max_length: int,
        max_new_tokens: int,
        batch_size: int,
        sort_by_length: bool = True,
        adapter_mode: str = "persistent",
    ) -> AdapterEvaluationResult:
        if eval_features is None and examples is None:
            raise ValueError("evaluate_adapter requires eval_features or examples")
        if eval_features is not None and examples is not None:
            raise ValueError("Pass either eval_features or examples, not both")
        if adapter_mode not in {"persistent", "temporary"}:
            raise ValueError(f"Unsupported adapter_mode: {adapter_mode}")

        previous_adapter = self._active_adapter_name if adapter_mode == "temporary" else None
        self._ensure_adapter_loaded(adapter_name=adapter_name, adapter_dir=adapter_dir)
        effective_features = (
            [dict(feature) for feature in eval_features]
            if eval_features is not None
            else _build_eval_features_from_examples(
                examples=examples or [],
                condition=condition,
                tokenizer=self._tokenizer,
                max_length=max_length,
            )
        )

        prompts = [str(feature["prompt"]) for feature in effective_features]
        prompt_length_hints = [int(feature.get("prompt_token_length", 0)) for feature in effective_features]
        generation_length_hints = [int(feature.get("target_token_length", 0)) for feature in effective_features]
        metric_rows: list[dict[str, float | None]] = []
        prediction_rows: list[dict[str, Any]] = []
        cap_hit_count = 0
        eos_hit_count = 0
        generated_token_lengths: list[int] = []
        try:
            with self._torch.inference_mode():
                generation_records = batched_greedy_generate_records(
                    model=self._model,
                    tokenizer=self._tokenizer,
                    device=self._device,
                    prompts=prompts,
                    max_length=max_length,
                    max_new_tokens=max_new_tokens,
                    batch_size=max(1, batch_size),
                    sort_by_length=sort_by_length,
                    prompt_length_hints=prompt_length_hints,
                    generation_length_hints=generation_length_hints,
                )
                for feature, generation in zip(effective_features, generation_records, strict=True):
                    metrics = score_prediction(
                        condition,
                        generation.text,
                        gold_answer=str(feature["gold_answer"]),
                        gold_support_idx=(int(feature["gold_support_idx"]) if feature.get("gold_support_idx") is not None else None),
                        gold_support_sentence=(
                            str(feature["gold_support_sentence"]) if feature.get("gold_support_sentence") is not None else None
                        ),
                    )
                    metric_rows.append(metrics)
                    prediction_rows.append(
                        {
                            "example_id": str(feature["example_id"]),
                            "condition": condition,
                            "prompt": str(feature["prompt"]),
                            "target": str(feature["target"]),
                            "prediction": generation.text,
                            "metrics": metrics,
                            "generation": {
                                "generated_token_count": generation.generated_token_count,
                                "prompt_token_length": generation.prompt_token_length,
                                "stopped_by_eos": generation.stopped_by_eos,
                                "hit_max_new_tokens": generation.hit_max_new_tokens,
                            },
                        }
                    )
                    generated_token_lengths.append(generation.generated_token_count)
                    if generation.hit_max_new_tokens:
                        cap_hit_count += 1
                    if generation.stopped_by_eos:
                        eos_hit_count += 1
            return AdapterEvaluationResult(
                metrics=mean_metric_rows(metric_rows),
                predictions=prediction_rows,
                generation_summary={
                    "generated_examples": len(effective_features),
                    "cap_hit_count": cap_hit_count,
                    "eos_hit_count": eos_hit_count,
                    "cap_hit_rate": cap_hit_count / len(effective_features) if effective_features else 0.0,
                    "eos_hit_rate": eos_hit_count / len(effective_features) if effective_features else 0.0,
                    "mean_generated_token_count": (
                        sum(generated_token_lengths) / len(generated_token_lengths) if generated_token_lengths else 0.0
                    ),
                    "max_generated_token_count": max(generated_token_lengths) if generated_token_lengths else 0,
                    "sort_by_length": sort_by_length,
                    "adapter_mode": adapter_mode,
                },
                feature_cache_hit=False,
                feature_cache_path="",
            )
        finally:
            if adapter_mode == "temporary":
                self._release_adapter(adapter_name, restore_adapter=previous_adapter)

    def evaluate_bundle(
        self,
        *,
        tasks: Sequence[EvalTaskSpec],
        features_by_task: Mapping[str, EvalFeatureBundle],
        sort_by_length: bool = True,
    ) -> dict[str, AdapterEvaluationResult]:
        results: dict[str, AdapterEvaluationResult] = {}
        for task in tasks:
            feature_bundle = features_by_task[task.name]
            result = self.evaluate_adapter(
                adapter_name=task.adapter_name,
                adapter_dir=task.adapter_dir,
                eval_features=feature_bundle.features,
                condition=task.condition,
                max_length=task.max_length,
                max_new_tokens=task.max_new_tokens,
                batch_size=task.batch_size,
                sort_by_length=sort_by_length,
                adapter_mode=task.adapter_mode,
            )
            results[task.name] = AdapterEvaluationResult(
                metrics=result.metrics,
                predictions=result.predictions,
                generation_summary=result.generation_summary,
                feature_cache_hit=feature_bundle.cache_hit,
                feature_cache_path=str(feature_bundle.cache_path),
            )
        return results


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=False) for row in rows)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")
