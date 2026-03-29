from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..artifacts import RunPaths, build_run_paths
from ..config import AppConfig, load_app_config
from ..feature_cache import default_feature_cache_dir
from .adapter_eval import (
    EvalFeatureBundle,
    LoadedAdapterEvaluator,
    write_environment_summary,
    write_json,
    write_jsonl,
)


def _payload_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None:
        raise KeyError(f"Missing required payload field: {key}")
    return str(value)


def _payload_int(payload: dict[str, Any], key: str, default: int) -> int:
    if key not in payload:
        return default
    return int(payload[key])


def _payload_float(payload: dict[str, Any], key: str, default: float) -> float:
    if key not in payload:
        return default
    return float(payload[key])


def _payload_bool(payload: dict[str, Any], key: str, default: bool) -> bool:
    if key not in payload:
        return default
    value = payload[key]
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean payload field {key!r} from {value!r}")


@dataclass(frozen=True, slots=True)
class BatchBenchmarkConfig:
    app: AppConfig
    base_model_name_or_path: str
    adapter_dir: Path
    adapter_name: str
    condition: str
    eval_dataset_path: Path
    local_files_only: bool
    max_length: int
    max_eval_examples: int
    max_new_tokens: int
    device: str
    torch_dtype: str
    reuse_feature_cache: bool
    feature_cache_dir: Path
    sort_eval_by_length: bool
    batch_sizes: tuple[int, ...]
    reference_batch_size: int
    stress_slice_size: int
    representative_slice_size: int
    minimum_speedup_fraction: float

    @classmethod
    def from_app_config(cls, app: AppConfig) -> "BatchBenchmarkConfig":
        payload = app.payload
        batch_sizes = tuple(sorted({int(item) for item in payload.get("batch_sizes", [4])}))
        if not batch_sizes:
            raise ValueError("payload.batch_sizes must contain at least one batch size")
        reference_batch_size = _payload_int(payload, "reference_batch_size", batch_sizes[0])
        if reference_batch_size not in batch_sizes:
            batch_sizes = tuple(sorted({*batch_sizes, reference_batch_size}))
        eval_dataset_path = Path(app.paths["eval_dataset"])
        return cls(
            app=app,
            base_model_name_or_path=_payload_string(payload, "base_model_name_or_path"),
            adapter_dir=Path(app.paths["adapter"]),
            adapter_name=str(payload.get("adapter_name", "benchmark_target")),
            condition=_payload_string(payload, "condition"),
            eval_dataset_path=eval_dataset_path,
            local_files_only=_payload_bool(payload, "local_files_only", True),
            max_length=_payload_int(payload, "max_length", 768),
            max_eval_examples=_payload_int(payload, "max_eval_examples", 1),
            max_new_tokens=_payload_int(payload, "max_new_tokens", 32),
            device=_payload_string(payload, "device"),
            torch_dtype=_payload_string(payload, "torch_dtype"),
            reuse_feature_cache=_payload_bool(payload, "reuse_feature_cache", True),
            feature_cache_dir=Path(app.paths["feature_cache_dir"])
            if "feature_cache_dir" in app.paths
            else default_feature_cache_dir(eval_dataset_path),
            sort_eval_by_length=_payload_bool(payload, "sort_eval_by_length", True),
            batch_sizes=batch_sizes,
            reference_batch_size=reference_batch_size,
            stress_slice_size=_payload_int(payload, "stress_slice_size", 512),
            representative_slice_size=_payload_int(payload, "representative_slice_size", 1024),
            minimum_speedup_fraction=_payload_float(payload, "minimum_speedup_fraction", 0.15),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkSlice:
    name: str
    features: list[dict[str, Any]]
    selection_policy: str


def _feature_sort_key(feature: Mapping[str, Any]) -> tuple[int, int, str]:
    return (
        int(feature.get("prompt_token_length", 0)),
        int(feature.get("target_token_length", 0)),
        str(feature.get("example_id", "")),
    )


def _stress_slice(features: Sequence[Mapping[str, Any]], size: int) -> list[dict[str, Any]]:
    if size >= len(features):
        return [dict(feature) for feature in features]
    ordered = sorted(features, key=_feature_sort_key, reverse=True)
    return [dict(feature) for feature in ordered[:size]]


def _length_stratified_slice(features: Sequence[Mapping[str, Any]], size: int) -> list[dict[str, Any]]:
    if size >= len(features):
        return [dict(feature) for feature in features]
    ordered = sorted(features, key=_feature_sort_key)
    if size <= 1:
        return [dict(ordered[len(ordered) // 2])]
    chosen_indices: list[int] = []
    last_index = len(ordered) - 1
    for offset in range(size):
        raw = round((offset * last_index) / (size - 1))
        index = min(max(int(raw), 0), last_index)
        if chosen_indices and index <= chosen_indices[-1]:
            index = min(chosen_indices[-1] + 1, last_index)
        chosen_indices.append(index)
    deduped_indices = sorted(set(chosen_indices))
    if len(deduped_indices) < size:
        for index in range(len(ordered)):
            if index not in deduped_indices:
                deduped_indices.append(index)
            if len(deduped_indices) == size:
                break
        deduped_indices.sort()
    return [dict(ordered[index]) for index in deduped_indices[:size]]


def _build_benchmark_slices(features: Sequence[Mapping[str, Any]], config: BatchBenchmarkConfig) -> tuple[BenchmarkSlice, ...]:
    return (
        BenchmarkSlice(
            name="stress",
            features=_stress_slice(features, config.stress_slice_size),
            selection_policy="longest_prompt_and_target_tail",
        ),
        BenchmarkSlice(
            name="representative",
            features=_length_stratified_slice(features, config.representative_slice_size),
            selection_policy="deterministic_length_stratified",
        ),
    )


def _prediction_texts(predictions: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(row["prediction"]) for row in predictions]


def _prediction_hash(predictions: Sequence[Mapping[str, Any]]) -> str:
    joined = "\n".join(_prediction_texts(predictions)).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def _metric_hash(metrics: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(metrics.keys()):
        digest.update(f"{key}={metrics[key]!r}\n".encode("utf-8"))
    return digest.hexdigest()


def _generated_token_count(predictions: Sequence[Mapping[str, Any]]) -> int:
    total = 0
    for row in predictions:
        generation = row.get("generation")
        if isinstance(generation, Mapping):
            total += int(generation.get("generated_token_count", 0))
    return total


def _throughput_examples_per_second(example_count: int, wall_seconds: float) -> float:
    if wall_seconds <= 0:
        return 0.0
    return example_count / wall_seconds


def _throughput_tokens_per_second(total_tokens: int, wall_seconds: float) -> float:
    if wall_seconds <= 0:
        return 0.0
    return total_tokens / wall_seconds


def _slice_summary(features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prompt_lengths = [int(feature.get("prompt_token_length", 0)) for feature in features]
    target_lengths = [int(feature.get("target_token_length", 0)) for feature in features]
    return {
        "example_count": len(features),
        "prompt_token_length": {
            "min": min(prompt_lengths) if prompt_lengths else 0,
            "max": max(prompt_lengths) if prompt_lengths else 0,
            "mean": (sum(prompt_lengths) / len(prompt_lengths)) if prompt_lengths else 0.0,
        },
        "target_token_length": {
            "min": min(target_lengths) if target_lengths else 0,
            "max": max(target_lengths) if target_lengths else 0,
            "mean": (sum(target_lengths) / len(target_lengths)) if target_lengths else 0.0,
        },
    }


def _ensure_run_layout(run_paths: RunPaths, source_config: Path) -> None:
    run_paths.run_dir.mkdir(parents=True, exist_ok=True)
    run_paths.control_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_config, run_paths.resolved_config)


def _gpu_memory_mib(evaluator: LoadedAdapterEvaluator) -> float | None:
    torch = evaluator._torch
    device = evaluator._device
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)


def _reset_peak_memory(evaluator: LoadedAdapterEvaluator) -> None:
    torch = evaluator._torch
    device = evaluator._device
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def _synchronize(evaluator: LoadedAdapterEvaluator) -> None:
    torch = evaluator._torch
    device = evaluator._device
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _capture_mismatch_examples(
    *,
    reference_predictions: Sequence[Mapping[str, Any]],
    candidate_predictions: Sequence[Mapping[str, Any]],
    limit: int = 3,
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for reference_row, candidate_row in zip(reference_predictions, candidate_predictions, strict=True):
        if reference_row["prediction"] == candidate_row["prediction"]:
            continue
        mismatches.append(
            {
                "example_id": str(reference_row["example_id"]),
                "reference_prediction": str(reference_row["prediction"]),
                "candidate_prediction": str(candidate_row["prediction"]),
            }
        )
        if len(mismatches) >= limit:
            break
    return mismatches


def _is_cuda_oom(error: RuntimeError) -> bool:
    text = str(error).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _select_recommended_batch_size(
    *,
    results_by_batch: Mapping[int, Mapping[str, Any]],
    reference_batch_size: int,
    minimum_speedup_fraction: float,
) -> dict[str, Any]:
    recommended = reference_batch_size
    accepted_batches = [reference_batch_size]
    rejection_reasons: dict[str, str] = {}
    reference = results_by_batch[reference_batch_size]
    reference_representative = reference["slices"]["representative"]
    reference_examples_per_second = float(reference_representative["throughput"]["examples_per_second"])

    for batch_size in sorted(results_by_batch.keys()):
        if batch_size == reference_batch_size:
            continue
        result = results_by_batch[batch_size]
        stress = result["slices"]["stress"]
        representative = result["slices"]["representative"]
        if result["status"] != "success":
            rejection_reasons[str(batch_size)] = str(result.get("error", "benchmark_failed"))
            continue
        if not bool(stress["exact_prediction_match"]):
            rejection_reasons[str(batch_size)] = "stress_predictions_changed"
            continue
        if not bool(representative["exact_prediction_match"]):
            rejection_reasons[str(batch_size)] = "representative_predictions_changed"
            continue
        if not bool(stress["exact_metric_match"]):
            rejection_reasons[str(batch_size)] = "stress_metrics_changed"
            continue
        if not bool(representative["exact_metric_match"]):
            rejection_reasons[str(batch_size)] = "representative_metrics_changed"
            continue
        throughput = float(representative["throughput"]["examples_per_second"])
        speedup_fraction = (
            ((throughput - reference_examples_per_second) / reference_examples_per_second)
            if reference_examples_per_second > 0
            else 0.0
        )
        if speedup_fraction < minimum_speedup_fraction:
            rejection_reasons[str(batch_size)] = f"speedup_below_threshold:{speedup_fraction:.4f}"
            continue
        accepted_batches.append(batch_size)
        recommended = batch_size

    return {
        "reference_batch_size": reference_batch_size,
        "minimum_speedup_fraction": minimum_speedup_fraction,
        "accepted_batches": accepted_batches,
        "recommended_batch_size": recommended,
        "rejection_reasons": rejection_reasons,
    }


def run_eval_batch_benchmark(config: BatchBenchmarkConfig) -> dict[str, Any]:
    run_paths = build_run_paths(config.app)
    if run_paths.run_dir.exists():
        raise FileExistsError(f"Benchmark run directory already exists: {run_paths.run_dir}")
    _ensure_run_layout(run_paths, config.app.source_path)
    write_environment_summary(run_paths.environment)

    evaluator = LoadedAdapterEvaluator(
        base_model_name_or_path=config.base_model_name_or_path,
        local_files_only=config.local_files_only,
        device=config.device,
        torch_dtype=config.torch_dtype,
        reuse_feature_cache=config.reuse_feature_cache,
    )
    feature_bundle = evaluator.get_eval_features(
        dataset_path=config.eval_dataset_path,
        condition=config.condition,
        max_length=config.max_length,
        max_examples=config.max_eval_examples,
        cache_dir=config.feature_cache_dir,
    )
    benchmark_slices = _build_benchmark_slices(feature_bundle.features, config)
    evaluator._ensure_adapter_loaded(adapter_name=config.adapter_name, adapter_dir=config.adapter_dir)

    reference_batch_size = config.reference_batch_size
    results_by_batch: dict[int, dict[str, Any]] = {}
    reference_payloads: dict[str, dict[str, Any]] = {}

    ordered_batch_sizes = (reference_batch_size,) + tuple(
        batch_size for batch_size in config.batch_sizes if batch_size != reference_batch_size
    )

    for batch_size in ordered_batch_sizes:
        batch_result: dict[str, Any] = {
            "batch_size": batch_size,
            "status": "success",
            "slices": {},
        }
        try:
            for benchmark_slice in benchmark_slices:
                _reset_peak_memory(evaluator)
                _synchronize(evaluator)
                started = time.perf_counter()
                evaluation = evaluator.evaluate_adapter(
                    adapter_name=config.adapter_name,
                    adapter_dir=config.adapter_dir,
                    eval_features=benchmark_slice.features,
                    condition=config.condition,
                    max_length=config.max_length,
                    max_new_tokens=config.max_new_tokens,
                    batch_size=batch_size,
                    sort_by_length=config.sort_eval_by_length,
                    adapter_mode="persistent",
                )
                _synchronize(evaluator)
                wall_seconds = time.perf_counter() - started
                total_generated_tokens = _generated_token_count(evaluation.predictions)
                slice_dir = run_paths.run_dir / benchmark_slice.name / f"batch_{batch_size:02d}"
                write_json(
                    slice_dir / "metrics.json",
                    {
                        "metrics": evaluation.metrics,
                        "generation_summary": evaluation.generation_summary,
                        "slice_summary": _slice_summary(benchmark_slice.features),
                        "throughput": {
                            "wall_seconds": wall_seconds,
                            "examples_per_second": _throughput_examples_per_second(len(benchmark_slice.features), wall_seconds),
                            "generated_tokens_per_second": _throughput_tokens_per_second(total_generated_tokens, wall_seconds),
                        },
                        "peak_memory_mib": _gpu_memory_mib(evaluator),
                    },
                )
                write_jsonl(slice_dir / "predictions.jsonl", evaluation.predictions)
                payload = {
                    "metrics": dict(evaluation.metrics),
                    "metric_hash": _metric_hash(evaluation.metrics),
                    "prediction_hash": _prediction_hash(evaluation.predictions),
                    "prediction_row_count": len(evaluation.predictions),
                    "generation_summary": dict(evaluation.generation_summary),
                    "slice_summary": _slice_summary(benchmark_slice.features),
                    "selection_policy": benchmark_slice.selection_policy,
                    "throughput": {
                        "wall_seconds": wall_seconds,
                        "examples_per_second": _throughput_examples_per_second(len(benchmark_slice.features), wall_seconds),
                        "generated_tokens_per_second": _throughput_tokens_per_second(total_generated_tokens, wall_seconds),
                    },
                    "peak_memory_mib": _gpu_memory_mib(evaluator),
                }
                if batch_size == reference_batch_size:
                    payload["exact_prediction_match"] = True
                    payload["exact_metric_match"] = True
                    payload["mismatch_examples"] = []
                    reference_payloads[benchmark_slice.name] = {
                        "metrics": dict(evaluation.metrics),
                        "predictions": list(evaluation.predictions),
                    }
                else:
                    reference_payload = reference_payloads[benchmark_slice.name]
                    payload["exact_prediction_match"] = _prediction_texts(evaluation.predictions) == _prediction_texts(
                        reference_payload["predictions"]
                    )
                    payload["exact_metric_match"] = dict(evaluation.metrics) == dict(reference_payload["metrics"])
                    payload["mismatch_examples"] = _capture_mismatch_examples(
                        reference_predictions=reference_payload["predictions"],
                        candidate_predictions=evaluation.predictions,
                    )
                batch_result["slices"][benchmark_slice.name] = payload
        except RuntimeError as error:
            if not _is_cuda_oom(error):
                raise
            batch_result["status"] = "oom"
            batch_result["error"] = str(error)
            torch = evaluator._torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results_by_batch[batch_size] = batch_result
        if batch_result["status"] == "oom":
            break

    selection = _select_recommended_batch_size(
        results_by_batch=results_by_batch,
        reference_batch_size=reference_batch_size,
        minimum_speedup_fraction=config.minimum_speedup_fraction,
    )
    metrics = {
        "status": "completed",
        "run_dir": str(run_paths.run_dir),
        "metrics_path": str(run_paths.metrics),
        "run_manifest_path": str(run_paths.run_manifest),
        "adapter_dir": str(config.adapter_dir),
        "adapter_name": config.adapter_name,
        "condition": config.condition,
        "eval_dataset_path": str(config.eval_dataset_path),
        "feature_cache_path": str(feature_bundle.cache_path),
        "feature_cache_hit": feature_bundle.cache_hit,
        "batch_sizes": list(config.batch_sizes),
        "reference_batch_size": reference_batch_size,
        "results_by_batch": {str(key): value for key, value in results_by_batch.items()},
        "selection": selection,
    }
    write_json(run_paths.metrics, metrics)
    write_json(
        run_paths.run_manifest,
        {
            "base_model_name_or_path": config.base_model_name_or_path,
            "adapter_dir": str(config.adapter_dir),
            "adapter_name": config.adapter_name,
            "condition": config.condition,
            "eval_dataset_path": str(config.eval_dataset_path),
            "max_length": config.max_length,
            "max_eval_examples": config.max_eval_examples,
            "max_new_tokens": config.max_new_tokens,
            "device": config.device,
            "torch_dtype": config.torch_dtype,
            "reuse_feature_cache": config.reuse_feature_cache,
            "feature_cache_dir": str(config.feature_cache_dir),
            "sort_eval_by_length": config.sort_eval_by_length,
            "batch_sizes": list(config.batch_sizes),
            "reference_batch_size": config.reference_batch_size,
            "stress_slice_size": config.stress_slice_size,
            "representative_slice_size": config.representative_slice_size,
            "minimum_speedup_fraction": config.minimum_speedup_fraction,
        },
    )
    return metrics


def load_batch_benchmark_config(path: str | Path) -> BatchBenchmarkConfig:
    app = load_app_config(path)
    return BatchBenchmarkConfig.from_app_config(app)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark evaluation batch sizes on a fixed adapter and validation slices.")
    parser.add_argument("--config", required=True, help="Path to the benchmark config YAML.")
    parser.add_argument("--json", action="store_true", help="Print the full JSON summary instead of only the recommendation.")
    args = parser.parse_args(argv)

    metrics = run_eval_batch_benchmark(load_batch_benchmark_config(args.config))
    if args.json:
        print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    else:
        print(metrics["selection"]["recommended_batch_size"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
