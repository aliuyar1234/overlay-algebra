from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
import json
from math import comb
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence

from ..adapter_io import (
    AdapterBundle,
    load_adapter_bundle,
    load_compact_delta_artifact,
    save_adapter_from_lora_factors,
    save_compact_delta_artifact,
)
from ..artifacts import RunPaths, build_run_paths
from ..config import AppConfig, load_app_config
from ..feature_cache import default_feature_cache_dir
from ..soar import (
    DenseDeltaFactors,
    combine_dense_factors,
    compose_factor_maps,
    dense_factor_map,
    factorize_dense_factors,
    simplex_grid,
)
from ..sweep_runtime import (
    SweepPlan,
    SweepRuntimeState,
    build_sweep_heartbeat,
    load_sweep_checkpoint,
    maybe_pause_sweep,
    resolve_sweep_checkpoint_dir,
    save_sweep_checkpoint,
)
from .adapter_eval import (
    EvalFeatureBundle,
    LoadedAdapterEvaluator,
    write_environment_summary,
    write_json,
    write_jsonl,
)


_PRIMARY_METRIC_BY_OVERLAY = {
    "J": "json_strict_schema_valid",
    "C": "citation_exact",
    "Q": "quote_exact",
}

_CANDIDATE_ARTIFACT_POLICIES = {"full", "minimal"}


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


def _payload_float_list(payload: dict[str, Any], key: str) -> tuple[float, ...]:
    value = payload.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError(f"Payload field {key!r} must be a list")
    return tuple(float(item) for item in value)


def _payload_choice_string(
    payload: dict[str, Any],
    key: str,
    *,
    default: str,
    allowed: set[str],
) -> str:
    if key not in payload:
        return default
    value = str(payload[key]).strip().lower()
    if value not in allowed:
        raise ValueError(f"Unsupported payload field {key!r}: {value!r}. Expected one of {sorted(allowed)}.")
    return value


def _parse_overlay_map(paths: Mapping[str, str], prefix: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for key, value in paths.items():
        lowered = key.lower()
        expected = f"{prefix}_"
        if not lowered.startswith(expected):
            continue
        suffix = lowered[len(expected) :].upper()
        if suffix in _PRIMARY_METRIC_BY_OVERLAY:
            result[suffix] = Path(value)
    return result


@dataclass(frozen=True, slots=True)
class CompositionTask:
    condition: str
    active_overlays: tuple[str, ...]
    max_new_tokens: int | None = None

    @property
    def name(self) -> str:
        return self.condition

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CompositionTask":
        condition = str(payload["condition"])
        if "active_overlays" in payload:
            raw_overlays = payload["active_overlays"]
            if not isinstance(raw_overlays, list):
                raise TypeError("composition active_overlays must be a list when provided")
            active_overlays = tuple(str(item) for item in raw_overlays)
        else:
            active_overlays = tuple(letter for letter in ("J", "C", "Q") if letter in condition)
        if not active_overlays:
            raise ValueError(f"Could not infer active overlays for composition {condition!r}")
        for overlay in active_overlays:
            if overlay not in _PRIMARY_METRIC_BY_OVERLAY:
                raise ValueError(f"Unsupported overlay {overlay!r} in composition task")
        return cls(
            condition=condition,
            active_overlays=active_overlays,
            max_new_tokens=(int(payload["max_new_tokens"]) if payload.get("max_new_tokens") is not None else None),
        )


@dataclass(frozen=True, slots=True)
class ComposeConfig:
    app: AppConfig
    base_model_name_or_path: str
    scaffold_adapter_dir: Path
    validation_dataset_path: Path
    local_files_only: bool
    alpha_grid: tuple[float, ...]
    merge_grid_step: float
    semantic_guardrail_points: float
    max_length: int
    default_max_new_tokens: int
    max_eval_examples: int
    eval_batch_size: int
    device: str
    torch_dtype: str
    reuse_feature_cache: bool
    feature_cache_dir: Path
    sort_eval_by_length: bool
    merge_search_policy: str
    candidate_artifact_policy: str
    synthesized_lora_dropout: float
    enforce_budget_parity: bool
    budget_parity_tolerance: int
    checkpoint_interval_units: int
    heartbeat_interval_seconds: float
    control_poll_seconds: float
    max_checkpoints_to_keep: int
    resume_latest: bool
    resume_checkpoint_path: Path | None
    compositions: tuple[CompositionTask, ...]
    residual_adapter_dirs: dict[str, Path]
    direct_adapter_dirs: dict[str, Path]
    scaffold_delta_artifact_dir: Path | None
    residual_delta_artifact_dirs: dict[str, Path]
    direct_delta_artifact_dirs: dict[str, Path]

    @classmethod
    def from_app_config(
        cls,
        app: AppConfig,
        *,
        resume_latest: bool = False,
        resume_checkpoint_path: Path | None = None,
    ) -> "ComposeConfig":
        payload = app.payload
        validation_dataset_path = Path(app.paths["eval_dataset"])
        raw_compositions = payload.get("compositions")
        if raw_compositions is None:
            raw_compositions = [
                {
                    "condition": "".join(str(item) for item in payload.get("active_overlays", [])),
                    "active_overlays": payload.get("active_overlays", []),
                }
            ]
        if not isinstance(raw_compositions, list):
            raise TypeError("payload.compositions must be a list")
        compositions = tuple(CompositionTask.from_mapping(item) for item in raw_compositions)
        configured_resume_checkpoint = Path(app.paths["resume_checkpoint"]) if "resume_checkpoint" in app.paths else None
        return cls(
            app=app,
            base_model_name_or_path=_payload_string(payload, "base_model_name_or_path"),
            scaffold_adapter_dir=Path(app.paths["scaffold_adapter"]),
            validation_dataset_path=validation_dataset_path,
            local_files_only=_payload_bool(payload, "local_files_only", True),
            alpha_grid=_payload_float_list(payload, "alpha_grid") or (0.5, 0.75, 1.0, 1.25),
            merge_grid_step=_payload_float(payload, "merge_grid_step", 0.25),
            semantic_guardrail_points=_payload_float(payload, "semantic_guardrail_points", 3.0),
            max_length=_payload_int(payload, "max_length", 256),
            default_max_new_tokens=_payload_int(payload, "max_new_tokens", 32),
            max_eval_examples=_payload_int(payload, "max_eval_examples", 1),
            eval_batch_size=_payload_int(payload, "eval_batch_size", 1),
            device=_payload_string(payload, "device"),
            torch_dtype=_payload_string(payload, "torch_dtype"),
            reuse_feature_cache=_payload_bool(payload, "reuse_feature_cache", True),
            feature_cache_dir=Path(app.paths["feature_cache_dir"])
            if "feature_cache_dir" in app.paths
            else default_feature_cache_dir(validation_dataset_path),
            sort_eval_by_length=_payload_bool(payload, "sort_eval_by_length", True),
            merge_search_policy=_payload_string(payload, "merge_search_policy") if "merge_search_policy" in payload else "match_soar_budget",
            candidate_artifact_policy=_payload_choice_string(
                payload,
                "candidate_artifact_policy",
                default="full",
                allowed=_CANDIDATE_ARTIFACT_POLICIES,
            ),
            synthesized_lora_dropout=_payload_float(payload, "synthesized_lora_dropout", 0.0),
            enforce_budget_parity=_payload_bool(payload, "enforce_budget_parity", True),
            budget_parity_tolerance=_payload_int(payload, "budget_parity_tolerance", 0),
            checkpoint_interval_units=_payload_int(payload, "checkpoint_interval_units", 1),
            heartbeat_interval_seconds=_payload_float(payload, "heartbeat_interval_seconds", 30.0),
            control_poll_seconds=_payload_float(payload, "control_poll_seconds", 5.0),
            max_checkpoints_to_keep=_payload_int(payload, "max_checkpoints_to_keep", 3),
            resume_latest=resume_latest or _payload_bool(payload, "resume_latest", False),
            resume_checkpoint_path=resume_checkpoint_path or configured_resume_checkpoint,
            compositions=compositions,
            residual_adapter_dirs=_parse_overlay_map(app.paths, "residual_adapter"),
            direct_adapter_dirs=_parse_overlay_map(app.paths, "direct_adapter"),
            scaffold_delta_artifact_dir=Path(app.paths["scaffold_delta_artifact"])
            if "scaffold_delta_artifact" in app.paths
            else None,
            residual_delta_artifact_dirs=_parse_overlay_map(app.paths, "residual_delta_artifact"),
            direct_delta_artifact_dirs=_parse_overlay_map(app.paths, "direct_delta_artifact"),
        )


@dataclass(frozen=True, slots=True)
class CompositionCandidate:
    system: str
    task_name: str
    key: str
    coefficients: dict[str, float]
    adapter_dir: Path | None
    metrics: dict[str, float]
    tie_distance: float

    @property
    def answer_f1(self) -> float:
        return float(self.metrics.get("answer_f1", 0.0))

    @property
    def semantic_f1_points(self) -> float:
        return self.answer_f1 * 100.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _coeff_distance(coefficients: Mapping[str, float], target: Mapping[str, float]) -> float:
    return sum((float(coefficients[key]) - float(target[key])) ** 2 for key in coefficients) ** 0.5


def _active_primary_metrics(active_overlays: Sequence[str]) -> list[str]:
    return [_PRIMARY_METRIC_BY_OVERLAY[overlay] for overlay in active_overlays]


def _mean_active_overlay_score(metrics: Mapping[str, Any], active_overlays: Sequence[str]) -> float:
    values = [float(metrics.get(_PRIMARY_METRIC_BY_OVERLAY[overlay], 0.0)) for overlay in active_overlays]
    return _mean(values)


def _soar_target(active_overlays: Sequence[str]) -> dict[str, float]:
    return {overlay: 1.0 for overlay in active_overlays}


def _merge_target(active_overlays: Sequence[str]) -> dict[str, float]:
    if not active_overlays:
        return {}
    uniform = 1.0 / len(active_overlays)
    return {overlay: uniform for overlay in active_overlays}


def _persist_full_candidate_artifacts(policy: str) -> bool:
    if policy not in _CANDIDATE_ARTIFACT_POLICIES:
        raise ValueError(f"Unsupported candidate_artifact_policy: {policy!r}")
    return policy == "full"


def choose_best_composition_candidate(
    *,
    active_overlays: Sequence[str],
    baseline_answer_f1: float,
    guardrail_points: float,
    candidates: Sequence[CompositionCandidate],
) -> tuple[CompositionCandidate, dict[str, Any]]:
    floor = baseline_answer_f1 * 100.0 - guardrail_points
    passing = [candidate for candidate in candidates if candidate.semantic_f1_points >= floor]

    if passing:
        chosen = sorted(
            passing,
            key=lambda item: (
                -_mean_active_overlay_score(item.metrics, active_overlays),
                item.tie_distance,
            ),
        )[0]
        selection = {
            "guardrail_floor_points": floor,
            "guardrail_satisfied": True,
            "selection_rule": (
                "prefer guardrail-passing candidates, maximize mean active overlay primary, "
                "tie-break toward target coefficients"
            ),
        }
        return chosen, selection

    chosen = sorted(
        candidates,
        key=lambda item: (
            -item.semantic_f1_points,
            item.tie_distance,
            -_mean_active_overlay_score(item.metrics, active_overlays),
        ),
    )[0]
    selection = {
        "guardrail_floor_points": floor,
        "guardrail_satisfied": False,
        "selection_rule": (
            "no candidate passed the guardrail; choose highest semantic F1, "
            "tie-break toward target coefficients, then mean active overlay primary"
        ),
    }
    return chosen, selection


def _slug_value(value: float) -> str:
    return f"{value:.2f}".replace("-", "neg_").replace(".", "_")


def _coeff_slug(prefix: str, coefficients: Mapping[str, float]) -> str:
    ordered = [f"{overlay.lower()}_{_slug_value(coefficients[overlay])}" for overlay in sorted(coefficients.keys())]
    return prefix + "__" + "__".join(ordered)


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _soar_candidate_coefficients(
    active_overlays: Sequence[str],
    alpha_grid: Sequence[float],
) -> list[dict[str, float]]:
    return [
        dict(zip(active_overlays, coeffs, strict=True))
        for coeffs in product(alpha_grid, repeat=len(active_overlays))
    ]


def _merge_candidate_coefficients(
    active_overlays: Sequence[str],
    merge_grid_step: float,
    *,
    policy: str,
    target_budget: int,
) -> list[dict[str, float]]:
    if policy == "simplex_grid_step":
        coefficient_rows = list(simplex_grid(len(active_overlays), step=merge_grid_step))
    elif policy == "match_soar_budget":
        coefficient_rows = _fixed_budget_simplex_points(len(active_overlays), target_budget)
    else:
        raise ValueError(f"Unsupported merge_search_policy: {policy}")
    return [dict(zip(active_overlays, coeffs, strict=True)) for coeffs in coefficient_rows]


def _simplex_lattice_count(dimensions: int, denominator: int) -> int:
    return comb(denominator + dimensions - 1, dimensions - 1)


def _simplex_lattice_points(dimensions: int, denominator: int) -> list[tuple[float, ...]]:
    if dimensions <= 0:
        raise ValueError("dimensions must be positive")
    if denominator <= 0:
        raise ValueError("denominator must be positive")

    step = 1.0 / denominator
    points: list[tuple[float, ...]] = []

    def rec(parts_left: int, units_left: int, prefix: list[float]) -> None:
        if parts_left == 1:
            points.append(tuple(prefix + [units_left * step]))
            return
        for unit in range(units_left + 1):
            rec(parts_left - 1, units_left - unit, prefix + [unit * step])

    rec(dimensions, denominator, [])
    return points


def _fixed_budget_simplex_points(dimensions: int, budget: int) -> list[tuple[float, ...]]:
    if dimensions <= 0:
        raise ValueError("dimensions must be positive")
    if budget <= 0:
        raise ValueError("budget must be positive")
    if dimensions == 1:
        return [(1.0,)]

    denominator = 1
    while _simplex_lattice_count(dimensions, denominator) < budget:
        denominator += 1

    lattice = _simplex_lattice_points(dimensions, denominator)
    if len(lattice) == budget:
        return lattice

    if budget == 1:
        return [lattice[len(lattice) // 2]]

    indices = [((len(lattice) - 1) * i) // (budget - 1) for i in range(budget)]
    return [lattice[index] for index in indices]


def _assert_tuning_budget_parity(
    *,
    task: CompositionTask,
    soar_candidate_count: int,
    merge_candidate_count: int,
    tolerance: int,
) -> None:
    if abs(soar_candidate_count - merge_candidate_count) <= tolerance:
        return
    raise ValueError(
        "Compose config violates the locked fairness rule against unequal tuning budgets: "
        f"{task.condition} has {soar_candidate_count} SOAR candidates versus {merge_candidate_count} merge "
        f"candidates (tolerance={tolerance}). Update the search policy or explicitly revise the docs before M5."
    )


def _global_rank_for_map(factor_map: Mapping[str, DenseDeltaFactors]) -> int:
    ranks = {factors.rank_bound for factors in factor_map.values()}
    if not ranks:
        raise ValueError("factor_map must not be empty")
    return max(ranks)


def _save_factor_map_as_adapter(
    *,
    output_dir: Path,
    template_bundle: AdapterBundle,
    factor_map: Mapping[str, DenseDeltaFactors],
    lora_dropout: float,
) -> tuple[Path, int]:
    global_rank = _global_rank_for_map(factor_map)
    lora_factors = {
        module_name: factorize_dense_factors(factors, global_rank, pad_to_rank=True)
        for module_name, factors in factor_map.items()
    }
    save_adapter_from_lora_factors(
        output_dir,
        template_bundle=template_bundle,
        lora_factors=lora_factors,
        global_rank=global_rank,
        lora_dropout=lora_dropout,
    )
    return output_dir, global_rank


def _write_candidate_metrics(
    *,
    candidate_dir: Path,
    metrics: Mapping[str, Any],
    generation_summary: Mapping[str, Any],
) -> None:
    write_json(
        candidate_dir / "metrics.json",
        {
            "metrics": dict(metrics),
            "generation_summary": dict(generation_summary),
        },
    )


def _write_candidate_predictions(
    *,
    candidate_dir: Path,
    predictions: Sequence[Mapping[str, Any]],
) -> None:
    write_jsonl(candidate_dir / "predictions.jsonl", [dict(row) for row in predictions])


def _candidate_factor_map(
    *,
    system: str,
    active_overlays: Sequence[str],
    coefficients: Mapping[str, float],
    scaffold_factors: Mapping[str, DenseDeltaFactors],
    residual_factor_maps: Mapping[str, Mapping[str, DenseDeltaFactors]],
    direct_factor_maps: Mapping[str, Mapping[str, DenseDeltaFactors]],
) -> dict[str, DenseDeltaFactors]:
    if system == "soar":
        return compose_factor_maps(
            scaffold_factors,
            residuals={overlay: residual_factor_maps[overlay] for overlay in active_overlays},
            alphas=coefficients,
        )
    if system == "merge":
        return _combine_direct_factors(
            active_overlays=active_overlays,
            direct_factor_maps=direct_factor_maps,
            weights=coefficients,
        )
    raise ValueError(f"Unsupported composition system: {system}")


def _ensure_run_layout(run_paths: RunPaths, source_config: Path) -> None:
    run_paths.run_dir.mkdir(parents=True, exist_ok=True)
    run_paths.adapter_dir.mkdir(parents=True, exist_ok=True)
    run_paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    run_paths.control_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_config, run_paths.resolved_config)


def _materialize_chosen_candidate(
    *,
    system: str,
    task: CompositionTask,
    chosen_record: dict[str, Any],
    output_dir: Path,
    scaffold_bundle: AdapterBundle,
    scaffold_factors: Mapping[str, DenseDeltaFactors],
    residual_factor_maps: Mapping[str, Mapping[str, DenseDeltaFactors]],
    direct_factor_maps: Mapping[str, Mapping[str, DenseDeltaFactors]],
    evaluator: LoadedAdapterEvaluator,
    feature_bundle: EvalFeatureBundle,
    config: ComposeConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    coefficients = {str(key): float(value) for key, value in dict(chosen_record["coefficients"]).items()}
    factor_map = _candidate_factor_map(
        system=system,
        active_overlays=task.active_overlays,
        coefficients=coefficients,
        scaffold_factors=scaffold_factors,
        residual_factor_maps=residual_factor_maps,
        direct_factor_maps=direct_factor_maps,
    )
    adapter_dir, global_rank = _save_factor_map_as_adapter(
        output_dir=output_dir / "adapter",
        template_bundle=scaffold_bundle,
        factor_map=factor_map,
        lora_dropout=config.synthesized_lora_dropout,
    )
    evaluation = evaluator.evaluate_adapter(
        adapter_name=f"{task.name.lower()}_{system}_chosen",
        adapter_dir=adapter_dir,
        eval_features=feature_bundle.features,
        condition=task.condition,
        max_length=config.max_length,
        max_new_tokens=_task_max_new_tokens(config, task),
        batch_size=config.eval_batch_size,
        sort_by_length=config.sort_eval_by_length,
        adapter_mode="temporary",
    )
    metrics = {"condition": task.condition, **evaluation.metrics}
    if metrics != dict(chosen_record["metrics"]):
        raise ValueError(
            "Chosen candidate replay changed metrics under deterministic compose selection: "
            f"{task.condition}/{system}/{chosen_record['chosen_key']}"
        )
    _write_candidate_metrics(
        candidate_dir=output_dir,
        metrics=metrics,
        generation_summary=evaluation.generation_summary,
    )
    _write_candidate_predictions(candidate_dir=output_dir, predictions=evaluation.predictions)

    materialized_record = dict(chosen_record)
    materialized_record["adapter_dir"] = str(adapter_dir)
    materialized_record["metrics_path"] = str(output_dir / "metrics.json")
    materialized_record["predictions_path"] = str(output_dir / "predictions.jsonl")
    materialized_record["chosen_predictions_path"] = str(output_dir / "predictions.jsonl")
    materialized_record["prediction_row_count"] = len(evaluation.predictions)
    materialized_record["generation_summary"] = dict(evaluation.generation_summary)
    materialized_record["global_rank"] = global_rank
    materialized_record["materialized_after_selection"] = True
    return materialized_record, evaluation.predictions


def _resolve_resume_checkpoint_path(config: ComposeConfig, run_paths: RunPaths) -> Path | None:
    if config.resume_checkpoint_path is not None:
        return resolve_sweep_checkpoint_dir(config.resume_checkpoint_path)
    if config.resume_latest:
        if run_paths.latest_checkpoint.exists():
            return resolve_sweep_checkpoint_dir(run_paths.latest_checkpoint)
        raise FileNotFoundError(
            f"--resume-latest was requested, but no latest checkpoint exists at {run_paths.latest_checkpoint}"
        )
    return None


def _assert_run_dir_available(run_paths: RunPaths, *, resume_checkpoint_dir: Path | None) -> None:
    if resume_checkpoint_dir is not None:
        return
    if run_paths.metrics.exists() or run_paths.run_manifest.exists() or run_paths.latest_checkpoint.exists():
        raise FileExistsError(
            "Run directory already contains completed or checkpointed state. "
            f"Move {run_paths.run_dir} aside or rerun with --resume-latest / --resume-checkpoint."
        )


def _load_or_extract_dense_factors(
    *,
    bundle: AdapterBundle,
    requested_artifact_dir: Path | None,
    fallback_artifact_dir: Path,
) -> tuple[dict[str, DenseDeltaFactors], Path]:
    artifact_dir = requested_artifact_dir or fallback_artifact_dir
    if (artifact_dir / "manifest.json").exists():
        factors_map, _manifest = load_compact_delta_artifact(artifact_dir)
        return factors_map, artifact_dir

    factors_map = dense_factor_map(bundle.lora_factors)
    save_compact_delta_artifact(
        artifact_dir,
        factors_map,
        source={
            "adapter_dir": str(bundle.adapter_dir),
            "base_model_name_or_path": bundle.base_model_name_or_path,
        },
    )
    return factors_map, artifact_dir


def _combine_direct_factors(
    *,
    active_overlays: Sequence[str],
    direct_factor_maps: Mapping[str, Mapping[str, DenseDeltaFactors]],
    weights: Mapping[str, float],
) -> dict[str, DenseDeltaFactors]:
    module_names = set(next(iter(direct_factor_maps.values())).keys())
    combined: dict[str, DenseDeltaFactors] = {}
    for module_name in sorted(module_names):
        terms = [(float(weights[overlay]), direct_factor_maps[overlay][module_name]) for overlay in active_overlays]
        combined[module_name] = combine_dense_factors(terms)
    return combined


def _task_max_new_tokens(config: ComposeConfig, task: CompositionTask) -> int:
    return task.max_new_tokens if task.max_new_tokens is not None else config.default_max_new_tokens


def _initial_compose_state(config: ComposeConfig) -> dict[str, Any]:
    return {
        "compositions": {
            task.name: {
                "condition": task.condition,
                "active_overlays": list(task.active_overlays),
                "prompt_only": None,
                "soar_candidates": {},
                "merge_candidates": {},
                "chosen_soar": None,
                "chosen_merge": None,
                "status": "pending",
            }
            for task in config.compositions
        }
    }


def _compose_total_units(config: ComposeConfig) -> int:
    total = 0
    for task in config.compositions:
        soar_candidates = _soar_candidate_coefficients(task.active_overlays, config.alpha_grid)
        merge_candidates = _merge_candidate_coefficients(
            task.active_overlays,
            config.merge_grid_step,
            policy=config.merge_search_policy,
            target_budget=len(soar_candidates),
        )
        total += 1
        total += len(soar_candidates)
        total += len(merge_candidates)
    return total


def _save_checkpoint_state(
    *,
    runtime: SweepRuntimeState,
    run_paths: RunPaths,
    state: Mapping[str, Any],
    keep_last: int,
    plan: SweepPlan,
    elapsed_runtime_seconds: float,
    active_item: str | None,
    phase: str | None,
    message: str | None = None,
    extra_progress: Mapping[str, Any] | None = None,
) -> None:
    save_sweep_checkpoint(
        runtime=runtime,
        checkpoints_dir=run_paths.checkpoints_dir,
        latest_checkpoint_path=run_paths.latest_checkpoint,
        state_payload=state,
        keep_last=keep_last,
        heartbeat_path=run_paths.heartbeat,
        control_dir=run_paths.control_dir,
        plan=plan,
        elapsed_runtime_seconds=elapsed_runtime_seconds,
        status="running",
        active_item=active_item,
        phase=phase,
        message=message,
        extra_progress=extra_progress,
    )


def compose_residuals(config: ComposeConfig) -> dict[str, Any]:
    run_paths = build_run_paths(config.app)
    resume_checkpoint_dir = _resolve_resume_checkpoint_path(config, run_paths)
    _assert_run_dir_available(run_paths, resume_checkpoint_dir=resume_checkpoint_dir)
    _ensure_run_layout(run_paths, config.app.source_path)
    write_environment_summary(run_paths.environment)

    scaffold_bundle = load_adapter_bundle(config.scaffold_adapter_dir)
    if scaffold_bundle.base_model_name_or_path != config.base_model_name_or_path:
        raise ValueError("Config base model does not match the scaffold adapter metadata.")

    required_overlays = {overlay for task in config.compositions for overlay in task.active_overlays}
    missing_residuals = sorted(required_overlays - set(config.residual_adapter_dirs.keys()))
    missing_direct = sorted(required_overlays - set(config.direct_adapter_dirs.keys()))
    if missing_residuals:
        raise KeyError(f"Missing residual adapters for overlays: {', '.join(missing_residuals)}")
    if missing_direct:
        raise KeyError(f"Missing direct adapters for overlays: {', '.join(missing_direct)}")

    delta_cache_root = run_paths.run_dir / "delta_cache"
    scaffold_factors, scaffold_delta_artifact_dir = _load_or_extract_dense_factors(
        bundle=scaffold_bundle,
        requested_artifact_dir=config.scaffold_delta_artifact_dir,
        fallback_artifact_dir=delta_cache_root / "scaffold",
    )
    residual_factor_maps: dict[str, dict[str, DenseDeltaFactors]] = {}
    direct_factor_maps: dict[str, dict[str, DenseDeltaFactors]] = {}
    residual_delta_artifact_dirs: dict[str, str] = {}
    direct_delta_artifact_dirs: dict[str, str] = {}
    for overlay in required_overlays:
        residual_bundle = load_adapter_bundle(config.residual_adapter_dirs[overlay])
        direct_bundle = load_adapter_bundle(config.direct_adapter_dirs[overlay])
        if residual_bundle.base_model_name_or_path != config.base_model_name_or_path:
            raise ValueError(f"Residual adapter {overlay} does not match the configured base model.")
        if direct_bundle.base_model_name_or_path != config.base_model_name_or_path:
            raise ValueError(f"Direct adapter {overlay} does not match the configured base model.")
        residual_factors, residual_artifact_dir = _load_or_extract_dense_factors(
            bundle=residual_bundle,
            requested_artifact_dir=config.residual_delta_artifact_dirs.get(overlay),
            fallback_artifact_dir=delta_cache_root / "residuals" / overlay.lower(),
        )
        direct_factors, direct_artifact_dir = _load_or_extract_dense_factors(
            bundle=direct_bundle,
            requested_artifact_dir=config.direct_delta_artifact_dirs.get(overlay),
            fallback_artifact_dir=delta_cache_root / "direct" / overlay.lower(),
        )
        residual_factor_maps[overlay] = residual_factors
        direct_factor_maps[overlay] = direct_factors
        residual_delta_artifact_dirs[overlay] = str(residual_artifact_dir)
        direct_delta_artifact_dirs[overlay] = str(direct_artifact_dir)

    candidate_budget_counts: dict[str, dict[str, int]] = {}
    for task in config.compositions:
        soar_candidate_count = len(_soar_candidate_coefficients(task.active_overlays, config.alpha_grid))
        merge_candidate_count = len(
            _merge_candidate_coefficients(
                task.active_overlays,
                config.merge_grid_step,
                policy=config.merge_search_policy,
                target_budget=soar_candidate_count,
            )
        )
        if config.enforce_budget_parity:
            _assert_tuning_budget_parity(
                task=task,
                soar_candidate_count=soar_candidate_count,
                merge_candidate_count=merge_candidate_count,
                tolerance=config.budget_parity_tolerance,
            )
        candidate_budget_counts[task.name] = {
            "soar": soar_candidate_count,
            "merge": merge_candidate_count,
        }

    plan = SweepPlan.from_counts(total_units=_compose_total_units(config), unit_label="evaluation_units")
    runtime = SweepRuntimeState()
    state = _initial_compose_state(config)
    if resume_checkpoint_dir is not None:
        runtime, state = load_sweep_checkpoint(resume_checkpoint_dir)
        if runtime.units_completed > plan.total_units:
            raise ValueError(
                "Resume checkpoint is ahead of the current composition plan: "
                f"{runtime.units_completed} > {plan.total_units}."
            )

    session_started = time.monotonic()
    paused_seconds = 0.0
    last_heartbeat_timestamp = session_started
    evaluator = LoadedAdapterEvaluator(
        base_model_name_or_path=config.base_model_name_or_path,
        local_files_only=config.local_files_only,
        device=config.device,
        torch_dtype=config.torch_dtype,
        reuse_feature_cache=config.reuse_feature_cache,
    )
    feature_bundles: dict[str, EvalFeatureBundle] = {}

    def elapsed_runtime_seconds() -> float:
        return runtime.accumulated_runtime_seconds + max(0.0, time.monotonic() - session_started - paused_seconds)

    write_json(run_paths.run_dir / "state.json", state)
    heartbeat = build_sweep_heartbeat(
        status="preparing",
        plan=plan,
        runtime=runtime,
        heartbeat_path=run_paths.heartbeat,
        control_dir=run_paths.control_dir,
        elapsed_runtime_seconds=elapsed_runtime_seconds(),
        message="Composition sweep state is ready; loading adapters and validation examples.",
    )
    write_json(run_paths.heartbeat, heartbeat)

    try:
        for task in config.compositions:
            task_state = state["compositions"][task.name]
            composition_dir = run_paths.run_dir / "compositions" / task.name
            candidates_dir = composition_dir / "candidates"
            prompt_only_dir = composition_dir / "prompt_only"
            chosen_soar_dir = composition_dir / "soar"
            chosen_merge_dir = composition_dir / "merge"
            max_new_tokens = _task_max_new_tokens(config, task)

            feature_bundle = feature_bundles.get(task.condition)
            if feature_bundle is None:
                feature_bundle = evaluator.get_eval_features(
                    dataset_path=config.validation_dataset_path,
                    condition=task.condition,
                    max_length=config.max_length,
                    max_examples=config.max_eval_examples,
                    cache_dir=config.feature_cache_dir,
                )
                feature_bundles[task.condition] = feature_bundle

            alpha_candidates = _soar_candidate_coefficients(task.active_overlays, config.alpha_grid)
            merge_candidates = _merge_candidate_coefficients(
                task.active_overlays,
                config.merge_grid_step,
                policy=config.merge_search_policy,
                target_budget=len(alpha_candidates),
            )

            if task_state["prompt_only"] is None:
                prompt_result = evaluator.evaluate_adapter(
                    adapter_name=f"prompt_only_{task.name.lower()}",
                    adapter_dir=config.scaffold_adapter_dir,
                    eval_features=feature_bundle.features,
                    condition=task.condition,
                    max_length=config.max_length,
                    max_new_tokens=max_new_tokens,
                    batch_size=config.eval_batch_size,
                    sort_by_length=config.sort_eval_by_length,
                )
                prompt_metrics = {"condition": task.condition, **prompt_result.metrics}
                _write_candidate_metrics(
                    candidate_dir=prompt_only_dir,
                    metrics=prompt_metrics,
                    generation_summary=prompt_result.generation_summary,
                )
                _write_candidate_predictions(candidate_dir=prompt_only_dir, predictions=prompt_result.predictions)
                task_state["prompt_only"] = {
                    "metrics": prompt_metrics,
                    "generation_summary": dict(prompt_result.generation_summary),
                    "metrics_path": str(prompt_only_dir / "metrics.json"),
                    "predictions_path": str(prompt_only_dir / "predictions.jsonl"),
                }
                runtime.units_completed += 1
                write_json(run_paths.run_dir / "state.json", state)
                _save_checkpoint_state(
                    runtime=runtime,
                    run_paths=run_paths,
                    state=state,
                    keep_last=config.max_checkpoints_to_keep,
                    plan=plan,
                    elapsed_runtime_seconds=elapsed_runtime_seconds(),
                    active_item=task.name,
                    phase="prompt_only_eval",
                )
                last_heartbeat_timestamp = time.monotonic()

            prompt_only_metrics = dict(task_state["prompt_only"]["metrics"])
            baseline_answer_f1 = float(prompt_only_metrics.get("answer_f1", 0.0))
            active_overlays = task.active_overlays
            for coefficients in alpha_candidates:
                candidate_key = _coeff_slug("alpha", coefficients)
                if candidate_key in task_state["soar_candidates"]:
                    continue
                if run_paths.pause_request.exists():
                    paused_seconds += maybe_pause_sweep(
                        pause_request_path=run_paths.pause_request,
                        heartbeat_path=run_paths.heartbeat,
                        control_dir=run_paths.control_dir,
                        plan=plan,
                        runtime=runtime,
                        elapsed_runtime_seconds=elapsed_runtime_seconds(),
                        active_item=task.name,
                        phase="soar_tuning",
                        control_poll_seconds=config.control_poll_seconds,
                        extra_progress={"system": "soar"},
                    )
                    last_heartbeat_timestamp = time.monotonic()

                factor_map = _candidate_factor_map(
                    system="soar",
                    active_overlays=active_overlays,
                    coefficients=coefficients,
                    scaffold_factors=scaffold_factors,
                    residual_factor_maps=residual_factor_maps,
                    direct_factor_maps=direct_factor_maps,
                )
                candidate_dir = candidates_dir / "soar" / candidate_key
                persist_full_candidate_artifacts = _persist_full_candidate_artifacts(config.candidate_artifact_policy)
                adapter_dir, global_rank = _save_factor_map_as_adapter(
                    output_dir=candidate_dir / "adapter",
                    template_bundle=scaffold_bundle,
                    factor_map=factor_map,
                    lora_dropout=config.synthesized_lora_dropout,
                )
                if persist_full_candidate_artifacts:
                    save_compact_delta_artifact(
                        candidate_dir / "delta_artifact",
                        factor_map,
                        source={
                            "system": "soar",
                            "condition": task.condition,
                            "coefficients": coefficients,
                            "scaffold_delta_artifact_dir": str(scaffold_delta_artifact_dir),
                            "residual_delta_artifact_dirs": {
                                overlay: residual_delta_artifact_dirs[overlay] for overlay in active_overlays
                            },
                        },
                    )
                evaluation = evaluator.evaluate_adapter(
                    adapter_name=f"{task.name.lower()}_{candidate_key}",
                    adapter_dir=adapter_dir,
                    eval_features=feature_bundle.features,
                    condition=task.condition,
                    max_length=config.max_length,
                    max_new_tokens=max_new_tokens,
                    batch_size=config.eval_batch_size,
                    sort_by_length=config.sort_eval_by_length,
                    adapter_mode="temporary",
                )
                metrics = {"condition": task.condition, **evaluation.metrics}
                _write_candidate_metrics(
                    candidate_dir=candidate_dir,
                    metrics=metrics,
                    generation_summary=evaluation.generation_summary,
                )
                if persist_full_candidate_artifacts:
                    _write_candidate_predictions(candidate_dir=candidate_dir, predictions=evaluation.predictions)
                target = _soar_target(active_overlays)
                task_state["soar_candidates"][candidate_key] = {
                    "coefficients": coefficients,
                    "adapter_dir": (str(adapter_dir) if persist_full_candidate_artifacts else None),
                    "delta_artifact_dir": (str(candidate_dir / "delta_artifact") if persist_full_candidate_artifacts else None),
                    "metrics": metrics,
                    "generation_summary": dict(evaluation.generation_summary),
                    "metrics_path": str(candidate_dir / "metrics.json"),
                    "predictions_path": (str(candidate_dir / "predictions.jsonl") if persist_full_candidate_artifacts else None),
                    "mean_active_overlay_primary": _mean_active_overlay_score(metrics, active_overlays),
                    "tie_distance_to_target": _coeff_distance(coefficients, target),
                    "global_rank": global_rank,
                    "candidate_artifact_policy": config.candidate_artifact_policy,
                }
                if not persist_full_candidate_artifacts:
                    shutil.rmtree(adapter_dir, ignore_errors=True)
                runtime.units_completed += 1
                if runtime.units_completed % max(config.checkpoint_interval_units, 1) == 0:
                    write_json(run_paths.run_dir / "state.json", state)
                    _save_checkpoint_state(
                        runtime=runtime,
                        run_paths=run_paths,
                        state=state,
                        keep_last=config.max_checkpoints_to_keep,
                        plan=plan,
                        elapsed_runtime_seconds=elapsed_runtime_seconds(),
                        active_item=task.name,
                        phase="soar_tuning",
                        extra_progress={"system": "soar", "candidate": candidate_key},
                    )
                    last_heartbeat_timestamp = time.monotonic()

                if time.monotonic() - last_heartbeat_timestamp >= max(config.heartbeat_interval_seconds, 1.0):
                    heartbeat = build_sweep_heartbeat(
                        status="running",
                        plan=plan,
                        runtime=runtime,
                        heartbeat_path=run_paths.heartbeat,
                        control_dir=run_paths.control_dir,
                        elapsed_runtime_seconds=elapsed_runtime_seconds(),
                        active_item=task.name,
                        phase="soar_tuning",
                        extra_progress={"system": "soar", "candidate": candidate_key},
                    )
                    write_json(run_paths.heartbeat, heartbeat)
                    last_heartbeat_timestamp = time.monotonic()

            for coefficients in merge_candidates:
                candidate_key = _coeff_slug("merge", coefficients)
                if candidate_key in task_state["merge_candidates"]:
                    continue
                if run_paths.pause_request.exists():
                    paused_seconds += maybe_pause_sweep(
                        pause_request_path=run_paths.pause_request,
                        heartbeat_path=run_paths.heartbeat,
                        control_dir=run_paths.control_dir,
                        plan=plan,
                        runtime=runtime,
                        elapsed_runtime_seconds=elapsed_runtime_seconds(),
                        active_item=task.name,
                        phase="merge_tuning",
                        control_poll_seconds=config.control_poll_seconds,
                        extra_progress={"system": "merge"},
                    )
                    last_heartbeat_timestamp = time.monotonic()

                factor_map = _candidate_factor_map(
                    system="merge",
                    active_overlays=active_overlays,
                    coefficients=coefficients,
                    scaffold_factors=scaffold_factors,
                    residual_factor_maps=residual_factor_maps,
                    direct_factor_maps=direct_factor_maps,
                )
                candidate_dir = candidates_dir / "merge" / candidate_key
                persist_full_candidate_artifacts = _persist_full_candidate_artifacts(config.candidate_artifact_policy)
                adapter_dir, global_rank = _save_factor_map_as_adapter(
                    output_dir=candidate_dir / "adapter",
                    template_bundle=scaffold_bundle,
                    factor_map=factor_map,
                    lora_dropout=config.synthesized_lora_dropout,
                )
                if persist_full_candidate_artifacts:
                    save_compact_delta_artifact(
                        candidate_dir / "delta_artifact",
                        factor_map,
                        source={
                            "system": "merge",
                            "condition": task.condition,
                            "weights": coefficients,
                            "direct_delta_artifact_dirs": {
                                overlay: direct_delta_artifact_dirs[overlay] for overlay in active_overlays
                            },
                        },
                    )
                evaluation = evaluator.evaluate_adapter(
                    adapter_name=f"{task.name.lower()}_{candidate_key}",
                    adapter_dir=adapter_dir,
                    eval_features=feature_bundle.features,
                    condition=task.condition,
                    max_length=config.max_length,
                    max_new_tokens=max_new_tokens,
                    batch_size=config.eval_batch_size,
                    sort_by_length=config.sort_eval_by_length,
                    adapter_mode="temporary",
                )
                metrics = {"condition": task.condition, **evaluation.metrics}
                _write_candidate_metrics(
                    candidate_dir=candidate_dir,
                    metrics=metrics,
                    generation_summary=evaluation.generation_summary,
                )
                if persist_full_candidate_artifacts:
                    _write_candidate_predictions(candidate_dir=candidate_dir, predictions=evaluation.predictions)
                target = _merge_target(active_overlays)
                task_state["merge_candidates"][candidate_key] = {
                    "coefficients": coefficients,
                    "adapter_dir": (str(adapter_dir) if persist_full_candidate_artifacts else None),
                    "delta_artifact_dir": (str(candidate_dir / "delta_artifact") if persist_full_candidate_artifacts else None),
                    "metrics": metrics,
                    "generation_summary": dict(evaluation.generation_summary),
                    "metrics_path": str(candidate_dir / "metrics.json"),
                    "predictions_path": (str(candidate_dir / "predictions.jsonl") if persist_full_candidate_artifacts else None),
                    "mean_active_overlay_primary": _mean_active_overlay_score(metrics, active_overlays),
                    "tie_distance_to_target": _coeff_distance(coefficients, target),
                    "global_rank": global_rank,
                    "candidate_artifact_policy": config.candidate_artifact_policy,
                }
                if not persist_full_candidate_artifacts:
                    shutil.rmtree(adapter_dir, ignore_errors=True)
                runtime.units_completed += 1
                if runtime.units_completed % max(config.checkpoint_interval_units, 1) == 0:
                    write_json(run_paths.run_dir / "state.json", state)
                    _save_checkpoint_state(
                        runtime=runtime,
                        run_paths=run_paths,
                        state=state,
                        keep_last=config.max_checkpoints_to_keep,
                        plan=plan,
                        elapsed_runtime_seconds=elapsed_runtime_seconds(),
                        active_item=task.name,
                        phase="merge_tuning",
                        extra_progress={"system": "merge", "candidate": candidate_key},
                    )
                    last_heartbeat_timestamp = time.monotonic()

                if time.monotonic() - last_heartbeat_timestamp >= max(config.heartbeat_interval_seconds, 1.0):
                    heartbeat = build_sweep_heartbeat(
                        status="running",
                        plan=plan,
                        runtime=runtime,
                        heartbeat_path=run_paths.heartbeat,
                        control_dir=run_paths.control_dir,
                        elapsed_runtime_seconds=elapsed_runtime_seconds(),
                        active_item=task.name,
                        phase="merge_tuning",
                        extra_progress={"system": "merge", "candidate": candidate_key},
                    )
                    write_json(run_paths.heartbeat, heartbeat)
                    last_heartbeat_timestamp = time.monotonic()

            if task_state["chosen_soar"] is None:
                soar_rows = [
                    CompositionCandidate(
                        system="soar",
                        task_name=task.name,
                        key=candidate_key,
                        coefficients=dict(candidate["coefficients"]),
                        adapter_dir=(Path(candidate["adapter_dir"]) if candidate.get("adapter_dir") else None),
                        metrics=dict(candidate["metrics"]),
                        tie_distance=float(candidate["tie_distance_to_target"]),
                    )
                    for candidate_key, candidate in task_state["soar_candidates"].items()
                ]
                chosen_soar, selection = choose_best_composition_candidate(
                    active_overlays=active_overlays,
                    baseline_answer_f1=baseline_answer_f1,
                    guardrail_points=config.semantic_guardrail_points,
                    candidates=soar_rows,
                )
                chosen_record = dict(task_state["soar_candidates"][chosen_soar.key])
                chosen_record["selection"] = selection
                chosen_record["chosen_key"] = chosen_soar.key
                chosen_record["primary_metrics"] = _active_primary_metrics(active_overlays)
                chosen_record["guardrail_floor_points"] = selection["guardrail_floor_points"]
                chosen_record["guardrail_satisfied"] = selection["guardrail_satisfied"]
                chosen_record["active_overlays"] = list(active_overlays)
                chosen_record["mean_active_overlay_primary"] = _mean_active_overlay_score(chosen_soar.metrics, active_overlays)
                chosen_soar_dir.mkdir(parents=True, exist_ok=True)
                if chosen_record.get("predictions_path") and chosen_record.get("adapter_dir"):
                    chosen_predictions = _read_jsonl_rows(Path(chosen_record["predictions_path"]))
                    shutil.copytree(Path(chosen_record["adapter_dir"]), chosen_soar_dir / "adapter", dirs_exist_ok=True)
                    _write_candidate_metrics(
                        candidate_dir=chosen_soar_dir,
                        metrics=dict(chosen_soar.metrics),
                        generation_summary=chosen_record.get("generation_summary", {}),
                    )
                    _write_candidate_predictions(candidate_dir=chosen_soar_dir, predictions=chosen_predictions)
                    chosen_record["chosen_predictions_path"] = str(chosen_soar_dir / "predictions.jsonl")
                    chosen_record["prediction_row_count"] = len(chosen_predictions)
                    chosen_record["materialized_after_selection"] = False
                else:
                    chosen_record, chosen_predictions = _materialize_chosen_candidate(
                        system="soar",
                        task=task,
                        chosen_record=chosen_record,
                        output_dir=chosen_soar_dir,
                        scaffold_bundle=scaffold_bundle,
                        scaffold_factors=scaffold_factors,
                        residual_factor_maps=residual_factor_maps,
                        direct_factor_maps=direct_factor_maps,
                        evaluator=evaluator,
                        feature_bundle=feature_bundle,
                        config=config,
                    )
                task_state["chosen_soar"] = chosen_record
                write_json(chosen_soar_dir / "selection.json", chosen_record)

            if task_state["chosen_merge"] is None:
                merge_rows = [
                    CompositionCandidate(
                        system="merge",
                        task_name=task.name,
                        key=candidate_key,
                        coefficients=dict(candidate["coefficients"]),
                        adapter_dir=(Path(candidate["adapter_dir"]) if candidate.get("adapter_dir") else None),
                        metrics=dict(candidate["metrics"]),
                        tie_distance=float(candidate["tie_distance_to_target"]),
                    )
                    for candidate_key, candidate in task_state["merge_candidates"].items()
                ]
                chosen_merge, selection = choose_best_composition_candidate(
                    active_overlays=active_overlays,
                    baseline_answer_f1=baseline_answer_f1,
                    guardrail_points=config.semantic_guardrail_points,
                    candidates=merge_rows,
                )
                chosen_record = dict(task_state["merge_candidates"][chosen_merge.key])
                chosen_record["selection"] = selection
                chosen_record["chosen_key"] = chosen_merge.key
                chosen_record["primary_metrics"] = _active_primary_metrics(active_overlays)
                chosen_record["guardrail_floor_points"] = selection["guardrail_floor_points"]
                chosen_record["guardrail_satisfied"] = selection["guardrail_satisfied"]
                chosen_record["active_overlays"] = list(active_overlays)
                chosen_record["mean_active_overlay_primary"] = _mean_active_overlay_score(chosen_merge.metrics, active_overlays)
                chosen_merge_dir.mkdir(parents=True, exist_ok=True)
                if chosen_record.get("predictions_path") and chosen_record.get("adapter_dir"):
                    chosen_predictions = _read_jsonl_rows(Path(chosen_record["predictions_path"]))
                    shutil.copytree(Path(chosen_record["adapter_dir"]), chosen_merge_dir / "adapter", dirs_exist_ok=True)
                    _write_candidate_metrics(
                        candidate_dir=chosen_merge_dir,
                        metrics=dict(chosen_merge.metrics),
                        generation_summary=chosen_record.get("generation_summary", {}),
                    )
                    _write_candidate_predictions(candidate_dir=chosen_merge_dir, predictions=chosen_predictions)
                    chosen_record["chosen_predictions_path"] = str(chosen_merge_dir / "predictions.jsonl")
                    chosen_record["prediction_row_count"] = len(chosen_predictions)
                    chosen_record["materialized_after_selection"] = False
                else:
                    chosen_record, chosen_predictions = _materialize_chosen_candidate(
                        system="merge",
                        task=task,
                        chosen_record=chosen_record,
                        output_dir=chosen_merge_dir,
                        scaffold_bundle=scaffold_bundle,
                        scaffold_factors=scaffold_factors,
                        residual_factor_maps=residual_factor_maps,
                        direct_factor_maps=direct_factor_maps,
                        evaluator=evaluator,
                        feature_bundle=feature_bundle,
                        config=config,
                    )
                task_state["chosen_merge"] = chosen_record
                write_json(chosen_merge_dir / "selection.json", chosen_record)

            task_state["status"] = "completed"
            write_json(
                composition_dir / "tuning_manifest.json",
                {
                    "condition": task.condition,
                    "active_overlays": list(active_overlays),
                    "prompt_only": task_state["prompt_only"],
                    "soar_candidates": task_state["soar_candidates"],
                    "merge_candidates": task_state["merge_candidates"],
                    "chosen_soar": dict(task_state["chosen_soar"]),
                    "chosen_merge": dict(task_state["chosen_merge"]),
                    "semantic_guardrail_points": config.semantic_guardrail_points,
                    "alpha_grid": list(config.alpha_grid),
                    "merge_grid_step": config.merge_grid_step,
                    "merge_search_policy": config.merge_search_policy,
                    "reuse_feature_cache": config.reuse_feature_cache,
                    "feature_cache_dir": str(config.feature_cache_dir),
                    "sort_eval_by_length": config.sort_eval_by_length,
                    "candidate_artifact_policy": config.candidate_artifact_policy,
                    "budget_parity": {
                        "enforced": config.enforce_budget_parity,
                        "tolerance": config.budget_parity_tolerance,
                        "soar_candidate_count": candidate_budget_counts[task.name]["soar"],
                        "merge_candidate_count": candidate_budget_counts[task.name]["merge"],
                    },
                    "validation_dataset_path": str(config.validation_dataset_path),
                    "validation_example_count": len(feature_bundle.features),
                },
            )
            write_json(run_paths.run_dir / "state.json", state)
            _save_checkpoint_state(
                runtime=runtime,
                run_paths=run_paths,
                state=state,
                keep_last=config.max_checkpoints_to_keep,
                plan=plan,
                elapsed_runtime_seconds=elapsed_runtime_seconds(),
                active_item=task.name,
                phase="selection",
                extra_progress={"condition": task.condition},
            )
            last_heartbeat_timestamp = time.monotonic()

        runtime.accumulated_runtime_seconds = elapsed_runtime_seconds()
        summary_rows: dict[str, Any] = {}
        aggregate_predictions: list[dict[str, Any]] = []
        for task in config.compositions:
            task_state = state["compositions"][task.name]
            summary_rows[task.name] = {
                "condition": task.condition,
                "active_overlays": list(task.active_overlays),
                "prompt_only_validation_metrics": task_state["prompt_only"]["metrics"],
                "budget_parity": {
                    "enforced": config.enforce_budget_parity,
                    "tolerance": config.budget_parity_tolerance,
                    "soar_candidate_count": candidate_budget_counts[task.name]["soar"],
                    "merge_candidate_count": candidate_budget_counts[task.name]["merge"],
                },
                "chosen_soar": dict(task_state["chosen_soar"]),
                "chosen_merge": dict(task_state["chosen_merge"]),
            }
            for system_name in ("chosen_soar", "chosen_merge"):
                for row in _read_jsonl_rows(Path(task_state[system_name]["chosen_predictions_path"])):
                    aggregate_row = dict(row)
                    aggregate_row["system"] = "soar" if system_name == "chosen_soar" else "merge"
                    aggregate_predictions.append(aggregate_row)

        metrics = {
            "status": "composed",
            "compositions": summary_rows,
            "validation_dataset_path": str(config.validation_dataset_path),
            "validation_example_count": (
                max(len(bundle.features) for bundle in feature_bundles.values()) if feature_bundles else 0
            ),
            "elapsed_runtime_seconds": runtime.accumulated_runtime_seconds,
        }
        tuning_manifest = {
            "alpha_grid": list(config.alpha_grid),
            "merge_grid_step": config.merge_grid_step,
            "merge_search_policy": config.merge_search_policy,
            "semantic_guardrail_points": config.semantic_guardrail_points,
            "validation_dataset_path": str(config.validation_dataset_path),
            "validation_example_count": (
                max(len(bundle.features) for bundle in feature_bundles.values()) if feature_bundles else 0
            ),
            "reuse_feature_cache": config.reuse_feature_cache,
            "feature_cache_dir": str(config.feature_cache_dir),
            "sort_eval_by_length": config.sort_eval_by_length,
            "candidate_artifact_policy": config.candidate_artifact_policy,
            "budget_parity_enforced": config.enforce_budget_parity,
            "budget_parity_tolerance": config.budget_parity_tolerance,
            "scaffold_delta_artifact_dir": str(scaffold_delta_artifact_dir),
            "residual_delta_artifact_dirs": residual_delta_artifact_dirs,
            "direct_delta_artifact_dirs": direct_delta_artifact_dirs,
            "compositions": summary_rows,
        }
        run_manifest = {
            "stage": "compose",
            "experiment_name": config.app.experiment_name,
            "milestone": config.app.milestone,
            "seed": config.app.seed,
            "base_model_name_or_path": config.base_model_name_or_path,
            "scaffold_adapter_dir": str(config.scaffold_adapter_dir),
            "validation_dataset_path": str(config.validation_dataset_path),
            "device": config.device,
            "torch_dtype": config.torch_dtype,
            "local_files_only": config.local_files_only,
            "alpha_grid": list(config.alpha_grid),
            "merge_grid_step": config.merge_grid_step,
            "merge_search_policy": config.merge_search_policy,
            "semantic_guardrail_points": config.semantic_guardrail_points,
            "max_length": config.max_length,
            "max_eval_examples": config.max_eval_examples,
            "eval_batch_size": config.eval_batch_size,
            "reuse_feature_cache": config.reuse_feature_cache,
            "feature_cache_dir": str(config.feature_cache_dir),
            "sort_eval_by_length": config.sort_eval_by_length,
            "candidate_artifact_policy": config.candidate_artifact_policy,
            "enforce_budget_parity": config.enforce_budget_parity,
            "budget_parity_tolerance": config.budget_parity_tolerance,
            "checkpoint_interval_units": config.checkpoint_interval_units,
            "heartbeat_interval_seconds": config.heartbeat_interval_seconds,
            "control_poll_seconds": config.control_poll_seconds,
            "max_checkpoints_to_keep": config.max_checkpoints_to_keep,
            "resume_latest": config.resume_latest,
            "resume_checkpoint_path": str(config.resume_checkpoint_path) if config.resume_checkpoint_path else None,
            "resumed_from_checkpoint": runtime.resumed_from_checkpoint,
            "metrics_path": str(run_paths.metrics),
            "tuning_manifest_path": str(run_paths.tuning_manifest),
            "state_path": str(run_paths.run_dir / "state.json"),
        }
        write_json(run_paths.metrics, metrics)
        write_json(run_paths.tuning_manifest, tuning_manifest)
        write_json(run_paths.run_manifest, run_manifest)
        write_jsonl(run_paths.predictions, aggregate_predictions)
        final_heartbeat = build_sweep_heartbeat(
            status="completed",
            plan=plan,
            runtime=runtime,
            heartbeat_path=run_paths.heartbeat,
            control_dir=run_paths.control_dir,
            elapsed_runtime_seconds=runtime.accumulated_runtime_seconds,
            message="Composition sweep completed successfully.",
        )
        write_json(run_paths.heartbeat, final_heartbeat)

        return {
            "run_dir": str(run_paths.run_dir),
            "metrics_path": str(run_paths.metrics),
            "tuning_manifest_path": str(run_paths.tuning_manifest),
            "heartbeat_path": str(run_paths.heartbeat),
            "latest_checkpoint_path": str(run_paths.latest_checkpoint),
            "compositions": {task.name: state["compositions"][task.name]["status"] for task in config.compositions},
            "status": "composed",
            "resumed_from_checkpoint": runtime.resumed_from_checkpoint,
        }
    except Exception as exc:
        runtime.accumulated_runtime_seconds = elapsed_runtime_seconds()
        failure_heartbeat = build_sweep_heartbeat(
            status="failed",
            plan=plan,
            runtime=runtime,
            heartbeat_path=run_paths.heartbeat,
            control_dir=run_paths.control_dir,
            elapsed_runtime_seconds=runtime.accumulated_runtime_seconds,
            message=str(exc),
        )
        write_json(run_paths.heartbeat, failure_heartbeat)
        raise


def load_compose_config(
    path: str | Path,
    *,
    resume_latest: bool = False,
    resume_checkpoint_path: Path | None = None,
) -> ComposeConfig:
    app = load_app_config(path)
    return ComposeConfig.from_app_config(
        app,
        resume_latest=resume_latest,
        resume_checkpoint_path=resume_checkpoint_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune SOAR pair/triple composition and merge baselines on validation.")
    parser.add_argument("--config", required=True, help="Path to a composition YAML config.")
    parser.add_argument("--json", action="store_true", help="Print a JSON summary.")
    parser.add_argument("--resume-latest", action="store_true", help="Resume from the latest saved checkpoint.")
    parser.add_argument("--resume-checkpoint", help="Resume from an explicit checkpoint path.")
    args = parser.parse_args()

    config = load_compose_config(
        Path(args.config),
        resume_latest=args.resume_latest,
        resume_checkpoint_path=Path(args.resume_checkpoint) if args.resume_checkpoint else None,
    )
    summary = compose_residuals(config)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        print(f"Wrote composition sweep outputs to {summary['run_dir']}.")
        print(f"Metrics: {summary['metrics_path']}")
        print(f"Tuning manifest: {summary['tuning_manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
