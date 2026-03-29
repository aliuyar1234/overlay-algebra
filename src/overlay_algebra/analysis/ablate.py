from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
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
from ..soar import (
    DenseDeltaFactors,
    LoRAFactors,
    compose_factor_maps,
    dense_factor_map,
    factorize_dense_factors,
    residualize_factor_map,
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
    LoadedAdapterEvaluator,
    write_environment_summary,
    write_json,
)
from .compose import _parse_overlay_map
from .residualize import BetaCandidate, choose_best_beta


_PRIMARY_METRIC_BY_OVERLAY = {
    "J": "json_strict_schema_valid",
    "C": "citation_exact",
    "Q": "quote_exact",
}


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


def _payload_int_list(payload: dict[str, Any], key: str) -> tuple[int, ...]:
    value = payload.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError(f"Payload field {key!r} must be a list")
    return tuple(int(item) for item in value)


@dataclass(frozen=True, slots=True)
class AblationTask:
    kind: str
    overlay_name: str | None
    rank_grid: tuple[int, ...]
    manifest_paths: tuple[Path, ...]
    report_paths: dict[str, Path]

    @property
    def key(self) -> str:
        overlay = self.overlay_name.lower() if self.overlay_name is not None else "global"
        return f"{self.kind}_{overlay}"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AblationTask":
        raw_report_paths = payload.get("report_paths", {})
        if raw_report_paths is None:
            raw_report_paths = {}
        if not isinstance(raw_report_paths, Mapping):
            raise TypeError("ablation task report_paths must be a mapping when provided")
        manifest_paths = payload.get("manifest_paths", [])
        if manifest_paths is None:
            manifest_paths = []
        if not isinstance(manifest_paths, list):
            raise TypeError("ablation task manifest_paths must be a list when provided")
        return cls(
            kind=str(payload["kind"]),
            overlay_name=(str(payload["overlay_name"]) if payload.get("overlay_name") is not None else None),
            rank_grid=tuple(int(item) for item in payload.get("rank_grid", [])),
            manifest_paths=tuple(Path(str(item)) for item in manifest_paths),
            report_paths={str(key): Path(str(value)) for key, value in raw_report_paths.items()},
        )


@dataclass(frozen=True, slots=True)
class AblationConfig:
    app: AppConfig
    base_model_name_or_path: str
    scaffold_adapter_dir: Path
    validation_dataset_path: Path
    local_files_only: bool
    beta_grid: tuple[float, ...]
    semantic_guardrail_points: float
    max_length: int
    max_new_tokens: int
    max_eval_examples: int
    eval_batch_size: int
    device: str
    torch_dtype: str
    synthesized_lora_dropout: float
    checkpoint_interval_units: int
    heartbeat_interval_seconds: float
    control_poll_seconds: float
    max_checkpoints_to_keep: int
    resume_latest: bool
    resume_checkpoint_path: Path | None
    tasks: tuple[AblationTask, ...]
    overlay_adapter_dirs: dict[str, Path]
    residual_adapter_dirs: dict[str, Path]
    scaffold_delta_artifact_dir: Path | None
    overlay_delta_artifact_dirs: dict[str, Path]
    residual_delta_artifact_dirs: dict[str, Path]

    @classmethod
    def from_app_config(
        cls,
        app: AppConfig,
        *,
        resume_latest: bool = False,
        resume_checkpoint_path: Path | None = None,
    ) -> "AblationConfig":
        payload = app.payload
        raw_tasks = payload.get("tasks")
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise TypeError("payload.tasks must be a non-empty list")
        configured_resume_checkpoint = Path(app.paths["resume_checkpoint"]) if "resume_checkpoint" in app.paths else None
        return cls(
            app=app,
            base_model_name_or_path=_payload_string(payload, "base_model_name_or_path"),
            scaffold_adapter_dir=Path(app.paths["scaffold_adapter"]),
            validation_dataset_path=Path(app.paths["eval_dataset"]),
            local_files_only=_payload_bool(payload, "local_files_only", True),
            beta_grid=_payload_float_list(payload, "beta_grid") or (0.5, 0.75, 1.0, 1.25),
            semantic_guardrail_points=_payload_float(payload, "semantic_guardrail_points", 3.0),
            max_length=_payload_int(payload, "max_length", 256),
            max_new_tokens=_payload_int(payload, "max_new_tokens", 32),
            max_eval_examples=_payload_int(payload, "max_eval_examples", 1),
            eval_batch_size=_payload_int(payload, "eval_batch_size", 1),
            device=_payload_string(payload, "device"),
            torch_dtype=_payload_string(payload, "torch_dtype"),
            synthesized_lora_dropout=_payload_float(payload, "synthesized_lora_dropout", 0.0),
            checkpoint_interval_units=_payload_int(payload, "checkpoint_interval_units", 1),
            heartbeat_interval_seconds=_payload_float(payload, "heartbeat_interval_seconds", 30.0),
            control_poll_seconds=_payload_float(payload, "control_poll_seconds", 5.0),
            max_checkpoints_to_keep=_payload_int(payload, "max_checkpoints_to_keep", 3),
            resume_latest=resume_latest or _payload_bool(payload, "resume_latest", False),
            resume_checkpoint_path=resume_checkpoint_path or configured_resume_checkpoint,
            tasks=tuple(AblationTask.from_mapping(item) for item in raw_tasks),
            overlay_adapter_dirs=_parse_overlay_map(app.paths, "overlay_adapter"),
            residual_adapter_dirs=_parse_overlay_map(app.paths, "residual_adapter"),
            scaffold_delta_artifact_dir=Path(app.paths["scaffold_delta_artifact"])
            if "scaffold_delta_artifact" in app.paths
            else None,
            overlay_delta_artifact_dirs=_parse_overlay_map(app.paths, "overlay_delta_artifact"),
            residual_delta_artifact_dirs=_parse_overlay_map(app.paths, "residual_delta_artifact"),
        )


def _ensure_run_layout(run_paths: RunPaths, source_config: Path) -> None:
    run_paths.run_dir.mkdir(parents=True, exist_ok=True)
    run_paths.adapter_dir.mkdir(parents=True, exist_ok=True)
    run_paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    run_paths.control_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_config, run_paths.resolved_config)


def _resolve_resume_checkpoint_path(config: AblationConfig, run_paths: RunPaths) -> Path | None:
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


def _save_factor_map_as_adapter(
    *,
    output_dir: Path,
    template_bundle: AdapterBundle,
    factor_map: Mapping[str, DenseDeltaFactors],
    lora_dropout: float,
) -> tuple[Path, int]:
    ranks = {factors.rank_bound for factors in factor_map.values()}
    if not ranks:
        raise ValueError("factor_map must not be empty")
    global_rank = max(ranks)
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


def _initial_state(config: AblationConfig) -> dict[str, Any]:
    return {
        "tasks": {
            task.key: {
                "kind": task.kind,
                "overlay_name": task.overlay_name,
                "results": None,
                "status": "pending",
            }
            for task in config.tasks
        },
        "baselines": {},
    }


def _task_units(task: AblationTask, config: AblationConfig) -> int:
    _ = config
    _ = task
    return 1


def _total_units(config: AblationConfig) -> int:
    return sum(_task_units(task, config) for task in config.tasks)


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


def _aligned_raw_factor_residual(
    overlay_factors: Mapping[str, LoRAFactors],
    scaffold_factors: Mapping[str, LoRAFactors],
    *,
    beta: float,
) -> dict[str, LoRAFactors]:
    residuals: dict[str, LoRAFactors] = {}
    for module_name, full_factors in overlay_factors.items():
        scaffold = scaffold_factors[module_name]
        if full_factors.A.shape != scaffold.A.shape or full_factors.B.shape != scaffold.B.shape:
            raise ValueError("Raw-factor subtraction requires identical LoRA factor shapes per module.")
        residuals[module_name] = LoRAFactors(
            A=full_factors.A - float(beta) * scaffold.A,
            B=full_factors.B - float(beta) * scaffold.B,
            alpha=float(full_factors.rank),
        )
    return residuals


def _cosine_similarity_dense(a: DenseDeltaFactors, b: DenseDeltaFactors) -> float:
    a_dense = a.dense_delta()
    b_dense = b.dense_delta()
    numerator = float((a_dense * b_dense).sum())
    a_norm = float((a_dense * a_dense).sum()) ** 0.5
    b_norm = float((b_dense * b_dense).sum()) ** 0.5
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return numerator / (a_norm * b_norm)


def _baseline_metrics(
    *,
    baseline_cache: dict[str, Any],
    evaluator: LoadedAdapterEvaluator,
    config: AblationConfig,
    eval_features: Sequence[Mapping[str, Any]],
    overlay_name: str,
) -> dict[str, Any]:
    if overlay_name in baseline_cache:
        return dict(baseline_cache[overlay_name])
    result = evaluator.evaluate_adapter(
        adapter_name=f"baseline_{overlay_name.lower()}",
        adapter_dir=config.scaffold_adapter_dir,
        eval_features=eval_features,
        condition=overlay_name,
        max_length=config.max_length,
        max_new_tokens=config.max_new_tokens,
        batch_size=config.eval_batch_size,
    )
    metrics = {"condition": overlay_name, **result.metrics}
    baseline_cache[overlay_name] = metrics
    return metrics


def _run_factor_subtraction_task(
    *,
    task: AblationTask,
    task_dir: Path,
    config: AblationConfig,
    evaluator: LoadedAdapterEvaluator,
    eval_features: Sequence[Mapping[str, Any]],
    scaffold_bundle: AdapterBundle,
    overlay_bundles: Mapping[str, AdapterBundle],
    scaffold_dense_factors: Mapping[str, DenseDeltaFactors],
    baseline_cache: dict[str, Any],
) -> dict[str, Any]:
    if task.overlay_name is None:
        raise ValueError("factor_subtraction task requires overlay_name")
    overlay_name = task.overlay_name
    baseline_metrics = _baseline_metrics(
        baseline_cache=baseline_cache,
        evaluator=evaluator,
        config=config,
        eval_features=eval_features,
        overlay_name=overlay_name,
    )
    candidates: list[BetaCandidate] = []
    candidate_rows = []
    for beta in config.beta_grid:
        raw_residual = _aligned_raw_factor_residual(
            overlay_bundles[overlay_name].lora_factors,
            scaffold_bundle.lora_factors,
            beta=beta,
        )
        residual_dense = dense_factor_map(raw_residual)
        factor_map = compose_factor_maps(
            scaffold_dense_factors,
            residuals={overlay_name: residual_dense},
            alphas={overlay_name: 1.0},
        )
        candidate_dir = task_dir / f"beta_{str(beta).replace('.', '_')}"
        adapter_dir, global_rank = _save_factor_map_as_adapter(
            output_dir=candidate_dir / "adapter",
            template_bundle=scaffold_bundle,
            factor_map=factor_map,
            lora_dropout=config.synthesized_lora_dropout,
        )
        save_compact_delta_artifact(
            candidate_dir / "delta_artifact",
            factor_map,
            source={
                "task_kind": task.kind,
                "overlay_name": overlay_name,
                "beta": beta,
            },
        )
        evaluation = evaluator.evaluate_adapter(
            adapter_name=f"factor_{overlay_name.lower()}_{str(beta).replace('.', '_')}",
            adapter_dir=adapter_dir,
            eval_features=eval_features,
            condition=overlay_name,
            max_length=config.max_length,
            max_new_tokens=config.max_new_tokens,
            batch_size=config.eval_batch_size,
            adapter_mode="temporary",
        )
        metrics = {"condition": overlay_name, **evaluation.metrics}
        candidate = BetaCandidate(beta=beta, adapter_dir=adapter_dir, metrics=metrics, predictions=evaluation.predictions)
        candidates.append(candidate)
        candidate_rows.append(
            {
                "beta": beta,
                "adapter_dir": str(adapter_dir),
                "metrics": metrics,
                "global_rank": global_rank,
            }
        )

    chosen, selection = choose_best_beta(
        overlay_name,
        baseline_answer_f1=float(baseline_metrics.get("answer_f1", 0.0)),
        guardrail_points=config.semantic_guardrail_points,
        candidates=candidates,
    )
    result = {
        "overlay_name": overlay_name,
        "baseline_prompt_only_metrics": baseline_metrics,
        "chosen_beta": chosen.beta,
        "selection": selection,
        "primary_metric": _PRIMARY_METRIC_BY_OVERLAY[overlay_name],
        "candidates": candidate_rows,
        "chosen_metrics": chosen.metrics,
        "chosen_adapter_dir": str(chosen.adapter_dir),
    }
    write_json(task_dir / "metrics.json", result)
    return result


def _run_rank_sweep_task(
    *,
    task: AblationTask,
    task_dir: Path,
    config: AblationConfig,
    evaluator: LoadedAdapterEvaluator,
    eval_features: Sequence[Mapping[str, Any]],
    scaffold_bundle: AdapterBundle,
    overlay_dense_factors: Mapping[str, Mapping[str, DenseDeltaFactors]],
    scaffold_dense_factors: Mapping[str, DenseDeltaFactors],
    baseline_cache: dict[str, Any],
) -> dict[str, Any]:
    if task.overlay_name is None:
        raise ValueError("rank_sweep task requires overlay_name")
    overlay_name = task.overlay_name
    baseline_metrics = _baseline_metrics(
        baseline_cache=baseline_cache,
        evaluator=evaluator,
        config=config,
        eval_features=eval_features,
        overlay_name=overlay_name,
    )
    rank_results = []
    for rank in task.rank_grid:
        candidates: list[BetaCandidate] = []
        for beta in config.beta_grid:
            residual_map = residualize_factor_map(
                overlay_dense_factors[overlay_name],
                scaffold_dense_factors,
                beta=beta,
                rank=rank,
            )
            factor_map = compose_factor_maps(
                scaffold_dense_factors,
                residuals={overlay_name: residual_map},
                alphas={overlay_name: 1.0},
            )
            candidate_dir = task_dir / f"rank_{rank}" / f"beta_{str(beta).replace('.', '_')}"
            adapter_dir, _global_rank = _save_factor_map_as_adapter(
                output_dir=candidate_dir / "adapter",
                template_bundle=scaffold_bundle,
                factor_map=factor_map,
                lora_dropout=config.synthesized_lora_dropout,
            )
            evaluation = evaluator.evaluate_adapter(
                adapter_name=f"rank_{overlay_name.lower()}_{rank}_{str(beta).replace('.', '_')}",
                adapter_dir=adapter_dir,
                eval_features=eval_features,
                condition=overlay_name,
                max_length=config.max_length,
                max_new_tokens=config.max_new_tokens,
                batch_size=config.eval_batch_size,
                adapter_mode="temporary",
            )
            metrics = {"condition": overlay_name, **evaluation.metrics}
            candidates.append(BetaCandidate(beta=beta, adapter_dir=adapter_dir, metrics=metrics, predictions=evaluation.predictions))
        chosen, selection = choose_best_beta(
            overlay_name,
            baseline_answer_f1=float(baseline_metrics.get("answer_f1", 0.0)),
            guardrail_points=config.semantic_guardrail_points,
            candidates=candidates,
        )
        rank_results.append(
            {
                "rank": rank,
                "chosen_beta": chosen.beta,
                "selection": selection,
                "chosen_metrics": chosen.metrics,
                "chosen_adapter_dir": str(chosen.adapter_dir),
            }
        )
    result = {
        "overlay_name": overlay_name,
        "baseline_prompt_only_metrics": baseline_metrics,
        "rank_results": rank_results,
        "primary_metric": _PRIMARY_METRIC_BY_OVERLAY[overlay_name],
    }
    write_json(task_dir / "metrics.json", result)
    return result


def _run_energy_summary_task(
    *,
    task: AblationTask,
    task_dir: Path,
    scaffold_dense_factors: Mapping[str, DenseDeltaFactors],
    overlay_dense_factors: Mapping[str, Mapping[str, DenseDeltaFactors]],
    residual_dense_factors: Mapping[str, Mapping[str, DenseDeltaFactors]],
) -> dict[str, Any]:
    if task.overlay_name is not None:
        overlays = [task.overlay_name]
    else:
        overlays = sorted(overlay_dense_factors.keys())
    rows: dict[str, Any] = {}
    for overlay in overlays:
        module_rows = {}
        for module_name, scaffold_factors in scaffold_dense_factors.items():
            direct = overlay_dense_factors[overlay][module_name]
            residual = residual_dense_factors.get(overlay, {}).get(module_name)
            module_rows[module_name] = {
                "scaffold_rank_bound": scaffold_factors.rank_bound,
                "direct_rank_bound": direct.rank_bound,
                "scaffold_direct_cosine": _cosine_similarity_dense(scaffold_factors, direct),
                "residual_rank_bound": residual.rank_bound if residual is not None else None,
                "residual_direct_cosine": _cosine_similarity_dense(residual, direct) if residual is not None else None,
            }
        rows[overlay] = module_rows
    result = {"overlay_energy_overlap": rows}
    write_json(task_dir / "metrics.json", result)
    return result


def _run_manifest_summary_task(*, task: AblationTask, task_dir: Path) -> dict[str, Any]:
    manifests = [_read_manifest(path) for path in task.manifest_paths]
    result = {"manifest_count": len(manifests), "manifests": manifests}
    write_json(task_dir / "metrics.json", result)
    return result


def _run_report_summary_task(*, task: AblationTask, task_dir: Path) -> dict[str, Any]:
    summary = {
        name: json.loads(path.read_text(encoding="utf-8-sig"))
        for name, path in task.report_paths.items()
    }
    result = {"reports": summary}
    write_json(task_dir / "metrics.json", result)
    return result


def _read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def run_ablations(config: AblationConfig) -> dict[str, Any]:
    run_paths = build_run_paths(config.app)
    resume_checkpoint_dir = _resolve_resume_checkpoint_path(config, run_paths)
    _assert_run_dir_available(run_paths, resume_checkpoint_dir=resume_checkpoint_dir)
    _ensure_run_layout(run_paths, config.app.source_path)
    write_environment_summary(run_paths.environment)

    scaffold_bundle = load_adapter_bundle(config.scaffold_adapter_dir)
    if scaffold_bundle.base_model_name_or_path != config.base_model_name_or_path:
        raise ValueError("Config base model does not match the scaffold adapter metadata.")

    required_overlays = sorted({task.overlay_name for task in config.tasks if task.overlay_name is not None})
    overlay_bundles = {overlay: load_adapter_bundle(config.overlay_adapter_dirs[overlay]) for overlay in required_overlays if overlay in config.overlay_adapter_dirs}
    missing_overlays = sorted(set(required_overlays) - set(overlay_bundles.keys()))
    if missing_overlays:
        raise KeyError(f"Missing overlay adapters for: {', '.join(missing_overlays)}")

    delta_cache_root = run_paths.run_dir / "delta_cache"
    scaffold_dense_factors, _scaffold_delta_artifact_dir = _load_or_extract_dense_factors(
        bundle=scaffold_bundle,
        requested_artifact_dir=config.scaffold_delta_artifact_dir,
        fallback_artifact_dir=delta_cache_root / "scaffold",
    )
    overlay_dense_factors: dict[str, dict[str, DenseDeltaFactors]] = {}
    residual_dense_factors: dict[str, dict[str, DenseDeltaFactors]] = {}
    for overlay, bundle in overlay_bundles.items():
        dense_map, _artifact_dir = _load_or_extract_dense_factors(
            bundle=bundle,
            requested_artifact_dir=config.overlay_delta_artifact_dirs.get(overlay),
            fallback_artifact_dir=delta_cache_root / "overlay" / overlay.lower(),
        )
        overlay_dense_factors[overlay] = dense_map
        if overlay in config.residual_adapter_dirs:
            residual_bundle = load_adapter_bundle(config.residual_adapter_dirs[overlay])
            residual_map, _residual_dir = _load_or_extract_dense_factors(
                bundle=residual_bundle,
                requested_artifact_dir=config.residual_delta_artifact_dirs.get(overlay),
                fallback_artifact_dir=delta_cache_root / "residual" / overlay.lower(),
            )
            residual_dense_factors[overlay] = residual_map

    plan = SweepPlan.from_counts(total_units=_total_units(config), unit_label="ablation_units")
    runtime = SweepRuntimeState()
    state = _initial_state(config)
    if resume_checkpoint_dir is not None:
        runtime, state = load_sweep_checkpoint(resume_checkpoint_dir)
        if runtime.units_completed > plan.total_units:
            raise ValueError(
                "Resume checkpoint is ahead of the current ablation plan: "
                f"{runtime.units_completed} > {plan.total_units}."
            )

    session_started = time.monotonic()
    paused_seconds = 0.0
    last_heartbeat_timestamp = session_started

    def elapsed_runtime_seconds() -> float:
        return runtime.accumulated_runtime_seconds + max(0.0, time.monotonic() - session_started - paused_seconds)

    write_json(run_paths.run_dir / "state.json", state)
    evaluator = LoadedAdapterEvaluator(
        base_model_name_or_path=config.base_model_name_or_path,
        local_files_only=config.local_files_only,
        device=config.device,
        torch_dtype=config.torch_dtype,
    )
    eval_feature_cache: dict[str, list[dict[str, Any]]] = {}
    try:
        for task in config.tasks:
            task_state = state["tasks"][task.key]
            if task_state["status"] == "completed":
                continue
            if run_paths.pause_request.exists():
                paused_seconds += maybe_pause_sweep(
                    pause_request_path=run_paths.pause_request,
                    heartbeat_path=run_paths.heartbeat,
                    control_dir=run_paths.control_dir,
                    plan=plan,
                    runtime=runtime,
                    elapsed_runtime_seconds=elapsed_runtime_seconds(),
                    active_item=task.key,
                    phase=task.kind,
                    control_poll_seconds=config.control_poll_seconds,
                )
                last_heartbeat_timestamp = time.monotonic()

            task_dir = run_paths.run_dir / "tasks" / task.key
            if task.kind == "factor_subtraction":
                if task.overlay_name is None:
                    raise ValueError("factor_subtraction task requires overlay_name")
                eval_features = eval_feature_cache.get(task.overlay_name)
                if eval_features is None:
                    eval_features = evaluator.get_eval_features(
                        dataset_path=config.validation_dataset_path,
                        condition=task.overlay_name,
                        max_length=config.max_length,
                        max_examples=config.max_eval_examples,
                    ).features
                    eval_feature_cache[task.overlay_name] = eval_features
                result = _run_factor_subtraction_task(
                    task=task,
                    task_dir=task_dir,
                    config=config,
                    evaluator=evaluator,
                    eval_features=eval_features,
                    scaffold_bundle=scaffold_bundle,
                    overlay_bundles=overlay_bundles,
                    scaffold_dense_factors=scaffold_dense_factors,
                    baseline_cache=state["baselines"],
                )
            elif task.kind == "rank_sweep":
                if task.overlay_name is None:
                    raise ValueError("rank_sweep task requires overlay_name")
                eval_features = eval_feature_cache.get(task.overlay_name)
                if eval_features is None:
                    eval_features = evaluator.get_eval_features(
                        dataset_path=config.validation_dataset_path,
                        condition=task.overlay_name,
                        max_length=config.max_length,
                        max_examples=config.max_eval_examples,
                    ).features
                    eval_feature_cache[task.overlay_name] = eval_features
                result = _run_rank_sweep_task(
                    task=task,
                    task_dir=task_dir,
                    config=config,
                    evaluator=evaluator,
                    eval_features=eval_features,
                    scaffold_bundle=scaffold_bundle,
                    overlay_dense_factors=overlay_dense_factors,
                    scaffold_dense_factors=scaffold_dense_factors,
                    baseline_cache=state["baselines"],
                )
            elif task.kind == "energy_summary":
                result = _run_energy_summary_task(
                    task=task,
                    task_dir=task_dir,
                    scaffold_dense_factors=scaffold_dense_factors,
                    overlay_dense_factors=overlay_dense_factors,
                    residual_dense_factors=residual_dense_factors,
                )
            elif task.kind == "manifest_summary":
                result = _run_manifest_summary_task(task=task, task_dir=task_dir)
            elif task.kind == "report_summary":
                result = _run_report_summary_task(task=task, task_dir=task_dir)
            else:
                raise ValueError(f"Unsupported ablation task kind: {task.kind}")

            task_state["results"] = result
            task_state["status"] = "completed"
            runtime.units_completed += _task_units(task, config)
            write_json(run_paths.run_dir / "state.json", state)
            _save_checkpoint_state(
                runtime=runtime,
                run_paths=run_paths,
                state=state,
                keep_last=config.max_checkpoints_to_keep,
                plan=plan,
                elapsed_runtime_seconds=elapsed_runtime_seconds(),
                active_item=task.key,
                phase=task.kind,
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
                    active_item=task.key,
                    phase=task.kind,
                )
                write_json(run_paths.heartbeat, heartbeat)
                last_heartbeat_timestamp = time.monotonic()

        runtime.accumulated_runtime_seconds = elapsed_runtime_seconds()
        metrics = {
            "status": "completed",
            "tasks": state["tasks"],
            "elapsed_runtime_seconds": runtime.accumulated_runtime_seconds,
        }
        run_manifest = {
            "stage": "ablate",
            "experiment_name": config.app.experiment_name,
            "milestone": config.app.milestone,
            "seed": config.app.seed,
            "base_model_name_or_path": config.base_model_name_or_path,
            "validation_dataset_path": str(config.validation_dataset_path),
            "device": config.device,
            "torch_dtype": config.torch_dtype,
            "local_files_only": config.local_files_only,
            "beta_grid": list(config.beta_grid),
            "semantic_guardrail_points": config.semantic_guardrail_points,
            "max_length": config.max_length,
            "max_new_tokens": config.max_new_tokens,
            "max_eval_examples": config.max_eval_examples,
            "eval_batch_size": config.eval_batch_size,
            "checkpoint_interval_units": config.checkpoint_interval_units,
            "heartbeat_interval_seconds": config.heartbeat_interval_seconds,
            "control_poll_seconds": config.control_poll_seconds,
            "max_checkpoints_to_keep": config.max_checkpoints_to_keep,
            "resumed_from_checkpoint": runtime.resumed_from_checkpoint,
            "metrics_path": str(run_paths.metrics),
        }
        write_json(run_paths.metrics, metrics)
        write_json(run_paths.run_manifest, run_manifest)
        final_heartbeat = build_sweep_heartbeat(
            status="completed",
            plan=plan,
            runtime=runtime,
            heartbeat_path=run_paths.heartbeat,
            control_dir=run_paths.control_dir,
            elapsed_runtime_seconds=runtime.accumulated_runtime_seconds,
            message="Ablation sweep completed successfully.",
        )
        write_json(run_paths.heartbeat, final_heartbeat)
        return {
            "run_dir": str(run_paths.run_dir),
            "metrics_path": str(run_paths.metrics),
            "heartbeat_path": str(run_paths.heartbeat),
            "latest_checkpoint_path": str(run_paths.latest_checkpoint),
            "status": "completed",
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


def load_ablation_config(
    path: str | Path,
    *,
    resume_latest: bool = False,
    resume_checkpoint_path: Path | None = None,
) -> AblationConfig:
    app = load_app_config(path)
    return AblationConfig.from_app_config(
        app,
        resume_latest=resume_latest,
        resume_checkpoint_path=resume_checkpoint_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run analysis and ablation sweeps over saved overlay artifacts.")
    parser.add_argument("--config", required=True, help="Path to an ablation YAML config.")
    parser.add_argument("--json", action="store_true", help="Print a JSON summary.")
    parser.add_argument("--resume-latest", action="store_true", help="Resume from the latest saved checkpoint.")
    parser.add_argument("--resume-checkpoint", help="Resume from an explicit checkpoint path.")
    args = parser.parse_args()

    config = load_ablation_config(
        Path(args.config),
        resume_latest=args.resume_latest,
        resume_checkpoint_path=Path(args.resume_checkpoint) if args.resume_checkpoint else None,
    )
    summary = run_ablations(config)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        print(f"Wrote ablation outputs to {summary['run_dir']}.")
        print(f"Metrics: {summary['metrics_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
