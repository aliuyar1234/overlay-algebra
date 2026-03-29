from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import importlib.util
import json
from pathlib import Path
import platform
import random
import shutil
import time
from typing import Any, Iterable, Mapping, Sequence

from ..adapter_io import (
    AdapterBundle,
    load_adapter_bundle,
    load_compact_delta_artifact,
    save_adapter_from_lora_factors,
    save_compact_delta_artifact,
)
from ..artifacts import RunPaths, build_run_paths
from ..config import AppConfig, load_app_config
from ..data.processed import ProcessedExample, load_processed_examples
from ..feature_cache import default_feature_cache_dir
from ..metrics import score_prediction
from ..prompts import render_prompt
from ..soar import (
    DenseDeltaFactors,
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
from .adapter_eval import EvalTaskSpec, LoadedAdapterEvaluator


_PRIMARY_METRIC_BY_CONDITION = {
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


@dataclass(frozen=True, slots=True)
class ResidualizeConfig:
    app: AppConfig
    overlay_name: str
    base_model_name_or_path: str
    scaffold_adapter_dir: Path
    overlay_adapter_dir: Path
    eval_dataset_path: Path
    local_files_only: bool
    beta_grid: tuple[float, ...]
    residual_rank: int
    composed_rank: int
    synthesized_lora_dropout: float
    semantic_guardrail_points: float
    max_length: int
    max_new_tokens: int
    max_eval_examples: int
    eval_batch_size: int
    device: str
    torch_dtype: str
    reuse_feature_cache: bool
    feature_cache_dir: Path
    sort_eval_by_length: bool
    checkpoint_interval_units: int
    heartbeat_interval_seconds: float
    control_poll_seconds: float
    max_checkpoints_to_keep: int
    resume_latest: bool
    resume_checkpoint_path: Path | None
    scaffold_delta_artifact_dir: Path | None
    overlay_delta_artifact_dir: Path | None

    @classmethod
    def from_app_config(
        cls,
        app: AppConfig,
        *,
        resume_latest: bool = False,
        resume_checkpoint_path: Path | None = None,
    ) -> "ResidualizeConfig":
        payload = app.payload
        overlay_name = _payload_string(payload, "overlay_name")
        if overlay_name not in _PRIMARY_METRIC_BY_CONDITION:
            raise ValueError("overlay_name must be one of J, C, Q")
        configured_resume_checkpoint = Path(app.paths["resume_checkpoint"]) if "resume_checkpoint" in app.paths else None
        return cls(
            app=app,
            overlay_name=overlay_name,
            base_model_name_or_path=_payload_string(payload, "base_model_name_or_path"),
            scaffold_adapter_dir=Path(app.paths["scaffold_adapter"]),
            overlay_adapter_dir=Path(app.paths["overlay_adapter"]),
            eval_dataset_path=Path(app.paths["eval_dataset"]),
            local_files_only=_payload_bool(payload, "local_files_only", True),
            beta_grid=_payload_float_list(payload, "beta_grid") or (0.5, 0.75, 1.0, 1.25),
            residual_rank=_payload_int(payload, "residual_rank", 16),
            composed_rank=_payload_int(payload, "composed_rank", 32),
            synthesized_lora_dropout=_payload_float(payload, "synthesized_lora_dropout", 0.0),
            semantic_guardrail_points=_payload_float(payload, "semantic_guardrail_points", 3.0),
            max_length=_payload_int(payload, "max_length", 256),
            max_new_tokens=_payload_int(payload, "max_new_tokens", 48),
            max_eval_examples=_payload_int(payload, "max_eval_examples", 1),
            eval_batch_size=_payload_int(payload, "eval_batch_size", 1),
            device=_payload_string(payload, "device"),
            torch_dtype=_payload_string(payload, "torch_dtype"),
            reuse_feature_cache=_payload_bool(payload, "reuse_feature_cache", True),
            feature_cache_dir=Path(app.paths["feature_cache_dir"])
            if "feature_cache_dir" in app.paths
            else default_feature_cache_dir(Path(app.paths["eval_dataset"])),
            sort_eval_by_length=_payload_bool(payload, "sort_eval_by_length", True),
            checkpoint_interval_units=_payload_int(payload, "checkpoint_interval_units", 1),
            heartbeat_interval_seconds=_payload_float(payload, "heartbeat_interval_seconds", 30.0),
            control_poll_seconds=_payload_float(payload, "control_poll_seconds", 5.0),
            max_checkpoints_to_keep=_payload_int(payload, "max_checkpoints_to_keep", 3),
            resume_latest=resume_latest or _payload_bool(payload, "resume_latest", False),
            resume_checkpoint_path=resume_checkpoint_path or configured_resume_checkpoint,
            scaffold_delta_artifact_dir=Path(app.paths["scaffold_delta_artifact"])
            if "scaffold_delta_artifact" in app.paths
            else None,
            overlay_delta_artifact_dir=Path(app.paths["overlay_delta_artifact"])
            if "overlay_delta_artifact" in app.paths
            else None,
        )


@dataclass(frozen=True, slots=True)
class BetaCandidate:
    beta: float
    adapter_dir: Path
    metrics: dict[str, float]
    predictions: list[dict[str, Any]]

    @property
    def overlay_primary(self) -> float:
        return float(self.metrics.get(_PRIMARY_METRIC_BY_CONDITION[self.metrics["condition"]], 0.0))

    @property
    def answer_f1(self) -> float:
        return float(self.metrics.get("answer_f1", 0.0))

    @property
    def semantic_f1_points(self) -> float:
        return self.answer_f1 * 100.0


def _import_eval_stack() -> tuple[Any, Any, Any, Any]:
    missing = [
        name
        for name in ("torch", "transformers", "peft")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeError(
            "Missing M3 runtime dependencies: "
            + ", ".join(missing)
            + ". Use the CUDA-enabled runtime before running residualization."
        )

    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    peft = importlib.import_module("peft")
    return torch, transformers.AutoModelForCausalLM, transformers.AutoTokenizer, peft.PeftModel


def _resolve_dtype(torch_module: Any, dtype_name: str) -> Any:
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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass

    try:
        torch = importlib.import_module("torch")
    except Exception:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=False) for row in rows)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _evaluate_adapter_dirs(
    config: ResidualizeConfig,
    *,
    adapter_dirs: Mapping[str, Path],
) -> tuple[dict[str, tuple[dict[str, float], list[dict[str, Any]]]], dict[str, Any]]:
    evaluator = LoadedAdapterEvaluator(
        base_model_name_or_path=config.base_model_name_or_path,
        local_files_only=config.local_files_only,
        device=config.device,
        torch_dtype=config.torch_dtype,
        reuse_feature_cache=config.reuse_feature_cache,
    )
    feature_bundle = evaluator.get_eval_features(
        dataset_path=config.eval_dataset_path,
        condition=config.overlay_name,
        max_length=config.max_length,
        max_examples=config.max_eval_examples,
        cache_dir=config.feature_cache_dir,
    )
    tasks = [
        EvalTaskSpec(
            name=adapter_name,
            adapter_name=adapter_name,
            adapter_dir=adapter_dir,
            condition=config.overlay_name,
            max_length=config.max_length,
            max_new_tokens=config.max_new_tokens,
            batch_size=config.eval_batch_size,
            adapter_mode="temporary",
        )
        for adapter_name, adapter_dir in adapter_dirs.items()
    ]
    feature_map = {task.name: feature_bundle for task in tasks}
    bundle_results = evaluator.evaluate_bundle(
        tasks=tasks,
        features_by_task=feature_map,
        sort_by_length=config.sort_eval_by_length,
    )
    results = {
        task_name: (result.metrics, result.predictions)
        for task_name, result in bundle_results.items()
    }
    return (
        results,
        {
            "feature_cache_hit": feature_bundle.cache_hit,
            "feature_cache_path": str(feature_bundle.cache_path),
            "feature_cache_key": feature_bundle.descriptor.cache_key(),
            "feature_cache_descriptor": feature_bundle.descriptor.as_dict(),
            "eval_example_count": len(feature_bundle.features),
            "sort_eval_by_length": config.sort_eval_by_length,
            "generation_summaries": {
                task_name: result.generation_summary for task_name, result in bundle_results.items()
            },
        },
    )


def _slug_beta(beta: float) -> str:
    return str(beta).replace("-", "neg_").replace(".", "_")


def _overlay_primary_metric_name(condition: str) -> str:
    if condition not in _PRIMARY_METRIC_BY_CONDITION:
        raise ValueError(f"Unsupported single-overlay condition: {condition}")
    return _PRIMARY_METRIC_BY_CONDITION[condition]


def choose_best_beta(
    overlay_name: str,
    *,
    baseline_answer_f1: float,
    guardrail_points: float,
    candidates: Sequence[BetaCandidate],
) -> tuple[BetaCandidate, dict[str, Any]]:
    primary_metric = _overlay_primary_metric_name(overlay_name)
    floor = baseline_answer_f1 * 100.0 - guardrail_points
    passing = [candidate for candidate in candidates if candidate.semantic_f1_points >= floor]

    if passing:
        chosen = sorted(
            passing,
            key=lambda item: (-float(item.metrics.get(primary_metric, 0.0)), abs(item.beta - 1.0)),
        )[0]
        selection = {
            "guardrail_floor_points": floor,
            "guardrail_satisfied": True,
            "primary_metric": primary_metric,
            "selection_rule": "prefer guardrail-passing candidates, maximize overlay primary, tie-break beta closest to 1.0",
        }
        return chosen, selection

    chosen = sorted(
        candidates,
        key=lambda item: (-item.semantic_f1_points, abs(item.beta - 1.0), -float(item.metrics.get(primary_metric, 0.0))),
    )[0]
    selection = {
        "guardrail_floor_points": floor,
        "guardrail_satisfied": False,
        "primary_metric": primary_metric,
        "selection_rule": "no candidate passed the guardrail; choose highest semantic F1, tie-break beta closest to 1.0",
    }
    return chosen, selection


def _ensure_run_layout(run_paths: RunPaths, source_config: Path) -> None:
    run_paths.run_dir.mkdir(parents=True, exist_ok=True)
    run_paths.adapter_dir.mkdir(parents=True, exist_ok=True)
    run_paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    run_paths.control_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_config, run_paths.resolved_config)


def _resolve_resume_checkpoint_path(config: ResidualizeConfig, run_paths: RunPaths) -> Path | None:
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


def _initial_state(config: ResidualizeConfig) -> dict[str, Any]:
    return {
        "status": "pending",
        "prompt_only": None,
        "candidates": {
            _slug_beta(beta): {
                "beta": beta,
                "status": "pending",
                "adapter_dir": None,
                "metrics": None,
                "metrics_path": None,
                "predictions_path": None,
                "generation_summary": None,
                "overlay_primary": None,
                "semantic_f1_points": None,
                "guardrail_satisfied": None,
            }
            for beta in config.beta_grid
        },
        "chosen_beta": None,
        "selection": None,
    }


def _residualize_total_units(config: ResidualizeConfig) -> int:
    return 1 + len(config.beta_grid)


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


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


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


def residualize_overlay(config: ResidualizeConfig) -> dict[str, Any]:
    _set_seed(config.app.seed)

    run_paths = build_run_paths(config.app)
    resume_checkpoint_dir = _resolve_resume_checkpoint_path(config, run_paths)
    _assert_run_dir_available(run_paths, resume_checkpoint_dir=resume_checkpoint_dir)
    _ensure_run_layout(run_paths, config.app.source_path)
    write_environment_summary(run_paths.environment)

    scaffold_bundle = load_adapter_bundle(config.scaffold_adapter_dir)
    overlay_bundle = load_adapter_bundle(config.overlay_adapter_dir)
    if scaffold_bundle.base_model_name_or_path != overlay_bundle.base_model_name_or_path:
        raise ValueError("Scaffold and overlay adapters must share the same base model.")
    if scaffold_bundle.base_model_name_or_path != config.base_model_name_or_path:
        raise ValueError("Config base model does not match the adapter metadata.")

    delta_cache_dir = run_paths.run_dir / "delta_cache"
    scaffold_factors, scaffold_artifact_dir = _load_or_extract_dense_factors(
        bundle=scaffold_bundle,
        requested_artifact_dir=config.scaffold_delta_artifact_dir,
        fallback_artifact_dir=delta_cache_dir / "scaffold",
    )
    overlay_factors, overlay_artifact_dir = _load_or_extract_dense_factors(
        bundle=overlay_bundle,
        requested_artifact_dir=config.overlay_delta_artifact_dir,
        fallback_artifact_dir=delta_cache_dir / "overlay",
    )

    plan = SweepPlan.from_counts(total_units=_residualize_total_units(config), unit_label="evaluation_units")
    runtime = SweepRuntimeState()
    state = _initial_state(config)
    if resume_checkpoint_dir is not None:
        runtime, state = load_sweep_checkpoint(resume_checkpoint_dir)
        if runtime.units_completed > plan.total_units:
            raise ValueError(
                "Resume checkpoint is ahead of the current residualization plan: "
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
    feature_bundle = evaluator.get_eval_features(
        dataset_path=config.eval_dataset_path,
        condition=config.overlay_name,
        max_length=config.max_length,
        max_examples=config.max_eval_examples,
        cache_dir=config.feature_cache_dir,
    )

    def elapsed_runtime_seconds() -> float:
        return runtime.accumulated_runtime_seconds + max(0.0, time.monotonic() - session_started - paused_seconds)

    _write_json(run_paths.run_dir / "state.json", state)
    _write_json(
        run_paths.heartbeat,
        build_sweep_heartbeat(
            status="preparing",
            plan=plan,
            runtime=runtime,
            heartbeat_path=run_paths.heartbeat,
            control_dir=run_paths.control_dir,
            elapsed_runtime_seconds=elapsed_runtime_seconds(),
            active_item=config.overlay_name,
            phase="residualize",
            message="Residualization state is ready; loading adapters and cached eval features.",
        ),
    )

    residual_by_beta: dict[float, dict[str, DenseDeltaFactors]] = {}
    residual_rank = int(config.residual_rank)
    composed_rank = int(config.composed_rank)
    candidates_root = run_paths.run_dir / "candidates"
    primary_metric = _overlay_primary_metric_name(config.overlay_name)

    try:
        if state["prompt_only"] is None:
            if run_paths.pause_request.exists():
                paused_seconds += maybe_pause_sweep(
                    pause_request_path=run_paths.pause_request,
                    heartbeat_path=run_paths.heartbeat,
                    control_dir=run_paths.control_dir,
                    plan=plan,
                    runtime=runtime,
                    elapsed_runtime_seconds=elapsed_runtime_seconds(),
                    active_item=config.overlay_name,
                    phase="prompt_only_eval",
                    control_poll_seconds=config.control_poll_seconds,
                )
                last_heartbeat_timestamp = time.monotonic()

            prompt_result = evaluator.evaluate_adapter(
                adapter_name=f"prompt_only_{config.overlay_name.lower()}",
                adapter_dir=config.scaffold_adapter_dir,
                eval_features=feature_bundle.features,
                condition=config.overlay_name,
                max_length=config.max_length,
                max_new_tokens=config.max_new_tokens,
                batch_size=config.eval_batch_size,
                sort_by_length=config.sort_eval_by_length,
            )
            prompt_dir = run_paths.run_dir / "prompt_only"
            prompt_metrics = {"condition": config.overlay_name, **prompt_result.metrics}
            _write_json(
                prompt_dir / "metrics.json",
                {
                    "condition": config.overlay_name,
                    "metrics": prompt_metrics,
                    "generation_summary": prompt_result.generation_summary,
                    "feature_cache_hit": feature_bundle.cache_hit,
                    "feature_cache_path": str(feature_bundle.cache_path),
                    "feature_cache_key": feature_bundle.descriptor.cache_key(),
                    "feature_cache_descriptor": feature_bundle.descriptor.as_dict(),
                },
            )
            _write_jsonl(prompt_dir / "predictions.jsonl", prompt_result.predictions)
            state["prompt_only"] = {
                "metrics": prompt_metrics,
                "metrics_path": str(prompt_dir / "metrics.json"),
                "predictions_path": str(prompt_dir / "predictions.jsonl"),
                "generation_summary": prompt_result.generation_summary,
            }
            runtime.units_completed += 1
            _write_json(run_paths.run_dir / "state.json", state)
            _save_checkpoint_state(
                runtime=runtime,
                run_paths=run_paths,
                state=state,
                keep_last=config.max_checkpoints_to_keep,
                plan=plan,
                elapsed_runtime_seconds=elapsed_runtime_seconds(),
                active_item=config.overlay_name,
                phase="prompt_only_eval",
            )
            last_heartbeat_timestamp = time.monotonic()

        baseline_answer_f1 = float(state["prompt_only"]["metrics"].get("answer_f1", 0.0))
        for beta in config.beta_grid:
            beta_key = _slug_beta(beta)
            candidate_state = state["candidates"][beta_key]
            if candidate_state["status"] == "completed":
                continue
            if run_paths.pause_request.exists():
                paused_seconds += maybe_pause_sweep(
                    pause_request_path=run_paths.pause_request,
                    heartbeat_path=run_paths.heartbeat,
                    control_dir=run_paths.control_dir,
                    plan=plan,
                    runtime=runtime,
                    elapsed_runtime_seconds=elapsed_runtime_seconds(),
                    active_item=config.overlay_name,
                    phase="beta_tuning",
                    control_poll_seconds=config.control_poll_seconds,
                    extra_progress={"beta": beta},
                )
                last_heartbeat_timestamp = time.monotonic()

            residual_map = residualize_factor_map(overlay_factors, scaffold_factors, beta=beta, rank=residual_rank)
            residual_by_beta[beta] = residual_map
            composed_map = compose_factor_maps(
                scaffold_factors,
                residuals={config.overlay_name: residual_map},
                alphas={config.overlay_name: 1.0},
            )
            composed_lora = {
                module_name: factorize_dense_factors(factors, composed_rank, pad_to_rank=True)
                for module_name, factors in composed_map.items()
            }
            candidate_dir = candidates_root / f"beta_{beta_key}"
            adapter_dir = candidate_dir / "adapter"
            save_adapter_from_lora_factors(
                adapter_dir,
                template_bundle=scaffold_bundle,
                lora_factors=composed_lora,
                global_rank=composed_rank,
                lora_dropout=config.synthesized_lora_dropout,
            )
            save_compact_delta_artifact(
                candidate_dir / "delta_artifact",
                composed_map,
                source={
                    "overlay_name": config.overlay_name,
                    "beta": beta,
                    "composed_rank": composed_rank,
                    "scaffold_delta_artifact_dir": str(scaffold_artifact_dir),
                    "overlay_delta_artifact_dir": str(overlay_artifact_dir),
                },
            )

            evaluation = evaluator.evaluate_adapter(
                adapter_name=f"{config.overlay_name.lower()}_{beta_key}",
                adapter_dir=adapter_dir,
                eval_features=feature_bundle.features,
                condition=config.overlay_name,
                max_length=config.max_length,
                max_new_tokens=config.max_new_tokens,
                batch_size=config.eval_batch_size,
                sort_by_length=config.sort_eval_by_length,
                adapter_mode="temporary",
            )
            metrics = {"condition": config.overlay_name, **evaluation.metrics}
            _write_json(
                candidate_dir / "metrics.json",
                {
                    "condition": config.overlay_name,
                    "beta": beta,
                    "metrics": metrics,
                    "generation_summary": evaluation.generation_summary,
                    "feature_cache_hit": feature_bundle.cache_hit,
                    "feature_cache_path": str(feature_bundle.cache_path),
                    "feature_cache_key": feature_bundle.descriptor.cache_key(),
                    "feature_cache_descriptor": feature_bundle.descriptor.as_dict(),
                },
            )
            _write_jsonl(candidate_dir / "predictions.jsonl", evaluation.predictions)
            semantic_f1_points = float(metrics.get("answer_f1", 0.0)) * 100.0
            candidate_state.update(
                {
                    "status": "completed",
                    "adapter_dir": str(adapter_dir),
                    "metrics": metrics,
                    "metrics_path": str(candidate_dir / "metrics.json"),
                    "predictions_path": str(candidate_dir / "predictions.jsonl"),
                    "generation_summary": evaluation.generation_summary,
                    "overlay_primary": float(metrics.get(primary_metric, 0.0)),
                    "semantic_f1_points": semantic_f1_points,
                    "guardrail_satisfied": semantic_f1_points >= (baseline_answer_f1 * 100.0 - config.semantic_guardrail_points),
                }
            )
            runtime.units_completed += 1
            _write_json(run_paths.run_dir / "state.json", state)
            if runtime.units_completed % max(config.checkpoint_interval_units, 1) == 0:
                _save_checkpoint_state(
                    runtime=runtime,
                    run_paths=run_paths,
                    state=state,
                    keep_last=config.max_checkpoints_to_keep,
                    plan=plan,
                    elapsed_runtime_seconds=elapsed_runtime_seconds(),
                    active_item=config.overlay_name,
                    phase="beta_tuning",
                    extra_progress={"beta": beta},
                )
                last_heartbeat_timestamp = time.monotonic()
            if time.monotonic() - last_heartbeat_timestamp >= max(config.heartbeat_interval_seconds, 1.0):
                _write_json(
                    run_paths.heartbeat,
                    build_sweep_heartbeat(
                        status="running",
                        plan=plan,
                        runtime=runtime,
                        heartbeat_path=run_paths.heartbeat,
                        control_dir=run_paths.control_dir,
                        elapsed_runtime_seconds=elapsed_runtime_seconds(),
                        active_item=config.overlay_name,
                        phase="beta_tuning",
                        extra_progress={"beta": beta},
                    ),
                )
                last_heartbeat_timestamp = time.monotonic()

        if state["chosen_beta"] is None:
            beta_candidates = [
                BetaCandidate(
                    beta=beta,
                    adapter_dir=Path(str(state["candidates"][_slug_beta(beta)]["adapter_dir"])),
                    metrics=dict(state["candidates"][_slug_beta(beta)]["metrics"]),
                    predictions=[],
                )
                for beta in config.beta_grid
            ]
            chosen_candidate, selection = choose_best_beta(
                config.overlay_name,
                baseline_answer_f1=baseline_answer_f1,
                guardrail_points=config.semantic_guardrail_points,
                candidates=beta_candidates,
            )
            state["chosen_beta"] = chosen_candidate.beta
            state["selection"] = selection
            state["status"] = "completed"
            _write_json(run_paths.run_dir / "state.json", state)
            _save_checkpoint_state(
                runtime=runtime,
                run_paths=run_paths,
                state=state,
                keep_last=config.max_checkpoints_to_keep,
                plan=plan,
                elapsed_runtime_seconds=elapsed_runtime_seconds(),
                active_item=config.overlay_name,
                phase="selection",
                extra_progress={"chosen_beta": chosen_candidate.beta},
            )

        chosen_beta = float(state["chosen_beta"])
        selection = dict(state["selection"])
        chosen_candidate_state = state["candidates"][_slug_beta(chosen_beta)]
        if chosen_beta not in residual_by_beta:
            residual_by_beta[chosen_beta] = residualize_factor_map(
                overlay_factors,
                scaffold_factors,
                beta=chosen_beta,
                rank=residual_rank,
            )
        chosen_residual_factors = residual_by_beta[chosen_beta]
        chosen_composed_factors = compose_factor_maps(
            scaffold_factors,
            residuals={config.overlay_name: chosen_residual_factors},
            alphas={config.overlay_name: 1.0},
        )
        save_compact_delta_artifact(
            delta_cache_dir / "residual",
            chosen_residual_factors,
            source={
                "overlay_name": config.overlay_name,
                "beta": chosen_beta,
                "residual_rank": residual_rank,
                "scaffold_artifact_dir": str(scaffold_artifact_dir),
                "overlay_artifact_dir": str(overlay_artifact_dir),
            },
        )
        save_compact_delta_artifact(
            delta_cache_dir / "composed",
            chosen_composed_factors,
            source={
                "overlay_name": config.overlay_name,
                "beta": chosen_beta,
                "composed_rank": composed_rank,
                "residual_artifact_dir": str(delta_cache_dir / "residual"),
            },
        )

        residual_adapter_dir = run_paths.run_dir / "residual_adapter"
        residual_lora = {
            module_name: factorize_dense_factors(factors, residual_rank, pad_to_rank=True)
            for module_name, factors in chosen_residual_factors.items()
        }
        save_adapter_from_lora_factors(
            residual_adapter_dir,
            template_bundle=scaffold_bundle,
            lora_factors=residual_lora,
            global_rank=residual_rank,
            lora_dropout=config.synthesized_lora_dropout,
        )
        shutil.copytree(Path(str(chosen_candidate_state["adapter_dir"])), run_paths.adapter_dir, dirs_exist_ok=True)

        candidate_rows = [
            {
                "beta": beta,
                "adapter_dir": str(state["candidates"][_slug_beta(beta)]["adapter_dir"]),
                "metrics": dict(state["candidates"][_slug_beta(beta)]["metrics"]),
                "metrics_path": str(state["candidates"][_slug_beta(beta)]["metrics_path"]),
                "predictions_path": str(state["candidates"][_slug_beta(beta)]["predictions_path"]),
                "overlay_primary": float(state["candidates"][_slug_beta(beta)]["overlay_primary"]),
                "semantic_f1_points": float(state["candidates"][_slug_beta(beta)]["semantic_f1_points"]),
                "guardrail_satisfied": bool(state["candidates"][_slug_beta(beta)]["guardrail_satisfied"]),
            }
            for beta in config.beta_grid
        ]
        chosen_predictions = _read_jsonl_rows(Path(str(chosen_candidate_state["predictions_path"])))
        runtime.accumulated_runtime_seconds = elapsed_runtime_seconds()
        summary = {
            "status": "residualized",
            "overlay_name": config.overlay_name,
            "chosen_beta": chosen_beta,
            "primary_metric": primary_metric,
            "validation_metrics": dict(chosen_candidate_state["metrics"]),
            "guardrail_floor_points": selection["guardrail_floor_points"],
            "guardrail_satisfied": selection["guardrail_satisfied"],
            "scaffold_prompt_only_validation_metrics": dict(state["prompt_only"]["metrics"]),
            "residual_adapter_dir": str(residual_adapter_dir),
            "composed_adapter_dir": str(run_paths.adapter_dir),
            "feature_cache_hit": bool(feature_bundle.cache_hit),
            "feature_cache_path": str(feature_bundle.cache_path),
            "elapsed_runtime_seconds": runtime.accumulated_runtime_seconds,
        }
        tuning_manifest = {
            "overlay_name": config.overlay_name,
            "primary_metric": primary_metric,
            "beta_grid": list(config.beta_grid),
            "chosen_beta": chosen_beta,
            "selection": selection,
            "semantic_guardrail_points": config.semantic_guardrail_points,
            "prompt_only": dict(state["prompt_only"]),
            "candidates": candidate_rows,
            "feature_cache_hit": bool(feature_bundle.cache_hit),
            "feature_cache_path": str(feature_bundle.cache_path),
            "feature_cache_key": feature_bundle.descriptor.cache_key(),
            "feature_cache_descriptor": feature_bundle.descriptor.as_dict(),
            "validation_dataset_path": str(config.eval_dataset_path),
            "validation_example_count": len(feature_bundle.features),
            "scaffold_delta_artifact_dir": str(scaffold_artifact_dir),
            "overlay_delta_artifact_dir": str(overlay_artifact_dir),
            "residual_delta_artifact_dir": str(delta_cache_dir / "residual"),
            "composed_delta_artifact_dir": str(delta_cache_dir / "composed"),
        }
        run_manifest = {
            "stage": "residualize",
            "experiment_name": config.app.experiment_name,
            "milestone": config.app.milestone,
            "seed": config.app.seed,
            "base_model_name_or_path": config.base_model_name_or_path,
            "overlay_name": config.overlay_name,
            "scaffold_adapter_dir": str(config.scaffold_adapter_dir),
            "overlay_adapter_dir": str(config.overlay_adapter_dir),
            "eval_dataset_path": str(config.eval_dataset_path),
            "device": config.device,
            "torch_dtype": config.torch_dtype,
            "local_files_only": config.local_files_only,
            "beta_grid": list(config.beta_grid),
            "residual_rank": residual_rank,
            "composed_rank": composed_rank,
            "synthesized_lora_dropout": config.synthesized_lora_dropout,
            "semantic_guardrail_points": config.semantic_guardrail_points,
            "max_length": config.max_length,
            "max_new_tokens": config.max_new_tokens,
            "max_eval_examples": config.max_eval_examples,
            "eval_batch_size": config.eval_batch_size,
            "reuse_feature_cache": config.reuse_feature_cache,
            "feature_cache_dir": str(config.feature_cache_dir),
            "sort_eval_by_length": config.sort_eval_by_length,
            "checkpoint_interval_units": config.checkpoint_interval_units,
            "heartbeat_interval_seconds": config.heartbeat_interval_seconds,
            "control_poll_seconds": config.control_poll_seconds,
            "max_checkpoints_to_keep": config.max_checkpoints_to_keep,
            "resume_latest": config.resume_latest,
            "resume_checkpoint_path": str(config.resume_checkpoint_path) if config.resume_checkpoint_path else None,
            "resumed_from_checkpoint": runtime.resumed_from_checkpoint,
            "metrics_path": str(run_paths.metrics),
            "predictions_path": str(run_paths.predictions),
            "tuning_manifest_path": str(run_paths.tuning_manifest),
            "state_path": str(run_paths.run_dir / "state.json"),
        }
        state["status"] = "completed"
        state["chosen_predictions_path"] = str(run_paths.predictions)
        state["residual_adapter_dir"] = str(residual_adapter_dir)
        state["composed_adapter_dir"] = str(run_paths.adapter_dir)
        _write_json(run_paths.run_dir / "state.json", state)
        _save_checkpoint_state(
            runtime=runtime,
            run_paths=run_paths,
            state=state,
            keep_last=config.max_checkpoints_to_keep,
            plan=plan,
            elapsed_runtime_seconds=runtime.accumulated_runtime_seconds,
            active_item=config.overlay_name,
            phase="finalizing",
            message="Residualization outputs are ready; writing final artifacts.",
            extra_progress={"chosen_beta": state["chosen_beta"]},
        )
        _write_json(run_paths.metrics, summary)
        _write_json(run_paths.tuning_manifest, tuning_manifest)
        _write_json(run_paths.run_manifest, run_manifest)
        _write_jsonl(run_paths.predictions, chosen_predictions)
        _write_json(
            run_paths.heartbeat,
            build_sweep_heartbeat(
                status="completed",
                plan=plan,
                runtime=runtime,
                heartbeat_path=run_paths.heartbeat,
                control_dir=run_paths.control_dir,
                elapsed_runtime_seconds=runtime.accumulated_runtime_seconds,
                message="Residualization completed successfully.",
            ),
        )
        return {
            "run_dir": str(run_paths.run_dir),
            "metrics_path": str(run_paths.metrics),
            "predictions_path": str(run_paths.predictions),
            "tuning_manifest_path": str(run_paths.tuning_manifest),
            "heartbeat_path": str(run_paths.heartbeat),
            "latest_checkpoint_path": str(run_paths.latest_checkpoint),
            "status": "residualized",
            "overlay_name": config.overlay_name,
            "chosen_beta": chosen_beta,
            "residual_adapter_dir": str(residual_adapter_dir),
            "composed_adapter_dir": str(run_paths.adapter_dir),
            "resumed_from_checkpoint": runtime.resumed_from_checkpoint,
        }
    except Exception as exc:
        runtime.accumulated_runtime_seconds = elapsed_runtime_seconds()
        _write_json(
            run_paths.heartbeat,
            build_sweep_heartbeat(
                status="failed",
                plan=plan,
                runtime=runtime,
                heartbeat_path=run_paths.heartbeat,
                control_dir=run_paths.control_dir,
                elapsed_runtime_seconds=runtime.accumulated_runtime_seconds,
                message=str(exc),
            ),
        )
        raise


def load_residualize_config(
    path: str | Path,
    *,
    resume_latest: bool = False,
    resume_checkpoint_path: Path | None = None,
) -> ResidualizeConfig:
    app = load_app_config(path)
    return ResidualizeConfig.from_app_config(
        app,
        resume_latest=resume_latest,
        resume_checkpoint_path=resume_checkpoint_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run M3 dense-delta residualization for one overlay.")
    parser.add_argument("--config", required=True, help="Path to a residualization YAML config.")
    parser.add_argument("--json", action="store_true", help="Print a JSON summary.")
    parser.add_argument("--resume-latest", action="store_true", help="Resume from the latest saved checkpoint.")
    parser.add_argument("--resume-checkpoint", help="Resume from an explicit checkpoint path.")
    args = parser.parse_args()

    if args.resume_latest and args.resume_checkpoint:
        parser.error("--resume-latest and --resume-checkpoint cannot be used together")

    config = load_residualize_config(
        Path(args.config),
        resume_latest=args.resume_latest,
        resume_checkpoint_path=Path(args.resume_checkpoint) if args.resume_checkpoint else None,
    )
    summary = residualize_overlay(config)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        print(
            f"Residualized overlay {config.overlay_name} with beta={summary['chosen_beta']} "
            f"into {summary['composed_adapter_dir']}."
        )
        print(f"Tuning manifest: {summary['tuning_manifest_path']}")
        print(f"Metrics: {summary['metrics_path']}")
        print(f"Heartbeat: {summary['heartbeat_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
