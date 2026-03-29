from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..analysis.adapter_eval import LoadedAdapterEvaluator, write_environment_summary, write_json, write_jsonl
from ..artifacts import RunPaths, build_run_paths
from ..config import AppConfig, load_app_config
from ..feature_cache import default_feature_cache_dir
from ..sweep_runtime import (
    SweepPlan,
    SweepRuntimeState,
    build_sweep_heartbeat,
    load_sweep_checkpoint,
    maybe_pause_sweep,
    resolve_sweep_checkpoint_dir,
    save_sweep_checkpoint,
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
class BundleEvalTask:
    name: str
    adapter_dir: Path
    condition: str
    max_new_tokens: int
    batch_size: int
    adapter_name: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, default_batch_size: int) -> "BundleEvalTask":
        name = str(payload["name"])
        return cls(
            name=name,
            adapter_dir=Path(str(payload["adapter_dir"])),
            condition=str(payload["condition"]),
            max_new_tokens=int(payload["max_new_tokens"]),
            batch_size=int(payload.get("batch_size", default_batch_size)),
            adapter_name=str(payload.get("adapter_name", name)),
        )


@dataclass(frozen=True, slots=True)
class BundleEvalConfig:
    app: AppConfig
    base_model_name_or_path: str
    eval_dataset_path: Path
    local_files_only: bool
    max_length: int
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
    tasks: tuple[BundleEvalTask, ...]

    @classmethod
    def from_app_config(
        cls,
        app: AppConfig,
        *,
        resume_latest: bool = False,
        resume_checkpoint_path: Path | None = None,
    ) -> "BundleEvalConfig":
        payload = app.payload
        eval_dataset_path = Path(app.paths["eval_dataset"])
        raw_tasks = payload.get("tasks")
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise TypeError("payload.tasks must be a non-empty list")
        configured_resume_checkpoint = Path(app.paths["resume_checkpoint"]) if "resume_checkpoint" in app.paths else None
        default_batch_size = _payload_int(payload, "eval_batch_size", 1)
        return cls(
            app=app,
            base_model_name_or_path=_payload_string(payload, "base_model_name_or_path"),
            eval_dataset_path=eval_dataset_path,
            local_files_only=_payload_bool(payload, "local_files_only", True),
            max_length=_payload_int(payload, "max_length", 256),
            max_eval_examples=_payload_int(payload, "max_eval_examples", 1),
            eval_batch_size=default_batch_size,
            device=_payload_string(payload, "device"),
            torch_dtype=_payload_string(payload, "torch_dtype"),
            reuse_feature_cache=_payload_bool(payload, "reuse_feature_cache", True),
            feature_cache_dir=Path(app.paths["feature_cache_dir"])
            if "feature_cache_dir" in app.paths
            else default_feature_cache_dir(eval_dataset_path),
            sort_eval_by_length=_payload_bool(payload, "sort_eval_by_length", True),
            checkpoint_interval_units=_payload_int(payload, "checkpoint_interval_units", 1),
            heartbeat_interval_seconds=_payload_float(payload, "heartbeat_interval_seconds", 30.0),
            control_poll_seconds=_payload_float(payload, "control_poll_seconds", 5.0),
            max_checkpoints_to_keep=_payload_int(payload, "max_checkpoints_to_keep", 3),
            resume_latest=resume_latest or _payload_bool(payload, "resume_latest", False),
            resume_checkpoint_path=resume_checkpoint_path or configured_resume_checkpoint,
            tasks=tuple(BundleEvalTask.from_mapping(item, default_batch_size=default_batch_size) for item in raw_tasks),
        )


def _ensure_run_layout(run_paths: RunPaths, source_config: Path) -> None:
    run_paths.run_dir.mkdir(parents=True, exist_ok=True)
    run_paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    run_paths.control_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_config, run_paths.resolved_config)


def _resolve_resume_checkpoint_path(config: BundleEvalConfig, run_paths: RunPaths) -> Path | None:
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


def _initial_state(config: BundleEvalConfig) -> dict[str, Any]:
    return {
        "tasks": {
            task.name: {
                "status": "pending",
                "condition": task.condition,
                "adapter_dir": str(task.adapter_dir),
                "max_new_tokens": task.max_new_tokens,
                "batch_size": task.batch_size,
                "metrics": None,
                "metrics_path": None,
                "predictions_path": None,
                "feature_cache_hit": None,
                "feature_cache_path": None,
                "generation_summary": None,
            }
            for task in config.tasks
        }
    }


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _collect_aggregate_predictions(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    aggregate_predictions: list[dict[str, Any]] = []
    for task_name, task_state in state["tasks"].items():
        if task_state.get("status") != "completed":
            continue
        predictions_path = task_state.get("predictions_path")
        if not predictions_path:
            continue
        for row in _read_jsonl_rows(Path(str(predictions_path))):
            aggregate_row = dict(row)
            aggregate_row["task_name"] = task_name
            aggregate_predictions.append(aggregate_row)
    return aggregate_predictions


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
) -> Path:
    return save_sweep_checkpoint(
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


def run_bundle_eval(config: BundleEvalConfig) -> dict[str, Any]:
    run_paths = build_run_paths(config.app)
    resume_checkpoint_dir = _resolve_resume_checkpoint_path(config, run_paths)
    _assert_run_dir_available(run_paths, resume_checkpoint_dir=resume_checkpoint_dir)
    _ensure_run_layout(run_paths, config.app.source_path)
    write_environment_summary(run_paths.environment)

    plan = SweepPlan.from_counts(total_units=len(config.tasks), unit_label="eval_tasks")
    runtime = SweepRuntimeState()
    state = _initial_state(config)
    if resume_checkpoint_dir is not None:
        runtime, state = load_sweep_checkpoint(resume_checkpoint_dir)
        if runtime.units_completed > plan.total_units:
            raise ValueError(
                "Resume checkpoint is ahead of the current bundle-eval plan: "
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
    feature_bundles: dict[str, Any] = {}

    def elapsed_runtime_seconds() -> float:
        return runtime.accumulated_runtime_seconds + max(0.0, time.monotonic() - session_started - paused_seconds)

    write_json(run_paths.run_dir / "state.json", state)
    write_json(
        run_paths.heartbeat,
        build_sweep_heartbeat(
            status="preparing",
            plan=plan,
            runtime=runtime,
            heartbeat_path=run_paths.heartbeat,
            control_dir=run_paths.control_dir,
            elapsed_runtime_seconds=elapsed_runtime_seconds(),
            message="Bundle eval state is ready; loading adapters and cached eval features.",
        ),
    )

    try:
        for task in config.tasks:
            task_state = state["tasks"][task.name]
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
                    active_item=task.name,
                    phase="bundle_eval",
                    control_poll_seconds=config.control_poll_seconds,
                )
                last_heartbeat_timestamp = time.monotonic()

            feature_bundle = feature_bundles.get(task.condition)
            if feature_bundle is None:
                feature_bundle = evaluator.get_eval_features(
                    dataset_path=config.eval_dataset_path,
                    condition=task.condition,
                    max_length=config.max_length,
                    max_examples=config.max_eval_examples,
                    cache_dir=config.feature_cache_dir,
                )
                feature_bundles[task.condition] = feature_bundle

            evaluation = evaluator.evaluate_adapter(
                adapter_name=task.adapter_name,
                adapter_dir=task.adapter_dir,
                eval_features=feature_bundle.features,
                condition=task.condition,
                max_length=config.max_length,
                max_new_tokens=task.max_new_tokens,
                batch_size=task.batch_size,
                sort_by_length=config.sort_eval_by_length,
            )
            task_dir = run_paths.run_dir / "tasks" / task.name
            metrics_payload = {
                "condition": task.condition,
                "metrics": evaluation.metrics,
                "generation_summary": evaluation.generation_summary,
                "feature_cache_hit": feature_bundle.cache_hit,
                "feature_cache_path": str(feature_bundle.cache_path),
                "feature_cache_key": feature_bundle.descriptor.cache_key(),
                "feature_cache_descriptor": feature_bundle.descriptor.as_dict(),
            }
            write_json(task_dir / "metrics.json", metrics_payload)
            write_jsonl(task_dir / "predictions.jsonl", evaluation.predictions)

            task_state["status"] = "completed"
            task_state["metrics"] = evaluation.metrics
            task_state["metrics_path"] = str(task_dir / "metrics.json")
            task_state["predictions_path"] = str(task_dir / "predictions.jsonl")
            task_state["feature_cache_hit"] = feature_bundle.cache_hit
            task_state["feature_cache_path"] = str(feature_bundle.cache_path)
            task_state["generation_summary"] = evaluation.generation_summary
            runtime.units_completed += 1
            write_json(run_paths.run_dir / "state.json", state)
            if runtime.units_completed % max(config.checkpoint_interval_units, 1) == 0:
                _save_checkpoint_state(
                    runtime=runtime,
                    run_paths=run_paths,
                    state=state,
                    keep_last=config.max_checkpoints_to_keep,
                    plan=plan,
                    elapsed_runtime_seconds=elapsed_runtime_seconds(),
                    active_item=task.name,
                    phase="bundle_eval",
                )
                last_heartbeat_timestamp = time.monotonic()

            if time.monotonic() - last_heartbeat_timestamp >= max(config.heartbeat_interval_seconds, 1.0):
                write_json(
                    run_paths.heartbeat,
                    build_sweep_heartbeat(
                        status="running",
                        plan=plan,
                        runtime=runtime,
                        heartbeat_path=run_paths.heartbeat,
                        control_dir=run_paths.control_dir,
                        elapsed_runtime_seconds=elapsed_runtime_seconds(),
                        active_item=task.name,
                        phase="bundle_eval",
                    ),
                )
                last_heartbeat_timestamp = time.monotonic()

        runtime.accumulated_runtime_seconds = elapsed_runtime_seconds()
        aggregate_predictions = _collect_aggregate_predictions(state)
        metrics = {
            "status": "evaluated",
            "tasks": {task.name: state["tasks"][task.name] for task in config.tasks},
            "eval_dataset_path": str(config.eval_dataset_path),
            "elapsed_runtime_seconds": runtime.accumulated_runtime_seconds,
        }
        run_manifest = {
            "stage": "bundle_eval",
            "experiment_name": config.app.experiment_name,
            "milestone": config.app.milestone,
            "seed": config.app.seed,
            "base_model_name_or_path": config.base_model_name_or_path,
            "eval_dataset_path": str(config.eval_dataset_path),
            "device": config.device,
            "torch_dtype": config.torch_dtype,
            "local_files_only": config.local_files_only,
            "max_length": config.max_length,
            "max_eval_examples": config.max_eval_examples,
            "reuse_feature_cache": config.reuse_feature_cache,
            "feature_cache_dir": str(config.feature_cache_dir),
            "sort_eval_by_length": config.sort_eval_by_length,
            "checkpoint_interval_units": config.checkpoint_interval_units,
            "heartbeat_interval_seconds": config.heartbeat_interval_seconds,
            "control_poll_seconds": config.control_poll_seconds,
            "max_checkpoints_to_keep": config.max_checkpoints_to_keep,
            "resumed_from_checkpoint": runtime.resumed_from_checkpoint,
            "tasks": [
                {
                    "name": task.name,
                    "adapter_dir": str(task.adapter_dir),
                    "adapter_name": task.adapter_name,
                    "condition": task.condition,
                    "max_new_tokens": task.max_new_tokens,
                    "batch_size": task.batch_size,
                }
                for task in config.tasks
            ],
            "metrics_path": str(run_paths.metrics),
            "predictions_path": str(run_paths.predictions),
        }
        write_json(run_paths.metrics, metrics)
        write_json(run_paths.run_manifest, run_manifest)
        write_jsonl(run_paths.predictions, aggregate_predictions)
        write_json(
            run_paths.heartbeat,
            build_sweep_heartbeat(
                status="completed",
                plan=plan,
                runtime=runtime,
                heartbeat_path=run_paths.heartbeat,
                control_dir=run_paths.control_dir,
                elapsed_runtime_seconds=runtime.accumulated_runtime_seconds,
                message="Bundle evaluation completed successfully.",
            ),
        )
        return {
            "run_dir": str(run_paths.run_dir),
            "metrics_path": str(run_paths.metrics),
            "predictions_path": str(run_paths.predictions),
            "heartbeat_path": str(run_paths.heartbeat),
            "latest_checkpoint_path": str(run_paths.latest_checkpoint),
            "status": "evaluated",
            "resumed_from_checkpoint": runtime.resumed_from_checkpoint,
        }
    except Exception as exc:
        runtime.accumulated_runtime_seconds = elapsed_runtime_seconds()
        write_json(
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


def load_bundle_eval_config(
    path: str | Path,
    *,
    resume_latest: bool = False,
    resume_checkpoint_path: Path | None = None,
) -> BundleEvalConfig:
    app = load_app_config(path)
    return BundleEvalConfig.from_app_config(
        app,
        resume_latest=resume_latest,
        resume_checkpoint_path=resume_checkpoint_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate multiple adapters in one load-once session.")
    parser.add_argument("--config", required=True, help="Path to a bundled eval YAML config.")
    parser.add_argument("--json", action="store_true", help="Print a JSON summary.")
    parser.add_argument("--resume-latest", action="store_true", help="Resume from the latest saved checkpoint.")
    parser.add_argument("--resume-checkpoint", help="Resume from an explicit checkpoint path.")
    args = parser.parse_args()

    config = load_bundle_eval_config(
        Path(args.config),
        resume_latest=args.resume_latest,
        resume_checkpoint_path=Path(args.resume_checkpoint) if args.resume_checkpoint else None,
    )
    summary = run_bundle_eval(config)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        print(f"Wrote bundled eval outputs to {summary['run_dir']}.")
        print(f"Metrics: {summary['metrics_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
