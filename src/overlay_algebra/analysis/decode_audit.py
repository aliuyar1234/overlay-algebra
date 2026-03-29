from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..artifacts import RunPaths, build_run_paths
from ..config import AppConfig, load_app_config
from ..feature_cache import default_feature_cache_dir
from ..parsers import ParseError, parse_condition
from ..sweep_runtime import (
    SweepPlan,
    SweepRuntimeState,
    build_sweep_heartbeat,
    load_sweep_checkpoint,
    maybe_pause_sweep,
    resolve_sweep_checkpoint_dir,
    save_sweep_checkpoint,
)
from .adapter_eval import LoadedAdapterEvaluator, write_environment_summary, write_json, write_jsonl


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
class DecodeAuditTask:
    name: str
    adapter_dir: Path
    condition: str
    budgets: tuple[int, ...]
    batch_size: int
    adapter_name: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, default_batch_size: int) -> "DecodeAuditTask":
        raw_budgets = payload.get("budgets")
        if not isinstance(raw_budgets, list) or not raw_budgets:
            raise TypeError("decode audit task budgets must be a non-empty list")
        normalized_budgets = tuple(sorted({int(item) for item in raw_budgets}))
        name = str(payload["name"])
        return cls(
            name=name,
            adapter_dir=Path(str(payload["adapter_dir"])),
            condition=str(payload["condition"]),
            budgets=normalized_budgets,
            batch_size=int(payload.get("batch_size", default_batch_size)),
            adapter_name=str(payload.get("adapter_name", name)),
        )


@dataclass(frozen=True, slots=True)
class DecodeAuditConfig:
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
    tasks: tuple[DecodeAuditTask, ...]

    @classmethod
    def from_app_config(
        cls,
        app: AppConfig,
        *,
        resume_latest: bool = False,
        resume_checkpoint_path: Path | None = None,
    ) -> "DecodeAuditConfig":
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
            tasks=tuple(DecodeAuditTask.from_mapping(item, default_batch_size=default_batch_size) for item in raw_tasks),
        )


def _ensure_run_layout(run_paths: RunPaths, source_config: Path) -> None:
    run_paths.run_dir.mkdir(parents=True, exist_ok=True)
    run_paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    run_paths.control_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_config, run_paths.resolved_config)


def _resolve_resume_checkpoint_path(config: DecodeAuditConfig, run_paths: RunPaths) -> Path | None:
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


def _initial_state(config: DecodeAuditConfig) -> dict[str, Any]:
    return {
        "tasks": {
            task.name: {
                "status": "pending",
                "condition": task.condition,
                "adapter_dir": str(task.adapter_dir),
                "feature_cache_hit": None,
                "feature_cache_path": None,
                "budgets": {
                    str(budget): {
                        "status": "pending",
                        "metrics": None,
                        "metrics_path": None,
                        "predictions_path": None,
                        "generation_summary": None,
                    }
                    for budget in task.budgets
                },
                "comparison_path": None,
            }
            for task in config.tasks
        }
    }


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


def _read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _prediction_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        [
            {
                "example_id": str(row["example_id"]),
                "prediction": str(row["prediction"]),
            }
            for row in rows
        ],
        ensure_ascii=True,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_success_rate(condition: str, rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        return 0.0
    successes = 0
    for row in rows:
        try:
            parse_condition(str(row["prediction"]), condition)
        except ParseError:
            continue
        successes += 1
    return successes / len(rows)


def _compare_prediction_rows(
    *,
    reference_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reference_map = {str(row["example_id"]): str(row["prediction"]) for row in reference_rows}
    candidate_map = {str(row["example_id"]): str(row["prediction"]) for row in candidate_rows}
    shared_ids = [example_id for example_id in reference_map if example_id in candidate_map]
    changed_ids = [example_id for example_id in shared_ids if reference_map[example_id] != candidate_map[example_id]]
    return {
        "shared_example_count": len(shared_ids),
        "changed_example_count": len(changed_ids),
        "changed_example_rate": (len(changed_ids) / len(shared_ids) if shared_ids else 0.0),
        "identical_prediction_rate": (
            (len(shared_ids) - len(changed_ids)) / len(shared_ids) if shared_ids else 0.0
        ),
        "changed_example_ids_sample": changed_ids[:20],
        "reference_prediction_sha256": _prediction_digest(reference_rows),
        "candidate_prediction_sha256": _prediction_digest(candidate_rows),
    }


def _task_budget_summary(task_state: Mapping[str, Any], task: DecodeAuditTask) -> dict[str, Any]:
    reference_budget = max(task.budgets)
    budget_rows: dict[int, list[dict[str, Any]]] = {}
    budgets_summary: dict[str, Any] = {}
    for budget in task.budgets:
        budget_key = str(budget)
        budget_state = task_state["budgets"][budget_key]
        rows = _read_prediction_rows(Path(str(budget_state["predictions_path"])))
        budget_rows[budget] = rows
        budgets_summary[budget_key] = {
            "metrics": dict(budget_state["metrics"]),
            "generation_summary": dict(budget_state["generation_summary"]),
            "parse_success_rate": _parse_success_rate(task.condition, rows),
            "prediction_sha256": _prediction_digest(rows),
        }

    reference_rows = budget_rows[reference_budget]
    reference_comparisons = {}
    for budget in task.budgets:
        if budget == reference_budget:
            continue
        reference_comparisons[str(budget)] = _compare_prediction_rows(
            reference_rows=reference_rows,
            candidate_rows=budget_rows[budget],
        )
    return {
        "task_name": task.name,
        "condition": task.condition,
        "reference_budget": reference_budget,
        "budgets": budgets_summary,
        "reference_comparisons": reference_comparisons,
    }


def run_decode_audit(config: DecodeAuditConfig) -> dict[str, Any]:
    run_paths = build_run_paths(config.app)
    resume_checkpoint_dir = _resolve_resume_checkpoint_path(config, run_paths)
    _assert_run_dir_available(run_paths, resume_checkpoint_dir=resume_checkpoint_dir)
    _ensure_run_layout(run_paths, config.app.source_path)
    write_environment_summary(run_paths.environment)

    total_units = sum(len(task.budgets) for task in config.tasks)
    plan = SweepPlan.from_counts(total_units=total_units, unit_label="decode_budget_evals")
    runtime = SweepRuntimeState()
    state = _initial_state(config)
    if resume_checkpoint_dir is not None:
        runtime, state = load_sweep_checkpoint(resume_checkpoint_dir)
        if runtime.units_completed > plan.total_units:
            raise ValueError(
                "Resume checkpoint is ahead of the current decode-audit plan: "
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
            message="Decode audit state is ready; loading adapters and cached eval features.",
        ),
    )

    try:
        for task in config.tasks:
            task_state = state["tasks"][task.name]
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
            task_state["feature_cache_hit"] = feature_bundle.cache_hit
            task_state["feature_cache_path"] = str(feature_bundle.cache_path)

            for budget in task.budgets:
                budget_key = str(budget)
                budget_state = task_state["budgets"][budget_key]
                if budget_state["status"] == "completed":
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
                        phase="decode_audit",
                        control_poll_seconds=config.control_poll_seconds,
                        extra_progress={"max_new_tokens": budget},
                    )
                    last_heartbeat_timestamp = time.monotonic()

                evaluation = evaluator.evaluate_adapter(
                    adapter_name=task.adapter_name,
                    adapter_dir=task.adapter_dir,
                    eval_features=feature_bundle.features,
                    condition=task.condition,
                    max_length=config.max_length,
                    max_new_tokens=budget,
                    batch_size=task.batch_size,
                    sort_by_length=config.sort_eval_by_length,
                )
                budget_dir = run_paths.run_dir / "tasks" / task.name / f"budget_{budget}"
                metrics_payload = {
                    "condition": task.condition,
                    "max_new_tokens": budget,
                    "metrics": evaluation.metrics,
                    "generation_summary": evaluation.generation_summary,
                    "feature_cache_hit": feature_bundle.cache_hit,
                    "feature_cache_path": str(feature_bundle.cache_path),
                    "feature_cache_key": feature_bundle.descriptor.cache_key(),
                    "feature_cache_descriptor": feature_bundle.descriptor.as_dict(),
                }
                write_json(budget_dir / "metrics.json", metrics_payload)
                write_jsonl(budget_dir / "predictions.jsonl", evaluation.predictions)

                budget_state["status"] = "completed"
                budget_state["metrics"] = evaluation.metrics
                budget_state["metrics_path"] = str(budget_dir / "metrics.json")
                budget_state["predictions_path"] = str(budget_dir / "predictions.jsonl")
                budget_state["generation_summary"] = evaluation.generation_summary
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
                        phase="decode_audit",
                        extra_progress={"max_new_tokens": budget},
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
                            phase="decode_audit",
                            extra_progress={"max_new_tokens": budget},
                        ),
                    )
                    last_heartbeat_timestamp = time.monotonic()

            comparison = _task_budget_summary(task_state, task)
            comparison_path = run_paths.run_dir / "tasks" / task.name / "comparison.json"
            write_json(comparison_path, comparison)
            task_state["comparison_path"] = str(comparison_path)
            task_state["status"] = "completed"
            write_json(run_paths.run_dir / "state.json", state)

        runtime.accumulated_runtime_seconds = elapsed_runtime_seconds()
        metrics = {
            "status": "completed",
            "tasks": {
                task.name: json.loads(Path(str(state["tasks"][task.name]["comparison_path"])).read_text(encoding="utf-8"))
                for task in config.tasks
            },
            "eval_dataset_path": str(config.eval_dataset_path),
            "elapsed_runtime_seconds": runtime.accumulated_runtime_seconds,
        }
        run_manifest = {
            "stage": "decode_audit",
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
                    "budgets": list(task.budgets),
                    "batch_size": task.batch_size,
                }
                for task in config.tasks
            ],
            "metrics_path": str(run_paths.metrics),
        }
        write_json(run_paths.metrics, metrics)
        write_json(run_paths.run_manifest, run_manifest)
        write_json(
            run_paths.heartbeat,
            build_sweep_heartbeat(
                status="completed",
                plan=plan,
                runtime=runtime,
                heartbeat_path=run_paths.heartbeat,
                control_dir=run_paths.control_dir,
                elapsed_runtime_seconds=runtime.accumulated_runtime_seconds,
                message="Decode audit completed successfully.",
            ),
        )
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


def load_decode_audit_config(
    path: str | Path,
    *,
    resume_latest: bool = False,
    resume_checkpoint_path: Path | None = None,
) -> DecodeAuditConfig:
    app = load_app_config(path)
    return DecodeAuditConfig.from_app_config(
        app,
        resume_latest=resume_latest,
        resume_checkpoint_path=resume_checkpoint_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit decode budgets for saved adapters on a validation split.")
    parser.add_argument("--config", required=True, help="Path to a decode-audit YAML config.")
    parser.add_argument("--json", action="store_true", help="Print a JSON summary.")
    parser.add_argument("--resume-latest", action="store_true", help="Resume from the latest saved checkpoint.")
    parser.add_argument("--resume-checkpoint", help="Resume from an explicit checkpoint path.")
    args = parser.parse_args()

    config = load_decode_audit_config(
        Path(args.config),
        resume_latest=args.resume_latest,
        resume_checkpoint_path=Path(args.resume_checkpoint) if args.resume_checkpoint else None,
    )
    summary = run_decode_audit(config)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        print(f"Wrote decode-audit outputs to {summary['run_dir']}.")
        print(f"Metrics: {summary['metrics_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
