from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .artifacts import RunPaths, build_run_paths
from .config import AppConfig, load_app_config


REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_HEARTBEAT_STALE_SECONDS = 10 * 60
RECENT_RUNNING_MAX_AGE_SECONDS = 6 * 60 * 60
PAPER_DEFAULT_SCOPE = "jq_first_reduced_scope"

_RESUMABLE_COMMANDS = {
    "overlay_algebra.train.fit",
    "overlay_algebra.analysis.decode_audit",
    "overlay_algebra.analysis.residualize",
    "overlay_algebra.eval.bundle",
    "overlay_algebra.analysis.compose",
    "overlay_algebra.analysis.ablate",
}

_TRAIN_COMMAND = "overlay_algebra.train.fit"
_EVAL_COMMAND = "overlay_algebra.eval.run"
_RESIDUALIZE_COMMAND = "overlay_algebra.analysis.residualize"

_PAPER_PRIORITY_STEP_KEYS = (
    "scaffold",
    "full_j",
    "full_c",
    "full_q",
    "decode_audit_q_pre",
    "residualize_j",
    "residualize_q",
    "prompt_only_j_eval",
    "direct_j_eval",
    "soar_j_eval",
    "report_j",
    "prompt_only_q_eval",
    "direct_q_eval",
    "soar_q_eval",
    "report_q",
    "full_single_overlay",
    "claim_checks",
)

_DEPRIORITIZED_STEP_KEYS = (
    "residualize_c",
    "prompt_only_c_eval",
    "direct_c_eval",
    "soar_c_eval",
    "report_c",
)


@dataclass(frozen=True, slots=True)
class M4StepSpec:
    key: str
    label: str
    config_path: Path


@dataclass(frozen=True, slots=True)
class M4StepState:
    spec: M4StepSpec
    command: str
    artifact_kind: str
    artifact_path: Path
    status: str
    percent_complete: float | None = None
    progress_completed: int | None = None
    progress_total: int | None = None
    progress_unit: str | None = None
    eta_seconds: float | None = None
    latest_checkpoint: Path | None = None
    updated_at: str | None = None
    stale: bool = False
    resumable: bool = False
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class RunningRun:
    run_dir: Path
    heartbeat_path: Path
    updated_at: str | None
    age_seconds: float | None
    stale: bool
    progress_completed: int | None = None
    progress_total: int | None = None
    progress_unit: str | None = None
    eta_seconds: float | None = None
    latest_checkpoint: Path | None = None
    detail: str | None = None


def _paper_priority_states(step_states: Sequence[M4StepState]) -> list[M4StepState]:
    by_key = {state.spec.key: state for state in step_states}
    prioritized: list[M4StepState] = []
    for key in _PAPER_PRIORITY_STEP_KEYS:
        state = by_key.get(key)
        if state is not None:
            prioritized.append(state)
    if prioritized:
        return prioritized
    return list(step_states)


def _paper_priority_specs(step_specs: Sequence[M4StepSpec]) -> list[M4StepSpec]:
    by_key = {spec.key: spec for spec in step_specs}
    prioritized: list[M4StepSpec] = []
    for key in _PAPER_PRIORITY_STEP_KEYS:
        spec = by_key.get(key)
        if spec is not None:
            prioritized.append(spec)
    if prioritized:
        return prioritized
    return list(step_specs)


def _active_tree_states(step_states: Sequence[M4StepState]) -> list[M4StepState]:
    active = [state for state in step_states if state.spec.key not in _DEPRIORITIZED_STEP_KEYS]
    if active:
        return active
    return list(step_states)


def _deprioritized_states(step_states: Sequence[M4StepState]) -> list[M4StepState]:
    return [state for state in step_states if state.spec.key in _DEPRIORITIZED_STEP_KEYS and state.status != "completed"]


def _pick_next_step(step_states: Sequence[M4StepState]) -> M4StepState | None:
    running_steps = [state for state in step_states if state.status == "running"]
    if running_steps:
        return next((state for state in step_states if state.status not in {"completed", "running"}), None)
    return next((state for state in step_states if state.status != "completed"), None)


def _resolve_repo_path(repo_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def _absolute_run_paths(repo_root: Path, config: AppConfig) -> RunPaths:
    raw = build_run_paths(config)
    return RunPaths(
        run_dir=_resolve_repo_path(repo_root, raw.run_dir),
        adapter_dir=_resolve_repo_path(repo_root, raw.adapter_dir),
        checkpoints_dir=_resolve_repo_path(repo_root, raw.checkpoints_dir),
        latest_checkpoint=_resolve_repo_path(repo_root, raw.latest_checkpoint),
        heartbeat=_resolve_repo_path(repo_root, raw.heartbeat),
        control_dir=_resolve_repo_path(repo_root, raw.control_dir),
        pause_request=_resolve_repo_path(repo_root, raw.pause_request),
        resolved_config=_resolve_repo_path(repo_root, raw.resolved_config),
        run_manifest=_resolve_repo_path(repo_root, raw.run_manifest),
        metrics=_resolve_repo_path(repo_root, raw.metrics),
        predictions=_resolve_repo_path(repo_root, raw.predictions),
        stdout=_resolve_repo_path(repo_root, raw.stdout),
        environment=_resolve_repo_path(repo_root, raw.environment),
        tuning_manifest=_resolve_repo_path(repo_root, raw.tuning_manifest),
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.fromtimestamp(float(text), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_updated_at(value: Any) -> str | None:
    parsed = _parse_datetime(value)
    if parsed is None:
        return None
    return parsed.isoformat()


def _newest_updated_at(*values: Any) -> str | None:
    parsed_values = [parsed for value in values if (parsed := _parse_datetime(value)) is not None]
    if not parsed_values:
        return None
    return max(parsed_values).isoformat()


def _age_seconds(updated_at: str | None) -> float | None:
    parsed = _parse_datetime(updated_at)
    if parsed is None:
        return None
    return max((datetime.now(timezone.utc) - parsed).total_seconds(), 0.0)


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    total = int(round(max(seconds, 0.0)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _is_recent_running_heartbeat(heartbeat: Mapping[str, Any], *, stale_seconds: float) -> tuple[bool, bool]:
    if str(heartbeat.get("status")) != "running":
        return False, False
    age = _age_seconds(heartbeat.get("updated_at"))
    stale = age is not None and age > stale_seconds
    return not stale, stale


def _heartbeat_progress(heartbeat: Mapping[str, Any]) -> tuple[int | None, int | None, str | None]:
    progress = heartbeat.get("progress")
    if not isinstance(progress, Mapping):
        return None, None, None
    if "optimizer_steps_total" in progress:
        return (
            int(progress.get("optimizer_steps_completed", 0)),
            int(progress["optimizer_steps_total"]),
            "optimizer_steps",
        )
    if "units_total" in progress:
        return (
            int(progress.get("units_completed", 0)),
            int(progress["units_total"]),
            str(progress.get("unit_label", "units")),
        )
    return None, None, None


def _percent_complete(completed: int | None, total: int | None) -> float | None:
    if completed is None or total is None or total <= 0:
        return None
    return (completed / total) * 100.0


def _checkpoint_progress(latest_checkpoint: Path | None) -> tuple[int | None, str | None]:
    if latest_checkpoint is None:
        return None, None
    payload = _read_json(latest_checkpoint)
    if not isinstance(payload, Mapping):
        return None, None
    units_completed_raw = payload.get("units_completed")
    units_completed = None
    if isinstance(units_completed_raw, (int, float)) and not isinstance(units_completed_raw, bool):
        units_completed = int(units_completed_raw)
    return units_completed, _normalize_updated_at(payload.get("updated_at"))


def _file_updated_at(path: Path) -> str | None:
    try:
        return _normalize_updated_at(path.stat().st_mtime)
    except OSError:
        return None


def _recent_run_activity(run_dir: Path) -> tuple[str | None, str | None]:
    newest_updated_at: str | None = None
    newest_relative_path: str | None = None
    candidate_roots = (
        run_dir / "checkpoints",
        run_dir / "compositions",
        run_dir / "delta_cache",
        run_dir / "adapter",
        run_dir / "state.json",
        run_dir / "metrics.json",
        run_dir / "run_manifest.json",
        run_dir / "tuning_manifest.json",
    )
    for root in candidate_roots:
        if not root.exists():
            continue
        paths = root.rglob("*") if root.is_dir() else (root,)
        for path in paths:
            if not path.is_file():
                continue
            updated_at = _file_updated_at(path)
            if updated_at is None:
                continue
            if newest_updated_at is None or _parse_datetime(updated_at) > _parse_datetime(newest_updated_at):
                newest_updated_at = updated_at
                try:
                    newest_relative_path = str(path.relative_to(run_dir))
                except ValueError:
                    newest_relative_path = str(path)
    return newest_updated_at, newest_relative_path


def _run_completion_ok(command: str, run_paths: RunPaths) -> bool:
    required = [run_paths.metrics, run_paths.run_manifest]
    if command in {_TRAIN_COMMAND, _RESIDUALIZE_COMMAND}:
        required.append(run_paths.adapter_dir)
    if command == _RESIDUALIZE_COMMAND:
        required.extend([run_paths.tuning_manifest, run_paths.predictions])
    if command == _EVAL_COMMAND:
        required.append(run_paths.predictions)
    return all(path.exists() for path in required)


def _report_completion_ok(output_dir: Path) -> bool:
    return (output_dir / "metrics.json").exists() and (output_dir / "report.md").exists()


def _report_partial(output_dir: Path) -> bool:
    if not output_dir.exists():
        return False
    return any(output_dir.iterdir())


def _step_state_for_run(spec: M4StepSpec, config: AppConfig, run_paths: RunPaths) -> M4StepState:
    heartbeat = _read_json(run_paths.heartbeat)
    latest_checkpoint = run_paths.latest_checkpoint if run_paths.latest_checkpoint.exists() else None
    heartbeat_updated_at = (
        _normalize_updated_at(heartbeat.get("updated_at"))
        if isinstance(heartbeat, Mapping)
        else None
    )
    checkpoint_completed, checkpoint_updated_at = _checkpoint_progress(latest_checkpoint)
    updated_at = _newest_updated_at(heartbeat_updated_at, checkpoint_updated_at)

    if _run_completion_ok(config.command, run_paths):
        return M4StepState(
            spec=spec,
            command=config.command,
            artifact_kind="run",
            artifact_path=run_paths.run_dir,
            status="completed",
            latest_checkpoint=latest_checkpoint,
            updated_at=updated_at,
            resumable=config.command in _RESUMABLE_COMMANDS,
        )

    if isinstance(heartbeat, Mapping):
        is_recent, is_stale = _is_recent_running_heartbeat(heartbeat, stale_seconds=ACTIVE_HEARTBEAT_STALE_SECONDS)
        completed, total, unit = _heartbeat_progress(heartbeat)
        if checkpoint_completed is not None and (completed is None or checkpoint_completed > completed):
            completed = checkpoint_completed
        eta_seconds = heartbeat.get("timing", {}).get("eta_seconds") if isinstance(heartbeat.get("timing"), Mapping) else None
        detail = str(heartbeat.get("message")) if heartbeat.get("message") else None
        if is_recent:
            return M4StepState(
                spec=spec,
                command=config.command,
                artifact_kind="run",
                artifact_path=run_paths.run_dir,
                status="running",
                percent_complete=_percent_complete(completed, total),
                progress_completed=completed,
                progress_total=total,
                progress_unit=unit,
                eta_seconds=float(eta_seconds) if isinstance(eta_seconds, (int, float)) else None,
                latest_checkpoint=latest_checkpoint,
                updated_at=updated_at,
                stale=False,
                resumable=config.command in _RESUMABLE_COMMANDS,
                detail=detail,
            )
        if is_stale:
            return M4StepState(
                spec=spec,
                command=config.command,
                artifact_kind="run",
                artifact_path=run_paths.run_dir,
                status="partial",
                percent_complete=_percent_complete(completed, total),
                progress_completed=completed,
                progress_total=total,
                progress_unit=unit,
                eta_seconds=float(eta_seconds) if isinstance(eta_seconds, (int, float)) else None,
                latest_checkpoint=latest_checkpoint,
                updated_at=updated_at,
                stale=True,
                resumable=config.command in _RESUMABLE_COMMANDS,
                detail="stale running heartbeat",
            )
        if str(heartbeat.get("status")) == "failed":
            return M4StepState(
                spec=spec,
                command=config.command,
                artifact_kind="run",
                artifact_path=run_paths.run_dir,
                status="failed",
                percent_complete=_percent_complete(completed, total),
                progress_completed=completed,
                progress_total=total,
                progress_unit=unit,
                latest_checkpoint=latest_checkpoint,
                updated_at=updated_at,
                stale=False,
                resumable=config.command in _RESUMABLE_COMMANDS,
                detail=str(heartbeat.get("message")) if heartbeat.get("message") else None,
            )

    if latest_checkpoint is not None or (run_paths.run_dir.exists() and any(run_paths.run_dir.iterdir())):
        heartbeat_completed, heartbeat_total, heartbeat_unit = (
            _heartbeat_progress(heartbeat) if isinstance(heartbeat, Mapping) else (None, None, None)
        )
        completed = heartbeat_completed
        if checkpoint_completed is not None and (completed is None or checkpoint_completed > completed):
            completed = checkpoint_completed
        return M4StepState(
            spec=spec,
            command=config.command,
            artifact_kind="run",
            artifact_path=run_paths.run_dir,
            status="partial",
            percent_complete=_percent_complete(completed, heartbeat_total),
            progress_completed=completed,
            progress_total=heartbeat_total,
            progress_unit=heartbeat_unit,
            latest_checkpoint=latest_checkpoint,
            updated_at=updated_at,
            resumable=config.command in _RESUMABLE_COMMANDS,
        )

    return M4StepState(
        spec=spec,
        command=config.command,
        artifact_kind="run",
        artifact_path=run_paths.run_dir,
        status="pending",
        resumable=config.command in _RESUMABLE_COMMANDS,
    )


def _step_state_for_report(spec: M4StepSpec, config: AppConfig, repo_root: Path) -> M4StepState:
    output_dir = _resolve_repo_path(repo_root, config.paths["output_dir"])
    if _report_completion_ok(output_dir):
        status = "completed"
    elif _report_partial(output_dir):
        status = "partial"
    else:
        status = "pending"
    return M4StepState(
        spec=spec,
        command=config.command,
        artifact_kind="report",
        artifact_path=output_dir,
        status=status,
    )


def evaluate_step_state(spec: M4StepSpec, *, repo_root: Path = REPO_ROOT) -> M4StepState:
    config = load_app_config(spec.config_path)
    if config.milestone != "M4":
        raise ValueError(f"M4 step {spec.key!r} points to non-M4 config {spec.config_path}")
    if config.command.startswith("overlay_algebra.reports."):
        return _step_state_for_report(spec, config, repo_root)
    return _step_state_for_run(spec, config, _absolute_run_paths(repo_root, config))


def build_default_m4_steps(repo_root: Path = REPO_ROOT) -> list[M4StepSpec]:
    def step(key: str, label: str, relative_config: str) -> M4StepSpec:
        return M4StepSpec(key=key, label=label, config_path=repo_root / relative_config)

    return [
        step("scaffold", "Train scaffold", "configs/train/scaffold.yaml"),
        step("full_j", "Train direct J", "configs/train/full_J.yaml"),
        step("full_c", "Train direct C", "configs/train/full_C.yaml"),
        step("full_q", "Train direct Q", "configs/train/full_Q.yaml"),
        step(
            "decode_audit_q_pre",
            "Run pre-residualization Q decode audit",
            "configs/analysis/decode_audit_q_pre_residualize.yaml",
        ),
        step("residualize_j", "Residualize J", "configs/analysis/full_residuals_J.yaml"),
        step("residualize_c", "Residualize C", "configs/analysis/full_residuals_C.yaml"),
        step("residualize_q", "Residualize Q", "configs/analysis/full_residuals_Q.yaml"),
        step("prompt_only_j_eval", "Eval prompt-only J", "configs/eval/full_prompt_only_J.yaml"),
        step("direct_j_eval", "Eval direct J", "configs/eval/full_direct_J.yaml"),
        step("soar_j_eval", "Eval SOAR J", "configs/eval/full_soar_J.yaml"),
        step("report_j", "Build J recovery report", "configs/reports/full_single_overlay_recovery_J.yaml"),
        step("prompt_only_c_eval", "Eval prompt-only C", "configs/eval/full_prompt_only_C.yaml"),
        step("direct_c_eval", "Eval direct C", "configs/eval/full_direct_C.yaml"),
        step("soar_c_eval", "Eval SOAR C", "configs/eval/full_soar_C.yaml"),
        step("report_c", "Build C recovery report", "configs/reports/full_single_overlay_recovery_C.yaml"),
        step("prompt_only_q_eval", "Eval prompt-only Q", "configs/eval/full_prompt_only_Q.yaml"),
        step("direct_q_eval", "Eval direct Q", "configs/eval/full_direct_Q.yaml"),
        step("soar_q_eval", "Eval SOAR Q", "configs/eval/full_soar_Q.yaml"),
        step("report_q", "Build Q recovery report", "configs/reports/full_single_overlay_recovery_Q.yaml"),
        step("full_single_overlay", "Build full M4 overlay study report", "configs/reports/full_single_overlay_JQ.yaml"),
        step("claim_checks", "Build M4 claim checks", "configs/reports/c1_c4_c5.yaml"),
    ]


def list_recent_running_runs(
    repo_root: Path = REPO_ROOT,
    *,
    stale_seconds: float = ACTIVE_HEARTBEAT_STALE_SECONDS,
) -> list[RunningRun]:
    runs_root = repo_root / "runs"
    if not runs_root.exists():
        return []
    records: list[RunningRun] = []
    for heartbeat_path in runs_root.glob("*/heartbeat.json"):
        heartbeat = _read_json(heartbeat_path)
        if not isinstance(heartbeat, Mapping):
            continue
        is_recent, is_stale = _is_recent_running_heartbeat(heartbeat, stale_seconds=stale_seconds)
        latest_checkpoint = heartbeat_path.parent / "checkpoints" / "latest.json"
        latest_checkpoint = latest_checkpoint if latest_checkpoint.exists() else None
        checkpoint_completed, checkpoint_updated_at = _checkpoint_progress(latest_checkpoint)
        heartbeat_updated_at = _normalize_updated_at(heartbeat.get("updated_at"))
        activity_updated_at = None
        activity_detail = None
        if is_stale:
            activity_updated_at, activity_path = _recent_run_activity(heartbeat_path.parent)
            if activity_path is not None:
                activity_detail = f"recent filesystem activity at {activity_path}"
        updated_at = _newest_updated_at(heartbeat_updated_at, checkpoint_updated_at, activity_updated_at)
        age_seconds = _age_seconds(updated_at)
        if age_seconds is not None and age_seconds > RECENT_RUNNING_MAX_AGE_SECONDS:
            continue
        is_stale = age_seconds is not None and age_seconds > stale_seconds
        is_recent = not is_stale
        if not is_recent and not is_stale:
            continue
        progress_completed, progress_total, progress_unit = _heartbeat_progress(heartbeat)
        if checkpoint_completed is not None and (progress_completed is None or checkpoint_completed > progress_completed):
            progress_completed = checkpoint_completed
        eta_seconds = heartbeat.get("timing", {}).get("eta_seconds") if isinstance(heartbeat.get("timing"), Mapping) else None
        detail = str(heartbeat.get("message")) if heartbeat.get("message") else activity_detail
        records.append(
            RunningRun(
                run_dir=heartbeat_path.parent,
                heartbeat_path=heartbeat_path,
                updated_at=updated_at,
                age_seconds=age_seconds,
                stale=is_stale,
                progress_completed=progress_completed,
                progress_total=progress_total,
                progress_unit=progress_unit,
                eta_seconds=float(eta_seconds) if isinstance(eta_seconds, (int, float)) else None,
                latest_checkpoint=latest_checkpoint,
                detail=detail,
            )
        )
    records.sort(key=lambda item: item.updated_at or "", reverse=True)
    return records


def collect_m4_status(
    *,
    repo_root: Path = REPO_ROOT,
    steps: Sequence[M4StepSpec] | None = None,
) -> dict[str, Any]:
    step_specs = list(steps) if steps is not None else build_default_m4_steps(repo_root)
    step_states = [evaluate_step_state(spec, repo_root=repo_root) for spec in step_specs]
    paper_step_states = _paper_priority_states(step_states)
    active_tree_step_states = _active_tree_states(step_states)
    deprioritized_step_states = _deprioritized_states(step_states)
    completed_steps = [state for state in step_states if state.status == "completed"]
    paper_completed_steps = [state for state in paper_step_states if state.status == "completed"]
    running_steps = [state for state in step_states if state.status == "running"]
    failed_steps = [state for state in step_states if state.status == "failed"]
    partial_steps = [state for state in step_states if state.status == "partial"]
    next_step = _pick_next_step(paper_step_states)
    tree_next_step = _pick_next_step(active_tree_step_states)
    recent_runs = list_recent_running_runs(repo_root)
    step_run_dirs = {
        state.artifact_path.resolve()
        for state in step_states
        if state.artifact_kind == "run"
    }
    external_running_runs = [
        {
            "run_dir": str(item.run_dir),
            "updated_at": item.updated_at,
            "age_seconds": item.age_seconds,
            "stale": item.stale,
            "progress_completed": item.progress_completed,
            "progress_total": item.progress_total,
            "progress_unit": item.progress_unit,
            "eta_seconds": item.eta_seconds,
            "latest_checkpoint": str(item.latest_checkpoint) if item.latest_checkpoint else None,
            "detail": item.detail,
        }
        for item in recent_runs
        if item.run_dir.resolve() not in step_run_dirs
    ]

    if running_steps:
        overall_status = "running"
    elif failed_steps:
        overall_status = "failed"
    elif len(completed_steps) == len(step_states):
        overall_status = "completed"
    elif partial_steps:
        overall_status = "resume_available"
    else:
        overall_status = "ready"

    current_step = running_steps[0] if running_steps else None
    return {
        "status_scope": PAPER_DEFAULT_SCOPE,
        "overall_status": overall_status,
        "completed_steps": len(completed_steps),
        "total_steps": len(step_states),
        "paper_completed_steps": len(paper_completed_steps),
        "paper_total_steps": len(paper_step_states),
        "current_step": _step_state_payload(current_step) if current_step is not None else None,
        "next_step": {
            "key": next_step.spec.key,
            "label": next_step.spec.label,
            "status": next_step.status,
        }
        if next_step is not None
        else None,
        "tree_next_step": {
            "key": tree_next_step.spec.key,
            "label": tree_next_step.spec.label,
            "status": tree_next_step.status,
        }
        if tree_next_step is not None
        else None,
        "remaining_steps": [
            {"key": state.spec.key, "label": state.spec.label, "status": state.status}
            for state in paper_step_states
            if state.status != "completed"
        ],
        "deprioritized_steps": [
            {"key": state.spec.key, "label": state.spec.label, "status": state.status}
            for state in deprioritized_step_states
        ],
        "tree_remaining_steps": [
            {"key": state.spec.key, "label": state.spec.label, "status": state.status}
            for state in active_tree_step_states
            if state.status != "completed"
        ],
        "steps": [_step_state_payload(state) for state in step_states],
        "external_running_runs": external_running_runs,
    }


def _step_state_payload(state: M4StepState) -> dict[str, Any]:
    return {
        "key": state.spec.key,
        "label": state.spec.label,
        "command": state.command,
        "artifact_kind": state.artifact_kind,
        "artifact_path": str(state.artifact_path),
        "status": state.status,
        "percent_complete": state.percent_complete,
        "progress_completed": state.progress_completed,
        "progress_total": state.progress_total,
        "progress_unit": state.progress_unit,
        "eta_seconds": state.eta_seconds,
        "latest_checkpoint": str(state.latest_checkpoint) if state.latest_checkpoint else None,
        "updated_at": state.updated_at,
        "stale": state.stale,
        "resumable": state.resumable,
        "detail": state.detail,
    }


def plan_m4_actions(
    *,
    repo_root: Path = REPO_ROOT,
    steps: Sequence[M4StepSpec] | None = None,
) -> list[dict[str, Any]]:
    if steps is not None:
        step_specs = list(steps)
    else:
        step_specs = _paper_priority_specs(build_default_m4_steps(repo_root))
    actions: list[dict[str, Any]] = []
    for state in [evaluate_step_state(spec, repo_root=repo_root) for spec in step_specs]:
        if state.status == "completed":
            action = "skip"
        elif state.status == "running":
            action = "wait"
        elif state.status == "partial" and state.resumable and state.latest_checkpoint is not None:
            action = "resume"
        else:
            action = "run"
        actions.append(
            {
                "key": state.spec.key,
                "label": state.spec.label,
                "status": state.status,
                "action": action,
                "artifact_path": str(state.artifact_path),
                "latest_checkpoint": str(state.latest_checkpoint) if state.latest_checkpoint else None,
            }
        )
    return actions


def _invoke_step(
    spec: M4StepSpec,
    *,
    repo_root: Path,
    state: M4StepState,
    runner: Callable[..., Any],
) -> None:
    config = load_app_config(spec.config_path)
    command = [
        sys.executable,
        "-m",
        config.command,
        "--config",
        str(spec.config_path),
    ]
    if state.status == "partial" and state.resumable and state.latest_checkpoint is not None:
        command.append("--resume-latest")
    runner(command, cwd=str(repo_root), check=True)


def run_m4_pipeline(
    *,
    repo_root: Path = REPO_ROOT,
    dry_run: bool = False,
    steps: Sequence[M4StepSpec] | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    if steps is not None:
        step_specs = list(steps)
    else:
        step_specs = _paper_priority_specs(build_default_m4_steps(repo_root))
    actions = plan_m4_actions(repo_root=repo_root, steps=step_specs)
    recent_running_runs = [
        item
        for item in list_recent_running_runs(repo_root)
        if not item.stale
    ]
    if dry_run:
        return {
            "status": "dry_run",
            "status_scope": PAPER_DEFAULT_SCOPE,
            "actions": actions,
            "active_runs": [
                {
                    "run_dir": str(item.run_dir),
                    "updated_at": item.updated_at,
                    "age_seconds": item.age_seconds,
                    "progress_completed": item.progress_completed,
                    "progress_total": item.progress_total,
                    "progress_unit": item.progress_unit,
                    "eta_seconds": item.eta_seconds,
                    "latest_checkpoint": str(item.latest_checkpoint) if item.latest_checkpoint else None,
                    "detail": item.detail,
                }
                for item in recent_running_runs
            ],
        }

    if recent_running_runs:
        return {
            "status": "running",
            "status_scope": PAPER_DEFAULT_SCOPE,
            "message": "A recent running heartbeat already exists; not starting another M4 step.",
            "active_runs": [
                {
                    "run_dir": str(item.run_dir),
                    "updated_at": item.updated_at,
                    "age_seconds": item.age_seconds,
                    "progress_completed": item.progress_completed,
                    "progress_total": item.progress_total,
                    "progress_unit": item.progress_unit,
                    "eta_seconds": item.eta_seconds,
                    "latest_checkpoint": str(item.latest_checkpoint) if item.latest_checkpoint else None,
                    "detail": item.detail,
                }
                for item in recent_running_runs
            ],
        }

    executed: list[str] = []
    skipped: list[str] = []
    for spec in step_specs:
        state = evaluate_step_state(spec, repo_root=repo_root)
        if state.status == "completed":
            skipped.append(spec.key)
            continue
        if state.status == "running":
            return {
                "status": "running",
                "status_scope": PAPER_DEFAULT_SCOPE,
                "message": f"Step {spec.key} is already running.",
                "executed_steps": executed,
                "skipped_steps": skipped,
            }
        _invoke_step(spec, repo_root=repo_root, state=state, runner=runner)
        executed.append(spec.key)

    final_status = collect_m4_status(repo_root=repo_root, steps=step_specs)
    return {
        "status": final_status["overall_status"],
        "status_scope": PAPER_DEFAULT_SCOPE,
        "executed_steps": executed,
        "skipped_steps": skipped,
        "completed_steps": final_status["completed_steps"],
        "total_steps": final_status["total_steps"],
        "paper_completed_steps": final_status["paper_completed_steps"],
        "paper_total_steps": final_status["paper_total_steps"],
        "next_step": final_status["next_step"],
    }


def _step_payload_by_key(summary: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    for item in summary.get("steps", []):
        if isinstance(item, Mapping) and str(item.get("key")) == key:
            return item
    return None


def watch_m4_pipeline(
    *,
    repo_root: Path = REPO_ROOT,
    steps: Sequence[M4StepSpec] | None = None,
    runner: Callable[..., Any] = subprocess.run,
    poll_seconds: float = 30.0,
    partial_timeout_seconds: float = 3 * 60 * 60,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if poll_seconds <= 0.0:
        raise ValueError("poll_seconds must be positive")
    if partial_timeout_seconds <= 0.0:
        raise ValueError("partial_timeout_seconds must be positive")

    step_specs = list(steps) if steps is not None else _paper_priority_specs(build_default_m4_steps(repo_root))
    waiting_on_partial_key: str | None = None
    waiting_on_partial_started_at: float | None = None

    while True:
        summary = collect_m4_status(repo_root=repo_root, steps=step_specs)
        overall_status = str(summary.get("overall_status", "ready"))
        if overall_status in {"completed", "failed"}:
            return {
                "status": overall_status,
                "status_scope": PAPER_DEFAULT_SCOPE,
                "message": f"M4 watch observed terminal status: {overall_status}.",
                "next_step": summary.get("next_step"),
                "completed_steps": summary.get("completed_steps"),
                "total_steps": summary.get("total_steps"),
                "paper_completed_steps": summary.get("paper_completed_steps"),
                "paper_total_steps": summary.get("paper_total_steps"),
            }

        current_step = summary.get("current_step")
        if isinstance(current_step, Mapping):
            waiting_on_partial_key = None
            waiting_on_partial_started_at = None
            sleep_fn(poll_seconds)
            continue

        next_step = summary.get("next_step")
        if not isinstance(next_step, Mapping):
            return {
                "status": overall_status,
                "status_scope": PAPER_DEFAULT_SCOPE,
                "message": "No next M4 step is available.",
                "next_step": None,
                "completed_steps": summary.get("completed_steps"),
                "total_steps": summary.get("total_steps"),
                "paper_completed_steps": summary.get("paper_completed_steps"),
                "paper_total_steps": summary.get("paper_total_steps"),
            }

        next_state = _step_payload_by_key(summary, str(next_step["key"]))
        if (
            isinstance(next_state, Mapping)
            and str(next_state.get("status")) == "partial"
            and not bool(next_state.get("resumable"))
        ):
            step_key = str(next_state["key"])
            if waiting_on_partial_key != step_key:
                waiting_on_partial_key = step_key
                waiting_on_partial_started_at = time.monotonic()
            elif waiting_on_partial_started_at is not None:
                waited_seconds = time.monotonic() - waiting_on_partial_started_at
                if waited_seconds >= partial_timeout_seconds:
                    return {
                        "status": "blocked",
                        "status_scope": PAPER_DEFAULT_SCOPE,
                        "message": (
                            f"Non-resumable partial step {step_key} did not complete within "
                            f"{int(partial_timeout_seconds)}s."
                        ),
                        "next_step": summary.get("next_step"),
                        "completed_steps": summary.get("completed_steps"),
                        "total_steps": summary.get("total_steps"),
                        "paper_completed_steps": summary.get("paper_completed_steps"),
                        "paper_total_steps": summary.get("paper_total_steps"),
                    }
            sleep_fn(poll_seconds)
            continue

        waiting_on_partial_key = None
        waiting_on_partial_started_at = None
        result = run_m4_pipeline(repo_root=repo_root, steps=step_specs, runner=runner)
        if result.get("status") == "running":
            sleep_fn(poll_seconds)
            continue
        return result


def _render_status_text(summary: Mapping[str, Any]) -> str:
    lines = [f"M4 status: {summary['overall_status']}"]
    scope = summary.get("status_scope")
    if scope:
        lines.append(f"Scope: {scope}")
    current = summary.get("current_step")
    if isinstance(current, Mapping):
        lines.append(f"Current: {current['label']} ({current['key']})")
        if current.get("percent_complete") is not None:
            lines.append(
                f"Progress: {current['progress_completed']}/{current['progress_total']} "
                f"{current['progress_unit']} ({float(current['percent_complete']):.1f}%)"
            )
        lines.append(f"ETA: {_format_duration(current.get('eta_seconds'))}")
        if current.get("latest_checkpoint"):
            lines.append(f"Latest checkpoint: {current['latest_checkpoint']}")
    lines.append(
        f"Completed steps: {summary['paper_completed_steps']}/{summary['paper_total_steps']} "
        f"(paper), {summary['completed_steps']}/{summary['total_steps']} (tree)"
    )
    next_step = summary.get("next_step")
    if isinstance(next_step, Mapping):
        lines.append(f"Paper next: {next_step['label']} ({next_step['key']})")
    tree_next = summary.get("tree_next_step")
    if isinstance(tree_next, Mapping) and (
        not isinstance(next_step, Mapping) or tree_next.get("key") != next_step.get("key")
    ):
        lines.append(f"Tree next: {tree_next['label']} ({tree_next['key']})")
    remaining = summary.get("remaining_steps", [])
    if remaining:
        lines.append("Paper remaining:")
        for item in remaining:
            lines.append(f"- {item['key']}: {item['label']} [{item['status']}]")
    deprioritized = summary.get("deprioritized_steps", [])
    if deprioritized:
        lines.append("Deprioritized:")
        for item in deprioritized:
            lines.append(f"- {item['key']}: {item['label']} [{item['status']}]")
    tree_remaining = summary.get("tree_remaining_steps", [])
    if tree_remaining and any(
        item not in remaining and item not in deprioritized for item in tree_remaining
    ):
        lines.append("Tree-only remaining:")
        for item in tree_remaining:
            if item in remaining or item in deprioritized:
                continue
            lines.append(f"- {item['key']}: {item['label']} [{item['status']}]")
    external = summary.get("external_running_runs", [])
    if external:
        lines.append("Other active runs:")
        for item in external:
            stale_suffix = " [stale]" if item.get("stale") else ""
            lines.append(f"- {item['run_dir']} (updated {item['updated_at']}){stale_suffix}")
            if item.get("progress_completed") is not None and item.get("progress_total") is not None:
                lines.append(
                    f"  progress: {item['progress_completed']}/{item['progress_total']} {item.get('progress_unit') or 'units'}"
                )
            if item.get("eta_seconds") is not None:
                lines.append(f"  ETA: {_format_duration(item['eta_seconds'])}")
            if item.get("latest_checkpoint"):
                lines.append(f"  latest checkpoint: {item['latest_checkpoint']}")
            if item.get("detail"):
                lines.append(f"  detail: {item['detail']}")
    return "\n".join(lines)


def _render_dry_run_text(summary: Mapping[str, Any]) -> str:
    scope = summary.get("status_scope")
    if scope:
        lines = [f"M4 dry run ({scope}):"]
    else:
        lines = ["M4 dry run:"]
    for action in summary["actions"]:
        lines.append(f"- {action['key']}: {action['action']} [{action['status']}]")
    active_runs = summary.get("active_runs", [])
    if active_runs:
        lines.append("Active runs:")
        for item in active_runs:
            lines.append(f"- {item['run_dir']} (updated {item['updated_at']})")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Small M4-only runner and status helper.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    run_parser = subparsers.add_parser("run", help="Run or continue the checked-in M4 sequence.")
    run_parser.add_argument("--dry-run", action="store_true", help="Show the planned M4 actions without running them.")

    watch_parser = subparsers.add_parser(
        "watch",
        help="Wait through the current M4 step if needed, then continue the checked-in sequence unattended.",
    )
    watch_parser.add_argument(
        "--poll-seconds",
        type=float,
        default=30.0,
        help="How often to poll M4 status while waiting for an active step to finish.",
    )
    watch_parser.add_argument(
        "--partial-timeout-seconds",
        type=float,
        default=3 * 60 * 60,
        help="Maximum time to wait for a non-resumable partial step before stopping.",
    )

    status_parser = subparsers.add_parser("status", help="Show M4 progress from saved artifacts.")
    status_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    args = parser.parse_args(argv)

    if args.subcommand == "run":
        summary = run_m4_pipeline(dry_run=args.dry_run)
        if args.dry_run:
            print(_render_dry_run_text(summary))
        else:
            if summary["status"] == "running":
                print(summary["message"])
                for item in summary.get("active_runs", []):
                    print(f"Active run: {item['run_dir']}")
            else:
                print(f"M4 run status: {summary['status']}")
                if summary.get("executed_steps"):
                    print("Executed steps:")
                    for key in summary["executed_steps"]:
                        print(f"- {key}")
                if summary.get("skipped_steps"):
                    print("Skipped steps:")
                    for key in summary["skipped_steps"]:
                        print(f"- {key}")
                if summary.get("next_step"):
                    next_step = summary["next_step"]
                    print(f"Next step: {next_step['key']} ({next_step['label']})")
        return 0

    if args.subcommand == "watch":
        summary = watch_m4_pipeline(
            poll_seconds=args.poll_seconds,
            partial_timeout_seconds=args.partial_timeout_seconds,
        )
        print(f"M4 watch status: {summary['status']}")
        message = summary.get("message")
        if message:
            print(message)
        if summary.get("next_step"):
            next_step = summary["next_step"]
            print(f"Next step: {next_step['key']} ({next_step['label']})")
        return 0 if summary["status"] not in {"failed", "blocked"} else 1

    summary = collect_m4_status()
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(_render_status_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
