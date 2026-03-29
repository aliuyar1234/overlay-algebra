from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping

from .train_runtime import prune_old_checkpoints, write_heartbeat


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class SweepPlan:
    total_units: int
    unit_label: str

    @classmethod
    def from_counts(cls, *, total_units: int, unit_label: str) -> "SweepPlan":
        if total_units <= 0:
            raise ValueError("total_units must be positive")
        if not unit_label.strip():
            raise ValueError("unit_label must not be empty")
        return cls(total_units=total_units, unit_label=unit_label.strip())


@dataclass(slots=True)
class SweepRuntimeState:
    units_completed: int = 0
    accumulated_runtime_seconds: float = 0.0
    latest_checkpoint_dir: str | None = None
    latest_checkpoint_units_completed: int = 0
    resumed_from_checkpoint: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "units_completed": self.units_completed,
            "accumulated_runtime_seconds": self.accumulated_runtime_seconds,
            "latest_checkpoint_dir": self.latest_checkpoint_dir,
            "latest_checkpoint_units_completed": self.latest_checkpoint_units_completed,
            "resumed_from_checkpoint": self.resumed_from_checkpoint,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SweepRuntimeState":
        return cls(
            units_completed=int(payload.get("units_completed", 0)),
            accumulated_runtime_seconds=float(payload.get("accumulated_runtime_seconds", 0.0)),
            latest_checkpoint_dir=(
                str(payload["latest_checkpoint_dir"]) if payload.get("latest_checkpoint_dir") is not None else None
            ),
            latest_checkpoint_units_completed=int(payload.get("latest_checkpoint_units_completed", 0)),
            resumed_from_checkpoint=(
                str(payload["resumed_from_checkpoint"]) if payload.get("resumed_from_checkpoint") is not None else None
            ),
        )


def _safe_rate(completed: int, elapsed_seconds: float) -> float | None:
    if completed <= 0 or elapsed_seconds <= 0.0:
        return None
    return completed / elapsed_seconds


def _safe_eta(remaining: int, rate: float | None) -> float | None:
    if rate is None or rate <= 0.0 or remaining <= 0:
        return None
    return remaining / rate


def build_sweep_heartbeat(
    *,
    status: str,
    plan: SweepPlan,
    runtime: SweepRuntimeState,
    heartbeat_path: Path,
    control_dir: Path,
    elapsed_runtime_seconds: float | None = None,
    active_item: str | None = None,
    phase: str | None = None,
    message: str | None = None,
    extra_progress: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    effective_elapsed = (
        float(elapsed_runtime_seconds)
        if elapsed_runtime_seconds is not None
        else float(runtime.accumulated_runtime_seconds)
    )
    remaining_units = max(plan.total_units - runtime.units_completed, 0)
    rate = _safe_rate(runtime.units_completed, effective_elapsed)

    return {
        "updated_at": _utc_now_iso(),
        "status": status,
        "message": message,
        "active_item": active_item,
        "phase": phase,
        "progress": {
            "units_completed": runtime.units_completed,
            "units_total": plan.total_units,
            "units_remaining": remaining_units,
            "unit_label": plan.unit_label,
            "extra": dict(extra_progress or {}),
        },
        "timing": {
            "elapsed_runtime_seconds": effective_elapsed,
            "units_per_second": rate,
            "eta_seconds": _safe_eta(remaining_units, rate),
        },
        "checkpoint": {
            "latest_checkpoint_dir": runtime.latest_checkpoint_dir,
            "latest_checkpoint_units_completed": runtime.latest_checkpoint_units_completed,
            "resumed_from_checkpoint": runtime.resumed_from_checkpoint,
        },
        "paths": {
            "heartbeat": str(heartbeat_path),
            "control_dir": str(control_dir),
            "pause_request": str(control_dir / "pause.request"),
        },
    }


def publish_sweep_heartbeat(
    *,
    status: str,
    plan: SweepPlan,
    runtime: SweepRuntimeState,
    heartbeat_path: Path,
    control_dir: Path,
    elapsed_runtime_seconds: float | None = None,
    active_item: str | None = None,
    phase: str | None = None,
    message: str | None = None,
    extra_progress: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    heartbeat = build_sweep_heartbeat(
        status=status,
        plan=plan,
        runtime=runtime,
        heartbeat_path=heartbeat_path,
        control_dir=control_dir,
        elapsed_runtime_seconds=elapsed_runtime_seconds,
        active_item=active_item,
        phase=phase,
        message=message,
        extra_progress=extra_progress,
    )
    write_heartbeat(heartbeat_path, heartbeat)
    return heartbeat


def _checkpoint_dir_for(runtime: SweepRuntimeState, checkpoints_dir: Path) -> Path:
    return checkpoints_dir / f"unit_{runtime.units_completed:07d}"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_sweep_checkpoint(
    *,
    runtime: SweepRuntimeState,
    checkpoints_dir: Path,
    latest_checkpoint_path: Path,
    state_payload: Mapping[str, Any],
    keep_last: int,
    heartbeat_path: Path | None = None,
    control_dir: Path | None = None,
    plan: SweepPlan | None = None,
    elapsed_runtime_seconds: float | None = None,
    status: str | None = None,
    active_item: str | None = None,
    phase: str | None = None,
    message: str | None = None,
    extra_progress: Mapping[str, Any] | None = None,
) -> Path:
    checkpoint_dir = _checkpoint_dir_for(runtime, checkpoints_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_json(checkpoint_dir / "runtime_state.json", runtime.as_dict())
    _write_json(checkpoint_dir / "state.json", state_payload)
    _write_json(
        checkpoint_dir / "metadata.json",
        {
            "created_at": time.time(),
            "checkpoint_dir": str(checkpoint_dir),
            "units_completed": runtime.units_completed,
        },
    )

    runtime.latest_checkpoint_dir = str(checkpoint_dir)
    runtime.latest_checkpoint_units_completed = runtime.units_completed
    _write_json(
        latest_checkpoint_path,
        {
            "updated_at": time.time(),
            "checkpoint_dir": str(checkpoint_dir),
            "runtime_state_path": str(checkpoint_dir / "runtime_state.json"),
            "state_path": str(checkpoint_dir / "state.json"),
            "units_completed": runtime.units_completed,
        },
    )
    if any(value is not None for value in (heartbeat_path, control_dir, plan, status)):
        if heartbeat_path is None or control_dir is None or plan is None or status is None:
            raise ValueError(
                "heartbeat_path, control_dir, plan, and status must all be provided when publishing checkpoint progress."
            )
        publish_sweep_heartbeat(
            status=status,
            plan=plan,
            runtime=runtime,
            heartbeat_path=heartbeat_path,
            control_dir=control_dir,
            elapsed_runtime_seconds=elapsed_runtime_seconds,
            active_item=active_item,
            phase=phase,
            message=message,
            extra_progress=extra_progress,
        )
    prune_old_checkpoints(checkpoints_dir, keep_last=max(1, keep_last))
    return checkpoint_dir


def resolve_sweep_checkpoint_dir(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        return candidate
    if candidate.name in {"latest.json", "runtime_state.json", "state.json"}:
        if candidate.name == "latest.json":
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            return Path(payload["checkpoint_dir"])
        return candidate.parent
    raise ValueError(f"Cannot resolve sweep checkpoint directory from {candidate}")


def load_sweep_checkpoint(checkpoint_dir: str | Path) -> tuple[SweepRuntimeState, dict[str, Any]]:
    resolved = resolve_sweep_checkpoint_dir(checkpoint_dir)
    runtime_payload = json.loads((resolved / "runtime_state.json").read_text(encoding="utf-8"))
    state_payload = json.loads((resolved / "state.json").read_text(encoding="utf-8"))
    runtime = SweepRuntimeState.from_mapping(runtime_payload)
    runtime.latest_checkpoint_dir = str(resolved)
    runtime.latest_checkpoint_units_completed = runtime.units_completed
    runtime.resumed_from_checkpoint = str(resolved)
    return runtime, state_payload


def maybe_pause_sweep(
    *,
    pause_request_path: Path,
    heartbeat_path: Path,
    control_dir: Path,
    plan: SweepPlan,
    runtime: SweepRuntimeState,
    elapsed_runtime_seconds: float,
    active_item: str | None,
    phase: str | None,
    control_poll_seconds: float,
    extra_progress: Mapping[str, Any] | None = None,
) -> float:
    if not pause_request_path.exists():
        return 0.0

    paused_payload = publish_sweep_heartbeat(
        status="paused",
        plan=plan,
        runtime=runtime,
        heartbeat_path=heartbeat_path,
        control_dir=control_dir,
        elapsed_runtime_seconds=elapsed_runtime_seconds,
        active_item=active_item,
        phase=phase,
        message="Sweep is paused because control/pause.request exists. Remove the file to resume.",
        extra_progress=extra_progress,
    )
    pause_started = time.monotonic()
    while pause_request_path.exists():
        time.sleep(max(control_poll_seconds, 0.1))
        write_heartbeat(heartbeat_path, paused_payload)
    return time.monotonic() - pause_started
