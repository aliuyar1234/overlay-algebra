from __future__ import annotations

import json
from pathlib import Path

from overlay_algebra.sweep_runtime import (
    SweepPlan,
    SweepRuntimeState,
    build_sweep_heartbeat,
    load_sweep_checkpoint,
    resolve_sweep_checkpoint_dir,
    save_sweep_checkpoint,
)


def test_build_sweep_heartbeat_reports_eta() -> None:
    plan = SweepPlan.from_counts(total_units=20, unit_label="candidates")
    runtime = SweepRuntimeState(
        units_completed=5,
        accumulated_runtime_seconds=10.0,
        latest_checkpoint_dir="ckpt",
        latest_checkpoint_units_completed=5,
    )
    heartbeat = build_sweep_heartbeat(
        status="running",
        plan=plan,
        runtime=runtime,
        heartbeat_path=Path("heartbeat.json"),
        control_dir=Path("control"),
        elapsed_runtime_seconds=10.0,
        active_item="JC",
        phase="soar_tuning",
    )

    assert heartbeat["progress"]["units_remaining"] == 15
    assert heartbeat["timing"]["units_per_second"] == 0.5
    assert heartbeat["timing"]["eta_seconds"] == 30.0


def test_save_and_load_sweep_checkpoint_round_trip(tmp_path: Path) -> None:
    checkpoints_dir = tmp_path / "checkpoints"
    latest = tmp_path / "latest.json"
    runtime = SweepRuntimeState(units_completed=3)
    save_sweep_checkpoint(
        runtime=runtime,
        checkpoints_dir=checkpoints_dir,
        latest_checkpoint_path=latest,
        state_payload={"status": "partial", "rows": [1, 2, 3]},
        keep_last=2,
    )

    resolved = resolve_sweep_checkpoint_dir(latest)
    loaded_runtime, state = load_sweep_checkpoint(resolved)

    assert resolved.exists()
    assert loaded_runtime.units_completed == 3
    assert state["rows"] == [1, 2, 3]
    latest_payload = json.loads(latest.read_text(encoding="utf-8"))
    assert latest_payload["units_completed"] == 3


def test_save_sweep_checkpoint_can_publish_heartbeat(tmp_path: Path) -> None:
    checkpoints_dir = tmp_path / "checkpoints"
    latest = tmp_path / "latest.json"
    heartbeat_path = tmp_path / "heartbeat.json"
    runtime = SweepRuntimeState(units_completed=2)
    plan = SweepPlan.from_counts(total_units=5, unit_label="evaluation_units")

    save_sweep_checkpoint(
        runtime=runtime,
        checkpoints_dir=checkpoints_dir,
        latest_checkpoint_path=latest,
        state_payload={"status": "partial"},
        keep_last=2,
        heartbeat_path=heartbeat_path,
        control_dir=tmp_path / "control",
        plan=plan,
        elapsed_runtime_seconds=12.5,
        status="running",
        active_item="Q",
        phase="beta_tuning",
        extra_progress={"beta": 1.0},
    )

    heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert heartbeat["status"] == "running"
    assert heartbeat["active_item"] == "Q"
    assert heartbeat["phase"] == "beta_tuning"
    assert heartbeat["progress"]["units_completed"] == 2
    assert heartbeat["progress"]["units_total"] == 5
    assert heartbeat["progress"]["extra"] == {"beta": 1.0}
    assert heartbeat["checkpoint"]["latest_checkpoint_units_completed"] == 2
