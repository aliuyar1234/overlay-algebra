from __future__ import annotations

import json
from pathlib import Path

from overlay_algebra.train_runtime import (
    TrainingProgressPlan,
    TrainingRuntimeState,
    build_training_heartbeat,
    prune_old_checkpoints,
    resolve_checkpoint_dir,
    write_heartbeat,
)


def test_training_progress_plan_counts_micro_and_optimizer_steps() -> None:
    plan = TrainingProgressPlan.from_counts(
        train_examples=10,
        batch_size=4,
        gradient_accumulation_steps=2,
        train_epochs=3,
    )

    assert plan.batches_per_epoch == 3
    assert plan.total_micro_steps == 9
    assert plan.total_optimizer_steps == 5
    assert plan.total_examples == 30


def test_build_training_heartbeat_reports_eta(tmp_path: Path) -> None:
    plan = TrainingProgressPlan.from_counts(
        train_examples=100,
        batch_size=5,
        gradient_accumulation_steps=2,
        train_epochs=1,
    )
    runtime = TrainingRuntimeState(
        micro_steps_completed=10,
        optimizer_steps_completed=5,
        loss_sum=4.0,
        loss_count=10,
        last_loss=0.25,
        accumulated_training_seconds=20.0,
        latest_checkpoint_dir=str(tmp_path / "ckpt"),
        latest_checkpoint_micro_steps_completed=10,
        latest_checkpoint_optimizer_steps_completed=5,
    )
    heartbeat = build_training_heartbeat(
        condition="plain",
        status="running",
        plan=plan,
        runtime=runtime,
        train_examples_loaded=100,
        skipped_train_example_ids=["x1", "x2"],
        heartbeat_path=tmp_path / "heartbeat.json",
        control_dir=tmp_path / "control",
        current_epoch_index=1,
        current_batch_index_in_epoch=11,
        elapsed_training_seconds=20.0,
    )

    assert heartbeat["progress"]["micro_steps_remaining"] == 10
    assert heartbeat["progress"]["optimizer_steps_remaining"] == 5
    assert heartbeat["loss"]["mean_loss_so_far"] == 0.4
    assert heartbeat["timing"]["micro_steps_per_second"] == 0.5
    assert heartbeat["timing"]["eta_seconds"] == 20.0


def test_resolve_checkpoint_dir_accepts_manifest_and_state_file(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "step_0000001_micro_0000008"
    checkpoint_dir.mkdir()
    trainer_state = checkpoint_dir / "trainer_state.pt"
    trainer_state.write_text("placeholder", encoding="utf-8")
    latest = tmp_path / "latest.json"
    latest.write_text(json.dumps({"checkpoint_dir": str(checkpoint_dir)}), encoding="utf-8")

    assert resolve_checkpoint_dir(checkpoint_dir) == checkpoint_dir
    assert resolve_checkpoint_dir(trainer_state) == checkpoint_dir
    assert resolve_checkpoint_dir(latest) == checkpoint_dir


def test_prune_old_checkpoints_keeps_latest_lexicographic_dirs(tmp_path: Path) -> None:
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    for name in (
        "step_0000000_micro_0000000",
        "step_0000001_micro_0000008",
        "step_0000002_micro_0000016",
    ):
        (checkpoints_dir / name).mkdir()

    removed = prune_old_checkpoints(checkpoints_dir, keep_last=2)

    assert len(removed) == 1
    assert not (checkpoints_dir / "step_0000000_micro_0000000").exists()
    assert (checkpoints_dir / "step_0000001_micro_0000008").exists()
    assert (checkpoints_dir / "step_0000002_micro_0000016").exists()


def test_write_heartbeat_writes_json_file(tmp_path: Path) -> None:
    path = tmp_path / "heartbeat.json"
    write_heartbeat(path, {"status": "running", "progress": {"micro_steps_completed": 3}})

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "running"
    assert payload["progress"]["micro_steps_completed"] == 3
