from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class TrainingProgressPlan:
    train_examples: int
    batch_size: int
    gradient_accumulation_steps: int
    train_epochs: int
    batches_per_epoch: int
    total_micro_steps: int
    total_optimizer_steps: int
    total_examples: int

    @classmethod
    def from_counts(
        cls,
        *,
        train_examples: int,
        batch_size: int,
        gradient_accumulation_steps: int,
        train_epochs: int,
    ) -> "TrainingProgressPlan":
        if train_examples <= 0:
            raise ValueError("train_examples must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if train_epochs <= 0:
            raise ValueError("train_epochs must be positive")

        batches_per_epoch = math.ceil(train_examples / batch_size)
        total_micro_steps = batches_per_epoch * train_epochs
        total_optimizer_steps = math.ceil(total_micro_steps / gradient_accumulation_steps)
        return cls(
            train_examples=train_examples,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            train_epochs=train_epochs,
            batches_per_epoch=batches_per_epoch,
            total_micro_steps=total_micro_steps,
            total_optimizer_steps=total_optimizer_steps,
            total_examples=train_examples * train_epochs,
        )


@dataclass(slots=True)
class TrainingRuntimeState:
    micro_steps_completed: int = 0
    optimizer_steps_completed: int = 0
    loss_sum: float = 0.0
    loss_count: int = 0
    last_loss: float | None = None
    accumulated_training_seconds: float = 0.0
    latest_checkpoint_dir: str | None = None
    latest_checkpoint_micro_steps_completed: int = 0
    latest_checkpoint_optimizer_steps_completed: int = 0
    resumed_from_checkpoint: str | None = None

    def mean_loss(self) -> float | None:
        if self.loss_count == 0:
            return None
        return self.loss_sum / self.loss_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "micro_steps_completed": self.micro_steps_completed,
            "optimizer_steps_completed": self.optimizer_steps_completed,
            "loss_sum": self.loss_sum,
            "loss_count": self.loss_count,
            "last_loss": self.last_loss,
            "accumulated_training_seconds": self.accumulated_training_seconds,
            "latest_checkpoint_dir": self.latest_checkpoint_dir,
            "latest_checkpoint_micro_steps_completed": self.latest_checkpoint_micro_steps_completed,
            "latest_checkpoint_optimizer_steps_completed": self.latest_checkpoint_optimizer_steps_completed,
            "resumed_from_checkpoint": self.resumed_from_checkpoint,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TrainingRuntimeState":
        return cls(
            micro_steps_completed=int(payload.get("micro_steps_completed", 0)),
            optimizer_steps_completed=int(payload.get("optimizer_steps_completed", 0)),
            loss_sum=float(payload.get("loss_sum", 0.0)),
            loss_count=int(payload.get("loss_count", 0)),
            last_loss=(float(payload["last_loss"]) if payload.get("last_loss") is not None else None),
            accumulated_training_seconds=float(payload.get("accumulated_training_seconds", 0.0)),
            latest_checkpoint_dir=(
                str(payload["latest_checkpoint_dir"]) if payload.get("latest_checkpoint_dir") is not None else None
            ),
            latest_checkpoint_micro_steps_completed=int(payload.get("latest_checkpoint_micro_steps_completed", 0)),
            latest_checkpoint_optimizer_steps_completed=int(
                payload.get("latest_checkpoint_optimizer_steps_completed", 0)
            ),
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


def build_training_heartbeat(
    *,
    condition: str,
    status: str,
    plan: TrainingProgressPlan,
    runtime: TrainingRuntimeState,
    train_examples_loaded: int,
    skipped_train_example_ids: list[str],
    heartbeat_path: Path,
    control_dir: Path,
    current_epoch_index: int | None = None,
    current_batch_index_in_epoch: int | None = None,
    elapsed_training_seconds: float | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    effective_elapsed = (
        float(elapsed_training_seconds)
        if elapsed_training_seconds is not None
        else float(runtime.accumulated_training_seconds)
    )
    micro_steps_remaining = max(plan.total_micro_steps - runtime.micro_steps_completed, 0)
    optimizer_steps_remaining = max(plan.total_optimizer_steps - runtime.optimizer_steps_completed, 0)
    micro_rate = _safe_rate(runtime.micro_steps_completed, effective_elapsed)
    optimizer_rate = _safe_rate(runtime.optimizer_steps_completed, effective_elapsed)
    mean_loss = runtime.mean_loss()

    return {
        "updated_at": _utc_now_iso(),
        "status": status,
        "condition": condition,
        "message": message,
        "progress": {
            "epoch_index": current_epoch_index,
            "epochs_total": plan.train_epochs,
            "batch_index_in_epoch": current_batch_index_in_epoch,
            "batches_per_epoch": plan.batches_per_epoch,
            "micro_steps_completed": runtime.micro_steps_completed,
            "micro_steps_total": plan.total_micro_steps,
            "micro_steps_remaining": micro_steps_remaining,
            "optimizer_steps_completed": runtime.optimizer_steps_completed,
            "optimizer_steps_total": plan.total_optimizer_steps,
            "optimizer_steps_remaining": optimizer_steps_remaining,
            "examples_loaded": train_examples_loaded,
            "examples_used": plan.train_examples,
            "examples_skipped_no_target_tokens": len(skipped_train_example_ids),
        },
        "loss": {
            "last_loss": runtime.last_loss,
            "mean_loss_so_far": mean_loss,
            "loss_count": runtime.loss_count,
        },
        "timing": {
            "elapsed_training_seconds": effective_elapsed,
            "micro_steps_per_second": micro_rate,
            "optimizer_steps_per_second": optimizer_rate,
            "eta_seconds": _safe_eta(micro_steps_remaining, micro_rate),
        },
        "checkpoint": {
            "latest_checkpoint_dir": runtime.latest_checkpoint_dir,
            "latest_checkpoint_micro_steps_completed": runtime.latest_checkpoint_micro_steps_completed,
            "latest_checkpoint_optimizer_steps_completed": runtime.latest_checkpoint_optimizer_steps_completed,
            "resumed_from_checkpoint": runtime.resumed_from_checkpoint,
        },
        "paths": {
            "heartbeat": str(heartbeat_path),
            "control_dir": str(control_dir),
            "pause_request": str(control_dir / "pause.request"),
        },
    }


def write_heartbeat(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_checkpoint_dir(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        return candidate
    if candidate.name == "latest.json":
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        return Path(payload["checkpoint_dir"])
    if candidate.name == "trainer_state.pt":
        return candidate.parent
    raise ValueError(f"Cannot resolve checkpoint directory from {candidate}")


def prune_old_checkpoints(checkpoints_dir: Path, *, keep_last: int) -> list[str]:
    if keep_last < 1:
        raise ValueError("keep_last must be at least 1")
    checkpoint_dirs = sorted(
        [path for path in checkpoints_dir.iterdir() if path.is_dir()],
        key=lambda item: item.name,
    )
    to_remove = checkpoint_dirs[:-keep_last]
    removed = [str(path) for path in to_remove]
    for path in to_remove:
        shutil.rmtree(path)
    return removed
