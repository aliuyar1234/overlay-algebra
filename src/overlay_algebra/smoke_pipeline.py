from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import random
import shutil
import time
from typing import Any, Mapping, Sequence

from .artifacts import RunPaths, build_run_paths
from .config import AppConfig, load_app_config
from .data.processed import ProcessedExample, load_processed_examples
from .feature_cache import (
    FeatureCacheDescriptor,
    build_feature_cache_descriptor,
    default_feature_cache_dir,
    feature_cache_path,
    load_feature_cache_bundle,
    save_feature_cache_bundle,
)
from .hf_local import resolve_pretrained_source
from .inference import batched_greedy_generate_texts
from .metrics import score_prediction
from .prompts import prompt_protocol_fingerprint, render_prompt
from .train_runtime import (
    TrainingProgressPlan,
    TrainingRuntimeState,
    build_training_heartbeat,
    prune_old_checkpoints,
    resolve_checkpoint_dir,
    write_heartbeat,
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


def _payload_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"Payload field {key!r} must be a list")
    return [str(item) for item in value]


@dataclass(frozen=True, slots=True)
class TrainSmokeConfig:
    app: AppConfig
    base_model_name_or_path: str
    condition: str
    train_dataset_path: Path
    eval_dataset_path: Path
    local_files_only: bool
    target_modules: list[str]
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    learning_rate: float
    train_epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    max_length: int
    max_new_tokens: int
    max_train_examples: int
    max_eval_examples: int
    device: str
    torch_dtype: str
    reuse_feature_cache: bool
    feature_cache_dir: Path
    checkpoint_interval_optimizer_steps: int
    heartbeat_interval_seconds: float
    control_poll_seconds: float
    max_checkpoints_to_keep: int
    resume_latest: bool
    resume_checkpoint_path: Path | None

    @classmethod
    def from_app_config(
        cls,
        app: AppConfig,
        *,
        resume_latest: bool = False,
        resume_checkpoint_path: Path | None = None,
    ) -> "TrainSmokeConfig":
        payload = app.payload
        configured_resume_checkpoint = (
            Path(app.paths["resume_checkpoint"]) if "resume_checkpoint" in app.paths else None
        )
        effective_resume_checkpoint = resume_checkpoint_path or configured_resume_checkpoint
        train_dataset_path = Path(app.paths["train_dataset"])
        return cls(
            app=app,
            base_model_name_or_path=_payload_string(payload, "base_model_name_or_path"),
            condition=_payload_string(payload, "condition"),
            train_dataset_path=train_dataset_path,
            eval_dataset_path=Path(app.paths["eval_dataset"]),
            local_files_only=_payload_bool(payload, "local_files_only", True),
            target_modules=_payload_list(payload, "target_modules"),
            lora_rank=_payload_int(payload, "lora_rank", 16),
            lora_alpha=_payload_int(payload, "lora_alpha", 16),
            lora_dropout=_payload_float(payload, "lora_dropout", 0.05),
            learning_rate=_payload_float(payload, "learning_rate", 1e-4),
            train_epochs=_payload_int(payload, "train_epochs", 1),
            batch_size=_payload_int(payload, "batch_size", 1),
            gradient_accumulation_steps=_payload_int(payload, "gradient_accumulation_steps", 1),
            max_length=_payload_int(payload, "max_length", 256),
            max_new_tokens=_payload_int(payload, "max_new_tokens", 32),
            max_train_examples=_payload_int(payload, "max_train_examples", 1),
            max_eval_examples=_payload_int(payload, "max_eval_examples", 1),
            device=_payload_string(payload, "device"),
            torch_dtype=_payload_string(payload, "torch_dtype"),
            reuse_feature_cache=_payload_bool(payload, "reuse_feature_cache", True),
            feature_cache_dir=Path(app.paths["feature_cache_dir"])
            if "feature_cache_dir" in app.paths
            else default_feature_cache_dir(train_dataset_path),
            checkpoint_interval_optimizer_steps=_payload_int(payload, "checkpoint_interval_optimizer_steps", 250),
            heartbeat_interval_seconds=_payload_float(payload, "heartbeat_interval_seconds", 30.0),
            control_poll_seconds=_payload_float(payload, "control_poll_seconds", 5.0),
            max_checkpoints_to_keep=_payload_int(payload, "max_checkpoints_to_keep", 3),
            resume_latest=resume_latest or _payload_bool(payload, "resume_latest", False),
            resume_checkpoint_path=effective_resume_checkpoint,
        )


@dataclass(frozen=True, slots=True)
class EvalSmokeConfig:
    app: AppConfig
    base_model_name_or_path: str
    condition: str
    eval_dataset_path: Path
    adapter_dir: Path
    local_files_only: bool
    max_length: int
    max_new_tokens: int
    max_eval_examples: int
    eval_batch_size: int
    device: str
    torch_dtype: str
    reuse_feature_cache: bool
    feature_cache_dir: Path
    sort_eval_by_length: bool

    @classmethod
    def from_app_config(cls, app: AppConfig) -> "EvalSmokeConfig":
        payload = app.payload
        eval_dataset_path = Path(app.paths["eval_dataset"])
        return cls(
            app=app,
            base_model_name_or_path=_payload_string(payload, "base_model_name_or_path"),
            condition=_payload_string(payload, "condition"),
            eval_dataset_path=eval_dataset_path,
            adapter_dir=Path(app.paths["adapter_dir"]),
            local_files_only=_payload_bool(payload, "local_files_only", True),
            max_length=_payload_int(payload, "max_length", 256),
            max_new_tokens=_payload_int(payload, "max_new_tokens", 32),
            max_eval_examples=_payload_int(payload, "max_eval_examples", 1),
            eval_batch_size=_payload_int(payload, "eval_batch_size", 1),
            device=_payload_string(payload, "device"),
            torch_dtype=_payload_string(payload, "torch_dtype"),
            reuse_feature_cache=_payload_bool(payload, "reuse_feature_cache", True),
            feature_cache_dir=Path(app.paths["feature_cache_dir"])
            if "feature_cache_dir" in app.paths
            else default_feature_cache_dir(eval_dataset_path),
            sort_eval_by_length=_payload_bool(payload, "sort_eval_by_length", True),
        )


def _import_ml_stack() -> tuple[Any, Any, Any, Any, Any]:
    missing = [
        name
        for name in ("torch", "transformers", "peft")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeError(
            "Missing M2 runtime dependencies: "
            + ", ".join(missing)
            + ". Sync the environment before running real smoke jobs."
        )

    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    peft = importlib.import_module("peft")
    return (
        torch,
        transformers.AutoModelForCausalLM,
        transformers.AutoTokenizer,
        peft.LoraConfig,
        peft.get_peft_model,
    )


def _import_peft_reload() -> Any:
    if importlib.util.find_spec("peft") is None:
        raise RuntimeError("Missing dependency: peft")
    peft = importlib.import_module("peft")
    return peft.PeftModel


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


def _ensure_run_layout(run_paths: RunPaths, source_config: Path) -> None:
    run_paths.run_dir.mkdir(parents=True, exist_ok=True)
    run_paths.adapter_dir.mkdir(parents=True, exist_ok=True)
    run_paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    run_paths.control_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_config, run_paths.resolved_config)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=False) for row in rows)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _version_or_missing(name: str) -> str:
    try:
        return importlib.import_module(name).__version__
    except Exception:
        return "missing"


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


def _collect_rng_state(torch_module: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python_random_state": random.getstate(),
        "torch_random_state": torch_module.get_rng_state(),
    }
    if torch_module.cuda.is_available():
        payload["cuda_random_state"] = torch_module.cuda.get_rng_state_all()
    try:
        import numpy as np  # type: ignore

        payload["numpy_random_state"] = np.random.get_state()
    except Exception:
        payload["numpy_random_state"] = None
    return payload


def _restore_rng_state(torch_module: Any, payload: Mapping[str, Any]) -> None:
    python_random_state = payload.get("python_random_state")
    if python_random_state is not None:
        random.setstate(python_random_state)

    numpy_random_state = payload.get("numpy_random_state")
    if numpy_random_state is not None:
        try:
            import numpy as np  # type: ignore

            np.random.set_state(numpy_random_state)
        except Exception:
            pass

    torch_random_state = payload.get("torch_random_state")
    if torch_random_state is not None:
        torch_module.set_rng_state(torch_random_state)

    cuda_random_state = payload.get("cuda_random_state")
    if cuda_random_state is not None and torch_module.cuda.is_available():
        torch_module.cuda.set_rng_state_all(cuda_random_state)


def _move_optimizer_state_to_device(optimizer: Any, device: Any) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if hasattr(value, "to"):
                state[key] = value.to(device)


def _resolve_resume_checkpoint_path(config: TrainSmokeConfig, run_paths: RunPaths) -> Path | None:
    if config.resume_checkpoint_path is not None:
        return resolve_checkpoint_dir(config.resume_checkpoint_path)
    if config.resume_latest:
        if run_paths.latest_checkpoint.exists():
            return resolve_checkpoint_dir(run_paths.latest_checkpoint)
        raise FileNotFoundError(
            f"--resume-latest was requested, but no latest checkpoint exists at {run_paths.latest_checkpoint}"
        )
    return None


def _assert_run_dir_available_for_training(run_paths: RunPaths, *, resume_checkpoint_dir: Path | None) -> None:
    if resume_checkpoint_dir is not None:
        return
    if run_paths.metrics.exists() or run_paths.run_manifest.exists() or run_paths.latest_checkpoint.exists():
        raise FileExistsError(
            "Run directory already contains completed or checkpointed state. "
            f"Move {run_paths.run_dir} aside or rerun with --resume-latest / --resume-checkpoint."
        )


def _load_trainable_peft_model(base_model: Any, adapter_dir: Path) -> Any:
    PeftModel = _import_peft_reload()
    try:
        return PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)
    except TypeError:
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        model.train()
        return model


def _checkpoint_dir_for(runtime: TrainingRuntimeState, run_paths: RunPaths) -> Path:
    return run_paths.checkpoints_dir / (
        f"step_{runtime.optimizer_steps_completed:07d}_micro_{runtime.micro_steps_completed:07d}"
    )


def _write_latest_checkpoint_manifest(
    path: Path,
    *,
    checkpoint_dir: Path,
    runtime: TrainingRuntimeState,
    trainer_state_path: Path,
) -> None:
    _write_json(
        path,
        {
            "updated_at": time.time(),
            "checkpoint_dir": str(checkpoint_dir),
            "trainer_state_path": str(trainer_state_path),
            "adapter_dir": str(checkpoint_dir / "adapter"),
            "micro_steps_completed": runtime.micro_steps_completed,
            "optimizer_steps_completed": runtime.optimizer_steps_completed,
        },
    )


def _save_training_checkpoint(
    *,
    torch_module: Any,
    model: Any,
    optimizer: Any,
    runtime: TrainingRuntimeState,
    run_paths: RunPaths,
    keep_last: int,
) -> Path:
    checkpoint_dir = _checkpoint_dir_for(runtime, run_paths)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = checkpoint_dir / "adapter"
    model.save_pretrained(adapter_dir)
    trainer_state_path = checkpoint_dir / "trainer_state.pt"
    torch_module.save(
        {
            "runtime": runtime.as_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": _collect_rng_state(torch_module),
        },
        trainer_state_path,
    )
    _write_json(
        checkpoint_dir / "metadata.json",
        {
            "created_at": time.time(),
            "adapter_dir": str(adapter_dir),
            "trainer_state_path": str(trainer_state_path),
            "micro_steps_completed": runtime.micro_steps_completed,
            "optimizer_steps_completed": runtime.optimizer_steps_completed,
            "last_loss": runtime.last_loss,
        },
    )

    runtime.latest_checkpoint_dir = str(checkpoint_dir)
    runtime.latest_checkpoint_micro_steps_completed = runtime.micro_steps_completed
    runtime.latest_checkpoint_optimizer_steps_completed = runtime.optimizer_steps_completed
    _write_latest_checkpoint_manifest(
        run_paths.latest_checkpoint,
        checkpoint_dir=checkpoint_dir,
        runtime=runtime,
        trainer_state_path=trainer_state_path,
    )
    prune_old_checkpoints(run_paths.checkpoints_dir, keep_last=max(1, keep_last))
    return checkpoint_dir


def _load_training_checkpoint(
    *,
    torch_module: Any,
    model: Any,
    optimizer: Any,
    checkpoint_dir: Path,
    device: Any,
) -> TrainingRuntimeState:
    trainer_state = torch_module.load(checkpoint_dir / "trainer_state.pt", map_location="cpu")
    runtime = TrainingRuntimeState.from_mapping(trainer_state["runtime"])
    optimizer.load_state_dict(trainer_state["optimizer_state_dict"])
    _move_optimizer_state_to_device(optimizer, device)
    rng_state = trainer_state.get("rng_state")
    if isinstance(rng_state, Mapping):
        _restore_rng_state(torch_module, rng_state)
    runtime.latest_checkpoint_dir = str(checkpoint_dir)
    runtime.latest_checkpoint_micro_steps_completed = runtime.micro_steps_completed
    runtime.latest_checkpoint_optimizer_steps_completed = runtime.optimizer_steps_completed
    runtime.resumed_from_checkpoint = str(checkpoint_dir)
    return runtime


def _write_training_heartbeat(
    *,
    config: TrainSmokeConfig,
    run_paths: RunPaths,
    plan: TrainingProgressPlan,
    runtime: TrainingRuntimeState,
    train_examples_loaded: int,
    skipped_train_example_ids: list[str],
    status: str,
    elapsed_training_seconds: float,
    current_epoch_index: int | None = None,
    current_batch_index_in_epoch: int | None = None,
    message: str | None = None,
) -> None:
    payload = build_training_heartbeat(
        condition=config.condition,
        status=status,
        plan=plan,
        runtime=runtime,
        train_examples_loaded=train_examples_loaded,
        skipped_train_example_ids=skipped_train_example_ids,
        heartbeat_path=run_paths.heartbeat,
        control_dir=run_paths.control_dir,
        current_epoch_index=current_epoch_index,
        current_batch_index_in_epoch=current_batch_index_in_epoch,
        elapsed_training_seconds=elapsed_training_seconds,
        message=message,
    )
    write_heartbeat(run_paths.heartbeat, payload)


def _maybe_pause_training(
    *,
    config: TrainSmokeConfig,
    run_paths: RunPaths,
    plan: TrainingProgressPlan,
    runtime: TrainingRuntimeState,
    train_examples_loaded: int,
    skipped_train_example_ids: list[str],
    elapsed_training_seconds: float,
) -> float:
    if not run_paths.pause_request.exists():
        return 0.0

    _write_training_heartbeat(
        config=config,
        run_paths=run_paths,
        plan=plan,
        runtime=runtime,
        train_examples_loaded=train_examples_loaded,
        skipped_train_example_ids=skipped_train_example_ids,
        status="paused",
        elapsed_training_seconds=elapsed_training_seconds,
        message="Training is paused because control/pause.request exists. Remove the file to resume.",
    )
    pause_started = time.monotonic()
    while run_paths.pause_request.exists():
        time.sleep(max(config.control_poll_seconds, 0.1))
        _write_training_heartbeat(
            config=config,
            run_paths=run_paths,
            plan=plan,
            runtime=runtime,
            train_examples_loaded=train_examples_loaded,
            skipped_train_example_ids=skipped_train_example_ids,
            status="paused",
            elapsed_training_seconds=elapsed_training_seconds,
            message="Training is paused because control/pause.request exists. Remove the file to resume.",
        )
    return time.monotonic() - pause_started


def _cuda_available_text() -> str:
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return "unknown"
    try:
        return str(bool(torch.cuda.is_available()))
    except Exception:
        return "unknown"


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


def _load_examples(path: Path, *, limit: int) -> list[ProcessedExample]:
    examples = load_processed_examples(path, limit=limit)
    if not examples:
        raise ValueError(f"No processed examples found in {path}")
    return examples


def _build_supervised_features(
    tokenizer: Any,
    example: ProcessedExample,
    *,
    condition: str,
    max_length: int,
) -> dict[str, list[int]]:
    prompt = render_prompt(example.question, example.labeled_context, condition)
    target = example.targets[condition]
    eos = tokenizer.eos_token or ""

    prompt_tokens = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=max_length)
    full_tokens = tokenizer(
        prompt + target + eos,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )

    input_ids = list(full_tokens["input_ids"])
    attention_mask = list(full_tokens["attention_mask"])
    labels = list(input_ids)

    prompt_len = min(len(prompt_tokens["input_ids"]), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "example_id": example.example_id,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "supervised_token_count": sum(1 for label in labels if label != -100),
    }


def _collate_training_batch(batch: Sequence[dict[str, list[int]]], torch_module: Any, pad_token_id: int) -> dict[str, Any]:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    attention_masks = []
    labels = []

    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        attention_masks.append(item["attention_mask"] + [0] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch_module.tensor(input_ids, dtype=torch_module.long),
        "attention_mask": torch_module.tensor(attention_masks, dtype=torch_module.long),
        "labels": torch_module.tensor(labels, dtype=torch_module.long),
    }


def _batch_examples(items: Sequence[Any], batch_size: int) -> list[list[Any]]:
    return [list(items[index : index + batch_size]) for index in range(0, len(items), batch_size)]


def _partition_supervised_features(
    features: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    usable_features: list[dict[str, Any]] = []
    skipped_features: list[dict[str, Any]] = []
    for feature in features:
        if int(feature.get("supervised_token_count", 0)) > 0:
            usable_features.append(feature)
        else:
            skipped_features.append(feature)
    return usable_features, skipped_features


def _load_or_build_training_features(
    *,
    config: TrainSmokeConfig,
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], list[str], FeatureCacheDescriptor, Path, bool]:
    descriptor = build_feature_cache_descriptor(
        dataset_path=config.train_dataset_path,
        feature_kind="train_supervised",
        base_model_name_or_path=config.base_model_name_or_path,
        tokenizer_name_or_path=config.base_model_name_or_path,
        condition=config.condition,
        prompt_protocol_sha256=prompt_protocol_fingerprint(config.condition),
        max_length=config.max_length,
        max_examples=config.max_train_examples,
    )
    cache_path = feature_cache_path(config.feature_cache_dir, descriptor)
    if config.reuse_feature_cache:
        cached = load_feature_cache_bundle(cache_path, descriptor=descriptor)
        if cached is not None:
            metadata = cached["metadata"]
            skipped_ids = metadata.get("skipped_example_ids_no_target_tokens", [])
            if not isinstance(skipped_ids, list):
                skipped_ids = []
            return cached["features"], [str(item) for item in skipped_ids], descriptor, cache_path, True

    train_examples = _load_examples(config.train_dataset_path, limit=config.max_train_examples)
    all_train_features = [
        _build_supervised_features(tokenizer, example, condition=config.condition, max_length=config.max_length)
        for example in train_examples
    ]
    usable_features, skipped_features = _partition_supervised_features(all_train_features)
    skipped_train_example_ids = [str(feature.get("example_id", "unknown")) for feature in skipped_features]
    if config.reuse_feature_cache:
        save_feature_cache_bundle(
            cache_path,
            descriptor=descriptor,
            features=usable_features,
            metadata={
                "skipped_example_ids_no_target_tokens": skipped_train_example_ids,
                "train_examples_loaded": len(all_train_features),
                "train_examples_used": len(usable_features),
            },
        )
    return usable_features, skipped_train_example_ids, descriptor, cache_path, False


def _mean_loss(losses: Sequence[float]) -> float:
    if not losses:
        raise ValueError("Cannot summarize training losses because no optimization steps were recorded.")
    for loss in losses:
        if not math.isfinite(loss):
            raise ValueError(f"Encountered non-finite loss value {loss!r} in training summary.")
    return sum(losses) / len(losses)


def _mean_metric_rows(rows: Sequence[dict[str, float | None]]) -> dict[str, float]:
    metrics: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if value is None:
                continue
            metrics.setdefault(key, []).append(float(value))
    return {
        key: (sum(values) / len(values) if values else 0.0)
        for key, values in metrics.items()
    }


def train_smoke(config: TrainSmokeConfig) -> dict[str, Any]:
    torch, AutoModelForCausalLM, AutoTokenizer, LoraConfig, get_peft_model = _import_ml_stack()
    run_paths = build_run_paths(config.app)
    resume_checkpoint_dir = _resolve_resume_checkpoint_path(config, run_paths)
    _assert_run_dir_available_for_training(run_paths, resume_checkpoint_dir=resume_checkpoint_dir)
    _ensure_run_layout(run_paths, config.app.source_path)
    write_environment_summary(run_paths.environment)

    _set_seed(config.app.seed)
    resolved_model_source = resolve_pretrained_source(
        config.base_model_name_or_path,
        local_files_only=config.local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_source,
        local_files_only=config.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_features, skipped_train_example_ids, feature_cache_descriptor, feature_cache_path_used, feature_cache_hit = (
        _load_or_build_training_features(config=config, tokenizer=tokenizer)
    )
    if not train_features:
        raise ValueError(
            "All loaded training examples lost their supervised target tokens after truncation. "
            "Increase max_length or change the training slice."
        )
    train_examples_loaded = len(train_features) + len(skipped_train_example_ids)
    plan = TrainingProgressPlan.from_counts(
        train_examples=len(train_features),
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        train_epochs=config.train_epochs,
    )

    runtime = TrainingRuntimeState()
    session_started = time.monotonic()
    paused_seconds = 0.0
    last_heartbeat_timestamp = session_started

    def elapsed_training_seconds() -> float:
        return runtime.accumulated_training_seconds + max(0.0, time.monotonic() - session_started - paused_seconds)

    _write_training_heartbeat(
        config=config,
        run_paths=run_paths,
        plan=plan,
        runtime=runtime,
        train_examples_loaded=train_examples_loaded,
        skipped_train_example_ids=skipped_train_example_ids,
        status="preparing_model",
        elapsed_training_seconds=elapsed_training_seconds(),
        message=(
            "Loaded tokenized training features from cache; loading the model."
            if feature_cache_hit
            else "Tokenizer and training features are ready; loading the model."
        ),
    )

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            resolved_model_source,
            local_files_only=config.local_files_only,
            torch_dtype=_resolve_dtype(torch, config.torch_dtype),
        )
        base_model.config.use_cache = False

        if resume_checkpoint_dir is None:
            lora_config = LoraConfig(
                task_type="CAUSAL_LM",
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.target_modules,
            )
            model = get_peft_model(base_model, lora_config)
        else:
            model = _load_trainable_peft_model(base_model, resume_checkpoint_dir / "adapter")

        device = torch.device(config.device)
        model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=config.learning_rate,
        )

        if resume_checkpoint_dir is None:
            _save_training_checkpoint(
                torch_module=torch,
                model=model,
                optimizer=optimizer,
                runtime=runtime,
                run_paths=run_paths,
                keep_last=config.max_checkpoints_to_keep,
            )
        else:
            runtime = _load_training_checkpoint(
                torch_module=torch,
                model=model,
                optimizer=optimizer,
                checkpoint_dir=resume_checkpoint_dir,
                device=device,
            )
            if runtime.micro_steps_completed > plan.total_micro_steps:
                raise ValueError(
                    "Resume checkpoint is ahead of the current training plan: "
                    f"{runtime.micro_steps_completed} > {plan.total_micro_steps} micro-steps."
                )
            last_heartbeat_timestamp = time.monotonic()

        _write_training_heartbeat(
            config=config,
            run_paths=run_paths,
            plan=plan,
            runtime=runtime,
            train_examples_loaded=train_examples_loaded,
            skipped_train_example_ids=skipped_train_example_ids,
            status="running",
            elapsed_training_seconds=elapsed_training_seconds(),
            message=(
                "Training resumed from checkpoint."
                if resume_checkpoint_dir is not None
                else "Training started from a fresh adapter initialization."
            ),
        )

        train_batches = _batch_examples(train_features, config.batch_size)
        total_batches_per_epoch = len(train_batches)

        global_batch_index = runtime.micro_steps_completed
        while global_batch_index < plan.total_micro_steps:
            epoch_index = global_batch_index // total_batches_per_epoch
            batch_index_in_epoch = global_batch_index % total_batches_per_epoch

            if run_paths.pause_request.exists():
                safe_pause_boundary = (
                    runtime.micro_steps_completed == 0
                    or runtime.micro_steps_completed % config.gradient_accumulation_steps == 0
                )
                if safe_pause_boundary:
                    if runtime.micro_steps_completed != runtime.latest_checkpoint_micro_steps_completed:
                        _save_training_checkpoint(
                            torch_module=torch,
                            model=model,
                            optimizer=optimizer,
                            runtime=runtime,
                            run_paths=run_paths,
                            keep_last=config.max_checkpoints_to_keep,
                        )
                    paused_seconds += _maybe_pause_training(
                        config=config,
                        run_paths=run_paths,
                        plan=plan,
                        runtime=runtime,
                        train_examples_loaded=train_examples_loaded,
                        skipped_train_example_ids=skipped_train_example_ids,
                        elapsed_training_seconds=elapsed_training_seconds(),
                    )
                    last_heartbeat_timestamp = time.monotonic()
                elif (
                    time.monotonic() - last_heartbeat_timestamp
                    >= max(config.heartbeat_interval_seconds, 1.0)
                ):
                    _write_training_heartbeat(
                        config=config,
                        run_paths=run_paths,
                        plan=plan,
                        runtime=runtime,
                        train_examples_loaded=train_examples_loaded,
                        skipped_train_example_ids=skipped_train_example_ids,
                        status="pause_pending",
                        elapsed_training_seconds=elapsed_training_seconds(),
                        current_epoch_index=epoch_index + 1,
                        current_batch_index_in_epoch=batch_index_in_epoch + 1,
                        message=(
                            "Pause requested; waiting to reach the next checkpoint-safe optimizer-step boundary."
                        ),
                    )
                    last_heartbeat_timestamp = time.monotonic()

            batch_items = train_batches[batch_index_in_epoch]
            batch = _collate_training_batch(batch_items, torch, tokenizer.pad_token_id)
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            raw_loss = float(outputs.loss.item())
            if not math.isfinite(raw_loss):
                batch_ids = [str(item.get("example_id", "unknown")) for item in batch_items]
                raise RuntimeError(
                    "Encountered non-finite training loss "
                    f"for condition {config.condition} on examples {batch_ids}. "
                    "This usually means the target tokens were fully truncated or the optimizer became unstable."
                )

            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()
            runtime.loss_sum += raw_loss
            runtime.loss_count += 1
            runtime.last_loss = raw_loss
            runtime.micro_steps_completed += 1
            global_batch_index += 1

            optimizer_step_taken = False
            if runtime.micro_steps_completed % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                runtime.optimizer_steps_completed += 1
                optimizer_step_taken = True

            if optimizer_step_taken and config.checkpoint_interval_optimizer_steps > 0:
                if runtime.optimizer_steps_completed % config.checkpoint_interval_optimizer_steps == 0:
                    _save_training_checkpoint(
                        torch_module=torch,
                        model=model,
                        optimizer=optimizer,
                        runtime=runtime,
                        run_paths=run_paths,
                        keep_last=config.max_checkpoints_to_keep,
                    )

            if time.monotonic() - last_heartbeat_timestamp >= max(config.heartbeat_interval_seconds, 1.0):
                _write_training_heartbeat(
                    config=config,
                    run_paths=run_paths,
                    plan=plan,
                    runtime=runtime,
                    train_examples_loaded=train_examples_loaded,
                    skipped_train_example_ids=skipped_train_example_ids,
                    status="running",
                    elapsed_training_seconds=elapsed_training_seconds(),
                    current_epoch_index=epoch_index + 1,
                    current_batch_index_in_epoch=batch_index_in_epoch + 1,
                )
                last_heartbeat_timestamp = time.monotonic()

        if runtime.micro_steps_completed % config.gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            runtime.optimizer_steps_completed += 1

        runtime.accumulated_training_seconds = elapsed_training_seconds()
        _save_training_checkpoint(
            torch_module=torch,
            model=model,
            optimizer=optimizer,
            runtime=runtime,
            run_paths=run_paths,
            keep_last=config.max_checkpoints_to_keep,
        )

        model.save_pretrained(run_paths.adapter_dir)
        tokenizer.save_pretrained(run_paths.adapter_dir)

        trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total_params = sum(parameter.numel() for parameter in model.parameters())

        skipped_count = len(skipped_train_example_ids)
        metrics = {
            "status": "trained",
            "condition": config.condition,
            "train_examples": len(train_features),
            "train_examples_loaded": train_examples_loaded,
            "skipped_train_examples_no_target_tokens": skipped_count,
            "skipped_train_example_ids_no_target_tokens": skipped_train_example_ids,
            "train_steps": runtime.micro_steps_completed,
            "optimizer_steps": runtime.optimizer_steps_completed,
            "mean_train_loss": _mean_loss([runtime.loss_sum / runtime.loss_count]),
            "last_train_loss": runtime.last_loss,
            "trainable_parameters": trainable_params,
            "total_parameters": total_params,
            "elapsed_training_seconds": runtime.accumulated_training_seconds,
            "latest_checkpoint_dir": runtime.latest_checkpoint_dir,
            "feature_cache_hit": feature_cache_hit,
            "feature_cache_path": str(feature_cache_path_used),
        }

        manifest = {
            "stage": "train",
            "experiment_name": config.app.experiment_name,
            "milestone": config.app.milestone,
            "seed": config.app.seed,
            "base_model_name_or_path": config.base_model_name_or_path,
            "condition": config.condition,
            "train_dataset_path": str(config.train_dataset_path),
            "eval_dataset_path": str(config.eval_dataset_path),
            "adapter_dir": str(run_paths.adapter_dir),
            "checkpoints_dir": str(run_paths.checkpoints_dir),
            "latest_checkpoint_path": str(run_paths.latest_checkpoint),
            "heartbeat_path": str(run_paths.heartbeat),
            "control_dir": str(run_paths.control_dir),
            "pause_request_path": str(run_paths.pause_request),
            "device": config.device,
            "torch_dtype": config.torch_dtype,
            "local_files_only": config.local_files_only,
            "target_modules": config.target_modules,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "learning_rate": config.learning_rate,
            "train_epochs": config.train_epochs,
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "max_length": config.max_length,
            "max_new_tokens": config.max_new_tokens,
            "max_train_examples": config.max_train_examples,
            "max_eval_examples": config.max_eval_examples,
            "checkpoint_interval_optimizer_steps": config.checkpoint_interval_optimizer_steps,
            "heartbeat_interval_seconds": config.heartbeat_interval_seconds,
            "control_poll_seconds": config.control_poll_seconds,
            "max_checkpoints_to_keep": config.max_checkpoints_to_keep,
            "train_examples_loaded": train_examples_loaded,
            "train_examples_used": len(train_features),
            "skipped_train_examples_no_target_tokens": skipped_count,
            "skipped_train_example_ids_no_target_tokens": skipped_train_example_ids,
            "metrics_path": str(run_paths.metrics),
            "resumed_from_checkpoint": runtime.resumed_from_checkpoint,
            "reuse_feature_cache": config.reuse_feature_cache,
            "feature_cache_dir": str(config.feature_cache_dir),
            "feature_cache_path": str(feature_cache_path_used),
            "feature_cache_hit": feature_cache_hit,
            "feature_cache_key": feature_cache_descriptor.cache_key(),
            "feature_cache_descriptor": feature_cache_descriptor.as_dict(),
        }

        _write_json(run_paths.metrics, metrics)
        _write_json(run_paths.run_manifest, manifest)
        _write_training_heartbeat(
            config=config,
            run_paths=run_paths,
            plan=plan,
            runtime=runtime,
            train_examples_loaded=train_examples_loaded,
            skipped_train_example_ids=skipped_train_example_ids,
            status="completed",
            elapsed_training_seconds=runtime.accumulated_training_seconds,
            current_epoch_index=config.train_epochs,
            current_batch_index_in_epoch=total_batches_per_epoch,
            message="Training completed successfully.",
        )

        return {
            "run_dir": str(run_paths.run_dir),
            "adapter_dir": str(run_paths.adapter_dir),
            "metrics_path": str(run_paths.metrics),
            "run_manifest_path": str(run_paths.run_manifest),
            "heartbeat_path": str(run_paths.heartbeat),
            "latest_checkpoint_path": str(run_paths.latest_checkpoint),
            "train_examples": len(train_features),
            "train_examples_loaded": train_examples_loaded,
            "skipped_train_examples_no_target_tokens": skipped_count,
            "train_steps": runtime.micro_steps_completed,
            "optimizer_steps": runtime.optimizer_steps_completed,
            "status": "trained",
            "resumed_from_checkpoint": runtime.resumed_from_checkpoint,
            "feature_cache_hit": feature_cache_hit,
            "feature_cache_path": str(feature_cache_path_used),
        }
    except Exception as exc:
        runtime.accumulated_training_seconds = elapsed_training_seconds()
        _write_training_heartbeat(
            config=config,
            run_paths=run_paths,
            plan=plan,
            runtime=runtime,
            train_examples_loaded=train_examples_loaded,
            skipped_train_example_ids=skipped_train_example_ids,
            status="failed",
            elapsed_training_seconds=runtime.accumulated_training_seconds,
            message=str(exc),
        )
        raise


def evaluate_smoke(config: EvalSmokeConfig) -> dict[str, Any]:
    run_paths = build_run_paths(config.app)
    _ensure_run_layout(run_paths, config.app.source_path)
    write_environment_summary(run_paths.environment)
    _set_seed(config.app.seed)
    from .analysis.adapter_eval import LoadedAdapterEvaluator

    evaluator = LoadedAdapterEvaluator(
        base_model_name_or_path=config.base_model_name_or_path,
        local_files_only=config.local_files_only,
        device=config.device,
        torch_dtype=config.torch_dtype,
        reuse_feature_cache=config.reuse_feature_cache,
    )
    feature_bundle = evaluator.get_eval_features(
        dataset_path=config.eval_dataset_path,
        condition=config.condition,
        max_length=config.max_length,
        max_examples=config.max_eval_examples,
        cache_dir=config.feature_cache_dir,
    )
    evaluation = evaluator.evaluate_adapter(
        adapter_name=f"eval_{config.condition.lower()}",
        adapter_dir=config.adapter_dir,
        eval_features=feature_bundle.features,
        condition=config.condition,
        max_length=config.max_length,
        max_new_tokens=config.max_new_tokens,
        batch_size=config.eval_batch_size,
        sort_by_length=config.sort_eval_by_length,
    )
    summary = {
        "status": "evaluated",
        "condition": config.condition,
        "eval_examples": len(feature_bundle.features),
        "metrics": evaluation.metrics,
        "adapter_dir": str(config.adapter_dir),
        "feature_cache_hit": feature_bundle.cache_hit,
        "feature_cache_path": str(feature_bundle.cache_path),
        "generation_summary": evaluation.generation_summary,
    }

    manifest = {
        "stage": "eval",
        "experiment_name": config.app.experiment_name,
        "milestone": config.app.milestone,
        "seed": config.app.seed,
        "base_model_name_or_path": config.base_model_name_or_path,
        "condition": config.condition,
        "eval_dataset_path": str(config.eval_dataset_path),
        "adapter_dir": str(config.adapter_dir),
        "device": config.device,
        "torch_dtype": config.torch_dtype,
        "local_files_only": config.local_files_only,
        "max_length": config.max_length,
        "max_new_tokens": config.max_new_tokens,
        "max_eval_examples": config.max_eval_examples,
        "eval_batch_size": config.eval_batch_size,
        "reuse_feature_cache": config.reuse_feature_cache,
        "feature_cache_dir": str(config.feature_cache_dir),
        "feature_cache_path": str(feature_bundle.cache_path),
        "feature_cache_hit": feature_bundle.cache_hit,
        "feature_cache_key": feature_bundle.descriptor.cache_key(),
        "feature_cache_descriptor": feature_bundle.descriptor.as_dict(),
        "sort_eval_by_length": config.sort_eval_by_length,
        "metrics_path": str(run_paths.metrics),
        "predictions_path": str(run_paths.predictions),
    }

    _write_json(run_paths.metrics, summary)
    _write_json(run_paths.run_manifest, manifest)
    _write_jsonl(run_paths.predictions, evaluation.predictions)

    return {
        "run_dir": str(run_paths.run_dir),
        "metrics_path": str(run_paths.metrics),
        "predictions_path": str(run_paths.predictions),
        "run_manifest_path": str(run_paths.run_manifest),
        "eval_examples": len(feature_bundle.features),
        "status": "evaluated",
        "metrics": evaluation.metrics,
        "feature_cache_hit": feature_bundle.cache_hit,
        "feature_cache_path": str(feature_bundle.cache_path),
    }


def load_train_smoke_config(
    path: str | Path,
    *,
    resume_latest: bool = False,
    resume_checkpoint_path: str | Path | None = None,
) -> TrainSmokeConfig:
    app = load_app_config(path)
    return TrainSmokeConfig.from_app_config(
        app,
        resume_latest=resume_latest,
        resume_checkpoint_path=(Path(resume_checkpoint_path) if resume_checkpoint_path is not None else None),
    )


def load_eval_smoke_config(path: str | Path) -> EvalSmokeConfig:
    app = load_app_config(path)
    return EvalSmokeConfig.from_app_config(app)
