from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

from ..config import AppConfig, load_app_config


@dataclass(frozen=True, slots=True)
class PilotSliceConfig:
    experiment_name: str
    seed: int
    source_dataset_dir: Path
    output_dir: Path
    dataset_version: str
    sample_seed: int
    train_examples: int
    val_examples: int
    test_examples: int

    @classmethod
    def from_app_config(cls, config: AppConfig) -> "PilotSliceConfig":
        payload = config.payload
        sample_sizes = payload.get("sample_sizes", {})
        return cls(
            experiment_name=config.experiment_name,
            seed=config.seed,
            source_dataset_dir=Path(config.paths["source_dataset_dir"]),
            output_dir=Path(config.paths["output_dir"]),
            dataset_version=str(payload["dataset_version"]),
            sample_seed=int(payload.get("sample_seed", config.seed)),
            train_examples=int(sample_sizes.get("train", 500)),
            val_examples=int(sample_sizes.get("val", 200)),
            test_examples=int(sample_sizes.get("test", 500)),
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False, sort_keys=False) for record in records]
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _sample_key(seed: int, split_name: str, example_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}:{split_name}:{example_id}".encode("utf-8")).hexdigest()
    return digest, example_id


def select_pilot_records(records: list[dict[str, Any]], *, split_name: str, limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    if len(records) < limit:
        raise ValueError(
            f"Requested {limit} {split_name} examples, but only {len(records)} are available."
        )
    ordered = sorted(records, key=lambda row: _sample_key(seed, split_name, str(row["example_id"])))
    selected = ordered[:limit]
    return sorted(selected, key=lambda row: str(row["example_id"]))


def _audit_titles(split_records: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
    split_titles = {
        split_name: sorted({str(record["title"]) for record in records})
        for split_name, records in split_records.items()
    }
    overlaps = {
        "train_val": sorted(set(split_titles["train"]) & set(split_titles["val"])),
        "train_test": sorted(set(split_titles["train"]) & set(split_titles["test"])),
        "val_test": sorted(set(split_titles["val"]) & set(split_titles["test"])),
    }
    return {
        "split_titles": split_titles,
        "title_counts": {name: len(values) for name, values in split_titles.items()},
        "overlaps": overlaps,
        "passed": all(not values for values in overlaps.values()),
    }


def build_pilot_slice(config: PilotSliceConfig, *, source_path: Path | None = None) -> dict[str, Any]:
    source_dir = config.source_dataset_dir.resolve()
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_manifest_path = source_dir / "dataset_manifest.json"
    source_manifest = _read_json(source_manifest_path) if source_manifest_path.exists() else None

    source_splits = {
        "train": _read_jsonl(source_dir / "train.jsonl"),
        "val": _read_jsonl(source_dir / "val.jsonl"),
        "test": _read_jsonl(source_dir / "test.jsonl"),
    }
    sample_sizes = {
        "train": config.train_examples,
        "val": config.val_examples,
        "test": config.test_examples,
    }
    sampled = {
        split_name: select_pilot_records(records, split_name=split_name, limit=sample_sizes[split_name], seed=config.sample_seed)
        for split_name, records in source_splits.items()
    }

    audit = _audit_titles(sampled)
    if not audit["passed"]:
        raise RuntimeError(f"pilot slice title overlap audit failed: {audit['overlaps']}")

    hashes = {
        f"{split_name}_jsonl": _write_jsonl(output_dir / f"{split_name}.jsonl", records)
        for split_name, records in sampled.items()
    }
    audit_path = output_dir / "split_audit.json"
    _write_json(audit_path, audit)

    manifest = {
        "dataset_version": config.dataset_version,
        "experiment_name": config.experiment_name,
        "seed": config.seed,
        "sample_seed": config.sample_seed,
        "sample_policy": "deterministic hash sample by example_id within each existing split",
        "sample_sizes": sample_sizes,
        "counts": {
            "source_train_examples": len(source_splits["train"]),
            "source_val_examples": len(source_splits["val"]),
            "source_test_examples": len(source_splits["test"]),
            "pilot_train_examples": len(sampled["train"]),
            "pilot_val_examples": len(sampled["val"]),
            "pilot_test_examples": len(sampled["test"]),
        },
        "hashes": hashes,
        "artifact_paths": {
            "train_jsonl": str(output_dir / "train.jsonl"),
            "val_jsonl": str(output_dir / "val.jsonl"),
            "test_jsonl": str(output_dir / "test.jsonl"),
            "split_audit_json": str(audit_path),
        },
        "source_dataset_dir": str(source_dir),
        "source_manifest_path": str(source_manifest_path) if source_manifest is not None else None,
        "source_manifest_sha256": hashlib.sha256(
            json.dumps(source_manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if source_manifest is not None
        else None,
        "config_path": str(source_path) if source_path is not None else None,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest_path = output_dir / "dataset_manifest.json"
    _write_json(manifest_path, manifest)

    if source_path is not None:
        shutil.copyfile(source_path, output_dir / "resolved_config.yaml")

    return {
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "audit_path": str(audit_path),
        "counts": manifest["counts"],
        "sample_sizes": sample_sizes,
    }


def load_pilot_slice_config(path: str | Path) -> PilotSliceConfig:
    app_config = load_app_config(path)
    return PilotSliceConfig.from_app_config(app_config)
