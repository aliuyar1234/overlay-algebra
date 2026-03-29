from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping

from ..compiler import CONDITIONS, compile_targets, sid_from_index
from ..config import AppConfig, load_app_config
from ..parsers import ParseError, parse_condition
from .sentences import SentenceSpan, SentenceSplitter
from .squad import RawQaExample, load_squad_records, raw_source_identity


_LABEL_RE = re.compile(r"\[S[1-9][0-9]*\]")


@dataclass(frozen=True, slots=True)
class DatasetBuildConfig:
    experiment_name: str
    seed: int
    raw_train_path: Path
    raw_dev_path: Path
    output_dir: Path
    dataset_version: str
    source_scope: str
    val_fraction: float
    split_seed: int
    expected_spacy_version: str | None

    @classmethod
    def from_app_config(cls, config: AppConfig) -> "DatasetBuildConfig":
        payload = config.payload
        split_payload = payload.get("split", {})
        sentence_splitter = payload.get("sentence_splitter", {})
        return cls(
            experiment_name=config.experiment_name,
            seed=config.seed,
            raw_train_path=Path(config.paths["raw_train_path"]),
            raw_dev_path=Path(config.paths["raw_dev_path"]),
            output_dir=Path(config.paths["output_dir"]),
            dataset_version=str(payload["dataset_version"]),
            source_scope=str(payload.get("source_scope", "unknown")),
            val_fraction=float(split_payload.get("val_fraction", 0.1)),
            split_seed=int(split_payload.get("seed", config.seed)),
            expected_spacy_version=_optional_string(sentence_splitter.get("expected_version")),
        )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def _stable_val_titles(titles: set[str], *, val_fraction: float, seed: int) -> set[str]:
    ordered_titles = sorted(
        titles,
        key=lambda title: (hashlib.sha256(f"{seed}:{title}".encode("utf-8")).hexdigest(), title),
    )
    if len(ordered_titles) <= 1:
        return set()

    val_count = max(1, int(math.ceil(len(ordered_titles) * val_fraction)))
    val_count = min(val_count, len(ordered_titles) - 1)
    return set(ordered_titles[:val_count])


def render_labeled_context(sentences: list[SentenceSpan]) -> str:
    return "\n".join(f"[S{sentence.index}] {sentence.text}" for sentence in sentences)


def _roundtrip_targets(answer_text: str, support_idx: int, support_sentence: str, targets: Mapping[str, str]) -> None:
    expected_support = sid_from_index(support_idx)
    for condition in CONDITIONS:
        parsed = parse_condition(targets[condition], condition)
        if parsed["answer"] != answer_text:
            raise ParseError(f"round-trip answer mismatch for {condition}")
        if "support" in parsed and parsed["support"] != expected_support:
            raise ParseError(f"round-trip support mismatch for {condition}")
        if "quote" in parsed and parsed["quote"] != support_sentence:
            raise ParseError(f"round-trip quote mismatch for {condition}")


def _find_support_sentence(
    raw_example: RawQaExample,
    sentences: list[SentenceSpan],
) -> SentenceSpan | None:
    answer_start = raw_example.answer_start
    answer_end = answer_start + len(raw_example.answer_text)
    if raw_example.context[answer_start:answer_end] != raw_example.answer_text:
        return None

    matches = [
        sentence
        for sentence in sentences
        if answer_start >= sentence.start_char and answer_end <= sentence.end_char
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def process_raw_example(
    raw_example: RawQaExample,
    splitter: SentenceSplitter,
) -> tuple[dict[str, Any] | None, str | None]:
    sentences = splitter.split(raw_example.context)
    if not sentences:
        return None, "sentence_split_empty"

    if any(_LABEL_RE.search(sentence.text) for sentence in sentences):
        return None, "ambiguous_sentence_labels"

    support_sentence = _find_support_sentence(raw_example, sentences)
    if support_sentence is None:
        return None, "answer_not_in_exactly_one_sentence"

    if not " ".join(support_sentence.text.split()):
        return None, "support_sentence_empty_after_normalization"

    targets = compile_targets(
        raw_example.answer_text,
        support_sentence.index,
        support_sentence.text,
    )
    try:
        _roundtrip_targets(raw_example.answer_text, support_sentence.index, support_sentence.text, targets)
    except ParseError:
        return None, "parser_roundtrip_failed"

    record = {
        "example_id": raw_example.example_id,
        "title": raw_example.title,
        "question": raw_example.question,
        "context": raw_example.context,
        "context_sentences": [sentence.text for sentence in sentences],
        "labeled_context": render_labeled_context(sentences),
        "answer_text": raw_example.answer_text,
        "answer_start": raw_example.answer_start,
        "support_sent_idx": support_sentence.index,
        "support_sentence": support_sentence.text,
        "targets": targets,
    }
    return record, None


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


def _copy_resolved_config(source_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, output_dir / "resolved_config.yaml")


def build_processed_dataset(config: DatasetBuildConfig, *, source_path: Path | None = None) -> dict[str, Any]:
    splitter = SentenceSplitter(expected_spacy_version=config.expected_spacy_version)
    raw_train = load_squad_records(config.raw_train_path, source_split="official_train")
    raw_dev = load_squad_records(config.raw_dev_path, source_split="official_dev")

    processed_train_candidates: list[dict[str, Any]] = []
    processed_test: list[dict[str, Any]] = []
    dropped_reasons: dict[str, int] = {}

    def handle_example(raw_example: RawQaExample, destination: list[dict[str, Any]]) -> None:
        processed, reason = process_raw_example(raw_example, splitter)
        if processed is None:
            dropped_reasons[reason or "unknown"] = dropped_reasons.get(reason or "unknown", 0) + 1
            return
        destination.append(processed)

    for example in raw_train:
        handle_example(example, processed_train_candidates)
    for example in raw_dev:
        handle_example(example, processed_test)

    train_titles = {str(record["title"]) for record in processed_train_candidates}
    val_titles = _stable_val_titles(train_titles, val_fraction=config.val_fraction, seed=config.split_seed)

    processed_train = sorted(
        [record for record in processed_train_candidates if record["title"] not in val_titles],
        key=lambda record: str(record["example_id"]),
    )
    processed_val = sorted(
        [record for record in processed_train_candidates if record["title"] in val_titles],
        key=lambda record: str(record["example_id"]),
    )
    processed_test = sorted(processed_test, key=lambda record: str(record["example_id"]))

    split_records = {
        "train": processed_train,
        "val": processed_val,
        "test": processed_test,
    }
    audit = _audit_titles(split_records)
    if not audit["passed"]:
        raise RuntimeError(f"title overlap audit failed: {audit['overlaps']}")

    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    hashes = {
        "train_jsonl": _write_jsonl(output_dir / "train.jsonl", processed_train),
        "val_jsonl": _write_jsonl(output_dir / "val.jsonl", processed_val),
        "test_jsonl": _write_jsonl(output_dir / "test.jsonl", processed_test),
    }

    audit_path = output_dir / "split_audit.json"
    _write_json(audit_path, audit)

    manifest = {
        "dataset_version": config.dataset_version,
        "source_scope": config.source_scope,
        "experiment_name": config.experiment_name,
        "seed": config.seed,
        "raw_sources": {
            "train": raw_source_identity(config.raw_train_path),
            "dev": raw_source_identity(config.raw_dev_path),
        },
        "sentence_splitter": {
            "identity": splitter.identity,
            "version": splitter.version,
        },
        "split_policy": {
            "train_source": "official_train",
            "val_fraction": config.val_fraction,
            "split_seed": config.split_seed,
            "test_source": "official_dev",
        },
        "counts": {
            "raw_train_examples": len(raw_train),
            "raw_dev_examples": len(raw_dev),
            "processed_train_examples": len(processed_train),
            "processed_val_examples": len(processed_val),
            "processed_test_examples": len(processed_test),
            "dropped_examples": sum(dropped_reasons.values()),
        },
        "dropped_by_reason": dropped_reasons,
        "hashes": hashes,
        "artifact_paths": {
            "train_jsonl": str(output_dir / "train.jsonl"),
            "val_jsonl": str(output_dir / "val.jsonl"),
            "test_jsonl": str(output_dir / "test.jsonl"),
            "split_audit_json": str(audit_path),
        },
        "roundtrip_audit": {
            "checked_examples": len(processed_train) + len(processed_val) + len(processed_test),
            "failed_examples": 0,
        },
        "config_path": str(source_path) if source_path is not None else None,
    }
    manifest["manifest_sha256"] = _sha256_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True))

    manifest_path = output_dir / "dataset_manifest.json"
    _write_json(manifest_path, manifest)

    if source_path is not None:
        _copy_resolved_config(source_path, output_dir)

    return {
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "audit_path": str(audit_path),
        "counts": manifest["counts"],
        "dropped_by_reason": dropped_reasons,
        "sentence_splitter": splitter.identity,
    }


def load_dataset_build_config(path: str | Path) -> DatasetBuildConfig:
    app_config = load_app_config(path)
    return DatasetBuildConfig.from_app_config(app_config)


def run_split_audit(output_dir: Path) -> dict[str, Any]:
    records: dict[str, list[dict[str, Any]]] = {}
    for split_name in ("train", "val", "test"):
        path = output_dir / f"{split_name}.jsonl"
        split_records = []
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    split_records.append(json.loads(line))
        records[split_name] = split_records

    audit = _audit_titles(records)
    audit["output_dir"] = str(output_dir)
    return audit
