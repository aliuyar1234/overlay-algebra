from __future__ import annotations

import json
from pathlib import Path

import pytest

from overlay_algebra.data.dataset_builder import (
    DatasetBuildConfig,
    build_processed_dataset,
    process_raw_example,
    run_split_audit,
)
from overlay_algebra.data.sentences import SentenceSplitter
from overlay_algebra.data.squad import RawQaExample
from overlay_algebra.parsers import parse_condition


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def spacy_version() -> str:
    spacy = pytest.importorskip("spacy")
    return str(spacy.__version__)


def test_process_raw_example_drops_cross_sentence_answer(spacy_version: str) -> None:
    splitter = SentenceSplitter(expected_spacy_version=spacy_version)
    raw_example = RawQaExample(
        example_id="cross-sentence",
        title="Synthetic",
        question="Which phrase crosses the sentence boundary?",
        context="Alpha ends here. Beta begins there.",
        answer_text="here. Beta",
        answer_start=11,
        source_split="official_train",
    )

    processed, reason = process_raw_example(raw_example, splitter)
    assert processed is None
    assert reason == "answer_not_in_exactly_one_sentence"


def test_build_processed_dataset_outputs_roundtrip_files(tmp_path: Path, spacy_version: str) -> None:
    config = DatasetBuildConfig(
        experiment_name="fixture-build-test",
        seed=17,
        raw_train_path=REPO_ROOT / "tests" / "fixtures" / "raw" / "squad_v1_1" / "train-v1.1.json",
        raw_dev_path=REPO_ROOT / "tests" / "fixtures" / "raw" / "squad_v1_1" / "dev-v1.1.json",
        output_dir=tmp_path / "processed",
        dataset_version="fixture-test",
        source_scope="test-fixture",
        val_fraction=0.1,
        split_seed=17,
        expected_spacy_version=spacy_version,
    )

    summary = build_processed_dataset(config)
    audit = run_split_audit(config.output_dir)
    manifest = json.loads((config.output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))

    assert summary["counts"]["processed_train_examples"] == 1
    assert summary["counts"]["processed_val_examples"] == 1
    assert summary["counts"]["processed_test_examples"] == 1
    assert audit["passed"] is True
    assert manifest["roundtrip_audit"]["failed_examples"] == 0

    train_record = json.loads((config.output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert train_record["labeled_context"].startswith("[S1] ")
    assert parse_condition(train_record["targets"]["JCQ"], "JCQ")["support"].startswith("S")
