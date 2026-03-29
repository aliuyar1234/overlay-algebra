from __future__ import annotations

from pathlib import Path

import pytest

from overlay_algebra.data.pilot_slice import (
    PilotSliceConfig,
    build_pilot_slice,
    select_pilot_records,
)


FIXTURE_DATASET_DIR = Path("data/processed/squad_jcq_v1_fixture")


def test_select_pilot_records_is_deterministic() -> None:
    records = [
        {"example_id": "b", "title": "T1"},
        {"example_id": "a", "title": "T2"},
        {"example_id": "c", "title": "T3"},
    ]
    first = select_pilot_records(records, split_name="train", limit=2, seed=17)
    second = select_pilot_records(records, split_name="train", limit=2, seed=17)
    assert first == second
    assert [row["example_id"] for row in first] == sorted(row["example_id"] for row in first)


def test_select_pilot_records_raises_when_limit_exceeds_available() -> None:
    with pytest.raises(ValueError):
        select_pilot_records([{"example_id": "x", "title": "T1"}], split_name="val", limit=2, seed=17)


def test_build_pilot_slice_from_fixture_dataset(tmp_path: Path) -> None:
    if not FIXTURE_DATASET_DIR.exists():
        pytest.skip("fixture dataset shard is not present in this review/build context")

    config = PilotSliceConfig(
        experiment_name="pilot_fixture",
        seed=17,
        source_dataset_dir=FIXTURE_DATASET_DIR,
        output_dir=tmp_path / "pilot",
        dataset_version="fixture_pilot",
        sample_seed=17,
        train_examples=1,
        val_examples=1,
        test_examples=1,
    )
    summary = build_pilot_slice(config)
    assert summary["counts"]["pilot_train_examples"] == 1
    assert summary["counts"]["pilot_val_examples"] == 1
    assert summary["counts"]["pilot_test_examples"] == 1
    assert (tmp_path / "pilot" / "dataset_manifest.json").exists()
    assert (tmp_path / "pilot" / "split_audit.json").exists()
