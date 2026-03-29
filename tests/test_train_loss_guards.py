from __future__ import annotations

import math

import pytest

from overlay_algebra.smoke_pipeline import _mean_loss, _partition_supervised_features


def test_partition_supervised_features_skips_fully_masked_examples() -> None:
    usable, skipped = _partition_supervised_features(
        [
            {"example_id": "keep-1", "supervised_token_count": 3},
            {"example_id": "skip-1", "supervised_token_count": 0},
            {"example_id": "keep-2", "supervised_token_count": 1},
        ]
    )

    assert [feature["example_id"] for feature in usable] == ["keep-1", "keep-2"]
    assert [feature["example_id"] for feature in skipped] == ["skip-1"]


def test_mean_loss_rejects_nan_values() -> None:
    with pytest.raises(ValueError, match="non-finite loss"):
        _mean_loss([0.4, math.nan, 0.2])


def test_mean_loss_accepts_finite_values() -> None:
    assert _mean_loss([0.3, 0.5, 0.4]) == pytest.approx(0.4)
