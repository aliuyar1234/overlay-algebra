from __future__ import annotations

import numpy as np

from overlay_algebra.soar import (
    DenseDeltaFactors,
    LoRAFactors,
    compose_dense_maps,
    compose_factor_maps,
    dense_delta_from_lora,
    dense_factor_map,
    factorize_dense_delta,
    factorize_dense_factors,
    frobenius_distance,
    residualize_dense_map,
    residualize_factor_map,
    simplex_grid,
    tsvd_project,
)


def test_factor_gauge_invariance() -> None:
    rng = np.random.default_rng(7)
    A = rng.normal(size=(2, 3))
    B = rng.normal(size=(4, 2))
    Q = np.array([[2.0, 0.0], [0.0, 0.5]])
    A2 = np.linalg.inv(Q) @ A
    B2 = B @ Q
    d1 = dense_delta_from_lora(A, B, alpha=2.0)
    d2 = dense_delta_from_lora(A2, B2, alpha=2.0)
    assert np.allclose(d1, d2)


def test_toy_residual_recovery_dense() -> None:
    delta_s = {"m": np.array([[1.0, 0.0], [0.0, 0.0]])}
    delta_f = {"m": np.array([[1.0, 0.0], [0.0, 2.0]])}
    residual = residualize_dense_map(delta_f, delta_s, beta=1.0, rank=1)
    composed = compose_dense_maps(delta_s, residuals={"J": residual}, alphas={"J": 1.0})
    assert np.allclose(composed["m"], delta_f["m"], atol=1e-8)


def test_toy_residual_recovery_factorized() -> None:
    scaffold = {
        "m": LoRAFactors(
            A=np.array([[1.0, 0.0]]),
            B=np.array([[1.0], [0.0]]),
            alpha=1.0,
        )
    }
    overlay = {
        "m": LoRAFactors(
            A=np.array([[1.0, 0.0], [0.0, 1.0]]),
            B=np.array([[1.0, 0.0], [0.0, 2.0]]),
            alpha=2.0,
        )
    }
    scaffold_factors = dense_factor_map(scaffold)
    overlay_factors = dense_factor_map(overlay)
    residual = residualize_factor_map(overlay_factors, scaffold_factors, beta=1.0, rank=1)
    composed = compose_factor_maps(scaffold_factors, residuals={"J": residual}, alphas={"J": 1.0})
    assert np.allclose(composed["m"].dense_delta(), overlay["m"].dense_delta(), atol=1e-8)


def test_factorize_dense_delta_round_trip() -> None:
    delta = np.array([[3.0, 0.0], [0.0, 1.0]])
    factors = factorize_dense_delta(delta, rank=2)
    assert np.allclose(delta, factors.dense_delta(), atol=1e-8)


def test_factorize_dense_factors_round_trip_with_padding() -> None:
    factors = DenseDeltaFactors(
        left=np.array([[2.0], [0.0]]),
        right=np.array([[3.0, 0.0]]),
    )
    lora = factorize_dense_factors(factors, rank=2, pad_to_rank=True)
    assert lora.A.shape == (2, 2)
    assert lora.B.shape == (2, 2)
    assert np.allclose(factors.dense_delta(), lora.dense_delta(), atol=1e-8)


def test_tsvd_best_rank_one_shape() -> None:
    delta = np.array([[3.0, 0.0], [0.0, 1.0]])
    proj = tsvd_project(delta, rank=1)
    assert proj.shape == delta.shape
    assert np.linalg.matrix_rank(proj) == 1


def test_simplex_grid() -> None:
    grid = list(simplex_grid(2, step=0.5))
    assert (0.0, 1.0) in grid
    assert (0.5, 0.5) in grid
    assert (1.0, 0.0) in grid
    for item in grid:
        assert np.isclose(sum(item), 1.0)
        assert all(x >= 0 for x in item)


def test_frobenius_distance_zero_for_identical_matrices() -> None:
    delta = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert frobenius_distance(delta, delta) == 0.0
