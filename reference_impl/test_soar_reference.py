import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np

from soar_reference import (
    LoRAFactors,
    compose_dense_maps,
    dense_delta_from_lora,
    factorize_dense_delta,
    frobenius_distance,
    residualize_dense_map,
    simplex_grid,
    tsvd_project,
)


def test_factor_gauge_invariance():
    rng = np.random.default_rng(7)
    A = rng.normal(size=(2, 3))
    B = rng.normal(size=(4, 2))
    Q = np.array([[2.0, 0.0], [0.0, 0.5]])
    A2 = np.linalg.inv(Q) @ A
    B2 = B @ Q
    d1 = dense_delta_from_lora(A, B, alpha=2.0)
    d2 = dense_delta_from_lora(A2, B2, alpha=2.0)
    assert np.allclose(d1, d2)


def test_toy_residual_recovery():
    delta_s = {"m": np.array([[1.0, 0.0], [0.0, 0.0]])}
    delta_f = {"m": np.array([[1.0, 0.0], [0.0, 2.0]])}
    residual = residualize_dense_map(delta_f, delta_s, beta=1.0, rank=1)
    composed = compose_dense_maps(delta_s, residuals={"J": residual}, alphas={"J": 1.0})
    assert np.allclose(composed["m"], delta_f["m"], atol=1e-8)


def test_factorize_dense_delta_round_trip():
    delta = np.array([[3.0, 0.0], [0.0, 1.0]])
    fac = factorize_dense_delta(delta, rank=2)
    recon = fac.dense_delta()
    assert np.allclose(delta, recon, atol=1e-8)


def test_tsvd_best_rank_one_shape():
    delta = np.array([[3.0, 0.0], [0.0, 1.0]])
    proj = tsvd_project(delta, rank=1)
    assert proj.shape == delta.shape
    assert np.linalg.matrix_rank(proj) == 1


def test_simplex_grid():
    grid = list(simplex_grid(2, step=0.5))
    assert (0.0, 1.0) in grid
    assert (0.5, 0.5) in grid
    assert (1.0, 0.0) in grid
    for item in grid:
        assert np.isclose(sum(item), 1.0)
        assert all(x >= 0 for x in item)
