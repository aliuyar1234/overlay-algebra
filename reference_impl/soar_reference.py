"""Reference implementation for the core SOAR math.

This file is clarity-first and intentionally small.
It is NOT production-ready training code.
Its purpose is to anchor the exact math in executable form so future
implementation does not invent or drift from the method.

Conventions:
- A has shape [r, d_in]
- B has shape [d_out, r]
- effective dense delta = (alpha / r) * (B @ A)
- when factorizing a dense delta back to LoRA form, this reference sets
  alpha = rank so that alpha / rank = 1 and B @ A equals the dense delta

Only NumPy is used here on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np


Array = np.ndarray
DeltaMap = Dict[str, Array]


@dataclass(frozen=True)
class LoRAFactors:
    A: Array  # [r, d_in]
    B: Array  # [d_out, r]
    alpha: float

    def dense_delta(self) -> Array:
        if self.A.ndim != 2 or self.B.ndim != 2:
            raise ValueError("A and B must be rank-2 arrays.")
        if self.B.shape[1] != self.A.shape[0]:
            raise ValueError("B.shape[1] must equal A.shape[0] (the LoRA rank).")
        rank = self.A.shape[0]
        return (self.alpha / rank) * (self.B @ self.A)


def dense_delta_from_lora(A: Array, B: Array, alpha: float) -> Array:
    return LoRAFactors(A=A, B=B, alpha=alpha).dense_delta()


def dense_delta_map(adapter: Mapping[str, LoRAFactors]) -> DeltaMap:
    return {name: factors.dense_delta() for name, factors in adapter.items()}


def tsvd_project(matrix: Array, rank: int) -> Array:
    if rank <= 0:
        raise ValueError("rank must be positive")
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
    rr = min(rank, len(s))
    if rr == 0:
        return np.zeros_like(matrix)
    return U[:, :rr] @ np.diag(s[:rr]) @ Vt[:rr, :]


def factorize_dense_delta(delta: Array, rank: int) -> LoRAFactors:
    if rank <= 0:
        raise ValueError("rank must be positive")
    U, s, Vt = np.linalg.svd(delta, full_matrices=False)
    rr = min(rank, len(s))
    if rr == 0:
        raise ValueError("cannot factorize a rank-0 matrix with positive rank")
    sqrt_s = np.sqrt(s[:rr])
    B = U[:, :rr] * sqrt_s[None, :]
    A = sqrt_s[:, None] * Vt[:rr, :]
    alpha = float(rr)  # so alpha / rank = 1
    return LoRAFactors(A=A, B=B, alpha=alpha)


def residualize_dense_map(
    delta_full: Mapping[str, Array],
    delta_scaffold: Mapping[str, Array],
    beta: float,
    rank: int,
) -> DeltaMap:
    if set(delta_full.keys()) != set(delta_scaffold.keys()):
        raise ValueError("delta_full and delta_scaffold must share identical module keys.")
    out: DeltaMap = {}
    for name in delta_full:
        gamma = delta_full[name] - beta * delta_scaffold[name]
        out[name] = tsvd_project(gamma, rank=rank)
    return out


def compose_dense_maps(
    delta_scaffold: Mapping[str, Array],
    residuals: Mapping[str, Mapping[str, Array]],
    alphas: Mapping[str, float],
) -> DeltaMap:
    names = set(delta_scaffold.keys())
    for key, rmap in residuals.items():
        if set(rmap.keys()) != names:
            raise ValueError(f"Residual map {key!r} does not match scaffold module keys.")
    out: DeltaMap = {name: np.array(delta_scaffold[name], copy=True) for name in names}
    for overlay_name, coeff in alphas.items():
        if overlay_name not in residuals:
            raise KeyError(f"Missing residual map for overlay {overlay_name!r}")
        for module_name in names:
            out[module_name] = out[module_name] + coeff * residuals[overlay_name][module_name]
    return out


def frobenius_distance(a: Array, b: Array) -> float:
    return float(np.linalg.norm(a - b))


def simplex_grid(k: int, step: float = 0.25):
    """Generate nonnegative simplex weights summing to 1 on a coarse grid.

    Example:
        list(simplex_grid(2, step=0.5))
        -> [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if step <= 0 or step > 1:
        raise ValueError("step must be in (0, 1]")

    grid_units = int(round(1.0 / step))
    if not np.isclose(grid_units * step, 1.0):
        raise ValueError("step must divide 1.0 exactly in this simple reference impl.")

    def rec(parts_left: int, units_left: int, prefix):
        if parts_left == 1:
            yield tuple(prefix + [units_left * step])
            return
        for u in range(units_left + 1):
            yield from rec(parts_left - 1, units_left - u, prefix + [u * step])

    yield from rec(k, grid_units, [])


def _demo():
    delta_s = {"m": np.array([[1.0, 0.0], [0.0, 0.0]])}
    delta_f = {"m": np.array([[1.0, 0.0], [0.0, 2.0]])}
    residual = residualize_dense_map(delta_f, delta_s, beta=1.0, rank=1)
    composed = compose_dense_maps(delta_s, residuals={"J": residual}, alphas={"J": 1.0})
    print("Toy recovery distance:", frobenius_distance(composed["m"], delta_f["m"]))


if __name__ == "__main__":
    _demo()
