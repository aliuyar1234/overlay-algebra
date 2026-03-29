from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


Array = np.ndarray
DenseMap = Dict[str, Array]


@dataclass(frozen=True, slots=True)
class LoRAFactors:
    A: Array
    B: Array
    alpha: float

    @property
    def rank(self) -> int:
        if self.A.ndim != 2 or self.B.ndim != 2:
            raise ValueError("A and B must both be rank-2 arrays.")
        if self.B.shape[1] != self.A.shape[0]:
            raise ValueError("B.shape[1] must equal A.shape[0].")
        return int(self.A.shape[0])

    def dense_delta(self) -> Array:
        rank = self.rank
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        return (self.alpha / rank) * (self.B @ self.A)

    def dense_factors(self) -> "DenseDeltaFactors":
        rank = self.rank
        scale = float(self.alpha) / float(rank)
        return DenseDeltaFactors(left=np.array(self.B * scale, copy=True), right=np.array(self.A, copy=True))


@dataclass(frozen=True, slots=True)
class DenseDeltaFactors:
    left: Array
    right: Array

    @property
    def rank_bound(self) -> int:
        if self.left.ndim != 2 or self.right.ndim != 2:
            raise ValueError("left and right must both be rank-2 arrays.")
        if self.left.shape[1] != self.right.shape[0]:
            raise ValueError("left.shape[1] must equal right.shape[0].")
        return int(self.left.shape[1])

    def dense_delta(self) -> Array:
        _ = self.rank_bound
        return self.left @ self.right


def dense_delta_from_lora(A: Array, B: Array, alpha: float) -> Array:
    return LoRAFactors(A=A, B=B, alpha=alpha).dense_delta()


def dense_delta_map(adapter: Mapping[str, LoRAFactors]) -> DenseMap:
    return {name: factors.dense_delta() for name, factors in adapter.items()}


def dense_factor_map(adapter: Mapping[str, LoRAFactors]) -> dict[str, DenseDeltaFactors]:
    return {name: factors.dense_factors() for name, factors in adapter.items()}


def _check_identical_module_keys(*maps: Mapping[str, object]) -> list[str]:
    if not maps:
        return []
    names = list(maps[0].keys())
    expected = set(names)
    for mapping in maps[1:]:
        if set(mapping.keys()) != expected:
            raise ValueError("All module maps must share identical keys.")
    return names


def _svd_from_product(factors: DenseDeltaFactors) -> tuple[Array, Array, Array]:
    _ = factors.rank_bound
    q_left, r_left = np.linalg.qr(factors.left, mode="reduced")
    q_right, r_right = np.linalg.qr(factors.right.T, mode="reduced")
    small = r_left @ r_right.T
    u_hat, singular_values, vt_hat = np.linalg.svd(small, full_matrices=False)
    left_vectors = q_left @ u_hat
    right_vectors_t = vt_hat @ q_right.T
    return left_vectors, singular_values, right_vectors_t


def tsvd_project(matrix: Array, rank: int) -> Array:
    if rank <= 0:
        raise ValueError("rank must be positive")
    U, singular_values, Vt = np.linalg.svd(matrix, full_matrices=False)
    rr = min(rank, len(singular_values))
    if rr == 0:
        return np.zeros_like(matrix)
    return U[:, :rr] @ np.diag(singular_values[:rr]) @ Vt[:rr, :]


def _pad_lora_factors(factors: LoRAFactors, target_rank: int) -> LoRAFactors:
    if target_rank <= 0:
        raise ValueError("target_rank must be positive")
    if factors.rank > target_rank:
        raise ValueError("Cannot pad LoRA factors to a smaller rank.")
    if factors.rank == target_rank:
        return factors

    d_out, _ = factors.B.shape
    _, d_in = factors.A.shape
    padded_a = np.zeros((target_rank, d_in), dtype=factors.A.dtype)
    padded_b = np.zeros((d_out, target_rank), dtype=factors.B.dtype)
    padded_a[: factors.rank, :] = factors.A
    padded_b[:, : factors.rank] = factors.B
    return LoRAFactors(A=padded_a, B=padded_b, alpha=float(target_rank))


def factorize_dense_delta(delta: Array, rank: int, *, pad_to_rank: bool = False) -> LoRAFactors:
    if rank <= 0:
        raise ValueError("rank must be positive")
    if delta.ndim != 2:
        raise ValueError("delta must be a rank-2 array")
    U, singular_values, Vt = np.linalg.svd(delta, full_matrices=False)
    return _factorize_from_svd(U, singular_values, Vt, rank, pad_to_rank=pad_to_rank)


def factorize_dense_factors(
    factors: DenseDeltaFactors,
    rank: int,
    *,
    pad_to_rank: bool = False,
) -> LoRAFactors:
    if rank <= 0:
        raise ValueError("rank must be positive")
    left_vectors, singular_values, right_vectors_t = _svd_from_product(factors)
    return _factorize_from_svd(left_vectors, singular_values, right_vectors_t, rank, pad_to_rank=pad_to_rank)


def _factorize_from_svd(
    U: Array,
    singular_values: Array,
    Vt: Array,
    target_rank: int,
    *,
    pad_to_rank: bool,
) -> LoRAFactors:
    rr = min(target_rank, len(singular_values))
    if rr == 0:
        if not pad_to_rank:
            raise ValueError("Cannot factorize a rank-0 matrix without padding.")
        d_out = U.shape[0]
        d_in = Vt.shape[1]
        zeros = np.zeros((target_rank, d_in), dtype=np.float32)
        return LoRAFactors(
            A=zeros,
            B=np.zeros((d_out, target_rank), dtype=np.float32),
            alpha=float(target_rank),
        )

    sqrt_s = np.sqrt(singular_values[:rr])
    B = U[:, :rr] * sqrt_s[None, :]
    A = sqrt_s[:, None] * Vt[:rr, :]
    factors = LoRAFactors(A=A, B=B, alpha=float(rr))
    if pad_to_rank:
        return _pad_lora_factors(factors, target_rank)
    return factors


def project_dense_factors(factors: DenseDeltaFactors, rank: int) -> DenseDeltaFactors:
    return factorize_dense_factors(factors, rank, pad_to_rank=False).dense_factors()


def combine_dense_factors(terms: Sequence[tuple[float, DenseDeltaFactors]]) -> DenseDeltaFactors:
    if not terms:
        raise ValueError("terms must not be empty")
    left_parts = [float(coeff) * factors.left for coeff, factors in terms]
    right_parts = [factors.right for _, factors in terms]
    return DenseDeltaFactors(
        left=np.concatenate(left_parts, axis=1),
        right=np.concatenate(right_parts, axis=0),
    )


def residualize_dense_map(
    delta_full: Mapping[str, Array],
    delta_scaffold: Mapping[str, Array],
    beta: float,
    rank: int,
) -> DenseMap:
    _check_identical_module_keys(delta_full, delta_scaffold)
    out: DenseMap = {}
    for name in delta_full:
        gamma = delta_full[name] - beta * delta_scaffold[name]
        out[name] = tsvd_project(gamma, rank=rank)
    return out


def residualize_factor_map(
    delta_full: Mapping[str, DenseDeltaFactors],
    delta_scaffold: Mapping[str, DenseDeltaFactors],
    beta: float,
    rank: int,
) -> dict[str, DenseDeltaFactors]:
    names = _check_identical_module_keys(delta_full, delta_scaffold)
    out: dict[str, DenseDeltaFactors] = {}
    for name in names:
        gamma = combine_dense_factors(((1.0, delta_full[name]), (-float(beta), delta_scaffold[name])))
        out[name] = project_dense_factors(gamma, rank)
    return out


def compose_dense_maps(
    delta_scaffold: Mapping[str, Array],
    residuals: Mapping[str, Mapping[str, Array]],
    alphas: Mapping[str, float],
) -> DenseMap:
    names = _check_identical_module_keys(delta_scaffold, *residuals.values())
    out: DenseMap = {name: np.array(delta_scaffold[name], copy=True) for name in names}
    for overlay_name, coeff in alphas.items():
        if overlay_name not in residuals:
            raise KeyError(f"Missing residual map for overlay {overlay_name!r}")
        for name in names:
            out[name] = out[name] + float(coeff) * residuals[overlay_name][name]
    return out


def compose_factor_maps(
    delta_scaffold: Mapping[str, DenseDeltaFactors],
    residuals: Mapping[str, Mapping[str, DenseDeltaFactors]],
    alphas: Mapping[str, float],
) -> dict[str, DenseDeltaFactors]:
    names = _check_identical_module_keys(delta_scaffold, *residuals.values())
    out: dict[str, DenseDeltaFactors] = {}
    for name in names:
        terms: list[tuple[float, DenseDeltaFactors]] = [(1.0, delta_scaffold[name])]
        for overlay_name, coeff in alphas.items():
            if overlay_name not in residuals:
                raise KeyError(f"Missing residual map for overlay {overlay_name!r}")
            terms.append((float(coeff), residuals[overlay_name][name]))
        out[name] = combine_dense_factors(terms)
    return out


def frobenius_distance(a: Array, b: Array) -> float:
    return float(np.linalg.norm(a - b))


def simplex_grid(k: int, step: float = 0.25) -> Iterable[tuple[float, ...]]:
    if k <= 0:
        raise ValueError("k must be positive")
    if step <= 0 or step > 1:
        raise ValueError("step must be in (0, 1]")

    grid_units = int(round(1.0 / step))
    if not np.isclose(grid_units * step, 1.0):
        raise ValueError("step must divide 1.0 exactly in this grid helper.")

    def rec(parts_left: int, units_left: int, prefix: list[float]) -> Iterable[tuple[float, ...]]:
        if parts_left == 1:
            yield tuple(prefix + [units_left * step])
            return
        for unit in range(units_left + 1):
            yield from rec(parts_left - 1, units_left - unit, prefix + [unit * step])

    yield from rec(k, grid_units, [])
