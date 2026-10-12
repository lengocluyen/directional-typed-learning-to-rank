"""Convex linear preference learners with ridge, property-group and anchor penalties.

Objective for pairwise data with difference rows D (preferred minus non-preferred):

    mean loss(D theta) + (lam/2)||theta||^2 + gamma * sum_g ||theta_g||_2 + (mu/2)||theta - anchor||^2

with loss = log(1 + exp(-m)) (logistic) or max(0, 1 - m) (hinge, unit margin).
The conditional-logit variant replaces the pairwise loss by the multinomial
negative log-likelihood of the chosen alternative.

Smooth losses use accelerated proximal gradient with backtracking and restart
(block soft-thresholding is the exact proximal map of the group penalty). The
hinge loss uses CLARABEL through CVXPY.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import cvxpy as cp
import numpy as np
from scipy.optimize import minimize_scalar

Groups = Sequence[tuple[int, int]]


@dataclass(frozen=True)
class Penalty:
    lam: float
    gamma: float = 0.0
    mu: float = 0.0

    def __post_init__(self) -> None:
        if min(self.lam, self.gamma, self.mu) < 0:
            raise ValueError("Penalties must be non-negative")


def _group_norm_sum(theta: np.ndarray, groups: Groups) -> float:
    return float(sum(np.linalg.norm(theta[a:b]) for a, b in groups))


def _prox_group(v: np.ndarray, threshold: float, groups: Groups) -> np.ndarray:
    if threshold <= 0 or not groups:
        return v
    out = v.copy()
    for a, b in groups:
        norm = float(np.linalg.norm(v[a:b]))
        out[a:b] = 0.0 if norm <= threshold else v[a:b] * (1.0 - threshold / norm)
    return out


def _fista(
    smooth: Callable[[np.ndarray], tuple[float, np.ndarray]],
    groups: Groups,
    gamma: float,
    start: np.ndarray,
    *,
    max_iterations: int = 20_000,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, dict[str, object]]:
    def total(x: np.ndarray) -> float:
        return smooth(x)[0] + gamma * _group_norm_sum(x, groups)

    x = start.astype(float).copy()
    y = x.copy()
    t = 1.0
    lipschitz = 1.0
    f_x = total(x)
    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        f_y, g_y = smooth(y)
        while True:
            z = _prox_group(y - g_y / lipschitz, gamma / lipschitz, groups)
            diff = z - y
            f_z_smooth = smooth(z)[0]
            if f_z_smooth <= f_y + float(g_y @ diff) + 0.5 * lipschitz * float(diff @ diff) + 1e-14:
                break
            lipschitz *= 2.0
        f_z = f_z_smooth + gamma * _group_norm_sum(z, groups)
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        if f_z > f_x:  # monotone restart
            y, t = x.copy(), 1.0
            lipschitz *= 2.0
            continue
        step = float(np.linalg.norm(z - x))
        y = z + ((t - 1.0) / t_next) * (z - x)
        x, f_x, t = z, f_z, t_next
        lipschitz = max(lipschitz / 1.5, 1e-12)
        if step <= tolerance * max(1.0, float(np.linalg.norm(x))):
            converged = True
            break
    return x, {"iterations": iteration, "converged": converged, "objective": f_x}


def fit_pairwise(
    differences: np.ndarray,
    loss: str,
    penalty: Penalty,
    groups: Groups,
    *,
    anchor: np.ndarray | None = None,
    dimension: int | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit coefficients on pairwise difference rows (preferred minus non-preferred)."""
    rows = np.asarray(differences, dtype=float)
    d = int(dimension if dimension is not None else rows.shape[1])
    anchor_vec = np.zeros(d) if anchor is None else np.asarray(anchor, dtype=float)
    mu = penalty.mu if anchor is not None else 0.0
    if rows.size == 0:
        if anchor is None:
            raise ValueError("A fit without rows needs an anchor")
        return anchor_vec.copy(), {"status": "anchor_only", "rows": 0}
    n = rows.shape[0]
    if loss == "logistic":

        def smooth(theta: np.ndarray) -> tuple[float, np.ndarray]:
            margins = rows @ theta
            value = float(np.mean(np.logaddexp(0.0, -margins)))
            weights = 0.5 * (1.0 - np.tanh(0.5 * margins))  # sigmoid(-m), overflow-safe
            gradient = -(rows.T @ weights) / n
            value += 0.5 * penalty.lam * float(theta @ theta) + 0.5 * mu * float(
                (theta - anchor_vec) @ (theta - anchor_vec)
            )
            gradient = gradient + penalty.lam * theta + mu * (theta - anchor_vec)
            return value, gradient

        theta, info = _fista(smooth, groups, penalty.gamma, anchor_vec)
        return theta, {"status": "fista", "rows": n, **info}
    if loss == "hinge":
        beta = cp.Variable(d)
        objective = cp.sum(cp.pos(1.0 - rows @ beta)) / n + 0.5 * penalty.lam * cp.sum_squares(beta)
        if penalty.gamma > 0 and groups:
            objective = objective + penalty.gamma * cp.sum(cp.hstack([cp.norm(beta[a:b], 2) for a, b in groups]))
        if mu > 0:
            objective = objective + 0.5 * mu * cp.sum_squares(beta - anchor_vec)
        problem = cp.Problem(cp.Minimize(objective))
        problem.solve(solver="CLARABEL", tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9, max_iter=500)
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or beta.value is None:
            raise RuntimeError(f"Hinge fit failed with status {problem.status}")
        return np.asarray(beta.value, dtype=float).reshape(-1), {
            "status": str(problem.status),
            "rows": n,
            "objective": float(problem.value),
        }
    raise ValueError(f"Unknown loss {loss!r}")


def fit_choice(
    feature_sets: np.ndarray,
    chosen: np.ndarray,
    penalty: Penalty,
    groups: Groups,
    *,
    anchor: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Conditional logit on an (N, J, d) tensor with chosen indices of length N."""
    X = np.asarray(feature_sets, dtype=float)
    y = np.asarray(chosen, dtype=int)
    n, _, d = X.shape
    anchor_vec = np.zeros(d) if anchor is None else np.asarray(anchor, dtype=float)
    mu = penalty.mu if anchor is not None else 0.0
    rows = np.arange(n)

    def smooth(theta: np.ndarray) -> tuple[float, np.ndarray]:
        utilities = X @ theta
        top = utilities.max(axis=1, keepdims=True)
        log_norm = top[:, 0] + np.log(np.exp(utilities - top).sum(axis=1))
        value = float(np.mean(log_norm - utilities[rows, y]))
        probabilities = np.exp(utilities - log_norm[:, None])
        expected = np.einsum("nj,njd->nd", probabilities, X)
        gradient = (expected - X[rows, y]).mean(axis=0)
        value += 0.5 * penalty.lam * float(theta @ theta) + 0.5 * mu * float((theta - anchor_vec) @ (theta - anchor_vec))
        gradient = gradient + penalty.lam * theta + mu * (theta - anchor_vec)
        return value, gradient

    theta, info = _fista(smooth, groups, penalty.gamma, anchor_vec)
    return theta, {"status": "fista", "rows": n, **info}


# ----------------------------------------------------------------- evaluation
def fit_temperature(margins: np.ndarray) -> float:
    """One-parameter probability scale minimizing binary log loss of P = sigmoid(T m)."""
    values = np.asarray(margins, dtype=float)
    if values.size == 0 or not np.any(np.abs(values) > 1e-12):
        return 1.0

    def objective(log_t: float) -> float:
        return float(np.mean(np.logaddexp(0.0, -math.exp(log_t) * values)))

    result = minimize_scalar(objective, bounds=(-7.0, 7.0), method="bounded", options={"xatol": 1e-8})
    return float(math.exp(result.x))


def pairwise_metrics(margins: np.ndarray, temperature: float) -> dict[str, float]:
    values = np.asarray(margins, dtype=float)
    ties = np.abs(values) <= 1e-12
    scaled = np.clip(temperature * values, -40.0, 40.0)
    probability = 1.0 / (1.0 + np.exp(-scaled))
    return {
        "accuracy": float(np.mean(np.where(ties, 0.5, (values > 0).astype(float)))),
        "log_loss": float(np.mean(np.logaddexp(0.0, -scaled))),
        "brier": float(np.mean(np.square(1.0 - probability))),
        "tie_rate": float(np.mean(ties)),
    }
