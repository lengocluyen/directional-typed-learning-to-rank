"""Finite-audited utility learners for matched pair/choice experiments.

The linear objective is mean likelihood/hinge loss plus ridge, disjoint
property-group and anchor penalties. TD additionally restricts weights to the
simplex. Consequently its group penalty is constant, not a sparsity penalty.
CatBoost is a separate PairLogit *item utility* control: its leaf penalty is
not asserted equivalent to linear ridge and its unavailable group/anchor
penalties are rejected rather than silently ignored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import cvxpy as cp
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logsumexp

from dtltr_vehicle.learners import _fista, _prox_group
from .features import MODELS


def penalty_grid(settings: Mapping[str, object], model: str, adaptation: bool = False) -> list[dict[str, float]]:
    """Return the declared grid, eliminating mathematically meaningless cells."""
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}")
    grids = settings.get("penalties", settings.get("grids", settings.get("grid", settings)))
    if not isinstance(grids, Mapping):
        raise ValueError("Grid settings must be a mapping")

    def values(names, default):
        for name in names:
            if name in grids:
                value = grids[name]
                return list(value) if isinstance(value, (list, tuple)) else [value]
        return default

    lam = values(("lambda_grid", "lam_grid", "lambda", "lam", "l2"), [1e-4, 1e-2, 1.0])
    gamma = [0.0] if model in {"TD", "catboost_ref"} else values(("gamma_grid", "gamma", "group"), [0.0])
    mu = [0.0] if not adaptation or model == "catboost_ref" else values(("mu_grid", "mu", "anchor"), [0.1, 1.0, 10.0])
    if model == "catboost_ref":
        cfg = settings.get("catboost", {})
        if isinstance(cfg, Mapping):
            lam = cfg.get("l2_leaf_reg_grid", cfg.get("l2_leaf_reg", lam))
            lam = list(lam) if isinstance(lam, (list, tuple)) else [lam]
    out = []
    for l in lam:
        for g in gamma:
            for m in mu:
                p = {"lam": float(l), "gamma": float(g), "mu": float(m)}
                _penalty(p)
                if p not in out:
                    out.append(p)
    if not out:
        raise ValueError("Penalty grid must not be empty")
    return out


def _penalty(penalty: Mapping[str, float]) -> tuple[float, float, float]:
    values = (float(penalty.get("lam", penalty.get("lambda", 0.0))), float(penalty.get("gamma", 0.0)), float(penalty.get("mu", 0.0)))
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError("Penalty parameters must be finite and non-negative")
    return values


def _validate(X, groups, ndim, anchor):
    array = np.asarray(X, dtype=float)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"Expected finite {ndim}-dimensional feature data")
    d = array.shape[-1]
    group_list = [(int(a), int(b)) for a, b in groups]
    covered = [i for a, b in group_list for i in range(a, b)]
    if sorted(covered) != list(range(d)) or any(a < 0 or b <= a or b > d for a, b in group_list):
        raise ValueError("Groups must partition all feature coordinates exactly once")
    av = np.zeros(d) if anchor is None else np.asarray(anchor, dtype=float)
    if av.shape != (d,) or not np.all(np.isfinite(av)):
        raise ValueError("Anchor dimension/finite check failed")
    return array, group_list, av


def _simplex_project(value):
    ordered = np.sort(value)[::-1]
    cumulative = np.cumsum(ordered)-1
    index = np.arange(1,len(value)+1)
    eligible = np.nonzero(ordered-cumulative/index > 0)[0]
    rho = int(eligible[-1])
    return np.maximum(value-cumulative[rho]/(rho+1),0)


def _polish_simplex(smooth, start, lipschitz, target, max_iterations):
    """Solve the unchanged smooth simplex objective with audited projections.

    ``lipschitz`` is a valid Hessian upper bound, so each projected step has
    the standard convex descent guarantee. Unlike SLSQP's objective-change
    termination, stopping here requires the existing unit-step stationarity
    mapping to be small. The objective includes ridge and the original anchor.
    """
    theta = _simplex_project(np.asarray(start, dtype=float))
    lipschitz = max(float(lipschitz), np.finfo(float).eps)
    if not math.isfinite(lipschitz):
        raise RuntimeError("Simplex Lipschitz bound is non-finite")
    iterations = 0
    while True:
        _, gradient = smooth(theta)
        residual = float(np.linalg.norm(theta-_simplex_project(theta-gradient)))
        if residual <= target or iterations >= max_iterations:
            break
        theta = _simplex_project(theta-gradient/lipschitz)
        iterations += 1
    return theta, {"iterations": iterations, "stationarity_residual": residual,
                   "target": target, "target_met": residual <= target, "lipschitz_bound": lipschitz}


@dataclass
class Fit:
    theta: np.ndarray | None
    info: dict[str, object]
    estimator: object | None = field(default=None, repr=False)

    def score(self, X: np.ndarray) -> np.ndarray:
        array = np.asarray(X, dtype=float)
        if array.ndim < 1 or not np.all(np.isfinite(array)):
            raise ValueError("Score features must be finite")
        if self.theta is not None:
            if array.shape[-1] != len(self.theta):
                raise ValueError("Score feature dimension mismatch")
            out = array @ self.theta
        else:
            shape = array.shape[:-1]
            out = np.asarray(self.estimator.predict(array.reshape(-1, array.shape[-1])), dtype=float).reshape(shape)
        if not np.all(np.isfinite(out)):
            raise RuntimeError("Model produced non-finite utilities")
        return out

    def contributions(self, D: np.ndarray, encoder):
        """Exact additive utility contributions; unavailable for tree control."""
        if self.theta is None:
            return None
        array = np.asarray(D, dtype=float)
        if array.ndim == 1:
            return encoder.contributions(array, self.theta)
        if array.ndim != 2:
            raise ValueError("Contribution data must be a feature vector or matrix")
        return [encoder.contributions(row, self.theta) for row in array]


def _linear_fit(X, y, penalty, groups, anchor, model, config, *, multinomial):
    X, groups, av = _validate(X, groups, 3 if multinomial else 2, anchor)
    lam, gamma, mu_declared = _penalty(penalty)
    mu = mu_declared if anchor is not None else 0.0
    n, d = X.shape[0], X.shape[-1]
    simplex = model == "TD"
    loss = str(config.get("loss", "logistic"))
    if loss not in {"logistic", "hinge"} or (multinomial and loss != "logistic"):
        raise ValueError("Only pair logistic/hinge and multinomial logistic losses are supported")
    if simplex and any(b-a != 1 for a, b in groups):
        raise ValueError("TD simplex expects one negative distance coordinate per property")
    if simplex and anchor is not None and (np.min(av) < -1e-8 or abs(av.sum()-1) > 1e-6):
        raise ValueError("TD anchor must lie on the simplex")
    if n == 0:
        if anchor is None:
            raise ValueError("An empty fit requires an existing anchor")
        return Fit(av.copy(), {"status": "anchor_only", "converged": True, "rows": 0, "model": model})
    if multinomial:
        y = np.asarray(y)
        if y.shape != (n,) or not np.all(y == np.asarray(y, dtype=int)) or np.any(y < 0) or np.any(y >= X.shape[1]):
            raise ValueError("Chosen indices must be integral and in range")
        y = y.astype(int)
    start = av.copy() if anchor is not None else np.ones(d)/d if simplex else np.zeros(d)

    def smooth(theta):
        if multinomial:
            utilities = X @ theta
            norms = logsumexp(utilities, axis=1)
            probabilities = np.exp(utilities - norms[:, None])
            value = float(np.mean(norms - utilities[np.arange(n), y]))
            gradient = np.mean(np.einsum("nj,njd->nd", probabilities, X) - X[np.arange(n), y], axis=0)
        else:
            margins = X @ theta
            if loss == "hinge":
                value = float(np.mean(np.maximum(1-margins, 0)))
                gradient = -np.mean(X * (margins < 1)[:, None], axis=0)
            else:
                value = float(np.mean(np.logaddexp(0, -margins)))
                gradient = -(X.T @ expit(-margins))/n
        delta = theta - av
        value += lam/2 * float(theta @ theta) + mu/2 * float(delta @ delta)
        gradient = gradient + lam * theta + mu * delta
        return value, gradient

    def objective(theta):
        return smooth(theta)[0] + gamma * sum(float(np.linalg.norm(theta[a:b])) for a,b in groups)

    tolerance, max_iterations = float(config.get("tolerance", 1e-8)), int(config.get("max_iterations", 10000))
    if not math.isfinite(tolerance) or tolerance <= 0 or max_iterations < 1:
        raise ValueError("Solver tolerance must be positive and max_iterations at least one")

    def stationarity(theta):
        if loss == "hinge":
            return None  # solver certificate handles the nonsmooth data loss
        gradient = smooth(theta)[1]
        projected = _simplex_project(theta-gradient) if simplex else _prox_group(theta-gradient,gamma,groups)
        return float(np.linalg.norm(theta-projected)) / max(1.0,float(np.linalg.norm(theta)))

    fallback = False
    if simplex and loss == "logistic":
        # The group norm is constant on this simplex, so omit it from gradients.
        result = minimize(lambda t: smooth(t)[0], start, jac=lambda t: smooth(t)[1], method="SLSQP",
                          bounds=[(0, 1)]*d, constraints=[{"type": "eq", "fun": lambda t: t.sum()-1,
                                                                      "jac": lambda t: np.ones(d)}],
                          options={"ftol": min(tolerance, 1e-12), "maxiter": max_iterations})
        theta = np.asarray(result.x, dtype=float)
        info = {"status": "slsqp_simplex", "iterations": int(result.nit), "converged": bool(result.success), "solver_message": str(result.message)}
    elif gamma == 0 and loss == "logistic":
        result = minimize(lambda t: smooth(t), start, jac=True, method="L-BFGS-B",
                          options={"ftol": min(tolerance*0.01, 1e-12), "gtol": tolerance,
                                   "maxiter": max_iterations, "maxls": 50})
        theta = np.asarray(result.x, dtype=float)
        info = {"status": "lbfgs", "iterations": int(result.nit), "converged": bool(result.success), "solver_message": str(result.message)}
    elif loss == "logistic":
        theta, info = _fista(smooth, groups, gamma, start, max_iterations=max_iterations, tolerance=tolerance)
        info = {"status": "proximal_gradient", **info}
    else:
        theta, info = start, {"converged": False}
    residual = stationarity(theta) if np.all(np.isfinite(theta)) else math.inf
    audit_tolerance = float(config.get("stationarity_tolerance", 1e-4))
    if not math.isfinite(audit_tolerance) or audit_tolerance <= 0:
        raise ValueError("stationarity_tolerance must be finite and positive")
    if simplex and loss == "logistic" and np.all(np.isfinite(theta)):
        polish_target = min(1e-6, audit_tolerance*.1)
        if not info["converged"] or residual > polish_target:
            first_solver = {**info, "first_stationarity_residual": residual}
            if multinomial:
                # Softmax covariance has operator norm <= 1/2. Centering
                # alternatives removes utility offsets without changing it.
                centered = X-X.mean(axis=1, keepdims=True)
                bound = lam+mu+float(np.sum(centered*centered))/(2*n)
            else:
                # Binary logistic curvature <= 1/4; Frobenius norm bounds
                # the squared spectral norm of the difference design.
                bound = lam+mu+float(np.sum(X*X))/(4*n)
            theta, polish = _polish_simplex(smooth, theta, bound, polish_target, max_iterations)
            residual = stationarity(theta)
            info = {"status": "simplex_projected_gradient_polish", "converged": residual <= audit_tolerance,
                    "iterations": polish["iterations"], "first_solver": first_solver, "polish": polish}
            fallback = True
    if not info["converged"] or not np.all(np.isfinite(theta)) or (residual is not None and residual > audit_tolerance):
        info["first_stationarity_residual"] = residual
        fallback = True
        beta = cp.Variable(d)
        if multinomial:
            data_loss = sum(cp.log_sum_exp(X[i] @ beta) - X[i, y[i]] @ beta for i in range(n))/n
        elif loss == "hinge":
            data_loss = cp.sum(cp.pos(1-X@beta))/n
        else:
            data_loss = cp.sum(cp.logistic(-X@beta))/n
        total = data_loss + lam/2 * cp.sum_squares(beta)
        if gamma:
            total += gamma * sum(cp.norm(beta[a:b], 2) for a,b in groups)
        if mu:
            total += mu/2 * cp.sum_squares(beta-av)
        problem = cp.Problem(cp.Minimize(total), [beta >= 0, cp.sum(beta) == 1] if simplex else [])
        problem.solve(solver="CLARABEL", tol_gap_abs=1e-8, tol_gap_rel=1e-8, tol_feas=1e-8, max_iter=500)
        if beta.value is None or problem.status != cp.OPTIMAL:
            raise RuntimeError(f"Preference solver failed ({model}): {problem.status}; first solver {info}")
        theta = np.asarray(beta.value, dtype=float).reshape(-1)
        info = {"status": "clarabel", "converged": True, "iterations": int(problem.solver_stats.num_iters), "first_solver": info}
    if simplex:
        if np.min(theta) < -1e-6 or abs(theta.sum()-1) > 1e-6:
            raise RuntimeError("Simplex solution failed feasibility audit")
        theta = np.maximum(theta, 0)
        theta /= theta.sum()
    predictions = X @ theta
    value = objective(theta)
    if not np.all(np.isfinite(predictions)) or not math.isfinite(value):
        raise RuntimeError("Non-finite solution failed fit audit")
    residual = stationarity(theta)
    if residual is not None and residual > audit_tolerance:
        raise RuntimeError(f"Solution failed stationarity audit ({model}): residual={residual:.6g}, tolerance={audit_tolerance:.6g}")
    info.update({"model": model, "rows": n, "dimension": d, "objective": float(value), "finite": True,
                 "loss": "multinomial_logit" if multinomial else loss, "penalty": {"lam": lam, "gamma": gamma, "mu": mu},
                 "fallback": fallback, "simplex": simplex, "group_penalty_constant_on_simplex": simplex,
                 "gradient_norm": float(np.linalg.norm(smooth(theta)[1])), "coefficient_norm": float(np.linalg.norm(theta)),
                 "stationarity_residual": residual, "stationarity_tolerance": audit_tolerance,
                 "stationarity_definition": "relative unit-step proximal-gradient mapping; simplex projection for TD; solver certificate for hinge"})
    return Fit(theta, info)


def _catboost_pair(pair_items, penalty, seed, config):
    lam, gamma, mu = _penalty(penalty)
    if gamma or mu:
        raise ValueError("CatBoost has no matched linear group/anchor penalty; gamma and mu must be zero")
    items = np.asarray(pair_items, dtype=float)
    if items.ndim != 3 or items.shape[1] != 2 or not len(items) or not np.all(np.isfinite(items)):
        raise ValueError("CatBoost pair_items must be finite preferred/nonpreferred rows (N,2,d)")
    try:
        from catboost import CatBoostRanker, Pool
    except ImportError as exc:
        raise RuntimeError("catboost_ref requires optional dependency: pip install '.[modern-baselines]'") from exc
    nested = config.get("catboost", {})
    settings = {**nested, **{k: v for k,v in config.items() if k in {"iterations", "depth", "learning_rate", "thread_count", "task_type"}}}
    if "catboost_iterations" in config:
        settings["iterations"] = config["catboost_iterations"]
    estimator = CatBoostRanker(loss_function="PairLogit", iterations=int(settings.get("iterations", 150)),
                               depth=int(settings.get("depth", 4)), learning_rate=float(settings.get("learning_rate", 0.05)),
                               l2_leaf_reg=lam, random_seed=int(seed), thread_count=int(settings.get("thread_count", 1)),
                               task_type=str(settings.get("task_type", "CPU")), verbose=False, allow_writing_files=False)
    n, _, d = items.shape
    data = items.reshape(2*n, d)
    pool = Pool(data, label=np.tile([1,0], n), group_id=np.repeat(np.arange(n),2), pairs=[(2*i,2*i+1) for i in range(n)])
    estimator.fit(pool)
    values = np.asarray(estimator.predict(data), dtype=float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("CatBoost utility fit failed finite audit")
    margins = values[::2] - values[1::2]
    return Fit(None, {"status": "catboost_pairlogit", "model": "catboost_ref", "converged": True, "finite": True,
                      "convergence_definition": "completed declared boosting iterations; no convex stationarity claim",
                      "loss": "PairLogit", "rows": n, "dimension": d, "iterations": int(estimator.tree_count_),
                      "training_pair_log_loss": float(np.mean(np.logaddexp(0,-margins))),
                      "penalty": {"lam": lam, "gamma": 0.0, "mu": 0.0}, "lambda_meaning": "l2_leaf_reg; not equivalent to linear ridge",
                      "settings": settings, "exact_additive_contributions": False}, estimator)


def fit_pair(D, penalty, groups, anchor=None, model="DT", seed=1, config=None) -> Fit:
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}")
    config = {} if config is None else dict(config)
    if model == "catboost_ref":
        if anchor is not None:
            raise ValueError("CatBoost anchored adaptation is not supported by the matched linear procedure")
        if "pair_items" not in config:
            raise ValueError("CatBoost item utility requires config['pair_items']; feature differences alone are insufficient")
        array, _, _ = _validate(D, groups, 2, None)
        items = np.asarray(config["pair_items"], dtype=float)
        if items.shape != (len(array), 2, array.shape[1]) or not np.allclose(items[:,0]-items[:,1], array):
            raise ValueError("CatBoost pair_items must reconstruct the supplied preferred-minus-nonpreferred differences")
        return _catboost_pair(items, penalty, seed, config)
    return _linear_fit(D, None, penalty, groups, anchor, model, config, multinomial=False)


def fit_multinomial(X, y, penalty, groups, anchor=None, model="DT", seed=1, config=None) -> Fit:
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}")
    config = {} if config is None else dict(config)
    if model == "catboost_ref":
        if anchor is not None:
            raise ValueError("CatBoost anchored adaptation is not supported")
        array, _, _ = _validate(X, groups, 3, None)
        original = np.asarray(y)
        chosen = np.asarray(y, dtype=int)
        if chosen.shape != (len(array),) or not np.all(chosen == original) or np.any(chosen < 0) or np.any(chosen >= array.shape[1]):
            raise ValueError("Chosen indices shape/range check failed")
        pairs = np.stack([np.stack([items[win], items[lose]]) for items, win in zip(array, chosen, strict=True) for lose in range(array.shape[1]) if lose != win])
        fit = _catboost_pair(pairs, penalty, seed, config)
        fit.info.update({"choice_sets": len(array), "alternatives": array.shape[1],
                         "objective_difference": "PairLogit over chosen-vs-other alternatives; linear controls fit multinomial likelihood"})
        return fit
    return _linear_fit(X, y, penalty, groups, anchor, model, config, multinomial=True)


def fit_temperature(margins: np.ndarray) -> float:
    """Tune a positive utility multiplier on validation margins only."""
    values = np.asarray(margins, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("Pair calibration needs a finite vector of margins")
    if not len(values) or not np.any(np.abs(values)>1e-12):
        return 1.0
    result = minimize_scalar(lambda t: float(np.mean(np.logaddexp(0,-math.exp(t)*values))),
                             bounds=(-7,7), method="bounded", options={"xatol":1e-8})
    if not result.success or not math.isfinite(result.fun):
        raise RuntimeError("Pair calibration optimisation failed")
    return float(math.exp(result.x))


def pairwise_metrics(margins: np.ndarray, temperature: float = 1.0) -> dict[str, float]:
    values = np.asarray(margins, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Pair metrics need finite nonempty margins and a positive temperature")
    ties = np.abs(values)<=1e-12
    scaled = temperature*values
    return {"accuracy":float(np.mean(np.where(ties,.5,values>0))),
            "log_loss":float(np.mean(np.logaddexp(0,-scaled))),
            "brier":float(np.mean((1-expit(scaled))**2)), "tie_rate":float(np.mean(ties))}


def fit_choice_temperature(utilities: np.ndarray, y: np.ndarray) -> float:
    scores, chosen = np.asarray(utilities, dtype=float), np.asarray(y, dtype=int)
    if scores.ndim != 2 or chosen.shape != (len(scores),) or not np.all(np.isfinite(scores)):
        raise ValueError("Choice calibration needs finite utility matrix and aligned choices")
    if not len(scores) or np.all(np.ptp(scores, axis=1) <= 1e-12):
        return 1.0
    if np.any(chosen < 0) or np.any(chosen >= scores.shape[1]):
        raise ValueError("Choice calibration index outside alternative range")
    result = minimize_scalar(lambda t: float(np.mean(logsumexp(math.exp(t)*scores, axis=1) - math.exp(t)*scores[np.arange(len(scores)),chosen])),
                             bounds=(-7,7), method="bounded", options={"xatol": 1e-8})
    if not result.success or not math.isfinite(result.fun):
        raise RuntimeError("Choice calibration optimisation failed")
    return float(math.exp(result.x))


def choice_metrics(utilities: np.ndarray, y: np.ndarray, temperature: float = 1.0) -> dict[str, float]:
    scores, chosen = np.asarray(utilities, dtype=float), np.asarray(y, dtype=int)
    if scores.ndim != 2 or not len(scores) or chosen.shape != (len(scores),) or np.any(chosen < 0) or np.any(chosen >= scores.shape[1]):
        raise ValueError("Choice metrics need nonempty scores and aligned in-range choices")
    if not np.all(np.isfinite(scores)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Choice scores and positive temperature must be finite")
    scaled = temperature * scores
    norms = logsumexp(scaled, axis=1)
    probability = np.exp(scaled - norms[:,None])
    ties = np.isclose(scores, scores.max(axis=1, keepdims=True), rtol=0, atol=1e-12)
    correct = ties[np.arange(len(scores)), chosen] / ties.sum(axis=1)
    target = np.eye(scores.shape[1])[chosen]
    return {"accuracy": float(np.mean(correct)), "log_loss": float(np.mean(norms-scaled[np.arange(len(scores)),chosen])),
            "brier": float(np.mean(np.sum((probability-target)**2, axis=1))), "tie_rate": float(np.mean(ties.sum(axis=1)>1))}
