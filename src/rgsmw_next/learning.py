from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, Sequence

import cvxpy as cp
import numpy as np
from scipy.optimize import minimize

from .types import FitResult, Preference, require_dimension


# A 1e-9 OSQP target made the archival 14k-pair fits hit 200k iterations
# without adding meaningful precision to reported metrics.  The frozen 1e-6
# target remains several orders below the declared margins/effect sizes, and
# raw residuals are retained in every fit manifest.
OSQP_EPSILON = 1e-6
OSQP_MAX_ITERATIONS = 100_000

# CLARABEL is the deterministic independent conic fallback for the quadratic
# programs below.  Its settings are frozen here (and copied into every fit
# manifest) so a solver switch caused by an OSQP iteration limit is visible and
# reproducible rather than an undocumented runtime accident.
CLARABEL_TOLERANCE = 1e-8
CLARABEL_MAX_ITERATIONS = 1_000

# A candidate solution is accepted only when CVXPY reports an optimal status,
# all requested variables are finite, and every model constraint is satisfied
# within five times the primary solver's absolute tolerance.  This is also the
# tolerance already used by _clean_simplex for active non-negativity bounds.
SOLVER_ACCEPTANCE_FEASIBILITY_TOLERANCE = 5.0 * OSQP_EPSILON
_ACCEPTED_SOLVER_STATUSES = frozenset({cp.OPTIMAL, cp.OPTIMAL_INACCURATE})


@dataclass(frozen=True)
class _SolverOutcome:
    value: float
    status: str
    solver: str
    attempts: tuple[dict[str, object], ...]
    maximum_constraint_violation: float
    constraint_violations: tuple[dict[str, object], ...]


class ConvexSolverFailure(RuntimeError):
    """Raised after every configured deterministic QP solver has failed.

    ``attempts`` is deliberately JSON-safe so an orchestration layer may write
    it directly to a failure manifest.
    """

    def __init__(self, label: str, attempts: Sequence[dict[str, object]]) -> None:
        self.label = label
        self.attempts = tuple(dict(attempt) for attempt in attempts)
        summary = "; ".join(
            f"{attempt['solver']}={attempt['status']}"
            for attempt in self.attempts
        )
        super().__init__(f"{label} optimization failed after {summary}")


def _matrix(preferences: Iterable[Preference], dimension: int) -> tuple[list[Preference], np.ndarray]:
    rows = list(preferences)
    if not rows:
        return rows, np.empty((0, dimension), dtype=float)
    for preference in rows:
        require_dimension(preference.delta, dimension)
    return rows, np.asarray([preference.delta for preference in rows], dtype=float)


def _solver_options(solver: str) -> dict[str, object]:
    """Return the frozen, JSON-safe options for a supported CVXPY solver."""
    normalized = solver.upper()
    if normalized == "OSQP":
        return {
            "eps_abs": OSQP_EPSILON,
            "eps_rel": OSQP_EPSILON,
            "max_iter": OSQP_MAX_ITERATIONS,
            "polishing": True,
            "warm_start": False,
            "verbose": False,
        }
    if normalized == "CLARABEL":
        return {
            "tol_gap_abs": CLARABEL_TOLERANCE,
            "tol_gap_rel": CLARABEL_TOLERANCE,
            "tol_feas": CLARABEL_TOLERANCE,
            "max_iter": CLARABEL_MAX_ITERATIONS,
            "warm_start": False,
            "verbose": False,
        }
    # Preserve support for an explicitly configured CVXPY solver without
    # guessing solver-specific tolerances.  The independent CLARABEL fallback
    # remains available when this primary solver fails.
    return {"warm_start": False, "verbose": False}


def _solver_statistics(problem: cp.Problem) -> dict[str, object]:
    """Snapshot JSON-safe CVXPY diagnostics before another solver is tried."""
    stats = problem.solver_stats
    if stats is None:
        return {}
    diagnostics: dict[str, object] = {
        "reported_solver": None if stats.solver_name is None else str(stats.solver_name),
        "solver_iterations": None if stats.num_iters is None else int(stats.num_iters),
        "solver_solve_time_seconds": (
            None if stats.solve_time is None else float(stats.solve_time)
        ),
        "solver_setup_time_seconds": (
            None if stats.setup_time is None else float(stats.setup_time)
        ),
    }
    extra = stats.extra_stats
    info = getattr(extra, "info", None)
    if info is not None:
        for output_name, attribute in (
            ("solver_primal_residual", "prim_res"),
            ("solver_dual_residual", "dual_res"),
            ("solver_status_code", "status_val"),
        ):
            value = getattr(info, attribute, None)
            if value is not None:
                diagnostics[output_name] = (
                    int(value) if output_name == "solver_status_code" else float(value)
                )
    return diagnostics


def _constraint_diagnostics(
    constraints: Sequence[cp.Constraint],
) -> tuple[float | None, tuple[dict[str, object], ...]]:
    """Measure each CVXPY constraint's maximum post-solve violation."""
    rows: list[dict[str, object]] = []
    maxima: list[float] = []
    for index, constraint in enumerate(constraints):
        row: dict[str, object] = {
            "constraint_index": index,
            "constraint_type": type(constraint).__name__,
        }
        try:
            values = np.asarray(constraint.violation(), dtype=float)
            if values.size == 0:
                maximum = 0.0
            elif not bool(np.all(np.isfinite(values))):
                maximum = None
            else:
                maximum = float(np.max(np.maximum(values, 0.0)))
        except (ValueError, TypeError) as error:
            maximum = None
            row["diagnostic_error"] = f"{type(error).__name__}: {error}"
        row["maximum_violation"] = maximum
        rows.append(row)
        if maximum is None:
            return None, tuple(rows)
        maxima.append(maximum)
    return (max(maxima, default=0.0), tuple(rows))


def _solve_qp_with_fallback(
    problem: cp.Problem,
    *,
    primary_solver: str,
    required_variables: Sequence[cp.Variable],
    label: str,
) -> _SolverOutcome:
    """Solve a QP with OSQP (or the configured primary), then CLARABEL.

    Non-optimal statuses such as OSQP's ``user_limit``, solver exceptions,
    absent/non-finite decision variables, non-finite objectives, and materially
    infeasible solutions all trigger the independent fallback.  Every attempt
    is retained for the returned fit manifest.
    """
    primary = str(primary_solver).upper()
    attempt_order = [primary]
    if primary != "CLARABEL" and "CLARABEL" in cp.installed_solvers():
        attempt_order.append("CLARABEL")

    attempts: list[dict[str, object]] = []
    for solver in attempt_order:
        options = _solver_options(solver)
        attempt: dict[str, object] = {
            "attempt_index": len(attempts),
            "solver": solver,
            "options": dict(options),
        }
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                value = problem.solve(solver=solver, **options)
        except Exception as error:  # CVXPY wraps backend-specific exceptions.
            attempt.update(
                {
                    "status": "exception",
                    "accepted": False,
                    "rejection_reasons": ["solver_exception"],
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                    "warnings": [
                        {
                            "category": warning.category.__name__,
                            "message": str(warning.message),
                        }
                        for warning in caught_warnings
                    ],
                }
            )
            attempts.append(attempt)
            continue

        status = "none" if problem.status is None else str(problem.status)
        objective_value = (
            None
            if value is None or not np.isfinite(float(value))
            else float(value)
        )
        variables_available = all(variable.value is not None for variable in required_variables)
        variables_finite = variables_available and all(
            bool(np.all(np.isfinite(np.asarray(variable.value, dtype=float))))
            for variable in required_variables
        )
        maximum_violation, constraint_rows = _constraint_diagnostics(problem.constraints)
        rejection_reasons: list[str] = []
        if status not in _ACCEPTED_SOLVER_STATUSES:
            rejection_reasons.append("non_optimal_status")
        if objective_value is None:
            rejection_reasons.append("non_finite_or_missing_objective")
        if not variables_available:
            rejection_reasons.append("missing_required_variable")
        elif not variables_finite:
            rejection_reasons.append("non_finite_required_variable")
        if maximum_violation is None:
            rejection_reasons.append("constraint_violation_unavailable")
        elif maximum_violation > SOLVER_ACCEPTANCE_FEASIBILITY_TOLERANCE:
            rejection_reasons.append("constraint_violation_exceeds_tolerance")

        attempt.update(
            {
                "status": status,
                "objective": objective_value,
                "required_variables_available": variables_available,
                "required_variables_finite": variables_finite,
                "maximum_constraint_violation": maximum_violation,
                "constraint_violations": [dict(row) for row in constraint_rows],
                "accepted": not rejection_reasons,
                "rejection_reasons": rejection_reasons,
                "warnings": [
                    {
                        "category": warning.category.__name__,
                        "message": str(warning.message),
                    }
                    for warning in caught_warnings
                ],
                **_solver_statistics(problem),
            }
        )
        attempts.append(attempt)
        if not rejection_reasons:
            stats = problem.solver_stats
            actual_solver = (
                solver
                if stats is None or stats.solver_name is None
                else str(stats.solver_name)
            )
            assert objective_value is not None
            assert maximum_violation is not None
            return _SolverOutcome(
                value=objective_value,
                status=status,
                solver=actual_solver,
                attempts=tuple(attempts),
                maximum_constraint_violation=maximum_violation,
                constraint_violations=constraint_rows,
            )

    raise ConvexSolverFailure(label, attempts)


def _fallback_metadata(
    outcome: _SolverOutcome, configured_primary_solver: str
) -> dict[str, object]:
    return {
        "configured_primary_solver": str(configured_primary_solver).upper(),
        "actual_solver": outcome.solver,
        "fallback_used": len(outcome.attempts) > 1,
        "solver_attempts": [dict(attempt) for attempt in outcome.attempts],
        "solver_acceptance_feasibility_tolerance": (
            SOLVER_ACCEPTANCE_FEASIBILITY_TOLERANCE
        ),
        "maximum_constraint_violation": outcome.maximum_constraint_violation,
        "constraint_violations": [dict(row) for row in outcome.constraint_violations],
    }


def _validate_cap(dimension: int, w_max: float | None) -> None:
    if w_max is not None and w_max < (1.0 / dimension) - 1e-12:
        raise ValueError(f"w_max={w_max} makes the {dimension}-dimensional simplex infeasible")


def _clean_simplex(solution: np.ndarray) -> np.ndarray:
    # OSQP's frozen 1e-6 feasibility target can return sub-micro negative
    # coordinates at an active nonnegativity boundary.  Values beyond five
    # solver tolerances remain a hard failure; smaller residuals are recorded
    # in metadata and projected exactly onto the simplex here.
    if np.min(solution) < -5.0 * OSQP_EPSILON:
        raise RuntimeError(f"Solver returned a materially negative weight: {np.min(solution)}")
    cleaned = np.maximum(solution, 0.0)
    total = float(np.sum(cleaned))
    if total <= 0:
        raise RuntimeError("Solver returned a zero weight vector")
    cleaned /= total
    if abs(float(np.sum(cleaned)) - 1.0) > 1e-8:
        raise RuntimeError("Unable to normalize the learned simplex weights")
    return cleaned


def _solver_metadata(problem: cp.Problem, raw_solution: np.ndarray) -> dict[str, object]:
    """Extract JSON-safe optimization diagnostics before any simplex cleanup."""
    diagnostics: dict[str, object] = {
        **_solver_statistics(problem),
        "raw_solution_minimum": float(np.min(raw_solution)),
        "raw_solution_maximum": float(np.max(raw_solution)),
    }
    return diagnostics


@dataclass(frozen=True)
class GlobalSimplexLearner:
    dimension: int
    c: float = 1.0
    l2: float = 1e-3
    margin: float = 0.05
    w_max: float | None = None
    solver: str = "OSQP"

    def fit(self, preferences: Iterable[Preference]) -> FitResult:
        _validate_cap(self.dimension, self.w_max)
        rows, delta = _matrix(preferences, self.dimension)
        if not rows:
            raise ValueError("At least one strict preference is required to learn the global prior")
        weights = cp.Variable(self.dimension)
        slack = cp.Variable(len(rows))
        constraints: list[cp.Constraint] = [
            delta @ weights >= self.margin - slack,
            slack >= 0,
            weights >= 0,
            cp.sum(weights) == 1,
        ]
        if self.w_max is not None:
            constraints.append(weights <= self.w_max)
        objective = cp.Minimize(
            (self.c / len(rows)) * cp.sum(slack) + (self.l2 / 2.0) * cp.sum_squares(weights)
        )
        problem = cp.Problem(objective, constraints)
        outcome = _solve_qp_with_fallback(
            problem,
            primary_solver=self.solver,
            required_variables=(weights, slack),
            label="Global",
        )
        assert weights.value is not None
        raw_weights = np.asarray(weights.value, dtype=float).reshape(-1)
        fitted = _clean_simplex(raw_weights)
        margins = delta @ fitted
        return FitResult(
            weights=tuple(float(item) for item in fitted),
            objective=outcome.value,
            status=outcome.status,
            solver=outcome.solver,
            n_preferences=len(rows),
            metadata={
                "minimum_margin": float(np.min(margins)),
                "mean_margin": float(np.mean(margins)),
                "constraint_violation_rate": float(np.mean(margins < self.margin - 1e-7)),
                "c": self.c,
                "l2": self.l2,
                "margin": self.margin,
                "w_max": self.w_max,
                "configured_eps_abs": OSQP_EPSILON,
                "configured_eps_rel": OSQP_EPSILON,
                "configured_max_iterations": OSQP_MAX_ITERATIONS,
                "raw_simplex_sum_residual": abs(float(np.sum(raw_weights)) - 1.0),
                "raw_simplex_nonnegativity_violation": max(0.0, -float(np.min(raw_weights))),
                "postprocessed_simplex_sum_residual": abs(float(np.sum(fitted)) - 1.0),
                "postprocessed_simplex_nonnegativity_violation": max(
                    0.0, -float(np.min(fitted))
                ),
                "postprocessed_cap_violation": (
                    0.0
                    if self.w_max is None
                    else max(0.0, float(np.max(fitted)) - self.w_max)
                ),
                **_fallback_metadata(outcome, self.solver),
                **_solver_metadata(problem, raw_weights),
            },
        )


@dataclass(frozen=True)
class GlobalLogisticSimplexLearner:
    """Convex pairwise-logistic learner without one slack variable per row.

    This is preferable for large archival pair sets.  SLSQP optimizes only the
    predicate weights under the same simplex constraints and exposes its full
    stopping diagnostics in the returned fit manifest.
    """

    dimension: int
    c: float = 1.0
    l2: float = 1e-3
    margin: float = 0.0
    max_iterations: int = 2_000
    tolerance: float = 1e-10

    def fit(self, preferences: Iterable[Preference]) -> FitResult:
        rows, delta = _matrix(preferences, self.dimension)
        if not rows:
            raise ValueError("At least one strict preference is required")
        if self.c < 0 or self.l2 < 0:
            raise ValueError("c and l2 must be non-negative")
        if self.max_iterations <= 0 or self.tolerance <= 0:
            raise ValueError("max_iterations and tolerance must be positive")

        def objective(weights: np.ndarray) -> float:
            residual = self.margin - delta @ weights
            return float(
                self.c * np.mean(np.logaddexp(0.0, residual))
                + 0.5 * self.l2 * np.dot(weights, weights)
            )

        def gradient(weights: np.ndarray) -> np.ndarray:
            residual = np.clip(self.margin - delta @ weights, -40.0, 40.0)
            probability = 1.0 / (1.0 + np.exp(-residual))
            return (
                -(self.c / len(rows)) * (delta.T @ probability)
                + self.l2 * weights
            )

        initial = np.full(self.dimension, 1.0 / self.dimension, dtype=float)
        result = minimize(
            objective,
            initial,
            jac=gradient,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * self.dimension,
            constraints=(
                {
                    "type": "eq",
                    "fun": lambda weights: float(np.sum(weights) - 1.0),
                    "jac": lambda weights: np.ones_like(weights),
                },
            ),
            options={
                "ftol": self.tolerance,
                "maxiter": self.max_iterations,
                "disp": False,
            },
        )
        if not result.success or result.x is None:
            raise RuntimeError(
                f"Logistic simplex optimization failed ({result.status}): {result.message}"
            )
        fitted = _clean_simplex(np.asarray(result.x, dtype=float))
        margins = delta @ fitted
        return FitResult(
            weights=tuple(float(value) for value in fitted),
            objective=float(result.fun),
            status="optimal",
            solver="scipy_slsqp",
            n_preferences=len(rows),
            metadata={
                "loss": "mean_pairwise_logistic",
                "c": self.c,
                "l2": self.l2,
                "margin_offset": self.margin,
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
                "gradient_evaluations": int(result.njev),
                "optimizer_status_code": int(result.status),
                "optimizer_message": str(result.message),
                "configured_tolerance": self.tolerance,
                "configured_max_iterations": self.max_iterations,
                "gradient_l2_at_solution": float(np.linalg.norm(gradient(fitted))),
                "simplex_sum_residual": abs(float(np.sum(fitted)) - 1.0),
                "minimum_weight": float(np.min(fitted)),
                "minimum_margin": float(np.min(margins)),
                "mean_margin": float(np.mean(margins)),
            },
        )


@dataclass(frozen=True)
class PersonalizedSimplexLearner:
    global_weights: Sequence[float]
    c: float = 1.0
    mu: float = 1.0
    margin: float = 0.05
    w_max: float | None = None
    solver: str = "OSQP"

    def fit(self, preferences: Iterable[Preference]) -> FitResult:
        prior = np.asarray(self.global_weights, dtype=float)
        dimension = prior.size
        require_dimension(prior, dimension)
        if np.min(prior) < -1e-10 or not np.isclose(np.sum(prior), 1.0, atol=1e-8):
            raise ValueError("The global prior must be a probability-simplex vector")
        _validate_cap(dimension, self.w_max)
        rows, delta = _matrix(preferences, dimension)
        if not rows:
            return FitResult(
                weights=tuple(float(item) for item in prior),
                objective=0.0,
                status="prior_only",
                solver="none",
                n_preferences=0,
                metadata={"mu": self.mu, "support_size": 0},
            )
        weights = cp.Variable(dimension)
        slack = cp.Variable(len(rows))
        constraints: list[cp.Constraint] = [
            delta @ weights >= self.margin - slack,
            slack >= 0,
            weights >= 0,
            cp.sum(weights) == 1,
        ]
        if self.w_max is not None:
            constraints.append(weights <= self.w_max)
        objective = cp.Minimize(
            (self.c / len(rows)) * cp.sum(slack) + (self.mu / 2.0) * cp.sum_squares(weights - prior)
        )
        problem = cp.Problem(objective, constraints)
        outcome = _solve_qp_with_fallback(
            problem,
            primary_solver=self.solver,
            required_variables=(weights, slack),
            label="Personalized",
        )
        assert weights.value is not None
        raw_weights = np.asarray(weights.value, dtype=float).reshape(-1)
        fitted = _clean_simplex(raw_weights)
        margins = delta @ fitted
        return FitResult(
            weights=tuple(float(item) for item in fitted),
            objective=outcome.value,
            status=outcome.status,
            solver=outcome.solver,
            n_preferences=len(rows),
            metadata={
                "minimum_margin": float(np.min(margins)),
                "mean_margin": float(np.mean(margins)),
                "distance_from_global_l2": float(np.linalg.norm(fitted - prior)),
                "c": self.c,
                "mu": self.mu,
                "margin": self.margin,
                "w_max": self.w_max,
                "support_size": len(rows),
                "configured_eps_abs": OSQP_EPSILON,
                "configured_eps_rel": OSQP_EPSILON,
                "configured_max_iterations": OSQP_MAX_ITERATIONS,
                "raw_simplex_sum_residual": abs(float(np.sum(raw_weights)) - 1.0),
                "raw_simplex_nonnegativity_violation": max(0.0, -float(np.min(raw_weights))),
                "postprocessed_simplex_sum_residual": abs(float(np.sum(fitted)) - 1.0),
                "postprocessed_simplex_nonnegativity_violation": max(
                    0.0, -float(np.min(fitted))
                ),
                "postprocessed_cap_violation": (
                    0.0
                    if self.w_max is None
                    else max(0.0, float(np.max(fitted)) - self.w_max)
                ),
                **_fallback_metadata(outcome, self.solver),
                **_solver_metadata(problem, raw_weights),
            },
        )


@dataclass(frozen=True)
class UnconstrainedRankSVM:
    dimension: int
    c: float = 1.0
    l2: float = 1e-3
    margin: float = 0.05
    anchor: Sequence[float] | None = None
    anchor_strength: float = 0.0
    solver: str = "OSQP"

    def fit(self, preferences: Iterable[Preference]) -> FitResult:
        rows, delta = _matrix(preferences, self.dimension)
        if not rows:
            if self.anchor is None:
                raise ValueError("RankSVM requires preferences when no anchor is supplied")
            anchor = np.asarray(self.anchor, dtype=float)
            require_dimension(anchor, self.dimension)
            return FitResult(
                weights=tuple(float(item) for item in anchor),
                objective=0.0,
                status="anchor_only",
                solver="none",
                n_preferences=0,
            )
        beta = cp.Variable(self.dimension)
        slack = cp.Variable(len(rows))
        constraints = [delta @ beta >= self.margin - slack, slack >= 0]
        penalty = (self.l2 / 2.0) * cp.sum_squares(beta)
        if self.anchor is not None and self.anchor_strength > 0:
            anchor = np.asarray(self.anchor, dtype=float)
            require_dimension(anchor, self.dimension)
            penalty += (self.anchor_strength / 2.0) * cp.sum_squares(beta - anchor)
        objective = cp.Minimize((self.c / len(rows)) * cp.sum(slack) + penalty)
        problem = cp.Problem(objective, constraints)
        outcome = _solve_qp_with_fallback(
            problem,
            primary_solver=self.solver,
            required_variables=(beta, slack),
            label="RankSVM",
        )
        assert beta.value is not None
        fitted = np.asarray(beta.value, dtype=float).reshape(-1)
        return FitResult(
            weights=tuple(float(item) for item in fitted),
            objective=outcome.value,
            status=outcome.status,
            solver=outcome.solver,
            n_preferences=len(rows),
            metadata={
                "weight_l2": float(np.linalg.norm(fitted)),
                "negative_weight_count": int(np.sum(fitted < 0)),
                "c": self.c,
                "l2": self.l2,
                "margin": self.margin,
                "anchor_strength": self.anchor_strength,
                "configured_eps_abs": OSQP_EPSILON,
                "configured_eps_rel": OSQP_EPSILON,
                "configured_max_iterations": OSQP_MAX_ITERATIONS,
                **_fallback_metadata(outcome, self.solver),
                **_solver_metadata(problem, fitted),
            },
        )


def pairwise_accuracy(weights: Sequence[float], preferences: Iterable[Preference]) -> float:
    rows = list(preferences)
    if not rows:
        return float("nan")
    vector = np.asarray(weights, dtype=float)
    predictions = [float(np.dot(vector, preference.delta)) for preference in rows]
    return float(np.mean(np.asarray(predictions) > 0.0))


def identifiability_report(
    preferences: Iterable[Preference],
    dimension: int,
    relative_tolerance: float = 1e-8,
    absolute_tolerance: float = 1e-10,
) -> dict[str, object]:
    rows, delta = _matrix(preferences, dimension)
    target_rank = max(0, dimension - 1)
    if not rows:
        return {
            "n_preferences": 0,
            "dimension": dimension,
            "rank": 0,
            "rank_on_simplex_tangent": 0,
            "target_rank": target_rank,
            "full_tangent_rank": target_rank == 0,
            "margin_map_injective": target_rank == 0,
            "identifiable_on_simplex": False,
            "singular_values": [],
            "condition_number_nonzero": float("inf"),
            "null_directions": [],
            "diagnostic_scope": "exact_margin_map_not_binary_label_identification",
        }

    # Work in an orthonormal basis of {v : 1^T v = 0}.  Projecting in the
    # ambient space and reading Vh directly would always return the simplex
    # normal as a null vector, even when every feasible tangent direction is
    # observed.  QR gives a stable (dimension x dimension-1) tangent basis.
    if dimension <= 1:
        tangent_basis = np.empty((dimension, 0), dtype=float)
    else:
        contrasts = np.eye(dimension, dtype=float)[:, :-1] - np.eye(
            dimension, dtype=float
        )[:, [-1]]
        tangent_basis, _ = np.linalg.qr(contrasts, mode="reduced")
    tangent_design = delta @ tangent_basis
    _, singular_values, vh = np.linalg.svd(tangent_design, full_matrices=True)
    relative_cutoff = relative_tolerance * singular_values[0] if singular_values.size else 0.0
    tolerance = max(absolute_tolerance, float(relative_cutoff))
    rank = int(np.sum(singular_values > tolerance))
    nonzero = singular_values[singular_values > tolerance]
    condition = float(nonzero[0] / nonzero[-1]) if nonzero.size else float("inf")
    null_coordinates = vh[rank:] if rank < vh.shape[0] else np.empty((0, target_rank))
    null_directions = (null_coordinates @ tangent_basis.T).tolist()
    column_variance = np.var(delta, axis=0).tolist()
    full_tangent_rank = rank >= target_rank
    return {
        "n_preferences": len(rows),
        "dimension": dimension,
        "rank_on_simplex_tangent": rank,
        "target_rank": target_rank,
        "full_tangent_rank": full_tangent_rank,
        "margin_map_injective": full_tangent_rank,
        # Retained for compatibility with frozen artifacts.  Binary preference
        # signs do not, by themselves, identify a weight vector.
        "identifiable_on_simplex": full_tangent_rank,
        "diagnostic_scope": "exact_margin_map_not_binary_label_identification",
        "relative_tolerance": relative_tolerance,
        "configured_absolute_tolerance": absolute_tolerance,
        "absolute_tolerance": float(tolerance),
        "singular_values": singular_values.tolist(),
        "condition_number_nonzero": condition,
        "column_variance": column_variance,
        "null_directions": null_directions,
    }
