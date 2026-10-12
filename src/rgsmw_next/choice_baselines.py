from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from .distance import FittedMetric
from .preference_learning import PairwiseObservation
from .types import EntityRecord, Preference


@dataclass(frozen=True)
class FittedEntityFeatureEncoder:
    """Leakage-safe direct-utility encoding fitted on training entities only."""

    metric: FittedMetric
    columns: tuple[tuple[str, str], ...]
    numeric_centres: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        records: Iterable[EntityRecord],
        metric: FittedMetric,
    ) -> "FittedEntityFeatureEncoder":
        rows = list(records)
        columns: list[tuple[str, str]] = []
        centres: dict[str, float] = {}
        for predicate in metric.predicates:
            field_name = predicate.spec.field
            kind = predicate.spec.kind
            if kind in {"numeric", "count", "year", "date", "geographic"}:
                observed: list[float] = []
                for row in rows:
                    values = row.get(field_name)
                    if not values:
                        continue
                    value = values[0]
                    if hasattr(value, "toordinal"):
                        observed.append(float(value.toordinal()))  # type: ignore[union-attr]
                    elif isinstance(value, (int, float)):
                        observed.append(float(value))
                centres[field_name] = float(np.median(observed)) if observed else 0.0
                columns.extend(((field_name, "value"), (field_name, "missing")))
            elif kind == "ordinal":
                columns.extend(((field_name, "ordinal"), (field_name, "missing")))
            else:
                categories = sorted(
                    {
                        str(value)
                        for row in rows
                        for value in row.get(field_name)
                    }
                )
                columns.extend((field_name, f"value={value}") for value in categories)
                columns.append((field_name, "missing"))
        return cls(metric=metric, columns=tuple(columns), numeric_centres=centres)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(f"{field}:{component}" for field, component in self.columns)

    def transform(self, record: EntityRecord) -> np.ndarray:
        values: dict[tuple[str, str], float] = {}
        predicates = {predicate.spec.field: predicate for predicate in self.metric.predicates}
        for field_name, predicate in predicates.items():
            observed = record.get(field_name)
            kind = predicate.spec.kind
            if not observed:
                values[(field_name, "missing")] = 1.0
                continue
            if kind in {"numeric", "count", "year", "date"}:
                raw = observed[0]
                if hasattr(raw, "toordinal"):
                    numeric = float(raw.toordinal())  # type: ignore[union-attr]
                elif isinstance(raw, (int, float)):
                    numeric = float(raw)
                else:
                    raise TypeError(f"Direct numeric feature {field_name} is not numeric")
                denominator = max(
                    np.finfo(float).eps,
                    predicate.clip_multiplier * predicate.scale,
                )
                values[(field_name, "value")] = (
                    numeric - float(self.numeric_centres[field_name])
                ) / denominator
            elif kind == "geographic":
                # A coordinate has no meaningful signed scalar direction in
                # this generic encoder.  The explicit missing indicator remains
                # available; geographic direct-utility baselines require a
                # task-specific coordinate transform.
                values[(field_name, "value")] = 0.0
            elif kind == "ordinal":
                level = predicate.spec.ordinal.get(str(observed[0]))
                if level is None:
                    raise ValueError(f"Unknown ordinal value for {field_name}: {observed[0]!r}")
                levels = tuple(predicate.spec.ordinal.values())
                denominator = max(1, max(levels) - min(levels))
                values[(field_name, "ordinal")] = (level - min(levels)) / denominator
            else:
                for raw in observed:
                    key = (field_name, f"value={raw}")
                    if key in self.columns:
                        values[key] = 1.0
        return np.asarray([values.get(column, 0.0) for column in self.columns], dtype=float)


def direct_utility_preferences(
    observations: Iterable[PairwiseObservation],
    entities: Mapping[str, EntityRecord],
    encoder: FittedEntityFeatureEncoder,
) -> list[Preference]:
    output: list[Preference] = []
    for observation in observations:
        preferred = encoder.transform(entities[observation.preferred_id])
        nonpreferred = encoder.transform(entities[observation.nonpreferred_id])
        output.append(
            Preference(
                user_id=observation.user_id,
                query_id=observation.comparison_id,
                preferred_id=observation.preferred_id,
                nonpreferred_id=observation.nonpreferred_id,
                delta=tuple(float(value) for value in preferred - nonpreferred),
            )
        )
    return output


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(frozen=True)
class BradleyTerryFit:
    item_ids: tuple[str, ...]
    utilities: tuple[float, ...]
    objective: float
    iterations: int
    converged: bool
    metadata: Mapping[str, object] = field(default_factory=dict)

    def utility_map(self) -> dict[str, float]:
        return dict(zip(self.item_ids, self.utilities, strict=True))


def fit_bradley_terry(
    observations: Iterable[PairwiseObservation],
    item_ids: Sequence[str],
    *,
    l2: float = 1e-2,
    prior: Mapping[str, float] | None = None,
    prior_strength: float = 0.0,
    max_iterations: int = 20_000,
    tolerance: float = 1e-9,
) -> BradleyTerryFit:
    rows = list(observations)
    ordered = tuple(sorted(set(item_ids)))
    if not rows:
        if prior is None:
            raise ValueError("Bradley-Terry requires observations when no prior is supplied")
        utilities = np.asarray([float(prior.get(item, 0.0)) for item in ordered], dtype=float)
        utilities -= float(np.mean(utilities))
        return BradleyTerryFit(
            item_ids=ordered,
            utilities=tuple(float(value) for value in utilities),
            objective=0.0,
            iterations=0,
            converged=True,
            metadata={"status": "prior_only"},
        )
    if l2 < 0 or prior_strength < 0:
        raise ValueError("Regularization strengths must be non-negative")
    index = {item: position for position, item in enumerate(ordered)}
    preferred = np.asarray([index[row.preferred_id] for row in rows], dtype=int)
    nonpreferred = np.asarray([index[row.nonpreferred_id] for row in rows], dtype=int)
    prior_vector = np.asarray(
        [float(prior.get(item, 0.0)) if prior is not None else 0.0 for item in ordered],
        dtype=float,
    )
    prior_vector -= float(np.mean(prior_vector))
    utilities = prior_vector.copy()

    def objective_gradient(vector: np.ndarray) -> tuple[float, np.ndarray]:
        margins = vector[preferred] - vector[nonpreferred]
        loss = float(np.mean(np.logaddexp(0.0, -margins)))
        probability_wrong = _sigmoid(-margins)
        gradient = np.zeros_like(vector)
        np.add.at(gradient, preferred, -probability_wrong / len(rows))
        np.add.at(gradient, nonpreferred, probability_wrong / len(rows))
        if l2:
            loss += 0.5 * l2 * float(np.mean(np.square(vector)))
            gradient += (l2 / vector.size) * vector
        if prior_strength:
            difference = vector - prior_vector
            loss += 0.5 * prior_strength * float(np.mean(np.square(difference)))
            gradient += (prior_strength / vector.size) * difference
        gradient -= float(np.mean(gradient))
        return loss, gradient

    objective, gradient = objective_gradient(utilities)
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm <= tolerance:
            converged = True
            break
        step = 1.0
        while step >= 1e-12:
            proposal = utilities - step * gradient
            proposal -= float(np.mean(proposal))
            next_objective, next_gradient = objective_gradient(proposal)
            if next_objective <= objective - 1e-4 * step * gradient_norm**2:
                utilities = proposal
                objective = next_objective
                gradient = next_gradient
                break
            step *= 0.5
        else:
            converged = gradient_norm <= math.sqrt(tolerance)
            break
        if float(np.linalg.norm(step * gradient)) <= tolerance:
            converged = True
            break
    return BradleyTerryFit(
        item_ids=ordered,
        utilities=tuple(float(value) for value in utilities),
        objective=objective,
        iterations=iterations,
        converged=converged,
        metadata={
            "n_observations": len(rows),
            "l2": l2,
            "prior_strength": prior_strength,
            "centred_sum_residual": abs(float(np.sum(utilities))),
        },
    )


def evaluate_bradley_terry(
    fitted: BradleyTerryFit,
    observations: Iterable[PairwiseObservation],
) -> dict[str, float]:
    rows = list(observations)
    if not rows:
        return {"accuracy": float("nan"), "log_loss": float("nan"), "brier": float("nan")}
    utilities = fitted.utility_map()
    margins = np.asarray(
        [utilities[row.preferred_id] - utilities[row.nonpreferred_id] for row in rows],
        dtype=float,
    )
    probabilities = _sigmoid(margins)
    return {
        "accuracy": float(np.mean(margins > 0.0)),
        "tie_rate": float(np.mean(np.isclose(margins, 0.0, atol=1e-12))),
        "log_loss": float(np.mean(np.logaddexp(0.0, -margins))),
        "brier": float(np.mean(np.square(1.0 - probabilities))),
    }


@dataclass(frozen=True)
class ConditionalLogitFit:
    coefficients: tuple[float, ...]
    objective: float
    iterations: int
    converged: bool
    metadata: Mapping[str, object] = field(default_factory=dict)


def _choice_objective_gradient(
    coefficients: np.ndarray,
    feature_sets: Sequence[np.ndarray] | np.ndarray,
    chosen_indices: Sequence[int] | np.ndarray,
    l2: float,
) -> tuple[float, np.ndarray]:
    if isinstance(feature_sets, np.ndarray) and feature_sets.ndim == 3:
        chosen = np.asarray(chosen_indices, dtype=int)
        scores = feature_sets @ coefficients
        maxima = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(np.clip(scores - maxima, -40.0, 0.0))
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        row_indices = np.arange(feature_sets.shape[0])
        loss = float(
            np.mean(
                np.log(np.sum(exp_scores, axis=1))
                + maxima[:, 0]
                - scores[row_indices, chosen]
            )
        )
        residual = probabilities
        residual[row_indices, chosen] -= 1.0
        gradient = np.einsum("nmd,nm->d", feature_sets, residual) / feature_sets.shape[0]
        if l2:
            loss += 0.5 * l2 * float(np.dot(coefficients, coefficients))
            gradient += l2 * coefficients
        return loss, gradient

    loss = 0.0
    gradient = np.zeros_like(coefficients)
    for features, chosen in zip(feature_sets, chosen_indices, strict=True):
        scores = features @ coefficients
        shifted = scores - float(np.max(scores))
        exp_scores = np.exp(np.clip(shifted, -40.0, 0.0))
        probabilities = exp_scores / float(np.sum(exp_scores))
        loss += math.log(float(np.sum(exp_scores))) + float(np.max(scores)) - float(scores[chosen])
        gradient += features.T @ probabilities - features[chosen]
    loss /= len(feature_sets)
    gradient /= len(feature_sets)
    if l2:
        loss += 0.5 * l2 * float(np.dot(coefficients, coefficients))
        gradient += l2 * coefficients
    return loss, gradient


def _choice_hessian(
    coefficients: np.ndarray,
    feature_sets: Sequence[np.ndarray] | np.ndarray,
    l2: float,
) -> np.ndarray:
    """Exact conditional-logit Hessian for deterministic damped Newton steps."""

    dimension = coefficients.size
    hessian = np.eye(dimension, dtype=float) * l2
    if isinstance(feature_sets, np.ndarray) and feature_sets.ndim == 3:
        scores = feature_sets @ coefficients
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(np.clip(shifted, -40.0, 0.0))
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        expected = np.einsum("nm,nmd->nd", probabilities, feature_sets)
        centered = feature_sets - expected[:, None, :]
        hessian += (
            np.einsum("nm,nmd,nme->de", probabilities, centered, centered)
            / feature_sets.shape[0]
        )
        return hessian

    for features in feature_sets:
        scores = features @ coefficients
        shifted = scores - float(np.max(scores))
        exp_scores = np.exp(np.clip(shifted, -40.0, 0.0))
        probabilities = exp_scores / float(np.sum(exp_scores))
        expected = probabilities @ features
        centered = features - expected
        hessian += (centered.T * probabilities) @ centered / len(feature_sets)
    return hessian


def fit_conditional_logit(
    feature_sets: Sequence[np.ndarray],
    chosen_indices: Sequence[int],
    *,
    l2: float = 1e-2,
    max_iterations: int = 250,
    tolerance: float = 1e-8,
) -> ConditionalLogitFit:
    if not feature_sets or len(feature_sets) != len(chosen_indices):
        raise ValueError("Conditional logit requires aligned non-empty choice sets and labels")
    dimension = int(feature_sets[0].shape[1])
    for features, chosen in zip(feature_sets, chosen_indices, strict=True):
        if features.ndim != 2 or features.shape[1] != dimension:
            raise ValueError("All conditional-logit feature matrices must share a dimension")
        if chosen < 0 or chosen >= features.shape[0]:
            raise ValueError("Chosen alternative index is outside its choice set")
    stacked: np.ndarray | Sequence[np.ndarray]
    if len({features.shape[0] for features in feature_sets}) == 1:
        stacked = np.stack(feature_sets, axis=0)
    else:
        stacked = feature_sets
    chosen_array = np.asarray(chosen_indices, dtype=int)
    coefficients = np.zeros(dimension, dtype=float)
    objective, gradient = _choice_objective_gradient(
        coefficients, stacked, chosen_array, l2
    )
    if dimension >= 40:
        def objective_and_gradient(vector: np.ndarray) -> tuple[float, np.ndarray]:
            return _choice_objective_gradient(vector, stacked, chosen_array, l2)

        optimized = minimize(
            objective_and_gradient,
            coefficients,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": max_iterations,
                "ftol": tolerance,
                "gtol": tolerance,
                "maxls": 50,
            },
        )
        final_objective, final_gradient = _choice_objective_gradient(
            np.asarray(optimized.x, dtype=float), stacked, chosen_array, l2
        )
        return ConditionalLogitFit(
            coefficients=tuple(float(value) for value in optimized.x),
            objective=float(final_objective),
            iterations=int(optimized.nit),
            converged=bool(optimized.success),
            metadata={
                "choice_sets": len(feature_sets),
                "l2": l2,
                "optimizer": "scipy_l_bfgs_b",
                "optimizer_status_code": int(optimized.status),
                "optimizer_message": str(optimized.message),
                "function_evaluations": int(optimized.nfev),
                "gradient_evaluations": int(optimized.njev),
                "gradient_norm": float(np.linalg.norm(final_gradient)),
                "tolerance": tolerance,
                "maximum_iterations": max_iterations,
            },
        )
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        norm = float(np.linalg.norm(gradient))
        if norm <= tolerance:
            converged = True
            break
        hessian = _choice_hessian(coefficients, stacked, l2)
        try:
            direction = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            # A zero-penalty model can contain redundant one-hot columns.  The
            # minimum-norm Newton direction is deterministic and preserves the
            # convex objective without pretending the coefficients are unique.
            direction = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        directional_decrement = float(np.dot(gradient, direction))
        if not np.isfinite(directional_decrement) or directional_decrement <= 0.0:
            direction = gradient
            directional_decrement = norm**2
        step = 1.0
        while step >= 1e-12:
            proposal = coefficients - step * direction
            next_objective, next_gradient = _choice_objective_gradient(
                proposal, stacked, chosen_array, l2
            )
            if next_objective <= objective - 1e-4 * step * directional_decrement:
                coefficients = proposal
                objective = next_objective
                gradient = next_gradient
                break
            step *= 0.5
        else:
            converged = norm <= math.sqrt(tolerance)
            break
        if float(np.linalg.norm(step * direction)) <= tolerance * (
            1.0 + float(np.linalg.norm(coefficients))
        ):
            converged = True
            break
    return ConditionalLogitFit(
        coefficients=tuple(float(value) for value in coefficients),
        objective=objective,
        iterations=iterations,
        converged=converged,
        metadata={
            "choice_sets": len(feature_sets),
            "l2": l2,
            "optimizer": "damped_newton_exact_hessian",
            "gradient_norm": float(np.linalg.norm(gradient)),
            "tolerance": tolerance,
            "maximum_iterations": max_iterations,
        },
    )


def evaluate_choice_sets(
    fitted: ConditionalLogitFit,
    feature_sets: Sequence[np.ndarray],
    chosen_indices: Sequence[int],
) -> dict[str, float]:
    coefficients = np.asarray(fitted.coefficients, dtype=float)
    reciprocal_ranks: list[float] = []
    ranks: list[float] = []
    losses: list[float] = []
    top_one: list[float] = []
    briers: list[float] = []
    for features, chosen in zip(feature_sets, chosen_indices, strict=True):
        scores = features @ coefficients
        order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
        rank = order.index(chosen) + 1
        shifted = scores - float(np.max(scores))
        probabilities = np.exp(np.clip(shifted, -40.0, 0.0))
        probabilities /= float(np.sum(probabilities))
        target = np.zeros_like(probabilities)
        target[chosen] = 1.0
        ranks.append(float(rank))
        reciprocal_ranks.append(1.0 / rank)
        top_one.append(float(rank == 1))
        losses.append(-math.log(max(np.finfo(float).tiny, float(probabilities[chosen]))))
        briers.append(float(np.mean(np.square(probabilities - target))))
    return {
        "top1_accuracy": float(np.mean(top_one)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "mean_chosen_rank": float(np.mean(ranks)),
        "negative_log_likelihood": float(np.mean(losses)),
        "multiclass_brier": float(np.mean(briers)),
    }


@dataclass(frozen=True)
class LatentClassConditionalLogitFit:
    """Finite-mixture conditional-logit fit with global class membership.

    The archive used by the McFadden--Train experiment contains one choice set
    per row and no stable respondent identifier.  Consequently the mixing
    probabilities here are population-level probabilities; they must not be
    interpreted as personalized class assignments.
    """

    class_count: int
    coefficients: tuple[tuple[float, ...], ...]
    mixing_probabilities: tuple[float, ...]
    objective: float
    iterations: int
    converged: bool
    metadata: Mapping[str, object] = field(default_factory=dict)


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Numerically stable log-sum-exp without silently clipping probabilities."""

    maxima = np.max(values, axis=axis, keepdims=True)
    shifted = values - maxima
    output = maxima + np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    if axis is None:
        return np.asarray(output.squeeze(), dtype=float)
    return np.asarray(np.squeeze(output, axis=axis), dtype=float)


def _validate_choice_arrays(
    feature_sets: Sequence[np.ndarray],
    chosen_indices: Sequence[int],
) -> tuple[list[np.ndarray], np.ndarray, int]:
    if not feature_sets or len(feature_sets) != len(chosen_indices):
        raise ValueError("Latent-class logit requires aligned non-empty choice sets and labels")
    matrices = [np.asarray(features, dtype=float) for features in feature_sets]
    dimension = int(matrices[0].shape[1]) if matrices[0].ndim == 2 else -1
    if dimension <= 0:
        raise ValueError("Choice-set feature matrices must have a positive feature dimension")
    chosen = np.asarray(chosen_indices, dtype=int)
    for features, selected in zip(matrices, chosen, strict=True):
        if features.ndim != 2 or features.shape[1] != dimension or not features.shape[0]:
            raise ValueError("All choice-set feature matrices must be non-empty and aligned")
        if not np.all(np.isfinite(features)):
            raise ValueError("Choice-set features must all be finite")
        if selected < 0 or selected >= features.shape[0]:
            raise ValueError("Chosen alternative index is outside its choice set")
    return matrices, chosen, dimension


def _stack_if_rectangular(
    feature_sets: Sequence[np.ndarray],
) -> Sequence[np.ndarray] | np.ndarray:
    if len({features.shape[0] for features in feature_sets}) == 1:
        return np.stack(feature_sets, axis=0)
    return feature_sets


def _chosen_class_log_probabilities(
    coefficients: np.ndarray,
    feature_sets: Sequence[np.ndarray] | np.ndarray,
    chosen_indices: np.ndarray,
) -> np.ndarray:
    """Return N by K chosen-alternative log probabilities."""

    if isinstance(feature_sets, np.ndarray):
        scores = np.einsum("nmd,kd->nmk", feature_sets, coefficients)
        denominators = _logsumexp(scores, axis=1)
        selected = scores[np.arange(feature_sets.shape[0]), chosen_indices, :]
        return selected - denominators
    output = np.empty((len(feature_sets), coefficients.shape[0]), dtype=float)
    for index, (features, chosen) in enumerate(
        zip(feature_sets, chosen_indices, strict=True)
    ):
        scores = features @ coefficients.T
        output[index] = scores[chosen] - _logsumexp(scores, axis=0)
    return output


def _weighted_choice_objective_gradient(
    coefficients: np.ndarray,
    feature_sets: Sequence[np.ndarray] | np.ndarray,
    chosen_indices: np.ndarray,
    responsibilities: np.ndarray,
    l2: float,
) -> tuple[float, np.ndarray]:
    """Class-weighted conditional-logit M-step objective and analytic gradient."""

    sample_count = len(chosen_indices)
    if isinstance(feature_sets, np.ndarray):
        scores = feature_sets @ coefficients
        maxima = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - maxima)
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        rows = np.arange(sample_count)
        log_probabilities = (
            scores[rows, chosen_indices]
            - maxima[:, 0]
            - np.log(np.sum(exp_scores, axis=1))
        )
        residual = probabilities.copy()
        residual[rows, chosen_indices] -= 1.0
        weighted_residual = responsibilities[:, None] * residual
        loss = -float(np.dot(responsibilities, log_probabilities)) / sample_count
        gradient = (
            np.einsum("nmd,nm->d", feature_sets, weighted_residual) / sample_count
        )
    else:
        loss = 0.0
        gradient = np.zeros_like(coefficients)
        for features, chosen, weight in zip(
            feature_sets, chosen_indices, responsibilities, strict=True
        ):
            scores = features @ coefficients
            shifted = scores - float(np.max(scores))
            exp_scores = np.exp(shifted)
            probabilities = exp_scores / float(np.sum(exp_scores))
            loss -= float(weight) * (
                float(scores[chosen])
                - float(np.max(scores))
                - math.log(float(np.sum(exp_scores)))
            )
            probabilities[chosen] -= 1.0
            gradient += float(weight) * (features.T @ probabilities)
        loss /= sample_count
        gradient /= sample_count
    if l2:
        loss += 0.5 * l2 * float(np.dot(coefficients, coefficients))
        gradient += l2 * coefficients
    return float(loss), np.asarray(gradient, dtype=float)


def _fit_weighted_logit_class(
    feature_sets: Sequence[np.ndarray] | np.ndarray,
    chosen_indices: np.ndarray,
    responsibilities: np.ndarray,
    initial: np.ndarray,
    *,
    l2: float,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, dict[str, object]]:
    def objective_gradient(vector: np.ndarray) -> tuple[float, np.ndarray]:
        return _weighted_choice_objective_gradient(
            vector,
            feature_sets,
            chosen_indices,
            responsibilities,
            l2,
        )

    optimized = minimize(
        objective_gradient,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": max_iterations,
            "ftol": tolerance,
            "gtol": tolerance,
            "maxls": 50,
        },
    )
    objective, gradient = objective_gradient(np.asarray(optimized.x, dtype=float))
    return np.asarray(optimized.x, dtype=float), {
        "success": bool(optimized.success),
        "status_code": int(optimized.status),
        "message": str(optimized.message),
        "iterations": int(optimized.nit),
        "function_evaluations": int(optimized.nfev),
        "gradient_evaluations": int(optimized.njev),
        "objective": float(objective),
        "gradient_norm": float(np.linalg.norm(gradient)),
    }


def _latent_mixture_state(
    coefficients: np.ndarray,
    mixing_probabilities: np.ndarray,
    feature_sets: Sequence[np.ndarray] | np.ndarray,
    chosen_indices: np.ndarray,
    l2: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    class_log_probabilities = _chosen_class_log_probabilities(
        coefficients, feature_sets, chosen_indices
    )
    log_joint = class_log_probabilities + np.log(mixing_probabilities)[None, :]
    log_evidence = _logsumexp(log_joint, axis=1)
    responsibilities = np.exp(log_joint - log_evidence[:, None])
    objective = -float(np.mean(log_evidence))
    if l2:
        objective += 0.5 * l2 * float(np.sum(np.square(coefficients)))
    return objective, responsibilities, log_evidence


def _deterministic_initial_responsibilities(
    feature_sets: Sequence[np.ndarray],
    chosen_indices: np.ndarray,
    class_count: int,
    smoothing: float,
) -> np.ndarray:
    """Separate classes by ranked chosen-versus-offer feature contrasts."""

    contrasts = np.stack(
        [
            features[chosen] - np.mean(features, axis=0)
            for features, chosen in zip(feature_sets, chosen_indices, strict=True)
        ],
        axis=0,
    )
    coordinates = np.arange(1, contrasts.shape[1] + 1, dtype=float)
    direction = np.sin(np.sqrt(2.0) * coordinates) + np.cos(
        np.sqrt(3.0) * coordinates
    )
    norm = float(np.linalg.norm(direction))
    if norm <= np.finfo(float).eps:
        direction = np.ones_like(direction)
        norm = float(np.linalg.norm(direction))
    projections = contrasts @ (direction / norm)
    ordering = np.argsort(projections, kind="stable")
    ranks = np.empty(len(feature_sets), dtype=float)
    ranks[ordering] = np.arange(len(feature_sets), dtype=float)
    if len(feature_sets) > 1:
        ranks /= len(feature_sets) - 1
    else:
        ranks[:] = 0.5
    centres = (np.arange(class_count, dtype=float) + 0.5) / class_count
    bandwidth = max(0.05, 0.40 / class_count)
    logits = -np.square(ranks[:, None] - centres[None, :]) / (2.0 * bandwidth**2)
    logits -= np.max(logits, axis=1, keepdims=True)
    responsibilities = np.exp(logits)
    responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
    return (1.0 - smoothing) * responsibilities + smoothing / class_count


def fit_latent_class_conditional_logit(
    feature_sets: Sequence[np.ndarray],
    chosen_indices: Sequence[int],
    *,
    class_count: int = 2,
    l2: float = 1e-2,
    max_iterations: int = 500,
    tolerance: float = 1e-6,
    m_step_max_iterations: int = 250,
    m_step_tolerance: float = 1e-8,
    initialization_scale: float = 0.15,
    initialization_smoothing: float = 0.05,
    mixing_floor: float = 1e-8,
    minimum_class_share: float = 0.0,
) -> LatentClassConditionalLogitFit:
    """Fit a deterministic finite mixture of conditional logits with EM.

    The objective is mean observed-choice negative log likelihood plus one L2
    penalty per component.  Class-weighted convex M-steps use analytic-gradient
    L-BFGS-B.  A ranked, data-dependent initialization prevents the symmetric
    all-components-equal fixed point without introducing a random seed.
    """

    matrices, chosen, dimension = _validate_choice_arrays(feature_sets, chosen_indices)
    if class_count < 2:
        raise ValueError("Latent-class conditional logit requires at least two classes")
    if class_count > len(matrices):
        raise ValueError("class_count cannot exceed the number of choice sets")
    if l2 < 0 or max_iterations <= 0 or m_step_max_iterations <= 0:
        raise ValueError("Regularization must be non-negative and iteration limits positive")
    if tolerance <= 0 or m_step_tolerance <= 0:
        raise ValueError("Optimization tolerances must be positive")
    if initialization_scale < 0 or not 0 <= initialization_smoothing < 1:
        raise ValueError("Invalid deterministic-initialization configuration")
    if not 0 < mixing_floor < 1 / class_count:
        raise ValueError("mixing_floor must lie between zero and 1/class_count")
    if not 0 <= minimum_class_share < 1 / class_count:
        raise ValueError("minimum_class_share must lie in [0, 1/class_count)")

    stacked = _stack_if_rectangular(matrices)
    responsibilities = _deterministic_initial_responsibilities(
        matrices, chosen, class_count, initialization_smoothing
    )
    mixing_probabilities = np.mean(responsibilities, axis=0)
    mixing_probabilities = np.maximum(mixing_probabilities, mixing_floor)
    mixing_probabilities /= float(np.sum(mixing_probabilities))

    coordinates = np.arange(1, dimension + 1, dtype=float)
    coefficients = np.empty((class_count, dimension), dtype=float)
    initialization_diagnostics: list[dict[str, object]] = []
    all_m_steps_succeeded = True
    for class_index in range(class_count):
        perturbation = np.sin(
            (class_index + 1.0) * np.sqrt(2.0) * coordinates
        ) + np.cos((class_index + 2.0) * np.sqrt(3.0) * coordinates)
        perturbation /= max(np.finfo(float).eps, float(np.linalg.norm(perturbation)))
        coefficients[class_index], diagnostic = _fit_weighted_logit_class(
            stacked,
            chosen,
            responsibilities[:, class_index],
            initialization_scale * perturbation,
            l2=l2,
            max_iterations=m_step_max_iterations,
            tolerance=m_step_tolerance,
        )
        initialization_diagnostics.append(diagnostic)
        all_m_steps_succeeded &= bool(diagnostic["success"])

    objective, responsibilities, _ = _latent_mixture_state(
        coefficients, mixing_probabilities, stacked, chosen, l2
    )
    objective_trace = [float(objective)]
    em_diagnostics: list[dict[str, object]] = []
    converged = False
    valid = bool(np.isfinite(objective))
    failure_reason: str | None = None
    maximum_responsibility_change = float("inf")
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        previous_objective = objective
        previous_responsibilities = responsibilities
        mixing_probabilities = np.mean(previous_responsibilities, axis=0)
        mixing_probabilities = np.maximum(mixing_probabilities, mixing_floor)
        mixing_probabilities /= float(np.sum(mixing_probabilities))
        step_diagnostics: list[dict[str, object]] = []
        for class_index in range(class_count):
            coefficients[class_index], diagnostic = _fit_weighted_logit_class(
                stacked,
                chosen,
                previous_responsibilities[:, class_index],
                coefficients[class_index],
                l2=l2,
                max_iterations=m_step_max_iterations,
                tolerance=m_step_tolerance,
            )
            step_diagnostics.append(diagnostic)
            all_m_steps_succeeded &= bool(diagnostic["success"])
        objective, responsibilities, _ = _latent_mixture_state(
            coefficients, mixing_probabilities, stacked, chosen, l2
        )
        maximum_responsibility_change = float(
            np.max(np.abs(responsibilities - previous_responsibilities))
        )
        improvement = float(previous_objective - objective)
        objective_trace.append(float(objective))
        em_diagnostics.append(
            {
                "iteration": iterations,
                "objective": float(objective),
                "objective_improvement": improvement,
                "maximum_responsibility_change": maximum_responsibility_change,
                "m_steps": step_diagnostics,
            }
        )
        if not np.isfinite(objective) or not np.all(np.isfinite(responsibilities)):
            valid = False
            failure_reason = "non_finite_em_state"
            break
        monotonic_tolerance = 1e-8 * (1.0 + abs(previous_objective))
        if improvement < -monotonic_tolerance:
            valid = False
            failure_reason = "penalized_objective_increased"
            break
        if abs(improvement) <= tolerance * (1.0 + abs(previous_objective)):
            converged = True
            break

    effective_counts = np.sum(responsibilities, axis=0)
    effective_shares = effective_counts / len(matrices)
    if float(np.min(effective_shares)) < minimum_class_share:
        valid = False
        failure_reason = "collapsed_latent_class"
    if not all_m_steps_succeeded:
        valid = False
        failure_reason = failure_reason or "l_bfgs_m_step_nonconvergence"
    if not converged:
        failure_reason = failure_reason or "em_iteration_limit"

    # Canonical ordering removes label-switching ambiguity from serialized fits.
    ordering = sorted(
        range(class_count),
        key=lambda index: tuple(np.round(coefficients[index], 12)),
    )
    coefficients = coefficients[ordering]
    mixing_probabilities = mixing_probabilities[ordering]
    responsibilities = responsibilities[:, ordering]
    effective_counts = effective_counts[ordering]
    effective_shares = effective_shares[ordering]
    separations = [
        float(np.linalg.norm(coefficients[left] - coefficients[right]))
        for left in range(class_count)
        for right in range(left + 1, class_count)
    ]
    valid &= bool(
        np.all(np.isfinite(coefficients))
        and np.all(np.isfinite(mixing_probabilities))
        and abs(float(np.sum(mixing_probabilities)) - 1.0) <= 1e-10
    )
    if not valid and failure_reason is None:
        failure_reason = "invalid_final_parameters"

    return LatentClassConditionalLogitFit(
        class_count=class_count,
        coefficients=tuple(
            tuple(float(value) for value in row) for row in coefficients
        ),
        mixing_probabilities=tuple(float(value) for value in mixing_probabilities),
        objective=float(objective),
        iterations=iterations,
        converged=bool(converged and valid),
        metadata={
            "valid": bool(valid),
            "failure_reason": failure_reason,
            "scope": "global_population_mixture_without_respondent_identifiers",
            "choice_sets": len(matrices),
            "feature_dimension": dimension,
            "class_count": class_count,
            "l2": l2,
            "optimizer": "deterministic_em_with_analytic_gradient_l_bfgs_b_m_steps",
            "initialization": "ranked_chosen_offer_contrast",
            "initialization_scale": initialization_scale,
            "initialization_smoothing": initialization_smoothing,
            "mixing_floor": mixing_floor,
            "minimum_class_share": minimum_class_share,
            "objective_trace": objective_trace,
            "objective_monotone_within_tolerance": all(
                later <= earlier + 1e-8 * (1.0 + abs(earlier))
                for earlier, later in zip(objective_trace, objective_trace[1:])
            ),
            "maximum_responsibility_change": maximum_responsibility_change,
            "effective_class_counts": [float(value) for value in effective_counts],
            "effective_class_shares": [float(value) for value in effective_shares],
            "near_empty_class_warning": bool(np.min(effective_shares) < 1e-3),
            "minimum_pairwise_coefficient_distance": (
                min(separations) if separations else 0.0
            ),
            "initialization_m_steps": initialization_diagnostics,
            "em_diagnostics": em_diagnostics,
            "maximum_em_iterations": max_iterations,
            "m_step_maximum_iterations": m_step_max_iterations,
            "em_tolerance": tolerance,
            "m_step_tolerance": m_step_tolerance,
        },
    )


def latent_class_log_utilities(
    fitted: LatentClassConditionalLogitFit,
    feature_sets: Sequence[np.ndarray],
) -> list[np.ndarray]:
    """Return log mixture probabilities, usable as softmax utility scores."""

    coefficients = np.asarray(fitted.coefficients, dtype=float)
    mixing = np.asarray(fitted.mixing_probabilities, dtype=float)
    if coefficients.ndim != 2 or coefficients.shape[0] != fitted.class_count:
        raise ValueError("Latent-class coefficient matrix is malformed")
    if mixing.shape != (fitted.class_count,) or np.any(mixing <= 0):
        raise ValueError("Latent-class mixing probabilities are malformed")
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(mixing)):
        raise ValueError("Latent-class parameters must be finite")
    output: list[np.ndarray] = []
    for features in feature_sets:
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != coefficients.shape[1]:
            raise ValueError("Prediction feature dimensions do not match the fitted model")
        class_scores = matrix @ coefficients.T
        class_log_probabilities = class_scores - _logsumexp(
            class_scores, axis=0
        )[None, :]
        log_mixture = _logsumexp(
            class_log_probabilities + np.log(mixing)[None, :], axis=1
        )
        # The subtraction only removes accumulated floating-point normalization
        # error; exp(log_mixture) already sums to one mathematically.
        log_mixture -= float(_logsumexp(log_mixture))
        output.append(np.asarray(log_mixture, dtype=float))
    return output


def evaluate_latent_class_choice_sets(
    fitted: LatentClassConditionalLogitFit,
    feature_sets: Sequence[np.ndarray],
    chosen_indices: Sequence[int],
) -> dict[str, float]:
    """Evaluate exact mixture probabilities with standard choice metrics."""

    _, chosen, _ = _validate_choice_arrays(feature_sets, chosen_indices)
    log_utilities = latent_class_log_utilities(fitted, feature_sets)
    reciprocal_ranks: list[float] = []
    ranks: list[float] = []
    losses: list[float] = []
    top_one: list[float] = []
    briers: list[float] = []
    for scores, selected in zip(log_utilities, chosen, strict=True):
        order = sorted(
            range(len(scores)), key=lambda index: (-float(scores[index]), index)
        )
        rank = order.index(int(selected)) + 1
        probabilities = np.exp(scores)
        target = np.zeros_like(probabilities)
        target[selected] = 1.0
        ranks.append(float(rank))
        reciprocal_ranks.append(1.0 / rank)
        top_one.append(float(rank == 1))
        losses.append(-float(scores[selected]))
        briers.append(float(np.mean(np.square(probabilities - target))))
    return {
        "top1_accuracy": float(np.mean(top_one)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "mean_chosen_rank": float(np.mean(ranks)),
        "negative_log_likelihood": float(np.mean(losses)),
        "multiclass_brier": float(np.mean(briers)),
    }
