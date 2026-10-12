from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from .distance import FittedMetric
from .learning import GlobalSimplexLearner, PersonalizedSimplexLearner
from .types import EntityRecord, FitResult, Preference


@dataclass(frozen=True)
class PairwiseObservation:
    """One strict archival choice before a query/profile has been inferred."""

    user_id: str
    comparison_id: str
    preferred_id: str
    nonpreferred_id: str
    group_id: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class LatentProfileFit:
    prototype: EntityRecord
    weights: tuple[float, ...]
    objective: float
    status: str
    mode: str
    n_observations: int
    metadata: Mapping[str, object] = field(default_factory=dict)


def build_prototype_grid(
    records: Iterable[EntityRecord],
    fields: Sequence[str],
    *,
    max_candidates: int = 512,
) -> tuple[EntityRecord, ...]:
    """Build a deterministic finite ideal-point search space.

    Archival choice datasets generally provide alternatives but no explicit
    query entity.  A finite grid keeps latent-profile selection inspectable and
    reproducible.  If a full Cartesian product is too large, observed entities
    are retained and a deterministic evenly-spaced subset of combinations is
    added; no label is used during grid construction.
    """

    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    record_list = sorted(records, key=lambda item: item.entity_id)
    if not record_list:
        raise ValueError("At least one entity is required to build prototype candidates")
    if not fields:
        raise ValueError("At least one field is required to build prototype candidates")

    values_by_field: list[tuple[tuple[object, ...], ...]] = []
    for field_name in fields:
        unique: dict[str, tuple[object, ...]] = {}
        for record in record_list:
            value = tuple(record.get(field_name))
            if value:
                unique.setdefault(repr(value), value)
        if not unique:
            # Missing is a represented state in the metric.  Keeping it in the
            # grid is preferable to silently deleting a declared predicate.
            unique["()"] = ()
        values_by_field.append(tuple(unique[key] for key in sorted(unique)))

    combination_count = math.prod(len(values) for values in values_by_field)
    combinations = itertools.product(*values_by_field)
    if combination_count <= max_candidates:
        selected = list(combinations)
    else:
        # The product is generated lazily and sampled at deterministic integer
        # positions.  This prevents a combinatorial allocation while retaining
        # coverage across the lexicographically ordered grid.
        positions = set(
            int(value)
            for value in np.linspace(0, combination_count - 1, max_candidates, dtype=int)
        )
        selected = [combo for index, combo in enumerate(combinations) if index in positions]

    candidates: dict[tuple[tuple[str, str], ...], EntityRecord] = {}
    for index, combination in enumerate(selected):
        values = {
            field_name: tuple(value)
            for field_name, value in zip(fields, combination, strict=True)
            if value
        }
        signature = tuple((name, repr(values.get(name, ()))) for name in fields)
        candidates[signature] = EntityRecord(
            entity_id=f"latent_profile_{index:04d}",
            uri=f"urn:rgsmw:latent-profile:{index:04d}",
            values=values,
            source_file="generated_from_training_schema_only",
        )

    # Observed alternatives are also legal prototypes.  This is important when
    # the Cartesian grid is capped and makes the fallback auditable.
    for record in record_list:
        values = {name: tuple(record.get(name)) for name in fields if record.get(name)}
        signature = tuple((name, repr(values.get(name, ()))) for name in fields)
        candidates.setdefault(
            signature,
            EntityRecord(
                entity_id=f"observed_profile_{record.entity_id}",
                uri=f"urn:rgsmw:observed-profile:{record.entity_id}",
                values=values,
                source_file="observed_alternative_profile_candidate",
            ),
        )
    return tuple(candidates[key] for key in sorted(candidates))


def observations_to_preferences(
    observations: Iterable[PairwiseObservation],
    entities: Mapping[str, EntityRecord],
    prototype: EntityRecord,
    metric: FittedMetric,
) -> list[Preference]:
    output: list[Preference] = []
    for observation in observations:
        try:
            preferred = entities[observation.preferred_id]
            nonpreferred = entities[observation.nonpreferred_id]
        except KeyError as error:
            raise KeyError(
                f"Unknown alternative {error.args[0]!r} in comparison "
                f"{observation.comparison_id!r}"
            ) from error
        preferred_vector = metric.feature_vector(prototype, preferred)
        nonpreferred_vector = metric.feature_vector(prototype, nonpreferred)
        delta = nonpreferred_vector - preferred_vector
        output.append(
            Preference(
                user_id=observation.user_id,
                query_id=observation.comparison_id,
                preferred_id=observation.preferred_id,
                nonpreferred_id=observation.nonpreferred_id,
                delta=tuple(float(value) for value in delta),
            )
        )
    return output


def margin_metrics(
    weights: Sequence[float],
    preferences: Iterable[Preference],
    *,
    probability_scale: float = 1.0,
) -> dict[str, float]:
    rows = list(preferences)
    if not rows:
        return {"accuracy": float("nan"), "log_loss": float("nan"), "brier": float("nan")}
    if probability_scale <= 0:
        raise ValueError("probability_scale must be positive")
    vector = np.asarray(weights, dtype=float)
    margins = np.asarray([float(np.dot(vector, row.delta)) for row in rows], dtype=float)
    scaled = np.clip(probability_scale * margins, -40.0, 40.0)
    probabilities = 1.0 / (1.0 + np.exp(-scaled))
    return {
        "accuracy": float(np.mean(margins > 0.0)),
        "tie_rate": float(np.mean(np.isclose(margins, 0.0, atol=1e-12))),
        "mean_margin": float(np.mean(margins)),
        "log_loss": float(np.mean(np.logaddexp(0.0, -scaled))),
        "brier": float(np.mean(np.square(1.0 - probabilities))),
    }


def _profile_distance(
    left: EntityRecord,
    right: EntityRecord,
    metric: FittedMetric,
) -> float:
    return float(np.mean(metric.feature_vector(left, right)))


def _selection_key(
    score: float,
    prototype: EntityRecord,
    weights: Sequence[float],
) -> tuple[float, str, tuple[float, ...]]:
    return (
        round(float(score), 14),
        prototype.entity_id,
        tuple(round(float(value), 14) for value in weights),
    )


@dataclass(frozen=True)
class GlobalLatentProfileSimplexLearner:
    metric: FittedMetric
    prototypes: Sequence[EntityRecord]
    c: float = 5.0
    l2: float = 1e-2
    margin: float = 0.02
    w_max: float | None = None

    def fit(
        self,
        observations: Iterable[PairwiseObservation],
        entities: Mapping[str, EntityRecord],
    ) -> LatentProfileFit:
        rows = list(observations)
        if not rows:
            raise ValueError("At least one strict choice is required for global profile learning")
        if not self.prototypes:
            raise ValueError("At least one prototype candidate is required")

        candidates: list[tuple[tuple[float, str, tuple[float, ...]], EntityRecord, FitResult]] = []
        failures: list[dict[str, str]] = []
        for prototype in sorted(self.prototypes, key=lambda item: item.entity_id):
            preferences = observations_to_preferences(rows, entities, prototype, self.metric)
            try:
                fitted = GlobalSimplexLearner(
                    dimension=len(self.metric.predicates),
                    c=self.c,
                    l2=self.l2,
                    margin=self.margin,
                    w_max=self.w_max,
                ).fit(preferences)
            except (RuntimeError, ValueError) as error:
                failures.append({"prototype_id": prototype.entity_id, "error": str(error)})
                continue
            metrics = margin_metrics(fitted.weights, preferences)
            # The optimization objective is the primary selector.  Mean
            # logistic loss is recorded independently for diagnosis.
            key = _selection_key(fitted.objective, prototype, fitted.weights)
            candidates.append((key, prototype, fitted))
        if not candidates:
            raise RuntimeError("Every latent-profile optimization failed")
        _, prototype, fitted = min(candidates, key=lambda item: item[0])
        selected_preferences = observations_to_preferences(rows, entities, prototype, self.metric)
        return LatentProfileFit(
            prototype=prototype,
            weights=fitted.weights,
            objective=fitted.objective,
            status=fitted.status,
            mode="global_joint_profile_and_weights",
            n_observations=len(rows),
            metadata={
                "candidate_count": len(self.prototypes),
                "successful_candidates": len(candidates),
                "failed_candidates": failures,
                "selection_rule": "minimum_regularized_hinge_objective_then_prototype_id",
                "selected_margin_metrics": margin_metrics(fitted.weights, selected_preferences),
                "weight_fit_metadata": dict(fitted.metadata),
            },
        )


@dataclass(frozen=True)
class PersonalizedLatentProfileSimplexLearner:
    metric: FittedMetric
    prototypes: Sequence[EntityRecord]
    global_fit: LatentProfileFit
    c: float = 5.0
    mu: float = 1.0
    prototype_penalty: float = 0.25
    margin: float = 0.02
    w_max: float | None = None

    def fit(
        self,
        observations: Iterable[PairwiseObservation],
        entities: Mapping[str, EntityRecord],
        *,
        mode: str = "joint",
    ) -> LatentProfileFit:
        if mode not in {"weights_only", "profile_only", "joint"}:
            raise ValueError("mode must be weights_only, profile_only, or joint")
        rows = list(observations)
        if not rows:
            return LatentProfileFit(
                prototype=self.global_fit.prototype,
                weights=self.global_fit.weights,
                objective=0.0,
                status="prior_only",
                mode=mode,
                n_observations=0,
                metadata={"support_size": 0},
            )
        if not self.prototypes:
            raise ValueError("At least one prototype candidate is required")

        search_space = (
            (self.global_fit.prototype,)
            if mode == "weights_only"
            else tuple(sorted(self.prototypes, key=lambda item: item.entity_id))
        )
        candidates: list[
            tuple[tuple[float, str, tuple[float, ...]], EntityRecord, tuple[float, ...], object]
        ] = []
        for prototype in search_space:
            preferences = observations_to_preferences(rows, entities, prototype, self.metric)
            if mode == "profile_only":
                weights = tuple(float(value) for value in self.global_fit.weights)
                hinge = np.maximum(
                    0.0,
                    self.margin
                    - np.asarray(
                        [float(np.dot(weights, preference.delta)) for preference in preferences]
                    ),
                )
                base_objective = float(self.c * np.mean(hinge))
                fit_metadata: object = {"status": "global_weights_frozen"}
            else:
                fitted = PersonalizedSimplexLearner(
                    global_weights=self.global_fit.weights,
                    c=self.c,
                    mu=self.mu,
                    margin=self.margin,
                    w_max=self.w_max,
                ).fit(preferences)
                weights = fitted.weights
                base_objective = fitted.objective
                fit_metadata = dict(fitted.metadata)
            distance_from_prior = _profile_distance(
                self.global_fit.prototype, prototype, self.metric
            )
            objective = base_objective + self.prototype_penalty * distance_from_prior
            candidates.append(
                (
                    _selection_key(objective, prototype, weights),
                    prototype,
                    weights,
                    {
                        "base_objective": base_objective,
                        "profile_distance_from_global": distance_from_prior,
                        "weight_fit_metadata": fit_metadata,
                    },
                )
            )
        key, prototype, weights, diagnostics = min(candidates, key=lambda item: item[0])
        preferences = observations_to_preferences(rows, entities, prototype, self.metric)
        return LatentProfileFit(
            prototype=prototype,
            weights=weights,
            objective=float(key[0]),
            status="optimal_finite_profile_search",
            mode=mode,
            n_observations=len(rows),
            metadata={
                "support_size": len(rows),
                "candidate_count": len(search_space),
                "prototype_penalty": self.prototype_penalty,
                "selection_rule": "minimum_regularized_support_objective_then_prototype_id",
                "support_margin_metrics": margin_metrics(weights, preferences),
                **dict(diagnostics),
            },
        )


def evaluate_latent_fit(
    fitted: LatentProfileFit,
    observations: Iterable[PairwiseObservation],
    entities: Mapping[str, EntityRecord],
    metric: FittedMetric,
    *,
    probability_scale: float = 1.0,
) -> dict[str, float]:
    preferences = observations_to_preferences(observations, entities, fitted.prototype, metric)
    return margin_metrics(fitted.weights, preferences, probability_scale=probability_scale)

