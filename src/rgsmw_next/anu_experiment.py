from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .anu_car_preferences import (
    AnuCarPreference,
    AnuCarPreferenceDataset,
    AnuCarUser,
    anu_item_predicate_specs,
    anu_items_to_entity_records,
    anu_preferences_to_pairwise_observations,
    deterministic_support_evaluation_split,
    deterministic_user_grouped_folds,
)
from .archival_evaluation import (
    aggregate_repeated_cluster_rows,
    cluster_bootstrap_mean,
    paired_cluster_bootstrap,
)
from .choice_baselines import (
    BradleyTerryFit,
    FittedEntityFeatureEncoder,
    direct_utility_preferences,
    fit_bradley_terry,
)
from .distance import FittedMetric
from .learning import GlobalSimplexLearner, UnconstrainedRankSVM, identifiability_report
from .kernel_preference import (
    KernelItemCatalogue,
    fit_kernel_pairwise_rbf,
    select_kernel_anchor_strength,
    select_kernel_pairwise_rbf,
)
from .preference_learning import (
    GlobalLatentProfileSimplexLearner,
    LatentProfileFit,
    PairwiseObservation,
    PersonalizedLatentProfileSimplexLearner,
    build_prototype_grid,
    observations_to_preferences,
)
from .types import EntityRecord, FitResult, Preference


METHODS = (
    "uniform_label_free_profile",
    "global_simplex_fixed_profile",
    "global_latent_simplex",
    "personalized_simplex_weights",
    "personalized_simplex_profile",
    "personalized_simplex_joint",
    "global_ranksvm",
    "personalized_ranksvm_anchored",
    "demographic_interaction_ranksvm",
    "global_same_profile_typed_ranksvm",
    "personalized_same_profile_typed_ranksvm_anchored",
    "global_kernel_pairwise_rbf",
    "personalized_kernel_pairwise_rbf_anchored",
    "global_item_identity_bradley_terry",
    "personalized_item_identity_bradley_terry_anchored",
)

ANALYTICAL_RANDOM_REFERENCE = "analytical_random_choice_floor"
DEMOGRAPHIC_ATTRIBUTES = ("education", "age", "gender", "region")


@dataclass(frozen=True)
class FittedDemographicInteractionEncoder:
    """Direct item features crossed with archival, pre-choice demographics.

    The level vocabulary is learned from outer-fold *model-fit* users only.
    A held-out or calibration user whose level was not observed during fitting
    receives the global direct-item block and zeroes for that unseen
    interaction block.  Item attributes are fixed-catalogue covariates and are
    covered by the experiment's declared transductive catalogue scope.
    """

    item_encoder: FittedEntityFeatureEncoder
    levels: Mapping[str, tuple[int, ...]]

    @classmethod
    def fit(
        cls,
        item_encoder: FittedEntityFeatureEncoder,
        users: Iterable[AnuCarUser],
    ) -> "FittedDemographicInteractionEncoder":
        rows = tuple(users)
        if not rows:
            raise ValueError("Demographic interaction encoding requires model-fit users")
        levels = {
            attribute: tuple(
                sorted({int(getattr(user, attribute)) for user in rows})
            )
            for attribute in DEMOGRAPHIC_ATTRIBUTES
        }
        return cls(item_encoder=item_encoder, levels=levels)

    @property
    def feature_names(self) -> tuple[str, ...]:
        base = self.item_encoder.feature_names
        interactions = tuple(
            f"{attribute}={level}*{feature}"
            for attribute in DEMOGRAPHIC_ATTRIBUTES
            for level in self.levels[attribute]
            for feature in base
        )
        return (*base, *interactions)

    def transform_delta(
        self,
        preferred: EntityRecord,
        nonpreferred: EntityRecord,
        user: AnuCarUser,
    ) -> np.ndarray:
        base = self.item_encoder.transform(preferred) - self.item_encoder.transform(
            nonpreferred
        )
        blocks = [base]
        for attribute in DEMOGRAPHIC_ATTRIBUTES:
            observed = int(getattr(user, attribute))
            blocks.extend(
                base if observed == level else np.zeros_like(base)
                for level in self.levels[attribute]
            )
        return np.concatenate(blocks)


def demographic_interaction_preferences(
    observations: Iterable[PairwiseObservation],
    entities: Mapping[str, EntityRecord],
    users: Mapping[str, AnuCarUser],
    encoder: FittedDemographicInteractionEncoder,
) -> list[Preference]:
    """Create pairwise direct-utility contrasts without consulting test labels."""

    output: list[Preference] = []
    for observation in observations:
        try:
            user = users[observation.user_id]
        except KeyError as error:
            raise KeyError(
                f"No archived demographic row for {observation.user_id}"
            ) from error
        delta = encoder.transform_delta(
            entities[observation.preferred_id],
            entities[observation.nonpreferred_id],
            user,
        )
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


@dataclass(frozen=True)
class AnuExperimentConfig:
    """Predeclared protocol for the archival ANU preference experiment.

    The two source experiments are modelled separately.  Source-local user IDs
    are never used as cross-experiment participant identifiers because the
    archive does not provide a defensible linkage between them.
    """

    n_folds: int = 5
    fold_seed: int = 20260825
    support_sizes: tuple[int, ...] = (0, 2, 5, 10, 20)
    support_seeds: tuple[int, ...] = (20260821, 20260822, 20260823)
    minimum_correct_controls: int | None = None
    calibration_fraction: float = 0.20
    probability_scales: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
    prototype_grid_candidates: int = 12
    personalized_prototype_candidates: int = 4
    simplex_c: float = 5.0
    simplex_l2: float = 1e-2
    simplex_margin: float = 0.02
    personalized_weight_anchor: float = 1.0
    personalized_profile_penalty: float = 0.25
    ranksvm_c: float = 1.0
    ranksvm_l2: float = 1e-2
    ranksvm_margin: float = 0.02
    ranksvm_anchor_strength: float = 1.0
    kernel_length_scale_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0)
    kernel_l2_candidates: tuple[float, ...] = (1e-3, 1e-2, 1e-1)
    kernel_anchor_candidates: tuple[float, ...] = (0.1, 1.0, 10.0)
    kernel_anchor_calibration_support: int = 5
    kernel_jitter: float = 1e-9
    kernel_max_iterations: int = 2_000
    kernel_tolerance: float = 1e-9
    bradley_terry_l2: float = 1e-2
    bradley_terry_prior_strength: float = 1.0
    bradley_terry_max_iterations: int = 5_000
    bootstrap_replicates: int = 10_000
    bootstrap_confidence: float = 0.95
    bootstrap_seed: int = 20260825

    def __post_init__(self) -> None:
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least two")
        sizes = tuple(sorted(set(int(value) for value in self.support_sizes)))
        if not sizes or sizes[0] < 0:
            raise ValueError("support_sizes must contain non-negative integers")
        if sizes != self.support_sizes:
            raise ValueError("support_sizes must be sorted and unique")
        if 0 not in sizes:
            raise ValueError("support_sizes must include the zero-support condition")
        if not self.support_seeds or len(set(self.support_seeds)) != len(
            self.support_seeds
        ):
            raise ValueError("support_seeds must be non-empty and unique")
        if self.minimum_correct_controls is not None and self.minimum_correct_controls < 0:
            raise ValueError("minimum_correct_controls must be non-negative")
        if not 0.0 < self.calibration_fraction < 1.0:
            raise ValueError("calibration_fraction must lie strictly between zero and one")
        if not self.probability_scales or any(
            not math.isfinite(value) or value <= 0 for value in self.probability_scales
        ):
            raise ValueError("probability_scales must be finite and positive")
        if tuple(sorted(set(self.probability_scales))) != self.probability_scales:
            raise ValueError("probability_scales must be sorted and unique")
        positive_kernel_grids = {
            "kernel_length_scale_multipliers": self.kernel_length_scale_multipliers,
            "kernel_l2_candidates": self.kernel_l2_candidates,
            "kernel_anchor_candidates": self.kernel_anchor_candidates,
        }
        for name, values in positive_kernel_grids.items():
            if not values or any(
                not math.isfinite(value) or value <= 0 for value in values
            ):
                raise ValueError(f"{name} must contain finite positive values")
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be sorted and unique")
        if self.kernel_anchor_calibration_support <= 0:
            raise ValueError("kernel_anchor_calibration_support must be positive")
        if not math.isfinite(self.kernel_jitter) or self.kernel_jitter <= 0:
            raise ValueError("kernel_jitter must be finite and positive")
        if self.kernel_max_iterations <= 0:
            raise ValueError("kernel_max_iterations must be positive")
        if not math.isfinite(self.kernel_tolerance) or self.kernel_tolerance <= 0:
            raise ValueError("kernel_tolerance must be finite and positive")
        if self.prototype_grid_candidates <= 0:
            raise ValueError("prototype_grid_candidates must be positive")
        if self.personalized_prototype_candidates <= 0:
            raise ValueError("personalized_prototype_candidates must be positive")
        nonnegative = {
            "simplex_c": self.simplex_c,
            "simplex_l2": self.simplex_l2,
            "simplex_margin": self.simplex_margin,
            "personalized_weight_anchor": self.personalized_weight_anchor,
            "personalized_profile_penalty": self.personalized_profile_penalty,
            "ranksvm_c": self.ranksvm_c,
            "ranksvm_l2": self.ranksvm_l2,
            "ranksvm_margin": self.ranksvm_margin,
            "ranksvm_anchor_strength": self.ranksvm_anchor_strength,
            "bradley_terry_l2": self.bradley_terry_l2,
            "bradley_terry_prior_strength": self.bradley_terry_prior_strength,
        }
        if any(value < 0 for value in nonnegative.values()):
            raise ValueError("Learning penalties and margins must be non-negative")
        if self.bradley_terry_max_iterations <= 0:
            raise ValueError("bradley_terry_max_iterations must be positive")
        if self.bootstrap_replicates <= 0:
            raise ValueError("bootstrap_replicates must be positive")
        if not 0.0 < self.bootstrap_confidence < 1.0:
            raise ValueError("bootstrap_confidence must lie strictly between zero and one")


@dataclass(frozen=True)
class AnuExperimentResult:
    config: Mapping[str, object]
    manifest: Mapping[str, object]
    fold_models: tuple[Mapping[str, object], ...]
    adaptation_models: tuple[Mapping[str, object], ...]
    prediction_rows: tuple[Mapping[str, object], ...]
    summaries: tuple[Mapping[str, object], ...]
    paired_comparisons: tuple[Mapping[str, object], ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "rgsmw-anu-archival-experiment-1",
            "material_passport": {
                "origin": "archival ANU Car Preferences experiments 1 and 2",
                "verification_status": "ARCHIVAL_HUMAN_ORIGIN_NO_NEW_RECRUITMENT",
                "version": "official checksum-pinned archive snapshot declared by the acquisition manifest",
                "new_recruitment": False,
                "human_participation": (
                    "secondary analysis of archived human-origin preference responses"
                ),
                "ethics_note": (
                    "no new participants are recruited, but any required local secondary-data "
                    "ethics or data-use determination remains the researcher's responsibility"
                ),
                "scope_caveat": (
                    "source-local users are qualified by experiment and are not linked "
                    "across experiments"
                ),
            },
            "config": dict(self.config),
            "manifest": dict(self.manifest),
            "fold_models": [dict(row) for row in self.fold_models],
            "adaptation_models": [dict(row) for row in self.adaptation_models],
            "prediction_rows": [dict(row) for row in self.prediction_rows],
            "summaries": [dict(row) for row in self.summaries],
            "paired_comparisons": [dict(row) for row in self.paired_comparisons],
        }


def _digest(seed: int, *parts: object) -> str:
    raw = "\0".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _derived_seed(seed: int, *parts: object) -> int:
    return int(_digest(seed, *parts)[:8], 16)


def _group_digest(rows: Iterable[PairwiseObservation]) -> str:
    groups = sorted({row.group_id for row in rows})
    return hashlib.sha256("\n".join(groups).encode("utf-8")).hexdigest()


def _json_safe_identifiability(report: Mapping[str, object]) -> dict[str, object]:
    output = dict(report)
    condition = output.get("condition_number_nonzero")
    if isinstance(condition, (int, float)) and not math.isfinite(float(condition)):
        output["condition_number_nonzero"] = None
        output["condition_number_status"] = "undefined_no_nonzero_singular_values"
    else:
        output["condition_number_status"] = "finite"
    return output


def _metrics_from_margins(
    margins: Sequence[float] | np.ndarray,
    probability_scale: float,
) -> dict[str, float]:
    values = np.asarray(margins, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot evaluate an empty margin vector")
    scaled = np.clip(probability_scale * values, -40.0, 40.0)
    probabilities = 1.0 / (1.0 + np.exp(-scaled))
    return {
        "accuracy": float(np.mean(values > 0.0)),
        "tie_rate": float(np.mean(np.isclose(values, 0.0, atol=1e-12))),
        "mean_margin": float(np.mean(values)),
        "log_loss": float(np.mean(np.logaddexp(0.0, -scaled))),
        "brier": float(np.mean(np.square(1.0 - probabilities))),
    }


def _select_probability_scale(
    margins: Sequence[float] | np.ndarray,
    candidates: Sequence[float],
) -> float:
    values = np.asarray(margins, dtype=float)
    if values.size == 0:
        raise ValueError("A non-empty training-user calibration set is required")
    scored = [
        (
            float(np.mean(np.logaddexp(0.0, -np.clip(scale * values, -40.0, 40.0)))),
            float(scale),
        )
        for scale in candidates
    ]
    return min(scored, key=lambda row: (round(row[0], 14), row[1]))[1]


def _preference_margins(
    weights: Sequence[float], preferences: Iterable[Preference]
) -> np.ndarray:
    vector = np.asarray(weights, dtype=float)
    return np.asarray(
        [float(np.dot(vector, preference.delta)) for preference in preferences],
        dtype=float,
    )


def _latent_margins(
    fitted: LatentProfileFit,
    observations: Iterable[PairwiseObservation],
    entities: Mapping[str, EntityRecord],
    metric: FittedMetric,
) -> np.ndarray:
    preferences = observations_to_preferences(
        observations, entities, fitted.prototype, metric
    )
    return _preference_margins(fitted.weights, preferences)


def _bradley_terry_margins(
    fitted: BradleyTerryFit, observations: Iterable[PairwiseObservation]
) -> np.ndarray:
    utilities = fitted.utility_map()
    return np.asarray(
        [
            utilities[row.preferred_id] - utilities[row.nonpreferred_id]
            for row in observations
        ],
        dtype=float,
    )


def _label_free_medoid(
    records: Sequence[EntityRecord], metric: FittedMetric
) -> EntityRecord:
    weights = metric.uniform_weights()
    candidates: list[tuple[float, str, EntityRecord]] = []
    for candidate in records:
        distances = [
            metric.distance(candidate, other, weights)[0] for other in records
        ]
        candidates.append((float(np.mean(distances)), candidate.entity_id, candidate))
    return min(candidates, key=lambda row: (round(row[0], 14), row[1]))[2]


def _personalized_prototype_shortlist(
    global_fit: LatentProfileFit,
    prototypes: Sequence[EntityRecord],
    metric: FittedMetric,
    maximum: int,
) -> tuple[EntityRecord, ...]:
    """Shortlist without consulting any held-out user's support or evaluation labels."""

    scored = sorted(
        prototypes,
        key=lambda candidate: (
            round(
                float(np.mean(metric.feature_vector(global_fit.prototype, candidate))),
                14,
            ),
            candidate.entity_id,
        ),
    )
    ordered = [global_fit.prototype, *scored]
    unique: dict[str, EntityRecord] = {}
    for candidate in ordered:
        unique.setdefault(candidate.entity_id, candidate)
    return tuple(list(unique.values())[:maximum])


def _inner_user_split(
    train_user_keys: Sequence[tuple[str, str]],
    *,
    fraction: float,
    seed: int,
    experiment_id: str,
    fold: int,
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    if len(train_user_keys) < 2:
        raise ValueError(
            "Each outer training fold needs at least two users for model fitting and "
            "probability calibration"
        )
    ordered = sorted(
        train_user_keys,
        key=lambda key: (
            _digest(seed, "inner-calibration", experiment_id, fold, *key),
            key,
        ),
    )
    count = max(1, int(round(fraction * len(ordered))))
    count = min(count, len(ordered) - 1)
    calibration = tuple(ordered[:count])
    fit = tuple(ordered[count:])
    return fit, calibration


def _rows_for_users(
    rows: Sequence[AnuCarPreference], keys: set[tuple[str, str]]
) -> tuple[AnuCarPreference, ...]:
    return tuple(row for row in rows if row.user_key in keys)


def _as_observations(
    rows: Iterable[AnuCarPreference],
) -> tuple[PairwiseObservation, ...]:
    return anu_preferences_to_pairwise_observations(rows, control_policy="error")


def _collapse_consistent_pair_groups(
    rows: Iterable[AnuCarPreference],
) -> tuple[tuple[AnuCarPreference, ...], dict[str, object]]:
    """Give each user/item pair one vote and exclude contradictory repeats."""

    grouped: dict[tuple[str, str, str, str], list[AnuCarPreference]] = defaultdict(list)
    for row in rows:
        if row.control:
            raise ValueError("Pair-group collapse accepts non-control rows only")
        grouped[row.unordered_pair_key].append(row)
    retained: list[AnuCarPreference] = []
    conflicting: list[str] = []
    same_direction_repeats = 0
    for key, group in sorted(grouped.items()):
        directions = {(row.preferred, row.nonpreferred) for row in group}
        if len(directions) != 1:
            conflicting.append(":".join(key))
            continue
        same_direction_repeats += len(group) - 1
        retained.append(min(group, key=lambda row: row.source_line))
    return (
        tuple(retained),
        {
            "policy": "one_vote_per_user_unordered_item_pair",
            "input_rows": sum(len(group) for group in grouped.values()),
            "input_pair_groups": len(grouped),
            "retained_consistent_pair_groups": len(retained),
            "same_direction_repeat_rows_removed": same_direction_repeats,
            "conflicting_pair_groups_excluded": len(conflicting),
            "conflicting_pair_group_ids": conflicting,
        },
    )


def _latent_from_weight_fit(
    prototype: EntityRecord,
    fitted: FitResult,
    *,
    mode: str,
) -> LatentProfileFit:
    return LatentProfileFit(
        prototype=prototype,
        weights=fitted.weights,
        objective=fitted.objective,
        status=fitted.status,
        mode=mode,
        n_observations=fitted.n_preferences,
        metadata={"weight_fit_metadata": dict(fitted.metadata)},
    )


def _eligible_view(
    dataset: AnuCarPreferenceDataset,
    config: AnuExperimentConfig,
) -> tuple[
    tuple[tuple[str, str], ...],
    tuple[AnuCarPreference, ...],
    dict[str, object],
]:
    maximum_support = max(config.support_sizes)
    raw_noncontrol = tuple(row for row in dataset.preferences if not row.control)
    control_rows = [row for row in dataset.preferences if row.control]
    control_pair_keys = {row.unordered_pair_key for row in control_rows}
    quality_overlap_removed = 0
    if config.minimum_correct_controls is not None:
        if not control_rows:
            raise ValueError(
                "minimum_correct_controls requires a raw ANU view retaining control rows; "
                "otherwise quality-filter/evaluation overlap cannot be audited"
            )
        filtered_noncontrol = tuple(
            row for row in raw_noncontrol if row.unordered_pair_key not in control_pair_keys
        )
        quality_overlap_removed = len(raw_noncontrol) - len(filtered_noncontrol)
    else:
        filtered_noncontrol = raw_noncontrol
    noncontrol, pair_collapse_audit = _collapse_consistent_pair_groups(filtered_noncontrol)
    users = {(row.experiment_id, row.user_id): row for row in dataset.users}
    by_user: dict[tuple[str, str], list[AnuCarPreference]] = defaultdict(list)
    for row in noncontrol:
        by_user[row.user_key].append(row)

    eligible: list[tuple[str, str]] = []
    excluded: dict[str, list[str]] = defaultdict(list)
    group_counts: dict[tuple[str, str], int] = {}
    for key, user in sorted(users.items()):
        label = f"{key[0]}:{key[1]}"
        pair_groups = {row.unordered_pair_key for row in by_user.get(key, [])}
        group_counts[key] = len(pair_groups)
        if (
            config.minimum_correct_controls is not None
            and user.correct_controls < config.minimum_correct_controls
        ):
            excluded["below_minimum_correct_controls"].append(label)
            continue
        if len(pair_groups) <= maximum_support:
            excluded["insufficient_pair_groups_for_common_evaluation"].append(label)
            continue
        eligible.append(key)
    eligible_set = set(eligible)
    preferences = tuple(row for row in noncontrol if row.user_key in eligible_set)
    return (
        tuple(eligible),
        preferences,
        {
            "control_policy": "exclude",
            "control_rows_excluded_by_runner": len(control_rows),
            "minimum_correct_controls": config.minimum_correct_controls,
            "quality_filter_overlap_policy": (
                "all ordinary rows sharing a user/item pair with a control are removed "
                "when minimum_correct_controls is active"
                if config.minimum_correct_controls is not None
                else "not_applicable_no_quality_filter"
            ),
            "ordinary_rows_removed_for_quality_filter_independence": quality_overlap_removed,
            "pair_group_collapse": pair_collapse_audit,
            "maximum_support_pair_groups": maximum_support,
            "eligibility_rule": (
                "more unique non-control unordered item-pair groups than maximum support"
            ),
            "eligible_experiment_scoped_users": len(eligible),
            "eligible_preference_rows": len(preferences),
            "excluded_users": {name: rows for name, rows in sorted(excluded.items())},
            "eligible_pair_group_counts": {
                f"{key[0]}:{key[1]}": group_counts[key]
                for key in sorted(eligible_set)
            },
        },
    )


def _summary_scopes(
    rows: Sequence[Mapping[str, object]],
) -> Iterable[tuple[str, list[Mapping[str, object]]]]:
    experiments = sorted({str(row["experiment_id"]) for row in rows})
    for experiment in experiments:
        yield experiment, [row for row in rows if row["experiment_id"] == experiment]
    yield "pooled_experiment_scoped_users", list(rows)


def _summaries(
    rows: Sequence[Mapping[str, object]], config: AnuExperimentConfig
) -> tuple[dict[str, object], ...]:
    output: list[dict[str, object]] = []
    for scope, scoped in _summary_scopes(rows):
        for support_size in config.support_sizes:
            for method in METHODS:
                selected = [
                    row
                    for row in scoped
                    if row["support_size"] == support_size and row["method"] == method
                ]
                repeated = aggregate_repeated_cluster_rows(
                    selected,
                    cluster_key="user_key",
                    metric_keys=("accuracy", "log_loss", "brier", "tie_rate"),
                )
                metrics: dict[str, object] = {}
                for metric_name in ("accuracy", "log_loss", "brier", "tie_rate"):
                    values = {
                        user: scores[metric_name]
                        for user, scores in repeated.items()
                        if metric_name in scores
                    }
                    metrics[metric_name] = cluster_bootstrap_mean(
                        values,
                        seed=_derived_seed(
                            config.bootstrap_seed,
                            "summary",
                            scope,
                            support_size,
                            method,
                            metric_name,
                        ),
                        replicates=config.bootstrap_replicates,
                        confidence=config.bootstrap_confidence,
                    )
                output.append(
                    {
                        "scope": scope,
                        "support_size": support_size,
                        "method": method,
                        "support_repeats_averaged_before_inference": len(
                            config.support_seeds
                        ),
                        "metrics": metrics,
                    }
                )
    # A fair random chooser is an analytical reference, not an executed
    # system.  It therefore has no fabricated per-user rows, no bootstrap
    # interval, and no deterministic tie-break accuracy.  Its proper-score
    # expectations are exact for a reported probability of one half.
    for scope, _ in _summary_scopes(rows):
        for support_size in config.support_sizes:
            output.append(
                {
                    "scope": scope,
                    "support_size": support_size,
                    "method": ANALYTICAL_RANDOM_REFERENCE,
                    "result_type": "analytical_reference_not_executed_predictions",
                    "support_repeats_averaged_before_inference": 0,
                    "metrics": {
                        "accuracy": {
                            "mean": 0.5,
                            "status": "exact_expected_value_fair_random_choice",
                        },
                        "log_loss": {
                            "mean": float(math.log(2.0)),
                            "status": "exact_value_probability_one_half",
                        },
                        "brier": {
                            "mean": 0.25,
                            "status": "exact_value_probability_one_half",
                        },
                    },
                    "tie_rate": {
                        "status": (
                            "not_applicable_random_choice_has_no_deterministic_margin_"
                            "or_tie_break"
                        )
                    },
                    "inference": "not_applicable_analytical_reference",
                }
            )
    return tuple(output)


def _paired_comparisons(
    rows: Sequence[Mapping[str, object]], config: AnuExperimentConfig
) -> tuple[dict[str, object], ...]:
    reference = "global_latent_simplex"
    output: list[dict[str, object]] = []
    for scope, scoped in _summary_scopes(rows):
        for support_size in config.support_sizes:
            support_rows = [
                row for row in scoped if row["support_size"] == support_size
            ]
            for method in METHODS:
                if method == reference:
                    continue
                for metric_name in ("accuracy", "log_loss"):
                    first_rows = [row for row in support_rows if row["method"] == method]
                    reference_rows = [
                        row for row in support_rows if row["method"] == reference
                    ]
                    first_aggregate = aggregate_repeated_cluster_rows(
                        first_rows,
                        cluster_key="user_key",
                        metric_keys=(metric_name,),
                    )
                    reference_aggregate = aggregate_repeated_cluster_rows(
                        reference_rows,
                        cluster_key="user_key",
                        metric_keys=(metric_name,),
                    )
                    first = {
                        key: values[metric_name]
                        for key, values in first_aggregate.items()
                    }
                    second = {
                        key: values[metric_name]
                        for key, values in reference_aggregate.items()
                    }
                    inference = paired_cluster_bootstrap(
                        first,
                        second,
                        seed=_derived_seed(
                            config.bootstrap_seed,
                            "paired",
                            scope,
                            support_size,
                            method,
                            metric_name,
                        ),
                        replicates=config.bootstrap_replicates,
                        confidence=config.bootstrap_confidence,
                    )
                    output.append(
                        {
                            "scope": scope,
                            "support_size": support_size,
                            "method": method,
                            "reference": reference,
                            "metric": metric_name,
                            "favourable_direction": (
                                "positive" if metric_name == "accuracy" else "negative"
                            ),
                            "inference": inference,
                        }
                    )
    return tuple(output)


def run_anu_archival_experiment(
    dataset: AnuCarPreferenceDataset,
    config: AnuExperimentConfig | None = None,
) -> AnuExperimentResult:
    """Run leakage-controlled archival preference experiments.

    This function does not claim person-level longitudinal personalization.
    Its unit is the experiment-scoped user/session because the two ANU source
    experiments do not provide a verified respondent linkage.
    """

    protocol = config or AnuExperimentConfig()
    eligible_keys, preference_rows, exclusion_audit = _eligible_view(dataset, protocol)
    eligible_set = set(eligible_keys)
    eligible_users = tuple(
        user
        for user in dataset.users
        if (user.experiment_id, user.user_id) in eligible_set
    )
    if not eligible_users:
        raise ValueError("No ANU users satisfy the predeclared evaluation eligibility rule")
    fold_assignment = deterministic_user_grouped_folds(
        eligible_users, n_folds=protocol.n_folds, seed=protocol.fold_seed
    )
    fold_map = fold_assignment.as_mapping()

    all_records = anu_items_to_entity_records(dataset.items)
    user_lookup = {
        f"anu:{user.experiment_id}:user:{user.user_id}": user
        for user in eligible_users
    }
    experiments = sorted({key[0] for key in eligible_keys})
    prediction_rows: list[dict[str, object]] = []
    fold_models: list[dict[str, object]] = []
    adaptation_models: list[dict[str, object]] = []

    for experiment_id in experiments:
        experiment_keys = tuple(key for key in eligible_keys if key[0] == experiment_id)
        experiment_preferences = tuple(
            row for row in preference_rows if row.experiment_id == experiment_id
        )
        records = tuple(
            record
            for record in all_records
            if record.entity_id.startswith(f"anu:{experiment_id}:item:")
        )
        if not records:
            raise ValueError(f"Experiment {experiment_id} has no item records")
        specs = tuple(
            spec
            for spec in anu_item_predicate_specs()
            if any(record.get(spec.field) for record in records)
        )
        metric = FittedMetric.fit(records, specs)
        entities = {record.entity_id: record for record in records}
        item_ids = tuple(sorted(entities))
        encoder = FittedEntityFeatureEncoder.fit(records, metric)
        kernel_catalogue = KernelItemCatalogue.from_records(records, encoder)
        label_free_profile = _label_free_medoid(records, metric)
        prototypes = build_prototype_grid(
            records,
            metric.fields,
            max_candidates=protocol.prototype_grid_candidates,
        )

        for fold in range(protocol.n_folds):
            heldout_keys = tuple(key for key in experiment_keys if fold_map[key] == fold)
            train_keys = tuple(key for key in experiment_keys if fold_map[key] != fold)
            fit_keys, calibration_keys = _inner_user_split(
                train_keys,
                fraction=protocol.calibration_fraction,
                seed=protocol.fold_seed,
                experiment_id=experiment_id,
                fold=fold,
            )
            fit_rows = _rows_for_users(experiment_preferences, set(fit_keys))
            calibration_rows = _rows_for_users(
                experiment_preferences, set(calibration_keys)
            )
            heldout_rows = _rows_for_users(experiment_preferences, set(heldout_keys))
            fit_observations = _as_observations(fit_rows)
            calibration_observations = _as_observations(calibration_rows)
            model_fit_users = tuple(
                user
                for user in eligible_users
                if (user.experiment_id, user.user_id) in set(fit_keys)
            )
            demographic_encoder = FittedDemographicInteractionEncoder.fit(
                encoder, model_fit_users
            )

            fixed_preferences = observations_to_preferences(
                fit_observations, entities, label_free_profile, metric
            )
            fixed_weight_fit = GlobalSimplexLearner(
                dimension=len(metric.predicates),
                c=protocol.simplex_c,
                l2=protocol.simplex_l2,
                margin=protocol.simplex_margin,
            ).fit(fixed_preferences)
            fixed_latent_fit = _latent_from_weight_fit(
                label_free_profile,
                fixed_weight_fit,
                mode="global_weights_fixed_label_free_profile",
            )
            uniform_fit = LatentProfileFit(
                prototype=label_free_profile,
                weights=tuple(float(value) for value in metric.uniform_weights()),
                objective=0.0,
                status="label_free",
                mode="uniform_label_free_profile",
                n_observations=0,
            )
            global_latent_fit = GlobalLatentProfileSimplexLearner(
                metric=metric,
                prototypes=prototypes,
                c=protocol.simplex_c,
                l2=protocol.simplex_l2,
                margin=protocol.simplex_margin,
            ).fit(fit_observations, entities)
            same_profile_fit_preferences = observations_to_preferences(
                fit_observations,
                entities,
                global_latent_fit.prototype,
                metric,
            )
            global_same_profile_ranksvm_fit = UnconstrainedRankSVM(
                dimension=len(metric.predicates),
                c=protocol.ranksvm_c,
                l2=protocol.ranksvm_l2,
                margin=protocol.ranksvm_margin,
            ).fit(same_profile_fit_preferences)
            personalized_prototypes = _personalized_prototype_shortlist(
                global_latent_fit,
                prototypes,
                metric,
                protocol.personalized_prototype_candidates,
            )

            direct_fit_preferences = direct_utility_preferences(
                fit_observations, entities, encoder
            )
            global_ranksvm_fit = UnconstrainedRankSVM(
                dimension=len(encoder.feature_names),
                c=protocol.ranksvm_c,
                l2=protocol.ranksvm_l2,
                margin=protocol.ranksvm_margin,
            ).fit(direct_fit_preferences)
            demographic_fit_preferences = demographic_interaction_preferences(
                fit_observations,
                entities,
                user_lookup,
                demographic_encoder,
            )
            demographic_ranksvm_fit = UnconstrainedRankSVM(
                dimension=len(demographic_encoder.feature_names),
                c=protocol.ranksvm_c,
                l2=protocol.ranksvm_l2,
                margin=protocol.ranksvm_margin,
            ).fit(demographic_fit_preferences)
            global_bradley_fit = fit_bradley_terry(
                fit_observations,
                item_ids,
                l2=protocol.bradley_terry_l2,
                max_iterations=protocol.bradley_terry_max_iterations,
            )

            # Both predictive kernel hyperparameters and the personalization
            # anchor are selected entirely inside the outer-training fold.
            # Calibration users are atomic and never overlap model-fit or
            # outer-held-out users.
            kernel_selection = select_kernel_pairwise_rbf(
                fit_observations,
                calibration_observations,
                kernel_catalogue,
                length_scale_multipliers=protocol.kernel_length_scale_multipliers,
                l2_candidates=protocol.kernel_l2_candidates,
                jitter=protocol.kernel_jitter,
                max_iterations=protocol.kernel_max_iterations,
                tolerance=protocol.kernel_tolerance,
            )
            global_kernel_fit = kernel_selection.fit
            kernel_anchor_selection = select_kernel_anchor_strength(
                global_kernel_fit,
                calibration_observations,
                anchor_candidates=protocol.kernel_anchor_candidates,
                support_size_per_user=protocol.kernel_anchor_calibration_support,
                seed=_derived_seed(
                    protocol.fold_seed,
                    "kernel-anchor-calibration",
                    experiment_id,
                    fold,
                ),
                jitter=protocol.kernel_jitter,
                max_iterations=protocol.kernel_max_iterations,
                tolerance=protocol.kernel_tolerance,
            )

            calibration_fixed_preferences = observations_to_preferences(
                calibration_observations, entities, label_free_profile, metric
            )
            calibration_direct_preferences = direct_utility_preferences(
                calibration_observations, entities, encoder
            )
            calibration_demographic_preferences = demographic_interaction_preferences(
                calibration_observations,
                entities,
                user_lookup,
                demographic_encoder,
            )
            calibration_same_profile_preferences = observations_to_preferences(
                calibration_observations,
                entities,
                global_latent_fit.prototype,
                metric,
            )
            probability_scales = {
                "uniform": _select_probability_scale(
                    _preference_margins(
                        uniform_fit.weights, calibration_fixed_preferences
                    ),
                    protocol.probability_scales,
                ),
                "fixed_simplex": _select_probability_scale(
                    _preference_margins(
                        fixed_latent_fit.weights, calibration_fixed_preferences
                    ),
                    protocol.probability_scales,
                ),
                "latent_simplex": _select_probability_scale(
                    _latent_margins(
                        global_latent_fit,
                        calibration_observations,
                        entities,
                        metric,
                    ),
                    protocol.probability_scales,
                ),
                "ranksvm": _select_probability_scale(
                    _preference_margins(
                        global_ranksvm_fit.weights,
                        calibration_direct_preferences,
                    ),
                    protocol.probability_scales,
                ),
                "demographic_ranksvm": _select_probability_scale(
                    _preference_margins(
                        demographic_ranksvm_fit.weights,
                        calibration_demographic_preferences,
                    ),
                    protocol.probability_scales,
                ),
                "same_profile_typed_ranksvm": _select_probability_scale(
                    _preference_margins(
                        global_same_profile_ranksvm_fit.weights,
                        calibration_same_profile_preferences,
                    ),
                    protocol.probability_scales,
                ),
                "bradley_terry": _select_probability_scale(
                    _bradley_terry_margins(
                        global_bradley_fit, calibration_observations
                    ),
                    protocol.probability_scales,
                ),
            }

            fold_models.append(
                {
                    "experiment_id": experiment_id,
                    "fold": fold,
                    "heldout_users": [f"{key[0]}:{key[1]}" for key in heldout_keys],
                    "outer_training_users": [f"{key[0]}:{key[1]}" for key in train_keys],
                    "model_fit_users": [f"{key[0]}:{key[1]}" for key in fit_keys],
                    "calibration_users": [
                        f"{key[0]}:{key[1]}" for key in calibration_keys
                    ],
                    "model_fit_comparisons": len(fit_observations),
                    "calibration_comparisons": len(calibration_observations),
                    "active_predicates": list(metric.fields),
                    "metric": metric.scale_manifest(),
                    "fixed_label_free_profile": label_free_profile.entity_id,
                    "prototype_policy": {
                        "global_grid_cap_before_observed_item_union": protocol.prototype_grid_candidates,
                        "global_candidate_count_after_observed_item_union": len(prototypes),
                        "personalized_policy": (
                            "global selected profile plus nearest candidates under the "
                            "training-fold metric; no held-out labels consulted"
                        ),
                        "personalized_candidate_cap": protocol.personalized_prototype_candidates,
                        "personalized_candidate_ids": [
                            row.entity_id for row in personalized_prototypes
                        ],
                    },
                    "global_simplex_fixed_profile": {
                        "weights": list(fixed_latent_fit.weights),
                        "profile_id": fixed_latent_fit.prototype.entity_id,
                        "status": fixed_latent_fit.status,
                    },
                    "global_latent_simplex": {
                        "weights": list(global_latent_fit.weights),
                        "profile_id": global_latent_fit.prototype.entity_id,
                        "status": global_latent_fit.status,
                    },
                    "global_ranksvm": {
                        "scope": "direct_item_attribute_utility_no_user_covariates",
                        "feature_names": list(encoder.feature_names),
                        "weights": list(global_ranksvm_fit.weights),
                        "status": global_ranksvm_fit.status,
                    },
                    "demographic_interaction_ranksvm": {
                        "scope": (
                            "cold_start_direct_item_utility_conditioned_on_archived_"
                            "education_age_gender_region"
                        ),
                        "training_scope": "outer_fold_model_fit_users_only",
                        "heldout_inputs": (
                            "archived_pre_choice_demographics_and_fixed_catalogue_item_"
                            "attributes_only_no_heldout_choice_labels"
                        ),
                        "demographic_attributes": list(DEMOGRAPHIC_ATTRIBUTES),
                        "demographic_levels_learned_from_model_fit_users": {
                            key: list(values)
                            for key, values in demographic_encoder.levels.items()
                        },
                        "feature_names": list(demographic_encoder.feature_names),
                        "weights": list(demographic_ranksvm_fit.weights),
                        "status": demographic_ranksvm_fit.status,
                    },
                    "global_same_profile_typed_ranksvm": {
                        "scope": (
                            "unconstrained_weight_comparator_on_exact_global_latent_"
                            "simplex_profile_and_typed_distance_delta_components"
                        ),
                        "profile_id": global_latent_fit.prototype.entity_id,
                        "feature_names": list(metric.fields),
                        "weights": list(global_same_profile_ranksvm_fit.weights),
                        "status": global_same_profile_ranksvm_fit.status,
                    },
                    "global_kernel_pairwise_rbf": {
                        "scope": (
                            "nonlinear_typed_item_attribute_utility_in_the_fixed_"
                            "catalogue_transductive_setting"
                        ),
                        "claim_boundary": (
                            "convex finite-catalogue RBF kernel logistic preference "
                            "baseline; not a full GP or DP-GP"
                        ),
                        "training_scope": "outer_fold_model_fit_users_only",
                        "catalogue": kernel_catalogue.audit(),
                        "hyperparameter_selection": kernel_selection.audit(),
                        "personalization_anchor_selection": (
                            kernel_anchor_selection.audit()
                        ),
                        "fit": global_kernel_fit.audit(),
                        "probability_scale": {
                            "value": 1.0,
                            "reason": "intrinsic_pairwise_logistic_probability_link",
                        },
                    },
                    "global_item_identity_bradley_terry": {
                        "scope": (
                            "fixed_catalogue_transductive_item_identity_popularity_"
                            "baseline_no_item_attribute_generalization"
                        ),
                        "item_ids": list(global_bradley_fit.item_ids),
                        "utilities": list(global_bradley_fit.utilities),
                        "converged": global_bradley_fit.converged,
                    },
                    "probability_scale_selection": {
                        "unit": "outer-training experiment-scoped users held out from model fitting",
                        "criterion": "minimum mean binary log loss on fixed predeclared grid",
                        "selected": probability_scales,
                    },
                }
            )

            for support_seed in protocol.support_seeds:
                splits = {
                    size: deterministic_support_evaluation_split(
                        heldout_rows,
                        support_size_per_user=size,
                        seed=support_seed,
                        control_policy="exclude",
                        require_evaluation=True,
                    )
                    for size in protocol.support_sizes
                }
                maximum_split = splits[max(protocol.support_sizes)]
                common_evaluation = _as_observations(maximum_split.evaluation)
                evaluation_by_user: dict[str, list[PairwiseObservation]] = defaultdict(list)
                for observation in common_evaluation:
                    evaluation_by_user[observation.user_id].append(observation)
                maximum_support_groups = {
                    observation.group_id
                    for observation in _as_observations(maximum_split.support)
                }

                for support_size in protocol.support_sizes:
                    support_observations = _as_observations(splits[support_size].support)
                    if not {
                        observation.group_id for observation in support_observations
                    }.issubset(maximum_support_groups):
                        raise AssertionError("Nested support construction failed")
                    support_by_user: dict[str, list[PairwiseObservation]] = defaultdict(list)
                    for observation in support_observations:
                        support_by_user[observation.user_id].append(observation)

                    personalized_latent_learner = PersonalizedLatentProfileSimplexLearner(
                        metric=metric,
                        prototypes=personalized_prototypes,
                        global_fit=global_latent_fit,
                        c=protocol.simplex_c,
                        mu=protocol.personalized_weight_anchor,
                        prototype_penalty=protocol.personalized_profile_penalty,
                        margin=protocol.simplex_margin,
                    )

                    for experiment_key in heldout_keys:
                        source_user_id = experiment_key[1]
                        user_key = f"anu:{experiment_id}:user:{source_user_id}"
                        support = tuple(support_by_user.get(user_key, ()))
                        evaluation = tuple(evaluation_by_user.get(user_key, ()))
                        if not evaluation:
                            raise AssertionError(
                                f"Common evaluation is empty for {user_key}"
                            )
                        support_groups = {row.group_id for row in support}
                        evaluation_groups = {row.group_id for row in evaluation}
                        if support_groups & evaluation_groups:
                            raise AssertionError("Held-out support/evaluation leakage")

                        personalized_latent = {
                            mode: personalized_latent_learner.fit(
                                support, entities, mode=mode
                            )
                            for mode in ("weights_only", "profile_only", "joint")
                        }
                        evaluation_direct = direct_utility_preferences(
                            evaluation, entities, encoder
                        )
                        support_direct = direct_utility_preferences(
                            support, entities, encoder
                        )
                        evaluation_demographic = demographic_interaction_preferences(
                            evaluation,
                            entities,
                            user_lookup,
                            demographic_encoder,
                        )
                        support_same_profile = observations_to_preferences(
                            support,
                            entities,
                            global_latent_fit.prototype,
                            metric,
                        )
                        evaluation_same_profile = observations_to_preferences(
                            evaluation,
                            entities,
                            global_latent_fit.prototype,
                            metric,
                        )
                        personalized_ranksvm = UnconstrainedRankSVM(
                            dimension=len(encoder.feature_names),
                            c=protocol.ranksvm_c,
                            l2=protocol.ranksvm_l2,
                            margin=protocol.ranksvm_margin,
                            anchor=global_ranksvm_fit.weights,
                            anchor_strength=protocol.ranksvm_anchor_strength,
                        ).fit(support_direct)
                        personalized_same_profile_ranksvm = UnconstrainedRankSVM(
                            dimension=len(metric.predicates),
                            c=protocol.ranksvm_c,
                            l2=protocol.ranksvm_l2,
                            margin=protocol.ranksvm_margin,
                            anchor=global_same_profile_ranksvm_fit.weights,
                            anchor_strength=protocol.ranksvm_anchor_strength,
                        ).fit(support_same_profile)
                        personalized_kernel_fit = fit_kernel_pairwise_rbf(
                            support,
                            kernel_catalogue,
                            length_scale_multiplier=(
                                global_kernel_fit.length_scale_multiplier
                            ),
                            l2=global_kernel_fit.l2,
                            prior=global_kernel_fit,
                            anchor_strength=kernel_anchor_selection.anchor_strength,
                            jitter=protocol.kernel_jitter,
                            max_iterations=protocol.kernel_max_iterations,
                            tolerance=protocol.kernel_tolerance,
                        )
                        personalized_bradley = fit_bradley_terry(
                            support,
                            item_ids,
                            l2=protocol.bradley_terry_l2,
                            prior=global_bradley_fit.utility_map(),
                            prior_strength=protocol.bradley_terry_prior_strength,
                            max_iterations=protocol.bradley_terry_max_iterations,
                        )
                        joint_support_preferences = observations_to_preferences(
                            support,
                            entities,
                            personalized_latent["joint"].prototype,
                            metric,
                        )
                        adaptation_models.append(
                            {
                                "experiment_id": experiment_id,
                                "fold": fold,
                                "support_seed": support_seed,
                                "support_size": support_size,
                                "source_user_id": source_user_id,
                                "user_key": user_key,
                                "support_rows": len(support),
                                "support_pair_groups": len(support_groups),
                                "support_group_digest": _group_digest(support),
                                "simplex_identifiability": _json_safe_identifiability(
                                    identifiability_report(
                                        joint_support_preferences,
                                        dimension=len(metric.predicates),
                                    )
                                ),
                                "personalized_simplex": {
                                    mode: {
                                        "profile_id": fitted.prototype.entity_id,
                                        "weights": list(fitted.weights),
                                        "objective": fitted.objective,
                                        "status": fitted.status,
                                        "metadata": dict(fitted.metadata),
                                    }
                                    for mode, fitted in personalized_latent.items()
                                },
                                "personalized_ranksvm_anchored": {
                                    "weights": list(personalized_ranksvm.weights),
                                    "status": personalized_ranksvm.status,
                                    "metadata": dict(personalized_ranksvm.metadata),
                                },
                                "personalized_same_profile_typed_ranksvm_anchored": {
                                    "profile_id": global_latent_fit.prototype.entity_id,
                                    "feature_names": list(metric.fields),
                                    "weights": list(
                                        personalized_same_profile_ranksvm.weights
                                    ),
                                    "status": personalized_same_profile_ranksvm.status,
                                    "metadata": dict(
                                        personalized_same_profile_ranksvm.metadata
                                    ),
                                },
                                "personalized_kernel_pairwise_rbf_anchored": {
                                    **personalized_kernel_fit.audit(),
                                    "global_anchor_observation_digest_sha256": (
                                        global_kernel_fit.diagnostics.get(
                                            "observation_digest_sha256"
                                        )
                                    ),
                                    "selected_anchor_strength_source": (
                                        "outer_training_calibration_users_only"
                                    ),
                                },
                                "personalized_item_identity_bradley_terry_anchored": {
                                    "item_ids": list(personalized_bradley.item_ids),
                                    "utilities": list(personalized_bradley.utilities),
                                    "converged": personalized_bradley.converged,
                                    "metadata": dict(personalized_bradley.metadata),
                                },
                            }
                        )

                        margin_by_method = {
                            "uniform_label_free_profile": _latent_margins(
                                uniform_fit, evaluation, entities, metric
                            ),
                            "global_simplex_fixed_profile": _latent_margins(
                                fixed_latent_fit, evaluation, entities, metric
                            ),
                            "global_latent_simplex": _latent_margins(
                                global_latent_fit, evaluation, entities, metric
                            ),
                            "personalized_simplex_weights": _latent_margins(
                                personalized_latent["weights_only"],
                                evaluation,
                                entities,
                                metric,
                            ),
                            "personalized_simplex_profile": _latent_margins(
                                personalized_latent["profile_only"],
                                evaluation,
                                entities,
                                metric,
                            ),
                            "personalized_simplex_joint": _latent_margins(
                                personalized_latent["joint"],
                                evaluation,
                                entities,
                                metric,
                            ),
                            "global_ranksvm": _preference_margins(
                                global_ranksvm_fit.weights, evaluation_direct
                            ),
                            "personalized_ranksvm_anchored": _preference_margins(
                                personalized_ranksvm.weights, evaluation_direct
                            ),
                            "demographic_interaction_ranksvm": _preference_margins(
                                demographic_ranksvm_fit.weights,
                                evaluation_demographic,
                            ),
                            "global_same_profile_typed_ranksvm": _preference_margins(
                                global_same_profile_ranksvm_fit.weights,
                                evaluation_same_profile,
                            ),
                            "personalized_same_profile_typed_ranksvm_anchored": (
                                _preference_margins(
                                    personalized_same_profile_ranksvm.weights,
                                    evaluation_same_profile,
                                )
                            ),
                            "global_kernel_pairwise_rbf": global_kernel_fit.margins(
                                evaluation
                            ),
                            "personalized_kernel_pairwise_rbf_anchored": (
                                personalized_kernel_fit.margins(evaluation)
                            ),
                            "global_item_identity_bradley_terry": _bradley_terry_margins(
                                global_bradley_fit, evaluation
                            ),
                            "personalized_item_identity_bradley_terry_anchored": (
                                _bradley_terry_margins(
                                    personalized_bradley, evaluation
                                )
                            ),
                        }
                        scale_by_method = {
                            "uniform_label_free_profile": probability_scales["uniform"],
                            "global_simplex_fixed_profile": probability_scales[
                                "fixed_simplex"
                            ],
                            "global_latent_simplex": probability_scales[
                                "latent_simplex"
                            ],
                            "personalized_simplex_weights": probability_scales[
                                "latent_simplex"
                            ],
                            "personalized_simplex_profile": probability_scales[
                                "latent_simplex"
                            ],
                            "personalized_simplex_joint": probability_scales[
                                "latent_simplex"
                            ],
                            "global_ranksvm": probability_scales["ranksvm"],
                            "personalized_ranksvm_anchored": probability_scales[
                                "ranksvm"
                            ],
                            "demographic_interaction_ranksvm": probability_scales[
                                "demographic_ranksvm"
                            ],
                            "global_same_profile_typed_ranksvm": probability_scales[
                                "same_profile_typed_ranksvm"
                            ],
                            "personalized_same_profile_typed_ranksvm_anchored": (
                                probability_scales["same_profile_typed_ranksvm"]
                            ),
                            "global_kernel_pairwise_rbf": 1.0,
                            "personalized_kernel_pairwise_rbf_anchored": 1.0,
                            "global_item_identity_bradley_terry": probability_scales[
                                "bradley_terry"
                            ],
                            "personalized_item_identity_bradley_terry_anchored": (
                                probability_scales["bradley_terry"]
                            ),
                        }
                        for method in METHODS:
                            metrics = _metrics_from_margins(
                                margin_by_method[method], scale_by_method[method]
                            )
                            prediction_rows.append(
                                {
                                    "experiment_id": experiment_id,
                                    "fold": fold,
                                    "support_seed": support_seed,
                                    "support_size": support_size,
                                    "source_user_id": source_user_id,
                                    "user_key": user_key,
                                    "method": method,
                                    "support_rows": len(support),
                                    "support_pair_groups": len(support_groups),
                                    "evaluation_rows": len(evaluation),
                                    "evaluation_pair_groups": len(evaluation_groups),
                                    "support_group_digest": _group_digest(support),
                                    "evaluation_group_digest": _group_digest(evaluation),
                                    "probability_scale": scale_by_method[method],
                                    **metrics,
                                }
                            )

    stable_rows = tuple(
        sorted(
            prediction_rows,
            key=lambda row: (
                str(row["experiment_id"]),
                int(row["fold"]),
                int(row["support_seed"]),
                int(row["support_size"]),
                str(row["source_user_id"]),
                str(row["method"]),
            ),
        )
    )
    manifest = {
        "target_relation": "strict pairwise vehicle preference in the archived ANU tasks",
        "inference_unit": "experiment-scoped user/session",
        "participant_linkage_caveat": (
            "Equal source-local user numbers across experiments are not assumed to identify "
            "the same person; experiments are fitted separately."
        ),
        "outer_validation": (
            "balanced experiment-stratified user-held-out folds; no held-out user label "
            "enters global model fitting or probability calibration"
        ),
        "inner_calibration": (
            "outer-training users split atomically into model-fit and calibration users"
        ),
        "adaptation_rule": (
            "personalized models receive only nested support unordered-pair groups"
        ),
        "evaluation_rule": (
            "within each support seed, all support sizes use the common evaluation set "
            "remaining after the maximum support prefix"
        ),
        "repeated_pair_rule": (
            "all rows sharing a user-specific unordered item pair remain in one split role"
        ),
        "catalogue_scope": (
            "fixed-catalogue transductive evaluation: item attributes are known; choice "
            "labels, not alternatives, are held out"
        ),
        "demographic_covariate_scope": (
            "education, age, gender, and region are archived pre-choice covariates; "
            "interaction vocabularies and coefficients are learned only from outer-fold "
            "model-fit users, while held-out attributes are known without held-out labels"
        ),
        "controls": (
            "archived reversed control rows are excluded from model fitting, support, and "
            "evaluation"
        ),
        "exclusions": exclusion_audit,
        "fold_assignment": dict(fold_assignment.audit),
        "source_filter_audit": dict(dataset.audit.get("filter", {})),
        "methods": list(METHODS),
        "method_semantics": {
            "global_ranksvm": (
                "unconstrained direct item-attribute utility without user covariates"
            ),
            "personalized_ranksvm_anchored": (
                "support-adapted direct item-attribute utility anchored to global_ranksvm"
            ),
            "demographic_interaction_ranksvm": (
                "cold-start direct item-attribute utility with education/age/gender/region "
                "interactions and no held-out choice labels"
            ),
            "global_same_profile_typed_ranksvm": (
                "unconstrained weights on exactly the global_latent_simplex profile and "
                "typed-distance delta components; isolates the simplex constraint"
            ),
            "personalized_same_profile_typed_ranksvm_anchored": (
                "support-adapted unconstrained typed-distance weights on the same fixed "
                "global profile, anchored to global_same_profile_typed_ranksvm"
            ),
            "global_kernel_pairwise_rbf": (
                "nonlinear typed item-attribute utility learned with an RBF kernel and "
                "pairwise logistic loss; kernel bandwidth and RKHS penalty are selected "
                "on outer-training calibration users"
            ),
            "personalized_kernel_pairwise_rbf_anchored": (
                "support-adapted RBF kernel utility regularized toward the global kernel "
                "fit; anchor strength is selected by within-calibration-user adaptation"
            ),
            "global_item_identity_bradley_terry": (
                "fixed-catalogue transductive item-identity popularity baseline"
            ),
            "personalized_item_identity_bradley_terry_anchored": (
                "support-adapted item-identity utilities anchored to the global "
                "Bradley-Terry popularity baseline"
            ),
        },
        "analytical_references": {
            ANALYTICAL_RANDOM_REFERENCE: {
                "expected_accuracy": 0.5,
                "probability": 0.5,
                "log_loss": float(math.log(2.0)),
                "brier": 0.25,
                "prediction_rows": False,
                "tie_break_accuracy": "not_defined_or_reported",
                "reason": (
                    "an exact sanity floor is reported without manufacturing sampled "
                    "per-user predictions or deterministic tie wins"
                ),
            }
        },
        "inference": (
            "row metrics are averaged within experiment-scoped user across support seeds; "
            "percentile confidence intervals resample users"
        ),
        "pooled_summary_status": (
            "secondary; experiment-specific summaries remain primary because item designs differ"
        ),
    }
    return AnuExperimentResult(
        config=asdict(protocol),
        manifest=manifest,
        fold_models=tuple(
            sorted(
                fold_models,
                key=lambda row: (str(row["experiment_id"]), int(row["fold"])),
            )
        ),
        adaptation_models=tuple(
            sorted(
                adaptation_models,
                key=lambda row: (
                    str(row["experiment_id"]),
                    int(row["fold"]),
                    int(row["support_seed"]),
                    int(row["support_size"]),
                    str(row["source_user_id"]),
                ),
            )
        ),
        prediction_rows=stable_rows,
        summaries=_summaries(stable_rows, protocol),
        paired_comparisons=_paired_comparisons(stable_rows, protocol),
    )
