from __future__ import annotations

"""Dependency-light nonlinear pairwise preference baselines.

The implementation is deliberately named ``kernel_pairwise_rbf``.  It is a
finite-catalogue kernel logistic preference model, not a Gaussian process and
not the DP-GP model from the preference-learning literature.  This distinction
matters: the model supplies nonlinear utilities and logistic probabilities,
but it does not supply a GP posterior or posterior uncertainty.
"""

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from .choice_baselines import FittedEntityFeatureEncoder
from .preference_learning import PairwiseObservation
from .types import EntityRecord


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _stable_digest(seed: int, *parts: object) -> str:
    payload = "\0".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ordered_observations(
    observations: Iterable[PairwiseObservation],
) -> tuple[PairwiseObservation, ...]:
    rows = tuple(observations)
    self_comparisons = sorted(
        {row.comparison_id for row in rows if row.preferred_id == row.nonpreferred_id}
    )
    if self_comparisons:
        raise ValueError(
            "Strict pairwise preference observations cannot compare an item with itself: "
            f"{self_comparisons}"
        )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.user_id,
                row.group_id,
                row.comparison_id,
                row.preferred_id,
                row.nonpreferred_id,
            ),
        )
    )


def _observation_digest(observations: Sequence[PairwiseObservation]) -> str:
    payload = "\n".join(
        "\0".join(
            (
                row.user_id,
                row.group_id,
                row.comparison_id,
                row.preferred_id,
                row.nonpreferred_id,
            )
        )
        for row in observations
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finite_or_none(value: float) -> float | None:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


@dataclass(frozen=True)
class KernelItemCatalogue:
    """Typed, label-free representation of one fixed alternative catalogue.

    Numeric, ordinal, categorical, and explicit missingness columns are
    supplied by :class:`FittedEntityFeatureEncoder`.  Columns belonging to the
    same RDF predicate are divided by the square root of their block size so a
    categorical predicate with many one-hot columns does not receive more
    distance mass merely because it has a larger encoding block.
    """

    item_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_matrix: np.ndarray = field(repr=False, compare=False)
    squared_distances: np.ndarray = field(repr=False, compare=False)
    median_nonzero_distance: float
    feature_digest: str
    predicate_block_audit: tuple[Mapping[str, object], ...]

    @classmethod
    def from_records(
        cls,
        records: Iterable[EntityRecord],
        encoder: FittedEntityFeatureEncoder,
    ) -> "KernelItemCatalogue":
        ordered = tuple(sorted(records, key=lambda row: row.entity_id))
        if len(ordered) < 2:
            raise ValueError("A kernel preference catalogue requires at least two items")
        if len({row.entity_id for row in ordered}) != len(ordered):
            raise ValueError("Kernel catalogue item IDs must be unique")

        features = np.vstack([encoder.transform(row) for row in ordered]).astype(
            float, copy=False
        )
        if features.ndim != 2 or features.shape[1] == 0:
            raise ValueError("The typed item encoder produced no usable feature columns")
        if not np.all(np.isfinite(features)):
            raise ValueError("Kernel catalogue features must be finite")

        # Equalize the median non-zero squared distance of each varying typed
        # predicate.  This is a linear rescaling of explicit feature columns,
        # so the subsequent Gaussian RBF remains positive semidefinite.  It
        # also avoids making a categorical predicate less influential merely
        # because its one-hot vocabulary contains more levels.
        field_names = tuple(name.split(":", 1)[0] for name in encoder.feature_names)
        ordered_fields = tuple(dict.fromkeys(field_names))
        block_medians: dict[str, float | None] = {}
        for field_name in ordered_fields:
            positions = [
                index for index, observed in enumerate(field_names) if observed == field_name
            ]
            block = features[:, positions]
            differences = block[:, None, :] - block[None, :, :]
            block_squared = np.einsum("ijk,ijk->ij", differences, differences)
            upper = block_squared[np.triu_indices(len(ordered), k=1)]
            nonzero = upper[upper > np.finfo(float).eps]
            block_medians[field_name] = (
                float(np.median(nonzero)) if nonzero.size else None
            )
        active_fields = tuple(
            field_name
            for field_name in ordered_fields
            if block_medians[field_name] is not None
        )
        if not active_fields:
            raise ValueError(
                "All fixed-catalogue items have identical typed feature vectors; "
                "an attribute RBF baseline is not identifiable"
            )
        column_scale = np.zeros(features.shape[1], dtype=float)
        block_audit: list[dict[str, object]] = []
        for field_name in ordered_fields:
            median_squared = block_medians[field_name]
            scale = (
                1.0 / math.sqrt(len(active_fields) * median_squared)
                if median_squared is not None
                else 0.0
            )
            for index, observed in enumerate(field_names):
                if observed == field_name:
                    column_scale[index] = scale
            block_audit.append(
                {
                    "field": field_name,
                    "column_count": field_names.count(field_name),
                    "median_nonzero_squared_distance_before_scaling": median_squared,
                    "column_scale": scale,
                    "active": median_squared is not None,
                }
            )
        features = features * column_scale[None, :]
        differences = features[:, None, :] - features[None, :, :]
        squared = np.einsum("ijk,ijk->ij", differences, differences)
        squared = np.maximum(squared, 0.0)
        nonzero = np.sqrt(squared[np.triu_indices(len(ordered), k=1)])
        nonzero = nonzero[nonzero > np.finfo(float).eps]
        median = float(np.median(nonzero)) if nonzero.size else 1.0

        digest_payload = np.ascontiguousarray(features, dtype="<f8").tobytes()
        digest = hashlib.sha256(
            b"kernel-item-catalogue-v2\0"
            + "\n".join(row.entity_id for row in ordered).encode("utf-8")
            + b"\0"
            + "\n".join(encoder.feature_names).encode("utf-8")
            + b"\0"
            + digest_payload
        ).hexdigest()
        features.setflags(write=False)
        squared.setflags(write=False)
        return cls(
            item_ids=tuple(row.entity_id for row in ordered),
            feature_names=encoder.feature_names,
            feature_matrix=features,
            squared_distances=squared,
            median_nonzero_distance=median,
            feature_digest=digest,
            predicate_block_audit=tuple(block_audit),
        )

    def rbf_kernel(self, length_scale_multiplier: float) -> tuple[np.ndarray, float]:
        if not math.isfinite(length_scale_multiplier) or length_scale_multiplier <= 0:
            raise ValueError("RBF length-scale multiplier must be finite and positive")
        length_scale = self.median_nonzero_distance * float(length_scale_multiplier)
        kernel = np.exp(-self.squared_distances / (2.0 * length_scale**2))
        kernel = (kernel + kernel.T) / 2.0
        return kernel, length_scale

    def audit(self) -> dict[str, object]:
        return {
            "item_count": len(self.item_ids),
            "feature_count": len(self.feature_names),
            "feature_names": list(self.feature_names),
            "representation_schema": "typed-entity-feature-rbf-v2",
            "predicate_block_normalization": (
                "equal_median_nonzero_squared_distance_across_varying_predicates"
            ),
            "predicate_blocks": [dict(row) for row in self.predicate_block_audit],
            "median_nonzero_typed_distance": self.median_nonzero_distance,
            "feature_digest_sha256": self.feature_digest,
            "scope": "fixed_catalogue_transductive",
            "label_use": "none_catalogue_representation_is_label_free",
            "kernel": "exp(-squared_typed_distance/(2*length_scale^2))",
        }


@dataclass(frozen=True)
class KernelPairwiseFit:
    """One convex finite-catalogue RBF pairwise-logistic fit."""

    catalogue: KernelItemCatalogue = field(repr=False, compare=False)
    coefficients: tuple[float, ...]
    utilities: tuple[float, ...]
    length_scale_multiplier: float
    length_scale: float
    l2: float
    anchor_strength: float
    objective: float
    iterations: int
    converged: bool
    status: str
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def utility_map(self) -> dict[str, float]:
        return dict(zip(self.catalogue.item_ids, self.utilities, strict=True))

    def margins(self, observations: Iterable[PairwiseObservation]) -> np.ndarray:
        utilities = self.utility_map()
        rows = tuple(observations)
        unknown = sorted(
            {
                item_id
                for row in rows
                for item_id in (row.preferred_id, row.nonpreferred_id)
                if item_id not in utilities
            }
        )
        if unknown:
            raise ValueError(
                "Kernel preference evaluation contains items outside the fitted "
                f"catalogue: {unknown}"
            )
        return np.asarray(
            [utilities[row.preferred_id] - utilities[row.nonpreferred_id] for row in rows],
            dtype=float,
        )

    def probabilities(self, observations: Iterable[PairwiseObservation]) -> np.ndarray:
        """Return P(preferred beats nonpreferred) for each supplied row."""

        return _sigmoid(self.margins(observations))

    def audit(self) -> dict[str, object]:
        return {
            "model": "kernel_pairwise_rbf",
            "fit_scope": (
                "anchored_personalized_support"
                if self.anchor_strength > 0
                else "global_outer_training_users"
            ),
            "claim_boundary": (
                "finite-catalogue RBF kernel logistic preference model; not a full GP or DP-GP"
            ),
            "item_ids": list(self.catalogue.item_ids),
            "coefficients": list(self.coefficients),
            "utilities": list(self.utilities),
            "length_scale_multiplier": self.length_scale_multiplier,
            "length_scale": self.length_scale,
            "l2": self.l2,
            "anchor_strength": self.anchor_strength,
            "objective": self.objective,
            "iterations": self.iterations,
            "converged": self.converged,
            "status": self.status,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class KernelHyperparameterSelection:
    fit: KernelPairwiseFit
    selected_length_scale_multiplier: float
    selected_l2: float
    criterion: str
    candidates: tuple[Mapping[str, object], ...]
    provenance: Mapping[str, object] = field(default_factory=dict)

    def audit(self) -> dict[str, object]:
        return {
            "selected_length_scale_multiplier": self.selected_length_scale_multiplier,
            "selected_l2": self.selected_l2,
            "criterion": self.criterion,
            "candidates": [dict(row) for row in self.candidates],
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class KernelAnchorSelection:
    anchor_strength: float
    criterion: str
    support_size_per_calibration_user: int
    eligible_calibration_users: int
    candidates: tuple[Mapping[str, object], ...]
    provenance: Mapping[str, object] = field(default_factory=dict)

    def audit(self) -> dict[str, object]:
        return {
            "selected_anchor_strength": self.anchor_strength,
            "criterion": self.criterion,
            "support_size_per_calibration_user": self.support_size_per_calibration_user,
            "eligible_calibration_users": self.eligible_calibration_users,
            "candidates": [dict(row) for row in self.candidates],
            "provenance": dict(self.provenance),
        }


def _observation_design(
    observations: Sequence[PairwiseObservation],
    catalogue: KernelItemCatalogue,
    kernel: np.ndarray,
) -> np.ndarray:
    item_index = {item_id: index for index, item_id in enumerate(catalogue.item_ids)}
    unknown = sorted(
        {
            item_id
            for row in observations
            for item_id in (row.preferred_id, row.nonpreferred_id)
            if item_id not in item_index
        }
    )
    if unknown:
        raise ValueError(f"Pairwise observations reference unknown items: {unknown}")
    preferred = np.asarray([item_index[row.preferred_id] for row in observations], dtype=int)
    nonpreferred = np.asarray(
        [item_index[row.nonpreferred_id] for row in observations], dtype=int
    )
    return kernel[preferred] - kernel[nonpreferred]


def _spectral_diagnostics(matrix: np.ndarray) -> dict[str, object]:
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
    maximum = float(max(0.0, eigenvalues[-1]))
    tolerance = max(matrix.shape) * np.finfo(float).eps * max(1.0, maximum)
    positive = eigenvalues[eigenvalues > tolerance]
    condition = _finite_or_none(maximum / positive[0]) if positive.size else None
    return {
        "minimum_eigenvalue": _finite_or_none(eigenvalues[0]),
        "maximum_eigenvalue": _finite_or_none(eigenvalues[-1]),
        "effective_rank": int(positive.size),
        "condition_number_nonzero": condition,
        "eigenvalue_tolerance": float(tolerance),
    }


def fit_kernel_pairwise_rbf(
    observations: Iterable[PairwiseObservation],
    catalogue: KernelItemCatalogue,
    *,
    length_scale_multiplier: float = 1.0,
    l2: float = 1e-2,
    prior: KernelPairwiseFit | None = None,
    anchor_strength: float = 1.0,
    jitter: float = 1e-9,
    max_iterations: int = 2_000,
    tolerance: float = 1e-9,
) -> KernelPairwiseFit:
    """Fit global or anchored personalized nonlinear item utilities.

    Global fitting minimizes mean pairwise logistic loss plus the squared RKHS
    norm controlled by ``l2``.  Personalized fitting instead regularizes the
    deviation from ``prior``.  Consequently an empty personalized support set
    returns the global prior exactly rather than silently shrinking it.
    """

    rows = _ordered_observations(observations)
    if not math.isfinite(l2) or l2 < 0:
        raise ValueError("l2 must be finite and non-negative")
    if not math.isfinite(anchor_strength) or anchor_strength <= 0:
        raise ValueError("anchor_strength must be finite and positive")
    if not math.isfinite(jitter) or jitter <= 0:
        raise ValueError("jitter must be finite and positive")
    if max_iterations <= 0 or tolerance <= 0:
        raise ValueError("Optimizer limits must be positive")
    if prior is None and not rows:
        raise ValueError("A global kernel model requires at least one observation")
    if prior is not None:
        if prior.catalogue.feature_digest != catalogue.feature_digest:
            raise ValueError("Personalized prior and catalogue feature spaces differ")
        if not math.isclose(
            prior.length_scale_multiplier,
            length_scale_multiplier,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("Personalized fit must reuse the prior RBF length scale")
        if not math.isclose(prior.l2, l2, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("Personalized fit must report the global prior's selected l2")
        if len(prior.coefficients) != len(catalogue.item_ids) or not np.all(
            np.isfinite(np.asarray(prior.coefficients, dtype=float))
        ):
            raise ValueError("Personalized prior coefficients are invalid")

    kernel, length_scale = catalogue.rbf_kernel(length_scale_multiplier)
    regularizer = kernel + jitter * np.eye(len(catalogue.item_ids), dtype=float)
    design = _observation_design(rows, catalogue, kernel) if rows else np.empty(
        (0, len(catalogue.item_ids)), dtype=float
    )
    centre = (
        np.asarray(prior.coefficients, dtype=float)
        if prior is not None
        else np.zeros(len(catalogue.item_ids), dtype=float)
    )
    penalty = float(anchor_strength if prior is not None else l2)
    if penalty <= 0:
        raise ValueError("Global l2 must be positive for a uniquely stabilized kernel fit")

    if not rows:
        coefficients = centre.copy()
        utilities = kernel @ coefficients
        diagnostics = {
            **_spectral_diagnostics(kernel),
            "regularized_condition_number": _finite_or_none(
                np.linalg.cond(regularizer)
            ),
            "gradient_norm": 0.0,
            "n_observations": 0,
            "optimizer_message": "prior_only_empty_support",
            "jitter": jitter,
            "regularizer": "anchored_rkhs_deviation",
            "observation_digest_sha256": _observation_digest(rows),
            "training_user_count": 0,
            "training_pair_group_count": 0,
            "probability_link": "logistic_with_intrinsic_scale_one",
        }
        return KernelPairwiseFit(
            catalogue=catalogue,
            coefficients=tuple(float(value) for value in coefficients),
            utilities=tuple(float(value) for value in utilities),
            length_scale_multiplier=float(length_scale_multiplier),
            length_scale=length_scale,
            l2=float(l2),
            anchor_strength=float(anchor_strength),
            objective=0.0,
            iterations=0,
            converged=True,
            status="prior_only_empty_support",
            diagnostics=diagnostics,
        )

    # The experiment-scoped user/session is the independent unit.  Give every
    # training user equal total objective mass, then divide that mass equally
    # among the user's retained pair observations.  With a single user this is
    # ordinary mean pairwise logistic loss.
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row.user_id] += 1
    row_weights = np.asarray(
        [1.0 / (len(counts) * counts[row.user_id]) for row in rows], dtype=float
    )

    def objective_gradient(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        margins = design @ coefficients
        logistic = float(np.dot(row_weights, np.logaddexp(0.0, -margins)))
        wrong_probability = _sigmoid(-margins)
        difference = coefficients - centre
        penalty_value = 0.5 * penalty * float(difference @ regularizer @ difference)
        gradient = -(design.T @ (row_weights * wrong_probability))
        gradient += penalty * (regularizer @ difference)
        return logistic + penalty_value, gradient

    initial = centre.copy()
    result = minimize(
        objective_gradient,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(max_iterations),
            "ftol": float(tolerance),
            "gtol": float(tolerance),
            "maxls": 50,
        },
    )
    coefficients = np.asarray(result.x, dtype=float)
    objective, gradient = objective_gradient(coefficients)
    if not np.all(np.isfinite(coefficients)) or not math.isfinite(objective):
        raise RuntimeError("Kernel preference optimization produced non-finite output")
    utilities = kernel @ coefficients
    margins = design @ coefficients
    probabilities = _sigmoid(margins)
    curvature = probabilities * (1.0 - probabilities)
    hessian = (design.T * (row_weights * curvature)) @ design
    hessian += penalty * regularizer
    hessian_condition = _finite_or_none(np.linalg.cond(hessian))
    gradient_norm = float(np.linalg.norm(gradient))
    converged = bool(result.success) or gradient_norm <= math.sqrt(tolerance)
    status = "optimal" if converged else "optimizer_not_converged"
    diagnostics = {
        **_spectral_diagnostics(kernel),
        "regularized_condition_number": _finite_or_none(np.linalg.cond(regularizer)),
        "hessian_condition_number": hessian_condition,
        "gradient_norm": gradient_norm,
        "n_observations": len(rows),
        "n_function_evaluations": int(getattr(result, "nfev", 0)),
        "optimizer_message": str(result.message),
        "jitter": jitter,
        "regularizer": (
            "anchored_rkhs_deviation" if prior is not None else "global_rkhs_norm"
        ),
        "training_mean_probability_preferred": float(np.mean(probabilities)),
        "observation_digest_sha256": _observation_digest(rows),
        "training_user_count": len({row.user_id for row in rows}),
        "training_pair_group_count": len(
            {row.group_id or row.comparison_id for row in rows}
        ),
        "objective_normalization": (
            "equal_experiment_scoped_user_mass_then_equal_mass_per_user_observation"
        ),
        "probability_link": "logistic_with_intrinsic_scale_one",
    }
    return KernelPairwiseFit(
        catalogue=catalogue,
        coefficients=tuple(float(value) for value in coefficients),
        utilities=tuple(float(value) for value in utilities),
        length_scale_multiplier=float(length_scale_multiplier),
        length_scale=length_scale,
        l2=float(l2),
        anchor_strength=float(anchor_strength if prior is not None else 0.0),
        objective=float(objective),
        iterations=int(getattr(result, "nit", 0)),
        converged=converged,
        status=status,
        diagnostics=diagnostics,
    )


def kernel_pairwise_metrics(
    fitted: KernelPairwiseFit,
    observations: Iterable[PairwiseObservation],
) -> dict[str, float]:
    rows = tuple(observations)
    if not rows:
        raise ValueError("Kernel preference evaluation requires observations")
    margins = fitted.margins(rows)
    probabilities = _sigmoid(margins)
    return {
        "accuracy": float(np.mean(margins > 0.0)),
        "tie_rate": float(np.mean(np.isclose(margins, 0.0, atol=1e-12))),
        "mean_margin": float(np.mean(margins)),
        "log_loss": float(np.mean(np.logaddexp(0.0, -margins))),
        "brier": float(np.mean(np.square(1.0 - probabilities))),
    }


def _user_mean_kernel_metrics(
    fitted: KernelPairwiseFit,
    observations: Sequence[PairwiseObservation],
) -> dict[str, float]:
    by_user: dict[str, list[PairwiseObservation]] = defaultdict(list)
    for row in observations:
        by_user[row.user_id].append(row)
    if not by_user:
        raise ValueError("User-mean kernel evaluation requires observations")
    rows = [kernel_pairwise_metrics(fitted, by_user[user_id]) for user_id in sorted(by_user)]
    return {
        name: float(np.mean([row[name] for row in rows]))
        for name in ("accuracy", "tie_rate", "mean_margin", "log_loss", "brier")
    }


def select_kernel_pairwise_rbf(
    fit_observations: Iterable[PairwiseObservation],
    calibration_observations: Iterable[PairwiseObservation],
    catalogue: KernelItemCatalogue,
    *,
    length_scale_multipliers: Sequence[float] = (0.5, 1.0, 2.0),
    l2_candidates: Sequence[float] = (1e-3, 1e-2, 1e-1),
    jitter: float = 1e-9,
    max_iterations: int = 2_000,
    tolerance: float = 1e-9,
) -> KernelHyperparameterSelection:
    """Select global kernel hyperparameters on outer-training calibration users."""

    fit_rows = _ordered_observations(fit_observations)
    calibration_rows = _ordered_observations(calibration_observations)
    if not fit_rows or not calibration_rows:
        raise ValueError("Global kernel selection requires fit and calibration observations")
    fit_users = {row.user_id for row in fit_rows}
    calibration_users = {row.user_id for row in calibration_rows}
    overlap = sorted(fit_users & calibration_users)
    if overlap:
        raise ValueError(
            "Kernel hyperparameter selection requires user-atomic model-fit and "
            f"calibration partitions; overlapping users: {overlap}"
        )
    length_grid = tuple(sorted(set(float(value) for value in length_scale_multipliers)))
    l2_grid = tuple(sorted(set(float(value) for value in l2_candidates)))
    if not length_grid or any(not math.isfinite(value) or value <= 0 for value in length_grid):
        raise ValueError("Length-scale candidates must be finite and positive")
    if not l2_grid or any(not math.isfinite(value) or value <= 0 for value in l2_grid):
        raise ValueError("l2 candidates must be finite and positive")

    fitted_candidates: list[tuple[float, float, float, KernelPairwiseFit]] = []
    audit_rows: list[dict[str, object]] = []
    for multiplier in length_grid:
        for l2 in l2_grid:
            fitted = fit_kernel_pairwise_rbf(
                fit_rows,
                catalogue,
                length_scale_multiplier=multiplier,
                l2=l2,
                jitter=jitter,
                max_iterations=max_iterations,
                tolerance=tolerance,
            )
            metrics = _user_mean_kernel_metrics(fitted, calibration_rows)
            fitted_candidates.append((metrics["log_loss"], multiplier, l2, fitted))
            audit_rows.append(
                {
                    "length_scale_multiplier": multiplier,
                    "l2": l2,
                    "calibration_user_mean_log_loss": metrics["log_loss"],
                    "calibration_user_mean_accuracy": metrics["accuracy"],
                    "converged": fitted.converged,
                    "gradient_norm": fitted.diagnostics.get("gradient_norm"),
                }
            )
    converged_candidates = [row for row in fitted_candidates if row[3].converged]
    if not converged_candidates:
        raise RuntimeError("No global kernel hyperparameter candidate converged")
    selected = min(
        converged_candidates,
        key=lambda row: (round(row[0], 14), row[1], row[2]),
    )
    return KernelHyperparameterSelection(
        fit=selected[3],
        selected_length_scale_multiplier=selected[1],
        selected_l2=selected[2],
        criterion=(
            "minimum experiment-scoped-user-mean pairwise logistic log loss on "
            "outer-training calibration users"
        ),
        candidates=tuple(audit_rows),
        provenance={
            "selection_scope": "outer_training_users_only",
            "model_fit_user_count": len(fit_users),
            "calibration_user_count": len(calibration_users),
            "user_partitions_disjoint": True,
            "model_fit_observation_digest_sha256": _observation_digest(fit_rows),
            "calibration_observation_digest_sha256": _observation_digest(
                calibration_rows
            ),
            "catalogue_feature_digest_sha256": catalogue.feature_digest,
            "candidate_order": "length_scale_multiplier_then_l2_ascending",
        },
    )


def _calibration_adaptation_rows(
    observations: Sequence[PairwiseObservation],
    *,
    support_size_per_user: int,
    seed: int,
) -> tuple[tuple[str, tuple[PairwiseObservation, ...], tuple[PairwiseObservation, ...]], ...]:
    if support_size_per_user <= 0:
        raise ValueError("Calibration adaptation support size must be positive")
    by_user: dict[str, list[PairwiseObservation]] = defaultdict(list)
    for row in observations:
        by_user[row.user_id].append(row)
    output: list[
        tuple[str, tuple[PairwiseObservation, ...], tuple[PairwiseObservation, ...]]
    ] = []
    for user_id in sorted(by_user):
        groups: dict[str, list[PairwiseObservation]] = defaultdict(list)
        for row in by_user[user_id]:
            group_id = row.group_id or row.comparison_id
            groups[group_id].append(row)
        ordered_groups = sorted(
            groups,
            key=lambda group_id: (_stable_digest(seed, user_id, group_id), group_id),
        )
        # Keep at least one unordered pair group for calibration evaluation.
        take = min(support_size_per_user, len(ordered_groups) - 1)
        if take <= 0:
            continue
        support_ids = set(ordered_groups[:take])
        support = tuple(
            row
            for group_id in ordered_groups
            if group_id in support_ids
            for row in groups[group_id]
        )
        evaluation = tuple(
            row
            for group_id in ordered_groups
            if group_id not in support_ids
            for row in groups[group_id]
        )
        if support and evaluation:
            output.append((user_id, support, evaluation))
    return tuple(output)


def select_kernel_anchor_strength(
    global_fit: KernelPairwiseFit,
    calibration_observations: Iterable[PairwiseObservation],
    *,
    anchor_candidates: Sequence[float] = (0.1, 1.0, 10.0),
    support_size_per_user: int = 5,
    seed: int = 20260825,
    jitter: float = 1e-9,
    max_iterations: int = 2_000,
    tolerance: float = 1e-9,
) -> KernelAnchorSelection:
    """Tune support adaptation using only outer-training calibration users.

    Each calibration user is split atomically by unordered pair group.  The
    criterion first averages log loss within user and then across users, so a
    user with more repeated comparisons cannot dominate selection.
    """

    if not global_fit.converged:
        raise ValueError("Anchor selection requires a converged global kernel fit")
    candidates = tuple(sorted(set(float(value) for value in anchor_candidates)))
    if not candidates or any(not math.isfinite(value) or value <= 0 for value in candidates):
        raise ValueError("Anchor candidates must be finite and positive")
    ordered_observations = _ordered_observations(calibration_observations)
    calibration_rows = _calibration_adaptation_rows(
        ordered_observations,
        support_size_per_user=support_size_per_user,
        seed=seed,
    )
    if not calibration_rows:
        raise ValueError(
            "No calibration user has enough pair groups for anchor-strength selection"
        )
    scored: list[tuple[float, float]] = []
    audit_rows: list[dict[str, object]] = []
    for anchor in candidates:
        user_losses: list[float] = []
        user_accuracies: list[float] = []
        converged = 0
        for _, support, evaluation in calibration_rows:
            fitted = fit_kernel_pairwise_rbf(
                support,
                global_fit.catalogue,
                length_scale_multiplier=global_fit.length_scale_multiplier,
                l2=global_fit.l2,
                prior=global_fit,
                anchor_strength=anchor,
                jitter=jitter,
                max_iterations=max_iterations,
                tolerance=tolerance,
            )
            metrics = kernel_pairwise_metrics(fitted, evaluation)
            user_losses.append(metrics["log_loss"])
            user_accuracies.append(metrics["accuracy"])
            converged += int(fitted.converged)
        mean_loss = float(np.mean(user_losses))
        scored.append((mean_loss, anchor))
        audit_rows.append(
            {
                "anchor_strength": anchor,
                "calibration_user_mean_log_loss": mean_loss,
                "calibration_user_mean_accuracy": float(np.mean(user_accuracies)),
                "converged_user_fits": converged,
                "eligible_users": len(calibration_rows),
            }
        )
    selected = min(scored, key=lambda row: (round(row[0], 14), row[1]))
    return KernelAnchorSelection(
        anchor_strength=selected[1],
        criterion=(
            "minimum user-mean log loss after deterministic within-calibration-user "
            "support adaptation"
        ),
        support_size_per_calibration_user=support_size_per_user,
        eligible_calibration_users=len(calibration_rows),
        candidates=tuple(audit_rows),
        provenance={
            "selection_scope": "outer_training_calibration_users_only",
            "calibration_observation_digest_sha256": _observation_digest(
                ordered_observations
            ),
            "deterministic_support_seed": int(seed),
            "support_split_unit": "user_specific_unordered_pair_group",
            "aggregation": "mean_within_user_then_mean_across_users",
            "catalogue_feature_digest_sha256": global_fit.catalogue.feature_digest,
        },
    )
