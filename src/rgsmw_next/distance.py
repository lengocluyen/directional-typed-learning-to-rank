from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Sequence

import numpy as np

from .types import EntityRecord, PredicateSpec, Scalar, require_dimension


def _numeric_value(value: Scalar) -> float:
    if isinstance(value, date):
        return float(value.toordinal())
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"Expected a numeric or temporal value, received {type(value).__name__}")


def robust_scale(values: Sequence[float]) -> tuple[float, str]:
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        return 1.0, "constant_or_insufficient"
    q25, q75 = np.quantile(array, [0.25, 0.75])
    iqr = float(q75 - q25)
    if iqr > 0:
        return iqr, "iqr"
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median))) * 1.4826
    if mad > 0:
        return mad, "mad"
    observed_range = float(np.max(array) - np.min(array))
    if observed_range > 0:
        return observed_range, "range"
    return 1.0, "constant_or_insufficient"


def jaccard_distance(left: Sequence[Scalar], right: Sequence[Scalar]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def haversine_km(left: tuple[float, float], right: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, left)
    lat2, lon2 = map(math.radians, right)
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    h = math.sin(d_lat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2.0) ** 2
    return 2.0 * 6371.0088 * math.asin(min(1.0, math.sqrt(max(0.0, h))))


@dataclass(frozen=True)
class FittedPredicateMetric:
    spec: PredicateSpec
    scale: float = 1.0
    scale_source: str = "not_applicable"
    clip_multiplier: float = 2.0

    def distance(self, left: Sequence[Scalar], right: Sequence[Scalar]) -> float:
        if not left and not right:
            return 0.0
        if not left or not right:
            return 1.0
        kind = self.spec.kind
        if kind == "set":
            return jaccard_distance(left, right)
        if kind in {"numeric", "count", "year", "date"}:
            denominator = max(np.finfo(float).eps, self.clip_multiplier * self.scale)
            return min(1.0, abs(_numeric_value(left[0]) - _numeric_value(right[0])) / denominator)
        if kind == "ordinal":
            left_level = self.spec.ordinal.get(str(left[0]))
            right_level = self.spec.ordinal.get(str(right[0]))
            if left_level is None or right_level is None:
                unknown = [
                    str(value)
                    for value, level in ((left[0], left_level), (right[0], right_level))
                    if level is None
                ]
                raise ValueError(
                    f"Unknown value(s) for ordinal field {self.spec.field}: {unknown}"
                )
            levels = list(self.spec.ordinal.values())
            denominator = max(1, max(levels) - min(levels))
            return abs(left_level - right_level) / denominator
        if kind == "geographic":
            if not isinstance(left[0], tuple) or not isinstance(right[0], tuple):
                raise TypeError(
                    f"Geographic field {self.spec.field} requires (latitude, longitude) tuples"
                )
            denominator = max(np.finfo(float).eps, self.clip_multiplier * self.scale)
            return min(1.0, haversine_km(left[0], right[0]) / denominator)
        return 0.0 if left[0] == right[0] else 1.0


@dataclass(frozen=True)
class FittedMetric:
    predicates: tuple[FittedPredicateMetric, ...]

    @classmethod
    def fit(
        cls,
        records: Iterable[EntityRecord],
        specs: Iterable[PredicateSpec],
        clip_multiplier: float = 2.0,
    ) -> "FittedMetric":
        record_list = list(records)
        predicates: list[FittedPredicateMetric] = []
        for spec in specs:
            if not spec.metric:
                continue
            if spec.kind in {"numeric", "count", "year", "date"}:
                values = [
                    _numeric_value(record.get(spec.field)[0])
                    for record in record_list
                    if record.get(spec.field)
                ]
                scale, scale_source = robust_scale(values)
            elif spec.kind == "geographic":
                coordinates = [
                    record.get(spec.field)[0]
                    for record in record_list
                    if record.get(spec.field) and isinstance(record.get(spec.field)[0], tuple)
                ]
                pair_distances = [
                    haversine_km(coordinates[index - 1], coordinates[index])
                    for index in range(1, len(coordinates))
                ]
                scale, scale_source = robust_scale(pair_distances)
            else:
                scale, scale_source = 1.0, "not_applicable"
            predicates.append(
                FittedPredicateMetric(
                    spec=spec,
                    scale=scale,
                    scale_source=scale_source,
                    clip_multiplier=clip_multiplier,
                )
            )
        if not predicates:
            raise ValueError("No metric-enabled predicates were supplied")
        return cls(tuple(predicates))

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(item.spec.field for item in self.predicates)

    def feature_vector(self, left: EntityRecord, right: EntityRecord) -> np.ndarray:
        return np.asarray(
            [item.distance(left.get(item.spec.field), right.get(item.spec.field)) for item in self.predicates],
            dtype=float,
        )

    def distance(
        self,
        left: EntityRecord,
        right: EntityRecord,
        weights: Sequence[float],
    ) -> tuple[float, np.ndarray]:
        require_dimension(weights, len(self.predicates))
        weight_array = np.asarray(weights, dtype=float)
        if np.min(weight_array) < -1e-10:
            raise ValueError("Metric weights must be non-negative")
        if not np.isclose(np.sum(weight_array), 1.0, atol=1e-8):
            raise ValueError("Metric weights must sum to one")
        vector = self.feature_vector(left, right)
        contributions = weight_array * vector
        return float(np.sum(contributions)), contributions

    def uniform_weights(self) -> np.ndarray:
        return np.full(len(self.predicates), 1.0 / len(self.predicates), dtype=float)

    def scale_manifest(self) -> list[dict[str, object]]:
        return [
            {
                "field": item.spec.field,
                "kind": item.spec.kind,
                "scale": item.scale,
                "scale_source": item.scale_source,
                "clip_multiplier": item.clip_multiplier,
            }
            for item in self.predicates
        ]
