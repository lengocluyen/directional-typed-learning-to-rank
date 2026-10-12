"""Matched symmetric/directional typed bases with an explicit reference.

For a numeric property with reference value q and fitted scale c, let
t = clip((v - q) / c, -1, 1).  The symmetric family S0 keeps g_s = |t| / sqrt(2);
the directional family D0 keeps g_s and g_a = t / sqrt(2).  The pair (g_s, g_a)
is an orthogonal rotation of the below/above pair (h-, h+) = (max(-t, 0), max(t, 0)),
so a ridge or group penalty has the same geometry in both bases.  Every
categorical, set and missingness coordinate is identical in S0 and D0, which
makes S0 versus D0 an exact test of numerical direction.

The direct family is reference-free: standardized raw numerics, full one-hot
categoricals, multi-hot sets and missing indicators.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from rgsmw_next.distance import robust_scale
from rgsmw_next.types import EntityRecord

FAMILIES = ("S0", "D0", "direct")
NUMERIC_KINDS = frozenset({"numeric", "count", "year", "ordinal"})
ROOT_HALF = 1.0 / math.sqrt(2.0)


@dataclass(frozen=True)
class PropertySpec:
    field: str
    kind: str  # numeric | count | year | ordinal | categorical | set
    ordinal_levels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in NUMERIC_KINDS | {"categorical", "set"}:
            raise ValueError(f"Unsupported property kind {self.kind!r} for {self.field}")
        if self.kind == "ordinal" and len(self.ordinal_levels) < 2:
            raise ValueError(f"Ordinal property {self.field} needs at least two levels")


@dataclass(frozen=True)
class Reference:
    """Reference values: a float for numerics, a string for categoricals, a set for sets.

    ``None`` means the reference is missing for that property.
    """

    values: Mapping[str, object]
    label: str = "reference"

    def get(self, field_name: str) -> object:
        return self.values.get(field_name)


@dataclass(frozen=True)
class Group:
    field: str
    kind: str
    start: int
    stop: int
    names: tuple[str, ...]


@dataclass(frozen=True)
class FittedProperty:
    spec: PropertySpec
    median: float | None = None
    scale: float = 1.0
    scale_source: str = "not_applicable"
    vocabulary: tuple[str, ...] = ()
    mode: str | None = None
    modal_set: frozenset[str] = frozenset()


def _numeric(spec: PropertySpec, value: object) -> float:
    if spec.kind == "ordinal":
        try:
            return float(spec.ordinal_levels.index(str(value)))
        except ValueError as error:
            raise ValueError(f"Unknown ordinal level {value!r} for {spec.field}") from error
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        raise ValueError(f"Non-finite numeric value for {spec.field}")
    return number


def record_numeric(record: EntityRecord, spec: PropertySpec) -> float | None:
    values = record.get(spec.field)
    if not values:
        return None
    if len(values) > 1:
        raise ValueError(f"Scalar property {spec.field} has several values on {record.entity_id}")
    return _numeric(spec, values[0])


def record_category(record: EntityRecord, spec: PropertySpec) -> str | None:
    values = record.get(spec.field)
    if not values:
        return None
    if len(values) > 1:
        raise ValueError(f"Categorical property {spec.field} has several values on {record.entity_id}")
    return str(values[0])


def record_set(record: EntityRecord, spec: PropertySpec) -> frozenset[str]:
    return frozenset(str(value) for value in record.get(spec.field))


@dataclass(frozen=True)
class TypedBasis:
    """Encoder fitted on a declared, label-free set of catalogue records."""

    properties: tuple[FittedProperty, ...]
    clip_multiplier: float = 2.0
    _groups: Mapping[str, tuple[Group, ...]] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        records: Iterable[EntityRecord],
        specs: Sequence[PropertySpec],
        clip_multiplier: float = 2.0,
    ) -> "TypedBasis":
        rows = tuple(records)
        if not rows:
            raise ValueError("TypedBasis needs at least one record")
        fitted: list[FittedProperty] = []
        for spec in specs:
            if spec.kind in NUMERIC_KINDS:
                values = [v for v in (record_numeric(r, spec) for r in rows) if v is not None]
                if not values:
                    raise ValueError(f"Property {spec.field} is never observed")
                if spec.kind == "ordinal":
                    scale, source = float(max(1, len(spec.ordinal_levels) - 1)), "ordinal_level_range"
                else:
                    scale, source = robust_scale(values)
                fitted.append(
                    FittedProperty(spec, median=float(np.median(values)), scale=float(scale), scale_source=source)
                )
            elif spec.kind == "categorical":
                observed = [c for c in (record_category(r, spec) for r in rows) if c is not None]
                counts = Counter(observed)
                vocabulary = tuple(sorted(counts))
                mode = min(counts, key=lambda k: (-counts[k], k)) if counts else None
                fitted.append(FittedProperty(spec, vocabulary=vocabulary, mode=mode))
            else:
                sets = [record_set(r, spec) for r in rows]
                vocabulary = tuple(sorted({v for s in sets for v in s}))
                counts = Counter(s for s in sets if s)
                modal = min(counts, key=lambda s: (-counts[s], sorted(s))) if counts else frozenset()
                fitted.append(FittedProperty(spec, vocabulary=vocabulary, modal_set=modal))
        basis = cls(tuple(fitted), clip_multiplier)
        object.__setattr__(basis, "_groups", {f: basis._build_groups(f) for f in FAMILIES})
        return basis

    # ------------------------------------------------------------------ layout
    def _component_names(self, prop: FittedProperty, family: str) -> tuple[str, ...]:
        kind = prop.spec.kind
        if family == "direct":
            if kind in NUMERIC_KINDS:
                return ("standardized", "missing")
            if kind == "categorical":
                return (*(f"is={v}" for v in prop.vocabulary), "unseen", "missing")
            return (*(f"has={v}" for v in prop.vocabulary), "empty")
        if kind in NUMERIC_KINDS:
            return ("g_sym", "g_dir", "missing_mismatch") if family == "D0" else ("g_sym", "missing_mismatch")
        if kind == "categorical":
            return (*(f"is={v}" for v in prop.vocabulary[1:]), "unseen", "missing", "matches_reference")
        return ("jaccard_distance", "reference_containment", "candidate_containment", "empty")

    def _build_groups(self, family: str) -> tuple[Group, ...]:
        groups: list[Group] = []
        start = 0
        for prop in self.properties:
            names = self._component_names(prop, family)
            groups.append(Group(prop.spec.field, prop.spec.kind, start, start + len(names), names))
            start += len(names)
        return tuple(groups)

    def groups(self, family: str) -> tuple[Group, ...]:
        if family not in FAMILIES:
            raise ValueError(f"Unknown family {family!r}")
        return self._groups[family]

    def dimension(self, family: str) -> int:
        return self.groups(family)[-1].stop

    def feature_names(self, family: str) -> tuple[str, ...]:
        return tuple(f"{g.field}:{n}" for g in self.groups(family) for n in g.names)

    # --------------------------------------------------------------- references
    def default_reference(self) -> Reference:
        """Label-free catalogue reference: numeric medians, categorical modes, modal set."""
        values: dict[str, object] = {}
        for prop in self.properties:
            if prop.spec.kind in NUMERIC_KINDS:
                values[prop.spec.field] = prop.median
            elif prop.spec.kind == "categorical":
                values[prop.spec.field] = prop.mode
            else:
                values[prop.spec.field] = prop.modal_set
        return Reference(values, "catalogue_median_mode")

    def reference_from_records(
        self, records: Sequence[EntityRecord], fallback: Reference, label: str
    ) -> Reference:
        """Mean numeric values, modal category and majority set of the given records.

        Properties never observed in ``records`` fall back to ``fallback``.
        """
        if not records:
            return fallback
        values: dict[str, object] = {}
        for prop in self.properties:
            spec = prop.spec
            if spec.kind in NUMERIC_KINDS:
                observed = [v for v in (record_numeric(r, spec) for r in records) if v is not None]
                values[spec.field] = float(np.mean(observed)) if observed else fallback.get(spec.field)
            elif spec.kind == "categorical":
                counts = Counter(c for c in (record_category(r, spec) for r in records) if c is not None)
                values[spec.field] = (
                    min(counts, key=lambda k: (-counts[k], k)) if counts else fallback.get(spec.field)
                )
            else:
                sets = [record_set(r, spec) for r in records]
                counts = Counter(v for s in sets for v in s)
                majority = frozenset(v for v, n in counts.items() if n * 2 >= len(sets))
                values[spec.field] = majority if majority else fallback.get(spec.field)
        return Reference(values, label)

    # ---------------------------------------------------------------- encoding
    def _numeric_block(self, prop: FittedProperty, candidate: float | None, ref: object, family: str) -> list[float]:
        if family == "direct":
            if candidate is None:
                return [0.0, 1.0]
            assert prop.median is not None
            return [(candidate - prop.median) / max(prop.scale, np.finfo(float).eps), 0.0]
        reference = None if ref is None else _numeric(prop.spec, ref) if isinstance(ref, str) else float(ref)  # type: ignore[arg-type]
        mismatch = float((candidate is None) != (reference is None))
        if candidate is None or reference is None:
            return [0.0, 0.0, mismatch] if family == "D0" else [0.0, mismatch]
        denominator = max(np.finfo(float).eps, self.clip_multiplier * prop.scale)
        t = float(np.clip((candidate - reference) / denominator, -1.0, 1.0))
        if family == "D0":
            return [abs(t) * ROOT_HALF, t * ROOT_HALF, 0.0]
        return [abs(t) * ROOT_HALF, 0.0]

    def _categorical_block(self, prop: FittedProperty, candidate: str | None, ref: object, family: str) -> list[float]:
        known = candidate is not None and candidate in prop.vocabulary
        if family == "direct":
            return [
                *(float(candidate == v) for v in prop.vocabulary),
                float(candidate is not None and not known),
                float(candidate is None),
            ]
        return [
            *(float(candidate == v) for v in prop.vocabulary[1:]),
            float(candidate is not None and not known),
            float(candidate is None),
            float(candidate is not None and ref is not None and candidate == ref),
        ]

    def _set_block(self, prop: FittedProperty, candidate: frozenset[str], ref: object, family: str) -> list[float]:
        if family == "direct":
            return [*(float(v in candidate) for v in prop.vocabulary), float(not candidate)]
        reference = frozenset(ref) if ref else frozenset()  # type: ignore[arg-type]
        union = candidate | reference
        intersection = candidate & reference
        return [
            1.0 - (len(intersection) / len(union) if union else 1.0),
            len(intersection) / len(reference) if reference else 0.0,
            len(intersection) / len(candidate) if candidate else 0.0,
            float(not candidate),
        ]

    def transform(self, record: EntityRecord, reference: Reference | None, family: str) -> np.ndarray:
        if family != "direct" and reference is None:
            raise ValueError("Typed families need a reference")
        output: list[float] = []
        for prop in self.properties:
            spec = prop.spec
            ref = None if reference is None else reference.get(spec.field)
            if spec.kind in NUMERIC_KINDS:
                output.extend(self._numeric_block(prop, record_numeric(record, spec), ref, family))
            elif spec.kind == "categorical":
                output.extend(self._categorical_block(prop, record_category(record, spec), ref, family))
            else:
                output.extend(self._set_block(prop, record_set(record, spec), ref, family))
        vector = np.asarray(output, dtype=float)
        if vector.shape[0] != self.dimension(family) or not np.all(np.isfinite(vector)):
            raise RuntimeError(f"Invalid {family} feature vector for {record.entity_id}")
        return vector

    def matrix(self, records: Sequence[EntityRecord], reference: Reference | None, family: str) -> np.ndarray:
        return np.vstack([self.transform(r, reference, family) for r in records])

    # ------------------------------------------------------------ explanation
    def contributions(
        self, record: EntityRecord, reference: Reference | None, family: str, weights: Sequence[float]
    ) -> dict[str, float]:
        """Exact per-property contributions; they sum to the score."""
        theta = np.asarray(weights, dtype=float)
        x = self.transform(record, reference, family)
        return {g.field: float(math.fsum(theta[g.start:g.stop] * x[g.start:g.stop])) for g in self.groups(family)}

    def numeric_slopes(self, family: str, weights: Sequence[float]) -> dict[str, dict[str, float]]:
        """Utility change per unit increase below and above the reference (D0 only).

        With coefficients (alpha on g_sym, eta on g_dir) the slopes inside the clip
        interval are (eta - alpha) / (sqrt(2) c) below and (eta + alpha) / (sqrt(2) c) above.
        """
        if family != "D0":
            raise ValueError("Directional slopes are defined for D0 only")
        theta = np.asarray(weights, dtype=float)
        output: dict[str, dict[str, float]] = {}
        for prop, group in zip(self.properties, self.groups("D0"), strict=True):
            if prop.spec.kind not in NUMERIC_KINDS:
                continue
            alpha, eta = float(theta[group.start]), float(theta[group.start + 1])
            c = self.clip_multiplier * prop.scale
            output[prop.spec.field] = {
                "below_reference_slope": (eta - alpha) * ROOT_HALF / c,
                "above_reference_slope": (eta + alpha) * ROOT_HALF / c,
                "clip_half_width": c,
                "scale": prop.scale,
            }
        return output

    def manifest(self) -> dict[str, object]:
        return {
            "clip_multiplier": self.clip_multiplier,
            "properties": [
                {
                    "field": p.spec.field,
                    "kind": p.spec.kind,
                    "median": p.median,
                    "scale": p.scale,
                    "scale_source": p.scale_source,
                    "vocabulary": list(p.vocabulary),
                    "mode": p.mode,
                    "modal_set": sorted(p.modal_set),
                    "ordinal_levels": list(p.spec.ordinal_levels),
                }
                for p in self.properties
            ],
            "dimensions": {f: self.dimension(f) for f in FAMILIES},
        }
