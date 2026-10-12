"""Train-fitted, declared-property encoders for the matched experiment.

References may come from a test participant's *support* choices, but neither
those choices nor test candidates are used to estimate scales, vocabularies or
spline knots. Missing reference completion changes only the feature mapping;
the original missingness remains explicitly encoded. An absent set and an
observed empty set are distinguished through ``EntityRecord.values``.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from dtltr_vehicle.basis import (
    FittedProperty, NUMERIC_KINDS, PropertySpec, ROOT_HALF, Reference,
    _numeric, record_category, record_numeric, record_set,
)
from rgsmw_next.distance import robust_scale
from rgsmw_next.types import EntityRecord

MODELS = ("TD", "ST", "DT", "direct", "direct_ref", "spline", "spline_ref", "catboost_ref")


def _ref_numeric(spec: PropertySpec, value: object) -> float | None:
    if value is None:
        return None
    number = _numeric(spec, value) if isinstance(value, str) else float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite reference for {spec.field}")
    return number


@dataclass(frozen=True)
class Encoder:
    properties: tuple[FittedProperty, ...]
    model: str
    completion: str
    clipping: str
    spline_knots: Mapping[str, tuple[float, ...]]
    train_ids: tuple[str, ...]
    knots: int = 5
    clip_multiplier: float = 2.0
    knot_reference_source: str = "training_default"

    @classmethod
    def fit(
        cls, records: Iterable[EntityRecord], specs: Sequence[PropertySpec], model: str,
        completion: str = "default", clipping: str = "clipped", knots: int = 5,
        *, references: Sequence[Reference] | Reference | None = None,
    ) -> "Encoder":
        rows = tuple(records)
        if not rows or not specs:
            raise ValueError("Encoder.fit needs training records and declared properties")
        if model not in MODELS:
            raise ValueError(f"Unknown model {model!r}; expected {MODELS}")
        if completion not in {"raw", "default"} or clipping not in {"clipped", "unclipped"}:
            raise ValueError("completion must be raw/default and clipping clipped/unclipped")
        if not isinstance(knots, int) or knots < 0:
            raise ValueError("knots is a non-negative number of interior quantile knots")
        if len({p.field for p in specs}) != len(specs):
            raise ValueError("Registry property names must be unique")
        fitted = []
        for spec in specs:
            if spec.kind in NUMERIC_KINDS:
                values = [v for v in (record_numeric(r, spec) for r in rows) if v is not None]
                if spec.kind == "ordinal":
                    scale, source = float(len(spec.ordinal_levels) - 1), "ordinal_level_range"
                elif values:
                    scale, source = robust_scale(values)
                else:
                    scale, source = 1.0, "entirely_missing_training_property"
                fitted.append(FittedProperty(spec, median=float(np.median(values)) if values else 0.0,
                                             scale=max(float(scale), np.finfo(float).eps), scale_source=source))
            elif spec.kind == "categorical":
                counts = Counter(c for c in (record_category(r, spec) for r in rows) if c is not None)
                fitted.append(FittedProperty(spec, vocabulary=tuple(sorted(counts)),
                                             mode=min(counts, key=lambda k: (-counts[k], k)) if counts else None))
            else:
                sets = [record_set(r, spec) for r in rows]
                counts = Counter(s for s in sets if s)
                modal = min(counts, key=lambda s: (-counts[s], sorted(s))) if counts else frozenset()
                fitted.append(FittedProperty(spec, vocabulary=tuple(sorted({v for s in sets for v in s})), modal_set=modal))
        provisional = cls(tuple(fitted), model, completion, clipping, {}, tuple(sorted({r.entity_id for r in rows})), knots)
        refs = provisional._reference_sequence(rows, references)
        knot_map = {}
        for prop in fitted:
            if prop.spec.kind not in NUMERIC_KINDS:
                continue
            coordinates = []
            for record, ref in zip(rows, refs, strict=True):
                candidate = record_numeric(record, prop.spec)
                q, _ = provisional._reference(prop, ref)
                if candidate is None:
                    continue
                if model == "spline_ref":
                    qn = _ref_numeric(prop.spec, q)
                    if qn is None:
                        continue
                    coordinates.append(provisional._clip((candidate - qn) / (2 * prop.scale)))
                else:
                    coordinates.append(provisional._clip((candidate - float(prop.median)) / (2 * prop.scale)))
            quantiles = np.arange(1, knots + 1) / (knots + 1)
            knot_map[prop.spec.field] = tuple(float(k) for k in np.unique(np.quantile(coordinates, quantiles))) if coordinates and knots else ()
        return cls(tuple(fitted), model, completion, clipping, knot_map, provisional.train_ids,
                   knots, knot_reference_source="aligned_training_references" if references is not None else "training_default")

    @property
    def fields(self) -> list[str]:
        return [p.spec.field for p in self.properties]

    def _component_names(self, prop: FittedProperty) -> tuple[str, ...]:
        kind, vocabulary, model = prop.spec.kind, prop.vocabulary, self.model
        if model == "TD":
            return ("negative_distance",)
        if kind in NUMERIC_KINDS:
            missing = ("candidate_missing", "reference_missing", "missing_mismatch")
            if model == "ST":
                return ("g_sym", *missing)
            if model == "DT":
                return ("g_sym", "g_dir", *missing)
            if model == "direct":
                return ("candidate_standardized", "candidate_missing")
            if model == "spline":
                return ("candidate_standardized", *(f"candidate_hinge_{i}" for i in range(len(self.spline_knots.get(prop.spec.field, ())))), "candidate_missing")
            base = ("candidate_standardized", "reference_standardized", "g_sym", "g_dir", *missing)
            if model == "spline_ref":
                return (*base, *(f"deviation_hinge_{i}" for i in range(len(self.spline_knots.get(prop.spec.field, ())))))
            return base
        if kind == "categorical":
            candidate = (*(f"candidate={v}" for v in vocabulary), "candidate_unseen", "candidate_missing")
            if model in {"direct", "spline"}:
                return candidate
            if model in {"ST", "DT"}:
                return (*candidate, "matches_reference", "reference_missing")
            return (*candidate, *(f"reference={v}" for v in vocabulary), "reference_unseen", "reference_missing", "matches_reference")
        candidate = (*(f"candidate_has={v}" for v in vocabulary), "candidate_unseen_count", "candidate_empty", "candidate_missing")
        typed = ("jaccard_distance", "reference_containment", "candidate_containment", "candidate_empty", "candidate_missing", "reference_missing")
        if model in {"ST", "DT"}:
            return typed
        if model in {"direct", "spline"}:
            return candidate
        return (*candidate, *(f"reference_has={v}" for v in vocabulary), "reference_unseen_count", "reference_empty", "reference_missing", "jaccard_distance", "reference_containment", "candidate_containment")

    @property
    def names(self) -> list[str]:
        return [f"{p.spec.field}:{n}" for p in self.properties for n in self._component_names(p)]

    @property
    def groups(self) -> list[tuple[int, int]]:
        groups, start = [], 0
        for prop in self.properties:
            stop = start + len(self._component_names(prop))
            groups.append((start, stop))
            start = stop
        return groups

    @property
    def dimension(self) -> int:
        return len(self.names)

    def default_reference(self) -> Reference:
        return Reference({p.spec.field: p.median if p.spec.kind in NUMERIC_KINDS else p.mode if p.spec.kind == "categorical" else p.modal_set
                          for p in self.properties}, "training_median_mode")

    def reference_from_records(self, records: Sequence[EntityRecord], fallback: Reference | None = None, label: str = "support") -> Reference:
        """Compute a reference from explicitly supplied support records only."""
        fallback = self.default_reference() if fallback is None else fallback
        values = {}
        for prop in self.properties:
            spec = prop.spec
            if spec.kind in NUMERIC_KINDS:
                observed = [v for v in (record_numeric(r, spec) for r in records) if v is not None]
                values[spec.field] = float(np.mean(observed)) if observed else fallback.get(spec.field)
            elif spec.kind == "categorical":
                counts = Counter(v for v in (record_category(r, spec) for r in records) if v is not None)
                values[spec.field] = min(counts, key=lambda k: (-counts[k], k)) if counts else fallback.get(spec.field)
            else:
                counts = Counter(v for r in records for v in record_set(r, spec))
                values[spec.field] = frozenset(v for v, n in counts.items() if 2 * n >= len(records)) if counts else fallback.get(spec.field)
        return Reference(values, label)

    def _reference_sequence(self, records: Sequence[EntityRecord], references) -> list[Reference]:
        if references is None:
            return [self.default_reference()] * len(records)
        if isinstance(references, Reference):
            return [references] * len(records)
        refs = list(references)
        if len(refs) != len(records) or not all(isinstance(r, Reference) for r in refs):
            raise ValueError("References must be a Reference or align one-to-one with records")
        return refs

    def _reference(self, prop: FittedProperty, reference: Reference | None) -> tuple[object, bool]:
        value = None if reference is None else reference.get(prop.spec.field)
        missing = value is None
        if missing and self.completion == "default":
            value = self.default_reference().get(prop.spec.field)
        return value, missing

    def _clip(self, value: float) -> float:
        return float(np.clip(value, -1.0, 1.0)) if self.clipping == "clipped" else float(value)

    def transform(self, record: EntityRecord, reference: Reference | None = None) -> np.ndarray:
        if reference is None and self.model not in {"direct", "spline"}:
            reference = self.default_reference()
        output = []
        for prop in self.properties:
            spec, model = prop.spec, self.model
            q, qmissing = self._reference(prop, reference)
            if spec.kind in NUMERIC_KINDS:
                value, qnumber = record_numeric(record, spec), _ref_numeric(spec, q)
                missing = value is None
                z = 0.0 if missing else self._clip((value - float(prop.median)) / (2 * prop.scale))
                qz = 0.0 if qnumber is None else self._clip((qnumber - float(prop.median)) / (2 * prop.scale))
                t = 0.0 if missing or qnumber is None else self._clip((value - qnumber) / (2 * prop.scale))
                # Completion determines the distance-domain reference. Retain
                # original reference missingness separately as its own flag.
                mismatch = float(missing != (qnumber is None))
                if model == "TD":
                    output.append(-mismatch if missing or qnumber is None else -abs(t))
                elif model in {"ST", "DT"}:
                    output.extend([abs(t) * ROOT_HALF] + ([t * ROOT_HALF] if model == "DT" else []) + [float(missing), float(qmissing), mismatch])
                elif model == "direct":
                    output.extend([z, float(missing)])
                elif model == "spline":
                    output.extend([z, *(0.0 if missing else max(z-k, 0.0) for k in self.spline_knots.get(spec.field, ())), float(missing)])
                else:
                    output.extend([z, qz, abs(t) * ROOT_HALF, t * ROOT_HALF, float(missing), float(qmissing), mismatch])
                    if model == "spline_ref":
                        output.extend(0.0 if missing or qnumber is None else max(t-k, 0.0) for k in self.spline_knots.get(spec.field, ()))
            elif spec.kind == "categorical":
                value = record_category(record, spec)
                known = value in prop.vocabulary
                match = float(value is not None and q is not None and value == str(q))
                candidate = [*(float(value == v) for v in prop.vocabulary), float(value is not None and not known), float(value is None)]
                if model == "TD":
                    output.append(-float((value is None) != (q is None) or (value is not None and q is not None and value != str(q))))
                elif model in {"direct", "spline"}:
                    output.extend(candidate)
                elif model in {"ST", "DT"}:
                    output.extend([*candidate, match, float(qmissing)])
                else:
                    output.extend([*candidate, *(float(q is not None and str(q) == v) for v in prop.vocabulary),
                                   float(q is not None and str(q) not in prop.vocabulary), float(qmissing), match])
            else:
                value = record_set(record, spec)
                qset = frozenset(str(v) for v in q) if q is not None else frozenset()
                intersection, union = value & qset, value | qset
                distance = 1 - len(intersection) / len(union) if union else 0.0
                missing = spec.field not in record.values or spec.field in record.missing
                contains_q = len(intersection) / len(qset) if qset else 0.0
                contains_v = len(intersection) / len(value) if value else 0.0
                candidate = [*(float(v in value) for v in prop.vocabulary), float(len(value - set(prop.vocabulary))), float(not value), float(missing)]
                if model == "TD":
                    output.append(-float(missing != (q is None)) if missing or q is None else -distance)
                elif model in {"direct", "spline"}:
                    output.extend(candidate)
                elif model in {"ST", "DT"}:
                    output.extend([distance, contains_q, contains_v, float(not value), float(missing), float(qmissing)])
                else:
                    output.extend([*candidate, *(float(v in qset) for v in prop.vocabulary), float(len(qset - set(prop.vocabulary))),
                                   float(not qset), float(qmissing), distance, contains_q, contains_v])
        vector = np.asarray(output, dtype=float)
        if vector.shape != (self.dimension,) or not np.all(np.isfinite(vector)):
            raise RuntimeError(f"Invalid {self.model} features for {record.entity_id}")
        return vector

    def matrix(self, records: Sequence[EntityRecord], references: Sequence[Reference] | Reference | None = None) -> np.ndarray:
        rows = list(records)
        refs = self._reference_sequence(rows, references)
        return np.vstack([self.transform(r, q) for r, q in zip(rows, refs, strict=True)]) if rows else np.empty((0, self.dimension))

    def contributions(self, feature_difference: np.ndarray, theta: Sequence[float]) -> dict[str, float]:
        x, beta = np.asarray(feature_difference, dtype=float), np.asarray(theta, dtype=float)
        if x.shape != (self.dimension,) or beta.shape != x.shape:
            raise ValueError("Contributions require a single feature vector and matching coefficients")
        return {field: float(math.fsum(x[a:b] * beta[a:b])) for field, (a, b) in zip(self.fields, self.groups, strict=True)}

    def diagnostics(self, record: EntityRecord, reference: Reference | None = None) -> dict[str, object]:
        reference = self.default_reference() if reference is None else reference
        missing, ref_missing, outside, unseen, missing_sets = [], [], [], [], []
        for prop in self.properties:
            spec = prop.spec
            q, qm = self._reference(prop, reference)
            if qm:
                ref_missing.append(spec.field)
            if spec.kind in NUMERIC_KINDS:
                value, qn = record_numeric(record, spec), _ref_numeric(spec, q)
                if value is None:
                    missing.append(spec.field)
                if value is not None and qn is not None and abs((value-qn)/(2*prop.scale)) > 1:
                    outside.append(spec.field)
            elif spec.kind == "categorical":
                value = record_category(record, spec)
                if value is None:
                    missing.append(spec.field)
                elif value not in prop.vocabulary:
                    unseen.append(spec.field)
            else:
                if spec.field not in record.values or spec.field in record.missing:
                    missing.append(spec.field)
                    missing_sets.append(spec.field)
                if record_set(record, spec) - set(prop.vocabulary):
                    unseen.append(spec.field)
        return {"candidate_missing_fields": missing, "reference_missing_fields": ref_missing,
                "outside_clip_fields": outside, "unseen_fields": unseen, "missing_set_fields": missing_sets,
                "candidate_missing_count": len(missing), "reference_missing_count": len(ref_missing),
                "outside_clip_count": len(outside), "unseen_count": len(unseen)}

    def metadata(self) -> dict[str, object]:
        return {"version": 3, "model": self.model, "completion": self.completion, "clipping": self.clipping,
                "clip_multiplier": self.clip_multiplier, "dimension": self.dimension, "names": self.names,
                "fields": self.fields, "groups": [list(g) for g in self.groups], "knots": self.knots,
                "spline_basis": "continuous piecewise-linear quantile hinges", "knot_reference_source": self.knot_reference_source,
                "train_ids": list(self.train_ids), "train_id_sha256": hashlib.sha256("\n".join(self.train_ids).encode()).hexdigest(),
                "missing_policy": "candidate missing remains missing; original reference missingness retained after completion; absent sets distinguished from observed empty sets",
                "information_policy": "all reference-aware models use the same supplied candidate/reference attributes; maps have different capacities; catboost_ref equals direct_ref encoding",
                "properties": [{"field": p.spec.field, "kind": p.spec.kind, "ordinal_levels": list(p.spec.ordinal_levels),
                                "median": p.median, "scale": p.scale, "scale_source": p.scale_source,
                                "vocabulary": list(p.vocabulary), "mode": p.mode, "modal_set": sorted(p.modal_set),
                                "spline_knots": list(self.spline_knots.get(p.spec.field, ()))} for p in self.properties]}

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> "Encoder":
        """Restore a frozen encoder without re-estimating any training statistics."""
        properties, knots = [], {}
        for row in metadata["properties"]:
            spec = PropertySpec(row["field"], row["kind"], tuple(row.get("ordinal_levels", ())))
            properties.append(FittedProperty(spec, row["median"], float(row["scale"]), row["scale_source"], tuple(row["vocabulary"]), row["mode"], frozenset(row["modal_set"])))
            knots[spec.field] = tuple(float(k) for k in row.get("spline_knots", ()))
        return cls(tuple(properties), str(metadata["model"]), str(metadata["completion"]), str(metadata["clipping"]),
                   knots, tuple(metadata["train_ids"]), int(metadata["knots"]), float(metadata["clip_multiplier"]), str(metadata["knot_reference_source"]))
