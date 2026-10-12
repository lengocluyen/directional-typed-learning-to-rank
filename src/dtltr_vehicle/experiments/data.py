"""Declared-registry audits and an explicit interface for future observed data.

No observations are acquired or generated here.  A declared independent bundle
must provide provenance and preassigned train/development/calibration/test
splits.  Validation checks those declarations for internal consistency; it
cannot establish that a researcher has never previously examined the data.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

from rgsmw_next.types import EntityRecord
from ..basis import NUMERIC_KINDS, PropertySpec, Reference, record_category, record_numeric, record_set
from ..open_catalogue import OV, PROPERTIES, SEGMENTS

SPLITS = ("train", "development", "calibration", "test")


def read_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected an array of objects or JSONL objects: {path}")
    return rows


def record_from_row(row: dict, specs: list[PropertySpec] | None = None) -> EntityRecord:
    entity_id = str(row["entity_id"])
    if not entity_id:
        raise ValueError("An entity_id cannot be empty")
    values = {str(field): tuple(value) if isinstance(value, (list, tuple)) else (value,)
              for field, value in row.get("values", {}).items() if value is not None}
    record = EntityRecord(entity_id, str(row.get("uri", f"urn:{entity_id}")), values,
                          missing=dict(row.get("missing", {})))
    if specs is not None:
        known = {spec.field for spec in specs}
        if set(values) - known or set(record.missing) - known:
            raise ValueError(f"Undeclared property on {entity_id}")
        for spec in specs:
            if spec.kind in NUMERIC_KINDS:
                record_numeric(record, spec)
            elif spec.kind == "categorical":
                record_category(record, spec)
            elif any(not isinstance(value, str) for value in record.get(spec.field)):
                raise ValueError(f"Set values must be strings: {entity_id}/{spec.field}")
        for field, reason in record.missing.items():
            if record.get(field) or reason not in {"not_applicable", "not_reported", "unknown"}:
                raise ValueError(f"Inconsistent missing annotation: {entity_id}/{field}")
    return record


def raw_reference(record: EntityRecord, specs: list[PropertySpec]) -> Reference:
    """Reference from an observed item, retaining missing values without fallback."""
    values = {}
    for spec in specs:
        if spec.kind in NUMERIC_KINDS:
            values[spec.field] = record_numeric(record, spec)
        elif spec.kind == "categorical":
            values[spec.field] = record_category(record, spec)
        else:
            values[spec.field] = record_set(record, spec) if spec.field in record.values else None
    return Reference(values, f"raw:{record.entity_id}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_registry(settings: dict) -> dict:
    """Compare the analytical catalogue with explicitly declared RDF predicates.

    ADEME serializes numerics with six significant digits.  Numerical agreement
    therefore uses that declared serialization, and reports rounding separately
    instead of silently claiming a lossless decimal mapping.
    """
    paths = settings.get("paths", settings)
    records_path = Path(paths["records"])
    rdf_path = Path(paths.get("rdf", records_path.with_name("vehicles.ttl")))
    rows = read_rows(records_path)
    graph = Graph().parse(rdf_path, format="turtle")
    failures: list[dict] = []
    entity_counts = Counter(row["entity_id"] for row in rows)
    for entity_id, count in entity_counts.items():
        if count > 1:
            failures.append({"entity_id": entity_id, "issue": "duplicate_analytical_entity", "count": count})
    missing_reasons: Counter = Counter()
    rounding_values = 0
    rounding_max_abs = 0.0
    checked = 0
    subjects = set()
    annotation_count = 0
    for row in rows:
        entity_id = row["entity_id"]
        subject = URIRef(row.get("uri", str(OV[f"vehicle/{entity_id.split(':')[-1]}"])))
        subjects.add(subject)
        undeclared = (set(row.get("values", {})) | set(row.get("missing", {}))) - set(PROPERTIES)
        if undeclared:
            failures.append({"entity_id": entity_id, "issue": "undeclared_analytical_fields", "fields": sorted(undeclared)})
        if (subject, RDF.type, OV.Vehicle) not in graph:
            failures.append({"entity_id": entity_id, "issue": "missing_vehicle_type"})
        for field, (local, kind) in PROPERTIES.items():
            checked += 1
            actual = set(graph.objects(subject, OV[local]))
            value = row.get("values", {}).get(field)
            if value is None:
                expected = set()
            elif kind == "set":
                expected = {OV[f"energy/{v}"] for v in value}
            elif kind in {"numeric", "count"}:
                rounded = float(f"{float(value):.6g}")
                error = abs(float(value) - rounded)
                rounding_values += int(error > 0)
                rounding_max_abs = max(rounding_max_abs, error)
                expected = {Literal(f"{float(value):.6g}", datatype=XSD.decimal)}
            else:
                expected = {Literal(str(value))}
            equal = actual == expected
            if value is not None and kind in {"numeric", "count"}:
                # Turtle parsing can normalize decimal lexical forms (10 ->
                # 10.0). Audit decimal values and datatype, not spelling.
                try:
                    equal = (len(actual) == 1 and all(isinstance(v, Literal) and v.datatype == XSD.decimal
                             and Decimal(str(v)) == Decimal(f"{float(value):.6g}") for v in actual))
                except (InvalidOperation, ValueError):
                    equal = False
            if not equal:
                failures.append({"entity_id": entity_id, "field": field,
                                 "issue": "rdf_record_value_mismatch",
                                 "expected": sorted(map(str, expected)), "actual": sorted(map(str, actual))})
            if kind == "set" and any(not isinstance(v, URIRef) for v in actual):
                failures.append({"entity_id": entity_id, "field": field, "issue": "set_not_iri_valued"})
            if kind != "set" and any(not isinstance(v, Literal) for v in actual):
                failures.append({"entity_id": entity_id, "field": field, "issue": "scalar_not_literal"})
        actual_annotations = []
        for node in graph.objects(subject, OV.missingValue):
            props, reasons = list(graph.objects(node, OV.onProperty)), list(graph.objects(node, OV.reason))
            if len(props) != 1 or len(reasons) != 1:
                failures.append({"entity_id": entity_id, "issue": "invalid_missing_annotation"})
            else:
                actual_annotations.append((str(props[0]), str(reasons[0])))
        expected_annotations = [(str(OV[PROPERTIES[field][0]]), reason)
                                for field, reason in row.get("missing", {}).items() if field in PROPERTIES]
        if Counter(actual_annotations) != Counter(expected_annotations):
            failures.append({"entity_id": entity_id, "issue": "missing_annotation_mismatch"})
        for field, reason in row.get("missing", {}).items():
            annotation_count += 1
            missing_reasons[reason] += 1
            if row.get("values", {}).get(field) is not None:
                failures.append({"entity_id": entity_id, "field": field, "issue": "observed_value_has_missing_reason"})
    extra_subjects = set(graph.subjects(RDF.type, OV.Vehicle)) - subjects
    if extra_subjects:
        failures.append({"issue": "rdf_entities_without_analytical_record", "count": len(extra_subjects)})
    return {"dataset": "ademe_registry", "status": "pass" if not failures else "failed",
            "records": len(rows), "declared_properties": len(PROPERTIES), "checked_entity_properties": checked,
            "failures": failures, "failure_count": len(failures),
            "rdf_sha256": _sha256(rdf_path), "records_sha256": _sha256(records_path),
            "numeric_serialization": {"significant_digits": 6, "rounded_values": rounding_values,
                                      "maximum_absolute_rounding_error": rounding_max_abs},
            "missing_annotations": annotation_count, "missing_reasons": dict(missing_reasons),
            "encoder_fidelity": {"keeps_missingness_presence": True, "keeps_missing_reason_identity": False,
                                 "reason_annotations_collapsed": annotation_count,
                                 "meaning": "not_applicable and not_reported remain in source records/RDF but are not separate fitted coordinates"},
            "scope": "Declared literal/IRI registry mapping only; no schema inference or graph-neighbourhood reasoning."}


def load_external_bundle(path: str | Path) -> dict[str, Any]:
    """Validate a sealed observed-data bundle; reject leakage in declared cold splits.

    manifest.json supplies metadata and optional files names; schema.json is an
    array of {field, kind, ordinal_levels}. records.json and observations.json
    can be JSON arrays or JSONL. Labels encode whether candidate a was chosen.
    """
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    metadata = manifest.get("metadata", manifest)
    required = {"dataset_id", "source", "licence", "label_origin", "provenance", "split_mode", "reference_source"}
    if required - metadata.keys():
        raise ValueError(f"Missing external metadata: {sorted(required - metadata.keys())}")
    for key in ("dataset_id", "source", "licence", "reference_source"):
        if not isinstance(metadata[key], str) or not metadata[key].strip() or "REPLACE" in metadata[key]:
            raise ValueError(f"External metadata.{key} needs a nonempty actual value")
    if metadata["label_origin"] not in {"human", "observed", "human_observed", "archival_human"}:
        raise ValueError("External validation requires observed labels, not synthetic/generated labels")
    provenance = metadata["provenance"]
    if not isinstance(provenance, dict) or not {"independent", "exposure", "previously_examined"} <= provenance.keys():
        raise ValueError("Provenance must declare independent, exposure and previously_examined")
    if not isinstance(provenance["independent"], bool) or not isinstance(provenance["previously_examined"], bool):
        raise ValueError("Independence/exposure declarations must be booleans")
    if provenance["exposure"] not in {"unexamined", "development", "previously_examined"}:
        raise ValueError("Unknown exposure status")
    if provenance["independent"] and (provenance["previously_examined"] or provenance["exposure"] != "unexamined"):
        raise ValueError("Independent validation cannot be declared for previously examined/development data")
    if metadata["split_mode"] not in {"warm", "cold"}:
        raise ValueError("split_mode must be warm or cold")
    files = manifest.get("files", {})
    def named(name: str, default: str) -> Path:
        entry = files.get(name, default)
        filename = entry.get("path", default) if isinstance(entry, dict) else entry
        target = (root / filename).resolve()
        if not target.is_relative_to(root.resolve()):
            raise ValueError("External bundle file paths must remain inside the bundle")
        if isinstance(entry, dict) and "sha256" in entry and _sha256(target) != entry["sha256"]:
            raise ValueError(f"External bundle hash mismatch: {name}")
        return target
    schema_path, records_path, observations_path = named("schema", "schema.json"), named("records", "records.json"), named("observations", "observations.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if isinstance(schema, dict):
        schema = schema.get("properties", [])
    specs = [PropertySpec(str(s["field"]), str(s["kind"]), tuple(s.get("ordinal_levels", ()))) for s in schema]
    if not specs or len({s.field for s in specs}) != len(specs):
        raise ValueError("External schema must declare nonempty unique property fields")
    records = [record_from_row(row, specs) for row in read_rows(records_path)]
    indexed = {record.entity_id: record for record in records}
    if not records or len(indexed) != len(records):
        raise ValueError("External entity IDs must be unique and nonempty")
    observations = read_rows(observations_path)
    splits: dict[str, list[dict]] = {name: [] for name in SPLITS}
    entity_splits: dict[str, set] = {name: set() for name in SPLITS}
    user_splits: dict[str, set] = {name: set() for name in SPLITS}
    actual_user_splits: dict[str, set] = {name: set() for name in SPLITS}
    ids = set()
    for index, source in enumerate(observations):
        row = dict(source)
        needed = {"a", "b", "label", "split", "user_id"}
        if needed - row.keys():
            raise ValueError(f"Observation {index} lacks fields {sorted(needed-row.keys())}")
        split = row["split"]
        if split not in SPLITS:
            raise ValueError(f"Unknown observation split: {split}")
        if not row["user_id"] or row["a"] == row["b"] or row["label"] not in (0, 1):
            raise ValueError("Observation users must be named, alternatives distinct, labels binary")
        ref_id = row.get("reference_id", row.get("query_id"))
        if ref_id is None and "reference_values" not in row:
            raise ValueError("Observed pairs need a reference_id/query_id or directly recorded reference_values")
        involved = {str(row["a"]), str(row["b"])} | ({str(ref_id)} if ref_id is not None else set())
        if involved - indexed.keys():
            raise ValueError(f"Unknown observation entity IDs: {sorted(involved-indexed.keys())}")
        row["id"] = str(row.get("id", f"observation:{index}"))
        if row["id"] in ids:
            raise ValueError(f"Duplicate observation ID: {row['id']}")
        ids.add(row["id"])
        row["reference_id"] = None if ref_id is None else str(ref_id)
        row["a"], row["b"] = str(row["a"]), str(row["b"])
        row["user_id"] = str(row["user_id"])
        if "reference_values" in row:
            reference_record = record_from_row({"entity_id": f"reference:{row['id']}", "values": row["reference_values"]}, specs)
            row["reference"] = raw_reference(reference_record, specs)
        else:
            row["reference"] = raw_reference(indexed[row["reference_id"]], specs)
        splits[split].append(row)
        entity_splits[split].update(involved)
        user_splits[split].add(str(row.get("group_id", row["user_id"])))
        actual_user_splits[split].add(row["user_id"])
    if any(not rows for rows in splits.values()):
        raise ValueError("External bundle needs nonempty train/development/calibration/test observations")
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i+1:]:
            if metadata["split_mode"] == "cold" and entity_splits[left] & entity_splits[right]:
                raise ValueError(f"Cold split entities overlap: {left}/{right}")
            if metadata.get("user_disjoint", True) and (user_splits[left] & user_splits[right] or actual_user_splits[left] & actual_user_splits[right]):
                raise ValueError(f"User/group splits overlap: {left}/{right}")
    if not metadata.get("user_disjoint", True) and "user_overlap_reason" not in metadata:
        raise ValueError("Non-disjoint user groups require a documented user_overlap_reason")
    return {"records": records, "specs": specs, "observations": [row for name in SPLITS for row in splits[name]],
            "splits": splits, "metadata": metadata,
            "split_entities": {name: sorted(values) for name, values in entity_splits.items()},
            "split_users": {name: sorted(values) for name, values in user_splits.items()},
            "files": {str(p.name): _sha256(p) for p in (root/"manifest.json", schema_path, records_path, observations_path)}}


def external_bundle_template() -> dict:
    """Schema documentation only: contains no fabricated human observations."""
    return {"manifest": {"metadata": {"dataset_id": "REPLACE", "source": "REPLACE_WITH_SOURCE_AND_VERSION",
            "licence": "REPLACE_WITH_PERMITTED_USE", "label_origin": "human_observed",
            "reference_source": "directly_recorded", "split_mode": "cold", "user_disjoint": True,
            "provenance": {"independent": True, "exposure": "unexamined", "previously_examined": False}}},
            "schema": [{"field": "declared_property", "kind": "numeric"}],
            "record_fields": ["entity_id", "uri", "values", "missing"],
            "observation_fields": ["id", "user_id", "group_id (optional)", "reference_id OR reference_values",
                                   "a", "b", "label (a chosen=1, b chosen=0)", "split"],
            "split_names": list(SPLITS), "observations": [],
            "notice": "Template only. Supply genuine permitted observed data; do not use this empty template as a validation dataset."}
