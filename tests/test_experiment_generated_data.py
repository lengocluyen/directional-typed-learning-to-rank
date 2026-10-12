"""Protocol checks using explicitly artificial unit-test fixtures, not evidence."""
import json
from pathlib import Path

import numpy as np
import pytest
from rdflib import Literal

from dtltr_vehicle.experiments.data import (
    audit_registry, external_bundle_template, load_external_bundle, raw_reference,
)
from dtltr_vehicle.experiments.features import Encoder
from dtltr_vehicle.experiments.generated import run_external, run_generated
from dtltr_vehicle.basis import PropertySpec, Reference
from dtltr_vehicle.open_catalogue import OV, PROPERTIES, SEGMENTS, to_graph
from rgsmw_next.types import EntityRecord


@pytest.fixture
def catalogue(tmp_path):
    rows = []
    for i in range(60):
        numeric = {field: float(10 + i * (j+1)) for j, (field, (_, kind)) in enumerate(PROPERTIES.items()) if kind in {"numeric", "count"}}
        values = {**numeric, "energy_carrier": ["petrol", "electricity"] if i % 3 == 0 else ["petrol"],
                  "powertrain": "PHEV" if i % 3 == 0 else "ICE", "body_type": f"type:{i%3}",
                  "brand": f"brand:{i%4}", "gearbox": "auto", "segment": SEGMENTS[i%len(SEGMENTS)]}
        missing = {}
        if i % 3:
            values["electric_range_km"] = None
            missing["electric_range_km"] = "not_applicable"
        if i % 11 == 0:
            values["co2_g_km"] = None
            missing["co2_g_km"] = "not_reported"
        rows.append({"entity_id": f"ademe:{i:05d}", "label": f"Test vehicle {i}", "values": values, "missing": missing})
    records = tmp_path / "records.json"
    records.write_text(json.dumps(rows), encoding="utf-8")
    rdf = tmp_path / "vehicles.ttl"
    to_graph(rows).serialize(rdf, format="turtle")
    return records, rdf, rows


def _settings(records, models=None):
    return {"paths": {"records": str(records)}, "models": models or ["TD", "ST", "DT", "direct_ref", "spline_ref"],
            "completion": ["raw", "default"], "clipping": ["clipped"], "knots": 2,
            "penalties": {"lam": [.1], "gamma": [0.0], "mu": [0.0]},
            "sizes": {"train": 15, "development": 8, "calibration": 8, "test": 12},
            "catboost_iteration_grid": [10], "noise_fraction": .25}


def test_generated_four_way_entity_and_label_matching(catalogue):
    result = run_generated(_settings(catalogue[0]), seed=7, rule="gain_loss")
    splits = result["splits"]
    names = list(splits)
    for i, name in enumerate(names):
        for other in names[i+1:]:
            assert not set(splits[name]["entity_ids"]) & set(splits[other]["entity_ids"])
    expected_train = set(splits["train"]["entity_ids"])
    for fit in result["fits"]:
        assert set(fit["encoder"]["train_ids"]) == expected_train
        assert fit["test_used_in_selection"] is False
        assert fit["calibration_row_ids"] == splits["calibration"]["row_ids"]
    # Every model/policy uses exactly the same raw test pairs, truth and labels.
    grouped = {}
    for row in result["predictions"]:
        fingerprint = (row["query_id"], row["a"], row["b"], row["label"], row["truth_margin_evaluation_only"])
        grouped.setdefault(row["row_id"], set()).add(fingerprint)
        if row["pairwise_contributions"] is not None:
            assert sum(row["pairwise_contributions"].values()) == pytest.approx(row["margin"], abs=1e-8)
    assert len(grouped) == 12
    assert all(len(values) == 1 for values in grouped.values())
    assert result["diagnostics"]["metadata"]["raw_query_truth_preserved"]
    assert any(row["reference_incomplete"] for row in result["diagnostics"]["query_strata"])
    # Output must be usable by the immutable runner without NaN/in-memory types.
    json.dumps(result, allow_nan=False)


def test_registry_audit_detects_mapping_and_missing_annotation_errors(catalogue):
    records, rdf, rows = catalogue
    settings = {"paths": {"records": str(records), "rdf": str(rdf)}}
    result = audit_registry(settings)
    assert result["status"] == "pass"
    assert result["checked_entity_properties"] == 60 * len(PROPERTIES)
    assert result["encoder_fidelity"]["keeps_missing_reason_identity"] is False
    assert result["missing_reasons"]["not_reported"] > 0
    graph = to_graph(rows)
    subject = OV["vehicle/00000"]
    graph.add((subject, OV.priceEur, Literal("unexpected")))
    for annotation in list(graph.objects(subject, OV.missingValue)):
        graph.remove((subject, OV.missingValue, annotation))
    graph.serialize(rdf, format="turtle")
    failed = audit_registry(settings)
    assert failed["status"] == "failed"
    assert {r["issue"] for r in failed["failures"]} >= {"rdf_record_value_mismatch", "missing_annotation_mismatch"}


def test_raw_reference_and_clip_strata_preserve_actual_missingness():
    specs = [PropertySpec("x", "numeric"), PropertySpec("tags", "set")]
    train = [EntityRecord(str(i), f"urn:{i}", {"x": (float(i),), "tags": ("a",)}) for i in range(6)]
    raw = raw_reference(EntityRecord("q", "urn:q", {"tags": ()}), specs)
    assert raw.get("x") is None
    assert raw.get("tags") == frozenset()
    encoder = Encoder.fit(train, specs, "DT", completion="default")
    diagnostic = encoder.diagnostics(EntityRecord("heldout", "urn:heldout", {"x": (1000.,), "tags": ("new",)}), raw)
    assert diagnostic["reference_missing_count"] == 1
    assert diagnostic["outside_clip_count"] == 1
    assert diagnostic["unseen_count"] == 1


@pytest.fixture
def external(tmp_path):
    root = tmp_path / "external"
    root.mkdir()
    metadata = external_bundle_template()["manifest"]["metadata"]
    metadata.update({"dataset_id": "unit-test-fixture-only", "source": "artificial unit-test fixture, not research data",
                     "licence": "test fixture", "provenance": {"independent": False, "exposure": "development", "previously_examined": True}})
    (root/"manifest.json").write_text(json.dumps({"metadata": metadata}), encoding="utf-8")
    (root/"schema.json").write_text(json.dumps([{"field": "x", "kind": "numeric"}, {"field": "category", "kind": "categorical"}]), encoding="utf-8")
    records = [{"entity_id": str(i), "values": {"x": float(i), "category": "a" if i%2 else "b"}} for i in range(12)]
    (root/"records.json").write_text(json.dumps(records), encoding="utf-8")
    observations = [{"id": f"pair:{i}", "user_id": f"u:{i}", "reference_id": str(3*i),
                     "a": str(3*i+1), "b": str(3*i+2), "label": i%2, "split": split}
                    for i, split in enumerate(("train", "development", "calibration", "test"))]
    (root/"observations.json").write_text(json.dumps(observations), encoding="utf-8")
    return root


def test_external_validated_run_has_no_invented_truth(external):
    bundle = load_external_bundle(external)
    assert len(bundle["observations"]) == 4
    settings = _settings(external, models=["DT", "direct_ref", "spline_ref"])
    settings["paths"]["external"] = str(external)
    result = run_external(settings, 1)
    assert result["diagnostics"]["independence_status"] == "development"
    assert all("truth_margin_evaluation_only" not in row for row in result["predictions"])
    assert all(row["accuracy_clean"] is None for row in result["metrics"])
    json.dumps(result, allow_nan=False)


def test_runner_external_dispatch_all_models_finite_test_fixture(external):
    """Integration smoke on artificial fixture labels; never paper evidence."""
    pytest.importorskip("catboost")
    from dtltr_vehicle.experiments.generated import MODELS
    from dtltr_vehicle.experiments.runner import canonical, execute_task

    settings = _settings(external, models=list(MODELS))
    settings["paths"]["external"] = str(external)
    settings.update({"completion": ["raw"], "clipping": ["clipped"],
                     "catboost_iterations": 2, "catboost_iteration_grid": [2],
                     "catboost": {"depth": 1, "thread_count": 1}, "stability_probe_count": 2})
    task = {"suite": "external", "seed": 1, "id": "test-only-external-dispatch"}
    result = execute_task(task, settings)
    assert result["planned_task"] == task
    assert {row["model"] for row in result["fits"]} == set(MODELS)
    assert len(result["fits"]) == len(MODELS)
    catboost = next(row for row in result["fits"] if row["model"] == "catboost_ref")
    assert catboost["fit"]["iterations"] == 2
    assert all(row["fit"]["finite"] and row["fit"]["converged"] for row in result["fits"])
    assert result["diagnostics"]["metadata"]["dataset_id"] == "unit-test-fixture-only"
    assert result["diagnostics"]["independence_status"] == "development"
    assert all("truth_margin_evaluation_only" not in row for row in result["predictions"])
    assert all(0 <= row["p_preferred"] <= 1 and np.isfinite(row["log_loss"]) for row in result["predictions"])
    assert json.loads(canonical(result))["planned_task"] == task
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("mutation,match", [
    ("unknown", "Unknown observation entity"),
    ("cold_overlap", "Cold split entities overlap"),
    ("user_overlap", "User/group splits overlap"),
    ("missing_provenance", "Provenance must declare"),
    ("false_independence", "Independent validation cannot"),
    ("generated", "requires observed labels"),
])
def test_external_rejects_invalid_provenance_and_split_leakage(external, mutation, match):
    observations_path = external/"observations.json"
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    manifest_path = external/"manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "unknown":
        observations[-1]["a"] = "absent"
    elif mutation == "cold_overlap":
        observations[-1]["reference_id"] = "0"
    elif mutation == "user_overlap":
        observations[-1]["user_id"] = observations[0]["user_id"]
    elif mutation == "missing_provenance":
        del manifest["metadata"]["provenance"]["exposure"]
    elif mutation == "false_independence":
        manifest["metadata"]["provenance"]["independent"] = True
    elif mutation == "generated":
        manifest["metadata"]["label_origin"] = "generated"
    observations_path.write_text(json.dumps(observations), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_external_bundle(external)


def test_empty_external_template_is_explicitly_not_a_dataset():
    template = external_bundle_template()
    assert template["observations"] == []
    assert "Template only" in template["notice"]
