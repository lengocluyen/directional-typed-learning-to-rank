import json

import numpy as np
import pytest

from dtltr_vehicle.experiments.reporting import _axis, bootstrap_mean, paired_task, stability_rows, summarize
from dtltr_vehicle.experiments.runner import ROOT, atomic_json, canonical, load_settings, output_guard, task_plan


def test_profiles_have_bounded_nonempty_tasks_and_explicit_catboost_adaptation_exclusion():
    expected = {"smoke": 7, "pilot": 25, "full": 101}
    for name, count in expected.items():
        cfg = load_settings(ROOT / "configs" / f"{name}.yaml")
        tasks = task_plan(cfg)
        assert len(tasks) == count
        assert len({t["id"] for t in tasks}) == count
        assert "catboost_ref" not in cfg["anu_models"]
        assert "catboost_ref" in cfg["models"]
        assert all(t["suite"] != "external" for t in tasks)


def test_frozen_output_guard_and_atomic_finite_output(tmp_path):
    with pytest.raises(ValueError, match="frozen"):
        output_guard(ROOT / "runs" / "dtltr_vehicle_v2" / "new")
    path = tmp_path / "task.json"
    atomic_json(path, {"value": np.array([1., 2.])})
    assert json.loads(path.read_text())["value"] == [1., 2.]
    with pytest.raises(ValueError):
        canonical({"bad": float("nan")})


def test_generator_rules_never_pool_in_same_comparison():
    row = {"dataset": "ademe", "protocol": "generated", "model": "DT", "rule": "gain_loss"}
    assert _axis(row) != _axis({**row, "rule": "offset_ideal_point"})


def test_paired_summary_collapses_draws_within_independent_units():
    rows = []
    for draw in range(5):
        for unit, a, b in [("u1", 1., 0.), ("u2", 0., 1.)]:
            for model, value in [("DT", a), ("ST", b)]:
                rows.append({"dataset": "anu", "model": model, "unit_id": unit,
                             "row_id": unit + ":p", "draw": draw, "budget": 2,
                             "accuracy": value, "log_loss": 1. - value})
    result = paired_task(rows)
    accuracy = next(r for r in result if r["measure"] == "accuracy")
    assert accuracy["units"] == {"u1": 1., "u2": -1.}
    assert accuracy["paired_rows"] == 10
    assert len(accuracy["units"]) == 2  # Not ten independent participants.


def test_conditional_bootstrap_is_deterministic_and_single_unit_has_no_interval():
    assert bootstrap_mean([.1, .2, -.05], 200, 1) == bootstrap_mean([.1, .2, -.05], 200, 1)
    assert bootstrap_mean([.1], 200) == (None, None)


def test_generated_pair_difference_is_per_observation_not_query_reweighting():
    rows = []
    for i, (query, effect) in enumerate([("q1", 1.), ("q1", 1.), ("q2", -1.)]):
        for model, value in [("DT", max(effect, 0)), ("ST", max(-effect, 0))]:
            rows.append({"dataset": "generated", "model": model, "unit_id": query,
                         "row_id": str(i), "accuracy": value})
    comparison = paired_task(rows)[0]
    assert comparison["row_mean"] == pytest.approx(1/3)
    assert np.mean(list(comparison["units"].values())) == 0


def test_stability_accepts_both_probe_formats_and_ignores_nondecomposable_trees():
    base = {"dataset": "generated", "model": "DT"}
    parts = {"price": 1., "power": 2.}
    rows = [{**base, "seed": 1, "stability_probes": [{"row_id": "probe", "pairwise_contributions": parts}]},
            {**base, "seed": 2, "probe_contributions": {"probe": parts}},
            {**base, "model": "catboost_ref", "probe_contributions": {"probe": None}}]
    summary = stability_rows(rows)
    assert len(summary) == 1
    assert summary[0]["common_probes"] == 1


def test_generated_strata_never_enter_overall_seed_means(tmp_path, monkeypatch):
    import dtltr_vehicle.experiments.reporting as reporting
    tasks = [{"id": str(i), "suite": "generated", "seed": i} for i in range(2)]
    manifest = {"status": "complete", "fingerprint": "test", "tasks": tasks, "completed": {}}
    atomic_json(tmp_path / "configuration.json", {})
    for task in tasks:
        base = {"dataset": "generated", "protocol": "generated", "model": "DT", "accuracy": .75}
        atomic_json(tmp_path / "tasks" / (task["id"] + ".json"),
                    {"metrics": [base, {**base, "stratum": "missing", "accuracy": .1}], "predictions": []})
        from dtltr_vehicle.provenance import sha256
        manifest["completed"][task["id"]] = sha256(tmp_path / "tasks" / (task["id"] + ".json"))
    atomic_json(tmp_path / "manifest.json", manifest)
    summarize(tmp_path)
    import csv
    rows = list(csv.DictReader((tmp_path / "reports" / "metrics.csv").open()))
    assert len(rows) == 1
    assert float(rows[0]["accuracy"]) == .75
    assert int(rows[0]["accuracy_seeds"]) == 2


def test_resume_rejects_changed_identity_and_audit_failure(tmp_path, monkeypatch):
    import dtltr_vehicle.experiments.runner as runner
    import dtltr_vehicle.experiments.data as data
    cfg = ROOT / "configs" / "smoke.yaml"
    monkeypatch.setattr(runner, "input_hashes", lambda _: {"fixture": "hash"})
    monkeypatch.setattr(runner, "source_hashes", lambda: {"fixture.py": "before"})
    _, output, first = runner.prepare(cfg, tmp_path / "run")
    _, _, resumed = runner.prepare(cfg, output, resume=True)
    assert first["fingerprint"] == resumed["fingerprint"]
    monkeypatch.setattr(runner, "source_hashes", lambda: {"fixture.py": "after"})
    with pytest.raises(ValueError, match="Resume rejected"):
        runner.prepare(cfg, output, resume=True)
    monkeypatch.setattr(data, "audit_registry", lambda _: {"status": "failed", "failure_count": 1})
    with pytest.raises(ValueError, match="RDF registry audit failed"):
        runner.execute_task({"suite": "audit"}, {})


def test_lock_checks_process_without_signalling_it(tmp_path, monkeypatch):
    import os
    import socket
    import dtltr_vehicle.experiments.runner as runner
    assert runner._process_alive(os.getpid())  # Safe on Windows as well as Linux.
    atomic_json(tmp_path / ".running.lock", {"host": socket.gethostname(), "pid": os.getpid()})
    monkeypatch.setattr(runner, "_process_alive", lambda _: True)
    with pytest.raises(RuntimeError, match="Another runner is active"):
        runner._acquire_lock(tmp_path)
    monkeypatch.setattr(runner, "_process_alive", lambda _: False)
    lock = runner._acquire_lock(tmp_path)
    assert lock.exists()


def test_generated_probability_metrics_preserve_extreme_wrong_margin():
    from dtltr_vehicle.experiments.generated import pairwise_metrics
    result = pairwise_metrics(np.array([-100., 100.]))
    assert result["log_loss"] == pytest.approx(50.)


def test_resource_hashes_and_corrupted_report_inputs_are_checked(tmp_path):
    import dtltr_vehicle.experiments.runner as runner
    assert "src/rgsmw_next/config/anu_car_preferences.yaml" in runner.source_hashes()
    path = tmp_path / "tasks" / "test.json"
    atomic_json(path, {"metrics": [], "predictions": []})
    manifest = {"completed": {"test": runner.sha256(path)}}
    assert runner.verified_task_path(tmp_path, manifest, "test") == path
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupted"):
        runner.verified_task_path(tmp_path, manifest, "test")


def test_compact_diagnostics_preserves_exact_fit_and_encoding_audit_counts():
    from dtltr_vehicle.experiments.reporting import compact_diagnostics

    identity = {"dataset": "fixture", "protocol": "test-only", "model": "DT", "completion": "raw",
                "clipping": "clipped", "budget": 5, "stage": "adaptation_test",
                "penalty": {"lam": .1, "gamma": .01, "mu": 1.}}
    audits = [{**identity, "converged": True, "finite": True, "fallback": False, "status": "lbfgs",
               "stationarity_residual": 1e-8, "stationarity_tolerance": 1e-6},
              {**identity, "converged": True, "finite": True, "fallback": True, "status": "clarabel",
               "stationarity_residual": 2e-6, "stationarity_tolerance": 1e-6,
               "first_solver": {"coefficients": list(range(1000))}},
              {**identity, "converged": False, "finite": False, "status": "stopped", "stationarity_residual": None},
              {**identity, "stage": "population_tuning", "budget": None,
               "penalty": {"lam": .2, "gamma": 0., "mu": 0.}, "status": "anchor_only"}]
    strata_identity = {key: identity[key] for key in ("dataset", "protocol", "model", "completion", "clipping", "budget")}
    strata = [{**strata_identity, "reference_incomplete": True, "candidate_incomplete": True,
               "outside_clip": True, "reference_missing_count": 1, "candidate_missing_count": 3,
               "outside_clip_count": 1, "unseen_count": 2,
               "encoding_diagnostics": [{"reference_missing_fields": ["price"], "candidate_missing_fields": ["mass", "range"],
                                          "outside_clip_fields": ["price"], "unseen_fields": ["brand"]},
                                         {"reference_missing_fields": ["price"], "candidate_missing_fields": ["price"], "unseen_fields": ["brand"]}]},
              {**strata_identity, "reference_incomplete": True, "candidate_incomplete": False,
               "outside_clip": True, "reference_missing_count": 2, "candidate_missing_count": 0,
               "outside_clip_count": 4, "unseen_count": 0,
               "encoding_diagnostics": [{"reference_missing_fields": ["price", "mass"], "outside_clip_fields": ["price", "mass"]},
                                         {"reference_missing_fields": ["price", "mass"], "outside_clip_fields": ["price", "mass"]}]}]
    diagnostics = {"fit_audit": audits, "query_strata": strata, "status": "test-only"}
    before = json.dumps(diagnostics, sort_keys=True)
    compact = compact_diagnostics(diagnostics)
    assert json.dumps(diagnostics, sort_keys=True) == before  # Full task audit is untouched.
    assert diagnostics["fit_audit"] is audits
    assert "fit_audit" not in compact and "query_strata" not in compact
    assert compact_diagnostics(compact) == compact  # Compact exports are safe to re-read.
    assert compact["fit_audit_summary"]["raw_rows"] == 4
    groups = compact["fit_audit_summary"]["groups"]
    assert sum(row["fits"] for row in groups) == 4
    adaptation = next(row for row in groups if row["stage"] == "adaptation_test")
    assert adaptation["fits"] == 3
    assert adaptation["converged"] == 2 and adaptation["nonconverged"] == 1
    assert adaptation["finite"] == 2 and adaptation["finite_failures"] == 1
    assert adaptation["fallbacks"] == 1
    assert adaptation["status_counts"] == {"lbfgs": 1, "clarabel": 1, "stopped": 1}
    assert adaptation["maximum_stationarity_residual"] == 2e-6
    assert adaptation["stationarity_tolerance_violations"] == 1
    assert adaptation["stationarity_checked"] == 2 and adaptation["stationarity_missing"] == 1
    population = next(row for row in groups if row["stage"] == "population_tuning")
    assert population["convergence_unknown"] == population["finite_unknown"] == 1
    query = compact["query_strata_summary"]["groups"][0]
    assert query["encoding_evaluations"] == 2
    assert query["reference_incomplete"] == 2 and query["candidate_incomplete"] == 1
    assert query["outside_clip"] == 2
    assert query["reference_missing_count"] == 3 and query["candidate_missing_count"] == 3
    assert query["outside_clip_count"] == 5 and query["unseen_count"] == 2
    assert query["maximum_outside_clip_count"] == 4
    assert query["field_counts"]["reference_missing"] == {"price": 2, "mass": 1}
    assert query["field_counts"]["candidate_missing"] == {"mass": 1, "range": 1, "price": 1}
    assert query["field_counts"]["outside_clip"] == {"price": 3, "mass": 2}
    assert query["field_counts"]["unseen"] == {"brand": 2}
    assert "first_solver" not in json.dumps(compact)
    assert compact["status"] == "test-only"
    json.dumps(compact, allow_nan=False)


def test_compact_diagnostics_flags_nonfinite_stationarity_without_emitting_nan():
    from dtltr_vehicle.experiments.reporting import compact_diagnostics
    compact = compact_diagnostics({"fit_audit": [{"finite": True, "stationarity_residual": float("nan"),
                                                 "status": "artificial-invalid-audit"}]})
    row = compact["fit_audit_summary"]["groups"][0]
    assert row["finite_failures"] == row["stationarity_nonfinite"] == 1
    assert row["maximum_stationarity_residual"] is None
    json.dumps(compact, allow_nan=False)


def test_report_compacts_diagnostics_and_retains_full_task_audit(tmp_path, monkeypatch):
    import dtltr_vehicle.experiments.reporting as reporting
    from dtltr_vehicle.provenance import sha256
    task = {"id": "test-audit", "suite": "anu", "seed": 1}
    repeated = [{"model": "DT", "stage": "population_tuning", "finite": True, "converged": True,
                 "status": "lbfgs", "penalty": {"lam": .1, "gamma": 0., "mu": 0.},
                 "unit_id": str(i), "stationarity_residual": 1e-9, "stationarity_tolerance": 1e-6,
                 "first_solver": {"solver_detail": "x"*1000}} for i in range(100)]
    path = tmp_path / "tasks" / "test-audit.json"
    atomic_json(path, {"metrics": [], "predictions": [], "diagnostics": {"fit_audit": repeated}})
    original_hash = sha256(path)
    atomic_json(tmp_path / "manifest.json", {"status": "complete", "fingerprint": "test-only", "tasks": [task],
                                             "completed": {"test-audit": original_hash}})
    atomic_json(tmp_path / "configuration.json", {})
    reporting.summarize(tmp_path)
    report_path = tmp_path / "reports" / "diagnostics.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))[0]["diagnostics"]
    assert report["fit_audit_summary"]["raw_rows"] == 100
    assert report["fit_audit_summary"]["groups"][0]["fits"] == 100
    assert "fit_audit" not in report
    assert sha256(path) == original_hash
    assert len(json.loads(path.read_text(encoding="utf-8"))["diagnostics"]["fit_audit"]) == 100
    assert report_path.stat().st_size < path.stat().st_size / 10
