"""Unit-aware summaries; repeated splits are not independent replications."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import itertools
import json
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.stats import spearmanr

AXES = ("dataset", "protocol", "rule", "reference", "completion", "clipping", "budget")
MEASURES = ("accuracy", "log_loss", "brier")


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def compact_diagnostics(diagnostics):
    """Aggregate repeated fit/encoding audits without retaining their raw rows.

    This pure helper never changes the full task's diagnostics. Counts concern
    executed fits or encoding evaluations, not independent participants. It is
    idempotent for compact task summaries, which already contain aggregates.
    """
    if not isinstance(diagnostics, dict):
        raise ValueError("Task diagnostics must be a mapping")
    compact = {key: value for key, value in diagnostics.items() if key not in {"fit_audit", "query_strata"}}
    if "fit_audit" in diagnostics:
        audits = diagnostics["fit_audit"]
        if not isinstance(audits, list):
            raise ValueError("fit_audit must be an array")
        groups = {}
        for row in audits:
            identity = {key: row.get(key) for key in (*AXES, "model", "stage")}
            penalty = row.get("penalty", {})
            if not isinstance(penalty, dict):
                penalty = {}
            identity["penalty"] = {key: penalty.get(key) for key in ("lam", "gamma", "mu")}
            key = json.dumps(identity, sort_keys=True)
            if key not in groups:
                groups[key] = {**identity, "fits": 0, "converged": 0, "nonconverged": 0,
                               "convergence_unknown": 0, "finite": 0, "finite_failures": 0,
                               "finite_unknown": 0, "fallbacks": 0, "status_counts": Counter(),
                               "stationarity_checked": 0, "stationarity_missing": 0,
                               "stationarity_nonfinite": 0, "stationarity_tolerance_missing": 0,
                               "stationarity_tolerance_violations": 0,
                               "maximum_stationarity_residual": None,
                               "minimum_stationarity_tolerance": None, "maximum_stationarity_tolerance": None}
            summary = groups[key]
            summary["fits"] += 1
            convergence = row.get("converged")
            summary["converged" if convergence is True else "nonconverged" if convergence is False else "convergence_unknown"] += 1
            residual, tolerance = row.get("stationarity_residual"), row.get("stationarity_tolerance")
            numeric_failures = any(isinstance(row.get(field), (int, float)) and not math.isfinite(row[field])
                                   for field in ("objective", "coefficient_norm", "stationarity_residual"))
            finite = row.get("finite")
            summary["finite_failures" if finite is False or numeric_failures else "finite" if finite is True else "finite_unknown"] += 1
            summary["fallbacks"] += int(bool(row.get("fallback", "first_solver" in row)))
            summary["status_counts"][str(row.get("status", "unknown"))] += 1
            if residual is None:
                summary["stationarity_missing"] += 1
            elif not _finite_number(residual):
                summary["stationarity_nonfinite"] += 1
            else:
                summary["stationarity_checked"] += 1
                previous = summary["maximum_stationarity_residual"]
                summary["maximum_stationarity_residual"] = float(residual) if previous is None else max(previous, float(residual))
                if not _finite_number(tolerance) or tolerance < 0:
                    summary["stationarity_tolerance_missing"] += 1
                else:
                    summary["stationarity_tolerance_violations"] += int(residual > tolerance)
                    lower, upper = summary["minimum_stationarity_tolerance"], summary["maximum_stationarity_tolerance"]
                    summary["minimum_stationarity_tolerance"] = float(tolerance) if lower is None else min(lower, float(tolerance))
                    summary["maximum_stationarity_tolerance"] = float(tolerance) if upper is None else max(upper, float(tolerance))
        rows = []
        for key in sorted(groups):
            summary = groups[key]
            summary["status_counts"] = dict(sorted(summary["status_counts"].items()))
            rows.append(summary)
        compact["fit_audit_summary"] = {"raw_rows": len(audits), "groups": rows,
                                         "meaning": "Executed fits grouped by stage/model/policy/budget/penalty; counts are not independent replicates."}
    if "query_strata" in diagnostics:
        evaluations = diagnostics["query_strata"]
        if not isinstance(evaluations, list):
            raise ValueError("query_strata must be an array")
        groups = {}
        for row in evaluations:
            identity = {key: row.get(key) for key in (*AXES, "model")}
            key = json.dumps(identity, sort_keys=True)
            if key not in groups:
                groups[key] = {**identity, "encoding_evaluations": 0,
                               "reference_incomplete": 0, "candidate_incomplete": 0, "outside_clip": 0,
                               "reference_missing_count": 0, "candidate_missing_count": 0,
                               "outside_clip_count": 0, "unseen_count": 0,
                               "maximum_reference_missing_count": 0, "maximum_candidate_missing_count": 0,
                               "maximum_outside_clip_count": 0, "maximum_unseen_count": 0,
                               "field_counts": {name: Counter() for name in ("reference_missing", "candidate_missing", "outside_clip", "unseen")}}
            summary = groups[key]
            summary["encoding_evaluations"] += 1
            for flag in ("reference_incomplete", "candidate_incomplete", "outside_clip"):
                summary[flag] += int(bool(row.get(flag, False)))
            for count in ("reference_missing_count", "candidate_missing_count", "outside_clip_count", "unseen_count"):
                value = row.get(count, 0)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError(f"Invalid query audit count: {count}")
                summary[count] += value
                summary["maximum_"+count] = max(summary["maximum_"+count], value)
            reference_fields = set()
            for encoding in row.get("encoding_diagnostics", []):
                reference_fields.update(encoding.get("reference_missing_fields", []))
                for name in ("candidate_missing", "outside_clip", "unseen"):
                    summary["field_counts"][name].update(encoding.get(name+"_fields", []))
            # The shared reference is counted once per query, whereas candidate
            # fields count both alternatives' encoding evaluations.
            summary["field_counts"]["reference_missing"].update(reference_fields)
        rows = []
        for key in sorted(groups):
            summary = groups[key]
            summary["field_counts"] = {name: dict(sorted(values.items())) for name, values in summary["field_counts"].items()}
            rows.append(summary)
        compact["query_strata_summary"] = {"raw_rows": len(evaluations), "groups": rows,
                                             "meaning": "Repeated query encodings by model/policy; outside_clip denotes outside the fitted interval even with unclipped features; field counts count reference once and candidate alternatives separately."}
    if "fit_audit" in diagnostics or "query_strata" in diagnostics:
        compact["diagnostics_compaction"] = {"version": 1, "raw_audits": "retained in full task JSON only",
                                              "fit_or_encoding_rows_retained": False}
    return compact


def _axis(row):
    return tuple(str(row.get(k, "none")) for k in AXES)


def _group(row):
    return (*_axis(row), str(row.get("model", "unknown")))


def _is_generated(axis):
    return "generated" in axis[0].lower() or "generated" in axis[1].lower()


def bootstrap_mean(values, draws=2000, seed=741):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None, None
    if len(values) == 1:
        return None, None
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, draws, 256):
        count = min(256, draws - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        samples.extend(values[indices].mean(axis=1).tolist())
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(lo), float(hi)


def csv_write(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def paired_task(predictions):
    """Average paired differences within users before any across-repeat inference."""
    grouped = defaultdict(dict)
    for row in predictions:
        case = tuple(str(row.get(k, "none")) for k in ("unit_id", "row_id", "draw", "budget"))
        grouped[(_axis(row), str(row.get("model")))][case] = row
    result = []
    axes = {axis for axis, _ in grouped}
    for axis in axes:
        first = grouped.get((axis, "DT"), {})
        if not first:
            continue
        for other_axis, model in list(grouped):
            if other_axis != axis or model == "DT":
                continue
            second = grouped[(axis, model)]
            common = sorted(set(first) & set(second))
            for measure in MEASURES:
                units = defaultdict(list)
                for case in common:
                    a, b = first[case].get(measure), second[case].get(measure)
                    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
                        units[case[0]].append(float(a) - float(b))
                if units:
                    result.append({"axis": axis, "comparison": f"DT - {model}", "measure": measure,
                                   "units": {u: statistics.mean(v) for u, v in units.items()},
                                   "row_mean": statistics.mean(v for values in units.values() for v in values),
                                   "paired_rows": sum(map(len, units.values()))})
    return result


def stability_rows(fits):
    collections = defaultdict(list)
    for fit in fits:
        probes = fit.get("probe_contributions")
        if not isinstance(probes, dict):
            probes = {p["row_id"]: p["pairwise_contributions"]
                      for p in fit.get("stability_probes", [])
                      if isinstance(p.get("pairwise_contributions"), dict)}
        if isinstance(probes, dict) and probes:
            probes = {p: values for p, values in probes.items() if isinstance(values, dict) and values}
        if probes:
            collections[_group(fit)].append((str(fit.get("seed", fit.get("repeat", "none"))) + ":" + str(fit.get("fold", "none")), probes))
    rows = []
    for group, members in collections.items():
        for (run_a, probes_a), (run_b, probes_b) in itertools.combinations(members, 2):
            common = sorted(set(probes_a) & set(probes_b))
            if not common:
                continue
            fields = sorted(set.intersection(*(set(probes_a[p]) & set(probes_b[p]) for p in common)))
            if len(fields) < 2:
                continue
            a = np.array([[probes_a[p][f] for f in fields] for p in common], dtype=float)
            b = np.array([[probes_b[p][f] for f in fields] for p in common], dtype=float)
            ia, ib = np.std(a, axis=0), np.std(b, axis=0)
            corr = None
            if np.ptp(ia) > 0 and np.ptp(ib) > 0:
                value = float(spearmanr(ia, ib).statistic)
                corr = value if np.isfinite(value) else None
            nonzero = (np.abs(a) > 1e-10) & (np.abs(b) > 1e-10)
            agreement = float(np.mean(np.sign(a[nonzero]) == np.sign(b[nonzero]))) if nonzero.any() else None
            rows.append({**dict(zip(AXES, group[:-1])), "model": group[-1], "fit_a": run_a, "fit_b": run_b,
                         "common_probes": len(common), "importance_spearman": corr,
                         "nonzero_sign_agreement": agreement,
                         "interpretation": "Descriptive comparisons of overlapping refits, not independent replicates"})
    return rows


def summarize(output):
    from .runner import atomic_json, read_json, verified_task_path
    output = Path(output).resolve()
    manifest = read_json(output / "manifest.json")
    settings = read_json(output / "configuration.json")
    completed = manifest.get("completed", {})
    aggregate = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    generated_values = defaultdict(lambda: defaultdict(list))
    paired = defaultdict(lambda: defaultdict(list))
    split_effects = defaultdict(lambda: defaultdict(list))
    calibration = defaultdict(lambda: [0, 0.0, 0.0])
    raw_metrics, fits, diagnostics = [], [], []
    n_predictions, reconstruction_max = 0, 0.0
    for task in manifest["tasks"]:
        if task["id"] not in completed:
            continue
        path = verified_task_path(output, manifest, task["id"], compact_ok=True)
        result = read_json(path)
        rows = result.get("metrics", [])
        raw_metrics.extend(rows)
        fits.extend(result.get("fits", []))
        diagnostics.append({"task": task, "diagnostics": compact_diagnostics(result.get("diagnostics", {}))})
        predictions = result.get("predictions", [])
        n_predictions += len(predictions)
        if not predictions and result.get("prediction_rows", 0):
            raise ValueError("Compact export: existing reports can be read, but recomputing requires the full task files")
        if task["suite"] == "generated":
            for row in rows:
                if "stratum" in row:
                    continue
                group = _group(row)
                for measure in MEASURES:
                    value = row.get(measure)
                    if isinstance(value, (int, float)):
                        generated_values[group][measure].append(value)
        else:
            local = defaultdict(lambda: defaultdict(list))
            for row in predictions:
                key = (_group(row), str(row.get("unit_id", row.get("row_id", "none"))))
                for measure in MEASURES:
                    value = row.get(measure)
                    if isinstance(value, (int, float)):
                        local[key][measure].append(value)
            for (group, unit), measures in local.items():
                for measure, values in measures.items():
                    aggregate[group][measure][unit].append(statistics.mean(values))
        for row in predictions:
            group = _group(row)
            confidence = row.get("confidence")
            correct = row.get("accuracy_observed", row.get("accuracy"))
            if isinstance(confidence, (float, int)) and isinstance(correct, (float, int)):
                bin_id = min(9, int(float(confidence) * 10))
                stats = calibration[(group, bin_id)]
                stats[0] += 1
                stats[1] += float(confidence)
                stats[2] += float(correct)
            parts = row.get("pairwise_contributions", row.get("contributions"))
            margin = row.get("margin", row.get("raw_margin"))
            if isinstance(parts, dict) and isinstance(margin, (float, int)):
                reconstruction_max = max(reconstruction_max, abs(sum(parts.values()) - float(margin)))
        for contrast in paired_task(predictions):
            key = (contrast["axis"], contrast["comparison"], contrast["measure"])
            units = contrast["units"]
            if _is_generated(contrast["axis"]):
                split_effects[key][str(task.get("seed", "none"))].append(contrast["row_mean"])
            else:
                split_effects[key][str(task.get("seed", "none"))].extend(units.values())
                for unit, value in units.items():
                    paired[key][unit].append(value)
    summary_rows = []
    for group, measures in aggregate.items():
        row = {**dict(zip(AXES, group[:-1])), "model": group[-1]}
        for measure, units in measures.items():
            values = [statistics.mean(v) for v in units.values()]
            row[measure] = statistics.mean(values)
            row[measure + "_units"] = len(values)
        row["aggregation"] = "Equal unit weight after averaging draws and repeated test appearances"
        summary_rows.append(row)
    for group, measures in generated_values.items():
        row = {**dict(zip(AXES, group[:-1])), "model": group[-1], "aggregation": "Descriptive mean of generated seed results"}
        for measure, values in measures.items():
            row[measure] = statistics.mean(values)
            row[measure + "_seed_sd"] = statistics.stdev(values) if len(values) > 1 else None
            row[measure + "_seeds"] = len(values)
        summary_rows.append(row)
    contrasts = []
    for key, repeat_units in split_effects.items():
        axis, comparison, measure = key
        effects = [statistics.mean(v) for v in repeat_units.values()]
        units = paired.get(key, {})
        values = [statistics.mean(v) for v in units.values()]
        low, high = bootstrap_mean(values, int(settings.get("bootstrap_draws", 2000))) if values else (None, None)
        contrasts.append({**dict(zip(AXES, axis)), "comparison": comparison, "metric": measure,
                          "difference": statistics.mean(values) if values else statistics.mean(effects),
                          "ci_low": low, "ci_high": high, "units": len(values), "repeats": len(effects),
                          "repeat_min": min(effects), "repeat_max": max(effects),
                          "repeat_sd": statistics.stdev(effects) if len(effects) > 1 else None,
                          "inference": "Conditional paired unit bootstrap of averaged saved predictions; not full training uncertainty" if values else "Generated seeds described without a population CI"})
    bins = [{**dict(zip(AXES, group[:-1])), "model": group[-1], "bin": bin_id, "predictions": v[0],
             "confidence": v[1] / v[0], "observed_accuracy": v[2] / v[0],
             "interpretation": "Descriptive repeated held-out predictions"} for (group, bin_id), v in calibration.items()]
    stability = stability_rows(fits)
    reports = output / "reports"
    csv_write(reports / "metrics.csv", summary_rows)
    csv_write(reports / "raw_metrics.csv", raw_metrics)
    csv_write(reports / "strata.csv", [r for r in raw_metrics if "stratum" in r])
    csv_write(reports / "contrasts.csv", contrasts)
    csv_write(reports / "calibration.csv", bins)
    csv_write(reports / "stability.csv", stability)
    tuning = [{k: v for k, v in fit.items() if k not in {"theta", "probe_contributions", "stability_probes", "encoder"}} for fit in fits]
    csv_write(reports / "tuning.csv", tuning)
    atomic_json(reports / "diagnostics.json", diagnostics)
    atomic_json(reports / "fits.json", fits)
    report = ["# DT-LTR experiment report", "", f"Status: **{manifest['status']}**; {len(completed)}/{len(manifest['tasks'])} tasks saved.", "",
              "## Material passport", "", "Mode: code execution and reproducibility validation. Source: local public archival data and generated catalogue mechanisms.",
              f"Fingerprint: `{manifest['fingerprint']}`. Predictions: {n_predictions:,}. Fits: {len(fits):,}.", "",
              "## Interpretation", "", "Archival tasks remain development reanalyses. Repeated splits share data and are not independent replications. Generated truth is used for evaluation only.",
              "Human contrasts average repeated test appearances within each unit. Their paired bootstrap conditions on the saved fitted procedure; repeat ranges describe sensitivity to refitting and partitioning.",
              "Generated accuracy concerns clean strict generating preferences; log loss/Brier concern observed noisy labels. These endpoints must not be interchanged.",
              "The design proxy remains confounded with presentation order. Independent human validation is not supplied by this suite.", "",
              "## Evidence still pending", ""]
    report += [f"- {item}" for item in manifest.get("pending_evidence", [])]
    report += ["", "## DT comparisons", "", "Positive accuracy differences favour DT; negative log loss/Brier differences favour DT.", "",
               "| Dataset / mechanism | Reference/policy | Budget | Comparison | Metric | Difference | Conditional 95% interval | Repeat range |",
               "|---|---|---:|---|---|---:|---|---|"]
    for row in sorted(contrasts, key=lambda r: tuple(str(r.get(k)) for k in (*AXES, "comparison", "metric"))):
        interval = "not estimated" if row["ci_low"] is None else f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
        dataset = row['dataset'] + (" / " + row['rule'] if row['rule'] != "none" else "")
        report.append(f"| {dataset} | {row['reference']}/{row['completion']}/{row['clipping']} | {row['budget']} | {row['comparison']} | {row['metric']} | {row['difference']:.4f} | {interval} | [{row['repeat_min']:.4f}, {row['repeat_max']:.4f}] |")
    report += ["", "## Encoding and stability", "", f"Maximum checked pairwise reconstruction error: {reconstruction_max:.3g}. Canonical probe comparisons: {len(stability)}.",
               "Undefined rank correlations remain empty when all property effects are constant. CatBoost has no intrinsic property-additive decomposition in this suite.",
               "See diagnostics.json for RDF correspondence, incomplete-query and clipping strata, fit convergence and split-role audits.", "",
               "## Outputs", "", "- metrics.csv: unit-aware human outcomes and descriptive generated seed means.",
               "- contrasts.csv: matched DT comparisons, paired conditional intervals and refit sensitivity.",
               "- strata.csv: separate incomplete-query, missing-candidate and clipping strata; never pooled into overall outcomes.",
               "- calibration.csv: confidence/correctness bins; no independent-repeat inference.",
               "- stability.csv: repeated canonical-probe contribution comparisons.",
               "- tuning.csv and fits.json: selected penalties, boundary/convergence diagnostics and encoder metadata.",
               "- This release exports numerical reports only.", ""]
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    result = {"status": manifest["status"], "output": str(output), "completed_tasks": len(completed),
              "expected_tasks": len(manifest["tasks"]), "metric_rows": len(summary_rows), "contrast_rows": len(contrasts),
              "prediction_rows": n_predictions, "plots": 0, "pending_evidence": manifest.get("pending_evidence", [])}
    atomic_json(reports / "summary.json", result)
    return result

