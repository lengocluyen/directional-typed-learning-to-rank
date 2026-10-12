"""Matched generated-mechanism experiments and frozen observed-bundle validation.

Four disjoint entity pools protect generated test query and candidate entities.
Every representation sees the same triples and noisy labels.  Rule utilities
are evaluation truth only; they never enter fitting, model selection or
probability calibration.  The observed-bundle entry point never invents truth.
"""
from __future__ import annotations

import hashlib
import itertools
import math
import time
from pathlib import Path

import numpy as np

from rgsmw_next.distance import FittedMetric
from ..generated_study import RULES, NOISE_FRACTION, Rules, predicate_specs, property_specs
from .data import SPLITS, load_external_bundle, raw_reference, read_rows, record_from_row
from .features import Encoder
from .models import fit_pair, penalty_grid, fit_temperature, pairwise_metrics

MODELS = ("TD", "ST", "DT", "direct", "direct_ref", "spline", "spline_ref", "catboost_ref")


def _rows(pool, n, rng):
    if len(pool) < 3 or n < 1:
        raise ValueError("Each generated split needs at least three entities and one observation")
    return [tuple(int(x) for x in rng.choice(len(pool), size=3, replace=False)) for _ in range(n)]


def _split_entities(records, seed, fractions):
    fractions = np.asarray(fractions, dtype=float)
    if fractions.shape != (4,) or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1):
        raise ValueError("entity_split must give four positive fractions summing to one")
    if len(records) < 12:
        raise ValueError("Generated four-way holdout needs at least twelve entities")
    # Sort by an independently seeded hash: reproducible regardless of file order.
    ordered = sorted(records, key=lambda r: hashlib.sha256(f"{seed}:entity-pool:{r.entity_id}".encode()).digest())
    counts = np.floor(fractions * len(records)).astype(int)
    counts[-1] += len(records) - int(counts.sum())
    if np.any(counts < 3):
        raise ValueError("Configured fractions leave a split with fewer than three entities")
    offset = 0
    pools = {}
    for name, count in zip(SPLITS, counts, strict=True):
        pools[name] = ordered[offset:offset+int(count)]
        offset += int(count)
    return pools


def _policy_values(settings, key, default):
    values = settings.get(key, default)
    return list(values) if isinstance(values, (list, tuple)) else [values]


def _pair_arrays(encoder, rows, indexed):
    references = [row["reference"] for row in rows]
    left = encoder.matrix([indexed[row["a"]] for row in rows], references)
    right = encoder.matrix([indexed[row["b"]] for row in rows], references)
    sign = np.array([2 * row["label"] - 1 for row in rows], dtype=float)
    differences = (left - right) * sign[:, None]
    pair_items = np.stack([np.where((sign > 0)[:, None], left, right),
                           np.where((sign > 0)[:, None], right, left)], axis=1)
    return {"left": left, "right": right, "sign": sign, "D": differences, "pair_items": pair_items}


def _margin(fit, arrays):
    return np.asarray(fit.score(arrays["left"])) - np.asarray(fit.score(arrays["right"]))


def _selection_boundary(selected, candidates):
    out = {}
    for key, value in selected.items():
        values = sorted({row[key] for row in candidates if key in row})
        out[key] = {"value": value, "minimum": values[0], "maximum": values[-1],
                    "at_minimum": len(values) > 1 and value == values[0],
                    "at_maximum": len(values) > 1 and value == values[-1], "grid_size": len(values)}
    return out


def _query_strata(encoder, row, indexed):
    diagnostics = [encoder.diagnostics(indexed[row[alternative]], row["reference"]) for alternative in ("a", "b")]
    reference_missing = max(int(d.get("reference_missing_count", 0)) for d in diagnostics)
    candidate_missing = sum(int(d.get("candidate_missing_count", 0)) for d in diagnostics)
    outside_clip = sum(int(d.get("outside_clip_count", 0)) for d in diagnostics)
    unseen = sum(int(d.get("unseen_count", 0)) for d in diagnostics)
    return {"reference_incomplete": reference_missing > 0, "reference_missing_count": reference_missing,
            "candidate_incomplete": candidate_missing > 0, "candidate_missing_count": candidate_missing,
            "outside_clip": outside_clip > 0, "outside_clip_count": outside_clip,
            "unseen_count": unseen, "encoding_diagnostics": diagnostics}


def _stability_probe_rows(records, specs, count):
    """Fixed label-free pairs: evaluation diagnostics only, shared across seeds."""
    ordered = sorted(records, key=lambda r: r.entity_id)
    if len(ordered) < 3 or count < 1:
        return []
    rng = np.random.default_rng(190711)
    seen, rows = set(), []
    while len(rows) < min(count, len(ordered)*(len(ordered)-1)*(len(ordered)-2)):
        q, a, b = (ordered[i] for i in rng.choice(len(ordered), 3, replace=False))
        fingerprint = (q.entity_id, a.entity_id, b.entity_id)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        rows.append({"id": "probe:"+hashlib.sha256("\0".join(fingerprint).encode()).hexdigest()[:20],
                     "reference_id": q.entity_id, "reference": raw_reference(q, specs),
                     "a": a.entity_id, "b": b.entity_id, "label": 1})
    return rows


def _evaluate(settings, seed, observations, records, specs, task, splits, metadata, *, generated=False):
    models = settings.get("generated_models" if generated else "external_models", settings.get("models", list(MODELS)))
    completions = _policy_values(settings, "completion", ["raw", "default"])
    clippings = _policy_values(settings, "clipping", ["clipped", "unclipped"])
    if any(model not in MODELS for model in models):
        raise ValueError(f"Unknown model(s): {models}")
    if not set(completions) <= {"raw", "default"} or not set(clippings) <= {"clipped", "unclipped"}:
        raise ValueError("completion is raw/default; clipping is clipped/unclipped")
    indexed = {record.entity_id: record for record in records}
    train_ids = set(splits["train"]["entity_ids"])
    train_records = [indexed[entity_id] for entity_id in sorted(train_ids)]
    if not train_records:
        raise ValueError("No training entities")
    metrics, predictions, fits, strata_rows = [], [], [], []
    # The probe contains no choice labels or rule truth, and does not enter any
    # preprocessing, model selection or calibration. Pool exposure is explicit.
    probes = _stability_probe_rows(records, specs, int(settings.get("stability_probe_count", 20)))
    for model, completion, clipping in itertools.product(models, completions, clippings):
        started = time.perf_counter()
        knots_values = _policy_values(settings, "knots", 5) if model.startswith("spline") else [5]
        if model.startswith("spline") and "knot_grid" in settings:
            knots_values = _policy_values(settings, "knot_grid", 5)
        penalties = list(penalty_grid(settings, model, adaptation=False))
        iterations_values = settings.get("catboost_iteration_grid", [settings.get("catboost_iterations", 200)]) if model == "catboost_ref" else [0]
        if isinstance(iterations_values, int):
            iterations_values = [iterations_values]
        candidates = []
        best = None
        for knots in knots_values:
            encoding_started = time.perf_counter()
            # No development/calibration/test entities or references fit scales,
            # vocabularies, imputation defaults or spline knot positions.
            encoder = Encoder.fit(train_records, specs, model, completion=completion, clipping=clipping,
                                  knots=int(knots))
            if model == "spline_ref":
                # Unique training records keep the same scale/vocabulary weight
                # across all models. For reference-relative spline knots use
                # the first TRAIN pair involving each candidate; unused train
                # entities receive the label-free TRAIN default reference.
                references_by_candidate = {}
                for row in observations["train"]:
                    for alternative in ("a", "b"):
                        references_by_candidate.setdefault(row[alternative], row["reference"])
                aligned_refs = [references_by_candidate.get(r.entity_id, encoder.default_reference()) for r in train_records]
                encoder = Encoder.fit(train_records, specs, model, completion=completion, clipping=clipping,
                                      knots=int(knots), references=aligned_refs)
            encoded = {name: _pair_arrays(encoder, rows, indexed) for name, rows in observations.items()}
            encoding_seconds = time.perf_counter() - encoding_started
            for penalty, iterations in itertools.product(penalties, iterations_values):
                fit_started = time.perf_counter()
                config = dict(settings)
                config.update({"pair_items": encoded["train"]["pair_items"], "iterations": int(iterations),
                               "catboost_iterations": int(iterations)})
                fit = fit_pair(encoded["train"]["D"], penalty, encoder.groups, model=model, seed=seed, config=config)
                fit_seconds = time.perf_counter() - fit_started
                dev_margins = _margin(fit, encoded["development"]) * encoded["development"]["sign"]
                dev_loss = float(np.mean(np.logaddexp(0.0, -dev_margins)))
                parameters = {**penalty, **({"knots": int(knots)} if model.startswith("spline") else {}),
                              **({"iterations": int(iterations)} if model == "catboost_ref" else {})}
                candidates.append({**parameters, "development_log_loss": dev_loss, "fit_seconds": fit_seconds})
                if best is None or dev_loss < best[0] - 1e-12:
                    best = (dev_loss, fit, encoder, encoded, parameters, encoding_seconds, fit_seconds)
        if best is None:
            raise ValueError(f"Empty tuning grid for {model}")
        dev_loss, fit, encoder, encoded, parameters, encoding_seconds, fit_seconds = best
        calibration_margins = _margin(fit, encoded["calibration"]) * encoded["calibration"]["sign"]
        temperature = fit_temperature(calibration_margins)
        test_margins = _margin(fit, encoded["test"])
        observed = pairwise_metrics(test_margins * encoded["test"]["sign"], temperature)
        clean = np.array([row.get("truth", math.nan) for row in observations["test"]])
        strict = np.abs(clean) > 1e-12
        strict_correct = np.where(np.abs(test_margins) <= 1e-12, .5, (test_margins * clean > 0).astype(float))
        n_clean = int(strict.sum())
        accuracy_clean = float(np.mean(strict_correct[strict])) if n_clean else None
        common = {"dataset": metadata["dataset_id"], "protocol": task["protocol"], "seed": seed, "model": model,
                  "reference": metadata.get("reference_source", "raw_catalogue_query"), "completion": completion,
                  "clipping": clipping, "budget": len(observations["train"]), "rule": task.get("rule", "observed")}
        total_seconds = time.perf_counter() - started
        metrics.append({**common, "n": n_clean if generated else len(test_margins), "n_observed": len(test_margins),
                        "accuracy": accuracy_clean if generated else observed["accuracy"],
                        "accuracy_clean": accuracy_clean if generated else None,
                        "accuracy_clean_strict_preferences": accuracy_clean if generated else None,
                        "accuracy_observed": observed["accuracy"], "log_loss": observed["log_loss"], "brier": observed["brier"],
                        "model_tie_rate": observed["tie_rate"], "truth_ties_excluded": int(len(clean)-n_clean) if generated else 0,
                        "seconds": total_seconds, "dimension": len(encoder.names)})
        contributions = fit.contributions(encoded["test"]["left"] - encoded["test"]["right"], encoder)
        if contributions is None:
            contributions = [None] * len(test_margins)
        if isinstance(contributions, dict):
            contributions = [contributions]
        for index, (row, margin, contribution) in enumerate(zip(observations["test"], test_margins, contributions, strict=True)):
            probability = float(1.0 / (1.0 + np.exp(-np.clip(temperature * margin, -40, 40))))
            p_preferred = probability if row["label"] else 1-probability
            observed_margin = float(margin) * (2*row["label"]-1)
            accuracy_observed = .5 if abs(float(margin)) <= 1e-12 else float(observed_margin > 0)
            clean_correct = None if generated and not strict[index] else float(strict_correct[index]) if generated else accuracy_observed
            strata = _query_strata(encoder, row, indexed)
            prediction = {**common, "row_id": row["id"], "user_id": row.get("user_id"), "query_id": row.get("reference_id"),
                          "unit_id": row["reference_id"] if generated else str(row.get("group_id", row["user_id"])),
                          "a": row["a"], "b": row["b"], "label": row["label"], "margin": float(margin),
                          "probability_a": probability, "p_preferred": p_preferred,
                          "accuracy": clean_correct, "accuracy_observed": accuracy_observed,
                          "log_loss": float(np.logaddexp(0, -temperature*observed_margin)),
                          "brier": float((1-p_preferred)**2), "confidence": max(probability, 1-probability),
                          "pairwise_contributions": contribution,
                          "reference_incomplete": strata["reference_incomplete"], "candidate_incomplete": strata["candidate_incomplete"],
                          "outside_clip": strata["outside_clip"], "reference_missing_count": strata["reference_missing_count"],
                          "outside_clip_count": strata["outside_clip_count"]}
            if generated:
                prediction.update({"truth_margin_evaluation_only": float(clean[index]),
                                   "clean_label_evaluation_only": None if not strict[index] else int(clean[index] > 0),
                                   "truth_tie": bool(not strict[index]),
                                   "strict_clean_correct": None if not strict[index] else float(strict_correct[index])})
            predictions.append(prediction)
            strata_rows.append({**common, "row_id": row["id"], **strata})
        for stratum in ("reference_incomplete", "candidate_incomplete", "outside_clip"):
            for value in (False, True):
                indices = np.array([r[stratum] == value for r in strata_rows[-len(test_margins):]], dtype=bool)
                if not np.any(indices):
                    continue
                observed_stratum = pairwise_metrics(test_margins[indices] * encoded["test"]["sign"][indices], temperature)
                clean_keep = indices & strict
                stratum_accuracy = float(np.mean(strict_correct[clean_keep])) if generated and np.any(clean_keep) else (None if generated else observed_stratum["accuracy"])
                metrics.append({**common, "stratum": stratum, "stratum_value": value,
                                "n": int(clean_keep.sum()) if generated else int(indices.sum()), "n_observed": int(indices.sum()),
                                "accuracy": stratum_accuracy, "accuracy_clean": stratum_accuracy if generated else None,
                                "accuracy_observed": observed_stratum["accuracy"], "log_loss": observed_stratum["log_loss"],
                                "brier": observed_stratum["brier"], "truth_ties_excluded": int(np.sum(indices & ~strict)) if generated else 0,
                                "seconds": total_seconds, "dimension": len(encoder.names)})
        boundary = _selection_boundary(parameters, candidates)
        probe_arrays = _pair_arrays(encoder, probes, indexed) if probes else None
        probe_margins = _margin(fit, probe_arrays) if probes else []
        probe_contributions = fit.contributions(probe_arrays["left"]-probe_arrays["right"], encoder) if probes else []
        if probe_contributions is None:
            probe_contributions = [None] * len(probes)
        probe_results = [{"row_id": row["id"], "query_id": row["reference_id"], "a": row["a"], "b": row["b"],
                          "margin": float(margin), "pairwise_contributions": contribution}
                         for row, margin, contribution in zip(probes, probe_margins, probe_contributions, strict=True)]
        fits.append({**common, "selected": parameters, "selection_boundary": boundary,
                     "development_log_loss": dev_loss, "temperature": temperature,
                     "temperature_boundary": not (math.exp(-7)*1.01 < temperature < math.exp(7)*0.99),
                     "tuning_candidates": candidates, "encoder": encoder.metadata(), "fit": fit.info,
                     "coefficient_vector": None if fit.theta is None else np.asarray(fit.theta).tolist(),
                     "property_fields": list(encoder.fields), "feature_names": list(encoder.names),
                     "fit_train_entity_ids": sorted(train_ids), "selection_row_ids": splits["development"]["row_ids"],
                     "calibration_row_ids": splits["calibration"]["row_ids"], "encoding_seconds": encoding_seconds,
                     "selected_fit_seconds": fit_seconds, "total_seconds": total_seconds,
                     "stability_probes": probe_results,
                     "stability_probe_use": "descriptive contributions on common label-free raw catalogue pairs; includes all entity pools; never selection/calibration",
                     "test_used_in_selection": False, "model_capacity": "nonlinear_tree" if model == "catboost_ref" else "additive",
                     "decomposition": "not_an_exact_additive_model" if model == "catboost_ref" else "exact_pairwise_property_sum"})
    return {"metrics": metrics, "predictions": predictions, "fits": fits, "splits": splits,
            "diagnostics": {"query_strata": strata_rows, "metadata": metadata,
                            "truth_use": "evaluation only; never fitting/selection/calibration" if generated else "no unobserved ground truth invented",
                            "comparisons": "Compare within completion/clipping blocks; distinguish representation from learner capacity."},
            "task": task}


def run_generated(settings: dict, seed: int, rule: str) -> dict:
    if rule not in RULES:
        raise ValueError(f"Unknown generated rule: {rule}")
    records = [record_from_row(row) for row in read_rows(settings["paths"]["records"])]
    if len({r.entity_id for r in records}) != len(records):
        raise ValueError("Catalogue entity IDs must be unique")
    specs = property_specs()
    pools = _split_entities(records, seed, settings.get("entity_split", [.60, .15, .10, .15]))
    sizes = settings.get("sizes", {"train": 3000, "development": 1000, "calibration": 1000, "test": 2000})
    # The generating mechanism is fixed from the raw declared catalogue. Its
    # fitted metric/scales are never exposed to any experimental encoder.
    truth_metric = FittedMetric.fit(records, predicate_specs())
    generating = Rules(records, truth_metric, seed)
    rng = np.random.default_rng(seed)
    sampled = {name: _rows(pools[name], int(sizes[name]), rng) for name in SPLITS}
    truths = {name: np.array([generating.utility(rule, pools[name][q], pools[name][a]) -
                             generating.utility(rule, pools[name][q], pools[name][b]) for q, a, b in triples])
              for name, triples in sampled.items()}
    noise_fraction = float(settings.get("noise_fraction", NOISE_FRACTION))
    if not math.isfinite(noise_fraction) or noise_fraction < 0:
        raise ValueError("noise_fraction must be finite and nonnegative")
    noise_scale = noise_fraction * float(np.std(truths["train"]))
    if noise_fraction > 0:
        noise_scale = max(1e-9, noise_scale)
    observations = {}
    splits = {}
    for name in SPLITS:
        noisy = (truths[name] + rng.logistic(0, noise_scale, len(sampled[name])) > 0).astype(int)
        observations[name] = [{"id": f"generated:{seed}:{name}:{i}", "user_id": None,
                               "reference_id": pools[name][q].entity_id, "reference": raw_reference(pools[name][q], specs),
                               "a": pools[name][a].entity_id, "b": pools[name][b].entity_id,
                               "label": int(noisy[i]), "truth": float(truths[name][i])}
                              for i, (q, a, b) in enumerate(sampled[name])]
        splits[name] = {"entity_ids": sorted(r.entity_id for r in pools[name]),
                        "row_ids": [r["id"] for r in observations[name]], "observations": len(observations[name])}
    metadata = {"dataset_id": "ademe_generated", "label_origin": "generated",
                "reference_source": "raw_catalogue_query", "exposure": "development",
                "noise_scale": noise_scale, "noise_fraction": noise_fraction,
                "raw_query_truth_preserved": True, "entity_disjoint_four_way": True,
                "truth_metric_catalogue_use": "defines fixed generated mechanism only; not passed to fitted methods",
                "accuracy_definition": "accuracy on strict clean preferences excluding generating ties; predicted ties receive half credit",
                "probability_definition": "observed noisy binary labels; separate calibration pool"}
    return _evaluate(settings, seed, observations, records, specs,
                     {"suite": "generated", "seed": seed, "rule": rule, "protocol": "generated_entity_holdout"},
                     splits, metadata, generated=True)


def run_external(settings: dict, seed: int) -> dict:
    """Run identical matched controls on externally supplied observed splits."""
    bundle = load_external_bundle(settings["paths"]["external"])
    metadata = bundle["metadata"]
    splits = {name: {"entity_ids": bundle["split_entities"][name], "user_ids": bundle["split_users"][name],
                     "row_ids": [r["id"] for r in bundle["splits"][name]], "observations": len(bundle["splits"][name])}
              for name in SPLITS}
    result = _evaluate(settings, seed, bundle["splits"], bundle["records"], bundle["specs"],
                       {"suite": "external", "seed": seed, "protocol": f"external_{metadata['split_mode']}_holdout"},
                       splits, {**metadata, "bundle_hashes": bundle["files"]})
    result["diagnostics"]["independence_status"] = "declared_independent" if metadata["provenance"]["independent"] else "development"
    result["diagnostics"]["independence_validation_limit"] = "Consistency checked; previous researcher exposure cannot be independently verified by software."
    return result
