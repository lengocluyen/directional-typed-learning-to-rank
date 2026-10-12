"""Repeated, grouped development reanalyses of the archival vehicle choices.

Shared data adapters and eligibility rules are reused.  Four
disjoint roles separate population fitting, penalty selection, probability
calibration and test evaluation.  These exposed archives are development data;
repeated refits do not make them an independent confirmatory sample.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import time
from collections import Counter, defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logsumexp

from rgsmw_next.anu_car_preferences import (
    anu_item_predicate_specs, anu_items_to_entity_records,
    deterministic_user_grouped_folds, load_anu_car_preferences,
)
from rgsmw_next.anu_experiment import AnuExperimentConfig, _as_observations, _eligible_view
from rgsmw_next.mcfadden_train import load_mcfadden_train
from rgsmw_next.preference_learning import PairwiseObservation

from ..basis import PropertySpec, Reference, TypedBasis
from ..mcfadden_study import CATEGORICAL, NUMERIC, _chosen_index, _records
from .features import Encoder
from .models import fit_multinomial, fit_pair, penalty_grid


def _digest(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode("utf-8")).hexdigest()


def _ordered(values: Iterable[str], seed: int, purpose: str) -> list[str]:
    return sorted(set(values), key=lambda value: (_digest(seed, purpose, value), value))


def _four_roles(units: Iterable[str], seed: int, *, test: Sequence[str] | None = None) -> dict[str, list[str]]:
    """Keep whole units in separate roles; never reuse development for calibration."""
    ordered = _ordered(units, seed, "role")
    if test is None:
        if len(ordered) < 4:
            raise ValueError("Four disjoint roles require at least four grouped units")
        test_count = max(1, round(0.2 * len(ordered)))
        test = ordered[:test_count]
    else:
        if not set(test).issubset(ordered) or not test:
            raise ValueError("Test units must be a non-empty subset of supplied units")
    rest = [unit for unit in ordered if unit not in set(test)]
    if len(rest) < 3:
        raise ValueError("Outer training partition needs fit, development and calibration units")
    dev_count = min(max(1, round(0.2 * len(rest))), len(rest) - 2)
    cal_count = min(max(1, round(0.2 * len(rest))), len(rest) - dev_count - 1)
    roles = {"development": rest[:dev_count], "calibration": rest[dev_count:dev_count + cal_count],
             "train": rest[dev_count + cal_count:], "test": sorted(test)}
    assert len(set(itertools.chain.from_iterable(roles.values()))) == sum(map(len, roles.values()))
    return roles


def _budgets(settings: dict) -> list[int]:
    values = list(settings.get("budgets", [0, 2, 5, 10, 20]))
    if any(int(v) != v or v < 0 for v in values) or values != sorted(set(values)) or 0 not in values:
        raise ValueError("budgets must be sorted unique non-negative integers including zero")
    return values


def _support_views(rows: Sequence[PairwiseObservation], maximum: int, draws: int, seed: int) -> dict:
    """Reserve test queries once, then permute only the support pool for draws.

    Evaluation rows are identical across draws and all nested budgets. Unused
    support rows stay reserved and are never moved into evaluation at budget 0.
    """
    if draws < 1:
        raise ValueError("support_draws must be positive")
    grouped: dict[str, list[PairwiseObservation]] = defaultdict(list)
    for row in rows:
        grouped[row.user_id].append(row)
    views = {}
    for user, observations in sorted(grouped.items()):
        groups = [row.group_id for row in observations]
        if len(groups) != len(set(groups)):
            raise ValueError("Support splitting requires one row per collapsed pair group")
        ordered = sorted(observations, key=lambda row: (_digest(seed, "pool", user, row.group_id), row.group_id))
        if len(ordered) <= maximum:
            raise ValueError(f"{user} has insufficient observations for maximum support {maximum}")
        pool, evaluation = ordered[:maximum], ordered[maximum:]
        support_draws = [sorted(pool, key=lambda row: (_digest(seed, "draw", draw, user, row.group_id), row.group_id))
                         for draw in range(draws)]
        views[user] = {"support": support_draws, "evaluation": evaluation,
                       "support_ids": [r.group_id for r in pool],
                       "evaluation_ids": [r.group_id for r in evaluation]}
    return views


def _grid(settings: dict, model: str, *, adaptation: bool = False) -> list[dict[str, float]]:
    """Use shared grids, retaining the selected population penalty in adaptation."""
    adapted = {**settings, "grids": settings.get("penalties", settings.get("grids", {}))}
    cells = penalty_grid(adapted, model, adaptation=adaptation)
    if adaptation:
        return [{"mu": value} for value in sorted({cell["mu"] for cell in cells})]
    return cells


def _boundary(value: float, grid: Sequence[float]) -> bool:
    return len(set(grid)) > 1 and value in (min(grid), max(grid))


def _audit_fit(fit, stage: str, audit: list[dict], **identity) -> None:
    info = dict(fit.info)
    if info.get("converged") is False:
        raise RuntimeError(f"Non-converged {stage} fit: {info}")
    if fit.theta is not None and not np.all(np.isfinite(fit.theta)):
        raise RuntimeError(f"Non-finite coefficients in {stage} fit")
    audit.append({"stage": stage, **identity, **info})


def _pair_statistics(margins: np.ndarray, temperature: float = 1.0) -> dict[str, np.ndarray]:
    values = np.asarray(margins, dtype=float)
    ties = np.abs(values) <= 1e-12
    scaled = values * temperature
    probabilities = expit(scaled)
    return {"accuracy": np.where(ties, 0.5, (values > 0).astype(float)),
            "log_loss": np.logaddexp(0.0, -scaled), "brier": (1.0 - probabilities) ** 2,
            "tie_rate": ties.astype(float), "probability": probabilities,
            "confidence": np.maximum(probabilities, 1.0 - probabilities)}


def _temperature(utilities: np.ndarray, chosen: np.ndarray | None = None) -> tuple[float, dict]:
    if (not np.any(np.abs(utilities) > 1e-12) or
            (chosen is not None and np.all(np.ptp(utilities, axis=1) <= 1e-12))):
        return 1.0, {"status": "all_ties", "boundary": False}
    def loss(log_scale):
        scaled = math.exp(log_scale) * utilities
        if chosen is None:
            return float(np.mean(np.logaddexp(0.0, -scaled)))
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(chosen)), chosen]))
    result = minimize_scalar(loss, bounds=(-7.0, 7.0), method="bounded", options={"xatol": 1e-8})
    if not result.success or not math.isfinite(result.fun):
        raise RuntimeError(f"Probability calibration failed: {result.message}")
    return math.exp(result.x), {"status": "bounded", "log_scale": float(result.x),
                                "boundary": abs(result.x) > 6.99, "log_loss": float(result.fun)}


def _pair_items(encoder: Encoder, rows: Sequence[PairwiseObservation], entities: Mapping,
                reference: Reference) -> np.ndarray:
    if not rows:
        return np.zeros((0, 2, len(encoder.names)))
    return np.stack([[encoder.transform(entities[row.preferred_id], reference),
                      encoder.transform(entities[row.nonpreferred_id], reference)] for row in rows])


def _pair_margin(fit, items: np.ndarray) -> np.ndarray:
    return np.asarray(fit.score(items[:, 0])) - np.asarray(fit.score(items[:, 1]))


def _fit_config(settings: dict, items: np.ndarray | None = None) -> dict:
    config = dict(settings)
    for source, target in (("catboost_iterations", "iterations"), ("catboost_depth", "depth"),
                           ("catboost_thread_count", "thread_count")):
        if source in settings:
            config[target] = settings[source]
    if items is not None:
        config["pair_items"] = items
    return config


def _contributions(encoder: Encoder, differences: np.ndarray, theta) -> list[dict] | None:
    if theta is None:
        return None
    return [encoder.contributions(row, theta) for row in differences]


def _safe_reference(reference: Reference) -> dict:
    return {k: sorted(v) if isinstance(v, (set, frozenset)) else v for k, v in reference.values.items()}


def _settings_values(settings: dict, key: str, default: list) -> list:
    values = settings.get(key, default)
    if isinstance(values, str):
        values = [values]
    if not values:
        raise ValueError(f"{key} cannot be empty")
    return list(values)


def run_anu(settings: dict, seed: int, experiment: str, fold: int) -> dict:
    """Repeated user-grouped ANU reanalysis with independent calibration users."""
    started = time.monotonic()
    budgets = _budgets(settings)
    draws = int(settings.get("support_draws", 3))
    config = AnuExperimentConfig(support_sizes=tuple(budgets))
    dataset = load_anu_car_preferences(settings["paths"]["anu"], verify_file_checksums=True)
    eligible_keys, preference_rows, eligibility = _eligible_view(dataset, config)
    users = [u for u in dataset.users if u.experiment_id == experiment and (u.experiment_id, u.user_id) in eligible_keys]
    eligible_user_ids = {u.user_id for u in users}
    limit = settings.get("limits", {}).get("users")
    if limit is not None:
        users = sorted(users, key=lambda u: _digest(settings.get("cohort_seed", 20261010), "limit", u.experiment_id, u.user_id))[:int(limit)]
    if len(users) < 4:
        raise ValueError(f"ANU {experiment} needs at least four eligible users")
    identifiers = {u.user_id: f"anu:{experiment}:user:{u.user_id}" for u in users}
    folds = int(settings.get("folds", 5))
    if folds == 1:
        if fold != 0:
            raise ValueError("Single-fold smoke accepts only fold 0")
        roles = _four_roles(identifiers.values(), seed)
    else:
        if not 0 <= fold < folds:
            raise ValueError("Requested fold is outside configured outer folds")
        fold_map = deterministic_user_grouped_folds(users, n_folds=folds, seed=seed).as_mapping()
        test = [identifiers[u.user_id] for u in users if fold_map[(experiment, u.user_id)] == fold]
        roles = _four_roles(identifiers.values(), int(_digest(seed, fold)[:8], 16), test=test)
    selected = set(identifiers)
    rows = _as_observations(r for r in preference_rows if r.experiment_id == experiment and r.user_id in selected)
    by_role = {name: [r for r in rows if r.user_id in units] for name, units in roles.items()}
    views = {name: _support_views(by_role[name], max(budgets), draws, seed)
             for name in ("development", "calibration", "test")}
    records = tuple(r for r in anu_items_to_entity_records(dataset.items)
                    if r.entity_id.startswith(f"anu:{experiment}:item:"))
    specs = tuple(PropertySpec(spec.field, "numeric" if spec.kind == "numeric" else "categorical")
                  for spec in anu_item_predicate_specs() if any(r.get(spec.field) for r in records))
    basis = TypedBasis.fit(records, specs)
    q0 = basis.default_reference()
    entities = {r.entity_id: r for r in records}
    models = _settings_values(settings, "anu_models", settings.get("models", ["ST", "DT", "direct_ref", "spline_ref"]))
    if any("catboost" in str(model).lower() for model in models):
        raise ValueError("ANU CatBoost has no matched coefficient-anchor adaptation; choose explicit nonadaptive study instead")
    fits, predictions, metrics, audits = [], [], [], []
    for reference_mode in _settings_values(settings, "anu_references", ["catalogue"]):
        if reference_mode not in {"catalogue", "support"}:
            raise ValueError(f"Unsupported ANU reference {reference_mode}")
        train_rows, train_refs = list(by_role["train"]), [q0] * len(by_role["train"])
        if reference_mode == "support" and max(budgets) > 0:
            grouped = defaultdict(list)
            for row in train_rows:
                grouped[row.user_id].append(row)
            train_rows, train_refs = [], []
            ref_budget = min(int(settings.get("training_reference_budget", 5)), max(budgets))
            if ref_budget < 1:
                raise ValueError("Support-derived population reference needs positive training_reference_budget")
            for user, group in sorted(grouped.items()):
                ordered = sorted(group, key=lambda r: _digest(seed, "train_ref", user, r.group_id))
                ref = basis.reference_from_records([entities[r.preferred_id] for r in ordered[:ref_budget]], q0, "support-derived")
                train_rows.extend(ordered[ref_budget:])
                train_refs.extend([ref] * (len(ordered) - ref_budget))
        for completion, clipping, model in itertools.product(
                _settings_values(settings, "completion", ["default"]),
                _settings_values(settings, "clipping", ["clipped"]), models):
            cell_started = time.monotonic()
            common = {"dataset": f"anu-{experiment}", "seed": seed, "repeat": seed, "fold": fold,
                      "model": model, "completion": completion, "clipping": clipping, "reference": reference_mode,
                      "loss": settings.get("loss", "logistic"), "protocol": "archival_session_holdout",
                      "information_access": "reference-free control" if model in {"direct", "spline"} else "matched reference-aware"}
            # The ANU catalogue is known at deployment: label-free transductive item encoding.
            encoder = Encoder.fit(records, specs, model, completion=completion, clipping=clipping,
                                  knots=int(settings.get("knots", 5)))
            train_items = np.stack([[encoder.transform(entities[r.preferred_id], ref),
                                     encoder.transform(entities[r.nonpreferred_id], ref)]
                                    for r, ref in zip(train_rows, train_refs, strict=True)])
            train_D = train_items[:, 0] - train_items[:, 1]
            def ref_for(support):
                return (basis.reference_from_records([entities[r.preferred_id] for r in support], q0, "support-derived")
                        if reference_mode == "support" and support else q0)
            encoded_views = {}
            def evaluate(fit, role: str, budget: int, draw: int, penalty: dict, *, audit_stage: str):
                blocks = []
                for user, view in sorted(views[role].items()):
                    support = view["support"][draw][:budget]
                    cache_key = (role, user, budget, draw)
                    if cache_key not in encoded_views:
                        reference = ref_for(support)
                        support_items = _pair_items(encoder, support, entities, reference)
                        test_items = _pair_items(encoder, view["evaluation"], entities, reference)
                        encoded_views[cache_key] = (reference, support_items, test_items)
                    reference, support_items, test_items = encoded_views[cache_key]
                    if not settings.get("cache_features", True):
                        encoded_views.pop(cache_key)
                    adapted = fit
                    if support:
                        adapted = fit_pair(support_items[:, 0] - support_items[:, 1], penalty, encoder.groups,
                                           anchor=fit.theta, model=model, seed=seed,
                                           config=_fit_config(settings, support_items))
                        _audit_fit(adapted, audit_stage, audits, **common, unit_id=user, budget=budget, draw=draw,
                                   penalty=penalty)
                    blocks.append((user, view["evaluation"], _pair_margin(adapted, test_items),
                                   test_items[:, 0] - test_items[:, 1], adapted, reference))
                return blocks
            candidates = []
            for penalty in _grid(settings, model):
                fit = fit_pair(train_D, penalty, encoder.groups, model=model, seed=seed,
                               config=_fit_config(settings, train_items))
                _audit_fit(fit, "population_tuning", audits, **common, penalty=penalty)
                blocks = evaluate(fit, "development", 0, 0, penalty, audit_stage="development_adaptation")
                dev_loss = float(np.mean([np.mean(np.logaddexp(0.0, -block[2])) for block in blocks]))
                candidates.append((dev_loss, penalty, fit))
            dev_loss, penalty, population = min(candidates, key=lambda c: (round(c[0], 12), -c[1]["lam"], c[1]["gamma"]))
            mu_candidates = []
            for addition in _grid(settings, model, adaptation=True):
                adapted_penalty = {**penalty, **addition}
                losses = []
                for budget, draw in itertools.product([b for b in budgets if b > 0], range(draws)):
                    blocks = evaluate(population, "development", budget, draw, adapted_penalty,
                                      audit_stage="adaptation_tuning")
                    losses.extend(float(np.mean(np.logaddexp(0.0, -block[2]))) for block in blocks)
                mu_candidates.append((float(np.mean(losses)) if losses else dev_loss, adapted_penalty))
            adaptation_loss, penalty = min(mu_candidates, key=lambda c: (round(c[0], 12), -c[1]["mu"]))
            temperature, calibration = {}, {}
            for budget in budgets:
                margins = np.concatenate([block[2] for draw in range(draws)
                                          for block in evaluate(population, "calibration", budget, draw, penalty,
                                                                audit_stage="probability_calibration_adaptation")])
                temperature[budget], calibration[budget] = _temperature(margins)
            first = records[0]
            probes = []
            for second in records[1:min(len(records), 6)]:
                difference = encoder.transform(first, q0) - encoder.transform(second, q0)
                contributions = _contributions(encoder, difference[None, :], population.theta)
                probes.append({"row_id": f"{first.entity_id}|{second.entity_id}",
                               "contributions": contributions[0] if contributions else None,
                               "margin": float(_pair_margin(population, np.stack([[encoder.transform(first, q0),
                                                                                 encoder.transform(second, q0)]]))[0])})
            population_grids = {key: [candidate[1][key] for candidate in candidates] for key in ("lam", "gamma")}
            adaptation_grid = [candidate[1]["mu"] for candidate in mu_candidates]
            fits.append({**common, "selected_penalty": penalty, "theta": population.theta.tolist() if population.theta is not None else None,
                         "encoder": encoder.metadata(), "temperature": temperature, "calibration": calibration,
                         "development_log_loss": dev_loss, "adaptation_development_log_loss": adaptation_loss,
                         "boundary": {**{key: _boundary(penalty[key], population_grids[key]) for key in ("lam", "gamma")},
                                      "mu": _boundary(penalty["mu"], adaptation_grid)},
                         "canonical_probes": probes,
                         "probe_contributions": {p["row_id"]: p["contributions"] for p in probes},
                         "population_info": dict(population.info),
                         "training_pair_groups": [r.group_id for r in train_rows]})
            for budget, draw in itertools.product(budgets, range(draws)):
                blocks = evaluate(population, "test", budget, draw, penalty, audit_stage="test_support_adaptation")
                cell_predictions = []
                for user, observations, margins, differences, adapted, reference in blocks:
                    statistics = _pair_statistics(margins, temperature[budget])
                    contributions = _contributions(encoder, differences, adapted.theta)
                    for i, row in enumerate(observations):
                        cell_predictions.append({**common, "unit_id": user, "row_id": row.group_id,
                            "budget": budget, "draw": draw, "n": 1, "margin": float(margins[i]),
                            **{name: float(values[i]) for name, values in statistics.items()},
                            "contributions": contributions[i] if contributions else None,
                            "reference_values": _safe_reference(reference)})
                predictions.extend(cell_predictions)
                metrics.append({**common, "budget": budget, "draw": draw, "n": len(cell_predictions),
                                "units": len(blocks), "seconds": time.monotonic() - cell_started,
                                "selected_penalty": penalty,
                                **{name: float(np.mean([r[name] for r in cell_predictions]))
                                   for name in ("accuracy", "log_loss", "brier", "tie_rate")}})
    return {"metrics": metrics, "predictions": predictions, "fits": fits,
            "splits": {"unit": "experiment-scoped ANU session", "roles": roles,
                       "support": {role: {user: {"support_ids": view["support_ids"],
                                                 "evaluation_ids": view["evaluation_ids"]}
                                           for user, view in units.items()} for role, units in views.items()},
                       "query_policy": "same reserved queries across budgets, draws and methods"},
            "diagnostics": {"eligibility": eligibility, "fit_audit": audits,
                            "cohort": {"eligible_users": len(eligible_user_ids), "included_users": len(users),
                                       "label_free_cap_excluded": sorted(eligible_user_ids - selected),
                                       "cohort_seed": int(settings.get("cohort_seed", 20261010))},
                            "encoder_scope": "label-free fixed catalogue; transductive known-item evaluation",
                            "status": "development reanalysis of previously exposed archival data",
                            "reference_interpretation": "support-derived predictive reference is not an observed psychological reference",
                            "seconds": time.monotonic() - started},
            "task": {"suite": "anu", "experiment": experiment, "fold": fold, "seed": seed}}


def _choice_group(choice_set) -> str:
    return (f"respondent:{choice_set.respondent_id}" if choice_set.respondent_id is not None
            else f"choice-set:{choice_set.choice_set_id}")


def _design_proxy(choice_set, q0: Reference) -> Reference:
    """Unlabelled survey-design proxy, inseparable from first-listed presentation."""
    counts = Counter(str(a.item_features["type"]) for a in choice_set.alternatives)
    repeated = [value for value, count in counts.items() if count == 3]
    if len(repeated) != 1:
        raise ValueError(f"Choice set {choice_set.choice_set_id} lacks a unique threefold body-type design proxy")
    return Reference({**q0.values, "type": repeated[0]}, "survey-design-proxy")


def _choice_statistics(scores: np.ndarray, chosen: np.ndarray, temperature: float) -> dict[str, np.ndarray]:
    utilities = np.asarray(scores, dtype=float)
    scaled = temperature * utilities
    log_probability = scaled - logsumexp(scaled, axis=1, keepdims=True)
    probability = np.exp(log_probability)
    maxima = utilities.max(axis=1, keepdims=True)
    tied = np.abs(utilities - maxima) <= 1e-12
    rows = np.arange(len(chosen))
    accuracy = tied[rows, chosen].astype(float) / tied.sum(axis=1)
    observed = np.zeros_like(probability)
    observed[rows, chosen] = 1.0
    return {"accuracy": accuracy, "log_loss": -log_probability[rows, chosen],
            "brier": ((probability - observed) ** 2).sum(axis=1),
            "tie_rate": (tied.sum(axis=1) > 1).astype(float),
            "probability": probability[rows, chosen], "confidence": probability.max(axis=1)}


def run_mcfadden(settings: dict, seed: int, reference_mode: str) -> dict:
    """Grouped refit/tune/calibrate/test conditional choices, with fixed grid."""
    started = time.monotonic()
    if reference_mode not in {"catalogue", "design_proxy"}:
        raise ValueError(f"Unsupported McFadden reference {reference_mode}")
    load_options = {"download": False}
    if settings.get("mcfadden_respondent_id_column"):
        load_options["respondent_id_column"] = settings["mcfadden_respondent_id_column"]
    dataset = load_mcfadden_train(settings["paths"]["mcfadden"], **load_options)
    all_sets = list(dataset.choice_sets)
    presence = {cs.respondent_id is not None for cs in all_sets}
    if len(presence) > 1:
        raise ValueError("Respondent identifiers cannot be partially present")
    grouped = defaultdict(list)
    for cs in all_sets:
        grouped[_choice_group(cs)].append(cs)
    units = _ordered(grouped, int(settings.get("cohort_seed", 20261010)), "limit")
    limit = settings.get("limits", {}).get("mcfadden_sets")
    if limit is not None:
        retained, count = [], 0
        for unit in units:
            if count + len(grouped[unit]) > int(limit):
                break
            retained.append(unit)
            count += len(grouped[unit])
        units = retained
    roles = _four_roles(units, seed)
    parts = {role: sorted([cs for unit in members for cs in grouped[unit]], key=lambda cs: cs.choice_set_id)
             for role, members in roles.items()}
    training_records = [record for cs in parts["train"] for record in _records(cs)]
    specs = [PropertySpec(f, "numeric") for f in NUMERIC] + [PropertySpec(f, "categorical") for f in CATEGORICAL]
    basis = TypedBasis.fit(training_records, specs)
    q0 = basis.default_reference()
    references = {role: [q0 if reference_mode == "catalogue" else _design_proxy(cs, q0) for cs in choices]
                  for role, choices in parts.items()}
    chosen = {role: np.array([_chosen_index(cs) for cs in choices]) for role, choices in parts.items()}
    probe_sets = sorted([cs for unit in units for cs in grouped[unit]],
                        key=lambda cs: _digest(settings.get("cohort_seed", 20261010), "probe", cs.choice_set_id))[:5]
    models = _settings_values(settings, "mcfadden_models", settings.get("models", ["ST", "DT", "direct_ref", "spline_ref"]))
    predictions, metrics, fits, audits = [], [], [], []
    for completion, clipping, model in itertools.product(
            _settings_values(settings, "completion", ["default"]),
            _settings_values(settings, "clipping", ["clipped"]), models):
        cell_started = time.monotonic()
        common = {"dataset": "mcfadden", "seed": seed, "repeat": seed, "fold": 0, "model": model,
                  "completion": completion, "clipping": clipping, "reference": reference_mode, "budget": 0, "draw": 0,
                  "loss": "PairLogit" if model == "catboost_ref" else "multinomial_logit",
                  "protocol": "archival_choice_group_holdout",
                  "information_access": "reference-free control" if model in {"direct", "spline"} else "matched reference-aware"}
        encoder = Encoder.fit(training_records, specs, model, completion=completion, clipping=clipping,
                              knots=int(settings.get("knots", 5)),
                              references=[ref for cs, ref in zip(parts["train"], references["train"], strict=True)
                                          for _ in cs.alternatives])
        X = {role: np.stack([encoder.matrix(_records(cs), ref)
                             for cs, ref in zip(sets, references[role], strict=True)]) for role, sets in parts.items()}
        candidates = []
        for penalty in _grid(settings, model):
            fit = fit_multinomial(X["train"], chosen["train"], penalty, encoder.groups, model=model,
                                  seed=seed, config=_fit_config(settings))
            _audit_fit(fit, "population_tuning", audits, **common, penalty=penalty)
            score = np.asarray(fit.score(X["development"]))
            loss = float(np.mean(_choice_statistics(score, chosen["development"], 1.0)["log_loss"]))
            candidates.append((loss, penalty, fit))
        dev_loss, penalty, population = min(candidates, key=lambda c: (round(c[0], 12), -c[1]["lam"], c[1]["gamma"]))
        calibration_utilities = np.asarray(population.score(X["calibration"]))
        temperature, calibration = _temperature(calibration_utilities, chosen["calibration"])
        scores = np.asarray(population.score(X["test"]))
        statistics = _choice_statistics(scores, chosen["test"], temperature)
        test_differences = X["test"][np.arange(len(chosen["test"])), chosen["test"]] - X["test"].mean(axis=1)
        contributions = _contributions(encoder, test_differences, population.theta)
        for i, cs in enumerate(parts["test"]):
            predictions.append({**common, "unit_id": _choice_group(cs), "row_id": cs.choice_set_id, "n": 1,
                                **{name: float(values[i]) for name, values in statistics.items()},
                                "utilities": scores[i].tolist(), "chosen_index": int(chosen["test"][i]),
                                "contributions": contributions[i] if contributions else None,
                                "contribution_target": "chosen alternative minus mean offered utility",
                                "reference_values": _safe_reference(references["test"][i])})
        metrics.append({**common, "n": len(parts["test"]), "units": len(roles["test"]),
                        "seconds": time.monotonic() - cell_started, "selected_penalty": penalty,
                        **{name: float(np.mean(values)) for name, values in statistics.items() if name not in {"probability", "confidence"}}})
        grids = {key: [candidate[1][key] for candidate in candidates] for key in ("lam", "gamma")}
        probe_contributions = {}
        for cs in probe_sets:
            reference = q0 if reference_mode == "catalogue" else _design_proxy(cs, q0)
            matrix = encoder.matrix(_records(cs), reference)
            probe = _contributions(encoder, (matrix[0] - matrix[1])[None, :], population.theta)
            probe_contributions[f"mcf:{cs.choice_set_id}:1|2"] = probe[0] if probe else None
        fits.append({**common, "selected_penalty": penalty,
                     "theta": population.theta.tolist() if population.theta is not None else None,
                     "encoder": encoder.metadata(), "temperature": temperature, "calibration": calibration,
                     "development_log_loss": dev_loss, "population_info": dict(population.info),
                     "probe_contributions": probe_contributions,
                     "boundary": {key: _boundary(penalty[key], grids[key]) for key in ("lam", "gamma")}})
    proxy_audit = None
    if reference_mode == "design_proxy":
        proxy_audit = {"first_listed_matches_proxy": all(
            sorted(cs.alternatives, key=lambda a: a.alternative_id)[0].item_features["type"] == ref.get("type")
            for role in parts for cs, ref in zip(parts[role], references[role], strict=True)),
            "interpretation": "survey-design proxy; body-type-reference and presentation effects remain confounded"}
    return {"metrics": metrics, "predictions": predictions, "fits": fits,
            "splits": {"unit": "explicit respondent" if True in presence else "archived choice set",
                       "roles": roles, "choice_sets": {role: [cs.choice_set_id for cs in choices] for role, choices in parts.items()},
                       "respondent_limitation": None if True in presence else
                       "Bundled schema has no verified panel respondent ID; choice-set grouping cannot audit unknown respondent overlap."},
            "diagnostics": {"fit_audit": audits, "provenance": dict(dataset.provenance),
                            "cohort": {"available_choice_sets": len(all_sets),
                                       "included_choice_sets": sum(len(grouped[unit]) for unit in units),
                                       "cohort_seed": int(settings.get("cohort_seed", 20261010))},
                            "design_proxy": proxy_audit, "encoder_scope": "train alternatives only; no held-out item preprocessing",
                            "status": "development reanalysis of previously exposed archival data", "seconds": time.monotonic() - started},
            "task": {"suite": "mcfadden", "seed": seed, "reference": reference_mode}}
