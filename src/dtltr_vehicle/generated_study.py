"""Generated-rule stress tests on the open ADEME RDF catalogue.

Labels are GENERATED from declared rules, not human choices. Each query is a
reference vehicle q with two candidates. Vehicles are split per seed into a
training pool (70 %) and a test pool (30 %), so every test query and candidate is
a vehicle unseen during training. Clean-label accuracy is measured against the
noiseless rule; models are trained on logistic-noise labels.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Callable, Mapping

import numpy as np

from rgsmw_next.distance import FittedMetric, robust_scale
from rgsmw_next.learning import GlobalSimplexLearner
from rgsmw_next.types import EntityRecord, PredicateSpec, Preference

from .basis import PropertySpec, TypedBasis
from .learners import Penalty, fit_pairwise
from .open_catalogue import PROPERTIES, SEGMENTS

RULES = (
    "additive_similarity",
    "sparse_similarity",
    "signed_linear",
    "gain_loss",
    "offset_ideal_point",
    "lexicographic",
    "informative_missingness",
    "query_interaction",
)
SEEDS = (11, 22, 33, 44, 55)
N_TRAIN, N_VALID, N_TEST = 3000, 1000, 2000
LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
LOSS_AVERSION = 2.25
NOISE_FRACTION = 0.25
# desirable direction of numeric attributes used by directional rules (+1 higher is better)
DIRECTION = {"price_eur": -1, "max_power_kw": 1, "co2_g_km": -1, "fuel_consumption_l100km": -1, "electric_range_km": 1, "mass_kg": -1}


def load_records(path: str) -> list[EntityRecord]:
    rows = json.loads(open(path, encoding="utf-8").read())
    records = []
    for row in rows:
        values = {}
        for field, value in row["values"].items():
            if value is None:
                continue
            values[field] = tuple(value) if isinstance(value, list) else (value,)
        records.append(EntityRecord(entity_id=row["entity_id"], uri=f"urn:{row['entity_id']}", values=values))
    return records


def property_specs() -> list[PropertySpec]:
    return [
        PropertySpec(f, kind, SEGMENTS if kind == "ordinal" else ()) for f, (_, kind) in PROPERTIES.items()
    ]


def predicate_specs() -> list[PredicateSpec]:
    return [
        PredicateSpec(f, f, kind, ordinal={level: i for i, level in enumerate(SEGMENTS)} if kind == "ordinal" else {})
        for f, (_, kind) in PROPERTIES.items()
    ]


def _hash_unit(*parts: object) -> float:
    return int(hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()[:12], 16) / 16**12


def _num(record: EntityRecord, field: str) -> float | None:
    values = record.get(field)
    return float(values[0]) if values else None


class Rules:
    def __init__(self, records: list[EntityRecord], metric: FittedMetric, seed: int):
        self.metric = metric
        self.scale = {f: robust_scale([v for v in (_num(r, f) for r in records) if v is not None])[0] for f in DIRECTION}
        rng = np.random.default_rng(seed)
        k = len(metric.predicates)
        self.dense = rng.dirichlet(np.ones(k))
        sparse = np.zeros(k)
        sparse[rng.choice(k, size=3, replace=False)] = rng.dirichlet(np.ones(3))
        self.sparse = sparse
        self.signed = {f: DIRECTION[f] * float(rng.uniform(0.5, 1.5)) for f in DIRECTION}
        self.weights = {f: float(rng.uniform(0.5, 1.5)) for f in DIRECTION}
        self.offset = {f: DIRECTION[f] * 0.5 for f in DIRECTION}

    def z(self, record: EntityRecord, field: str) -> float | None:
        value = _num(record, field)
        return None if value is None else value / self.scale[field]

    def utility(self, rule: str, q: EntityRecord, e: EntityRecord) -> float:
        if rule == "additive_similarity":
            return -float(self.dense @ self.metric.feature_vector(q, e))
        if rule == "sparse_similarity":
            return -float(self.sparse @ self.metric.feature_vector(q, e))
        if rule == "signed_linear":
            return float(sum(w * (self.z(e, f) or 0.0) for f, w in self.signed.items()))
        if rule == "gain_loss":
            total = 0.0
            for f, sign in DIRECTION.items():
                zq, ze = self.z(q, f), self.z(e, f)
                if zq is None or ze is None:
                    continue
                gain = sign * (ze - zq)
                total += self.weights[f] * (max(gain, 0.0) - LOSS_AVERSION * max(-gain, 0.0))
            return total
        if rule == "offset_ideal_point":
            total = 0.0
            for f in DIRECTION:
                zq, ze = self.z(q, f), self.z(e, f)
                if zq is None or ze is None:
                    total -= 1.0
                    continue
                total -= self.weights[f] * abs(ze - (zq + self.offset[f]))
            return total
        if rule == "lexicographic":
            match = float(q.get("powertrain") == e.get("powertrain"))
            return 10.0 * match - (self.z(e, "price_eur") or 0.0) / 10.0
        if rule == "informative_missingness":
            plug_in = float(bool(e.get("electric_range_km")))
            return 1.5 * plug_in - abs((self.z(e, "price_eur") or 0.0) - (self.z(q, "price_eur") or 0.0))
        if rule == "query_interaction":
            premium = 1.0 if SEGMENTS.index(q.get("segment")[0]) >= 4 else -1.0
            return premium * (self.z(e, "price_eur") or 0.0) + 0.5 * (self.z(e, "max_power_kw") or 0.0)
        raise ValueError(rule)


def _sample(pool: list[EntityRecord], n: int, rng: np.random.Generator) -> list[tuple[int, int, int]]:
    out = []
    while len(out) < n:
        q, a, b = rng.choice(len(pool), size=3, replace=False)
        out.append((int(q), int(a), int(b)))
    return out


def run_rule(records_path: str, rule: str, seed: int) -> dict[str, object]:
    records = load_records(records_path)
    train_pool = [r for r in records if _hash_unit(seed, "pool", r.entity_id) < 0.7]
    test_pool = [r for r in records if _hash_unit(seed, "pool", r.entity_id) >= 0.7]
    # Label-free fitting on the training pool only.
    metric = FittedMetric.fit(train_pool, predicate_specs())
    basis = TypedBasis.fit(train_pool, property_specs())
    rules = Rules(records, metric, seed)
    rng = np.random.default_rng(seed)
    splits = {
        "train": (train_pool, _sample(train_pool, N_TRAIN, rng)),
        "valid": (train_pool, _sample(train_pool, N_VALID, rng)),
        "test": (test_pool, _sample(test_pool, N_TEST, rng)),
    }
    diffs = {}
    for name, (pool, queries) in splits.items():
        diffs[name] = np.array([rules.utility(rule, pool[q], pool[a]) - rules.utility(rule, pool[q], pool[b]) for q, a, b in queries])
    noise_scale = NOISE_FRACTION * float(np.std(diffs["train"])) or 1e-9
    noisy = {name: (d + rng.logistic(0.0, noise_scale, size=d.size) > 0) for name, d in diffs.items()}

    def pair_rows(name: str, feature: Callable[[EntityRecord, EntityRecord], np.ndarray]) -> np.ndarray:
        pool, queries = splits[name]
        rows = []
        for (q, a, b), a_wins in zip(queries, noisy[name], strict=True):
            delta = feature(pool[q], pool[a]) - feature(pool[q], pool[b])
            rows.append(delta if a_wins else -delta)
        return np.array(rows)

    def clean_accuracy(name: str, feature: Callable, theta: np.ndarray) -> float:
        pool, queries = splits[name]
        margins = np.array([(feature(pool[q], pool[a]) - feature(pool[q], pool[b])) @ theta for q, a, b in queries])
        truth = diffs[name]
        keep = np.abs(truth) > 1e-12
        correct = np.where(np.abs(margins) <= 1e-12, 0.5, (np.sign(margins) == np.sign(truth)).astype(float))
        return float(np.mean(correct[keep]))

    results: dict[str, object] = {"rule": rule, "seed": seed, "train_vehicles": len(train_pool), "test_vehicles": len(test_pool),
                                  "test_ties_excluded": int(np.sum(np.abs(diffs["test"]) <= 1e-12)), "models": {}}
    families: dict[str, Callable[[EntityRecord, EntityRecord], np.ndarray]] = {
        "S0": lambda q, e: basis.transform(e, basis.reference_from_records([q], basis.default_reference(), "query"), "S0"),
        "D0": lambda q, e: basis.transform(e, basis.reference_from_records([q], basis.default_reference(), "query"), "D0"),
        "Direct": lambda q, e: basis.transform(e, None, "direct"),
    }
    cache: dict = {}

    def cached(name: str, fn: Callable) -> Callable:
        def inner(q: EntityRecord, e: EntityRecord) -> np.ndarray:
            key = (name, q.entity_id, e.entity_id)
            if key not in cache:
                cache[key] = fn(q, e)
            return cache[key]

        return inner

    for name, fn in families.items():
        feature = cached(name, fn)
        family = "direct" if name == "Direct" else name
        groups = [(g.start, g.stop) for g in basis.groups(family)]
        train_rows, valid_rows = pair_rows("train", feature), pair_rows("valid", feature)
        best = None
        for lam in LAMBDA_GRID:
            theta, _ = fit_pairwise(train_rows, "logistic", Penalty(lam), groups)
            loss = float(np.mean(np.logaddexp(0.0, -valid_rows @ theta)))
            if best is None or loss < best[0] - 1e-12:
                best = (loss, lam, theta)
        _, lam, theta = best
        entry = {"lam": lam, "clean_accuracy": clean_accuracy("test", feature, theta)}
        if name == "D0" and rule == "gain_loss":
            entry["slopes"] = {f: v for f, v in basis.numeric_slopes("D0", theta).items() if f in DIRECTION}
            entry["generating_weights"] = rules.weights
            entry["generating_scales"] = rules.scale
        results["models"][name] = entry  # type: ignore[index]

    # Nonnegative typed-distance simplex (August baseline): distance to q, smaller is better.
    dist = cached("dist", lambda q, e: metric.feature_vector(q, e))
    prefs = [Preference("u", f"t{i}", "a", "b", tuple(float(x) for x in -row)) for i, row in enumerate(pair_rows("train", dist))]
    simplex = GlobalSimplexLearner(dimension=len(metric.predicates), c=5.0, l2=1e-2, margin=0.02).fit(prefs)
    results["models"]["Simplex"] = {"clean_accuracy": clean_accuracy("test", dist, -np.asarray(simplex.weights))}  # type: ignore[index]

    # CatBoost PairLogit on D0 query-relative features plus absolute candidate features.
    from catboost import CatBoostRanker, Pool

    both = cached("both", lambda q, e: np.concatenate([families["D0"](q, e), families["Direct"](q, e)]))

    def pool_for(name: str) -> Pool:
        pool, queries = splits[name]
        x, pairs, group = [], [], []
        for i, ((q, a, b), a_wins) in enumerate(zip(queries, noisy[name], strict=True)):
            x.extend([both(pool[q], pool[a]), both(pool[q], pool[b])])
            group.extend([i, i])
            pairs.append([2 * i, 2 * i + 1] if a_wins else [2 * i + 1, 2 * i])
        return Pool(np.array(x), group_id=group, pairs=pairs)

    model = CatBoostRanker(loss_function="PairLogit", iterations=600, depth=6, learning_rate=0.05, random_seed=seed, verbose=False, thread_count=1)
    model.fit(pool_for("train"), eval_set=pool_for("valid"), use_best_model=True)
    pool, queries = splits["test"]
    scores = model.predict(np.array([both(pool[q], pool[c]) for q, a, b in queries for c in (a, b)]))
    margins = scores[0::2] - scores[1::2]
    truth = diffs["test"]
    keep = np.abs(truth) > 1e-12
    correct = np.where(np.abs(margins) <= 1e-12, 0.5, (np.sign(margins) == np.sign(truth)).astype(float))
    results["models"]["CatBoost"] = {"clean_accuracy": float(np.mean(correct[keep])), "best_iteration": int(model.get_best_iteration())}  # type: ignore[index]
    return results
