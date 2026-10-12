"""McFadden–Train vehicle-choice study.

Same grouped 60/20/20 split as the August run (seed 20260825). Every family is a
conditional logit on its own feature map, so differences come from the
representation, not the loss.
"""

from __future__ import annotations

import numpy as np

from rgsmw_next.choice_baselines import fit_latent_class_conditional_logit, latent_class_log_utilities
from rgsmw_next.mcfadden_train import deterministic_grouped_choice_split, evaluate_choice_scores, load_mcfadden_train
from rgsmw_next.types import EntityRecord

from .basis import PropertySpec, TypedBasis
from .learners import Penalty, fit_choice

SEED = 20260825
NUMERIC = ("price", "range", "acc", "speed", "pollution", "space", "cost", "station")
CATEGORICAL = ("type", "fuel", "size")
DEMOGRAPHICS = ("college", "hsg2", "coml5")
LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
GAMMA_GRID = (0.0, 1e-3, 1e-2)
LATENT_CLASSES = (2, 3)
LATENT_L2 = (1e-3, 1e-2, 1e-1)
BOOTSTRAP_FITS = 200
BOOTSTRAP_DRAWS = 2_000
FAMILIES = ("Direct-CL", "Direct+demo-CL", "S0-CL", "D0-CL", "D0ctx-CL")


def _records(choice_set) -> list[EntityRecord]:
    ordered = sorted(choice_set.alternatives, key=lambda a: a.alternative_id)
    return [
        EntityRecord(
            entity_id=f"mcf:{choice_set.choice_set_id}:{a.alternative_id}",
            uri=f"urn:mcf:{choice_set.choice_set_id}:{a.alternative_id}",
            values={
                **{f: (float(a.item_features[f]),) for f in NUMERIC},
                **{f: (str(a.item_features[f]),) for f in CATEGORICAL},
            },
        )
        for a in ordered
    ]


def _chosen_index(choice_set) -> int:
    ordered = sorted(choice_set.alternatives, key=lambda a: a.alternative_id)
    return next(i for i, a in enumerate(ordered) if a.chosen)


def _tensor(basis: TypedBasis, sets, family: str) -> np.ndarray:
    q0 = basis.default_reference()
    blocks = []
    for cs in sets:
        records = _records(cs)
        if family in ("Direct-CL", "Direct+demo-CL"):
            x = basis.matrix(records, None, "direct")
            if family == "Direct+demo-CL":
                demo = [float(cs.shared_features[k]) for k in DEMOGRAPHICS]
                x = np.hstack([x] + [x * v for v in demo])
        elif family == "D0ctx-CL":
            x = basis.matrix(records, basis.reference_from_records(records, q0, "choice_set_mean"), "D0")
        else:
            x = basis.matrix(records, q0, family.split("-")[0])
        blocks.append(x)
    return np.stack(blocks)


def _groups(basis: TypedBasis, family: str) -> list[tuple[int, int]]:
    if family == "Direct+demo-CL":
        d = basis.dimension("direct")
        return [(g.start + k * d, g.stop + k * d) for k in range(1 + len(DEMOGRAPHICS)) for g in basis.groups("direct")]
    name = {"Direct-CL": "direct", "S0-CL": "S0", "D0-CL": "D0", "D0ctx-CL": "D0"}[family]
    return [(g.start, g.stop) for g in basis.groups(name)]


def _log_loss(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    u = X @ theta
    top = u.max(axis=1, keepdims=True)
    log_norm = top[:, 0] + np.log(np.exp(u - top).sum(axis=1))
    return float(np.mean(log_norm - u[np.arange(len(y)), y]))


def _scores(sets, utilities: np.ndarray) -> dict[str, dict[str, float]]:
    out = {}
    for cs, row in zip(sets, utilities, strict=True):
        ordered = sorted(cs.alternatives, key=lambda a: a.alternative_id)
        out[cs.choice_set_id] = {a.alternative_id: float(v) for a, v in zip(ordered, row, strict=True)}
    return out


def _paired(first: list[dict], second: list[dict], metric: str, seed: int) -> dict[str, float]:
    a = np.array([r[metric] for r in first])
    b = np.array([r[metric] for r in second])
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(BOOTSTRAP_DRAWS, diff.size))
    boots = diff[idx].mean(axis=1)
    return {
        "difference": float(diff.mean()),
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
        "choice_sets": int(diff.size),
    }


def run(root: str) -> dict[str, object]:
    dataset = load_mcfadden_train(root, download=False)
    split = deterministic_grouped_choice_split(dataset.choice_sets, seed=SEED, train_fraction=0.6, dev_fraction=0.2)
    by_id = {cs.choice_set_id: cs for cs in dataset.choice_sets}
    parts = {name: [by_id[i] for i in sorted(split[name], key=lambda s: int(s))] for name in ("train", "dev", "test")}
    basis = TypedBasis.fit([r for cs in parts["train"] for r in _records(cs)], [PropertySpec(f, "numeric") for f in NUMERIC] + [PropertySpec(f, "categorical") for f in CATEGORICAL])
    y = {name: np.array([_chosen_index(cs) for cs in sets]) for name, sets in parts.items()}

    fitted: dict[str, dict] = {}
    evaluations: dict[str, dict] = {}
    for family in FAMILIES:
        X = {name: _tensor(basis, sets, family) for name, sets in parts.items()}
        groups = _groups(basis, family)
        candidates = []
        for lam in LAMBDA_GRID:
            for gamma in GAMMA_GRID:
                theta, info = fit_choice(X["train"], y["train"], Penalty(lam, gamma), groups)
                candidates.append((_log_loss(X["dev"], y["dev"], theta), lam, gamma, theta, info))
        dev_loss, lam, gamma, theta, info = min(candidates, key=lambda c: (round(c[0], 12), -c[1], c[2]))
        evaluation = evaluate_choice_scores(parts["test"], _scores(parts["test"], X["test"] @ theta))
        fitted[family] = {
            "lam": lam,
            "gamma": gamma,
            "dev_log_loss": dev_loss,
            "converged": bool(info.get("converged")),
            "boundary_lam": lam in (LAMBDA_GRID[0], LAMBDA_GRID[-1]),
            "theta": theta.tolist(),
            "dimension": int(theta.size),
        }
        evaluations[family] = evaluation
        if family in ("D0-CL", "D0ctx-CL"):
            # Bootstrap refits over training choice sets for gain/loss slope intervals.
            rng = np.random.default_rng(SEED + (1 if family == "D0ctx-CL" else 0))
            slopes = []
            for _ in range(BOOTSTRAP_FITS):
                idx = rng.integers(0, len(y["train"]), size=len(y["train"]))
                boot, _ = fit_choice(X["train"][idx], y["train"][idx], Penalty(lam, gamma), groups)
                slopes.append(basis.numeric_slopes("D0", boot))
            point = basis.numeric_slopes("D0", theta)
            fitted[family]["slopes"] = {
                field: {
                    **point[field],
                    **{
                        f"{key}_ci": [
                            float(np.quantile([s[field][key] for s in slopes], 0.025)),
                            float(np.quantile([s[field][key] for s in slopes], 0.975)),
                        ]
                        for key in ("below_reference_slope", "above_reference_slope")
                    },
                    "asymmetry_above_minus_below": point[field]["above_reference_slope"] - point[field]["below_reference_slope"],
                    "asymmetry_ci": [
                        float(np.quantile([s[field]["above_reference_slope"] - s[field]["below_reference_slope"] for s in slopes], 0.025)),
                        float(np.quantile([s[field]["above_reference_slope"] - s[field]["below_reference_slope"] for s in slopes], 0.975)),
                    ],
                }
                for field in point
            }

    # Latent-class conditional logit on direct features (existing implementation).
    X = {name: _tensor(basis, sets, "Direct-CL") for name, sets in parts.items()}
    lc_candidates = []
    for classes in LATENT_CLASSES:
        for l2 in LATENT_L2:
            fit = fit_latent_class_conditional_logit(list(X["train"]), list(y["train"]), class_count=classes, l2=l2)
            dev_logp = latent_class_log_utilities(fit, list(X["dev"]))
            dev_loss = float(-np.mean([row[k] - np.logaddexp.reduce(row) for row, k in zip(dev_logp, y["dev"], strict=True)]))
            lc_candidates.append((dev_loss, classes, l2, fit))
    dev_loss, classes, l2, lc_fit = min(lc_candidates, key=lambda c: (round(c[0], 12), c[1], -c[2]))
    test_logp = np.array(latent_class_log_utilities(lc_fit, list(X["test"])))
    evaluations["LatentClass-CL"] = evaluate_choice_scores(parts["test"], _scores(parts["test"], test_logp))
    fitted["LatentClass-CL"] = {"classes": classes, "l2": l2, "dev_log_loss": dev_loss}

    summary = {name: ev["metrics"] for name, ev in evaluations.items()}
    per_set = {name: ev["per_choice_set"] for name, ev in evaluations.items()}
    contrasts = {}
    pairs = {
        "P4_D0_minus_S0": ("D0-CL", "S0-CL"),
        "P5_D0_minus_Direct": ("D0-CL", "Direct-CL"),
        "P6_D0ctx_minus_D0": ("D0ctx-CL", "D0-CL"),
        "D0_minus_LatentClass": ("D0-CL", "LatentClass-CL"),
        "D0ctx_minus_Direct": ("D0ctx-CL", "Direct-CL"),
        "D0_minus_DirectDemo": ("D0-CL", "Direct+demo-CL"),
    }
    for k, (name, (a, b)) in enumerate(pairs.items()):
        contrasts[name] = {
            metric: _paired(per_set[a], per_set[b], metric, SEED + 100 * k + j)
            for j, metric in enumerate(("top1_accuracy", "log_loss", "ndcg_at_6"))
        }

    # Worked decomposition for three test choice sets (D0-CL).
    theta = np.asarray(fitted["D0-CL"]["theta"])
    q0 = basis.default_reference()
    examples = []
    for cs in parts["test"][:3]:
        records = _records(cs)
        examples.append(
            {
                "choice_set_id": cs.choice_set_id,
                "chosen_index": _chosen_index(cs),
                "alternatives": [
                    {"values": {k: v[0] for k, v in r.values.items()}, "contributions": basis.contributions(r, q0, "D0", theta)}
                    for r in records
                ],
            }
        )
    return {
        "summary": {"test_metrics": summary, "fitted": fitted, "basis": basis.manifest(), "test_choice_sets": len(parts["test"])},
        "contrasts": contrasts,
        "examples": {"reference": {k: v for k, v in q0.values.items()}, "choice_sets": examples},
    }
