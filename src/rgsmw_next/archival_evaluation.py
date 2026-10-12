from __future__ import annotations

from collections import defaultdict
from typing import Hashable, Iterable, Mapping, Sequence, TypeVar

import numpy as np


T = TypeVar("T")


def deterministic_group_folds(
    group_ids: Iterable[str],
    *,
    n_folds: int = 5,
    seed: int = 20260825,
) -> dict[str, int]:
    groups = np.asarray(sorted(set(group_ids)), dtype=object)
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    if groups.size < n_folds:
        raise ValueError("The number of groups must be at least n_folds")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    return {str(group): int(index % n_folds) for index, group in enumerate(groups)}


def nested_support_split(
    rows: Sequence[T],
    support_sizes: Sequence[int],
    *,
    seed: int,
    stable_key: callable,
) -> tuple[dict[int, list[T]], list[T]]:
    """Create nested support prefixes and one untouched common evaluation set."""

    sizes = tuple(sorted(set(int(value) for value in support_sizes)))
    if not sizes or sizes[0] < 0:
        raise ValueError("support_sizes must contain non-negative integers")
    maximum = sizes[-1]
    ordered = sorted(rows, key=stable_key)
    if len(ordered) <= maximum:
        raise ValueError(
            f"Need more than {maximum} rows to retain a common evaluation set; "
            f"received {len(ordered)}"
        )
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(ordered))
    shuffled = [ordered[int(index)] for index in permutation]
    support = {size: list(shuffled[:size]) for size in sizes}
    evaluation = list(shuffled[maximum:])
    for smaller, larger in zip(sizes, sizes[1:]):
        if support[smaller] != support[larger][:smaller]:
            raise RuntimeError("Internal error: support sets are not nested")
    return support, evaluation


def cluster_bootstrap_mean(
    values: Mapping[Hashable, float],
    *,
    seed: int = 20260825,
    replicates: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, float]:
    if not values:
        raise ValueError("At least one cluster is required")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    keys = sorted(values, key=str)
    observed = np.asarray([float(values[key]) for key in keys], dtype=float)
    if not np.all(np.isfinite(observed)):
        raise ValueError("Bootstrap values must be finite")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(keys), size=(replicates, len(keys)))
    means = np.mean(observed[indices], axis=1)
    alpha = 1.0 - confidence
    return {
        "mean": float(np.mean(observed)),
        "sample_sd": float(np.std(observed, ddof=1)) if observed.size > 1 else 0.0,
        "ci_low": float(np.quantile(means, alpha / 2.0)),
        "ci_high": float(np.quantile(means, 1.0 - alpha / 2.0)),
        "clusters": int(observed.size),
        "bootstrap_replicates": int(replicates),
        "confidence": float(confidence),
    }


def paired_cluster_bootstrap(
    first: Mapping[Hashable, float],
    second: Mapping[Hashable, float],
    *,
    seed: int = 20260825,
    replicates: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, float]:
    if set(first) != set(second):
        raise ValueError("Paired systems must contain exactly the same clusters")
    differences = {key: float(first[key]) - float(second[key]) for key in first}
    summary = cluster_bootstrap_mean(
        differences,
        seed=seed,
        replicates=replicates,
        confidence=confidence,
    )
    return {"direction": "first_minus_second", **summary}


def aggregate_repeated_cluster_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    cluster_key: str,
    metric_keys: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Average repeated support draws within cluster before inference."""

    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        cluster = str(row[cluster_key])
        for metric in metric_keys:
            value = float(row[metric])
            if np.isfinite(value):
                grouped[cluster][metric].append(value)
    output: dict[str, dict[str, float]] = {}
    for cluster in sorted(grouped):
        output[cluster] = {
            metric: float(np.mean(values))
            for metric, values in grouped[cluster].items()
            if values
        }
    return output


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down family-wise adjustment without external dependencies."""

    ordered = sorted(
        ((name, float(value)) for name, value in p_values.items()),
        key=lambda item: (item[1], item[0]),
    )
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        if not 0.0 <= value <= 1.0:
            raise ValueError("p-values must be in [0, 1]")
        running = max(running, (total - index) * value)
        adjusted[name] = min(1.0, running)
    return adjusted

