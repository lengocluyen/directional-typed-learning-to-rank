from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping, Sequence


Scalar = str | int | float | date | tuple[float, float]


@dataclass(frozen=True)
class PredicateSpec:
    legacy: str
    field: str
    kind: str
    unit: str | None = None
    metric: bool = True
    ordinal: Mapping[str, int] = field(default_factory=dict)
    threshold: float | None = None
    source_note: str | None = None


@dataclass
class EntityRecord:
    entity_id: str
    uri: str
    values: dict[str, tuple[Scalar, ...]]
    raw_values: dict[str, tuple[str, ...]] = field(default_factory=dict)
    missing: dict[str, str] = field(default_factory=dict)
    source_file: str = ""
    source_sha256: str = ""

    def get(self, field_name: str) -> tuple[Scalar, ...]:
        return self.values.get(field_name, ())


@dataclass(frozen=True)
class Preference:
    user_id: str
    query_id: str
    preferred_id: str
    nonpreferred_id: str
    delta: tuple[float, ...]


@dataclass(frozen=True)
class RankedEntity:
    entity_id: str
    distance: float
    contributions: tuple[float, ...]


@dataclass(frozen=True)
class FitResult:
    weights: tuple[float, ...]
    objective: float
    status: str
    solver: str
    n_preferences: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


def require_dimension(weights: Sequence[float], expected: int) -> None:
    if len(weights) != expected:
        raise ValueError(f"Expected {expected} weights, received {len(weights)}")

