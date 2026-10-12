"""Archival McFadden--Train stated vehicle-choice data adapter.

The module deliberately depends only on the Python standard library and NumPy.
It reads the compact ``car.csv.gz`` file bundled by Choice-Learn without
installing Choice-Learn, pandas, or TensorFlow.  The source is represented as
choice sets rather than as independent vehicle records: every archived row
contains six offered alternatives and exactly one chosen alternative.

The bundled CSV has a ``rownames`` column but no explicit respondent identifier
column.  Consequently, ``respondent_id`` is ``None`` by default.  Callers may
name an explicit respondent column when adapting another, independently
verified copy of the source; the parser never silently treats an R row name as
a longitudinal respondent identifier.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import math
import re
import shutil
import ssl
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np

try:
    import certifi
except ModuleNotFoundError:  # A verified offline cache does not need certifi.
    certifi = None  # type: ignore[assignment]


ALTERNATIVE_IDS = tuple(str(index) for index in range(1, 7))
SHARED_FEATURE_NAMES = ("college", "hsg2", "coml5")
CATEGORICAL_ITEM_FEATURE_NAMES = ("type", "fuel")
NUMERIC_ITEM_FEATURE_NAMES = (
    "price",
    "range",
    "acc",
    "speed",
    "pollution",
    "size",
    "space",
    "cost",
    "station",
)
ITEM_FEATURE_NAMES = CATEGORICAL_ITEM_FEATURE_NAMES + NUMERIC_ITEM_FEATURE_NAMES
CHOICE_PATTERN = re.compile(r"choice([1-6])\Z")
ARCHIVAL_CHOICE_SET_COUNT = 4_654
FEATURE_SEMANTICS: Mapping[str, str] = {
    "college": "Binary indicator for college education.",
    "hsg2": "Binary indicator for household size greater than two.",
    "coml5": "Binary indicator for a commute shorter than five miles per day.",
    "type": "Vehicle body type (six archived categories).",
    "fuel": "Fuel/propulsion type (gasoline, methanol, CNG, or electric).",
    "price": "Vehicle price divided by the logarithm of household income.",
    "range": "Driving range between refuelling/recharging in the archived scale.",
    "acc": "Acceleration time to 30 mph in the archived scale.",
    "speed": "Highest attainable speed in the archived scale.",
    "pollution": "Tailpipe emissions as a fraction of a new gasoline vehicle.",
    "size": "Ordinal size: 0 mini, 1 subcompact, 2 compact, 3 mid-size/large.",
    "space": "Luggage space as a fraction of a comparable gasoline vehicle.",
    "cost": "Operating cost per mile in the archived scale.",
    "station": "Fraction of stations able to refuel/recharge the alternative.",
}


def _wide_feature_columns() -> tuple[str, ...]:
    return tuple(
        f"{feature}{alternative_id}"
        for feature in ITEM_FEATURE_NAMES
        for alternative_id in ALTERNATIVE_IDS
    )


CHOICE_LEARN_COLUMNS = (
    "rownames",
    "choice",
    *SHARED_FEATURE_NAMES,
    *_wide_feature_columns(),
)


@dataclass(frozen=True)
class VehicleChoiceSource:
    """Checksum-pinned packaging source and bibliographic provenance."""

    name: str
    filename: str
    url: str
    sha256: str
    repository: str
    commit: str
    repository_path: str
    landing_page: str
    paper_doi: str
    software_license: str
    dataset_license_note: str


DEFAULT_SOURCE = VehicleChoiceSource(
    name="McFadden--Train stated car preferences (Choice-Learn packaging)",
    filename="car.csv.gz",
    url=(
        "https://raw.githubusercontent.com/artefactory/choice-learn/"
        "e45b330a42ac2c3566ce9cf7cd9acf95b79df761/"
        "choice_learn/datasets/data/car.csv.gz"
    ),
    sha256="cbdab01357dab84e19f70c2761a6b6421f73bde530987007a4dbdfb642f0a554",
    repository="https://github.com/artefactory/choice-learn",
    commit="e45b330a42ac2c3566ce9cf7cd9acf95b79df761",
    repository_path="choice_learn/datasets/data/car.csv.gz",
    landing_page=(
        "https://artefactory.github.io/choice-learn/references/datasets/"
        "references_base/#choice_learn.datasets.base.load_car_preferences"
    ),
    paper_doi=(
        "10.1002/1099-1255(200009/10)15:5<447::AID-JAE570>3.0.CO;2-1"
    ),
    software_license="MIT (Choice-Learn repository)",
    dataset_license_note=(
        "The Choice-Learn repository is MIT-licensed, and the Journal of Applied "
        "Econometrics archive is listed without a download restriction. A separate "
        "dataset-specific redistribution licence was not found in the bundled file; "
        "verify redistribution terms before republishing the archive."
    ),
)


ChoiceValue = str | int | float


@dataclass(frozen=True)
class VehicleChoiceAlternative:
    """One offered alternative in a six-item stated-choice set."""

    alternative_id: str
    chosen: bool
    item_features: Mapping[str, ChoiceValue]


@dataclass(frozen=True)
class VehicleChoiceSet:
    """One archived choice task and its shared/item-specific features."""

    choice_set_id: str
    source_row_id: str
    respondent_id: str | None
    chosen_alternative_id: str
    shared_features: Mapping[str, ChoiceValue]
    alternatives: tuple[VehicleChoiceAlternative, ...]


@dataclass(frozen=True)
class McFaddenTrainDataset:
    """Parsed archival choice sets with explicit source provenance."""

    choice_sets: tuple[VehicleChoiceSet, ...]
    shared_feature_names: tuple[str, ...]
    item_feature_names: tuple[str, ...]
    provenance: Mapping[str, object]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def acquire_mcfadden_train(
    root: str | Path,
    *,
    download: bool = True,
    source: VehicleChoiceSource = DEFAULT_SOURCE,
) -> dict[str, object]:
    """Acquire a checksum-pinned archive or validate an existing offline cache.

    Set ``download=False`` for offline operation.  A cached archive is never
    trusted by filename alone: its SHA-256 digest is checked on every call.
    """

    archive_directory = Path(root) / "archives"
    archive_directory.mkdir(parents=True, exist_ok=True)
    archive = archive_directory / source.filename
    downloaded = False
    if not archive.exists():
        if not download:
            raise FileNotFoundError(
                f"Missing {archive}; copy the checksum-pinned {source.filename} "
                "there or enable downloads"
            )
        partial = archive.with_suffix(archive.suffix + ".part")
        try:
            tls_context = ssl.create_default_context()
            if certifi is None:
                raise RuntimeError(
                    "Online acquisition requires certifi; install the frozen project "
                    "environment or place the archive in the offline cache"
                )
            tls_context.load_verify_locations(cafile=certifi.where())
            request = urllib.request.Request(
                source.url,
                headers={
                    "User-Agent": "rgsmw-next/0.1 archival-choice-fetcher",
                    "Accept": "application/octet-stream",
                },
            )
            with urllib.request.urlopen(
                request, timeout=120, context=tls_context
            ) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output)
            partial.replace(archive)
            downloaded = True
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise ConnectionError(
                f"Unable to download {source.url} with verified TLS. Copy "
                f"{source.filename} into {archive_directory} and retry with "
                f"download=False. Original error: {error}"
            ) from error
        finally:
            if partial.exists():
                partial.unlink()

    actual_sha256 = sha256_file(archive)
    if actual_sha256.lower() != source.sha256.lower():
        raise ValueError(
            f"Checksum mismatch for {archive}: expected {source.sha256}, "
            f"observed {actual_sha256}"
        )
    return {
        "schema_version": "mcfadden-train-acquisition-1",
        "archive": str(archive),
        "bytes": archive.stat().st_size,
        "sha256": actual_sha256,
        "downloaded": downloaded,
        "cache_mode": "downloaded" if downloaded else "verified_existing",
        "source": _source_provenance(source),
    }


def _source_provenance(source: VehicleChoiceSource) -> dict[str, str]:
    return {
        "name": source.name,
        "url": source.url,
        "repository": source.repository,
        "commit": source.commit,
        "repository_path": source.repository_path,
        "landing_page": source.landing_page,
        "paper_doi": source.paper_doi,
        "software_license": source.software_license,
        "dataset_license_note": source.dataset_license_note,
    }


def _open_choice_csv(path: Path):
    with path.open("rb") as handle:
        magic = handle.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def _required_columns(
    choice_set_id_column: str, respondent_id_column: str | None
) -> set[str]:
    required = set(CHOICE_LEARN_COLUMNS)
    required.add(choice_set_id_column)
    if respondent_id_column is not None:
        required.add(respondent_id_column)
    return required


def _parse_binary(value: str, *, field: str, line_number: int) -> int:
    normalized = value.strip()
    if normalized not in {"0", "1"}:
        raise ValueError(
            f"{field} must be 0 or 1 at CSV line {line_number}; observed {value!r}"
        )
    return int(normalized)


def _parse_numeric(value: str, *, field: str, line_number: int) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(
            f"{field} must be numeric at CSV line {line_number}; observed {value!r}"
        ) from error
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite at CSV line {line_number}")
    return parsed


def parse_mcfadden_train(
    path: str | Path,
    *,
    choice_set_id_column: str = "rownames",
    respondent_id_column: str | None = None,
    strict_schema: bool = True,
    expected_sha256: str | None = None,
    source: VehicleChoiceSource = DEFAULT_SOURCE,
) -> McFaddenTrainDataset:
    """Parse Choice-Learn's single-wide archival CSV or CSV.GZ representation.

    ``respondent_id_column`` defaults to ``None`` because the pinned file does
    not contain an explicit respondent ID.  Passing a column is an explicit
    provenance decision by the caller and is recorded in the returned dataset.
    """

    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    actual_sha256 = sha256_file(source_path)
    if expected_sha256 is not None and actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"Checksum mismatch for {source_path}: expected {expected_sha256}, "
            f"observed {actual_sha256}"
        )

    choice_sets: list[VehicleChoiceSet] = []
    observed_ids: set[str] = set()
    with _open_choice_csv(source_path) as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        required = _required_columns(choice_set_id_column, respondent_id_column)
        missing = sorted(required - set(fieldnames))
        permitted_extra = {
            column
            for column in (choice_set_id_column, respondent_id_column)
            if column is not None
        }
        unexpected = sorted(set(fieldnames) - set(CHOICE_LEARN_COLUMNS) - permitted_extra)
        if missing:
            raise ValueError(f"Missing required vehicle-choice columns: {missing}")
        if strict_schema and unexpected:
            raise ValueError(f"Unexpected vehicle-choice columns: {unexpected}")

        for line_number, row in enumerate(reader, start=2):
            source_row_id = str(row["rownames"]).strip()
            choice_set_id = str(row[choice_set_id_column]).strip()
            if not source_row_id or not choice_set_id:
                raise ValueError(f"Blank source/choice-set ID at CSV line {line_number}")
            if choice_set_id in observed_ids:
                raise ValueError(
                    f"Duplicate choice_set_id {choice_set_id!r} at CSV line {line_number}"
                )
            observed_ids.add(choice_set_id)

            respondent_id = None
            if respondent_id_column is not None:
                respondent_id = str(row[respondent_id_column]).strip()
                if not respondent_id:
                    raise ValueError(
                        f"Blank respondent ID at CSV line {line_number}"
                    )

            choice_match = CHOICE_PATTERN.fullmatch(str(row["choice"]).strip())
            if choice_match is None:
                raise ValueError(
                    f"choice must be choice1..choice6 at CSV line {line_number}; "
                    f"observed {row['choice']!r}"
                )
            chosen_alternative_id = choice_match.group(1)
            shared_features: dict[str, ChoiceValue] = {
                feature: _parse_binary(
                    str(row[feature]), field=feature, line_number=line_number
                )
                for feature in SHARED_FEATURE_NAMES
            }
            alternatives: list[VehicleChoiceAlternative] = []
            for alternative_id in ALTERNATIVE_IDS:
                item_features: dict[str, ChoiceValue] = {}
                for feature in CATEGORICAL_ITEM_FEATURE_NAMES:
                    value = str(row[f"{feature}{alternative_id}"]).strip()
                    if not value:
                        raise ValueError(
                            f"Blank {feature}{alternative_id} at CSV line {line_number}"
                        )
                    item_features[feature] = value
                for feature in NUMERIC_ITEM_FEATURE_NAMES:
                    item_features[feature] = _parse_numeric(
                        str(row[f"{feature}{alternative_id}"]),
                        field=f"{feature}{alternative_id}",
                        line_number=line_number,
                    )
                alternatives.append(
                    VehicleChoiceAlternative(
                        alternative_id=alternative_id,
                        chosen=alternative_id == chosen_alternative_id,
                        item_features=item_features,
                    )
                )
            if sum(alternative.chosen for alternative in alternatives) != 1:
                raise AssertionError("Parser constructed a non-unique chosen alternative")
            choice_sets.append(
                VehicleChoiceSet(
                    choice_set_id=choice_set_id,
                    source_row_id=source_row_id,
                    respondent_id=respondent_id,
                    chosen_alternative_id=chosen_alternative_id,
                    shared_features=shared_features,
                    alternatives=tuple(alternatives),
                )
            )

    if not choice_sets:
        raise ValueError(f"No vehicle-choice rows found in {source_path}")
    respondent_status = (
        "explicit_source_column"
        if respondent_id_column is not None
        else "absent_from_bundled_schema"
    )
    provenance: dict[str, object] = {
        "schema_version": "mcfadden-train-choice-data-1",
        "parser_version": "mcfadden_train_parser_v1",
        "format": "Choice-Learn single-wide CSV with six alternatives",
        "source_file": str(source_path),
        "source_bytes": source_path.stat().st_size,
        "source_sha256": actual_sha256,
        "source": _source_provenance(source),
        "choice_set_id_column": choice_set_id_column,
        "choice_set_id_semantics": (
            "One identifier per archived six-alternative choice task. The default "
            "copies Choice-Learn's R rownames column."
        ),
        "respondent_id_column": respondent_id_column,
        "respondent_id_status": respondent_status,
        "documented_design": (
            "Cross-section; Ecdat documentation describes each observation as an "
            "individual. The bundled Choice-Learn CSV exposes no separate repeated-"
            "respondent/panel identifier."
        ),
        "feature_semantics": dict(FEATURE_SEMANTICS),
        "references": {
            "choice_learn_loader": source.landing_page,
            "ecdat_data_dictionary": (
                "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Car.html"
            ),
            "journal_article": (
                "https://onlinelibrary.wiley.com/doi/abs/"
                "10.1002/1099-1255%28200009%2F10%2915%3A5%3C447%3A%3AAID-JAE570%3E3.0.CO%3B2-1"
            ),
            "journal_data_archive": "http://qed.econ.queensu.ca/jae/2000-v15.5/",
        },
        "normalization": "none; archived categorical strings and numeric values preserved",
        "expected_archival_choice_sets": ARCHIVAL_CHOICE_SET_COUNT,
        "header": list(fieldnames),
    }
    return McFaddenTrainDataset(
        choice_sets=tuple(choice_sets),
        shared_feature_names=SHARED_FEATURE_NAMES,
        item_feature_names=ITEM_FEATURE_NAMES,
        provenance=provenance,
    )


def load_mcfadden_train(
    root: str | Path,
    *,
    download: bool = True,
    respondent_id_column: str | None = None,
    source: VehicleChoiceSource = DEFAULT_SOURCE,
) -> McFaddenTrainDataset:
    """Acquire and parse the pinned archive in one dependency-light call."""

    acquisition = acquire_mcfadden_train(root, download=download, source=source)
    return parse_mcfadden_train(
        str(acquisition["archive"]),
        respondent_id_column=respondent_id_column,
        expected_sha256=source.sha256,
        source=source,
    )


def audit_mcfadden_train(dataset: McFaddenTrainDataset) -> dict[str, object]:
    """Return structural, feature, choice, respondent, and provenance checks."""

    choice_ids: set[str] = set()
    source_row_ids: set[str] = set()
    alternatives_per_set: Counter[int] = Counter()
    chosen_counts: Counter[str] = Counter()
    chosen_type_counts: Counter[str] = Counter()
    chosen_fuel_counts: Counter[str] = Counter()
    categorical_values: dict[str, Counter[str]] = {
        feature: Counter() for feature in CATEGORICAL_ITEM_FEATURE_NAMES
    }
    numeric_values: dict[str, list[float]] = {
        feature: [] for feature in NUMERIC_ITEM_FEATURE_NAMES
    }
    shared_values: dict[str, Counter[int]] = {
        feature: Counter() for feature in SHARED_FEATURE_NAMES
    }
    respondents: Counter[str] = Counter()
    respondent_presence: set[bool] = set()

    for choice_set in dataset.choice_sets:
        if choice_set.choice_set_id in choice_ids:
            raise ValueError(f"Duplicate choice set in dataset: {choice_set.choice_set_id}")
        if choice_set.source_row_id in source_row_ids:
            raise ValueError(f"Duplicate source row in dataset: {choice_set.source_row_id}")
        choice_ids.add(choice_set.choice_set_id)
        source_row_ids.add(choice_set.source_row_id)
        respondent_presence.add(choice_set.respondent_id is not None)
        if choice_set.respondent_id is not None:
            respondents[choice_set.respondent_id] += 1
        alternatives_per_set[len(choice_set.alternatives)] += 1
        alternative_ids = {
            alternative.alternative_id for alternative in choice_set.alternatives
        }
        if alternative_ids != set(ALTERNATIVE_IDS):
            raise ValueError(
                f"Choice set {choice_set.choice_set_id} does not contain alternatives 1..6"
            )
        chosen = [alternative for alternative in choice_set.alternatives if alternative.chosen]
        if len(chosen) != 1 or chosen[0].alternative_id != choice_set.chosen_alternative_id:
            raise ValueError(
                f"Choice set {choice_set.choice_set_id} has inconsistent chosen item"
            )
        chosen_counts[choice_set.chosen_alternative_id] += 1
        chosen_type_counts[str(chosen[0].item_features["type"])] += 1
        chosen_fuel_counts[str(chosen[0].item_features["fuel"])] += 1
        for feature in SHARED_FEATURE_NAMES:
            shared_values[feature][int(choice_set.shared_features[feature])] += 1
        for alternative in choice_set.alternatives:
            for feature in CATEGORICAL_ITEM_FEATURE_NAMES:
                categorical_values[feature][str(alternative.item_features[feature])] += 1
            for feature in NUMERIC_ITEM_FEATURE_NAMES:
                numeric_values[feature].append(float(alternative.item_features[feature]))

    if len(respondent_presence) > 1:
        raise ValueError("Respondent IDs are present for only a subset of choice sets")
    respondents_present = respondent_presence == {True}
    repeated_respondents = (
        {identifier: count for identifier, count in sorted(respondents.items()) if count > 1}
        if respondents_present
        else None
    )
    return {
        "schema_version": "mcfadden-train-audit-1",
        "status": "valid",
        "choice_sets": len(dataset.choice_sets),
        "matches_archival_choice_set_count": (
            len(dataset.choice_sets) == ARCHIVAL_CHOICE_SET_COUNT
        ),
        "alternatives": sum(
            len(choice_set.alternatives) for choice_set in dataset.choice_sets
        ),
        "alternatives_per_choice_set": {
            str(count): frequency for count, frequency in sorted(alternatives_per_set.items())
        },
        "all_choice_sets_have_six_alternatives": alternatives_per_set == {6: len(dataset.choice_sets)},
        "chosen_alternative_counts": dict(sorted(chosen_counts.items())),
        "chosen_type_counts": dict(sorted(chosen_type_counts.items())),
        "chosen_fuel_counts": dict(sorted(chosen_fuel_counts.items())),
        "shared_feature_counts": {
            feature: {str(value): count for value, count in sorted(counter.items())}
            for feature, counter in shared_values.items()
        },
        "categorical_item_values": {
            feature: dict(sorted(counter.items()))
            for feature, counter in categorical_values.items()
        },
        "numeric_item_ranges": {
            feature: {
                "minimum": min(values),
                "maximum": max(values),
                "observations": len(values),
            }
            for feature, values in numeric_values.items()
        },
        "respondent_ids": {
            "status": (
                "explicit_source_column"
                if respondents_present
                else "absent_from_bundled_schema"
            ),
            "unique": len(respondents) if respondents_present else None,
            "repeated": repeated_respondents,
            "repeated_count": (
                len(repeated_respondents or {}) if respondents_present else None
            ),
            "claim": (
                "Repeated respondents are reported only from an explicitly named "
                "source column; none are inferred from rownames."
            ),
        },
        "provenance": dict(dataset.provenance),
    }


def choice_sets_to_long_rows(
    choice_sets: Iterable[VehicleChoiceSet],
) -> tuple[dict[str, object], ...]:
    """Flatten choice sets to dependency-neutral rows for model adapters.

    Feature mappings remain separately namespaced so callers can convert an
    alternative to an ``EntityRecord`` or a tabular representation without
    confusing respondent-shared and alternative-specific attributes.
    """

    rows: list[dict[str, object]] = []
    for choice_set in choice_sets:
        for alternative in choice_set.alternatives:
            rows.append(
                {
                    "choice_set_id": choice_set.choice_set_id,
                    "source_row_id": choice_set.source_row_id,
                    "respondent_id": choice_set.respondent_id,
                    "alternative_id": alternative.alternative_id,
                    "chosen": alternative.chosen,
                    "shared_features": dict(choice_set.shared_features),
                    "item_features": dict(alternative.item_features),
                }
            )
    return tuple(rows)


SplitGroup = Literal["auto", "choice_set", "respondent"]


def _split_sizes(group_count: int, train_fraction: float, dev_fraction: float) -> tuple[int, int]:
    if group_count < 3:
        raise ValueError("At least three independent groups are required for train/dev/test")
    if train_fraction <= 0 or dev_fraction <= 0 or train_fraction + dev_fraction >= 1:
        raise ValueError("train_fraction and dev_fraction must be positive and sum to less than one")
    train_count = max(1, int(math.floor(group_count * train_fraction)))
    dev_count = max(1, int(math.floor(group_count * dev_fraction)))
    while train_count + dev_count > group_count - 1:
        if train_count >= dev_count and train_count > 1:
            train_count -= 1
        elif dev_count > 1:
            dev_count -= 1
        else:
            raise ValueError("Fractions leave no non-empty test partition")
    return train_count, dev_count


def _grouping_key(choice_set: VehicleChoiceSet, group_by: str) -> str:
    if group_by == "choice_set":
        return choice_set.choice_set_id
    if group_by == "respondent":
        if choice_set.respondent_id is None:
            raise ValueError("Respondent grouping requires an explicit respondent ID on every row")
        return choice_set.respondent_id
    raise ValueError(f"Unknown group_by value: {group_by}")


def _resolve_group_by(choice_sets: Sequence[VehicleChoiceSet], group_by: SplitGroup) -> str:
    presence = {choice_set.respondent_id is not None for choice_set in choice_sets}
    if len(presence) > 1:
        raise ValueError("Respondent IDs are present for only a subset of choice sets")
    if group_by == "auto":
        return "respondent" if presence == {True} else "choice_set"
    if group_by == "respondent" and presence != {True}:
        raise ValueError("Respondent grouping requested but no explicit respondent IDs exist")
    if group_by not in {"choice_set", "respondent"}:
        raise ValueError(f"Unknown group_by value: {group_by}")
    return group_by


def deterministic_grouped_choice_split(
    choice_sets: Sequence[VehicleChoiceSet],
    *,
    seed: int = 20260825,
    train_fraction: float = 0.6,
    dev_fraction: float = 0.2,
    group_by: SplitGroup = "auto",
) -> dict[str, object]:
    """Create a stable, leakage-free train/dev/test split by verified groups."""

    rows = tuple(choice_sets)
    if not rows:
        raise ValueError("Cannot split an empty vehicle-choice dataset")
    resolved_group = _resolve_group_by(rows, group_by)
    grouped: dict[str, list[str]] = defaultdict(list)
    choice_set_ids: set[str] = set()
    for choice_set in rows:
        if choice_set.choice_set_id in choice_set_ids:
            raise ValueError(f"Duplicate choice_set_id: {choice_set.choice_set_id}")
        choice_set_ids.add(choice_set.choice_set_id)
        grouped[_grouping_key(choice_set, resolved_group)].append(choice_set.choice_set_id)
    ordered_groups = sorted(
        grouped,
        key=lambda identifier: (
            hashlib.sha256(
                f"mcfadden-train-split-v1\0{seed}\0{identifier}".encode("utf-8")
            ).digest(),
            identifier,
        ),
    )
    train_count, dev_count = _split_sizes(
        len(ordered_groups), train_fraction, dev_fraction
    )
    group_partitions = {
        "train": tuple(ordered_groups[:train_count]),
        "dev": tuple(ordered_groups[train_count : train_count + dev_count]),
        "test": tuple(ordered_groups[train_count + dev_count :]),
    }
    choice_partitions = {
        partition: tuple(
            sorted(
                choice_set_id
                for group_id in group_ids
                for choice_set_id in grouped[group_id]
            )
        )
        for partition, group_ids in group_partitions.items()
    }
    return {
        "schema_version": "mcfadden-train-grouped-split-1",
        "seed": seed,
        "group_by": resolved_group,
        "grouping_note": (
            "Explicit respondent IDs are used only when supplied by a verified "
            "source column; otherwise each documented cross-sectional choice set "
            "is its own split group."
        ),
        "fractions": {
            "train": train_fraction,
            "dev": dev_fraction,
            "test": 1.0 - train_fraction - dev_fraction,
        },
        "train": choice_partitions["train"],
        "dev": choice_partitions["dev"],
        "test": choice_partitions["test"],
        "group_ids": group_partitions,
        "group_counts": {
            partition: len(group_ids) for partition, group_ids in group_partitions.items()
        },
        "choice_set_counts": {
            partition: len(ids) for partition, ids in choice_partitions.items()
        },
    }


def _choice_outcome(
    choice_set: VehicleChoiceSet,
    scores: Mapping[str, float],
    *,
    higher_is_better: bool,
) -> dict[str, object]:
    alternatives = {alternative.alternative_id for alternative in choice_set.alternatives}
    if set(scores) != alternatives:
        missing = sorted(alternatives - set(scores))
        unexpected = sorted(set(scores) - alternatives)
        raise ValueError(
            f"Scores for {choice_set.choice_set_id} do not match its alternatives; "
            f"missing={missing}, unexpected={unexpected}"
        )
    utilities: dict[str, float] = {}
    for alternative_id, raw_score in scores.items():
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError(
                f"Non-finite score for {choice_set.choice_set_id}/{alternative_id}"
            )
        utilities[alternative_id] = score if higher_is_better else -score
    ranking = sorted(utilities, key=lambda item: (-utilities[item], item))
    chosen_rank = ranking.index(choice_set.chosen_alternative_id) + 1
    maximum = max(utilities.values())
    denominator = sum(math.exp(value - maximum) for value in utilities.values())
    chosen_probability = (
        math.exp(utilities[choice_set.chosen_alternative_id] - maximum) / denominator
    )
    return {
        "choice_set_id": choice_set.choice_set_id,
        "respondent_id": choice_set.respondent_id,
        "chosen_alternative_id": choice_set.chosen_alternative_id,
        "predicted_alternative_id": ranking[0],
        "chosen_rank": chosen_rank,
        "top1_accuracy": float(chosen_rank == 1),
        "hit_at_3": float(chosen_rank <= 3),
        "reciprocal_rank": 1.0 / chosen_rank,
        "ndcg_at_6": 1.0 / math.log2(chosen_rank + 1.0),
        "chosen_probability": chosen_probability,
        "log_loss": -math.log(max(chosen_probability, np.finfo(float).tiny)),
    }


def evaluate_choice_scores(
    choice_sets: Sequence[VehicleChoiceSet],
    scores: Mapping[str, Mapping[str, float]],
    *,
    choice_set_ids: Iterable[str] | None = None,
    higher_is_better: bool = True,
) -> dict[str, object]:
    """Evaluate one score per alternative on frozen six-alternative choice sets."""

    by_id = {choice_set.choice_set_id: choice_set for choice_set in choice_sets}
    selected = tuple(sorted(set(choice_set_ids) if choice_set_ids is not None else by_id))
    if not selected:
        raise ValueError("No choice sets selected for evaluation")
    missing_choice_sets = sorted(set(selected) - set(by_id))
    if missing_choice_sets:
        raise KeyError(f"Unknown choice_set_ids: {missing_choice_sets}")
    missing_scores = sorted(set(selected) - set(scores))
    if missing_scores:
        raise KeyError(f"Missing scores for choice sets: {missing_scores}")
    outcomes = [
        _choice_outcome(
            by_id[choice_set_id],
            scores[choice_set_id],
            higher_is_better=higher_is_better,
        )
        for choice_set_id in selected
    ]
    metric_names = (
        "top1_accuracy",
        "hit_at_3",
        "reciprocal_rank",
        "ndcg_at_6",
        "chosen_probability",
        "log_loss",
    )
    return {
        "schema_version": "mcfadden-train-choice-evaluation-1",
        "choice_sets": len(outcomes),
        "alternatives_per_choice_set": 6,
        "higher_is_better": higher_is_better,
        "tie_break": "ascending alternative_id",
        "random_top1_reference": 1.0 / 6.0,
        "metrics": {
            metric: float(np.mean([float(row[metric]) for row in outcomes]))
            for metric in metric_names
        },
        "per_choice_set": outcomes,
    }


def paired_grouped_choice_bootstrap(
    choice_sets: Sequence[VehicleChoiceSet],
    proposed_scores: Mapping[str, Mapping[str, float]],
    baseline_scores: Mapping[str, Mapping[str, float]],
    *,
    choice_set_ids: Iterable[str] | None = None,
    metric: str = "top1_accuracy",
    group_by: SplitGroup = "auto",
    higher_is_better: bool = True,
    seed: int = 20260825,
    replications: int = 2_000,
) -> dict[str, object]:
    """Paired cluster bootstrap over explicit respondents or choice sets."""

    if replications <= 0:
        raise ValueError("replications must be positive")
    selected_ids = tuple(
        sorted(
            set(choice_set_ids)
            if choice_set_ids is not None
            else {choice_set.choice_set_id for choice_set in choice_sets}
        )
    )
    selected_set = set(selected_ids)
    selected_rows = tuple(
        choice_set for choice_set in choice_sets if choice_set.choice_set_id in selected_set
    )
    if len(selected_rows) != len(selected_ids):
        observed = {choice_set.choice_set_id for choice_set in selected_rows}
        raise KeyError(f"Unknown choice_set_ids: {sorted(selected_set - observed)}")
    resolved_group = _resolve_group_by(selected_rows, group_by)
    proposed = evaluate_choice_scores(
        selected_rows,
        proposed_scores,
        higher_is_better=higher_is_better,
    )
    baseline = evaluate_choice_scores(
        selected_rows,
        baseline_scores,
        higher_is_better=higher_is_better,
    )
    proposed_rows = {
        str(row["choice_set_id"]): row for row in proposed["per_choice_set"]
    }
    baseline_rows = {
        str(row["choice_set_id"]): row for row in baseline["per_choice_set"]
    }
    if metric not in proposed_rows[selected_ids[0]]:
        raise KeyError(f"Unknown per-choice metric: {metric}")
    grouped_differences: dict[str, list[float]] = defaultdict(list)
    for choice_set in selected_rows:
        proposed_value = proposed_rows[choice_set.choice_set_id][metric]
        baseline_value = baseline_rows[choice_set.choice_set_id][metric]
        if not isinstance(proposed_value, (int, float)) or not isinstance(
            baseline_value, (int, float)
        ):
            raise TypeError(f"Metric {metric!r} is not numeric")
        grouped_differences[_grouping_key(choice_set, resolved_group)].append(
            float(proposed_value) - float(baseline_value)
        )
    group_ids = tuple(sorted(grouped_differences))
    observed = float(
        np.mean([value for values in grouped_differences.values() for value in values])
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(replications, dtype=float)
    for replication in range(replications):
        sampled = rng.choice(group_ids, size=len(group_ids), replace=True)
        values = [
            value
            for group_id in sampled
            for value in grouped_differences[str(group_id)]
        ]
        draws[replication] = float(np.mean(values))
    return {
        "schema_version": "mcfadden-train-paired-bootstrap-1",
        "metric": metric,
        "contrast": "proposed_minus_baseline",
        "mean_difference": observed,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "choice_sets": len(selected_rows),
        "groups": len(group_ids),
        "group_by": resolved_group,
        "seed": seed,
        "bootstrap_replications": replications,
    }


__all__ = [
    "ALTERNATIVE_IDS",
    "ARCHIVAL_CHOICE_SET_COUNT",
    "CATEGORICAL_ITEM_FEATURE_NAMES",
    "CHOICE_LEARN_COLUMNS",
    "DEFAULT_SOURCE",
    "FEATURE_SEMANTICS",
    "ITEM_FEATURE_NAMES",
    "McFaddenTrainDataset",
    "NUMERIC_ITEM_FEATURE_NAMES",
    "SHARED_FEATURE_NAMES",
    "VehicleChoiceAlternative",
    "VehicleChoiceSet",
    "VehicleChoiceSource",
    "acquire_mcfadden_train",
    "audit_mcfadden_train",
    "choice_sets_to_long_rows",
    "deterministic_grouped_choice_split",
    "evaluate_choice_scores",
    "load_mcfadden_train",
    "paired_grouped_choice_bootstrap",
    "parse_mcfadden_train",
    "sha256_file",
]
