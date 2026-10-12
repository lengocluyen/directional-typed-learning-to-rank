from __future__ import annotations

import csv
import hashlib
import json
import shutil
import ssl
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml

from .preference_learning import PairwiseObservation
from .types import EntityRecord, PredicateSpec

try:
    import certifi
except ModuleNotFoundError:  # Offline/cache-only use does not need a CA bundle.
    certifi = None  # type: ignore[assignment]


MANIFEST_SCHEMA = "rgsmw-anu-car-preferences-1"

_BODY_TYPE = {1: "sedan", 2: "suv", 3: "hatchback"}
_TRANSMISSION = {1: "manual", 2: "automatic"}
_FUEL = {1: "hybrid", 2: "non_hybrid"}
_LAYOUT = {1: "awd", 2: "fwd"}


@dataclass(frozen=True)
class AnuCarItem:
    experiment_id: str
    item_id: str
    body_type: int
    transmission: int
    engine_capacity_l: float
    fuel_consumed: int
    engine_transmission_layout: int | None
    source_line: int

    @property
    def item_features(self) -> dict[str, int | float]:
        features: dict[str, int | float] = {
            "body_type": self.body_type,
            "transmission": self.transmission,
            "engine_capacity_l": self.engine_capacity_l,
            "fuel_consumed": self.fuel_consumed,
        }
        if self.engine_transmission_layout is not None:
            features["engine_transmission_layout"] = self.engine_transmission_layout
        return features


@dataclass(frozen=True)
class AnuCarUser:
    experiment_id: str
    user_id: str
    education: int
    age: int
    gender: int
    region: int
    correct_controls: int
    source_line: int

    @property
    def user_attributes(self) -> dict[str, int]:
        return {
            "education": self.education,
            "age": self.age,
            "gender": self.gender,
            "region": self.region,
            "correct_controls": self.correct_controls,
        }


@dataclass(frozen=True)
class AnuCarPreference:
    experiment_id: str
    user_id: str
    preferred: str
    nonpreferred: str
    control: bool
    source_line: int

    @property
    def user_key(self) -> tuple[str, str]:
        return (self.experiment_id, self.user_id)

    @property
    def unordered_pair_key(self) -> tuple[str, str, str, str]:
        first, second = sorted((self.preferred, self.nonpreferred))
        return (self.experiment_id, self.user_id, first, second)


@dataclass(frozen=True)
class AnuFilterPolicy:
    """Predeclared exclusions for the archival observations.

    The default retains every source row.  ``minimum_correct_controls`` follows
    the actual user-file field and deliberately avoids the ambiguous
    ``limit``/"mistakes" wording in the archive's optional Matlab helper.
    """

    experiment_ids: tuple[str, ...] = ()
    include_controls: bool = True
    minimum_correct_controls: int | None = None
    minimum_preferences_per_user: int | None = None
    require_complete_user_attributes: bool = False

    def __post_init__(self) -> None:
        if self.minimum_correct_controls is not None and self.minimum_correct_controls < 0:
            raise ValueError("minimum_correct_controls must be non-negative")
        if (
            self.minimum_preferences_per_user is not None
            and self.minimum_preferences_per_user < 1
        ):
            raise ValueError("minimum_preferences_per_user must be positive")
        if len(set(self.experiment_ids)) != len(self.experiment_ids):
            raise ValueError("experiment_ids contains duplicates")


@dataclass(frozen=True)
class AnuCarPreferenceDataset:
    items: tuple[AnuCarItem, ...]
    users: tuple[AnuCarUser, ...]
    preferences: tuple[AnuCarPreference, ...]
    audit: Mapping[str, object]


@dataclass(frozen=True)
class AnuFoldAssignment:
    experiment_id: str
    user_id: str
    fold: int


@dataclass(frozen=True)
class AnuUserGroupedFolds:
    assignments: tuple[AnuFoldAssignment, ...]
    n_folds: int
    seed: int
    audit: Mapping[str, object]

    def as_mapping(self) -> dict[tuple[str, str], int]:
        return {
            (assignment.experiment_id, assignment.user_id): assignment.fold
            for assignment in self.assignments
        }


@dataclass(frozen=True)
class AnuSupportEvaluationSplit:
    support: tuple[AnuCarPreference, ...]
    evaluation: tuple[AnuCarPreference, ...]
    excluded_controls: tuple[AnuCarPreference, ...]
    support_size_per_user: int
    seed: int
    audit: Mapping[str, object]


def default_anu_manifest_path() -> Path:
    return Path(str(files("rgsmw_next").joinpath("config/anu_car_preferences.yaml")))


def load_anu_manifest(path: str | Path | None = None) -> dict[str, object]:
    manifest_path = Path(path) if path else default_anu_manifest_path()
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"Unsupported ANU car-preference manifest: {manifest_path}")
    if not isinstance(payload.get("archives"), dict) or not isinstance(
        payload.get("experiments"), dict
    ):
        raise ValueError(f"Incomplete ANU car-preference manifest: {manifest_path}")
    return payload


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> tuple[str, ...]:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    extracted: list[str] = []
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            member_path = Path(member.filename)
            target = (destination / member_path).resolve()
            unix_mode = member.external_attr >> 16
            if member_path.is_absolute() or (root != target and root not in target.parents):
                raise ValueError(f"Unsafe path in {archive.name}: {member.filename}")
            if unix_mode & 0o170000 == 0o120000:
                raise ValueError(f"Symbolic link is not allowed in {archive.name}: {member.filename}")
            if member.is_dir() or member.filename.startswith("__MACOSX/"):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted.append(member.filename)
    return tuple(sorted(extracted))


def _clear_extracted_directory(destination: Path, extracted_root: Path) -> None:
    resolved_destination = destination.resolve()
    resolved_root = extracted_root.resolve()
    if resolved_destination == resolved_root or resolved_root not in resolved_destination.parents:
        raise ValueError(f"Refusing to clear extraction path outside cache: {destination}")
    if destination.exists():
        shutil.rmtree(destination)


def acquire_anu_car_preferences(
    root: str | Path,
    manifest_path: str | Path | None = None,
    download: bool = True,
) -> dict[str, object]:
    """Acquire checksum-pinned official archives and extract them safely.

    In offline mode callers copy the exact archives into ``root/archives``.
    TLS verification is never disabled.
    """

    root_path = Path(root)
    archives_dir = root_path / "archives"
    extracted_dir = root_path / "extracted"
    archives_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_anu_manifest(manifest_path)
    acquired: dict[str, object] = {}
    for archive_id, raw_spec in sorted(dict(manifest["archives"]).items()):
        spec = dict(raw_spec)
        archive_path = archives_dir / str(spec["filename"])
        expected = str(spec["sha256"]).casefold()
        if not archive_path.exists():
            if not download:
                raise FileNotFoundError(
                    f"Missing {archive_path}; enable download or copy the checksum-pinned "
                    "official archive into the cache"
                )
            partial = archive_path.with_suffix(archive_path.suffix + ".part")
            try:
                if certifi is None:
                    raise RuntimeError(
                        "Online ANU download requires certifi; install the locked project "
                        "environment or use the offline cache"
                    )
                tls_context = ssl.create_default_context()
                tls_context.load_verify_locations(cafile=certifi.where())
                request = urllib.request.Request(
                    str(spec["url"]),
                    headers={"User-Agent": "rgsmw-next/0.1 reproducible-dataset-fetcher"},
                )
                with urllib.request.urlopen(
                    request, timeout=120, context=tls_context
                ) as response, partial.open("wb") as output:
                    shutil.copyfileobj(response, output)
                partial.replace(archive_path)
            except urllib.error.URLError as error:
                raise ConnectionError(
                    f"Unable to download {spec['url']} with verified TLS. Copy "
                    f"{spec['filename']} into {archives_dir} and retry offline. "
                    f"Original error: {error}"
                ) from error
            finally:
                if partial.exists():
                    partial.unlink()
        actual = sha256_file(archive_path)
        if actual.casefold() != expected:
            raise ValueError(
                f"Checksum mismatch for {archive_path}: expected {expected}, observed {actual}"
            )
        destination = extracted_dir / str(archive_id)
        marker = destination / ".archive_sha256"
        extracted_members: tuple[str, ...]
        if not marker.exists() or marker.read_text(encoding="ascii").strip() != actual:
            _clear_extracted_directory(destination, extracted_dir)
            extracted_members = _safe_extract(archive_path, destination)
            marker.write_text(actual + "\n", encoding="ascii", newline="\n")
        else:
            extracted_members = tuple(
                sorted(
                    str(path.relative_to(destination)).replace("\\", "/")
                    for path in destination.rglob("*")
                    if path.is_file() and path != marker
                )
            )
        acquired[str(archive_id)] = {
            "url": str(spec["url"]),
            "archive": str(archive_path),
            "bytes": archive_path.stat().st_size,
            "sha256": actual,
            "extracted": str(destination),
            "members": list(extracted_members),
        }
    return {
        "schema_version": MANIFEST_SCHEMA,
        "source": dict(manifest.get("source", {})),
        "archives": acquired,
    }


_USER_HEADERS = {
    "user_id": ("User ID",),
    "education": ("Education",),
    "age": ("Age",),
    "gender": ("Gender",),
    "region": ("Region",),
    "correct_controls": ("# Correct Controls",),
}
_ITEM_HEADERS = {
    "item_id": ("Item ID",),
    "body_type": ("BodyType", "Body Type"),
    "transmission": ("Transmission",),
    "engine_capacity_l": ("Engin Capacity", "Engine Capacity"),
    "fuel_consumed": ("Fuel Consumed",),
}
_LAYOUT_HEADER = {"engine_transmission_layout": ("Engine/Transmission Layout",)}
_PREFERENCE_HEADERS = {
    "user_id": ("User ID",),
    "preferred": ("Item1 ID",),
    "nonpreferred": ("Item2 ID",),
    "control": ("Is Control",),
}


def _read_csv(
    path: Path, header_aliases: Mapping[str, Sequence[str]]
) -> list[tuple[int, dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        observed = tuple((name or "").strip() for name in (reader.fieldnames or ()))
        if not observed:
            raise ValueError(f"CSV has no header: {path}")
        mapping: dict[str, str] = {}
        for canonical, aliases in header_aliases.items():
            matches = [name for name in observed if name in aliases]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one header for {canonical!r} in {path}; observed {observed}"
                )
            mapping[canonical] = matches[0]
        expected_observed = set(mapping.values())
        unexpected = sorted(set(observed) - expected_observed)
        if unexpected:
            raise ValueError(f"Unexpected columns in {path}: {unexpected}")
        rows: list[tuple[int, dict[str, str]]] = []
        for line_number, raw in enumerate(reader, start=2):
            if None in raw:
                raise ValueError(f"Too many CSV fields at {path}:{line_number}")
            row = {
                canonical: str(raw[header] or "").strip()
                for canonical, header in mapping.items()
            }
            if not any(row.values()):
                continue
            if not all(row.values()):
                raise ValueError(f"Blank required CSV value at {path}:{line_number}")
            rows.append((line_number, row))
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    return rows


def _integer(value: str, path: Path, line: int, field: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"Invalid integer {field} at {path}:{line}: {value!r}") from error
    return parsed


def _float(value: str, path: Path, line: int, field: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"Invalid numeric {field} at {path}:{line}: {value!r}") from error
    return parsed


def _parse_users(path: Path, experiment_id: str) -> tuple[AnuCarUser, ...]:
    users: list[AnuCarUser] = []
    for line, row in _read_csv(path, _USER_HEADERS):
        user = AnuCarUser(
            experiment_id=experiment_id,
            user_id=row["user_id"],
            education=_integer(row["education"], path, line, "education"),
            age=_integer(row["age"], path, line, "age"),
            gender=_integer(row["gender"], path, line, "gender"),
            region=_integer(row["region"], path, line, "region"),
            correct_controls=_integer(
                row["correct_controls"], path, line, "correct_controls"
            ),
            source_line=line,
        )
        if user.education not in range(4):
            raise ValueError(f"Education code outside 0..3 at {path}:{line}")
        if user.age not in range(5):
            raise ValueError(f"Age code outside 0..4 at {path}:{line}")
        if user.gender not in range(3):
            raise ValueError(f"Gender code outside 0..2 at {path}:{line}")
        if user.region not in range(5):
            raise ValueError(f"Region code outside 0..4 at {path}:{line}")
        if user.correct_controls < 0:
            raise ValueError(f"Negative correct-controls count at {path}:{line}")
        users.append(user)
    return tuple(users)


def _parse_items(
    path: Path, experiment_id: str, has_layout: bool
) -> tuple[AnuCarItem, ...]:
    aliases = dict(_ITEM_HEADERS)
    if has_layout:
        aliases.update(_LAYOUT_HEADER)
    items: list[AnuCarItem] = []
    for line, row in _read_csv(path, aliases):
        item = AnuCarItem(
            experiment_id=experiment_id,
            item_id=row["item_id"],
            body_type=_integer(row["body_type"], path, line, "body_type"),
            transmission=_integer(row["transmission"], path, line, "transmission"),
            engine_capacity_l=_float(
                row["engine_capacity_l"], path, line, "engine_capacity_l"
            ),
            fuel_consumed=_integer(
                row["fuel_consumed"], path, line, "fuel_consumed"
            ),
            engine_transmission_layout=(
                _integer(
                    row["engine_transmission_layout"],
                    path,
                    line,
                    "engine_transmission_layout",
                )
                if has_layout
                else None
            ),
            source_line=line,
        )
        maximum_body_type = 3 if has_layout else 2
        if item.body_type not in range(1, maximum_body_type + 1):
            raise ValueError(f"Body-type code outside expected range at {path}:{line}")
        if item.transmission not in {1, 2}:
            raise ValueError(f"Transmission code outside 1..2 at {path}:{line}")
        if item.fuel_consumed not in {1, 2}:
            raise ValueError(f"Fuel code outside 1..2 at {path}:{line}")
        if item.engine_capacity_l <= 0:
            raise ValueError(f"Non-positive engine capacity at {path}:{line}")
        if has_layout and item.engine_transmission_layout not in {1, 2}:
            raise ValueError(f"Layout code outside 1..2 at {path}:{line}")
        items.append(item)
    return tuple(items)


def _parse_preferences(path: Path, experiment_id: str) -> tuple[AnuCarPreference, ...]:
    preferences: list[AnuCarPreference] = []
    for line, row in _read_csv(path, _PREFERENCE_HEADERS):
        control = _integer(row["control"], path, line, "control")
        if control not in {0, 1}:
            raise ValueError(f"Control flag must be 0 or 1 at {path}:{line}")
        if row["preferred"] == row["nonpreferred"]:
            raise ValueError(f"Self-preference at {path}:{line}")
        preferences.append(
            AnuCarPreference(
                experiment_id=experiment_id,
                user_id=row["user_id"],
                preferred=row["preferred"],
                nonpreferred=row["nonpreferred"],
                control=bool(control),
                source_line=line,
            )
        )
    return tuple(preferences)


def _composite_id(experiment_id: str, identifier: str) -> str:
    return f"{experiment_id}:{identifier}"


def _validate_references(
    users: Sequence[AnuCarUser],
    items: Sequence[AnuCarItem],
    preferences: Sequence[AnuCarPreference],
) -> None:
    user_keys = [(user.experiment_id, user.user_id) for user in users]
    item_keys = [(item.experiment_id, item.item_id) for item in items]
    if len(set(user_keys)) != len(user_keys):
        raise ValueError("Duplicate composite user ID in ANU data")
    if len(set(item_keys)) != len(item_keys):
        raise ValueError("Duplicate composite item ID in ANU data")
    user_set = set(user_keys)
    item_set = set(item_keys)
    for preference in preferences:
        if preference.user_key not in user_set:
            raise ValueError(
                f"Preference references absent user {_composite_id(*preference.user_key)}"
            )
        for item_id in (preference.preferred, preference.nonpreferred):
            if (preference.experiment_id, item_id) not in item_set:
                raise ValueError(
                    "Preference references absent item "
                    f"{_composite_id(preference.experiment_id, item_id)}"
                )


def _statistics(
    items: Sequence[AnuCarItem],
    users: Sequence[AnuCarUser],
    preferences: Sequence[AnuCarPreference],
) -> dict[str, object]:
    experiment_ids = sorted(
        {item.experiment_id for item in items}
        | {user.experiment_id for user in users}
        | {preference.experiment_id for preference in preferences}
    )
    result: dict[str, object] = {}
    for experiment_id in experiment_ids:
        experiment_items = [item for item in items if item.experiment_id == experiment_id]
        experiment_users = [user for user in users if user.experiment_id == experiment_id]
        experiment_preferences = [
            preference
            for preference in preferences
            if preference.experiment_id == experiment_id
        ]
        pair_groups: dict[tuple[str, str, str, str], list[AnuCarPreference]] = defaultdict(list)
        exact_rows: Counter[tuple[str, str, str, bool]] = Counter()
        for preference in experiment_preferences:
            pair_groups[preference.unordered_pair_key].append(preference)
            exact_rows[
                (
                    preference.user_id,
                    preference.preferred,
                    preference.nonpreferred,
                    preference.control,
                )
            ] += 1
        noncontrol_conflicts = 0
        for rows in pair_groups.values():
            directions = {
                (row.preferred, row.nonpreferred) for row in rows if not row.control
            }
            if len(directions) > 1:
                noncontrol_conflicts += 1
        result[experiment_id] = {
            "items": len(experiment_items),
            "users": len(experiment_users),
            "preferences": len(experiment_preferences),
            "control_preferences": sum(row.control for row in experiment_preferences),
            "noncontrol_preferences": sum(not row.control for row in experiment_preferences),
            "unordered_user_pair_groups": len(pair_groups),
            "repeated_unordered_user_pair_groups": sum(
                len(rows) > 1 for rows in pair_groups.values()
            ),
            "conflicting_noncontrol_pair_groups": noncontrol_conflicts,
            "exact_duplicate_rows_beyond_first": sum(
                count - 1 for count in exact_rows.values() if count > 1
            ),
            "missing_user_attributes": {
                field: sum(getattr(user, field) == 0 for user in experiment_users)
                for field in ("education", "age", "gender", "region")
            },
            "observed_correct_controls": sorted(
                {user.correct_controls for user in experiment_users}
            ),
            "observed_item_features": {
                field: sorted(
                    {
                        value
                        for item in experiment_items
                        if (value := getattr(item, field)) is not None
                    }
                )
                for field in (
                    "body_type",
                    "transmission",
                    "engine_capacity_l",
                    "fuel_consumed",
                    "engine_transmission_layout",
                )
            },
        }
    return result


def _filter_dataset(
    dataset: AnuCarPreferenceDataset, policy: AnuFilterPolicy
) -> AnuCarPreferenceDataset:
    available_experiments = sorted({user.experiment_id for user in dataset.users})
    selected = set(policy.experiment_ids or available_experiments)
    unknown = selected - set(available_experiments)
    if unknown:
        raise KeyError(f"Unknown ANU experiment IDs: {sorted(unknown)}")

    eligible_before_minimum: list[AnuCarPreference] = [
        preference
        for preference in dataset.preferences
        if preference.experiment_id in selected
        and (policy.include_controls or not preference.control)
    ]
    preference_counts = Counter(preference.user_key for preference in eligible_before_minimum)
    excluded_users: dict[str, list[str]] = defaultdict(list)
    retained_user_keys: set[tuple[str, str]] = set()
    for user in dataset.users:
        key = (user.experiment_id, user.user_id)
        label = _composite_id(*key)
        if user.experiment_id not in selected:
            excluded_users["experiment_not_selected"].append(label)
            continue
        if (
            policy.minimum_correct_controls is not None
            and user.correct_controls < policy.minimum_correct_controls
        ):
            excluded_users["below_minimum_correct_controls"].append(label)
            continue
        if policy.require_complete_user_attributes and any(
            getattr(user, field) == 0 for field in ("education", "age", "gender", "region")
        ):
            excluded_users["incomplete_user_attributes"].append(label)
            continue
        if (
            policy.minimum_preferences_per_user is not None
            and preference_counts[key] < policy.minimum_preferences_per_user
        ):
            excluded_users["below_minimum_preferences"].append(label)
            continue
        retained_user_keys.add(key)

    excluded_preferences: Counter[str] = Counter()
    retained_preferences: list[AnuCarPreference] = []
    for preference in dataset.preferences:
        if preference.experiment_id not in selected:
            excluded_preferences["experiment_not_selected"] += 1
        elif preference.user_key not in retained_user_keys:
            excluded_preferences["user_filtered"] += 1
        elif not policy.include_controls and preference.control:
            excluded_preferences["control_filtered"] += 1
        else:
            retained_preferences.append(preference)
    retained_users = tuple(
        user
        for user in dataset.users
        if (user.experiment_id, user.user_id) in retained_user_keys
    )
    retained_items = tuple(
        item for item in dataset.items if item.experiment_id in selected
    )
    if not retained_users or not retained_preferences:
        raise ValueError("ANU filter policy removes every user or preference")
    audit = dict(dataset.audit)
    audit["filter"] = {
        "policy": {
            **asdict(policy),
            "experiment_ids": list(policy.experiment_ids),
        },
        "input": {
            "items": len(dataset.items),
            "users": len(dataset.users),
            "preferences": len(dataset.preferences),
        },
        "output": {
            "items": len(retained_items),
            "users": len(retained_users),
            "preferences": len(retained_preferences),
        },
        "excluded_users": {
            reason: sorted(values) for reason, values in sorted(excluded_users.items())
        },
        "excluded_preferences": dict(sorted(excluded_preferences.items())),
    }
    audit["statistics"] = _statistics(
        retained_items, retained_users, retained_preferences
    )
    return AnuCarPreferenceDataset(
        items=retained_items,
        users=retained_users,
        preferences=tuple(retained_preferences),
        audit=audit,
    )


def load_anu_car_preferences(
    root: str | Path,
    policy: AnuFilterPolicy | None = None,
    manifest_path: str | Path | None = None,
    verify_file_checksums: bool = True,
) -> AnuCarPreferenceDataset:
    """Parse both experiments from an acquired cache and return an audited view."""

    root_path = Path(root)
    manifest = load_anu_manifest(manifest_path)
    items: list[AnuCarItem] = []
    users: list[AnuCarUser] = []
    preferences: list[AnuCarPreference] = []
    file_audit: dict[str, object] = {}
    for experiment_id, raw_spec in sorted(dict(manifest["experiments"]).items()):
        spec = dict(raw_spec)
        directory = root_path / "extracted" / str(spec["archive"])
        paths = {
            kind: directory / str(spec[kind])
            for kind in ("users", "items", "preferences")
        }
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"Missing acquired ANU file: {path}")
        expected_hashes = {
            str(name): str(digest).casefold()
            for name, digest in dict(spec.get("file_sha256", {})).items()
        }
        observed_hashes = {path.name: sha256_file(path) for path in paths.values()}
        if verify_file_checksums:
            for filename, observed in observed_hashes.items():
                expected = expected_hashes.get(filename)
                if not expected:
                    raise ValueError(f"No expected checksum for {filename} in manifest")
                if observed.casefold() != expected:
                    raise ValueError(
                        f"Checksum mismatch for {paths[next(key for key, value in paths.items() if value.name == filename)]}: "
                        f"expected {expected}, observed {observed}"
                    )
        users.extend(_parse_users(paths["users"], str(experiment_id)))
        items.extend(
            _parse_items(
                paths["items"], str(experiment_id), bool(spec.get("has_layout", False))
            )
        )
        preferences.extend(_parse_preferences(paths["preferences"], str(experiment_id)))
        file_audit[str(experiment_id)] = {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": observed_hashes[path.name],
            }
            for name, path in sorted(paths.items())
        }

    stable_items = tuple(sorted(items, key=lambda row: (row.experiment_id, row.item_id)))
    stable_users = tuple(sorted(users, key=lambda row: (row.experiment_id, row.user_id)))
    stable_preferences = tuple(
        sorted(
            preferences,
            key=lambda row: (row.experiment_id, row.user_id, row.source_line),
        )
    )
    _validate_references(stable_users, stable_items, stable_preferences)
    raw_audit: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA,
        "source": dict(manifest.get("source", {})),
        "files": file_audit,
        "statistics": _statistics(stable_items, stable_users, stable_preferences),
        "integrity": {
            "composite_ids": "experiment_id plus source-local ID",
            "preference_direction": "preferred is Item1 ID; nonpreferred is Item2 ID",
            "control_rows_retained_in_raw_parse": True,
            "archived_cleaning_helper_note": (
                "clear_controls.m compares the user-file correct-controls field against "
                "a minimum threshold although its comments describe a mistake limit; this "
                "adapter exposes minimum_correct_controls by its observed meaning"
            ),
            "validation": "all preference user/item references resolved",
        },
        "codebooks": dict(manifest.get("codebooks", {})),
    }
    raw_dataset = AnuCarPreferenceDataset(
        items=stable_items,
        users=stable_users,
        preferences=stable_preferences,
        audit=raw_audit,
    )
    return _filter_dataset(raw_dataset, policy or AnuFilterPolicy())


def write_anu_audit(dataset: AnuCarPreferenceDataset, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(dataset.audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _stable_digest(seed: int, *parts: str) -> str:
    payload = "\0".join((str(seed), *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deterministic_user_grouped_folds(
    users: Iterable[AnuCarUser],
    n_folds: int = 5,
    seed: int = 20260825,
) -> AnuUserGroupedFolds:
    """Build balanced folds independently within each experiment.

    Composite ``(experiment_id, user_id)`` keys are the assignment unit; input
    order and Python hash randomisation cannot affect the result.
    """

    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    grouped: dict[str, dict[str, AnuCarUser]] = defaultdict(dict)
    for user in users:
        if user.user_id in grouped[user.experiment_id]:
            raise ValueError(
                f"Duplicate user {_composite_id(user.experiment_id, user.user_id)}"
            )
        grouped[user.experiment_id][user.user_id] = user
    if not grouped:
        raise ValueError("No users supplied for fold assignment")
    assignments: list[AnuFoldAssignment] = []
    by_experiment: dict[str, object] = {}
    for experiment_id, user_map in sorted(grouped.items()):
        if len(user_map) < n_folds:
            raise ValueError(
                f"Experiment {experiment_id} has fewer users than n_folds={n_folds}"
            )
        ordered = sorted(
            user_map,
            key=lambda user_id: (
                _stable_digest(seed, "fold", experiment_id, user_id),
                user_id,
            ),
        )
        offset = int(_stable_digest(seed, "offset", experiment_id)[:8], 16) % n_folds
        fold_counts: Counter[int] = Counter()
        for index, user_id in enumerate(ordered):
            fold = (index + offset) % n_folds
            assignments.append(AnuFoldAssignment(experiment_id, user_id, fold))
            fold_counts[fold] += 1
        by_experiment[experiment_id] = {
            "users": len(ordered),
            "fold_counts": {
                str(fold): fold_counts[fold] for fold in range(n_folds)
            },
        }
    stable = tuple(
        sorted(assignments, key=lambda row: (row.experiment_id, row.user_id))
    )
    return AnuUserGroupedFolds(
        assignments=stable,
        n_folds=n_folds,
        seed=seed,
        audit={
            "unit": "composite experiment_id/user_id",
            "algorithm": "sha256 ordering with per-experiment balanced round-robin",
            "n_folds": n_folds,
            "seed": seed,
            "experiments": by_experiment,
        },
    )


def deterministic_support_evaluation_split(
    preferences: Iterable[AnuCarPreference],
    support_size_per_user: int,
    seed: int = 20260825,
    control_policy: str = "exclude",
    require_evaluation: bool = True,
) -> AnuSupportEvaluationSplit:
    """Separate support and evaluation without leaking an unordered pair.

    ``support_size_per_user`` counts unique unordered item-pair groups. Repeated
    rows, including reversed controls when retained, stay together in one role.
    The audit reports both group and row counts so duplicates cannot silently
    change the effective observation budget.
    """

    if support_size_per_user < 0:
        raise ValueError("support_size_per_user must be non-negative")
    if control_policy not in {"exclude", "include"}:
        raise ValueError("control_policy must be 'exclude' or 'include'")
    rows = tuple(preferences)
    if not rows:
        raise ValueError("No preferences supplied for support/evaluation split")
    by_user: dict[tuple[str, str], list[AnuCarPreference]] = defaultdict(list)
    excluded_controls: list[AnuCarPreference] = []
    for preference in rows:
        if preference.control and control_policy == "exclude":
            excluded_controls.append(preference)
        else:
            by_user[preference.user_key].append(preference)
    support: list[AnuCarPreference] = []
    evaluation: list[AnuCarPreference] = []
    user_audit: dict[str, object] = {}
    for (experiment_id, user_id), user_rows in sorted(by_user.items()):
        groups: dict[tuple[str, str, str, str], list[AnuCarPreference]] = defaultdict(list)
        for preference in user_rows:
            groups[preference.unordered_pair_key].append(preference)
        minimum_groups = support_size_per_user + (1 if require_evaluation else 0)
        if len(groups) < minimum_groups:
            raise ValueError(
                f"User {_composite_id(experiment_id, user_id)} has {len(groups)} eligible "
                f"pair groups; {minimum_groups} required"
            )
        ordered_keys = sorted(
            groups,
            key=lambda key: (
                _stable_digest(seed, "support", *key),
                key,
            ),
        )
        support_keys = set(ordered_keys[:support_size_per_user])
        for key in ordered_keys:
            target = support if key in support_keys else evaluation
            target.extend(groups[key])
        label = _composite_id(experiment_id, user_id)
        user_audit[label] = {
            "eligible_rows": len(user_rows),
            "pair_groups": len(groups),
            "support_pair_groups": len(support_keys),
            "support_rows": sum(len(groups[key]) for key in support_keys),
            "evaluation_pair_groups": len(groups) - len(support_keys),
            "evaluation_rows": sum(
                len(groups[key]) for key in ordered_keys if key not in support_keys
            ),
            "repeated_pair_groups": sum(len(group) > 1 for group in groups.values()),
        }
    if not by_user:
        raise ValueError("No eligible non-control preferences remain")

    def stable_rows(values: Iterable[AnuCarPreference]) -> tuple[AnuCarPreference, ...]:
        return tuple(
            sorted(
                values,
                key=lambda row: (row.experiment_id, row.user_id, row.source_line),
            )
        )

    stable_support = stable_rows(support)
    stable_evaluation = stable_rows(evaluation)
    support_pairs = {row.unordered_pair_key for row in stable_support}
    evaluation_pairs = {row.unordered_pair_key for row in stable_evaluation}
    if support_pairs & evaluation_pairs:
        raise AssertionError("Support/evaluation pair-group leakage")
    return AnuSupportEvaluationSplit(
        support=stable_support,
        evaluation=stable_evaluation,
        excluded_controls=stable_rows(excluded_controls),
        support_size_per_user=support_size_per_user,
        seed=seed,
        audit={
            "algorithm": "sha256 deterministic unordered-pair grouping",
            "seed": seed,
            "support_size_unit": "unordered item-pair groups per user",
            "support_size_per_user": support_size_per_user,
            "control_policy": control_policy,
            "excluded_control_rows": len(excluded_controls),
            "require_evaluation": require_evaluation,
            "users": user_audit,
        },
    )


def anu_preferences_to_pairwise_observations(
    preferences: Iterable[AnuCarPreference],
    *,
    control_policy: str = "error",
) -> tuple[PairwiseObservation, ...]:
    """Convert archival rows without colliding IDs across experiments.

    Control observations were intentionally reversed repeats in the original
    questionnaire.  They are therefore rejected by default instead of being
    silently treated as ordinary preference labels.  Callers that have an
    audited exclusion policy may pass ``exclude``; ``include`` is available
    only for explicit diagnostic analyses.
    """

    if control_policy not in {"error", "exclude", "include"}:
        raise ValueError("control_policy must be 'error', 'exclude', or 'include'")
    output: list[PairwiseObservation] = []
    for preference in preferences:
        if preference.control:
            if control_policy == "error":
                raise ValueError(
                    "Control rows are present; filter them audibly or set an explicit "
                    "control_policy"
                )
            if control_policy == "exclude":
                continue
        experiment = preference.experiment_id
        first, second = sorted((preference.preferred, preference.nonpreferred))
        output.append(
            PairwiseObservation(
                user_id=f"anu:{experiment}:user:{preference.user_id}",
                comparison_id=(
                    f"anu:{experiment}:user:{preference.user_id}:"
                    f"source-line:{preference.source_line:06d}"
                ),
                preferred_id=f"anu:{experiment}:item:{preference.preferred}",
                nonpreferred_id=f"anu:{experiment}:item:{preference.nonpreferred}",
                group_id=(
                    f"anu:{experiment}:user:{preference.user_id}:"
                    f"unordered-pair:{first}:{second}"
                ),
                metadata={
                    "dataset": "anu_car_preferences",
                    "experiment_id": experiment,
                    "source_user_id": preference.user_id,
                    "source_preferred_item_id": preference.preferred,
                    "source_nonpreferred_item_id": preference.nonpreferred,
                    "control": preference.control,
                    "source_line": preference.source_line,
                },
            )
        )
    return tuple(output)


def anu_item_predicate_specs() -> tuple[PredicateSpec, ...]:
    """Typed schema for item features shown in the archival questionnaires."""

    return (
        PredicateSpec("BodyType", "body_type", "categorical"),
        PredicateSpec("Transmission", "transmission", "categorical"),
        PredicateSpec(
            "Engin Capacity",
            "engine_capacity_l",
            "numeric",
            unit="litre",
            source_note="The official CSV header spells 'Engin Capacity'.",
        ),
        PredicateSpec("Fuel Consumed", "fuel_consumed", "categorical"),
        PredicateSpec(
            "Engine/Transmission Layout",
            "engine_transmission_layout",
            "categorical",
            source_note="Collected only in experiment 2; structurally missing in experiment 1.",
        ),
    )


def anu_items_to_entity_records(
    items: Iterable[AnuCarItem],
) -> tuple[EntityRecord, ...]:
    """Convert alternatives to typed records matching qualified observation IDs."""

    records: list[EntityRecord] = []
    seen: set[str] = set()
    for item in items:
        entity_id = f"anu:{item.experiment_id}:item:{item.item_id}"
        if entity_id in seen:
            raise ValueError(f"Duplicate qualified ANU item ID: {entity_id}")
        seen.add(entity_id)
        try:
            values: dict[str, tuple[str | float, ...]] = {
                "body_type": (_BODY_TYPE[item.body_type],),
                "transmission": (_TRANSMISSION[item.transmission],),
                "engine_capacity_l": (float(item.engine_capacity_l),),
                "fuel_consumed": (_FUEL[item.fuel_consumed],),
            }
            if item.engine_transmission_layout is not None:
                values["engine_transmission_layout"] = (
                    _LAYOUT[item.engine_transmission_layout],
                )
        except KeyError as error:
            raise ValueError(
                f"Unmapped item code {error.args[0]!r} for {entity_id}"
            ) from error
        raw_values = {
            "body_type": (str(item.body_type),),
            "transmission": (str(item.transmission),),
            "engine_capacity_l": (str(item.engine_capacity_l),),
            "fuel_consumed": (str(item.fuel_consumed),),
        }
        missing: dict[str, str] = {}
        if item.engine_transmission_layout is None:
            missing["engine_transmission_layout"] = "not_collected_in_experiment_1"
        else:
            raw_values["engine_transmission_layout"] = (
                str(item.engine_transmission_layout),
            )
        records.append(
            EntityRecord(
                entity_id=entity_id,
                uri=f"urn:rgsmw:anu-car-preferences:{item.experiment_id}:item:{item.item_id}",
                values=values,
                raw_values=raw_values,
                missing=missing,
                source_file=f"{item.experiment_id}:items.csv",
            )
        )
    return tuple(sorted(records, key=lambda record: record.entity_id))
