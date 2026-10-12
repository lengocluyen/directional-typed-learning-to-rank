"""Run manifests: source hashes, environment and timing."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("numpy", "scipy", "cvxpy", "clarabel", "pandas", "rdflib", "pyshacl", "catboost")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_inventory() -> dict[str, str]:
    files = sorted((ROOT / "src" / "dtltr_vehicle").glob("*.py")) + [
        ROOT / "scripts" / "run_dtltr_vehicle.py",
        ROOT / "study" / "DTLTR_VEHICLE_V2_PROTOCOL.md",
    ]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in files if p.exists()}


def environment() -> dict[str, object]:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": packages}


def write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=1, sort_keys=True, default=_default)
    path.write_text(text, encoding="utf-8")
    return sha256(path)


def _default(value: object) -> object:
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def write_manifest(output: Path, study: str, outputs: Mapping[str, str], started: datetime, extra: Mapping[str, object]) -> None:
    write_json(
        output / "manifest.json",
        {
            "study": study,
            "status": "complete",
            "protocol": "study/DTLTR_VEHICLE_V2_PROTOCOL.md",
            "started_utc": started.isoformat(timespec="seconds"),
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "environment": environment(),
            "source_files": source_inventory(),
            "outputs": dict(outputs),
            **extra,
        },
    )
