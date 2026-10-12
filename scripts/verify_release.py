"""Verify the fixed public inventory using only the Python standard library."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def verified_files(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    entries = manifest["files"]
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Release inventory is empty or invalid")
    paths = []
    for name, entry in sorted(entries.items()):
        relative = Path(name)
        target = root / relative
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or not target.resolve().is_relative_to(root):
            raise ValueError(f"Unsafe release path: {name}")
        if target.is_symlink() or not target.is_file():
            raise ValueError(f"Missing/nonregular release file: {name}")
        if target.stat().st_size != entry["bytes"] or digest(target) != entry["sha256"]:
            raise ValueError(f"Release file changed: {name}")
        paths.append(target)
    return paths


def main():
    paths = verified_files()
    print(json.dumps({"status": "verified", "files": len(paths),
                      "bytes": sum(path.stat().st_size for path in paths)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
