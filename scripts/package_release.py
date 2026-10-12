"""Create an archive from reviewed files only, excluding local acquired data/runs."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_release import ROOT, digest, verified_files


def build_archive(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    files = verified_files(root)
    files += [root / "RELEASE_MANIFEST.json", root / "SHA256SUMS"]
    if output in files:
        raise ValueError("Output must not replace a release input")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive mode avoids accidentally replacing an earlier deposited ZIP.
    with output.open("xb") as stream:
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(files):
                archive.write(path, arcname=(Path("dtltr") / path.relative_to(root)).as_posix())
    checksum = digest(output)
    output.with_name(output.name + ".sha256").write_text(f"{checksum}  {output.name}\n", encoding="ascii")
    return {"archive": str(output), "sha256": checksum, "bytes": output.stat().st_size, "files": len(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(build_archive(ROOT, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
