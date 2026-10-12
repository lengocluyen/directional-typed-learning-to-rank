"""Release integrity and exclusion checks with explicitly artificial fixtures."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from package_release import build_archive
from verify_release import verified_files


def test_local_implementation_imports_never_use_another_checkout():
    for name in ("dtltr_vehicle.basis", "dtltr_vehicle.learners", "dtltr_vehicle.open_catalogue",
                 "dtltr_vehicle.experiments.features", "dtltr_vehicle.experiments.models",
                 "dtltr_vehicle.experiments.human", "dtltr_vehicle.experiments.generated",
                 "rgsmw_next.anu_car_preferences", "rgsmw_next.mcfadden_train"):
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(ROOT / "src")


def test_numerical_reporting_has_no_figure_generator():
    from dtltr_vehicle.experiments import reporting
    assert not hasattr(reporting, "make_plots")


def test_archive_excludes_unlisted_archival_data_and_outputs_and_rejects_changes(tmp_path):
    root = tmp_path / "fixture"
    root.mkdir()
    payload = root / "README.md"
    payload.write_text("Artificial release fixture\n", encoding="utf-8")
    manifest = {"files": {"README.md": {"bytes": payload.stat().st_size,
                 "sha256": hashlib.sha256(payload.read_bytes()).hexdigest()}}}
    (root / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "SHA256SUMS").write_text("fixture\n", encoding="utf-8")
    for name in ("data/archival/anu/archives/fixture.zip", "runs/example/figures/fixture.png", ".env"):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("Artificial excluded file", encoding="utf-8")
    output = tmp_path / "release.zip"
    build_archive(root, output)
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {"dtltr/README.md", "dtltr/RELEASE_MANIFEST.json", "dtltr/SHA256SUMS"}
    with pytest.raises(FileExistsError):
        build_archive(root, output)
    payload.write_text("Changed fixture", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        verified_files(root)
