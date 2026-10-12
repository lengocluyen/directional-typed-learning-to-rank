"""Immutable plans, resumable tasks, source/data provenance and summary exports."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import tarfile
import traceback

import yaml

from dtltr_vehicle.provenance import environment, sha256
from . import SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[3]
MODELS = {"TD", "ST", "DT", "direct", "direct_ref", "spline", "spline_ref", "catboost_ref"}
SUITES = {"audit", "anu", "mcfadden", "generated", "external"}


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(value):
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_json_default)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=_json_default), encoding="utf-8")
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_settings(path):
    settings = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("Configuration must be a mapping")
    for name in ("seeds", "models", "suites", "completion", "clipping"):
        if not isinstance(settings.get(name), list) or not settings[name]:
            raise ValueError(f"{name} must be a nonempty list")
        if len(set(settings[name])) != len(settings[name]):
            raise ValueError(f"{name} contains duplicate values")
    if set(settings["models"]) - MODELS:
        raise ValueError("Unknown model in configuration")
    if set(settings["suites"]) - SUITES:
        raise ValueError("Unknown suite in configuration")
    if set(settings["completion"]) - {"raw", "default"} or set(settings["clipping"]) - {"clipped", "unclipped"}:
        raise ValueError("Unknown encoding policy")
    if any(not isinstance(s, int) or isinstance(s, bool) or s < 0 for s in settings["seeds"]):
        raise ValueError("Seeds must be nonnegative integers")
    if int(settings.get("folds", 5)) < 1 or int(settings.get("support_draws", 1)) < 1:
        raise ValueError("folds and support_draws must be positive")
    budgets = settings.get("budgets", [0, 2, 5, 10, 20])
    if budgets != sorted(set(budgets)) or any(not isinstance(b, int) or b < 0 for b in budgets):
        raise ValueError("Budgets must be sorted unique nonnegative integers")
    for name in ("lam", "gamma", "mu"):
        values = settings.get("penalties", {}).get(name)
        if not values or any(not isinstance(v, (float, int)) or not 0 <= v < float("inf") for v in values):
            raise ValueError(f"penalties.{name} must contain finite nonnegative numbers")
    for name in ("anu_models", "mcfadden_models", "generated_models", "external_models"):
        if name in settings and (not settings[name] or set(settings[name]) - MODELS):
            raise ValueError(f"Invalid {name}")
    return settings


def source_hashes():
    files = list((ROOT / "src" / "dtltr_vehicle").rglob("*.py"))
    files += list((ROOT / "src" / "rgsmw_next").glob("*.py"))
    files += [p for p in (ROOT / "src" / "rgsmw_next" / "config").rglob("*") if p.is_file()]
    files += [ROOT / "scripts" / "run_experiment.py", ROOT / "pyproject.toml", ROOT / "uv.lock",
              ROOT / "study" / "EXPERIMENT_PROTOCOL.md", ROOT / "requirements.txt",
              ROOT / "scripts" / "acquire_data.py", ROOT / "scripts" / "demo.py"]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(files) if p.is_file()}


def resolve_settings(settings):
    out = json.loads(canonical(settings))
    out["paths"] = {k: str((ROOT / v).resolve()) if not Path(v).is_absolute() else str(Path(v).resolve())
                    for k, v in settings.get("paths", {}).items() if v}
    return out


def input_hashes(settings):
    needed = {"audit": ["records", "rdf"], "generated": ["records"], "anu": ["anu"],
              "mcfadden": ["mcfadden"], "external": ["external"]}
    names = {k for suite in settings["suites"] for k in needed[suite]}
    hashes = {}
    for name in sorted(names):
        value = settings.get("paths", {}).get(name)
        if not value:
            if name == "external":
                continue
            raise ValueError(f"Missing paths.{name}")
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"Input {name}: {path}")
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        if not files:
            raise ValueError(f"Empty input directory: {path}")
        hashes[name] = {(p.relative_to(path).as_posix() if path.is_dir() else path.name): sha256(p) for p in files}
    return hashes


def task_plan(settings):
    tasks = []
    for suite in settings["suites"]:
        if suite == "external" and not settings.get("paths", {}).get("external"):
            continue
        if suite == "audit":
            tasks.append({"suite": suite})
        for seed in settings["seeds"]:
            if suite == "anu":
                for experiment in settings.get("anu_experiments", ["exp1", "exp2"]):
                    for fold in range(int(settings.get("folds", 5))):
                        tasks.append({"suite": suite, "seed": seed, "experiment": experiment, "fold": fold})
            elif suite == "mcfadden":
                for reference in settings.get("mcfadden_references", ["catalogue", "design_proxy"]):
                    tasks.append({"suite": suite, "seed": seed, "reference": reference})
            elif suite == "generated":
                for rule in settings.get("rules", ["gain_loss"]):
                    tasks.append({"suite": suite, "seed": seed, "rule": rule})
            elif suite == "external":
                tasks.append({"suite": suite, "seed": seed})
    for task in tasks:
        task["id"] = hashlib.sha256(canonical(task).encode()).hexdigest()[:20]
    if not tasks:
        raise ValueError("No executable tasks selected")
    return tasks


def output_guard(output):
    output = Path(output).resolve()
    if output == ROOT or output == ROOT / "runs":
        raise ValueError("Output must be a dedicated new experiment directory")
    for legacy in ("dtltr_vehicle_v2", "archival_frozen_20260826", "archival_secondary_20260826",
                   "paper1_typed_extension_20260827", "paper1_modern_baselines_20260827_corrected"):
        if output.is_relative_to(ROOT / "runs" / legacy):
            raise ValueError("The frozen earlier studies cannot be overwritten")
    return output


def prepare(config, output, suites=None, external=None, resume=False):
    declared = load_settings(config)
    if suites:
        declared["suites"] = suites
    if external:
        declared.setdefault("paths", {})["external"] = str(Path(external).resolve())
        if "external" not in declared["suites"]:
            declared["suites"].append("external")
    settings = resolve_settings(declared)
    data = input_hashes(settings)
    sources = source_hashes()
    env = environment()
    identity = {"schema": SCHEMA_VERSION, "configuration": declared, "source_files": sources,
                "input_files": data, "packages": env["packages"], "python": env["python"]}
    fingerprint = hashlib.sha256(canonical(identity).encode()).hexdigest()
    output = output_guard(output)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"Run exists: {output}; use --resume with the unchanged configuration")
        old = read_json(manifest_path)
        if old.get("fingerprint") != fingerprint:
            raise ValueError("Resume rejected: configuration, code, data or runtime changed. Use a NEW output directory.")
        return settings, output, old
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Nonempty output without a manifest: {output}")
    output.mkdir(parents=True, exist_ok=True)
    tasks = task_plan(settings)
    manifest = {"schema": SCHEMA_VERSION, "fingerprint": fingerprint, "status": "planned", "created_utc": utc(),
                "environment": env, "identity": identity, "tasks": tasks, "completed": {}, "summary_hashes": {}, "failed": {},
                "pending_evidence": ["independent human reference/presentation validation"],
                "interpretation": "Development reanalysis and generated mechanisms; never independent confirmatory evidence merely because splits are held out."}
    if not settings.get("paths", {}).get("external"):
        manifest["pending_evidence"].append("independent external domain: no bundle supplied")
    atomic_json(output / "configuration.json", settings)
    atomic_json(manifest_path, manifest)
    return settings, output, manifest


def execute_task(task, settings):
    suite = task["suite"]
    if suite == "audit":
        from .data import audit_registry
        audit = audit_registry(settings)
        if audit.get("status") != "pass":
            raise ValueError(f"RDF registry audit failed: {audit}")
        result = {"diagnostics": {"registry_audit": audit}}
    elif suite == "anu":
        from .human import run_anu
        result = run_anu(settings, task["seed"], task["experiment"], task["fold"])
    elif suite == "mcfadden":
        from .human import run_mcfadden
        result = run_mcfadden(settings, task["seed"], task["reference"])
    elif suite == "generated":
        from .generated import run_generated
        result = run_generated(settings, task["seed"], task["rule"])
    else:
        from .generated import run_external
        result = run_external(settings, task["seed"])
    result["planned_task"] = task
    for name in ("metrics", "predictions", "fits"):
        result.setdefault(name, [])
    canonical(result)  # Reject NaN/Infinity before accepting a task.
    return result


def verified_task_path(output, manifest, task_id, compact_ok=False):
    output = Path(output)
    path = output / "tasks" / (task_id + ".json")
    expected = manifest["completed"].get(task_id)
    if not path.is_file() and compact_ok:
        path = output / "task_summaries" / (task_id + ".json")
        expected = manifest.get("summary_hashes", {}).get(task_id)
    if not expected or not path.is_file() or sha256(path) != expected:
        raise ValueError(f"Completed task corrupted or missing: {task_id}; preserve this run and investigate")
    return path


def _process_alive(pid):
    if os.name == "nt":
        # os.kill(pid, 0) can terminate a Windows process. Query a handle instead.
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            error = ctypes.get_last_error()
            if error == 87:  # ERROR_INVALID_PARAMETER: no such process.
                return False
            raise OSError(error, "Cannot inspect the existing Windows runner")
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise ctypes.WinError(ctypes.get_last_error())
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False


def _acquire_lock(output):
    lock = output / ".running.lock"
    if lock.exists():
        old = read_json(lock)
        if old.get("host") == socket.gethostname():
            try:
                alive = _process_alive(old["pid"])
            except OSError as exc:
                raise RuntimeError("Cannot verify the previous runner; inspect .running.lock") from exc
            if alive:
                raise RuntimeError(f"Another runner is active (PID {old['pid']})")
            lock.unlink()
        else:
            raise RuntimeError("Run locked on another host; verify it has stopped before removing .running.lock")
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(canonical({"pid": os.getpid(), "host": socket.gethostname(), "utc": utc()}))
    return lock


def run(config, output, workers=1, resume=False, suites=None, external=None):
    if workers < 1:
        raise ValueError("workers must be positive")
    settings, output, manifest = prepare(config, output, suites, external, resume)
    lock = _acquire_lock(output)
    try:
        pending = []
        for task in manifest["tasks"]:
            saved = manifest["completed"].get(task["id"])
            if saved:
                verified_task_path(output, manifest, task["id"])
            else:
                pending.append(task)
        manifest["status"] = "running"
        manifest["last_started_utc"] = utc()
        atomic_json(output / "manifest.json", manifest)
        print(f"{len(manifest['tasks'])} tasks; {len(pending)} pending; {workers} CPU workers", flush=True)

        def persist(task, result):
            path = output / "tasks" / (task["id"] + ".json")
            atomic_json(path, result)
            summary = {k: v for k, v in result.items() if k != "predictions"}
            from .reporting import compact_diagnostics
            summary["diagnostics"] = compact_diagnostics(result.get("diagnostics", {}))
            summary["prediction_rows"] = len(result["predictions"])
            atomic_json(output / "task_summaries" / path.name, summary)
            manifest["completed"][task["id"]] = sha256(path)
            manifest.setdefault("summary_hashes", {})[task["id"]] = sha256(output / "task_summaries" / path.name)
            manifest["failed"].pop(task["id"], None)
            atomic_json(output / "manifest.json", manifest)
            print(f"completed {len(manifest['completed'])}/{len(manifest['tasks'])}: {task}", flush=True)

        def fail(task, exc):
            manifest["status"] = "failed"
            manifest["failed"][task["id"]] = {"error": str(exc), "traceback": traceback.format_exc(), "utc": utc()}
            atomic_json(output / "manifest.json", manifest)
            print(f"FAILED {task}: {exc}\nSaved failure in {output / 'manifest.json'}; cancelling queued tasks.", flush=True)

        if workers == 1:
            for task in pending:
                try:
                    persist(task, execute_task(task, settings))
                except Exception as exc:
                    fail(task, exc)
                    raise
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(execute_task, task, settings): task for task in pending}
                for future in as_completed(futures):
                    # Release each completed Future and its potentially large result.
                    task = futures.pop(future)
                    try:
                        persist(task, future.result())
                    except Exception as exc:
                        fail(task, exc)
                        for other in futures:
                            other.cancel()
                        raise
        manifest["status"] = "complete"
        manifest["finished_utc"] = utc()
        atomic_json(output / "manifest.json", manifest)
        from .reporting import summarize
        result = summarize(output)
        print(canonical(result), flush=True)
        return result
    finally:
        lock.unlink(missing_ok=True)


def export_summary(output):
    output = output_guard(output)
    manifest = read_json(output / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Export requires a complete run; summarize can review partial runs")
    for task in manifest["tasks"]:
        verified_task_path(output, manifest, task["id"])
        summary = output / "task_summaries" / (task["id"] + ".json")
        if not summary.is_file() or sha256(summary) != manifest.get("summary_hashes", {}).get(task["id"]):
            raise ValueError(f"Compact task summary corrupted or missing: {task['id']}")
    files = [output / "manifest.json", output / "configuration.json"]
    for folder in ("reports", "task_summaries"):
        files += [p for p in (output / folder).rglob("*") if p.is_file()]
    destination = output / "exports" / (output.name + "-summary.tgz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        for path in sorted(files):
            archive.add(path, arcname=(Path(output.name) / path.relative_to(output)).as_posix())
    checksum = sha256(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(f"{checksum}  {destination.name}\n", encoding="utf-8")
    return {"archive": str(destination), "sha256": checksum, "bytes": destination.stat().st_size, "files": len(files)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--suite", choices=sorted(SUITES), action="append")
        p.add_argument("--external-bundle")
        p.add_argument("--resume", action="store_true")
        if name == "run":
            p.add_argument("--workers", type=int, default=1)
    for name in ("summarize", "export"):
        p = sub.add_parser(name)
        p.add_argument("--output", required=True)
    p = sub.add_parser("validate-bundle")
    p.add_argument("--bundle", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.config, args.output, args.workers, args.resume, args.suite, args.external_bundle)
    elif args.command == "plan":
        _, output, manifest = prepare(args.config, args.output, args.suite, args.external_bundle, args.resume)
        print(canonical({"output": str(output), "tasks": len(manifest["tasks"]), "fingerprint": manifest["fingerprint"],
                         "pending_evidence": manifest["pending_evidence"]}))
    elif args.command == "summarize":
        from .reporting import summarize
        print(canonical(summarize(args.output)))
    elif args.command == "export":
        print(canonical(export_summary(args.output)))
    else:
        from .data import load_external_bundle
        bundle = load_external_bundle(args.bundle)
        print(canonical({"status": "validated", "bundle": str(Path(args.bundle).resolve()),
                         "type": type(bundle).__name__}))
    return 0
