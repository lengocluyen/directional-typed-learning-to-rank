"""Acquire or verify checksum-pinned human archives; never part of a release ZIP."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", "anu", "mcfadden"), default="all")
    parser.add_argument("--download", action="store_true", help="Allow verified HTTPS acquisition if a cache is missing")
    args = parser.parse_args()
    results = {}
    if args.dataset in ("all", "anu"):
        from rgsmw_next.anu_car_preferences import acquire_anu_car_preferences
        result = acquire_anu_car_preferences(ROOT / "data/archival/anu", download=args.download)
        results["anu"] = {"status": "verified", "archives": len(result["archives"])}
    if args.dataset in ("all", "mcfadden"):
        from rgsmw_next.mcfadden_train import acquire_mcfadden_train
        result = acquire_mcfadden_train(ROOT / "data/archival/mcfadden", download=args.download)
        results["mcfadden"] = {"status": "verified", "sha256": result["sha256"]}
    print(json.dumps(results, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
