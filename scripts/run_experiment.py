"""Run DT-LTR experiments from this checkout, including on Windows spawn workers."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")

from dtltr_vehicle.experiments.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
