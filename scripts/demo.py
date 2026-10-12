"""Small artificial example of fitting, ranking and exact score inspection."""
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np
from dtltr_vehicle.basis import PropertySpec
from dtltr_vehicle.experiments.features import Encoder
from dtltr_vehicle.experiments.models import fit_pair
from rgsmw_next.types import EntityRecord

def main():
    # Artificial numbers and labels: demonstration only, not paper evidence.
    records = [EntityRecord(str(i), f"urn:demo:{i}",
               {"cost": (float(10+i),), "category": ("a" if i % 2 else "b",),
                "tags": ("x", "y") if i % 2 else ("x",)}) for i in range(8)]
    specs = [PropertySpec("cost", "numeric"), PropertySpec("category", "categorical"), PropertySpec("tags", "set")]
    encoder = Encoder.fit(records, specs, "DT")
    features = encoder.matrix(records, encoder.default_reference())
    pairs = [(a,b) for a in range(8) for b in range(a+1,8)]
    fit = fit_pair(np.array([features[a]-features[b] for a,b in pairs]),
                   {"lam": .1, "gamma": 0., "mu": 0.}, encoder.groups, model="DT")
    utilities = fit.score(features)
    contribution = fit.contributions(features[0]-features[1], encoder)
    reconstruction_error = abs(sum(contribution.values())-float(utilities[0]-utilities[1]))
    assert reconstruction_error < 1e-10
    print(json.dumps({"label_origin": "artificial demonstration", "model": "DT-LTR",
                      "ranking": [records[i].entity_id for i in np.argsort(-utilities)],
                      "pairwise_property_contributions": contribution,
                      "reconstruction_error": reconstruction_error}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
