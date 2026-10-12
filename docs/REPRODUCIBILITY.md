# Reproducibility

This standalone release implements the experiments for *Directional typed
learning to rank for heterogeneous RDF entities*. Start with the commands in
the [README](../README.md), the [method pipeline](../README.md#method-pipeline) and the
[protocol](../study/EXPERIMENT_PROTOCOL.md). The active runner is
`scripts/run_experiment.py`; its implementation is `dtltr_vehicle.experiments`.
Configurations are `configs/smoke.yaml`, `configs/pilot.yaml`,
`configs/full.yaml` and `configs/ablation-smoke.yaml`.

## Frozen reference and fresh runs

The aggregate files in `reference_results/` come from the completed development
study (101/101 tasks). `frozen_study.json` records that study's fingerprint,
settings, input checksums and original source inventory. Its four CSV reports
retain their original numerical values. These historical identities describe
the computation that produced the reference; they are not current resume
manifests.

Encoders, estimators, penalties, support sampling, calibration, metrics and
controlled utility definitions preserve the experimental procedure. Release
changes, including the standalone package/entry-point paths, data paths and
removal of plotting code, are recorded in `SOURCE_LINEAGE.json`. Some transitive
dependencies may install a plotting library, but this release contains no
figure-generation scripts or functions.

A fresh run receives a new identity from source files, configuration, input
checksums and the Python/package environment. Use a new output directory for
each changed identity. `--resume` accepts only the same unchanged release run
and verifies saved completed tasks. It cannot resume the original development
study. `plan` saves an immutable manifest without fitting; executing that plan
requires `run` with the same arguments and `--resume`.

`VALIDATION.json` records the actual checks performed on this release and their
scope. Availability of the full configuration is distinct from rerunning the
expensive full study or obtaining independent observed validation.

## Experimental scope

The full profile retains eight representations and all four blocks crossing
raw/default reference completion with clipped/unclipped numerical coordinates.
ANU uses seven additive representations: CatBoost is excluded because tree
warm starts do not implement the same quadratic coefficient anchor.
Hyperparameter grids, seeds, role partitions and generator sizes are declared
in the configurations. An external suite creates no tasks unless a genuine
observed-data bundle is explicitly supplied.

Typed maps are fitted from training entities in generated, McFadden and external
experiments. ANU is the declared exception: label-free preprocessing uses its
known fixed catalogue of 10 cars in exp1 and 20 in exp2, so it measures
transductive known-item prediction.

TD learns nonnegative property weights constrained to sum to one, fixing its
overall coefficient scale. ST/DT coefficient magnitudes are fitted under
regularization. This distinction concerns utility scale, independently of the
fixed numerical preprocessing scales. A positive postfit probability
temperature preserves ranks while rescaling utilities for probability
prediction. No learned utility-scale TD comparator is included.

Generated query-dependent utilities define label and evaluation truth only;
their full-catalogue truth map is never passed to fitted methods, development
selection or calibration. Five of the eight mechanisms also draw utility
parameters by seed; repeats for those five vary both mechanisms and sampled data.
Generated clean-preference accuracy and noisy-label log loss are different
endpoints.

Human paired-bootstrap intervals condition on saved fits. Repeated splits do
not add independent participants, and the McFadden design proxy remains
confounded with presentation. Additive property contributions reconstruct
uncalibrated utility differences exactly; they describe the fitted model rather
than causal effects or observed psychological preferences. See the protocol
for the remaining interpretation limits.

## Optional catalogue rebuild

Use the bundled pinned files to reproduce the frozen inputs. Rebuilding is
optional and uses a new output directory:

```sh
uv run --frozen --extra dev python -c "from pathlib import Path; from dtltr_vehicle.open_catalogue import build; p=Path('runs/rebuilt-catalogue'); p.mkdir(parents=True); build(Path('data/catalogue/raw/ademe_car_labelling.csv'), p)"
```

A rebuild can change Turtle blank-node identifiers or ordering while preserving
graph meaning. The registry audit checks semantic correspondence and the
declared six-significant-digit numerical serialization. It does not claim exact
lexical preservation or infer arbitrary RDF schemas.
