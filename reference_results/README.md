# Frozen numerical reference

These aggregate CSVs are byte-identical copies from the completed development
study. `metrics.csv` gives unit-aware outcomes, `contrasts.csv` matched
differences and conditional intervals, `calibration.csv` reliability bins,
and `strata.csv` descriptive subgroup outcomes. Column definitions and
aggregation are implemented in
[dtltr_vehicle.experiments.reporting](../src/dtltr_vehicle/experiments/reporting.py).

`frozen_study.json` preserves the original computational provenance, including
historical source paths and the run fingerprint. It is not a resume manifest
for this release. Per-session labels, predictions and coefficient vectors are
excluded from this reference folder. Fresh runs write to their own output
directories and do not modify these files.

Human intervals condition on the saved fitted procedure. Generated seeds are
descriptive mechanism replications, with seed-varying utility parameters in
five mechanisms. The archival outcomes are development reanalyses, not
independent validation. See the
[protocol](../study/EXPERIMENT_PROTOCOL.md) and
[method pipeline](../README.md#method-pipeline) before interpreting the reports.
