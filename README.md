# Directional typed learning to rank (DT-LTR)

DT-LTR learns to rank entities described by numerical, ordinal, categorical
and set-valued properties. It compares each candidate with a reference profile,
constructs a feature group for each property, and learns an additive utility
from observed preferences. The resulting score gives both a ranking and an
exact decomposition into named property contributions.

This repository contains the implementation and experiments for **Directional
typed learning to rank for heterogeneous RDF entities**, evaluated on vehicle
preferences and a validated RDF catalogue.

## Method pipeline

![DT-LTR method pipeline: entity descriptions, a reference and observed choices feed typed feature construction, additive utility learning, ranking and property contributions.](docs/assets/dtltr-pipeline.svg)

*Pipeline reused from Figure 1 of the paper.*

| Stage in the figure | What the method does |
|---|---|
| **Input** | Read entity descriptions, a reference profile `q`, and preference observations. A declared registry identifies the properties and their analytical types. |
| **Feature construction** | Normalise values, retain missingness and construct reference-relative property groups. Numerical and ordinal groups distinguish deviations below and above the reference; categorical and set groups represent matching and overlap. |
| **Learning** | Fit coefficients from pairwise preferences or choices within a set. The additive utility combines the weighted property groups; optional support adaptation personalises the population coefficients. |
| **Outputs** | Rank candidates by decreasing utility. Property contributions sum exactly to the score, and their differences reconstruct a pairwise ranking margin. |

The vehicle experiments use archival ANU pairwise preferences, McFadden--Train
choice sets, and generated noisy comparisons over the bundled ADEME catalogue.
Matched symmetric, direct, spline and CatBoost controls assess the contribution
of the representation. Generated preferences are controlled mechanism tests.

## Install and check

Use Python **3.12** and [uv](https://docs.astral.sh/uv/). Run these commands from this folder in a Linux shell or Windows PowerShell;
the same commands work in both:

```sh
uv sync --frozen --extra dev
uv run --frozen --extra dev python scripts/verify_release.py
uv run --frozen --extra dev python scripts/demo.py
uv run --frozen --extra dev python -m pytest -q
```

The verifier checks the fixed release inventory. The demo fits and inspects a
small artificial example; its labels are not research evidence. The locked
environment uses `uv.lock`, which pins direct and transitive dependencies.
The recorded validation scope is in `VALIDATION.json`.

## First experiment: bundled data only

```sh
uv run --frozen --extra dev python -u scripts/run_experiment.py run --config configs/smoke.yaml --suite audit --suite generated --output runs/generated-smoke --workers 2
```

This runs **three tasks**: the RDF registry audit and two generated mechanisms.
All eight representations are evaluated in the generated tasks. No human-data
download is needed. Smoke runs check execution on small samples.

## Archival and full experiments

Read [data sources and terms](data/README.md). Acquisition is an explicit opt-in:

```sh
uv run --frozen --extra dev python scripts/acquire_data.py --dataset all --download
uv run --frozen --extra dev python -u scripts/run_experiment.py run --config configs/full.yaml --output runs/full --workers 4
```
