# Data

| Dataset | Included | Source and terms |
|---|---|---|
| ADEME catalogue | CSV snapshot, 2,309 analytical records, RDF and SHACL | [ADEME](https://www.data.gouv.fr/datasets/ademe-car-labelling), Licence Ouverte 2.0; provenance in `catalogue/DATASET.json` |
| ANU car preferences | Pinned acquisition metadata and loader | [Official research page](https://users.cecs.anu.edu.au/~u4940058/CarPreferences.html); research use with citation is stated, but redistribution is not explicitly licensed |
| McFadden--Train car preferences | Pinned acquisition metadata and loader | [Choice-Learn packaging](https://github.com/artefactory/choice-learn/tree/e45b330a42ac2c3566ce9cf7cd9acf95b79df761/choice_learn/datasets/data); software MIT does not by itself establish a separate data licence |

Human-data acquisition is optional. The bundled ADEME catalogue supports the
audit/generated smoke command in the [run guide](../README.md). Installation,
the demo and the experiment runner do not download human archives.

To opt in, download the two human datasets directly into your local cache:

```sh
uv run --frozen --extra dev python scripts/acquire_data.py --dataset all --download
```

To work offline, place the exact `exp1-prefs.zip` and `exp2-prefs.zip` in
`data/archival/anu/archives/`, and `car.csv.gz` in
`data/archival/mcfadden/archives/`. Then omit `--download`:

```sh
uv run --frozen --extra dev python scripts/acquire_data.py --dataset all
```

All archives and extracted ANU files are checksum-verified. TLS verification
remains enabled. A changed file is rejected rather than substituted. No data
are downloaded by the experiment runner itself.

ANU: Abbasnejad, Sanner, Bonilla and Poupart (2013), *Learning Community-based
Preferences via Dirichlet Process Mixtures of Gaussian Processes*, IJCAI.
McFadden--Train data: preserve the provider's Brownstone--Train and
McFadden--Train bibliographic provenance in the loader. These archival results
are development reanalyses. ADEME generated preferences are controlled
mechanisms, not newly observed human judgments.

Human archive checksums:

- `exp1-prefs.zip`: `e3ab308d4cfbf01d64e4b8c80aaf15ae2b724c95554b80a6f1545b2d20440082`
- `exp2-prefs.zip`: `ccbef141b41eb41550dc4ff98163386111db6c31682bfe81535d01d3e30979d8`
- `car.csv.gz`: `cbdab01357dab84e19f70c2761a6b6421f73bde530987007a4dbdfb642f0a554`
