from pathlib import Path

import numpy as np
import pytest
from pyshacl import validate
from rdflib import Graph

from dtltr_vehicle.generated_study import DIRECTION, LOSS_AVERSION, Rules, load_records, predicate_specs
from dtltr_vehicle.open_catalogue import OV, SHAPES, completeness_audit, load_rows, to_graph
from rgsmw_next.distance import FittedMetric
from rgsmw_next.types import EntityRecord

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "catalogue" / "raw" / "ademe_car_labelling.csv"

pytestmark = pytest.mark.skipif(not CSV.exists(), reason="ADEME snapshot not present")


@pytest.fixture(scope="module")
def rows():
    return load_rows(CSV)


def test_deduplication_and_missing_semantics(rows):
    records, audit = rows
    assert audit["source_rows"] == audit["distinct_vehicles"] + audit["exact_duplicates_removed"]
    assert completeness_audit(records)["violations"] == 0
    for row in records:
        powertrain = row["values"]["powertrain"]
        if powertrain == "BEV":
            assert row["missing"].get("fuel_consumption_l100km") == "not_applicable"
            assert row["values"]["energy_carrier"] == ("electricity",)
        if powertrain == "ICE":
            assert row["missing"].get("electric_range_km") == "not_applicable"


def test_subset_graph_conforms_and_keeps_sets(rows):
    records, _ = rows
    subset = [r for r in records if r["values"]["powertrain"] in ("PHEV", "ICE")][:40]
    graph = to_graph(subset)
    shapes = Graph().parse(data=SHAPES, format="turtle")
    conforms, _, report = validate(graph, shacl_graph=shapes, inference="none")
    assert conforms, report
    phev = next(r for r in subset if r["values"]["powertrain"] == "PHEV")
    subject = OV[f"vehicle/{phev['entity_id'].split(':')[1]}"]
    assert len(list(graph.objects(subject, OV.energyCarrier))) == 2


def test_gain_loss_rule_is_loss_averse():
    def car(price):
        return EntityRecord("x", "urn:x", {"price_eur": (price,), "mass_kg": (1500.0,), "powertrain": ("ICE",)})

    records = [car(p) for p in (20000.0, 30000.0, 40000.0, 50000.0)]
    metric = FittedMetric.fit(records, [s for s in predicate_specs() if s.field in ("price_eur", "mass_kg", "powertrain")])
    rules = Rules(records, metric, seed=1)
    reference = car(30000.0)
    gain = rules.utility("gain_loss", reference, car(25000.0)) - rules.utility("gain_loss", reference, reference)
    loss = rules.utility("gain_loss", reference, reference) - rules.utility("gain_loss", reference, car(35000.0))
    assert DIRECTION["price_eur"] == -1
    assert loss == pytest.approx(LOSS_AVERSION * gain)
