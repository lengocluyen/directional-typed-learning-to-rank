"""Open RDF vehicle catalogue from ADEME Car Labelling (Licence Ouverte 2.0).

Each distinct vehicle version becomes an ``ov:Vehicle`` with typed literals,
IRI-valued energy carriers (a set-valued property), and explicit missing-value
annotations that distinguish "not applicable" (e.g. electric range of a petrol
car) from "not reported". The graph is validated against SHACL shapes with
pySHACL (no inference).
"""

from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path

import pandas as pd
from pyshacl import validate
from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, RDF, RDFS, XSD

OV = Namespace("https://w3id.org/rgsmw/open-vehicle/")
SOURCE = URIRef("https://www.data.gouv.fr/datasets/ademe-car-labelling")
LICENCE = URIRef("https://www.etalab.gouv.fr/licence-ouverte-open-licence/")

SEGMENTS = ("ECONOMIQUE", "INFERIEURE", "MOYENNE INFERIEURE", "MOYENNE SUPERIEURE", "SUPERIEURE", "LUXE")
ENERGY = {
    "ESSENCE": (("petrol",), "ICE"),
    "GAZOLE": (("diesel",), "ICE"),
    "SUPERETHANOL": (("ethanol_e85", "petrol"), "ICE"),
    "ESS+G.P.L.": (("lpg", "petrol"), "ICE"),
    "ESS+ELEC HNR": (("electricity", "petrol"), "HEV"),
    "GAZ+ELEC HNR": (("diesel", "electricity"), "HEV"),
    "ELEC+ESSENC HR": (("electricity", "petrol"), "PHEV"),
    "ELEC+GAZOLE HR": (("diesel", "electricity"), "PHEV"),
    "ELECTRIC": (("electricity",), "BEV"),
}
# field -> (RDF property local name, kind)
PROPERTIES = {
    "price_eur": ("priceEur", "numeric"),
    "energy_carrier": ("energyCarrier", "set"),
    "powertrain": ("powertrain", "categorical"),
    "body_type": ("bodyType", "categorical"),
    "segment": ("segment", "ordinal"),
    "brand": ("brand", "categorical"),
    "gearbox": ("gearbox", "categorical"),
    "engine_displacement_cc": ("engineDisplacementCc", "numeric"),
    "max_power_kw": ("maxPowerKw", "numeric"),
    "mass_kg": ("massKg", "numeric"),
    "co2_g_km": ("co2GramPerKm", "numeric"),
    "fuel_consumption_l100km": ("fuelConsumptionL100km", "numeric"),
    "electric_range_km": ("electricRangeKm", "numeric"),
    "fiscal_power_cv": ("fiscalPowerCv", "count"),
}
OPTIONAL_NUMERIC = ("engine_displacement_cc", "max_power_kw", "co2_g_km", "fuel_consumption_l100km", "electric_range_km")


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def _number(value: object) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().replace(" ", "").replace(",", ".")
    if not text:
        return None
    number = float(text)
    return number if math.isfinite(number) else None


def _mean(*values: object) -> float | None:
    numbers = [n for n in (_number(v) for v in values) if n is not None]
    return float(sum(numbers) / len(numbers)) if numbers else None


def load_rows(csv_path: Path) -> tuple[list[dict], dict[str, int]]:
    frame = pd.read_csv(csv_path, sep=";", encoding="utf-8", dtype=str, keep_default_na=False)
    frame.columns = [_ascii(c) for c in frame.columns]
    rows: list[dict] = []
    seen: set[str] = set()
    audit = {"source_rows": int(len(frame)), "exact_duplicates_removed": 0}
    for index, raw in frame.iterrows():
        key = json.dumps(raw.to_dict(), sort_keys=True, ensure_ascii=False)
        if key in seen:
            audit["exact_duplicates_removed"] += 1
            continue
        seen.add(key)
        energy = raw["Energie"].strip()
        if energy not in ENERGY:
            raise ValueError(f"Unmapped energy label {energy!r}")
        carriers, powertrain = ENERGY[energy]
        plug_in = powertrain in ("PHEV", "BEV")
        combustion = powertrain != "BEV"
        displacement = _number(raw["Cylindree"])
        values: dict[str, object] = {
            "price_eur": _number(raw["Prix vehicule"]),
            "energy_carrier": carriers,
            "powertrain": powertrain,
            "body_type": raw["Carrosserie"].strip() or None,
            "segment": raw["Gamme"].strip() or None,
            "brand": raw["Marque"].strip() or None,
            "gearbox": raw["Type de boite"].strip() or None,
            "engine_displacement_cc": displacement if displacement and displacement > 0 else None,
            "max_power_kw": _number(raw["Puissance maximale"]),
            "mass_kg": _number(raw["Poids a vide"]),
            "co2_g_km": _mean(raw["CO2 vitesse mixte Min"], raw["CO2 vitesse mixte Max"]),
            "fuel_consumption_l100km": _mean(raw["Conso vitesse mixte Min"], raw["Conso vitesse mixte Max"]),
            "electric_range_km": _mean(raw["Autonomie elec Min"], raw["Autonomie elec Max"]),
            "fiscal_power_cv": _number(raw["Puissance fiscale"]),
        }
        applicable = {
            "engine_displacement_cc": combustion,
            "fuel_consumption_l100km": combustion,
            "electric_range_km": plug_in,
            "co2_g_km": True,
            "max_power_kw": True,
        }
        missing = {
            field: ("not_applicable" if not applicable[field] else "not_reported")
            for field in OPTIONAL_NUMERIC
            if values[field] is None
        }
        # A combustion or plug-in value reported where it cannot apply is treated as not applicable.
        for field in OPTIONAL_NUMERIC:
            if values[field] is not None and not applicable[field]:
                values[field] = None
                missing[field] = "not_applicable"
        if values["price_eur"] is None or values["mass_kg"] is None:
            raise ValueError(f"Row {index} lacks price or mass")
        if values["segment"] is not None and values["segment"] not in SEGMENTS:
            raise ValueError(f"Unknown segment {values['segment']!r}")
        rows.append(
            {
                "entity_id": f"ademe:{len(rows):05d}",
                "source_row": int(index) + 2,
                "label": f"{raw['Marque'].strip()} {raw['Description Commerciale'].strip()}",
                "values": values,
                "missing": missing,
            }
        )
    audit["distinct_vehicles"] = len(rows)
    return rows, audit


SHAPES = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix ov: <https://w3id.org/rgsmw/open-vehicle/> .

ov:VehicleShape a sh:NodeShape ;
  sh:targetClass ov:Vehicle ;
  sh:property [ sh:path ov:priceEur ; sh:minCount 1 ; sh:maxCount 1 ; sh:datatype xsd:decimal ; sh:minExclusive 0 ] ;
  sh:property [ sh:path ov:massKg ; sh:minCount 1 ; sh:maxCount 1 ; sh:datatype xsd:decimal ; sh:minExclusive 0 ] ;
  sh:property [ sh:path ov:energyCarrier ; sh:minCount 1 ; sh:nodeKind sh:IRI ] ;
  sh:property [ sh:path ov:powertrain ; sh:minCount 1 ; sh:maxCount 1 ; sh:in ( "ICE" "HEV" "PHEV" "BEV" ) ] ;
  sh:property [ sh:path ov:segment ; sh:maxCount 1 ;
                sh:in ( "ECONOMIQUE" "INFERIEURE" "MOYENNE INFERIEURE" "MOYENNE SUPERIEURE" "SUPERIEURE" "LUXE" ) ] ;
  sh:property [ sh:path ov:bodyType ; sh:maxCount 1 ; sh:nodeKind sh:Literal ] ;
  sh:property [ sh:path ov:brand ; sh:minCount 1 ; sh:maxCount 1 ; sh:nodeKind sh:Literal ] ;
  sh:property [ sh:path ov:gearbox ; sh:maxCount 1 ; sh:nodeKind sh:Literal ] ;
  sh:property [ sh:path ov:engineDisplacementCc ; sh:maxCount 1 ; sh:datatype xsd:decimal ; sh:minExclusive 0 ] ;
  sh:property [ sh:path ov:maxPowerKw ; sh:maxCount 1 ; sh:datatype xsd:decimal ; sh:minExclusive 0 ] ;
  sh:property [ sh:path ov:co2GramPerKm ; sh:maxCount 1 ; sh:datatype xsd:decimal ; sh:minInclusive 0 ] ;
  sh:property [ sh:path ov:fuelConsumptionL100km ; sh:maxCount 1 ; sh:datatype xsd:decimal ; sh:minInclusive 0 ] ;
  sh:property [ sh:path ov:electricRangeKm ; sh:maxCount 1 ; sh:datatype xsd:decimal ; sh:minExclusive 0 ] ;
  sh:property [ sh:path ov:fiscalPowerCv ; sh:maxCount 1 ; sh:datatype xsd:decimal ; sh:minExclusive 0 ] ;
  sh:property [ sh:path ov:missingValue ; sh:node ov:MissingValueShape ] .

ov:MissingValueShape a sh:NodeShape ;
  sh:property [ sh:path ov:onProperty ; sh:minCount 1 ; sh:maxCount 1 ; sh:nodeKind sh:IRI ] ;
  sh:property [ sh:path ov:reason ; sh:minCount 1 ; sh:maxCount 1 ; sh:in ( "not_applicable" "not_reported" ) ] .
"""


def to_graph(rows: list[dict]) -> Graph:
    graph = Graph()
    graph.bind("ov", OV)
    graph.bind("dcterms", DCTERMS)
    dataset = OV["dataset/ademe-car-labelling"]
    graph.add((dataset, DCTERMS.source, SOURCE))
    graph.add((dataset, DCTERMS.license, LICENCE))
    for row in rows:
        subject = OV[f"vehicle/{row['entity_id'].split(':')[1]}"]
        graph.add((subject, RDF.type, OV.Vehicle))
        graph.add((subject, RDFS.label, Literal(row["label"])))
        graph.add((subject, DCTERMS.isPartOf, dataset))
        for field, value in row["values"].items():
            local, kind = PROPERTIES[field]
            if value is None:
                continue
            if kind == "set":
                for carrier in value:
                    graph.add((subject, OV[local], OV[f"energy/{carrier}"]))
            elif kind in ("numeric", "count"):
                graph.add((subject, OV[local], Literal(f"{float(value):.6g}", datatype=XSD.decimal)))
            else:
                graph.add((subject, OV[local], Literal(str(value))))
        for field, reason in row["missing"].items():
            node = BNode()
            graph.add((subject, OV.missingValue, node))
            graph.add((node, OV.onProperty, OV[PROPERTIES[field][0]]))
            graph.add((node, OV.reason, Literal(reason)))
    return graph


def completeness_audit(rows: list[dict]) -> dict[str, object]:
    """Every optional numeric property is either observed or explicitly annotated."""
    violations = [
        (row["entity_id"], field)
        for row in rows
        for field in OPTIONAL_NUMERIC
        if (row["values"][field] is None) != (field in row["missing"])
    ]
    coverage = {
        field: sum(row["values"][field] is not None and row["values"][field] != () for row in rows) / len(rows)
        for field in PROPERTIES
    }
    reasons = {
        field: {r: sum(row["missing"].get(field) == r for row in rows) for r in ("not_applicable", "not_reported")}
        for field in OPTIONAL_NUMERIC
    }
    return {"violations": len(violations), "coverage": coverage, "missing_reasons": reasons}


def build(csv_path: Path, output: Path) -> list[str]:
    rows, audit = load_rows(Path(csv_path))
    graph = to_graph(rows)
    graph.serialize(output / "vehicles.ttl", format="turtle")
    (output / "shapes.ttl").write_text(SHAPES.strip() + "\n", encoding="utf-8")
    shapes = Graph().parse(data=SHAPES, format="turtle")
    conforms, _, report = validate(graph, shacl_graph=shapes, inference="none")
    summary = {
        **audit,
        "triples": len(graph),
        "shacl_conforms": bool(conforms),
        "shacl_report_head": report[:2000],
        "completeness": completeness_audit(rows),
        "licence": str(LICENCE),
        "source": str(SOURCE),
    }
    (output / "catalogue_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (output / "records.json").write_text(json.dumps(rows, indent=0, ensure_ascii=False), encoding="utf-8")
    if not conforms:
        raise RuntimeError("SHACL validation failed; see catalogue_summary.json")
    return ["vehicles.ttl", "shapes.ttl", "catalogue_summary.json", "records.json"]
