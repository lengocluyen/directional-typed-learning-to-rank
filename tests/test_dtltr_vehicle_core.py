import math

import cvxpy as cp
import numpy as np
import pytest

from dtltr_vehicle.basis import PropertySpec, Reference, TypedBasis
from dtltr_vehicle.learners import Penalty, fit_choice, fit_pairwise, fit_temperature, pairwise_metrics
from rgsmw_next.types import EntityRecord


def _records():
    rows = [
        ("a", 10.0, "red", ("x", "y"), "low"),
        ("b", 20.0, "blue", ("y",), "mid"),
        ("c", 30.0, "red", (), "high"),
        ("d", None, "green", ("x",), "mid"),
        ("e", 25.0, None, ("x", "z"), "low"),
    ]
    out = []
    for key, price, colour, feats, seg in rows:
        values = {"feature": tuple(feats), "segment": (seg,)}
        if price is not None:
            values["price"] = (price,)
        if colour is not None:
            values["colour"] = (colour,)
        out.append(EntityRecord(entity_id=key, uri=f"urn:{key}", values=values))
    return out


SPECS = (
    PropertySpec("price", "numeric"),
    PropertySpec("colour", "categorical"),
    PropertySpec("feature", "set"),
    PropertySpec("segment", "ordinal", ("low", "mid", "high")),
)


def test_s0_is_d0_without_antisymmetric_coordinates_and_rotation_is_orthogonal():
    records = _records()
    basis = TypedBasis.fit(records, SPECS)
    ref = basis.default_reference()
    names_s0 = basis.feature_names("S0")
    names_d0 = basis.feature_names("D0")
    assert set(names_s0) < set(names_d0)
    assert {n for n in names_d0 if n not in names_s0} == {"price:g_dir", "segment:g_dir"}
    for record in records:
        s0 = dict(zip(names_s0, basis.transform(record, ref, "S0")))
        d0 = dict(zip(names_d0, basis.transform(record, ref, "D0")))
        for name, value in s0.items():
            assert d0[name] == pytest.approx(value)
        # (g_sym, g_dir) is an orthogonal rotation of (h-, h+)
        g_s, g_a = d0["price:g_sym"], d0["price:g_dir"]
        h_minus, h_plus = (g_s - g_a) / math.sqrt(2), (g_s + g_a) / math.sqrt(2)
        assert min(h_minus, h_plus) == pytest.approx(0.0, abs=1e-12)
        assert g_s**2 + g_a**2 == pytest.approx(h_minus**2 + h_plus**2)


def test_contributions_reconstruct_scores_exactly_for_every_family():
    records = _records()
    basis = TypedBasis.fit(records, SPECS)
    ref = basis.reference_from_records(records[:2], basis.default_reference(), "user")
    rng = np.random.default_rng(0)
    for family in ("S0", "D0", "direct"):
        theta = rng.normal(size=basis.dimension(family))
        for record in records:
            parts = basis.contributions(record, None if family == "direct" else ref, family, theta)
            score = float(theta @ basis.transform(record, None if family == "direct" else ref, family))
            assert math.fsum(parts.values()) == pytest.approx(score, abs=1e-12)


def test_missing_values_are_not_zero_values():
    records = _records()
    basis = TypedBasis.fit(records, SPECS)
    ref = Reference({"price": 20.0, "colour": "red", "feature": frozenset({"x"}), "segment": "mid"})
    missing_price = dict(zip(basis.feature_names("D0"), basis.transform(records[3], ref, "D0")))
    assert missing_price["price:missing_mismatch"] == 1.0
    assert missing_price["price:g_sym"] == 0.0 and missing_price["price:g_dir"] == 0.0


def test_directional_slopes_match_finite_differences():
    records = _records()
    basis = TypedBasis.fit(records, (PropertySpec("price", "numeric"),))
    ref = Reference({"price": 20.0})
    theta = np.array([0.7, -1.3, 0.0])
    slopes = basis.numeric_slopes("D0", theta)["price"]

    def score(v):
        record = EntityRecord("t", "urn:t", {"price": (v,)})
        return float(theta @ basis.transform(record, ref, "D0"))

    assert (score(19.0) - score(18.0)) == pytest.approx(slopes["below_reference_slope"])
    assert (score(22.0) - score(21.0)) == pytest.approx(slopes["above_reference_slope"])


def test_logistic_group_lasso_matches_cvxpy():
    rng = np.random.default_rng(1)
    D = rng.normal(size=(60, 6))
    groups = [(0, 2), (2, 5), (5, 6)]
    anchor = rng.normal(size=6) * 0.3
    penalty = Penalty(lam=0.05, gamma=0.03, mu=0.4)
    theta, info = fit_pairwise(D, "logistic", penalty, groups, anchor=anchor)
    beta = cp.Variable(6)
    objective = (
        cp.sum(cp.logistic(-(D @ beta))) / 60
        + 0.025 * cp.sum_squares(beta)
        + 0.03 * sum(cp.norm(beta[a:b], 2) for a, b in groups)
        + 0.2 * cp.sum_squares(beta - anchor)
    )
    cp.Problem(cp.Minimize(objective)).solve(solver="CLARABEL")
    assert info["converged"]
    assert np.allclose(theta, beta.value, atol=1e-5)


def test_conditional_logit_matches_cvxpy():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 4, 5))
    y = rng.integers(0, 4, size=80)
    theta, info = fit_choice(X, y, Penalty(lam=0.1, gamma=0.02), [(0, 3), (3, 5)])
    beta = cp.Variable(5)
    utilities = cp.hstack([X[i] @ beta for i in range(80)])
    loss = sum(cp.log_sum_exp(X[i] @ beta) - X[i, y[i]] @ beta for i in range(80)) / 80
    problem = cp.Problem(cp.Minimize(loss + 0.05 * cp.sum_squares(beta) + 0.02 * (cp.norm(beta[0:3], 2) + cp.norm(beta[3:5], 2))))
    problem.solve(solver="CLARABEL")
    assert info["converged"]
    assert np.allclose(theta, beta.value, atol=1e-5)


def test_hinge_and_temperature_and_metrics():
    rng = np.random.default_rng(3)
    true = np.array([1.0, -2.0, 0.5])
    D = rng.normal(size=(200, 3))
    D = D * np.sign(D @ true)[:, None]
    theta, _ = fit_pairwise(D, "hinge", Penalty(lam=0.01), [(0, 1), (1, 2), (2, 3)])
    assert np.mean(D @ theta > 0) > 0.95
    temperature = fit_temperature(np.array([2.0, 1.0, -0.5, 3.0]))
    assert temperature > 0
    metrics = pairwise_metrics(np.array([1.0, 0.0, -1.0]), 1.0)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["tie_rate"] == pytest.approx(1 / 3)
