"""Leakage, matching and grouping checks for the archival reanalysis."""

from dataclasses import replace

import numpy as np
import pytest

from dtltr_vehicle.experiments import human
from rgsmw_next.anu_car_preferences import AnuCarItem, AnuCarPreference, AnuCarPreferenceDataset, AnuCarUser
from rgsmw_next.mcfadden_train import McFaddenTrainDataset, VehicleChoiceAlternative, VehicleChoiceSet
from rgsmw_next.preference_learning import PairwiseObservation


def settings():
    return {"paths": {"anu": "unused", "mcfadden": "unused"}, "models": ["DT", "direct_ref", "spline_ref"],
            "completion": ["default"], "clipping": ["clipped"], "knots": 2,
            "penalties": {"lam": [0.1], "gamma": [0.0], "mu": [0.1, 1.0]},
            "budgets": [0, 1, 2], "support_draws": 2, "folds": 2, "limits": {}}


def anu_dataset():
    items = tuple(AnuCarItem("exp1", str(i), 1 + i % 2, 1 + i % 2, 1.0 + i / 4, 1 + i % 2, None, i)
                  for i in range(1, 7))
    users = tuple(AnuCarUser("exp1", str(u), 1, 2, 1, 10, 5, u) for u in range(1, 11))
    preferences = []
    for user in users:
        for a, b in ((1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (2, 6)):
            high, low = (b, a) if int(user.user_id) % 2 else (a, b)
            preferences.append(AnuCarPreference("exp1", user.user_id, str(high), str(low), False, len(preferences) + 1))
    return AnuCarPreferenceDataset(items, users, tuple(preferences), {})


def mcf_dataset(respondents=False):
    sets = []
    for i in range(20):
        alternatives = []
        for j in range(6):
            values = {field: float(1 + j + i / 30) for field in human.NUMERIC}
            values.update({"type": ("regcar", "regcar", "regcar", "sportuv", "truck", "van")[j],
                           "fuel": ("gasoline", "electric", "cng")[j % 3], "size": str(j % 4)})
            alternatives.append(VehicleChoiceAlternative(str(j + 1), j == i % 6, values))
        sets.append(VehicleChoiceSet(str(i), str(i), str(i // 2) if respondents else None,
                                    str(i % 6 + 1), {"college": 0, "hsg2": 0, "coml5": 0}, tuple(alternatives)))
    return McFaddenTrainDataset(tuple(sets), (), (), {"fixture": True})


def test_four_roles_are_disjoint_deterministic_and_cover_the_cohort():
    a = human._four_roles([str(i) for i in range(30)], 5)
    b = human._four_roles(reversed([str(i) for i in range(30)]), 5)
    assert a == b
    units = [unit for role in a.values() for unit in role]
    assert len(units) == len(set(units)) == 30
    assert all(a.values())
    with pytest.raises(ValueError, match="four grouped"):
        human._four_roles(["1", "2", "3"], 5)


def test_support_draws_change_only_reserved_pool_and_budgets_are_nested():
    rows = [PairwiseObservation("user", f"row{i}", str(i), str(i + 1), f"group{i}") for i in range(10)]
    view = human._support_views(rows, maximum=4, draws=4, seed=3)["user"]
    evaluation = set(view["evaluation_ids"])
    assert len(evaluation) == 6
    for support in view["support"]:
        assert {r.group_id for r in support} == set(view["support_ids"])
        assert not evaluation.intersection(r.group_id for r in support)
        assert {r.group_id for r in support[:2]} <= {r.group_id for r in support[:4]}
    with pytest.raises(ValueError, match="collapsed pair"):
        human._support_views(rows + rows[:1], 4, 1, 3)


def test_fractional_ties_and_probability_metrics():
    pair = human._pair_statistics(np.array([0.0, 2.0, -2.0]))
    assert pair["accuracy"].tolist() == [0.5, 1.0, 0.0]
    scores = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0], [2.0, 2.0, 0.0]])
    choice = human._choice_statistics(scores, np.array([0, 1, 2]), 1.0)
    assert choice["accuracy"].tolist() == [1 / 3, 0.5, 0.0]
    assert np.all(np.isfinite(choice["log_loss"]))
    assert choice["brier"][0] == pytest.approx(2 / 3)


def test_anu_end_to_end_has_matched_test_queries_and_exact_contributions(monkeypatch):
    monkeypatch.setattr(human, "load_anu_car_preferences", lambda *a, **k: anu_dataset())
    config = settings()
    config["anu_references"] = ["catalogue", "support"]
    result = human.run_anu(config, 5, "exp1", 0)
    role_units = [u for units in result["splits"]["roles"].values() for u in units]
    assert len(role_units) == len(set(role_units)) == 10
    identities = {}
    for row in result["predictions"]:
        identity = (row["model"], row["reference"], row["budget"], row["draw"])
        identities.setdefault(identity, set()).add((row["unit_id"], row["row_id"]))
        assert sum(row["contributions"].values()) == pytest.approx(row["margin"], abs=1e-8)
    assert all(value == next(iter(identities.values())) for value in identities.values())
    assert len(result["metrics"]) == 3 * 2 * 3 * 2
    assert all(fit["calibration"] for fit in result["fits"])
    assert all(audit.get("converged", True) for audit in result["diagnostics"]["fit_audit"])


def test_anu_test_labels_do_not_select_population_or_adaptation_penalties(monkeypatch):
    dataset = anu_dataset()
    monkeypatch.setattr(human, "load_anu_car_preferences", lambda *a, **k: dataset)
    config = settings()
    config["models"] = ["DT"]
    first = human.run_anu(config, 6, "exp1", 0)
    test_ids = {user.rsplit(":", 1)[1] for user in first["splits"]["roles"]["test"]}
    changed = replace(dataset, preferences=tuple(replace(row, preferred=row.nonpreferred, nonpreferred=row.preferred)
                                                 if row.user_id in test_ids else row for row in dataset.preferences))
    monkeypatch.setattr(human, "load_anu_car_preferences", lambda *a, **k: changed)
    second = human.run_anu(config, 6, "exp1", 0)
    assert first["fits"][0]["selected_penalty"] == second["fits"][0]["selected_penalty"]
    assert first["fits"][0]["theta"] == second["fits"][0]["theta"]
    assert first["fits"][0]["temperature"] == second["fits"][0]["temperature"]


def test_calibration_labels_do_not_tune_population_or_adaptation(monkeypatch):
    dataset = anu_dataset()
    monkeypatch.setattr(human, "load_anu_car_preferences", lambda *a, **k: dataset)
    config = settings()
    config["models"] = ["DT"]
    first = human.run_anu(config, 8, "exp1", 0)
    calibration_ids = {user.rsplit(":", 1)[1] for user in first["splits"]["roles"]["calibration"]}
    changed = replace(dataset, preferences=tuple(replace(row, preferred=row.nonpreferred, nonpreferred=row.preferred)
                                                 if row.user_id in calibration_ids else row for row in dataset.preferences))
    monkeypatch.setattr(human, "load_anu_car_preferences", lambda *a, **k: changed)
    second = human.run_anu(config, 8, "exp1", 0)
    assert first["fits"][0]["selected_penalty"] == second["fits"][0]["selected_penalty"]
    assert first["fits"][0]["theta"] == second["fits"][0]["theta"]
    # The reserved calibration role can alter the probability scale only.
    assert first["fits"][0]["temperature"] != second["fits"][0]["temperature"]


def test_anu_feature_cache_preserves_predictions_and_selection(monkeypatch):
    monkeypatch.setattr(human, "load_anu_car_preferences", lambda *a, **k: anu_dataset())
    config = settings()
    config["models"] = ["DT"]
    config["anu_references"] = ["support"]
    calls = []
    original = human._pair_items
    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(human, "_pair_items", spy)
    uncached = human.run_anu({**config, "cache_features": False}, 4, "exp1", 0)
    uncached_calls = len(calls)
    calls.clear()
    cached = human.run_anu(config, 4, "exp1", 0)
    assert cached["predictions"] == uncached["predictions"]
    assert cached["fits"] == uncached["fits"]
    assert len(calls) < uncached_calls


def test_mcfadden_actual_respondents_stay_grouped_and_encoder_is_train_only(monkeypatch):
    dataset = mcf_dataset(respondents=True)
    monkeypatch.setattr(human, "load_mcfadden_train", lambda *a, **k: dataset)
    config = settings()
    result = human.run_mcfadden(config, 7, "design_proxy")
    assert result["splits"]["unit"] == "explicit respondent"
    assert result["splits"]["respondent_limitation"] is None
    parts = result["splits"]["choice_sets"]
    for respondent in range(10):
        roles = [role for role, ids in parts.items() if str(respondent * 2) in ids or str(respondent * 2 + 1) in ids]
        assert len(roles) == 1
    train_ids = {f"mcf:{cs}:{alternative}" for cs in parts["train"] for alternative in range(1, 7)}
    for fit in result["fits"]:
        assert set(fit["encoder"]["train_ids"]) == train_ids
    assert result["diagnostics"]["design_proxy"]["first_listed_matches_proxy"] is True
    assert "confounded" in result["diagnostics"]["design_proxy"]["interpretation"]


def test_mcfadden_absent_panel_identifier_is_not_invented(monkeypatch):
    monkeypatch.setattr(human, "load_mcfadden_train", lambda *a, **k: mcf_dataset())
    config = settings()
    config["models"] = ["DT"]
    result = human.run_mcfadden(config, 9, "catalogue")
    assert result["splits"]["unit"] == "archived choice set"
    assert "unknown respondent overlap" in result["splits"]["respondent_limitation"]
    assert all(row["unit_id"].startswith("choice-set:") for row in result["predictions"])


def test_mcfadden_test_attributes_cannot_estimate_scales_knots_or_coefficients(monkeypatch):
    dataset = mcf_dataset()
    monkeypatch.setattr(human, "load_mcfadden_train", lambda *a, **k: dataset)
    config = settings()
    config["models"] = ["spline_ref"]
    first = human.run_mcfadden(config, 12, "catalogue")
    test_ids = set(first["splits"]["choice_sets"]["test"])
    changed = replace(dataset, choice_sets=tuple(
        replace(cs, alternatives=tuple(replace(a, item_features={**a.item_features, "price": 100000.0})
                                       for a in cs.alternatives)) if cs.choice_set_id in test_ids else cs
        for cs in dataset.choice_sets))
    monkeypatch.setattr(human, "load_mcfadden_train", lambda *a, **k: changed)
    second = human.run_mcfadden(config, 12, "catalogue")
    assert first["fits"][0]["encoder"] == second["fits"][0]["encoder"]
    assert first["fits"][0]["theta"] == second["fits"][0]["theta"]
    assert first["fits"][0]["selected_penalty"] == second["fits"][0]["selected_penalty"]
    assert set(first["fits"][0]["probe_contributions"]) == set(second["fits"][0]["probe_contributions"])


def test_design_proxy_never_uses_choice_label_and_rejects_ambiguous_design():
    cs = mcf_dataset().choice_sets[0]
    q0 = human.Reference({"type": "truck"})
    ref = human._design_proxy(cs, q0)
    relabelled = replace(cs, chosen_alternative_id="6", alternatives=tuple(replace(a, chosen=a.alternative_id == "6") for a in cs.alternatives))
    assert ref == human._design_proxy(relabelled, q0)
    with pytest.raises(ValueError, match="unique threefold"):
        human._design_proxy(replace(cs, alternatives=tuple(replace(a, item_features={**a.item_features, "type": str(i)})
                                                         for i, a in enumerate(cs.alternatives))), q0)
