"""Scientific invariants for train-only encoding and matched utility learning."""
import json
import math

import cvxpy as cp
import numpy as np
import pytest

from dtltr_vehicle.basis import PropertySpec, Reference
from dtltr_vehicle.experiments.features import Encoder
from dtltr_vehicle.experiments.models import (
    choice_metrics, fit_choice_temperature, fit_multinomial, fit_pair, penalty_grid,
)
from rgsmw_next.types import EntityRecord


SPECS = (PropertySpec("x", "numeric"), PropertySpec("category", "categorical"),
         PropertySpec("tags", "set"), PropertySpec("rank", "ordinal", ("low", "mid", "high")))


def rows():
    return [EntityRecord(str(i), f"urn:{i}", {"x": (float(i),), "category": ("a" if i%2 else "b",),
                      "tags": ("p", "q") if i%2 else (), "rank": ("low", "mid", "high")[i%3:i%3+1]}) for i in range(8)]


def test_train_only_statistics_and_roundtrip_holdout_does_not_modify_registry():
    train = rows()
    enc = Encoder.fit(train, SPECS, "spline_ref", references=[Reference({"x":float(i)/2}) for i in range(8)])
    frozen = json.dumps(enc.metadata(), sort_keys=True)
    holdout = EntityRecord("holdout", "urn:holdout", {"x": (10000.,), "category": ("new",), "tags": ("unknown",), "rank": ("high",)})
    vector = enc.transform(holdout, Reference({"x": -10000., "category":"new", "tags":frozenset({"unknown"})}))
    assert np.all(np.isfinite(vector))
    assert json.dumps(enc.metadata(), sort_keys=True) == frozen
    assert "new" not in enc.properties[1].vocabulary
    assert "unknown" not in enc.properties[2].vocabulary
    assert "holdout" not in enc.train_ids
    restored = Encoder.from_metadata(json.loads(frozen))
    assert restored.metadata() == enc.metadata()
    assert np.array_equal(restored.transform(holdout, enc.default_reference()), enc.transform(holdout, enc.default_reference()))
    # Numeric deviation knots come from TRAIN aligned references, not holdout deviations.
    numeric = enc.properties[0]
    deviations = np.array([i/(4*numeric.scale) for i in range(8)])
    expected = np.unique(np.quantile(np.clip(deviations,-1,1), np.arange(1,6)/6))
    assert enc.spline_knots["x"] == pytest.approx(expected)


def test_symmetric_features_are_exact_subset_and_missing_empty_policies_are_explicit():
    train = rows()
    st, dt = (Encoder.fit(train, SPECS, m, completion="raw") for m in ("ST","DT"))
    q = Reference({"x":3., "category":"a", "tags":frozenset({"p"}), "rank":"mid"})
    assert set(st.names) < set(dt.names)
    assert set(dt.names)-set(st.names) == {"x:g_dir", "rank:g_dir"}
    for r in train:
        sv = dict(zip(st.names, st.transform(r,q)))
        dv = dict(zip(dt.names, dt.transform(r,q)))
        assert all(dv[name] == pytest.approx(value) for name,value in sv.items())
    absent = EntityRecord("missing", "urn:missing", {})
    observed_empty = EntityRecord("empty", "urn:empty", {"tags": ()})
    av = dict(zip(dt.names, dt.transform(absent,q)))
    ev = dict(zip(dt.names, dt.transform(observed_empty,q)))
    assert av["tags:candidate_missing"] == 1
    assert ev["tags:candidate_missing"] == 0
    assert av["x:g_sym"] == av["x:g_dir"] == 0
    assert av["x:candidate_missing"] == av["x:missing_mismatch"] == 1


def test_raw_vs_default_reference_and_clipping_change_only_declared_feature_policies():
    train, ref = rows(), Reference({})
    raw = Encoder.fit(train, SPECS, "DT", completion="raw")
    default = Encoder.fit(train, SPECS, "DT", completion="default")
    ar, ad = dict(zip(raw.names,raw.transform(train[0],ref))), dict(zip(default.names,default.transform(train[0],ref)))
    assert ar["x:g_sym"] == 0 and ad["x:g_sym"] > 0
    assert ar["x:reference_missing"] == ad["x:reference_missing"] == 1
    holdout = EntityRecord("outside", "urn:outside", {"x":(10000.,)})
    unclipped = Encoder.fit(train,SPECS,"DT",clipping="unclipped")
    clipped = Encoder.fit(train,SPECS,"DT",clipping="clipped")
    assert abs(clipped.transform(holdout, clipped.default_reference())[1]) <= 1/math.sqrt(2)
    assert abs(unclipped.transform(holdout, unclipped.default_reference())[1]) > 1
    assert clipped.diagnostics(holdout,clipped.default_reference())["outside_clip_count"] == 1


def test_all_model_maps_are_finite_and_tree_uses_exact_same_information_as_direct_ref():
    train = rows()
    for model in ("TD","ST","DT","direct","direct_ref","spline","spline_ref","catboost_ref"):
        enc = Encoder.fit(train,SPECS,model)
        X = enc.matrix(train,enc.default_reference())
        assert X.shape == (len(train),enc.dimension)
        assert enc.groups[-1][1] == len(enc.names)
        assert np.all(np.isfinite(X))
    direct, tree = (Encoder.fit(train,SPECS,m) for m in ("direct_ref","catboost_ref"))
    assert tree.names == direct.names
    assert np.array_equal(tree.matrix(train),direct.matrix(train))


def test_group_logistic_matches_independent_convex_solution_and_contributions_reconstruct():
    rng = np.random.default_rng(411)
    D = rng.normal(size=(75,5))
    groups = [(0,2),(2,5)]
    anchor = rng.normal(size=5)*.2
    penalty = {"lam":.07,"gamma":.03,"mu":.1}
    fit = fit_pair(D,penalty,groups,anchor=anchor)
    beta = cp.Variable(5)
    objective = cp.sum(cp.logistic(-D@beta))/len(D) + .035*cp.sum_squares(beta) + .03*(cp.norm(beta[:2])+cp.norm(beta[2:])) + .05*cp.sum_squares(beta-anchor)
    problem = cp.Problem(cp.Minimize(objective))
    problem.solve(solver="CLARABEL")
    assert fit.info["converged"] and fit.info["finite"]
    assert fit.info["objective"] == pytest.approx(problem.value,abs=2e-7)
    assert np.allclose(fit.theta,beta.value,atol=3e-5)
    enc = Encoder.fit(rows(),SPECS,"DT")
    pairs = enc.matrix(rows())[1:]-enc.matrix(rows())[:-1]
    fitted = fit_pair(pairs,{"lam":.1},enc.groups)
    parts = fitted.contributions(pairs,enc)
    assert [sum(p.values()) for p in parts] == pytest.approx(fitted.score(pairs),abs=1e-12)


def test_td_simplex_objective_and_anchor_match_convex_reference():
    rng = np.random.default_rng(519)
    D = rng.uniform(-1,1,size=(90,4))
    anchor = np.array([.1,.2,.3,.4])
    penalty = {"lam":.02,"gamma":.3,"mu":.5}
    fit = fit_pair(D,penalty,[(i,i+1) for i in range(4)],anchor=anchor,model="TD")
    beta = cp.Variable(4)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.logistic(-D@beta))/len(D) + .01*cp.sum_squares(beta) + .3*cp.norm1(beta) + .25*cp.sum_squares(beta-anchor)),[beta>=0,cp.sum(beta)==1])
    problem.solve(solver="CLARABEL")
    assert np.all(fit.theta >= 0) and sum(fit.theta) == pytest.approx(1)
    assert fit.info["objective"] == pytest.approx(problem.value,abs=2e-7)
    assert np.allclose(fit.theta,beta.value,atol=5e-5)
    assert fit.info["group_penalty_constant_on_simplex"]
    grid = penalty_grid({"lambda_grid":[.01,.1],"gamma_grid":[0,10],"mu_grid":[.1,1]},"TD",adaptation=True)
    assert len(grid)==4 and all(p["gamma"]==0 for p in grid)


def test_flexible_spline_represents_nonlinear_preference_linear_direct_cannot():
    train = [EntityRecord(str(i),f"urn:{i}",{"x":(float(x),)}) for i,x in enumerate(np.linspace(-3,3,121))]
    specs = [PropertySpec("x","numeric")]
    direct, spline = (Encoder.fit(train,specs,m,clipping="unclipped",knots=5) for m in ("direct","spline"))
    rng = np.random.default_rng(929)
    pairs = rng.integers(0,len(train),size=(1200,2))
    utilities = -np.linspace(-3,3,len(train))**2
    pairs = pairs[np.abs(utilities[pairs[:,0]]-utilities[pairs[:,1]])>.2]
    pairs = np.where((utilities[pairs[:,0]]>utilities[pairs[:,1]])[:,None],pairs,pairs[:,::-1])
    accuracy = {}
    for encoder in (direct,spline):
        X = encoder.matrix(train)
        D = X[pairs[:,0]]-X[pairs[:,1]]
        fit = fit_pair(D,{"lam":1e-4},encoder.groups,model=encoder.model)
        accuracy[encoder.model] = np.mean(fit.score(D)>0)
    assert accuracy["spline"] > .95
    assert accuracy["direct"] < .65


def test_multinomial_loss_matches_convex_reference_and_temperature_handles_ties():
    rng=np.random.default_rng(155)
    X=rng.normal(size=(40,3,4))
    y=rng.integers(0,3,size=40)
    fit=fit_multinomial(X,y,{"lam":.1},[(0,2),(2,4)],model="direct")
    beta=cp.Variable(4)
    problem=cp.Problem(cp.Minimize(sum(cp.log_sum_exp(X[i]@beta)-X[i,y[i]]@beta for i in range(40))/40 + .05*cp.sum_squares(beta)))
    problem.solve(solver="CLARABEL")
    assert fit.info["objective"]==pytest.approx(problem.value,abs=1e-7)
    utilities=fit.score(X)
    assert utilities.shape==(40,3)
    t=fit_choice_temperature(utilities,y)
    assert t>0 and choice_metrics(utilities,y,t)["log_loss"] <= choice_metrics(utilities,y)["log_loss"]+1e-8
    ties=choice_metrics(np.zeros((2,3)),np.array([0,1]))
    assert ties["accuracy"]==pytest.approx(1/3) and ties["log_loss"]==pytest.approx(math.log(3))


def test_unsupported_tree_combinations_and_nonfinite_inputs_fail_explicitly():
    with pytest.raises(ValueError,match="pair_items"):
        fit_pair(np.ones((2,2)),{"lam":1},[(0,2)],model="catboost_ref")
    with pytest.raises(ValueError,match="finite"):
        fit_pair(np.array([[np.nan,1.]]),{"lam":1},[(0,2)])
    with pytest.raises(ValueError,match="partition"):
        fit_pair(np.ones((2,2)),{"lam":1},[(0,1)])
    with pytest.raises(ValueError,match="non-negative"):
        fit_pair(np.ones((2,2)),{"lam":-1},[(0,2)])
    with pytest.raises(ValueError,match="Unknown model"):
        fit_pair(np.ones((2,2)),{"lam":1},[(0,2)],model="invented")
    grid=penalty_grid({"penalties":{"lam":[.003,.007],"gamma":[0,.02],"mu":[.4,2.0]}},"DT",adaptation=True)
    assert len(grid)==8 and {p["lam"] for p in grid}=={.003,.007}


def test_nonconverged_primary_solver_uses_audited_fallback_and_hinge_remains_utility_model():
    rng=np.random.default_rng(81)
    D=rng.normal(size=(30,3))
    fit=fit_pair(D,{"lam":.1},[(0,1),(1,3)],config={"max_iterations":1})
    assert fit.info["fallback"] and fit.info["status"]=="clarabel"
    assert fit.info["stationarity_residual"] < fit.info["stationarity_tolerance"]
    hinge=fit_pair(D,{"lam":.1,"gamma":.01},[(0,1),(1,3)],config={"loss":"hinge"})
    assert hinge.info["loss"]=="hinge" and hinge.info["converged"]
    assert hinge.info["stationarity_residual"] is None


def _full_anu_td_failure():
    """Actual full-profile ANU exp1/seed11/fold4 adaptation, fit call 589.

    Captured with the production one-thread BLAS settings. Old SLSQP stopped
    after two iterations at residual 0.0008883118340984472; its conic fallback
    returned optimal_inaccurate. These are public archived feature differences,
    not participant identifiers or a fabricated mechanism.
    """
    D=np.array([[1,0,0,0],[1,-1,-1,1],[1,1,0,0],[0,1,-.5,0],[-1,0,.5,1],
                [-1,-1,.5,1],[-1,0,0,0],[0,0,1,1],[0,-1,.5,1],[-1,0,-.5,1],
                [0,-1,0,1],[1,-1,-.5,0],[0,-1,0,1],[0,-1,-1,0],[0,0,-1,0],
                [-1,0,1,0],[1,0,.5,0],[0,-1,.5,1],[-1,-1,-.5,1],[0,0,0,1]],dtype=float)
    anchor=np.array([0.,.9498488007495616,.05015119925043844,1.3986208025063007e-17])
    return D,anchor,{"lam":.01,"gamma":0.,"mu":100.}


def test_real_full_anu_strong_anchor_failure_reaches_the_same_objective_kkt_solution():
    from scipy.optimize import root
    from scipy.special import expit
    D,anchor,penalty=_full_anu_td_failure()
    fit=fit_pair(D,penalty,[(i,i+1) for i in range(4)],anchor=anchor,model="TD")

    def gradient(theta):
        return -(D.T@expit(-D@theta))/len(D)+.01*theta+100*(theta-anchor)

    # Independent active-set KKT equations: coordinate zero is on the boundary,
    # the remaining coordinates share one multiplier, and weights sum to one.
    def equations(active):
        theta=np.r_[0.,active]
        g=gradient(theta)
        return np.array([g[1]-g[2],g[1]-g[3],active.sum()-1])

    solution=root(equations,[.94,.05,.01],tol=1e-11)
    expected=np.r_[0.,solution.x]
    assert solution.success
    assert np.min(expected)>=0 and abs(expected.sum()-1)<1e-12
    assert np.linalg.norm(equations(solution.x))<1e-10
    assert gradient(expected)[0]>=gradient(expected)[1]-1e-10
    assert np.allclose(fit.theta,expected,atol=1e-8)
    assert fit.info["stationarity_tolerance"]==1e-4
    assert fit.info["stationarity_residual"]<=1e-6
    assert fit.info["penalty"]==penalty and fit.info["converged"]


def test_real_full_anu_loose_slsqp_is_polished_without_accepting_an_inaccurate_conic_fit(monkeypatch):
    from scipy.optimize import minimize as scipy_minimize
    import dtltr_vehicle.experiments.models as models
    D,anchor,penalty=_full_anu_td_failure()

    def reproduce_old_stop(*args,**kwargs):
        assert kwargs["method"]=="SLSQP"
        kwargs["options"]={**kwargs["options"],"ftol":1e-8}
        return scipy_minimize(*args,**kwargs)

    monkeypatch.setattr(models,"minimize",reproduce_old_stop)
    fit=fit_pair(D,penalty,[(i,i+1) for i in range(4)],anchor=anchor,model="TD")
    assert fit.info["status"]=="simplex_projected_gradient_polish"
    assert fit.info["first_solver"]["first_stationarity_residual"]>1e-4
    assert fit.info["polish"]["target_met"]
    assert fit.info["polish"]["iterations"]>0
    assert fit.info["stationarity_residual"]<=1e-6
    assert fit.info["stationarity_tolerance"]==1e-4
    assert np.min(fit.theta)>=0 and abs(fit.theta.sum()-1)<1e-12


def test_projected_simplex_polish_multinomial_matches_independent_convex_objective(monkeypatch):
    from types import SimpleNamespace
    import dtltr_vehicle.experiments.models as models
    rng=np.random.default_rng(837)
    X=rng.normal(size=(18,3,4))
    y=rng.integers(0,3,size=18)
    anchor=np.array([.2,.1,.3,.4])
    monkeypatch.setattr(models,"minimize",lambda *args,**kwargs: SimpleNamespace(x=anchor.copy(),success=True,nit=0,message="regression: intentionally unoptimized warm start"))
    fit=fit_multinomial(X,y,{"lam":.01,"mu":50},[(i,i+1) for i in range(4)],anchor=anchor,model="TD")
    beta=cp.Variable(4)
    objective=sum(cp.log_sum_exp(X[i]@beta)-X[i,y[i]]@beta for i in range(len(X)))/len(X)+.005*cp.sum_squares(beta)+25*cp.sum_squares(beta-anchor)
    problem=cp.Problem(cp.Minimize(objective),[beta>=0,cp.sum(beta)==1])
    problem.solve(solver="CLARABEL")
    assert problem.status==cp.OPTIMAL
    assert fit.info["status"]=="simplex_projected_gradient_polish"
    assert fit.info["stationarity_residual"]<=1e-6
    assert fit.info["objective"]==pytest.approx(problem.value,abs=1e-7)
    assert np.allclose(fit.theta,beta.value,atol=1e-5)


def test_catboost_is_item_utility_pairlogit_not_classifier_or_difference_regressor():
    pytest.importorskip("catboost")
    rng=np.random.default_rng(517)
    pairs=rng.normal(size=(50,2,3))
    pairs=np.where((pairs[:,0,0]>pairs[:,1,0])[:,None,None],pairs,pairs[:,::-1])
    D=pairs[:,0]-pairs[:,1]
    fit=fit_pair(D,{"lam":1},[(0,1),(1,3)],model="catboost_ref",config={"pair_items":pairs,"iterations":30,"depth":3})
    assert fit.theta is None and fit.info["loss"]=="PairLogit"
    assert fit.score(pairs).shape==(50,2)
    assert np.mean(fit.score(pairs)[:,0]>fit.score(pairs)[:,1])>.9
    assert fit.contributions(D,None) is None
    with pytest.raises(ValueError,match="gamma and mu"):
        fit_pair(D,{"lam":1,"gamma":.1},[(0,1),(1,3)],model="catboost_ref",config={"pair_items":pairs})
