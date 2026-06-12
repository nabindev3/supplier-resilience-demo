"""Smoke tests for the DEA + allocation + FCM engine and the stage-1/2 models.

    python test_model.py    # no pytest needed
    python -m pytest        # also works

The stage-1 tests inject a fixed demand instead of running the Prophet fit,
so the whole file stays fast.
"""

from data import SUPPLIERS, DEFAULT_DEMAND, dea_arrays
from dea import ccr_input_efficiency
from allocation import allocate, stress_test
from fcm import FCM
from fcm_data import CONCEPTS, weight_matrix
from stage1 import DemandOptimizationEngine
from stage2 import (ingest_stage1, baseline_apc, buyer_budget,
                    baseline_supplier_profits)


def _efficiency():
    inp, out = dea_arrays(SUPPLIERS)
    return ccr_input_efficiency(inp, out)


def test_dea_scores_in_unit_interval():
    eff = _efficiency()
    assert eff, "no scores returned"
    for name, score in eff.items():
        assert 0.0 < score <= 1.0 + 1e-9, f"{name} score out of range: {score}"


def test_allocation_meets_demand():
    eff = _efficiency()
    plan = allocate(SUPPLIERS, DEFAULT_DEMAND, eff, efficiency_weight=0.0)
    assert plan["unmet_units"] == 0
    assert abs(sum(plan["allocation"].values()) - DEFAULT_DEMAND) < 1e-6


def test_resilient_plan_diversifies():
    eff = _efficiency()
    cost = allocate(SUPPLIERS, DEFAULT_DEMAND, eff, efficiency_weight=0.0)
    res = allocate(
        SUPPLIERS, DEFAULT_DEMAND, eff,
        efficiency_weight=1.5, max_share=0.4, min_suppliers=3,
    )
    assert len(res["active_suppliers"]) >= 3
    assert len(res["active_suppliers"]) > len(cost["active_suppliers"])
    # diversification shouldn't be free
    assert res["purchasing_cost"] >= cost["purchasing_cost"]


def test_resilience_pays_off_under_disruption():
    eff = _efficiency()
    cost = allocate(SUPPLIERS, DEFAULT_DEMAND, eff, efficiency_weight=0.0)
    res = allocate(
        SUPPLIERS, DEFAULT_DEMAND, eff,
        efficiency_weight=1.5, max_share=0.4, min_suppliers=3,
    )
    disruption = {"S6": 0.0}  # the cheap high-volume supplier goes down
    cost_hit = stress_test(SUPPLIERS, DEFAULT_DEMAND, cost["allocation"], disruption)
    res_hit = stress_test(SUPPLIERS, DEFAULT_DEMAND, res["allocation"], disruption)
    assert res_hit["realized_service_level"] > cost_hit["realized_service_level"]


def test_fcm_converges_to_unit_interval():
    fcm = FCM(CONCEPTS, weight_matrix())
    final, traj = fcm.run(max_steps=100)
    assert len(traj) < 101, "did not converge"
    assert all(0.0 <= v <= 1.0 for v in final)


def test_fcm_diversification_reduces_risk_raises_resilience():
    fcm = FCM(CONCEPTS, weight_matrix())
    sc = fcm.scenario("Supplier diversification")
    assert sc["Disruption risk"][2] < 0
    assert sc["Resilience"][2] > 0


def test_nhl_preserves_structure_and_bounds():
    fcm = FCM(CONCEPTS, weight_matrix())
    final, _ = fcm.run()
    W2 = fcm.nhl_step(final)
    W0 = weight_matrix()
    assert (W2[W0 == 0] == 0).all()
    assert (abs(W2) <= 1.0).all()


def _stage1_engine():
    # bypass the Prophet fit with a fixed annual demand
    e = DemandOptimizationEngine()
    e.demand_dist = {"D": 500_000.0, "D_lower": 450_000.0, "D_upper": 550_000.0,
                     "mean_daily": 1369.9, "horizon_days": 365}
    return e


def test_stage1_meets_demand_exactly():
    e = _stage1_engine()
    r = e.run(w1=1.0, w2=0.0)
    assert r["status"] == "Optimal"
    assert abs(sum(r["allocation"].values()) - 500_000.0) < 1.0


def test_stage1_weights_actually_trade_off():
    e = _stage1_engine()
    cost_run = e.run(w1=1.0, w2=0.0)
    eff_run = e.run(w1=0.0, w2=1.0)
    # paying more should buy a more efficient (and bigger) supplier base
    assert cost_run["Z1_cost"] <= eff_run["Z1_cost"]
    assert eff_run["Z2_efficiency"] >= cost_run["Z2_efficiency"]
    assert len(eff_run["selected"]) > len(cost_run["selected"])


def test_stage1_min_order_link_no_free_selections():
    # every selected supplier must carry real volume, otherwise Z2 would
    # just select everyone and order nothing from most of them
    e = _stage1_engine()
    r = e.run(w1=0.0, w2=1.0)
    for k in r["selected"]:
        assert r["allocation"][k] >= e.suppliers[k]["min_order"] * 365 - 1.0


def test_stage1_results_frame_consistent():
    e = _stage1_engine()
    e.run()
    df = e.results_frame()
    assert set(df["supplier"]) == set(e.result["selected"])
    assert (df["q_star"] > 0).all()
    assert abs(df["share_of_D"].sum() - 1.0) < 0.01


def test_stage2_supplier_profits_positive_and_consistent():
    e = _stage1_engine()
    _, plan = ingest_stage1(e)
    plan = baseline_supplier_profits(plan)
    # every supplier must make money at list price, or they wouldn't be
    # selling at it; and the game needs margin to bargain away
    assert (plan["unit_margin"] > 0).all()
    assert (plan["baseline_profit"] > 0).all()
    expected = (plan["unit_cost"] - plan["production_cost"]) * plan["q_star"]
    assert (abs(plan["baseline_profit"] - expected) < 1.0).all()
    # the buyer's required concession has to fit inside the total margin
    apc = baseline_apc(plan)
    gap = apc - buyer_budget(apc)
    assert plan["baseline_profit"].sum() > gap


def test_stage2_budget_forces_a_gap():
    e = _stage1_engine()
    _, plan = ingest_stage1(e)
    apc = baseline_apc(plan)
    assert buyer_budget(apc) < apc
    try:
        buyer_budget(apc, factor=1.0)
        assert False, "factor >= 1 should be rejected"
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [
        test_dea_scores_in_unit_interval,
        test_allocation_meets_demand,
        test_resilient_plan_diversifies,
        test_resilience_pays_off_under_disruption,
        test_fcm_converges_to_unit_interval,
        test_fcm_diversification_reduces_risk_raises_resilience,
        test_nhl_preserves_structure_and_bounds,
        test_stage1_meets_demand_exactly,
        test_stage1_weights_actually_trade_off,
        test_stage1_min_order_link_no_free_selections,
        test_stage1_results_frame_consistent,
        test_stage2_supplier_profits_positive_and_consistent,
        test_stage2_budget_forces_a_gap,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
