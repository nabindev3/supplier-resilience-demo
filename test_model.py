"""Smoke tests for the DEA + allocation + FCM engine and the stage-1/2 models.

    python test_model.py    # no pytest needed
    python -m pytest        # also works

The stage-1 tests inject a fixed demand instead of running the Prophet fit,
so the whole file stays fast.
"""

import pandas as pd

from data import SUPPLIERS, DEFAULT_DEMAND, dea_arrays
from dea import ccr_input_efficiency
from allocation import allocate, stress_test
from fcm import FCM
from fcm_data import CONCEPTS, weight_matrix
from stage1 import DemandOptimizationEngine, disruption_service
from stage2 import (ingest_stage1, baseline_apc, buyer_budget,
                    baseline_supplier_profits, profit_floors, buyer_utility,
                    supplier_utilities, nash_objective, bargaining_weights,
                    solve_nash, budget_constraint, floor_constraints,
                    total_savings, profit_sacrifices, negotiation_dashboard,
                    GameTheoryPricingEngine)


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


def test_stage2_floors_between_zero_and_baseline():
    e = _stage1_engine()
    _, plan = ingest_stage1(e)
    plan = profit_floors(baseline_supplier_profits(plan))
    # the floor must leave room to concede, but never pay below cost
    assert (plan["profit_floor"] > 0).all()
    assert (plan["profit_floor"] < plan["baseline_profit"]).all()
    assert (plan["floor_price"] > plan["production_cost"]).all()
    assert (plan["floor_price"] < plan["unit_cost"]).all()
    try:
        profit_floors(plan, factor=1.5)
        assert False, "factor outside (0,1) should be rejected"
    except ValueError:
        pass


def test_stage2_buyer_utility_signs():
    e = _stage1_engine()
    _, plan = ingest_stage1(e)
    plan = profit_floors(baseline_supplier_profits(plan))
    apc = baseline_apc(plan)
    budget = buyer_budget(apc)
    list_prices = dict(zip(plan["supplier"], plan["unit_cost"]))
    floor_prices = dict(zip(plan["supplier"], plan["floor_price"]))
    # list prices bust the budget by construction; floor prices must fit,
    # otherwise the bargaining set is empty and there is no game to solve
    assert buyer_utility(list_prices, plan, budget) < 0
    assert buyer_utility(floor_prices, plan, budget) > 0
    # and U_B at list prices is exactly B - APC
    assert abs(buyer_utility(list_prices, plan, budget) - (budget - apc)) < 1.0


def _bargaining_table():
    e = _stage1_engine()
    _, plan = ingest_stage1(e)
    plan = profit_floors(baseline_supplier_profits(plan))
    apc = baseline_apc(plan)
    return plan, buyer_budget(apc)


def test_stage2_supplier_utilities_at_the_edges():
    plan, budget = _bargaining_table()
    floor_prices = dict(zip(plan["supplier"], plan["floor_price"]))
    list_prices = dict(zip(plan["supplier"], plan["unit_cost"]))
    # zero at the walk-away point (within price-rounding), SAP - G at list
    for k, u in supplier_utilities(floor_prices, plan).items():
        assert abs(u) < 50.0, f"{k} floor utility should be ~0, got {u}"
    for _, r in plan.iterrows():
        u = supplier_utilities(list_prices, plan)[r["supplier"]]
        expected = r["baseline_profit"] - r["profit_floor"]
        assert abs(u - expected) < 1.0


def test_stage2_constraints_fence_the_bargaining_set():
    import numpy as np
    plan, budget = _bargaining_table()
    lo = plan["floor_price"].to_numpy()
    hi = plan["unit_cost"].to_numpy()
    bc = budget_constraint(plan, budget)
    fcs = floor_constraints(plan)
    # at list prices the budget is busted (constraint < 0), floors are slack
    assert bc["fun"](hi) < 0
    assert all(c["fun"](hi) > 0 for c in fcs)
    # at floor prices the budget is satisfied, every floor binds (~0)
    assert bc["fun"](lo) > 0
    assert all(abs(c["fun"](lo)) < 50.0 for c in fcs)
    # objective stays finite everywhere now (clipped), so SLSQP never sees NaN
    assert np.isfinite(nash_objective(lo, plan, budget))
    assert np.isfinite(nash_objective(hi, plan, budget))


def test_stage2_nash_solution_splits_surplus_equally():
    # utilities are linear in prices, so total surplus is fixed and the
    # symmetric Nash product must split it equally among all players --
    # a closed-form answer the numeric solve has to reproduce
    plan, budget = _bargaining_table()
    sol = solve_nash(plan, budget)
    assert sol["converged"]
    cost_at_floors = float((plan["floor_price"] * plan["q_star"]).sum())
    share = (budget - cost_at_floors) / (len(plan) + 1)
    assert abs(sol["buyer_utility"] - share) < 1.0
    for u in sol["supplier_utilities"].values():
        assert abs(u - share) < 1.0


def test_stage2_weighted_nash_splits_by_bargaining_power():
    # weighted version of the same closed form: with fixed total surplus the
    # weighted Nash solution gives each player surplus * a_i / sum(a), so
    # S01 (83% of the volume) must walk away with most of the supplier side
    plan, budget = _bargaining_table()
    w = bargaining_weights(plan)
    sol = solve_nash(plan, budget, weights=w)
    assert sol["converged"]
    cost_at_floors = float((plan["floor_price"] * plan["q_star"]).sum())
    surplus = budget - cost_at_floors
    total_w = sum(w.values())
    assert abs(sol["buyer_utility"] - surplus * w["buyer"] / total_w) < 1.0
    for k, u in sol["supplier_utilities"].items():
        assert abs(u - surplus * w[k] / total_w) < 1.0
    # and the asymmetry is real: S01 gets more than the others combined
    others = sum(u for k, u in sol["supplier_utilities"].items() if k != "S01")
    assert sol["supplier_utilities"]["S01"] > others


def test_disruption_service_rewards_diversification():
    # the bridge between the halves: when S01 dies post-commitment, the
    # efficiency-heavy (diversified) plan must keep more service than the
    # cost-only plan that leaned on S01
    e = _stage1_engine()
    cost_run = e.run(w1=1.0, w2=0.0)
    eff_run = e.run(w1=0.0, w2=1.0)
    sweep = {"demand": e.demand_dist, "runs": [
        {**cost_run, "w1": 1.0}, {**eff_run, "w1": 0.0}]}
    rows = disruption_service(sweep, disruption={"S01": 0.0})
    assert rows[1]["service"] > rows[0]["service"]
    assert 0.0 <= rows[0]["service"] <= 1.0


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


def test_stage2_savings_and_sacrifices_consistent():
    plan, budget = _bargaining_table()
    apc = baseline_apc(plan)
    prices = solve_nash(plan, budget, bargaining_weights(plan))["prices"]
    sav = total_savings(plan, prices, apc)
    # negotiation lowers the bill, but not below the budget floor it cleared
    assert 0 < sav["savings"] < apc
    assert sav["negotiated_cost"] <= budget + 1.0
    assert abs(sav["baseline_cost"] - sav["negotiated_cost"] - sav["savings"]) < 1.0
    sac = profit_sacrifices(plan, prices)
    for k, s in sac.items():
        # nobody concedes past their floor: sacrifice stays under (1 - 0.40)
        assert 0 <= s["sacrifice_pct"] <= 0.60 + 1e-6
        assert s["negotiated_profit"] >= plan.set_index("supplier").loc[
            k, "profit_floor"] - 1.0


def test_stage2_dashboard_shape_and_columns():
    plan, budget = _bargaining_table()
    prices = solve_nash(plan, budget)["prices"]
    df = negotiation_dashboard(plan, prices)
    assert len(df) == len(plan)
    for col in ("list_price", "nego_price", "price_drop_%", "sacrifice_%"):
        assert col in df.columns
    # negotiated price sits inside the bargaining range for every supplier
    assert (df["nego_price"] <= df["list_price"] + 1e-6).all()
    assert (df["price_drop_%"] >= -1e-6).all()


def test_game_theory_engine_matches_functions():
    # the packaged engine must reproduce the standalone-function result, and
    # caching the stage-1 engine keeps it from refitting
    e = _stage1_engine()
    gt = GameTheoryPricingEngine(stage1_engine=e, power="volume")
    sol = gt.solve()
    assert sol["converged"]
    plan, budget = gt.setup()["plan"], gt.setup()["budget"]
    ref = solve_nash(plan, budget, bargaining_weights(plan))
    for k in ref["prices"]:
        assert abs(gt.equilibrium_prices()[k] - ref["prices"][k]) < 1e-3
    # symmetric vs volume-weighted give genuinely different prices for S01
    gt_sym = GameTheoryPricingEngine(stage1_engine=e, power="equal")
    gt_sym.solve()
    assert gt.equilibrium_prices()["S01"] != gt_sym.equilibrium_prices()["S01"]
    assert isinstance(gt.dashboard(), pd.DataFrame)


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
        test_stage2_floors_between_zero_and_baseline,
        test_stage2_buyer_utility_signs,
        test_stage2_supplier_utilities_at_the_edges,
        test_stage2_constraints_fence_the_bargaining_set,
        test_stage2_nash_solution_splits_surplus_equally,
        test_stage2_weighted_nash_splits_by_bargaining_power,
        test_disruption_service_rewards_diversification,
        test_stage2_budget_forces_a_gap,
        test_stage2_savings_and_sacrifices_consistent,
        test_stage2_dashboard_shape_and_columns,
        test_game_theory_engine_matches_functions,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
