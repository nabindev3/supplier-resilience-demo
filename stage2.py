"""Stage 2 setup: the Nash bargaining game over the order price.

Stage 1 fixes who supplies what (q_k*). In the 2021 paper the price is then
negotiated between the buyer and each selected supplier. This module sets
the table for that game: pull q_k* from the stage-1 engine, price the
no-negotiation baseline, and give the buyer a budget below that baseline so
there is actually something to bargain over.
"""

import pandas as pd

from stage1 import DemandOptimizationEngine
from suppliers_config import SUPPLIERS

# budget = 95% of the list-price cost: tight enough to force everyone to the
# table, loose enough that a deal clearly exists
BUDGET_FACTOR = 0.95

# a supplier walks away rather than keep less than 40% of the profit they
# planned the year around. 0.40 is deliberately under ~0.44: above that the
# floor prices alone already blow the 95% budget and the bargaining set is
# empty (see docs/decisions.md)
PROFIT_FLOOR_FACTOR = 0.40


def ingest_stage1(engine=None, w1=0.6, w2=0.4):
    """q_k* and the selected suppliers, straight from the stage-1 engine.

    Pass an already-run engine to reuse its cached Prophet fit.
    """
    if engine is None:
        engine = DemandOptimizationEngine(w1=w1, w2=w2)
    if engine.result is None:
        engine.run()
    return engine, engine.results_frame()


def baseline_apc(plan: pd.DataFrame) -> float:
    """Annual purchasing cost at list prices: sum(unit_cost * q*).

    Purchasing only. Holding and setup stay out because only the price is on
    the table. Any bargained outcome has to beat this number for the buyer.
    """
    return round(float((plan["unit_cost"] * plan["q_star"]).sum()), 2)


def buyer_budget(apc: float, factor: float = BUDGET_FACTOR) -> float:
    """The buyer's hard ceiling B, set below the baseline APC on purpose:
    the status quo must be unaffordable or nobody has a reason to negotiate."""
    if factor >= 1.0:
        raise ValueError("factor must be < 1, otherwise no negotiation is forced")
    return round(apc * factor, 2)


def baseline_supplier_profits(plan: pd.DataFrame,
                              suppliers: dict = SUPPLIERS) -> pd.DataFrame:
    """SAP_k: each selected supplier's annual profit at its list price,

        SAP_k = (unit_cost_k - production_cost_k) * q_k*

    This is the supplier side of the negotiation: what each one banks if no
    deal is struck and the buyer somehow pays list anyway. Every price
    concession in the game comes straight out of this number, and a supplier
    will never go below zero margin — production cost is their hard floor,
    like the budget is the buyer's.

    Returns the plan with production_cost, unit_margin and baseline_profit
    columns added.
    """
    out = plan.copy()
    out["production_cost"] = out["supplier"].map(
        lambda k: suppliers[k]["production_cost"])
    out["unit_margin"] = (out["unit_cost"] - out["production_cost"]).round(2)
    out["baseline_profit"] = (out["unit_margin"] * out["q_star"]).round(2)
    return out


def profit_floors(plan: pd.DataFrame,
                  factor: float = PROFIT_FLOOR_FACTOR) -> pd.DataFrame:
    """G_k: the profit below which supplier k walks away.

        G_k = factor * SAP_k

    Tying the floor to baseline profit (rather than one absolute number)
    keeps the walk-away points proportionate: S01 plans its year around a
    $300k margin and won't keep the contract for pocket change, while S03
    can live with far less. Each G_k also pins a floor *price*,

        floor_price_k = production_cost_k + G_k / q_k*

    the lowest unit price at which the contract still clears the supplier's
    threshold. The game can push p_k no lower than this.

    Adds profit_floor and floor_price columns.
    """
    if not 0.0 < factor < 1.0:
        raise ValueError("factor must be in (0, 1): 0 means no floor, "
                         "1 means nobody concedes anything")
    out = plan.copy()
    out["profit_floor"] = (factor * out["baseline_profit"]).round(2)
    out["floor_price"] = (
        out["production_cost"] + out["profit_floor"] / out["q_star"]
    ).round(4)
    return out


def buyer_utility(prices: dict, plan: pd.DataFrame, budget: float) -> float:
    """U_B = B - total negotiated cost, the buyer's gain from the deal.

    `prices` maps supplier -> negotiated unit price p_k; quantities stay at
    q_k* (stage 1 already fixed them, only the price is on the table).
    Positive means the deal fits the budget; at list prices it's negative by
    construction, which is the whole reason the buyer is negotiating.
    """
    cost = float(sum(prices[k] * q for k, q
                     in zip(plan["supplier"], plan["q_star"])))
    return round(budget - cost, 2)


def frame_bargaining_problem(engine=None, w1=0.6, w2=0.4,
                             budget_factor=BUDGET_FACTOR,
                             floor_factor=PROFIT_FLOOR_FACTOR) -> dict:
    engine, plan = ingest_stage1(engine, w1, w2)
    plan = baseline_supplier_profits(plan)
    plan = profit_floors(plan, floor_factor)
    apc = baseline_apc(plan)
    budget = buyer_budget(apc, budget_factor)

    # the bargaining set is non-empty only if the buyer can afford every
    # supplier sitting exactly at their floor price
    cost_at_floors = float((plan["floor_price"] * plan["q_star"]).sum())

    return {
        "engine": engine,
        "plan": plan,
        "baseline_apc": apc,
        "budget": budget,
        "required_concession": round(apc - budget, 2),
        "supplier_profits": dict(zip(plan["supplier"], plan["baseline_profit"])),
        "profit_floors": dict(zip(plan["supplier"], plan["profit_floor"])),
        "cost_at_floors": round(cost_at_floors, 2),
        "bargaining_set_nonempty": cost_at_floors <= budget,
        "demand": engine.demand_dist,
    }


if __name__ == "__main__":
    setup = frame_bargaining_problem()
    plan = setup["plan"]

    print(f"{len(plan)} suppliers at the table, "
          f"D = {setup['demand']['D']:,.0f} units\n")
    print(plan.to_string(index=False))
    print(f"\nbaseline APC : ${setup['baseline_apc']:,.2f}")
    print(f"budget B     : ${setup['budget']:,.2f}  "
          f"({BUDGET_FACTOR:.0%} of baseline)")
    print(f"gap to close : ${setup['required_concession']:,.2f}")

    total_profit = sum(setup["supplier_profits"].values())
    print(f"\nper supplier: baseline profit SAP_k, walk-away floor G_k "
          f"({PROFIT_FLOOR_FACTOR:.0%}), floor price:")
    for _, r in plan.iterrows():
        print(f"  {r['supplier']}  SAP ${r['baseline_profit']:>11,.2f}   "
              f"G ${r['profit_floor']:>10,.2f}   "
              f"list ${r['unit_cost']:.2f} -> floor ${r['floor_price']:.4f}")

    print(f"\ncost with everyone at their floor: ${setup['cost_at_floors']:,.2f}")
    print(f"bargaining set non-empty: {setup['bargaining_set_nonempty']}")

    # buyer utility at the two ends of the bargaining range
    list_prices = dict(zip(plan["supplier"], plan["unit_cost"]))
    floor_prices = dict(zip(plan["supplier"], plan["floor_price"]))
    print(f"\nbuyer utility U_B = B - total cost:")
    print(f"  at list prices  : ${buyer_utility(list_prices, plan, setup['budget']):>12,.2f}")
    print(f"  at floor prices : ${buyer_utility(floor_prices, plan, setup['budget']):>12,.2f}")
