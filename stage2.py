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


def frame_bargaining_problem(engine=None, w1=0.6, w2=0.4,
                             budget_factor=BUDGET_FACTOR) -> dict:
    engine, plan = ingest_stage1(engine, w1, w2)
    plan = baseline_supplier_profits(plan)
    apc = baseline_apc(plan)
    budget = buyer_budget(apc, budget_factor)
    return {
        "engine": engine,
        "plan": plan,
        "baseline_apc": apc,
        "budget": budget,
        "required_concession": round(apc - budget, 2),
        "supplier_profits": dict(zip(plan["supplier"], plan["baseline_profit"])),
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
    print(f"\nbaseline supplier profits (SAP_k), {len(plan)} suppliers, "
          f"${total_profit:,.0f} total:")
    for k, p in setup["supplier_profits"].items():
        print(f"  {k}  ${p:>12,.2f}")
    # sanity: the game is only winnable if the suppliers' combined margin
    # exceeds the concession the buyer needs
    headroom = total_profit - setup["required_concession"]
    print(f"\nmargin headroom after closing the gap: ${headroom:,.0f} "
          f"({'feasible' if headroom > 0 else 'INFEASIBLE'})")
