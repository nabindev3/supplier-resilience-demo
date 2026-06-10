"""Stage 2 — Nash bargaining game over the order price (framing).

Stage 1 decides *who* supplies *how much* (q_k*). Stage 2 of Yousefi,
Jahangoshai Rezaee & Solimanpur (2021) then decides *at what price*: a
non-cooperative Nash bargaining game between the buyer and each selected
supplier, solved as a non-linear program.

This module frames that game — the first three of its building blocks:

  1. Ingest Stage-1 outputs   — pull q_k* and the selected supplier subset
     straight from `DemandOptimizationEngine` (no copy-pasted numbers; the
     bargaining table is whatever the optimiser actually chose).
  2. Baseline metrics         — the Annual Purchasing Cost (APC) the buyer
     would pay at list prices if no negotiation took place:
         APC = Σ_k  unit_cost_k · q_k*      (purchasing only — holding and
                                             setup are not on the table).
  3. Buyer's budget           — the hard ceiling B the buyer can spend.
     Set *below* the baseline APC (default 95%), so the status quo is
     unaffordable and a negotiated price concession is the only way to close
     the gap: that gap is what gives the game a non-empty bargaining set.
"""

from __future__ import annotations

import pandas as pd

from stage1 import DemandOptimizationEngine

# Budget = 95% of the no-negotiation cost: tight enough to force every
# selected supplier to the table, loose enough that a deal plainly exists.
BUDGET_FACTOR = 0.95


def ingest_stage1(
    engine: DemandOptimizationEngine | None = None,
    w1: float = 0.6,
    w2: float = 0.4,
) -> tuple[DemandOptimizationEngine, pd.DataFrame]:
    """Step 1 — pull q_k* and the selected suppliers from the Stage-1 engine.

    Accepts an already-run engine (to reuse its cached Prophet fit) or builds
    one at the given weights. Returns the engine and the structured frame of
    selected suppliers — the negotiation table.
    """
    if engine is None:
        engine = DemandOptimizationEngine(w1=w1, w2=w2)
    if engine.result is None:
        engine.run()
    return engine, engine.results_frame()


def baseline_apc(plan: pd.DataFrame) -> float:
    """Step 2 — Annual Purchasing Cost with no negotiation.

    What the buyer pays if every selected supplier charges list price for its
    committed volume: Σ unit_cost_k · q_k*. This is the buyer's *disagreement
    benchmark* — any bargained outcome must beat it to be worth the talks.
    """
    return round(float((plan["unit_cost"] * plan["q_star"]).sum()), 2)


def buyer_budget(apc: float, factor: float = BUDGET_FACTOR) -> float:
    """Step 3 — the buyer's absolute budget limit B.

    B = factor · APC with factor < 1, so the no-negotiation outcome violates
    the budget by construction. The shortfall (APC − B) is the concession the
    bargaining game must extract across the selected suppliers.
    """
    if factor >= 1.0:
        raise ValueError(
            f"factor must be < 1 to force a negotiation (got {factor})"
        )
    return round(apc * factor, 2)


def frame_bargaining_problem(
    engine: DemandOptimizationEngine | None = None,
    w1: float = 0.6,
    w2: float = 0.4,
    budget_factor: float = BUDGET_FACTOR,
) -> dict:
    """Steps 1–3 in one call: the inputs every later game step builds on."""
    engine, plan = ingest_stage1(engine, w1, w2)
    apc = baseline_apc(plan)
    budget = buyer_budget(apc, budget_factor)
    return {
        "engine": engine,
        "plan": plan,                       # q_k* per selected supplier
        "baseline_apc": apc,                # cost of not negotiating
        "budget": budget,                   # hard ceiling B < APC
        "required_concession": round(apc - budget, 2),
        "demand": engine.demand_dist,
    }


if __name__ == "__main__":
    print("Stage 2 — Nash bargaining game: framing the negotiation\n")

    setup = frame_bargaining_problem()
    plan = setup["plan"]

    print(f"Negotiation table ({len(plan)} selected suppliers, "
          f"D = {setup['demand']['D']:,.0f} units):\n")
    print(plan.to_string(index=False))

    print(f"\nBaseline APC (no negotiation): ${setup['baseline_apc']:>12,.2f}")
    print(f"Buyer's budget B ({BUDGET_FACTOR:.0%} of APC): "
          f"${setup['budget']:>12,.2f}")
    print(f"Required concession (APC − B): ${setup['required_concession']:>12,.2f}")
    print("\nThe status quo exceeds the budget by construction — the gap is "
          "what the\nbargaining game must close through per-supplier price "
          "concessions.")
