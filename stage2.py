"""Stage 2 setup: the Nash bargaining game over the order price.

Stage 1 fixes who supplies what (q_k*). In the 2021 paper the price is then
negotiated between the buyer and each selected supplier. This module sets
the table for that game: pull q_k* from the stage-1 engine, price the
no-negotiation baseline, and give the buyer a budget below that baseline so
there is actually something to bargain over.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

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

    Not rounded: this sits inside the Nash objective and the optimiser
    shouldn't see cent-sized flat spots.
    """
    cost = float(sum(prices[k] * q for k, q
                     in zip(plan["supplier"], plan["q_star"])))
    return budget - cost


def supplier_utilities(prices: dict, plan: pd.DataFrame) -> dict:
    """U_k = negotiated profit - G_k, supplier k's gain over walking away.

        U_k = (p_k - production_cost_k) * q_k* - G_k

    Zero exactly at the floor price, SAP_k - G_k at list price.
    """
    return {
        r["supplier"]: float(
            (prices[r["supplier"]] - r["production_cost"]) * r["q_star"]
            - r["profit_floor"])
        for _, r in plan.iterrows()
    }


def bargaining_weights(plan: pd.DataFrame, buyer_weight: float = 1.0) -> dict:
    """Bargaining power for the weighted Nash product.

    The buyer is one side of every deal and gets `buyer_weight`; the
    suppliers split a combined 1.0 in proportion to the volume they carry.
    Volume share is the most defensible proxy here: losing S01 (83% of D)
    hurts the buyer far more than losing S03 (3%), and that threat is
    exactly what bargaining power is. Swap in DEA scores or anything else —
    the solver doesn't care, only the proportional split does.
    """
    w = {"buyer": buyer_weight}
    total = float(plan["share_of_D"].sum())
    for _, r in plan.iterrows():
        w[r["supplier"]] = float(r["share_of_D"]) / total
    return w


_LOG_EPS = 1e-9  # clip utilities here so log stays finite during line search


def nash_objective(price_vector: np.ndarray, plan: pd.DataFrame,
                   budget: float, weights: dict | None = None) -> float:
    """Negative log of the (weighted) Nash product, what scipy minimises.

        maximise  U_B^a_B * prod_k U_k^a_k
        ==  minimise  -(a_B log U_B + sum a_k log U_k)

    `weights` maps "buyer" and each supplier to a bargaining power a_i;
    None means the symmetric game (all 1).

    Log form for two reasons: the raw product is around 1e21 here (five
    utilities of 1e3-1e5 each), so taking it to the solver directly is
    numerically hopeless; and the sum of logs turns the product into a sum,
    which the constrained solver handles cleanly. Utilities are clipped at a
    tiny positive epsilon, so if SLSQP's line search probes just outside the
    bargaining set the log stays finite (a NaN there would stall the solve).
    The optimum is strictly interior — every player keeps a positive share of
    the surplus — so the clip never binds at the solution; the budget and
    floor constraints, not this clip, are what hold the answer inside the set.

    `price_vector` follows the row order of `plan`. This is non-linear in
    the prices, which is why stage 2 needs scipy.optimize.minimize where
    stage 1 got away with PuLP.
    """
    prices = dict(zip(plan["supplier"], price_vector))
    u = {"buyer": buyer_utility(prices, plan, budget),
         **supplier_utilities(prices, plan)}
    if weights is None:
        weights = {k: 1.0 for k in u}
    return -float(sum(weights[k] * np.log(max(u[k], _LOG_EPS)) for k in u))


def budget_constraint(plan: pd.DataFrame, budget: float) -> dict:
    """Step: the buyer's total negotiated cost must stay within B.

        B - sum_k p_k * q_k* >= 0

    Returned as a scipy inequality-constraint dict so SLSQP enforces it
    directly, rather than leaning on the log barrier to keep cost under B.
    """
    q = plan["q_star"].to_numpy()
    return {"type": "ineq", "fun": lambda x: budget - float(np.dot(x, q))}


def floor_constraints(plan: pd.DataFrame) -> list[dict]:
    """Step: every supplier's negotiated profit must clear its floor G_k.

        (p_k - production_cost_k) * q_k* - G_k >= 0   for each k

    One inequality-constraint dict per supplier. These coincide with the
    lower price bounds (p_k >= floor_price_k), but stating them explicitly
    mirrors the game's walk-away condition and guards against the bound's
    4-decimal rounding letting a supplier dip a cent below its floor.
    """
    cons = []
    for i, (_, r) in enumerate(plan.iterrows()):
        prod, q, g = r["production_cost"], r["q_star"], r["profit_floor"]
        cons.append({
            "type": "ineq",
            "fun": lambda x, i=i, prod=prod, q=q, g=g: (x[i] - prod) * q - g,
        })
    return cons


def solve_nash(plan: pd.DataFrame, budget: float,
               weights: dict | None = None) -> dict:
    """Solve the bargaining game with constrained SLSQP.

    Search space (step: price bounds): each price between the supplier's
    walk-away price and its list asking price, [floor_price_k, unit_cost_k].
    Constraints: the budget ceiling and every supplier's profit floor, as
    explicit inequality dicts.

    Starts just inside the feasible wedge — the budget plane cuts the price
    box very close to the floor side, so the box midpoint is NOT feasible and
    a careless start gives SLSQP an infeasible point to recover from.
    """
    lo = plan["floor_price"].to_numpy()
    hi = plan["unit_cost"].to_numpy()
    x0 = lo + 0.03 * (hi - lo)
    constraints = [budget_constraint(plan, budget), *floor_constraints(plan)]
    res = minimize(nash_objective, x0, args=(plan, budget, weights),
                   method="SLSQP", bounds=list(zip(lo, hi)),
                   constraints=constraints,
                   options={"ftol": 1e-12, "maxiter": 1000})
    prices = dict(zip(plan["supplier"], res.x))
    return {
        "prices": {k: round(p, 4) for k, p in prices.items()},
        "buyer_utility": round(buyer_utility(prices, plan, budget), 2),
        "supplier_utilities": {k: round(u, 2) for k, u
                               in supplier_utilities(prices, plan).items()},
        "converged": bool(res.success),
        "iterations": int(res.nit),
    }


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


def total_savings(plan: pd.DataFrame, prices: dict,
                  baseline_apc: float) -> dict:
    """Step: the money the negotiation actually saved the buyer.

    Difference between the Stage-1 baseline cost (everyone at list price) and
    the Stage-2 negotiated cost (everyone at c_k*). This is the buyer's prize
    for running the game — distinct from U_B, which measures headroom against
    the budget B rather than against the old list-price bill.
    """
    negotiated = float(sum(prices[k] * q for k, q
                           in zip(plan["supplier"], plan["q_star"])))
    return {
        "baseline_cost": round(baseline_apc, 2),
        "negotiated_cost": round(negotiated, 2),
        "savings": round(baseline_apc - negotiated, 2),
        "savings_pct": round((baseline_apc - negotiated) / baseline_apc, 4),
    }


def profit_sacrifices(plan: pd.DataFrame, prices: dict) -> dict:
    """Step: how much profit each supplier gave up to secure the contract.

    Per supplier: baseline profit SAP_k vs negotiated profit at c_k*, and the
    sacrifice as a percentage of SAP_k. The floor caps this — a supplier can
    never be pushed past giving up (1 - floor_factor) of its profit.
    """
    out = {}
    for _, r in plan.iterrows():
        k = r["supplier"]
        negotiated_profit = (prices[k] - r["production_cost"]) * r["q_star"]
        sap = r["baseline_profit"]
        out[k] = {
            "baseline_profit": round(sap, 2),
            "negotiated_profit": round(negotiated_profit, 2),
            "sacrifice": round(sap - negotiated_profit, 2),
            "sacrifice_pct": round((sap - negotiated_profit) / sap, 4),
        }
    return out


def negotiation_dashboard(plan: pd.DataFrame, prices: dict) -> pd.DataFrame:
    """Step: the before/after table, one row per supplier.

    Lays the list price next to the negotiated price, and the baseline profit
    next to the negotiated profit, so the trade each supplier made is legible
    at a glance.
    """
    sac = profit_sacrifices(plan, prices)
    rows = []
    for _, r in plan.iterrows():
        k = r["supplier"]
        rows.append({
            "supplier": k,
            "q_star": r["q_star"],
            "list_price": r["unit_cost"],
            "nego_price": round(prices[k], 4),
            "price_drop_%": round(
                100 * (r["unit_cost"] - prices[k]) / r["unit_cost"], 2),
            "baseline_profit": sac[k]["baseline_profit"],
            "nego_profit": sac[k]["negotiated_profit"],
            "floor": r["profit_floor"],
            "sacrifice_%": round(100 * sac[k]["sacrifice_pct"], 2),
        })
    return pd.DataFrame(rows)


class GameTheoryPricingEngine:
    """Stage 2 as one standalone, modular step in the pipeline.

    Takes the Stage-1 allocation, frames the bargaining game, solves it, and
    reports the outcome:

        gt = GameTheoryPricingEngine(power="volume")
        gt.solve()
        df = gt.dashboard()          # before/after table
        gt.savings()                 # buyer's prize vs the list-price bill
        gt.sacrifices()              # what each supplier gave up

    `power` picks the bargaining-power model: "volume" (weighted by share of
    demand), "equal"/"symmetric" (all players weighted 1), or a ready-made
    weights dict. Pass an already-run Stage-1 engine to reuse its cached
    Prophet fit instead of refitting.
    """

    def __init__(self, stage1_engine=None, w1=0.6, w2=0.4,
                 budget_factor=BUDGET_FACTOR, floor_factor=PROFIT_FLOOR_FACTOR,
                 power="volume"):
        self.stage1_engine = stage1_engine
        self.w1, self.w2 = w1, w2
        self.budget_factor = budget_factor
        self.floor_factor = floor_factor
        self.power = power
        self.setup_ = None
        self.solution_ = None

    def setup(self) -> dict:
        """Frame the game (ingest, baseline, budget, profits, floors). Cached."""
        if self.setup_ is None:
            self.setup_ = frame_bargaining_problem(
                self.stage1_engine, self.w1, self.w2,
                self.budget_factor, self.floor_factor)
            self.stage1_engine = self.setup_["engine"]
        return self.setup_

    def _weights(self, plan: pd.DataFrame) -> dict | None:
        if self.power in ("equal", "symmetric"):
            return None
        if self.power == "volume":
            return bargaining_weights(plan)
        if isinstance(self.power, dict):
            return self.power
        raise ValueError(f"unknown bargaining power model: {self.power!r}")

    def solve(self) -> dict:
        """Run the constrained Nash solve; returns the equilibrium prices."""
        setup = self.setup()
        if not setup["bargaining_set_nonempty"]:
            raise ValueError("bargaining set is empty: no price clears every "
                             "supplier's floor within the buyer's budget")
        plan = setup["plan"]
        self.solution_ = solve_nash(plan, setup["budget"], self._weights(plan))
        return self.solution_

    def equilibrium_prices(self) -> dict:
        """The negotiated unit prices c_k* (solving first if needed)."""
        if self.solution_ is None:
            self.solve()
        return self.solution_["prices"]

    def savings(self) -> dict:
        return total_savings(self.setup()["plan"], self.equilibrium_prices(),
                             self.setup()["baseline_apc"])

    def sacrifices(self) -> dict:
        return profit_sacrifices(self.setup()["plan"],
                                 self.equilibrium_prices())

    def dashboard(self) -> pd.DataFrame:
        return negotiation_dashboard(self.setup()["plan"],
                                     self.equilibrium_prices())


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

    # the Nash solve. Utilities are all linear in prices, so U_B + sum(U_k)
    # is a constant: the total surplus B - cost_at_floors. That gives both
    # games a closed-form answer to check the optimiser against — the
    # symmetric product splits the surplus equally, the weighted one splits
    # it in proportion to bargaining power.
    budget = setup["budget"]
    surplus = budget - setup["cost_at_floors"]
    n_players = len(plan) + 1
    print(f"\ntotal surplus on the table: ${surplus:,.2f} "
          f"-> ${surplus / n_players:,.2f} each if split {n_players} ways")

    sym = solve_nash(plan, budget)
    print(f"\nsymmetric Nash ({sym['iterations']} iterations, "
          f"converged={sym['converged']}):")
    print(f"  buyer keeps ${sym['buyer_utility']:,.2f} under budget")
    for k in plan["supplier"]:
        print(f"  {k}  price ${sym['prices'][k]:.4f}  ->  "
              f"U ${sym['supplier_utilities'][k]:,.2f} above the floor")

    # the packaged engine: weighted game (power = volume share) plus the
    # before/after dashboard, savings and sacrifices in one place
    gt = GameTheoryPricingEngine(stage1_engine=setup["engine"], power="volume")
    sol = gt.solve()
    print(f"\nweighted Nash via GameTheoryPricingEngine "
          f"({sol['iterations']} iterations, converged={sol['converged']}):")
    print(gt.dashboard().to_string(index=False))

    sav = gt.savings()
    print(f"\nlist-price bill ${sav['baseline_cost']:,.2f}  ->  "
          f"negotiated ${sav['negotiated_cost']:,.2f}")
    print(f"total savings  ${sav['savings']:,.2f}  ({sav['savings_pct']:.2%})")
