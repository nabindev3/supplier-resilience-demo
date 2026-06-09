"""Stage 1 optimiser — forecast-driven supplier selection & order allocation.

Assembles the Stage-1 mixed-integer program in the order the modelling actually
happens. Each step is a discrete, inspectable method so the model can be built
(and reviewed) incrementally:

  1. Demand distribution  — annual requirement D and its confidence band, from
     the Prophet forecast (`forecast.annual_demand`).
  2. Supplier efficiency  — a DEA score per supplier, cost in / quality+delivery
     out (`dea.ccr_input_efficiency` over `suppliers_config.dea_arrays`).
  3. PuLP environment      — a cost-minimising `LpProblem`.
  4. Decision variables    — binary y_k (is supplier k selected?) and continuous
     q_k (units ordered from supplier k).

The objective and constraints layer onto this scaffold; this module owns the
setup so every later step starts from a clean, known state.
"""

from __future__ import annotations

import pulp

from suppliers_config import SUPPLIERS, dea_arrays, annual_capacity
from dea import ccr_input_efficiency
from forecast import annual_demand


def supplier_efficiency(suppliers: dict = SUPPLIERS) -> dict:
    """Step 2 — DEA efficiency per supplier (cost in, quality/delivery out)."""
    return ccr_input_efficiency(*dea_arrays(suppliers))


class Stage1Model:
    """Scaffold that builds the Stage-1 MILP one step at a time.

    Holds the demand target, the DEA efficiency scores, the PuLP problem, and
    the decision variables, so later steps (objective, constraints, solve) all
    operate on the same shared state.
    """

    def __init__(
        self,
        suppliers: dict,
        demand: float,
        efficiency: dict | None = None,
        periods: int = 365,
    ):
        self.suppliers = suppliers
        self.names = list(suppliers)
        self.demand = demand
        self.efficiency = efficiency or {}
        self.periods = periods
        # Annual capacity = per-day throughput x horizon, so it shares the
        # yearly basis of the forecasted demand D (ready for the capacity step).
        self.capacity = annual_capacity(suppliers, periods)
        self.prob: pulp.LpProblem | None = None
        self.y: dict[str, pulp.LpVariable] = {}   # binary    — supplier selection
        self.q: dict[str, pulp.LpVariable] = {}   # continuous — order quantity

    # -- step 3: initialise the PuLP environment --------------------------
    def init_problem(self) -> "Stage1Model":
        """Create a cost-minimising `LpProblem`."""
        self.prob = pulp.LpProblem("stage1_order_allocation", pulp.LpMinimize)
        return self

    # -- step 4: define the decision variables ----------------------------
    def define_variables(self) -> "Stage1Model":
        """Create y_k (binary selection) and q_k (continuous order quantity).

            y_k ∈ {0, 1}   1 if supplier k is selected, else 0
            q_k ≥ 0        units ordered from supplier k
        """
        self.y = {k: pulp.LpVariable(f"y_{k}", cat="Binary") for k in self.names}
        self.q = {
            k: pulp.LpVariable(f"q_{k}", lowBound=0, cat="Continuous")
            for k in self.names
        }
        return self


def build_stage1(demand: float | None = None) -> tuple[Stage1Model, dict]:
    """Run steps 1–4 and return the scaffolded model plus the demand distribution.

    Pass `demand` to override the forecast (e.g. to size against D_lower/D_upper);
    otherwise the expected annual demand D drives the model.
    """
    dist = annual_demand()
    target = dist["D"] if demand is None else demand
    eff = supplier_efficiency()
    model = (
        Stage1Model(SUPPLIERS, target, eff, periods=dist["horizon_days"])
        .init_problem()
        .define_variables()
    )
    return model, dist


if __name__ == "__main__":
    print("Stage 1 — forecast-driven supplier selection & order allocation\n")

    model, dist = build_stage1()

    # 1) demand distribution
    print(f"1) Demand distribution (Prophet, next {dist['horizon_days']} days)")
    print(f"   D (expected) : {dist['D']:>12,.0f} units")
    print(f"   90% interval : {dist['D_lower']:>12,.0f} .. {dist['D_upper']:,.0f} units")
    print(f"   uncertainty  : ±{(dist['D_upper'] - dist['D_lower']) / 2:>10,.0f} units\n")

    # 2) DEA efficiency
    print("2) DEA efficiency (cost in -> quality/delivery out)")
    for k, score in model.efficiency.items():
        flag = "  <- efficient frontier" if score >= 0.999 else ""
        print(f"   {k}: {score:.3f}{flag}")
    print()

    # 3) PuLP environment
    sense = "minimize" if model.prob.sense == pulp.LpMinimize else "maximize"
    print("3) PuLP environment")
    print(f"   problem : {model.prob.name}")
    print(f"   sense   : {sense} (cost)\n")

    # 4) decision variables
    print("4) Decision variables")
    print(f"   binary    y_k : {len(model.y):>2}  ({', '.join(model.y)})")
    print(f"   continuous q_k: {len(model.q):>2}  ({', '.join(model.q)})\n")

    annual_cap = sum(model.capacity.values())
    print(f"Capacity basis : {model.periods}-day horizon -> "
          f"{annual_cap:,.0f} annual units of capacity")
    print(f"               : demand D = {model.demand:,.0f}  "
          f"({'feasible' if annual_cap >= model.demand else 'INFEASIBLE'} "
          f"vs capacity)\n")
    print("Scaffold ready — objective and constraints layer on top next.")
