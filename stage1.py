"""Stage 1 optimiser — forecast-driven supplier selection & order allocation.

Builds and solves the Stage-1 mixed-integer program in the order the modelling
actually happens:

  1. Demand distribution  — annual requirement D and its confidence band, from
     the Prophet forecast (`forecast.annual_demand`).
  2. Supplier efficiency  — a DEA score per supplier, cost in / quality+delivery
     out (`dea.ccr_input_efficiency` over `suppliers_config.dea_arrays`).
  3. PuLP environment      — a cost-minimising `LpProblem`.
  4. Decision variables    — binary y_k (is supplier k selected?) and continuous
     q_k (units ordered from supplier k).
  5. Objectives            — Z1 (total annual cost: purchasing + holding +
     setup) and Z2 (sum of DEA efficiency over selected suppliers).
  6. Global criterion      — Z1 and Z2 combined into one weighted objective
     with adjustable weights w1 (cost) and w2 (efficiency), each objective
     normalised by its ideal value so the weights are comparable.
  7. Strict constraints    — Σ q_k = D (demand met exactly) and
     q_k ≤ capacity_k · y_k (nobody exceeds capacity; ordering links selection).
  8. Solve + risk sweep    — CBC finds q_k*; a sweep over 10 (w1, w2) combos
     from cost-driven to efficiency-driven traces the Pareto frontier.

Mirrors the two-objective structure of Yousefi, Jahangoshai Rezaee & Solimanpur
(2021) Stage 1 — cost minimisation fused with DEA efficiency — solved here with
the weighted global criterion method instead of their linearised QP.
"""

from __future__ import annotations

import pulp

from suppliers_config import SUPPLIERS, dea_arrays, annual_capacity
from dea import ccr_input_efficiency
from forecast import annual_demand

PARETO_PNG = "pareto_frontier.png"

# Normalised objective coefficients are tiny (price / Z1* ~ 1e-6); scaling the
# whole objective keeps them comfortably above CBC's numeric tolerances without
# changing the argmin.
OBJ_SCALE = 1_000.0


def supplier_efficiency(suppliers: dict = SUPPLIERS) -> dict:
    """Step 2 — DEA efficiency per supplier (cost in, quality/delivery out)."""
    return ccr_input_efficiency(*dea_arrays(suppliers))


class Stage1Model:
    """The Stage-1 MILP, built one inspectable step at a time.

    Holds the demand target, the DEA efficiency scores, the PuLP problem, and
    the decision variables, so objective and constraints all operate on the
    same shared state.
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
        # Annual basis: per-day capacity and min-order scale with the horizon
        # so they share the yearly basis of the forecasted demand D.
        self.capacity = annual_capacity(suppliers, periods)
        self.min_order = {
            k: s["min_order"] * periods for k, s in suppliers.items()
        }
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

    # -- step 5a: cost objective Z1 ----------------------------------------
    def z1_cost(self):
        """Total annual supply-chain cost (linear expression).

            Z1 = Σ_k  unit_cost_k · q_k          purchasing
               + Σ_k  holding_cost_k · q_k / 2   holding (average cycle stock:
                                                  half the annual order quantity
                                                  is carried on average)
               + Σ_k  setup_cost_k · y_k         setup / contracting (fixed,
                                                  incurred once if selected)
        """
        s = self.suppliers
        purchasing = pulp.lpSum(s[k]["unit_cost"] * self.q[k] for k in self.names)
        holding = pulp.lpSum(s[k]["holding_cost"] * self.q[k] * 0.5 for k in self.names)
        setup = pulp.lpSum(s[k]["setup_cost"] * self.y[k] for k in self.names)
        return purchasing + holding + setup

    # -- step 5b: efficiency objective Z2 ----------------------------------
    def z2_efficiency(self):
        """Sum of DEA efficiency scores over the *selected* suppliers.

            Z2 = Σ_k  eff_k · y_k    (maximise)

        The min-order linking constraint stops Z2 from collecting "free"
        efficiency points: selecting a supplier commits real order volume.
        """
        return pulp.lpSum(self.efficiency[k] * self.y[k] for k in self.names)

    # -- step 6: global criterion (weighted, normalised) -------------------
    def set_global_criterion(
        self,
        w1: float,
        w2: float,
        z1_ideal: float,
        z2_ideal: float,
        z1_nadir: float,
        z2_nadir: float,
    ) -> "Stage1Model":
        """Combine Z1 and Z2 into one objective via the global criterion method.

            minimise  w1 · (Z1 − Z1*) / (Z1ⁿ − Z1*)
                    + w2 · (Z2* − Z2) / (Z2* − Z2ⁿ)

        Each term is the deviation from that objective's ideal value (Z1* =
        minimum cost, Z2* = maximum efficiency sum), normalised by the
        objective's *range* across the Pareto set (ideal → nadir). Range
        normalisation maps both deviations onto [0, 1], so w1 and w2 sweep the
        frontier evenly — with ideal-value normalisation the objective with the
        proportionally wider range would dominate at almost any weight.
        """
        z1_range = max(z1_nadir - z1_ideal, 1e-9)
        z2_range = max(z2_ideal - z2_nadir, 1e-9)
        deviation = (
            w1 * (self.z1_cost() - z1_ideal) * (1.0 / z1_range)
            + w2 * (z2_ideal - self.z2_efficiency()) * (1.0 / z2_range)
        )
        self.prob += OBJ_SCALE * deviation
        return self

    # -- step 7: strict constraints -----------------------------------------
    def add_constraints(self) -> "Stage1Model":
        """Hard boundaries of the allocation.

            Σ_k q_k = D                          demand met exactly
            q_k ≤ capacity_k · y_k               capacity + selection linking
            q_k ≥ min_order_k · y_k              a selected supplier gets real
                                                 volume (no free Z2 points)
        """
        self.prob += (
            pulp.lpSum(self.q[k] for k in self.names) == self.demand,
            "demand_balance",
        )
        for k in self.names:
            self.prob += self.q[k] <= self.capacity[k] * self.y[k], f"cap_{k}"
            self.prob += self.q[k] >= self.min_order[k] * self.y[k], f"minq_{k}"
        return self

    # -- step 8: execute the solver -----------------------------------------
    def solve(self) -> dict:
        """Run CBC and extract the optimal allocation q_k* and selection y_k*."""
        self.prob.solve(pulp.PULP_CBC_CMD(msg=0))
        alloc = {k: round(self.q[k].value() or 0.0, 1) for k in self.names}
        selected = [k for k in self.names if (self.y[k].value() or 0) > 0.5]
        s = self.suppliers
        purchasing = sum(s[k]["unit_cost"] * alloc[k] for k in self.names)
        holding = sum(s[k]["holding_cost"] * alloc[k] * 0.5 for k in self.names)
        setup = sum(s[k]["setup_cost"] for k in selected)
        return {
            "status": pulp.LpStatus[self.prob.status],
            "allocation": alloc,
            "selected": selected,
            "Z1_cost": round(purchasing + holding + setup, 2),
            "Z1_breakdown": {
                "purchasing": round(purchasing, 2),
                "holding": round(holding, 2),
                "setup": round(setup, 2),
            },
            "Z2_efficiency": round(
                sum(self.efficiency[k] for k in selected), 4
            ),
        }


def _fresh_model(
    demand: float, eff: dict, periods: int, suppliers: dict = SUPPLIERS
) -> Stage1Model:
    """Steps 3, 4, 7 — a constrained model with variables, no objective yet."""
    return (
        Stage1Model(suppliers, demand, eff, periods)
        .init_problem()
        .define_variables()
        .add_constraints()
    )


def anchor_points(
    demand: float, eff: dict, periods: int, suppliers: dict = SUPPLIERS
) -> dict:
    """Solve each objective alone to anchor the global criterion.

    Ideal points (best case per objective):
        Z1* — minimum achievable cost (efficiency ignored).
        Z2* — maximum achievable efficiency sum (cost ignored).
    Nadir points (worst value each objective takes across the Pareto set),
    found lexicographically:
        Z2ⁿ — the efficiency sum of the pure cost-minimal plan.
        Z1ⁿ — the cheapest cost that still achieves Z2* (re-solve min Z1
              with Σ eff_k·y_k ≥ Z2* pinned).
    """
    m1 = _fresh_model(demand, eff, periods, suppliers)
    m1.prob += m1.z1_cost()
    r1 = m1.solve()
    z1_star, z2_nadir = r1["Z1_cost"], r1["Z2_efficiency"]

    m2 = _fresh_model(demand, eff, periods, suppliers)
    m2.prob += -m2.z2_efficiency()          # LpMinimize, so negate to maximise
    z2_star = m2.solve()["Z2_efficiency"]

    m3 = _fresh_model(demand, eff, periods, suppliers)
    m3.prob += m3.z2_efficiency() >= z2_star - 1e-6, "pin_z2_ideal"
    m3.prob += m3.z1_cost()
    z1_nadir = m3.solve()["Z1_cost"]

    return {
        "z1_ideal": z1_star,
        "z2_ideal": z2_star,
        "z1_nadir": z1_nadir,
        "z2_nadir": z2_nadir,
    }


def solve_weighted(
    w1: float, w2: float,
    demand: float, eff: dict, periods: int,
    anchors: dict,
    suppliers: dict = SUPPLIERS,
) -> dict:
    """One global-criterion solve at a given (w1, w2)."""
    model = _fresh_model(demand, eff, periods, suppliers).set_global_criterion(
        w1, w2,
        anchors["z1_ideal"], anchors["z2_ideal"],
        anchors["z1_nadir"], anchors["z2_nadir"],
    )
    result = model.solve()
    result["w1"], result["w2"] = round(w1, 3), round(w2, 3)
    return result


class DemandOptimizationEngine:
    """The whole Stage-1 workflow behind one clean interface.

    Encapsulates: Prophet demand distribution → DEA efficiency scoring →
    ideal/nadir anchoring → global-criterion MILP solve, and exposes the
    result as a structured DataFrame of the selected suppliers — the handoff
    artifact downstream stages (e.g. the Stage-2 bargaining game) consume.

        engine = DemandOptimizationEngine(w1=0.6, w2=0.4)
        plan = engine.run()                  # dict (solver output)
        df = engine.results_frame()          # DataFrame, one row per selected
                                             # supplier with q_k*, costs, DEA

    Expensive inputs (the Prophet fit, the DEA solves, the anchors) are
    computed once on first use and cached, so re-solving at different weights
    is cheap.
    """

    def __init__(
        self,
        suppliers: dict = SUPPLIERS,
        w1: float = 0.6,
        w2: float = 0.4,
    ):
        self.suppliers = suppliers
        self.w1, self.w2 = w1, w2
        self.demand_dist: dict | None = None     # D and its confidence band
        self.efficiency: dict | None = None      # DEA score per supplier
        self.anchors: dict | None = None         # ideal/nadir per objective
        self.result: dict | None = None          # last solve

    def _prepare(self) -> None:
        """Forecast, score, and anchor once; cache for subsequent solves."""
        if self.demand_dist is None:
            self.demand_dist = annual_demand()
        if self.efficiency is None:
            self.efficiency = supplier_efficiency(self.suppliers)
        if self.anchors is None:
            self.anchors = anchor_points(
                self.demand_dist["D"], self.efficiency,
                self.demand_dist["horizon_days"], self.suppliers,
            )

    def run(self, w1: float | None = None, w2: float | None = None) -> dict:
        """Execute the full pipeline; returns the solver result dict."""
        if w1 is not None:
            self.w1 = w1
        if w2 is not None:
            self.w2 = w2
        self._prepare()
        self.result = solve_weighted(
            self.w1, self.w2,
            self.demand_dist["D"], self.efficiency,
            self.demand_dist["horizon_days"], self.anchors, self.suppliers,
        )
        return self.result

    def results_frame(self) -> "pd.DataFrame":
        """The selected suppliers as a structured DataFrame.

        One row per *selected* supplier: optimal quantity q_k*, share of D,
        unit economics, annual cost components, and the DEA score. Sorted by
        allocated volume, ready to hand to Stage 2.
        """
        import pandas as pd

        if self.result is None:
            self.run()
        r, s = self.result, self.suppliers
        D = self.demand_dist["D"]
        rows = []
        for k in r["selected"]:
            q = r["allocation"][k]
            rows.append({
                "supplier": k,
                "q_star": q,
                "share_of_D": round(q / D, 4),
                "unit_cost": s[k]["unit_cost"],
                "purchasing_cost": round(s[k]["unit_cost"] * q, 2),
                "holding_cost": round(s[k]["holding_cost"] * q * 0.5, 2),
                "setup_cost": s[k]["setup_cost"],
                "dea_efficiency": self.efficiency[k],
                "defect_rate": s[k]["defect_rate"],
                "delivery_time": s[k]["delivery_time"],
            })
        return (
            pd.DataFrame(rows)
            .sort_values("q_star", ascending=False)
            .reset_index(drop=True)
        )


def risk_sweep(n: int = 10) -> dict:
    """The full Stage-1 pipeline plus the risk sweep.

    Forecasts D once, scores DEA once, anchors the ideal points, then sweeps n
    (w1, w2) combinations from purely cost-driven (w1=1) to purely
    efficiency-driven (w2=1), solving the global-criterion MILP at each point.
    """
    dist = annual_demand()
    eff = supplier_efficiency()
    periods = dist["horizon_days"]
    anchors = anchor_points(dist["D"], eff, periods)

    runs = []
    for i in range(n):
        w2 = i / (n - 1)
        w1 = 1.0 - w2
        runs.append(solve_weighted(w1, w2, dist["D"], eff, periods, anchors))
    return {"demand": dist, "efficiency": eff, **anchors, "runs": runs}


def plot_pareto(sweep: dict, path: str = PARETO_PNG) -> str:
    """Visualise the cost-vs-efficiency trade-off curve from the risk sweep."""
    import matplotlib
    matplotlib.use("Agg")                    # headless: render straight to file
    import matplotlib.pyplot as plt

    runs = sweep["runs"]
    # Several weights can land on the same Pareto point; group them so each
    # point gets one label ("w1=1.00-0.78") instead of overprinted text.
    points: dict[tuple, list[float]] = {}
    for r in runs:
        points.setdefault((r["Z1_cost"], r["Z2_efficiency"]), []).append(r["w1"])

    cost = [z1 / 1e6 for z1, _ in points]
    effs = [z2 for _, z2 in points]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(cost, effs, "o-", color="#1f77b4", linewidth=1.5, markersize=7)
    for ((z1, z2), weights) in points.items():
        label = (f"w1={max(weights):.2f}" if len(weights) == 1
                 else f"w1={max(weights):.2f}–{min(weights):.2f}")
        ax.annotate(
            label, (z1 / 1e6, z2),
            textcoords="offset points", xytext=(8, -4), fontsize=8,
        )
    ax.set_xlabel("Z1 — total annual cost ($M)")
    ax.set_ylabel("Z2 — Σ DEA efficiency of selected suppliers")
    ax.set_title(
        "Stage 1 Pareto frontier: cost vs supplier-base efficiency\n"
        f"(D = {sweep['demand']['D']:,.0f} units, global criterion sweep)"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("Stage 1 — forecast-driven supplier selection & order allocation\n")

    sweep = risk_sweep(n=10)
    dist = sweep["demand"]

    print(f"Demand D = {dist['D']:,.0f} units "
          f"(90% interval {dist['D_lower']:,.0f} .. {dist['D_upper']:,.0f})")
    print(f"Anchors: Z1 ${sweep['z1_ideal']:,.0f} (ideal) .. "
          f"${sweep['z1_nadir']:,.0f} (nadir)   "
          f"Z2 {sweep['z2_nadir']:.3f} (nadir) .. "
          f"{sweep['z2_ideal']:.3f} (ideal)\n")

    hdr = (f"{'w1':>5} {'w2':>5} {'Z1 cost ($)':>14} {'Z2 eff':>7} "
           f"{'#sup':>5}  selected")
    print(hdr)
    print("-" * (len(hdr) + 18))
    for r in sweep["runs"]:
        print(f"{r['w1']:>5.2f} {r['w2']:>5.2f} {r['Z1_cost']:>14,.0f} "
              f"{r['Z2_efficiency']:>7.3f} {len(r['selected']):>5}  "
              f"{', '.join(r['selected'])}")

    png = plot_pareto(sweep)
    print(f"\nPareto frontier saved -> {png}")
