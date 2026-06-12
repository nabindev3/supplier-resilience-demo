"""Stage 1: forecast-driven supplier selection and order allocation.

Prophet gives annual demand D (forecast.py), DEA scores the 10 candidates
(suppliers_config.py), and a MILP decides who supplies what. Two objectives,
following the structure of Yousefi et al. (2021):

    Z1 = purchasing + holding + setup cost              (minimise)
    Z2 = sum of DEA scores over selected suppliers      (maximise)

combined with the weighted global criterion method. risk_sweep() re-solves at
10 weight settings from cost-driven to efficiency-driven and plot_pareto()
draws the resulting trade-off curve.
"""

import pandas as pd
import pulp

from allocation import stress_test
from suppliers_config import SUPPLIERS, dea_arrays, annual_capacity
from dea import ccr_input_efficiency
from forecast import annual_demand

PARETO_PNG = "pareto_frontier.png"
RESILIENCE_PNG = "resilience_frontier.png"

# normalised objective coefficients come out tiny (unit_cost / Z1* is around
# 1e-6), scale up so CBC's tolerances don't swallow them
OBJ_SCALE = 1000.0


def supplier_efficiency(suppliers: dict = SUPPLIERS) -> dict:
    return ccr_input_efficiency(*dea_arrays(suppliers))


class Stage1Model:
    """One MILP instance.

    Variables: y_k binary (supplier selected) and q_k >= 0 (units ordered).
    Constraints: sum(q) == D, q_k <= capacity_k * y_k, q_k >= min_order_k * y_k.

    The min-order link is load-bearing for Z2: without it the solver would
    set y_k = 1 everywhere and collect efficiency points without ordering
    anything. Selecting a supplier has to commit real volume.
    """

    def __init__(self, suppliers: dict, demand: float, efficiency: dict,
                 periods: int = 365):
        self.suppliers = suppliers
        self.names = list(suppliers)
        self.demand = demand
        self.efficiency = efficiency
        # capacity/min_order are per day, D covers the whole horizon
        self.capacity = annual_capacity(suppliers, periods)
        self.min_order = {k: s["min_order"] * periods for k, s in suppliers.items()}

        self.prob = pulp.LpProblem("stage1_order_allocation", pulp.LpMinimize)
        self.y = {k: pulp.LpVariable(f"y_{k}", cat="Binary") for k in self.names}
        self.q = {k: pulp.LpVariable(f"q_{k}", lowBound=0) for k in self.names}

        self.prob += pulp.lpSum(self.q.values()) == demand, "demand_balance"
        for k in self.names:
            self.prob += self.q[k] <= self.capacity[k] * self.y[k], f"cap_{k}"
            self.prob += self.q[k] >= self.min_order[k] * self.y[k], f"minq_{k}"

    def z1_cost(self):
        """Purchasing + holding + setup. Holding charges q/2, the average
        cycle stock over the year."""
        s = self.suppliers
        return (
            pulp.lpSum(s[k]["unit_cost"] * self.q[k] for k in self.names)
            + pulp.lpSum(s[k]["holding_cost"] * 0.5 * self.q[k] for k in self.names)
            + pulp.lpSum(s[k]["setup_cost"] * self.y[k] for k in self.names)
        )

    def z2_efficiency(self):
        return pulp.lpSum(self.efficiency[k] * self.y[k] for k in self.names)

    def set_global_criterion(self, w1, w2, z1_ideal, z2_ideal, z1_nadir, z2_nadir):
        """min  w1*(Z1 - Z1*)/(Z1n - Z1*) + w2*(Z2* - Z2)/(Z2* - Z2n)

        Deviations are normalised by each objective's ideal-to-nadir range,
        not by the ideal value. I tried ideal-normalisation first and the
        sweep collapsed to ~3 points: cost only moves about 5% off its ideal
        while the efficiency sum moves about 80%, so the Z2 term dominated at
        almost any weight. Range scaling puts both on [0, 1].
        """
        z1_range = max(z1_nadir - z1_ideal, 1e-9)
        z2_range = max(z2_ideal - z2_nadir, 1e-9)
        self.prob += OBJ_SCALE * (
            w1 * (self.z1_cost() - z1_ideal) * (1.0 / z1_range)
            + w2 * (z2_ideal - self.z2_efficiency()) * (1.0 / z2_range)
        )
        return self

    def solve(self) -> dict:
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
            "Z2_efficiency": round(sum(self.efficiency[k] for k in selected), 4),
        }


def anchor_points(demand, eff, periods, suppliers=SUPPLIERS) -> dict:
    """Ideal and nadir values for both objectives.

    Z1* = min cost ignoring efficiency; Z2* = max efficiency ignoring cost.
    Nadirs come out of the same solves: Z2 at the cost-only optimum, and the
    cheapest Z1 that still reaches Z2* (re-solve with Z2 pinned).
    """
    m1 = Stage1Model(suppliers, demand, eff, periods)
    m1.prob += m1.z1_cost()
    r1 = m1.solve()
    z1_star, z2_nadir = r1["Z1_cost"], r1["Z2_efficiency"]

    m2 = Stage1Model(suppliers, demand, eff, periods)
    m2.prob += -m2.z2_efficiency()  # minimise the negative
    z2_star = m2.solve()["Z2_efficiency"]

    m3 = Stage1Model(suppliers, demand, eff, periods)
    m3.prob += m3.z2_efficiency() >= z2_star - 1e-6, "pin_z2"
    m3.prob += m3.z1_cost()
    z1_nadir = m3.solve()["Z1_cost"]

    return {
        "z1_ideal": z1_star,
        "z2_ideal": z2_star,
        "z1_nadir": z1_nadir,
        "z2_nadir": z2_nadir,
    }


def solve_weighted(w1, w2, demand, eff, periods, anchors, suppliers=SUPPLIERS) -> dict:
    model = Stage1Model(suppliers, demand, eff, periods).set_global_criterion(
        w1, w2,
        anchors["z1_ideal"], anchors["z2_ideal"],
        anchors["z1_nadir"], anchors["z2_nadir"],
    )
    result = model.solve()
    result["w1"], result["w2"] = round(w1, 3), round(w2, 3)
    return result


class DemandOptimizationEngine:
    """The whole stage-1 pipeline behind one interface.

        engine = DemandOptimizationEngine(w1=0.6, w2=0.4)
        engine.run()
        df = engine.results_frame()

    The Prophet fit, DEA scores and anchor solves are cached after the first
    run, so re-solving at different weights only costs a CBC call.
    """

    def __init__(self, suppliers: dict = SUPPLIERS, w1: float = 0.6, w2: float = 0.4):
        self.suppliers = suppliers
        self.w1, self.w2 = w1, w2
        self.demand_dist = None
        self.efficiency = None
        self.anchors = None
        self.result = None

    def _prepare(self):
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

    def results_frame(self) -> pd.DataFrame:
        """Selected suppliers as a DataFrame: q*, share of D, costs, DEA
        score. This is the handoff to stage 2."""
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
    """Solve at n weight settings from (w1=1, w2=0) to (w1=0, w2=1).

    The forecast and DEA scoring run once; only the MILP is re-solved.
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


def disruption_service(sweep: dict, disruption: dict | None = None) -> list[dict]:
    """Stress every plan on the frontier with the same disruption.

    This is what connects the two halves of the project: the weight sweep
    prices diversification (Z1 vs Z2), and the stress test shows what that
    diversification actually buys when a supplier fails *after* orders are
    committed. Default scenario: S01 fully down — the cheap supplier that
    every cost-leaning plan leans on hardest.

    Returns one row per sweep run with the realised service level added.
    """
    disruption = disruption or {"S01": 0.0}
    periods = sweep["demand"]["horizon_days"]
    D = sweep["demand"]["D"]
    annual = {k: {"capacity": s["capacity"] * periods}
              for k, s in SUPPLIERS.items()}
    rows = []
    for r in sweep["runs"]:
        hit = stress_test(annual, D, r["allocation"], disruption)
        rows.append({
            "w1": r["w1"],
            "Z1_cost": r["Z1_cost"],
            "Z2_efficiency": r["Z2_efficiency"],
            "n_suppliers": len(r["selected"]),
            "service": hit["realized_service_level"],
        })
    return rows


def plot_resilience(rows: list[dict], disruption_label: str = "S01 down",
                    path: str = RESILIENCE_PNG) -> str:
    """Cost vs realised service under disruption — the third axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = {}
    for r in rows:
        points.setdefault((r["Z1_cost"], r["service"], r["n_suppliers"]),
                          []).append(r["w1"])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    cost = [z1 / 1e6 for z1, _, _ in points]
    serv = [s * 100 for _, s, _ in points]
    ax.plot(cost, serv, "o-", color="#d62728", linewidth=1.5, markersize=7)
    for (z1, s, n), weights in points.items():
        label = (f"w1={max(weights):.2f}" if len(weights) == 1
                 else f"w1={max(weights):.2f}-{min(weights):.2f}")
        ax.annotate(f"{label} ({n} sup.)", (z1 / 1e6, s * 100),
                    textcoords="offset points", xytext=(8, -4), fontsize=8)
    ax.set_xlabel("Z1, total annual cost ($M)")
    ax.set_ylabel(f"realised service when {disruption_label} (%)")
    ax.set_title("What diversification buys: cost vs service under disruption")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_pareto(sweep: dict, path: str = PARETO_PNG) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # several weights can land on the same solution, label each point once
    points = {}
    for r in sweep["runs"]:
        points.setdefault((r["Z1_cost"], r["Z2_efficiency"]), []).append(r["w1"])

    cost = [z1 / 1e6 for z1, _ in points]
    effs = [z2 for _, z2 in points]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(cost, effs, "o-", color="#1f77b4", linewidth=1.5, markersize=7)
    for (z1, z2), weights in points.items():
        label = (f"w1={max(weights):.2f}" if len(weights) == 1
                 else f"w1={max(weights):.2f}-{min(weights):.2f}")
        ax.annotate(label, (z1 / 1e6, z2),
                    textcoords="offset points", xytext=(8, -4), fontsize=8)
    ax.set_xlabel("Z1, total annual cost ($M)")
    ax.set_ylabel("Z2, sum of DEA scores of selected suppliers")
    ax.set_title("Cost vs supplier-base efficiency "
                 f"(D = {sweep['demand']['D']:,.0f} units)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    sweep = risk_sweep(n=10)
    dist = sweep["demand"]

    print(f"D = {dist['D']:,.0f} units "
          f"({dist['D_lower']:,.0f} to {dist['D_upper']:,.0f} at 90%)")
    print(f"Z1 ideal ${sweep['z1_ideal']:,.0f}, nadir ${sweep['z1_nadir']:,.0f}; "
          f"Z2 ideal {sweep['z2_ideal']:.3f}, nadir {sweep['z2_nadir']:.3f}\n")

    # third axis: what each plan is worth when S01 fails post-commitment
    hits = disruption_service(sweep)
    for r, h in zip(sweep["runs"], hits):
        print(f"w1={r['w1']:.2f}  Z1 ${r['Z1_cost']:>12,.0f}  "
              f"Z2 {r['Z2_efficiency']:.3f}  service {h['service']:>5.0%}  "
              f"{len(r['selected'])} suppliers: {', '.join(r['selected'])}")

    print(f"\nwrote {plot_pareto(sweep)}")
    print(f"wrote {plot_resilience(hits)}")
