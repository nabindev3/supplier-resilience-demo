"""Order allocation MILP plus a post-commit disruption stress test.

The 2021 paper allocates orders to DEA-efficient suppliers at minimum cost
but is fully deterministic: no supplier ever fails. This module keeps that
allocation core and adds two resilience levers (a per-supplier share cap and
a minimum number of active suppliers) plus stress_test(), which knocks out a
supplier *after* orders are committed and counts what survives.

The `unmet` variable with a big penalty keeps the model feasible no matter
what, so a disruption shows up as lost service instead of an infeasible solve.
"""

import pulp

STOCKOUT_PENALTY = 1e6  # cost per unit of demand left unmet


def allocate(
    suppliers: dict,
    demand: float,
    efficiency: dict,
    efficiency_weight: float = 0.0,
    max_share: float = 1.0,
    min_suppliers: int = 1,
    disruption: dict | None = None,
):
    """Allocate `demand` units across suppliers.

    suppliers: {name: {"price", "capacity", "min_order"}}
    efficiency: DEA scores from dea.ccr_input_efficiency
    efficiency_weight: per-unit reward for using efficient suppliers
        (0 = pure cost minimisation, as in the 2021 model)
    max_share: cap on any one supplier's fraction of demand (1.0 = off)
    min_suppliers: minimum number of suppliers used (1 = off)
    disruption: {name: remaining_capacity_fraction}, e.g. {"S2": 0.0}
    """
    disruption = disruption or {}
    names = list(suppliers.keys())

    prob = pulp.LpProblem("order_allocation", pulp.LpMinimize)
    q = {j: pulp.LpVariable(f"q_{j}", lowBound=0) for j in names}
    use = {j: pulp.LpVariable(f"use_{j}", cat="Binary") for j in names}
    unmet = pulp.LpVariable("unmet", lowBound=0)

    prob += (
        pulp.lpSum(suppliers[j]["price"] * q[j] for j in names)
        + STOCKOUT_PENALTY * unmet
        - efficiency_weight * pulp.lpSum(efficiency[j] * q[j] for j in names)
    )

    prob += pulp.lpSum(q[j] for j in names) + unmet == demand

    for j in names:
        eff_cap = suppliers[j]["capacity"] * disruption.get(j, 1.0)
        prob += q[j] <= eff_cap * use[j]
        prob += q[j] >= suppliers[j]["min_order"] * use[j]
        prob += q[j] <= max_share * demand

    prob += pulp.lpSum(use[j] for j in names) >= min_suppliers

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    alloc = {j: round(q[j].value() or 0.0, 2) for j in names}
    unmet_units = round(unmet.value() or 0.0, 2)
    purchasing_cost = round(
        sum(suppliers[j]["price"] * alloc[j] for j in names), 2
    )
    served = demand - unmet_units
    return {
        "status": pulp.LpStatus[prob.status],
        "allocation": alloc,
        "purchasing_cost": purchasing_cost,
        "unmet_units": unmet_units,
        "service_level": round(served / demand, 4) if demand else 1.0,
        "active_suppliers": [j for j in names if (use[j].value() or 0) > 0.5],
    }


def stress_test(suppliers: dict, demand: float, allocation: dict, disruption: dict):
    """Apply a disruption to an already-committed plan, no re-optimisation.

    Orders were placed before the failure was known, so anything committed
    above a supplier's surviving capacity is lost. A plan concentrated on one
    supplier loses far more than a diversified one.
    """
    fulfilled = 0.0
    lost = 0.0
    for j, committed in allocation.items():
        surviving_cap = suppliers[j]["capacity"] * disruption.get(j, 1.0)
        kept = min(committed, surviving_cap)
        fulfilled += kept
        lost += committed - kept
    return {
        "fulfilled_units": round(fulfilled, 2),
        "lost_units": round(lost, 2),
        "realized_service_level": round(fulfilled / demand, 4) if demand else 1.0,
    }
