"""
Order-allocation module — this is where the demo *bridges* Yousefi's 2021
deterministic model into his CURRENT research agenda (2022-2026): supply-chain
resilience and disruption-risk management (Ontario Tech / formerly UBC Okanagan).

The 2021 paper allocates orders to DEA-efficient suppliers to minimise cost.
We keep that core, then add the dimension his own follow-up work opened up
(see Jahangoshai Rezaee et al., "Multi-stage hybrid model ... considering
disruption risks," Int. J. Production Economics, 2021):

  * a *disruption scenario* that knocks out / degrades supplier capacity, and
  * *resilience constraints* (single-sourcing cap + minimum active suppliers)
    that trade a little cost for the ability to absorb a disruption.

Solved as a MILP with PuLP/CBC. An `unmet` variable with a stockout penalty
keeps the model always feasible, so a disruption shows up honestly as lost
service rather than an infeasible solve.
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

    suppliers: {name: {"price": float, "capacity": float, "min_order": float}}
    efficiency: {name: DEA score in (0,1]}  -- from dea.ccr_input_efficiency
    efficiency_weight: reward (per unit) for routing orders to efficient
        suppliers; 0 == pure cost minimisation (faithful to the 2021 cost goal).
    max_share: cap on the fraction of demand any one supplier may serve
        (resilience lever; 1.0 disables it).
    min_suppliers: minimum number of suppliers that must be used
        (resilience lever; 1 disables it -> single sourcing allowed).
    disruption: {name: remaining_capacity_fraction}  e.g. {"S2": 0.0} downs S2.

    Returns dict with allocation, cost, unmet units, service level, status.
    """
    disruption = disruption or {}
    names = list(suppliers.keys())

    prob = pulp.LpProblem("order_allocation", pulp.LpMinimize)
    q = {j: pulp.LpVariable(f"q_{j}", lowBound=0) for j in names}
    use = {j: pulp.LpVariable(f"use_{j}", cat="Binary") for j in names}
    unmet = pulp.LpVariable("unmet", lowBound=0)

    # Objective: purchasing cost + stockout penalty - efficiency reward
    prob += (
        pulp.lpSum(suppliers[j]["price"] * q[j] for j in names)
        + STOCKOUT_PENALTY * unmet
        - efficiency_weight * pulp.lpSum(efficiency[j] * q[j] for j in names)
    )

    # Meet demand (or record the shortfall)
    prob += pulp.lpSum(q[j] for j in names) + unmet == demand

    for j in names:
        eff_cap = suppliers[j]["capacity"] * disruption.get(j, 1.0)
        prob += q[j] <= eff_cap * use[j]                       # capacity + link
        prob += q[j] >= suppliers[j]["min_order"] * use[j]     # min order if used
        prob += q[j] <= max_share * demand                     # anti single-source

    prob += pulp.lpSum(use[j] for j in names) >= min_suppliers  # multi-sourcing

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
    """Apply a disruption to an *already-committed* plan (no re-optimisation).

    This is the honest resilience metric: orders were placed before anyone knew
    a supplier would fail, so units committed above a supplier's surviving
    capacity are simply lost. A plan concentrated on one supplier loses far more
    than a diversified one -- the whole point of the resilience levers.

    Returns realised fulfilled units, lost units, and realised service level.
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
