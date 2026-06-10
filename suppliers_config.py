"""Stage 1 (part 2) — candidate supplier configuration.

A static, version-controlled catalogue of **10 candidate suppliers** for the
order-allocation stage. Each supplier carries the cost and performance metrics
the optimisation needs:

    capacity       per-day throughput (units/day); scale by the horizon for an
                   annual plan — see `annual_capacity`
    unit_cost      purchase price ($/unit)            — DEA input  (minimise)
    holding_cost   inventory carrying cost ($/unit held for the period)
    defect_rate    fraction of units defective (0–1)
    delivery_time  lead time in days
    quality        quality score 0–100                — DEA output (maximise)
    on_time        on-time delivery %                 — DEA output (maximise)
    setup_cost     annual fixed cost ($) of engaging the supplier at all
                   (contracting, onboarding, audits) — incurred once if y_k = 1
    min_order      minimum order quantity (units/day) if the supplier is used;
                   scales with the horizon like capacity

This is the richer of the two catalogues in the repo: `data.py` holds the
original 6-supplier DEA case from Yousefi et al. (2021); this file is the
expanded 10-supplier pool used by the demand-forecasting Stage-1 pipeline. The
spread is deliberate — cheap/slow/sloppy high-capacity vendors at one end,
pricey/fast/reliable niche vendors at the other — so cost-minimisation and
resilience pull in genuinely different directions.
"""

from __future__ import annotations

# name -> metrics. Costs in $, capacity/min_order in units, time in days,
# defect_rate as a fraction in [0, 1].
SUPPLIERS: dict[str, dict] = {
    "S01": {"capacity": 1200, "unit_cost":  8.00, "holding_cost": 0.90, "defect_rate": 0.040, "delivery_time": 14, "quality": 88, "on_time": 84, "setup_cost": 5000, "min_order": 100},
    "S02": {"capacity":  900, "unit_cost":  8.50, "holding_cost": 0.85, "defect_rate": 0.030, "delivery_time": 12, "quality": 90, "on_time": 87, "setup_cost": 4500, "min_order":  80},
    "S03": {"capacity":  500, "unit_cost":  9.00, "holding_cost": 0.70, "defect_rate": 0.022, "delivery_time":  9, "quality": 93, "on_time": 91, "setup_cost": 3500, "min_order":  50},
    "S04": {"capacity":  650, "unit_cost":  9.50, "holding_cost": 0.75, "defect_rate": 0.018, "delivery_time": 10, "quality": 94, "on_time": 90, "setup_cost": 3800, "min_order":  60},
    "S05": {"capacity":  400, "unit_cost": 10.00, "holding_cost": 0.60, "defect_rate": 0.015, "delivery_time":  7, "quality": 95, "on_time": 94, "setup_cost": 3000, "min_order":  50},
    "S06": {"capacity":  350, "unit_cost": 10.50, "holding_cost": 0.55, "defect_rate": 0.012, "delivery_time":  6, "quality": 96, "on_time": 95, "setup_cost": 2800, "min_order":  40},
    "S07": {"capacity":  300, "unit_cost": 11.50, "holding_cost": 0.50, "defect_rate": 0.008, "delivery_time":  5, "quality": 97, "on_time": 96, "setup_cost": 2500, "min_order":  40},
    "S08": {"capacity":  550, "unit_cost":  9.25, "holding_cost": 0.72, "defect_rate": 0.020, "delivery_time": 11, "quality": 93, "on_time": 89, "setup_cost": 3600, "min_order":  60},
    "S09": {"capacity":  800, "unit_cost":  8.75, "holding_cost": 0.80, "defect_rate": 0.028, "delivery_time": 13, "quality": 91, "on_time": 85, "setup_cost": 4200, "min_order":  80},
    "S10": {"capacity":  250, "unit_cost": 12.00, "holding_cost": 0.48, "defect_rate": 0.006, "delivery_time":  4, "quality": 98, "on_time": 98, "setup_cost": 2000, "min_order":  30},
}

# Total per-day pool throughput, useful for sanity-checking demand against what
# the supplier base can absorb (multiply by the horizon for the annual figure).
TOTAL_CAPACITY = sum(s["capacity"] for s in SUPPLIERS.values())


def annual_capacity(suppliers: dict = SUPPLIERS, periods: int = 365) -> dict:
    """Scale per-day throughput to capacity over a `periods`-day horizon.

    The forecast produces an *annual* demand D, so the allocation must compare it
    against annual capacity: each supplier can supply `capacity * periods` units
    over the horizon. Returns {supplier: annual_capacity}.
    """
    return {j: s["capacity"] * periods for j, s in suppliers.items()}


def allocation_view(suppliers: dict = SUPPLIERS) -> dict:
    """Adapt the rich config to the schema `allocation.allocate` expects.

    The allocator works in terms of `price`, `capacity`, and `min_order`; here
    `unit_cost` plays the role of `price`.
    """
    return {
        j: {
            "price": s["unit_cost"],
            "capacity": s["capacity"],
            "min_order": s["min_order"],
        }
        for j, s in suppliers.items()
    }


def dea_arrays(suppliers: dict = SUPPLIERS):
    """Split the catalogue into DEA input/output arrays.

    INPUT  to minimise : unit cost.
    OUTPUTS to maximise: quality score and on-time delivery %.
    Cost-in / quality+delivery-out is the classic supplier-DEA framing — it asks
    "which supplier turns each dollar into the most quality and reliability?".
    Mirrors `data.dea_arrays` so either catalogue can feed `dea.py`.
    """
    inputs = {j: [s["unit_cost"]] for j, s in suppliers.items()}
    outputs = {j: [s["quality"], s["on_time"]] for j, s in suppliers.items()}
    return inputs, outputs


if __name__ == "__main__":
    print(f"{len(SUPPLIERS)} candidate suppliers  |  total capacity "
          f"{TOTAL_CAPACITY:,} units\n")
    hdr = (f"{'id':<4} {'cap':>5} {'cost':>6} {'hold':>5} {'defect':>7} "
           f"{'lead':>5} {'qual':>5} {'ontime':>7} {'minQ':>5}")
    print(hdr)
    print("-" * len(hdr))
    for j, s in SUPPLIERS.items():
        print(f"{j:<4} {s['capacity']:>5} {s['unit_cost']:>6.2f} "
              f"{s['holding_cost']:>5.2f} {s['defect_rate']*100:>6.1f}% "
              f"{s['delivery_time']:>4}d {s['quality']:>5} {s['on_time']:>6}% "
              f"{s['min_order']:>5}")
