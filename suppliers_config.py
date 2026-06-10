"""The 10-supplier candidate pool for the forecast-driven stage-1 model.

data.py keeps the original 6-supplier case; this is the bigger pool. The
spread is intentional: cheap/slow/high-defect volume vendors at one end
(S01, S02, S09), expensive/fast/reliable niche vendors at the other (S07,
S10), so cost and efficiency actually pull in different directions.

Units: capacity and min_order are per day (scale by the horizon, see
annual_capacity), unit_cost and holding_cost in $/unit, setup_cost in $/year
if the supplier is used at all, delivery_time in days, defect_rate a
fraction, quality and on_time on 0-100 scales.
"""

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

TOTAL_CAPACITY = sum(s["capacity"] for s in SUPPLIERS.values())  # units/day


def annual_capacity(suppliers: dict = SUPPLIERS, periods: int = 365) -> dict:
    """Per-day capacity scaled to a `periods`-day horizon, since the forecast
    gives annual demand."""
    return {j: s["capacity"] * periods for j, s in suppliers.items()}


def allocation_view(suppliers: dict = SUPPLIERS) -> dict:
    """Adapt this config to what allocation.allocate expects (price/capacity/
    min_order)."""
    return {
        j: {
            "price": s["unit_cost"],
            "capacity": s["capacity"],
            "min_order": s["min_order"],
        }
        for j, s in suppliers.items()
    }


def dea_arrays(suppliers: dict = SUPPLIERS):
    """DEA split: cost in, quality and on-time delivery out.

    The question this asks is which supplier turns each dollar into the most
    quality and reliability.
    """
    inputs = {j: [s["unit_cost"]] for j, s in suppliers.items()}
    outputs = {j: [s["quality"], s["on_time"]] for j, s in suppliers.items()}
    return inputs, outputs


if __name__ == "__main__":
    print(f"{len(SUPPLIERS)} suppliers, {TOTAL_CAPACITY:,} units/day total\n")
    for j, s in SUPPLIERS.items():
        print(f"{j}  cap {s['capacity']:>5}/d  ${s['unit_cost']:>5.2f}/u  "
              f"defect {s['defect_rate']*100:.1f}%  lead {s['delivery_time']}d  "
              f"quality {s['quality']}  on-time {s['on_time']}%")
