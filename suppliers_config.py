"""The 10-supplier candidate pool for the forecast-driven stage-1 model.

data.py keeps the original 6-supplier case; this is the bigger pool. The
spread is intentional: cheap/slow/high-defect volume vendors at one end
(S01, S02, S09), expensive/fast/reliable niche vendors at the other (S07,
S10), so cost and efficiency actually pull in different directions.

Units: capacity and min_order are per day (scale by the horizon, see
annual_capacity), unit_cost and holding_cost in $/unit, setup_cost in $/year
if the supplier is used at all, delivery_time in days, defect_rate a
fraction, quality and on_time on 0-100 scales.

production_cost ($/unit) is what it costs the supplier to make one unit;
unit_cost minus production_cost is their margin, which is what stage 2
bargains over. Margins widen toward the premium end (about 9% for the volume
vendors up to ~14% for S10) — niche suppliers price further above cost, so
they have more room to concede before hitting their walk-away point.

Calibration note: the demand series is now real (UCI Online Retail II total
volume, ~6.2M units/year; see online_retail.py), so the three scale-dependent
fields — capacity, min_order and setup_cost — were multiplied by 12 from their
original illustrative values to match that magnitude. Every ratio is
unchanged, so the cost/quality/efficiency trade-offs the model exercises are
identical; only the scale is real. The per-unit economics (unit_cost,
production_cost, margins, quality, defect, lead time) stay as set — real
supplier production costs are commercially confidential and not public, so
those remain a calibrated assumption, documented as such.
"""

SUPPLIERS: dict[str, dict] = {
    "S01": {"capacity": 14400, "unit_cost":  8.00, "production_cost":  7.30, "holding_cost": 0.90, "defect_rate": 0.040, "delivery_time": 14, "quality": 88, "on_time": 84, "setup_cost": 60000, "min_order": 1200},
    "S02": {"capacity": 10800, "unit_cost":  8.50, "production_cost":  7.70, "holding_cost": 0.85, "defect_rate": 0.030, "delivery_time": 12, "quality": 90, "on_time": 87, "setup_cost": 54000, "min_order":  960},
    "S03": {"capacity":  6000, "unit_cost":  9.00, "production_cost":  8.05, "holding_cost": 0.70, "defect_rate": 0.022, "delivery_time":  9, "quality": 93, "on_time": 91, "setup_cost": 42000, "min_order":  600},
    "S04": {"capacity":  7800, "unit_cost":  9.50, "production_cost":  8.45, "holding_cost": 0.75, "defect_rate": 0.018, "delivery_time": 10, "quality": 94, "on_time": 90, "setup_cost": 45600, "min_order":  720},
    "S05": {"capacity":  4800, "unit_cost": 10.00, "production_cost":  8.80, "holding_cost": 0.60, "defect_rate": 0.015, "delivery_time":  7, "quality": 95, "on_time": 94, "setup_cost": 36000, "min_order":  600},
    "S06": {"capacity":  4200, "unit_cost": 10.50, "production_cost":  9.20, "holding_cost": 0.55, "defect_rate": 0.012, "delivery_time":  6, "quality": 96, "on_time": 95, "setup_cost": 33600, "min_order":  480},
    "S07": {"capacity":  3600, "unit_cost": 11.50, "production_cost":  9.95, "holding_cost": 0.50, "defect_rate": 0.008, "delivery_time":  5, "quality": 97, "on_time": 96, "setup_cost": 30000, "min_order":  480},
    "S08": {"capacity":  6600, "unit_cost":  9.25, "production_cost":  8.25, "holding_cost": 0.72, "defect_rate": 0.020, "delivery_time": 11, "quality": 93, "on_time": 89, "setup_cost": 43200, "min_order":  720},
    "S09": {"capacity":  9600, "unit_cost":  8.75, "production_cost":  7.90, "holding_cost": 0.80, "defect_rate": 0.028, "delivery_time": 13, "quality": 91, "on_time": 85, "setup_cost": 50400, "min_order":  960},
    "S10": {"capacity":  3000, "unit_cost": 12.00, "production_cost": 10.30, "holding_cost": 0.48, "defect_rate": 0.006, "delivery_time":  4, "quality": 98, "on_time": 98, "setup_cost": 24000, "min_order":  360},
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
