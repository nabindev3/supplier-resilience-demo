"""Illustrative supplier data (synthetic, in the spirit of the 2021 case data).

Six suppliers described by the criteria a DEA model needs:
  INPUTS  (minimise): unit price, lead time (days), defect rate (%)
  OUTPUTS (maximise): quality score (0-100), on-time delivery (%), capacity (u)

The same `capacity` and `price` feed the allocation model, so DEA scoring and
order allocation stay consistent.
"""

SUPPLIERS = {
    "S1": {"price": 10.0, "lead_time": 7,  "defect": 1.5, "quality": 92, "on_time": 97, "capacity": 400, "min_order": 50},
    "S2": {"price":  8.5, "lead_time": 12, "defect": 3.0, "quality": 80, "on_time": 88, "capacity": 600, "min_order": 50},
    "S3": {"price": 11.5, "lead_time": 5,  "defect": 0.8, "quality": 96, "on_time": 99, "capacity": 300, "min_order": 50},
    "S4": {"price":  9.0, "lead_time": 9,  "defect": 2.2, "quality": 85, "on_time": 91, "capacity": 500, "min_order": 50},
    "S5": {"price": 12.0, "lead_time": 6,  "defect": 1.0, "quality": 94, "on_time": 98, "capacity": 250, "min_order": 50},
    "S6": {"price":  8.0, "lead_time": 14, "defect": 4.0, "quality": 72, "on_time": 82, "capacity": 700, "min_order": 50},
}

DEFAULT_DEMAND = 1000


def dea_arrays(suppliers: dict):
    """Split the supplier table into DEA input/output arrays."""
    inputs = {j: [s["price"], s["lead_time"], s["defect"]] for j, s in suppliers.items()}
    outputs = {j: [s["quality"], s["on_time"], s["capacity"]] for j, s in suppliers.items()}
    return inputs, outputs
