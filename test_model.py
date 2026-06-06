"""Smoke tests for the DEA + allocation + resilience engine.

Runnable two ways:
    python test_model.py        # plain runner, no extra dependency
    python -m pytest            # if pytest is installed
"""

from data import SUPPLIERS, DEFAULT_DEMAND, dea_arrays
from dea import ccr_input_efficiency
from allocation import allocate, stress_test


def _efficiency():
    inp, out = dea_arrays(SUPPLIERS)
    return ccr_input_efficiency(inp, out)


def test_dea_scores_in_unit_interval():
    """Every CCR efficiency score must lie in (0, 1]."""
    eff = _efficiency()
    assert eff, "no scores returned"
    for name, score in eff.items():
        assert 0.0 < score <= 1.0 + 1e-9, f"{name} score out of range: {score}"


def test_allocation_meets_demand():
    """A feasible allocation should fully serve demand (no stockout)."""
    eff = _efficiency()
    plan = allocate(SUPPLIERS, DEFAULT_DEMAND, eff, efficiency_weight=0.0)
    assert plan["unmet_units"] == 0
    assert abs(sum(plan["allocation"].values()) - DEFAULT_DEMAND) < 1e-6


def test_resilient_plan_diversifies():
    """Resilience levers should force more suppliers and a share cap."""
    eff = _efficiency()
    cost = allocate(SUPPLIERS, DEFAULT_DEMAND, eff, efficiency_weight=0.0)
    res = allocate(
        SUPPLIERS, DEFAULT_DEMAND, eff,
        efficiency_weight=1.5, max_share=0.4, min_suppliers=3,
    )
    assert len(res["active_suppliers"]) >= 3
    assert len(res["active_suppliers"]) > len(cost["active_suppliers"])
    # diversification is not free
    assert res["purchasing_cost"] >= cost["purchasing_cost"]


def test_resilience_pays_off_under_disruption():
    """When a high-volume supplier fails, the resilient plan retains more service."""
    eff = _efficiency()
    cost = allocate(SUPPLIERS, DEFAULT_DEMAND, eff, efficiency_weight=0.0)
    res = allocate(
        SUPPLIERS, DEFAULT_DEMAND, eff,
        efficiency_weight=1.5, max_share=0.4, min_suppliers=3,
    )
    disruption = {"S6": 0.0}  # the cheap, high-volume supplier collapses
    cost_hit = stress_test(SUPPLIERS, DEFAULT_DEMAND, cost["allocation"], disruption)
    res_hit = stress_test(SUPPLIERS, DEFAULT_DEMAND, res["allocation"], disruption)
    assert res_hit["realized_service_level"] > cost_hit["realized_service_level"]


if __name__ == "__main__":
    tests = [
        test_dea_scores_in_unit_interval,
        test_allocation_meets_demand,
        test_resilient_plan_diversifies,
        test_resilience_pays_off_under_disruption,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
