"""
Streamlit proof-of-concept: DEA-based supplier selection & order allocation,
extended with disruption-resilience analysis.

Bridges Yousefi, Jahangoshai Rezaee & Solimanpur (2021), "Supplier selection
and order allocation using two-stage hybrid supply chain model and game-based
order price," Operational Research 21(1), 553-588 -- whose Stage 1 scores
suppliers with DEA and allocates orders to the efficient ones -- toward
Dr. Yousefi's current research agenda (supply-chain resilience & disruption-risk
management, Ontario Tech University).

Run:  streamlit run app.py
"""

import pandas as pd
import streamlit as st

from allocation import allocate, stress_test
from data import SUPPLIERS, DEFAULT_DEMAND, dea_arrays
from dea import ccr_input_efficiency

st.set_page_config(page_title="DEA Supplier Allocation + Resilience", layout="wide")

st.title("Supplier Selection, Order Allocation & Disruption Resilience")
st.caption(
    "DEA efficiency scoring + order allocation (after Yousefi et al., 2021) "
    "extended with disruption-resilience analysis — a proof-of-concept bridge "
    "from the 2021 two-stage model to current supply-chain resilience research."
)

# ---------------------------------------------------------------- supplier data
st.subheader("1 · Supplier data")
st.caption(
    "DEA inputs (minimise): price, lead time, defect %. "
    "DEA outputs (maximise): quality, on-time %, capacity. Edit any cell."
)
df = pd.DataFrame(SUPPLIERS).T.reset_index().rename(columns={"index": "supplier"})
edited = st.data_editor(df, hide_index=True, use_container_width=True, key="suppliers")
suppliers = {
    row["supplier"]: {
        "price": float(row["price"]),
        "lead_time": float(row["lead_time"]),
        "defect": float(row["defect"]),
        "quality": float(row["quality"]),
        "on_time": float(row["on_time"]),
        "capacity": float(row["capacity"]),
        "min_order": float(row["min_order"]),
    }
    for _, row in edited.iterrows()
}

# ----------------------------------------------------------------- DEA scoring
inp, out = dea_arrays(suppliers)
efficiency = ccr_input_efficiency(inp, out)

# --------------------------------------------------------------------- sidebar
st.sidebar.header("Scenario controls")
demand = st.sidebar.number_input("Total demand (units)", 100, 100000, DEFAULT_DEMAND, 100)
eff_weight = st.sidebar.slider(
    "Efficiency reward weight", 0.0, 5.0, 1.5, 0.1,
    help="0 = pure cost minimisation (the 2021 cost goal). Higher pulls orders "
         "toward DEA-efficient suppliers.",
)
st.sidebar.markdown("**Resilience levers**")
max_share = st.sidebar.slider(
    "Max share per supplier", 0.1, 1.0, 0.4, 0.05,
    help="Caps how much of demand any single supplier may serve (anti single-sourcing).",
)
min_suppliers = st.sidebar.slider("Minimum active suppliers", 1, len(suppliers), 3)
st.sidebar.markdown("**Disruption (hits the committed plan)**")
_disruption_choices = ["(none)"] + list(suppliers)
disrupted = st.sidebar.selectbox(
    "Supplier disrupted",
    _disruption_choices,
    index=len(_disruption_choices) - 1,  # default to the last (high-volume) supplier
    help="Defaults to a disruption so the cost-vs-resilience trade-off is visible on load.",
)
severity = st.sidebar.slider(
    "Remaining capacity of disrupted supplier", 0.0, 1.0, 0.0, 0.05,
    help="0 = supplier fully down.",
)
disruption = {disrupted: severity} if disrupted != "(none)" else {}

# ------------------------------------------------------------- DEA scores view
st.subheader("2 · DEA efficiency (input-oriented CCR)")
eff_df = pd.DataFrame(
    {"supplier": list(efficiency), "DEA efficiency": list(efficiency.values())}
)
c1, c2 = st.columns([1, 2])
c1.dataframe(eff_df, hide_index=True, use_container_width=True)
c2.bar_chart(eff_df.set_index("supplier"))

# ---------------------------------------------------- two competing allocations
cost_plan = allocate(suppliers, demand, efficiency, efficiency_weight=0.0)
res_plan = allocate(
    suppliers, demand, efficiency,
    efficiency_weight=eff_weight, max_share=max_share, min_suppliers=min_suppliers,
)

st.subheader("3 · Two allocation strategies")
plan_df = pd.DataFrame(
    {
        "supplier": list(suppliers),
        "Cost-only plan": [cost_plan["allocation"][j] for j in suppliers],
        "Resilient plan": [res_plan["allocation"][j] for j in suppliers],
    }
)
c3, c4 = st.columns([2, 1])
c3.bar_chart(plan_df.set_index("supplier"))
with c4:
    st.metric("Cost-only purchasing cost", f"${cost_plan['purchasing_cost']:,.0f}")
    st.metric(
        "Resilient purchasing cost",
        f"${res_plan['purchasing_cost']:,.0f}",
        delta=f"{res_plan['purchasing_cost'] - cost_plan['purchasing_cost']:,.0f}",
        delta_color="inverse",
    )
    st.caption(
        f"Cost-only uses {len(cost_plan['active_suppliers'])} supplier(s); "
        f"resilient uses {len(res_plan['active_suppliers'])}."
    )

# -------------------------------------------------------------- stress testing
st.subheader("4 · Disruption stress test")
if not disruption:
    st.info("Select a disrupted supplier in the sidebar to stress-test both plans.")
else:
    cost_hit = stress_test(suppliers, demand, cost_plan["allocation"], disruption)
    res_hit = stress_test(suppliers, demand, res_plan["allocation"], disruption)
    label = f"{disrupted} at {severity:.0%} capacity"
    st.caption(f"After orders are committed, **{label}**. Pre-committed units above surviving capacity are lost.")
    s1, s2 = st.columns(2)
    s1.metric(
        "Cost-only realised service",
        f"{cost_hit['realized_service_level']:.0%}",
        delta=f"-{cost_hit['lost_units']:.0f} units lost",
        delta_color="inverse",
    )
    s2.metric(
        "Resilient realised service",
        f"{res_hit['realized_service_level']:.0%}",
        delta=f"-{res_hit['lost_units']:.0f} units lost",
        delta_color="inverse",
    )
    premium = res_plan["purchasing_cost"] - cost_plan["purchasing_cost"]
    gain = res_hit["realized_service_level"] - cost_hit["realized_service_level"]
    st.success(
        f"The resilient plan costs **${premium:,.0f} more** up front but delivers "
        f"**{gain:+.0%} higher realised service** when {label}. "
        "This cost-vs-resilience trade-off is exactly the gap the 2021 deterministic "
        "model leaves open."
    )
