"""
Streamlit proof-of-concept with two linked views:

  1. DEA-based supplier selection & order allocation + disruption-resilience
     analysis  (bridges Yousefi, Jahangoshai Rezaee & Solimanpur 2021,
     Operational Research 21(1), 553-588).
  2. A Fuzzy Cognitive Map (FCM) of the causal drivers of resilience &
     sustainability — the methodology Dr. Yousefi uses in his blockchain /
     sustainable-supply-chain work (FCM + hybrid learning + DEA;
     e.g. Int. J. Production Economics 246, 2022; Eng. Applications of AI, 2024).

Run:  streamlit run app.py
"""

import pandas as pd
import streamlit as st

from allocation import allocate, stress_test
from data import SUPPLIERS, DEFAULT_DEMAND, dea_arrays
from dea import ccr_input_efficiency
from fcm import FCM
from fcm_data import CONCEPTS, weight_matrix, ENABLERS

st.set_page_config(page_title="DEA Allocation + Resilience + FCM", layout="wide")

st.title("Supplier Selection, Order Allocation & Disruption Resilience")
st.caption(
    "DEA efficiency + order allocation (after Yousefi et al., 2021), extended with "
    "a disruption stress test and a Fuzzy Cognitive Map of resilience/sustainability "
    "enablers — bridging the 2021 model toward Dr. Yousefi's current research."
)

# --------------------------------------------------------------------- sidebar
st.sidebar.header("Allocation controls")
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
min_suppliers = st.sidebar.slider("Minimum active suppliers", 1, len(SUPPLIERS), 3)
st.sidebar.markdown("**Disruption (hits the committed plan)**")
_choices = ["(none)"] + list(SUPPLIERS)
disrupted = st.sidebar.selectbox(
    "Supplier disrupted", _choices, index=len(_choices) - 1,
    help="Defaults to a disruption so the cost-vs-resilience trade-off is visible on load.",
)
severity = st.sidebar.slider(
    "Remaining capacity of disrupted supplier", 0.0, 1.0, 0.0, 0.05,
    help="0 = supplier fully down.",
)
disruption = {disrupted: severity} if disrupted != "(none)" else {}

tab_alloc, tab_fcm = st.tabs(["📦 Allocation & Resilience", "🕸️ Causal Map (FCM)"])

# ======================================================= TAB 1: allocation
with tab_alloc:
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

    inp, out = dea_arrays(suppliers)
    efficiency = ccr_input_efficiency(inp, out)

    st.subheader("2 · DEA efficiency (input-oriented CCR)")
    eff_df = pd.DataFrame(
        {"supplier": list(efficiency), "DEA efficiency": list(efficiency.values())}
    )
    c1, c2 = st.columns([1, 2])
    c1.dataframe(eff_df, hide_index=True, use_container_width=True)
    c2.bar_chart(eff_df.set_index("supplier"))

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

    st.subheader("4 · Disruption stress test")
    if not disruption:
        st.info("Select a disrupted supplier in the sidebar to stress-test both plans.")
    else:
        cost_hit = stress_test(suppliers, demand, cost_plan["allocation"], disruption)
        res_hit = stress_test(suppliers, demand, res_plan["allocation"], disruption)
        label = f"{disrupted} at {severity:.0%} capacity"
        st.caption(
            f"After orders are committed, **{label}**. Pre-committed units above "
            "surviving capacity are lost."
        )
        s1, s2 = st.columns(2)
        s1.metric(
            "Cost-only realised service", f"{cost_hit['realized_service_level']:.0%}",
            delta=f"-{cost_hit['lost_units']:.0f} units lost", delta_color="inverse",
        )
        s2.metric(
            "Resilient realised service", f"{res_hit['realized_service_level']:.0%}",
            delta=f"-{res_hit['lost_units']:.0f} units lost", delta_color="inverse",
        )
        premium = res_plan["purchasing_cost"] - cost_plan["purchasing_cost"]
        gain = res_hit["realized_service_level"] - cost_hit["realized_service_level"]
        st.success(
            f"The resilient plan costs **${premium:,.0f} more** up front but delivers "
            f"**{gain:+.0%} higher realised service** when {label}. This cost-vs-"
            "resilience trade-off is exactly the gap the 2021 deterministic model leaves open."
        )

# ============================================================ TAB 2: FCM
with tab_fcm:
    st.subheader("Causal map of resilience & sustainability enablers")
    st.caption(
        "A Fuzzy Cognitive Map (FCM): signed, weighted causal links between enablers "
        "and targets. Green = reinforcing, red = inhibiting. This is the methodology "
        "Dr. Yousefi uses to model blockchain enablers → sustainability targets "
        "(FCM + hybrid learning). Note the **Supplier diversification** node — the same "
        "lever the allocation model on the other tab exercises."
    )

    fcm = FCM(CONCEPTS, weight_matrix())
    st.graphviz_chart(fcm.to_dot(), use_container_width=True)

    st.markdown("#### What-if: activate one enabler and watch the system re-settle")
    ca, cb = st.columns([2, 1])
    enabler = ca.selectbox("Enabler to switch on", ENABLERS)
    level = cb.slider("Activation level", 0.0, 1.0, 1.0, 0.05)

    result = fcm.scenario(enabler, value=level)
    _, traj = fcm.run(clamp={enabler: level})

    g1, g2 = st.columns([3, 2])
    with g1:
        st.caption("State transitions over time (each line is a concept's activation).")
        st.line_chart(pd.DataFrame(traj, columns=CONCEPTS))
    with g2:
        st.caption("Downstream effect vs. baseline (Δ).")
        delta_df = pd.DataFrame(
            [(c, b, s, d) for c, (b, s, d) in result.items()],
            columns=["concept", "baseline", "scenario", "Δ"],
        )
        st.dataframe(delta_df, hide_index=True, use_container_width=True)

    st.caption(
        "Weights here are expert-defined; in Dr. Yousefi's work a hybrid learning "
        "algorithm tunes them. `fcm.nhl_step()` implements a Nonlinear Hebbian "
        "Learning update from that family."
    )
