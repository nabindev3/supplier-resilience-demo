"""Streamlit demo with two linked views: DEA-based order allocation with a
disruption stress test (after Yousefi et al. 2021), and a Fuzzy Cognitive
Map of resilience/sustainability enablers.

Run:  streamlit run app.py
"""

import numpy as np
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
    "DEA efficiency + order allocation (after Yousefi et al., 2021), extended "
    "with a disruption stress test and a causal map of resilience enablers."
)

# sidebar -----------------------------------------------------------------
st.sidebar.header("Allocation controls")
demand = st.sidebar.number_input("Total demand (units)", 100, 100000, DEFAULT_DEMAND, 100)
eff_weight = st.sidebar.slider(
    "Efficiency reward weight", 0.0, 5.0, 1.5, 0.1,
    help="0 = pure cost minimisation. Higher pulls orders toward DEA-efficient suppliers.",
)
st.sidebar.markdown("**Resilience levers**")
max_share = st.sidebar.slider(
    "Max share per supplier", 0.1, 1.0, 0.4, 0.05,
    help="Caps how much of demand any single supplier may serve.",
)
min_suppliers = st.sidebar.slider("Minimum active suppliers", 1, len(SUPPLIERS), 3)
st.sidebar.markdown("**Disruption (hits the committed plan)**")
_choices = ["(none)"] + list(SUPPLIERS)
disrupted = st.sidebar.selectbox(
    "Supplier disrupted", _choices, index=len(_choices) - 1,
    help="Defaults to a disruption so the trade-off is visible on load.",
)
severity = st.sidebar.slider(
    "Remaining capacity of disrupted supplier", 0.0, 1.0, 0.0, 0.05,
    help="0 = supplier fully down.",
)
disruption = {disrupted: severity} if disrupted != "(none)" else {}

tab_alloc, tab_fcm, tab_nash = st.tabs(
    ["Allocation & Resilience", "Causal Map (FCM)", "Nash Bargaining (Stage 2)"]
)

# tab 1: allocation -------------------------------------------------------
with tab_alloc:
    st.subheader("1. Supplier data")
    st.caption(
        "DEA inputs (minimise): price, lead time, defect %. "
        "Outputs (maximise): quality, on-time %, capacity. Edit any cell."
    )
    df = pd.DataFrame(SUPPLIERS).T.reset_index().rename(columns={"index": "supplier"})
    edited = st.data_editor(df, hide_index=True, width="stretch", key="suppliers")
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

    st.subheader("2. DEA efficiency (input-oriented CCR)")
    eff_df = pd.DataFrame(
        {"supplier": list(efficiency), "DEA efficiency": list(efficiency.values())}
    )
    c1, c2 = st.columns([1, 2])
    c1.dataframe(eff_df, hide_index=True, width="stretch")
    c2.bar_chart(eff_df.set_index("supplier"))

    cost_plan = allocate(suppliers, demand, efficiency, efficiency_weight=0.0)
    res_plan = allocate(
        suppliers, demand, efficiency,
        efficiency_weight=eff_weight, max_share=max_share, min_suppliers=min_suppliers,
    )

    st.subheader("3. Two allocation strategies")
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

    st.subheader("4. Disruption stress test")
    if not disruption:
        st.info("Select a disrupted supplier in the sidebar to stress-test both plans.")
    else:
        cost_hit = stress_test(suppliers, demand, cost_plan["allocation"], disruption)
        res_hit = stress_test(suppliers, demand, res_plan["allocation"], disruption)
        label = f"{disrupted} at {severity:.0%} capacity"
        st.caption(
            f"After orders are committed, **{label}**. Units committed above "
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
            f"The resilient plan costs **${premium:,.0f} more** up front but keeps "
            f"**{gain:+.0%} more realised service** when {label}. A deterministic "
            "model never sees this trade-off."
        )

# tab 2: FCM ---------------------------------------------------------------
with tab_fcm:
    st.subheader("Causal map of resilience & sustainability enablers")
    st.caption(
        "Signed, weighted causal links between enablers and targets. Green = "
        "reinforcing, red = inhibiting. The 'Supplier diversification' node is "
        "the same lever the allocation tab exercises."
    )

    fcm = FCM(CONCEPTS, weight_matrix())
    st.graphviz_chart(fcm.to_dot(), width="stretch")

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
        st.caption("Downstream effect vs baseline.")
        delta_df = pd.DataFrame(
            [(c, b, s, d) for c, (b, s, d) in result.items()],
            columns=["concept", "baseline", "scenario", "delta"],
        )
        st.dataframe(delta_df, hide_index=True, width="stretch")

    st.caption(
        "Weights are expert-defined here; in the original papers a hybrid "
        "learning algorithm tunes them. fcm.nhl_step() implements the Hebbian "
        "half of that."
    )


# tab 3: Nash bargaining ---------------------------------------------------
@st.cache_resource(show_spinner="Running Stage 1 (Prophet + MILP) once…")
def _bargaining_setup():
    """Frame the Stage-2 game once and cache it.

    The Prophet fit and MILP allocation behind this are expensive, so they run
    a single time; re-solving the Nash game at different bargaining powers
    afterwards is just an SLSQP call.
    """
    from stage2 import GameTheoryPricingEngine

    gt = GameTheoryPricingEngine(power="volume")
    setup = gt.setup()
    plan = setup["plan"]
    return {
        "plan": plan,
        "budget": setup["budget"],
        "baseline_apc": setup["baseline_apc"],
        "surplus": round(setup["budget"] - setup["cost_at_floors"], 2),
        "demand": setup["demand"]["D"],
        "volume_share": dict(zip(plan["supplier"], plan["share_of_D"])),
    }


def _nash_weights(plan, power, buyer_weight):
    """Bargaining powers with the supplier side normalised to sum to 1, so the
    buyer slider reads directly as 'buyer power vs the whole supplier side'."""
    if power == "Symmetric":
        n = len(plan)
        sup = {k: 1.0 / n for k in plan["supplier"]}
    else:  # volume-weighted
        total = float(plan["share_of_D"].sum())
        sup = {k: float(s) / total for k, s in zip(plan["supplier"], plan["share_of_D"])}
    return {"buyer": buyer_weight, **sup}


with tab_nash:
    st.subheader("Stage 2: the Nash bargaining game over price")
    st.caption(
        "Stage 1 fixed who supplies what; Stage 2 negotiates the unit price "
        "with each selected supplier. Constrained SLSQP maximises the weighted "
        "Nash product subject to the buyer's budget and every supplier's "
        "walk-away floor. This tab is the live solver, not a mock-up."
    )

    try:
        bs = _bargaining_setup()
    except Exception as exc:  # pragma: no cover - depends on Prophet at runtime
        st.error(
            "Couldn't run the Stage-1 → Stage-2 pipeline in this environment "
            f"({type(exc).__name__}: {exc}). It needs Prophet and the stage1/"
            "stage2 modules."
        )
    else:
        from stage2 import solve_nash, total_savings, negotiation_dashboard

        plan, budget, surplus = bs["plan"], bs["budget"], bs["surplus"]
        st.caption(
            f"D = {bs['demand']:,.0f} units · {len(plan)} suppliers at the table · "
            f"budget B = \\${budget:,.0f} · total surplus on the table = "
            f"\\${surplus:,.0f}."
        )

        ca, cb = st.columns([1, 2])
        power = ca.radio(
            "Bargaining power", ["Volume-weighted", "Symmetric"],
            help="Volume-weighted gives each supplier power equal to its share "
                 "of demand (losing S01 is the buyer's real threat). Symmetric "
                 "splits the supplier side evenly.",
        )
        buyer_weight = cb.slider(
            "Buyer power (w_B, relative to the whole supplier side)",
            0.1, 5.0, 1.0, 0.1,
            help="1.0 means buyer and supplier side are evenly matched, so the "
                 "buyer keeps half the surplus. Higher = the buyer extracts more.",
        )

        weights = _nash_weights(plan, power, buyer_weight)
        sup_weights = {k: v for k, v in weights.items() if k != "buyer"}
        w_sup_total = sum(sup_weights.values())  # normalised to 1

        # The utilities are all linear in price, so U_B + sum(U_k) is a constant
        # (the surplus). The whole N-dimensional price search collapses to one
        # scalar: how to split that fixed pie. Sweep the buyer's share t and the
        # weighted log-Nash objective traces a single concave hill — exactly the
        # point SLSQP climbs to.
        t = np.linspace(1e-3, 1 - 1e-3, 400)
        obj = buyer_weight * np.log(t * surplus)
        for wk in sup_weights.values():
            obj = obj + wk * np.log((wk / w_sup_total) * (1.0 - t) * surplus)
        t_star = buyer_weight / (buyer_weight + w_sup_total)

        # the live solve, for the prices/savings the curve is the landscape of
        sol = solve_nash(plan, budget, weights)
        sav = total_savings(plan, sol["prices"], bs["baseline_apc"])

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.plot(t * 100, obj, color="#1f77b4", linewidth=2)
        ax.axvline(t_star * 100, color="#d62728", linestyle="--", linewidth=1.3)
        ax.plot([t_star * 100], [obj.max()], "o", color="#d62728", markersize=9)
        ax.annotate(
            f"optimum: buyer keeps {t_star:.0%}\n(U_B = \\${t_star * surplus:,.0f})",
            (t_star * 100, obj.max()), textcoords="offset points",
            xytext=(12, -28), fontsize=9, color="#d62728",
        )
        ax.set_xlabel("buyer's share of the surplus  (%)")
        ax.set_ylabel("weighted Nash objective\n$w_B\\,\\log U_B + \\sum w_k\\,\\log U_k$")
        ax.set_title("The pie-split SLSQP is really solving (single concave optimum)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Concave with one peak — because $-(w_B\\log U_B + \\sum w_k\\log U_k)$ "
            "is convex (a sum of negative-logs of linear functions), so SLSQP "
            "can't land in a local optimum. The log form also tames the raw "
            "Nash product, which is ~$10^{21}$ at this scale and would swamp the "
            "solver's finite-difference gradients."
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Buyer keeps (U_B)", f"${sol['buyer_utility']:,.0f}",
                  help="Headroom under budget B after the negotiated deal.")
        m2.metric("List-price bill → negotiated",
                  f"${sav['negotiated_cost']:,.0f}",
                  delta=f"-${sav['savings']:,.0f}", delta_color="inverse")
        m3.metric("Buyer's savings", f"{sav['savings_pct']:.1%}",
                  help="Versus the Stage-1 list-price cost — the buyer's prize "
                       "for running the game.")

        st.markdown("#### Before/after: what each supplier conceded")
        dash = negotiation_dashboard(plan, sol["prices"])
        st.dataframe(dash, hide_index=True, width="stretch")
        st.caption(
            f"Solver: SLSQP, converged={sol['converged']} in {sol['iterations']} "
            "iterations. Each negotiated price sits between the supplier's "
            "walk-away floor and its list asking price; no supplier is pushed "
            "past giving up its floor share of profit."
        )
