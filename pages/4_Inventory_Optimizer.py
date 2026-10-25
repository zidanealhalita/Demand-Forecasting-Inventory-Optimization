"""Page 4 - Inventory Policy Optimizer: SS/ROP/EOQ, cost curve, Monte-Carlo what-if, policy back-test."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import data, services, ui, viz
from src import inventory as INV
from src import policy_backtest as PB

panel = data.load_panel()
f = ui.get_filters(panel)
ui.page_header(
    "Inventory Policy Optimizer",
    "How much safety stock, what reorder point and order quantity, and which service level is worth paying for - and is the proposed policy really better than the current one?",
)
sku, reg = ui.series_selector(panel, f, "inv")
P = services.series_params(sku, reg)
service = st.slider("Target service level (cycle service level)", 0.90, 0.995, 0.95, 0.005, format="%.3f", help="Probability of not stocking out during a replenishment cycle.")
z = float(INV.z_score(service))

tabs = st.tabs(["1. SS / ROP / EOQ", "2. Cost curve", "3. What-if simulator", "4. Policy comparison (back-test)"])

# ----------------------------------------------------------------------------------------------- 1
with tabs[0]:
    d, sd, L, sL, R = P["d_bar"], P["sigma_d"], P["L_bar"], P["sigma_L"], P["R"]
    sig_lt = INV.sigma_lead_time_demand(d, sd, L, sL)
    sig_dem_only = INV.sigma_lead_time_demand(d, sd, L, 0.0)
    ss = INV.safety_stock(z, sig_lt)
    rop = INV.reorder_point(d, L, ss)
    H = P["unit_cost"] * P["holding_pct"]
    eoq_raw, eoq_r = INV.eoq(d * 365, P["order_cost"], H, P["case_size"])
    S_up = INV.order_up_to_level(d, L, R, sd, sL, z)
    c = st.columns(4)
    c[0].metric("Mean demand / day (d)", f"{d:,.0f}", help="Last 90 days, non-stockout days.")
    c[1].metric("Demand-noise std / day", f"{sd:,.0f}", help="Std of the deviation from the 28-day mean (forecast-error proxy).")
    c[2].metric("Lead time mean / std", f"{L:.1f} / {sL:.1f} d", help="Observed on closed POs (planned: %.1f d)." % P["planned_L"])
    c[3].metric("Demand in lead time (mu_L)", f"{d * L:,.0f}")
    c = st.columns(4)
    c[0].metric("sigma_LT", f"{sig_lt:,.0f}", delta=f"{(sig_lt / sig_dem_only - 1) * 100:+.0f}% from lead-time variability", delta_color="off")
    c[1].metric("Safety stock (z=%.2f)" % z, f"{ss:,.0f} units", help=f"IDR {ss * P['unit_cost'] / 1e6:,.1f} m")
    c[2].metric("Reorder point", f"{rop:,.0f} units")
    c[3].metric("EOQ (rounded to cases of %d)" % P["case_size"], f"{eoq_r:,} units", help=f"Raw EOQ {eoq_raw:,.0f}. One order every {eoq_r / d:.0f} days.")
    st.markdown(
        f"""
**Formulas.** `sigma_LT = sqrt(L*sigma_d^2 + d^2*sigma_L^2) = sqrt({L:.1f}*{sd:,.0f}^2 + {d:,.0f}^2*{sL:.1f}^2) = {sig_lt:,.0f}`;
`SS = z*sigma_LT = {z:.2f} x {sig_lt:,.0f} = {ss:,.0f}`; `ROP = d*L + SS = {d * L:,.0f} + {ss:,.0f} = {rop:,.0f}`;
`EOQ = sqrt(2*D*S/H)` with D = {d * 365:,.0f}/yr, S = IDR {P['order_cost']:,.0f}, H = {P['unit_cost']:,.0f} x {P['holding_pct']:.0%} = IDR {H:,.0f}.
Without lead-time variability sigma_LT would be {sig_dem_only:,.0f}: **lead-time variability multiplies the safety stock by {sig_lt / sig_dem_only:.1f}x ({(1 - sig_dem_only / sig_lt) * 100:.0f}% of it exists only because supply is unreliable)**.
The network uses a periodic review every {R} days, so the equivalent order-up-to level is `S = d(L+R) + z*sqrt((L+R)*sigma_d^2 + d^2*sigma_L^2)` = **{S_up:,.0f} units**.
"""
    )
    if eoq_r / d > 2 * R:
        st.warning(
            f"EOQ implies one order every ~{eoq_r / d:.0f} days but the policy orders every {R} days: ordering cost (IDR {P['order_cost']:,.0f}/PO) is probably the biggest cost lever here - see the review-period results on the Recommendations page.",
            icon=":material/warning:",
        )

# ----------------------------------------------------------------------------------------------- 2
with tabs[1]:
    cc = INV.cost_curve(d, sd, L, sL, P["unit_cost"], P["holding_pct"], P["order_cost"], P["unit_margin"], P["case_size"])
    opt = INV.optimal_service_level(cc)
    at = cc.iloc[(cc["service_level"] - service).abs().argmin()]
    D = d * 365
    lost = lambda r: (D / r["Q"]) * sig_lt * INV.std_normal_loss(r["z"])
    fig = go.Figure()
    for col, name, colr in [("total_cost", "Total", viz.BLACK), ("holding_cost", "Holding", viz.BLUE), ("ordering_cost", "Ordering", viz.GREEN), ("stockout_cost", "Expected stockout (lost margin)", viz.VERMILION)]:
        fig.add_trace(go.Scatter(x=cc["service_level"] * 100, y=cc[col] / 1e6, mode="lines", name=name, line=dict(color=colr, width=3 if col == "total_cost" else 1.8)))
    fig.add_trace(go.Scatter(x=[opt["service_level"] * 100], y=[opt["total_cost"] / 1e6], mode="markers+text", name="Cost-optimal", marker=dict(color=viz.ORANGE, size=14, symbol="star", line=dict(color="white", width=1.5)), text=[f"optimum {opt['service_level'] * 100:.1f}%"], textposition="top center"))
    fig.add_trace(go.Scatter(x=[at["service_level"] * 100], y=[at["total_cost"] / 1e6], mode="markers", name="Selected", marker=dict(color=viz.BLACK, size=11, symbol="diamond-open", line=dict(width=2))))
    lo_, hi_ = cc.iloc[(cc["service_level"] - 0.95).abs().argmin()], cc.iloc[(cc["service_level"] - 0.99).abs().argmin()]
    viz.style(fig, f"Going from 95% to 99% service adds IDR {(hi_['total_cost'] - lo_['total_cost']) / 1e6:,.0f} m/yr ({(hi_['total_cost'] / lo_['total_cost'] - 1) * 100:+.0f}%) for {lost(lo_) - lost(hi_):,.0f} fewer lost units/yr", "Cycle service level (%)", "Expected annual cost (IDR m)", height=430)
    st.plotly_chart(fig)
    t = pd.DataFrame(
        [
            {"Point": "Selected", "Service": at["service_level"], "Safety stock (units)": at["safety_stock"], "Annual cost (IDR m)": at["total_cost"] / 1e6, "Lost units / yr": lost(at)},
            {"Point": "Cost-optimal", "Service": opt["service_level"], "Safety stock (units)": opt["safety_stock"], "Annual cost (IDR m)": opt["total_cost"] / 1e6, "Lost units / yr": lost(opt)},
            {"Point": "95%", "Service": lo_["service_level"], "Safety stock (units)": lo_["safety_stock"], "Annual cost (IDR m)": lo_["total_cost"] / 1e6, "Lost units / yr": lost(lo_)},
            {"Point": "99%", "Service": hi_["service_level"], "Safety stock (units)": hi_["safety_stock"], "Annual cost (IDR m)": hi_["total_cost"] / 1e6, "Lost units / yr": lost(hi_)},
        ]
    )
    st.dataframe(t, hide_index=True, column_config={"Service": st.column_config.NumberColumn(format="percent"), "Safety stock (units)": st.column_config.NumberColumn(format="%,.0f"), "Annual cost (IDR m)": st.column_config.NumberColumn(format="%,.1f"), "Lost units / yr": st.column_config.NumberColumn(format="%,.0f")})
    st.caption(
        "Stockout cost per unit = unit margin (price - cost); (Q, r) policy with Q = EOQ; shortage per cycle = sigma_LT x G(z), G = normal loss function. "
        "Holding is a small part of the cost because inventory is lean, so the curve is flat near the optimum: the choice of service level is a business decision about how much margin you accept to lose, not a technical one."
    )

# ----------------------------------------------------------------------------------------------- 3
with tabs[2]:
    st.caption("The policy is *planned* at today's parameters and the chosen service level; the sliders then change the *world* (unless you tick 'planner anticipates'). 90 simulated days, vectorised Monte-Carlo.")
    c1, c2, c3, c4 = st.columns(4)
    surge = c1.slider("Demand surge (%)", -30, 100, 20, 5) / 100
    lti = c2.slider("Lead-time increase (days)", 0, 10, 2)
    promo = c3.toggle("Promo in the period", value=False, help="7-day promo mid-period, then a 5-day pull-forward dip.")
    ant = c4.toggle("Planner anticipates", value=False, help="If on, the order-up-to level is re-planned with the shock.")
    runs = st.select_slider("Simulation runs", [1000, 2000, 5000, 10000], value=2000)
    mc = INV.MCInputs(P["d_bar"], P["sigma_rel"], P["dow_factor"], P["lt_samples"], P["R"], P["case_size"], P["unit_cost"], P["unit_margin"], P["holding_pct"], P["order_cost"], P["promo_uplift"], 0.10, P["start_dow"])
    base = INV.simulate_mc(mc, service, 0.0, 0.0, False, False, runs)
    sim = INV.simulate_mc(mc, service, surge, lti, promo, ant, runs)
    c = st.columns(5)
    c[0].metric("Stockout probability", viz.pct(sim["stockout_prob"], 0), delta=f"{(sim['stockout_prob'] - base['stockout_prob']) * 100:+.0f} pts vs no shock", delta_color="inverse", help="Share of runs with at least one stockout day in the 90 days.")
    c[1].metric("Expected lost units", f"{sim['lost_units_mean']:,.0f}", delta=f"{sim['lost_units_mean'] - base['lost_units_mean']:+,.0f}", delta_color="inverse")
    c[2].metric("Average inventory", f"{sim['avg_inventory']:,.0f} u", delta=f"{sim['avg_inventory'] - base['avg_inventory']:+,.0f}", delta_color="off")
    c[3].metric("Inventory days", f"{sim['inventory_days']:.1f} d")
    c[4].metric("Expected cost (90 d)", viz.idr_m(sim["cost_mean"]), delta=viz.idr_m(sim["cost_mean"] - base["cost_mean"]), delta_color="inverse", help="Holding + ordering + lost margin.")
    lr, lb_ = sim["lost_units_runs"], base["lost_units_runs"]
    fig = go.Figure()
    bins = dict(start=0, end=float(max(lr.max(), lb_.max(), 1)), size=float(max(lr.max(), lb_.max(), 1)) / 40)
    fig.add_trace(go.Histogram(x=lb_, xbins=bins, name="No shock", marker_color=viz.GREY, opacity=0.55))
    fig.add_trace(go.Histogram(x=lr, xbins=bins, name="Scenario", marker_color=viz.BLUE, opacity=0.7))
    fig.add_vline(x=sim["lost_units_p50"], line_dash="dash", line_color=viz.BLACK, annotation_text=f"P50 {sim['lost_units_p50']:,.0f}", annotation_position="top")
    fig.add_vline(x=sim["lost_units_p90"], line_dash="dash", line_color=viz.VERMILION, annotation_text=f"P90 {sim['lost_units_p90']:,.0f}", annotation_position="top")
    fig.update_layout(barmode="overlay")
    viz.style(fig, f"In 1 run out of 10 you lose {sim['lost_units_p90']:,.0f}+ units over 90 days (P90); typical (P50) is {sim['lost_units_p50']:,.0f}", "Lost units over 90 days", "Number of simulation runs", height=400)
    st.plotly_chart(fig)
    st.caption(f"Order-up-to level used: {sim['S_level']:,.0f} units (no-shock plan {base['S_level']:,.0f}). Fill rate in the scenario: {sim['fill_rate'] * 100:.1f}%. Cost components (90 d): holding {viz.idr_m(sim['cost_holding'])}, ordering {viz.idr_m(sim['cost_ordering'])}, lost margin {viz.idr_m(sim['cost_stockout'])}.")

# ----------------------------------------------------------------------------------------------- 4
with tabs[3]:
    st.markdown(
        "**Back-test over 2025** (stocking points: all 50 SKU x region; first 28 days = warm-up, not measured). Both policies face the **same true demand** "
        "(ground truth is used *only here*, as a real company would have to estimate it) and the **same lead-time draws** (20 bootstrap scenarios from observed POs). "
        "*Current:* order up to MA28 x (L+R+1) + ~1.24 x SD28 x sqrt(L+R+1) on planned lead time - fitted on the 14,714 historical POs. "
        "*Proposed:* Poisson calendar forecast (promo, payday, Ramadan/Lebaran, temperature, trend, monthly refit) + observed lead time and its variability + event-regime forecast-error buffer, "
        "same review days and case rounding."
    )
    beta = st.slider("Proposed policy: fill-rate target", 0.90, 0.99, services.DEFAULT_BETA, 0.01, format="%.2f", help="Default 0.95 = the company target in sku_master. z is solved from the loss function so the *expected fill rate* (not just cycle service) meets the target.")
    with st.spinner("Running the back-test (about 5 s the first time)..."):
        kb = services.kpi_frame("baseline")
        kp = services.kpi_frame("proposed", round(beta, 2))
        fr = services.frontier()
    ab, ap = PB.aggregate(kb), PB.aggregate(kp)
    mb, mp = ab.drop(columns="scenario").mean(), ap.drop(columns="scenario").mean()
    ann = 365 / float(kb["days"].iloc[0])
    cmp = pd.DataFrame(
        {
            "Metric": ["Fill rate", "Stockout days (% of days)", "Avg inventory (units)", "Avg inventory (IDR m)", "Inventory days", "Holding cost (IDR m/yr)", "Ordering cost (IDR m/yr)", "Lost margin (IDR m/yr)", "Total cost (IDR m/yr)"],
            "Current policy": [mb.fill_rate * 100, mb.stockout_day_rate * 100, mb.avg_inventory_units, mb.avg_inventory_idr / 1e6, mb.inventory_days, mb.holding_cost * ann / 1e6, mb.ordering_cost * ann / 1e6, mb.lost_margin * ann / 1e6, mb.total_cost * ann / 1e6],
            "Proposed policy": [mp.fill_rate * 100, mp.stockout_day_rate * 100, mp.avg_inventory_units, mp.avg_inventory_idr / 1e6, mp.inventory_days, mp.holding_cost * ann / 1e6, mp.ordering_cost * ann / 1e6, mp.lost_margin * ann / 1e6, mp.total_cost * ann / 1e6],
        }
    )
    cmp["Change"] = cmp["Proposed policy"] / cmp["Current policy"] - 1
    cmp.loc[[0, 1], "Change"] = (cmp.loc[[0, 1], "Proposed policy"] - cmp.loc[[0, 1], "Current policy"])  # in points
    c = st.columns(4)
    c[0].metric("Fill rate", f"{mp.fill_rate * 100:.1f}%", delta=f"{(mp.fill_rate - mb.fill_rate) * 100:+.1f} pts vs {mb.fill_rate * 100:.1f}%")
    c[1].metric("Avg inventory", viz.idr_m(mp.avg_inventory_idr), delta=f"{(mp.avg_inventory_idr / mb.avg_inventory_idr - 1) * 100:+.1f}%", delta_color="inverse")
    c[2].metric("Lost units / yr", f"{mp.lost_units * ann:,.0f}", delta=f"{(mp.lost_units / mb.lost_units - 1) * 100:+.0f}%", delta_color="inverse")
    c[3].metric("Total cost / yr", viz.idr_m(mp.total_cost * ann), delta=f"{(mp.total_cost / mb.total_cost - 1) * 100:+.1f}%", delta_color="inverse")
    st.dataframe(cmp, hide_index=True, column_config={"Current policy": st.column_config.NumberColumn(format="%,.1f"), "Proposed policy": st.column_config.NumberColumn(format="%,.1f"), "Change": st.column_config.NumberColumn(help="Fill rate and stockout days: percentage points; everything else: relative change.", format="%.3f")})
    st.caption("Results are the mean over 20 lead-time scenarios. Ordering cost is identical by construction (same review days); the gain comes from fewer lost sales and better-placed stock. Both service **and** cost are reported on purpose.")

    # frontier
    pf = fr[fr["policy"] == "Proposed policy"]
    cur = fr[fr["policy"] == "Current policy"].iloc[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pf["avg_inventory_idr"] / 1e6, y=pf["fill_rate"] * 100, mode="lines+markers+text", name="Proposed policy (target 90-99%)", line=dict(color=viz.BLUE, width=2.5), marker=dict(size=9), text=[f"{b:.0%}" for b in pf["beta"]], textposition="bottom right"))
    fig.add_trace(go.Scatter(x=[cur["avg_inventory_idr"] / 1e6], y=[cur["fill_rate"] * 100], mode="markers", name="Current policy", marker=dict(color=viz.VERMILION, size=16, symbol="diamond", line=dict(color="white", width=1.5))))
    x_iso = np.interp(cur["fill_rate"], pf["fill_rate"], pf["avg_inventory_idr"]) / 1e6
    viz.style(fig, f"At the same fill rate the proposed policy needs ~{(1 - x_iso / (cur['avg_inventory_idr'] / 1e6)) * 100:.0f}% less stock; at the same stock it fills {(np.interp(cur['avg_inventory_idr'], pf['avg_inventory_idr'], pf['fill_rate']) - cur['fill_rate']) * 100:.1f} pts more", "Average inventory (IDR m)", "Fill rate (%)", height=420)
    st.plotly_chart(fig)

    # cost stack
    stack = pd.DataFrame(
        {
            "Policy": ["Current policy"] * 3 + ["Proposed policy"] * 3,
            "Component": ["Holding", "Ordering", "Lost margin"] * 2,
            "IDR m / yr": [mb.holding_cost, mb.ordering_cost, mb.lost_margin, mp.holding_cost, mp.ordering_cost, mp.lost_margin],
        }
    )
    stack["IDR m / yr"] = stack["IDR m / yr"] * ann / 1e6
    fig = px.bar(stack, x="Policy", y="IDR m / yr", color="Component", color_discrete_map={"Holding": viz.BLUE, "Ordering": viz.GREEN, "Lost margin": viz.VERMILION}, barmode="stack")
    viz.style(fig, f"Lost margin falls {(1 - mp.lost_margin / mb.lost_margin) * 100:.0f}% while holding moves {(mp.holding_cost / mb.holding_cost - 1) * 100:+.0f}%; ordering is the largest block for both", "", "Annual cost (IDR m)", height=380)
    c1, c2 = st.columns([2, 3])
    c1.plotly_chart(fig)

    # by sku
    ks = kb.groupby(["sku_id", "region"]).mean(numeric_only=True)
    kq = kp.groupby(["sku_id", "region"]).mean(numeric_only=True)
    sc = pd.DataFrame(
        {
            "Fill (current)": ks["fill_rate"] * 100,
            "Fill (proposed)": kq["fill_rate"] * 100,
            "Inventory IDR m (current)": ks["avg_inventory_idr"] / 1e6,
            "Inventory IDR m (proposed)": kq["avg_inventory_idr"] / 1e6,
            "Cost change %": (kq["total_cost"] / ks["total_cost"] - 1) * 100,
        }
    )
    share_better = float(((kp.set_index(["scenario", "sku_id", "region"])["total_cost"] < kb.set_index(["scenario", "sku_id", "region"])["total_cost"]).groupby(level=[1, 2]).mean() >= 0.8).mean())
    worse = sc[sc["Cost change %"] > 0].sort_values("Cost change %", ascending=False)
    by_sku = kb.groupby("sku_id").agg(cb=("total_cost", "sum")).join(kp.groupby("sku_id").agg(cp=("total_cost", "sum")))
    by_sku["Cost change %"] = (by_sku["cp"] / by_sku["cb"] - 1) * 100
    fig = go.Figure(go.Bar(x=by_sku.index, y=by_sku["Cost change %"], marker_color=[viz.BLUE if v < 0 else viz.VERMILION for v in by_sku["Cost change %"]], text=[f"{v:+.0f}%" for v in by_sku["Cost change %"]], textposition="outside"))
    viz.style(fig, f"{(by_sku['Cost change %'] < 0).sum()} of 10 SKUs get cheaper in total; {len(worse)} of 50 stocking points do not", "SKU", "Total cost change vs current (%)", height=380, legend=False)
    c2.plotly_chart(fig)
    st.markdown(f"**Where the proposed policy does *not* win ({len(worse)} of 50 SKU-regions cost more at this target; {share_better * 100:.0f}% of stocking points are better in at least 80% of scenarios):**")
    if len(worse):
        st.dataframe(worse.round(2), column_config={c: st.column_config.NumberColumn(format="%.1f") for c in worse.columns})
        st.caption("These are mostly points with short lead times where the current policy already reaches ~96-98% fill (e.g. gallon water, chilled yogurt): the forecast has little left to add, so extra precision buys little and a lower buffer saves only a little holding cost. Keep the current policy there or lower the target service level.")

    # trajectory
    st.subheader("Stock trajectory of the selected SKU-region (replay of 2025)")
    b = services.backtest_base()
    w = b["world"]
    n = w.keys.index((sku, reg))
    pr_sim = services.proposed_sim(round(beta, 2))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w.dates, y=b["base"].oh_end[:, 0, n], name="Current policy: end-of-day stock", line=dict(color=viz.VERMILION, width=1.6)))
    fig.add_trace(go.Scatter(x=w.dates, y=pr_sim.oh_end[:, 0, n], name="Proposed policy: end-of-day stock", line=dict(color=viz.BLUE, width=1.6)))
    fig.add_trace(go.Scatter(x=w.dates, y=w.demand[:, n], name="True daily demand", line=dict(color=viz.BLACK, width=1, dash="dot")))
    leb = (w.days_to_lebaran >= -30) & (w.days_to_lebaran <= 14)
    if leb.any():
        fig.add_vrect(x0=w.dates[leb][0], x1=w.dates[leb][-1], fillcolor=viz.GREY, opacity=0.12, line_width=0, annotation_text="Ramadan-Lebaran window", annotation_position="top left")
    lost_b, lost_p = b["base"].lost[:, 0, n].sum(), pr_sim.lost[:, 0, n].sum()
    viz.style(fig, f"{sku} / {reg}: current policy loses {lost_b:,.0f} units in the replayed 2025, proposed loses {lost_p:,.0f}", "Date (2025)", "Units", height=420)
    st.plotly_chart(fig)
    with st.expander("How faithful is the simulator? (current policy replayed vs real 2025 history)"):
        s0 = slice(PB.WARMUP_DAYS, None)
        sim_so = (b["base"].lost[s0, 0] > 0).mean()
        act_so = w.actual_so[s0].mean()
        sim_inv, act_inv = b["base"].oh_end[s0, 0].mean(), w.actual_oh[s0].mean()
        sim_fill = 1 - b["base"].lost[s0, 0].sum() / w.demand[s0].sum()
        act_fill = 1 - (w.demand[s0] - w.actual_sales[s0]).sum() / w.demand[s0].sum()
        st.dataframe(pd.DataFrame({"Metric": ["Stockout days %", "Average stock (units / point)", "Fill rate % (true demand)"], "Simulated (real lead times)": [sim_so * 100, sim_inv, sim_fill * 100], "Actual 2025": [act_so * 100, act_inv, act_fill * 100]}), hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ["Simulated (real lead times)", "Actual 2025"]})
        st.caption("The baseline formula was fitted on historical POs; replaying it with the realised lead times and true demand reproduces the real stockout rate and stock level, which is what makes the comparison credible.")
ui.footer()
