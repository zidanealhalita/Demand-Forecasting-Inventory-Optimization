"""Page 5 - Supply Risk & Lead-Time analytics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import data, ui, viz
from src import supply as SUP

panel = data.load_panel()
po = data.load_purchase_orders()
sku_m = data.load_sku_master()
f = ui.get_filters(panel)
ui.page_header(
    "Supply Risk & Lead-Time Analytics",
    "Where is supply unreliable enough to justify a bigger buffer, a second source or a different review period - and what does that unreliability cost?",
)
st.caption(f"Filter: {f.label()} (date filter applies to PO order date).")

comp = SUP.completed_pos(po.merge(sku_m[["sku_id", "category"]], on="sku_id"))
cf = ui.apply_filters(comp, f, date_col="order_date")
if cf.empty:
    st.warning("No purchase orders for this filter combination.")
    ui.footer()
    st.stop()

# ------------------------------------------------------------------------------------------ KPIs
c = st.columns(5)
c[0].metric("Closed POs", f"{len(cf):,}")
c[1].metric("% late (any delay)", viz.pct(cf["is_late"].mean()))
c[2].metric("% disruptions (> planned + 3 d)", viz.pct(cf["is_disruption"].mean()))
c[3].metric("Mean delay vs plan", f"{cf['delay_days'].mean():+.2f} d")
c[4].metric("P90 delay vs plan", f"{np.percentile(cf['delay_days'], 90):+.0f} d")

# ------------------------------------------------------------------------------------------ boxplots
tab_r, tab_s, tab_d, tab_c, tab_a = st.tabs(["By region", "By SKU", "Disruptions", "Cost of unreliability", "Recommended actions"])
regs = SUP.reliability_table(cf, ["region"])
with tab_r:
    fig = px.box(cf, x="region", y="actual_lead_time_days", color_discrete_sequence=[viz.BLUE], points=False)
    ptab = cf.groupby("region")["planned_lead_time_days"].mean()
    fig.add_trace(go.Scatter(x=ptab.index, y=ptab.values, mode="markers", name="Mean planned", marker=dict(color=viz.VERMILION, size=11, symbol="diamond")))
    srt = regs.sort_values("sigma_L", ascending=False)
    viz.style(fig, f"{srt.iloc[0]['region']} lead times vary most (std {srt.iloc[0]['sigma_L']:.1f} d vs {srt.iloc[-1]['sigma_L']:.1f} d in {srt.iloc[-1]['region']}) and run {srt.iloc[0]['actual_lt']:.1f} d on average", "Region", "Actual lead time (days)", height=400)
    st.plotly_chart(fig)
    st.dataframe(regs, hide_index=True, column_config={"region": "Region", "pos": "POs", "pct_late": st.column_config.NumberColumn("% late", format="percent"), "pct_disrupted": st.column_config.NumberColumn("% disrupted", format="percent"), "mean_delay": st.column_config.NumberColumn("Mean delay (d)", format="%.2f"), "p90_delay": st.column_config.NumberColumn("P90 delay (d)", format="%.0f"), "planned_lt": st.column_config.NumberColumn("Planned LT", format="%.1f"), "actual_lt": st.column_config.NumberColumn("Actual LT", format="%.1f"), "sigma_L": st.column_config.NumberColumn("Std of LT (d)", format="%.2f")})
with tab_s:
    cf2 = cf.assign(delay=cf["delay_days"])
    fig = px.box(cf2, x="sku_id", y="delay", color="category", color_discrete_map=viz.CATEGORY_COLORS, points=False)
    fig.add_hline(y=0, line_color=viz.GREY)
    fig.add_hline(y=SUP.DISRUPTION_MARGIN_DAYS, line_dash="dash", line_color=viz.VERMILION, annotation_text="disruption threshold (+3 d)", annotation_position="top left")
    viz.style(fig, "Delay vs plan is centred on zero for every SKU; the risk is in the long right tail (disruptions)", "SKU", "Actual - planned lead time (days)", height=420)
    st.plotly_chart(fig)
    st.dataframe(SUP.reliability_table(cf, ["sku_id"]), hide_index=True, column_config={"pct_late": st.column_config.NumberColumn("% late", format="percent"), "pct_disrupted": st.column_config.NumberColumn("% disrupted", format="percent"), "mean_delay": st.column_config.NumberColumn(format="%.2f"), "p90_delay": st.column_config.NumberColumn(format="%.0f"), "planned_lt": st.column_config.NumberColumn(format="%.1f"), "actual_lt": st.column_config.NumberColumn(format="%.1f"), "sigma_L": st.column_config.NumberColumn(format="%.2f")})

# ------------------------------------------------------------------------------------------ disruptions
with tab_d:
    tr = SUP.disruption_trend(cf, "region")
    allm = SUP.disruption_trend(cf)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=allm["order_month"], y=allm["pct_disrupted"] * 100, name="All regions", marker_color=viz.SKY))
    sm = tr[tr["region"] == "Sumatera"]
    if len(sm):
        fig.add_trace(go.Scatter(x=sm["order_month"], y=sm["pct_disrupted"] * 100, name="Sumatera", mode="lines+markers", line=dict(color=viz.VERMILION, width=2)))
    overall = cf["is_disruption"].mean()
    viz.style(fig, f"{overall * 100:.1f}% of POs are disrupted (> planned + {SUP.DISRUPTION_MARGIN_DAYS} d); no trend - a steady background risk, not a deteriorating one" if allm["pct_disrupted"].std() < 0.03 else f"Disruption share ranges {allm['pct_disrupted'].min() * 100:.0f}-{allm['pct_disrupted'].max() * 100:.0f}% by month", "Order month", "% of POs disrupted", height=400)
    st.plotly_chart(fig)
    d2 = cf[cf["is_disruption"]]
    st.caption(f"Disrupted POs: {len(d2):,} of {len(cf):,}; average extra delay {d2['delay_days'].mean():.1f} d versus plan. The jump is mostly random, so it is better absorbed with buffer than predicted.")
    st.dataframe(d2.groupby("region").agg(disrupted_POs=("po_id", "size"), mean_delay=("delay_days", "mean"), max_delay=("delay_days", "max")).reset_index(), hide_index=True)

# ------------------------------------------------------------------------------------------ cost of unreliability
with tab_c:
    sl = st.slider("Service level for the buffer calculation", 0.90, 0.995, 0.95, 0.005, format="%.3f", key="sup_sl")
    pf = ui.apply_filters(panel, f, dates=False)
    u = SUP.unreliability_cost(pf, comp[comp["order_date"] <= f.date_to], sl)
    rc = SUP.region_ss_comparison(u)
    sum_row = rc[rc["region"] == "Sumatera"]
    prem = f"Sumatera needs {sum_row['premium_vs_ref'].iloc[0] * 100:+.0f}% more safety stock (in days of demand) than Jabodetabek" if len(sum_row) and (u["region"] == "Jabodetabek").any() else "Safety stock by region"
    fig = go.Figure()
    fig.add_trace(go.Bar(x=rc["region"], y=(rc["ss_idr"] - rc["extra_ss_idr"]) / 1e6, name="Buffer for demand noise", marker_color=viz.BLUE))
    fig.add_trace(go.Bar(x=rc["region"], y=rc["extra_ss_idr"] / 1e6, name="Extra buffer for lead-time variability", marker_color=viz.VERMILION))
    fig.update_layout(barmode="stack")
    viz.style(fig, f"{prem}; lead-time variability is {rc['lt_share_of_ss'].mean() * 100:.0f}% of the buffer on average", "Region", "Safety stock value (IDR m)", height=400)
    st.plotly_chart(fig)
    show = rc.rename(columns={"region": "Region", "ss_days": "Safety stock (days of demand)", "extra_ss_idr": "Extra SS due to LT variability (IDR)", "ss_idr": "Total SS (IDR)", "lt_share_of_ss": "LT share of SS", "premium_vs_ref": "Premium vs Jabodetabek"})
    st.dataframe(show, hide_index=True, column_config={"Safety stock (days of demand)": st.column_config.NumberColumn(format="%.2f"), "Extra SS due to LT variability (IDR)": st.column_config.NumberColumn(format="%,.0f"), "Total SS (IDR)": st.column_config.NumberColumn(format="%,.0f"), "LT share of SS": st.column_config.NumberColumn(format="percent"), "Premium vs Jabodetabek": st.column_config.NumberColumn(format="percent")})
    st.caption("SS = z sqrt((L+R) sigma_d^2 + d^2 sigma_L^2) with observed L and sigma_L, compared with the same formula at sigma_L = 0. Why lead-time variability matters more than demand variability here: demand noise averages out over the protection period, but a late PO delays *all* units at once - the d^2 sigma_L^2 term grows with volume while the demand term grows only with its square root.")

# ------------------------------------------------------------------------------------------ actions
with tab_a:
    rel = SUP.reliability_table(cf, ["sku_id", "region"])
    revenue = ui.apply_filters(panel, f, dates=False).groupby(["sku_id", "region"])["revenue_idr"].sum()
    act = SUP.action_table(rel, u, sku_m, revenue)
    view = act.sort_values("extra_holding_idr_year", ascending=False)
    st.caption("Rules: **Dual-source** if an A-class item (top 80% of revenue) has >= 5% disrupted POs or P90 delay >= 5 days; **Increase buffer** if mean delay > 0.5 day (planned lead time understated); **Lengthen review period** if the EOQ cycle is > 2x the review period; else **Monitor**. Impact = annual holding cost of the extra buffer paid for lead-time variability (an upper bound of what better supply could save).")
    cnt = view["action"].value_counts()
    st.dataframe(
        view[["sku_id", "region", "pct_disrupted", "p90_delay", "mean_delay", "sigma_L", "extra_ss_idr", "extra_holding_idr_year", "eoq_cycle_days", "review_days", "action"]],
        hide_index=True,
        column_config={"sku_id": "SKU", "region": "Region", "pct_disrupted": st.column_config.NumberColumn("% disrupted", format="percent"), "p90_delay": st.column_config.NumberColumn("P90 delay", format="%.0f"), "mean_delay": st.column_config.NumberColumn("Mean delay", format="%.2f"), "sigma_L": st.column_config.NumberColumn("LT std", format="%.2f"), "extra_ss_idr": st.column_config.NumberColumn("Extra SS (IDR)", format="%,.0f"), "extra_holding_idr_year": st.column_config.NumberColumn("Holding cost of extra SS (IDR/yr)", format="%,.0f"), "eoq_cycle_days": st.column_config.NumberColumn("EOQ cycle (d)", format="%.0f"), "review_days": "Review (d)", "action": "Action"},
    )
    st.caption("Action counts: " + "; ".join(f"{k}: {v}" for k, v in cnt.items()))
ui.footer()
