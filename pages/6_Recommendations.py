"""Page 6 - Recommendations: ranked actions with annual IDR impact (simulated), assumptions and method."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import data, services, ui, viz
from src import recommendations as RC
from src import supply as SUP

panel = data.load_panel()
f = ui.get_filters(panel)
ui.page_header(
    "Recommendations",
    "What should the planner change first, how much is it worth per year, and how sure are we?",
)
st.caption(f"Sidebar SKU / category / region filters apply to this list. Policy comparison fixed at the company fill-rate target ({services.DEFAULT_BETA:.0%}). **All IDR figures are simulated on synthetic data.**")

with st.spinner("Preparing the simulations behind the recommendations (about 15 s the first time, cached afterwards)..."):
    b = services.backtest_base()
    base_k = services.kpi_frame("baseline")
    prop_ks = {m: services.review_kpis(m) for m in (1, 2, 3)}
    leb = RC.leb_window_stats(b["world"], b["base"], services.proposed_sim(services.DEFAULT_BETA))
    shelf = data.load_sku_master().set_index("sku_id")["shelf_life_days"].to_dict()
    rd = {k: int(r) for k, r in zip(b["world"].keys, b["world"].R)}
    act, none = RC.build_actions(base_k, prop_ks, shelf, leb, rd)

cat = panel.drop_duplicates("sku_id").set_index("sku_id")["category"].to_dict()
act["category"] = act["sku_id"].map(cat)
none["category"] = none["sku_id"].map(cat)
mask = pd.Series(True, index=act.index)
mask_n = pd.Series(True, index=none.index)
if f.categories:
    mask &= act["category"].isin(f.categories); mask_n &= none["category"].isin(f.categories)
if f.skus:
    mask &= act["sku_id"].isin(f.skus); mask_n &= none["sku_id"].isin(f.skus)
if f.regions:
    mask &= act["region"].isin(f.regions); mask_n &= none["region"].isin(f.regions)
a, nn = act[mask].reset_index(drop=True), none[mask_n]
a["rank"] = np.arange(1, len(a) + 1)

if a.empty:
    st.warning("No actions for this filter combination.")
    ui.footer()
    st.stop()

fr = services.frontier()
cur = fr[fr["policy"] == "Current policy"].iloc[0]
pro = fr[(fr["policy"] == "Proposed policy") & np.isclose(fr["beta"], services.DEFAULT_BETA)].iloc[0]
pol = a[a["type"] == "Forecast-based replenishment"]
rev = a[a["type"] == "Review period"]
c = st.columns(4)
c[0].metric("Total simulated value / yr", viz.idr_m(a["impact_idr_year"].sum()), help="Sum of the robust actions listed below (no double counting: review-period gains are incremental to the replenishment policy).")
c[1].metric("Replenishment policy", viz.idr_m(pol["impact_idr_year"].sum()), help=f"{len(pol)} stocking points")
c[2].metric("Longer review period", viz.idr_m(rev["impact_idr_year"].sum()), help=f"{len(rev)} stocking points; sensitive to the ordering cost per PO")
c[3].metric("Lost units avoided / yr", f"{pol['lost_units_avoided_year'].sum():,.0f}", help="Replenishment actions only")

st.info(
    f"**Headline (network, simulated 2025):** the proposed forecast-based policy lifts fill rate from **{cur['fill_rate'] * 100:.1f}% to {pro['fill_rate'] * 100:.1f}%** "
    f"with **{(pro['avg_inventory_idr'] / cur['avg_inventory_idr'] - 1) * 100:+.0f}%** average inventory value and **{(pro['total_cost'] / cur['total_cost'] - 1) * 100:+.0f}%** total cost - "
    f"but it is *not* better everywhere ({len(nn)} stocking points in the current selection are kept on the current policy). "
    f"The single biggest structural lever is order frequency: ordering cost is {cur['ordering_cost'] / cur['total_cost'] * 100:.0f}% of today's total cost.",
    icon=":material/flag:",
)

# ---------------------------------------------------------------------------------- chart
top = a.head(15).iloc[::-1]
colors = [viz.BLUE if t == "Forecast-based replenishment" else viz.GREEN for t in top["type"]]
fig = go.Figure(go.Bar(x=top["impact_idr_year"] / 1e6, y=top["sku_id"] + " | " + top["region"], orientation="h", marker_color=colors, customdata=top["type"], hovertemplate="%{y}<br>%{x:,.1f} IDR m / yr<br>%{customdata}<extra></extra>"))
viz.style(fig, f"Top 15 actions are worth IDR {a.head(15)['impact_idr_year'].sum() / 1e6:,.0f} m / yr of the IDR {a['impact_idr_year'].sum() / 1e6:,.0f} m total (blue = replenishment policy, green = review period)", "Annual impact (IDR m, simulated)", "", height=520, legend=False)
st.plotly_chart(fig)

# ---------------------------------------------------------------------------------- action list
st.subheader("Ranked action list")
st.dataframe(
    a[["rank", "action", "type", "impact_idr_year", "robust_share", "method"]],
    hide_index=True,
    height=520,
    column_config={
        "rank": st.column_config.NumberColumn("#", width="small"),
        "action": st.column_config.TextColumn("Action", width="large"),
        "type": "Type",
        "impact_idr_year": st.column_config.NumberColumn("Impact (IDR / yr)", format="%,.0f"),
        "robust_share": st.column_config.NumberColumn("Robust in", format="percent", help="Share of the 20 lead-time scenarios in which the action reduces cost."),
        "method": st.column_config.TextColumn("Method", width="medium"),
    },
)

# ---------------------------------------------------------------------------------- honesty
st.subheader("Where the proposal does not help")
if len(nn):
    st.dataframe(
        nn[["sku_id", "region", "impact_idr_year", "robust_share", "fill_pts", "d_inv_idr_m", "reason"]],
        hide_index=True,
        column_config={"sku_id": "SKU", "region": "Region", "impact_idr_year": st.column_config.NumberColumn("Simulated impact (IDR / yr)", format="%,.0f"), "robust_share": st.column_config.NumberColumn("Better in", format="percent"), "fill_pts": st.column_config.NumberColumn("Fill rate change (pts)", format="%.2f"), "d_inv_idr_m": st.column_config.NumberColumn("Stock change (IDR m)", format="%.2f"), "reason": "Why"},
    )
    st.caption("Typically short-lead-time points where the current policy already reaches ~96-98% fill: the forecast adds little and a lower buffer saves only a little holding cost while risking sales. Keep the current policy or tune the target service level here.")
else:
    st.success("In this selection the proposal wins robustly everywhere.")

# ---------------------------------------------------------------------------------- method
with st.expander("Assumptions and method for every number", expanded=False):
    st.markdown(
        f"""
**Data.** Synthetic (generator seed fixed). SKUs have generic names; no real brands or company data.

**Simulation.** Day-by-day replay of 2025 for all 50 stocking points; both policies face the true (uncensored) daily demand and the same lead-time draws (20 bootstrap scenarios resampled from observed PO deviations). The first 28 days are a warm-up. Annualised with 365/{int(base_k['days'].iloc[0])}.

**Current policy** is reverse-engineered from 14,714 POs (order-up-to = a x MA28 x (L+R+1) + b x SD28 x sqrt(L+R+1), a = 0.974, b = 1.239); replayed with real lead times it reproduces the real 2025 stockout rate and stock level (see page 4).

**Proposed policy.** Poisson calendar forecast refitted monthly on stockout-excluded history; observed lead time and its variability; forecast-error buffer per regime (normal / Lebaran window / promo); z from a fill-rate target through the normal loss function. Same review days and case rounding.

**Cost.** Holding = stock x unit cost x holding %/365; ordering = number of POs x ordering cost per PO (sku_master); lost sales = lost units x (price - unit cost). No expiry write-offs are modelled (shelf-life is only used as a constraint for review-period changes: average cover <= {RC.SHELF_LIFE_MAX_SHARE:.0%} of shelf life).

**Review-period actions** stretch the review period by 2x or 3x (cap 28 days) and are kept only if cost falls in >= {RC.ROBUST_MIN:.0%} of scenarios, fill rate stays within {RC.FILL_TOLERANCE * 100:.1f} pt of the current policy and shelf-life is respected. They depend directly on the ordering cost per PO (IDR 250-600 k) and ignore truck-load, MOQ and supplier constraints - validate with real data before acting.

**Robustness filter.** A replenishment action is listed only if it lowers cost in >= {RC.ROBUST_MIN:.0%} of scenarios; otherwise the stocking point appears in the table above.

**Caveats.** One test year (2025) with one Lebaran; the fill-rate target was not tuned on the test year (it equals the company target); the Lebaran effect is learned from two earlier events.
"""
    )
ui.footer()
