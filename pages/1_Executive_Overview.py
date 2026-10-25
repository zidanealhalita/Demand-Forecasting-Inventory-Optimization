"""Page 1 - Executive Overview: KPIs, trend, stockout heatmap, problem list, auto-generated insights."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import data, ui, viz

panel = data.load_panel()
po = data.load_purchase_orders()
f = ui.get_filters(panel)
d = ui.apply_filters(panel, f)

ui.page_header(
    "Executive Overview",
    "Which SKU-regions are losing sales to stockouts or tying up cash in stock, and where should the planner look first?",
)
st.page_link("pages/6_Recommendations.py", label="Short on time? Jump to the ranked action list", icon=":material/task_alt:")
st.caption(f"Filter: {f.label()}")

if d.empty:
    st.warning("No data for this filter combination.")
    ui.footer()
    st.stop()

# ----------------------------------------------------------------------------- KPIs
n_days = d["date"].nunique()
units = d["units_sold"].sum()
revenue = d["revenue_idr"].sum()
so_rate = d["stockout_flag"].mean()
fill_proxy = 1 - so_rate
inv_days = d["on_hand_end"].sum() / units if units else np.nan
p = ui.apply_filters(po, f, dates=False)
open_pos = int(((p["order_date"] <= f.date_to) & (p["receipt_date"].isna() | (p["receipt_date"] > f.date_to))).sum())

c = st.columns(6)
c[0].metric("Total units sold", f"{units / 1e6:,.2f} m")
c[1].metric("Revenue", viz.idr_m(revenue))
c[2].metric("Fill rate (proxy)", viz.pct(fill_proxy), help="1 - share of SKU-region-days with a stockout. A proxy: the true fill rate needs unconstrained demand and is shown only in the validation appendix.")
c[3].metric("Avg inventory days", f"{inv_days:,.1f} d", help="Sum of end-of-day stock / units sold = days of cover.")
c[4].metric("Stockout days", viz.pct(so_rate), help="Share of SKU-region-days with stockout_flag = 1.")
c[5].metric("Open POs", f"{open_pos:,}", help="Purchase orders placed but not received at the end of the selected range.")

# ----------------------------------------------------------------------------- trend
m = d.assign(month=d["date"].dt.to_period("M").dt.to_timestamp()).groupby(["month", "category"])["units_sold"].sum().reset_index()
yr = d.groupby("year")["units_sold"].sum()
if len(yr) >= 2 and yr.iloc[-2] > 0:
    # compare the last two full calendar years present in the filtered range
    full = [y for y in yr.index if d[d["year"] == y]["date"].nunique() >= 360]
    if len(full) >= 2:
        g = yr[full[-1]] / yr[full[-2]] - 1
        peak = m.groupby("month")["units_sold"].sum().idxmax()
        t_title = f"Volume {'grew' if g >= 0 else 'fell'} {abs(g) * 100:.0f}% in {full[-1]} vs {full[-2]}; busiest month: {peak:%b %Y}"
    else:
        t_title = "Monthly volume by category (units)"
else:
    t_title = "Monthly volume by category (units)"
fig = px.line(m, x="month", y="units_sold", color="category", color_discrete_map=viz.CATEGORY_COLORS, markers=False)
fig.update_traces(line=dict(width=2))
viz.style(fig, t_title, "Month", "Units sold per month")
c1, c2 = st.columns([3, 2])
c1.plotly_chart(fig)

# ----------------------------------------------------------------------------- heatmap
hm = d.pivot_table(index="sku_id", columns="region", values="stockout_flag", aggfunc="mean")
worst = d.groupby(["sku_id", "region"])["stockout_flag"].mean().sort_values(ascending=False)
fig = go.Figure(
    go.Heatmap(
        z=hm.values * 100, x=hm.columns, y=hm.index, colorscale=viz.RISK_SCALE,
        text=np.round(hm.values * 100, 1), texttemplate="%{text}", hovertemplate="%{y} | %{x}: %{z:.1f}% stockout days<extra></extra>",
        colorbar=dict(title="% days"),
    )
)
reg_worst = d.groupby("region")["stockout_flag"].mean().sort_values(ascending=False)
hm_title = f"{reg_worst.index[0]} has the most stockouts ({reg_worst.iloc[0] * 100:.1f}% of days); worst cell {worst.index[0][0]} / {worst.index[0][1]} at {worst.iloc[0] * 100:.1f}%"
viz.style(fig, hm_title, "Region", "SKU", height=380, legend=False)
fig.update_yaxes(autorange="reversed")
c2.plotly_chart(fig)

# ----------------------------------------------------------------------------- problem table
g = d.groupby(["sku_id", "region"]).agg(
    stockout_days=("stockout_flag", "sum"),
    stockout_rate=("stockout_flag", "mean"),
    on_hand=("on_hand_end", "sum"),
    units=("units_sold", "sum"),
    rev=("revenue_idr", "sum"),
    n=("date", "size"),
).reset_index()
ok = d[d["stockout_flag"] == 0].groupby(["sku_id", "region"])["revenue_idr"].mean().rename("rev_day_ok")
g = g.merge(ok, on=["sku_id", "region"], how="left")
g["revenue_at_risk_idr"] = g["stockout_days"] * g["rev_day_ok"]
g["inventory_days"] = g["on_hand"] / g["units"]
top5 = g.sort_values("revenue_at_risk_idr", ascending=False).head(5)
st.subheader("Top-5 problem SKU-regions")
st.caption("Ranked by revenue at risk (proxy) = stockout days x average daily revenue on normal days. A proxy for lost sales, not the true loss.")
st.dataframe(
    top5.assign(
        stockout_rate=top5["stockout_rate"] * 100,
        revenue_at_risk_idr=top5["revenue_at_risk_idr"] / 1e6,
    )[["sku_id", "region", "stockout_days", "stockout_rate", "inventory_days", "revenue_at_risk_idr"]],
    hide_index=True,
    column_config={
        "sku_id": "SKU", "region": "Region",
        "stockout_days": st.column_config.NumberColumn("Stockout days"),
        "stockout_rate": st.column_config.NumberColumn("Stockout %", format="%.1f%%"),
        "inventory_days": st.column_config.NumberColumn("Inventory days", format="%.1f"),
        "revenue_at_risk_idr": st.column_config.NumberColumn("Revenue at risk (IDR m)", format="%.0f"),
    },
)

# ----------------------------------------------------------------------------- so what
leb = d["days_to_lebaran"].between(-30, 14)
ins = []
w = g.sort_values("stockout_rate", ascending=False).iloc[0]
avg_rate = d["stockout_flag"].mean()
ins.append(
    f"**Biggest leak:** {w['sku_id']} in {w['region']} is out of stock on {w['stockout_rate'] * 100:.1f}% of days "
    f"({w['stockout_rate'] / avg_rate:.1f}x the average of {avg_rate * 100:.1f}%), roughly IDR {w['revenue_at_risk_idr'] / 1e6:,.0f} m of revenue at risk (proxy)."
)
if leb.any() and (~leb).any():
    a, b = d.loc[leb, "stockout_flag"].mean(), d.loc[~leb, "stockout_flag"].mean()
    by_cat = d[leb].groupby("category")["stockout_flag"].mean().sort_values(ascending=False)
    extra = f" {by_cat.index[0]} is hit hardest at {by_cat.iloc[0] * 100:.1f}%." if len(by_cat) else ""
    ins.append(f"**Events matter:** stockout days run at {a * 100:.1f}% inside the Ramadan-Lebaran window versus {b * 100:.1f}% outside ({a / b:.1f}x).{extra}")
else:
    mm = d.groupby("month")["stockout_flag"].mean()
    ins.append(f"**Seasonality:** stockouts peak in month {int(mm.idxmax())} ({mm.max() * 100:.1f}%) versus {mm.min() * 100:.1f}% in month {int(mm.idxmin())}.")
shelf = d.groupby("sku_id").agg(inv=("on_hand_end", "sum"), u=("units_sold", "sum"), sl=("shelf_life_days", "first"))
shelf["days"] = shelf["inv"] / shelf["u"]
shelf["share"] = shelf["days"] / shelf["sl"]
hi_s, lo_s = shelf["share"].idxmax(), shelf["days"].idxmin()
ins.append(
    f"**Cash vs risk:** {hi_s} carries {shelf.loc[hi_s, 'days']:.1f} days of cover against a {shelf.loc[hi_s, 'sl']:.0f}-day shelf life "
    f"({shelf.loc[hi_s, 'share'] * 100:.1f}% of life) while {lo_s} runs on only {shelf.loc[lo_s, 'days']:.1f} days of cover - "
    f"the filtered network averages {inv_days:.1f} days of cover, so the lever is *better-placed* buffer rather than simply more stock."
)
st.subheader("So what?")
st.success("\n\n".join(f"{i}. {t}" for i, t in enumerate(ins, 1)), icon=":material/lightbulb:")
ui.footer()
