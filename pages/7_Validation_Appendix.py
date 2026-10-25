"""Appendix - Validation / data-science: uses ground_truth_demand.csv (not available in real life)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import data, services, ui, viz
from src import validation as VAL

panel = data.load_panel()
truth = data.load_truth()
f = ui.get_filters(panel)
ui.page_header(
    "Appendix - Validation with ground truth",
    "How badly does stockout censoring bias a forecast, does excluding stockout days fix it, and is the fill-rate proxy on the Overview trustworthy?",
)
st.warning(
    "**This page uses `ground_truth_demand.csv` - the unconstrained demand that a real company never observes.** It exists only to validate the method; no main page depends on it (the policy back-test uses it solely as the simulated 'world').",
    icon=":material/science:",
)

# ----------------------------------------------------------------------------------- censoring bias
st.subheader("1. Forecast bias: trained on censored sales vs true demand")
raw = services.censoring_bias()
raw = ui.apply_filters(raw.merge(panel.drop_duplicates("sku_id")[["sku_id", "category"]], on="sku_id"), f, dates=False)
model = st.radio("Model", ["glm", "ma28", "naive"], horizontal=True, format_func={"glm": "Poisson calendar regression", "ma28": "Moving average 28d", "naive": "Naive seasonal"}.get)
sm = VAL.summarise_bias(raw[raw["model"] == model], ["sku_id"])
tot = VAL.summarise_bias(raw[raw["model"] == model], [])
order = list(VAL.VARIANTS.values())  # keys: censored, excluded, truth
label = {v: k for k, v in VAL.VARIANTS.items()}
colors = {order[0]: viz.VERMILION, order[1]: viz.BLUE, order[2]: viz.GREY}
fig = go.Figure()
for v in order:
    g = sm[sm["variant"] == v]
    fig.add_trace(go.Bar(x=g["sku_id"], y=g["Bias"] * 100, name=label[v], marker_color=colors[v]))
fig.update_layout(barmode="group")
bias_a = float(tot.loc[tot["variant"] == order[0], "Bias"].iloc[0])
bias_b = float(tot.loc[tot["variant"] == order[1], "Bias"].iloc[0])
bias_c = float(tot.loc[tot["variant"] == order[2], "Bias"].iloc[0])
viz.style(fig, f"Training on censored sales under-forecasts true demand by {abs(bias_a) * 100:.1f}%; excluding stockout days cuts that to {bias_b * 100:+.1f}% (oracle: {bias_c * 100:+.1f}%)", "SKU", "Bias vs TRUE demand (%) - negative = under-forecast", height=430)
st.plotly_chart(fig)
wt = tot.set_index("variant").loc[order, ["Bias", "WAPE"]]
wt.index = [label[i] for i in wt.index]
st.dataframe(wt, column_config={"Bias": st.column_config.NumberColumn(format="percent"), "WAPE": st.column_config.NumberColumn(format="percent")})
st.caption("Rolling-origin, 4 x 28-day test windows at the end of 2025, all 50 stocking points, scored against the true demand on **all** days (including stockout days). Variant B is what pages 3 and 4 do. The remaining gap to the oracle comes from days where the shortage hides the true peak (e.g. around Lebaran) and from imputation limits.")
with st.expander("By region"):
    rg = VAL.summarise_bias(raw[raw["model"] == model], ["region"])
    rgp = rg.pivot(index="region", columns="variant", values="Bias")[order].rename(columns=label)
    st.dataframe(rgp, column_config={v: st.column_config.NumberColumn(format="percent") for v in rgp.columns})

# ----------------------------------------------------------------------------------- fill rate
st.subheader("2. Fill rate: truth vs proxy")
fr = ui.apply_filters(VAL.fill_rate_truth_vs_proxy(panel, truth).merge(panel.drop_duplicates("sku_id")[["sku_id", "category"]], on="sku_id"), f, dates=False)
tot_true = 1 - fr["lost"].sum() / fr["demand"].sum()
tot_proxy = (fr["proxy_fill"] * 1).mean()
c = st.columns(3)
c[0].metric("True fill rate (units)", viz.pct(tot_true, 2))
c[1].metric("Proxy: 1 - stockout-day rate", viz.pct(tot_proxy, 2))
c[2].metric("Proxy minus truth", f"{(tot_proxy - tot_true) * 100:+.2f} pts")
lo = float(min(fr["true_fill"].min(), fr["proxy_fill"].min())) * 100 - 0.5
fig = px.scatter(fr, x="true_fill", y="proxy_fill", color="category", color_discrete_map=viz.CATEGORY_COLORS, hover_data=["sku_id", "region"])
fig.update_traces(marker=dict(size=9, line=dict(width=1, color="white")))
fig.update_xaxes(tickformat=".1%")
fig.update_yaxes(tickformat=".1%")
mn, mx = min(fr["true_fill"].min(), fr["proxy_fill"].min()), max(fr["true_fill"].max(), fr["proxy_fill"].max())
fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", name="proxy = truth", line=dict(color=viz.BLACK, dash="dot")))
corr = float(np.corrcoef(fr["true_fill"], fr["proxy_fill"])[0, 1]) if len(fr) > 2 else float("nan")
viz.style(fig, f"The stockout-day proxy understates the true fill rate by {abs(tot_proxy - tot_true) * 100:.1f} pts but ranks stocking points well (corr {corr:.2f})", "True fill rate (1 - lost units / demand)", "Proxy fill rate (1 - stockout-day rate)", height=430)
st.plotly_chart(fig)
st.caption("A stockout *day* is binary but usually only the last hours of the day are short, so the proxy is conservative on level and fine for ranking and monitoring. Report the proxy as a monitoring KPI, not as the service level.")
ui.footer()
