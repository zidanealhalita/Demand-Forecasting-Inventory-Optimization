"""Page 3 - Forecasting: six models, rolling-origin back-test, event vs normal weeks, censoring toggle."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import data, services, ui, viz
from src import forecasting as FC

panel = data.load_panel()
f = ui.get_filters(panel)
ui.page_header(
    "Forecasting",
    "Which forecasting method should drive replenishment for this SKU-region, how wrong will it be, and does it hold up in event weeks?",
)

sku, reg = ui.series_selector(panel, f, "fc")
c1, c2, c3 = st.columns([1, 2, 2])
horizon = c1.radio("Horizon (days)", [14, 28, 56], index=1, horizontal=True)
anchor_label = c2.selectbox("Back-test window", list(services.ANCHORS), help="Rolling-origin: 4 consecutive test windows of the chosen horizon, newest last. The last one is the final holdout (for 56 days = the last 8 weeks). The Lebaran option shifts the windows so the Lebaran surge is inside the test.")
exclude = c3.toggle("Exclude stockout days when training", value=True, help="Stockout days are censored: units_sold < true demand. Excluding them (or imputing, for models that need an unbroken series) removes the downward bias.")
anchor = services.ANCHORS[anchor_label]

with st.spinner("Fitting 6 models on 4 rolling origins (a few seconds the first time, cached afterwards)..."):
    bt, lb, last, cov = services.forecast_backtest(sku, reg, horizon, anchor, exclude)
best = lb.index[0]
naive_w = lb.loc["naive", "WAPE"]

# ------------------------------------------------------------------------------------------ leaderboard
st.subheader("Leaderboard")
st.caption("Pooled over 4 rolling folds, scored on non-stockout days only (true demand is unobservable on stockout days). WAPE = sum|error| / sum(actual); Bias = sum(forecast - actual) / sum(actual); MASE < 1 beats the in-sample seasonal naive.")
show = lb[["Model", "WAPE", "MAPE", "Bias", "MASE", "WAPE (last fold)"]].copy()
st.dataframe(
    show,
    hide_index=True,
    column_config={
        "WAPE": st.column_config.NumberColumn(format="percent"),
        "MAPE": st.column_config.NumberColumn(format="percent"),
        "Bias": st.column_config.NumberColumn(format="percent"),
        "MASE": st.column_config.NumberColumn(format="%.2f"),
        "WAPE (last fold)": st.column_config.NumberColumn("WAPE (final holdout)", format="percent"),
    },
)

# ------------------------------------------------------------------------------------------ chart
st.subheader("Forecast vs actual (final holdout window)")
c1, c2 = st.columns([2, 1])
model = c1.selectbox("Model to plot", list(lb.index), index=0, format_func=lambda m: FC.MODEL_LABELS[m])
show_all = c2.checkbox("Overlay all models", value=False)
cur = last[last["model"] == model].sort_values("date")
fig = go.Figure()
fig.add_trace(go.Scatter(x=cur["date"], y=cur["hi"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=cur["date"], y=cur["lo"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(0,114,178,0.15)", name="80% prediction interval"))
if show_all:
    for m in lb.index:
        if m == model:
            continue
        g = last[last["model"] == m].sort_values("date")
        fig.add_trace(go.Scatter(x=g["date"], y=g["forecast"], mode="lines", name=FC.MODEL_LABELS[m], line=dict(width=1, dash="dot")))
fig.add_trace(go.Scatter(x=cur["date"], y=cur["forecast"], mode="lines", name=FC.MODEL_LABELS[model], line=dict(color=viz.BLUE, width=2.5)))
ok = cur[~cur["stockout"]]
fig.add_trace(go.Scatter(x=ok["date"], y=ok["actual"], mode="lines+markers", name="Actual (units sold)", line=dict(color=viz.BLACK, width=1.5), marker=dict(size=5)))
so = cur[cur["stockout"]]
if len(so):
    fig.add_trace(go.Scatter(x=so["date"], y=so["actual"], mode="markers", name="Stockout day (censored, not scored)", marker=dict(color=viz.VERMILION, size=9, symbol="x")))
ev = cur[cur["event"]]
if len(ev):
    fig.add_trace(go.Scatter(x=ev["date"], y=[cur["actual"].min() * 0.9] * len(ev), mode="markers", name="Event day", marker=dict(color=viz.ORANGE, size=6, symbol="square")))
w_last = lb.loc[model, "WAPE (last fold)"]
viz.style(fig, f"{FC.MODEL_LABELS[model]} misses the final {horizon}-day holdout by {w_last * 100:.0f}% WAPE; interval covers {cov.get(model, np.nan) * 100:.0f}% of days (target 80%)", "Date", "Units per day", height=430)
st.plotly_chart(fig)

# ------------------------------------------------------------------------------------------ event vs normal
st.subheader("Event days vs normal days")
ev_tab = lb[["Model", "WAPE event days", "WAPE normal days", "Bias event days", "Bias normal days"]].copy()
n_ev = int(bt[(bt["model"] == best) & bt["event"] & ~bt["stockout"]].shape[0])
n_no = int(bt[(bt["model"] == best) & ~bt["event"] & ~bt["stockout"]].shape[0])
fig = go.Figure()
fig.add_trace(go.Bar(x=[FC.MODEL_LABELS[m] for m in lb.index], y=lb["WAPE event days"], name=f"Event days (n={n_ev})", marker_color=viz.VERMILION))
fig.add_trace(go.Bar(x=[FC.MODEL_LABELS[m] for m in lb.index], y=lb["WAPE normal days"], name=f"Normal days (n={n_no})", marker_color=viz.BLUE))
fig.update_layout(barmode="group", bargap=0.25)
exog_models = ["sarimax", "glm", "gbm"]
ev_best = lb["WAPE event days"].idxmin()
viz.style(fig, f"On event days {FC.MODEL_LABELS[ev_best]} is best ({lb.loc[ev_best, 'WAPE event days'] * 100:.0f}% WAPE); the naive model reaches {lb.loc['naive', 'WAPE event days'] * 100:.0f}%", "Model", "WAPE (lower is better)", height=380)
fig.update_yaxes(tickformat=".0%")
st.plotly_chart(fig)
st.caption("Event day = promo or the 5 days after it, Ramadan-Lebaran window (-30 to +14 days), public holiday or year-end window.")

# ------------------------------------------------------------------------------------------ plain-words verdict
exo_best = lb.loc[[m for m in exog_models if m in lb.index], "WAPE event days"].min()
ets_ma_best = lb.loc[[m for m in ["naive", "ma28", "ets"] if m in lb.index], "WAPE event days"].min()
verdict = (
    f"**In plain words:** for {sku} in {reg}, the best overall model is **{FC.MODEL_LABELS[best]}** with a WAPE of {lb.loc[best, 'WAPE'] * 100:.1f}% "
    f"({(1 - lb.loc[best, 'WAPE'] / naive_w) * 100:.0f}% better than the naive same-weekday forecast at {naive_w * 100:.1f}%). "
    f"On event days the models that *know* the calendar (promo, payday, Lebaran, temperature) reach {exo_best * 100:.1f}% WAPE versus {ets_ma_best * 100:.1f}% for the pure time-series ones "
    f"({'a clear edge' if exo_best < ets_ma_best else 'no edge in this window'}). "
    f"Daily noise is large, so ~{lb['WAPE'].min() * 100:.0f}% is close to the floor for a single day; replenishment uses *sums over the lead time*, where errors average out."
)
st.success(verdict, icon=":material/lightbulb:")

# ------------------------------------------------------------------------------------------ censoring
st.subheader("Censored demand: effect of excluding stockout days")
with st.spinner("Comparing with and without exclusion..."):
    _, lb_other, _, _ = services.forecast_backtest(sku, reg, horizon, anchor, not exclude)
lb_ex, lb_in = (lb, lb_other) if exclude else (lb_other, lb)
cmp = pd.DataFrame(
    {
        "Model": [FC.MODEL_LABELS[m] for m in lb_ex.index],
        "Bias - stockout days kept": lb_in.loc[lb_ex.index, "Bias"].values,
        "Bias - stockout days excluded": lb_ex["Bias"].values,
        "WAPE - kept": lb_in.loc[lb_ex.index, "WAPE"].values,
        "WAPE - excluded": lb_ex["WAPE"].values,
    }
)
mean_kept, mean_ex = cmp["Bias - stockout days kept"].mean(), cmp["Bias - stockout days excluded"].mean()
st.dataframe(cmp, hide_index=True, column_config={c: st.column_config.NumberColumn(format="percent") for c in cmp.columns[1:]})
st.caption(
    f"Average bias across models: {mean_kept * 100:+.1f}% with censored days kept vs {mean_ex * 100:+.1f}% excluded. "
    "Keeping them teaches the model that demand is lower than it really was (stock ran out), which creates a negative bias and more stockouts later - a vicious circle. "
    "Excluding them (or imputing) is the standard fix; the validation appendix measures the remaining bias against the true demand."
)
with st.expander("Rolling-fold detail and method notes"):
    scored = bt[~bt["stockout"]]
    by_fold = scored.groupby(["model", "fold"]).apply(lambda g: FC.wape(g["actual"].values, g["forecast"].values), include_groups=False).unstack()
    by_fold.index = [FC.MODEL_LABELS[m] for m in by_fold.index]
    by_fold.columns = [f"Fold {c}" for c in by_fold.columns]
    st.dataframe(by_fold, column_config={c: st.column_config.NumberColumn(format="percent") for c in by_fold.columns})
    st.markdown(
        "- **Time-based split:** every fold is trained only on days before its origin; no shuffling.\n"
        "- **Future regressors:** promo plan and calendar are known in advance; temperature is replaced by the training-period climatology.\n"
        "- **Gradient boosting:** direct multi-horizon on ratio-to-level targets; lag statistics are computed at the forecast origin only.\n"
        "- **Prediction interval:** empirical 10-90% quantiles of relative errors from the *earlier* folds.\n"
        "- **MASE** is scaled by the in-sample seasonal-naive MAE (lag 7) on uncensored days."
    )
ui.footer()
