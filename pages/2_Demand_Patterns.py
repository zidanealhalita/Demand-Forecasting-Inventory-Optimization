"""Page 2 - Demand Patterns (EDA): seasonality, promotions, weather, ABC-XYZ."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import analytics as AN, data, services, ui, viz
from src.inventory import POLICY_MATRIX

panel = data.load_panel()
f = ui.get_filters(panel)
ui.page_header(
    "Demand Patterns",
    "Which predictable patterns (weekday, payday, Lebaran, promo, weather) should be built into the forecast and the buffer, and how should each SKU class be controlled?",
)
st.caption(f"Filter: {f.label()}. Patterns are measured on non-stockout days as an index vs a rolling baseline (1.00 = normal level).")

idx = ui.apply_filters(services.indexed_panel(), f)
if idx.empty:
    st.warning("No data for this filter combination.")
    ui.footer()
    st.stop()

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
tabs = st.tabs(["Seasonality", "Promotions", "Weather", "ABC-XYZ"])

# ------------------------------------------------------------------------------------------------ seasonality
with tabs[0]:
    c1, c2 = st.columns(2)
    wd = AN.weekday_profile(idx)
    wd["day"] = wd["dow"].map(dict(enumerate(DOW)))
    top = wd.loc[wd["idx28"].idxmax()]
    spread = wd.groupby("category")["idx28"].agg(lambda x: x.max() - x.min()).sort_values(ascending=False)
    fig = px.line(wd, x="day", y="idx28", color="category", color_discrete_map=viz.CATEGORY_COLORS, markers=True, category_orders={"day": DOW})
    viz.style(fig, f"Weekend lifts {spread.index[0]} the most ({spread.iloc[0] * 100:.0f} pts Mon-Sun spread); peak {top['category']} on {top['day']}", "Weekday", "Demand index (1.00 = 28-day mean)")
    c1.plotly_chart(fig)

    pdp = AN.payday_profile(idx)
    pw = idx.assign(pw=idx["is_payday_window"] == 1).groupby(["category", "pw"])["idx28"].mean().unstack()
    pw["lift"] = pw[True] / pw[False] - 1
    best = pw["lift"].idxmax()
    fig = px.line(pdp, x="day_of_month", y="idx28", color="category", color_discrete_map=viz.CATEGORY_COLORS)
    fig.add_vrect(x0=24.5, x1=31.5, fillcolor=viz.GREY, opacity=0.12, line_width=0)
    fig.add_vrect(x0=0.5, x1=2.5, fillcolor=viz.GREY, opacity=0.12, line_width=0)
    viz.style(fig, f"Payday window (day 25-2) lifts {best} demand by {pw.loc[best, 'lift'] * 100:.0f}% vs the rest of the month", "Day of month (grey = payday window)", "Demand index")
    c2.plotly_chart(fig)

    c3, c4 = st.columns(2)
    mp = AN.monthly_profile(idx)
    wm = mp[mp["category"] == "Water"]
    if len(wm):
        pk = wm.loc[wm["idx365"].idxmax()]
        hot = wm[wm["month"].isin([9, 10])]["idx365"].mean()
        mt = (f"Water runs {(hot - 1) * 100:+.0f}% vs its annual mean in the hot season (Sep-Oct); "
              f"its highest month is {int(pk['month'])} ({(pk['idx365'] - 1) * 100:+.0f}%)"
              + (", the Ramadan/Lebaran surge" if int(pk["month"]) in (3, 4, 5) else ""))
    else:
        mt = "Monthly profile by category"
    fig = px.line(mp, x="month", y="idx365", color="category", color_discrete_map=viz.CATEGORY_COLORS, markers=True)
    viz.style(fig, mt, "Calendar month", "Demand index (1.00 = 12-month centred mean)")
    fig.update_xaxes(dtick=1)
    c3.plotly_chart(fig)

    lp = AN.lebaran_profile(idx)
    if len(lp):
        pk = lp.loc[lp.groupby("category")["idx91m"].idxmax()].set_index("category")
        lead = pk["idx91m"].idxmax()
        lt = f"{lead} surges to {pk.loc[lead, 'idx91m']:.2f}x normal {abs(int(pk.loc[lead, 'days_to_lebaran']))} days before Lebaran, then dips below 1.0"
    else:
        lt = "Lebaran window"
    fig = px.line(lp, x="days_to_lebaran", y="idx91m", color="category", color_discrete_map=viz.CATEGORY_COLORS)
    fig.add_vrect(x0=-30, x1=-0.5, fillcolor=viz.GREY, opacity=0.12, line_width=0, annotation_text="Ramadan", annotation_position="top left")
    fig.add_vline(x=0, line_dash="dot", line_color=viz.BLACK)
    fig.add_hline(y=1, line_color=viz.GREY, line_width=1)
    viz.style(fig, lt, "Days relative to Lebaran (0 = Lebaran day; pooled 2023-2025)", "Demand index vs 91-day rolling median")
    c4.plotly_chart(fig)
    st.caption("Implication: build weekday, payday, Ramadan/Lebaran and promo features into the forecast (page 3) and size event buffers from the regime-specific forecast error (page 4).")

# ------------------------------------------------------------------------------------------------ promotions
with tabs[1]:
    ev = AN.promo_event_study(ui.apply_filters(panel, f))
    if ev.empty:
        st.info("No promo events in the current selection.")
    else:
        s = AN.promo_summary(ev)
        c = st.columns(4)
        c[0].metric("Uplift during promo", f"{s['uplift_during_pct'] * 100:+.0f}%", help="Mean index during the 7 promo days vs the pre-promo baseline.")
        c[1].metric("Dip, 5 days after", f"{s['dip_after_pct'] * 100:+.1f}%", help="Pull-forward: customers bought ahead of time.")
        c[2].metric("Uplift recovered as dip", f"{s['recovered_share'] * 100:.0f}%", help="Lost units in the 5 post-promo days / extra units during the promo.")
        c[3].metric("Net uplift (12 days)", f"{s['net_uplift_pct'] * 100:+.0f}%", help="Average lift over the promo week plus the 5-day dip window.")
        m = ev.groupby(["category", "rel_day"])["idx"].mean().reset_index()
        fig = px.line(m, x="rel_day", y="idx", color="category", color_discrete_map=viz.CATEGORY_COLORS, markers=True)
        fig.add_vrect(x0=-0.5, x1=6.5, fillcolor=viz.GREEN, opacity=0.08, line_width=0, annotation_text="promo (7 days)", annotation_position="top left")
        fig.add_vrect(x0=6.5, x1=11.5, fillcolor=viz.VERMILION, opacity=0.08, line_width=0, annotation_text="pull-forward dip", annotation_position="top left")
        fig.add_hline(y=1, line_color=viz.GREY, line_width=1)
        viz.style(fig, f"Promos lift demand {s['uplift_during_pct'] * 100:.0f}%, but {s['recovered_share'] * 100:.0f}% of the extra units come back as a dip - net lift only {s['net_uplift_pct'] * 100:.0f}%", "Days relative to promo start", "Demand index vs pre-promo baseline", height=420)
        st.plotly_chart(fig)
        tab = []
        for cat, g in ev.groupby("category"):
            ss = AN.promo_summary(g)
            tab.append({"Category": cat, "Events": g.groupby(["sku_id", "region"]).ngroups, "Uplift in promo": ss["uplift_during_pct"], "Dip after": ss["dip_after_pct"], "Recovered": ss["recovered_share"], "Net uplift": ss["net_uplift_pct"]})
        st.dataframe(pd.DataFrame(tab), hide_index=True, column_config={c_: st.column_config.NumberColumn(format="percent") for c_ in ["Uplift in promo", "Dip after", "Recovered", "Net uplift"]})
        st.caption("Caveat: promo days are often censored by stockouts (and excluded), so the gross uplift is, if anything, understated. Planner action: do not stock for the full promo uplift beyond the promo week - the following week is weaker.")

# ------------------------------------------------------------------------------------------------ weather
with tabs[2]:
    b, sl = AN.temperature_relation(idx)
    if b.empty:
        st.info("Not enough data in the current selection.")
    else:
        w = sl[sl["group"] == "Water SKUs"]
        o = sl[sl["group"].str.startswith("Other")]
        wt = f"Water demand moves {w['pct_per_degC'].iloc[0] * 100:+.1f}% per +1 degC" if len(w) else "Temperature relation"
        if len(o):
            wt += f" versus {o['pct_per_degC'].iloc[0] * 100:+.1f}% for non-water SKUs"
        fig = go.Figure()
        for grp, col in [("Water SKUs", viz.BLUE), ("Other SKUs (control)", viz.GREY)]:
            g = b[b["group"] == grp]
            fig.add_trace(go.Scatter(x=g["bin"], y=g["idx"], mode="lines+markers", name=grp, line=dict(color=col, width=2), marker=dict(size=8), error_y=dict(type="data", array=1.96 * g["se"], thickness=1)))
        fig.add_hline(y=1, line_color=viz.GREY, line_width=1)
        viz.style(fig, wt, "Temperature vs the region's average (degC, 0.5 degC bins)", "Demand index vs 12-month mean", height=400)
        st.plotly_chart(fig)
        st.dataframe(sl.rename(columns={"group": "Group", "pct_per_degC": "% change per +1 degC", "ci": "95% CI (+/-)"}), hide_index=True, column_config={"% change per +1 degC": st.column_config.NumberColumn(format="percent"), "95% CI (+/-)": st.column_config.NumberColumn(format="percent")})
        st.caption("Log-linear regression with weekday dummies on non-stockout days. The temperature also has an annual cycle (hottest around Sep-Oct), so the same effect shows up in the monthly profile. Planner action: include temperature (expected, from climatology) for water SKUs and pre-build before the hot season.")

# ------------------------------------------------------------------------------------------------ ABC-XYZ
with tabs[3]:
    c1, c2 = st.columns(2)
    level = c1.radio("Classify", ["SKU x Region", "SKU"], horizontal=True, help="SKU x Region = 50 stocking points (how a planner actually controls stock); SKU = 10 products.")
    gran = c2.radio("XYZ variability measured on", ["Weekly", "Daily"], horizontal=True, help="Weekly follows the standard definition; daily is more sensitive and splits the X class.")
    ax = AN.abc_xyz(ui.apply_filters(panel, f), level, gran)
    cnt = ax.groupby(["ABC", "XYZ"]).size().unstack(fill_value=0).reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0)
    fig = go.Figure(go.Heatmap(z=cnt.values, x=cnt.columns, y=cnt.index, text=cnt.values, texttemplate="%{text}", colorscale=viz.SEQ_SCALE, showscale=False, hovertemplate="%{y}%{x}: %{z} items<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    n_x = int((ax["XYZ"] == "X").sum())
    viz.style(fig, f"{n_x} of {len(ax)} items are X (CV < 0.5): demand is forecastable, so variability is not the problem - lead time and events are", "XYZ (variability: X < 0.5, Y 0.5-1.0, Z > 1.0 CV)", "ABC (value: A = top 80%, B = next 15%, C = last 5% of revenue)", height=320, legend=False)
    c1, c2 = st.columns([2, 3])
    c1.plotly_chart(fig)
    pol = pd.DataFrame(
        [{"Cell": k, "Items": int(cnt.loc[k[0], k[1]]), "Recommended control policy": v} for k, v in POLICY_MATRIX.items()]
    )
    c2.dataframe(pol[pol["Items"] > 0], hide_index=True, height=300)
    st.markdown("**Items**")
    st.dataframe(
        ax.reset_index(names="item")[["item", "class", "revenue_share", "cv"]],
        hide_index=True,
        column_config={"item": "Item", "class": "Class", "revenue_share": st.column_config.NumberColumn("Revenue share", format="percent"), "cv": st.column_config.NumberColumn("CV", format="%.2f")},
    )
    st.caption("Because this synthetic demand is smooth at weekly level, almost everything is X; switching to *daily* CV shows where daily volatility is higher. Revenue is concentrated: the A class carries 80% of revenue.")
ui.footer()
