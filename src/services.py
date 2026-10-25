"""Cached service layer used by the Streamlit pages (heavy computations run once per server process)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from . import data, features as F, forecasting as FC, policy_backtest as PB, analytics as AN, validation as VAL

DEFAULT_BETA = 0.95  # company target service level from sku_master.csv (used as the fill-rate target)
N_BOOT = 20  # lead-time bootstrap scenarios (+1 replay scenario)


@st.cache_data(show_spinner=False)
def calendar_features() -> pd.DataFrame:
    """Calendar feature frame for all dates."""
    return F.calendar_features(data.load_calendar())


@st.cache_data(show_spinner=False)
def indexed_panel() -> pd.DataFrame:
    """Panel with demand indices (used by the EDA page)."""
    return AN.add_indices(data.load_panel())


# ---------------------------------------------------------------------- forecasting page
ANCHORS = {
    "Latest data (Oct-Dec 2025)": None,
    "Lebaran 2025 stress test (ends 14 Apr 2025)": "2025-04-14",
}


@st.cache_data(show_spinner=False)
def forecast_backtest(sku: str, region: str, horizon: int, anchor: str | None, exclude: bool, n_folds: int = 4):
    """Rolling-origin back-test of all models for one SKU x region. Returns (long frame, leaderboard, last fold + PI, coverage)."""
    panel = data.load_panel()
    s = data.get_series(panel, sku, region)
    ctx = FC.SeriesContext.build(s, calendar_features())
    end = None if anchor is None else int(np.where(ctx.dates == pd.Timestamp(anchor))[0][0]) + 1
    bt = FC.rolling_origin_backtest(ctx, horizon, n_folds, end, exclude)
    lb = FC.leaderboard(bt, ctx)
    last = FC.add_intervals(bt)
    cov = FC.interval_coverage(last)
    return bt, lb, last, cov


# ---------------------------------------------------------------------- policy back-test
@st.cache_resource(show_spinner="Building the back-test world (50 stocking points, 2025)...")
def backtest_base():
    """World, lead-time scenarios, monthly forecast tables and the *current policy* simulation."""
    panel, truth, po, cal = data.load_panel(), data.load_truth(), data.load_purchase_orders(), data.load_calendar()
    world = PB.build_world(panel, truth, po, cal)
    lt = PB.lead_time_scenarios(world, n_boot=N_BOOT)
    tables = PB.build_forecast_tables(panel, po, calendar_features(), world)
    base = PB.simulate(world, lt, "baseline")
    return {"world": world, "lt": lt, "tables": tables, "base": base}


@st.cache_resource(show_spinner="Simulating proposed policy...")
def proposed_sim(beta: float):
    """SimResult of the proposed policy at a fill-rate target ``beta`` (review period unchanged)."""
    b = backtest_base()
    return PB.simulate(b["world"], b["lt"], "proposed", b["tables"], beta=float(beta))


@st.cache_data(show_spinner=False)
def kpi_frame(policy: str, beta: float = DEFAULT_BETA) -> pd.DataFrame:
    """Per (scenario, series) KPIs for 'baseline' or 'proposed' over the 20 bootstrap lead-time scenarios.

    Scenario 0 (replay of the realised history) is excluded here; it is used only for the fidelity check
    and the stock-trajectory chart.
    """
    b = backtest_base()
    res = b["base"] if policy == "baseline" else proposed_sim(beta)
    k = PB.kpis(b["world"], res)
    return k[k["scenario"] > 0].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def frontier(betas: tuple[float, ...] = (0.90, 0.93, 0.95, 0.97, 0.98, 0.99)) -> pd.DataFrame:
    """Aggregate KPIs (mean over scenarios) of the baseline and the proposed policy over a grid of targets."""
    rows = []
    kb = PB.aggregate(kpi_frame("baseline"))
    rows.append({"policy": "Current policy", "beta": np.nan, **kb.drop(columns="scenario").mean().to_dict()})
    for be in betas:
        k = PB.aggregate(kpi_frame("proposed", be))
        rows.append({"policy": "Proposed policy", "beta": be, **k.drop(columns="scenario").mean().to_dict()})
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner="Evaluating longer review periods (this runs once)...")
def review_sims(mult: int, beta: float = DEFAULT_BETA):
    """Proposed policy with the review period multiplied by ``mult`` (world, tables, sim)."""
    b = backtest_base()
    panel, po = data.load_panel(), data.load_purchase_orders()
    w2 = PB.with_review_multiplier(b["world"], mult)
    t2 = PB.build_forecast_tables(panel, po, calendar_features(), w2)
    res = PB.simulate(w2, b["lt"][1:], "proposed", t2, beta=beta)
    return {"world": w2, "res": res}


@st.cache_data(show_spinner=False)
def review_kpis(mult: int, beta: float = DEFAULT_BETA) -> pd.DataFrame:
    """KPIs of the proposed policy under review-period multiplier ``mult`` (bootstrap scenarios only)."""
    if mult == 1:
        return kpi_frame("proposed", beta)
    r = review_sims(mult, beta)
    return PB.kpis(r["world"], r["res"])


# ---------------------------------------------------------------------- validation appendix
@st.cache_data(show_spinner="Training on censored vs true demand for all 50 series...")
def censoring_bias() -> pd.DataFrame:
    """Forecast bias/WAPE vs the true demand when models are trained on censored sales, de-censored sales or truth."""
    return VAL.censoring_bias(data.load_panel(), data.load_truth(), calendar_features())


# ---------------------------------------------------------------------- inventory optimizer inputs
@st.cache_data(show_spinner=False)
def series_params(sku: str, region: str) -> dict:
    """Planning parameters of one SKU x region from sales_daily and purchase_orders only (no ground truth)."""
    panel = data.load_panel()
    s = data.get_series(panel, sku, region)
    po = data.load_purchase_orders()
    p = po[(po["sku_id"] == sku) & (po["region"] == region) & po["actual_lead_time_days"].notna()]
    stt = AN.demand_stats(s, 365)
    recent = s.tail(90)
    d_bar = float(recent["units_sold"].where(recent["stockout_flag"] == 0).mean())
    last = s.iloc[-1]
    y = s.tail(365)
    ok = y["stockout_flag"] == 0
    pr = y.loc[ok & (y["promo_flag"] == 1), "units_sold"].mean()
    npr = y.loc[ok & (y["promo_flag"] == 0), "units_sold"].mean()
    uplift = float(np.clip(pr / npr - 1, 0.05, 1.5)) if np.isfinite(pr) and npr > 0 else 0.3
    L = p["actual_lead_time_days"].values.astype(float)
    return {
        "d_bar": d_bar,
        "sigma_d": stt["sigma_d"],
        "sigma_rel": float(stt["sigma_d"] / stt["d_bar"]),
        "dow_factor": AN.dow_factor(s),
        "lt_samples": L,
        "L_bar": float(L.mean()),
        "sigma_L": float(L.std(ddof=1)),
        "planned_L": float(p["planned_lead_time_days"].mean()),
        "R": int(last["review_period_days"]),
        "case_size": int(last["case_size"]),
        "unit_cost": float(last["unit_cost_idr"]),
        "unit_margin": float(last["price_idr"] - last["unit_cost_idr"]),
        "holding_pct": float(last["holding_cost_pct_per_year"]),
        "order_cost": float(last["ordering_cost_idr"]),
        "shelf_life": float(last["shelf_life_days"]),
        "promo_uplift": uplift,
        "start_dow": int((s.index[-1] + pd.Timedelta(days=1)).dayofweek),
    }
