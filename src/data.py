"""Data loading, caching and joins.

All CSVs live in ``data_inventory/`` and are treated as read-only. The *main* pages only ever call
:func:`load_panel` (sales + master data + calendar). The ground-truth demand file is exposed through
a separate function, :func:`load_truth`, and is only imported by the validation appendix and by the
policy back-test simulator (where the true demand is the "world" the policies are tested against).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parents[1] / "data_inventory"
KEY = ["sku_id", "region"]

CATEGORY_ORDER = ["Water", "Dairy", "Dairy-Chilled", "Nutrition", "Plant-based"]


def _csv(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, **kw)


@st.cache_data(show_spinner=False)
def load_sku_master() -> pd.DataFrame:
    """SKU master (10 rows)."""
    return _csv("sku_master.csv")


@st.cache_data(show_spinner=False)
def load_region_master() -> pd.DataFrame:
    """Region / distribution-centre master (5 rows)."""
    return _csv("region_master.csv")


@st.cache_data(show_spinner=False)
def load_calendar() -> pd.DataFrame:
    """Calendar with Indonesian seasonality flags (payday, Ramadan, Lebaran, holidays)."""
    cal = _csv("calendar.csv", parse_dates=["date"])
    return cal


@st.cache_data(show_spinner=False)
def load_purchase_orders() -> pd.DataFrame:
    """Purchase orders incl. derived delay columns. ``receipt_date`` is NaT for open POs."""
    po = _csv("purchase_orders.csv", parse_dates=["order_date", "receipt_date"])
    po["delay_days"] = po["actual_lead_time_days"] - po["planned_lead_time_days"]
    po["is_open"] = po["receipt_date"].isna()
    return po


@st.cache_data(show_spinner=False)
def load_panel() -> pd.DataFrame:
    """Daily sales panel (date x sku x region) joined with SKU, region and calendar attributes.

    Adds ``revenue_idr`` (units x price), ``margin_idr`` (units x (price - unit cost)) and the
    ``category`` / ``sku_name`` labels. Does **not** contain any ground-truth demand.
    """
    sales = _csv("sales_daily.csv", parse_dates=["date"])
    sku = load_sku_master()
    reg = load_region_master()
    cal = load_calendar()
    df = (
        sales.merge(sku, on="sku_id", how="left")
        .merge(reg[["region", "volume_scale", "lead_time_multiplier"]], on="region", how="left")
        .merge(
            cal[
                [
                    "date",
                    "dow",
                    "day_of_month",
                    "month",
                    "year",
                    "is_weekend",
                    "is_payday_window",
                    "is_ramadan",
                    "days_to_lebaran",
                    "is_public_holiday",
                    "is_year_end_window",
                ]
            ],
            on="date",
            how="left",
        )
    )
    df["revenue_idr"] = df["units_sold"] * df["price_idr"]
    df["margin_idr"] = df["units_sold"] * (df["price_idr"] - df["unit_cost_idr"])
    df = df.sort_values(KEY + ["date"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_truth() -> pd.DataFrame:
    """Ground-truth (uncensored) demand. VALIDATION / SIMULATION ONLY - not available in real life."""
    return _csv("ground_truth_demand.csv", parse_dates=["date"]).sort_values(KEY + ["date"]).reset_index(drop=True)


def series_label(sku_id: str, region: str) -> str:
    """Human-readable label for a SKU x region stocking point."""
    return f"{sku_id} | {region}"


def get_series(panel: pd.DataFrame, sku_id: str, region: str) -> pd.DataFrame:
    """Return the daily rows of one SKU x region, indexed by date."""
    s = panel[(panel["sku_id"] == sku_id) & (panel["region"] == region)]
    return s.set_index("date").sort_index()
