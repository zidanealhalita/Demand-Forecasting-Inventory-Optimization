"""Exploratory analytics used by the Demand Patterns and Overview pages (sales_daily only, no ground truth).

Patterns are measured on *uncensored* days (stockout days are dropped) and expressed as an **index versus a
rolling baseline** so that volume differences between SKUs/regions and long-run growth do not blur them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .inventory import abc_classify, coefficient_of_variation, xyz_classify

KEY = ["sku_id", "region"]


def add_indices(panel: pd.DataFrame) -> pd.DataFrame:
    """Add demand indices vs rolling baselines (censored days excluded -> NaN).

    * ``idx28``  : units / centred 28-day mean (neutral for weekday and payday cycles)
    * ``idx91m`` : units / centred 91-day median (robust baseline for the Lebaran window)
    * ``idx365`` : units / centred 365-day mean (annual seasonality, e.g. temperature)
    """
    df = panel.sort_values(KEY + ["date"]).copy()
    df["u"] = df["units_sold"].where(df["stockout_flag"] == 0)
    g = df.groupby(KEY)["u"]
    df["idx28"] = df["u"] / g.transform(lambda x: x.rolling(28, center=True, min_periods=14).mean())
    df["idx91m"] = df["u"] / g.transform(lambda x: x.rolling(91, center=True, min_periods=45).median())
    df["idx365"] = df["u"] / g.transform(lambda x: x.rolling(365, center=True, min_periods=270).mean())
    return df


def weekday_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Mean index by category and weekday (0 = Monday)."""
    return df.groupby(["category", "dow"])["idx28"].mean().reset_index()


def payday_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Mean index by category and day of month (payday window = day 25 -> day 2)."""
    return df.groupby(["category", "day_of_month"])["idx28"].mean().reset_index()


def monthly_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Mean index vs the 365-day centred mean by category and calendar month."""
    return df.groupby(["category", "month"])["idx365"].mean().reset_index()


def lebaran_profile(df: pd.DataFrame, lo: int = -45, hi: int = 30) -> pd.DataFrame:
    """Mean index vs 91-day rolling median by category and days relative to Lebaran (all 3 years pooled)."""
    d = df[df["days_to_lebaran"].between(lo, hi)]
    return d.groupby(["category", "days_to_lebaran"])["idx91m"].mean().reset_index()


PROMO_PRE = 7
PROMO_POST = 14
PROMO_LEN = 7
DIP_DAYS = 5


def promo_event_study(panel: pd.DataFrame) -> pd.DataFrame:
    """Event study around every promo start: units / pre-promo baseline for day -7 ... +14.

    Baseline = mean units on non-promo, non-stockout days between day -28 and day -8 before the promo.
    Returns a long frame (category, rel_day, idx, n_events).
    """
    rows = []
    for (sku, reg), g in panel.sort_values("date").groupby(KEY):
        g = g.reset_index(drop=True)
        pf = g["promo_flag"].values
        u = g["units_sold"].where(g["stockout_flag"] == 0).values.astype(float)
        starts = np.where((pf[1:] == 1) & (pf[:-1] == 0))[0] + 1
        for s in starts:
            if s - 28 < 0 or s + PROMO_POST >= len(g):
                continue
            pre = u[s - 28 : s - 7]
            pre_ok = pre[(pf[s - 28 : s - 7] == 0) & ~np.isnan(pre)]
            if len(pre_ok) < 8:
                continue
            base = pre_ok.mean()
            for r in range(-PROMO_PRE, PROMO_POST + 1):
                v = u[s + r]
                if not np.isnan(v):
                    rows.append((g.loc[0, "category"], sku, reg, r, v / base))
    return pd.DataFrame(rows, columns=["category", "sku_id", "region", "rel_day", "idx"])


def promo_summary(ev: pd.DataFrame) -> dict[str, float]:
    """Gross uplift in the promo, pull-forward dip and net uplift (units-weighted via the mean index)."""
    m = ev.groupby("rel_day")["idx"].mean()
    promo_days = m.loc[0 : PROMO_LEN - 1]
    dip_days = m.loc[PROMO_LEN : PROMO_LEN + DIP_DAYS - 1]
    gross = float((promo_days - 1).sum())
    dip = float((dip_days - 1).sum())
    window = PROMO_LEN + DIP_DAYS
    return {
        "uplift_during_pct": float((promo_days - 1).mean()),
        "dip_after_pct": float((dip_days - 1).mean()),
        "recovered_share": float(-dip / gross) if gross > 0 else float("nan"),
        "net_uplift_pct": float((gross + dip) / window),
        "pre_trend": float(m.loc[-PROMO_PRE:-1].mean()),
    }


def temperature_relation(df: pd.DataFrame, bin_width: float = 0.5, min_n: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Binned relation between temperature deviation (vs the region's mean) and the annual-baseline index.

    Returns (binned means by group, regression slopes). Groups: 'Water SKUs' vs 'Other SKUs' (control).
    The slope is % change in demand per +1 degC from a log-linear OLS with weekday dummies.
    """
    d = df.dropna(subset=["idx365"]).copy()
    d["dev"] = d["avg_temp_c"] - d.groupby("region")["avg_temp_c"].transform("mean")
    d["group"] = np.where(d["category"] == "Water", "Water SKUs", "Other SKUs (control)")
    d["bin"] = (np.round(d["dev"] / bin_width) * bin_width).clip(-3, 3)
    b = d.groupby(["group", "bin"]).agg(idx=("idx365", "mean"), n=("idx365", "size"), sd=("idx365", "std")).reset_index()
    b = b[b["n"] >= min_n]
    b["se"] = b["sd"] / np.sqrt(b["n"])
    slopes = []
    for grp, g in d.groupby("group"):
        X = pd.get_dummies(g["dow"], prefix="d", drop_first=True).astype(float)
        X["dev"] = g["dev"].values
        X = sm.add_constant(X)
        res = sm.OLS(np.log(g["idx365"].clip(lower=0.05)).values, X.values).fit()
        j = list(X.columns).index("dev")
        slopes.append({"group": grp, "pct_per_degC": float(np.expm1(res.params[j])), "ci": float(1.96 * res.bse[j])})
    return b, pd.DataFrame(slopes)


def abc_xyz(panel: pd.DataFrame, level: str = "SKU x Region", granularity: str = "Weekly") -> pd.DataFrame:
    """ABC (revenue share) x XYZ (CV of weekly or daily demand) classification of items.

    ``level`` is 'SKU' (10 items) or 'SKU x Region' (50 stocking points).
    """
    d = panel.copy()
    keys = ["sku_id"] if level == "SKU" else KEY
    rev = d.groupby(keys)["revenue_idr"].sum()
    d["item"] = d[keys].astype(str).agg(" | ".join, axis=1)
    if granularity == "Weekly":
        w = d.groupby(["item", pd.Grouper(key="date", freq="W")])["units_sold"].sum().reset_index()
        cv = w.groupby("item")["units_sold"].apply(coefficient_of_variation)
    else:
        cv = d.groupby("item")["units_sold"].apply(coefficient_of_variation)
    rev.index = [" | ".join(map(str, i)) if isinstance(i, tuple) else str(i) for i in rev.index]
    out = pd.DataFrame({"revenue_idr": rev, "cv": cv}).dropna()
    out["ABC"] = abc_classify(out["revenue_idr"])
    out["XYZ"] = xyz_classify(out["cv"])
    out["class"] = out["ABC"] + out["XYZ"]
    out["revenue_share"] = out["revenue_idr"] / out["revenue_idr"].sum()
    return out.sort_values("revenue_idr", ascending=False)


def demand_stats(series: pd.DataFrame, window: int = 365) -> dict[str, float]:
    """Mean daily demand and daily noise (std of the deviation from a centred 28-day mean) on uncensored days.

    ``series`` is one SKU x region indexed by date. Uses sales only (stockout days dropped).
    """
    s = series.tail(window)
    u = s["units_sold"].where(s["stockout_flag"] == 0)
    base = s["units_sold"].where(s["stockout_flag"] == 0).rolling(28, center=True, min_periods=14).mean()
    resid = (u - base).dropna()
    d_bar = float(u.mean())
    return {"d_bar": d_bar, "sigma_d": float(resid.std()), "cv_noise": float(resid.std() / d_bar)}


def dow_factor(series: pd.DataFrame) -> np.ndarray:
    """Length-7 multiplicative weekday profile (mean 1) from uncensored days."""
    u = series["units_sold"].where(series["stockout_flag"] == 0)
    f = u.groupby(series.index.dayofweek).mean()
    f = f.reindex(range(7)).fillna(f.mean())
    return (f / f.mean()).values
