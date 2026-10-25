"""Validation appendix: uses ``ground_truth_demand.csv`` (NOT available in real life)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import forecasting as FC

MODELS = ["naive", "ma28", "glm"]
VARIANTS = {
    "A. Trained on raw (censored) sales": "censored",
    "B. Censored, stockout days excluded": "excluded",
    "C. Trained on true demand (oracle)": "truth",
}


def _ctx_true(ctx: FC.SeriesContext, y_true: np.ndarray) -> FC.SeriesContext:
    return FC.SeriesContext(
        dates=ctx.dates, y=y_true.astype(float), stockout=np.zeros(len(y_true), bool), cal=ctx.cal, promo=ctx.promo, temp=ctx.temp
    )


def censoring_bias(
    panel: pd.DataFrame, truth: pd.DataFrame, cal_feats: pd.DataFrame, horizon: int = 28, n_folds: int = 4
) -> pd.DataFrame:
    """Forecast error versus the **true** demand for three training variants (rolling-origin, last 4 x 28 days).

    Returns one row per (sku, region, model, variant) with the sums needed to aggregate WAPE and bias.
    """
    tr = truth.set_index(["sku_id", "region", "date"])["units_demand_unconstrained"]
    rows = []
    for (sku, reg), g in panel.groupby(["sku_id", "region"]):
        s = g.set_index("date").sort_index()
        ctx = FC.SeriesContext.build(s, cal_feats)
        y_true = tr.loc[(sku, reg)].reindex(s.index).values
        ctx_t = _ctx_true(ctx, y_true)
        n = len(ctx.y)
        acc: dict[tuple[str, str], list[float]] = {}
        for k in range(n_folds):
            test_end = n - (n_folds - 1 - k) * horizon
            origin = test_end - horizon
            yt = y_true[origin:test_end]
            fc = {
                "censored": FC.run_models(ctx, origin, horizon, False, MODELS),
                "excluded": FC.run_models(ctx, origin, horizon, True, MODELS),
                "truth": FC.run_models(ctx_t, origin, horizon, False, MODELS),
            }
            for v, d in fc.items():
                for m, f in d.items():
                    a = acc.setdefault((m, v), [0.0, 0.0, 0.0])
                    a[0] += np.abs(f - yt).sum()
                    a[1] += (f - yt).sum()
                    a[2] += yt.sum()
        for (m, v), (ae, e, y) in acc.items():
            rows.append({"sku_id": sku, "region": reg, "model": m, "variant": v, "abs_err": ae, "err": e, "demand": y})
    return pd.DataFrame(rows)


def summarise_bias(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Aggregate WAPE and bias (volume weighted) of :func:`censoring_bias` output, always split by variant."""
    g = df.groupby(list(by) + ["variant"])[["abs_err", "err", "demand"]].sum()
    g["WAPE"] = g["abs_err"] / g["demand"]
    g["Bias"] = g["err"] / g["demand"]
    return g.reset_index()


def fill_rate_truth_vs_proxy(panel: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """True fill rate (1 - lost/demand) vs the stockout-day proxy (1 - stockout-day rate) per SKU x region."""
    m = panel.merge(truth, on=["date", "sku_id", "region"])
    g = m.groupby(["sku_id", "region"]).agg(
        demand=("units_demand_unconstrained", "sum"),
        lost=("lost_sales_units", "sum"),
        so_rate=("stockout_flag", "mean"),
    )
    g["true_fill"] = 1 - g["lost"] / g["demand"]
    g["proxy_fill"] = 1 - g["so_rate"]
    return g.reset_index()
