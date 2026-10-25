"""Benchmark all forecasting models on all 50 SKU x region series (run: python scripts/forecast_benchmark.py).

Pooled WAPE (volume weighted) on non-stockout days, rolling-origin 4 folds x 28 days, stockout days excluded in
training. Two anchors: latest data, and a Lebaran-2025 stress test (last fold ends 14 Apr 2025).
Writes ``docs/forecast_benchmark.csv`` and prints a table.
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import data, features as F, forecasting as FC


def main() -> pd.DataFrame:
    panel = data.load_panel()
    cal = F.calendar_features(data.load_calendar())
    rows = []
    for (sku, reg), g in panel.groupby(["sku_id", "region"]):
        ctx = FC.SeriesContext.build(g.set_index("date").sort_index(), cal)
        for anchor_name, anchor in (("latest", None), ("lebaran_2025", "2025-04-14")):
            end = None if anchor is None else int(np.where(ctx.dates == pd.Timestamp(anchor))[0][0]) + 1
            bt = FC.rolling_origin_backtest(ctx, 28, 4, end, True)
            sc = bt[~bt["stockout"]]
            for (m, ev), d in sc.groupby(["model", "event"]):
                rows.append({"sku_id": sku, "region": reg, "anchor": anchor_name, "model": m, "event": ev,
                             "abs_err": np.abs(d.actual - d.forecast).sum(), "err": (d.forecast - d.actual).sum(), "actual": d.actual.sum()})
    r = pd.DataFrame(rows)
    out = []
    for (a, m), d in r.groupby(["anchor", "model"]):
        ev, no = d[d.event], d[~d.event]
        out.append({"anchor": a, "model": m, "WAPE": d.abs_err.sum() / d.actual.sum(), "Bias": d.err.sum() / d.actual.sum(),
                    "WAPE_event_days": ev.abs_err.sum() / ev.actual.sum(), "WAPE_normal_days": no.abs_err.sum() / no.actual.sum()})
    t = pd.DataFrame(out).sort_values(["anchor", "WAPE"])
    t.to_csv(ROOT / "docs" / "forecast_benchmark.csv", index=False)
    return t


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(main().round(4).to_string(index=False))
