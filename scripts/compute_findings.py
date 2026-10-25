"""Recompute every number quoted in README.md / FINDINGS.md (run: python scripts/compute_findings.py).

Prints a JSON blob so the documents can be refreshed after any change to the data or the models.
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import analytics as AN, data, services as S, supply as SUP, validation as VAL
from src import policy_backtest as PB, recommendations as RC


def main() -> dict:
    out: dict = {}
    panel, po, truth = data.load_panel(), data.load_purchase_orders(), data.load_truth()
    b = S.backtest_base()
    w = b["world"]
    kb = S.kpi_frame("baseline")
    ab = PB.aggregate(kb).drop(columns="scenario").mean()
    ann = 365 / float(kb["days"].iloc[0])
    out["current"] = {"fill_rate": ab.fill_rate, "inventory_days": ab.inventory_days, "stockout_day_rate": ab.stockout_day_rate,
                      "lost_units_year": ab.lost_units * ann, "avg_inventory_idr_m": ab.avg_inventory_idr / 1e6,
                      "ordering_share_of_cost": ab.ordering_cost / ab.total_cost, "total_cost_idr_m_year": ab.total_cost * ann / 1e6}
    # Lebaran concentration (replay scenario 0 = what happened) for dairy UHT
    d = w.days_to_lebaran
    win = (d >= -30) & (d <= 14)
    mlk = [i for i, k in enumerate(w.keys) if k[0].startswith("MLK")]
    lost = b["base"].lost[:, 0, :]
    out["lebaran_dairy"] = {"lost_in_window": float(lost[win][:, mlk].sum()), "lost_total_2025": float(lost[:, mlk].sum()),
                            "window_share_of_days": float(win.mean()), "window_share_of_lost": float(lost[win][:, mlk].sum() / lost[:, mlk].sum())}
    out["lebaran_all"] = {"window_share_of_lost": float(lost[win].sum() / lost.sum()), "window_share_of_days": float(win.mean())}
    fr = S.frontier()
    pf = fr[fr.policy == "Proposed policy"]
    cur = fr[fr.policy == "Current policy"].iloc[0]
    rows = {}
    for beta in (0.93, 0.95, 0.97):
        r = pf[np.isclose(pf.beta, beta)].iloc[0]
        rows[str(beta)] = {"fill_rate": r.fill_rate, "d_fill_pts": (r.fill_rate - cur.fill_rate) * 100,
                           "d_inventory_pct": (r.avg_inventory_idr / cur.avg_inventory_idr - 1) * 100,
                           "d_total_cost_pct": (r.total_cost / cur.total_cost - 1) * 100,
                           "d_lost_units_pct": (r.lost_units / cur.lost_units - 1) * 100,
                           "inventory_days": r.inventory_days}
    out["proposed"] = rows
    # per-scenario range + series that do not win at 0.95
    kp = S.kpi_frame("proposed", 0.95)
    sc = PB.aggregate(kp).total_cost / PB.aggregate(kb).total_cost - 1
    out["scenario_range_cost_pct_095"] = [float(sc.min() * 100), float(sc.max() * 100)]
    c0 = kb.set_index(["scenario", "sku_id", "region"]).total_cost
    c1 = kp.set_index(["scenario", "sku_id", "region"]).total_cost
    robust = (c1 < c0).groupby(level=[1, 2]).mean()
    out["series_not_robust_095"] = int((robust < 0.8).sum())
    out["series_robust_095"] = int((robust >= 0.8).sum())
    # iso-service / iso-inventory
    out["iso_fill_inventory_saving_pct"] = float((1 - np.interp(cur.fill_rate, pf.fill_rate, pf.avg_inventory_idr) / cur.avg_inventory_idr) * 100)
    out["iso_inventory_fill_gain_pts"] = float((np.interp(cur.avg_inventory_idr, pf.avg_inventory_idr, pf.fill_rate) - cur.fill_rate) * 100)
    # simulator fidelity
    s0 = slice(PB.WARMUP_DAYS, None)
    out["fidelity"] = {"sim_stockout": float((b["base"].lost[s0, 0] > 0).mean()), "actual_stockout": float(w.actual_so[s0].mean()),
                       "sim_stock": float(b["base"].oh_end[s0, 0].mean()), "actual_stock": float(w.actual_oh[s0].mean())}
    # supply
    comp = SUP.completed_pos(po)
    u = SUP.unreliability_cost(panel, comp, 0.95)
    rc = SUP.region_ss_comparison(u)
    out["supply"] = {r.region: {"ss_days": r.ss_days, "premium": r.premium_vs_ref, "lt_share": r.lt_share_of_ss, "extra_ss_idr_m": r.extra_ss_idr / 1e6} for r in rc.itertuples()}
    rel = SUP.reliability_table(comp, ["region"])
    out["lead_time"] = {r.region: {"mean": r.actual_lt, "std": r.sigma_L, "disrupted": r.pct_disrupted, "late": r.pct_late} for r in rel.itertuples()}
    out["disruption_rate"] = float(comp.is_disruption.mean())
    # promo
    ev = AN.promo_event_study(panel)
    out["promo"] = AN.promo_summary(ev)
    # patterns
    idx = S.indexed_panel()
    pw = idx.assign(pw=idx.is_payday_window == 1).groupby(["category", "pw"]).idx28.mean().unstack()
    out["payday_lift"] = (pw[True] / pw[False] - 1).round(3).to_dict()
    lp = AN.lebaran_profile(idx)
    pk = lp.loc[lp.groupby("category").idx91m.idxmax()].set_index("category")
    out["lebaran_peak"] = {c: {"idx": float(r.idx91m), "days": int(r.days_to_lebaran)} for c, r in pk.iterrows()}
    _, sl = AN.temperature_relation(idx)
    out["temp_slope_pct_per_degC"] = {r.group: float(r.pct_per_degC * 100) for r in sl.itertuples()}
    # validation
    raw = S.censoring_bias()
    for m in ("glm", "ma28"):
        t = VAL.summarise_bias(raw[raw.model == m], [])
        out[f"censoring_{m}"] = {r.variant: {"bias": r.Bias, "wape": r.WAPE} for r in t.itertuples()}
    f = VAL.fill_rate_truth_vs_proxy(panel, truth)
    out["fill"] = {"true": float(1 - f.lost.sum() / f.demand.sum()), "proxy": float(f.proxy_fill.mean()), "corr": float(np.corrcoef(f.true_fill, f.proxy_fill)[0, 1])}
    # recommendations
    bk = S.kpi_frame("baseline")
    ks = {m: S.review_kpis(m) for m in (1, 2, 3)}
    leb = RC.leb_window_stats(w, b["base"], S.proposed_sim(0.95))
    sku = data.load_sku_master().set_index("sku_id").shelf_life_days.to_dict()
    rd = {k: int(r) for k, r in zip(w.keys, w.R)}
    act, none = RC.build_actions(bk, ks, sku, leb, rd)
    out["actions"] = {"n": len(act), "total_idr_m": float(act.impact_idr_year.sum() / 1e6), "by_type": (act.groupby("type").impact_idr_year.sum() / 1e6).round(1).to_dict(),
                      "n_by_type": act.type.value_counts().to_dict(), "none": len(none), "top3": act.head(3).action.tolist()}
    return out


if __name__ == "__main__":
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return str(o)
    print(json.dumps(main(), indent=1, default=conv))
