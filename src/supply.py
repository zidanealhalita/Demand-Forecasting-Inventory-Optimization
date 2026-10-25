"""Supply-risk analytics: lead-time reliability, disruption detection, cost of unreliability, actions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .analytics import demand_stats
from .inventory import eoq, z_score

DISRUPTION_MARGIN_DAYS = 3  # a PO is a "disruption" if delay > planned + 3 days


def completed_pos(po: pd.DataFrame) -> pd.DataFrame:
    """Closed purchase orders with delay, late and disruption flags."""
    c = po[po["actual_lead_time_days"].notna()].copy()
    c["delay_days"] = c["actual_lead_time_days"] - c["planned_lead_time_days"]
    c["is_late"] = c["delay_days"] > 0
    c["is_disruption"] = c["delay_days"] > DISRUPTION_MARGIN_DAYS
    c["order_month"] = c["order_date"].dt.to_period("M").dt.to_timestamp()
    return c


def reliability_table(c: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Per group: #POs, % late, % disruptions, mean / P90 delay, lead-time mean and std."""
    g = c.groupby(by)
    return g.agg(
        pos=("po_id", "size"),
        pct_late=("is_late", "mean"),
        pct_disrupted=("is_disruption", "mean"),
        mean_delay=("delay_days", "mean"),
        p90_delay=("delay_days", lambda x: float(np.percentile(x, 90))),
        planned_lt=("planned_lead_time_days", "mean"),
        actual_lt=("actual_lead_time_days", "mean"),
        sigma_L=("actual_lead_time_days", "std"),
    ).reset_index()


def disruption_trend(c: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Share of disrupted POs by order month (optionally split by ``by``)."""
    keys = ["order_month"] + ([by] if by else [])
    return c.groupby(keys).agg(pct_disrupted=("is_disruption", "mean"), pos=("po_id", "size")).reset_index()


def unreliability_cost(
    panel: pd.DataFrame, c: pd.DataFrame, service_level: float = 0.95
) -> pd.DataFrame:
    """Extra safety stock caused by lead-time variability, per SKU x region.

    ``SS_with = z sqrt((L+R) sigma_d^2 + d^2 sigma_L^2)`` versus ``SS_without = z sqrt((L+R) sigma_d^2)``
    with the *observed* mean lead time L and review period R. The difference is the buffer paid purely for
    unreliable supply. Annual holding cost = extra units x unit cost x holding %.
    """
    z = float(z_score(service_level))
    rows = []
    lt = c.groupby(["sku_id", "region"])["actual_lead_time_days"].agg(["mean", "std"])
    for (sku, reg), s in panel.groupby(["sku_id", "region"]):
        s = s.set_index("date").sort_index()
        st = demand_stats(s)
        first = s.iloc[0]
        L, sL = lt.loc[(sku, reg), "mean"], lt.loc[(sku, reg), "std"]
        R = float(first["review_period_days"])
        P = L + R
        d, sd = st["d_bar"], st["sigma_d"]
        ss_with = z * np.sqrt(P * sd**2 + d**2 * sL**2)
        ss_without = z * np.sqrt(P * sd**2)
        extra = ss_with - ss_without
        rows.append(
            {
                "sku_id": sku,
                "region": reg,
                "d_bar": d,
                "L_bar": L,
                "sigma_L": sL,
                "R": R,
                "ss_units": ss_with,
                "ss_days": ss_with / d,
                "ss_without_lt_units": ss_without,
                "extra_ss_units": extra,
                "extra_ss_idr": extra * first["unit_cost_idr"],
                "ss_idr": ss_with * first["unit_cost_idr"],
                "extra_holding_idr_year": extra * first["unit_cost_idr"] * first["holding_cost_pct_per_year"],
                "lt_share_of_ss": extra / ss_with if ss_with > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def region_ss_comparison(u: pd.DataFrame, ref: str = "Jabodetabek") -> pd.DataFrame:
    """Safety stock in *days of demand* per region (demand-weighted) and the % premium vs ``ref``."""
    g = u.groupby("region").apply(
        lambda d: pd.Series(
            {
                "ss_days": float((d["ss_days"] * d["d_bar"]).sum() / d["d_bar"].sum()),
                "extra_ss_idr": d["extra_ss_idr"].sum(),
                "ss_idr": d["ss_idr"].sum(),
                "lt_share_of_ss": float(d["extra_ss_units"].sum() / d["ss_units"].sum()),
            }
        ),
        include_groups=False,
    )
    g["premium_vs_ref"] = g["ss_days"] / g.loc[ref, "ss_days"] - 1
    return g.reset_index()


def action_table(
    rel: pd.DataFrame, u: pd.DataFrame, sku_master: pd.DataFrame, revenue: pd.Series
) -> pd.DataFrame:
    """Rule-based action per SKU x region.

    Rules (thresholds shown in the page caption):
    * **Dual-source**: >= 5 % disrupted POs *or* P90 delay >= 5 days, on an A-class (top 80 % revenue) item.
    * **Buffer increase**: mean delay > 0.5 day (planned lead time understated) - raise planned L / safety stock.
    * **Review-period change**: EOQ cycle > 2x the review period (orders too frequent for the ordering cost).
    * otherwise **Monitor**.
    """
    t = rel.merge(u[["sku_id", "region", "d_bar", "extra_ss_idr", "extra_holding_idr_year", "lt_share_of_ss"]], on=["sku_id", "region"])
    sm = sku_master.set_index("sku_id")
    eoq_days, rp = [], []
    for _, r in t.iterrows():
        m = sm.loc[r["sku_id"]]
        H = m["unit_cost_idr"] * m["holding_cost_pct_per_year"]
        _, q = eoq(r["d_bar"] * 365, m["ordering_cost_idr"], H, int(m["case_size"]))
        eoq_days.append(q / r["d_bar"])
        rp.append(m["review_period_days"])
    t["eoq_cycle_days"], t["review_days"] = eoq_days, rp
    rev = revenue.sort_values(ascending=False)
    a_class = set(rev[(rev.cumsum() - rev) / rev.sum() < 0.8].index)
    t["is_A"] = [(s, g) in a_class for s, g in zip(t["sku_id"], t["region"])]
    acts = []
    for _, r in t.iterrows():
        a = []
        if r["is_A"] and (r["pct_disrupted"] >= 0.05 or r["p90_delay"] >= 5):
            a.append("Dual-source / alternate supplier")
        if r["mean_delay"] > 0.5:
            a.append("Increase buffer (planned LT understated)")
        if r["eoq_cycle_days"] > 2 * r["review_days"]:
            a.append("Lengthen review period")
        acts.append(" + ".join(a) if a else "Monitor")
    t["action"] = acts
    return t
