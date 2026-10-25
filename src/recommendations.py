"""Turn simulation results into a ranked, quantified action list (annual IDR impact, all *simulated*)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import policy_backtest as PB

KEY = ["sku_id", "region"]
ROBUST_MIN = 0.8  # share of lead-time scenarios in which the action must reduce cost
FILL_TOLERANCE = 0.002  # review-period stretch may not lose more than 0.2 pt of fill rate
SHELF_LIFE_MAX_SHARE = 0.30  # average inventory days must stay below 30 % of the shelf life


def _per_series(k: pd.DataFrame) -> pd.DataFrame:
    """Mean over scenarios per series of the KPIs used here (scenario order re-indexed 0..)."""
    return k.groupby(KEY).mean(numeric_only=True)


def _scen_costs(k: pd.DataFrame) -> pd.DataFrame:
    """Total cost per (series x scenario) as a wide frame (rows = series, columns = scenario position)."""
    d = k.copy()
    d["pos"] = d.groupby(KEY).cumcount()
    return d.pivot_table(index=KEY, columns="pos", values="total_cost")


def leb_window_stats(world: PB.World, base: PB.SimResult, prop: PB.SimResult, scen: slice = slice(1, None)) -> pd.DataFrame:
    """Lebaran window (30 days before to 14 days after) lost units and the pre-Lebaran order-up-to uplift."""
    d = world.days_to_lebaran
    win = (d >= -30) & (d <= 14)
    pre = (d >= -14) & (d <= -1)
    lb = base.lost[win][:, scen].sum(axis=0).mean(axis=0)
    lp = prop.lost[win][:, scen].sum(axis=0).mean(axis=0)
    sb = np.nanmean(np.where(pre[:, None, None], base.S_used, np.nan)[:, scen], axis=(0, 1))
    sp = np.nanmean(np.where(pre[:, None, None], prop.S_used, np.nan)[:, scen], axis=(0, 1))
    out = pd.DataFrame({"leb_lost_base": lb, "leb_lost_prop": lp, "pre_leb_S_base": sb, "pre_leb_S_prop": sp})
    out.index = pd.MultiIndex.from_tuples(world.keys, names=KEY)
    return out


def build_actions(
    base_k: pd.DataFrame,
    prop_ks: dict[int, pd.DataFrame],
    shelf_life: dict[str, float],
    leb: pd.DataFrame,
    review_days: dict[tuple[str, str], int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the ranked action list and the list of series where no robust gain was found.

    ``prop_ks`` maps review-period multiplier (1, 2, 3) to proposed-policy KPIs (same scenarios as ``base_k``).
    """
    days = float(base_k["days"].iloc[0])
    ann = 365.0 / days
    b = _per_series(base_k)
    bc = _scen_costs(base_k)
    p1 = _per_series(prop_ks[1])
    c1 = _scen_costs(prop_ks[1])
    rows, none_rows = [], []
    for key in b.index:
        sku, reg = key
        impact = (b.loc[key, "total_cost"] - p1.loc[key, "total_cost"]) * ann
        robust = float((c1.loc[key].values < bc.loc[key].values).mean())
        d_inv_u = p1.loc[key, "avg_inventory_units"] - b.loc[key, "avg_inventory_units"]
        d_inv_i = p1.loc[key, "avg_inventory_idr"] - b.loc[key, "avg_inventory_idr"]
        lost_av = (b.loc[key, "lost_units"] - p1.loc[key, "lost_units"]) * ann
        d_fill = (p1.loc[key, "fill_rate"] - b.loc[key, "fill_rate"]) * 100
        if impact > 0 and robust >= ROBUST_MIN:
            lw = leb.loc[key]
            leb_av = (lw["leb_lost_base"] - lw["leb_lost_prop"])
            lost_total_b = (b.loc[key, "lost_units"] - p1.loc[key, "lost_units"])
            event_led = lost_total_b > 0 and leb_av / lost_total_b >= 0.4 and lw["pre_leb_S_prop"] > lw["pre_leb_S_base"]
            if event_led:
                txt = (
                    f"Raise the order-up-to level for {sku} in {reg} before Lebaran "
                    f"(+{lw['pre_leb_S_prop'] - lw['pre_leb_S_base']:,.0f} units in the 14 days before; "
                    f"avoids ~{leb_av:,.0f} lost units in the Lebaran window)"
                )
            elif d_inv_u > 0.03 * b.loc[key, "avg_inventory_units"]:
                txt = (
                    f"Raise safety stock for {sku} in {reg} (+{d_inv_u:,.0f} units average stock; "
                    f"avoids ~{lost_av:,.0f} lost units/yr, fill rate {d_fill:+.1f} pts)"
                )
            elif d_inv_u < -0.03 * b.loc[key, "avg_inventory_units"]:
                txt = (
                    f"Reduce stock of {sku} in {reg} ({d_inv_u:,.0f} units, frees IDR {-d_inv_i / 1e6:,.1f} m of inventory) "
                    f"while fill rate moves {d_fill:+.1f} pts"
                )
            else:
                txt = f"Re-time the buffer for {sku} in {reg}: same average stock, better placed on event/promo days (fill rate {d_fill:+.1f} pts)"
            rows.append(
                dict(type="Forecast-based replenishment", sku_id=sku, region=reg, action=txt, impact_idr_year=impact,
                     lost_units_avoided_year=lost_av, inv_change_units=d_inv_u, inv_change_idr=d_inv_i, robust_share=robust,
                     method="Simulated 2025 back-test, current vs proposed policy, 20 lead-time scenarios")
            )
        else:
            none_rows.append(dict(sku_id=sku, region=reg, impact_idr_year=impact, robust_share=robust, fill_pts=d_fill,
                                  d_inv_idr_m=d_inv_i / 1e6,
                                  reason="costs more or gain not robust across scenarios - keep current policy here"))
        # review-period stretch (incremental to the proposed policy at the current review period)
        best = None
        for m in (2, 3):
            if m not in prop_ks:
                continue
            pm = _per_series(prop_ks[m])
            cm = _scen_costs(prop_ks[m])
            inc = (p1.loc[key, "total_cost"] - pm.loc[key, "total_cost"]) * ann
            rob = float((cm.loc[key].values < c1.loc[key].values).mean())
            ok_fill = pm.loc[key, "fill_rate"] >= b.loc[key, "fill_rate"] - FILL_TOLERANCE
            ok_shelf = pm.loc[key, "inventory_days"] <= SHELF_LIFE_MAX_SHARE * shelf_life[sku]
            if inc > 0 and rob >= ROBUST_MIN and ok_fill and ok_shelf and (best is None or inc > best[1]):
                best = (m, inc, rob, pm.loc[key])
        if best is not None:
            m, inc, rob, pr = best
            r0 = review_days[key]
            rows.append(
                dict(type="Review period", sku_id=sku, region=reg,
                     action=(f"Lengthen the review period for {sku} in {reg} from {r0} to {min(r0 * m, 28)} days "
                             f"(orders/yr {p1.loc[key, 'orders'] * ann:,.0f} -> {pr['orders'] * ann:,.0f}; fill rate stays >= current)"),
                     impact_idr_year=inc, lost_units_avoided_year=np.nan,
                     inv_change_units=pr["avg_inventory_units"] - p1.loc[key, "avg_inventory_units"],
                     inv_change_idr=pr["avg_inventory_idr"] - p1.loc[key, "avg_inventory_idr"], robust_share=rob,
                     method="Simulated: proposed policy with review period x%d; ordering cost IDR/PO from sku_master" % m)
            )
    act = pd.DataFrame(rows).sort_values("impact_idr_year", ascending=False).reset_index(drop=True)
    act.insert(0, "rank", np.arange(1, len(act) + 1))
    return act, pd.DataFrame(none_rows).sort_values("impact_idr_year")
