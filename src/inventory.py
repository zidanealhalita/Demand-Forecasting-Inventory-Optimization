"""Inventory maths: safety stock, reorder point, EOQ, cost curve, Monte-Carlo what-if, ABC-XYZ.

Notation (daily units): ``d_bar`` mean demand, ``sigma_d`` std of daily demand (forecast error),
``L_bar`` mean lead time, ``sigma_L`` std of lead time, ``R`` review period, ``z`` service factor.
All functions are pure NumPy / SciPy and unit-tested in ``tests/test_inventory.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats

# --------------------------------------------------------------------------------------------------
# Core formulas
# --------------------------------------------------------------------------------------------------


def z_score(service_level: float | np.ndarray) -> float | np.ndarray:
    """Standard-normal quantile for a cycle service level (e.g. 0.95 -> 1.645)."""
    return stats.norm.ppf(service_level)


def sigma_lead_time_demand(d_bar: float, sigma_d: float, L_bar: float, sigma_L: float) -> float:
    r"""Std of demand during the (variable) lead time: :math:`\sqrt{\bar L\,\sigma_d^2 + \bar d^2\,\sigma_L^2}`.

    The first term is demand uncertainty accumulated over ``L_bar`` days, the second is the extra uncertainty
    caused by the lead time itself being random.
    """
    return float(np.sqrt(L_bar * sigma_d**2 + (d_bar**2) * sigma_L**2))


def safety_stock(z: float, sigma_lt: float) -> float:
    """Safety stock ``SS = z * sigma_LT``."""
    return float(z * sigma_lt)


def reorder_point(d_bar: float, L_bar: float, ss: float) -> float:
    """Reorder point ``ROP = d_bar * L_bar + SS``."""
    return float(d_bar * L_bar + ss)


def eoq(annual_demand: float, order_cost: float, unit_holding_cost: float, case_size: int = 1) -> tuple[float, int]:
    """Economic order quantity ``sqrt(2 D S / H)`` and the same rounded **up** to a multiple of ``case_size``.

    Returns ``(raw_eoq, rounded_eoq)``.
    """
    raw = float(np.sqrt(2.0 * annual_demand * order_cost / unit_holding_cost))
    case = max(int(case_size), 1)
    rounded = int(np.ceil(raw / case) * case)
    return raw, max(rounded, case)


def order_up_to_level(d_bar: float, L_bar: float, R: float, sigma_d: float, sigma_L: float, z: float) -> float:
    """Periodic-review order-up-to level ``S = d_bar (L+R) + z sqrt((L+R) sigma_d^2 + d_bar^2 sigma_L^2)``."""
    P = L_bar + R
    return float(d_bar * P + z * np.sqrt(P * sigma_d**2 + (d_bar**2) * sigma_L**2))


def std_normal_loss(z: float | np.ndarray) -> float | np.ndarray:
    """Standard-normal loss function ``G(z) = phi(z) - z (1 - Phi(z))`` = expected shortage per unit of sigma."""
    return stats.norm.pdf(z) - z * (1.0 - stats.norm.cdf(z))


def inverse_std_normal_loss(target: float | np.ndarray, lo: float = -1.0, hi: float = 4.5) -> np.ndarray:
    """Solve ``G(z) = target`` (vectorised via a lookup table + interpolation; G is monotone decreasing)."""
    grid = np.linspace(lo, hi, 2001)
    g = std_normal_loss(grid)
    t = np.clip(np.asarray(target, float), g.min(), g.max())
    return np.interp(t, g[::-1], grid[::-1])


def cost_curve(
    d_bar: float,
    sigma_d: float,
    L_bar: float,
    sigma_L: float,
    unit_cost: float,
    holding_pct: float,
    order_cost: float,
    unit_margin: float,
    case_size: int = 1,
    service_levels: np.ndarray | None = None,
) -> pd.DataFrame:
    """Expected annual cost vs cycle service level for a (Q, r) policy with Q = EOQ.

    * holding   = H * (Q/2 + SS)               with ``H = unit_cost * holding_pct``
    * ordering  = S * D / Q
    * stockout  = (D / Q) * sigma_LT * G(z) * unit_margin   (expected lost units per cycle x cycles per year)
    """
    sl = np.linspace(0.80, 0.999, 120) if service_levels is None else np.asarray(service_levels, float)
    D = d_bar * 365.0
    H = unit_cost * holding_pct
    _, Q = eoq(D, order_cost, H, case_size)
    sig = sigma_lead_time_demand(d_bar, sigma_d, L_bar, sigma_L)
    z = z_score(sl)
    ss = z * sig
    hold = H * (Q / 2.0 + ss)
    order = np.full_like(sl, order_cost * D / Q)
    short = (D / Q) * sig * std_normal_loss(z) * unit_margin
    return pd.DataFrame(
        {
            "service_level": sl,
            "z": z,
            "safety_stock": ss,
            "holding_cost": hold,
            "ordering_cost": order,
            "stockout_cost": short,
            "total_cost": hold + order + short,
            "Q": Q,
        }
    )


def optimal_service_level(curve: pd.DataFrame) -> pd.Series:
    """Row of the cost curve with the minimum total expected cost."""
    return curve.loc[curve["total_cost"].idxmin()]


# --------------------------------------------------------------------------------------------------
# ABC - XYZ classification
# --------------------------------------------------------------------------------------------------


def abc_classify(value: pd.Series, a_cut: float = 0.80, b_cut: float = 0.95) -> pd.Series:
    """ABC class by cumulative value share (A = first 80 %, B = next 15 %, C = last 5 %).

    An item gets the class of the cumulative share **before** it is added, so the item that crosses a
    boundary stays in the higher class and there is always at least one A item.
    """
    v = value.astype(float)
    order = v.sort_values(ascending=False)
    share_before = (order.cumsum() - order) / order.sum()
    cls = np.where(share_before < a_cut, "A", np.where(share_before < b_cut, "B", "C"))
    return pd.Series(cls, index=order.index).reindex(value.index)


def xyz_classify(cv: pd.Series, x_cut: float = 0.5, y_cut: float = 1.0) -> pd.Series:
    """XYZ class by coefficient of variation: X < 0.5, Y in [0.5, 1.0], Z > 1.0."""
    c = cv.astype(float)
    return pd.Series(np.where(c < x_cut, "X", np.where(c <= y_cut, "Y", "Z")), index=c.index)


def coefficient_of_variation(x: pd.Series | np.ndarray) -> float:
    """Sample coefficient of variation std/mean (0 if mean is 0)."""
    a = np.asarray(x, float)
    m = a.mean()
    return float(a.std(ddof=1) / m) if m > 0 and len(a) > 1 else 0.0


POLICY_MATRIX: dict[str, str] = {
    "AX": "Tight control: high service (98-99%), forecast-driven replenishment, frequent review.",
    "AY": "High service with event buffers (Lebaran / promo); review forecast weekly.",
    "AZ": "Protect revenue but avoid over-stocking: collaborate with sales on demand signals, shorter review period.",
    "BX": "Standard service (95-97%), automated order-up-to policy.",
    "BY": "Standard service (95%), moderate safety stock, monitor monthly.",
    "BZ": "Moderate service (90-95%); buffer only for known events.",
    "CX": "Low-touch: lean policy (90-93%), larger less frequent orders (EOQ).",
    "CY": "Low-touch: 90% service, minimal safety stock.",
    "CZ": "Make-to-order / lowest stock: do not hold safety stock, review range regularly.",
}


# --------------------------------------------------------------------------------------------------
# Monte-Carlo what-if simulator (vectorised across runs)
# --------------------------------------------------------------------------------------------------


@dataclass
class MCInputs:
    """Inputs of the what-if simulation for one SKU x region."""

    d_bar: float  # planning mean daily demand
    sigma_rel: float  # relative daily demand noise (std / mean), from forecast errors
    dow_factor: np.ndarray  # length-7 multiplicative weekday profile (mean 1)
    lt_samples: np.ndarray  # empirical lead-time observations (days)
    R: int  # review period
    case_size: int
    unit_cost: float
    unit_margin: float
    holding_pct: float
    order_cost: float
    promo_uplift: float = 0.35  # relative uplift during promo
    post_promo_dip: float = 0.10  # relative dip in the 5 days after a promo
    start_dow: int = 0


def simulate_mc(
    p: MCInputs,
    service_level: float = 0.95,
    demand_surge: float = 0.0,
    lt_increase: float = 0.0,
    promo_on: bool = False,
    anticipate: bool = False,
    runs: int = 2000,
    days: int = 90,
    warmup: int = 28,
    seed: int = 42,
) -> dict:
    """Monte-Carlo simulation of a periodic-review order-up-to policy for ``days`` days.

    The policy is *planned* from the baseline statistics (``d_bar``, observed lead-time distribution) at the
    chosen ``service_level``. The *world* may then differ: ``demand_surge`` (+x % demand), ``lt_increase``
    (extra lead-time days) and ``promo_on`` (a 7-day promo in the middle of the horizon followed by a
    5-day dip). If ``anticipate`` is True the planner knows about the shock and re-plans.

    Everything is vectorised over ``runs`` (array shape ``(runs,)``); only the day loop is in Python.
    Returns summary KPIs plus per-run arrays for the histogram.
    """
    rng = np.random.default_rng(seed)
    L_obs = np.asarray(p.lt_samples, float)
    L_bar, sig_L = float(L_obs.mean()), float(L_obs.std(ddof=1)) if len(L_obs) > 1 else 0.0
    d_plan = p.d_bar * ((1 + demand_surge) if anticipate else 1.0)
    L_plan = L_bar + (lt_increase if anticipate else 0.0)
    P = L_plan + p.R
    z = float(z_score(service_level))
    S_level = d_plan * P + z * d_plan * np.sqrt(P * p.sigma_rel**2 + sig_L**2)
    if anticipate and promo_on:
        S_level += d_plan * p.promo_uplift * 7  # pre-build for the promo week
    total_days = warmup + days
    promo_start = warmup + days // 2
    # --- world demand: (runs, total_days) gamma noise with mean 1 and cv = sigma_rel
    k = 1.0 / max(p.sigma_rel, 1e-6) ** 2
    noise = rng.gamma(shape=k, scale=1.0 / k, size=(runs, total_days))
    mult = np.ones(total_days)
    if promo_on:
        mult[promo_start : promo_start + 7] *= 1 + p.promo_uplift
        mult[promo_start + 7 : promo_start + 12] *= 1 - p.post_promo_dip
    dow = (p.start_dow + np.arange(total_days)) % 7
    mean_t = p.d_bar * (1 + demand_surge) * p.dow_factor[dow] * mult
    demand = np.rint(noise * mean_t[None, :])
    # --- lead times: bootstrap from the empirical distribution (+ shock)
    lt = rng.choice(L_obs, size=(runs, total_days), replace=True) + lt_increase
    lt = np.maximum(np.rint(lt), 1).astype(int)
    max_lt = int(lt.max()) + 1
    arrivals = np.zeros((runs, total_days + max_lt + 1))
    on_hand = np.full(runs, S_level - d_plan * p.R / 2.0)
    pipeline = np.zeros(runs)
    rows = np.arange(runs)
    inv_hist = np.zeros((runs, total_days))
    lost = np.zeros((runs, total_days))
    n_orders = np.zeros(runs)
    for t in range(total_days):
        if t % p.R == 0:  # review at start of day: order up to S
            q = np.maximum(S_level - (on_hand + pipeline), 0.0)
            q = np.ceil(q / p.case_size) * p.case_size
            arrivals[rows, t + lt[:, t]] += q
            pipeline += q
            n_orders += (q > 0)
        arr = arrivals[:, t]
        on_hand += arr
        pipeline -= arr
        sold = np.minimum(on_hand, demand[:, t])
        lost[:, t] = demand[:, t] - sold
        on_hand = on_hand - sold
        inv_hist[:, t] = on_hand
    sl = slice(warmup, total_days)
    lost_u = lost[:, sl].sum(axis=1)
    dem_u = demand[:, sl].sum(axis=1)
    avg_inv = inv_hist[:, sl].mean(axis=1)
    so_days = (lost[:, sl] > 0).sum(axis=1)
    orders_meas = np.maximum(n_orders - np.ceil(warmup / p.R), 0)
    hold_cost = avg_inv * p.unit_cost * p.holding_pct * days / 365.0
    order_cost = orders_meas * p.order_cost
    stock_cost = lost_u * p.unit_margin
    total = hold_cost + order_cost + stock_cost
    mean_daily = dem_u.mean() / days
    return {
        "S_level": float(S_level),
        "stockout_prob": float((lost_u > 0).mean()),
        "stockout_days_mean": float(so_days.mean()),
        "lost_units_mean": float(lost_u.mean()),
        "lost_units_p50": float(np.percentile(lost_u, 50)),
        "lost_units_p90": float(np.percentile(lost_u, 90)),
        "fill_rate": float(1 - lost_u.sum() / dem_u.sum()),
        "avg_inventory": float(avg_inv.mean()),
        "inventory_days": float(avg_inv.mean() / mean_daily) if mean_daily > 0 else float("nan"),
        "cost_mean": float(total.mean()),
        "cost_p90": float(np.percentile(total, 90)),
        "cost_holding": float(hold_cost.mean()),
        "cost_ordering": float(order_cost.mean()),
        "cost_stockout": float(stock_cost.mean()),
        "lost_units_runs": lost_u,
        "runs": runs,
    }
