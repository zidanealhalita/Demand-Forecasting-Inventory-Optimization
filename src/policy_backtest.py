"""Policy back-test: *current* replenishment policy vs a *proposed* forecast-based policy.

Both policies are run through the same day-by-day simulator over 2025 (all 50 SKU x region stocking
points at once, vectorised over series and lead-time scenarios) against the **ground-truth** daily demand
- the only place besides the validation appendix where ground truth is used. A real company would have to
estimate unconstrained demand (e.g. from stock-out-corrected sales); here we use the truth so that the
comparison is fair and not distorted by censoring.

Current policy (reverse-engineered from ``purchase_orders.csv``)
    Periodic review every ``R`` days, order up to
    ``S = a * MA28 * T + b * SD28 * sqrt(T)`` with ``T = L + R + 1`` (a = 0.974, b = 1.239 fitted on the POs; nominal 1 and 1.28) where MA28 / SD28 are the trailing 28-day mean
    and std of **sales** (so the policy inherits the censoring bias) and ``L`` is the *planned* lead time.

Proposed policy
    Same review calendar and the same case-size rounding (so ordering cost is identical), but

    1. demand forecast = Poisson calendar regression (promo, payday, Ramadan/Lebaran, holidays, temperature,
       trend), re-fitted monthly on stockout-excluded history, with a 28-day level correction;
    2. protection period ``P = L_obs + R`` using the *observed* mean lead time (not the planned one);
    3. safety stock ``z * d_bar * sqrt(P sigma_rel^2 + sigma_L^2)`` which includes **lead-time variability**
       (``sigma_L`` observed) and a forecast-error ``sigma_rel`` estimated out-of-sample per regime
       (normal / Lebaran window / promo) -> an *event-calendar* buffer;
    4. ``z`` solved from a **fill-rate target** via the standard-normal loss function.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import features as F
from . import forecasting as FC
from .inventory import inverse_std_normal_loss

TEST_START = pd.Timestamp("2025-01-01")
WARMUP_DAYS = 28
HIST = 28  # trailing history window used by both policies
BASELINE_Z = 1.2816  # nominal 90 % one-sided service factor
# Calibration of the current policy on the 14,714 historical POs (OLS of implied order-up-to level on
# MA28*T and SD28*sqrt(T), T = L_planned + R + 1): coefficients 0.974 and 1.239 (close to 1 and 1.28).
# With them the simulator reproduces 2025 history (stockout rate 7.15 % vs 7.17 %, avg stock +1 %).
BASELINE_A = 0.974
BASELINE_B = 1.239
EXT = 50  # days predicted beyond each monthly block (look-ahead for the protection window)
RHO_BOUNDS = (0.75, 1.35)


# --------------------------------------------------------------------------------------------------
# World (inputs shared by both policies)
# --------------------------------------------------------------------------------------------------


@dataclass
class World:
    """Everything the simulator needs, shaped ``(T, N)`` (days x series) unless noted."""

    keys: list[tuple[str, str]]
    dates: pd.DatetimeIndex
    demand: np.ndarray
    price: np.ndarray
    unit_cost: np.ndarray  # (N,)
    holding_pct: np.ndarray  # (N,)
    order_cost: np.ndarray  # (N,)
    case_size: np.ndarray  # (N,)
    review: np.ndarray  # bool (T, N): review/order days (taken from the historical PO calendar)
    R: np.ndarray  # (N,)
    Lp: np.ndarray  # (N,) planned lead time
    oh0: np.ndarray  # (N,)
    pipe0: list[tuple[int, int, float]]  # (series idx, arrival day idx, qty) already in transit
    hist_sales: np.ndarray  # (HIST, N) actual sales just before the test window
    hist_so: np.ndarray  # (HIST, N) actual stockout flags
    lt_actual: np.ndarray  # (T, N) realised lead time of the PO placed that day (planned if none)
    lt_devs: list[np.ndarray] = field(default_factory=list)  # per series: observed (actual-planned) deviations pre-test
    days_to_lebaran: np.ndarray | None = None  # (T,)
    actual_sales: np.ndarray | None = None  # (T, N) real units_sold (for replay validation)
    actual_oh: np.ndarray | None = None  # (T, N) real on_hand_end
    actual_so: np.ndarray | None = None  # (T, N) real stockout flag


def _series_frames(panel: pd.DataFrame, keys: list[tuple[str, str]]) -> dict[tuple[str, str], pd.DataFrame]:
    return {k: g.set_index("date").sort_index() for k, g in panel.groupby(["sku_id", "region"])}


def build_world(
    panel: pd.DataFrame,
    truth: pd.DataFrame,
    po: pd.DataFrame,
    cal: pd.DataFrame,
    start: pd.Timestamp = TEST_START,
) -> World:
    """Assemble the simulation inputs for the test window ``[start, end of data]``."""
    keys = sorted(panel[["sku_id", "region"]].drop_duplicates().itertuples(index=False, name=None))
    frames = _series_frames(panel, keys)
    tr = truth.set_index(["sku_id", "region", "date"])["units_demand_unconstrained"]
    dates = pd.DatetimeIndex(sorted(panel.loc[panel["date"] >= start, "date"].unique()))
    T, N = len(dates), len(keys)
    demand = np.zeros((T, N))
    price = np.zeros((T, N))
    review = np.zeros((T, N), bool)
    lt_actual = np.zeros((T, N))
    oh0 = np.zeros(N)
    hist_s = np.zeros((HIST, N))
    hist_o = np.zeros((HIST, N))
    unit_cost = np.zeros(N)
    hold = np.zeros(N)
    ocost = np.zeros(N)
    case = np.zeros(N)
    R = np.zeros(N)
    Lp = np.zeros(N)
    a_sales = np.zeros((T, N))
    a_oh = np.zeros((T, N))
    a_so = np.zeros((T, N))
    pipe0: list[tuple[int, int, float]] = []
    devs: list[np.ndarray] = []
    d2i = {d: i for i, d in enumerate(dates)}
    for n, (sku, reg) in enumerate(keys):
        s = frames[(sku, reg)]
        test = s.loc[s.index >= start]
        pre = s.loc[s.index < start]
        demand[:, n] = tr.loc[(sku, reg)].reindex(dates).values
        price[:, n] = test["price_idr"].values
        a_sales[:, n] = test["units_sold"].values
        a_oh[:, n] = test["on_hand_end"].values
        a_so[:, n] = test["stockout_flag"].values
        oh0[n] = pre["on_hand_end"].iloc[-1]
        hist_s[:, n] = pre["units_sold"].iloc[-HIST:].values
        hist_o[:, n] = pre["stockout_flag"].iloc[-HIST:].values
        row = test.iloc[0]
        unit_cost[n], hold[n], ocost[n] = row["unit_cost_idr"], row["holding_cost_pct_per_year"], row["ordering_cost_idr"]
        case[n], R[n] = row["case_size"], row["review_period_days"]
        p = po[(po["sku_id"] == sku) & (po["region"] == reg)]
        Lp[n] = p["planned_lead_time_days"].mode().iloc[0]
        ptest = p[p["order_date"] >= start]
        idx = ptest["order_date"].map(d2i).dropna().astype(int).values
        review[idx, n] = True
        lt_actual[:, n] = Lp[n]
        v = ptest["actual_lead_time_days"].values
        ok = ~np.isnan(v)
        lt_actual[idx[ok], n] = v[ok]
        # in transit at the start of the window
        tr_po = p[(p["order_date"] < start) & (p["receipt_date"] >= start)]
        for _, r in tr_po.iterrows():
            pipe0.append((n, int((r["receipt_date"] - start).days), float(r["qty_ordered"])))
        done = p[(p["order_date"] < start) & p["actual_lead_time_days"].notna()]
        devs.append((done["actual_lead_time_days"] - done["planned_lead_time_days"]).values)
    dtl = cal.set_index("date").loc[dates, "days_to_lebaran"].values
    return World(
        keys=keys, dates=dates, demand=demand, price=price, unit_cost=unit_cost, holding_pct=hold, order_cost=ocost,
        case_size=case, review=review, R=R, Lp=Lp, oh0=oh0, pipe0=pipe0, hist_sales=hist_s, hist_so=hist_o,
        lt_actual=lt_actual, lt_devs=devs, days_to_lebaran=dtl, actual_sales=a_sales, actual_oh=a_oh, actual_so=a_so,
    )


def lead_time_scenarios(world: World, n_boot: int = 24, seed: int = 42) -> np.ndarray:
    """Lead-time scenarios ``(K, T, N)`` (integer days >= 1). Scenario 0 = the realised history (replay);
    scenarios 1.. bootstrap the observed deviations (actual - planned) of each series. Both policies receive
    the *same* array (common random numbers)."""
    rng = np.random.default_rng(seed)
    T, N = world.demand.shape
    out = np.zeros((n_boot + 1, T, N), int)
    out[0] = np.maximum(np.rint(world.lt_actual), 1)
    for n in range(N):
        d = world.lt_devs[n]
        draws = rng.choice(d, size=(n_boot, T)) if len(d) else np.zeros((n_boot, T))
        out[1:, :, n] = np.maximum(np.rint(world.Lp[n] + draws), 1)
    return out


# --------------------------------------------------------------------------------------------------
# Forecast tables for the proposed policy (monthly re-fit, strictly causal)
# --------------------------------------------------------------------------------------------------


@dataclass
class ForecastTables:
    """Static (path-independent) inputs of the proposed policy, shaped ``(T, N)``; ``G_hist`` is ``(HIST, N)``."""

    FS: np.ndarray  # forecast demand summed over the protection period
    C: np.ndarray  # sqrt(P sigma_rel^2 + sigma_L^2) / P : safety-stock factor per unit of mean demand
    c_over_R: np.ndarray  # R / sqrt(P sigma_rel^2 + sigma_L^2): drives z via the loss function
    G: np.ndarray  # one-step model prediction for each test day (used for the level correction)
    G_hist: np.ndarray
    P: np.ndarray
    sigma_rel: np.ndarray
    sigma_L: np.ndarray
    L_bar: np.ndarray
    oos: pd.DataFrame  # out-of-sample residuals (series, date, rel_error, class) for the appendix/diagnostics


def _class_array(ctx: FC.SeriesContext) -> np.ndarray:
    """Regime of each day: 2 = Lebaran window, 1 = promo / post-promo, 0 = normal."""
    leb = ctx.cal.loc[ctx.dates, "in_leb_window"].values > 0
    pro = (ctx.promo["promo"].values > 0) | (ctx.promo["post_promo"].values > 0)
    return np.where(leb, 2, np.where(pro, 1, 0))


def _frac_window_sum(cs: np.ndarray, start: int, P: float) -> tuple[float, float]:
    """Sum over [start, start+P) of a series with cumulative sum ``cs`` (length n+1) incl. fractional end.
    Returns (sum, covered_length); windows beyond the data are truncated."""
    n = len(cs) - 1
    end = start + P
    if start >= n:
        return 0.0, 0.0
    end_c = min(end, n)
    lo = int(np.floor(end_c))
    frac = end_c - lo
    total = cs[lo] - cs[start]
    if lo < n:
        total += frac * (cs[lo + 1] - cs[lo])
    return float(total), float(end_c - start)


def build_forecast_tables(
    panel: pd.DataFrame,
    po: pd.DataFrame,
    cal_feats: pd.DataFrame,
    world: World,
    first_origin: str = "2024-02-01",
    alpha: float = 2.0,
) -> ForecastTables:
    """Fit the monthly Poisson models and derive forecast sums, error scales and lead-time statistics."""
    T, N = world.demand.shape
    FS = np.zeros((T, N)); C = np.zeros((T, N)); CR = np.zeros((T, N)); G = np.zeros((T, N))
    Gh = np.zeros((HIST, N)); Pm = np.zeros((T, N)); SR = np.zeros((T, N)); SL = np.zeros((T, N)); LB = np.zeros((T, N))
    oos_all_list = []
    frames = _series_frames(panel, world.keys)
    for n, (sku, reg) in enumerate(world.keys):
        s = frames[(sku, reg)]
        ctx = FC.SeriesContext.build(s, cal_feats)
        dates = ctx.dates
        y, so = ctx.y, ctx.stockout
        nobs = len(dates)
        t0 = dates[0]
        tmean = float(ctx.temp[:365].mean())
        X_all = F.design_matrix(dates, ctx.cal, ctx.promo, ctx.temp, tmean, t0)
        tcol = list(X_all.columns).index("temp_dev")
        Xv = X_all.values
        cls = _class_array(ctx)
        first = int(np.searchsorted(dates, pd.Timestamp(first_origin)))
        month_starts = [i for i in range(first, nobs) if dates[i].day == 1]
        # per-block predictions
        blocks = []  # (origin idx, block end idx, G array from origin)
        series_oos = []
        for b, o in enumerate(month_starts):
            end_b = month_starts[b + 1] if b + 1 < len(month_starts) else nobs
            pred_end = min(nobs, end_b + EXT)
            w = (~so[:o]).astype(float)
            beta = FC.fit_poisson_ridge(Xv[:o], y[:o], w=w, alpha=alpha)
            clim = F.temperature_climatology(dates[:o], ctx.temp[:o], dates[o:pred_end])
            Xp = Xv[o:pred_end].copy()
            Xp[:, tcol] = clim - tmean
            Gb = FC.predict_poisson(Xp, beta)
            blocks.append((o, end_b, Gb))
            # out-of-sample residuals inside the month
            rng_ = np.arange(o, end_b)
            m = ~so[rng_]
            rel = (y[rng_] - Gb[: end_b - o]) / Gb[: end_b - o]
            series_oos.append(pd.DataFrame({"series": f"{sku}|{reg}", "block": b, "idx": rng_[m], "date": dates[rng_[m]], "rel_error": rel[m], "cls": cls[rng_[m]]}))
        oos = pd.concat(series_oos, ignore_index=True)
        # observed lead-time statistics, causal
        p = po[(po["sku_id"] == sku) & (po["region"] == reg) & po["receipt_date"].notna()].sort_values("receipt_date")
        rec = p["receipt_date"].values
        alt = p["actual_lead_time_days"].values
        # indicator cumulative sums for regime shares (calendar & promo plan are known in advance)
        cs_leb = np.concatenate([[0], np.cumsum(cls == 2)]).astype(float)
        cs_pro = np.concatenate([[0], np.cumsum(cls == 1)]).astype(float)
        for b, (o, end_b, Gb) in enumerate(blocks):
            hist = oos[oos["block"] < b]
            sr_n = _scale(hist, 0, None)
            sr_all = _scale(hist, None, None)
            s_norm = sr_n if sr_n is not None else (sr_all if sr_all is not None else 0.22)
            s_leb = _scale(hist, 2, 1.0 * s_norm) or 1.6 * s_norm
            s_pro = _scale(hist, 1, 1.0 * s_norm) or 1.3 * s_norm
            k = np.searchsorted(rec, np.datetime64(dates[o]))
            if k >= 25:
                Lbar, sL = float(alt[:k].mean()), float(alt[:k].std(ddof=1))
            else:
                Lbar, sL = float(world.Lp[n]), 1.5
            P = Lbar + float(world.R[n])
            cs_G = np.concatenate([[0], np.cumsum(Gb)])
            for i in range(o, end_b):
                j = i - o
                fs, cov = _frac_window_sum(cs_G, j, P)
                if cov <= 0:
                    fs, cov = Gb[-1] * P, P
                fs = fs * (P / cov)
                # regime shares over the protection window (index in global day space)
                lv, c1 = _frac_window_sum(cs_leb, i, P)
                pv, c2 = _frac_window_sum(cs_pro, i, P)
                c1 = c1 if c1 > 0 else 1.0
                sh_leb, sh_pro = lv / c1, pv / (c2 if c2 > 0 else 1.0)
                sh_norm = max(1.0 - sh_leb - sh_pro, 0.0)
                srel = np.sqrt(sh_norm * s_norm**2 + sh_leb * s_leb**2 + sh_pro * s_pro**2)
                c = np.sqrt(P * srel**2 + sL**2)
                di = i - int(np.searchsorted(dates, world.dates[0]))
                if 0 <= di < T:
                    FS[di, n], C[di, n], CR[di, n] = fs, c / P, world.R[n] / c
                    G[di, n], Pm[di, n], SR[di, n], SL[di, n], LB[di, n] = Gb[j], P, srel, sL, Lbar
                elif -HIST <= di < 0:
                    Gh[HIST + di, n] = Gb[j]
        oos_all_list.append(oos)
    oos_all = pd.concat(oos_all_list, ignore_index=True)
    return ForecastTables(FS, C, CR, G, Gh, Pm, SR, SL, LB, oos_all)


def _scale(h: pd.DataFrame, cls: int | None, fallback: float | None, min_n: int = 25) -> float | None:
    """Std of relative out-of-sample errors for a regime; ``fallback`` if too few samples."""
    x = h["rel_error"] if cls is None else h.loc[h["cls"] == cls, "rel_error"]
    if len(x) >= min_n:
        return float(x.std(ddof=1))
    return fallback


# --------------------------------------------------------------------------------------------------
# Day-by-day simulator (vectorised over scenarios and series)
# --------------------------------------------------------------------------------------------------


@dataclass
class SimResult:
    """Daily simulation traces shaped ``(T, K, N)``."""

    oh_end: np.ndarray
    sold: np.ndarray
    lost: np.ndarray
    order_q: np.ndarray
    S_used: np.ndarray
    policy: str
    beta: float | None = None


def simulate(
    world: World,
    lt: np.ndarray,
    policy: str,
    tables: ForecastTables | None = None,
    beta: float = 0.97,
    z_floor: float = 0.0,
) -> SimResult:
    """Run ``policy`` in {"baseline", "proposed"} against the ground-truth demand.

    Order of events each day: (1) on a review day compute the order-up-to level ``S`` from data up to
    yesterday and order ``ceil((S - IP)/case) * case``; the order arrives ``lt`` days later; (2) arrivals
    due today are added to stock; (3) demand is served from stock, the rest is lost (no back-orders);
    (4) histories (sales, stockout flag) are updated.
    """
    T, N = world.demand.shape
    K = lt.shape[0]
    maxA = T + int(lt.max()) + 3
    arr = np.zeros((K, N, maxA))
    oh = np.tile(world.oh0, (K, 1)).astype(float)
    pipe = np.zeros((K, N))
    for n, day, q in world.pipe0:
        arr[:, n, day] += q
        pipe[:, n] += q
    hs = np.repeat(world.hist_sales[:, None, :], K, axis=1).astype(float)  # (HIST,K,N)
    hso = np.repeat(world.hist_so[:, None, :], K, axis=1).astype(float)
    if policy == "proposed":
        assert tables is not None
        hG = np.repeat(tables.G_hist[:, None, :], K, axis=1)
        zt = inverse_std_normal_loss(np.clip((1 - beta) * tables.c_over_R, 1e-6, None))
        zt = np.maximum(zt, z_floor)
    oh_end = np.zeros((T, K, N)); sold_a = np.zeros((T, K, N)); lost_a = np.zeros((T, K, N))
    oq = np.zeros((T, K, N)); S_used = np.full((T, K, N), np.nan)
    kk, nn = np.meshgrid(np.arange(K), np.arange(N), indexing="ij")
    Lp, R, case = world.Lp, world.R, world.case_size
    for t in range(T):
        rv = world.review[t]
        if rv.any():
            if policy == "baseline":
                ma = hs.mean(axis=0)
                sd = hs.std(axis=0, ddof=1)
                Tb = (Lp + R + 1.0)[None, :]
                S = BASELINE_A * ma * Tb + BASELINE_B * sd * np.sqrt(Tb)
            else:
                unc = 1.0 - hso
                num = (hs * unc).sum(axis=0)
                den = (hG * unc).sum(axis=0)
                rho = np.clip(np.where(den > 1e-9, num / np.maximum(den, 1e-9), 1.0), *RHO_BOUNDS)
                S = rho * tables.FS[t][None, :] * (1.0 + zt[t][None, :] * tables.C[t][None, :] / 1.0)
            ip = oh + pipe
            q = np.maximum(S - ip, 0.0)
            q = np.ceil(q / case[None, :] - 1e-9) * case[None, :]
            q = q * rv[None, :]
            ai = t + lt[:, t, :]
            arr[kk, nn, ai] += q
            pipe += q
            oq[t] = q
            S_used[t] = np.where(rv[None, :], S, np.nan)
        a = arr[:, :, t]
        oh += a
        pipe -= a
        d = world.demand[t][None, :]
        sold = np.minimum(oh, d)
        lost = d - sold
        oh = oh - sold
        oh_end[t], sold_a[t], lost_a[t] = oh, sold, lost
        slot = t % HIST
        hs[slot] = sold
        hso[slot] = (lost > 0)
        if policy == "proposed":
            hG[slot] = tables.G[t][None, :]
    return SimResult(oh_end, sold_a, lost_a, oq, S_used, policy, beta if policy == "proposed" else None)


# --------------------------------------------------------------------------------------------------
# KPIs
# --------------------------------------------------------------------------------------------------


def kpis(world: World, res: SimResult, warmup: int = WARMUP_DAYS) -> pd.DataFrame:
    """Per (scenario, series) KPIs over the measured window (after ``warmup`` days)."""
    sl = slice(warmup, None)
    T, K, N = res.oh_end.shape
    days = T - warmup
    dem = world.demand[sl][:, None, :]
    sold = res.sold[sl]
    lost = res.lost[sl]
    inv = res.oh_end[sl]
    margin = (world.price[sl] - world.unit_cost[None, :])[:, None, :]
    dem_sum = np.broadcast_to(dem.sum(0), (K, N))
    fill = sold.sum(0) / dem_sum
    so_days = (lost > 0).sum(0) / days
    avg_inv = inv.mean(0)
    inv_val = avg_inv * world.unit_cost[None, :]
    avg_dem = dem.mean(0)
    hold = inv.sum(0) * (world.unit_cost * world.holding_pct / 365.0)[None, :]
    n_orders = (res.order_q[sl] > 0).sum(0)
    ordc = n_orders * world.order_cost[None, :]
    lostm = (lost * margin).sum(0)
    lost_units = lost.sum(0)
    rows = []
    for k in range(K):
        for n in range(N):
            rows.append((k, n))
    kk = np.array([r[0] for r in rows]); nn = np.array([r[1] for r in rows])
    out = pd.DataFrame(
        {
            "scenario": kk,
            "sku_id": [world.keys[n][0] for n in nn],
            "region": [world.keys[n][1] for n in nn],
            "demand_units": dem_sum[kk, nn],
            "lost_units": lost_units[kk, nn],
            "fill_rate": fill[kk, nn],
            "stockout_day_rate": so_days[kk, nn],
            "avg_inventory_units": avg_inv[kk, nn],
            "avg_inventory_idr": inv_val[kk, nn],
            "inventory_days": avg_inv[kk, nn] / avg_dem[0, nn],
            "holding_cost": hold[kk, nn],
            "ordering_cost": ordc[kk, nn],
            "lost_margin": lostm[kk, nn],
            "orders": n_orders[kk, nn],
        }
    )
    out["total_cost"] = out["holding_cost"] + out["ordering_cost"] + out["lost_margin"]
    out["days"] = days
    return out


def aggregate(k: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Aggregate per-series KPIs (demand-weighted fill rate; sums for costs)."""
    by = by or ["scenario"]
    g = k.groupby(by)
    agg = g.agg(
        demand_units=("demand_units", "sum"),
        lost_units=("lost_units", "sum"),
        avg_inventory_units=("avg_inventory_units", "sum"),
        avg_inventory_idr=("avg_inventory_idr", "sum"),
        holding_cost=("holding_cost", "sum"),
        ordering_cost=("ordering_cost", "sum"),
        lost_margin=("lost_margin", "sum"),
        total_cost=("total_cost", "sum"),
        stockout_day_rate=("stockout_day_rate", "mean"),
        orders=("orders", "sum"),
    )
    agg["fill_rate"] = 1 - agg["lost_units"] / agg["demand_units"]
    agg["inventory_days"] = k.assign(w=k["avg_inventory_units"]).groupby(by).apply(
        lambda d: d["avg_inventory_units"].sum() / (d["demand_units"].sum() / d["days"].iloc[0]), include_groups=False
    )
    return agg.reset_index()


# --------------------------------------------------------------------------------------------------
# Review-period what-if (order less often): same engine, new review calendar
# --------------------------------------------------------------------------------------------------


def with_review_multiplier(world: World, m: int, max_days: int = 28) -> World:
    """Copy of ``world`` where every series is reviewed every ``m * R`` days (capped at ``max_days``).

    The first review day of each series is kept (same phase); ordering cost scales with the number of orders.
    """
    from dataclasses import replace

    T, N = world.demand.shape
    R_new = np.minimum(world.R * m, max_days).astype(int)
    review = np.zeros((T, N), bool)
    for n in range(N):
        first = int(np.argmax(world.review[:, n]))
        review[first :: R_new[n], n] = True
    return replace(world, review=review, R=R_new.astype(float))
