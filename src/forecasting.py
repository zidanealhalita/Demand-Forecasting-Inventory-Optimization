"""Forecasting models, metrics and the rolling-origin back-test.

Six models are compared on a strictly time-based split (no shuffling; all features are known at the
forecast origin):

1. ``naive``  - same weekday last week
2. ``ma28``   - moving average of the last 28 days
3. ``ets``    - Holt-Winters / ETS with weekly seasonality (statsmodels)
4. ``sarimax``- SARIMAX with exogenous regressors (promo, payday, Ramadan/Lebaran, temperature)
5. ``gbm``    - gradient boosting (direct multi-horizon, origin-level lag statistics only)
6. ``glm``    - Poisson regression on calendar + promo + temperature (also the engine behind the
   proposed inventory policy)

Censored demand: ``units_sold`` is capped by available stock. With ``exclude_stockouts=True`` the
stockout days are removed from training (SARIMAX / GLM / GBM) or imputed with the recent same-weekday
level (naive / MA / ETS, which need an unbroken series). Scoring always ignores stockout days in the test
window because true demand is unobservable there.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from . import features as F

MODEL_LABELS: dict[str, str] = {
    "naive": "Naive seasonal (same weekday last week)",
    "ma28": "Moving average 28d",
    "ets": "Holt-Winters (ETS, weekly)",
    "sarimax": "SARIMAX + exogenous",
    "gbm": "Gradient boosting (lags + calendar)",
    "glm": "Poisson regression (calendar)",
}
ALL_MODELS = list(MODEL_LABELS)

# --------------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------------


def wape(y: np.ndarray, f: np.ndarray) -> float:
    """Weighted absolute percentage error: sum|y-f| / sum|y|. Volume-weighted and safe when y has zeros."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    d = np.abs(y).sum()
    return float(np.abs(y - f).sum() / d) if d > 0 else float("nan")


def mape(y: np.ndarray, f: np.ndarray) -> float:
    """Mean absolute percentage error, ignoring days with y == 0 (undefined)."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    m = y > 0
    return float(np.mean(np.abs(y[m] - f[m]) / y[m])) if m.any() else float("nan")


def bias_pct(y: np.ndarray, f: np.ndarray) -> float:
    """Bias = sum(f - y) / sum(y). Positive = over-forecast, negative = under-forecast."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    d = y.sum()
    return float((f - y).sum() / d) if d > 0 else float("nan")


def mase(y: np.ndarray, f: np.ndarray, scale: float) -> float:
    """MASE: MAE of the forecast divided by the in-sample MAE of the seasonal naive (lag 7) forecast."""
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(f, float))) / scale) if scale > 0 else float("nan")


def seasonal_naive_scale(y: np.ndarray, valid: np.ndarray | None = None, season: int = 7) -> float:
    """In-sample mean absolute error of the lag-``season`` naive forecast (MASE denominator)."""
    y = np.asarray(y, float)
    if len(y) <= season:
        return float("nan")
    e = np.abs(y[season:] - y[:-season])
    if valid is not None:
        ok = np.asarray(valid, bool)
        e = e[ok[season:] & ok[:-season]]
    return float(e.mean()) if len(e) else float("nan")


def metric_row(y: np.ndarray, f: np.ndarray, scale: float) -> dict[str, float]:
    """All four leaderboard metrics for one model."""
    return {
        "WAPE": wape(y, f),
        "MAPE": mape(y, f),
        "Bias": bias_pct(y, f),
        "MASE": mase(y, f, scale),
    }


# --------------------------------------------------------------------------------------------------
# Poisson ridge regression (IRLS) - fast, vectorised, no sklearn overhead
# --------------------------------------------------------------------------------------------------


def fit_poisson_ridge(
    X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, alpha: float = 1.0, iters: int = 30
) -> np.ndarray:
    """Poisson (log-link) regression with an L2 penalty on all coefficients except the intercept (col 0).

    Solved by iteratively re-weighted least squares. ``w`` are observation weights (0 drops a day, used to
    exclude censored stockout days). Returns the coefficient vector.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    w = np.ones(n) if w is None else np.asarray(w, float)
    pen = np.eye(p) * alpha
    pen[0, 0] = 0.0
    beta = np.zeros(p)
    tot_w = w.sum()
    beta[0] = np.log(max((w * y).sum() / tot_w, 1e-6)) if tot_w > 0 else 0.0
    for _ in range(iters):
        eta = np.clip(X @ beta, -20, 20)
        mu = np.exp(eta)
        W = w * mu
        z = eta + (y - mu) / np.maximum(mu, 1e-9)
        A = X.T @ (W[:, None] * X) + pen
        b = X.T @ (W * z)
        new = np.linalg.solve(A, b)
        if np.max(np.abs(new - beta)) < 1e-7:
            beta = new
            break
        beta = new
    return beta


def predict_poisson(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Predicted mean demand ``exp(X @ beta)``."""
    return np.exp(np.clip(np.asarray(X, float) @ beta, -20, 20))


# --------------------------------------------------------------------------------------------------
# Series context & training-data preparation
# --------------------------------------------------------------------------------------------------


@dataclass
class SeriesContext:
    """All aligned arrays of one SKU x region needed by the models."""

    dates: pd.DatetimeIndex
    y: np.ndarray  # units_sold (censored)
    stockout: np.ndarray  # bool
    cal: pd.DataFrame  # calendar features indexed by date (full range)
    promo: pd.DataFrame  # promo / post_promo aligned to dates
    temp: np.ndarray  # observed avg temperature

    @classmethod
    def build(cls, s: pd.DataFrame, cal_feats: pd.DataFrame) -> "SeriesContext":
        """Build from one series frame (indexed by date) as returned by ``data.get_series``."""
        promo = F.promo_features(s["promo_flag"].reset_index(drop=True))
        promo.index = s.index
        return cls(
            dates=s.index,
            y=s["units_sold"].astype(float).values,
            stockout=s["stockout_flag"].astype(bool).values,
            cal=cal_feats,
            promo=promo,
            temp=s["avg_temp_c"].astype(float).values,
        )

    @property
    def event(self) -> np.ndarray:
        """Event-day flag for the full range."""
        return F.event_day_flag(self.cal.loc[self.dates], self.promo)


def impute_censored(y: np.ndarray, censored: np.ndarray, window: int = 28) -> np.ndarray:
    """Replace censored days with the mean of the same weekday over the previous ``window`` uncensored days.

    Strictly causal (uses only earlier days); falls back to the overall causal mean.
    """
    out = np.asarray(y, float).copy()
    n = len(out)
    cens = np.asarray(censored, bool)
    for t in np.flatnonzero(cens):
        lo = max(0, t - window)
        idx = np.arange(t - 7, lo - 1, -7)
        idx = idx[(idx >= 0) & (~cens[idx])]
        if len(idx):
            out[t] = out[idx].mean()
        else:
            prev = np.flatnonzero(~cens[lo:t]) + lo
            out[t] = out[prev].mean() if len(prev) else (out[~cens].mean() if (~cens).any() else out[t])
    return out


# --------------------------------------------------------------------------------------------------
# Individual models: every function returns an array of length ``h``
# --------------------------------------------------------------------------------------------------


def forecast_naive(y_imp: np.ndarray, h: int) -> np.ndarray:
    """Repeat the last observed week."""
    last = y_imp[-7:]
    return np.resize(last, h).astype(float)


def forecast_ma(y_fit: np.ndarray, h: int, window: int = 28) -> np.ndarray:
    """Flat forecast = mean of the last ``window`` valid days."""
    tail = y_fit[-window:]
    m = np.nanmean(tail) if np.isfinite(tail).any() else np.nanmean(y_fit)
    return np.full(h, m)


def forecast_ets(y_imp: np.ndarray, h: int, max_len: int = 540) -> np.ndarray:
    """Holt-Winters with damped trend and weekly multiplicative seasonality (additive if zeros present)."""
    y = np.asarray(y_imp[-max_len:], float)
    seasonal = "mul" if (y > 0).all() else "add"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ExponentialSmoothing(
                y, trend="add", damped_trend=True, seasonal=seasonal, seasonal_periods=7, initialization_method="estimated"
            ).fit(optimized=True)
            f = m.forecast(h)
        f = np.asarray(f, float)
        if not np.isfinite(f).all():
            raise ValueError("non-finite ETS forecast")
        return np.maximum(f, 0)
    except Exception:  # pragma: no cover - extremely defensive fallback
        return forecast_ma(y, h)


def forecast_sarimax(
    y_fit: np.ndarray, X_train: pd.DataFrame, X_future: pd.DataFrame, h: int, max_len: int = 540
) -> np.ndarray:
    """SARIMAX(1,0,1)x(1,0,0,7) with exogenous regressors. NaN (censored) days are handled by the Kalman filter."""
    y = np.asarray(y_fit[-max_len:], float)
    Xt = X_train.iloc[-max_len:].values.astype(float)
    scale = np.nanmean(y)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = SARIMAX(
                y / scale,
                exog=Xt,
                order=(1, 0, 1),
                seasonal_order=(1, 0, 0, 7),
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = mod.fit(disp=False, maxiter=40, method="lbfgs")
            f = res.forecast(h, exog=X_future.values.astype(float)) * scale
        f = np.asarray(f, float)
        if not np.isfinite(f).all():
            raise ValueError("non-finite SARIMAX forecast")
        return np.maximum(f, 0)
    except Exception:  # pragma: no cover
        return forecast_ma(y, h)


def forecast_glm(
    ctx: SeriesContext,
    origin: int,
    h: int,
    valid: np.ndarray,
    alpha: float = 2.0,
    train_len: int | None = None,
) -> np.ndarray:
    """Poisson calendar regression trained on days ``[0, origin)`` and applied to ``[origin, origin+h)``."""
    lo = 0 if train_len is None else max(0, origin - train_len)
    d_tr = ctx.dates[lo:origin]
    d_te = ctx.dates[origin : origin + h]
    t0 = ctx.dates[0]
    tmean = float(ctx.temp[lo:origin].mean())
    clim = F.temperature_climatology(ctx.dates[:origin], ctx.temp[:origin], d_te)
    Xtr = F.design_matrix(d_tr, ctx.cal, ctx.promo.iloc[lo:origin], ctx.temp[lo:origin], tmean, t0).values
    Xte = F.design_matrix(d_te, ctx.cal, ctx.promo.iloc[origin : origin + h], clim, tmean, t0).values
    beta = fit_poisson_ridge(Xtr, ctx.y[lo:origin], w=valid[lo:origin].astype(float), alpha=alpha)
    return predict_poisson(Xte, beta)


_GBM_CACHE: dict = {}


def forecast_gbm(ctx: SeriesContext, origin: int, h: int, valid: np.ndarray) -> np.ndarray:
    """Direct multi-horizon gradient boosting on ratio-to-level targets (origin statistics + target-day calendar)."""
    d_tr = ctx.dates[:origin]
    tmean = float(ctx.temp[:origin].mean())
    clim_all = F.temperature_climatology(ctx.dates[:origin], ctx.temp[:origin], ctx.dates[: origin + h])
    temp_dev = np.concatenate([ctx.temp[:origin], clim_all[origin : origin + h]]) - tmean
    prom = ctx.promo.iloc[: origin + h]
    X, Y = F.gbm_training_set(
        ctx.y[:origin], d_tr, ctx.cal, ctx.promo.iloc[:origin], temp_dev[:origin], valid[:origin], horizon=h
    )
    model = HistGradientBoostingRegressor(
        max_iter=120, learning_rate=0.08, max_depth=4, min_samples_leaf=20, l2_regularization=1.0, random_state=42
    )
    model.fit(X, Y)
    o = origin - 1
    ofeat, base = F.gbm_origin_row(ctx.y[:origin], valid[:origin], o)
    cal_cols = [f"dow_{k}" for k in range(1, 7)] + ["payday", "holiday", "year_end"] + F.LEB_COLS
    j = np.arange(1, h + 1)
    d_te = ctx.dates[origin : origin + h]
    cal_arr = ctx.cal.loc[d_te, cal_cols].values
    pro = prom[["promo", "post_promo"]].values[origin : origin + h]
    feats = np.column_stack(
        [np.full(h, ofeat[0]), np.full(h, ofeat[1]), np.full(h, ofeat[2]), np.full(h, ofeat[3]), j, cal_arr, pro, temp_dev[origin : origin + h]]
    )
    return np.maximum(model.predict(feats), 0) * base


def run_models(
    ctx: SeriesContext,
    origin: int,
    h: int,
    exclude_stockouts: bool,
    models: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Fit every requested model on days ``[0, origin)`` and forecast days ``[origin, origin+h)``."""
    models = models or ALL_MODELS
    y = ctx.y[:origin]
    cens = ctx.stockout[:origin] if exclude_stockouts else np.zeros(origin, bool)
    valid_full = ~(ctx.stockout if exclude_stockouts else np.zeros(len(ctx.y), bool))
    y_fit = np.where(cens, np.nan, y)
    y_imp = impute_censored(y, cens) if cens.any() else y.copy()
    out: dict[str, np.ndarray] = {}
    if "naive" in models:
        out["naive"] = forecast_naive(y_imp, h)
    if "ma28" in models:
        out["ma28"] = forecast_ma(y_fit, h)
    if "ets" in models:
        out["ets"] = forecast_ets(y_imp, h)
    if "sarimax" in models:
        d_te = ctx.dates[origin : origin + h]
        tmean = float(ctx.temp[:origin].mean())
        clim = F.temperature_climatology(ctx.dates[:origin], ctx.temp[:origin], d_te)
        Xtr = F.exog_matrix(ctx.dates[:origin], ctx.cal, ctx.promo.iloc[:origin], ctx.temp[:origin], tmean)
        Xte = F.exog_matrix(d_te, ctx.cal, ctx.promo.iloc[origin : origin + h], clim, tmean)
        out["sarimax"] = forecast_sarimax(y_fit, Xtr, Xte, h)
    if "gbm" in models:
        out["gbm"] = forecast_gbm(ctx, origin, h, valid_full)
    if "glm" in models:
        out["glm"] = forecast_glm(ctx, origin, h, valid_full)
    return out


# --------------------------------------------------------------------------------------------------
# Rolling-origin back-test
# --------------------------------------------------------------------------------------------------


def rolling_origin_backtest(
    ctx: SeriesContext,
    horizon: int = 28,
    n_folds: int = 4,
    end_idx: int | None = None,
    exclude_stockouts: bool = True,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Rolling-origin evaluation: ``n_folds`` consecutive test windows of ``horizon`` days, newest last.

    ``end_idx`` is the (exclusive) end of the last test window (default: end of data). Each fold is trained
    only on days before its own origin. Returns a long frame with one row per (fold, day, model).
    """
    n = len(ctx.y)
    end_idx = n if end_idx is None else min(end_idx, n)
    rows = []
    ev = ctx.event
    for k in range(n_folds):
        test_end = end_idx - (n_folds - 1 - k) * horizon
        origin = test_end - horizon
        if origin < 120:
            continue
        fc = run_models(ctx, origin, horizon, exclude_stockouts, models)
        sl = slice(origin, test_end)
        for m, f in fc.items():
            rows.append(
                pd.DataFrame(
                    {
                        "fold": k + 1,
                        "date": ctx.dates[sl],
                        "step": np.arange(1, horizon + 1),
                        "model": m,
                        "forecast": f,
                        "actual": ctx.y[sl],
                        "stockout": ctx.stockout[sl],
                        "event": ev[sl],
                        "origin_idx": origin,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def leaderboard(bt: pd.DataFrame, ctx: SeriesContext) -> pd.DataFrame:
    """Pooled metrics per model (scored on non-stockout days), plus the last fold only (= final holdout)."""
    first_origin = int(bt["origin_idx"].min())
    scale = seasonal_naive_scale(ctx.y[:first_origin], ~ctx.stockout[:first_origin])
    sc = bt[~bt["stockout"]]
    last = sc[sc["fold"] == sc["fold"].max()]
    rows = []
    for m, g in sc.groupby("model"):
        r = metric_row(g["actual"].values, g["forecast"].values, scale)
        gl = last[last["model"] == m]
        r["WAPE (last fold)"] = wape(gl["actual"].values, gl["forecast"].values)
        ev = g[g["event"]]
        no = g[~g["event"]]
        r["WAPE event days"] = wape(ev["actual"].values, ev["forecast"].values) if len(ev) else np.nan
        r["WAPE normal days"] = wape(no["actual"].values, no["forecast"].values) if len(no) else np.nan
        r["Bias event days"] = bias_pct(ev["actual"].values, ev["forecast"].values) if len(ev) else np.nan
        r["Bias normal days"] = bias_pct(no["actual"].values, no["forecast"].values) if len(no) else np.nan
        r["model"] = m
        rows.append(r)
    lb = pd.DataFrame(rows).set_index("model").sort_values("WAPE")
    lb.insert(0, "Model", [MODEL_LABELS[m] for m in lb.index])
    return lb


def add_intervals(bt: pd.DataFrame, level: float = 0.8) -> pd.DataFrame:
    """Add out-of-sample prediction intervals to the last fold.

    For each model the interval width comes from the empirical quantiles of the forecast errors made in
    the *earlier* folds (so the displayed fold is genuinely out-of-sample for the interval too).
    Returns the last-fold frame with ``lo``/``hi`` columns.
    """
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    kmax = bt["fold"].max()
    outs = []
    for m, g in bt.groupby("model"):
        prev = g[(g["fold"] < kmax) & (~g["stockout"])]
        cur = g[g["fold"] == kmax].copy()
        rel = (prev["actual"] - prev["forecast"]) / np.maximum(prev["forecast"], 1.0)
        if len(rel) >= 10:
            ql, qh = np.quantile(rel, [lo_q, hi_q])
        else:  # pragma: no cover
            ql, qh = stats.norm.ppf([lo_q, hi_q]) * 0.2
        cur["lo"] = np.maximum(cur["forecast"] * (1 + ql), 0)
        cur["hi"] = cur["forecast"] * (1 + qh)
        outs.append(cur)
    return pd.concat(outs, ignore_index=True)


def interval_coverage(last: pd.DataFrame) -> pd.Series:
    """Share of (non-stockout) last-fold actuals that fall inside the prediction interval, per model."""
    g = last[~last["stockout"]]
    return g.groupby("model").apply(lambda d: float(((d["actual"] >= d["lo"]) & (d["actual"] <= d["hi"])).mean()), include_groups=False)
