"""Calendar, promo, temperature and lag features.

Everything here is *causal*: features for day ``t`` only use information that is known before ``t``
(the calendar and the promo plan are known in advance; temperature for the future is replaced by a
climatology computed on the training period; lag features are built from the forecast origin).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Lebaran window bins on ``days_to_lebaran`` (negative = days before Lebaran).
# Derived from the demand shape: run-up from ~30 days before (Ramadan), peak ~5 days before, dip afterwards.
LEB_BINS: dict[str, tuple[int, int]] = {
    "leb_far": (-30, -11),
    "leb_run": (-10, -6),
    "leb_peak": (-5, -3),
    "leb_pre": (-2, -1),
    "leb_post1": (0, 7),
    "leb_post2": (8, 14),
}
LEB_COLS = list(LEB_BINS)
POST_PROMO_DAYS = 5


def calendar_features(cal: pd.DataFrame) -> pd.DataFrame:
    """Build the date-indexed calendar feature frame used by every model.

    Columns: ``dow`` dummies (``dow_1..dow_6``), ``is_weekend``, ``payday``, ``holiday``, ``year_end``,
    the six Lebaran bins, and two annual Fourier pairs (``s1,c1,s2,c2``).
    """
    c = cal.copy()
    c["date"] = pd.to_datetime(c["date"])
    c = c.set_index("date").sort_index()
    out = pd.DataFrame(index=c.index)
    for k in range(1, 7):
        out[f"dow_{k}"] = (c["dow"] == k).astype(float)
    out["is_weekend"] = c["is_weekend"].astype(float)
    out["payday"] = c["is_payday_window"].astype(float)
    out["holiday"] = c["is_public_holiday"].astype(float)
    out["year_end"] = c["is_year_end_window"].astype(float)
    d = c["days_to_lebaran"]
    for name, (lo, hi) in LEB_BINS.items():
        out[name] = d.between(lo, hi).astype(float)
    doy = c.index.dayofyear.values
    for k in (1, 2):
        out[f"s{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        out[f"c{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    out["in_leb_window"] = d.between(-30, 14).astype(float)
    return out


def promo_features(promo: pd.Series | np.ndarray) -> pd.DataFrame:
    """Promo flag plus a *post-promo* flag (1-5 days after a promo spell ended = pull-forward dip)."""
    p = pd.Series(np.asarray(promo, dtype=float))
    post = np.zeros(len(p))
    # a day is "post-promo" when it is not promo but a promo day occurred within the previous 5 days
    recent = p.rolling(POST_PROMO_DAYS, min_periods=1).max().shift(1).fillna(0.0)
    post = ((p == 0) & (recent > 0)).astype(float).values
    idx = promo.index if isinstance(promo, pd.Series) else None
    out = pd.DataFrame({"promo": p.values, "post_promo": post})
    if idx is not None:
        out.index = idx
    return out


def event_day_flag(cal_feats: pd.DataFrame, promo_feats: pd.DataFrame) -> np.ndarray:
    """Boolean array: 'event day' = promo / post-promo / Ramadan-Lebaran window / public holiday / year-end.

    Used to report forecast error on event days vs normal days separately.
    """
    f = (
        (promo_feats["promo"].values > 0)
        | (promo_feats["post_promo"].values > 0)
        | (cal_feats["in_leb_window"].values > 0)
        | (cal_feats["holiday"].values > 0)
        | (cal_feats["year_end"].values > 0)
    )
    return f


def temperature_climatology(
    train_dates: pd.DatetimeIndex, train_temp: np.ndarray, target_dates: pd.DatetimeIndex, smooth: int = 15
) -> np.ndarray:
    """Day-of-year mean temperature learned on the training window, circularly smoothed.

    Used as the *expected* temperature for forecast days (the real future temperature is unknown).
    """
    doy = np.minimum(train_dates.dayofyear.values, 365) - 1
    sums = np.bincount(doy, weights=train_temp, minlength=365)
    cnt = np.bincount(doy, minlength=365)
    clim = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)
    if np.isnan(clim).any():  # fewer than one year of data: fill gaps by circular interpolation
        s = pd.Series(clim)
        s = pd.concat([s, s, s]).interpolate(limit_direction="both").iloc[365:730].values
        clim = s
    k = np.ones(smooth) / smooth
    ext = np.concatenate([clim[-smooth:], clim, clim[:smooth]])
    sm = np.convolve(ext, k, mode="same")[smooth:-smooth]
    tdoy = np.minimum(target_dates.dayofyear.values, 365) - 1
    return sm[tdoy]


def design_matrix(
    dates: pd.DatetimeIndex,
    cal_feats: pd.DataFrame,
    promo_feats: pd.DataFrame,
    temp: np.ndarray,
    temp_mean: float,
    t0: pd.Timestamp,
) -> pd.DataFrame:
    """Full regression design (intercept first) for the Poisson-calendar model."""
    cf = cal_feats.loc[dates]
    X = pd.DataFrame(index=dates)
    X["intercept"] = 1.0
    X["trend"] = (dates - t0).days.values / 365.25
    for c in [f"dow_{k}" for k in range(1, 7)] + ["payday", "holiday", "year_end"] + LEB_COLS + ["s1", "c1", "s2", "c2"]:
        X[c] = cf[c].values
    X["promo"] = promo_feats["promo"].values
    X["post_promo"] = promo_feats["post_promo"].values
    X["temp_dev"] = np.asarray(temp, dtype=float) - temp_mean
    return X


EXOG_COLS = ["promo", "post_promo", "payday", "is_weekend", "temp_dev"] + LEB_COLS


def exog_matrix(
    dates: pd.DatetimeIndex,
    cal_feats: pd.DataFrame,
    promo_feats: pd.DataFrame,
    temp: np.ndarray,
    temp_mean: float,
) -> pd.DataFrame:
    """Exogenous regressors for SARIMAX: promo, payday, Ramadan/Lebaran bins, temperature (+weekend)."""
    cf = cal_feats.loc[dates]
    X = pd.DataFrame(index=dates)
    X["promo"] = promo_feats["promo"].values
    X["post_promo"] = promo_feats["post_promo"].values
    X["payday"] = cf["payday"].values
    X["is_weekend"] = cf["is_weekend"].values
    X["temp_dev"] = np.asarray(temp, dtype=float) - temp_mean
    for c in LEB_COLS:
        X[c] = cf[c].values
    return X[EXOG_COLS]


def gbm_training_set(
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    cal_feats: pd.DataFrame,
    promo_feats: pd.DataFrame,
    temp_dev: np.ndarray,
    valid: np.ndarray,
    horizon: int,
    origin_step: int = 7,
    min_history: int = 56,
) -> tuple[np.ndarray, np.ndarray]:
    """Direct multi-horizon training set for gradient boosting (no leakage).

    For every forecast origin ``o`` (the last observed day) and step ``j`` in 1..horizon, features are:
    calendar/promo/temperature of the *target* day, the step ``j``, and *origin* statistics
    (7/28-day means, same-weekday mean of the last 4 weeks, relative trend) computed only from
    data up to ``o``. The target is ``y[o+j]``. Rows whose target day is censored (``valid`` False) are dropped.
    """
    n = len(y)
    yy = np.where(valid, y, np.nan)
    cs = pd.Series(yy)
    m7 = cs.rolling(7, min_periods=3).mean().values
    m28 = cs.rolling(28, min_periods=10).mean().values
    m56 = cs.rolling(56, min_periods=20).mean().values
    cf = cal_feats.loc[dates]
    cal_cols = [f"dow_{k}" for k in range(1, 7)] + ["payday", "holiday", "year_end"] + LEB_COLS
    cal_arr = cf[cal_cols].values
    pro = promo_feats[["promo", "post_promo"]].values
    X, Y = [], []
    for o in range(min_history, n - 1, origin_step):
        hmax = min(horizon, n - 1 - o)
        if hmax < 1:
            continue
        base = m28[o] if np.isfinite(m28[o]) else np.nanmean(yy[: o + 1])
        if not np.isfinite(base) or base <= 0:
            continue
        origin_feats = _origin_features(yy, o, m7, m28, m56, base)
        j = np.arange(1, hmax + 1)
        tgt = o + j
        ok = valid[tgt]
        if not ok.any():
            continue
        feats = np.column_stack(
            [
                np.full(hmax, origin_feats[0]),
                np.full(hmax, origin_feats[1]),
                np.full(hmax, origin_feats[2]),
                np.full(hmax, origin_feats[3]),
                j,
                cal_arr[tgt],
                pro[tgt],
                temp_dev[tgt],
            ]
        )
        X.append(feats[ok])
        Y.append((y[tgt] / base)[ok])  # model the ratio to the 28d level -> scale-free
    return np.vstack(X), np.concatenate(Y)


def _origin_features(yy: np.ndarray, o: int, m7, m28, m56, base: float) -> tuple[float, float, float, float]:
    """Origin-only statistics (all use data up to and including index ``o``)."""
    r7 = m7[o] / base if np.isfinite(m7[o]) else 1.0
    r56 = m56[o] / base if np.isfinite(m56[o]) else 1.0
    wk = yy[max(0, o - 27) : o + 1]
    # same-weekday mean of the last 4 weeks relative to level, for the weekday of the target is handled by dow dummies;
    # here we keep the dispersion of the last 28 days as a volatility proxy
    vol = np.nanstd(wk) / base if np.isfinite(np.nanstd(wk)) else 0.2
    slope = (m28[o] - m28[max(0, o - 28)]) / base if np.isfinite(m28[max(0, o - 28)]) else 0.0
    return r7, r56, vol, slope


def gbm_origin_row(
    y: np.ndarray,
    valid: np.ndarray,
    o: int,
) -> tuple[tuple[float, float, float, float], float]:
    """Origin statistics and the 28-day level at index ``o`` (used at prediction time)."""
    yy = np.where(valid, y, np.nan)
    cs = pd.Series(yy)
    m7 = cs.rolling(7, min_periods=3).mean().values
    m28 = cs.rolling(28, min_periods=10).mean().values
    m56 = cs.rolling(56, min_periods=20).mean().values
    base = m28[o] if np.isfinite(m28[o]) else np.nanmean(yy[: o + 1])
    return _origin_features(yy, o, m7, m28, m56, base), float(base)
