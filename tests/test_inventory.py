"""Unit tests: every formula is checked against a hand-calculated example.

Run with ``pytest -q`` from the project root.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import features as F  # noqa: E402
from src import forecasting as FC  # noqa: E402
from src import inventory as INV  # noqa: E402

# ------------------------------------------------------------------------------------------ inventory formulas


def test_sigma_lead_time_demand_hand_calc():
    # sqrt(4*20^2 + 100^2*1^2) = sqrt(1600 + 10000) = sqrt(11600) = 107.7033
    assert INV.sigma_lead_time_demand(100, 20, 4, 1) == pytest.approx(107.7033, abs=1e-3)


def test_sigma_without_lead_time_variability_reduces_to_sqrt_L_sigma():
    assert INV.sigma_lead_time_demand(100, 20, 4, 0) == pytest.approx(2 * 20)


def test_z_score_and_safety_stock():
    assert INV.z_score(0.95) == pytest.approx(1.6449, abs=1e-4)
    assert INV.z_score(0.90) == pytest.approx(1.2816, abs=1e-4)
    assert INV.safety_stock(1.6449, 107.7033) == pytest.approx(177.17, abs=0.01)


def test_reorder_point():
    # d*L + SS = 100*4 + 177.17
    assert INV.reorder_point(100, 4, 177.17) == pytest.approx(577.17)


def test_eoq_hand_calc_and_case_rounding():
    # sqrt(2*12000*100/2) = sqrt(1,200,000) = 1095.445 ; ceil(1095.445/24) = 46 cases = 1104
    raw, rounded = INV.eoq(12000, 100, 2, case_size=24)
    assert raw == pytest.approx(1095.445, abs=1e-3)
    assert rounded == 1104
    assert INV.eoq(12000, 100, 2, case_size=1)[1] == 1096


def test_eoq_increases_with_ordering_cost_and_decreases_with_holding():
    a = INV.eoq(1000, 100, 1)[0]
    assert INV.eoq(1000, 400, 1)[0] == pytest.approx(2 * a)
    assert INV.eoq(1000, 100, 4)[0] == pytest.approx(a / 2)


def test_order_up_to_level():
    # P = 4+3 = 7 ; S = 100*7 + 1.6449*sqrt(7*400 + 10000*1) = 700 + 1.6449*sqrt(12800)
    s = INV.order_up_to_level(100, 4, 3, 20, 1, 1.6449)
    assert s == pytest.approx(700 + 1.6449 * np.sqrt(12800), abs=1e-6)


def test_normal_loss_function_and_inverse():
    assert INV.std_normal_loss(0.0) == pytest.approx(0.398942, abs=1e-6)
    assert INV.std_normal_loss(1.0) == pytest.approx(0.083315, abs=1e-6)
    assert float(INV.inverse_std_normal_loss(0.083315)) == pytest.approx(1.0, abs=2e-3)


def test_cost_curve_shape_and_optimum():
    cc = INV.cost_curve(100, 20, 4, 1, unit_cost=1000, holding_pct=0.2, order_cost=50000, unit_margin=500, case_size=1)
    assert cc["holding_cost"].is_monotonic_increasing  # more safety stock -> more holding
    assert cc["stockout_cost"].is_monotonic_decreasing  # higher service -> fewer expected shortages
    opt = INV.optimal_service_level(cc)
    assert opt["total_cost"] == cc["total_cost"].min()
    assert 0.80 <= opt["service_level"] <= 0.999
    # total = holding + ordering + stockout
    assert np.allclose(cc["total_cost"], cc["holding_cost"] + cc["ordering_cost"] + cc["stockout_cost"])


# ------------------------------------------------------------------------------------------ ABC / XYZ


def test_abc_classification_hand_calc():
    v = pd.Series([50, 30, 10, 5, 3, 2], index=list("abcdef"))
    # cumulative share *before* each item (sorted desc): 0, .50, .80, .90, .95, .98
    assert INV.abc_classify(v).tolist() == ["A", "A", "B", "B", "C", "C"]


def test_abc_always_has_an_a_item():
    assert INV.abc_classify(pd.Series([100.0], index=["x"])).iloc[0] == "A"
    assert INV.abc_classify(pd.Series([90, 10], index=["x", "y"])).tolist() == ["A", "B"]


def test_abc_keeps_original_order():
    v = pd.Series([2, 50, 30], index=["c", "a", "b"])
    out = INV.abc_classify(v)
    assert list(out.index) == ["c", "a", "b"]
    assert out["a"] == "A"


def test_xyz_thresholds():
    cv = pd.Series([0.1, 0.49, 0.5, 1.0, 1.01], index=list("abcde"))
    assert INV.xyz_classify(cv).tolist() == ["X", "X", "Y", "Y", "Z"]


def test_coefficient_of_variation():
    # mean 5, sample std = sqrt(sum((x-5)^2)/(n-1)); x = [4,5,6] -> std 1 -> CV 0.2
    assert INV.coefficient_of_variation([4, 5, 6]) == pytest.approx(0.2)
    assert INV.coefficient_of_variation([0, 0, 0]) == 0.0


# ------------------------------------------------------------------------------------------ forecast metrics


def test_wape_hand_calc_and_zero_guard():
    y, f = np.array([100, 200, 0]), np.array([110, 180, 10])
    assert FC.wape(y, f) == pytest.approx(40 / 300)
    assert np.isnan(FC.wape(np.zeros(3), f))


def test_mape_ignores_zero_actuals():
    y, f = np.array([100, 200, 0]), np.array([110, 180, 10])
    assert FC.mape(y, f) == pytest.approx((0.10 + 0.10) / 2)


def test_bias_sign_convention():
    assert FC.bias_pct(np.array([100, 100]), np.array([110, 130])) == pytest.approx(0.2)  # over-forecast
    assert FC.bias_pct(np.array([100, 100]), np.array([90, 70])) == pytest.approx(-0.2)  # under-forecast


def test_mase_and_scale():
    y = np.arange(1, 15, dtype=float)  # lag-7 differences are all 7
    assert FC.seasonal_naive_scale(y) == pytest.approx(7.0)
    assert FC.mase(np.array([10, 12.0]), np.array([11, 10.0]), 2.0) == pytest.approx(0.75)


def test_wape_is_volume_weighted_unlike_mape():
    # tiny day with a huge relative error barely moves WAPE but dominates MAPE
    y, f = np.array([1000.0, 1.0]), np.array([1000.0, 3.0])
    assert FC.wape(y, f) == pytest.approx(2 / 1001)
    assert FC.mape(y, f) == pytest.approx(1.0)


# ------------------------------------------------------------------------------------------ forecasting building blocks


def test_naive_and_ma_forecasts():
    y = np.arange(1.0, 15.0)
    assert FC.forecast_naive(y, 10).tolist() == [8, 9, 10, 11, 12, 13, 14, 8, 9, 10]
    assert FC.forecast_ma(y, 3, window=4).tolist() == [12.5, 12.5, 12.5]


def test_ma_ignores_censored_nan():
    y = np.array([10.0, 10.0, np.nan, 10.0])
    assert FC.forecast_ma(y, 2, window=4).tolist() == [10.0, 10.0]


def test_impute_censored_is_causal_and_uses_same_weekday():
    y = np.array([10, 20, 30, 40, 50, 60, 70, 11, 21, 31, 41, 51, 61, 5], dtype=float)
    cens = np.zeros(14, bool)
    cens[13] = True  # day 13 (weekday == day 6) censored: should use day 6 only (=70) - never future data
    out = FC.impute_censored(y, cens)
    assert out[13] == 70.0
    assert np.array_equal(out[:13], y[:13])


def test_poisson_ridge_recovers_coefficients():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    y = rng.poisson(np.exp(3.0 + 0.4 * x))
    beta = FC.fit_poisson_ridge(X, y, alpha=1e-6)
    assert beta[0] == pytest.approx(3.0, abs=0.03)
    assert beta[1] == pytest.approx(0.4, abs=0.03)


def test_poisson_zero_weight_drops_observations():
    X = np.ones((6, 1))
    y = np.array([10, 10, 10, 10, 1000, 1000.0])
    w = np.array([1, 1, 1, 1, 0, 0.0])
    beta = FC.fit_poisson_ridge(X, y, w=w, alpha=0.0)
    assert np.exp(beta[0]) == pytest.approx(10.0, rel=1e-4)


def test_post_promo_flag_marks_five_days_after_a_promo():
    promo = np.zeros(20)
    promo[3:6] = 1
    pf = F.promo_features(promo)
    assert pf["post_promo"].values[6:11].tolist() == [1, 1, 1, 1, 1]
    assert pf["post_promo"].values[11] == 0
    assert pf["post_promo"].values[:6].sum() == 0  # no leakage from the future into earlier days


def test_temperature_climatology_uses_training_days_only():
    dates = pd.date_range("2023-01-01", periods=730)
    temp = np.where(dates.year == 2023, 25.0, 30.0)
    clim = F.temperature_climatology(dates[:365], temp[:365], dates[365:370])
    assert np.allclose(clim, 25.0)  # the (hotter) 2024 data is not used


def test_calendar_features_lebaran_bins_are_mutually_exclusive():
    cal = pd.read_csv(Path(__file__).resolve().parents[1] / "data_inventory" / "calendar.csv", parse_dates=["date"])
    cf = F.calendar_features(cal)
    assert (cf[F.LEB_COLS].sum(axis=1) <= 1).all()


# ------------------------------------------------------------------------------------------ Monte-Carlo


def _mc(**kw):
    p = INV.MCInputs(d_bar=100, sigma_rel=0.25, dow_factor=np.ones(7), lt_samples=np.array([3, 4, 4, 5, 8.0]), R=3, case_size=1, unit_cost=1000, unit_margin=500, holding_pct=0.2, order_cost=10000)
    return INV.simulate_mc(p, runs=1500, seed=1, **kw)


def test_mc_higher_service_level_means_fewer_lost_units_and_more_inventory():
    lo, hi = _mc(service_level=0.90), _mc(service_level=0.99)
    assert hi["lost_units_mean"] < lo["lost_units_mean"]
    assert hi["avg_inventory"] > lo["avg_inventory"]


def test_mc_shock_hurts_when_not_anticipated_and_helps_when_anticipated():
    base = _mc(service_level=0.95)
    shock = _mc(service_level=0.95, demand_surge=0.3, lt_increase=2)
    plan = _mc(service_level=0.95, demand_surge=0.3, lt_increase=2, anticipate=True)
    assert shock["lost_units_mean"] > base["lost_units_mean"]
    assert plan["lost_units_mean"] < shock["lost_units_mean"]


def test_mc_is_reproducible_and_vectorised_over_runs():
    a, b = _mc(service_level=0.95), _mc(service_level=0.95)
    assert a["lost_units_mean"] == b["lost_units_mean"]
    assert len(a["lost_units_runs"]) == 1500
    assert a["lost_units_p90"] >= a["lost_units_p50"]


def test_mc_percentiles_and_probability_bounds():
    r = _mc(service_level=0.95)
    assert 0 <= r["stockout_prob"] <= 1
    assert 0 <= r["fill_rate"] <= 1
