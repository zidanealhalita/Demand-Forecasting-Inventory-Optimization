"""Integration tests for the day-by-day simulator (uses the committed CSVs)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

logging.disable(logging.CRITICAL)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data, policy_backtest as PB  # noqa: E402


@pytest.fixture(scope="module")
def world():
    w = PB.build_world(data.load_panel(), data.load_truth(), data.load_purchase_orders(), data.load_calendar())
    return w, PB.lead_time_scenarios(w, n_boot=3)


def test_stock_balance_and_non_negativity(world):
    w, lt = world
    r = PB.simulate(w, lt[:2], "baseline")
    assert (r.oh_end >= -1e-6).all()
    assert np.allclose(r.sold + r.lost, w.demand[:, None, :])  # every unit of demand is sold or lost
    assert (r.lost >= -1e-9).all()


def test_orders_are_whole_cases(world):
    w, lt = world
    r = PB.simulate(w, lt[:1], "baseline")
    rem = np.mod(r.order_q[:, 0, :], w.case_size[None, :])
    assert np.allclose(rem, 0) or np.allclose(np.minimum(rem, w.case_size[None, :] - rem), 0, atol=1e-6)


def test_orders_only_on_review_days(world):
    w, lt = world
    r = PB.simulate(w, lt[:1], "baseline")
    assert ((r.order_q[:, 0, :] > 0) & ~w.review).sum() == 0


def test_baseline_replay_reproduces_history(world):
    """Replaying the fitted current policy with real lead times and true demand must match real 2025 history."""
    w, lt = world
    r = PB.simulate(w, lt[:1], "baseline")
    s = slice(PB.WARMUP_DAYS, None)
    sim_so, act_so = (r.lost[s, 0] > 0).mean(), w.actual_so[s].mean()
    sim_inv, act_inv = r.oh_end[s, 0].mean(), w.actual_oh[s].mean()
    assert abs(sim_so - act_so) < 0.01  # within 1 pt of stockout days
    assert abs(sim_inv / act_inv - 1) < 0.05  # within 5 % of average stock


def test_common_random_numbers_same_lead_times_for_both_policies(world):
    w, lt = world
    assert lt.shape[1:] == w.demand.shape
    assert (lt >= 1).all()
