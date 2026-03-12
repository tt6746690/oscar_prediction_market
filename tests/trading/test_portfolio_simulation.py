"""Tests for portfolio_simulation — MC sampling, CVaR, metrics."""

import numpy as np

from oscar_prediction_market.one_offs.d20260225_buy_hold_backtest.portfolio_simulation import (
    CategoryScenario,
    compute_cvar,
    compute_portfolio_mc_metrics,
    sample_portfolio_pnl,
)


def test_sample_portfolio_pnl_deterministic():
    """With deterministic seed, MC samples are reproducible."""
    scenarios = {
        "cat_a": CategoryScenario(
            winners=np.array(["W1", "W2"]),
            probs=np.array([0.6, 0.4]),
            pnls=np.array([100.0, -50.0]),
        ),
    }
    rng = np.random.default_rng(42)
    pnls = sample_portfolio_pnl(scenarios, n_samples=10000, rng=rng)

    assert pnls.shape == (10000,)
    # With 60% chance of +100 and 40% chance of -50, EV = 40
    assert abs(np.mean(pnls) - 40.0) < 5.0  # within ~$5 of EV


def test_sample_portfolio_pnl_two_categories():
    """Portfolio sums P&L across independent categories.

    Category A: always wins (+100)
    Category B: always loses (-30)
    Portfolio should always have +70.
    """
    scenarios = {
        "cat_a": CategoryScenario(
            winners=np.array(["sure_win"]),
            probs=np.array([1.0]),
            pnls=np.array([100.0]),
        ),
        "cat_b": CategoryScenario(
            winners=np.array(["sure_lose"]),
            probs=np.array([1.0]),
            pnls=np.array([-30.0]),
        ),
    }
    rng = np.random.default_rng(42)
    pnls = sample_portfolio_pnl(scenarios, n_samples=100, rng=rng)

    np.testing.assert_array_almost_equal(pnls, 70.0)


def test_compute_cvar_worst_case():
    """CVaR at alpha=0 is the worst single outcome (minimum)."""
    pnls = np.array([-100.0, -50.0, 0.0, 50.0, 100.0])
    assert compute_cvar(pnls, alpha=0.0) == -100.0


def test_compute_cvar_5pct():
    """CVaR at 5% is the mean of the worst 5% of outcomes.

    With 100 samples sorted, worst 5 = first 5 samples.
    """
    pnls = np.arange(100.0)  # 0, 1, 2, ..., 99
    cvar = compute_cvar(pnls, alpha=0.05)
    # Worst 5%: {0, 1, 2, 3, 4} → mean = 2.0
    assert abs(cvar - 2.0) < 0.01


def test_compute_portfolio_mc_metrics():
    """Smoke test that all expected metric keys are returned."""
    pnls = np.array([10.0, -20.0, 30.0, -5.0, 15.0])
    metrics = compute_portfolio_mc_metrics(pnls, bankroll=1000.0)

    expected_keys = {
        "mean_pnl",
        "median_pnl",
        "std_pnl",
        "min_pnl",
        "max_pnl",
        "cvar_05",
        "cvar_10",
        "prob_profit",
        "prob_loss_10pct_bankroll",
        "prob_loss_20pct_bankroll",
        "total_capital_deployed",
        "pct_bankroll_deployed",
        "expected_roic",
    }
    assert set(metrics.keys()) == expected_keys
    assert metrics["mean_pnl"] == np.mean(pnls)
    assert metrics["prob_profit"] == 0.6  # 3 out of 5 are positive
