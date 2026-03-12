"""Monte Carlo portfolio simulation for category-based prediction markets.

Simulates portfolio P&L by independently sampling winners per category
from probability distributions, looking up per-winner P&L, and summing
across categories.
"""

from typing import NamedTuple

import numpy as np


class CategoryScenario(NamedTuple):
    """One category's winner outcomes with probabilities and P&L values.

    Each index i corresponds to one possible winner:
    - winners[i]: the winner name/label
    - probs[i]: probability of that winner (must sum to 1)
    - pnls[i]: P&L in dollars if that winner wins
    """

    winners: np.ndarray  # shape (n_outcomes,) — outcome labels
    probs: np.ndarray  # shape (n_outcomes,) — must sum to 1
    pnls: np.ndarray  # shape (n_outcomes,) — P&L per outcome


def sample_portfolio_pnl(
    category_scenarios: dict[str, CategoryScenario],
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Monte Carlo portfolio P&L over mutually exclusive category outcomes.

    For each sample, independently draws one winner per category from the
    probability distribution, looks up the corresponding P&L, and sums across
    categories. Categories are assumed independent (each has its own winner).

    Args:
        category_scenarios: Map from category name to scenario data.
        n_samples: Number of Monte Carlo draws.
        rng: NumPy random generator for reproducibility.

    Returns:
        Array of shape (n_samples,) with total portfolio P&L per draw.
    """
    total_pnls = np.zeros(n_samples)
    for _cat, scenario in category_scenarios.items():
        indices = rng.choice(len(scenario.winners), size=n_samples, p=scenario.probs)
        total_pnls += scenario.pnls[indices]
    return total_pnls


def compute_cvar(pnls: np.ndarray, alpha: float) -> float:
    """Conditional Value at Risk — mean of worst alpha-fraction of outcomes.

    CVaR_α is the expected loss given that we're in the worst α-percentile of outcomes.
    For α=0.05, this is the expected P&L in the worst 5% of scenarios.

    Args:
        pnls: Array of P&L values.
        alpha: Tail fraction (e.g., 0.05 for 5% CVaR). Must be in (0, 1].
            If alpha is 0, returns the minimum P&L (worst case).

    Returns:
        Mean P&L of the worst alpha-fraction of outcomes.
    """
    if alpha <= 0:
        return float(np.min(pnls))
    n_tail = max(1, int(np.floor(alpha * len(pnls))))
    sorted_pnls = np.sort(pnls)
    return float(np.mean(sorted_pnls[:n_tail]))


def compute_portfolio_mc_metrics(
    pnls: np.ndarray,
    bankroll: float,
    capital_deployed: float | None = None,
) -> dict[str, float]:
    """Compute full set of MC metrics from sampled portfolio P&L.

    Args:
        pnls: Array of portfolio P&L samples (from sample_portfolio_pnl).
        bankroll: Total bankroll (e.g., $1000 per category × N categories).
        capital_deployed: Actual capital deployed (for ROIC). If None, uses bankroll.

    Returns:
        Dict with keys: mean_pnl, median_pnl, std_pnl, min_pnl, max_pnl,
        cvar_05, cvar_10, prob_profit, prob_loss_10pct_bankroll,
        prob_loss_20pct_bankroll, total_capital_deployed,
        pct_bankroll_deployed, expected_roic.
    """
    capital = capital_deployed if capital_deployed is not None else bankroll
    return {
        "mean_pnl": float(np.mean(pnls)),
        "median_pnl": float(np.median(pnls)),
        "std_pnl": float(np.std(pnls)),
        "min_pnl": float(np.min(pnls)),
        "max_pnl": float(np.max(pnls)),
        "cvar_05": compute_cvar(pnls, 0.05),
        "cvar_10": compute_cvar(pnls, 0.10),
        "prob_profit": float(np.mean(pnls > 0)),
        "prob_loss_10pct_bankroll": float(np.mean(pnls < -0.10 * bankroll)),
        "prob_loss_20pct_bankroll": float(np.mean(pnls < -0.20 * bankroll)),
        "total_capital_deployed": capital,
        "pct_bankroll_deployed": capital / bankroll if bankroll > 0 else 0.0,
        "expected_roic": float(np.mean(pnls)) / capital if capital > 0 else 0.0,
    }
