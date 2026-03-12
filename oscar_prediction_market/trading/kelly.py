"""Kelly criterion position sizing for binary option portfolios.

The Kelly criterion answers: "given an edge, how much should I bet?" It
maximizes long-run geometric growth of capital. Bet too little and you
leave money on the table; bet too much and a losing streak wipes you out.

For a binary option (pays $1 if correct, $0 otherwise) at price ``p``
with model probability ``q``, the Kelly fraction is::

    f* = (q - p) / (1 - p)

This is the fraction of bankroll to wager. In practice, **fractional Kelly**
(e.g., 0.25x) is standard because:

1. The model's probability estimates are noisy -- full Kelly is optimal only
   if the model is perfectly calibrated.
2. Fractional Kelly dramatically reduces variance while sacrificing only a
   small amount of expected growth (growth scales as f - f^2/2).

This module supports two modes:

- **Independent Kelly**: sizes each outcome separately, ignoring correlations.
  Simple and fast, but oversizes when outcomes are mutually exclusive.

- **Multi-outcome Kelly**: jointly optimizes across all outcomes by maximizing
  expected log-wealth, subject to the constraint that exactly one outcome wins.
  This correctly accounts for the "portfolio effect" -- buying outcome A
  implicitly hedges against outcome B.

Both functions return ``list[KellyAllocation]`` — one allocation per input edge.

Usage::

    from oscar_prediction_market.trading.kelly import (
        independent_kelly,
        multi_outcome_kelly,
    )
    from oscar_prediction_market.trading.schema import KellyConfig

    config = KellyConfig(
        bankroll=1000, kelly_fraction=0.25, kelly_mode="multi_outcome",
        buy_edge_threshold=0.05, max_position_per_outcome=250,
        max_total_exposure=500,
    )
    edges = [...]  # list[Edge] from edge.py

    allocations = independent_kelly(edges, config)
    allocations = multi_outcome_kelly(edges, config)
"""

import math
from typing import Any

import numpy as np
from scipy.optimize import minimize

from oscar_prediction_market.trading.edge import Edge
from oscar_prediction_market.trading.schema import (
    KellyAllocation,
    KellyConfig,
    PositionDirection,
)

# ============================================================================
# Independent Kelly
# ============================================================================


def _kelly_fraction_binary(model_prob: float, implied_prob: float) -> float:
    """Raw Kelly fraction for a binary bet.

    Derivation: you bet fraction f of bankroll W on a binary contract at
    price p (= implied_prob). If you win (prob q = model_prob), your
    wealth becomes W + f*W*(1-p)/p. If you lose, wealth becomes W - f*W.

    Maximizing E[log(W')] = q*log(1 + f*(1-p)/p) + (1-q)*log(1-f) gives::

        f* = (q - p) / (1 - p)

    This is the fraction of bankroll to risk. Note:

    - f* = 0 when q = p (no edge)
    - f* = 1 when q = 1 (certain win)
    - f* < 0 when q < p (negative edge -- don't bet)
    """
    if implied_prob >= 1.0 or implied_prob <= 0.0:
        return 0.0
    f = (model_prob - implied_prob) / (1 - implied_prob)
    return max(0.0, f)


def _zero_allocation(edge: Edge) -> KellyAllocation:
    """Create a zero-contract allocation (ineligible or below threshold)."""
    return KellyAllocation(
        outcome=edge.outcome,
        direction=edge.direction,
        model_prob=edge.model_prob,
        execution_price=edge.execution_price,
        net_edge=edge.net_edge,
        recommended_contracts=0,
    )


def _cap_total_exposure(
    allocations: list[KellyAllocation],
    max_total_exposure: float,
) -> list[KellyAllocation]:
    """Scale down contract counts so total cost (incl. fees) stays within budget.

    After individual Kelly sizing, the total cost may exceed
    ``max_total_exposure``. This function proportionally scales down all
    active positions. Positions that scale to zero contracts are
    filtered out (zeroed).

    Total cost includes fees: each contract costs ``execution_price + fee``.

    Zero-contract allocations are passed through unchanged.

    Args:
        allocations: Per-outcome allocations (may exceed budget).
        max_total_exposure: Maximum total USD outlay allowed.

    Returns:
        New list of allocations with scaled-down contract counts.
    """
    total = sum(a.recommended_contracts * (a.execution_price + a.fee) for a in allocations)
    if total <= max_total_exposure:
        return allocations
    scale = max_total_exposure / total
    return [
        a.model_copy(update={"recommended_contracts": math.floor(a.recommended_contracts * scale)})
        if a.recommended_contracts > 0
        else a
        for a in allocations
    ]


def independent_kelly(
    edges: list[Edge],
    config: KellyConfig,
) -> list[KellyAllocation]:
    """Size positions independently for each outcome using Kelly criterion.

    Treats each outcome as a standalone binary bet and applies Kelly
    independently. This is conceptually simple and works well when:

    - You're betting on a small number of outcomes
    - The outcomes are nearly independent
    - You want a quick, conservative estimate

    Supports both YES and NO edges. For NO edges, ``model_prob`` is the
    probability the nominee loses and ``execution_price`` is the NO
    price. The Kelly formula is the same — it's direction-agnostic.

    Limitation: in a mutually exclusive market (exactly one winner), bets
    are anti-correlated -- if you buy A and B, one must lose. Independent
    Kelly ignores this and tends to oversize the total portfolio. The
    ``multi_outcome_kelly()`` function handles this correctly.

    After sizing, positions are capped at per-outcome and total exposure
    limits from config.

    Args:
        edges: Edge results for each outcome. May include both YES and NO edges.
        config: Kelly configuration parameters.

    Returns:
        List of KellyAllocation, one per input edge (zero contracts if ineligible).
    """
    allocations: list[KellyAllocation] = []

    for edge in edges:
        if edge.net_edge < config.buy_edge_threshold:
            allocations.append(_zero_allocation(edge))
            continue

        # Net edge Kelly: use implied prob that includes fees
        implied_prob_with_fee = edge.implied_prob + edge.fee
        kelly_raw = _kelly_fraction_binary(edge.model_prob, implied_prob_with_fee)
        kelly_applied = kelly_raw * config.kelly_fraction

        # Convert to contracts
        price_dollars = edge.execution_price
        if price_dollars <= 0:
            allocations.append(_zero_allocation(edge))
            continue

        raw_contracts = kelly_applied * config.bankroll / price_dollars

        # Cap at per-outcome max (including fee cost per contract)
        unit_cost = price_dollars + edge.fee
        max_contracts_by_cap = config.max_position_per_outcome / unit_cost
        n_contracts = min(math.floor(raw_contracts), math.floor(max_contracts_by_cap))
        n_contracts = max(0, n_contracts)

        allocations.append(
            KellyAllocation(
                outcome=edge.outcome,
                direction=edge.direction,
                model_prob=edge.model_prob,
                execution_price=edge.execution_price,
                net_edge=edge.net_edge,
                recommended_contracts=n_contracts,
            )
        )

    # Cap total exposure
    allocations = _cap_total_exposure(allocations, config.max_total_exposure)

    return allocations


# ============================================================================
# Multi-Outcome Kelly (Optimization)
# ============================================================================


def multi_outcome_kelly(
    edges: list[Edge],
    config: KellyConfig,
) -> list[KellyAllocation]:
    """Jointly optimize position sizes across mutually exclusive outcomes.

    In the Oscar Best Picture market, exactly one outcome wins and the rest
    lose. If we hold contracts on outcomes A, B, and C:

    - If A wins: we profit on A but lose our entire outlay on B and C
    - If nobody we bet on wins: we lose everything

    Independent Kelly ignores this structure. Multi-outcome Kelly maximizes
    expected log-wealth across ALL scenarios::

        E[log(W)] = sum_i  q_i * log(W_if_i_wins)
                  + q_none  * log(W_if_none_win)

    **YES+NO joint optimization**: This function handles both YES and NO
    positions. When outcome k wins:

    - YES on k pays $1 per contract.
    - NO on j (for j != k) pays $1 per contract (those nominees lost).
    - NO on k expires worthless (that nominee won).

    When no bet-on outcome wins:

    - All YES positions expire.
    - All NO positions pay $1 per contract.

    This correctly captures the hedging benefit of holding YES on A and
    NO on B: if A wins, both profit (YES on A pays, NO on B pays because
    B lost). The optimizer naturally balances YES/NO exposure because
    over-allocating to NO creates drag (paying for certainty) while
    under-allocating misses the hedge.

    The optimizer also implicitly enforces position netting: it's never
    optimal to hold both YES and NO on the same outcome (the combined
    cost exceeds $1 after fees, guaranteeing a loss). Post-optimization,
    we verify this and zero out any violating positions as a safety check.

    This is solved numerically via scipy SLSQP (Sequential Least Squares
    Programming), subject to constraints:

    1. Total cost <= max_total_exposure
    2. Total cost < 99% of bankroll (solvency buffer)
    3. 0 <= contracts_per_outcome <= max by position cap

    The optimizer starts from independent Kelly fractions (scaled down) as
    an initial guess, which helps convergence. If optimization fails, it
    falls back to those independent estimates.

    Args:
        edges: Edge results for each outcome. May include both YES and NO
            edges for the same outcome.
        config: Kelly configuration.

    Returns:
        List of KellyAllocation, one per input edge (zero contracts if ineligible).
    """
    # Filter to outcomes with positive edge
    eligible = [e for e in edges if e.net_edge >= config.buy_edge_threshold]

    if not eligible:
        return [_zero_allocation(e) for e in edges]

    n = len(eligible)
    directions = [e.direction for e in eligible]
    outcomes = [e.outcome for e in eligible]
    prices = np.array([e.execution_price for e in eligible])
    fees = np.array([e.fee for e in eligible])

    # All-in cost per contract: price + fee. Fee is constant at a given price
    # level (Kalshi's ceil(rate × P × (1-P)) depends only on price, not quantity
    # per contract). Precomputed to avoid redundant calculation in the objective
    # function, total-cost constraint, and solvency constraint.
    unit_cost = prices + fees

    # Build a map from outcome name to win probability.
    # For YES edges, model_prob is the win prob directly.
    # For NO edges, model_prob is (1 - win_prob), so win_prob = 1 - model_prob.
    outcome_win_probs: dict[str, float] = {}
    for e in eligible:
        if e.direction == PositionDirection.YES:
            outcome_win_probs[e.outcome] = e.model_prob
        elif e.outcome not in outcome_win_probs:
            outcome_win_probs[e.outcome] = 1.0 - e.model_prob

    # Probability that none of the eligible outcomes win
    prob_none = 1.0 - sum(outcome_win_probs.values())
    prob_none = max(prob_none, 0.01)

    w0 = config.bankroll

    _Float1D = np.ndarray[tuple[int], np.dtype[np.float64]]

    def neg_expected_log_wealth(n_contracts: _Float1D, /, *_: Any, **__: Any) -> float:
        """Negative expected log-wealth (for minimization).

        Handles mixed YES/NO portfolios. For each scenario (outcome k wins),
        YES on k and NO on all j != k pay $1. When no outcome wins, all
        NO positions pay $1.
        """
        # Total cost including fees (unit_cost precomputed above)
        total_cost = (unit_cost * n_contracts).sum()

        if total_cost >= w0:
            return 1e10  # infeasible — would bankrupt

        # Scenario: none of our outcomes win → all NO pay $1, all YES expire
        no_payouts_total = sum(
            n_contracts[i] * 1.0 for i in range(n) if directions[i] == PositionDirection.NO
        )
        wealth_if_none_win = w0 - total_cost + no_payouts_total
        if wealth_if_none_win <= 0:
            return 1e10

        log_wealth = prob_none * np.log(wealth_if_none_win)

        # Scenario: outcome k wins
        for outcome_name, win_prob in outcome_win_probs.items():
            if win_prob <= 0:
                continue
            payout = 0.0
            for i in range(n):
                if directions[i] == PositionDirection.YES and outcomes[i] == outcome_name:
                    # YES on winner pays $1 per contract (gross payout)
                    payout += n_contracts[i] * 1.0
                elif directions[i] == PositionDirection.NO and outcomes[i] != outcome_name:
                    # NO on losers pays $1 per contract (gross payout)
                    payout += n_contracts[i] * 1.0
                # YES on loser: expires, already in total_cost
                # NO on winner: expires, already in total_cost

            wealth_if_k_wins = w0 - total_cost + payout
            if wealth_if_k_wins <= 0:
                return 1e10
            log_wealth += win_prob * np.log(wealth_if_k_wins)

        return -log_wealth

    # Constraints
    constraints: list[dict[str, Any]] = []

    # Total cost <= max_total_exposure
    def total_cost_constraint(n_contracts: _Float1D) -> float:
        return config.max_total_exposure - (unit_cost * n_contracts).sum()

    constraints.append({"type": "ineq", "fun": total_cost_constraint})

    # Total cost < bankroll (solvency)
    def solvency_constraint(n_contracts: _Float1D) -> float:
        return w0 * 0.99 - (unit_cost * n_contracts).sum()  # Keep 1% buffer

    constraints.append({"type": "ineq", "fun": solvency_constraint})

    # Bounds: 0 <= n_i <= max_contracts_per_outcome
    bounds = []
    for e in eligible:
        price_dollars = e.execution_price
        if price_dollars > 0:
            max_by_cap = config.max_position_per_outcome / price_dollars
        else:
            max_by_cap = 0.0
        bounds.append((0.0, max_by_cap))

    # Initial guess: independent Kelly fractions (scaled down)
    x0 = np.zeros(n)
    for i, e in enumerate(eligible):
        implied_with_fee = e.implied_prob + e.fee
        f_raw = _kelly_fraction_binary(e.model_prob, implied_with_fee)
        f_applied = f_raw * config.kelly_fraction
        price_dollars = e.execution_price
        if price_dollars > 0:
            x0[i] = f_applied * w0 / price_dollars
        x0[i] = min(x0[i], bounds[i][1])

    result = minimize(
        neg_expected_log_wealth,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-10},
    )  # type: ignore[call-overload]

    optimal_n = result.x if result.success else x0

    # Safety: enforce position netting (no YES+NO on same outcome)
    # The optimizer shouldn't produce this (it's suboptimal after fees),
    # but enforce as a safety check.
    _enforce_position_netting(optimal_n, outcomes, directions, prices)

    # Build allocations
    eligible_set = {(e.outcome, e.direction) for e in eligible}
    final_allocations: list[KellyAllocation] = []

    eligible_idx = 0
    for edge in edges:
        if (edge.outcome, edge.direction) in eligible_set:
            raw_n = optimal_n[eligible_idx]
            n_contracts_int = max(0, math.floor(raw_n))

            final_allocations.append(
                KellyAllocation(
                    outcome=edge.outcome,
                    direction=edge.direction,
                    model_prob=edge.model_prob,
                    execution_price=edge.execution_price,
                    net_edge=edge.net_edge,
                    recommended_contracts=n_contracts_int,
                )
            )
            eligible_idx += 1
        else:
            final_allocations.append(_zero_allocation(edge))

    return final_allocations


def _enforce_position_netting(
    n_contracts: np.ndarray[tuple[int], np.dtype[np.float64]],
    outcomes: list[str],
    directions: list[PositionDirection],
    prices: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> None:
    """Zero out the smaller side if both YES and NO exist for the same outcome.

    Mutates ``n_contracts`` in place. The optimizer should never produce both
    YES and NO on the same outcome (it's always -EV after fees), but this
    safety check prevents pathological edge cases.
    """
    outcome_indices: dict[str, dict[PositionDirection, int]] = {}
    for i, (outcome, direction) in enumerate(zip(outcomes, directions, strict=True)):
        outcome_indices.setdefault(outcome, {})[direction] = i

    for _outcome, idx_by_dir in outcome_indices.items():
        if PositionDirection.YES in idx_by_dir and PositionDirection.NO in idx_by_dir:
            yes_idx = idx_by_dir[PositionDirection.YES]
            no_idx = idx_by_dir[PositionDirection.NO]
            yes_value = n_contracts[yes_idx] * prices[yes_idx]
            no_value = n_contracts[no_idx] * prices[no_idx]
            # Keep the side with higher notional value, zero the other
            if yes_value >= no_value:
                n_contracts[no_idx] = 0.0
            else:
                n_contracts[yes_idx] = 0.0
