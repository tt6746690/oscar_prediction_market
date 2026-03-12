"""Tests for Kelly criterion sizing.

Kelly answers: "Given an edge, what fraction of my bankroll should I bet?"

For binary bet at price p (implied prob) with true prob q:
  f* = (q - p) / (1 - p)

Example: q=0.30, p=0.20 -> f* = 0.10/0.80 = 0.125 (12.5% of bankroll)

In practice, fractional Kelly (0.25x) is used because:
- Model probabilities are noisy estimates, not ground truth
- Growth scales as f - f²/2, so 0.25x Kelly gives 44% of optimal growth
  but far less variance
"""

import pytest

from oscar_prediction_market.trading.edge import (
    Edge,
)
from oscar_prediction_market.trading.kelly import (
    _kelly_fraction_binary,
    independent_kelly,
    multi_outcome_kelly,
)
from oscar_prediction_market.trading.schema import (
    FeeType,
    KellyConfig,
    KellyMode,
    PositionDirection,
)

# ============================================================================
# Section 4: Kelly Criterion
#
# Kelly answers: "Given an edge, what fraction of my bankroll should I bet?"
#
# For binary bet at price p (implied prob) with true prob q:
#   f* = (q - p) / (1 - p)
#
# Example: q=0.30, p=0.20 -> f* = 0.10/0.80 = 0.125 (12.5% of bankroll)
#
# In practice, fractional Kelly (0.25x) is used because:
# - Model probabilities are noisy estimates, not ground truth
# - Growth scales as f - f²/2, so 0.25x Kelly gives 44% of optimal growth
#   but far less variance
# ============================================================================


class TestKellyFractionBinary:
    """Tests for _kelly_fraction_binary(model_prob, implied_prob)."""

    def test_no_edge_means_no_bet(self) -> None:
        """When model and market agree (q = p), Kelly says bet nothing.

        f* = (q - p) / (1 - p) = 0 / (1 - p) = 0
        """
        assert _kelly_fraction_binary(0.25, 0.25) == 0.0

    def test_clear_edge_moderate_price(self) -> None:
        """Classic scenario: model at 50%, market at 30%.

        f* = (0.50 - 0.30) / (1 - 0.30) = 0.20 / 0.70 ≈ 0.286

        Interpretation: bet 28.6% of your bankroll. On $1,000, that's
        $286 in contract outlay. This is FULL Kelly — very aggressive.
        """
        f = _kelly_fraction_binary(0.50, 0.30)
        assert f == pytest.approx(0.2 / 0.7, abs=1e-6)

    def test_certain_win_bets_everything(self) -> None:
        """If you're certain (q=1.0), Kelly says bet your entire bankroll.

        f* = (1.0 - p) / (1 - p) = 1.0 regardless of price.
        """
        assert _kelly_fraction_binary(1.0, 0.30) == pytest.approx(1.0, abs=1e-6)
        assert _kelly_fraction_binary(1.0, 0.90) == pytest.approx(1.0, abs=1e-6)

    def test_negative_edge_returns_zero(self) -> None:
        """When the market is smarter than you (q < p), Kelly says don't bet.

        f* = (0.15 - 0.25) / (1 - 0.25) = -0.133 -> clamped to 0

        This is the mathematical proof that you shouldn't gamble
        on negative-edge bets, no matter how sure you feel.
        """
        assert _kelly_fraction_binary(0.15, 0.25) == 0.0

    def test_small_edge_high_price_is_conservative(self) -> None:
        """Tiny edge on an expensive contract -> very small bet.

        q=0.82, p=0.80 -> f* = 0.02 / 0.20 = 0.10 (10%)

        Even though the 2pp edge looks small, Kelly sizes proportionally.
        With 0.25x fractional Kelly, this becomes 2.5% of bankroll.
        """
        f = _kelly_fraction_binary(0.82, 0.80)
        assert f == pytest.approx(0.10, abs=1e-6)

    def test_extreme_prices_handled(self) -> None:
        """Edge cases: implied prob at 0 or 1 should not crash.

        The function guards against degenerate prices (0 or 1) by returning 0.
        At implied_prob=0 the denominator (1-p) is fine, but a free contract
        doesn't exist in practice, so the function conservatively returns 0.
        At implied_prob=1 you'd pay the full $1 payout — no edge possible.
        """
        assert _kelly_fraction_binary(0.50, 0.0) == 0.0  # Guarded: returns 0
        assert _kelly_fraction_binary(0.50, 1.0) == 0.0  # Guarded: returns 0


class TestIndependentKelly:
    """
    Independent Kelly sizes each outcome separately, treating each as an
    isolated binary bet. Simple and conservative, but ignores the constraint
    that exactly one outcome wins (outcomes are mutually exclusive).
    """

    def _make_edge(
        self,
        outcome: str,
        model_prob: float,
        price: float,
    ) -> Edge:
        """Helper: create an Edge for testing."""
        return Edge(
            outcome=outcome,
            direction=PositionDirection.YES,
            model_prob=model_prob,
            execution_price=price,
            fee_type=FeeType.TAKER,
        )

    def test_single_outcome_full_chain(self) -> None:
        """Trace the full chain: edge -> Kelly fraction -> contracts -> outlay.

        Scenario (inspired by backtest Marty Supreme, Dec 2025):
          Model: 30%, Market: $0.10, Bankroll: $1,000, 0.25x Kelly

        Step 1 - Edge:
          fee at $0.10 = ceil(0.07 × 0.10 × 0.90 × 100) / 100 = $0.01
          implied_prob_with_fee = 0.10 + 0.01 = 0.11
          gross_edge = 0.30 - 0.10 = 0.20
          net_edge = 0.20 - 0.01 = 0.19

        Step 2 - Kelly:
          f* = (0.30 - 0.11) / (1 - 0.11) = 0.19 / 0.89 ≈ 0.2135
          f_applied = 0.2135 * 0.25 = 0.0534

        Step 3 - Contracts:
          raw = 0.0534 * $1000 / $0.10 = 533.7 -> 533 contracts
          capped at max_position = $250 / $0.10 = 2500 (no cap hit)
        """
        edge = self._make_edge("Marty Supreme", model_prob=0.30, price=0.10)
        config = KellyConfig(
            bankroll=1000,
            kelly_fraction=0.25,
            kelly_mode=KellyMode.INDEPENDENT,
            buy_edge_threshold=0.05,
            max_position_per_outcome=250,
            max_total_exposure=500,
        )

        result = independent_kelly([edge], config)

        assert len(result) == 1

        alloc = result[0]
        assert alloc.outcome == "Marty Supreme"
        assert alloc.recommended_contracts > 0
        assert alloc.outlay_dollars > 0
        assert alloc.outlay_dollars <= config.max_position_per_outcome

        # Verify the outlay = contracts * price
        assert alloc.outlay_dollars == pytest.approx(alloc.recommended_contracts * 0.10, abs=0.01)

    def test_buy_edge_threshold_filters_weak_signals(self) -> None:
        """Outcomes with net edge below buy_edge_threshold get zero allocation.

        Scenario: Model 12%, market $0.08.
        net_edge = 0.04 - 0.07 = -0.03 (below buy_edge_threshold=0.05)
        Result: 0 contracts. Don't trade when the edge is too thin.
        """
        edge = self._make_edge("Weak Signal", model_prob=0.12, price=0.08)
        config = KellyConfig(
            bankroll=1000,
            kelly_fraction=0.25,
            kelly_mode=KellyMode.INDEPENDENT,
            buy_edge_threshold=0.05,
            max_position_per_outcome=250,
            max_total_exposure=500,
        )

        result = independent_kelly([edge], config)

        assert result[0].recommended_contracts == 0
        assert sum(a.outlay_dollars for a in result) == 0

    def test_position_cap_limits_single_outcome(self) -> None:
        """Per-outcome cap prevents concentration in a single bet.

        Even with huge edge, each outcome is capped at max_position_per_outcome
        dollars of outlay. This prevents a single bad model estimate from
        destroying the portfolio.
        """
        # Huge edge -> Kelly wants lots of contracts
        edge = self._make_edge("Strong Signal", model_prob=0.80, price=0.10)
        config = KellyConfig(
            bankroll=10000,
            kelly_fraction=0.25,
            kelly_mode=KellyMode.INDEPENDENT,
            buy_edge_threshold=0.05,
            max_position_per_outcome=100,  # Tight cap
            max_total_exposure=500,
        )

        result = independent_kelly([edge], config)
        alloc = result[0]

        # Outlay should not exceed the per-outcome cap
        assert alloc.outlay_dollars <= 100

    def test_total_exposure_cap_scales_down_portfolio(self) -> None:
        """Total portfolio outlay is capped, scaling all positions proportionally.

        Scenario: Two outcomes each want $200 -> total $400.
        But max_total_exposure = $300. Both get scaled down proportionally.
        """
        edges = [
            self._make_edge("A", model_prob=0.40, price=0.15),
            self._make_edge("B", model_prob=0.35, price=0.10),
        ]
        config = KellyConfig(
            bankroll=5000,
            kelly_fraction=0.25,
            kelly_mode=KellyMode.INDEPENDENT,
            buy_edge_threshold=0.05,
            max_position_per_outcome=500,
            max_total_exposure=100,  # Very tight cap
        )

        result = independent_kelly(edges, config)
        assert sum(a.outlay_dollars for a in result) <= 100


class TestMultiOutcomeKelly:
    """
    Multi-outcome Kelly jointly optimizes across outcomes, accounting for the
    constraint that exactly one outcome wins. This correctly sizes a portfolio
    of mutually exclusive bets.

    Key insight: if you bet on outcomes A and B, at most one can win. Your
    loss on the loser is certain. Independent Kelly doesn't account for this
    anti-correlation and will oversize the portfolio. Multi-outcome Kelly
    maximizes expected log-wealth jointly:

      E[log(W)] = Σᵢ qᵢ * log(W_if_i_wins) + q_none * log(W_if_none_win)
    """

    def _make_edge(
        self,
        outcome: str,
        model_prob: float,
        price: float,
    ) -> Edge:
        return Edge(
            outcome=outcome,
            direction=PositionDirection.YES,
            model_prob=model_prob,
            execution_price=price,
            fee_type=FeeType.TAKER,
        )

    def test_multi_outcome_produces_valid_allocations(self) -> None:
        """Multi-outcome optimizer produces feasible allocations.

        Multi-outcome Kelly jointly optimizes across outcomes. It may allocate
        MORE than independent Kelly (which post-hoc applies exposure caps that
        can aggressively scale down). The optimizer maximizes E[log(W)] subject
        to constraints, producing a different portfolio shape.

        Key checks:
        - Only outcomes with sufficient edge get allocations
        - Total outlay respects max_total_exposure constraint
        """
        edges = [
            self._make_edge("A", model_prob=0.35, price=0.15),
            self._make_edge("B", model_prob=0.25, price=0.10),
            self._make_edge("C", model_prob=0.10, price=0.05),
        ]
        config = KellyConfig(
            bankroll=1000,
            kelly_fraction=0.25,
            kelly_mode=KellyMode.MULTI_OUTCOME,
            buy_edge_threshold=0.05,
            max_position_per_outcome=250,
            max_total_exposure=500,
        )

        allocations = multi_outcome_kelly(edges, config)

        total_outlay = sum(a.outlay_dollars for a in allocations)
        assert total_outlay <= config.max_total_exposure + 1  # Constraint

        # C has net_edge < buy_edge_threshold -> 0 allocation
        alloc_c = [a for a in allocations if a.outcome == "C"][0]
        assert alloc_c.recommended_contracts == 0

    def test_no_eligible_outcomes_returns_zero(self) -> None:
        """When no outcome has sufficient edge, both methods return zero."""
        edges = [
            self._make_edge("A", model_prob=0.05, price=0.05),
            self._make_edge("B", model_prob=0.03, price=0.03),
        ]
        config = KellyConfig(
            bankroll=1000,
            kelly_fraction=0.25,
            kelly_mode=KellyMode.MULTI_OUTCOME,
            buy_edge_threshold=0.10,
            max_position_per_outcome=250,
            max_total_exposure=500,
        )

        allocations = multi_outcome_kelly(edges, config)
        assert sum(a.outlay_dollars for a in allocations) == 0
        assert all(a.recommended_contracts == 0 for a in allocations)

    def test_single_outcome_both_methods_positive(self) -> None:
        """With one eligible outcome, both methods allocate contracts.

        The exact contract count may differ because independent Kelly applies
        per-outcome and total exposure caps post-hoc, while multi-outcome uses
        optimizer constraints. Both should be positive and reasonable.
        """
        edges = [self._make_edge("Solo", model_prob=0.40, price=0.15)]
        config = KellyConfig(
            bankroll=1000,
            kelly_fraction=0.25,
            kelly_mode=KellyMode.MULTI_OUTCOME,
            buy_edge_threshold=0.05,
            max_position_per_outcome=250,
            max_total_exposure=500,
        )

        ind = independent_kelly(edges, config)
        multi = multi_outcome_kelly(edges, config)

        assert ind[0].recommended_contracts > 0
        assert multi[0].recommended_contracts > 0
        assert sum(a.outlay_dollars for a in ind) <= config.max_position_per_outcome
        assert sum(a.outlay_dollars for a in multi) <= config.max_total_exposure + 1


# Note: TestExpectedLogGrowth and related tests using OutcomeAllocation
# and _expected_log_growth were removed in the schema refactor.
# These internals are now tested indirectly through the kelly allocation
# functions which jointly optimize expected log-growth.


class TestKellyAllocationFee:
    """Tests for the KellyAllocation.fee computed field."""

    def test_fee_derived_from_edge_components(self) -> None:
        """fee = model_prob - execution_price - net_edge.

        Example: model_prob=0.35, execution_price=0.25, net_edge=0.09
        fee = 0.35 - 0.25 - 0.09 = 0.01
        """
        from oscar_prediction_market.trading.schema import KellyAllocation

        alloc = KellyAllocation(
            outcome="Test",
            direction=PositionDirection.YES,
            model_prob=0.35,
            execution_price=0.25,
            net_edge=0.09,
            recommended_contracts=10,
        )
        assert alloc.fee == pytest.approx(0.01, abs=1e-5)

    def test_fee_zero_when_no_fee(self) -> None:
        """When net_edge = model_prob - execution_price (no fee), fee should be 0."""
        from oscar_prediction_market.trading.schema import KellyAllocation

        alloc = KellyAllocation(
            outcome="Test",
            direction=PositionDirection.YES,
            model_prob=0.40,
            execution_price=0.20,
            net_edge=0.20,  # full edge, no fees
            recommended_contracts=5,
        )
        assert alloc.fee == pytest.approx(0.0, abs=1e-5)


class TestCapTotalExposureIncludesFees:
    """Tests that _cap_total_exposure accounts for fees in the cost calculation.

    Bug fix: previously, _cap_total_exposure summed outlay_dollars (= contracts ×
    execution_price), ignoring fees. Since actual cost = contracts × (price + fee),
    the total could exceed the budget when the cap binds.

    Now it sums contracts × (execution_price + fee) so the cap is correct.
    """

    def test_cap_includes_fees_in_total(self) -> None:
        """Total cost with fees exceeds budget → positions should be scaled down.

        Setup: 2 allocations at $0.20 with $0.01 fee each, 100 contracts each.
        Total outlay (no fee): 2 × 100 × 0.20 = $40
        Total cost (with fee): 2 × 100 × 0.21 = $42
        Budget: $41 — under fee-inclusive cost but over fee-exclusive cost.

        Old behavior: $40 <= $41, no scaling (but actual cost is $42!).
        New behavior: $42 > $41, scale down by 41/42 ≈ 0.976.
          A: floor(100 × 0.976) = 97
          B: floor(100 × 0.976) = 97
        """
        from oscar_prediction_market.trading.kelly import (
            _cap_total_exposure,
        )
        from oscar_prediction_market.trading.schema import KellyAllocation

        allocations = [
            KellyAllocation(
                outcome="A",
                direction=PositionDirection.YES,
                model_prob=0.30,
                execution_price=0.20,
                net_edge=0.09,  # fee = 0.30 - 0.20 - 0.09 = 0.01
                recommended_contracts=100,
            ),
            KellyAllocation(
                outcome="B",
                direction=PositionDirection.YES,
                model_prob=0.30,
                execution_price=0.20,
                net_edge=0.09,  # fee = 0.01
                recommended_contracts=100,
            ),
        ]

        capped = _cap_total_exposure(allocations, max_total_exposure=41.0)

        # Should have been scaled down since 200 × 0.21 = $42 > $41
        total_cost = sum(a.recommended_contracts * (a.execution_price + a.fee) for a in capped)
        assert total_cost <= 41.0
        # Each should be ~97 contracts
        for a in capped:
            assert a.recommended_contracts < 100

    def test_per_outcome_cap_includes_fees(self) -> None:
        """Independent Kelly per-outcome cap should account for fees.

        With max_position_per_outcome=$50, price=$0.20, fee=$0.01:
        Max contracts by cap = $50 / $0.21 = 238 (not 250 from $50/$0.20).
        """
        edges = [
            Edge(
                outcome="Big",
                direction=PositionDirection.YES,
                model_prob=0.80,
                execution_price=0.20,
                fee_type=FeeType.TAKER,
            )
        ]
        config = KellyConfig(
            bankroll=100000,
            kelly_fraction=1.0,  # full Kelly for max sizing
            kelly_mode=KellyMode.INDEPENDENT,
            buy_edge_threshold=0.0,
            max_position_per_outcome=50,
            max_total_exposure=100000,
        )

        result = independent_kelly(edges, config)
        alloc = result[0]

        # With fees, max contracts should be floor(50 / (0.20 + fee))
        fee = alloc.fee
        max_with_fee = int(50 / (0.20 + fee))
        assert alloc.recommended_contracts <= max_with_fee
