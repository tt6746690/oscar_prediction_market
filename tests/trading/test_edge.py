"""Tests for edge computation.

Edge = model's estimated probability - market's implied probability - fees

For a contract priced at $P:
  implied_prob = P
  gross_edge = model_prob - implied_prob
  fee (in dollars)
  net_edge = gross_edge - fee

Positive net edge means the market is underpricing relative to our model,
even after accounting for transaction costs. This is the fundamental
prerequisite for any trade.
"""

import pytest

from oscar_prediction_market.trading.edge import (
    Edge,
)
from oscar_prediction_market.trading.schema import (
    FeeType,
    PositionDirection,
)

# ============================================================================
# Section 3: Edge Computation
#
# Edge = model's estimated probability - market's implied probability - fees
#
# For a contract priced at $P:
#   implied_prob = P
#   gross_edge = model_prob - implied_prob
#   fee (in dollars)
#   net_edge = gross_edge - fee
#
# Positive net edge means the market is underpricing relative to our model,
# even after accounting for transaction costs. This is the fundamental
# prerequisite for any trade.
# ============================================================================


class TestEdge:
    def test_positive_edge_buy_yes(self) -> None:
        """Model sees more value than the market — classic buy opportunity.

        Scenario (inspired by early DGA-period Marty Supreme):
          Model says 30% win probability.
          Market at $0.20 (implies 20%).
          Fee at $0.20 = ceil(0.07 × 0.20 × 0.80 × 100) / 100 = $0.02.

        gross_edge = 0.30 - 0.20 = 0.10 (10pp)
        fee = 0.02
        net_edge = 0.10 - 0.02 = 0.08 (8pp)

        With the variance-based formula, the fee at $0.20 is only $0.02
        (vs the old 7c minimum), leaving much more net edge.
        """
        result = Edge(
            outcome="Marty Supreme",
            direction=PositionDirection.YES,
            model_prob=0.30,
            execution_price=0.20,
            fee_type=FeeType.TAKER,
        )

        assert result.gross_edge == pytest.approx(0.10, abs=1e-6)
        assert result.fee == 0.02
        assert result.fee == pytest.approx(0.02, abs=1e-6)
        assert result.net_edge == pytest.approx(0.08, abs=1e-6)
        assert result.implied_prob == pytest.approx(0.20, abs=1e-6)

    def test_no_edge_after_fees(self) -> None:
        """Model slightly favors the contract, but fees consume the edge.

        Scenario: Model says 10%, market at $0.08. Gross edge = 2pp.
        Fee = ceil(0.07 × 0.08 × 0.92 × 100) / 100 = $0.01.
        fee = 0.01. Net edge = 0.02 - 0.01 = 0.01.

        With such thin net edge (1pp), this is marginal at best —
        likely below any practical buy_edge_threshold.
        """
        result = Edge(
            outcome="Longshot",
            direction=PositionDirection.YES,
            model_prob=0.10,
            execution_price=0.08,
            fee_type=FeeType.TAKER,
        )

        assert result.gross_edge == pytest.approx(0.02, abs=1e-6)
        assert result.fee == 0.01
        assert result.net_edge == pytest.approx(0.01, abs=1e-6)

    def test_market_agrees_with_model(self) -> None:
        """Model and market agree — zero gross edge, negative net edge.

        When model_prob = implied_prob, there's no informational
        advantage. Fees make it a guaranteed loss.
        """
        result = Edge(
            outcome="Consensus",
            direction=PositionDirection.YES,
            model_prob=0.25,
            execution_price=0.25,
            fee_type=FeeType.TAKER,
        )

        assert result.gross_edge == pytest.approx(0.0, abs=1e-6)
        assert result.net_edge < 0  # Fees make it negative

    def test_sell_edge_via_no_direction(self) -> None:
        """Selling YES is equivalent to buying NO.

        Scenario: Market at $0.30 (implies 30%), model says only 15%.
        For NO side: model_prob = 1 - 0.15 = 0.85, NO price = $1.00 - $0.30 = $0.70.
        gross_edge = 0.85 - 0.70 = 0.15
        Fee at $0.70 = ceil(0.07 × 0.70 × 0.30 × 100) / 100 = $0.02.
        net_edge = 0.15 - 0.02 = 0.13.
        """
        result = Edge(
            outcome="Overpriced",
            direction=PositionDirection.NO,
            model_prob=0.85,
            execution_price=0.70,
            fee_type=FeeType.TAKER,
        )

        assert result.gross_edge == pytest.approx(0.15, abs=1e-6)
        assert result.fee == 0.02
        assert result.net_edge == pytest.approx(0.13, abs=1e-6)

    def test_spread_adjusted_price_reduces_edge(self) -> None:
        """Caller adjusts execution price for spread before computing edge.

        Scenario: Market close at $0.20, spread penalty $0.02 per side.
        Buy execution price = $0.22 (worse for buyer).
        Edge computed at $0.22 is lower than at $0.20.

        Spread handling is the caller's responsibility
        (generate_signals adjusts prices directionally).
        """
        at_close = Edge(
            outcome="Test",
            direction=PositionDirection.YES,
            model_prob=0.30,
            execution_price=0.20,
            fee_type=FeeType.TAKER,
        )
        with_spread = Edge(
            outcome="Test",
            direction=PositionDirection.YES,
            model_prob=0.30,
            execution_price=0.22,  # close + $0.02 spread (caller-adjusted)
            fee_type=FeeType.TAKER,
        )

        assert with_spread.execution_price == 0.22
        assert with_spread.implied_prob == pytest.approx(0.22, abs=1e-6)
        assert with_spread.net_edge < at_close.net_edge

    def test_maker_fee_type(self) -> None:
        """Maker fee is much smaller, leaving more net edge.

        At $0.20: taker fee = $0.02, maker fee = ceil(0.0175 × 0.20 × 0.80 × 100) / 100
        = $0.01. Maker saves $0.01 per contract.
        """
        taker_edge = Edge(
            outcome="Test",
            direction=PositionDirection.YES,
            model_prob=0.30,
            execution_price=0.20,
            fee_type=FeeType.TAKER,
        )
        maker_edge = Edge(
            outcome="Test",
            direction=PositionDirection.YES,
            model_prob=0.30,
            execution_price=0.20,
            fee_type=FeeType.MAKER,
        )

        assert maker_edge.fee < taker_edge.fee
        assert maker_edge.net_edge > taker_edge.net_edge
