"""Tests for execution price computation from orderbook.

When you place a market order, you "walk the book" — consuming liquidity
at successively worse prices. VWAP (volume-weighted average price) tells
you the true average price you'll pay across all filled levels.

Kalshi uses a dual-sided orderbook: YES bids and NO bids.
Buying YES = taking from NO bids (converted: no_bid at P -> yes_ask at 100-P)
Selling YES = hitting YES bids directly
"""

import pytest

from oscar_prediction_market.trading.edge import (
    get_execution_price,
)
from oscar_prediction_market.trading.kalshi_client import (
    Orderbook,
)
from oscar_prediction_market.trading.schema import (
    Side,
)

# ============================================================================
# Section 2: Execution Price from Orderbook
#
# When you place a market order, you "walk the book" — consuming liquidity
# at successively worse prices. VWAP (volume-weighted average price) tells
# you the true average price you'll pay across all filled levels.
#
# Kalshi uses a dual-sided orderbook: YES bids and NO bids.
# Buying YES = taking from NO bids (converted: no_bid at P -> yes_ask at 100-P)
# Selling YES = hitting YES bids directly
# ============================================================================


class TestGetExecutionPrice:
    def test_single_level_full_fill(self) -> None:
        """All contracts filled at one price level -> VWAP = that price.

        Scenario: Orderbook has 50 No contracts at 75c (= Yes ask at 25c).
        You buy 10 YES contracts. All fill at 25c.
        """
        orderbook = Orderbook(yes=[], no=[[75, 50]])
        result = get_execution_price(orderbook, side=Side.BUY, n_contracts=10)

        assert result.execution_price == 0.25
        assert result.n_contracts_fillable == 10
        assert result.is_partial is False
        assert result.levels_consumed == 1

    def test_multi_level_fill_shows_price_impact(self) -> None:
        """Large orders walk through multiple price levels, getting worse fills.

        Scenario: You want 30 contracts. Orderbook has:
          No @ 75c (qty 10) -> Yes ask @ 25c  (cheapest, consumed first)
          No @ 74c (qty 15) -> Yes ask @ 26c
          No @ 73c (qty 20) -> Yes ask @ 27c

        VWAP = (10*25 + 15*26 + 5*27) / 30 = (250 + 390 + 135) / 30 = 25.83c

        This is why large orders on illiquid contracts are expensive.
        """
        orderbook = Orderbook(yes=[], no=[[75, 10], [74, 15], [73, 20]])
        result = get_execution_price(orderbook, side=Side.BUY, n_contracts=30)

        assert result.execution_price == pytest.approx(0.2583, abs=0.0001)
        assert result.n_contracts_fillable == 30
        assert result.is_partial is False
        assert result.levels_consumed == 3

    def test_partial_fill_when_book_too_thin(self) -> None:
        """If the orderbook doesn't have enough depth, you get a partial fill.

        Scenario: You want 20 contracts but only 5 available at 25c.
        Result: partial fill flagged, only 5 contracts fillable.
        """
        orderbook = Orderbook(yes=[], no=[[75, 5]])
        result = get_execution_price(orderbook, side=Side.BUY, n_contracts=20)

        assert result.n_contracts_fillable == 5
        assert result.is_partial is True
        assert result.execution_price == 0.25

    def test_empty_orderbook(self) -> None:
        """No liquidity at all -> zero fillable contracts."""
        orderbook = Orderbook(yes=[], no=[])
        result = get_execution_price(orderbook, side=Side.BUY, n_contracts=10)

        assert result.n_contracts_fillable == 0
        assert result.is_partial is True
        assert result.execution_price == 0

    def test_sell_side_hits_yes_bids(self) -> None:
        """Selling YES contracts hits YES bids (best bid first, descending).

        Scenario: Sell 10 YES. YES bids at 24c (qty 6) and 22c (qty 10).
        VWAP = (6*24 + 4*22) / 10 = (144 + 88) / 10 = 23.2c
        """
        orderbook = Orderbook(yes=[[24, 6], [22, 10]], no=[])
        result = get_execution_price(orderbook, side=Side.SELL, n_contracts=10)

        assert result.execution_price == pytest.approx(0.232, abs=0.001)
        assert result.n_contracts_fillable == 10
        assert result.is_partial is False
