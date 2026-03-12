"""Tests for spread estimation from historical trades."""

from datetime import UTC, datetime, timedelta

import pytest

from oscar_prediction_market.trading.edge import (
    estimate_spread_from_trades,
)
from oscar_prediction_market.trading.oscar_market import TradeRecord

# ============================================================================
# Section 5: Spread Estimation from Historical Trades
#
# In backtesting we don't have live orderbooks, so we estimate the bid-ask
# spread from trade data. When a buy-taker trade at 26c is followed by a
# sell-taker trade at 24c, the spread was ~2c. We take the median gap
# (robust to outliers) and halve it for one-way penalty.
# ============================================================================


class TestEstimateSpreadFromTrades:
    def _make_trades(
        self,
        sides: list[str],
        prices: list[float],
        ticker: str = "TEST",
    ) -> list[TradeRecord]:
        """Build typed trade records from alternating sides and prices."""
        start = datetime(2026, 1, 1, tzinfo=UTC)
        return [
            TradeRecord(
                timestamp=start + timedelta(hours=i),
                date=(start + timedelta(hours=i)).date(),
                ticker=ticker,
                nominee=ticker,
                taker_side=side,
                yes_price=price,
                count=1,
            )
            for i, (side, price) in enumerate(zip(sides, prices, strict=False))
        ]

    def test_alternating_buys_sells_clear_spread(self) -> None:
        """Alternating buy/sell at 26c/24c -> spread = $0.02, one-way = $0.01.

        The function computes |price_i - price_{i-1}| at each side change,
        takes the median, and halves it (one-way penalty). Returns dollars.
        """
        # Generate alternating buy/sell pattern with consistent 2c spread
        sides = ["yes", "no"] * 15  # 30 trades (above min_trades_required=20)
        prices = [0.26, 0.24] * 15
        trades = self._make_trades(sides, prices)

        spreads = estimate_spread_from_trades(trades, default_spread=0.04, min_trades_required=20)

        assert "TEST" in spreads
        assert spreads["TEST"] == pytest.approx(0.01, abs=0.001)

    def test_insufficient_trades_uses_default(self) -> None:
        """With too few trades, falls back to default_spread / 2."""
        sides = ["yes", "no", "yes"]
        prices = [0.26, 0.24, 0.26]
        trades = self._make_trades(sides, prices)

        spreads = estimate_spread_from_trades(trades, default_spread=0.04, min_trades_required=20)

        assert spreads["TEST"] == 0.02  # $0.04 default / 2 = $0.02

    def test_empty_trades(self) -> None:
        """Empty input returns empty dict."""
        assert estimate_spread_from_trades([], default_spread=0.04, min_trades_required=20) == {}

    def test_variable_spread_uses_median(self) -> None:
        """With variable spreads, the median is used (robust to outliers).

        Spreads: [2, 2, 2, 2, 10]  -> median = 2, one-way = $0.01
        The outlier (10c) doesn't distort the estimate.
        """
        sides = ["yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no"] * 3
        prices = [0.26, 0.24, 0.26, 0.24, 0.26, 0.24, 0.26, 0.24, 0.26, 0.16] * 3
        trades = self._make_trades(sides, prices)

        spreads = estimate_spread_from_trades(trades, default_spread=0.04, min_trades_required=20)
        assert spreads["TEST"] == pytest.approx(0.01, abs=0.005)
