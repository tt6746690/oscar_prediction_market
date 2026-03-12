"""Tests for inspect_trade module: per-fill P&L computation."""

from datetime import UTC, datetime

import pytest

from oscar_prediction_market.trading.inspect_trade import _fill_pnl
from oscar_prediction_market.trading.schema import (
    Fill,
    PositionDirection,
    TradeAction,
)


def _make_fill(
    outcome: str = "A",
    direction: PositionDirection = PositionDirection.YES,
    action: TradeAction = TradeAction.BUY,
    contracts: int = 10,
    price: float = 0.20,
    fee_dollars: float = 0.0,
) -> Fill:
    return Fill(
        timestamp=datetime(2026, 2, 10, 21, 0, tzinfo=UTC),
        outcome=outcome,
        direction=direction,
        action=action,
        contracts=contracts,
        price=price,
        fee_dollars=fee_dollars,
        cash_delta=0.0,
        reason="test",
    )


class TestFillPnlBuyYes:
    """P&L for BUY YES fills.

    A YES contract at price p pays $1 if the outcome wins:
      - Winner: profit = (1.0 - p) per contract
      - Loser:  loss   = -p per contract
    """

    def test_winner_profit(self) -> None:
        """BUY YES at $0.20 on the winner → +$0.80 per contract.

        10 contracts × (1.0 - 0.20) = 10 × $0.80 = $8.00
        """
        fill = _make_fill(direction=PositionDirection.YES, price=0.20, contracts=10)
        pnl = _fill_pnl(fill, winner="A")
        assert pnl == pytest.approx(8.0)

    def test_loser_loss(self) -> None:
        """BUY YES at $0.20 on a loser → -$0.20 per contract.

        10 contracts × -0.20 = -$2.00
        """
        fill = _make_fill(direction=PositionDirection.YES, price=0.20, contracts=10)
        pnl = _fill_pnl(fill, winner="B")
        assert pnl == pytest.approx(-2.0)

    def test_fee_subtracted(self) -> None:
        """Fees are an additional cost on top of settlement P&L."""
        fill = _make_fill(
            direction=PositionDirection.YES, price=0.20, contracts=10, fee_dollars=0.50
        )
        pnl = _fill_pnl(fill, winner="A")
        assert pnl == pytest.approx(8.0 - 0.50)


class TestFillPnlBuyNo:
    """P&L for BUY NO fills.

    A NO contract at NO-price p pays $1 if the outcome loses:
      - Outcome loses (NO wins):  profit = (1.0 - p) per contract
      - Outcome wins  (NO loses): loss   = -p per contract
    """

    def test_no_wins_profit(self) -> None:
        """BUY NO at $0.80, outcome loses → NO wins → +$0.20 per contract.

        10 contracts × (1.0 - 0.80) = 10 × $0.20 = $2.00
        """
        fill = _make_fill(direction=PositionDirection.NO, price=0.80, contracts=10)
        pnl = _fill_pnl(fill, winner="B")  # A loses → NO on A wins
        assert pnl == pytest.approx(2.0)

    def test_no_loses_loss(self) -> None:
        """BUY NO at $0.80, outcome wins → NO loses → -$0.80 per contract.

        10 contracts × -0.80 = -$8.00
        """
        fill = _make_fill(direction=PositionDirection.NO, price=0.80, contracts=10)
        pnl = _fill_pnl(fill, winner="A")  # A wins → NO on A loses
        assert pnl == pytest.approx(-8.0)

    def test_cheap_no_contract(self) -> None:
        """BUY NO at $0.10 (underdog), outcome loses → big profit.

        5 contracts × (1.0 - 0.10) = 5 × $0.90 = $4.50
        """
        fill = _make_fill(
            outcome="Underdog",
            direction=PositionDirection.NO,
            price=0.10,
            contracts=5,
        )
        pnl = _fill_pnl(fill, winner="Other")
        assert pnl == pytest.approx(4.50)


class TestFillPnlSellReturnsZero:
    """SELL fills return 0 P&L (settlement P&L doesn't apply to exited positions)."""

    def test_sell_yes_returns_zero(self) -> None:
        fill = _make_fill(action=TradeAction.SELL, direction=PositionDirection.YES)
        assert _fill_pnl(fill, winner="A") == 0.0

    def test_sell_no_returns_zero(self) -> None:
        fill = _make_fill(action=TradeAction.SELL, direction=PositionDirection.NO)
        assert _fill_pnl(fill, winner="B") == 0.0
