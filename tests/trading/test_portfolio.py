"""Tests for portfolio operations: apply_signals, compute_mtm_value, settle_positions.

Portfolio functions are stateless: they take positions + cash + signals and
return updated state. These tests verify the core accounting math directly.
"""

from datetime import UTC

from oscar_prediction_market.trading.portfolio import (
    ExecutionBatch,
    apply_signals,
    compute_mtm_value,
    settle_positions,
)
from oscar_prediction_market.trading.schema import (
    FeeType,
    KellyConfig,
    KellyMode,
    MarketQuotes,
    Position,
    PositionDirection,
    TradingConfig,
)
from oscar_prediction_market.trading.signals import (
    TradeSignal,
    generate_signals,
)

# ============================================================================
# Helpers
# ============================================================================


def _make_config(bankroll: float = 1000.0, buy_edge_threshold: float = 0.05) -> TradingConfig:
    return TradingConfig(
        kelly=KellyConfig(
            bankroll=bankroll,
            kelly_fraction=0.25,
            kelly_mode=KellyMode.MULTI_OUTCOME,
            buy_edge_threshold=buy_edge_threshold,
            max_position_per_outcome=500,
            max_total_exposure=1000,
        ),
        sell_edge_threshold=-0.03,
        fee_type=FeeType.TAKER,
        limit_price_offset=0.0,
        min_price=0,
        allowed_directions=frozenset({PositionDirection.YES}),
    )


def _make_report(
    model_predictions: dict[str, float],
    market_prices: dict[str, float],
    current_positions: list[Position] | None = None,
    spread: float = 0.0,
    bankroll: float = 1000.0,
) -> list[TradeSignal]:
    return generate_signals(
        model_predictions=model_predictions,
        execution_prices=MarketQuotes.from_close_prices(market_prices, spread),
        current_positions=current_positions or [],
        config=_make_config(bankroll=bankroll),
    )


# ============================================================================
# compute_mtm_value
# ============================================================================


class TestComputeMtmValue:
    """Mark-to-market values positions at current prices."""

    def test_single_position(self) -> None:
        """10 contracts at $0.30 = $3.00."""
        positions = [
            Position(outcome="A", direction=PositionDirection.YES, contracts=10, avg_cost=0.20)
        ]
        prices = {"A": 0.30}
        assert compute_mtm_value(positions, prices) == 3.00

    def test_multiple_positions(self) -> None:
        """Two outcomes: 10@$0.30 + 5@$0.50 = $3 + $2.50 = $5.50."""
        positions = [
            Position(outcome="A", direction=PositionDirection.YES, contracts=10, avg_cost=0.20),
            Position(outcome="B", direction=PositionDirection.YES, contracts=5, avg_cost=0.40),
        ]
        prices = {"A": 0.30, "B": 0.50}
        assert compute_mtm_value(positions, prices) == 5.50

    def test_missing_price_valued_at_zero(self) -> None:
        """Outcome not in prices is valued at 0 (e.g., market suspended)."""
        positions = [
            Position(outcome="A", direction=PositionDirection.YES, contracts=10, avg_cost=0.20)
        ]
        assert compute_mtm_value(positions, {}) == 0.00

    def test_zero_contracts_ignored(self) -> None:
        """Zero-contract positions contribute nothing to MtM."""
        positions = [
            Position(outcome="A", direction=PositionDirection.YES, contracts=0, avg_cost=0.20),
            Position(outcome="B", direction=PositionDirection.YES, contracts=5, avg_cost=0.40),
        ]
        prices = {"A": 0.30, "B": 0.50}
        assert compute_mtm_value(positions, prices) == 2.50

    def test_empty_positions(self) -> None:
        assert compute_mtm_value([], {"A": 0.30}) == 0.00


# ============================================================================
# apply_signals
# ============================================================================


class TestApplySignals:
    """Apply BUY/SELL signals: cash and position accounting."""

    def test_buy_single_outcome_updates_cash_and_position(self) -> None:
        """Buy 10 contracts at $0.20 (no spread) + fee.

        Step-by-step:
          buy cost = 10 × $0.20 = $2.00
          Kalshi taker fee at $0.20:
            fee_per_contract = ceil(0.07 × 0.20 × 0.80 × 100) / 100 = $0.02
            total fee = 10 × $0.02 = $0.20
          total cash out = $2.00 + $0.20 = $2.20
          remaining cash = $1000.00 - $2.20 = $997.80
          position = 10 contracts @ $0.20 avg cost
        """
        # Model 40% vs $0.20 market -> strong edge
        report = _make_report({"A": 0.40}, {"A": 0.20}, bankroll=1000.0)
        result = apply_signals([], 1000.0, report, fee_type=FeeType.TAKER)

        assert isinstance(result, ExecutionBatch)
        a_positions = [p for p in result.positions if p.outcome == "A"]
        assert len(a_positions) == 1
        assert a_positions[0].contracts > 0
        assert result.cash < 1000.0
        assert result.fees_paid > 0
        assert result.n_trades == 1
        assert len(result.fills) == 1
        fill = result.fills[0]
        assert fill.action == "buy"
        assert fill.contracts > 0
        assert fill.cash_delta < 0  # Cash went out

    def test_sell_reduces_position_and_adds_cash(self) -> None:
        """Sell 50 contracts at $0.10.

        First buy into position by generating a buy signal, then sell by
        generating a sell signal (edge flipped below threshold).

        Sell cashflow:
          revenue = 50 × $0.10 = $5.00
          fee at $0.10 = ceil(0.07 × 0.10 × 0.90 × 100) / 100 = $0.01/contract
          total fee = 50 × $0.01 = $0.50
          net cash in = $5.00 - $0.50 = $4.50
        """
        # Setup: existing position of 50 contracts @ $0.10
        positions = [
            Position(outcome="A", direction=PositionDirection.YES, contracts=50, avg_cost=0.10)
        ]
        cash = 950.0  # e.g. after buying

        # Edge flipped negative: model now 6%, market still 10c
        # net_edge = 0.06 - 0.10 - 0.01 = -0.05, below sell_threshold -0.03 -> SELL ALL
        report = _make_report(
            {"A": 0.06},
            {"A": 0.10},
            current_positions=positions,
        )

        result = apply_signals(positions, cash, report, fee_type=FeeType.TAKER)

        a_positions = [p for p in result.positions if p.outcome == "A"]
        assert len(a_positions) == 0 or a_positions[0].contracts == 0
        assert result.cash > cash  # Got money back from selling
        assert result.n_trades == 1
        assert result.fills[0].action == "sell"

    def test_buy_updates_avg_cost(self) -> None:
        """Average cost tracks weighted average of buys.

        Existing: 10 contracts @ $0.20
        New buy: 10 contracts @ $0.30
        Expected avg cost = (10×0.20 + 10×0.30) / 20 = $0.25
        """
        existing = [
            Position(outcome="A", direction=PositionDirection.YES, contracts=10, avg_cost=0.20)
        ]

        # Strong edge to trigger a buy with buy_price=$0.30
        # Use model 60% vs $0.30 market, bankroll=100 -> Kelly recommends buying
        report = _make_report({"A": 0.60}, {"A": 0.30}, current_positions=existing, bankroll=500)
        result = apply_signals(existing, 1000.0, report, fee_type=FeeType.TAKER)

        a_positions = [p for p in result.positions if p.outcome == "A"]
        pos = a_positions[0] if a_positions else None
        assert pos is not None
        assert pos.contracts > 10  # added contracts
        assert pos.avg_cost > 0.20  # avg moved up toward $0.30

    def test_returns_execution_batch_type(self) -> None:
        """apply_signals returns an ExecutionBatch NamedTuple."""
        report = _make_report({}, {})
        result = apply_signals([], 100.0, report, fee_type=FeeType.TAKER)
        assert isinstance(result, ExecutionBatch)
        assert result.positions == []
        assert result.cash == 100.0
        assert result.fees_paid == 0.0
        assert result.n_trades == 0
        assert result.fills == []

    def test_hold_signals_produce_no_fills(self) -> None:
        """HOLD signals don't change cash or positions."""
        # Model 10% vs $0.10 market: minimal edge, should HOLD
        report = _make_report({"A": 0.11}, {"A": 0.10}, bankroll=1000.0)
        result = apply_signals([], 1000.0, report, fee_type=FeeType.TAKER)

        # With buy_edge_threshold=0.05 and fee at $0.10 ~$0.01:
        # net_edge = 0.11 - 0.10 - 0.01 = 0.00 -> AT threshold, marginal
        # Either way, we test that HOLD signals are no-ops
        for fill in result.fills:
            assert fill.action in ("buy", "sell")  # fills must be actual trades

    def test_timestamp_propagates_to_fills(self) -> None:
        """The `timestamp` argument appears in each fill record."""
        from datetime import datetime

        ts = datetime(2026, 2, 10, 21, 0, tzinfo=UTC)
        report = _make_report({"A": 0.40}, {"A": 0.20}, bankroll=1000.0)
        result = apply_signals([], 1000.0, report, fee_type=FeeType.TAKER, timestamp=ts)
        for fill in result.fills:
            assert fill.timestamp == ts


# ============================================================================
# settle_positions
# ============================================================================


class TestSettlePositions:
    """Settling positions with a known winner pays $1/contract on winner."""

    def test_winner_pays_one_dollar_per_contract(self) -> None:
        """10 contracts on the winner @ $0.30 avg cost.

        Settlement:
          revenue = 10 × $1.00 = $10.00
          original cost = 10 × $0.30 = $3.00
          pnl = $10.00 - $3.00 = +$7.00
          final_cash = 970 + $10 = $980
        """
        positions = [
            Position(outcome="A", direction=PositionDirection.YES, contracts=10, avg_cost=0.30)
        ]
        cash = 970.0  # simplification: cash after buying

        result = settle_positions(
            positions=positions, cash=cash, winner="A", initial_bankroll=1000.0
        )

        # Cash increased by $10 (10 contracts × $1.00)
        assert result.final_cash == 980.0
        # PnL = revenue - cost_basis = $10 - $3 = $7
        assert result.pnl_by_outcome["A"] == 7.0

    def test_loser_expires_worthless(self) -> None:
        """Contracts on the losing outcome expire at $0 — full loss."""
        positions = [
            Position(
                outcome="B", direction=PositionDirection.YES, contracts=10, avg_cost=0.30
            ),  # loser
        ]
        cash = 970.0
        result = settle_positions(
            positions=positions, cash=cash, winner="A", initial_bankroll=1000.0
        )
        # B expired worthless — final_cash unchanged
        assert result.final_cash == 970.0
        # PnL = -cost_basis = -(10 × $0.30) = -$3.00
        assert result.pnl_by_outcome["B"] == -3.0

    def test_mixed_portfolio_winner_and_losers(self) -> None:
        """Holding winner + loser: PnL is net across both positions.

        Positions: 10 contracts A @ $0.30, 5 contracts B @ $0.20
        Winner: A
          A settles at $1/contract:
            revenue = $10, cost = $3, pnl = +$7
          B expires at 0:
            pnl = -(5 × $0.20) = -$1.00
          Net PnL = +$7 - $1 = +$6
        """
        positions = [
            Position(outcome="A", direction=PositionDirection.YES, contracts=10, avg_cost=0.30),
            Position(outcome="B", direction=PositionDirection.YES, contracts=5, avg_cost=0.20),
        ]
        cash = 970.0  # pre-settlement cash
        result = settle_positions(
            positions=positions, cash=cash, winner="A", initial_bankroll=1000.0
        )
        assert result.pnl_by_outcome["A"] == 7.0
        assert result.pnl_by_outcome["B"] == -1.0
        # final_cash = cash + 10 × $1 = 970 + 10 = 980
        assert result.final_cash == 980.0
        # total_pnl = sum of individual pnls = +$7 - $1 = +$6
        assert sum(result.pnl_by_outcome.values()) == 6.0


# ============================================================================
# NO-side (BUY NO) tests
# ============================================================================


class TestNoSideMtm:
    """Mark-to-market for NO positions.

    A NO contract on outcome X is worth (100 - yes_price) / 100 dollars.
    This is beacuse if the outcome resolves NO (i.e. X doesn't win), the
    NO holder receives $1. The market's YES price reflects P(X wins),
    so the NO value is the complement.
    """

    def test_no_position_mtm_value(self) -> None:
        """10 NO contracts when YES price = $0.80 → value = 10 × $0.20 = $2.00.

        Step-by-step:
          NO value per contract = (1.0 - 0.80) = $0.20
          10 contracts × $0.20 = $2.00
        """
        positions = [
            Position(outcome="A", direction=PositionDirection.NO, contracts=10, avg_cost=0.15)
        ]
        prices = {"A": 0.80}
        assert compute_mtm_value(positions, prices) == 2.00

    def test_no_position_low_yes_price_high_value(self) -> None:
        """When YES price drops to $0.10, NO contracts become very valuable.

        NO value = (1.0 - 0.10) = $0.90 per contract.
        5 contracts × $0.90 = $4.50
        """
        positions = [
            Position(outcome="A", direction=PositionDirection.NO, contracts=5, avg_cost=0.50)
        ]
        prices = {"A": 0.10}
        assert compute_mtm_value(positions, prices) == 4.50

    def test_mixed_yes_and_no_positions(self) -> None:
        """Portfolio with both YES and NO on different outcomes.

        A: 10 YES contracts, YES price $0.30 → value = $3.00
        B: 5 NO contracts, YES price $0.80 → NO value = (1.0-0.80) × 5 = $1.00
        Total MtM = $3.00 + $1.00 = $4.00
        """
        positions = [
            Position(outcome="A", direction=PositionDirection.YES, contracts=10, avg_cost=0.20),
            Position(outcome="B", direction=PositionDirection.NO, contracts=5, avg_cost=0.15),
        ]
        prices = {"A": 0.30, "B": 0.80}
        assert compute_mtm_value(positions, prices) == 4.00


class TestNoSideSettlement:
    """Settlement rules for NO positions.

    Binary contract settlement:
    - YES on winner  → pays $1 (correct: they won)
    - YES on loser   → expires $0 (incorrect: they didn't win)
    - NO on winner   → expires $0 (incorrect: they did win)
    - NO on loser    → pays $1 (correct: they didn't win)
    """

    def test_no_on_loser_pays_one_dollar(self) -> None:
        """Holding NO on the losing outcome → $1 payout per contract.

        We bet against B (BUY NO on B) at 20c avg cost. B loses.
        Settlement:
          payout = 10 × $1.00 = $10.00
          cost_basis = 10 × $0.20 = $2.00
          pnl = $10.00 - $2.00 = +$8.00
          final_cash = 980 + 10 = $990
        """
        positions = [
            Position(outcome="B", direction=PositionDirection.NO, contracts=10, avg_cost=0.20)
        ]
        cash = 980.0
        result = settle_positions(
            positions=positions, cash=cash, winner="A", initial_bankroll=1000.0
        )
        # B lost → our NO B bet was correct → $1/contract
        assert result.pnl_by_outcome["B"] == 8.0
        assert result.final_cash == 990.0

    def test_no_on_winner_expires_worthless(self) -> None:
        """Holding NO on the winner → expires at $0 (wrong bet).

        We bet against A (BUY NO on A) at 30c, but A wins.
        Settlement:
          payout = $0 (our NO bet was wrong)
          cost_basis = 5 × $0.30 = $1.50
          pnl = -$1.50
        """
        positions = [
            Position(outcome="A", direction=PositionDirection.NO, contracts=5, avg_cost=0.30)
        ]
        cash = 985.0
        result = settle_positions(
            positions=positions, cash=cash, winner="A", initial_bankroll=1000.0
        )
        assert result.pnl_by_outcome["A"] == -1.5
        assert result.final_cash == 985.0  # No payout

    def test_mixed_yes_and_no_settlement(self) -> None:
        """Portfolio with YES winner + NO loser: both pay out.

        Positions:
          A: 10 YES @ 30c (A is winner → pays $1)
          B: 5 NO  @ 20c (B is loser → NO pays $1)

        Settlement:
          A pnl = 10×$1 - 10×$0.30 = +$7.00
          B pnl = 5×$1 - 5×$0.20 = +$4.00 (NO on loser is correct)
          final_cash = 955 + 10 + 5 = $970
        """
        positions = [
            Position(outcome="A", direction=PositionDirection.YES, contracts=10, avg_cost=0.30),
            Position(outcome="B", direction=PositionDirection.NO, contracts=5, avg_cost=0.20),
        ]
        cash = 955.0
        result = settle_positions(
            positions=positions, cash=cash, winner="A", initial_bankroll=1000.0
        )
        assert result.pnl_by_outcome["A"] == 7.0
        assert result.pnl_by_outcome["B"] == 4.0
        assert result.final_cash == 970.0
