"""Tests for signal generation (integration tests).

The signal pipeline orchestrates: edge -> Kelly -> position delta.
Input: model predictions, market prices, current positions
Output: BUY/SELL/HOLD signals with exact contract deltas

These tests use scenarios inspired by the trade signal backtest experiment
(d20260214) where the pipeline was run across 10 temporal snapshots.
"""

from oscar_prediction_market.trading.schema import (
    FeeType,
    KellyConfig,
    KellyMode,
    MarketQuotes,
    Position,
    PositionDirection,
    TradeAction,
    TradingConfig,
)
from oscar_prediction_market.trading.signals import (
    generate_signals,
)

# ============================================================================
# Section 6: Signal Generation (Integration Tests)
#
# The signal pipeline orchestrates: edge -> Kelly -> position delta.
# Input: model predictions, market prices, current positions
# Output: BUY/SELL/HOLD signals with exact contract deltas
#
# These tests use scenarios inspired by the trade signal backtest experiment
# (d20260214) where the pipeline was run across 10 temporal snapshots.
# ============================================================================


class TestGenerateSignals:
    """Integration tests that trace the full signal pipeline."""

    def test_fresh_portfolio_with_clear_edge(self) -> None:
        """Starting from zero positions with strong model conviction.

        Inspired by the early backtest snapshots (Dec 2025) where the
        model identified edge on long-shot outcomes like Marty Supreme
        (model 30% vs market 10c) and Sinners (model 15% vs market 5c).

        Expected: BUY signals for outcomes with net_edge > buy_edge_threshold,
        HOLD/no action for the rest.
        """
        model_predictions = {
            "Frontrunner": 0.40,
            "Contender": 0.25,
            "Longshot": 0.08,
        }
        market_prices = {
            "Frontrunner": 0.35,  # model agrees with market -> little edge
            "Contender": 0.10,  # model >> market -> edge
            "Longshot": 0.05,  # model slightly above market -> too thin after fees
        }
        config = TradingConfig(
            kelly=KellyConfig(
                bankroll=1000,
                kelly_fraction=0.25,
                kelly_mode=KellyMode.MULTI_OUTCOME,
                buy_edge_threshold=0.05,
                max_position_per_outcome=250,
                max_total_exposure=500,
            ),
            sell_edge_threshold=-0.03,
            fee_type=FeeType.TAKER,
            limit_price_offset=0.0,
            min_price=0,
            allowed_directions=frozenset({PositionDirection.YES}),
        )

        report = generate_signals(
            model_predictions=model_predictions,
            execution_prices=MarketQuotes.from_close_prices(market_prices),
            current_positions=[],
            config=config,
        )

        assert isinstance(report, list)
        assert len(report) == 3

        signal_map = {s.outcome: s for s in report}

        # Contender has large edge (model 25% vs market $0.10) -> BUY
        contender = signal_map["Contender"]
        assert contender.action == "buy"
        assert contender.delta_contracts > 0
        assert contender.net_edge > 0.05

        # Longshot: model 8% vs $0.05 -> gross 3pp, fee at $0.05 = $0.01, net = 2pp.
        # Below buy_edge_threshold=0.05 -> HOLD.
        longshot = signal_map["Longshot"]
        assert longshot.action == "hold"
        assert longshot.delta_contracts == 0

    def test_sell_signal_when_edge_flips(self) -> None:
        """When a held position's edge drops below sell threshold, generate SELL.

        Scenario: We bought Falling Star at $0.10 when model said 25%.
        Now model updated to 6%, market still at $0.10.
        Fee at $0.10 = ceil(0.07 × 0.10 × 0.90 × 100) / 100 = $0.01.
        net_edge = 0.06 - 0.10 - 0.01 = -0.05, below sell_threshold of -0.03.
        -> SELL entire position.

        This mimics the backtest where early positions were sold after model
        re-estimation with new data.
        """
        model_predictions = {"Falling Star": 0.06}
        market_prices = {"Falling Star": 0.10}
        current_positions = [
            Position(
                outcome="Falling Star",
                direction=PositionDirection.YES,
                contracts=50,
                avg_cost=0.10,
            ),
        ]
        config = TradingConfig(
            kelly=KellyConfig(
                bankroll=1000,
                kelly_fraction=0.25,
                kelly_mode=KellyMode.MULTI_OUTCOME,
                buy_edge_threshold=0.05,
                max_position_per_outcome=250,
                max_total_exposure=500,
            ),
            sell_edge_threshold=-0.03,
            fee_type=FeeType.TAKER,
            limit_price_offset=0.0,
            min_price=0,
            allowed_directions=frozenset({PositionDirection.YES}),
        )

        report = generate_signals(
            model_predictions=model_predictions,
            execution_prices=MarketQuotes.from_close_prices(market_prices),
            current_positions=current_positions,
            config=config,
        )

        signal = report[0]
        assert signal.action == "sell"
        assert signal.delta_contracts == -50  # Sell entire position
        assert signal.current_contracts == 50
        assert signal.target_contracts == 0

    def test_reduce_position_when_edge_drops_below_buy_threshold(self) -> None:
        """When edge drops below buy_edge_threshold, Kelly targets 0 -> SELL to reduce.

        Scenario: We hold 20 contracts at $0.12. Edge has dropped:
          model 22%, market $0.15.
          Fee at $0.15 = ceil(0.07 × 0.15 × 0.85 × 100) / 100 = $0.01.
          net_edge = 0.22 - 0.15 - 0.01 = 0.06
          This is above sell_edge_threshold (-0.03), but below buy_edge_threshold (0.10).

        Because Kelly recommends 0 contracts (edge insufficient), the target
        is 0 and delta is -20 -> SELL with reason "reducing position."

        This is the correct behavior: if you wouldn't initiate a NEW position
        at this edge level, you should exit the existing one too.
        """
        model_predictions = {"Middling": 0.22}
        market_prices = {"Middling": 0.15}
        current_positions = [
            Position(
                outcome="Middling", direction=PositionDirection.YES, contracts=20, avg_cost=0.12
            ),
        ]
        config = TradingConfig(
            kelly=KellyConfig(
                bankroll=1000,
                kelly_fraction=0.25,
                kelly_mode=KellyMode.MULTI_OUTCOME,
                buy_edge_threshold=0.10,  # High threshold
                max_position_per_outcome=250,
                max_total_exposure=500,
            ),
            sell_edge_threshold=-0.03,
            fee_type=FeeType.TAKER,
            limit_price_offset=0.0,
            min_price=0,
            allowed_directions=frozenset({PositionDirection.YES}),
        )

        report = generate_signals(
            model_predictions=model_predictions,
            execution_prices=MarketQuotes.from_close_prices(market_prices),
            current_positions=current_positions,
            config=config,
        )

        signal = report[0]
        assert signal.action == "sell"
        assert signal.delta_contracts == -20  # Sell entire position
        assert signal.current_contracts == 20
        assert signal.target_contracts == 0

    def test_no_positions_no_predictions_empty_report(self) -> None:
        """Empty inputs produce empty report."""
        report = generate_signals(
            model_predictions={},
            execution_prices=MarketQuotes.from_close_prices({}),
            current_positions=[],
            config=TradingConfig(
                kelly=KellyConfig(
                    bankroll=1000,
                    kelly_fraction=0.25,
                    kelly_mode=KellyMode.MULTI_OUTCOME,
                    buy_edge_threshold=0.05,
                    max_position_per_outcome=250,
                    max_total_exposure=500,
                ),
                sell_edge_threshold=-0.03,
                fee_type=FeeType.TAKER,
                limit_price_offset=0.0,
                min_price=0,
                allowed_directions=frozenset({PositionDirection.YES}),
            ),
        )

        assert report == []

    def test_multi_outcome_portfolio_construction(self) -> None:
        """Build a portfolio across multiple outcomes with varying edge.

        Inspired by the average model backtest, which held 2-3 positions
        simultaneously across different Oscar BP outcomes. Verifies that:
        - Multiple BUY signals can coexist
        - Total outlay respects the exposure cap
        - Signals are sorted by action priority (BUY first, then HOLD)
        """
        model_predictions = {
            "Strong A": 0.35,
            "Strong B": 0.30,
            "Weak C": 0.06,
        }
        market_prices = {
            "Strong A": 0.12,
            "Strong B": 0.10,
            "Weak C": 0.05,
        }
        config = TradingConfig(
            kelly=KellyConfig(
                bankroll=1000,
                kelly_fraction=0.25,
                kelly_mode=KellyMode.MULTI_OUTCOME,
                buy_edge_threshold=0.05,
                max_position_per_outcome=250,
                max_total_exposure=300,
            ),
            sell_edge_threshold=-0.03,
            fee_type=FeeType.TAKER,
            limit_price_offset=0.0,
            min_price=0,
            allowed_directions=frozenset({PositionDirection.YES}),
        )

        report = generate_signals(
            model_predictions=model_predictions,
            execution_prices=MarketQuotes.from_close_prices(market_prices),
            current_positions=[],
            config=config,
        )

        buys = [s for s in report if s.action == "buy"]
        assert len(buys) >= 1  # At least one buy signal

        # Total target outlay should respect exposure cap
        total_target_outlay = sum(s.target_contracts * s.execution_price for s in buys)
        assert total_target_outlay <= 300 + 1  # Small rounding tolerance

        # Signals sorted: BUY before HOLD
        actions = [s.action for s in report]
        if TradeAction.BUY in actions and TradeAction.HOLD in actions:
            first_buy = actions.index(TradeAction.BUY)
            last_hold = len(actions) - 1 - actions[::-1].index(TradeAction.HOLD)
            assert first_buy < last_hold

    def test_spread_penalty_reduces_signals(self) -> None:
        """Adding a spread penalty can eliminate marginal buy signals.

        A outcome that has enough edge without spread penalty may not
        survive once we account for realistic bid-ask costs.
        """
        model_predictions = {"Marginal": 0.20}
        market_prices = {"Marginal": 0.12}

        config_no_spread = TradingConfig(
            kelly=KellyConfig(
                bankroll=1000,
                kelly_fraction=0.25,
                kelly_mode=KellyMode.MULTI_OUTCOME,
                buy_edge_threshold=0.05,
                max_position_per_outcome=250,
                max_total_exposure=500,
            ),
            sell_edge_threshold=-0.03,
            fee_type=FeeType.TAKER,
            limit_price_offset=0.0,
            min_price=0,
            allowed_directions=frozenset({PositionDirection.YES}),
        )
        config_with_spread = TradingConfig(
            kelly=KellyConfig(
                bankroll=1000,
                kelly_fraction=0.25,
                kelly_mode=KellyMode.MULTI_OUTCOME,
                buy_edge_threshold=0.05,
                max_position_per_outcome=250,
                max_total_exposure=500,
            ),
            sell_edge_threshold=-0.03,
            fee_type=FeeType.TAKER,
            limit_price_offset=0.0,
            min_price=0,
            allowed_directions=frozenset({PositionDirection.YES}),
        )

        report_no_spread = generate_signals(
            model_predictions=model_predictions,
            execution_prices=MarketQuotes.from_close_prices(market_prices, spread=0),
            current_positions=[],
            config=config_no_spread,
        )
        report_with_spread = generate_signals(
            model_predictions=model_predictions,
            execution_prices=MarketQuotes.from_close_prices(market_prices, spread=0.04),
            current_positions=[],
            config=config_with_spread,
        )

        # With spread penalty, the edge is reduced
        sig_no = report_no_spread[0]
        sig_with = report_with_spread[0]
        assert sig_with.net_edge < sig_no.net_edge

        # The spread version should have fewer or smaller buys
        no_spread_contracts = sum(s.delta_contracts for s in report_no_spread if s.action == "buy")
        with_spread_contracts = sum(
            s.delta_contracts for s in report_with_spread if s.action == "buy"
        )
        assert with_spread_contracts <= no_spread_contracts


class TestNoSideSignals:
    """Tests for NO-side (BUY NO) signal generation.

    On Kalshi, buying NO on outcome X is a bet *against* X. The NO price
    is the complement of the YES price: ``no_ask = 100 - yes_bid``.

    The model produces YES probabilities (prob the nominee wins). A BUY NO
    signal appears when model_prob is low enough that ``(1 - model_prob)``
    exceeds the NO ask price by more than the buy edge threshold + fees.
    """

    def test_no_buy_signal_for_overpriced_outcome(self) -> None:
        """When an outcome is overpriced (model << market), buy NO.

        Scenario: market prices Frontrunner at 90c (YES), model says 60%.
        - YES edge: 0.60 - 0.90 - fee = deeply negative → no YES buy.
        - NO model prob: 1 - 0.60 = 0.40
        - NO ask price: 100 - yes_bid. With yes_bid ~ 90c:
            no_ask = 100 - 90 = 10c.
        - NO edge: 0.40 - 0.10 - fee = large positive → BUY NO.

        This is the canonical "fade the favorite" trade.
        """
        model_predictions = {"Frontrunner": 0.60}
        market_prices = {"Frontrunner": 0.90}
        config = TradingConfig(
            kelly=KellyConfig(
                bankroll=1000,
                kelly_fraction=0.25,
                kelly_mode=KellyMode.MULTI_OUTCOME,
                buy_edge_threshold=0.05,
                max_position_per_outcome=250,
                max_total_exposure=500,
            ),
            sell_edge_threshold=-0.03,
            fee_type=FeeType.TAKER,
            limit_price_offset=0.0,
            min_price=0,
            allowed_directions=frozenset({PositionDirection.YES, PositionDirection.NO}),
        )

        report = generate_signals(
            model_predictions=model_predictions,
            execution_prices=MarketQuotes.from_close_prices(market_prices),
            current_positions=[],
            config=config,
        )

        buys = [s for s in report if s.action == TradeAction.BUY]
        assert len(buys) == 1
        buy = buys[0]
        assert buy.direction == PositionDirection.NO
        assert buy.delta_contracts > 0
        assert buy.net_edge > 0.05

    def test_yes_only_mode_ignores_no_edge(self) -> None:
        """With TradingSide.YES, NO edges are never computed.

        Same scenario as above (Frontrunner overpriced), but trading_side=YES.
        The NO edge exists but is suppressed by the config.
        """
        model_predictions = {"Frontrunner": 0.60}
        market_prices = {"Frontrunner": 0.90}
        config = TradingConfig(
            kelly=KellyConfig(
                bankroll=1000,
                kelly_fraction=0.25,
                kelly_mode=KellyMode.MULTI_OUTCOME,
                buy_edge_threshold=0.05,
                max_position_per_outcome=250,
                max_total_exposure=500,
            ),
            sell_edge_threshold=-0.03,
            fee_type=FeeType.TAKER,
            limit_price_offset=0.0,
            min_price=0,
            allowed_directions=frozenset({PositionDirection.YES}),
        )

        report = generate_signals(
            model_predictions=model_predictions,
            execution_prices=MarketQuotes.from_close_prices(market_prices),
            current_positions=[],
            config=config,
        )

        # No BUY signals — YES edge is negative, NO edge suppressed
        buys = [s for s in report if s.action == TradeAction.BUY]
        assert len(buys) == 0

    def test_direction_flip_yes_to_no(self) -> None:
        """When model conviction flips, generate SELL YES + BUY NO.

        Scenario: We hold YES at 20c. Model updated: now only 10% likely.
        Market still at 20c.

        YES edge = 0.10 - 0.20 - fee = deeply negative → Kelly targets 0 YES.
        NO model prob = 0.90, NO ask = 100 - 20 = 80c → large positive edge.
        Kelly may or may not allocate NO depending on optimizer behavior.

        The essential behavior: the YES position is fully liquidated because
        the YES edge is far below sell_edge_threshold.
        """
        model_predictions = {"Flip": 0.10}
        market_prices = {"Flip": 0.20}
        current_positions = [
            Position(outcome="Flip", direction=PositionDirection.YES, contracts=30, avg_cost=0.20),
        ]
        config = TradingConfig(
            kelly=KellyConfig(
                bankroll=1000,
                kelly_fraction=0.25,
                kelly_mode=KellyMode.MULTI_OUTCOME,
                buy_edge_threshold=0.05,
                max_position_per_outcome=250,
                max_total_exposure=500,
            ),
            sell_edge_threshold=-0.03,
            fee_type=FeeType.TAKER,
            limit_price_offset=0.0,
            min_price=0,
            allowed_directions=frozenset({PositionDirection.YES, PositionDirection.NO}),
        )

        report = generate_signals(
            model_predictions=model_predictions,
            execution_prices=MarketQuotes.from_close_prices(market_prices),
            current_positions=current_positions,
            config=config,
        )

        sells = [s for s in report if s.action == TradeAction.SELL]

        # Must have a SELL for the YES position (edge deeply negative)
        assert len(sells) == 1
        assert sells[0].direction == PositionDirection.YES
        assert sells[0].delta_contracts == -30  # Sell all

    def test_sell_no_position_when_edge_flips(self) -> None:
        """Holding NO, edge drops below sell threshold → SELL NO.

        Scenario: We hold NO at 10c (we bet against Longshot).
        Now model says Longshot is 95% likely → NO model prob = 5%.
        NO sell price (= no_bid) ≈ 10c - spread ≈ 10c (no spread here).
        net_edge for NO = 0.05 - 0.10 - fee → negative → below sell_threshold → SELL.
        """
        model_predictions = {"Longshot": 0.95}
        market_prices = {"Longshot": 0.90}
        current_positions = [
            Position(
                outcome="Longshot", direction=PositionDirection.NO, contracts=50, avg_cost=0.10
            ),
        ]
        config = TradingConfig(
            kelly=KellyConfig(
                bankroll=1000,
                kelly_fraction=0.25,
                kelly_mode=KellyMode.MULTI_OUTCOME,
                buy_edge_threshold=0.05,
                max_position_per_outcome=250,
                max_total_exposure=500,
            ),
            sell_edge_threshold=-0.03,
            fee_type=FeeType.TAKER,
            limit_price_offset=0.0,
            min_price=0,
            allowed_directions=frozenset({PositionDirection.YES, PositionDirection.NO}),
        )

        report = generate_signals(
            model_predictions=model_predictions,
            execution_prices=MarketQuotes.from_close_prices(market_prices),
            current_positions=current_positions,
            config=config,
        )

        sells = [s for s in report if s.action == TradeAction.SELL]
        assert len(sells) == 1
        assert sells[0].direction == PositionDirection.NO
        assert sells[0].delta_contracts == -50
