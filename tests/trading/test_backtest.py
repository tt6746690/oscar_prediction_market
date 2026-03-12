"""Tests for BacktestEngine: integration tests with synthetic 2-outcome markets.

BacktestEngine is the simulation core. These tests verify:
1. Basic buy-and-hold scenario produces correct fills and trade log
2. Directional spread: buys use close + spread, sells use close - spread
3. _cap_buy_contracts is pure (returns new list, doesn't mutate)
4. ExecutionBatch fills correctly aggregated into BacktestResult.trade_log
5. Multi-snapshot evolution: positions accumulated, then liquidated on edge flip

The engine is category-agnostic - tests use synthetic "A" and "B" outcomes.
"""

from datetime import UTC, datetime

import pytest

from oscar_prediction_market.trading.backtest import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    BacktestSimulationConfig,
    MarketSnapshot,
    _cap_buy_contracts,
)
from oscar_prediction_market.trading.backtest_configs import (
    generate_fast_trading_configs,
    generate_full_trading_configs,
)
from oscar_prediction_market.trading.schema import (
    NEVER_SELL_THRESHOLD,
    BankrollMode,
    FeeType,
    KellyConfig,
    KellyMode,
    PositionDirection,
    TradeAction,
    TradingConfig,
)
from oscar_prediction_market.trading.signals import (
    TradeSignal,
)

# ============================================================================
# Helpers
# ============================================================================


def _make_backtest_config(
    bankroll: float = 1000.0,
    spread_penalty: float = 0.0,
    bankroll_mode: str = "fixed",
    buy_edge_threshold: float = 0.05,
    sell_edge_threshold: float = -0.03,
    max_contracts_per_day: int | None = None,
) -> BacktestConfig:
    return BacktestConfig(
        trading=TradingConfig(
            kelly=KellyConfig(
                bankroll=bankroll,
                kelly_fraction=0.25,
                kelly_mode=KellyMode.MULTI_OUTCOME,
                buy_edge_threshold=buy_edge_threshold,
                max_position_per_outcome=500,
                max_total_exposure=1000,
            ),
            sell_edge_threshold=sell_edge_threshold,
            fee_type=FeeType.TAKER,
            limit_price_offset=0.0,
            min_price=0,
            allowed_directions=frozenset({PositionDirection.YES}),
        ),
        simulation=BacktestSimulationConfig(
            spread_penalty=spread_penalty,
            bankroll_mode=bankroll_mode,  # type: ignore[arg-type]
            max_contracts_per_day=max_contracts_per_day,
        ),
    )


# ============================================================================
# _cap_buy_contracts (pure function)
# ============================================================================


class TestCapBuyContracts:
    """Verify _cap_buy_contracts is a pure function (no mutation)."""

    def _make_signal(
        self,
        outcome: str,
        delta: int,
        price: float = 0.20,
    ) -> TradeSignal:
        """Create a TradeSignal with the given delta.

        Since delta_contracts is a computed field (= target - current),
        we set current_contracts=0 and target_contracts=abs(delta) for buys,
        or current_contracts=abs(delta) and target_contracts=0 for sells.
        """
        if delta > 0:
            action = TradeAction.BUY
            current = 0
            target = delta
        else:
            action = TradeAction.SELL
            current = abs(delta)
            target = 0
        return TradeSignal(
            outcome=outcome,
            ticker="",
            direction=PositionDirection.YES,
            action=action,
            model_prob=0.30,
            execution_price=price,
            net_edge=0.10,
            current_contracts=current,
            target_contracts=target,
            reason="test",
        )

    def test_no_cap_when_under_limit(self) -> None:
        """When total buys <= max_contracts, returns signals unchanged."""
        signals = [self._make_signal("A", 5), self._make_signal("B", 3)]
        capped = _cap_buy_contracts(signals, max_contracts=10)
        # No capping needed: 5+3=8 <= 10
        a = next(s for s in capped if s.outcome == "A")
        b = next(s for s in capped if s.outcome == "B")
        assert a.delta_contracts == 5
        assert b.delta_contracts == 3

    def test_scales_down_buys_proportionally(self) -> None:
        """With max=5 and total buy=10, both deltas should be halved (floor).

        A=6, B=4, total=10, max=5, scale=0.5
          A_scaled = floor(6 * 0.5) = 3
          B_scaled = floor(4 * 0.5) = 2
        """
        signals = [self._make_signal("A", 6), self._make_signal("B", 4)]
        capped = _cap_buy_contracts(signals, max_contracts=5)
        a = next(s for s in capped if s.outcome == "A")
        b = next(s for s in capped if s.outcome == "B")
        assert a.delta_contracts == 3
        assert b.delta_contracts == 2

    def test_sell_signals_never_capped(self) -> None:
        """SELL signals pass through unchanged regardless of cap."""
        signals = [
            self._make_signal("A", 8),  # BUY
            self._make_signal("B", -5),  # SELL
        ]
        capped = _cap_buy_contracts(signals, max_contracts=3)
        sell = next(s for s in capped if s.outcome == "B")
        assert sell.delta_contracts == -5  # not changed

    def test_original_signals_not_mutated(self) -> None:
        """Pure function: original signal list is unchanged after capping."""
        signals = [self._make_signal("A", 10)]
        original_delta = signals[0].delta_contracts

        capped = _cap_buy_contracts(signals, max_contracts=3)

        # Original signals should be unchanged
        assert signals[0].delta_contracts == original_delta
        # Capped has the new value
        assert capped[0].delta_contracts == 3

    def test_returns_new_list_when_capping(self) -> None:
        """_cap_buy_contracts returns a different list when capping occurs."""
        signals = [self._make_signal("A", 10)]
        capped = _cap_buy_contracts(signals, max_contracts=5)
        assert capped is not signals

    def test_returns_same_list_when_no_cap_needed(self) -> None:
        """When no capping needed, returns the original list (no copy overhead)."""
        signals = [self._make_signal("A", 3)]
        capped = _cap_buy_contracts(signals, max_contracts=10)
        assert capped is signals


# ============================================================================
# BacktestEngine (integration tests)
# ============================================================================


class TestBacktestEngine:
    """Integration tests for BacktestEngine.run()."""

    @staticmethod
    def _make_moments(
        data: dict[str, tuple[dict[str, float], dict[str, float]]],
    ) -> list[MarketSnapshot]:
        """Helper: build moments from {date_str: (predictions, prices)} dict."""
        from datetime import date as date_cls

        moments = []
        for date_str, (preds, prices) in sorted(data.items()):
            d = date_cls.fromisoformat(date_str)
            ts = datetime(d.year, d.month, d.day, 21, 0, tzinfo=UTC)
            moments.append(
                MarketSnapshot(
                    timestamp=ts,
                    predictions=preds,
                    prices=prices,
                )
            )
        return moments

    def test_single_snapshot_buy_fills_trade_log(self) -> None:
        """One snapshot with strong model conviction should generate fills.

        Model: A=40%, market $0.20 -> strong edge -> BUY
        Expects at least one fill in trade_log.
        """
        config = _make_backtest_config(bankroll=1000.0, spread_penalty=0.0)
        engine = BacktestEngine(config)

        moments = self._make_moments(
            {"2026-01-10": ({"A": 0.40, "B": 0.05}, {"A": 0.20, "B": 0.05})}
        )
        result = engine.run(moments=moments)

        assert isinstance(result, BacktestResult)
        assert len(result.portfolio_history) == 1
        # Should have bought A (strong edge), maybe not B (minimal edge)
        buys = [f for f in result.trade_log if f.action == "buy"]
        assert len(buys) >= 1
        assert any(f.outcome == "A" for f in buys)

    def test_directional_spread_buys_at_ask_sells_at_bid(self) -> None:
        """Spread is applied directionally: buys at close+spread, sells at close-spread.

        Setup:
          Day 1: A at $0.20, model=40% -> BUY at 0.20+0.02=$0.22
          Day 2: A at $0.20, model=5% -> edge flipped -> SELL at 0.20-0.02=$0.18
          spread = $0.02 each way

        The buy fill should have price = $0.22.
        The sell fill should have price = $0.18.
        """
        config = _make_backtest_config(
            bankroll=1000.0,
            spread_penalty=0.02,
            sell_edge_threshold=-0.01,  # low threshold: sell quickly on edge drop
        )
        engine = BacktestEngine(config)

        moments = self._make_moments(
            {
                "2026-01-10": ({"A": 0.40}, {"A": 0.20}),  # Day 1: strong edge -> BUY
                "2026-01-11": ({"A": 0.05}, {"A": 0.20}),  # Day 2: edge flipped -> SELL
            }
        )
        result = engine.run(moments=moments)

        buys = [f for f in result.trade_log if f.action == "buy"]
        sells = [f for f in result.trade_log if f.action == "sell"]

        assert len(buys) >= 1, "Expected at least one BUY fill"
        assert len(sells) >= 1, "Expected at least one SELL fill"

        # Buy price should be close + spread = $0.22
        buy_fill = next(f for f in buys if f.outcome == "A")
        assert buy_fill.price == pytest.approx(0.22)

        # Sell price should be close - spread = $0.18
        sell_fill = next(f for f in sells if f.outcome == "A")
        assert sell_fill.price == pytest.approx(0.18)

    def test_max_contracts_per_day_caps_buys(self) -> None:
        """max_contracts_per_day limits total buy contracts per snapshot.

        With two outcomes each wanting 10+ contracts and a cap of 5,
        the total across both fills should not exceed 5.
        """
        config = _make_backtest_config(
            bankroll=5000.0,
            spread_penalty=0.0,
            max_contracts_per_day=5,
        )
        engine = BacktestEngine(config)

        moments = self._make_moments(
            {"2026-01-10": ({"A": 0.40, "B": 0.35}, {"A": 0.10, "B": 0.10})}
        )
        result = engine.run(moments=moments)

        total_buys = sum(f.contracts for f in result.trade_log if f.action == "buy")
        assert total_buys <= 5

    def test_empty_predictions_returns_empty_result(self) -> None:
        """No predictions -> no snapshots recorded."""
        config = _make_backtest_config()
        engine = BacktestEngine(config)
        result = engine.run(moments=[])
        assert len(result.portfolio_history) == 0
        assert len(result.trade_log) == 0

    def test_trade_log_aggregates_fills_across_snapshots(self) -> None:
        """Fills from all snapshots are accumulated in trade_log."""
        config = _make_backtest_config(bankroll=1000.0, spread_penalty=0.0)
        engine = BacktestEngine(config)

        moments = self._make_moments(
            {
                "2026-01-10": ({"A": 0.40}, {"A": 0.20}),
                "2026-01-15": ({"A": 0.40}, {"A": 0.20}),  # same strong edge -> may BUY again
            }
        )
        result = engine.run(moments=moments)

        # All fills should have a timestamp field
        for fill in result.trade_log:
            assert fill.timestamp.date().isoformat() in ("2026-01-10", "2026-01-15")

    def test_two_snapshot_trajectory(self) -> None:
        """Full two-day trajectory: buy day 1, hold day 2.

        Day 1: A at 20c, model 40% -> BUY
        Day 2: A at 25c, model 40% (now lower edge) -> probably HOLD
        Should have 2 snapshots and positive MtM on day 2.
        """
        config = _make_backtest_config(bankroll=1000.0, spread_penalty=0.0)
        engine = BacktestEngine(config)

        moments = self._make_moments(
            {
                "2026-01-10": ({"A": 0.40}, {"A": 0.20}),
                "2026-01-11": ({"A": 0.40}, {"A": 0.25}),  # price moved up
            }
        )
        result = engine.run(moments=moments)

        assert len(result.portfolio_history) == 2
        # After day 2, MtM should reflect the price increase
        snap2 = result.portfolio_history[-1]
        assert snap2.mark_to_market_value >= 0


class TestSettleWithMissingPrices:
    """Tests for _compute_settlements universe expansion.

    Bug fix: previously, _compute_settlements only used the last snapshot's
    prices as the settlement universe. If the actual winner dropped out of
    the price feed (no candle data), settle(winner) would raise KeyError.

    Fix: the settlement universe is now the union of all moments' prices,
    predictions, and position outcomes.
    """

    @staticmethod
    def _make_moments(
        data: dict[str, tuple[dict[str, float], dict[str, float]]],
    ) -> list[MarketSnapshot]:
        """Helper: build moments from {date_str: (predictions, prices)} dict."""
        from datetime import date as date_cls

        moments = []
        for date_str, (preds, prices) in sorted(data.items()):
            d = date_cls.fromisoformat(date_str)
            ts = datetime(d.year, d.month, d.day, 21, 0, tzinfo=UTC)
            moments.append(
                MarketSnapshot(
                    timestamp=ts,
                    predictions=preds,
                    prices=prices,
                )
            )
        return moments

    def test_settle_winner_missing_from_last_snapshot_prices(self) -> None:
        """Winner present in predictions but absent from last snapshot's prices.

        Scenario: 3 outcomes A, B, C.
        - Day 1 prices: {A: 0.20, B: 0.15, C: 0.10}  (all visible)
        - Day 2 prices: {A: 0.25, B: 0.18}            (C dropped — no candle)
        - Predictions always include A, B, C

        Before the fix, settle("C") would raise KeyError. Now it should
        succeed because C is in the predictions and day 1 prices.

        With model prob 50% vs 10% price, A has huge edge → engine buys A.
        Settling with winner="C" → A loses, net loss.
        Settling with winner="A" → A pays out, net profit.
        Both should work without KeyError.
        """
        config = _make_backtest_config(bankroll=1000.0, spread_penalty=0.0)
        engine = BacktestEngine(config)

        moments = self._make_moments(
            {
                "2026-01-10": (
                    {"A": 0.50, "B": 0.25, "C": 0.10},
                    {"A": 0.20, "B": 0.15, "C": 0.10},
                ),
                "2026-01-11": (
                    {"A": 0.50, "B": 0.25, "C": 0.10},
                    {"A": 0.25, "B": 0.18},  # C removed from prices
                ),
            }
        )
        result = engine.run(moments=moments)

        # C should be in the settlement universe despite missing from last prices
        assert "C" in result.settlements
        assert "A" in result.settlements
        assert "B" in result.settlements

        # settle("C") should not raise
        settle_c = result.settle("C")
        assert settle_c.total_pnl is not None  # any value, just no error

        # settle("A") should produce profit (bought A at 20c with 50% model prob)
        settle_a = result.settle("A")
        assert settle_a.total_pnl > 0

    def test_settle_winner_only_in_predictions(self) -> None:
        """Winner never had price data — only appears in predictions.

        If a nominee was in predictions but never appeared in any price
        snapshot, it should still be in the settlement universe (from
        the predictions union).
        """
        config = _make_backtest_config(bankroll=1000.0, spread_penalty=0.0)
        engine = BacktestEngine(config)

        moments = self._make_moments(
            {
                "2026-01-10": (
                    {"A": 0.50, "B": 0.25, "D": 0.10},  # D in preds
                    {"A": 0.20, "B": 0.15},  # D never in prices
                ),
            }
        )
        result = engine.run(moments=moments)

        # D should be in settlement universe from predictions
        assert "D" in result.settlements
        settle_d = result.settle("D")
        # A was bought, D won → A loses, so PnL is negative
        assert settle_d.total_pnl <= 0


# ============================================================================
# BacktestConfig: label + shared grid generation
# ============================================================================


class TestBacktestConfigLabel:
    """Tests for the label computed field on BacktestConfig.

    BacktestConfig.label produces a compact human-readable string encoding
    all key parameters.  This is the same format previously generated by
    BacktestGridConfig.label, now moved directly onto BacktestConfig.
    """

    def test_label_encodes_all_dimensions(self) -> None:
        """Label includes fee, kelly_fraction, edge, min_price, kelly_mode,
        bankroll_mode, and allowed_directions.

        Using the helper ``_make_backtest_config`` which builds:
          fee=taker, kf=0.25, bet=0.05, mp=0, km=multi_outcome, bm=fixed, side=yes
        """
        cfg = _make_backtest_config(bankroll=1000.0)
        label = cfg.label
        assert "fee=taker" in label
        assert "kf=0.25" in label
        assert "bet=0.05" in label
        assert "mp=0" in label
        assert "km=multi_outcome" in label
        assert "bm=fixed" in label
        assert "side=yes" in label

    def test_label_yes_only(self) -> None:
        """YES-only directions produce side=yes."""
        config = BacktestConfig(
            trading=TradingConfig(
                kelly=KellyConfig(
                    bankroll=100,
                    kelly_fraction=0.1,
                    kelly_mode=KellyMode.INDEPENDENT,
                    buy_edge_threshold=0.05,
                    max_position_per_outcome=50,
                    max_total_exposure=100,
                ),
                sell_edge_threshold=-1.0,
                fee_type=FeeType.MAKER,
                limit_price_offset=0.0,
                min_price=0,
                allowed_directions=frozenset({PositionDirection.YES}),
            ),
            simulation=BacktestSimulationConfig(
                spread_penalty=0.0,
                bankroll_mode="fixed",  # type: ignore[arg-type]
            ),
        )
        assert config.label.endswith("side=yes_lpo=0.0")

    def test_label_no_only(self) -> None:
        """NO-only directions produce side=no."""
        config = BacktestConfig(
            trading=TradingConfig(
                kelly=KellyConfig(
                    bankroll=100,
                    kelly_fraction=0.1,
                    kelly_mode=KellyMode.INDEPENDENT,
                    buy_edge_threshold=0.05,
                    max_position_per_outcome=50,
                    max_total_exposure=100,
                ),
                sell_edge_threshold=-1.0,
                fee_type=FeeType.MAKER,
                limit_price_offset=0.0,
                min_price=0,
                allowed_directions=frozenset({PositionDirection.NO}),
            ),
            simulation=BacktestSimulationConfig(
                spread_penalty=0.0,
                bankroll_mode="fixed",  # type: ignore[arg-type]
            ),
        )
        assert config.label.endswith("side=no_lpo=0.0")

    def test_label_all_directions(self) -> None:
        """Both YES+NO directions produce side=all."""
        config = BacktestConfig(
            trading=TradingConfig(
                kelly=KellyConfig(
                    bankroll=100,
                    kelly_fraction=0.1,
                    kelly_mode=KellyMode.INDEPENDENT,
                    buy_edge_threshold=0.05,
                    max_position_per_outcome=50,
                    max_total_exposure=100,
                ),
                sell_edge_threshold=-1.0,
                fee_type=FeeType.MAKER,
                limit_price_offset=0.0,
                min_price=0,
                allowed_directions=frozenset({PositionDirection.YES, PositionDirection.NO}),
            ),
            simulation=BacktestSimulationConfig(
                spread_penalty=0.0,
                bankroll_mode="fixed",  # type: ignore[arg-type]
            ),
        )
        assert config.label.endswith("side=all_lpo=0.0")


class TestBacktestConfigGridGeneration:
    """Tests for shared backtest config grid builders."""

    def test_generate_fast_trading_configs(self) -> None:
        configs = generate_fast_trading_configs(bankroll=2000.0, spread_penalty=0.03)

        assert len(configs) == 18
        assert all(cfg.trading.kelly.bankroll == 2000.0 for cfg in configs)
        assert all(cfg.simulation.spread_penalty == 0.03 for cfg in configs)
        assert all(cfg.trading.sell_edge_threshold == NEVER_SELL_THRESHOLD for cfg in configs)
        assert all(cfg.trading.fee_type == FeeType.MAKER for cfg in configs)
        assert all(
            cfg.trading.allowed_directions == frozenset({PositionDirection.YES}) for cfg in configs
        )
        assert {cfg.trading.kelly.kelly_fraction for cfg in configs} == {0.10, 0.20, 0.35}
        assert {cfg.trading.kelly.buy_edge_threshold for cfg in configs} == {0.05, 0.08, 0.12}

    def test_generate_full_trading_configs(self) -> None:
        configs = generate_full_trading_configs(bankroll=1000.0, spread_penalty=0.02)

        assert len(configs) == 588
        sample = configs[0]
        assert sample.trading.kelly.bankroll == 1000.0
        assert sample.trading.kelly.max_position_per_outcome == 500.0
        assert sample.trading.kelly.max_total_exposure == 1000.0
        assert sample.trading.sell_edge_threshold == NEVER_SELL_THRESHOLD
        assert sample.simulation.spread_penalty == 0.02
        assert sample.simulation.bankroll_mode == BankrollMode.FIXED
