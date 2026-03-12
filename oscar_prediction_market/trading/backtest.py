"""Backtest engine: simulate trading strategies over historical data.

Core backtest framework for prediction market trading strategies. Contains
configuration, engine, and result types.

Architecture::

    TradingConfig (schema.py)            — strategy params (live + backtest)
    BacktestSimulationConfig             — backtest-only params (spread, bankroll mode)
    BacktestConfig                       — embeds TradingConfig + BacktestSimulationConfig
    MarketSnapshot                       — one decision point: timestamp + predictions + prices
    BacktestEngine                       — stateless simulation runner
    BacktestResult                       — portfolio history, settlements, trade log

The engine is category-agnostic — it works with any set of outcomes
represented as ``dict[str, float]`` keyed by outcome name.

The engine iterates over ``MarketSnapshot`` objects, each representing
a specific UTC timestamp with predictions and prices. This datetime-based
design accurately captures intraday execution (e.g., trading at 12:30 UTC
after a ceremony ends) as well as daily-close trading.

Backtest vs live trading: execution fills at price ± spread; no
orderbook depth constraints; ``spread_penalty`` approximates bid-ask
cost. See docstrings for details.
"""

import logging
import math
from datetime import datetime

from pydantic import BaseModel, Field, computed_field

from oscar_prediction_market.trading.portfolio import (
    PortfolioSnapshot,
    apply_signals,
    compute_mtm_value,
    settle_positions,
)
from oscar_prediction_market.trading.schema import (
    BankrollMode,
    Fill,
    MarketQuotes,
    Position,
    PositionDirection,
    SettlementResult,
    TradingConfig,
)
from oscar_prediction_market.trading.signals import (
    TradeSignal,
    generate_signals,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


class BacktestSimulationConfig(BaseModel):
    """Backtest-only parameters that don't apply to live trading.

    These control simulation mechanics: how to approximate execution
    costs, whether bankroll compounds, and liquidity constraints.

    ``spread_penalty`` approximates the bid-ask spread. In a backtest
    we only have daily close prices (no orderbook), so we penalize execution
    by assuming buys happen at ``close + spread`` and sells at
    ``close - spread``. Typical values: 0.01–0.03 dollars for liquid markets.

    ``bankroll_mode`` controls whether Kelly sizes against the initial bankroll
    (path-independent, good for ablation grids) or current mark-to-market
    wealth (realistic compounding). Fixed mode is the default for experiments
    because it makes results independent of snapshot ordering.
    """

    model_config = {"extra": "forbid"}

    spread_penalty: float = Field(
        ...,
        ge=0,
        description=(
            "One-way spread penalty in dollars. Buys execute at close + spread, "
            "sells at close - spread."
        ),
    )
    bankroll_mode: BankrollMode = Field(
        ...,
        description="'fixed' for path-independent sizing, 'dynamic' for compounding",
    )
    max_contracts_per_day: int | None = Field(
        default=None,
        ge=1,
        description="Cap on total buy contracts per snapshot. None = unlimited.",
    )


class BacktestConfig(BaseModel):
    """Complete configuration for a backtest run.

    Embeds ``TradingConfig`` (strategy parameters shared with live) and
    ``BacktestSimulationConfig`` (backtest-only execution mechanics).
    """

    model_config = {"extra": "forbid"}

    trading: TradingConfig = Field(..., description="Strategy parameters shared with live trading")
    simulation: BacktestSimulationConfig = Field(
        ..., description="Backtest-only simulation mechanics"
    )

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def label(self) -> str:
        """Short label for this config, suitable for display and filenames."""
        kc = self.trading.kelly
        side_str = (
            "yes"
            if self.trading.allowed_directions == frozenset({PositionDirection.YES})
            else "no"
            if self.trading.allowed_directions == frozenset({PositionDirection.NO})
            else "all"
        )
        return (
            f"fee={self.trading.fee_type.value}_kf={kc.kelly_fraction}"
            f"_bet={kc.buy_edge_threshold}_mp={self.trading.min_price}"
            f"_km={kc.kelly_mode.value}_bm={self.simulation.bankroll_mode.value}"
            f"_side={side_str}_lpo={self.trading.limit_price_offset}"
        )


# ============================================================================
# Result
# ============================================================================


class BacktestResult(BaseModel):
    """Complete result of a backtest run.

    Contains the full time series of portfolio snapshots, hypothetical
    settlements for every outcome, and a trade log.

    Key methods:

    - ``settle(winner)`` — retrieve settlement P&L for a specific winner.
    """

    model_config = {"extra": "forbid"}

    initial_bankroll: float = Field(..., description="Starting capital in USD")
    portfolio_history: list[PortfolioSnapshot] = Field(
        default_factory=list,
        description="Time series of portfolio snapshots, one per trading moment",
    )
    settlements: dict[str, SettlementResult] = Field(
        default_factory=dict,
        description="Hypothetical settlement for each outcome as potential winner",
    )
    trade_log: list[Fill] = Field(
        default_factory=list,
        description="Chronological record of all trade executions",
    )

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def final_wealth(self) -> float:
        """Wealth at the last snapshot (before settlement)."""
        if self.portfolio_history:
            return self.portfolio_history[-1].total_wealth
        return self.initial_bankroll

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def total_return_pct(self) -> float:
        """Mark-to-market return percentage (before settlement)."""
        return round((self.final_wealth - self.initial_bankroll) / self.initial_bankroll * 100, 1)

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def total_fees_paid(self) -> float:
        """Total fees paid across all trades, computed from trade log."""
        return round(sum(f.fee_dollars for f in self.trade_log), 2)

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def total_trades(self) -> int:
        """Total number of trades executed, computed from trade log."""
        return len(self.trade_log)

    def settle(self, winner: str) -> SettlementResult:
        """Get the settlement result for a specific winner."""
        if winner not in self.settlements:
            available = sorted(self.settlements.keys())
            raise KeyError(f"Winner '{winner}' not in settlement universe. Available: {available}")
        return self.settlements[winner]


# ============================================================================
# Market Snapshot
# ============================================================================


class MarketSnapshot(BaseModel):
    """A single decision point in the backtest timeline.

    Bundles the UTC timestamp, predictions, and market prices into one
    object. The engine iterates over a sorted list of these — it never
    constructs them internally.

    Construction is the caller's responsibility (see ``build_trading_moments``
    in the experiment's data_prep module). This keeps the engine decoupled
    from signal-delay logic and price-source selection.
    """

    model_config = {"extra": "forbid"}

    timestamp: datetime = Field(..., description="UTC datetime of this snapshot")
    predictions: dict[str, float] = Field(
        ..., description="{outcome: model_probability} for this snapshot"
    )
    prices: dict[str, float] = Field(
        ..., description="{outcome: yes_price} in dollars at this timestamp"
    )


# ============================================================================
# Engine Helpers
# ============================================================================


def _cap_buy_contracts(signals: list[TradeSignal], max_contracts: int) -> list[TradeSignal]:
    """Cap total buy contracts per snapshot and return an updated signal list.

    Scales down all BUY signal targets proportionally if the total buy delta
    exceeds ``max_contracts``. SELL signals are never capped (always allow
    unwinding).

    Since ``delta_contracts`` and ``outlay_dollars`` are computed fields on
    ``TradeSignal``, we only need to update ``target_contracts`` — the
    computed values adjust automatically.

    Returns a new signal list; the original signals are not modified.
    """
    total_buy = sum(s.delta_contracts for s in signals if s.delta_contracts > 0)
    if total_buy <= max_contracts:
        return signals

    scale = max_contracts / total_buy
    new_signals = []
    for signal in signals:
        if signal.delta_contracts > 0:
            scaled_delta = max(0, math.floor(signal.delta_contracts * scale))
            new_target = signal.current_contracts + scaled_delta
            new_signals.append(signal.model_copy(update={"target_contracts": new_target}))
        else:
            new_signals.append(signal)
    return new_signals


# ============================================================================
# Engine
# ============================================================================


class BacktestEngine:
    """Core backtest simulation engine.

    Stateless runner — all state is passed in via ``run()`` arguments and
    returned in ``BacktestResult``. Safe to reuse across parallel workers.

    The simulation loop:

    1. For each ``MarketSnapshot`` (sorted by timestamp):
       a. Read predictions and market prices from the snapshot
       b. Compute current bankroll (fixed or dynamic mode)
       c. Generate trade signals via the edge/kelly/signals pipeline
       d. Apply max_contracts_per_day cap if configured
       e. Execute signals: update positions, cash, record fills
       f. Mark positions to market, record snapshot
    2. Compute hypothetical settlements for every outcome
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    def run(
        self,
        moments: list[MarketSnapshot],
        spread_penalties: dict[str, float] | None = None,
        ticker_map: dict[str, str] | None = None,
    ) -> BacktestResult:
        """Run the backtest simulation.

        Args:
            moments: Chronologically sorted list of ``MarketSnapshot`` objects.
                Each moment bundles a UTC timestamp, predictions, and prices.
            spread_penalties: Optional per-outcome spread penalties in cents.
            ticker_map: Optional {outcome: kalshi_ticker} for labeling.

        Returns:
            BacktestResult with full portfolio history, settlements, and trade log.
        """
        tc = self.config.trading
        sc = self.config.simulation
        initial_bankroll = tc.kelly.bankroll

        positions: list[Position] = []
        cash = initial_bankroll
        portfolio_history: list[PortfolioSnapshot] = []
        all_fills: list[Fill] = []

        if not moments:
            return BacktestResult(
                initial_bankroll=initial_bankroll,
            )

        for moment in moments:
            predictions = moment.predictions
            market_prices = moment.prices

            if not predictions or not market_prices:
                logger.warning("Missing data for %s, skipping", moment.timestamp)
                continue

            # Determine spread penalty
            if spread_penalties:
                avg_spread = sum(
                    spread_penalties.get(t, sc.spread_penalty) for t in predictions
                ) / len(predictions)
            else:
                avg_spread = sc.spread_penalty

            # Determine bankroll for this moment
            if sc.bankroll_mode == BankrollMode.DYNAMIC and portfolio_history:
                mtm = compute_mtm_value(positions, market_prices)
                current_bankroll = cash + mtm
            else:
                current_bankroll = initial_bankroll

            # Build signal config and market quotes
            market_quotes = MarketQuotes.from_close_prices(market_prices, avg_spread)
            signal_config = tc.model_copy(
                update={
                    "kelly": tc.kelly.model_copy(update={"bankroll": max(1.0, current_bankroll)})
                }
            )

            # Generate signals
            signals = generate_signals(
                model_predictions=predictions,
                execution_prices=market_quotes,
                current_positions=positions,
                config=signal_config,
                ticker_map=ticker_map,
            )

            # Apply max_contracts_per_day cap
            if sc.max_contracts_per_day is not None:
                signals = _cap_buy_contracts(signals, sc.max_contracts_per_day)

            # Apply signals
            tr = apply_signals(
                positions, cash, signals, fee_type=tc.fee_type, timestamp=moment.timestamp
            )

            positions = tr.positions
            cash = tr.cash
            all_fills.extend(tr.fills)

            # Mark-to-market
            mtm = compute_mtm_value(positions, market_prices)

            snapshot = PortfolioSnapshot(
                timestamp=moment.timestamp,
                positions=[p.model_copy() for p in positions],
                cash=cash,
                mark_to_market_value=mtm,
            )
            portfolio_history.append(snapshot)

            logger.info(
                "  %s: cash=$%.2f  MtM=$%.2f  wealth=$%.2f  positions=%d",
                moment.timestamp.isoformat(),
                cash,
                mtm,
                snapshot.total_wealth,
                snapshot.n_positions,
            )

        # Compute hypothetical settlements
        settlements = self._compute_settlements(
            positions=positions,
            cash=cash,
            moments=moments,
        )

        return BacktestResult(
            initial_bankroll=initial_bankroll,
            portfolio_history=portfolio_history,
            settlements=settlements,
            trade_log=all_fills,
        )

    def _compute_settlements(
        self,
        positions: list[Position],
        cash: float,
        moments: list[MarketSnapshot],
    ) -> dict[str, SettlementResult]:
        """Compute hypothetical settlement for every possible winner.

        The settlement universe is the union of:
        - All outcomes that appeared in any moment's prices
        - All outcomes that appeared in any moment's predictions
        - All outcomes with open positions

        This prevents KeyError when the actual winner is missing from the
        last snapshot's prices (e.g., due to no candle data at that time).
        """
        if not positions:
            return {}

        # Build comprehensive universe: prices + predictions + positions
        outcomes_universe: set[str] = set()
        for m in moments:
            outcomes_universe.update(m.prices.keys())
            outcomes_universe.update(m.predictions.keys())
        outcomes_universe.update(p.outcome for p in positions if p.contracts > 0)

        settlements: dict[str, SettlementResult] = {}
        initial = self.config.trading.kelly.bankroll

        for outcome in sorted(outcomes_universe):
            settlements[outcome] = settle_positions(
                positions, cash, winner=outcome, initial_bankroll=initial
            )

        return settlements
