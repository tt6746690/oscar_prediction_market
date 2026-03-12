"""Portfolio operations: apply trades, settle positions, mark-to-market.

Stateless functions that operate on position lists. Used by both the
backtest engine and (eventually) live trading portfolio management.

Key types:

- ``PortfolioSnapshot`` — portfolio state at a point in time
- ``ExecutionBatch`` — NamedTuple result of applying trade signals

Key functions:

- ``apply_signals()``  — execute trade signals against a position set
- ``settle_positions()`` — resolve all positions when winner is declared
- ``compute_mtm_value()`` — mark-to-market against current prices
"""

from datetime import UTC, datetime
from typing import NamedTuple

from pydantic import BaseModel, Field, computed_field

from oscar_prediction_market.trading.kalshi_client import (
    estimate_fee,
)
from oscar_prediction_market.trading.schema import (
    FeeType,
    Fill,
    Position,
    PositionDirection,
    SettlementResult,
    TradeAction,
)
from oscar_prediction_market.trading.signals import TradeSignal

# ============================================================================
# Portfolio State
# ============================================================================


class PortfolioSnapshot(BaseModel):
    """Portfolio state at a single point in time during a backtest.

    Captures the immutable state of positions, cash, and valuation at one
    timestamp. The backtest engine stores a list of these to reconstruct
    the portfolio trajectory.

    Running totals (fees, trade count) are *not* stored here — they belong
    on ``BacktestResult`` which computes them from the trade log. This avoids
    redundant bookkeeping and potential sync bugs between running totals and
    the trade log.
    """

    model_config = {"extra": "forbid"}

    timestamp: datetime = Field(..., description="UTC datetime of this snapshot")
    positions: list[Position] = Field(
        default_factory=list, description="Open positions at this point in time"
    )
    cash: float = Field(
        default=0,
        description="Available cash in USD after all trades at this snapshot",
    )
    mark_to_market_value: float = Field(
        default=0,
        description="Current value of all open positions at prevailing market prices in USD",
    )

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def total_wealth(self) -> float:
        """Cash + mark-to-market value of positions."""
        return round(self.cash + self.mark_to_market_value, 2)

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def n_positions(self) -> int:
        """Number of outcomes with non-zero positions."""
        return sum(1 for p in self.positions if p.contracts > 0)


class ExecutionBatch(NamedTuple):
    """Result of applying trade signals to a portfolio.

    Returned by ``apply_signals()`` — bundles updated state and execution
    record. A NamedTuple because it's a pure data carrier with no behavior
    or validation beyond what the producing function guarantees.
    """

    positions: list[Position]
    cash: float
    fees_paid: float
    n_trades: int
    fills: list[Fill]


# ============================================================================
# Functions
# ============================================================================


def compute_mtm_value(
    positions: list[Position],
    market_prices: dict[str, float],
) -> float:
    """Mark-to-market: value positions at current market prices.

    - **YES** contracts are worth ``yes_price`` dollars each.
    - **NO** contracts are worth ``1.0 - yes_price`` dollars each.

    ``market_prices`` always contains YES-side prices in dollars (i.e. the
    probability the outcome wins). NO value is derived as the complement.

    Outcomes not in ``market_prices`` are valued at zero.
    """
    total = 0.0
    for pos in positions:
        if pos.contracts <= 0:
            continue
        yes_price = market_prices.get(pos.outcome, 0)
        if pos.direction == PositionDirection.YES:
            total += pos.contracts * yes_price
        else:
            # NO contract value = 1.0 - YES price
            total += pos.contracts * (1.0 - yes_price)
    return round(total, 2)


def apply_signals(
    positions: list[Position],
    cash: float,
    signals: list[TradeSignal],
    fee_type: FeeType,
    timestamp: datetime | None = None,
) -> ExecutionBatch:
    """Apply trade signals to update positions and cash.

    Processes all BUY and SELL signals from the signal list:

    - BUY: subtract cost + fees from cash, add contracts to position
    - SELL: add revenue - fees to cash, reduce position

    Args:
        positions: Current open positions.
        cash: Available cash in USD.
        signals: Trade signals from ``generate_signals()``.
        fee_type: Fee schedule to apply (taker or maker).
        timestamp: Execution timestamp for fills. Falls back to
            ``datetime.min`` if not provided.

    Returns:
        ``ExecutionBatch`` with updated positions, cash, fees, trade count, and fills.
    """
    fill_ts = timestamp or datetime.min.replace(tzinfo=UTC)
    if len({p.outcome for p in positions}) != len(positions):
        raise ValueError("Duplicate outcome in current_positions")
    pos_by_outcome = {p.outcome: p.model_copy() for p in positions}
    fees_paid = 0.0
    n_trades = 0
    fills: list[Fill] = []

    for signal in signals:
        if signal.delta_contracts == 0:
            continue

        outcome = signal.outcome
        pos = pos_by_outcome.get(
            outcome,
            Position(outcome=outcome, direction=signal.direction, contracts=0, avg_cost=0),
        )

        if signal.action == TradeAction.BUY and signal.delta_contracts > 0:
            # Safety: a BUY must not silently overwrite a position with a
            # different direction. Direction flips must go through an explicit
            # SELL-then-BUY sequence emitted by generate_signals.
            if pos.contracts > 0 and pos.direction != signal.direction:
                raise ValueError(
                    f"BUY {signal.direction.value} on '{outcome}' conflicts with existing "
                    f"{pos.direction.value} position ({pos.contracts} contracts). "
                    f"Direction flips require SELL first."
                )
            # Buy: subtract cost from cash, add to position.
            cost = signal.delta_contracts * signal.execution_price
            fee = estimate_fee(
                signal.execution_price,
                fee_type=fee_type,
                n_contracts=signal.delta_contracts,
            )

            # Update average cost
            old_total_cost = pos.contracts * pos.avg_cost
            new_total_cost = old_total_cost + signal.delta_contracts * signal.execution_price
            new_contracts = pos.contracts + signal.delta_contracts
            new_avg = new_total_cost / new_contracts if new_contracts > 0 else 0

            pos = Position(
                outcome=outcome,
                direction=signal.direction,
                contracts=new_contracts,
                avg_cost=round(new_avg, 4),
            )
            cash_delta = -(cost + fee)
            cash += cash_delta
            fees_paid += fee
            n_trades += 1

            fills.append(
                Fill(
                    timestamp=fill_ts,
                    outcome=outcome,
                    direction=signal.direction,
                    action=TradeAction.BUY,
                    contracts=signal.delta_contracts,
                    price=signal.execution_price,
                    fee_dollars=round(fee, 4),
                    cash_delta=round(cash_delta, 4),
                    reason=signal.reason,
                )
            )

        elif signal.action == TradeAction.SELL and signal.delta_contracts < 0:
            # Sell: add revenue to cash, reduce position.
            sell_contracts = abs(signal.delta_contracts)
            revenue = sell_contracts * signal.execution_price
            fee = estimate_fee(
                signal.execution_price,
                fee_type=fee_type,
                n_contracts=sell_contracts,
            )

            remaining = pos.contracts - sell_contracts
            pos = Position(
                outcome=outcome,
                direction=pos.direction,
                contracts=max(0, remaining),
                avg_cost=pos.avg_cost if remaining > 0 else 0,
            )
            cash_delta = revenue - fee
            cash += cash_delta
            fees_paid += fee
            n_trades += 1

            fills.append(
                Fill(
                    timestamp=fill_ts,
                    outcome=outcome,
                    direction=signal.direction,
                    action=TradeAction.SELL,
                    contracts=sell_contracts,
                    price=signal.execution_price,
                    fee_dollars=round(fee, 4),
                    cash_delta=round(cash_delta, 4),
                    reason=signal.reason,
                )
            )

        pos_by_outcome[outcome] = pos

    # Clean up zero positions
    final_positions = [v for v in pos_by_outcome.values() if v.contracts > 0]

    return ExecutionBatch(
        positions=final_positions,
        cash=round(cash, 2),
        fees_paid=round(fees_paid, 2),
        n_trades=n_trades,
        fills=fills,
    )


def settle_positions(
    positions: list[Position],
    cash: float,
    winner: str,
    initial_bankroll: float,
) -> SettlementResult:
    """Settle all positions assuming ``winner`` won.

    Settlement rules for binary contracts:

    - **YES on winner** → pays $1 per contract (correct: they won)
    - **YES on loser**  → expires worthless at $0 (incorrect: they didn't win)
    - **NO on winner**  → expires worthless at $0 (incorrect: they did win)
    - **NO on loser**   → pays $1 per contract (correct: they didn't win)

    P&L = payout − cost_basis for each position, where cost_basis is
    ``pos.outlay_dollars`` (contracts × avg_cost).

    Args:
        positions: Current open positions (each has ``outcome`` and ``direction``).
        cash: Available cash at settlement time.
        winner: Name of the winning outcome.
        initial_bankroll: Starting bankroll used to compute ``return_pct``.

    Returns:
        :class:`SettlementResult` with final cash, total P&L, return %,
        and per-outcome P&L breakdown.
    """
    pnl_by_outcome: dict[str, float] = {}
    final_cash = cash

    for pos in positions:
        if pos.contracts <= 0:
            continue
        cost_basis = pos.outlay_dollars

        # Determine if this position pays out
        is_winner = pos.outcome == winner
        if pos.direction == PositionDirection.YES:
            # YES on winner → $1, YES on loser → $0
            pays_out = is_winner
        else:
            # NO on winner → $0 (they did win), NO on loser → $1 (they didn't)
            pays_out = not is_winner

        if pays_out:
            payout = pos.contracts * 1.0
            pnl = payout - cost_basis
            final_cash += payout
        else:
            pnl = -cost_basis

        pnl_by_outcome[pos.outcome] = round(pnl, 2)

    final_cash = round(final_cash, 2)
    total_pnl = final_cash - initial_bankroll
    return SettlementResult(
        winner=winner,
        initial_bankroll=initial_bankroll,
        final_cash=final_cash,
        total_pnl=round(total_pnl, 2),
        pnl_by_outcome=pnl_by_outcome,
    )
