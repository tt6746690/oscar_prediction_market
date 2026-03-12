"""Inspect individual trades from a backtest result.

Given a ``BacktestResult``, formats a human-readable trade log showing
each fill with its edge, cost, and eventual P&L after settlement.
Useful for debugging why specific trades were made and which contributed
to overall profit/loss.

Usage (in a one-off script or notebook)::

    result = BacktestEngine(config).run(predictions, prices)
    settlement = result.settle(winner_model_name)
    print(format_trade_log(result, settlement))

    # Or for a quick summary table:
    df = trade_log_to_dataframe(result, winner_model_name)
"""

import pandas as pd

from oscar_prediction_market.trading.backtest import BacktestResult
from oscar_prediction_market.trading.schema import (
    Fill,
    PositionDirection,
    SettlementResult,
    TradeAction,
)


def _fill_pnl(fill: Fill, winner: str) -> float:
    """Compute settlement P&L of a single BUY fill given the known winner.

    SELL fills are ignored (return 0.0) because the P&L from a position
    that was sold before settlement is already captured in the cash_delta
    of the SELL fill itself. Computing settlement-style P&L for exited
    positions would double-count.

    For a BUY YES at price p (in dollars):
        - If the outcome wins: profit = (1.0 - p) per contract
        - If the outcome loses: loss = -p per contract

    For a BUY NO at NO-price p (in dollars):
        - If the outcome loses (NO wins): profit = (1.0 - p) per contract
        - If the outcome wins (NO loses): loss = -p per contract

    Fee is always an additional cost.
    """
    if fill.action == TradeAction.SELL:
        return 0.0

    is_winner = fill.outcome == winner
    p = fill.price

    if fill.direction == PositionDirection.YES:
        if is_winner:
            gross = (1.0 - p) * fill.contracts
        else:
            gross = -p * fill.contracts
    else:  # NO
        if is_winner:
            # Outcome won → NO position loses → lost our cost
            gross = -p * fill.contracts
        else:
            # Outcome lost → NO position wins → payout $1 minus cost
            gross = (1.0 - p) * fill.contracts

    return gross - fill.fee_dollars


def format_trade_log(
    result: BacktestResult,
    settlement: SettlementResult,
    winner: str,
) -> str:
    """Format a human-readable trade log with per-fill P&L.

    Args:
        result: BacktestResult from BacktestEngine.run().
        settlement: SettlementResult from result.settle(winner).
        winner: The name of the winning outcome.

    Returns:
        Multi-line string with one line per fill plus a summary.

    Example::

        2025-01-20  BUY YES  Anora          10 @ 52.0¢  edge=+8.2pp  → +$4.80
        2025-01-25  BUY YES  The Brutalist    5 @ 19.0¢  edge=+5.3pp  → -$0.95
        ─────────────────────────────────────────────────────────────────────
        Total: 15 fills, P&L = +$3.85, Fees = $0.42
    """
    lines: list[str] = []

    for fill in result.trade_log:
        pnl = _fill_pnl(fill, winner)
        pnl_sign = "+" if pnl >= 0 else ""
        is_winner_marker = " ★" if fill.outcome == winner else ""

        ts_str = fill.timestamp.strftime("%Y-%m-%d %H:%M")
        lines.append(
            f"{ts_str}  {fill.action.value:4s} {fill.direction.value:3s}  "
            f"{fill.outcome:<25s} {fill.contracts:3d} @ {fill.price * 100:5.1f}¢  "
            f"edge={fill.reason:>12s}  → {pnl_sign}${pnl:.2f}{is_winner_marker}"
        )

    lines.append("─" * 80)
    lines.append(
        f"Total: {len(result.trade_log)} fills, "
        f"P&L = {'+' if settlement.total_pnl >= 0 else ''}${settlement.total_pnl:.2f}, "
        f"Fees = ${result.total_fees_paid:.2f}"
    )

    return "\n".join(lines)


def trade_log_to_dataframe(
    result: BacktestResult,
    winner: str,
) -> pd.DataFrame:
    """Convert trade log to a DataFrame for analysis.

    Columns: date, outcome, direction, action, contracts, price,
    fee_dollars, cash_delta, edge_reason, pnl, is_winner.

    Args:
        result: BacktestResult from BacktestEngine.run().
        winner: The name of the winning outcome.

    Returns:
        DataFrame with one row per fill.
    """
    rows = []
    for fill in result.trade_log:
        pnl = _fill_pnl(fill, winner)
        rows.append(
            {
                "timestamp": fill.timestamp.isoformat(),
                "outcome": fill.outcome,
                "direction": fill.direction.value,
                "action": fill.action.value,
                "contracts": fill.contracts,
                "price": fill.price,
                "fee_dollars": round(fill.fee_dollars, 4),
                "cash_delta": round(fill.cash_delta, 4),
                "edge_reason": fill.reason,
                "pnl": round(pnl, 4),
                "is_winner": fill.outcome == winner,
            }
        )
    return pd.DataFrame(rows)
