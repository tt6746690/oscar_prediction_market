"""Deep-dive analysis of trade signal backtest: best/worst configs + GBT vs LR.

Produces:
1. Per-config temporal position & PnL plots for best and worst configs
2. Detailed per-trade log with model prob, market price, edge, PnL
3. GBT vs LR edge distribution comparison
4. Per-outcome contribution analysis
5. Comparison with old (full feature) vs new (additive_3) temporal models

Usage:
    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\
d20260214_trade_signal_ablation.analyze_deep_dive \
        --results-dir storage/d20260214_trade_signal_ablation/results \
        --models-dir storage/d20260214_trade_signal_ablation/models \
        --output-dir storage/d20260214_trade_signal_ablation
"""

import argparse
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oscar_prediction_market.data.schema import OscarCategory
from oscar_prediction_market.one_offs.analysis_utils.style import (
    AWARDS_SEASON_EVENTS,
    apply_style,
)
from oscar_prediction_market.one_offs.d20260214_trade_signal_ablation.price_helpers import (
    get_market_prices_on_date,
)
from oscar_prediction_market.one_offs.d20260214_trade_signal_backtest.generate_signals import (
    BacktestConfig,
)
from oscar_prediction_market.one_offs.legacy_snapshot_loading import (
    load_snapshot_predictions,
    load_weighted_predictions,
)
from oscar_prediction_market.trading.edge import (
    estimate_spread_from_trades,
)
from oscar_prediction_market.trading.kalshi_client import (
    estimate_fee,
)
from oscar_prediction_market.trading.market_data import OSCAR_MARKETS
from oscar_prediction_market.trading.oscar_market import (
    OscarMarket,
)
from oscar_prediction_market.trading.portfolio import (
    apply_signals,
    compute_mtm_value,
    settle_positions,
)
from oscar_prediction_market.trading.schema import (
    KellyConfig,
    KellyMode,
    Position,
    PositionDirection,
    TradingConfig,
)
from oscar_prediction_market.trading.signals import (
    MarketQuotes,
    generate_signals,
)

logger = logging.getLogger(__name__)

apply_style()

# Snapshot dates — derived from shared events
SNAPSHOT_DATES = list(AWARDS_SEASON_EVENTS.keys())


# ============================================================================
# Detailed Single-Config Runner (with per-trade logging)
# ============================================================================


def run_detailed_backtest(
    config: BacktestConfig,
    daily_prices: pd.DataFrame,
    spread_penalties: dict[str, float],
    median_spread: float,
    snapshot_dates: list[str],
    models_dir: Path,
) -> dict[str, Any]:
    """Run backtest for a single config with detailed per-trade + per-outcome logging.

    Unlike run_single_config in run_ablation.py, this tracks:
    - Every individual trade (nominee, action, contracts, price, fee, edge)
    - Per-outcome model prob and market price at every snapshot
    - Per-outcome position and MtM value at every snapshot
    - Cumulative PnL breakdown
    """
    model_type = config.model_types[0]

    positions: list[Position] = []
    cash = config.bankroll_dollars
    total_fees = 0.0
    total_trades = 0

    # Detailed tracking
    trade_log: list[dict[str, Any]] = []
    snapshot_details: list[dict[str, Any]] = []

    for snap_date in snapshot_dates:
        target_date = date.fromisoformat(snap_date)

        # Get market prices
        market_prices = get_market_prices_on_date(daily_prices, target_date)
        if not market_prices:
            continue

        # Load predictions
        predictions = load_weighted_predictions(
            models_dir,
            snap_date,
            model_type,
            market_prices,
            market_blend_alpha=config.market_blend_alpha,
            normalize_probabilities=config.normalize_probabilities,
        )
        if not predictions:
            continue

        # Spread
        default_spread = (
            median_spread
            if config.spread_penalty_mode == "trade_data"
            else config.fixed_spread_penalty
        )
        avg_spread = (
            sum(spread_penalties.get(t, default_spread) for t in predictions) / len(predictions)
            if predictions
            else default_spread
        )

        # Bankroll
        mtm_pre = compute_mtm_value(positions, market_prices)
        current_bankroll = max(1.0, cash + mtm_pre)

        # Signal config and execution prices
        execution_prices = MarketQuotes.from_close_prices(market_prices, avg_spread)
        signal_config = TradingConfig(
            kelly=KellyConfig(
                bankroll=current_bankroll,
                kelly_fraction=config.kelly_fraction,
                kelly_mode=KellyMode(config.kelly_mode),
                buy_edge_threshold=config.min_edge,
                max_position_per_outcome=config.max_position_per_outcome_dollars,
                max_total_exposure=config.max_total_exposure_dollars,
            ),
            sell_edge_threshold=config.sell_edge_threshold,
            fee_type=config.fee_type,  # type: ignore[arg-type]
            limit_price_offset=0.0,
            min_price=config.min_price,
            allowed_directions=frozenset({PositionDirection.YES}),
        )

        # Generate signals
        report = generate_signals(
            model_predictions=predictions,
            execution_prices=execution_prices,
            current_positions=positions,
            config=signal_config,
        )

        # Log per-outcome data BEFORE applying signals
        per_outcome: dict[str, dict[str, Any]] = {}
        pos_by_outcome = {p.outcome: p for p in positions}
        for nominee in sorted(set(predictions.keys()) | set(market_prices.keys())):
            model_p = predictions.get(nominee)
            market_p = market_prices.get(nominee)
            pos = pos_by_outcome.get(nominee)
            edge_val = None
            if model_p is not None and market_p is not None:
                edge_val = model_p - market_p

            per_outcome[nominee] = {
                "model_prob": model_p,
                "market_price": market_p,
                "edge": edge_val,
                "position_contracts": pos.contracts if pos else 0,
                "position_avg_cost": pos.avg_cost if pos else 0,
            }

        # Log each signal/trade
        for signal in report:
            if signal.delta_contracts != 0:
                fee_amount = estimate_fee(
                    signal.execution_price,
                    n_contracts=abs(signal.delta_contracts),
                    fee_type=config.fee_type,
                )
                trade_log.append(
                    {
                        "date": snap_date,
                        "nominee": signal.outcome,
                        "action": signal.action,
                        "contracts": signal.delta_contracts,
                        "price": signal.execution_price,
                        "model_prob": predictions.get(signal.outcome),
                        "market_price": market_prices.get(signal.outcome),
                        "net_edge": signal.net_edge,
                        "fee_dollars": fee_amount,
                        "outlay_dollars": signal.outlay_dollars,
                    }
                )

        # Apply signals
        tr = apply_signals(
            positions,
            cash,
            report,
            fee_type=config.fee_type,
            timestamp=datetime.combine(target_date, datetime.min.time()),
        )
        positions = tr.positions
        cash = tr.cash
        total_fees += tr.fees_paid
        total_trades += tr.n_trades

        # Post-trade snapshot
        mtm_post = compute_mtm_value(positions, market_prices)
        total_wealth = cash + mtm_post

        # Update per_outcome with post-trade positions
        post_pos_by_outcome = {p.outcome: p for p in positions}
        for nominee in per_outcome:
            pos = post_pos_by_outcome.get(nominee)
            per_outcome[nominee]["post_position_contracts"] = pos.contracts if pos else 0

        snapshot_details.append(
            {
                "date": snap_date,
                "event": AWARDS_SEASON_EVENTS.get(snap_date, snap_date),
                "cash": round(cash, 2),
                "mtm": round(mtm_post, 2),
                "total_wealth": round(total_wealth, 2),
                "total_fees": round(total_fees, 2),
                "total_trades": total_trades,
                "n_positions": sum(1 for p in positions if p.contracts > 0),
                "per_outcome": per_outcome,
            }
        )

    # Compute settlements
    settlements: dict[str, dict[str, Any]] = {}
    if positions:
        last_date = date.fromisoformat(snapshot_dates[-1])
        last_prices = get_market_prices_on_date(daily_prices, last_date)
        for nominee_name in sorted(set(last_prices.keys())):
            settlement = settle_positions(
                positions, cash, winner=nominee_name, initial_bankroll=config.bankroll_dollars
            )
            settlements[nominee_name] = settlement.model_dump()

    final_wealth = (
        snapshot_details[-1]["total_wealth"] if snapshot_details else config.bankroll_dollars
    )

    return {
        "model_type": model_type,
        "final_wealth": final_wealth,
        "total_return_pct": round(
            (final_wealth - config.bankroll_dollars) / config.bankroll_dollars * 100, 1
        ),
        "total_fees": round(total_fees, 2),
        "total_trades": total_trades,
        "trade_log": trade_log,
        "snapshots": snapshot_details,
        "settlements": settlements,
    }


# ============================================================================
# Plotting: Per-Config Temporal Deep Dive
# ============================================================================


def plot_config_deep_dive(
    result: dict[str, Any],
    config_id: str,
    output_dir: Path,
) -> None:
    """Multi-panel deep-dive plot for a single config.

    Panel 1: Wealth curve (cash + MtM) over time
    Panel 2: Per-outcome positions (stacked bar) over time
    Panel 3: Per-outcome model prob vs market price with trade markers
    Panel 4: Cumulative fees and trade count
    """
    snapshots = result["snapshots"]
    trades = result["trade_log"]
    if not snapshots:
        return

    dates = [s["date"] for s in snapshots]
    date_labels = [f"{d}\n{AWARDS_SEASON_EVENTS.get(d, '')}" for d in dates]

    # Collect all nominees that ever had a position or trade
    active_outcomes: set[str] = set()
    for t in trades:
        active_outcomes.add(t["nominee"])
    for s in snapshots:
        for nom, data in s["per_outcome"].items():
            if data.get("position_contracts", 0) > 0 or data.get("post_position_contracts", 0) > 0:
                active_outcomes.add(nom)

    if not active_outcomes:
        logger.info("No active positions for %s, skipping deep-dive plot", config_id)
        return

    active_outcomes_sorted = sorted(active_outcomes)
    cmap = matplotlib.colormaps["tab10"]
    colors = {
        nom: cmap(i / max(1, len(active_outcomes_sorted) - 1))
        for i, nom in enumerate(active_outcomes_sorted)
    }

    fig, axes = plt.subplots(4, 1, figsize=(16, 20), sharex=True)

    # --- Panel 1: Wealth curve ---
    ax = axes[0]
    cash_vals = [s["cash"] for s in snapshots]
    wealth_vals = [s["total_wealth"] for s in snapshots]

    ax.fill_between(range(len(dates)), 0, cash_vals, alpha=0.3, color="green", label="Cash")
    ax.fill_between(
        range(len(dates)),
        cash_vals,
        wealth_vals,
        alpha=0.3,
        color="blue",
        label="MtM Value",
    )
    ax.plot(range(len(dates)), wealth_vals, "ko-", markersize=4, label="Total Wealth")
    ax.axhline(y=1000, color="red", linestyle="--", alpha=0.5, label="Initial ($1000)")

    # Annotate trade events
    for i, s in enumerate(snapshots):
        if s["total_trades"] > (snapshots[i - 1]["total_trades"] if i > 0 else 0):
            new_trades = s["total_trades"] - (snapshots[i - 1]["total_trades"] if i > 0 else 0)
            ax.annotate(
                f"+{new_trades}T",
                xy=(i, wealth_vals[i]),
                xytext=(0, 10),
                textcoords="offset points",
                fontsize=8,
                color="darkred",
                ha="center",
            )

    ax.set_ylabel("Dollars ($)")
    ax.set_title(
        f"Backtest Deep Dive: {config_id}\n"
        f"Return: {result['total_return_pct']:+.1f}% | "
        f"Fees: ${result['total_fees']:.0f} | "
        f"Trades: {result['total_trades']}"
    )
    ax.legend(loc="upper right")

    # --- Panel 2: Position contracts per nominee (stacked bar) ---
    ax = axes[1]
    bottom = np.zeros(len(dates))
    bar_tops: dict[str, list[float]] = {}
    for nom in active_outcomes_sorted:
        contracts = []
        for s in snapshots:
            nom_data = s["per_outcome"].get(nom, {})
            contracts.append(nom_data.get("post_position_contracts", 0))
        contracts_arr = np.array(contracts, dtype=float)
        ax.bar(
            range(len(dates)),
            contracts_arr,
            bottom=bottom,
            label=nom,
            color=colors[nom],
            alpha=0.7,
        )
        bar_tops[nom] = list(bottom + contracts_arr)
        bottom += contracts_arr

    # Overlay buy/sell markers on position bars
    for t in trades:
        if t["date"] not in dates:
            continue
        idx = dates.index(t["date"])
        nom = t["nominee"]
        is_buy = t["action"] == "BUY"
        marker = "^" if is_buy else "v"
        color = "green" if is_buy else "red"
        y_pos = bar_tops.get(nom, [0] * len(dates))[idx]
        offset = 50 if is_buy else -50
        delta_str = f"+{t['contracts']}" if is_buy else str(t["contracts"])
        ax.scatter(
            [idx],
            [y_pos + offset],
            marker=marker,
            color=color,
            s=60,
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
        )
        ax.annotate(
            delta_str,
            xy=(idx, y_pos + offset),
            xytext=(0, 8 if is_buy else -8),
            textcoords="offset points",
            fontsize=5,
            color=color,
            ha="center",
            va="bottom" if is_buy else "top",
        )

    ax.set_ylabel("Contracts Held")
    ax.set_title("Position Evolution by Outcome")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    # --- Panel 3: Model prob vs market price for active nominees ---
    ax = axes[2]
    for nom in active_outcomes_sorted:
        model_probs = []
        market_probs = []
        for s in snapshots:
            nom_data = s["per_outcome"].get(nom, {})
            mp = nom_data.get("model_prob")
            mkp = nom_data.get("market_price")
            model_probs.append(mp * 100 if mp is not None else None)
            market_probs.append(mkp * 100 if mkp is not None else None)

        # Model (solid)
        valid_model = [(i, p) for i, p in enumerate(model_probs) if p is not None]
        if valid_model:
            xi, yi = zip(*valid_model, strict=True)
            ax.plot(xi, yi, "-o", color=colors[nom], markersize=3, label=f"{nom} (model)")

        # Market (dashed)
        valid_market = [(i, p) for i, p in enumerate(market_probs) if p is not None]
        if valid_market:
            xi, yi = zip(*valid_market, strict=True)
            ax.plot(xi, yi, "--", color=colors[nom], alpha=0.5)

    # Overlay buy/sell markers
    for t in trades:
        if t["date"] in dates:
            idx = dates.index(t["date"])
            marker = "^" if t["action"] == "BUY" else "v"
            color = "green" if t["action"] == "BUY" else "red"
            y_val = (
                t["model_prob"] * 100
                if t["model_prob"] is not None
                else (t["market_price"] * 100 if t["market_price"] is not None else None)
            )
            if y_val is not None:
                ax.scatter(
                    [idx],
                    [y_val],
                    marker=marker,
                    color=color,
                    s=100,
                    zorder=5,
                    edgecolors="black",
                    linewidths=0.5,
                )

    ax.set_ylabel("Probability / Price (%/cents)")
    ax.set_title("Model Prob (solid) vs Market Price (dashed) — Active Outcomes")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    # --- Panel 4: Cumulative fees and trade count ---
    ax = axes[3]
    cum_fees = [s["total_fees"] for s in snapshots]
    cum_trades = [s["total_trades"] for s in snapshots]

    ax2 = ax.twinx()
    ax.bar(range(len(dates)), cum_fees, alpha=0.4, color="orange", label="Cum. Fees ($)")
    ax2.plot(range(len(dates)), cum_trades, "rs-", markersize=4, label="Cum. Trades")

    ax.set_ylabel("Cumulative Fees ($)")
    ax2.set_ylabel("Cumulative Trades")
    ax.set_xlabel("Snapshot Date")
    ax.set_title("Fee Drag and Trading Activity")

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # X-axis labels
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(date_labels, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    safe_id = config_id.replace("/", "_").replace(" ", "_")
    fname = f"deep_dive_{safe_id}.png"
    plt.savefig(output_dir / fname, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", output_dir / fname)


def print_trade_log(result: dict[str, Any], config_id: str) -> None:
    """Print detailed trade log for a config."""
    trades = result["trade_log"]
    if not trades:
        print(f"\n  {config_id}: No trades")
        return

    print(f"\n{'=' * 100}")
    print(f"Trade Log: {config_id}")
    print(f"{'=' * 100}")
    header = (
        f"  {'Date':<12} {'Outcome':<30} {'Act':>4} {'Qty':>5} "
        f"{'Price':>6} {'Model':>6} {'Mkt':>6} {'Edge':>7} "
        f"{'Fee':>6} {'Outlay':>8}"
    )
    print(header)
    print(f"  {'-' * 96}")

    for t in trades:
        model_str = f"{t['model_prob'] * 100:.1f}" if t["model_prob"] else "N/A"
        mkt_str = f"{t['market_price'] * 100:.0f}" if t["market_price"] else "N/A"
        print(
            f"  {t['date']:<12} {t['nominee']:<30} {t['action']:>4} "
            f"{t['contracts']:>+5} {t['price'] * 100:>5.0f}c "
            f"{model_str:>5}% {mkt_str:>5}c {t['net_edge']:>+6.1%} "
            f"${t['fee_dollars']:>5.2f} ${t['outlay_dollars']:>+7.2f}"
        )

    total_buy = sum(t["outlay_dollars"] for t in trades if t["action"] == "BUY")
    total_sell = sum(abs(t["outlay_dollars"]) for t in trades if t["action"] == "SELL")
    total_fees = sum(t["fee_dollars"] for t in trades)
    print(f"\n  Total bought: ${total_buy:+.2f}")
    print(f"  Total sold:   ${total_sell:.2f}")
    print(f"  Total fees:   ${total_fees:.2f}")


# ============================================================================
# GBT vs LR Edge Distribution Analysis
# ============================================================================


def analyze_gbt_vs_lr(
    models_dir: Path,
    daily_prices: pd.DataFrame,
    spread_penalties: dict[str, float],
    median_spread: float,
    output_dir: Path,
) -> None:
    """Compare GBT vs LR: edge distributions, probability calibration, trade profiles.

    Investigates why similar accuracy/Brier leads to different trading outcomes.
    """
    print(f"\n{'=' * 80}")
    print("GBT vs LR: WHY DOES TRADING PERFORMANCE DIFFER?")
    print(f"{'=' * 80}")

    # Collect edges and predictions at every snapshot
    records: list[dict[str, Any]] = []

    for snap_date in SNAPSHOT_DATES:
        target_date = date.fromisoformat(snap_date)
        market_prices = get_market_prices_on_date(daily_prices, target_date)
        if not market_prices:
            continue

        for model_type in ["lr", "gbt"]:
            preds = load_snapshot_predictions(models_dir, model_type, snap_date)
            if not preds:
                continue

            for nominee, model_p in preds.items():
                market_p = market_prices.get(nominee)
                if market_p is None:
                    continue

                # Raw edge (before fees/spread)
                raw_edge = model_p - market_p

                # Approximate net edge (after fees)
                # Taker fee ~7%: fee ≈ 0.07 * P * (1-P) per contract
                p = market_p
                taker_fee = 0.07 * p * (1 - p)
                maker_fee = 0.0175 * p * (1 - p)
                spread_cost = spread_penalties.get(nominee, median_spread)

                net_edge_taker = raw_edge - taker_fee - spread_cost
                net_edge_maker = raw_edge - maker_fee - spread_cost

                records.append(
                    {
                        "date": snap_date,
                        "model": model_type,
                        "nominee": nominee,
                        "model_prob": model_p,
                        "market_price": market_p,
                        "market_prob": p,
                        "raw_edge": raw_edge,
                        "net_edge_taker": net_edge_taker,
                        "net_edge_maker": net_edge_maker,
                    }
                )

    df = pd.DataFrame(records)
    if df.empty:
        print("No data for GBT vs LR analysis")
        return

    # --- 1. Edge distribution comparison ---
    print("\n--- Edge Distribution (all snapshots) ---")
    for model in ["lr", "gbt"]:
        sub = df[df["model"] == model]
        print(
            f"  {model.upper()}: mean raw edge={sub['raw_edge'].mean():+.3f}, "
            f"std={sub['raw_edge'].std():.3f}, "
            f"positive edges={len(sub[sub['raw_edge'] > 0])}/{len(sub)}, "
            f"mean net(maker)={sub['net_edge_maker'].mean():+.3f}, "
            f"actionable(maker, >5%)={len(sub[sub['net_edge_maker'] > 0.05])}"
        )

    # --- 2. Probability comparison for key nominees ---
    print("\n--- Probability Comparison: Key Outcomes (Final Snapshot) ---")
    final = df[df["date"] == SNAPSHOT_DATES[-1]]
    key_outcomes = ["One Battle after Another", "Sinners", "Hamnet", "Marty Supreme"]
    header = f"  {'Outcome':<30} {'LR':>7} {'GBT':>7} {'Market':>7} {'LR Edge':>8} {'GBT Edge':>9}"
    print(header)
    print(f"  {'-' * 72}")
    for nom in key_outcomes:
        lr_row = final[(final["model"] == "lr") & (final["nominee"] == nom)]
        gbt_row = final[(final["model"] == "gbt") & (final["nominee"] == nom)]
        if lr_row.empty or gbt_row.empty:
            continue
        lr_p = lr_row.iloc[0]["model_prob"]
        gbt_p = gbt_row.iloc[0]["model_prob"]
        mkt = lr_row.iloc[0]["market_prob"]
        print(
            f"  {nom:<30} {lr_p:>6.1%} {gbt_p:>6.1%} {mkt:>6.1%} "
            f"{lr_p - mkt:>+7.1%} {gbt_p - mkt:>+8.1%}"
        )

    # --- 3. Edge distribution plots ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Raw edge histogram
    ax = axes[0, 0]
    for model in ["lr", "gbt"]:
        sub = df[df["model"] == model]
        ax.hist(
            sub["raw_edge"] * 100,
            bins=40,
            alpha=0.5,
            label=model.upper(),
            edgecolor="black",
        )
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    ax.axvline(x=5, color="green", linestyle=":", alpha=0.5, label="5% threshold")
    ax.set_xlabel("Raw Edge (pp)")
    ax.set_ylabel("Count")
    ax.set_title("Raw Edge Distribution (model - market)")
    ax.legend()

    # Panel 2: Edge by nominee (box plot)
    ax = axes[0, 1]
    edge_by_outcome: dict[str, dict[str, list[float]]] = {}
    for _, row in df.iterrows():
        nom = row["nominee"]
        model = row["model"]
        if nom not in edge_by_outcome:
            edge_by_outcome[nom] = {"lr": [], "gbt": []}
        edge_by_outcome[nom][model].append(row["raw_edge"] * 100)

    # Only show nominees with meaningful edges
    noms_to_plot = sorted(
        edge_by_outcome.keys(),
        key=lambda n: abs(np.mean(edge_by_outcome[n]["lr"] + edge_by_outcome[n]["gbt"])),
        reverse=True,
    )[:8]

    x_pos = np.arange(len(noms_to_plot))
    width = 0.35
    lr_means = [np.mean(edge_by_outcome[n]["lr"]) for n in noms_to_plot]
    gbt_means = [np.mean(edge_by_outcome[n]["gbt"]) for n in noms_to_plot]

    ax.bar(x_pos - width / 2, lr_means, width, label="LR", alpha=0.7)
    ax.bar(x_pos + width / 2, gbt_means, width, label="GBT", alpha=0.7)
    ax.set_ylabel("Mean Raw Edge (pp)")
    ax.set_title("Mean Edge by Outcome")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [n[:20] + "..." if len(n) > 20 else n for n in noms_to_plot],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax.legend()

    # Panel 3: Edge over time for frontrunner
    ax = axes[1, 0]
    frontrunner = "One Battle after Another"
    for model in ["lr", "gbt"]:
        sub = df[(df["model"] == model) & (df["nominee"] == frontrunner)]
        if not sub.empty:
            ax.plot(
                range(len(sub)),
                np.asarray(sub["raw_edge"].values) * 100,
                "o-",
                label=f"{model.upper()}",
                markersize=4,
            )
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("Raw Edge (pp)")
    ax.set_title(f"Edge Over Time: {frontrunner}")
    x_labels = [f"{d[:5]}\n{d[5:]}" for d in SNAPSHOT_DATES[: len(sub)]]
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.legend()

    # Panel 4: Model prob vs market scatter (all snapshots)
    ax = axes[1, 1]
    for model in ["lr", "gbt"]:
        sub = df[df["model"] == model]
        ax.scatter(
            sub["market_prob"] * 100,
            sub["model_prob"] * 100,
            alpha=0.3,
            s=15,
            label=model.upper(),
        )
    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, label="Perfect calibration")
    ax.set_xlabel("Market Price (cents)")
    ax.set_ylabel("Model Probability (%)")
    ax.set_title("Model vs Market: All Snapshots")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "gbt_vs_lr_analysis.png", bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {output_dir / 'gbt_vs_lr_analysis.png'}")

    # --- 4. Why GBT trades less than LR ---
    print("\n--- Actionable Edge Count (net edge > 5%, maker fees) ---")
    header = f"  {'Date':<12} {'LR':>4} {'GBT':>4} {'LR nominees (>5% edge)':>40} {'GBT nominees (>5% edge)':>40}"
    print(header)
    for snap_date in SNAPSHOT_DATES:
        for model in ["lr", "gbt"]:
            sub = df[(df["date"] == snap_date) & (df["model"] == model)]
            actionable = sub[sub["net_edge_maker"] > 0.05]
            if model == "lr":
                lr_count = len(actionable)
                lr_noms = ", ".join(
                    f"{r['nominee'][:15]}({r['net_edge_maker']:.0%})"
                    for _, r in actionable.iterrows()
                )
            else:
                gbt_count = len(actionable)
                gbt_noms = ", ".join(
                    f"{r['nominee'][:15]}({r['net_edge_maker']:.0%})"
                    for _, r in actionable.iterrows()
                )
        print(
            f"  {snap_date:<12} {lr_count:>4} {gbt_count:>4} {lr_noms[:40]:>40} {gbt_noms[:40]:>40}"
        )

    # --- 5. Probability spread analysis ---
    print("\n--- Probability Spread (how diffuse are model predictions?) ---")
    for snap_date in [SNAPSHOT_DATES[0], SNAPSHOT_DATES[5], SNAPSHOT_DATES[-1]]:
        print(f"\n  {snap_date}:")
        for model in ["lr", "gbt"]:
            sub = df[(df["date"] == snap_date) & (df["model"] == model)]
            if sub.empty:
                continue
            probs = np.asarray(sub["model_prob"].values)
            max_p = float(probs.max())
            entropy = float(-np.sum(probs * np.log2(np.clip(probs, 1e-10, 1))))
            spread = float(probs.std())
            print(
                f"    {model.upper()}: max={max_p:.1%}, "
                f"entropy={entropy:.2f}bits, std={spread:.3f}, "
                f"top3: {sorted(probs, reverse=True)[:3]}"
            )


# ============================================================================
# Edge Over Time — All Outcomes
# ============================================================================


def plot_edge_over_time_all_outcomes(
    models_dir: Path,
    daily_prices: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Subplot grid: raw edge over time for every nominee, LR vs GBT."""
    import math

    # Collect edges for all nominees/dates/models
    records: list[dict[str, Any]] = []
    for snap_date in SNAPSHOT_DATES:
        target_date = date.fromisoformat(snap_date)
        market_prices = get_market_prices_on_date(daily_prices, target_date)
        if not market_prices:
            continue

        for model_type in ["lr", "gbt"]:
            preds = load_snapshot_predictions(models_dir, model_type, snap_date)
            if not preds:
                continue
            for nominee, model_p in preds.items():
                market_p = market_prices.get(nominee)
                if market_p is None:
                    continue
                records.append(
                    {
                        "date": snap_date,
                        "model": model_type,
                        "nominee": nominee,
                        "raw_edge": model_p - market_p,
                    }
                )

    if not records:
        return

    df = pd.DataFrame(records)
    nominees = sorted(df["nominee"].unique())
    ncols = 3
    nrows = math.ceil(len(nominees) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows), sharex=True)
    axes_flat = axes.flatten() if nrows > 1 else (axes if ncols > 1 else [axes])

    event_labels = [AWARDS_SEASON_EVENTS.get(d, d[5:]) for d in SNAPSHOT_DATES]

    for i, nominee in enumerate(nominees):
        ax = axes_flat[i]
        for model_type, color, marker in [("lr", "tab:blue", "o"), ("gbt", "tab:orange", "o")]:
            sub = df[(df["model"] == model_type) & (df["nominee"] == nominee)].sort_values("date")
            if sub.empty:
                continue
            x_indices = [SNAPSHOT_DATES.index(d) for d in sub["date"]]
            ax.plot(
                x_indices,
                np.asarray(sub["raw_edge"].values) * 100,
                f"{marker}-",
                color=color,
                markersize=4,
                label=model_type.upper() if i == 0 else None,
            )

        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        ax.axhline(y=5, color="green", linestyle=":", alpha=0.5)

        nom_short = nominee[:25] + "..." if len(nominee) > 25 else nominee
        ax.set_title(nom_short, fontsize=9)
        if i % ncols == 0:
            ax.set_ylabel("Raw Edge (pp)")

    # Add legend to first subplot
    if len(axes_flat) > 0:
        axes_flat[0].legend(fontsize=7)

    # Set shared x-axis labels on bottom row
    for ax in axes_flat:
        ax.set_xticks(range(len(SNAPSHOT_DATES)))
        ax.set_xticklabels(event_labels, rotation=45, ha="right", fontsize=7)

    # Hide unused axes
    for j in range(len(nominees), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Raw Edge Over Time by Outcome (blue=LR, orange=GBT)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "edge_over_time_all_outcomes.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'edge_over_time_all_outcomes.png'}")


# ============================================================================
# Old vs New Model Comparison
# ============================================================================


def compare_old_vs_new(
    new_csv: Path,
    old_csv: Path | None,
    output_dir: Path,
) -> None:
    """Compare temporal predictions from old (full feature) vs new (additive_3) models.

    Generates comparison plots of:
    - Brier score progression
    - Feature count progression
    - CV accuracy progression
    - Jitter (consecutive prediction stability)
    """
    if old_csv is None or not old_csv.exists():
        print("\nNo old model predictions CSV found, skipping old vs new comparison")
        return

    print(f"\n{'=' * 80}")
    print("OLD vs NEW MODEL COMPARISON (Full Features vs Additive_3)")
    print(f"{'=' * 80}")

    new_df = pd.read_csv(new_csv)
    old_df = pd.read_csv(old_csv)

    new_df["snapshot_date"] = pd.to_datetime(new_df["snapshot_date"]).dt.date
    old_df["snapshot_date"] = pd.to_datetime(old_df["snapshot_date"]).dt.date

    # Filter to 2026 test preds only
    new_test = new_df[new_df["year"] == 2026].copy()
    old_test = old_df[old_df["year"] == 2026].copy()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for model in ["lr", "gbt"]:
        # --- Extract per-snapshot summaries ---
        for label, test_df, style in [
            ("Old (full)", old_test, "--"),
            ("New (add3)", new_test, "-"),
        ]:
            model_data = test_df[test_df["model_type"] == model]
            snapshots = sorted(model_data["snapshot_date"].unique())

            dates = list(range(len(snapshots)))
            briers = []
            feats = []
            accs = []
            for d in snapshots:
                snap = model_data[model_data["snapshot_date"] == d]
                briers.append(snap.iloc[0].get("cv_brier", float("nan")))
                feats.append(snap.iloc[0].get("feature_count", 0))
                accs.append(snap.iloc[0].get("cv_accuracy", 0))

            # CV Brier
            axes[0, 0].plot(
                dates, briers, f"{style}o", label=f"{model.upper()} {label}", markersize=3
            )
            # Feature count
            axes[0, 1].plot(
                dates, feats, f"{style}o", label=f"{model.upper()} {label}", markersize=3
            )
            # CV Accuracy
            axes[1, 0].plot(
                dates,
                [a * 100 for a in accs],
                f"{style}o",
                label=f"{model.upper()} {label}",
                markersize=3,
            )

    # Compute jitter (mean absolute consecutive prediction change)
    jitters: dict[str, list[float]] = {}
    for label, test_df in [("Old", old_test), ("New", new_test)]:
        for model in ["lr", "gbt"]:
            model_data = test_df[test_df["model_type"] == model]
            snapshots = sorted(model_data["snapshot_date"].unique())

            snap_jitters = []
            for i in range(1, len(snapshots)):
                prev = model_data[model_data["snapshot_date"] == snapshots[i - 1]]
                curr = model_data[model_data["snapshot_date"] == snapshots[i]]

                merged = prev.merge(curr, on="title", suffixes=("_prev", "_curr"))
                if not merged.empty:
                    jit = np.mean(
                        np.abs(
                            merged["probability_curr"].values - merged["probability_prev"].values
                        )
                    )
                    snap_jitters.append(jit)

            key = f"{model.upper()} {label}"
            jitters[key] = snap_jitters

    ax = axes[1, 1]
    for key, jit_vals in jitters.items():
        style = "--" if "Old" in key else "-"
        ax.plot(range(len(jit_vals)), jit_vals, f"{style}o", label=key, markersize=3)
    ax.set_ylabel("Mean Abs Jitter")
    ax.set_title("Prediction Jitter (consecutive snapshot change)")
    ax.legend()

    axes[0, 0].set_ylabel("CV Brier Score")
    axes[0, 0].set_title("CV Brier Score Over Time")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_ylabel("Feature Count")
    axes[0, 1].set_title("Selected Feature Count Over Time")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].set_ylabel("CV Accuracy (%)")
    axes[1, 0].set_title("CV Accuracy Over Time")
    axes[1, 0].legend(fontsize=8)

    for ax in axes.flatten():
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "old_vs_new_comparison.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'old_vs_new_comparison.png'}")

    # Print summary table
    print("\n--- Summary: Old (Full) vs New (Additive_3) at final snapshot ---")
    header = f"  {'Model':<12} {'Config':<12} {'Feat':>5} {'Brier':>7} {'Acc':>7} {'Jitter':>8}"
    print(header)
    print(f"  {'-' * 55}")

    for model in ["lr", "gbt"]:
        for label, test_df in [("Old (full)", old_test), ("New (add3)", new_test)]:
            model_data = test_df[test_df["model_type"] == model]
            if model_data.empty:
                continue
            last_snap = model_data["snapshot_date"].max()
            last = model_data[model_data["snapshot_date"] == last_snap].iloc[0]
            key = f"{model.upper()} {label.split('(')[1].rstrip(')')}"
            jit_key = f"{model.upper()} {label.split(' ')[0]}"
            mean_jit = np.mean(jitters.get(jit_key, [0]))
            print(
                f"  {model.upper():<12} {label:<12} "
                f"{int(last.get('feature_count', 0)):>5} "
                f"{last.get('cv_brier', 0):>7.4f} "
                f"{last.get('cv_accuracy', 0):>6.1%} "
                f"{mean_jit:>8.4f}"
            )


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Deep-dive backtest analysis")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--models-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--old-predictions",
        type=str,
        default=None,
        help="Path to old model_predictions_timeseries.csv (d20260211) for comparison",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ablation results to find best/worst configs
    results_path = results_dir / "ablation_results.json"
    with open(results_path) as f:
        ablation_data = json.load(f)

    results = ablation_data["results"]
    sorted_results = sorted(results, key=lambda x: -x["total_return_pct"])

    best_result = sorted_results[0]
    worst_result = sorted_results[-1]

    print(f"Best config:  {best_result['config_id']} ({best_result['total_return_pct']:+.1f}%)")
    print(f"Worst config: {worst_result['config_id']} ({worst_result['total_return_pct']:+.1f}%)")

    # Load shared market data
    print("\nLoading market data...")
    bp_data = OSCAR_MARKETS.get_category_data(OscarCategory.BEST_PICTURE, 2026)
    mkt = OscarMarket(event_ticker=bp_data.event_ticker, nominee_tickers=bp_data.nominee_tickers)
    daily_candles = mkt.get_daily_prices(
        start_date=date(2025, 12, 1),
        end_date=date(2026, 2, 15),
    )
    daily_prices = pd.DataFrame(
        [{"date": c.date, "nominee": c.nominee, "close": c.close} for c in daily_candles]
    )

    trades = mkt.fetch_trade_history(
        start_date=date(2025, 12, 1),
        end_date=date(2026, 2, 15),
    )
    ticker_to_nominee = bp_data.ticker_to_nominee

    spread_penalties: dict[str, float] = {}
    if trades:
        ticker_spreads = estimate_spread_from_trades(
            trades, default_spread=0.04, min_trades_required=20
        )
        for ticker, spread in ticker_spreads.items():
            nominee = ticker_to_nominee.get(ticker, ticker)
            spread_penalties[nominee] = spread

    median_spread = (
        sorted(spread_penalties.values())[len(spread_penalties) // 2] if spread_penalties else 2.0
    )

    # Re-run best and worst configs with detailed tracking
    print("\n--- Re-running best config with detailed tracking ---")
    best_config = BacktestConfig.model_validate(best_result["config"])
    best_detailed = run_detailed_backtest(
        best_config,
        daily_prices,
        spread_penalties,
        median_spread,
        SNAPSHOT_DATES,
        models_dir,
    )
    print_trade_log(best_detailed, best_result["config_id"])
    plot_config_deep_dive(best_detailed, best_result["config_id"], output_dir)

    print("\n--- Re-running worst config with detailed tracking ---")
    worst_config = BacktestConfig.model_validate(worst_result["config"])
    worst_detailed = run_detailed_backtest(
        worst_config,
        daily_prices,
        spread_penalties,
        median_spread,
        SNAPSHOT_DATES,
        models_dir,
    )
    print_trade_log(worst_detailed, worst_result["config_id"])
    plot_config_deep_dive(worst_detailed, worst_result["config_id"], output_dir)

    # Settlement analysis for best/worst
    print(f"\n{'=' * 80}")
    print("SETTLEMENT SCENARIOS")
    print(f"{'=' * 80}")
    for label, result in [
        ("BEST", best_detailed),
        ("WORST", worst_detailed),
    ]:
        print(
            f"\n  {label} ({result['model_type'].upper()}, return={result['total_return_pct']:+.1f}%):"
        )
        for winner, s in sorted(
            result["settlements"].items(),
            key=lambda x: -x[1]["return_pct"],
        ):
            if abs(s["return_pct"]) > 1:
                print(f"    If {winner:<30s} wins: {s['return_pct']:+.1f}%")

    # GBT vs LR analysis
    analyze_gbt_vs_lr(
        models_dir,
        daily_prices,
        spread_penalties,
        median_spread,
        output_dir,
    )

    # Edge over time — all outcomes
    plot_edge_over_time_all_outcomes(models_dir, daily_prices, output_dir)

    # Old vs New model comparison
    new_csv = output_dir / "model_predictions_timeseries.csv"
    old_csv = Path("storage/d20260211_temporal_model_snapshots/model_predictions_timeseries.csv")
    compare_old_vs_new(new_csv, old_csv, output_dir)

    # Save detailed results for reference
    for label, result, config_id in [
        ("best", best_detailed, best_result["config_id"]),
        ("worst", worst_detailed, worst_result["config_id"]),
    ]:
        out_path = output_dir / f"deep_dive_{label}.json"
        with open(out_path, "w") as f:
            json.dump({"config_id": config_id, **result}, f, indent=2, default=str)
        logger.info("Saved %s", out_path)

    print(f"\nAll deep-dive outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
