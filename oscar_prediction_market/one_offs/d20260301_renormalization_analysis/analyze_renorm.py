"""Compare buy-hold backtest PnL with and without probability renormalization.

After name-matching filters predictions to the matched nominee subset, the
probabilities may no longer sum to 1.  This script answers: does rescaling them
back to sum=1 meaningfully change trading outcomes?

For each (year, category, model), we run two backtests per snapshot entry point:
  1. **raw** — predictions as loaded (no renormalization after filtering)
  2. **renorm** — filtered predictions rescaled so sum(p_i) = 1

We use a single representative config (moderate sizing, maker fees, YES-only)
and aggregate PnL across entry points.  This is not a full parameter sweep —
it's a sanity check on whether renormalization is first-order important.

Usage::

    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\\
d20260301_renormalization_analysis.analyze_renorm
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import pandas as pd

from oscar_prediction_market.one_offs.d20260225_buy_hold_backtest.run_backtests import (
    ALL_ENSEMBLES,
    BACKTEST_MODEL_TYPES,
)
from oscar_prediction_market.one_offs.d20260225_buy_hold_backtest.year_config import (
    YEAR_CONFIGS,
)
from oscar_prediction_market.trading.backtest import (
    BacktestConfig,
    BacktestEngine,
    BacktestSimulationConfig,
)
from oscar_prediction_market.trading.name_matching import (
    match_nominees,
)
from oscar_prediction_market.trading.oscar_data import (
    build_market_prices,
    estimate_category_spreads,
    fetch_and_cache_market_data,
    get_winner_kalshi_name,
)
from oscar_prediction_market.trading.oscar_moments import build_entry_moment
from oscar_prediction_market.trading.oscar_prediction_source import (
    load_all_snapshot_predictions,
    load_ensemble_predictions,
    load_nomination_dataset,
    translate_predictions,
)
from oscar_prediction_market.trading.schema import (
    NEVER_SELL_THRESHOLD,
    BankrollMode,
    FeeType,
    KellyConfig,
    KellyMode,
    PositionDirection,
    TradingConfig,
)

logger = logging.getLogger(__name__)

BANKROLL = 1000.0


def _baseline_config(spread_penalty: float) -> BacktestConfig:
    """Single representative config: moderate Kelly, maker fees, YES-only."""
    return BacktestConfig(
        trading=TradingConfig(
            kelly=KellyConfig(
                bankroll=BANKROLL,
                kelly_fraction=0.20,
                kelly_mode=KellyMode.INDEPENDENT,
                buy_edge_threshold=0.05,
                max_position_per_outcome=BANKROLL * 0.5,
                max_total_exposure=BANKROLL * 1.0,
            ),
            fee_type=FeeType.MAKER,
            limit_price_offset=0.01,
            sell_edge_threshold=NEVER_SELL_THRESHOLD,
            min_price=0.0,
            allowed_directions=frozenset({PositionDirection.YES}),
        ),
        simulation=BacktestSimulationConfig(
            spread_penalty=spread_penalty,
            bankroll_mode=BankrollMode.FIXED,
        ),
    )


def _renormalize(preds: dict[str, float]) -> dict[str, float]:
    """Rescale probabilities to sum to 1."""
    total = sum(preds.values())
    if total <= 0:
        return preds
    return {k: v / total for k, v in preds.items()}


def _renormalize_all_snapshots(
    preds_by_snapshot: dict[str, dict[str, float]],
    matched_names: set[str],
) -> dict[str, dict[str, float]]:
    """Renormalize predictions after filtering to matched nominees.

    For each snapshot: filter to matched names, then rescale to sum=1.
    Returns a new dict with the renormalized predictions.
    """
    result: dict[str, dict[str, float]] = {}
    for snap_key, preds in preds_by_snapshot.items():
        filtered = {k: v for k, v in preds.items() if k in matched_names}
        result[snap_key] = _renormalize(filtered)
    return result


def run_analysis() -> pd.DataFrame:
    """Run the renormalization comparison across years and models.

    Returns:
        DataFrame with columns: year, category, model, renorm, total_pnl, n_trades,
        prob_sum_mean, prob_sum_std (the last two measure how far raw probs are from 1).
    """
    rows: list[dict] = []

    for year in [2024, 2025]:
        year_config = YEAR_CONFIGS[year]
        snapshot_keys = year_config.snapshot_keys()
        snapshot_key_strs = [k.dir_name for k in snapshot_keys]
        lag_hours = 6.0

        snapshot_availability = {
            s.dir_name: s.event_datetime_utc + timedelta(hours=lag_hours) for s in snapshot_keys
        }

        for category in year_config.categories:
            print(f"\n--- {year} / {category.name} ---")

            # Fetch market data (daily only — hourly not needed for renorm comparison)
            fetch_result = fetch_and_cache_market_data(
                category,
                ceremony_year=year,
                start_date=year_config.calendar.oscar_nominations_date_local,
                end_date=year_config.calendar.oscar_ceremony_date_local,
                market_data_dir=year_config.market_data_dir,
                fetch_hourly=False,
            )
            if fetch_result is None:
                continue

            market, daily_candles, hourly_candles, trades_df = fetch_result
            spread_by_ticker, mean_spread = estimate_category_spreads(trades_df)
            cfg = _baseline_config(mean_spread)

            dataset = load_nomination_dataset(
                year_config.datasets_dir, category, snapshot_key_strs[0]
            )

            # Build model runs: (label, preds_by_snapshot)
            model_runs: list[tuple[str, dict[str, dict[str, float]]]] = []
            for mt in BACKTEST_MODEL_TYPES:
                try:
                    preds = load_all_snapshot_predictions(
                        category,
                        mt,
                        snapshot_key_strs,
                        year_config.models_dir,
                        dataset=dataset,
                        ceremony_year=year,
                    )
                except (KeyError, FileNotFoundError) as exc:
                    print(f"  {mt.short_name}: skip ({exc})")
                    continue
                model_runs.append((mt.short_name, preds))

            for ens_label, ens_types in ALL_ENSEMBLES:
                try:
                    preds = load_ensemble_predictions(
                        category,
                        ens_types,
                        snapshot_key_strs,
                        year_config.models_dir,
                        dataset=dataset,
                        ceremony_year=year,
                    )
                except (KeyError, FileNotFoundError) as exc:
                    print(f"  {ens_label}: skip ({exc})")
                    continue
                model_runs.append((ens_label, preds))

            for short_name, preds_by_snapshot in model_runs:
                if not preds_by_snapshot:
                    continue

                first_snap_preds = next(iter(preds_by_snapshot.values()))
                model_names = list(first_snap_preds.keys())

                nominee_map = match_nominees(
                    model_names=model_names,
                    kalshi_names=list(market.nominee_tickers.keys()),
                    category=category,
                    ceremony_year=year,
                )
                if not nominee_map:
                    continue

                matched_kalshi_names = set(nominee_map.values())

                # Translate predictions from model-name space to kalshi-name space
                preds_by_snapshot = translate_predictions(preds_by_snapshot, nominee_map)

                try:
                    if year_config.winners is None:
                        raise ValueError(f"No winners for {year}")
                    winner = get_winner_kalshi_name(
                        category, matched_kalshi_names, year_config.winners
                    )
                except ValueError:
                    continue

                spread_penalties = {
                    kn: spread_by_ticker[t]
                    for kn in matched_kalshi_names
                    if (t := market.nominee_tickers.get(kn)) and t in spread_by_ticker
                }

                market_prices_by_date = build_market_prices(daily_candles)

                # Compute probability sum stats (before renorm)
                prob_sums: list[float] = []
                for _snap_key, snap_pred in preds_by_snapshot.items():
                    filtered = {k: v for k, v in snap_pred.items() if k in matched_kalshi_names}
                    if filtered:
                        prob_sums.append(sum(filtered.values()))

                prob_sum_mean = sum(prob_sums) / len(prob_sums) if prob_sums else 0.0
                prob_sum_std = (
                    (sum((s - prob_sum_mean) ** 2 for s in prob_sums) / len(prob_sums)) ** 0.5
                    if prob_sums
                    else 0.0
                )

                # Renormalized predictions
                renorm_preds = _renormalize_all_snapshots(preds_by_snapshot, matched_kalshi_names)

                # Run both variants
                for renorm_label, pred_source in [
                    ("raw", preds_by_snapshot),
                    ("renorm", renorm_preds),
                ]:
                    total_pnl = 0.0
                    total_trades = 0

                    for key in snapshot_keys:
                        moment = build_entry_moment(
                            snapshot_key=key,
                            snapshot_availability=snapshot_availability,
                            preds_by_snapshot=pred_source,
                            matched_names=matched_kalshi_names,
                            market_prices_by_date=market_prices_by_date,
                            hourly_candles=hourly_candles,
                        )
                        if moment is None:
                            continue

                        engine = BacktestEngine(cfg)
                        result = engine.run(
                            moments=[moment],
                            spread_penalties=spread_penalties if spread_penalties else None,
                        )

                        if result.total_trades > 0:
                            settlement = result.settle(winner)
                            total_pnl += settlement.total_pnl
                            total_trades += result.total_trades

                    rows.append(
                        {
                            "year": year,
                            "category": category.slug,
                            "model": short_name,
                            "renorm": renorm_label,
                            "total_pnl": round(total_pnl, 2),
                            "n_trades": total_trades,
                            "prob_sum_mean": round(prob_sum_mean, 4),
                            "prob_sum_std": round(prob_sum_std, 4),
                        }
                    )

                print(f"  {short_name}: prob_sum={prob_sum_mean:.4f}±{prob_sum_std:.4f}")

    return pd.DataFrame(rows)


def print_comparison(df: pd.DataFrame) -> None:
    """Print a clear comparison of raw vs renormalized results."""
    if df.empty:
        print("No results to compare.")
        return

    print("\n" + "=" * 80)
    print("RENORMALIZATION IMPACT ANALYSIS")
    print("=" * 80)

    # Probability sum diagnostics
    raw_df = df[df["renorm"] == "raw"]
    print("\n--- Probability Sum Diagnostics (before renorm) ---")
    print("  How far are filtered predictions from summing to 1.0?")
    diag = raw_df.groupby("model")["prob_sum_mean"].agg(["mean", "min", "max"])
    for model, row in diag.iterrows():
        print(f"  {model:30s}  mean={row['mean']:.4f}  range=[{row['min']:.4f}, {row['max']:.4f}]")

    # Per-model aggregate comparison
    print("\n--- Aggregate PnL: Raw vs Renormalized (per model, summed over years+categories) ---")
    pivot = df.pivot_table(
        index="model", columns="renorm", values="total_pnl", aggfunc="sum"
    ).round(2)
    if "raw" in pivot.columns and "renorm" in pivot.columns:
        pivot["delta"] = (pivot["renorm"] - pivot["raw"]).round(2)
        pivot["delta_pct"] = ((pivot["delta"] / pivot["raw"].abs().clip(lower=0.01)) * 100).round(1)
    print(pivot.to_string())

    # Per-year comparison
    print("\n--- Aggregate PnL by Year ---")
    pivot_year = df.pivot_table(
        index=["year", "model"], columns="renorm", values="total_pnl", aggfunc="sum"
    ).round(2)
    if "raw" in pivot_year.columns and "renorm" in pivot_year.columns:
        pivot_year["delta"] = (pivot_year["renorm"] - pivot_year["raw"]).round(2)
    print(pivot_year.to_string())

    # Trade count comparison
    print("\n--- Trade Count: Raw vs Renormalized ---")
    trade_pivot = df.pivot_table(index="model", columns="renorm", values="n_trades", aggfunc="sum")
    print(trade_pivot.to_string())

    # Summary
    if "raw" in pivot.columns and "renorm" in pivot.columns:
        total_raw = pivot["raw"].sum()
        total_renorm = pivot["renorm"].sum()
        print("\n--- Summary ---")
        print(f"  Total PnL (raw):    ${total_raw:.2f}")
        print(f"  Total PnL (renorm): ${total_renorm:.2f}")
        print(f"  Delta:              ${total_renorm - total_raw:.2f}")
        if abs(total_raw) > 0.01:
            print(f"  Relative change:    {(total_renorm - total_raw) / abs(total_raw) * 100:.1f}%")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    df = run_analysis()

    # Save results
    out_dir = Path("storage/d20260301_renormalization_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results.csv", index=False)
    print(f"\nResults saved to {out_dir / 'results.csv'}")

    print_comparison(df)


if __name__ == "__main__":
    main()
