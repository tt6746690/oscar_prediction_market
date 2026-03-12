"""Audit timing leakage risk with uniform lagged execution windows.

This script is intentionally one-off scoped (no framework changes). It compares:

1) Current backtest convention: same-day daily-close pricing on snapshot date
2) Lagged intraday pricing: execute at inferred_event_time + lag_hours

The goal is to test whether model-market divergence persists after a delay,
rather than relying on immediate post-award repricing windows.

Usage::

    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\
    d20260223_timing_leakage_audit.audit_uniform_lag_hours

Example with custom lags/time::

    uv run python -m oscar_prediction_market.one_offs.\
        d20260223_timing_leakage_audit.audit_uniform_lag_hours \
      --lags 1 6 12 24 \
      --default-event-time-et 21:00
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from oscar_prediction_market.data.awards_calendar import CALENDARS
from oscar_prediction_market.data.oscar_winners import WINNERS_BY_YEAR
from oscar_prediction_market.data.schema import OscarCategory
from oscar_prediction_market.modeling.models import ModelType
from oscar_prediction_market.one_offs.d20260220_backtest_strategies import (
    BACKTEST_MODEL_TYPES,
)
from oscar_prediction_market.trading.backtest import (
    BacktestConfig,
    BacktestEngine,
    BacktestSimulationConfig,
    MarketSnapshot,
)
from oscar_prediction_market.trading.name_matching import match_nominees
from oscar_prediction_market.trading.oscar_data import (
    build_market_prices,
    estimate_category_spreads,
    fetch_and_cache_market_data,
    get_winner_kalshi_name,
)
from oscar_prediction_market.trading.oscar_market import Candle, OscarMarket
from oscar_prediction_market.trading.oscar_prediction_source import (
    load_all_snapshot_predictions,
    load_nomination_dataset,
    translate_predictions,
)
from oscar_prediction_market.trading.schema import (
    BankrollMode,
    FeeType,
    KellyConfig,
    KellyMode,
    PositionDirection,
    TradingConfig,
)
from oscar_prediction_market.trading.temporal_model import (
    get_post_nomination_snapshot_dates,
)

logger = logging.getLogger(__name__)

CEREMONY_YEAR = 2025
BANKROLL = 1000.0
CALENDAR = CALENDARS[CEREMONY_YEAR]
WINNERS = WINNERS_BY_YEAR[CEREMONY_YEAR]
ET = ZoneInfo("America/New_York")

_BACKTEST_STORAGE = Path("storage/d20260220_backtest_strategies") / str(CEREMONY_YEAR)
DATASETS_DIR = _BACKTEST_STORAGE / "datasets"
MODELS_DIR = _BACKTEST_STORAGE / "models"
MARKET_DATA_DIR = _BACKTEST_STORAGE / "market_data"

EXP_DIR = Path("storage/d20260223_timing_leakage_audit")

DEFAULT_CATEGORIES: list[OscarCategory] = [
    OscarCategory.BEST_PICTURE,
    OscarCategory.DIRECTING,
    OscarCategory.ACTOR_LEADING,
    OscarCategory.ACTRESS_LEADING,
    OscarCategory.ACTOR_SUPPORTING,
    OscarCategory.ACTRESS_SUPPORTING,
]
DEFAULT_LAGS_HOURS = [1, 6, 12, 24]
DEFAULT_EVENT_TIME_ET = "21:00"


def _parse_time_hhmm(value: str) -> tuple[int, int]:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid HH:MM time: {value!r}")
    hour = int(parts[0])
    minute = int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Invalid HH:MM time: {value!r}")
    return hour, minute


def _event_datetime_utc(event_date: date, event_time_et: str) -> datetime:
    hour, minute = _parse_time_hhmm(event_time_et)
    event_dt_et = datetime(
        event_date.year,
        event_date.month,
        event_date.day,
        hour,
        minute,
        tzinfo=ET,
    )
    return event_dt_et.astimezone(UTC)


def _infer_event_time_et(
    hourly_candles: list[Candle],
    event_date: date,
    ticker_universe: set[str],
    default_time_et: str,
) -> str:
    """Infer likely event release time from largest market-wide hourly move.

    Heuristic:
    - Restrict to [event_date 12:00 ET, event_date+1 06:00 ET]
    - Compute absolute hourly close change per ticker
    - Aggregate mean abs-change across tickers for each timestamp
    - Pick timestamp with max aggregate move

    Returns HH:MM ET. Falls back to default when data is insufficient.
    """
    if not hourly_candles:
        return default_time_et

    # Build DataFrame locally for vectorized groupby/diff
    rows = [
        {"timestamp": c.timestamp, "ticker": c.ticker, "close": c.close}
        for c in hourly_candles
        if c.ticker in ticker_universe
    ]
    if not rows:
        return default_time_et

    df = pd.DataFrame(rows)
    df = df.sort_values(["ticker", "timestamp"]).copy()
    df["abs_move"] = df.groupby("ticker")["close"].diff().abs()
    df["timestamp_et"] = df["timestamp"].dt.tz_convert(ET)

    start_et = datetime(event_date.year, event_date.month, event_date.day, 12, 0, tzinfo=ET)
    end_et = start_et + timedelta(hours=18)
    df = df[(df["timestamp_et"] >= start_et) & (df["timestamp_et"] <= end_et)]
    if df.empty:
        return default_time_et

    by_ts = df.groupby("timestamp", as_index=True)["abs_move"].mean()
    by_ts = by_ts.dropna()
    if by_ts.empty:
        return default_time_et

    best_ts_utc = pd.Timestamp(by_ts.idxmax())
    best_ts_et = best_ts_utc.tz_convert(ET)
    return best_ts_et.strftime("%H:%M")


def _prices_at_target_timestamp(
    hourly_candles: list[Candle],
    target_ts_utc: datetime,
    ticker_to_name: dict[str, str],
) -> tuple[dict[str, float], dict[str, str], int, int]:
    """Get kalshi-name-space prices using first hourly candle >= target timestamp.

    Returns:
        prices_by_name,
        selected_ts_by_name (ISO UTC),
        n_after_target,
        n_fallback_before_target
    """
    prices_by_name: dict[str, float] = {}
    selected_ts_by_name: dict[str, str] = {}
    n_after_target = 0
    n_fallback_before_target = 0

    for ticker, name in ticker_to_name.items():
        ticker_candles = sorted(
            (c for c in hourly_candles if c.ticker == ticker),
            key=lambda c: c.timestamp,
        )
        if not ticker_candles:
            continue

        after = [c for c in ticker_candles if c.timestamp >= target_ts_utc]
        if after:
            candle = after[0]
            n_after_target += 1
        else:
            before = [c for c in ticker_candles if c.timestamp < target_ts_utc]
            if not before:
                continue
            candle = before[-1]
            n_fallback_before_target += 1

        if candle.close <= 0:
            continue

        prices_by_name[name] = candle.close
        selected_ts_by_name[name] = candle.timestamp.isoformat()

    return prices_by_name, selected_ts_by_name, n_after_target, n_fallback_before_target


def _fetch_hourly_candles_safe(
    *,
    market: OscarMarket,
    tickers: list[str],
    start_date: date,
    end_date: date,
) -> list[Candle]:
    """Fetch hourly candles robustly, falling back to per-ticker fetch on API limits."""
    try:
        return market.fetch_candlestick_history(
            start_date=start_date,
            end_date=end_date,
            period_interval=60,
            tickers=tickers,
        )
    except requests.HTTPError as exc:
        logger.warning("Batch hourly fetch failed (%s). Falling back to per-ticker fetch.", exc)

    all_candles: list[Candle] = []
    for ticker in tickers:
        try:
            candles = market.fetch_candlestick_history(
                start_date=start_date,
                end_date=end_date,
                period_interval=60,
                tickers=[ticker],
            )
        except requests.HTTPError as exc:
            logger.warning("Hourly fetch failed for ticker=%s (%s)", ticker, exc)
            continue

        all_candles.extend(candles)

    return all_candles


def _build_backtest_config() -> BacktestConfig:
    kelly_config = KellyConfig(
        bankroll=BANKROLL,
        kelly_fraction=0.25,
        kelly_mode=KellyMode.INDEPENDENT,
        buy_edge_threshold=0.08,
        max_position_per_outcome=BANKROLL * 0.5,
        max_total_exposure=BANKROLL,
    )
    trading_config = TradingConfig(
        kelly=kelly_config,
        sell_edge_threshold=-1.0,
        fee_type=FeeType.MAKER,
        limit_price_offset=0.01,
        min_price=0,
        allowed_directions=frozenset({PositionDirection.YES}),
    )
    sim_config = BacktestSimulationConfig(
        spread_penalty=0.015,
        bankroll_mode=BankrollMode.FIXED,
    )
    return BacktestConfig(trading=trading_config, simulation=sim_config)


def _run_one_snapshot_settlement(
    predictions: dict[str, float],
    market_prices: dict[str, float],
    winner: str,
    spread_penalties: dict[str, float],
) -> tuple[float, float, int]:
    if not predictions or not market_prices:
        return 0.0, 0.0, 0

    snapshot_key = "2025-02-01"
    engine = BacktestEngine(_build_backtest_config())
    moments = [
        MarketSnapshot(
            timestamp=datetime.combine(date.fromisoformat(snapshot_key), datetime.min.time()),
            predictions=predictions,
            prices=market_prices,
        )
    ]
    result = engine.run(
        moments=moments,
        spread_penalties=spread_penalties if spread_penalties else None,
    )
    try:
        settlement = result.settle(winner)
    except KeyError:
        return 0.0, 0.0, int(result.total_trades)

    return float(settlement.total_pnl), float(settlement.return_pct), int(result.total_trades)


def _compute_edge_metrics(
    predictions: dict[str, float],
    market_prices: dict[str, float],
) -> tuple[float, float, int]:
    edges_pp: list[float] = []
    for name, model_prob in predictions.items():
        if name not in market_prices:
            continue
        edge_pp = (model_prob - market_prices[name]) * 100.0
        edges_pp.append(edge_pp)

    if not edges_pp:
        return 0.0, 0.0, 0

    max_positive = max(max(v, 0.0) for v in edges_pp)
    mean_edge = sum(edges_pp) / len(edges_pp)
    n_positive = sum(1 for v in edges_pp if v > 0)
    return max_positive, mean_edge, n_positive


def audit_category_model(
    category: OscarCategory,
    model_type: ModelType,
    lags_hours: list[int],
    default_event_time_et: str,
    infer_event_time: bool,
    snapshot_info: list[tuple[date, list[str]]],
) -> tuple[list[dict], list[dict]]:
    cat_slug = category.slug
    model_short = model_type.short_name

    fetch_result = fetch_and_cache_market_data(
        category,
        ceremony_year=CEREMONY_YEAR,
        start_date=CALENDAR.oscar_nominations_date_local,
        end_date=CALENDAR.oscar_ceremony_date_local,
        market_data_dir=MARKET_DATA_DIR,
    )
    if fetch_result is None:
        return [], []

    market, daily_candles, _hourly_candles, trades_df = fetch_result
    spread_by_ticker, _mean_spread = estimate_category_spreads(trades_df)

    snapshot_dates = [d for d, _ in snapshot_info]
    snapshot_strs = [str(d) for d in snapshot_dates]

    # Load nomination records & build name-mapping data
    dataset = load_nomination_dataset(DATASETS_DIR, category, snapshot_strs[0])

    preds_by_snapshot = load_all_snapshot_predictions(
        category,
        model_type,
        snapshot_strs,
        MODELS_DIR,
        dataset=dataset,
        ceremony_year=CEREMONY_YEAR,
    )

    first_preds = next(iter(preds_by_snapshot.values()))
    model_names = list(first_preds.keys())
    kalshi_names = list(market.nominee_tickers.keys())

    nominee_map = match_nominees(
        model_names=model_names,
        kalshi_names=kalshi_names,
        category=category,
        ceremony_year=CEREMONY_YEAR,
    )
    if not nominee_map:
        logger.warning("No nominee mapping: category=%s model=%s", cat_slug, model_short)
        return [], []

    matched_kalshi_names = set(nominee_map.values())
    preds_by_snapshot = translate_predictions(preds_by_snapshot, nominee_map)
    winner_kalshi_name = get_winner_kalshi_name(category, matched_kalshi_names, WINNERS)
    spread_penalties = {
        kn: spread_by_ticker[t]
        for kn in matched_kalshi_names
        if (t := market.nominee_tickers.get(kn)) and t in spread_by_ticker
    }

    market_prices_by_date = build_market_prices(daily_candles)

    tickers = list(market.nominee_tickers.values())
    hourly_candles = _fetch_hourly_candles_safe(
        market=market,
        start_date=CALENDAR.oscar_nominations_date_local - timedelta(days=1),
        end_date=CALENDAR.oscar_ceremony_date_local,
        tickers=tickers,
    )

    ticker_to_name = {
        market.nominee_tickers[kn]: kn
        for kn in matched_kalshi_names
        if kn in market.nominee_tickers
    }

    rows: list[dict] = []
    timing_rows: list[dict] = []

    for snap_date, event_labels in snapshot_info:
        snap_str = str(snap_date)
        preds_raw = preds_by_snapshot.get(snap_str, {})
        if not preds_raw:
            continue

        preds = {name: prob for name, prob in preds_raw.items() if name in matched_kalshi_names}
        if not preds:
            continue

        daily_prices = market_prices_by_date.get(snap_str, {})
        if daily_prices:
            pnl, ret_pct, trades = _run_one_snapshot_settlement(
                predictions=preds,
                market_prices=daily_prices,
                winner=winner_kalshi_name,
                spread_penalties=spread_penalties,
            )
            max_pos_edge, mean_edge, n_pos = _compute_edge_metrics(preds, daily_prices)
            rows.append(
                {
                    "category": cat_slug,
                    "model_type": model_short,
                    "snapshot_date": snap_str,
                    "event_labels": " | ".join(event_labels),
                    "price_source": "daily_close_same_date",
                    "lag_hours": 0,
                    "event_time_et": None,
                    "target_timestamp_utc": None,
                    "n_priced_outcomes": len(daily_prices),
                    "n_after_target": None,
                    "n_fallback_before_target": None,
                    "total_pnl": round(pnl, 2),
                    "return_pct": round(ret_pct, 2),
                    "total_trades": trades,
                    "max_positive_edge_pp": round(max_pos_edge, 2),
                    "mean_edge_pp": round(mean_edge, 2),
                    "n_positive_edges": n_pos,
                }
            )

        if not hourly_candles:
            continue

        inferred_time_et = (
            _infer_event_time_et(
                hourly_candles=hourly_candles,
                event_date=snap_date,
                ticker_universe=set(ticker_to_name.keys()),
                default_time_et=default_event_time_et,
            )
            if infer_event_time
            else default_event_time_et
        )

        timing_rows.append(
            {
                "category": cat_slug,
                "model_type": model_short,
                "snapshot_date": snap_str,
                "event_labels": " | ".join(event_labels),
                "event_time_basis": "inferred" if infer_event_time else "fixed",
                "event_time_et": inferred_time_et,
            }
        )

        event_ts_utc = _event_datetime_utc(snap_date, inferred_time_et)

        for lag in lags_hours:
            target_ts = event_ts_utc + timedelta(hours=int(lag))
            lag_prices, selected_ts, n_after, n_fallback = _prices_at_target_timestamp(
                hourly_candles=hourly_candles,
                target_ts_utc=target_ts,
                ticker_to_name=ticker_to_name,
            )
            if not lag_prices:
                continue

            pnl, ret_pct, trades = _run_one_snapshot_settlement(
                predictions=preds,
                market_prices=lag_prices,
                winner=winner_kalshi_name,
                spread_penalties=spread_penalties,
            )
            max_pos_edge, mean_edge, n_pos = _compute_edge_metrics(preds, lag_prices)

            rows.append(
                {
                    "category": cat_slug,
                    "model_type": model_short,
                    "snapshot_date": snap_str,
                    "event_labels": " | ".join(event_labels),
                    "price_source": "intraday_lag_hours",
                    "lag_hours": int(lag),
                    "event_time_et": inferred_time_et,
                    "target_timestamp_utc": target_ts.isoformat(),
                    "n_priced_outcomes": len(lag_prices),
                    "n_after_target": n_after,
                    "n_fallback_before_target": n_fallback,
                    "selected_timestamps_utc": json.dumps(selected_ts, sort_keys=True),
                    "total_pnl": round(pnl, 2),
                    "return_pct": round(ret_pct, 2),
                    "total_trades": trades,
                    "max_positive_edge_pp": round(max_pos_edge, 2),
                    "mean_edge_pp": round(mean_edge, 2),
                    "n_positive_edges": n_pos,
                }
            )

    return rows, timing_rows


def _build_summary(audit_df: pd.DataFrame) -> dict:
    summary: dict = {}

    baseline = audit_df[
        (audit_df["price_source"] == "daily_close_same_date") & (audit_df["lag_hours"] == 0)
    ].copy()
    lagged = audit_df[audit_df["price_source"] == "intraday_lag_hours"].copy()

    summary["baseline"] = {
        "rows": int(len(baseline)),
        "total_pnl_sum": round(float(baseline["total_pnl"].sum()), 2)
        if not baseline.empty
        else 0.0,
        "pct_profitable": (
            round(float((baseline["total_pnl"] > 0).mean()) * 100.0, 1)
            if not baseline.empty
            else 0.0
        ),
    }

    if lagged.empty:
        summary["lag_grid"] = []
        summary["delta_vs_baseline"] = []
        return summary

    lag_stats = (
        lagged.groupby("lag_hours", as_index=False)
        .agg(
            rows=("total_pnl", "size"),
            total_pnl_sum=("total_pnl", "sum"),
            total_pnl_mean=("total_pnl", "mean"),
            median_pnl=("total_pnl", "median"),
            pct_profitable=("total_pnl", lambda s: (s > 0).mean() * 100.0),
            mean_edge_pp=("mean_edge_pp", "mean"),
            mean_max_positive_edge_pp=("max_positive_edge_pp", "mean"),
        )
        .sort_values("lag_hours")
    )
    summary["lag_grid"] = [
        {
            "lag_hours": int(row["lag_hours"]),
            "rows": int(row["rows"]),
            "total_pnl_sum": round(float(row["total_pnl_sum"]), 2),
            "total_pnl_mean": round(float(row["total_pnl_mean"]), 2),
            "median_pnl": round(float(row["median_pnl"]), 2),
            "pct_profitable": round(float(row["pct_profitable"]), 1),
            "mean_edge_pp": round(float(row["mean_edge_pp"]), 2),
            "mean_max_positive_edge_pp": round(float(row["mean_max_positive_edge_pp"]), 2),
        }
        for row in lag_stats.to_dict("records")
    ]

    baseline_keyed = baseline[["category", "model_type", "snapshot_date", "total_pnl"]].rename(
        columns={"total_pnl": "baseline_pnl"}
    )

    merged = lagged.merge(
        baseline_keyed,
        on=["category", "model_type", "snapshot_date"],
        how="left",
    )
    merged["delta_vs_baseline"] = merged["total_pnl"] - merged["baseline_pnl"]

    delta_stats = (
        merged.groupby("lag_hours", as_index=False)
        .agg(
            mean_delta=("delta_vs_baseline", "mean"),
            median_delta=("delta_vs_baseline", "median"),
            sum_delta=("delta_vs_baseline", "sum"),
        )
        .sort_values("lag_hours")
    )

    summary["delta_vs_baseline"] = [
        {
            "lag_hours": int(row["lag_hours"]),
            "mean_delta": round(float(row["mean_delta"]), 2),
            "median_delta": round(float(row["median_delta"]), 2),
            "sum_delta": round(float(row["sum_delta"]), 2),
        }
        for row in delta_stats.to_dict("records")
    ]

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit uniform lag timing assumptions")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=[c.slug for c in DEFAULT_CATEGORIES],
        help="Category slugs (default: 6 core categories from backtest README)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[m.short_name for m in BACKTEST_MODEL_TYPES],
        help="Model short names (lr, clogit, gbt, cal_sgbt)",
    )
    parser.add_argument(
        "--lags",
        nargs="+",
        type=int,
        default=DEFAULT_LAGS_HOURS,
        help="Lag grid in hours",
    )
    parser.add_argument(
        "--default-event-time-et",
        default=DEFAULT_EVENT_TIME_ET,
        help="Fallback assumed event release time in ET (HH:MM)",
    )
    parser.add_argument(
        "--no-infer-event-time",
        action="store_true",
        help="Disable market-based event-time inference; use fixed event time",
    )
    parser.add_argument(
        "--output-subdir",
        default=None,
        help=(
            "Output subdirectory under storage/d20260223_timing_leakage_audit/2025. "
            "Default: lag_audit_inferred or lag_audit_fixed_2100 based on mode."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("oscar_prediction_market.trading.backtest").setLevel(logging.WARNING)

    categories: list[OscarCategory] = []
    for slug in args.categories:
        try:
            categories.append(OscarCategory.from_slug(slug))
        except (KeyError, ValueError) as e:
            raise ValueError(f"Unknown category slug: {slug}") from e

    model_types: list[ModelType] = []
    for short in args.models:
        try:
            model_types.append(ModelType.from_short_name(short))
        except ValueError as e:
            raise ValueError(f"Unknown model short name: {short}") from e

    lags_hours = sorted(set(args.lags))
    infer_event_time = not args.no_infer_event_time
    output_subdir = args.output_subdir
    if output_subdir is None:
        output_subdir = "lag_audit_inferred" if infer_event_time else "lag_audit_fixed_2100"
    audit_dir = EXP_DIR / "2025" / output_subdir

    snapshot_info = get_post_nomination_snapshot_dates(CALENDAR)

    print("=" * 80)
    print("Uniform Lag Audit (Timing Leakage Sensitivity)")
    print("=" * 80)
    print(f"Categories: {[c.slug for c in categories]}")
    print(f"Models: {[m.short_name for m in model_types]}")
    print(f"Lags (hours): {lags_hours}")
    print(f"Event time basis: {'inferred from market' if infer_event_time else 'fixed'}")
    print(f"Default event time ET: {args.default_event_time_et}")

    all_rows: list[dict] = []
    all_timing_rows: list[dict] = []

    total_jobs = len(categories) * len(model_types)
    job_idx = 0

    for category in categories:
        cat_slug = category.slug
        for model_type in model_types:
            model_short = model_type.short_name
            job_idx += 1
            print(f"\n[{job_idx}/{total_jobs}] category={cat_slug} model={model_short}")

            rows, timing_rows = audit_category_model(
                category=category,
                model_type=model_type,
                lags_hours=lags_hours,
                default_event_time_et=args.default_event_time_et,
                infer_event_time=infer_event_time,
                snapshot_info=snapshot_info,
            )
            print(f"  Produced rows: {len(rows)}")
            all_rows.extend(rows)
            all_timing_rows.extend(timing_rows)

    audit_dir.mkdir(parents=True, exist_ok=True)

    assumptions = {
        "ceremony_year": CEREMONY_YEAR,
        "categories": [c.slug for c in categories],
        "models": [m.short_name for m in model_types],
        "lags_hours": lags_hours,
        "event_time_basis": "inferred" if infer_event_time else "fixed",
        "default_event_time_et": args.default_event_time_et,
        "notes": [
            "Baseline uses same-day daily-close prices from period_interval=1440 candles.",
            "Lag audit uses hourly candles and first observed candle at/after target timestamp.",
            "Target timestamp = (event_date + inferred_or_fixed_event_time_et) + lag_hours.",
        ],
    }

    audit_df = pd.DataFrame(all_rows)
    timing_df = pd.DataFrame(all_timing_rows)

    csv_path = audit_dir / "uniform_lag_audit.csv"
    timing_csv_path = audit_dir / "inferred_event_times.csv"
    assumptions_path = audit_dir / "inputs_snapshot_event_times.json"
    summary_path = audit_dir / "uniform_lag_audit_summary.json"

    if not audit_df.empty:
        audit_df.to_csv(csv_path, index=False)
        print(f"\nSaved audit rows: {len(audit_df)} -> {csv_path}")
    else:
        print("\nWARNING: No audit rows generated")

    if not timing_df.empty:
        timing_df = timing_df.drop_duplicates().sort_values(
            ["category", "model_type", "snapshot_date"]
        )
        timing_df.to_csv(timing_csv_path, index=False)
        print(f"Saved timing rows: {len(timing_df)} -> {timing_csv_path}")

    assumptions_path.write_text(json.dumps(assumptions, indent=2))
    print(f"Saved assumptions -> {assumptions_path}")

    summary = _build_summary(audit_df) if not audit_df.empty else {}
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary -> {summary_path}")

    if summary:
        print("\n=== Lag summary ===")
        for row in summary.get("lag_grid", []):
            print(
                "  lag={lag_hours:>2}h rows={rows:>3} sum_pnl={total_pnl_sum:>+8.2f} "
                "mean_pnl={total_pnl_mean:>+7.2f} profitable={pct_profitable:>5.1f}% "
                "mean_edge={mean_edge_pp:>+6.2f}pp".format(**row)
            )


if __name__ == "__main__":
    main()
