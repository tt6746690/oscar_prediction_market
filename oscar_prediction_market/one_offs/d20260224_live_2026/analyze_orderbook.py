"""Fetch live orderbook data for recommended positions and analyze liquidity.

Reads position_summary.csv (produced by run_buy_hold.py), fetches current
orderbooks from Kalshi, and reports execution quality metrics: spread,
depth, VWAP at target size, slippage, and market vs limit order
recommendations.

Usage::

    uv run python -m oscar_prediction_market.one_offs.d20260224_live_2026.analyze_orderbook
    uv run python -m oscar_prediction_market.one_offs.d20260224_live_2026.analyze_orderbook --config recommended --bankroll 1000
    uv run python -m oscar_prediction_market.one_offs.d20260224_live_2026.analyze_orderbook --config all
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from oscar_prediction_market.data.schema import OscarCategory
from oscar_prediction_market.one_offs.d20260224_live_2026 import (
    CEREMONY_YEAR,
    EXP_DIR,
)
from oscar_prediction_market.one_offs.d20260224_live_2026.recommended_configs import (
    OPTION_MODELS,
    RECOMMENDED_CONFIGS,
)
from oscar_prediction_market.trading.edge import get_execution_price
from oscar_prediction_market.trading.kalshi_client import (
    KalshiPublicClient,
    Orderbook,
)
from oscar_prediction_market.trading.market_data import OSCAR_MARKETS
from oscar_prediction_market.trading.oscar_market import OscarMarket
from oscar_prediction_market.trading.schema import Side

_ET = ZoneInfo("America/New_York")

CATEGORY_DISPLAY: dict[str, str] = {
    "best_picture": "Best Picture",
    "directing": "Directing",
    "actor_leading": "Actor (Leading)",
    "actress_leading": "Actress (Leading)",
    "actor_supporting": "Actor (Supporting)",
    "actress_supporting": "Actress (Supporting)",
    "original_screenplay": "Original Screenplay",
    "cinematography": "Cinematography",
    "animated_feature": "Animated Feature",
}

# Spread threshold (cents) for market vs limit order recommendation.
SPREAD_THRESHOLD_CENTS = 2


# ============================================================================
# Helpers
# ============================================================================


def _load_positions(results_dir: Path) -> pd.DataFrame:
    """Load position_summary.csv."""
    path = results_dir / "position_summary.csv"
    if not path.exists():
        print(f"ERROR: {path} not found. Run run_buy_hold.py first.", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path)


def _filter_latest_positions(
    pos_df: pd.DataFrame,
    config_label: str,
    model_type: str,
) -> pd.DataFrame:
    """Get latest-snapshot positions for a specific config and model.

    For each (category, outcome, direction), keep only the row from the
    most recent entry_snapshot.
    """
    mask = (pos_df["config_label"] == config_label) & (pos_df["model_type"] == model_type)
    filtered = pos_df[mask].copy()
    if filtered.empty:
        return filtered

    # Keep only the latest snapshot per (category, outcome, direction)
    filtered = (
        filtered.sort_values("entry_snapshot")
        .groupby(["category", "outcome", "direction"])
        .last()
        .reset_index()
    )
    return filtered


def _build_oscar_markets(
    categories: list[str],
    client: KalshiPublicClient,
) -> dict[str, OscarMarket]:
    """Build OscarMarket instances for each category slug."""
    markets: dict[str, OscarMarket] = {}
    for slug in categories:
        try:
            cat_enum = OscarCategory.from_slug(slug)
        except KeyError:
            print(f"  WARNING: Unknown category slug '{slug}', skipping")
            continue
        try:
            data = OSCAR_MARKETS.get_category_data(cat_enum, CEREMONY_YEAR)
        except KeyError:
            print(f"  WARNING: No market data for {slug} in {CEREMONY_YEAR}")
            continue
        markets[slug] = OscarMarket.from_registry_data(data, client=client)
    return markets


def _get_orderbook_for_nominee(
    mkt: OscarMarket,
    nominee: str,
    depth: int = 10,
) -> tuple[str | None, Orderbook | None]:
    """Look up ticker and fetch orderbook for a nominee.

    Returns (ticker, orderbook) or (None, None) if the nominee isn't found
    or the API call fails.
    """
    ticker = mkt.nominee_tickers.get(nominee)
    if ticker is None:
        return None, None
    try:
        ob = mkt.client.get_orderbook(ticker, depth=depth)
    except Exception as e:
        print(f"  WARNING: Failed to fetch orderbook for {ticker}: {e}")
        return ticker, None
    return ticker, ob


def _orderbook_stats(ob: Orderbook) -> dict[str, float | int | None]:
    """Extract summary stats from an orderbook.

    Returns dict with: best_yes_bid, best_yes_ask, best_no_bid, best_no_ask,
    spread, midpoint, yes_bid_depth, yes_ask_depth, no_bid_depth, no_ask_depth.
    All prices in cents.
    """
    yes_bids = ob.yes  # [[price_cents, qty], ...]
    no_bids = ob.no  # NO bids = YES asks at 100-P

    best_yes_bid = yes_bids[0][0] if yes_bids else 0
    best_no_bid = no_bids[0][0] if no_bids else 0

    # YES ask = 100 - best NO bid
    best_yes_ask = (100 - best_no_bid) if best_no_bid > 0 else 100
    # NO ask = 100 - best YES bid
    best_no_ask = (100 - best_yes_bid) if best_yes_bid > 0 else 100

    yes_spread = (best_yes_ask - best_yes_bid) if (best_yes_bid > 0 and best_no_bid > 0) else None
    yes_midpoint = (best_yes_bid + best_yes_ask) / 2 if yes_spread is not None else None

    no_spread = (best_no_ask - best_no_bid) if (best_no_bid > 0 and best_yes_bid > 0) else None
    no_midpoint = (best_no_bid + best_no_ask) / 2 if no_spread is not None else None

    yes_bid_depth = sum(level[1] for level in yes_bids)
    yes_ask_depth = sum(level[1] for level in no_bids)  # NO bids = YES asks
    no_bid_depth = sum(level[1] for level in no_bids)
    no_ask_depth = sum(level[1] for level in yes_bids)  # YES bids = NO asks

    return {
        "best_yes_bid": best_yes_bid,
        "best_yes_ask": best_yes_ask,
        "best_no_bid": best_no_bid,
        "best_no_ask": best_no_ask,
        "yes_spread": yes_spread,
        "yes_midpoint": yes_midpoint,
        "no_spread": no_spread,
        "no_midpoint": no_midpoint,
        "yes_bid_depth": yes_bid_depth,
        "yes_ask_depth": yes_ask_depth,
        "no_bid_depth": no_bid_depth,
        "no_ask_depth": no_ask_depth,
    }


def _analyze_position(
    ob: Orderbook,
    direction: str,
    n_contracts: int,
) -> dict[str, float | int | str | None]:
    """Analyze execution for a single position against its orderbook.

    For YES direction: we buy YES contracts → consume YES asks (= NO bids).
    For NO direction: we buy NO contracts → consume NO asks (= YES bids).
    In both cases we use Side.BUY, but the relevant side of the book differs.

    The get_execution_price function works on YES contracts:
      - Side.BUY consumes NO bids (= YES asks) → used for buying YES
      - Side.SELL consumes YES bids (= NO asks) → used for buying NO

    For buying NO, we need to flip: buying NO at price P is equivalent to
    selling YES at price (1-P). But get_execution_price with Side.SELL
    walks YES bids descending, which gives us the NO ask prices ascending.

    Returns dict with analysis metrics.
    """
    stats = _orderbook_stats(ob)

    if direction == "yes":
        # Buy YES → consume NO bids (converted to YES asks)
        best_bid = stats["best_yes_bid"]
        best_ask = stats["best_yes_ask"]
        spread = stats["yes_spread"]
        midpoint = stats["yes_midpoint"]
        depth = stats["yes_ask_depth"]  # Depth available to buy into
    else:
        best_bid = stats["best_no_bid"]
        best_ask = stats["best_no_ask"]
        spread = stats["no_spread"]
        midpoint = stats["no_midpoint"]
        depth = stats["no_ask_depth"]  # Depth available to buy into

    result: dict[str, float | int | str | None] = {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "midpoint": midpoint,
        "depth": depth,
    }

    if spread is None or midpoint is None:
        result["vwap_cents"] = None
        result["slippage_cents"] = None
        result["recommendation"] = "No market — orderbook empty"
        return result

    # Compute VWAP execution price
    if direction == "yes":
        exec_result = get_execution_price(ob, Side.BUY, n_contracts)
    else:
        # For NO buy: we need a separate orderbook interpretation.
        # Construct a flipped orderbook: swap yes↔no so BUY walks NO asks.
        flipped_ob = Orderbook(yes=ob.no, no=ob.yes)
        exec_result = get_execution_price(flipped_ob, Side.BUY, n_contracts)

    vwap_cents = exec_result.execution_price * 100 if exec_result.n_contracts_fillable > 0 else None
    fillable = exec_result.n_contracts_fillable

    slippage = None
    if vwap_cents is not None and midpoint is not None:
        slippage = round(vwap_cents - midpoint, 2)

    result["vwap_cents"] = round(vwap_cents, 2) if vwap_cents is not None else None
    result["slippage_cents"] = slippage
    result["fillable"] = fillable
    result["is_partial"] = exec_result.is_partial
    result["levels_consumed"] = exec_result.levels_consumed

    # Recommendation
    depth_int = int(depth) if depth is not None else 0
    if depth_int == 0:
        result["recommendation"] = "No liquidity"
    elif spread is not None and spread <= SPREAD_THRESHOLD_CENTS and depth_int >= n_contracts:
        result["recommendation"] = "Market order OK"
    else:
        result["recommendation"] = "Use limit order"

    return result


# ============================================================================
# Report generation
# ============================================================================


def _format_cents(value: float | int | None) -> str:
    """Format a value in cents for display."""
    if value is None:
        return "—"
    return f"{value:.1f}¢" if isinstance(value, float) else f"{value}¢"


def _format_config_header(config_name: str) -> str:
    """Format config name with key parameters."""
    cfg = RECOMMENDED_CONFIGS[config_name]
    model = OPTION_MODELS[config_name]
    kc = cfg.trading.kelly
    dirs = "YES+NO" if len(cfg.trading.allowed_directions) > 1 else "YES only"
    return (
        f"## Config: {config_name.title()} "
        f"(kf={kc.kelly_fraction}, e={kc.buy_edge_threshold}, "
        f"{kc.kelly_mode.value}, {dirs}, {cfg.trading.fee_type.value}, model={model})"
    )


def generate_report(
    pos_df: pd.DataFrame,
    config_names: list[str],
    client: KalshiPublicClient,
    bankroll: float,
) -> str:
    """Generate the full orderbook analysis markdown report.

    Args:
        pos_df: Full position_summary DataFrame.
        config_names: Which config(s) to analyze.
        client: Shared KalshiPublicClient instance.
        bankroll: Per-category bankroll (for context only — positions already sized).

    Returns:
        Markdown string.
    """
    now_et = datetime.now(_ET)
    lines: list[str] = [
        "# Orderbook & Liquidity Analysis",
        "",
        f"**Fetched:** {now_et.strftime('%Y-%m-%d %H:%M')} ET",
        f"**Bankroll per category:** ${bankroll:,.0f}",
        "",
    ]

    # Build OscarMarket instances (shared across configs)
    all_categories = sorted(pos_df["category"].unique())
    markets = _build_oscar_markets(list(all_categories), client)

    for config_name in config_names:
        cfg = RECOMMENDED_CONFIGS[config_name]
        model_type = OPTION_MODELS[config_name]

        lines.append(_format_config_header(config_name))
        lines.append("")

        latest_positions = _filter_latest_positions(pos_df, cfg.label, model_type)

        if latest_positions.empty:
            lines.append("*No positions for this config.*")
            lines.append("")
            continue

        # Group by category
        for cat_slug in sorted(latest_positions["category"].unique()):
            display_name = CATEGORY_DISPLAY.get(cat_slug, cat_slug)
            lines.append(f"### {display_name}")
            lines.append("")

            cat_positions = latest_positions[latest_positions["category"] == cat_slug]
            mkt = markets.get(cat_slug)

            if mkt is None:
                lines.append(f"*No market data for {cat_slug}.*")
                lines.append("")
                continue

            # Table header
            lines.append(
                "| Position | Snapshot | Contracts | Side | "
                "Best Bid | Best Ask | Spread | Midpoint | "
                "Depth | VWAP | Slippage | Fill |"
            )
            lines.append(
                "|----------|----------|-----------|------|"
                "----------|----------|--------|----------|"
                "-------|------|----------|------|"
            )

            recommendations: list[str] = []

            for _, row in cat_positions.iterrows():
                outcome = row["outcome"]
                direction = row["direction"]
                n_contracts = int(row["contracts"])
                snapshot = row["entry_snapshot"]

                ticker, ob = _get_orderbook_for_nominee(mkt, outcome)

                if ob is None:
                    lines.append(
                        f"| {direction.upper()} {outcome} | {snapshot} | "
                        f"{n_contracts:,} | Buy {direction.upper()} | "
                        f"— | — | — | — | — | — | — | — |"
                    )
                    recommendations.append(
                        f"- **{outcome}**: No orderbook data"
                        + (" (ticker not found)" if ticker is None else "")
                    )
                    continue

                analysis = _analyze_position(ob, direction, n_contracts)

                # Extract numeric values with explicit casts for type safety
                _best_bid = analysis["best_bid"]
                _best_ask = analysis["best_ask"]
                _spread = analysis["spread"]
                _midpoint = analysis["midpoint"]
                _depth = analysis["depth"]
                _vwap = analysis["vwap_cents"]
                _slippage = analysis["slippage_cents"]
                _fillable = analysis.get("fillable", 0)
                _is_partial = analysis.get("is_partial", True)

                best_bid_s = _format_cents(_best_bid if not isinstance(_best_bid, str) else None)
                best_ask_s = _format_cents(_best_ask if not isinstance(_best_ask, str) else None)
                spread_s = _format_cents(_spread if not isinstance(_spread, str) else None)
                midpoint_str = _format_cents(_midpoint if not isinstance(_midpoint, str) else None)
                depth_str = (
                    f"{_depth:,}" if _depth is not None and not isinstance(_depth, str) else "—"
                )
                vwap_s = _format_cents(_vwap if not isinstance(_vwap, str) else None)
                slippage_str = (
                    f"{_slippage:+.1f}¢"
                    if _slippage is not None and not isinstance(_slippage, str)
                    else "—"
                )
                fill_str = f"{_fillable:,}/{n_contracts:,}" if _is_partial else f"{n_contracts:,} ✓"

                lines.append(
                    f"| {direction.upper()} {outcome} | {snapshot} | "
                    f"{n_contracts:,} | Buy {direction.upper()} | "
                    f"{best_bid_s} | {best_ask_s} | {spread_s} | {midpoint_str} | "
                    f"{depth_str} | {vwap_s} | {slippage_str} | {fill_str} |"
                )

                # Build recommendation
                rec = str(analysis["recommendation"] or "")
                recommendations.append(
                    _build_recommendation(outcome, direction, n_contracts, analysis, rec)
                )

            lines.append("")

            # Recommendations
            if recommendations:
                lines.append("**Recommendations:**")
                lines.append("")
                for rec in recommendations:
                    lines.append(rec)
                lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _build_recommendation(
    outcome: str,
    direction: str,
    n_contracts: int,
    analysis: dict[str, float | int | str | None],
    rec: str,
) -> str:
    """Build a recommendation string for one position."""
    # Extract numeric values with explicit float() casts for type safety.
    spread_raw = analysis["spread"]
    midpoint_raw = analysis["midpoint"]
    best_bid_raw = analysis["best_bid"]
    vwap_raw = analysis["vwap_cents"]

    spread_v: float | None = float(spread_raw) if isinstance(spread_raw, (int, float)) else None
    midpoint_v: float | None = (
        float(midpoint_raw) if isinstance(midpoint_raw, (int, float)) else None
    )
    best_bid_v: float | None = (
        float(best_bid_raw) if isinstance(best_bid_raw, (int, float)) else None
    )
    vwap_v: float | None = float(vwap_raw) if isinstance(vwap_raw, (int, float)) else None

    parts: list[str] = [f"- **{direction.upper()} {outcome}** ({n_contracts:,} contracts): {rec}"]

    if rec == "Market order OK":
        if vwap_v is not None:
            cost = n_contracts * vwap_v / 100
            parts.append(f"  - VWAP fill: {vwap_v:.1f}¢ → cost ${cost:,.0f}")
        return "\n".join(parts)

    if rec == "Use limit order" and midpoint_v is not None:
        # Show limit order price options
        mid = midpoint_v
        aggressive: float | None = (best_bid_v + 1) if best_bid_v else None

        mid_cost = n_contracts * mid / 100
        parts.append(f"  - Midpoint fill: {mid:.1f}¢ → cost ${mid_cost:,.0f}")

        if aggressive is not None:
            agg_cost = n_contracts * aggressive / 100
            parts.append(
                f"  - Best bid+1¢: {aggressive:.0f}¢ → cost ${agg_cost:,.0f} (may take longer to fill)"
            )

        if vwap_v is not None:
            vwap_cost = n_contracts * vwap_v / 100
            parts.append(f"  - Market order VWAP: {vwap_v:.1f}¢ → cost ${vwap_cost:,.0f}")

        spread_display = spread_v if spread_v is not None else 0
        parts.append(
            f"  - Suggested: Place limit at {mid:.1f}¢ (midpoint), "
            f"adjust if not filled in 1hr. Spread is {spread_display}¢."
        )
        return "\n".join(parts)

    if rec in ("No liquidity", "No market — orderbook empty"):
        parts.append("  - Wait for liquidity or post limit order at model fair value.")

    return "\n".join(parts)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze live orderbook for recommended positions."
    )
    parser.add_argument(
        "--config",
        default="edge_20_taker",
        choices=list(RECOMMENDED_CONFIGS.keys()) + ["all"],
        help="Which trading config to analyze (default: recommended).",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Per-category bankroll in dollars (default: 1000).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=10,
        help="Orderbook depth to fetch (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write markdown output (default: EXP_DIR/results/orderbook_analysis.md).",
    )
    args = parser.parse_args()

    # Determine config(s) to analyze
    if args.config == "all":
        config_names = list(RECOMMENDED_CONFIGS.keys())
    else:
        config_names = [args.config]

    print(f"Loading positions from {EXP_DIR}/results/position_summary.csv ...")
    results_dir = Path(EXP_DIR) / "results"
    pos_df = _load_positions(results_dir)
    print(f"  {len(pos_df)} position rows loaded")

    # Show which configs we're analyzing
    for name in config_names:
        cfg = RECOMMENDED_CONFIGS[name]
        model = OPTION_MODELS[name]
        n = len(_filter_latest_positions(pos_df, cfg.label, model))
        print(f"  Config '{name}' ({model}): {n} latest-snapshot positions")

    print()
    print("Fetching live orderbooks from Kalshi ...")
    client = KalshiPublicClient()

    report = generate_report(pos_df, config_names, client, args.bankroll)

    # Print to stdout
    print()
    print(report)

    # Write markdown file (to storage results dir by default)
    default_output = str(Path(EXP_DIR) / "results" / "orderbook_analysis.md")
    output_path = args.output or default_output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(report)
    print(f"\nReport written to {output_path}")


if __name__ == "__main__":
    main()
