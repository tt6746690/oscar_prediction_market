"""Generate Markdown trading report for live 2026 Oscar predictions.

Reads the buy-hold backtest results (scenario_pnl.csv, model_vs_market.csv,
position_summary.csv, scenario_pnl_agg.csv) and produces one report per
recommended config (taker + maker). Each report uses the config's fee model
(taker or maker with limit_price_offset) showing:

1. Live portfolio sections (account summary, positions, adjustments) — actionable
2. Model positions from backtest (scaled to bankroll)
3. Edge detail per category (avg_ensemble: raw edge, spread, fees, net edge)
4. Category allocation — maxedge_100 weights and bankroll
5. Model probabilities vs market prices (all models)
6. Scenario P&L per category (what happens if nominee X wins?)
7. Monte Carlo portfolio simulation (with allocation scaling)

Reports are written to the one-off source directory by default:
``oscar_prediction_market/one_offs/d20260224_live_2026/reports/``

Usage::

    cd "$(git rev-parse --show-toplevel)"

    # Generate report
    uv run python -m oscar_prediction_market.one_offs.\\
d20260224_live_2026.generate_report

    # Custom bankroll
    uv run python -m oscar_prediction_market.one_offs.\\
d20260224_live_2026.generate_report --bankroll 5000
"""

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from oscar_prediction_market.one_offs.d20260224_live_2026 import (
    CEREMONY_YEAR,
    EXP_DIR,
)
from oscar_prediction_market.one_offs.d20260224_live_2026.live_portfolio import (
    LivePortfolio,
    PositionAdjustment,
    compute_adjustments,
    compute_target_positions,
    fetch_live_portfolio,
)
from oscar_prediction_market.one_offs.d20260224_live_2026.recommended_configs import (
    OPTION_MODELS,
    RECOMMENDED_CONFIGS,
)
from oscar_prediction_market.one_offs.d20260225_buy_hold_backtest.portfolio_simulation import (
    CategoryScenario,
    compute_portfolio_mc_metrics,
    sample_portfolio_pnl,
)
from oscar_prediction_market.trading.edge import Edge
from oscar_prediction_market.trading.schema import FeeType, PositionDirection

# ============================================================================
# Constants
# ============================================================================

#: Per-category bankroll used in the backtest (hardcoded in run_buy_hold.py).
BACKTEST_BANKROLL_PER_CATEGORY = 1000.0

#: Display names for categories.
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

#: Kalshi market URLs for each category (series-level, auto-redirects to current event).
KALSHI_URLS: dict[str, str] = {
    "best_picture": "https://kalshi.com/markets/kxoscarpic",
    "directing": "https://kalshi.com/markets/kxoscardir",
    "actor_leading": "https://kalshi.com/markets/kxoscaracto",
    "actress_leading": "https://kalshi.com/markets/kxoscaractr",
    "actor_supporting": "https://kalshi.com/markets/kxoscarsupacto",
    "actress_supporting": "https://kalshi.com/markets/kxoscarsupactr",
    "original_screenplay": "https://kalshi.com/markets/kxoscarsplay",
    "cinematography": "https://kalshi.com/markets/kxoscarcine",
    "animated_feature": "https://kalshi.com/markets/kxoscaranimated",
}

#: Config label → human-readable name mapping.
CONFIG_LABEL_TO_NAME: dict[str, str] = {
    cfg.label: name for name, cfg in RECOMMENDED_CONFIGS.items()
}

#: Default output directory — one-off source dir / reports.
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "reports"


def _short_snapshot(sk: str) -> str:
    """Format snapshot key for display.

    '2026-03-01_sag' → 'SAG (Mar 1)'
    'live_2026-03-03T15:30Z' → 'Live (Mar 3 15:30 ET)'
    """
    if sk.startswith("live_"):
        # Parse ISO timestamp: live_2026-03-03T15:30Z
        ts_str = sk[5:]  # Remove "live_" prefix
        try:
            utc_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            et_dt = utc_dt.astimezone(ZoneInfo("America/New_York"))
            month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr"}
            return (
                f"Live ({month_names.get(et_dt.month, str(et_dt.month))} "
                f"{et_dt.day} {et_dt.strftime('%H:%M')} ET)"
            )
        except (ValueError, IndexError):
            return f"Live ({ts_str})"
    # Original format: 2026-03-01_sag
    idx = sk.index("_")
    date_part = sk[:idx]
    event = sk[idx + 1 :]
    _year, month_s, day_s = date_part.split("-")
    month = int(month_s)
    day = int(day_s)
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr"}
    return f"{event.upper()} ({month_names.get(month, str(month))} {day})"


# ============================================================================
# Data loading
# ============================================================================


def load_results(
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all 4 result CSVs.

    Returns:
        (scenario_pnl, model_vs_market, position_summary, scenario_pnl_agg)
    """
    pnl = pd.read_csv(results_dir / "scenario_pnl.csv")
    mvm = pd.read_csv(results_dir / "model_vs_market.csv")
    pos = pd.read_csv(results_dir / "position_summary.csv")
    agg = pd.read_csv(results_dir / "scenario_pnl_agg.csv")
    return pnl, mvm, pos, agg


# ============================================================================
# Monte Carlo portfolio simulation
# ============================================================================


def monte_carlo_portfolio(
    scenario_pnl: pd.DataFrame,
    model_vs_market: pd.DataFrame,
    position_summary: pd.DataFrame,
    config_label: str,
    model_type: str,
    entry_snapshot: str,
    scale_factor: float,
    bankroll: float,
    n_samples: int = 10_000,
    rng_seed: int = 42,
    allocation_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Run Monte Carlo simulation of portfolio P&L across all categories.

    Uses ``CategoryScenario`` + ``sample_portfolio_pnl`` from
    ``trading.portfolio_simulation`` for the MC sampling, then adds
    per-capital loss metrics on top.

    Returns:
        Dict with canonical keys from compute_portfolio_mc_metrics
        (mean_pnl, median_pnl, std_pnl, min_pnl, max_pnl, cvar_05,
        cvar_10, prob_profit, prob_loss_10pct_bankroll,
        prob_loss_20pct_bankroll, total_capital_deployed,
        pct_bankroll_deployed, expected_roic) plus
        prob_loss_10pct_capital, prob_loss_20pct_capital.
    """
    rng = np.random.default_rng(rng_seed)

    # Filter to this config/model/entry
    pnl_slice = scenario_pnl[
        (scenario_pnl["config_label"] == config_label)
        & (scenario_pnl["model_type"] == model_type)
        & (scenario_pnl["entry_snapshot"] == entry_snapshot)
    ]

    # Get model probabilities per category
    probs_slice = model_vs_market[
        (model_vs_market["model_type"] == model_type)
        & (model_vs_market["snapshot_key"] == entry_snapshot)
    ]

    # Compute total capital deployed from position_summary
    pos_slice = position_summary[
        (position_summary["config_label"] == config_label)
        & (position_summary["model_type"] == model_type)
        & (position_summary["entry_snapshot"] == entry_snapshot)
    ]
    raw_capital_deployed = pos_slice["outlay_dollars"].sum() if len(pos_slice) > 0 else 0.0
    if allocation_weights and len(pos_slice) > 0:
        weighted_capital = 0.0
        for cat, group in pos_slice.groupby("category"):
            w = allocation_weights.get(str(cat), 1.0)
            weighted_capital += group["outlay_dollars"].sum() * w
        total_capital_deployed = weighted_capital * scale_factor
    else:
        total_capital_deployed = raw_capital_deployed * scale_factor

    categories = sorted(pnl_slice["category"].unique())

    # Build per-category CategoryScenario objects
    cat_scenarios: dict[str, CategoryScenario] = {}
    for cat in categories:
        cat_pnl = pnl_slice[pnl_slice["category"] == cat]
        cat_probs = probs_slice[probs_slice["category"] == cat]

        prob_map = dict(zip(cat_probs["nominee"], cat_probs["model_prob"], strict=False))
        winners = sorted(cat_pnl["assumed_winner"].unique())
        winners = [w for w in winners if w != "none"]

        if not winners:
            continue

        pnl_by_winner = {}
        for _, row in cat_pnl.iterrows():
            if row["assumed_winner"] != "none":
                pnl_by_winner[row["assumed_winner"]] = row["total_pnl"]

        probs = np.array([prob_map.get(w, 0.0) for w in winners])
        pnls = np.array([pnl_by_winner.get(w, 0.0) for w in winners])

        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = np.ones(len(winners)) / len(winners)

        cat_scenarios[cat] = CategoryScenario(
            winners=np.array(winners),
            probs=probs,
            pnls=pnls,
        )

    # Apply allocation weights to per-category P&L
    if allocation_weights:
        for cat, scenario in cat_scenarios.items():
            w = allocation_weights.get(cat, 1.0)
            cat_scenarios[cat] = CategoryScenario(
                winners=scenario.winners,
                probs=scenario.probs,
                pnls=scenario.pnls * w,
            )

    empty_result: dict[str, float] = {
        "mean_pnl": 0.0,
        "median_pnl": 0.0,
        "std_pnl": 0.0,
        "min_pnl": 0.0,
        "max_pnl": 0.0,
        "cvar_05": 0.0,
        "cvar_10": 0.0,
        "prob_profit": 0.0,
        "prob_loss_10pct_bankroll": 0.0,
        "prob_loss_20pct_bankroll": 0.0,
        "prob_loss_10pct_capital": 0.0,
        "prob_loss_20pct_capital": 0.0,
        "total_capital_deployed": total_capital_deployed,
        "pct_bankroll_deployed": (total_capital_deployed / bankroll * 100) if bankroll > 0 else 0.0,
        "expected_roic": 0.0,
    }

    if not cat_scenarios:
        return empty_result

    # Monte Carlo sampling via trading.portfolio_simulation
    total_pnls = sample_portfolio_pnl(cat_scenarios, n_samples, rng)

    # Apply scaling
    total_pnls = total_pnls * scale_factor

    # Compute standard metrics
    mc = compute_portfolio_mc_metrics(total_pnls, bankroll, total_capital_deployed)

    # Add per-capital loss metrics not in the standard MC metrics
    mc["prob_loss_10pct_capital"] = (
        float(np.mean(total_pnls < -0.10 * total_capital_deployed))
        if total_capital_deployed > 0
        else 0.0
    )
    mc["prob_loss_20pct_capital"] = (
        float(np.mean(total_pnls < -0.20 * total_capital_deployed))
        if total_capital_deployed > 0
        else 0.0
    )

    return mc


# ============================================================================
# Category allocation (maxedge_100)
# ============================================================================


def compute_allocation_weights(
    model_vs_market: pd.DataFrame,
    position_summary: pd.DataFrame,
    model_type: str,
    config_label: str,
    entry_snapshot: str,
) -> pd.DataFrame:
    """Compute maxedge_100 per-category allocation weights.

    The maxedge_100 strategy allocates bankroll proportionally to
    ``max(model_prob - market_prob)`` per category — the single nominee
    where the model most exceeds the market price.  Categories with no
    positions (inactive) receive zero weight. Weights are normalized so
    ``sum = n_categories``, meaning weight=1.0 is equivalent to uniform.

    Logic inlined from ``d20260305_portfolio_kelly.shared.compute_weights``
    to avoid cross-one-off import coupling.

    Returns:
        DataFrame with columns: category, max_edge, is_active, weight,
        display_name.
    """
    # Compute max edge per category from model vs market
    snapshot_col = "snapshot_key"
    mvm_slice = model_vs_market[
        (model_vs_market["model_type"] == model_type)
        & (model_vs_market[snapshot_col] == entry_snapshot)
    ]

    max_edges: dict[str, float] = {}
    for cat, group in mvm_slice.groupby("category"):
        edge = (group["model_prob"] - group["market_prob"]).max()
        max_edges[str(cat)] = float(edge)

    # Determine active categories (have positions)
    pos_slice = position_summary[
        (position_summary["config_label"] == config_label)
        & (position_summary["model_type"] == model_type)
        & (position_summary["entry_snapshot"] == entry_snapshot)
    ]
    active_cats = set(pos_slice["category"].unique())

    # Build per-category data
    all_cats = sorted(CATEGORY_DISPLAY.keys())
    n = len(all_cats)
    rows = []
    for cat in all_cats:
        is_active = cat in active_cats
        me = max(max_edges.get(cat, 0.0), 0.0)
        rows.append(
            {
                "category": cat,
                "max_edge": me,
                "is_active": is_active,
                "display_name": CATEGORY_DISPLAY.get(cat, cat),
            }
        )

    df = pd.DataFrame(rows)

    # Signal-proportional weights for active categories
    signal = np.where(df["is_active"], df["max_edge"].clip(lower=0), 0.0)
    total_signal = signal.sum()

    if total_signal > 0:
        weights = signal / total_signal * n
    else:
        # Fallback: equal among active
        n_active = df["is_active"].sum()
        if n_active > 0:
            weights = np.where(df["is_active"], n / n_active, 0.0)
        else:
            weights = np.ones(n)

    df["weight"] = weights
    return df


# ============================================================================
# Report sections
# ============================================================================


def _header(
    config_name: str,
    bankroll: float,
    entry_timestamp: str,
    now_str: str,
) -> str:
    """Report header with config details, bankroll, and dates."""
    cfg = RECOMMENDED_CONFIGS[config_name]
    kc = cfg.trading.kelly
    n_categories = len(CATEGORY_DISPLAY)
    bankroll_per_cat = bankroll / n_categories

    km = "multi_outcome" if "multi" in kc.kelly_mode.value else "independent"
    dirs = "YES + NO" if len(cfg.trading.allowed_directions) > 1 else "YES only"

    return f"""# 2026 Oscar Trading Report — {config_name.title()}

**Generated:** {now_str}
**Market prices fetched:** {entry_timestamp}
**Ceremony:** March 15, 2026

**Config:** {config_name.title()}
| Parameter | Value |
|-----------|-------|
| Kelly Fraction | {kc.kelly_fraction:.2f} |
| Edge Threshold | {kc.buy_edge_threshold:.2f} |
| Kelly Mode | {km} |
| Directions | {dirs} |
| Fee Type | {cfg.trading.fee_type.value} |
| Limit Price Offset | {cfg.trading.limit_price_offset:.2f} |
| Allocation | maxedge_100 (signal-proportional) |

**Bankroll:** ${bankroll:,.0f}
**Bankroll per category:** ${bankroll_per_cat:,.2f} ({n_categories} categories)
"""


def _model_vs_market_section(
    mvm: pd.DataFrame,
    entry_snapshot: str,
) -> str:
    """Model probabilities vs market prices for all models at latest entry.

    avg_ensemble column is placed right after Market and its header is bolded.
    Remaining model columns are in alphabetical order.
    """
    lines = [f"## Model Probabilities vs Market Prices — {_short_snapshot(entry_snapshot)}\n"]

    snapshot_data = mvm[mvm["snapshot_key"] == entry_snapshot]
    categories = sorted(snapshot_data["category"].unique())

    for cat in categories:
        cat_data = snapshot_data[snapshot_data["category"] == cat].copy()
        display_name = CATEGORY_DISPLAY.get(cat, cat)
        url = KALSHI_URLS.get(cat, "")
        heading = f"[{display_name}]({url})" if url else display_name
        lines.append(f"### {heading}\n")

        nominees = sorted(cat_data["nominee"].unique())
        all_models = sorted(cat_data["model_type"].unique())

        # avg_ensemble first (bolded), then remaining alphabetically
        other_models = [m for m in all_models if m != "avg_ensemble"]
        ordered_models = (["avg_ensemble"] if "avg_ensemble" in all_models else []) + other_models

        # Build header
        header = "| Nominee | Market |"
        sep = "|---------|--------|"
        for m in ordered_models:
            label = f" **{m}** |" if m == "avg_ensemble" else f" {m} |"
            header += label
            sep += "--------|"
        lines.append(header)
        lines.append(sep)

        for nominee in nominees:
            nom_data = cat_data[cat_data["nominee"] == nominee]
            market_p = nom_data["market_prob"].iloc[0] if len(nom_data) > 0 else 0
            row = f"| {nominee} | {market_p:5.1%} |"
            for m in ordered_models:
                m_data = nom_data[nom_data["model_type"] == m]
                if len(m_data) > 0:
                    p = m_data["model_prob"].iloc[0]
                    diff = p - market_p
                    arrow = "▲" if diff > 0.05 else ("▼" if diff < -0.05 else " ")
                    row += f" {p:5.1%}{arrow}|"
                else:
                    row += "   —   |"
            lines.append(row)
        lines.append("")

    return "\n".join(lines)


def _compute_edge_rows(
    mvm: pd.DataFrame,
    entry_snapshot: str,
    config_name: str,
    model_type: str = "avg_ensemble",
    min_raw_edge: float = 0.10,
) -> list[dict]:
    """Compute per-nominee edge detail for a single config's fee model.

    Constructs canonical ``Edge`` objects using the config's fee_type and
    limit_price_offset. Both YES-side and NO-side edges are evaluated.

    For YES direction:
      - offset > 0 (maker): exec_price = market_p - hs + offset  (bid + offset)
      - offset = 0 (taker): exec_price = market_p + hs  (ask)
    For NO direction:
      - offset > 0 (maker): exec_price = 1.0 - (market_p + hs) + offset  (no_bid + offset)
      - offset = 0 (taker): exec_price = 1.0 - (market_p - hs)  (no_ask)

    Returns list of dicts with keys: category, nominee, direction, model_prob,
    market_prob, half_spread, raw_edge, exec_price, fee, net_edge, edge_obj.
    """
    cfg = RECOMMENDED_CONFIGS[config_name]
    fee_type = cfg.trading.fee_type
    offset = cfg.trading.limit_price_offset

    snapshot_data = mvm[(mvm["snapshot_key"] == entry_snapshot) & (mvm["model_type"] == model_type)]

    if "half_spread" not in snapshot_data.columns:
        raise ValueError(
            "Column 'half_spread' missing from model_vs_market data. "
            "Re-run the backtest with spread data included."
        )

    rows: list[dict] = []
    for _, r in snapshot_data.iterrows():
        model_p = float(r["model_prob"])
        market_p = float(r["market_prob"])
        hs = float(r["half_spread"])

        # YES side
        if offset > 0:
            # Maker: bid + offset
            yes_exec = max(0.0, min(1.0, market_p - hs + offset))
        else:
            # Taker: ask
            yes_exec = min(1.0, market_p + hs)
        yes_raw = model_p - market_p

        # NO side
        if offset > 0:
            # Maker: no_bid + offset
            no_exec = max(0.0, min(1.0, 1.0 - (market_p + hs) + offset))
        else:
            # Taker: no_ask
            no_exec = min(1.0, 1.0 - (market_p - hs))
        no_raw = (1.0 - model_p) - (1.0 - market_p)  # = market_p - model_p

        for direction, raw, exec_p, dir_model_p, dir_market_p in [
            ("YES", yes_raw, yes_exec, model_p, market_p),
            ("NO", no_raw, no_exec, 1.0 - model_p, 1.0 - market_p),
        ]:
            if raw >= min_raw_edge:
                edge = Edge(
                    outcome=r["nominee"],
                    direction=PositionDirection(direction.lower()),
                    model_prob=dir_model_p,
                    execution_price=max(0.0, min(1.0, exec_p)),
                    fee_type=fee_type,
                )
                rows.append(
                    {
                        "category": r["category"],
                        "nominee": r["nominee"],
                        "direction": direction,
                        "model_prob": dir_model_p,
                        "market_prob": dir_market_p,
                        "half_spread": hs,
                        "raw_edge": raw,
                        "exec_price": exec_p,
                        "fee": edge.fee,
                        "net_edge": edge.net_edge,
                        "edge_obj": edge,
                    }
                )

    # Sort by raw_edge descending
    rows.sort(key=lambda x: -x["raw_edge"])
    return rows


def _edge_detail_section(
    mvm: pd.DataFrame,
    entry_snapshot: str,
    config_name: str,
    model_type: str = "avg_ensemble",
) -> str:
    """Per-category edge detail table for the primary model.

    Shows raw edge, spread, fee, net edge, and threshold status
    for all nominees with > 10% raw edge, using the config's fee model.
    """
    cfg = RECOMMENDED_CONFIGS[config_name]
    threshold = cfg.trading.kelly.buy_edge_threshold
    fee_type = cfg.trading.fee_type
    offset = cfg.trading.limit_price_offset

    edge_rows = _compute_edge_rows(mvm, entry_snapshot, config_name, model_type)
    if not edge_rows:
        return ""

    # Build caption describing fee model
    if offset > 0:
        fee_desc = f"maker fees, bid+{offset * 100:.0f}¢"
    else:
        fee_desc = "taker fees"

    lines = [f"## Edge Detail ({model_type}) — {_short_snapshot(entry_snapshot)}\n"]
    lines.append(
        f"*Threshold: {threshold:.0%} net edge. Shows nominees with > 10% raw edge. "
        f"Spread = orderbook half-spread. "
        f"Fee model: {fee_desc} ({fee_type.value}). Fee = Kalshi variance-based fee.*\n"
    )

    lines.append(
        "| Category | Nominee | Dir | Raw Edge | Spread | Exec | Fee | **Net Edge** | Status |"
    )
    lines.append(
        "|----------|---------|-----|----------|--------|------|-----|-------------|--------|"
    )

    for r in edge_rows:
        display = CATEGORY_DISPLAY.get(r["category"], r["category"])
        status = "✓ PASS" if r["net_edge"] >= threshold else "✗ below"
        lines.append(
            f"| {display} | {r['nominee']} | {r['direction']} | "
            f"{r['raw_edge']:+.1%} | {r['half_spread']:.2%} | "
            f"${r['exec_price']:.3f} | ${r['fee']:.2f} | "
            f"**{r['net_edge']:+.1%}** | {status} |"
        )

    lines.append("")
    return "\n".join(lines)


def _positions_section(
    pos: pd.DataFrame,
    agg: pd.DataFrame,
    config_name: str,
    entry_snapshot: str,
    scale_factor: float,
    bankroll: float,
    allocation_weights: dict[str, float] | None = None,
) -> str:
    """Positions table — one row per position, with allocation weight."""
    cfg = RECOMMENDED_CONFIGS[config_name]
    model = OPTION_MODELS[config_name]
    label = cfg.label

    lines = ["## Positions\n"]

    pos_slice = pos[
        (pos["config_label"] == label)
        & (pos["model_type"] == model)
        & (pos["entry_snapshot"] == entry_snapshot)
    ]

    lines.append(
        "| Category | Direction | Nominee | Contracts | Capital ($) | Weight | Allocated ($) | % Bankroll |"
    )
    lines.append(
        "|----------|-----------|---------|-----------|-------------|--------|---------------|------------|"
    )

    total_capital = 0.0
    categories = sorted(CATEGORY_DISPLAY.keys())

    for cat in categories:
        display = CATEGORY_DISPLAY[cat]
        url = KALSHI_URLS.get(cat, "")
        linked_display = f"[{display}]({url})" if url else display
        cat_pos = pos_slice[pos_slice["category"] == cat]

        if len(cat_pos) == 0:
            lines.append(
                f"| {linked_display} | — | *No trades (edge < threshold)* | — | — | — | — | — |"
            )
            continue

        cat_weight = allocation_weights.get(cat, 1.0) if allocation_weights else 1.0
        for i, (_, p) in enumerate(cat_pos.iterrows()):
            direction = "YES" if p["direction"] == "yes" else "NO"
            name = p["outcome"]
            contracts = int(round(p["contracts"] * scale_factor))
            capital = p["outlay_dollars"] * scale_factor
            allocated = capital * cat_weight
            pct_bankroll = allocated / bankroll * 100 if bankroll > 0 else 0
            total_capital += allocated

            # First row for this category shows the category name
            cat_label = linked_display if i == 0 else ""
            lines.append(
                f"| {cat_label} | {direction} | {name} | "
                f"{contracts:,} | ${capital:,.2f} | {cat_weight:.2f} | ${allocated:,.2f} | {pct_bankroll:.1f}% |"
            )

    lines.append("")
    pct_total = total_capital / bankroll * 100 if bankroll > 0 else 0
    lines.append(
        f"**Total allocated capital: ${total_capital:,.2f} ({pct_total:.1f}% of ${bankroll:,.0f} bankroll)**\n"
    )

    return "\n".join(lines)


def _allocation_section(
    alloc: pd.DataFrame,
    mvm: pd.DataFrame,
    entry_snapshot: str,
    bankroll: float,
    scale_factor: float,
    model_type: str = "avg_ensemble",
) -> str:
    """Category allocation table showing maxedge_100 weights and bankroll.

    The maxedge_100 strategy allocates proportionally to the maximum
    model-market edge per category, concentrating capital where the model
    sees the best buy-side opportunity.

    See: d20260305_portfolio_kelly for validation of this strategy.
    """
    n_categories = len(CATEGORY_DISPLAY)
    uniform_per_cat = bankroll / n_categories

    # Compute max net edge per category using taker fees
    # Use edge_20_taker for the allocation section since allocation weights
    # are computed from raw edges (model - market), independent of fee type.
    alloc_config = (
        "edge_20_taker"
        if "edge_20_taker" in RECOMMENDED_CONFIGS
        else list(RECOMMENDED_CONFIGS.keys())[0]
    )
    edge_rows = _compute_edge_rows(mvm, entry_snapshot, alloc_config, model_type, min_raw_edge=0.0)
    max_net_by_cat: dict[str, float] = {}
    for r in edge_rows:
        cat = r["category"]
        if cat not in max_net_by_cat or r["net_edge"] > max_net_by_cat[cat]:
            max_net_by_cat[cat] = r["net_edge"]

    lines = ["## Category Allocation — maxedge_100\n"]
    lines.append(
        "*Bankroll allocated proportionally to max(model_prob − market_prob) per category. "
        "Weight 1.0 = uniform allocation. Max Net Edge = after spread + taker fee. See "
        "[d20260305_portfolio_kelly](../d20260305_portfolio_kelly/) for validation.*\n"
    )

    lines.append("| Category | Max Raw Edge | Max Net Edge | Active? | Weight | Bankroll |")
    lines.append("|----------|-------------|-------------|---------|--------|----------|")

    total_allocated = 0.0
    for _, row in alloc.iterrows():
        display = row["display_name"]
        url = KALSHI_URLS.get(row["category"], "")
        linked = f"[{display}]({url})" if url else display
        active_str = "✓" if row["is_active"] else "—"
        allocated = uniform_per_cat * row["weight"]
        total_allocated += allocated
        net_edge = max_net_by_cat.get(row["category"], 0.0)
        lines.append(
            f"| {linked} | {row['max_edge']:+.1%} | {net_edge:+.1%} | {active_str} | "
            f"{row['weight']:.2f} | ${allocated:,.0f} |"
        )

    lines.append(f"| **Total** | | | | | **${total_allocated:,.0f}** |")
    lines.append("")
    lines.append(
        f"> Total bankroll: ${bankroll:,.0f}. Uniform would be ${uniform_per_cat:,.0f}/category.\n"
    )

    return "\n".join(lines)


def _scenario_pnl_section(
    scenario_pnl: pd.DataFrame,
    pos: pd.DataFrame,
    config_name: str,
    entry_snapshot: str,
    scale_factor: float,
    bankroll: float,
    allocation_weights: dict[str, float] | None = None,
) -> str:
    """Detailed scenario P&L for each category — what if nominee X wins?"""
    cfg = RECOMMENDED_CONFIGS[config_name]
    model = OPTION_MODELS[config_name]
    label = cfg.label

    lines = ["## Scenario P&L — What If Winner Is…\n"]

    pnl_slice = scenario_pnl[
        (scenario_pnl["config_label"] == label)
        & (scenario_pnl["model_type"] == model)
        & (scenario_pnl["entry_snapshot"] == entry_snapshot)
    ]

    # Get per-category capital deployed for ROIC
    pos_slice = pos[
        (pos["config_label"] == label)
        & (pos["model_type"] == model)
        & (pos["entry_snapshot"] == entry_snapshot)
    ]

    categories = sorted(pnl_slice["category"].unique())

    for cat in categories:
        display = CATEGORY_DISPLAY.get(cat, cat)
        url = KALSHI_URLS.get(cat, "")
        heading = f"[{display}]({url})" if url else display
        cat_pnl = pnl_slice[pnl_slice["category"] == cat].copy()

        if len(cat_pnl) == 0 or cat_pnl["total_trades"].max() == 0:
            lines.append(f"### {heading}\n")
            lines.append("*No trades — model agrees with market (edge < threshold).*\n")
            continue

        # Capital deployed in this category
        cat_pos = pos_slice[pos_slice["category"] == cat]
        cat_weight = allocation_weights.get(cat, 1.0) if allocation_weights else 1.0
        cat_capital = (
            cat_pos["outlay_dollars"].sum() * scale_factor * cat_weight if len(cat_pos) > 0 else 0.0
        )

        lines.append(f"### {heading}\n")
        lines.append("| If Winner Is… | P&L | Return % | ROIC % |")
        lines.append("|---------------|-----|----------|--------|")

        cat_pnl = cat_pnl.sort_values("total_pnl", ascending=False)
        for _, row in cat_pnl.iterrows():
            winner = row["assumed_winner"]
            if winner == "none":
                continue
            pnl = row["total_pnl"] * scale_factor * cat_weight
            ret = row["return_pct"] * scale_factor * cat_weight
            roic = (pnl / cat_capital * 100) if cat_capital > 0 else 0.0
            lines.append(f"| {winner} | ${pnl:+,.0f} | {ret:+.1f}% | {roic:+.1f}% |")

        lines.append("")
        lines.append(
            f"> Capital deployed in {display}: ${cat_capital:,.2f}. "
            f"ROIC % = P&L / capital deployed × 100.\n"
        )

    return "\n".join(lines)


def _mc_section(
    scenario_pnl: pd.DataFrame,
    model_vs_market: pd.DataFrame,
    position_summary: pd.DataFrame,
    config_name: str,
    entry_snapshot: str,
    bankroll: float,
    scale_factor: float,
    allocation_weights: dict[str, float] | None = None,
) -> str:
    """Monte Carlo portfolio simulation results — enhanced metrics."""
    cfg = RECOMMENDED_CONFIGS[config_name]
    model = OPTION_MODELS[config_name]

    lines = ["## Monte Carlo Portfolio Simulation\n"]
    lines.append(
        "*10,000 samples. Winners drawn per category from avg_ensemble model probabilities.*\n"
    )

    mc = monte_carlo_portfolio(
        scenario_pnl,
        model_vs_market,
        position_summary,
        config_label=cfg.label,
        model_type=model,
        entry_snapshot=entry_snapshot,
        scale_factor=scale_factor,
        bankroll=bankroll,
        allocation_weights=allocation_weights,
    )

    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| E[P&L] | ${mc['mean_pnl']:+,.0f} |")
    lines.append(f"| Median P&L | ${mc['median_pnl']:+,.0f} |")
    lines.append(f"| Std Dev | ${mc['std_pnl']:,.0f} |")
    lines.append(f"| CVaR-5% | ${mc['cvar_05']:+,.0f} |")
    lines.append(f"| CVaR-10% | ${mc['cvar_10']:+,.0f} |")
    lines.append(f"| $ Deployed | ${mc['total_capital_deployed']:,.0f} |")
    pct_deployed = mc["pct_bankroll_deployed"] * 100
    lines.append(f"| % Bankroll Deployed | {pct_deployed:.1f}% |")
    roic = mc["expected_roic"] * 100
    lines.append(f"| E[ROIC] | {roic:+.1f}% |")
    lines.append(f"| P(profit) | {mc['prob_profit']:.0%} |")
    lines.append(f"| P(loss>10% bankroll) | {mc['prob_loss_10pct_bankroll']:.0%} |")
    lines.append(f"| P(loss>20% bankroll) | {mc['prob_loss_20pct_bankroll']:.0%} |")
    lines.append(f"| P(loss>10% capital) | {mc['prob_loss_10pct_capital']:.0%} |")
    lines.append(f"| P(loss>20% capital) | {mc['prob_loss_20pct_capital']:.0%} |")
    lines.append("")

    lines.append(
        "> `E[P&L]`: Expected profit/loss from MC simulation. "
        "`CVaR-5%`: Average P&L in the worst 5% of simulations.\n"
        "> `$ Deployed`: Total capital used to purchase contracts. "
        "`% Bankroll`: Capital / total bankroll.\n"
        "> `E[ROIC]`: Expected return on invested capital (`E[P&L]` / `$ Deployed`).\n"
        "> `P(profit)`: Probability of positive total P&L. "
        "`P(loss>X% bankroll)`: Probability of losing more than X% of bankroll.\n"
        "> `P(loss>X% capital)`: Probability of losing more than X% of capital deployed.\n"
    )

    return "\n".join(lines)


def _model_comparison_section(
    agg: pd.DataFrame,
    config_name: str,
    entry_snapshot: str,
    scale_factor: float,
) -> str:
    """Compare all 6 models at the latest entry point using this config's parameters."""
    cfg = RECOMMENDED_CONFIGS[config_name]
    kc = cfg.trading.kelly
    config_label = cfg.label

    lines = [f"## Model Comparison — {_short_snapshot(entry_snapshot)}\n"]
    lines.append(
        f"*All models shown with the {config_name} config "
        f"(kf={kc.kelly_fraction}, e={kc.buy_edge_threshold}).*\n"
    )

    models = sorted(agg["model_type"].unique())

    header = "| Category |"
    sep = "|----------|"
    for m in models:
        header += f" {m} |"
        sep += "--------|"
    lines.append(header)
    lines.append(sep)

    # Track totals per model
    model_totals: dict[str, float] = dict.fromkeys(models, 0.0)
    model_has_data: dict[str, bool] = dict.fromkeys(models, False)

    for cat in sorted(CATEGORY_DISPLAY.keys()):
        display = CATEGORY_DISPLAY[cat]
        url = KALSHI_URLS.get(cat, "")
        linked = f"[{display}]({url})" if url else display
        row = f"| {linked} |"
        for m in models:
            cat_agg = agg[
                (agg["category"] == cat)
                & (agg["model_type"] == m)
                & (agg["entry_snapshot"] == entry_snapshot)
                & (agg["config_label"] == config_label)
            ]
            if len(cat_agg) > 0 and cat_agg.iloc[0]["total_trades"] > 0:
                mean_pnl = cat_agg.iloc[0]["mean_pnl"] * scale_factor
                model_totals[m] += mean_pnl
                model_has_data[m] = True
                row += f" ${mean_pnl:+,.0f} |"
            else:
                row += " — |"
        lines.append(row)

    # TOTAL row
    total_row = "| **TOTAL** |"
    for m in models:
        if model_has_data[m]:
            total_row += f" **${model_totals[m]:+,.0f}** |"
        else:
            total_row += " — |"
    lines.append(total_row)
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# Live portfolio sections
# ============================================================================


def _extract_market_prices(
    model_vs_market: pd.DataFrame,
    entry_snapshot: str,
    model_type: str = "avg_ensemble",
) -> dict[tuple[str, str], float]:
    """Extract YES mid-prices from model_vs_market data.

    Returns {(category_slug, nominee): yes_price_dollars}.
    Uses the first model_type that has data for each (category, nominee) pair.
    """
    mvm = model_vs_market[
        (model_vs_market["snapshot_key"] == entry_snapshot)
        & (model_vs_market["model_type"] == model_type)
    ]
    prices: dict[tuple[str, str], float] = {}
    for _, row in mvm.iterrows():
        prices[(row["category"], row["nominee"])] = float(row["market_prob"])
    return prices


def _account_summary_section(portfolio: LivePortfolio) -> str:
    """Account-level summary: wallet balance, positions MTM, total equity."""
    lines = ["## Account Summary (Kalshi)\n"]
    lines.append("| Item | Value |")
    lines.append("|------|-------|")
    lines.append(f"| Wallet Balance | ${portfolio.wallet_balance:,.2f} |")
    lines.append(f"| Positions Value (MTM) | ${portfolio.portfolio_value:,.2f} |")
    lines.append(f"| **Total Equity** | **${portfolio.total_equity:,.2f}** |")
    lines.append(f"| Total Fees Paid | ${portfolio.total_fees_paid:,.2f} |")
    lines.append(f"| Realized P&L (closed) | ${portfolio.total_realized_pnl:+,.2f} |")
    lines.append("")
    return "\n".join(lines)


def _current_positions_section(
    portfolio: LivePortfolio,
    market_prices: dict[tuple[str, str], float],
) -> str:
    """Table of actual Kalshi holdings with unrealized P&L."""
    lines = ["## Current Kalshi Positions\n"]

    if not portfolio.positions:
        lines.append("*No open positions.*\n")
        return "\n".join(lines)

    lines.append(
        "| Category | Nominee | Dir | Contracts | Avg Cost | Mkt Price | Value | Unrealized |"
    )
    lines.append(
        "|----------|---------|-----|-----------|----------|-----------|-------|------------|"
    )

    total_value = 0.0
    total_unrealized = 0.0

    # Sort by category then nominee
    sorted_positions = sorted(portfolio.positions, key=lambda p: (p.category_slug, p.nominee))

    for pos in sorted_positions:
        display = CATEGORY_DISPLAY.get(pos.category_slug, pos.category_slug)
        url = KALSHI_URLS.get(pos.category_slug, "")
        linked = f"[{display}]({url})" if url else display
        dir_str = "YES" if pos.direction.value == "yes" else "NO"

        yes_price = market_prices.get((pos.category_slug, pos.nominee), 0.5)
        if pos.direction.value == "yes":
            current_price = yes_price
        else:
            current_price = 1.0 - yes_price

        value = pos.contracts * current_price
        cost_basis = pos.contracts * pos.avg_cost
        unrealized = value - cost_basis

        total_value += value
        total_unrealized += unrealized

        lines.append(
            f"| {linked} | {pos.nominee} | {dir_str} | {pos.contracts} "
            f"| ${pos.avg_cost:.4f} | ${current_price:.4f} "
            f"| ${value:,.2f} | ${unrealized:+,.2f} |"
        )

    lines.append("")
    lines.append(
        f"**Total position value:** ${total_value:,.2f} | "
        f"**Total unrealized P&L:** ${total_unrealized:+,.2f}\n"
    )

    return "\n".join(lines)


def _position_adjustments_section(
    portfolio: LivePortfolio,
    adjustments: list[PositionAdjustment],
    buy_fee_type: FeeType = FeeType.TAKER,
) -> str:
    """Trade instructions to move from current holdings to model-recommended targets.

    Each row is a single tradeable action. Direction reversals produce two rows
    (sell then buy). HOLD rows are included for completeness.
    """
    lines = ["## Position Adjustments — Current → Target\n"]
    lines.append(
        f"**Target bankroll:** ${portfolio.total_equity:,.2f} "
        f"(wallet: ${portfolio.wallet_balance:,.2f} + positions: "
        f"${portfolio.portfolio_value:,.2f})\n"
    )

    if not adjustments:
        lines.append("*No adjustments needed — current portfolio matches target.*\n")
        return "\n".join(lines)

    lines.append(
        "| Category | Nominee | Current | Target | Action "
        "| Qty | Est. Price | Est. Cost | Est. Fee |"
    )
    lines.append(
        "|----------|---------|---------|--------|--------"
        "|-----|------------|-----------|----------|"
    )

    total_buy_cost = 0.0
    total_sell_proceeds = 0.0
    total_fees = 0.0

    for adj in adjustments:
        display = CATEGORY_DISPLAY.get(adj.category_slug, adj.category_slug)
        url = adj.kalshi_url or KALSHI_URLS.get(adj.category_slug, "")
        linked = f"[{display}]({url})" if url else display

        if adj.action == "HOLD":
            lines.append(
                f"| {linked} | {adj.nominee} | {adj.current_desc} "
                f"| {adj.target_desc} | HOLD | — | — | — | — |"
            )
            continue

        is_sell = adj.action.startswith("SELL")
        if is_sell:
            total_sell_proceeds += adj.est_cost
        else:
            total_buy_cost += adj.est_cost
        total_fees += adj.est_fee

        lines.append(
            f"| {linked} | {adj.nominee} | {adj.current_desc} | {adj.target_desc} "
            f"| **{adj.action}** | {adj.delta_contracts} "
            f"| ${adj.est_price:.2f} | ${adj.est_cost:,.2f} | ${adj.est_fee:.2f} |"
        )

    # Summary
    net_cash = total_sell_proceeds - total_buy_cost - total_fees
    cash_after = portfolio.wallet_balance + net_cash

    lines.append("")
    lines.append("| Summary | |")
    lines.append("|---------|------|")
    lines.append(f"| Total buy cost | ${total_buy_cost:,.2f} |")
    lines.append(f"| Total sell proceeds | ${total_sell_proceeds:,.2f} |")
    lines.append(f"| Total est. fees | ${total_fees:,.2f} |")
    lines.append(f"| **Net cash impact** | **${net_cash:+,.2f}** |")
    lines.append(f"| Wallet balance now | ${portfolio.wallet_balance:,.2f} |")
    lines.append(f"| **Wallet after adjustments** | **${cash_after:,.2f}** |")
    lines.append("")

    lines.append(
        f"> *Buy fees estimated using {buy_fee_type.value} rates. "
        f"Sell fees always use taker rates.*\n"
    )

    if cash_after < 0:
        lines.append(
            "> ⚠️ **Insufficient cash.** Net outflow exceeds wallet balance. "
            "Sell positions first, or reduce target sizing.\n"
        )

    return "\n".join(lines)


# ============================================================================
# Main report assembly
# ============================================================================


def generate_config_report(
    config_name: str,
    bankroll: float,
    results_dir: Path,
    output_dir: Path,
    now: datetime,
    scenario_pnl: pd.DataFrame,
    model_vs_market: pd.DataFrame,
    position_summary: pd.DataFrame,
    scenario_pnl_agg: pd.DataFrame,
    snapshot_filter: str | None = None,
    live_portfolio: LivePortfolio | None = None,
) -> str:
    """Generate a Markdown report for a single config.

    Args:
        config_name: Config name (e.g. "edge_20").
        bankroll: Total bankroll in dollars.
        results_dir: Directory containing result CSVs.
        output_dir: Directory to write report files.
        now: Datetime when report generation started.
        scenario_pnl: Full scenario P&L DataFrame.
        model_vs_market: Model-vs-market DataFrame.
        position_summary: Position summary DataFrame.
        scenario_pnl_agg: Aggregated scenario P&L DataFrame.
        snapshot_filter: Specific entry_snapshot to use. If None, uses the latest.
        live_portfolio: If provided, adds account summary, current positions,
            and position adjustment sections to the report.

    Returns:
        Path to the generated report file.
    """
    # Determine which entry snapshot to report on
    if snapshot_filter:
        latest_snapshot = snapshot_filter
    else:
        available_in_data = sorted(scenario_pnl["entry_snapshot"].unique())
        latest_snapshot = available_in_data[-1]

    # Scale factor: user bankroll / total backtest bankroll
    n_categories = len(CATEGORY_DISPLAY)
    scale_factor = bankroll / (BACKTEST_BANKROLL_PER_CATEGORY * n_categories)

    # Get entry_timestamp from data
    ts_rows = scenario_pnl[scenario_pnl["entry_snapshot"] == latest_snapshot]
    entry_timestamp = (
        ts_rows["entry_timestamp"].iloc[0]
        if "entry_timestamp" in ts_rows.columns and len(ts_rows) > 0
        else latest_snapshot
    )

    now_str = now.strftime("%Y-%m-%d %H:%M ET")

    # Compute allocation weights
    cfg = RECOMMENDED_CONFIGS[config_name]
    model = OPTION_MODELS[config_name]
    alloc = compute_allocation_weights(
        model_vs_market,
        position_summary,
        model_type=model,
        config_label=cfg.label,
        entry_snapshot=latest_snapshot,
    )
    alloc_weight_map = dict(zip(alloc["category"], alloc["weight"], strict=True))

    # Assembly
    sections = [
        _header(config_name, bankroll, entry_timestamp, now_str),
    ]

    # Live portfolio sections (actionable — what to do NOW)
    if live_portfolio is not None:
        market_prices = _extract_market_prices(model_vs_market, latest_snapshot, model)
        # Compute execution prices based on config
        if cfg.trading.limit_price_offset > 0:
            # Maker: bid + offset. With mid-price approximation, bid ≈ mid - spread.
            # We don't have per-nominee spread here, so use mid as approximation.
            # The actual bid+offset is handled by the signal pipeline during backtest.
            execution_prices = market_prices  # approximate: targets come from backtest scaling
        else:
            execution_prices = market_prices  # taker: ask ≈ mid (for target position scaling)
        targets = compute_target_positions(
            position_summary,
            config_label=cfg.label,
            model_type=model,
            entry_snapshot=latest_snapshot,
            bankroll=live_portfolio.total_equity,
            backtest_bankroll_per_category=BACKTEST_BANKROLL_PER_CATEGORY,
            n_categories=n_categories,
            allocation_weights=alloc_weight_map,
            execution_prices=execution_prices,
            ceremony_year=CEREMONY_YEAR,
        )
        adjustments = compute_adjustments(
            live_portfolio,
            targets,
            market_prices,
            buy_fee_type=cfg.trading.fee_type,
        )
        sections.extend(
            [
                _account_summary_section(live_portfolio),
                _current_positions_section(live_portfolio, market_prices),
                _position_adjustments_section(
                    live_portfolio, adjustments, buy_fee_type=cfg.trading.fee_type
                ),
            ]
        )

    # Model positions from backtest
    sections.append(
        _positions_section(
            position_summary,
            scenario_pnl_agg,
            config_name,
            latest_snapshot,
            scale_factor,
            bankroll,
            allocation_weights=alloc_weight_map,
        ),
    )

    # Edge scanner
    sections.append(
        _edge_detail_section(
            model_vs_market,
            latest_snapshot,
            config_name,
        ),
    )

    # Category allocation
    sections.append(
        _allocation_section(
            alloc,
            model_vs_market,
            latest_snapshot,
            bankroll,
            scale_factor,
        ),
    )

    # Model comparison (all models)
    sections.append(
        _model_vs_market_section(model_vs_market, latest_snapshot),
    )

    # Scenario analysis
    sections.extend(
        [
            _scenario_pnl_section(
                scenario_pnl,
                position_summary,
                config_name,
                latest_snapshot,
                scale_factor,
                bankroll,
                allocation_weights=alloc_weight_map,
            ),
            _mc_section(
                scenario_pnl,
                model_vs_market,
                position_summary,
                config_name,
                latest_snapshot,
                bankroll,
                scale_factor,
                allocation_weights=alloc_weight_map,
            ),
        ]
    )

    report = "\n".join(sections)

    # Write report
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = now.strftime("%Y-%m-%dT%H:%M")
    report_path = output_dir / f"{timestamp_str}_{config_name}.md"
    report_path.write_text(report)
    print(f"Report written to {report_path}")

    return str(report_path)


def generate_reports(
    bankroll: float,
    results_dir: Path,
    output_dir: Path,
    config_names: list[str] | None = None,
    snapshot_filter: str | None = None,
    live_portfolio: LivePortfolio | None = None,
) -> list[str]:
    """Generate Markdown reports for one or more configs.

    Args:
        bankroll: Total bankroll in dollars.
        results_dir: Directory containing result CSVs.
        output_dir: Directory to write report files.
        config_names: List of config names to generate. If None, generates for all recommended configs.
        snapshot_filter: Specific entry_snapshot to use. If None, uses the latest.
        live_portfolio: If provided, adds live portfolio sections to each report.

    Returns:
        List of paths to generated report files.
    """
    if config_names is None:
        config_names = list(RECOMMENDED_CONFIGS.keys())

    scenario_pnl, model_vs_market, position_summary, scenario_pnl_agg = load_results(results_dir)
    now = datetime.now(ZoneInfo("US/Eastern"))

    paths = []
    for name in config_names:
        path = generate_config_report(
            config_name=name,
            bankroll=bankroll,
            results_dir=results_dir,
            output_dir=output_dir,
            now=now,
            scenario_pnl=scenario_pnl,
            model_vs_market=model_vs_market,
            position_summary=position_summary,
            scenario_pnl_agg=scenario_pnl_agg,
            snapshot_filter=snapshot_filter,
            live_portfolio=live_portfolio,
        )
        paths.append(path)

    # Also write a CSV summary for downstream use
    if snapshot_filter:
        latest_snapshot = snapshot_filter
    else:
        latest_snapshot = sorted(scenario_pnl["entry_snapshot"].unique())[-1]
    n_categories = len(CATEGORY_DISPLAY)
    scale_factor = bankroll / (BACKTEST_BANKROLL_PER_CATEGORY * n_categories)
    # Compute allocation for CSV summary
    cfg_0 = RECOMMENDED_CONFIGS[config_names[0]]
    model_0 = OPTION_MODELS[config_names[0]]
    alloc = compute_allocation_weights(
        model_vs_market,
        position_summary,
        model_type=model_0,
        config_label=cfg_0.label,
        entry_snapshot=latest_snapshot,
    )
    alloc_weight_map = dict(zip(alloc["category"], alloc["weight"], strict=True))

    _write_csv_summary(
        scenario_pnl_agg,
        latest_snapshot,
        scale_factor,
        results_dir,
        config_names,
        allocation_weights=alloc_weight_map,
    )

    return paths


def _write_csv_summary(
    agg: pd.DataFrame,
    entry_snapshot: str,
    scale_factor: float,
    output_dir: Path,
    config_names: list[str],
    allocation_weights: dict[str, float] | None = None,
) -> None:
    """Write a concise CSV summary for the latest entry point."""
    rows = []
    for config_name in config_names:
        cfg = RECOMMENDED_CONFIGS[config_name]
        model = OPTION_MODELS[config_name]
        label = cfg.label

        config_agg = agg[
            (agg["config_label"] == label)
            & (agg["model_type"] == model)
            & (agg["entry_snapshot"] == entry_snapshot)
        ]

        for _, row in config_agg.iterrows():
            cat = row["category"]
            weight = allocation_weights.get(cat, 1.0) if allocation_weights else 1.0
            rows.append(
                {
                    "config": config_name,
                    "category": cat,
                    "model": model,
                    "trades": int(row["total_trades"]),
                    "capital_deployed": round(row["capital_deployed"] * scale_factor, 2),
                    "mean_pnl": round(row["mean_pnl"] * scale_factor, 2),
                    "min_pnl": round(row["min_pnl"] * scale_factor, 2),
                    "max_pnl": round(row["max_pnl"] * scale_factor, 2),
                    "allocation_weight": round(weight, 4),
                    "allocated_capital": round(row["capital_deployed"] * scale_factor * weight, 2),
                    "allocated_mean_pnl": round(row["mean_pnl"] * scale_factor * weight, 2),
                }
            )

    summary_df = pd.DataFrame(rows)
    summary_path = output_dir / "report_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary CSV written to {summary_path}")


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 2026 Oscar trading report")
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Total bankroll in dollars (default: $1000).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        choices=list(RECOMMENDED_CONFIGS.keys()),
        help="Generate report for a specific config (default: all).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help=f"Results directory (default: {EXP_DIR}/results/).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for reports (default: one-off dir reports/).",
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default=None,
        help="Specific entry_snapshot to use (default: latest available).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use the latest live entry_snapshot (auto-selects latest 'live_*' entry).",
    )
    parser.add_argument(
        "--live-portfolio",
        action="store_true",
        help="Fetch current Kalshi portfolio and show position adjustments in report.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else Path(EXP_DIR) / "results"
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR

    config_names = [args.config] if args.config else None

    # Resolve snapshot filter
    snapshot_filter: str | None = None
    if args.live:
        pnl = pd.read_csv(results_dir / "scenario_pnl.csv")
        live_entries = sorted(s for s in pnl["entry_snapshot"].unique() if s.startswith("live_"))
        if live_entries:
            snapshot_filter = live_entries[-1]
            print(f"Using live entry: {snapshot_filter}")
        else:
            print("WARNING: No live entries found in data. Using latest available.")
    elif args.snapshot:
        snapshot_filter = args.snapshot

    # Fetch live portfolio if requested
    live_portfolio: LivePortfolio | None = None
    if args.live_portfolio:
        print("Fetching live Kalshi portfolio...")
        live_portfolio = fetch_live_portfolio(CEREMONY_YEAR)
        print(
            f"  Total equity: ${live_portfolio.total_equity:,.2f} "
            f"(wallet: ${live_portfolio.wallet_balance:,.2f}, "
            f"positions: ${live_portfolio.portfolio_value:,.2f})"
        )
        print(f"  Open positions: {len(live_portfolio.positions)}")

    paths = generate_reports(
        bankroll=args.bankroll,
        results_dir=results_dir,
        output_dir=output_dir,
        config_names=config_names,
        snapshot_filter=snapshot_filter,
        live_portfolio=live_portfolio,
    )
    print(f"\nDone! {len(paths)} report(s) generated.")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
