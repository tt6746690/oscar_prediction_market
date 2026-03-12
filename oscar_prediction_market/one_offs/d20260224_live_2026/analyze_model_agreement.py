"""Analyze where the 4 base model types agree and disagree on predictions.

Identifies risky positions where model uncertainty is highest by comparing
clogit, lr, gbt, and cal_sgbt predictions for each nominee.  Cross-references
with active positions to flag trades sitting on high-disagreement nominees.

Usage::

    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\\
d20260224_live_2026.analyze_model_agreement

    # Custom bankroll / config
    uv run python -m oscar_prediction_market.one_offs.\\
d20260224_live_2026.analyze_model_agreement --bankroll 5000 --config recommended
"""

import argparse
from pathlib import Path

import pandas as pd

from oscar_prediction_market.one_offs.d20260224_live_2026 import (
    AVAILABLE_SNAPSHOT_KEY_STRS,
    EXP_DIR,
)
from oscar_prediction_market.one_offs.d20260224_live_2026.generate_report import (
    BACKTEST_BANKROLL_PER_CATEGORY,
    CATEGORY_DISPLAY,
)
from oscar_prediction_market.one_offs.d20260224_live_2026.recommended_configs import (
    OPTION_MODELS,
    RECOMMENDED_CONFIGS,
)

# ============================================================================
# Constants
# ============================================================================

#: The 4 base model types to compare.
BASE_MODELS = ["clogit", "lr", "gbt", "cal_sgbt"]

#: Thresholds for disagreement classification (percentage points).
HIGH_DISAGREE_THRESHOLD = 0.15  # 15pp spread
MEDIUM_DISAGREE_THRESHOLD = 0.10  # 10pp spread
HIGH_RISK_THRESHOLD = 0.20  # 20pp spread

#: Number of modeled categories used in the backtest.
NUM_CATEGORIES = len(CATEGORY_DISPLAY)

SCRIPT_DIR = Path(__file__).resolve().parent


# ============================================================================
# Data loading
# ============================================================================


def load_data(
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load model_vs_market.csv and position_summary.csv."""
    mvm = pd.read_csv(results_dir / "model_vs_market.csv")
    pos = pd.read_csv(results_dir / "position_summary.csv")
    return mvm, pos


# ============================================================================
# Analysis
# ============================================================================


def compute_nominee_spreads(
    mvm: pd.DataFrame,
    snapshot_key: str,
) -> pd.DataFrame:
    """Compute per-nominee model spread across the 4 base models.

    Returns a DataFrame with columns:
        category, nominee, market_prob, clogit, lr, gbt, cal_sgbt,
        spread, std, direction_agree
    """
    # Filter to latest snapshot and base models only
    mask = (mvm["snapshot_key"] == snapshot_key) & (mvm["model_type"].isin(BASE_MODELS))
    df = mvm.loc[mask, ["category", "nominee", "model_type", "model_prob", "market_prob"]].copy()

    if df.empty:
        print(f"WARNING: No data for snapshot_key={snapshot_key} with base models.")
        return pd.DataFrame()

    # Pivot: one row per (category, nominee), columns = model types
    pivoted = df.pivot_table(
        index=["category", "nominee"],
        columns="model_type",
        values="model_prob",
        aggfunc="first",
    ).reset_index()

    # Get market_prob (same for all models for a given nominee)
    market_probs = df.groupby(["category", "nominee"])["market_prob"].first().reset_index()
    pivoted = pivoted.merge(market_probs, on=["category", "nominee"], how="left")

    # Compute spread stats across base models
    model_cols = [c for c in BASE_MODELS if c in pivoted.columns]
    model_values = pivoted[model_cols]

    pivoted["spread"] = model_values.max(axis=1) - model_values.min(axis=1)
    pivoted["std"] = model_values.std(axis=1)

    # Direction agreement: do all models agree on direction vs market?
    # All above market OR all below market
    def _direction_agree(row: pd.Series) -> bool:
        mp = row["market_prob"]
        above = sum(1 for c in model_cols if row[c] > mp)
        below = sum(1 for c in model_cols if row[c] < mp)
        equal = sum(1 for c in model_cols if row[c] == mp)
        n = len(model_cols)
        return (above + equal == n) or (below + equal == n)

    pivoted["direction_agree"] = pivoted.apply(_direction_agree, axis=1)
    pivoted["high_disagreement"] = pivoted["spread"] > HIGH_DISAGREE_THRESHOLD

    return pivoted


def compute_category_risk(nominee_spreads: pd.DataFrame) -> pd.DataFrame:
    """Compute per-category risk assessment.

    Returns a DataFrame with columns:
        category, max_spread, n_high_disagree, risk_level, outlier_notes
    """
    model_cols = [c for c in BASE_MODELS if c in nominee_spreads.columns]
    rows = []

    for cat, group in nominee_spreads.groupby("category"):
        max_spread = group["spread"].max()
        n_high = int(group["high_disagreement"].sum())

        # Risk level
        if max_spread > HIGH_RISK_THRESHOLD:
            risk_level = "HIGH"
        elif max_spread > MEDIUM_DISAGREE_THRESHOLD:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Find outlier models per nominee (>15pp from mean of others)
        outlier_notes_list = []
        for _, row in group.iterrows():
            probs = {m: row[m] for m in model_cols if pd.notna(row[m])}
            if len(probs) < 2:
                continue
            for model, prob in probs.items():
                others = [v for k, v in probs.items() if k != model]
                mean_others = sum(others) / len(others) if others else prob
                if abs(prob - mean_others) > HIGH_DISAGREE_THRESHOLD:
                    direction = "high" if prob > mean_others else "low"
                    outlier_notes_list.append(f"{model} {direction} on {row['nominee']}")

        notes = "; ".join(outlier_notes_list[:3])  # cap at 3 notes
        if len(outlier_notes_list) > 3:
            notes += f" (+{len(outlier_notes_list) - 3} more)"

        rows.append(
            {
                "category": cat,
                "max_spread": max_spread,
                "n_high_disagree": n_high,
                "risk_level": risk_level,
                "outlier_notes": notes,
            }
        )

    return pd.DataFrame(rows)


def compute_position_risk(
    nominee_spreads: pd.DataFrame,
    position_summary: pd.DataFrame,
    config_label: str,
    model_type: str,
    snapshot_key: str,
    scale_factor: float,
) -> pd.DataFrame:
    """Cross-reference active positions with model disagreement.

    Returns a DataFrame with columns:
        category, nominee, direction, contracts, capital,
        clogit, lr, gbt, cal_sgbt, spread, risk
    """
    model_cols = [c for c in BASE_MODELS if c in nominee_spreads.columns]

    # Filter positions to the target config/model/snapshot
    pos_mask = (
        (position_summary["model_type"] == model_type)
        & (position_summary["entry_snapshot"] == snapshot_key)
        & (position_summary["config_label"] == config_label)
    )
    positions = position_summary.loc[pos_mask].copy()

    if positions.empty:
        return pd.DataFrame()

    # Merge with nominee spreads
    merged = positions.merge(
        nominee_spreads[
            ["category", "nominee", "market_prob", "spread", "direction_agree"] + model_cols
        ],
        left_on=["category", "outcome"],
        right_on=["category", "nominee"],
        how="left",
    )

    # Scale capital
    merged["capital"] = merged["outlay_dollars"] * scale_factor

    # Assign risk level
    def _risk_label(spread: float) -> str:
        if pd.isna(spread):
            return "UNKNOWN"
        if spread > HIGH_RISK_THRESHOLD:
            return "HIGH"
        if spread > MEDIUM_DISAGREE_THRESHOLD:
            return "MEDIUM"
        return "LOW"

    merged["risk"] = merged["spread"].apply(_risk_label)

    return merged


# ============================================================================
# Markdown generation
# ============================================================================


def _fmt_pct(val: float | None, decimals: int = 1) -> str:
    """Format a probability as a percentage string."""
    if val is None or pd.isna(val):
        return "—"
    return f"{val * 100:.{decimals}f}%"


def _fmt_pp(val: float | None, decimals: int = 1) -> str:
    """Format a spread/std as percentage points."""
    if val is None or pd.isna(val):
        return "—"
    return f"{val * 100:.{decimals}f}pp"


def generate_markdown(
    nominee_spreads: pd.DataFrame,
    category_risk: pd.DataFrame,
    position_risk: pd.DataFrame,
    snapshot_key: str,
    config_name: str,
    bankroll: float,
) -> str:
    """Generate the full Markdown report."""
    model_cols = [c for c in BASE_MODELS if c in nominee_spreads.columns]
    lines: list[str] = []

    lines.append(f"# Model Agreement Analysis — {snapshot_key}")
    lines.append("")
    lines.append(
        f"Compares the 4 base models ({', '.join(BASE_MODELS)}) to identify "
        f"where predictions diverge most."
    )
    lines.append(f"- **Snapshot:** {snapshot_key}")
    lines.append(f"- **Config:** {config_name}")
    lines.append(f"- **Bankroll:** ${bankroll:,.0f}")
    lines.append(f"- **High disagreement threshold:** {HIGH_DISAGREE_THRESHOLD * 100:.0f}pp spread")
    lines.append("")

    # Section 1: Category risk summary
    lines.append("## Section 1: Model Agreement Summary")
    lines.append("")
    lines.append("| Category | Risk Level | Max Spread | # High Disagree Nominees | Notes |")
    lines.append("|----------|-----------|------------|--------------------------|-------|")

    for _, row in category_risk.sort_values("max_spread", ascending=False).iterrows():
        cat_display = CATEGORY_DISPLAY.get(row["category"], row["category"])
        risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(row["risk_level"], "")
        lines.append(
            f"| {cat_display} | {risk_emoji} {row['risk_level']} "
            f"| {_fmt_pp(row['max_spread'])} "
            f"| {row['n_high_disagree']} "
            f"| {row['outlier_notes']} |"
        )
    lines.append("")

    # Section 2: Top 15 highest disagreement nominees
    lines.append("## Section 2: Top 15 Highest Disagreement Nominees")
    lines.append("")
    header = "| Category | Nominee | Market"
    for m in model_cols:
        header += f" | {m}"
    header += " | Spread | Direction Agreement? |"
    lines.append(header)
    sep = "|----------|---------|-------"
    for _ in model_cols:
        sep += "|------"
    sep += "|--------|---------------------|"
    lines.append(sep)

    top15 = nominee_spreads.nlargest(15, "spread")
    for _, row in top15.iterrows():
        cat_display = CATEGORY_DISPLAY.get(row["category"], row["category"])
        agree_str = "Yes" if row["direction_agree"] else "**No**"
        cols = f"| {cat_display} | {row['nominee']} | {_fmt_pct(row['market_prob'])}"
        for m in model_cols:
            cols += f" | {_fmt_pct(row[m])}"
        cols += f" | {_fmt_pp(row['spread'])} | {agree_str} |"
        lines.append(cols)
    lines.append("")

    # Section 3: Position risk analysis
    lines.append(f"## Section 3: Position Risk Analysis ({config_name} config)")
    lines.append("")

    if position_risk.empty:
        lines.append("*No active positions found for this config/snapshot.*")
        lines.append("")
    else:
        header = "| Category | Position | Dir | Contracts | Capital"
        for m in model_cols:
            header += f" | {m}"
        header += " | Spread | Risk |"
        lines.append(header)
        sep = "|----------|----------|-----|-----------|-------"
        for _ in model_cols:
            sep += "|------"
        sep += "|--------|------|"
        lines.append(sep)

        for _, row in position_risk.sort_values("spread", ascending=False).iterrows():
            cat_display = CATEGORY_DISPLAY.get(row["category"], row["category"])
            risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(row["risk"], "")
            cols = (
                f"| {cat_display} "
                f"| {row['outcome']} "
                f"| {row['direction']} "
                f"| {row['contracts']:,.0f} "
                f"| ${row['capital']:,.2f}"
            )
            for m in model_cols:
                val = row.get(m)
                cols += f" | {_fmt_pct(val)}"
            cols += f" | {_fmt_pp(row['spread'])} | {risk_emoji} {row['risk']} |"
            lines.append(cols)
        lines.append("")

        # Commentary
        high_risk_positions = position_risk[position_risk["risk"] == "HIGH"]
        med_risk_positions = position_risk[position_risk["risk"] == "MEDIUM"]
        if not high_risk_positions.empty:
            lines.append(
                f"**⚠ {len(high_risk_positions)} position(s) with HIGH model disagreement** "
                f"— models diverge by >20pp on these nominees. "
                f"Consider reducing size or hedging."
            )
            lines.append("")
        if not med_risk_positions.empty:
            lines.append(
                f"**{len(med_risk_positions)} position(s) with MEDIUM disagreement** "
                f"(10-20pp spread). Monitor closely."
            )
            lines.append("")

    # Section 4: Key observations
    lines.append("## Section 4: Key Observations")
    lines.append("")

    # Categories where all models agree
    low_risk_cats = category_risk[category_risk["risk_level"] == "LOW"]
    if not low_risk_cats.empty:
        cat_names = [CATEGORY_DISPLAY.get(c, c) for c in low_risk_cats["category"]]
        lines.append(
            f"- **Low risk (all 4 models agree):** {', '.join(cat_names)}. "
            f"Max spread <{MEDIUM_DISAGREE_THRESHOLD * 100:.0f}pp in these categories."
        )

    # Categories with highest disagreement
    high_risk_cats = category_risk[category_risk["risk_level"] == "HIGH"]
    if not high_risk_cats.empty:
        for _, row in high_risk_cats.iterrows():
            cat_display = CATEGORY_DISPLAY.get(row["category"], row["category"])
            lines.append(
                f"- **High risk — {cat_display}:** "
                f"Max spread {_fmt_pp(row['max_spread'])}, "
                f"{row['n_high_disagree']} nominee(s) with >15pp disagreement. "
                f"{row['outlier_notes']}"
            )

    med_risk_cats = category_risk[category_risk["risk_level"] == "MEDIUM"]
    if not med_risk_cats.empty:
        for _, row in med_risk_cats.iterrows():
            cat_display = CATEGORY_DISPLAY.get(row["category"], row["category"])
            lines.append(
                f"- **Medium risk — {cat_display}:** "
                f"Max spread {_fmt_pp(row['max_spread'])}. "
                f"{row['outlier_notes']}"
            )

    # Active positions on high-disagreement nominees
    if not position_risk.empty:
        risky_positions = position_risk[position_risk["risk"].isin(["HIGH", "MEDIUM"])]
        if not risky_positions.empty:
            lines.append("")
            lines.append("### Action Items — Positions on Disagreement Nominees")
            lines.append("")
            for _, row in risky_positions.sort_values("spread", ascending=False).iterrows():
                cat_display = CATEGORY_DISPLAY.get(row["category"], row["category"])
                lines.append(
                    f"- **{cat_display} / {row['outcome']}** ({row['direction']}, "
                    f"${row['capital']:,.2f}): "
                    f"spread={_fmt_pp(row['spread'])} ({row['risk']}). "
                    f"clogit={_fmt_pct(row.get('clogit'))}, "
                    f"lr={_fmt_pct(row.get('lr'))}, "
                    f"gbt={_fmt_pct(row.get('gbt'))}, "
                    f"cal_sgbt={_fmt_pct(row.get('cal_sgbt'))}"
                )
        else:
            lines.append("")
            lines.append(
                "- **All active positions sit on low-disagreement nominees.** No action items."
            )

    lines.append("")
    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze model agreement across 4 base model types."
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Total bankroll in dollars (default: 1000).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="edge_20_taker",
        choices=list(RECOMMENDED_CONFIGS),
        help="Trading config name (default: recommended).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: storage results dir).",
    )
    args = parser.parse_args()

    bankroll: float = args.bankroll
    config_name: str = args.config

    # Resolve config label and model type
    config = RECOMMENDED_CONFIGS[config_name]
    config_label = config.label
    model_type = OPTION_MODELS[config_name]

    # Scale factor: bankroll → backtest scale
    backtest_total = BACKTEST_BANKROLL_PER_CATEGORY * NUM_CATEGORIES
    scale_factor = bankroll / backtest_total

    # Latest snapshot
    snapshot_key = AVAILABLE_SNAPSHOT_KEY_STRS[-1]
    print(f"Analyzing model agreement for snapshot: {snapshot_key}")
    print(f"Config: {config_name} ({config_label}), model: {model_type}")
    print(f"Bankroll: ${bankroll:,.0f}, scale factor: {scale_factor:.4f}")
    print()

    # Load data
    results_dir = Path(EXP_DIR) / "results"
    mvm, pos = load_data(results_dir)

    # Compute analyses
    nominee_spreads = compute_nominee_spreads(mvm, snapshot_key)
    if nominee_spreads.empty:
        print("ERROR: No nominee spread data available. Exiting.")
        return

    category_risk = compute_category_risk(nominee_spreads)
    position_risk = compute_position_risk(
        nominee_spreads, pos, config_label, model_type, snapshot_key, scale_factor
    )

    # Generate markdown
    md = generate_markdown(
        nominee_spreads,
        category_risk,
        position_risk,
        snapshot_key,
        config_name,
        bankroll,
    )

    # Print to stdout
    print(md)

    # Write to storage results directory (alongside other CSVs)
    output_path = (
        Path(args.output)
        if args.output
        else Path(EXP_DIR) / "results" / "model_agreement_analysis.md"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md)
    print(f"\nReport written to: {output_path}")


if __name__ == "__main__":
    main()
