"""Analyze buy-and-hold backtest results.

Produces summary tables, temporal analysis, model comparison, and
comparison with the rebalancing backtest (d20260220).

Usage::

    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\\
d20260225_buy_hold_backtest.analyze --ceremony-year 2025
"""

import argparse
from pathlib import Path

import pandas as pd

from oscar_prediction_market.one_offs.d20260225_buy_hold_backtest.year_config import (
    YEAR_CONFIGS,
    YearConfig,
)

BANKROLL = 1000.0
BACKTEST_EXP_DIR = Path("storage/d20260220_backtest_strategies")


def load_data(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load entry-level and aggregate P&L data."""
    entry = pd.read_csv(results_dir / "entry_pnl.csv")
    agg = pd.read_csv(results_dir / "aggregate_pnl.csv")
    return entry, agg


def load_rebalancing_data(ceremony_year: int) -> pd.DataFrame | None:
    """Load rebalancing (d20260220) results for comparison."""
    rebal_dir = BACKTEST_EXP_DIR / str(ceremony_year) / "results_inferred_6h"
    path = rebal_dir / "daily_pnl.csv"
    if path.exists():
        return pd.read_csv(path)
    print(f"  (Rebalancing results not found at {path})")
    return None


# ============================================================================
# 1. Best aggregate config per category × model
# ============================================================================


def analyze_best_configs(agg: pd.DataFrame) -> pd.DataFrame:
    """Find best config per (category, model) by aggregate P&L."""
    print("\n" + "=" * 80)
    print("1. BEST AGGREGATE CONFIG PER CATEGORY × MODEL")
    print("=" * 80)

    # Filter to configs that actually traded
    active = agg[agg["entries_with_trades"] > 0].copy()

    idx = active.groupby(["category", "model_type"])["total_pnl"].idxmax()
    best = active.loc[idx].sort_values(["category", "total_pnl"], ascending=[True, False])

    for cat in sorted(best["category"].unique()):
        cat_best = best[best["category"] == cat]
        print(f"\n  {cat}:")
        for _, r in cat_best.iterrows():
            print(
                f"    {r['model_type']:<12} "
                f"P&L=${r['total_pnl']:>+9.2f} ({r['return_pct']:>+6.1f}%) "
                f"trades={r['total_trades']:>3}  entries={r['entries_with_trades']}/{r['n_entries']}  "
                f"fees=${r['total_fees']:.2f}  "
                f"kf={r['kelly_fraction']} bet={r['buy_edge_threshold']} "
                f"{r['kelly_mode']} {r['fee_type']} {r['allowed_directions']}"
            )

    return best


# ============================================================================
# 2. Temporal analysis: P&L by entry point
# ============================================================================


def analyze_temporal(entry: pd.DataFrame) -> None:
    """Analyze P&L across entry points (snapshots)."""
    print("\n" + "=" * 80)
    print("2. TEMPORAL ANALYSIS: P&L BY ENTRY POINT")
    print("=" * 80)

    # Use a reference config: maker, kf=0.25, bet=0.08, multi_outcome, yes
    ref_mask = (
        (entry["fee_type"] == "maker")
        & (entry["kelly_fraction"] == 0.25)
        & (entry["buy_edge_threshold"] == 0.08)
        & (entry["kelly_mode"] == "multi_outcome")
        & (entry["allowed_directions"] == "yes")
        & (entry["bankroll_mode"] == "fixed")
    )
    ref = entry[ref_mask].copy()

    if ref.empty:
        # Fallback to any reasonable config
        ref_mask = (
            (entry["fee_type"] == "maker")
            & (entry["kelly_fraction"] == 0.20)
            & (entry["buy_edge_threshold"] == 0.06)
            & (entry["kelly_mode"] == "multi_outcome")
            & (entry["allowed_directions"] == "yes")
        )
        ref = entry[ref_mask].copy()

    if ref.empty:
        print("  No reference config found")
        return

    print("\n  Reference config: maker, kf=0.25, bet=0.08, multi_outcome, YES side")
    print(f"  (Using {len(ref)} rows)\n")

    # Pivot: entry_snapshot × (category, model) → P&L
    pivot = ref.pivot_table(
        index="entry_snapshot",
        columns=["category", "model_type"],
        values="total_pnl",
        aggfunc="first",
    )

    # Show ensemble P&L by entry point × category
    models = ["avg_ensemble"]
    for model in models:
        cols = [
            (cat, model)
            for cat in sorted(ref["category"].unique())
            if (cat, model) in pivot.columns
        ]
        if not cols:
            continue
        sub = pivot[cols].copy()
        sub.columns = [c[0] for c in sub.columns]  # drop model level
        sub["TOTAL"] = sub.sum(axis=1)
        print("\n  Ensemble P&L by entry point:")
        print(f"  {'Entry':<12}", end="")
        for col in sub.columns:
            print(f" {col[:12]:>12}", end="")
        print()
        for snap, row in sub.iterrows():
            print(f"  {snap:<12}", end="")
            for val in row:
                if pd.isna(val) or val == 0:
                    print(f" {'—':>12}", end="")
                else:
                    print(f" ${val:>+10.2f}", end="")
            print()

    # Aggregate across all models: which entry point is best?
    print("\n  Average P&L across all models per entry point (same config):")
    entry_avg = ref.groupby("entry_snapshot")["total_pnl"].agg(["mean", "median", "sum", "count"])
    entry_avg.columns = ["Mean P&L", "Median P&L", "Total P&L", "N"]
    for snap, row in entry_avg.iterrows():
        print(
            f"    {snap}  mean=${row['Mean P&L']:>+7.2f}  "
            f"median=${row['Median P&L']:>+7.2f}  "
            f"total=${row['Total P&L']:>+8.2f}  n={int(row['N'])}"
        )


# ============================================================================
# 3. Model comparison
# ============================================================================


def analyze_model_comparison(agg: pd.DataFrame) -> None:
    """Compare models across categories."""
    print("\n" + "=" * 80)
    print("3. MODEL COMPARISON (AGGREGATE P&L, BEST CONFIG PER MODEL)")
    print("=" * 80)

    # For each model, find its best config, then sum P&L across categories
    sorted(agg["model_type"].unique())

    # Method: For each (category, model), use best config. Sum across categories.
    active = agg[agg["entries_with_trades"] > 0].copy()
    idx = active.groupby(["category", "model_type"])["total_pnl"].idxmax()
    best = active.loc[idx]

    model_totals = (
        best.groupby("model_type")
        .agg(
            total_pnl=("total_pnl", "sum"),
            total_trades=("total_trades", "sum"),
            total_fees=("total_fees", "sum"),
            n_categories=("category", "nunique"),
            avg_return_pct=("return_pct", "mean"),
        )
        .sort_values("total_pnl", ascending=False)
    )

    print("\n  Model ranking (sum of best-config P&L across all categories):")
    for model, row in model_totals.iterrows():
        print(
            f"    {model:<14} P&L=${row['total_pnl']:>+10.2f}  "
            f"avg_return={row['avg_return_pct']:>+6.1f}%  "
            f"trades={int(row['total_trades']):>4}  "
            f"fees=${row['total_fees']:>6.2f}  "
            f"categories={int(row['n_categories'])}"
        )

    # Per-category model rankings
    print("\n  Per-category model rankings (best config P&L):")
    for cat in sorted(best["category"].unique()):
        cat_data = best[best["category"] == cat].sort_values("total_pnl", ascending=False)
        print(f"\n    {cat}:")
        for _, r in cat_data.iterrows():
            print(
                f"      {r['model_type']:<14} P&L=${r['total_pnl']:>+9.2f} ({r['return_pct']:>+6.1f}%)"
            )


# ============================================================================
# 4. Config sensitivity analysis
# ============================================================================


def analyze_config_sensitivity(agg: pd.DataFrame) -> None:
    """Analyze sensitivity to key config parameters."""
    print("\n" + "=" * 80)
    print("4. CONFIG SENSITIVITY ANALYSIS")
    print("=" * 80)

    # Filter to ensemble only for cleaner signal
    ens = agg[agg["model_type"] == "avg_ensemble"].copy()
    if ens.empty:
        ens = agg.copy()

    active = ens[ens["entries_with_trades"] > 0]

    # By kelly_fraction
    print("\n  a) By kelly_fraction (avg across categories, ensemble):")
    by_kf = active.groupby("kelly_fraction")["return_pct"].agg(["mean", "median", "std", "count"])
    for kf, row in by_kf.iterrows():
        print(
            f"    kf={kf:<5}  mean={row['mean']:>+6.1f}%  "
            f"median={row['median']:>+6.1f}%  std={row['std']:>5.1f}%  n={int(row['count'])}"
        )

    # By buy_edge_threshold
    print("\n  b) By buy_edge_threshold:")
    by_bet = active.groupby("buy_edge_threshold")["return_pct"].agg(["mean", "median", "std"])
    for bet, row in by_bet.iterrows():
        print(
            f"    bet={bet:<5}  mean={row['mean']:>+6.1f}%  "
            f"median={row['median']:>+6.1f}%  std={row['std']:>5.1f}%"
        )

    # By kelly_mode
    print("\n  c) By kelly_mode:")
    by_km = active.groupby("kelly_mode")["return_pct"].agg(["mean", "median", "std"])
    for km, row in by_km.iterrows():
        print(
            f"    {km:<16}  mean={row['mean']:>+6.1f}%  "
            f"median={row['median']:>+6.1f}%  std={row['std']:>5.1f}%"
        )

    # By fee_type
    print("\n  d) By fee_type:")
    by_fee = active.groupby("fee_type")["return_pct"].agg(["mean", "median", "std"])
    for fee, row in by_fee.iterrows():
        print(
            f"    {fee:<8}  mean={row['mean']:>+6.1f}%  "
            f"median={row['median']:>+6.1f}%  std={row['std']:>5.1f}%"
        )

    # By allowed_directions
    print("\n  e) By allowed_directions:")
    by_side = active.groupby("allowed_directions")["return_pct"].agg(["mean", "median", "std"])
    for side, row in by_side.iterrows():
        print(
            f"    {side:<5}  mean={row['mean']:>+6.1f}%  "
            f"median={row['median']:>+6.1f}%  std={row['std']:>5.1f}%"
        )


# ============================================================================
# 5. Comparison with rebalancing backtest (d20260220)
# ============================================================================


def analyze_vs_rebalancing(agg: pd.DataFrame, year_config: YearConfig) -> None:
    """Compare buy-and-hold vs rebalancing from d20260220."""
    print("\n" + "=" * 80)
    print("5. BUY-AND-HOLD vs REBALANCING (d20260220)")
    print("=" * 80)

    rebal = load_rebalancing_data(ceremony_year=year_config.ceremony_year)
    if rebal is None:
        return

    # For fair comparison: use best config per (category, model) from each approach
    active_bh = agg[agg["entries_with_trades"] > 0]
    idx_bh = active_bh.groupby(["category", "model_type"])["total_pnl"].idxmax()
    best_bh = active_bh.loc[idx_bh][
        ["category", "model_type", "total_pnl", "total_trades", "total_fees"]
    ].copy()
    best_bh = best_bh.rename(
        columns={
            "total_pnl": "bh_pnl",
            "total_trades": "bh_trades",
            "total_fees": "bh_fees",
        }
    )

    active_rb = rebal[rebal["total_trades"] > 0]
    if "total_pnl" in active_rb.columns:
        idx_rb = active_rb.groupby(["category", "model_type"])["total_pnl"].idxmax()
        best_rb = active_rb.loc[idx_rb][
            ["category", "model_type", "total_pnl", "total_trades", "total_fees"]
        ].copy()
        best_rb = best_rb.rename(
            columns={
                "total_pnl": "rb_pnl",
                "total_trades": "rb_trades",
                "total_fees": "rb_fees",
            }
        )
    else:
        print("  Rebalancing CSV missing total_pnl column")
        return

    merged = pd.merge(best_bh, best_rb, on=["category", "model_type"], how="outer")
    merged["pnl_diff"] = merged["bh_pnl"].fillna(0) - merged["rb_pnl"].fillna(0)

    print("\n  Best-config P&L comparison (Buy-Hold vs Rebalancing):")
    print(
        f"  {'Category':<25} {'Model':<14} {'BH P&L':>10} {'RB P&L':>10} {'Diff':>10} {'BH Trades':>10} {'RB Trades':>10}"
    )
    print(f"  {'-' * 25} {'-' * 14} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")

    for _, r in merged.sort_values(["category", "model_type"]).iterrows():
        bh_pnl = r["bh_pnl"] if pd.notna(r["bh_pnl"]) else 0
        rb_pnl = r["rb_pnl"] if pd.notna(r["rb_pnl"]) else 0
        bh_tr = int(r["bh_trades"]) if pd.notna(r["bh_trades"]) else 0
        rb_tr = int(r["rb_trades"]) if pd.notna(r["rb_trades"]) else 0
        print(
            f"  {r['category']:<25} {r['model_type']:<14} "
            f"${bh_pnl:>+9.2f} ${rb_pnl:>+9.2f} ${r['pnl_diff']:>+9.2f} "
            f"{bh_tr:>10} {rb_tr:>10}"
        )

    # Summary
    bh_total = merged["bh_pnl"].fillna(0).sum()
    rb_total = merged["rb_pnl"].fillna(0).sum()
    bh_trades_total = merged["bh_trades"].fillna(0).sum()
    rb_trades_total = merged["rb_trades"].fillna(0).sum()
    print("\n  TOTAL (best config per cat×model):")
    print(f"    Buy-Hold:    P&L=${bh_total:>+10.2f}  trades={int(bh_trades_total)}")
    print(f"    Rebalancing: P&L=${rb_total:>+10.2f}  trades={int(rb_trades_total)}")
    print(f"    Difference:  ${bh_total - rb_total:>+10.2f}")

    # Ensemble-only comparison
    ens_bh = best_bh[best_bh["model_type"] == "avg_ensemble"]
    ens_rb = best_rb[best_rb["model_type"] == "avg_ensemble"]
    if not ens_bh.empty and not ens_rb.empty:
        print("\n  Ensemble-only comparison:")
        ens_merged = pd.merge(ens_bh, ens_rb, on=["category", "model_type"], how="outer")
        for _, r in ens_merged.sort_values("category").iterrows():
            bh_p = r["bh_pnl"] if pd.notna(r["bh_pnl"]) else 0
            rb_p = r["rb_pnl"] if pd.notna(r["rb_pnl"]) else 0
            print(
                f"    {r['category']:<25} BH=${bh_p:>+9.2f}  RB=${rb_p:>+9.2f}  diff=${bh_p - rb_p:>+9.2f}"
            )
        print(
            f"    {'TOTAL':<25} BH=${ens_bh['bh_pnl'].sum():>+9.2f}  RB=${ens_rb['rb_pnl'].sum():>+9.2f}"
        )


# ============================================================================
# 6. Risk analysis: worst-case entries
# ============================================================================


def analyze_risk(entry: pd.DataFrame) -> None:
    """Analyze worst-case losses and risk profile."""
    print("\n" + "=" * 80)
    print("6. RISK ANALYSIS")
    print("=" * 80)

    # Use a reasonable config subset
    active = entry[entry["total_trades"] > 0].copy()

    # Worst individual entries
    worst = active.nsmallest(20, "total_pnl")
    print("\n  20 worst individual entries:")
    for _, r in worst.iterrows():
        print(
            f"    {r['category']:<25} {r['model_type']:<12} {r['entry_snapshot']}  "
            f"P&L=${r['total_pnl']:>+9.2f}  trades={r['total_trades']}  "
            f"kf={r['kelly_fraction']} bet={r['buy_edge_threshold']}"
        )

    # Max loss per category
    print("\n  Max loss per category (across all configs, any entry):")
    max_loss = active.groupby("category")["total_pnl"].min()
    for cat, loss in max_loss.sort_values().items():
        print(f"    {cat:<25} ${loss:>+9.2f}")

    # Win rate by entry point
    print("\n  Win rate by entry point (% of active entries with P&L > 0):")
    for snap in sorted(active["entry_snapshot"].unique()):
        snap_data = active[active["entry_snapshot"] == snap]
        win_rate = (snap_data["total_pnl"] > 0).mean() * 100
        avg_pnl = snap_data["total_pnl"].mean()
        print(
            f"    {snap}  win_rate={win_rate:>5.1f}%  avg P&L=${avg_pnl:>+7.2f}  n={len(snap_data)}"
        )


# ============================================================================
# 7. Robust config recommendation
# ============================================================================


def analyze_robust_config(agg: pd.DataFrame) -> None:
    """Find configs that perform well across multiple categories."""
    print("\n" + "=" * 80)
    print("7. ROBUST CONFIG RECOMMENDATION")
    print("=" * 80)

    # Configs that are profitable in most categories (ensemble only)
    ens = agg[agg["model_type"] == "avg_ensemble"].copy()
    if ens.empty:
        print("  No ensemble data")
        return

    # For each config, count categories where it's profitable
    config_stats = ens.groupby("config_label").agg(
        total_pnl=("total_pnl", "sum"),
        mean_return=("return_pct", "mean"),
        positive_cats=("total_pnl", lambda x: (x > 0).sum()),
        n_cats=("total_pnl", "count"),
        total_trades=("total_trades", "sum"),
    )
    config_stats["win_rate"] = config_stats["positive_cats"] / config_stats["n_cats"] * 100

    # Sort by: 1) most profitable categories, 2) total P&L
    config_stats = config_stats.sort_values(
        ["positive_cats", "total_pnl"], ascending=[False, False]
    )

    print("\n  Top 15 most robust configs (ensemble, sorted by categories profitable then P&L):")
    for i, (label, row) in enumerate(config_stats.head(15).iterrows()):
        # Parse config label to show key params
        print(
            f"    {i + 1:>2}. profitable in {int(row['positive_cats'])}/{int(row['n_cats'])} cats  "
            f"P&L=${row['total_pnl']:>+9.2f}  "
            f"avg_ret={row['mean_return']:>+5.1f}%  "
            f"trades={int(row['total_trades']):>3}  "
            f"{str(label)[:60]}"
        )

    # Best config that's profitable in >= 7 categories
    robust = config_stats[config_stats["positive_cats"] >= 7]
    if not robust.empty:
        best_robust = robust.iloc[0]
        print(
            f"\n  RECOMMENDED CONFIG (profitable in {int(best_robust['positive_cats'])}+ categories):"
        )
        print(f"    {robust.index[0]}")
        print(f"    Total P&L: ${best_robust['total_pnl']:>+.2f}")
        print(f"    Avg return: {best_robust['mean_return']:>+.1f}%")
    else:
        # Relax to 5+
        robust = config_stats[config_stats["positive_cats"] >= 5]
        if not robust.empty:
            best_robust = robust.iloc[0]
            print(
                f"\n  RECOMMENDED CONFIG (profitable in {int(best_robust['positive_cats'])}+ categories):"
            )
            print(f"    {robust.index[0]}")
            print(f"    Total P&L: ${best_robust['total_pnl']:>+.2f}")


# ============================================================================
# 8. Best entry window analysis
# ============================================================================


def analyze_best_entry_window(entry: pd.DataFrame) -> None:
    """Which entry point (snapshot) produces the best returns?"""
    print("\n" + "=" * 80)
    print("8. BEST ENTRY WINDOW")
    print("=" * 80)

    # For each entry point, consider ensemble with a reasonable config range
    ens = entry[entry["model_type"] == "avg_ensemble"].copy()
    if ens.empty:
        ens = entry.copy()

    active = ens[ens["total_trades"] > 0]

    # Average return per entry point (across all configs)
    print("\n  Average return by entry point (across all configs with trades):")
    by_entry = (
        active.groupby(["entry_snapshot", "entry_events"])
        .agg(
            mean_pnl=("total_pnl", "mean"),
            median_pnl=("total_pnl", "median"),
            max_pnl=("total_pnl", "max"),
            min_pnl=("total_pnl", "min"),
            mean_return=("return_pct", "mean"),
            n_configs=("total_pnl", "count"),
        )
        .reset_index()
    )

    for _, r in by_entry.iterrows():
        print(
            f"    {r['entry_snapshot']}  {r['entry_events'][:35]:<37}"
            f"mean=${r['mean_pnl']:>+7.2f}  max=${r['max_pnl']:>+8.2f}  "
            f"min=${r['min_pnl']:>+8.2f}  n={int(r['n_configs'])}"
        )

    # By entry point × category (ensemble, reference config)
    ref = active[
        (active["kelly_fraction"] == 0.25)
        & (active["buy_edge_threshold"] == 0.08)
        & (active["kelly_mode"] == "multi_outcome")
        & (active["fee_type"] == "maker")
    ]
    if not ref.empty:
        print("\n  Entry × Category P&L (ensemble, kf=0.25/bet=0.08/multi/maker):")
        pivot = ref.pivot_table(
            index="entry_snapshot",
            columns="category",
            values="total_pnl",
            aggfunc="first",
        )
        # Shorten category names
        short_cats = {c: c[:10] for c in pivot.columns}
        pivot = pivot.rename(columns=short_cats)
        pivot["TOTAL"] = pivot.sum(axis=1)
        print(f"    {'Entry':<12}", end="")
        for col in pivot.columns:
            print(f" {col:>10}", end="")
        print()
        for snap, row in pivot.iterrows():
            print(f"    {snap:<12}", end="")
            for val in row:
                if pd.isna(val) or val == 0:
                    print(f" {'—':>10}", end="")
                else:
                    print(f" ${val:>+8.1f}", end="")
            print()


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Run all analyses."""
    parser = argparse.ArgumentParser(description="Analyze buy-and-hold backtest results")
    parser.add_argument(
        "--ceremony-year",
        type=int,
        default=2025,
        choices=sorted(YEAR_CONFIGS.keys()),
        help="Ceremony year to analyze (default: 2025).",
    )
    args = parser.parse_args()

    year_config = YEAR_CONFIGS[args.ceremony_year]
    entry, agg = load_data(year_config.results_dir)
    print(f"Loaded {len(entry)} entry rows, {len(agg)} aggregate rows")
    print(f"Categories: {sorted(entry['category'].unique())}")
    print(f"Models: {sorted(entry['model_type'].unique())}")
    print(f"Entry points: {sorted(entry['entry_snapshot'].unique())}")

    analyze_best_configs(agg)
    analyze_temporal(entry)
    analyze_model_comparison(agg)
    analyze_config_sensitivity(agg)
    analyze_vs_rebalancing(agg, year_config)
    analyze_risk(entry)
    analyze_robust_config(agg)
    analyze_best_entry_window(entry)

    print("\n\nAnalysis complete.")


if __name__ == "__main__":
    main()
