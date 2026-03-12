"""Compare P&L results across signal-delay modes.

Usage::

    cd "$(git rev-parse --show-toplevel)"
    uv run python oscar_prediction_market/one_offs/\
d20260220_backtest_strategies/compare_delay_modes.py
"""

from pathlib import Path

import pandas as pd

BASE = Path("storage/d20260220_backtest_strategies/2025")

# Well-known mode labels for nicer display; auto-discovered dirs use their name.
_KNOWN_LABELS: dict[str, str] = {
    "delay_0": "Same-day (delay=0)",
    "delay_1": "Next-day (delay=1)",
    "inferred_6h": "Inferred+6h",
}


def _discover_modes(base: Path) -> dict[str, str]:
    """Auto-discover results_* directories under *base*.

    Returns ``{mode_key: display_label}`` in sorted order.  Known modes get
    human-friendly labels; unknown ones use the directory suffix as-is.
    """
    modes: dict[str, str] = {}
    if not base.exists():
        return modes
    for d in sorted(base.iterdir()):
        if d.is_dir() and d.name.startswith("results_"):
            key = d.name.removeprefix("results_")
            modes[key] = _KNOWN_LABELS.get(key, key)
    return modes


def main() -> None:
    MODES = _discover_modes(BASE)
    if not MODES:
        print(f"No results_* directories found in {BASE}")
        return

    all_dfs: dict[str, pd.DataFrame] = {}
    for name in MODES:
        path = BASE / f"results_{name}" / "daily_pnl.csv"
        if not path.exists():
            print(f"SKIP {name}: {path} not found")
            continue
        all_dfs[name] = pd.read_csv(path)

    if not all_dfs:
        print("No results found!")
        return

    # --- Full breakdown per mode ---
    print("=" * 100)
    print("SIGNAL DELAY COMPARISON: Best P&L per (category, model_type)")
    print("=" * 100)

    for name, label in MODES.items():
        if name not in all_dfs:
            continue
        df = all_dfs[name]
        best = df.loc[df.groupby(["category", "model_type"])["total_pnl"].idxmax()]
        total = best["total_pnl"].sum()
        print(f"\n--- {label} ---")
        print(f"Sum of best P&L across all cat × model: ${total:+,.0f}")
        for _, r in best.sort_values(["category", "model_type"]).iterrows():
            print(
                f"  {r['category']:<25} {r['model_type']:<10} "
                f"P&L=${r['total_pnl']:>+9.2f} ({r['return_pct']:>+6.1f}%) "
                f"trades={int(r['total_trades'])}"
            )

    # --- Pivot table: cal_sgbt model, best config per category ---
    print("\n" + "=" * 100)
    print("COMPARISON TABLE: cal_sgbt best config P&L per category")
    print("=" * 100)

    rows = []
    for name, label in MODES.items():
        if name not in all_dfs:
            continue
        df = all_dfs[name]
        df_sgbt = df[df["model_type"] == "cal_sgbt"]
        best = df_sgbt.loc[df_sgbt.groupby("category")["total_pnl"].idxmax()]
        for _, r in best.iterrows():
            rows.append(
                {
                    "mode": label,
                    "category": r["category"],
                    "pnl": r["total_pnl"],
                    "return_pct": r["return_pct"],
                    "trades": int(r["total_trades"]),
                }
            )

    if rows:
        comp = pd.DataFrame(rows)
        pvt = comp.pivot(index="category", columns="mode", values="pnl")
        labels = [lbl for lbl in MODES.values() if lbl in pvt.columns]
        pvt = pvt[labels]
        print(pvt.to_string(float_format=lambda x: f"${x:+.0f}"))
        print()
        print("Totals:")
        for col in pvt.columns:
            print(f"  {col}: ${pvt[col].sum():+,.0f}")

    # --- Aggregate across all models: best single config per mode ---
    print("\n" + "=" * 100)
    print("AGGREGATE: Best single config P&L summed across all 9 categories")
    print("=" * 100)

    for name, label in MODES.items():
        if name not in all_dfs:
            continue
        df = all_dfs[name]
        # For each config, sum P&L across all categories (matching on config params)
        config_cols = [
            "kelly_fraction",
            "buy_edge_threshold",
            "fee_type",
            "kelly_mode",
            "allowed_directions",
            "min_price",
            "bankroll_mode",
        ]
        avail_cols = [c for c in config_cols if c in df.columns]

        # Group by model_type and config, sum total_pnl across categories
        print(f"\n--- {label} ---")
        for model_type in sorted(df["model_type"].unique()):
            try:
                model_grouped = (
                    df[df["model_type"] == model_type].groupby(avail_cols)["total_pnl"].sum()
                )
                best_total = model_grouped.max()
                n_cats = df[df["model_type"] == model_type]["category"].nunique()
                print(
                    f"  {model_type:<15} Best aggregate P&L=${best_total:>+9.2f} across {n_cats} categories"
                )
            except (KeyError, ValueError) as e:
                print(f"  {model_type:<15} ERROR: {e}")


if __name__ == "__main__":
    main()
