"""Generate markdown tables for buy-and-hold backtest READMEs.

Reads CSV results and prints formatted markdown tables to stdout,
ready to be piped/pasted into year-specific or cross-year READMEs.

Usage::

    cd "$(git rev-parse --show-toplevel)"

    # Per-year tables:
    uv run python -m oscar_prediction_market.one_offs.\\
d20260225_buy_hold_backtest.generate_tables --ceremony-year 2025

    # Cross-year tables:
    uv run python -m oscar_prediction_market.one_offs.\\
d20260225_buy_hold_backtest.generate_tables --mode cross-year
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from oscar_prediction_market.one_offs.d20260225_buy_hold_backtest.year_config import (
    BUY_HOLD_EXP_DIR,
    YEAR_CONFIGS,
    YearConfig,
)

BANKROLL = 1000.0

# Config parameter columns used for grouping
CONFIG_PARAMS = [
    "model_type",
    "config_label",
    "fee_type",
    "kelly_fraction",
    "buy_edge_threshold",
    "kelly_mode",
    "bankroll_mode",
    "allowed_directions",
]


# ============================================================================
# Data loading helpers
# ============================================================================


def load_entry_pnl(cfg: YearConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.results_dir / "entry_pnl.csv")
    return df[df["bankroll_mode"] == "fixed"]


def load_aggregate_pnl(cfg: YearConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.results_dir / "aggregate_pnl.csv")
    return df[df["bankroll_mode"] == "fixed"]


def load_risk_profile(cfg: YearConfig) -> pd.DataFrame:
    return pd.read_csv(cfg.results_dir / "risk_profile.csv")


def _portfolio_pnl(agg: pd.DataFrame) -> pd.DataFrame:
    """Sum total_pnl across categories per (model, config) to get portfolio P&L.

    Returns one row per (model_type, config_label, ...) with ``portfolio_pnl``.
    """
    group_cols = [c for c in CONFIG_PARAMS if c in agg.columns]
    sum_cols = ["total_pnl", "total_fees", "total_trades"]
    if "capital_deployed" in agg.columns:
        sum_cols.append("capital_deployed")
    if "total_bankroll_deployed" in agg.columns:
        sum_cols.append("total_bankroll_deployed")

    portfolio = agg.groupby(group_cols, as_index=False)[sum_cols].sum()
    portfolio = portfolio.rename(columns={"total_pnl": "portfolio_pnl"})
    return portfolio


# ============================================================================
# Markdown formatting helpers
# ============================================================================


def _fmt(v: float, decimals: int = 2) -> str:
    """Format a float with fixed decimals."""
    return f"{v:,.{decimals}f}"


def _fmt_pct(v: float, decimals: int = 1) -> str:
    return f"{v:.{decimals}f}%"


def _fmt_dollar(v: float, decimals: int = 2) -> str:
    return f"${v:,.{decimals}f}"


def _print_md_table(headers: list[str], rows: list[list[str]], align: list[str] | None = None):
    """Print a markdown table.

    Args:
        headers: Column headers.
        rows: List of rows, each a list of strings.
        align: Per-column alignment: "l", "r", or "c". Defaults to left.
    """
    if align is None:
        align = ["l"] * len(headers)

    # Header
    print("| " + " | ".join(headers) + " |")

    # Alignment row
    sep_parts = []
    for a in align:
        if a == "r":
            sep_parts.append("---:")
        elif a == "c":
            sep_parts.append(":---:")
        else:
            sep_parts.append(":---")
    print("| " + " | ".join(sep_parts) + " |")

    # Data rows
    for row in rows:
        print("| " + " | ".join(str(c) for c in row) + " |")


def _short_model(model: str) -> str:
    """Shorten model type names for table display."""
    abbrevs = {
        "avg_ensemble": "avg_ens",
        "cal_sgbt": "cal_sgbt",
        "clogit": "clogit",
        "clogit_cal_sgbt_ensemble": "clog_sgbt",
        "lr": "lr",
        "sgbt": "sgbt",
    }
    return abbrevs.get(model, model)


def _short_side(s: str) -> str:
    return {"yes": "Y", "no": "N", "all": "A"}.get(s, s)


def _short_kelly_mode(km: str) -> str:
    return {"independent": "ind", "multi_outcome": "multi"}.get(km, km)


def _config_summary(row: pd.Series) -> str:
    """Compact config summary like 'lr/Y/ind kf=0.25 e=0.08'."""
    model = _short_model(str(row.get("best_model", row.get("model_type", ""))))
    side = _short_side(str(row.get("best_side", row.get("allowed_directions", ""))))
    km = _short_kelly_mode(str(row.get("best_km", row.get("kelly_mode", ""))))
    kf = row.get("best_kf", row.get("kelly_fraction", ""))
    edge = row.get("best_edge", row.get("buy_edge_threshold", ""))
    return f"{model}/{side}/{km} kf={kf} e={edge}"


# ============================================================================
# Per-year tables
# ============================================================================


def table_portfolio_summary(cfg: YearConfig) -> None:
    """Table 1: Portfolio P&L Summary by Model."""
    print(f"\n## 1. Portfolio P&L Summary by Model ({cfg.ceremony_year})\n")

    agg = load_aggregate_pnl(cfg)
    portfolio = _portfolio_pnl(agg)

    has_capital = "capital_deployed" in portfolio.columns
    has_bankroll = "total_bankroll_deployed" in portfolio.columns

    rows = []
    for model in sorted(portfolio["model_type"].unique()):
        m = portfolio[portfolio["model_type"] == model]
        pnl = m["portfolio_pnl"]
        n_configs = len(pnl)
        pct_profitable = (pnl > 0).mean() * 100

        row: list[str] = [
            _short_model(model),
            str(n_configs),
            _fmt_dollar(pnl.max()),
            _fmt_dollar(pnl.mean()),
            _fmt_dollar(pnl.median()),
            _fmt_dollar(pnl.quantile(0.1)),
            _fmt_dollar(pnl.quantile(0.9)),
            _fmt_pct(pct_profitable),
        ]

        if has_capital and has_bankroll:
            deployed = m["capital_deployed"]
            bankroll = m["total_bankroll_deployed"]
            mask = deployed > 0
            if mask.any():
                roi_deployed = (m.loc[mask, "portfolio_pnl"] / deployed[mask]).mean() * 100
                util = (deployed[mask] / bankroll[mask]).mean() * 100
                row.append(_fmt_pct(roi_deployed))
                row.append(_fmt_pct(util))
            else:
                row.extend(["n/a", "n/a"])

        rows.append(row)

    headers = ["Model", "#Cfg", "Best P&L", "Mean", "Median", "P10", "P90", "% Prof"]
    align = ["l", "r", "r", "r", "r", "r", "r", "r"]
    if has_capital and has_bankroll:
        headers.extend(["Mean ROI", "Util %"])
        align.extend(["r", "r"])

    _print_md_table(headers, rows, align)


def table_top_portfolio_pnl(cfg: YearConfig, n: int = 15) -> None:
    """Table 3: Best Configs by Portfolio P&L (Top N)."""
    print(f"\n## 3. Best Configs by Portfolio P&L (Top {n}, {cfg.ceremony_year})\n")

    agg = load_aggregate_pnl(cfg)
    portfolio = _portfolio_pnl(agg)
    top = portfolio.nlargest(n, "portfolio_pnl").reset_index(drop=True)

    rows = []
    for i, r in top.iterrows():
        rows.append(
            [
                str(int(i) + 1),  # type: ignore[arg-type, call-overload]
                _short_model(r["model_type"]),
                _fmt(r["kelly_fraction"]),
                _fmt(r["buy_edge_threshold"]),
                _short_kelly_mode(r.get("kelly_mode", "")),
                _short_side(r.get("allowed_directions", "")),
                r.get("fee_type", ""),
                _fmt_dollar(r["portfolio_pnl"]),
                _fmt_dollar(r["total_fees"]),
                str(int(r["total_trades"])),
            ]
        )

    headers = [
        "#",
        "Model",
        "KF",
        "Edge",
        "KM",
        "Side",
        "Fee",
        "Portfolio P&L",
        "Fees",
        "Trades",
    ]
    align = ["r", "l", "r", "r", "l", "l", "l", "r", "r", "r"]
    _print_md_table(headers, rows, align)


def table_parameter_sensitivity(cfg: YearConfig) -> None:
    """Table 4: Parameter Sensitivity (Portfolio-Level).

    For each parameter, groups portfolio-level P&L by parameter value and
    shows mean, median, and % of configs that are profitable.
    """
    print(f"\n## 4. Parameter Sensitivity ({cfg.ceremony_year})\n")

    agg = load_aggregate_pnl(cfg)
    portfolio = _portfolio_pnl(agg)

    params = [
        "fee_type",
        "kelly_fraction",
        "buy_edge_threshold",
        "kelly_mode",
        "allowed_directions",
    ]

    for param in params:
        if param not in portfolio.columns:
            continue
        print(f"\n### {param}\n")

        grouped = portfolio.groupby(param)["portfolio_pnl"]
        rows = []
        for val, grp in sorted(grouped, key=lambda x: str(x[0])):
            rows.append(
                [
                    str(val),
                    str(len(grp)),
                    _fmt_dollar(grp.mean()),
                    _fmt_dollar(grp.median()),
                    _fmt_pct((grp > 0).mean() * 100),
                ]
            )

        headers = ["Value", "#Cfg", "Mean P&L", "Median P&L", "% Prof"]
        align = ["l", "r", "r", "r", "r"]
        _print_md_table(headers, rows, align)


def table_entry_timing(cfg: YearConfig) -> None:
    """Table 5: Entry Timing — Marginal Portfolio P&L per snapshot.

    For each entry snapshot, sums P&L across categories per (model, config)
    to get a portfolio P&L per entry, then reports summary stats.
    """
    print(f"\n## 5. Entry Timing ({cfg.ceremony_year})\n")

    entry = load_entry_pnl(cfg)

    # Sum P&L across categories per (model, config, entry_snapshot)
    group_cols = [c for c in CONFIG_PARAMS if c in entry.columns] + [
        "entry_snapshot",
        "entry_events",
    ]
    entry_portfolio = entry.groupby(group_cols, as_index=False)["total_pnl"].sum()
    entry_portfolio = entry_portfolio.rename(columns={"total_pnl": "portfolio_pnl"})  # type: ignore[call-overload]

    # Get snapshot ordering from first occurrence
    snapshot_order = entry.drop_duplicates("entry_snapshot")[
        ["entry_snapshot", "entry_events", "entry_timestamp"]
    ].reset_index(drop=True)

    rows = []
    for _, snap_row in snapshot_order.iterrows():
        snap = snap_row["entry_snapshot"]
        mask = entry_portfolio["entry_snapshot"] == snap
        pnl = entry_portfolio.loc[mask, "portfolio_pnl"]
        if pnl.empty:
            continue

        rows.append(
            [
                snap,
                snap_row["entry_events"],
                str(len(pnl)),
                _fmt_dollar(pnl.mean()),
                _fmt_dollar(pnl.median()),
                _fmt_dollar(pnl.max()),
                _fmt_pct((pnl > 0).mean() * 100),
            ]
        )

    headers = ["Entry Snapshot", "Events", "#Cfg", "Mean P&L", "Median", "Best", "% Prof"]
    align = ["l", "l", "r", "r", "r", "r", "r"]
    _print_md_table(headers, rows, align)


def table_category_model_matrix(cfg: YearConfig) -> None:
    """Table 6: Per-Category P&L Matrix (Model × Category).

    Shows the best P&L for each (model, category) from aggregate data,
    with a TOTAL row summing across categories.
    """
    print(f"\n## 6. Category × Model P&L Matrix ({cfg.ceremony_year})\n")

    agg = load_aggregate_pnl(cfg)

    # Best P&L per (model, category)
    best = agg.groupby(["model_type", "category"])["total_pnl"].max().unstack(fill_value=0)

    # Sort categories alphabetically, models alphabetically
    cats = sorted(best.columns)
    models = sorted(best.index)

    # Shorten category names to fit table
    cat_short = {c: c.replace("_", " ").title()[:12] for c in cats}

    headers = ["Model"] + [cat_short[c] for c in cats] + ["TOTAL"]
    align = ["l"] + ["r"] * (len(cats) + 1)

    rows = []
    for model in models:
        row = [_short_model(model)]
        total = 0.0
        for cat in cats:
            val = best.loc[model, cat] if cat in best.columns else 0.0
            total += val
            row.append(_fmt_dollar(val))
        row.append(_fmt_dollar(total))
        rows.append(row)

    # TOTAL row
    total_row = ["**TOTAL**"]
    grand_total = 0.0
    for cat in cats:
        col_total = best[cat].sum() if cat in best.columns else 0.0
        grand_total += col_total
        total_row.append(_fmt_dollar(col_total))
    total_row.append(f"**{_fmt_dollar(grand_total)}**")
    rows.append(total_row)

    _print_md_table(headers, rows, align)


def table_capital_utilization(cfg: YearConfig) -> None:
    """Table 7: Capital Utilization.

    Shows per-model averages of capital deployed, bankroll allocated,
    utilization, ROI on deployed capital, and ROI on allocated bankroll.
    Also shows per-entry-point average ROI.
    """
    agg = load_aggregate_pnl(cfg)

    if "capital_deployed" not in agg.columns:
        print(f"\n## 7. Capital Utilization ({cfg.ceremony_year})\n")
        print("_capital_deployed column not available in this dataset._\n")
        return

    print(f"\n## 7. Capital Utilization ({cfg.ceremony_year})\n")

    portfolio = _portfolio_pnl(agg)
    has_bankroll = "total_bankroll_deployed" in portfolio.columns

    # Per-model summary
    rows = []
    for model in sorted(portfolio["model_type"].unique()):
        m = portfolio[portfolio["model_type"] == model]
        deployed = m["capital_deployed"]
        pnl = m["portfolio_pnl"]

        mask = deployed > 0
        mean_deployed = deployed.mean()
        roi_deployed = (pnl[mask] / deployed[mask]).mean() * 100 if mask.any() else 0.0

        row = [
            _short_model(model),
            _fmt_dollar(mean_deployed),
        ]

        if has_bankroll:
            bankroll = m["total_bankroll_deployed"]
            mean_bankroll = bankroll.mean()
            util_pct = (deployed[mask] / bankroll[mask]).mean() * 100 if mask.any() else 0.0
            roi_bankroll = (pnl[mask] / bankroll[mask]).mean() * 100 if mask.any() else 0.0
            row.extend(
                [
                    _fmt_dollar(mean_bankroll),
                    _fmt_pct(util_pct),
                    _fmt_pct(roi_deployed),
                    _fmt_pct(roi_bankroll),
                ]
            )
        else:
            row.extend([_fmt_pct(roi_deployed)])

        rows.append(row)

    headers = ["Model", "Avg Deployed"]
    align = ["l", "r"]
    if has_bankroll:
        headers.extend(["Avg Bankroll", "Util %", "ROI Deployed", "ROI Bankroll"])
        align.extend(["r", "r", "r", "r"])
    else:
        headers.append("ROI Deployed")
        align.append("r")

    print("### Per-Model Summary\n")
    _print_md_table(headers, rows, align)

    # Per-entry-point average ROI
    if "capital_deployed" in agg.columns:
        entry_df = load_entry_pnl(cfg)
        if "capital_deployed" in entry_df.columns:
            print("\n### Per-Entry-Point Average ROI\n")
            entry_df = entry_df[entry_df["bankroll_mode"] == "fixed"]
            mask = entry_df["capital_deployed"] > 0
            if mask.any():
                active = entry_df[mask].copy()
                active["roi"] = active["total_pnl"] / active["capital_deployed"] * 100

                snap_roi = active.groupby("entry_snapshot")["roi"].agg(["mean", "median", "count"])
                snap_roi = snap_roi.reset_index()

                rows = []
                for _, r in snap_roi.iterrows():
                    rows.append(
                        [
                            r["entry_snapshot"],
                            str(int(r["count"])),
                            _fmt_pct(r["mean"]),
                            _fmt_pct(r["median"]),
                        ]
                    )

                _print_md_table(
                    ["Entry", "#Active", "Mean ROI", "Median ROI"],
                    rows,
                    ["l", "r", "r", "r"],
                )


def table_pareto_frontier(cfg: YearConfig) -> None:
    """Table 13: EV Scoring Pareto Frontier."""
    print(f"\n## 13. EV Scoring Pareto Frontier ({cfg.ceremony_year})\n")

    # Prefer unified cvar00 naming; fall back to legacy name
    path = Path(cfg.results_dir) / "pareto_frontier_cvar00_blend.csv"
    if not path.exists():
        path = Path(cfg.results_dir) / "pareto_frontier_blend.csv"
    if not path.exists():
        print("_Data not available._\n")
        return

    df = pd.read_csv(path)
    # Support both old (best_worst_pnl) and new (best_risk) column names
    risk_col = "best_risk" if "best_risk" in df.columns else "best_worst_pnl"

    rows = []
    for _, r in df.iterrows():
        cap_deployed = r.get("best_capital_deployed", 0)
        actual = r.get("best_total_pnl", 0)
        roic = (actual / cap_deployed * 100) if cap_deployed > 0 else 0.0
        rows.append(
            [
                _fmt(r["loss_bound_pct"], 0),
                str(int(r["n_feasible"])),
                _fmt_dollar(r["best_ev"]),
                _fmt_dollar(r[risk_col]),
                _fmt_dollar(actual),
                _fmt_pct(r["best_deployment_rate"]),
                _fmt_pct(roic),
                _config_summary(r),
            ]
        )

    headers = [
        "L (%)",
        "#Feasible",
        "Best EV ($)",
        "Worst ($)",
        "Actual ($)",
        "Deploy%",
        "ROIC%",
        "Config summary",
    ]
    align = ["r", "r", "r", "r", "r", "r", "r", "l"]
    _print_md_table(headers, rows, align)


def table_cvar_pareto(cfg: YearConfig) -> None:
    """Table 14: CVaR Pareto Frontier."""
    print(f"\n## 14. CVaR Pareto Frontier ({cfg.ceremony_year})\n")

    path = Path(cfg.results_dir) / "pareto_frontier_cvar05_blend.csv"
    if not path.exists():
        print("_Data not available._\n")
        return

    df = pd.read_csv(path)
    # Support both old (best_cvar) and new (best_risk) column names
    risk_col = "best_risk" if "best_risk" in df.columns else "best_cvar"

    rows = []
    for _, r in df.iterrows():
        cap_deployed = r.get("best_capital_deployed", 0)
        actual = r.get("best_total_pnl", 0)
        roic = (actual / cap_deployed * 100) if cap_deployed > 0 else 0.0
        rows.append(
            [
                _fmt(r["loss_bound_pct"], 0),
                str(int(r["n_feasible"])),
                _fmt_dollar(r["best_ev"]),
                _fmt_dollar(r[risk_col]),
                _fmt_dollar(actual),
                _fmt_pct(r["best_deployment_rate"]),
                _fmt_pct(roic),
                _config_summary(r),
            ]
        )

    headers = [
        "L (%)",
        "#Feasible",
        "Best EV ($)",
        "CVaR-5% ($)",
        "Actual ($)",
        "Deploy%",
        "ROIC%",
        "Config summary",
    ]
    align = ["r", "r", "r", "r", "r", "r", "r", "l"]
    _print_md_table(headers, rows, align)


def table_top_ev_configs(cfg: YearConfig, n: int = 15) -> None:
    """Table 15: Top Configs by EV (per-entry normalized)."""
    print(f"\n## 15. Top Configs by EV (Top {n}, {cfg.ceremony_year})\n")

    scores_path = Path(cfg.results_dir) / "portfolio_scores.csv"
    if not scores_path.exists():
        print("_Data not available._\n")
        return

    scores = pd.read_csv(scores_path)

    # Optionally join CVaR data
    cvar_path = Path(cfg.results_dir) / "portfolio_cvar.csv"
    has_cvar = cvar_path.exists()
    if has_cvar:
        cvar = pd.read_csv(cvar_path)
        merge_cols = [c for c in CONFIG_PARAMS if c in scores.columns and c in cvar.columns]
        scores = scores.merge(cvar, on=merge_cols, how="left")

    top = scores.nlargest(n, "ev_pnl_blend").reset_index(drop=True)

    rows = []
    for i, r in top.iterrows():
        bankroll_deployed = r.get("total_bankroll_deployed", 0)
        actual_pnl = r.get("total_pnl", 0)
        roi_pct = (actual_pnl / bankroll_deployed * 100) if bankroll_deployed > 0 else 0.0

        row = [
            str(int(i) + 1),  # type: ignore[arg-type, call-overload]
            _short_model(r["model_type"]),
            _fmt(r["kelly_fraction"]),
            _fmt(r["buy_edge_threshold"]),
            _short_kelly_mode(r.get("kelly_mode", "")),
            _short_side(r.get("allowed_directions", "")),
            _fmt_dollar(r["ev_pnl_blend"]),
            _fmt_dollar(r.get("worst_pnl", 0)),
        ]
        if has_cvar:
            row.append(_fmt_dollar(r.get("cvar_5", 0)))
        row.extend(
            [
                _fmt_dollar(actual_pnl),
                _fmt_pct(r.get("deployment_rate", 0)),
                _fmt_pct(roi_pct),
            ]
        )
        rows.append(row)

    headers = ["#", "Model", "KF", "Edge", "KM", "Side", "EV ($)", "Worst ($)"]
    align = ["r", "l", "r", "r", "l", "l", "r", "r"]
    if has_cvar:
        headers.append("CVaR-5% ($)")
        align.append("r")
    headers.extend(["Actual ($)", "Deploy%", "ROI%"])
    align.extend(["r", "r", "r"])
    _print_md_table(headers, rows, align)


# ============================================================================
# Cross-year tables
# ============================================================================


def _load_both_years_portfolio() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load aggregate P&L for both years and compute portfolio-level P&L."""
    cfg_2024 = YEAR_CONFIGS[2024]
    cfg_2025 = YEAR_CONFIGS[2025]

    agg_2024 = load_aggregate_pnl(cfg_2024)
    agg_2025 = load_aggregate_pnl(cfg_2025)

    port_2024 = _portfolio_pnl(agg_2024)
    port_2025 = _portfolio_pnl(agg_2025)

    return port_2024, port_2025


def _merge_years(port_2024: pd.DataFrame, port_2025: pd.DataFrame) -> pd.DataFrame:
    """Merge two portfolio DataFrames on config columns, suffixed by year."""
    merge_cols = [c for c in CONFIG_PARAMS if c in port_2024.columns and c in port_2025.columns]
    merged = port_2024.merge(
        port_2025,
        on=merge_cols,
        suffixes=("_2024", "_2025"),
        how="inner",
    )
    merged["combined_pnl"] = merged["portfolio_pnl_2024"] + merged["portfolio_pnl_2025"]
    return merged


def table_cross_year_profitability() -> None:
    """Table 8: Cross-Year Config Profitability."""
    print("\n## 8. Cross-Year Config Profitability\n")

    port_2024, port_2025 = _load_both_years_portfolio()
    merged = _merge_years(port_2024, port_2025)

    both = ((merged["portfolio_pnl_2024"] > 0) & (merged["portfolio_pnl_2025"] > 0)).sum()
    only_2024 = ((merged["portfolio_pnl_2024"] > 0) & (merged["portfolio_pnl_2025"] <= 0)).sum()
    only_2025 = ((merged["portfolio_pnl_2024"] <= 0) & (merged["portfolio_pnl_2025"] > 0)).sum()
    neither = ((merged["portfolio_pnl_2024"] <= 0) & (merged["portfolio_pnl_2025"] <= 0)).sum()
    total = len(merged)

    # Overall summary
    rows = [
        ["Both years profitable", str(both), _fmt_pct(both / total * 100)],
        ["2024 only", str(only_2024), _fmt_pct(only_2024 / total * 100)],
        ["2025 only", str(only_2025), _fmt_pct(only_2025 / total * 100)],
        ["Neither", str(neither), _fmt_pct(neither / total * 100)],
        ["**Total**", str(total), "100.0%"],
    ]
    _print_md_table(["Outcome", "Count", "%"], rows, ["l", "r", "r"])

    # Per-model breakdown
    print("\n### Per-Model Breakdown\n")
    model_rows = []
    for model in sorted(merged["model_type"].unique()):
        m = merged[merged["model_type"] == model]
        n = len(m)
        b = ((m["portfolio_pnl_2024"] > 0) & (m["portfolio_pnl_2025"] > 0)).sum()
        model_rows.append(
            [
                _short_model(model),
                str(n),
                str(b),
                _fmt_pct(b / n * 100) if n > 0 else "n/a",
                _fmt_dollar(m["combined_pnl"].mean()),
                _fmt_dollar(m["combined_pnl"].median()),
            ]
        )

    _print_md_table(
        ["Model", "#Cfg", "Both Prof", "Both %", "Mean Combined", "Median Combined"],
        model_rows,
        ["l", "r", "r", "r", "r", "r"],
    )


def table_model_comparison_across_years() -> None:
    """Table 9: Model Comparison Across Years."""
    print("\n## 9. Model Comparison Across Years\n")

    port_2024, port_2025 = _load_both_years_portfolio()
    merged = _merge_years(port_2024, port_2025)

    rows = []
    for model in sorted(merged["model_type"].unique()):
        m = merged[merged["model_type"] == model]
        n = len(m)

        both_rate = ((m["portfolio_pnl_2024"] > 0) & (m["portfolio_pnl_2025"] > 0)).mean() * 100

        # P&L stats
        mean_comb = m["combined_pnl"].mean()
        best_comb = m["combined_pnl"].max()
        mean_2024 = m["portfolio_pnl_2024"].mean()
        mean_2025 = m["portfolio_pnl_2025"].mean()

        # Spearman rank correlation of per-config P&L across years
        if n >= 3:
            rho, _ = stats.spearmanr(m["portfolio_pnl_2024"], m["portfolio_pnl_2025"])
            rho_str = _fmt(float(rho), 3)  # type: ignore[arg-type]  # scipy SpearmanrResult typed as object
        else:
            rho_str = "n/a"

        rows.append(
            [
                _short_model(model),
                _fmt_pct(both_rate),
                _fmt_dollar(mean_comb),
                _fmt_dollar(best_comb),
                _fmt_dollar(mean_2024),
                _fmt_dollar(mean_2025),
                rho_str,
            ]
        )

    headers = [
        "Model",
        "Both %",
        "Mean Combined",
        "Best Combined",
        "Mean 2024",
        "Mean 2025",
        "Spearman ρ",
    ]
    align = ["l", "r", "r", "r", "r", "r", "r"]
    _print_md_table(headers, rows, align)


def table_top_combined_pnl(n: int = 15) -> None:
    """Table 11: Top Configs by Combined P&L."""
    print(f"\n## 11. Top Configs by Combined P&L (Top {n})\n")

    port_2024, port_2025 = _load_both_years_portfolio()
    merged = _merge_years(port_2024, port_2025)
    top = merged.nlargest(n, "combined_pnl").reset_index(drop=True)

    rows = []
    for i, r in top.iterrows():
        rows.append(
            [
                str(int(i) + 1),  # type: ignore[arg-type, call-overload]
                _short_model(r["model_type"]),
                _fmt(r["kelly_fraction"]),
                _fmt(r.get("buy_edge_threshold", 0)),
                _short_kelly_mode(r.get("kelly_mode", "")),
                _short_side(r.get("allowed_directions", "")),
                r.get("fee_type", ""),
                _fmt_dollar(r["portfolio_pnl_2024"]),
                _fmt_dollar(r["portfolio_pnl_2025"]),
                _fmt_dollar(r["combined_pnl"]),
            ]
        )

    headers = [
        "#",
        "Model",
        "KF",
        "Edge",
        "KM",
        "Side",
        "Fee",
        "P&L 2024",
        "P&L 2025",
        "Combined",
    ]
    align = ["r", "l", "r", "r", "l", "l", "l", "r", "r", "r"]
    _print_md_table(headers, rows, align)


def table_cross_year_rank_correlation() -> None:
    """Table 12: Cross-Year Rank Correlation (Spearman) per model."""
    print("\n## 12. Cross-Year Rank Correlation (Spearman ρ)\n")

    port_2024, port_2025 = _load_both_years_portfolio()
    merged = _merge_years(port_2024, port_2025)

    rows = []
    for model in sorted(merged["model_type"].unique()):
        m = merged[merged["model_type"] == model]
        n = len(m)
        if n >= 3:
            rho, pval = stats.spearmanr(m["portfolio_pnl_2024"], m["portfolio_pnl_2025"])
            rho_f, pval_f = float(rho), float(pval)  # type: ignore[arg-type]  # scipy SpearmanrResult typed as object
            rows.append(
                [
                    _short_model(model),
                    str(n),
                    _fmt(rho_f, 4),
                    f"{pval_f:.2e}" if pval_f < 0.001 else _fmt(pval_f, 4),
                ]
            )
        else:
            rows.append([_short_model(model), str(n), "n/a", "n/a"])

    # Also add an overall row
    if len(merged) >= 3:
        rho, pval = stats.spearmanr(merged["portfolio_pnl_2024"], merged["portfolio_pnl_2025"])
        rho_f, pval_f = float(rho), float(pval)
        rows.append(
            [
                "**ALL**",
                str(len(merged)),
                _fmt(rho_f, 4),
                f"{pval_f:.2e}" if pval_f < 0.001 else _fmt(pval_f, 4),
            ]
        )

    headers = ["Model", "#Cfg", "Spearman ρ", "p-value"]
    align = ["l", "r", "r", "r"]
    _print_md_table(headers, rows, align)


def _load_portfolio_scores_for_years() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load portfolio_scores.csv for 2024 and 2025, keeping deploy/ROIC columns."""
    cols = [
        "model_type",
        "config_label",
        "capital_deployed",
        "deployment_rate",
        "total_pnl",
        "roic",
    ]
    scores_2024 = pd.read_csv(
        YEAR_CONFIGS[2024].results_dir / "portfolio_scores.csv",
        usecols=cols,
    )
    scores_2025 = pd.read_csv(
        YEAR_CONFIGS[2025].results_dir / "portfolio_scores.csv",
        usecols=cols,
    )
    return scores_2024, scores_2025


def _enrich_pareto_with_deploy(
    df: pd.DataFrame,
    scores_2024: pd.DataFrame,
    scores_2025: pd.DataFrame,
) -> pd.DataFrame:
    """Join capital deployment and ROIC from portfolio_scores onto a Pareto DF.

    Pareto CSVs use ``best_model`` / ``best_config``; portfolio_scores uses
    ``model_type`` / ``config_label``.

    Adds per-year columns: deploy_YYYY (capital_deployed), deploy_pct_YYYY
    (deployment_rate), actual_YYYY (total_pnl), roic_YYYY.
    """
    df = df.merge(
        scores_2024.rename(
            columns={
                "capital_deployed": "deploy_2024",
                "deployment_rate": "deploy_pct_2024",
                "total_pnl": "actual_2024",
                "roic": "roic_2024",
            }
        ),
        left_on=["best_model", "best_config"],
        right_on=["model_type", "config_label"],
        how="left",
    ).drop(columns=["model_type", "config_label"])
    df = df.merge(
        scores_2025.rename(
            columns={
                "capital_deployed": "deploy_2025",
                "deployment_rate": "deploy_pct_2025",
                "total_pnl": "actual_2025",
                "roic": "roic_2025",
            }
        ),
        left_on=["best_model", "best_config"],
        right_on=["model_type", "config_label"],
        how="left",
    ).drop(columns=["model_type", "config_label"])
    return df


def _load_cross_year_scores() -> pd.DataFrame | None:
    """Load cross_year_scenario_scores.csv if available.

    Handles column naming compatibility: the scoring code produces ``total_pnl``
    columns (e.g. ``total_pnl_2024``, ``avg_total_pnl``) but the extended
    analysis tables expect the legacy ``actual_pnl`` naming. This function
    normalizes by creating ``actual_pnl`` aliases if they are missing.
    """
    path = Path(BUY_HOLD_EXP_DIR) / "cross_year_scenario_scores.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)

    # Normalize: total_pnl -> actual_pnl (alias for backwards compatibility)
    renames = {}
    for col in list(df.columns):
        if "total_pnl" in col and col.replace("total_pnl", "actual_pnl") not in df.columns:
            renames[col] = col.replace("total_pnl", "actual_pnl")
    if renames:
        df = df.rename(columns=renames)

    return df


def _compute_pct_profitable(
    cross_scores: pd.DataFrame,
    risk_col: str,
    loss_bound_frac: float,
) -> float:
    """% of feasible configs that are profitable in both years.

    Args:
        cross_scores: cross_year_scenario_scores.csv DataFrame.
        risk_col: Risk column stem (e.g., ``worst_pnl`` or ``cvar_5``).
        loss_bound_frac: Loss bound as a fraction (e.g., 0.10 for 10%).

    Returns:
        Percentage (0-100) of feasible configs profitable in both years.
    """
    col_2024 = f"{risk_col}_2024"
    col_2025 = f"{risk_col}_2025"

    bankroll_2024 = cross_scores["total_bankroll_deployed_2024"].iloc[0]
    bankroll_2025 = cross_scores["total_bankroll_deployed_2025"].iloc[0]

    feasible = cross_scores[
        (cross_scores[col_2024] >= -loss_bound_frac * bankroll_2024)
        & (cross_scores[col_2025] >= -loss_bound_frac * bankroll_2025)
    ]
    if len(feasible) == 0:
        return 0.0
    return (feasible["profitable_both"].sum() / len(feasible)) * 100


def table_cross_year_pareto() -> None:
    """Table 16: Cross-Year Pareto Comparison (worst-case constraint)."""
    print("\n## 16. Cross-Year Pareto Comparison\n")

    # Prefer unified cvar00 naming; fall back to legacy name
    path = Path(BUY_HOLD_EXP_DIR) / "cross_year_pareto_cvar00_blend.csv"
    if not path.exists():
        path = Path(BUY_HOLD_EXP_DIR) / "cross_year_pareto_blend.csv"
    if not path.exists():
        print("_Data not available._\n")
        return

    df = pd.read_csv(path)

    # Support both old and new column names
    risk_key_2024 = "best_risk_2024" if "best_risk_2024" in df.columns else "best_worst_pnl_2024"
    risk_key_2025 = "best_risk_2025" if "best_risk_2025" in df.columns else "best_worst_pnl_2025"

    # Enrich with deployment and ROIC from per-year portfolio scores
    scores_2024, scores_2025 = _load_portfolio_scores_for_years()
    df = _enrich_pareto_with_deploy(df, scores_2024, scores_2025)

    # Load cross-year scores for % profitable
    cross_scores = _load_cross_year_scores()

    rows = []
    for _, r in df.iterrows():
        deploy_str_2024 = f"${r.get('deploy_2024', 0):,.0f} ({r.get('deploy_pct_2024', 0):.0f}%)"
        deploy_str_2025 = f"${r.get('deploy_2025', 0):,.0f} ({r.get('deploy_pct_2025', 0):.0f}%)"

        pct_prof = ""
        if cross_scores is not None:
            L = r["loss_bound_pct"] / 100
            pct_prof = _fmt_pct(_compute_pct_profitable(cross_scores, "worst_pnl", L))

        rows.append(
            [
                _fmt(r["loss_bound_pct"], 0),
                str(int(r["n_feasible"])),
                _fmt_dollar(r["best_avg_ev"]),
                _fmt_dollar(r[risk_key_2024]),
                _fmt_dollar(r[risk_key_2025]),
                _fmt_dollar(r["best_total_pnl_2024"]),
                _fmt_dollar(r["best_total_pnl_2025"]),
                deploy_str_2024,
                deploy_str_2025,
                _fmt_pct(r.get("roic_2024", 0)),
                _fmt_pct(r.get("roic_2025", 0)),
                pct_prof,
                _config_summary(r),
            ]
        )

    headers = [
        "L (%)",
        "#Feasible",
        "Avg EV ($)",
        "Worst 2024 ($)",
        "Worst 2025 ($)",
        "Actual 2024 ($)",
        "Actual 2025 ($)",
        "Deploy 2024",
        "Deploy 2025",
        "ROIC 2024",
        "ROIC 2025",
        "% Profitable",
        "Config",
    ]
    align = ["r"] * 12 + ["l"]
    _print_md_table(headers, rows, align)


def table_cross_year_cvar_pareto() -> None:
    """Table 17: Cross-Year CVaR Pareto."""
    print("\n## 17. Cross-Year CVaR Pareto\n")

    path = Path(BUY_HOLD_EXP_DIR) / "cross_year_pareto_cvar05_blend.csv"
    if not path.exists():
        print("_Data not available._\n")
        return

    df = pd.read_csv(path)

    # Support both old and new column names
    risk_key_2024 = "best_risk_2024" if "best_risk_2024" in df.columns else "best_cvar_2024"
    risk_key_2025 = "best_risk_2025" if "best_risk_2025" in df.columns else "best_cvar_2025"

    # Enrich with deployment and ROIC from per-year portfolio scores
    scores_2024, scores_2025 = _load_portfolio_scores_for_years()
    df = _enrich_pareto_with_deploy(df, scores_2024, scores_2025)

    # Load cross-year scores for % profitable
    cross_scores = _load_cross_year_scores()

    rows = []
    for _, r in df.iterrows():
        deploy_str_2024 = f"${r.get('deploy_2024', 0):,.0f} ({r.get('deploy_pct_2024', 0):.0f}%)"
        deploy_str_2025 = f"${r.get('deploy_2025', 0):,.0f} ({r.get('deploy_pct_2025', 0):.0f}%)"

        pct_prof = ""
        if cross_scores is not None:
            L = r["loss_bound_pct"] / 100
            pct_prof = _fmt_pct(_compute_pct_profitable(cross_scores, "cvar_5", L))

        rows.append(
            [
                _fmt(r["loss_bound_pct"], 0),
                str(int(r["n_feasible"])),
                _fmt_dollar(r["best_avg_ev"]),
                _fmt_dollar(r[risk_key_2024]),
                _fmt_dollar(r[risk_key_2025]),
                _fmt_dollar(r["best_total_pnl_2024"]),
                _fmt_dollar(r["best_total_pnl_2025"]),
                deploy_str_2024,
                deploy_str_2025,
                _fmt_pct(r.get("roic_2024", 0)),
                _fmt_pct(r.get("roic_2025", 0)),
                pct_prof,
                _config_summary(r),
            ]
        )

    headers = [
        "L (%)",
        "#Feasible",
        "Avg EV ($)",
        "CVaR-5% 2024 ($)",
        "CVaR-5% 2025 ($)",
        "Actual 2024 ($)",
        "Actual 2025 ($)",
        "Deploy 2024",
        "Deploy 2025",
        "ROIC 2024",
        "ROIC 2025",
        "% Profitable",
        "Config",
    ]
    align = ["r"] * 12 + ["l"]
    _print_md_table(headers, rows, align)


# ============================================================================
# Extended cross-year analysis tables
# ============================================================================

#: Loss bounds for extended per-model Pareto analysis.
_EXT_LOSS_BOUNDS = [0.10, 0.20, 0.30, 0.40, 0.50]


def table_ev_inflation(scores_df: pd.DataFrame) -> None:
    """Table E1: EV Inflation — ratio of mean EV to mean actual PnL.

    High inflation means EV systematically overpredicts realized returns.
    Computed per-model and per-year as ``mean(ev_pnl_blend) / mean(actual_pnl)``.
    """
    print("\n## E1. EV Inflation (EV / Actual PnL Ratio)\n")

    rows: list[list[str]] = []
    for model in sorted(scores_df["model_type"].unique()):
        m = scores_df[scores_df["model_type"] == model]

        # Per-year inflation
        mean_ev_2024 = m["ev_pnl_blend_2024"].mean()
        mean_actual_2024 = m["actual_pnl_2024"].mean()
        infl_2024 = mean_ev_2024 / mean_actual_2024 if mean_actual_2024 != 0 else float("inf")

        mean_ev_2025 = m["ev_pnl_blend_2025"].mean()
        mean_actual_2025 = m["actual_pnl_2025"].mean()
        infl_2025 = mean_ev_2025 / mean_actual_2025 if mean_actual_2025 != 0 else float("inf")

        # Overall inflation (from cross-year averages)
        mean_ev_avg = m["avg_ev_pnl_blend"].mean()
        mean_actual_avg = m["avg_actual_pnl"].mean()
        infl_overall = mean_ev_avg / mean_actual_avg if mean_actual_avg != 0 else float("inf")

        rows.append(
            [
                _short_model(model),
                _fmt(infl_2024, 2) + "x",
                _fmt(infl_2025, 2) + "x",
                _fmt(infl_overall, 2) + "x",
                _fmt_dollar(mean_ev_avg),
                _fmt_dollar(mean_actual_avg),
            ]
        )

    # Sort by overall inflation ascending
    def _parse_inflation(row: list[str]) -> float:
        s = row[3].rstrip("x").replace(",", "")
        try:
            return float(s)
        except ValueError:
            return float("inf")

    rows.sort(key=_parse_inflation)

    headers = ["Model", "2024 Infl", "2025 Infl", "Overall Infl", "Mean EV", "Mean Actual"]
    align = ["l", "r", "r", "r", "r", "r"]
    _print_md_table(headers, rows, align)


def table_per_model_pareto(scores_df: pd.DataFrame, alpha: float = 0.05) -> None:
    """Table E2: Per-Model Pareto — best EV config at each loss bound, per model.

    For each model_type separately, finds the config with max avg_ev_pnl_blend
    such that avg_cvar_5 >= -L * bankroll, where bankroll is the average of the
    per-year deployed bankrolls (approx 8500).

    Includes a summary table comparing the best config from each model at L=20%.
    """
    print("\n## E2. Per-Model Pareto Frontier (CVaR-5% Constraint)\n")

    avg_bankroll = (
        scores_df["total_bankroll_deployed_2024"].iloc[0]
        + scores_df["total_bankroll_deployed_2025"].iloc[0]
    ) / 2  # ~8500

    summary_at_20: list[list[str]] = []

    for model in sorted(scores_df["model_type"].unique()):
        m = scores_df[scores_df["model_type"] == model]
        print(f"\n### {_short_model(model)} ({len(m)} configs)\n")

        rows: list[list[str]] = []
        for L in _EXT_LOSS_BOUNDS:
            max_loss = L * avg_bankroll
            feasible = m[m["avg_cvar_5"] >= -max_loss]

            if feasible.empty:
                rows.append(
                    [_fmt(L * 100, 0), "0", "\u2014", "\u2014", "\u2014", "\u2014", "\u2014"]
                )
                continue

            best_idx = feasible["avg_ev_pnl_blend"].idxmax()
            best = feasible.loc[best_idx]

            ev = best["avg_ev_pnl_blend"]
            actual = best["avg_actual_pnl"]
            infl = ev / actual if actual != 0 else float("inf")

            rows.append(
                [
                    _fmt(L * 100, 0),
                    str(len(feasible)),
                    _fmt_dollar(ev),
                    _fmt_dollar(actual),
                    _fmt_dollar(best["avg_cvar_5"]),
                    _fmt(infl, 2) + "x",
                    str(best["config_label"])[:50],
                ]
            )

            if abs(L - 0.20) < 1e-9:
                summary_at_20.append(
                    [
                        _short_model(model),
                        str(len(feasible)),
                        _fmt_dollar(ev),
                        _fmt_dollar(actual),
                        _fmt(infl, 2) + "x",
                    ]
                )

        headers = [
            "L (%)",
            "#Feasible",
            "Best EV ($)",
            "Actual ($)",
            "CVaR-5% ($)",
            "Inflation",
            "Config",
        ]
        align = ["r", "r", "r", "r", "r", "r", "l"]
        _print_md_table(headers, rows, align)

    # Summary at L=20%
    if summary_at_20:
        print("\n### Summary: Best Config per Model at L=20%\n")
        _print_md_table(
            ["Model", "#Feasible", "Best EV ($)", "Actual ($)", "Inflation"],
            summary_at_20,
            ["l", "r", "r", "r", "r"],
        )


def table_cross_model_ev_correlation(scores_df: pd.DataFrame) -> None:
    """Table E3: EV-Actual Correlation — Spearman rank correlation and EV-best capture.

    Compares cross-model (all 3528 configs pooled) vs within-model Spearman rho
    between avg_ev_pnl_blend and avg_actual_pnl.

    Also reports "EV-best actual capture": the actual PnL of the EV-best config
    as a percentage of the actual-best config's PnL within the same model.
    """
    print("\n## E3. EV vs Actual PnL Correlation (Spearman)\n")

    rows: list[list[str]] = []

    # Per model
    for model in sorted(scores_df["model_type"].unique()):
        m = scores_df[scores_df["model_type"] == model]
        n = len(m)

        if n >= 3:
            rho, pval = stats.spearmanr(m["avg_ev_pnl_blend"], m["avg_actual_pnl"])
            rho_f = float(rho)  # type: ignore[arg-type]  # scipy SpearmanrResult
            pval_f = float(pval)  # type: ignore[arg-type]  # scipy SpearmanrResult
        else:
            rho_f, pval_f = float("nan"), float("nan")

        # EV-best actual capture: actual_pnl of EV-best / actual_pnl of actual-best
        ev_best_idx = m["avg_ev_pnl_blend"].idxmax()
        actual_best_idx = m["avg_actual_pnl"].idxmax()
        ev_best_actual = float(m.loc[ev_best_idx, "avg_actual_pnl"])
        actual_best_actual = float(m.loc[actual_best_idx, "avg_actual_pnl"])
        capture = ev_best_actual / actual_best_actual * 100 if actual_best_actual != 0 else 0.0

        rho_str = _fmt(rho_f, 4) if not pd.isna(rho_f) else "n/a"
        if pd.isna(pval_f):
            pval_str = "n/a"
        elif pval_f < 0.001:
            pval_str = f"{pval_f:.2e}"
        else:
            pval_str = _fmt(pval_f, 4)

        rows.append(
            [
                _short_model(model),
                str(n),
                rho_str,
                pval_str,
                _fmt_pct(capture),
                _fmt_dollar(ev_best_actual),
                _fmt_dollar(actual_best_actual),
            ]
        )

    # Overall (cross-model)
    n_all = len(scores_df)
    rho_all, pval_all = stats.spearmanr(scores_df["avg_ev_pnl_blend"], scores_df["avg_actual_pnl"])
    rho_all_f = float(rho_all)  # type: ignore[arg-type]  # scipy SpearmanrResult
    pval_all_f = float(pval_all)  # type: ignore[arg-type]  # scipy SpearmanrResult

    ev_best_idx_all = scores_df["avg_ev_pnl_blend"].idxmax()
    actual_best_idx_all = scores_df["avg_actual_pnl"].idxmax()
    ev_best_actual_all = float(scores_df.loc[ev_best_idx_all, "avg_actual_pnl"])  # type: ignore[arg-type]  # pandas scalar
    actual_best_actual_all = float(scores_df.loc[actual_best_idx_all, "avg_actual_pnl"])  # type: ignore[arg-type]  # pandas scalar
    capture_all = (
        ev_best_actual_all / actual_best_actual_all * 100 if actual_best_actual_all != 0 else 0.0
    )

    pval_all_str = f"{pval_all_f:.2e}" if pval_all_f < 0.001 else _fmt(pval_all_f, 4)
    rows.append(
        [
            "**ALL**",
            str(n_all),
            _fmt(rho_all_f, 4),
            pval_all_str,
            _fmt_pct(capture_all),
            _fmt_dollar(ev_best_actual_all),
            _fmt_dollar(actual_best_actual_all),
        ]
    )

    headers = [
        "Model",
        "#Cfg",
        "Spearman \u03c1",
        "p-value",
        "EV-Best Capture %",
        "EV-Best Actual",
        "Best Actual",
    ]
    align = ["l", "r", "r", "r", "r", "r", "r"]
    _print_md_table(headers, rows, align)


def _load_category_pnl_matrix(
    year: int,
) -> tuple[np.ndarray, list[str], dict[tuple[str, str], int]]:
    """Load entry_pnl for a year and build a (n_configs, n_categories) PnL matrix.

    Returns:
        matrix: (n_configs, n_categories) array of total actual PnL per (config, cat).
        categories: Sorted list of category names.
        config_to_idx: Mapping from (model_type, config_label) to row index.
    """
    cfg = YEAR_CONFIGS[year]
    entry_df = load_entry_pnl(cfg)
    pnl_col = "actual_pnl" if "actual_pnl" in entry_df.columns else "total_pnl"

    cat_pnl = entry_df.groupby(["config_label", "model_type", "category"], as_index=False)[
        pnl_col
    ].sum()
    cat_pnl = cat_pnl.rename(columns={pnl_col: "category_pnl"})  # type: ignore[call-overload]  # pandas groupby sum returns DataFrame

    categories = sorted(cat_pnl["category"].unique())
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    config_keys = (
        cat_pnl[["model_type", "config_label"]]
        .drop_duplicates()
        .sort_values(["model_type", "config_label"])
        .reset_index(drop=True)
    )
    config_to_idx: dict[tuple[str, str], int] = {
        (row["model_type"], row["config_label"]): i for i, row in config_keys.iterrows()
    }

    n_configs = len(config_to_idx)
    n_cats = len(categories)
    matrix = np.zeros((n_configs, n_cats))

    for _, row in cat_pnl.iterrows():
        key = (str(row["model_type"]), str(row["config_label"]))
        if key in config_to_idx:
            ci = config_to_idx[key]
            cat_i = cat_to_idx[str(row["category"])]
            matrix[ci, cat_i] = row["category_pnl"]

    return matrix, categories, config_to_idx


def table_bootstrap_rank_stability(scores_df: pd.DataFrame, n_bootstrap: int = 1000) -> None:
    """Table E4: Bootstrap Rank Stability under category resampling.

    For each year, draws K categories with replacement (K = number of
    categories in that year), sums per-category actual PnL to get a
    bootstrap portfolio PnL, then ranks all configs.  Reports what
    fraction of bootstrap samples each config appears in top-10/25/50.

    Configs reported: EV-best per model + top-5 by actual PnL.
    """
    print(f"\n## E4. Bootstrap Rank Stability ({n_bootstrap:,} samples)\n")

    # Build config index aligned with scores_df
    configs = scores_df[["model_type", "config_label"]].drop_duplicates().reset_index(drop=True)
    scores_config_to_idx: dict[tuple[str, str], int] = {
        (row["model_type"], row["config_label"]): int(i)  # type: ignore[call-overload]  # iterrows index is int
        for i, row in configs.iterrows()
    }
    n_configs = len(configs)

    # Load per-category PnL matrices for each year
    year_data: dict[int, tuple[np.ndarray, int]] = {}  # year -> (matrix aligned, n_cats)
    for year in [2024, 2025]:
        raw_matrix, categories, raw_config_idx = _load_category_pnl_matrix(year)
        n_cats = len(categories)

        # Align to scores_df config order
        aligned = np.zeros((n_configs, n_cats))
        for key, raw_i in raw_config_idx.items():
            if key in scores_config_to_idx:
                aligned[scores_config_to_idx[key]] = raw_matrix[raw_i]
        year_data[year] = (aligned, n_cats)

    rng = np.random.default_rng(42)

    # Bootstrap: resample categories, sum PnL, rank
    bootstrap_rankings = np.zeros((n_bootstrap, n_configs), dtype=np.intp)
    for b in range(n_bootstrap):
        total_pnl = np.zeros(n_configs)
        for year in [2024, 2025]:
            matrix, n_cats = year_data[year]
            indices = rng.integers(0, n_cats, size=n_cats)
            total_pnl += matrix[:, indices].sum(axis=1)

        order = np.argsort(-total_pnl)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, n_configs + 1)
        bootstrap_rankings[b] = ranks

    # Identify interesting configs
    report_configs: list[tuple[str, pd.Series]] = []

    # EV-best per model
    for model in sorted(scores_df["model_type"].unique()):
        m = scores_df[scores_df["model_type"] == model]
        ev_best_idx = m["avg_ev_pnl_blend"].idxmax()
        report_configs.append((f"EV-best {_short_model(model)}", scores_df.loc[ev_best_idx]))

    # Top-5 by actual PnL
    top5_actual = scores_df.nlargest(5, "avg_actual_pnl")
    for i, (_, row) in enumerate(top5_actual.iterrows()):
        report_configs.append((f"Actual #{i + 1}", row))

    # Build table rows
    rows: list[list[str]] = []
    for label, config_row in report_configs:
        key = (config_row["model_type"], config_row["config_label"])
        if key not in scores_config_to_idx:
            continue
        ci = scores_config_to_idx[key]
        ranks = bootstrap_rankings[:, ci]

        pct_top10 = (ranks <= 10).mean() * 100
        pct_top25 = (ranks <= 25).mean() * 100
        pct_top50 = (ranks <= 50).mean() * 100
        median_rank = float(np.median(ranks))

        rows.append(
            [
                label,
                _short_model(config_row["model_type"]),
                _fmt_dollar(config_row["avg_ev_pnl_blend"]),
                _fmt_dollar(config_row["avg_actual_pnl"]),
                _fmt(median_rank, 0),
                _fmt_pct(pct_top10),
                _fmt_pct(pct_top25),
                _fmt_pct(pct_top50),
            ]
        )

    headers = [
        "Config",
        "Model",
        "Avg EV",
        "Avg Actual",
        "Med Rank",
        "Top-10 %",
        "Top-25 %",
        "Top-50 %",
    ]
    align = ["l", "l", "r", "r", "r", "r", "r", "r"]
    _print_md_table(headers, rows, align)


# ============================================================================
# Orchestration
# ============================================================================


def run_per_year(ceremony_year: int) -> None:
    """Generate all per-year tables for a given ceremony year."""
    cfg = YEAR_CONFIGS[ceremony_year]
    print(f"# Buy-and-Hold Backtest Tables — {ceremony_year}")

    table_portfolio_summary(cfg)
    table_top_portfolio_pnl(cfg)
    table_parameter_sensitivity(cfg)
    table_entry_timing(cfg)
    table_category_model_matrix(cfg)
    table_capital_utilization(cfg)
    table_pareto_frontier(cfg)
    table_cvar_pareto(cfg)
    table_top_ev_configs(cfg)


def run_cross_year(*, extended: bool = False) -> None:
    """Generate all cross-year comparison tables.

    Args:
        extended: If True, also generate extended analysis tables
            (EV inflation, per-model Pareto, EV correlation,
            bootstrap rank stability). These can be slow.
    """
    print("# Cross-Year Comparison Tables")

    table_cross_year_profitability()
    table_model_comparison_across_years()
    table_top_combined_pnl()
    table_cross_year_rank_correlation()

    # Pareto tables may fail if portfolio_scores.csv schema differs from expected
    try:
        table_cross_year_pareto()
    except Exception as exc:  # noqa: BLE001
        print(f"\n_Table 16 skipped: {exc}_\n")
    try:
        table_cross_year_cvar_pareto()
    except Exception as exc:  # noqa: BLE001
        print(f"\n_Table 17 skipped: {exc}_\n")

    if extended:
        scores_df = _load_cross_year_scores()
        if scores_df is not None:
            print("\n# Extended Cross-Year Analysis")
            table_ev_inflation(scores_df)
            table_per_model_pareto(scores_df)
            table_cross_model_ev_correlation(scores_df)
            table_bootstrap_rank_stability(scores_df)
        else:
            print("\n_Extended tables skipped: cross_year_scenario_scores.csv not found._")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate markdown tables for buy-and-hold backtest READMEs."
    )
    parser.add_argument(
        "--ceremony-year",
        type=int,
        choices=sorted(YEAR_CONFIGS.keys()),
        help="Ceremony year for per-year tables.",
    )
    parser.add_argument(
        "--mode",
        choices=["per-year", "cross-year"],
        default="per-year",
        help="Table mode: per-year (default) or cross-year.",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Generate extended analysis tables (EV inflation, per-model Pareto, "
        "EV correlation, bootstrap rank stability). Can be slow.",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default=None,
        help=(
            "Override experiment directory for input data. "
            "Reads per-year results from <exp-dir>/<year>/results/ and "
            "cross-year CSVs from <exp-dir>/. "
            "Default: storage/d20260225_buy_hold_backtest."
        ),
    )
    args = parser.parse_args()

    # Patch module-level paths if --exp-dir is set, so all data-loading
    # functions read from the overridden directory.
    if args.exp_dir:
        import oscar_prediction_market.one_offs.d20260225_buy_hold_backtest.year_config as _yc

        exp_dir = Path(args.exp_dir)
        _yc.BUY_HOLD_EXP_DIR = exp_dir  # type: ignore[attr-defined]  # runtime override
        globals()["BUY_HOLD_EXP_DIR"] = exp_dir

    if args.mode == "cross-year":
        run_cross_year(extended=args.extended)
    else:
        if args.ceremony_year is None:
            parser.error("--ceremony-year is required for per-year mode.")
        run_per_year(args.ceremony_year)


if __name__ == "__main__":
    main()
