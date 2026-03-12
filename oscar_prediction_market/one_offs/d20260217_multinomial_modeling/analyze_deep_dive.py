"""Deep-dive analysis for multinomial modeling experiment.

Extends the basic CV analysis with:
1.  Model-vs-market comparison -- per-nominee model prob vs Kalshi market price
2.  Divergence heatmaps -- model-market gap per nominee x date, per model type
3.  Probability distribution anatomy -- how prob mass is spread across nominees
4.  Brier decomposition -- reliability + resolution per model type
5.  Rank agreement analysis -- do models agree on ordering? Spearman rho
6.  Feature importance evolution -- heatmap of top features across snapshots
7.  Per-trade backtest deep dive -- trade-by-trade replay for selected models
8.  Settlement scenario heatmap -- refined per-model x winner matrix
9.  Wealth & edge comparison -- model-specific wealth curves with trade annotations
10. Probability concentration -- entropy + Herfindahl + top-1/top-3 share over time

Usage:
    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\
        d20260217_multinomial_modeling.analyze_deep_dive \
        --exp-dir storage/d20260217_multinomial_modeling \
        --output-dir storage/d20260217_multinomial_modeling/deep_dive
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore[import-untyped]

from oscar_prediction_market.one_offs.analysis_utils.data_loading import (
    build_model_market_df,
    load_all_test_predictions,
    load_backtest_results,
    load_cv_metrics,
    load_cv_predictions,
    load_feature_importances,
    load_market_prices,
)
from oscar_prediction_market.one_offs.analysis_utils.plot_calibration import (
    plot_reliability_diagrams,
)
from oscar_prediction_market.one_offs.analysis_utils.plot_temporal import (
    plot_feature_importance_heatmap,
    plot_metrics_over_time_detailed,
)
from oscar_prediction_market.one_offs.analysis_utils.style import (
    AWARDS_SEASON_EVENTS,
    MODEL_DISPLAY,
    apply_style,
    get_model_display,
)
from oscar_prediction_market.one_offs.d20260217_multinomial_modeling.metrics import (
    extract_key_metrics,
)
from oscar_prediction_market.one_offs.d20260217_multinomial_modeling.plot_comparison import (
    plot_binary_vs_multinomial,
    plot_prob_distribution_anatomy,
)
from oscar_prediction_market.one_offs.d20260217_multinomial_modeling.plot_trading import (
    plot_divergence_heatmaps,
    plot_edge_distributions,
    plot_model_vs_market,
    plot_position_evolution,
    plot_settlement_heatmap,
    plot_wealth_curves,
)

apply_style()

# ============================================================================
# Constants
# ============================================================================

MODEL_TYPES = ["lr", "gbt", "conditional_logit", "softmax_gbt", "calibrated_softmax_gbt"]


# ============================================================================
# Unique analysis functions (not in shared module)
# ============================================================================


def analyze_rank_agreement(preds_df: pd.DataFrame, output_dir: Path) -> str:
    """Pairwise rank agreement between all model types at each snapshot.

    Returns markdown table for README.
    """
    test = preds_df[preds_df["year"] == 2026].copy()
    if test.empty:
        return ""

    dates = sorted(test["snapshot_date"].unique())

    pair_rhos: dict[tuple[str, str], list[float]] = {}
    for i, m1 in enumerate(MODEL_TYPES):
        for m2 in MODEL_TYPES[i + 1 :]:
            pair_rhos[(m1, m2)] = []

    for d in dates:
        snap = test[test["snapshot_date"] == d]
        for i, m1 in enumerate(MODEL_TYPES):
            for m2 in MODEL_TYPES[i + 1 :]:
                df1 = snap[snap["model_type"] == m1].sort_values("title")
                df2 = snap[snap["model_type"] == m2].sort_values("title")
                common = set(df1["title"]) & set(df2["title"])
                if len(common) < 3:
                    continue
                df1c = df1[df1["title"].isin(common)].sort_values("title")
                df2c = df2[df2["title"].isin(common)].sort_values("title")
                r = stats.spearmanr(df1c["probability"].values, df2c["probability"].values)
                pair_rhos[(m1, m2)].append(float(r.statistic))  # type: ignore[arg-type]

    lines = ["| Model Pair | Mean Spearman rho | Min | Max |"]
    lines.append("|------------|-------------------|-----|-----|")
    for (m1, m2), rhos in sorted(pair_rhos.items()):
        if rhos:
            lines.append(
                f"| {get_model_display(m1)} <-> {get_model_display(m2)} "
                f"| {np.mean(rhos):.3f} | {np.min(rhos):.3f} | {np.max(rhos):.3f} |"
            )
    table = "\n".join(lines)
    print("\n  Rank Agreement (pairwise Spearman rho):")
    print(table)
    return table


def analyze_backtest_trades(bt_results: dict, output_dir: Path) -> str:
    """Analyze per-model trading behavior from backtest results.

    Returns markdown with trade counts, position sizes, and wealth curves comparison.
    """
    backtests = bt_results.get("backtests", {})
    config = bt_results.get("config", {})
    initial_bankroll = config.get("bankroll_dollars", 1000)

    lines = [
        "| Model | Mode | Final Wealth | PnL | Return | Trades | Fees | Open Contracts | Open MtM |"
    ]
    lines.append(
        "|-------|------|-------------|-----|--------|--------|------|----------------|----------|"
    )

    for run_key in sorted(backtests.keys()):
        snaps = backtests[run_key]["snapshots"]
        if not snaps:
            continue
        last = snaps[-1]
        model = last["model_type"]
        mode = last["bankroll_mode"]
        wealth = last["total_wealth"]
        pnl = wealth - initial_bankroll
        ret = pnl / initial_bankroll * 100
        trades = last["total_trades"]
        fees = last["total_fees_paid"]
        open_contracts = sum(p["contracts"] for p in last["positions"].values())
        mtm = last["mark_to_market_value"]

        lines.append(
            f"| {MODEL_DISPLAY.get(model, model)} | {mode} | ${wealth:.2f} "
            f"| ${pnl:+.2f} | {ret:+.1f}% | {trades} | ${fees:.2f} "
            f"| {open_contracts} | ${mtm:.2f} |"
        )

    table = "\n".join(lines)
    print("\n  Backtest Trade Summary:")
    print(table)
    return table


def analyze_prob_concentration(preds_df: pd.DataFrame, output_dir: Path) -> str:
    """Probability concentration: entropy, Herfindahl, top-1/3 share.

    Returns markdown table for README.
    """
    test = preds_df[preds_df["year"] == 2026].copy()
    if test.empty:
        return ""

    lines = ["| Model | Entropy (bits) | Herfindahl | Top-1 Share | Top-3 Share | Prob Sum |"]
    lines.append("|-------|----------------|------------|-------------|-------------|----------|")

    for mt in MODEL_TYPES:
        mt_data = test[test["model_type"] == mt]
        dates = sorted(mt_data["snapshot_date"].unique())

        all_entropy: list[float] = []
        all_herf: list[float] = []
        all_top1: list[float] = []
        all_top3: list[float] = []
        all_psum: list[float] = []

        for d in dates:
            snap = mt_data[mt_data["snapshot_date"] == d]
            probs = snap["probability"].values
            psum = float(probs.sum())
            all_psum.append(psum)
            p_norm = probs / psum if psum > 0 else probs
            p_pos = p_norm[p_norm > 0]

            entropy = float(-np.sum(p_pos * np.log2(p_pos)))
            all_entropy.append(entropy)
            all_herf.append(float(np.sum(p_norm**2)))
            sorted_p = np.sort(p_norm)[::-1]
            all_top1.append(float(sorted_p[0]) if len(sorted_p) > 0 else 0.0)
            all_top3.append(
                float(sorted_p[:3].sum()) if len(sorted_p) >= 3 else float(sorted_p.sum())
            )

        lines.append(
            f"| {get_model_display(mt)} | {np.mean(all_entropy):.2f} "
            f"| {np.mean(all_herf):.3f} "
            f"| {np.mean(all_top1):.1%} "
            f"| {np.mean(all_top3):.1%} "
            f"| {np.mean(all_psum):.2f} +/- {np.std(all_psum):.2f} |"
        )

    table = "\n".join(lines)
    print("\n  Probability Concentration:")
    print(table)
    return table


def analyze_top_pick_agreement(preds_df: pd.DataFrame, output_dir: Path) -> str:
    """Top-1 and top-3 pick agreement across all model types per snapshot.

    Returns markdown table.
    """
    test = preds_df[preds_df["year"] == 2026].copy()
    if test.empty:
        return ""

    dates = sorted(test["snapshot_date"].unique())

    lines = [
        "| Date | Event | LR Top Pick | GBT Top Pick | CLogit Top Pick | "
        "SoftGBT Top Pick | CalSGBT Top Pick | All Agree? |"
    ]
    lines.append(
        "|------|-------|-------------|--------------|-----------------|"
        "------------------|------------------|------------|"
    )

    for d in dates:
        snap = test[test["snapshot_date"] == d]
        event = AWARDS_SEASON_EVENTS.get(d, "")
        tops: dict[str, str] = {}
        for mt in MODEL_TYPES:
            mt_snap = snap[snap["model_type"] == mt]
            if not mt_snap.empty:
                tops[mt] = mt_snap.loc[mt_snap["probability"].idxmax(), "title"]
            else:
                tops[mt] = "---"

        all_agree = len(set(tops.values()) - {"---"}) <= 1
        short = {mt: (t[:18] + ".." if len(t) > 18 else t) for mt, t in tops.items()}
        lines.append(
            f"| {d} | {event} | {short.get('lr', '---')} "
            f"| {short.get('gbt', '---')} "
            f"| {short.get('conditional_logit', '---')} "
            f"| {short.get('softmax_gbt', '---')} "
            f"| {short.get('calibrated_softmax_gbt', '---')} "
            f"| {'Yes' if all_agree else 'No'} |"
        )

    table = "\n".join(lines)
    print("\n  Top Pick Agreement:")
    print(table)
    return table


def analyze_marginal_info(exp_dir: Path, output_dir: Path) -> str:
    """Delta(accuracy) and Delta(Brier) per event transition, all models.

    Returns markdown table.
    """
    all_metrics: dict[str, list[tuple[str, float, float]]] = {}
    for mt in MODEL_TYPES:
        mt_dir = exp_dir / "models" / mt
        if not mt_dir.exists():
            continue
        entries: list[tuple[str, float, float]] = []
        for date_dir in sorted(mt_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            m = load_cv_metrics(exp_dir, mt, date_dir.name)
            if m:
                km = extract_key_metrics(m)
                entries.append(
                    (
                        date_dir.name,
                        km.get("accuracy", 0),
                        km.get("brier_score", 0),
                    )
                )
        all_metrics[mt] = entries

    lines = ["| Date | Event | LR Dacc | GBT Dacc | CLogit Dacc | SoftGBT Dacc | CalSGBT Dacc |"]
    lines.append(
        "|------|-------|---------|----------|-------------|--------------|--------------|"
    )

    all_dates = sorted({d for entries in all_metrics.values() for d, _, _ in entries})
    prev: dict[str, tuple[float, float]] = {}

    for d in all_dates:
        event = AWARDS_SEASON_EVENTS.get(d, "")
        parts = [f"| {d} | {event}"]
        for mt in MODEL_TYPES:
            entry = [(dd, a, b) for dd, a, b in all_metrics.get(mt, []) if dd == d]
            if entry:
                _, acc, brier = entry[0]
                if mt in prev:
                    delta = (acc - prev[mt][0]) * 100
                    parts.append(f" {delta:+.1f}pp")
                else:
                    parts.append(" ---")
                prev[mt] = (acc, brier)
            else:
                parts.append(" ---")
        lines.append(" |".join(parts) + " |")

    table = "\n".join(lines)
    print("\n  Marginal Information Value:")
    print(table)
    return table


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Run all deep-dive analyses."""
    parser = argparse.ArgumentParser(description="Deep-dive multinomial modeling analysis")
    parser.add_argument("--exp-dir", type=str, required=True, help="Experiment directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MULTINOMIAL MODELING --- DEEP DIVE ANALYSIS")
    print("=" * 70)
    print(f"  Experiment dir: {exp_dir}")
    print(f"  Output dir: {output_dir}")

    # 1. Load all predictions
    print("\nLoading predictions...")
    preds_df = load_all_test_predictions(exp_dir, MODEL_TYPES)
    print(f"  {len(preds_df)} prediction rows")

    # 2. Load market prices
    print("Loading Kalshi market prices...")
    market_df = load_market_prices()
    print(f"  {len(market_df)} price rows")

    # 3. Build model-market joined data
    print("Building model-market dataset...")
    mm_df = build_model_market_df(preds_df, market_df)
    print(f"  {len(mm_df)} joined rows")

    # 4. Load feature importances
    print("Loading feature importances...")
    fi_data = load_feature_importances(exp_dir, MODEL_TYPES)
    for mt, dates_dict in fi_data.items():
        print(f"  {mt}: {len(dates_dict)} snapshots")

    # 5. Load backtest results
    print("Loading backtest results...")
    bt_results = load_backtest_results(exp_dir)
    if bt_results:
        n_runs = len(bt_results.get("backtests", {}))
        print(f"  {n_runs} backtest runs")
    else:
        print("  No backtest results found")

    # ---- Run all analyses ----
    results: dict[str, str] = {}

    print("\n--- 1. Model vs Market ---")
    plot_model_vs_market(mm_df, MODEL_TYPES, output_path=output_dir / "model_vs_market.png")

    print("\n--- 2. Divergence Heatmaps ---")
    plot_divergence_heatmaps(mm_df, MODEL_TYPES, output_path=output_dir / "divergence_heatmaps.png")

    print("\n--- 3. Probability Distribution Anatomy ---")
    plot_prob_distribution_anatomy(
        preds_df, MODEL_TYPES, output_path=output_dir / "prob_anatomy.png"
    )

    print("\n--- 4. Calibration Reliability Diagrams ---")
    # Load CV predictions for latest snapshot per model
    snapshots = sorted(
        {
            d.name
            for mt in MODEL_TYPES
            if (exp_dir / "models" / mt).exists()
            for d in (exp_dir / "models" / mt).iterdir()
            if d.is_dir()
        }
    )
    latest_snap = snapshots[-1] if snapshots else ""
    cv_preds: dict[str, pd.DataFrame] = {}
    for mt in MODEL_TYPES:
        cv = load_cv_predictions(exp_dir, mt, latest_snap)
        if cv is not None:
            cv_preds[mt] = cv
    if cv_preds:
        plot_reliability_diagrams(cv_preds, output_path=output_dir / "reliability_diagrams.png")

    print("\n--- 5. Rank Agreement ---")
    results["rank_agreement"] = analyze_rank_agreement(preds_df, output_dir)

    print("\n--- 6. Feature Importance Evolution ---")
    for mt, dates_dict in fi_data.items():
        if dates_dict:
            plot_feature_importance_heatmap(
                dates_dict,
                get_model_display(mt),
                output_path=output_dir / f"feature_importance_{mt}.png",
            )

    print("\n--- 7. CV Metrics Over Time ---")
    # Build the dict structure plot_metrics_over_time_detailed expects
    all_cv: dict[str, dict[str, dict[str, float]]] = {}
    for mt in MODEL_TYPES:
        all_cv[mt] = {}
        mt_dir = exp_dir / "models" / mt
        if not mt_dir.exists():
            continue
        for date_dir in sorted(mt_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            metrics = load_cv_metrics(exp_dir, mt, date_dir.name)
            if metrics:
                km = extract_key_metrics(metrics)
                all_cv[mt][date_dir.name] = {
                    "accuracy": km.get("accuracy", 0),
                    "brier": km.get("brier_score", 0),
                    "log_loss": km.get("log_loss", 0),
                    "winner_prob": km.get("mean_winner_prob", 0),
                }

    plot_metrics_over_time_detailed(
        all_cv, MODEL_TYPES, output_path=output_dir / "cv_metrics_detailed.png"
    )

    # Build markdown table of per-date per-model metrics
    all_dates = sorted({d for mt_data in all_cv.values() for d in mt_data})
    metric_lines = [
        "| Date | Event | LR Acc | GBT Acc | CLogit Acc "
        "| SoftGBT Acc | CalSGBT Acc | LR Brier | CLogit Brier |"
    ]
    metric_lines.append(
        "|------|-------|--------|---------|------------"
        "|-------------|-------------|----------|--------------|"
    )
    for d in all_dates:
        event = AWARDS_SEASON_EVENTS.get(d, "")
        lr_acc = all_cv.get("lr", {}).get(d, {}).get("accuracy", 0)
        gbt_acc = all_cv.get("gbt", {}).get(d, {}).get("accuracy", 0)
        cl_acc = all_cv.get("conditional_logit", {}).get(d, {}).get("accuracy", 0)
        sg_acc = all_cv.get("softmax_gbt", {}).get(d, {}).get("accuracy", 0)
        csg_acc = all_cv.get("calibrated_softmax_gbt", {}).get(d, {}).get("accuracy", 0)
        lr_br = all_cv.get("lr", {}).get(d, {}).get("brier", 0)
        cl_br = all_cv.get("conditional_logit", {}).get(d, {}).get("brier", 0)
        metric_lines.append(
            f"| {d} | {event} | {lr_acc:.1%} | {gbt_acc:.1%} | {cl_acc:.1%} | {sg_acc:.1%} "
            f"| {csg_acc:.1%} | {lr_br:.4f} | {cl_br:.4f} |"
        )
    results["metrics_table"] = "\n".join(metric_lines)

    print("\n--- 8. Top Pick Agreement ---")
    results["top_pick"] = analyze_top_pick_agreement(preds_df, output_dir)

    print("\n--- 9. Marginal Info Value ---")
    results["marginal_info"] = analyze_marginal_info(exp_dir, output_dir)

    print("\n--- 10. Probability Concentration ---")
    results["prob_concentration"] = analyze_prob_concentration(preds_df, output_dir)

    print("\n--- 11. Binary vs Multinomial Side-by-Side ---")
    plot_binary_vs_multinomial(
        preds_df,
        model_a="lr",
        model_b="conditional_logit",
        output_path=output_dir / "binary_vs_multinomial.png",
    )

    print("\n--- 12. Edge Distributions ---")
    plot_edge_distributions(mm_df, MODEL_TYPES, output_path=output_dir / "edge_distributions.png")

    if bt_results:
        print("\n--- 13. Backtest Trade Summary ---")
        results["backtest_trades"] = analyze_backtest_trades(bt_results, output_dir)

        print("\n--- 14. Settlement Heatmap ---")
        plot_settlement_heatmap(
            bt_results, output_path=output_dir / "settlement_heatmap_detailed.png"
        )

        print("\n--- 15. Wealth Curves (annotated) ---")
        plot_wealth_curves(
            bt_results, MODEL_TYPES, output_path=output_dir / "wealth_curves_annotated.png"
        )

        print("\n--- 16. Position Evolution ---")
        for mt in MODEL_TYPES:
            plot_position_evolution(bt_results, mt, output_path=output_dir / f"positions_{mt}.png")

    # Save results summary for README
    summary_path = output_dir / "deep_dive_tables.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved tables to {summary_path}")

    print("\n" + "=" * 70)
    print("DEEP DIVE COMPLETE")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
