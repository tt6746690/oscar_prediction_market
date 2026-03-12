"""Comprehensive analysis of temporal model snapshots vs Kalshi market prices.

Analyses (original):
1. Correlation analysis — per-nominee Pearson r, MAE, RMSE
2. Event impact — per-event model + market deltas
3. Feature evolution — feature count, CV accuracy, entropy (3-panel plot)
4. Model agreement — LR vs GBT top-pick alignment
5. Trading signals — divergences > 10pp

Analyses (new):
6. Brier score comparison — CV Brier over time
7. Feature importance evolution — top features per model per snapshot (heatmap)
8. Marginal information value — Δ(CV metrics) per event
9. Market-adjusted model — α-blend model + market, sweep α
10. Calibration reliability diagram — predicted vs actual from CV
11. Rank comparison — Spearman rank correlation per snapshot

Usage:
    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\
d20260211_temporal_model_snapshots.analysis \
        --predictions storage/d20260211_temporal_model_snapshots/model_predictions_timeseries.csv \
        --models-dir storage/d20260211_temporal_model_snapshots/models \
        --output-dir storage/d20260211_temporal_model_snapshots
"""

import argparse
import math
from datetime import date
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats  # type: ignore[import-untyped]

from oscar_prediction_market.one_offs.analysis_utils.data_loading import (
    get_market_prob,
    load_market_prices,
)
from oscar_prediction_market.one_offs.analysis_utils.plot_calibration import (
    plot_reliability_diagrams,
)
from oscar_prediction_market.one_offs.analysis_utils.plot_temporal import (
    plot_feature_importance_heatmap,
)
from oscar_prediction_market.one_offs.analysis_utils.style import (
    AWARDS_SEASON_EVENTS,
    apply_style,
    get_model_display,
)

# ============================================================================
# Constants
# ============================================================================

apply_style()


# ============================================================================
# Data Loading
# ============================================================================


def load_predictions(predictions_path: Path) -> pd.DataFrame:
    """Load model predictions timeseries CSV.

    Returns DataFrame with columns: snapshot_date, model_type, feature_count,
    cv_accuracy, cv_logloss, cv_brier, ceremony, year, title, probability, rank
    """
    df = pd.read_csv(predictions_path)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.date
    return df


def build_model_market_df(preds_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """Join model predictions with market prices at each snapshot date.

    Returns DataFrame with: snapshot_date, model_type, title, model_prob, market_prob,
    divergence (model - market).
    """
    rows: list[dict] = []
    # Only keep 2026 test predictions (ceremony 98)
    test_preds = preds_df[preds_df["year"] == 2026].copy()

    for _, row in test_preds.iterrows():
        snapshot = row["snapshot_date"]
        market_prob = get_market_prob(market_df, str(snapshot), row["title"])
        rows.append(
            {
                "snapshot_date": snapshot,
                "model_type": row["model_type"],
                "title": row["title"],
                "model_prob": row["probability"],
                "market_prob": market_prob,
                "divergence": (
                    row["probability"] - market_prob if market_prob is not None else None
                ),
                "feature_count": row["feature_count"],
                "cv_accuracy": row["cv_accuracy"],
                "cv_brier": row.get("cv_brier"),
                "rank": row["rank"],
            }
        )

    df = pd.DataFrame(rows)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.date
    return df


def load_feature_importances(models_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Load feature importance for each model/date.

    Returns: {model_type: {snapshot_date: {feature: importance}}}
    """
    result: dict[str, dict[str, dict[str, float]]] = {}

    for model_dir in sorted(models_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_type = model_dir.name
        result[model_type] = {}

        for date_dir in sorted(model_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            snapshot_date = date_dir.name

            # Find feature importance CSV
            run_name = f"{model_type}_{snapshot_date}"
            fi_path = date_dir / run_name / "5_final_predict" / "feature_importance.csv"
            if not fi_path.exists():
                # Try alternative paths
                candidates = list(date_dir.glob("*/5_final_predict/feature_importance.csv"))
                if candidates:
                    fi_path = candidates[0]
                else:
                    continue

            fi_df = pd.read_csv(fi_path)
            importance_col = (
                "abs_coefficient" if "abs_coefficient" in fi_df.columns else "importance"
            )
            importances = {
                row["feature"]: float(row[importance_col]) for _, row in fi_df.iterrows()
            }
            result[model_type][snapshot_date] = importances

    return result


def load_cv_predictions(models_dir: Path) -> pd.DataFrame:
    """Load CV predictions from the final snapshot for reliability diagram.

    Reads 4_selected_cv/predictions.csv from the latest snapshot.
    """
    all_rows: list[pd.DataFrame] = []

    for model_dir in sorted(models_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_type = model_dir.name

        # Use the latest snapshot
        date_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir()])
        if not date_dirs:
            continue
        latest_dir = date_dirs[-1]
        snapshot_date = latest_dir.name

        run_name = f"{model_type}_{snapshot_date}"
        cv_path = latest_dir / run_name / "4_selected_cv" / "predictions.csv"
        if not cv_path.exists():
            candidates = list(latest_dir.glob("*/4_selected_cv/predictions.csv"))
            if candidates:
                cv_path = candidates[0]
            else:
                continue

        cv_df = pd.read_csv(cv_path)
        cv_df["model_type"] = model_type
        cv_df["snapshot_date"] = snapshot_date
        all_rows.append(cv_df)

    if all_rows:
        return pd.concat(all_rows, ignore_index=True)
    return pd.DataFrame()


# ============================================================================
# Analysis 1: Correlation Analysis
# ============================================================================


def analyze_correlation(mm_df: pd.DataFrame, output_dir: Path) -> None:
    """Per-nominee: Pearson r, MAE, RMSE between model and market."""
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS (Model P(win) vs Market Price)")
    print("=" * 70)

    valid = mm_df.dropna(subset=["market_prob"])
    if valid.empty:
        print("No market data available for correlation analysis.")
        return

    header = f"{'Nominee':<30} {'LR r':>7} {'GBT r':>7} {'LR MAE':>8} {'GBT MAE':>8} {'n':>4}"
    print(header)
    print("-" * len(header))

    nominees = sorted(valid["title"].unique())
    for nominee in nominees:
        parts = []
        parts.append(f"{nominee:<30}")
        for model in ["lr", "gbt"]:
            sub = valid[(valid["title"] == nominee) & (valid["model_type"] == model)]
            if len(sub) > 2:
                r, _ = stats.pearsonr(sub["model_prob"], sub["market_prob"])
                mae = np.mean(np.abs(sub["model_prob"] - sub["market_prob"]))
                parts.append(f"{r:>+7.2f}")
            else:
                parts.append(f"{'N/A':>7}")

        for model in ["lr", "gbt"]:
            sub = valid[(valid["title"] == nominee) & (valid["model_type"] == model)]
            if len(sub) > 0:
                mae = np.mean(np.abs(sub["model_prob"] - sub["market_prob"]))
                parts.append(f"{mae * 100:>7.1f}pp")
            else:
                parts.append(f"{'N/A':>8}")

        n = len(valid[(valid["title"] == nominee) & (valid["model_type"] == "lr")])
        parts.append(f"{n:>4}")
        print(" ".join(parts))


# ============================================================================
# Analysis 2: Event Impact
# ============================================================================


def analyze_event_impact(mm_df: pd.DataFrame, output_dir: Path) -> None:
    """Per-event: change in model prob and market price for top nominees."""
    print("\n" + "=" * 70)
    print("EVENT IMPACT ANALYSIS")
    print("=" * 70)

    dates = sorted(mm_df["snapshot_date"].unique())
    if len(dates) < 2:
        print("Need at least 2 snapshots for event impact analysis.")
        return

    # Track top nominees
    top_nominees = set()
    for d in dates:
        for model in ["lr", "gbt"]:
            sub = mm_df[(mm_df["snapshot_date"] == d) & (mm_df["model_type"] == model)]
            if not sub.empty:
                top = sub.nsmallest(3, "rank")["title"].tolist()
                top_nominees.update(top)

    # Show impact per event transition
    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        curr_date = dates[i]
        event = AWARDS_SEASON_EVENTS.get(str(curr_date), str(curr_date))
        print(f"\n{curr_date} — {event}")

        for model in ["lr", "gbt"]:
            prev = mm_df[(mm_df["snapshot_date"] == prev_date) & (mm_df["model_type"] == model)]
            curr = mm_df[(mm_df["snapshot_date"] == curr_date) & (mm_df["model_type"] == model)]

            if prev.empty or curr.empty:
                continue

            prev_dict = dict(zip(prev["title"], prev["model_prob"], strict=False))
            curr_dict = dict(zip(curr["title"], curr["model_prob"], strict=False))

            # Show biggest movers
            deltas = []
            for title in set(prev_dict) & set(curr_dict):
                delta = curr_dict[title] - prev_dict[title]
                deltas.append((title, delta))

            deltas.sort(key=lambda x: abs(x[1]), reverse=True)
            top_movers = [d for d in deltas[:3] if abs(d[1]) > 0.01]

            if top_movers:
                movers_str = ", ".join(f"{t}: {d:+.1%}" for t, d in top_movers)
                print(f"  {model.upper()}: {movers_str}")


# ============================================================================
# Analysis 3: Feature Evolution (3-panel plot)
# ============================================================================


def analyze_feature_evolution(preds_df: pd.DataFrame, output_dir: Path) -> None:
    """Feature count, CV accuracy, and prediction entropy over time."""
    print("\n" + "=" * 70)
    print("FEATURE EVOLUTION")
    print("=" * 70)

    # Get per-snapshot summary (from test predictions)
    test = preds_df[preds_df["year"] == 2026].copy()
    if test.empty:
        print("No 2026 test predictions found.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for model in ["lr", "gbt"]:
        model_data = test[test["model_type"] == model]
        snapshots = sorted(model_data["snapshot_date"].unique())

        feat_counts = []
        cv_accs = []
        entropies = []

        for d in snapshots:
            snap = model_data[model_data["snapshot_date"] == d]
            feat_counts.append(snap.iloc[0]["feature_count"])
            cv_accs.append(snap.iloc[0]["cv_accuracy"])

            # Compute prediction entropy
            probs = snap["probability"].values
            probs = probs[probs > 0]  # Avoid log(0)
            entropy = -np.sum(probs * np.log2(probs))
            entropies.append(entropy)

        date_labels = [str(d) for d in snapshots]
        label = model.upper()

        axes[0].plot(date_labels, feat_counts, "o-", label=label)
        axes[1].plot(date_labels, [a * 100 for a in cv_accs], "o-", label=label)
        axes[2].plot(date_labels, entropies, "o-", label=label)

        print(f"\n{label}:")
        print(f"  Features: {feat_counts}")
        print(f"  CV Accuracy: {[f'{a:.1%}' for a in cv_accs]}")
        print(f"  Entropy: {[f'{e:.2f}' for e in entropies]}")

    axes[0].set_ylabel("Feature Count")
    axes[0].set_title("Feature Evolution Over Awards Season")
    axes[0].legend()

    axes[1].set_ylabel("CV Accuracy (%)")
    axes[1].legend()

    axes[2].set_ylabel("Prediction Entropy (bits)")
    axes[2].set_xlabel("Snapshot Date")
    axes[2].legend()

    # Add event annotations
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "feature_evolution.png", bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {output_dir / 'feature_evolution.png'}")


# ============================================================================
# Analysis 4: Model Agreement
# ============================================================================


def analyze_model_agreement(mm_df: pd.DataFrame, output_dir: Path) -> None:
    """LR vs GBT top-pick agreement at each snapshot."""
    print("\n" + "=" * 70)
    print("MODEL AGREEMENT (LR vs GBT)")
    print("=" * 70)

    dates = sorted(mm_df["snapshot_date"].unique())
    agreements = 0
    total = 0

    header = f"{'Date':<12} {'Event':<25} {'LR Top Pick':<30} {'GBT Top Pick':<30} {'Agree':>5}"
    print(header)
    print("-" * len(header))

    for d in dates:
        event = AWARDS_SEASON_EVENTS.get(str(d), "")
        lr_sub = mm_df[(mm_df["snapshot_date"] == d) & (mm_df["model_type"] == "lr")]
        gbt_sub = mm_df[(mm_df["snapshot_date"] == d) & (mm_df["model_type"] == "gbt")]

        if lr_sub.empty or gbt_sub.empty:
            continue

        lr_top = lr_sub.loc[lr_sub["rank"].idxmin(), "title"]
        gbt_top = gbt_sub.loc[gbt_sub["rank"].idxmin(), "title"]
        agree = lr_top == gbt_top

        print(f"{str(d):<12} {event:<25} {lr_top:<30} {gbt_top:<30} {'Yes' if agree else 'No':>5}")

        if agree:
            agreements += 1
        total += 1

    if total > 0:
        print(f"\nAgreement: {agreements}/{total} ({agreements / total:.0%})")


# ============================================================================
# Analysis 5: Trading Signals
# ============================================================================


def analyze_trading_signals(mm_df: pd.DataFrame, output_dir: Path) -> None:
    """Divergences > 10pp between model and market."""
    print("\n" + "=" * 70)
    print("TRADING SIGNALS (>10pp model-market divergence)")
    print("=" * 70)

    valid = mm_df.dropna(subset=["market_prob"])
    signals = valid[abs(valid["divergence"]) > 0.10].copy()

    if signals.empty:
        print("No signals found.")
        return

    signals = signals.sort_values("divergence", key=abs, ascending=False)

    print(f"\n{len(signals)} signals total:")
    header = f"{'Date':<12} {'Nominee':<30} {'Model':>5} {'Model P':>8} {'Mkt P':>8} {'Gap':>8}"
    print(header)
    print("-" * len(header))

    for _, row in signals.head(20).iterrows():
        print(
            f"{str(row['snapshot_date']):<12} {row['title']:<30} "
            f"{row['model_type'].upper():>5} "
            f"{row['model_prob']:>7.1%} {row['market_prob']:>7.1%} "
            f"{row['divergence']:>+7.1%}"
        )


# ============================================================================
# Analysis 6: Brier Score Comparison (CV Brier over time)
# ============================================================================


def analyze_brier_score(preds_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot CV Brier score and log-loss over time for both models."""
    print("\n" + "=" * 70)
    print("BRIER SCORE COMPARISON (CV performance over time)")
    print("=" * 70)

    test = preds_df[preds_df["year"] == 2026].copy()
    if test.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model in ["lr", "gbt"]:
        model_data = test[test["model_type"] == model]
        snapshots = sorted(model_data["snapshot_date"].unique())

        briers = []
        loglosses = []
        for d in snapshots:
            snap = model_data[model_data["snapshot_date"] == d]
            b = snap.iloc[0].get("cv_brier")
            ll = snap.iloc[0].get("cv_logloss")
            briers.append(b if b is not None else float("nan"))
            loglosses.append(ll if ll is not None else float("nan"))

        date_labels = [str(d) for d in snapshots]
        label = model.upper()

        axes[0].plot(date_labels, briers, "o-", label=label)
        axes[1].plot(date_labels, loglosses, "o-", label=label)

        print(f"{label}: Brier {briers[-1]:.4f} → {briers[0]:.4f}" if briers else "")

    axes[0].set_ylabel("CV Brier Score")
    axes[0].set_title("CV Brier Score Over Time")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].set_ylabel("CV Log-Loss")
    axes[1].set_title("CV Log-Loss Over Time")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "brier_score_comparison.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'brier_score_comparison.png'}")


# ============================================================================
# Analysis 7: Feature Importance Evolution (heatmap)
# ============================================================================


def analyze_feature_importance_evolution(
    fi_data: dict[str, dict[str, dict[str, float]]], output_dir: Path
) -> None:
    """Heatmap of top feature importances evolving over snapshots."""
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE EVOLUTION")
    print("=" * 70)

    for model_type, dates_dict in fi_data.items():
        if not dates_dict:
            continue

        fname = f"feature_importance_evolution_{model_type}.png"
        plot_feature_importance_heatmap(
            dates_dict,
            model_display_name=get_model_display(model_type),
            output_path=output_dir / fname,
        )
        print(f"Saved: {output_dir / fname}")

        # Print top features at first and last snapshot
        dates = sorted(dates_dict.keys())
        first_date = dates[0]
        last_date = dates[-1]
        print(f"\n{model_type.upper()} — Top features:")
        print(f"  {first_date}: {sorted(dates_dict[first_date].items(), key=lambda x: -x[1])[:5]}")
        print(f"  {last_date}: {sorted(dates_dict[last_date].items(), key=lambda x: -x[1])[:5]}")


# ============================================================================
# Analysis 8: Marginal Information Value
# ============================================================================


def analyze_marginal_info(preds_df: pd.DataFrame, output_dir: Path) -> None:
    """Δ(CV metrics) per event — which events improve model most."""
    print("\n" + "=" * 70)
    print("MARGINAL INFORMATION VALUE (Δ CV metrics per event)")
    print("=" * 70)

    test = preds_df[preds_df["year"] == 2026].copy()
    if test.empty:
        return

    header = (
        f"{'Date':<12} {'Event':<25} "
        f"{'LR Δacc':>8} {'LR ΔBrier':>10} "
        f"{'GBT Δacc':>9} {'GBT ΔBrier':>10}"
    )
    print(header)
    print("-" * len(header))

    # Collect per-model results
    lr_results: list[tuple[date, float, float | None]] = []
    gbt_results: list[tuple[date, float, float | None]] = []

    for model in ["lr", "gbt"]:
        model_data = test[test["model_type"] == model]
        snapshots = sorted(model_data["snapshot_date"].unique())

        results: list[tuple[date, float, float | None]] = []
        for d in snapshots:
            snap = model_data[model_data["snapshot_date"] == d]
            acc = snap.iloc[0]["cv_accuracy"]
            brier = snap.iloc[0].get("cv_brier")
            results.append((d, acc, brier))

        if model == "lr":
            lr_results = results
        else:
            gbt_results = results

    # Print side-by-side
    dates = sorted({d for d, _, _ in lr_results} | {d for d, _, _ in gbt_results})
    lr_dict = {d: (a, b) for d, a, b in lr_results}
    gbt_dict = {d: (a, b) for d, a, b in gbt_results}

    prev_lr: tuple[float, float | None] | None = None
    prev_gbt: tuple[float, float | None] | None = None

    for d in dates:
        event = AWARDS_SEASON_EVENTS.get(str(d), str(d))
        lr_acc, lr_brier = lr_dict.get(d, (0, None))
        gbt_acc, gbt_brier = gbt_dict.get(d, (0, None))

        lr_da = f"{(lr_acc - prev_lr[0]) * 100:>+7.1f}pp" if prev_lr else f"{'—':>8}"
        gbt_da = f"{(gbt_acc - prev_gbt[0]) * 100:>+7.1f}pp" if prev_gbt else f"{'—':>9}"

        lr_db = (
            f"{lr_brier - prev_lr[1]:>+10.4f}"
            if prev_lr and prev_lr[1] is not None and lr_brier is not None
            else f"{'—':>10}"
        )
        gbt_db = (
            f"{gbt_brier - prev_gbt[1]:>+10.4f}"
            if prev_gbt and prev_gbt[1] is not None and gbt_brier is not None
            else f"{'—':>10}"
        )

        print(f"{str(d):<12} {event:<25} {lr_da} {lr_db} {gbt_da} {gbt_db}")

        prev_lr = (lr_acc, lr_brier)
        prev_gbt = (gbt_acc, gbt_brier)


# ============================================================================
# Analysis 9: Market-Adjusted Model (α-blend)
# ============================================================================


def analyze_market_blend(mm_df: pd.DataFrame, output_dir: Path) -> None:
    """Sweep α for P_blend = α * P_market + (1-α) * P_model.

    Alpha-blending combines model predictions with market prices to find the
    optimal weighting. α=0 trusts the model fully; α=1 trusts the market
    fully. The optimal α minimizes RMSE of blended predictions vs the final
    snapshot's most-informed estimate.

    Interpretation:
      - Low optimal α → model adds genuine information beyond market prices.
      - High optimal α → market is more accurate; model adds noise.
    """
    print("\n" + "=" * 70)
    print("MARKET-ADJUSTED MODEL (α-blend analysis)")
    print("=" * 70)

    valid = mm_df.dropna(subset=["market_prob"])
    if valid.empty:
        print("No market data for blend analysis.")
        return

    alphas = np.linspace(0, 1, 21)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, model in enumerate(["lr", "gbt"]):
        model_data = valid[valid["model_type"] == model]
        dates = sorted(model_data["snapshot_date"].unique())
        if not dates:
            continue

        # Reference: final snapshot model predictions
        final_date = dates[-1]
        final = model_data[model_data["snapshot_date"] == final_date]
        final_probs = dict(zip(final["title"], final["model_prob"], strict=False))

        # For each date, compute MSE of blend vs final model and vs market
        mse_vs_final: dict[float, list[float]] = {a: [] for a in alphas}
        mse_vs_market: dict[float, list[float]] = {a: [] for a in alphas}

        for d in dates:
            snap = model_data[model_data["snapshot_date"] == d]
            for _, row in snap.iterrows():
                title = row["title"]
                model_p = row["model_prob"]
                market_p = row["market_prob"]
                final_p = final_probs.get(title)

                if final_p is None:
                    continue

                for a in alphas:
                    blend_p = a * market_p + (1 - a) * model_p
                    mse_vs_final[a].append((blend_p - final_p) ** 2)
                    mse_vs_market[a].append((blend_p - market_p) ** 2)

        rmse_final = [np.sqrt(np.mean(mse_vs_final[a])) for a in alphas]
        rmse_market = [np.sqrt(np.mean(mse_vs_market[a])) for a in alphas]

        axes[idx].plot(alphas, rmse_final, "b-", label="RMSE vs final model")
        axes[idx].plot(alphas, rmse_market, "r--", label="RMSE vs market")
        axes[idx].set_xlabel("α (market weight)")
        axes[idx].set_ylabel("RMSE")
        axes[idx].set_title(f"Blend Analysis — {model.upper()}")
        axes[idx].legend()
        axes[idx].axvline(0.5, color="gray", linestyle=":", alpha=0.5)

        # Find optimal α (minimize RMSE vs final)
        opt_idx = int(np.argmin(rmse_final))
        opt_alpha = alphas[opt_idx]
        print(
            f"{model.upper()}: Optimal α = {opt_alpha:.2f} "
            f"(RMSE vs final: {rmse_final[opt_idx]:.4f})"
        )

    plt.tight_layout()
    plt.savefig(output_dir / "market_blend_analysis.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'market_blend_analysis.png'}")


# ============================================================================
# Analysis 10: Calibration Reliability Diagram
# ============================================================================


def analyze_reliability(cv_preds_df: pd.DataFrame, output_dir: Path) -> None:
    """Calibration reliability diagram from CV predictions."""
    print("\n" + "=" * 70)
    print("CALIBRATION RELIABILITY DIAGRAM")
    print("=" * 70)

    if cv_preds_df.empty:
        print("No CV predictions available for reliability diagram.")
        return

    # Split into per-model DataFrames for shared function
    cv_predictions: dict[str, pd.DataFrame] = {}
    for model in cv_preds_df["model_type"].unique():
        cv_predictions[model] = cv_preds_df[cv_preds_df["model_type"] == model]

    plot_reliability_diagrams(cv_predictions, output_path=output_dir / "reliability_diagram.png")

    for model, df in cv_predictions.items():
        print(f"{model.upper()}: {len(df)} predictions")

    print(f"Saved: {output_dir / 'reliability_diagram.png'}")


# ============================================================================
# Analysis 11: Rank Comparison (Spearman)
# ============================================================================


def analyze_rank_comparison(mm_df: pd.DataFrame, output_dir: Path) -> None:
    """Spearman rank correlation between model and market at each snapshot."""
    print("\n" + "=" * 70)
    print("RANK COMPARISON (Spearman ρ)")
    print("=" * 70)

    valid = mm_df.dropna(subset=["market_prob"])
    if valid.empty:
        print("No market data for rank comparison.")
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    header = f"{'Date':<12} {'Event':<25} {'LR ρ':>7} {'GBT ρ':>7}"
    print(header)
    print("-" * len(header))

    for model in ["lr", "gbt"]:
        model_data = valid[valid["model_type"] == model]
        dates = sorted(model_data["snapshot_date"].unique())

        rhos: list[float] = []
        for d in dates:
            snap = model_data[model_data["snapshot_date"] == d]
            if len(snap) > 2:
                result = stats.spearmanr(snap["model_prob"], snap["market_prob"])
                rhos.append(float(result.statistic))  # type: ignore[arg-type]
            else:
                rhos.append(float("nan"))

        date_labels = [str(d) for d in dates]
        ax.plot(date_labels, rhos, "o-", label=model.upper())

        if model == "lr":
            lr_rhos = dict(zip(dates, rhos, strict=False))
        else:
            for i, d in enumerate(dates):
                event = AWARDS_SEASON_EVENTS.get(str(d), "")
                lr_r = lr_rhos.get(d, float("nan"))
                gbt_r = rhos[i]
                print(f"{str(d):<12} {event:<25} {lr_r:>+7.3f} {gbt_r:>+7.3f}")

    ax.set_ylabel("Spearman ρ")
    ax.set_xlabel("Snapshot Date")
    ax.set_title("Rank Correlation: Model vs Market")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "rank_comparison.png", bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {output_dir / 'rank_comparison.png'}")


# ============================================================================
# Main Plots: Model vs Market + Divergence Heatmaps
# ============================================================================


def plot_model_vs_market(mm_df: pd.DataFrame, output_dir: Path) -> None:
    """Subplot grid: model P vs market P per nominee, one subplot per nominee."""
    valid = mm_df.dropna(subset=["market_prob"])
    if valid.empty:
        return

    for model in ["lr", "gbt"]:
        model_data = valid[valid["model_type"] == model]
        nominees = sorted(model_data["title"].unique())
        ncols = 3
        nrows = math.ceil(len(nominees) / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows), sharex=True)
        axes_flat = axes.flatten() if nrows > 1 else (axes if ncols > 1 else [axes])
        cmap = matplotlib.colormaps["tab10"]

        for i, nominee in enumerate(nominees):
            ax = axes_flat[i]
            nom_data = model_data[model_data["title"] == nominee].sort_values("snapshot_date")
            dates = [str(d) for d in nom_data["snapshot_date"]]
            model_probs = nom_data["model_prob"].values * 100
            market_probs = nom_data["market_prob"].values * 100

            ax.plot(
                dates,
                model_probs,
                "-o",
                color=cmap(i / max(1, len(nominees) - 1)),
                markersize=4,
                label="Model",
            )
            ax.plot(dates, market_probs, "--", color="gray", alpha=0.7, label="Market")
            ax.fill_between(
                dates,
                model_probs,
                market_probs,
                alpha=0.10,
                color=cmap(i / max(1, len(nominees) - 1)),
            )

            nom_short = nominee[:25] + "..." if len(nominee) > 25 else nominee
            ax.set_title(nom_short, fontsize=9)
            ax.tick_params(axis="x", rotation=45, labelsize=7)
            if i % ncols == 0:
                ax.set_ylabel("Probability (%)")
            if i == 0:
                ax.legend(fontsize=7)

        # Hide unused axes
        for j in range(len(nominees), len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(
            f"Model vs Market — {model.upper()} (solid=model, dashed=market)",
            fontsize=13,
        )
        plt.tight_layout()
        fname = f"model_vs_market_{model}.png"
        plt.savefig(output_dir / fname, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_dir / fname}")


def plot_divergence_heatmap(mm_df: pd.DataFrame, output_dir: Path) -> None:
    """Nominee × date heatmaps showing model - market probability."""
    valid = mm_df.dropna(subset=["market_prob"])
    if valid.empty:
        return

    for model in ["lr", "gbt"]:
        model_data = valid[valid["model_type"] == model]
        nominees = sorted(model_data["title"].unique())
        dates = sorted(model_data["snapshot_date"].unique())

        matrix = np.full((len(nominees), len(dates)), np.nan)
        for i, nominee in enumerate(nominees):
            for j, d in enumerate(dates):
                sub = model_data[
                    (model_data["title"] == nominee) & (model_data["snapshot_date"] == d)
                ]
                if not sub.empty:
                    matrix[i, j] = sub.iloc[0]["divergence"] * 100  # In pp

        fig, ax = plt.subplots(figsize=(14, 8))
        vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels([str(d) for d in dates], rotation=45, ha="right")
        ax.set_yticks(range(len(nominees)))
        ax.set_yticklabels(nominees)
        ax.set_title(f"Divergence Heatmap — {model.upper()} (model - market, pp)")
        plt.colorbar(im, ax=ax, label="Divergence (pp)")

        # Annotate cells
        for i in range(len(nominees)):
            for j in range(len(dates)):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > vmax * 0.6 else "black"
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7, color=color)

        plt.tight_layout()
        fname = f"divergence_heatmap_{model}.png"
        plt.savefig(output_dir / fname, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_dir / fname}")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Run all analyses."""
    parser = argparse.ArgumentParser(
        description="Comprehensive analysis of temporal model snapshots vs market"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to model_predictions_timeseries.csv",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        required=True,
        help="Root directory with model outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for output plots and reports",
    )
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading model predictions...")
    preds_df = load_predictions(predictions_path)

    print("Loading market prices from Kalshi...")
    market_start = date(2025, 11, 25)
    market_end = date(2026, 2, 12)
    market_df = load_market_prices(market_start, market_end)

    print("Building model-market joined dataset...")
    mm_df = build_model_market_df(preds_df, market_df)

    print("Loading feature importances...")
    fi_data = load_feature_importances(models_dir)

    print("Loading CV predictions for reliability diagram...")
    cv_preds_df = load_cv_predictions(models_dir)

    # Run all analyses
    plot_model_vs_market(mm_df, output_dir)
    plot_divergence_heatmap(mm_df, output_dir)
    analyze_correlation(mm_df, output_dir)
    analyze_event_impact(mm_df, output_dir)
    analyze_feature_evolution(preds_df, output_dir)
    analyze_model_agreement(mm_df, output_dir)
    analyze_trading_signals(mm_df, output_dir)
    analyze_brier_score(preds_df, output_dir)
    analyze_feature_importance_evolution(fi_data, output_dir)
    analyze_marginal_info(preds_df, output_dir)
    analyze_market_blend(mm_df, output_dir)
    analyze_reliability(cv_preds_df, output_dir)
    analyze_rank_comparison(mm_df, output_dir)

    print("\n" + "=" * 70)
    print("ALL ANALYSES COMPLETE")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
