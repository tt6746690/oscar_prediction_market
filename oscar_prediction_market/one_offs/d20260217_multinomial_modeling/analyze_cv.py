"""Analyze CV results across model types for multinomial modeling comparison.

Loads cross-validation metrics from all model types x snapshot dates and produces:
1. Summary comparison table (accuracy, Brier, log loss, prob-sum)
2. Temporal stability plots (metrics across snapshot dates)
3. Post-hoc normalization analysis (what if we normalize binary model probs?)

Usage:
    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\
        d20260217_multinomial_modeling.analyze_cv \
        --exp-dir storage/d20260217_multinomial_modeling \
        --output-dir storage/d20260217_multinomial_modeling/cv_analysis
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from oscar_prediction_market.one_offs.analysis_utils.data_loading import (
    get_available_snapshots,
    load_cv_metrics,
    load_cv_predictions,
    load_test_predictions,
)
from oscar_prediction_market.one_offs.analysis_utils.plot_calibration import (
    plot_prob_sum_distribution,
)
from oscar_prediction_market.one_offs.analysis_utils.plot_temporal import (
    plot_metrics_over_time,
)
from oscar_prediction_market.one_offs.analysis_utils.style import (
    MODEL_DISPLAY,
    apply_style,
    get_model_display,
)
from oscar_prediction_market.one_offs.d20260217_multinomial_modeling.metrics import (
    compute_cv_metrics_from_df,
    extract_key_metrics,
)
from oscar_prediction_market.one_offs.d20260217_multinomial_modeling.plot_comparison import (
    plot_final_predictions_comparison,
)

logger = logging.getLogger(__name__)

apply_style()

MODEL_TYPES = ["lr", "gbt", "conditional_logit", "softmax_gbt", "calibrated_softmax_gbt"]


# ============================================================================
# Normalization analysis (unique to this one-off)
# ============================================================================


def compute_normalized_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    """Compute metrics after normalizing binary model probabilities to sum to 1.

    For each ceremony year, divides each nominee's probability by the sum of all
    probabilities in that ceremony, so they sum to 1.

    Returns the same key metrics as extract_key_metrics.
    """
    df = predictions.copy()

    # Normalize probabilities per ceremony
    ceremony_sums = df.groupby("ceremony")["probability"].transform("sum")
    df["probability"] = df["probability"] / ceremony_sums

    # Re-rank within each ceremony
    df["rank"] = df.groupby("ceremony")["probability"].rank(ascending=False, method="first")
    df["rank"] = df["rank"].astype(int)
    df["is_predicted_winner"] = df["rank"] == 1

    return compute_cv_metrics_from_df(df)


# ============================================================================
# Analysis
# ============================================================================


def build_comparison_table(
    exp_dir: Path, snapshot_dates: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build metric comparison tables across model types.

    Returns:
        per_snapshot: DataFrame with rows = (model_type, snapshot_date), columns = metrics
        averaged: DataFrame with rows = model_type, columns = metrics (averaged across snapshots)
    """
    rows: list[dict[str, object]] = []
    for model_type in MODEL_TYPES:
        for snap_date in snapshot_dates:
            metrics = load_cv_metrics(exp_dir, model_type, snap_date)
            if metrics is None:
                continue
            row: dict[str, object] = {**extract_key_metrics(metrics)}
            row["model_type"] = model_type
            row["snapshot_date"] = snap_date
            row["num_years"] = metrics.get("num_years", 0)
            rows.append(row)

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    per_snapshot = pd.DataFrame(rows)

    averaged = (
        per_snapshot.groupby("model_type")
        .agg(
            {
                "accuracy": "mean",
                "top_3_accuracy": "mean",
                "mrr": "mean",
                "brier_score": "mean",
                "log_loss": "mean",
                "mean_winner_prob": "mean",
                "prob_sum_mean": "mean",
                "prob_sum_std": "mean",
                "num_years": "first",
            }
        )
        .reindex(MODEL_TYPES)
    )

    return per_snapshot, averaged


def build_normalized_comparison(exp_dir: Path, snapshot_dates: list[str]) -> pd.DataFrame:
    """Compare binary models raw vs. post-hoc normalized.

    Only applies to lr and gbt since multinomial models already sum to 1.
    """
    rows: list[dict[str, object]] = []
    for model_type in ["lr", "gbt"]:
        for snap_date in snapshot_dates:
            preds = load_cv_predictions(exp_dir, model_type, snap_date)
            if preds is None:
                continue

            raw_row: dict[str, object] = {**compute_cv_metrics_from_df(preds)}
            raw_row["model_type"] = f"{model_type}_raw"
            raw_row["snapshot_date"] = snap_date
            rows.append(raw_row)

            norm_row: dict[str, object] = {**compute_normalized_metrics(preds)}
            norm_row["model_type"] = f"{model_type}_normalized"
            norm_row["snapshot_date"] = snap_date
            rows.append(norm_row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.groupby("model_type").agg(
        {
            "accuracy": "mean",
            "top_3_accuracy": "mean",
            "mrr": "mean",
            "brier_score": "mean",
            "log_loss": "mean",
            "mean_winner_prob": "mean",
            "prob_sum_mean": "mean",
        }
    )


def build_final_predictions_table(exp_dir: Path, snapshot_date: str) -> pd.DataFrame | None:
    """Compare 2026 predictions across model types for a given snapshot.

    Returns wide DataFrame: columns = model types, rows = nominees.
    """
    dfs = {}
    for model_type in MODEL_TYPES:
        preds = load_test_predictions(exp_dir, model_type, snapshot_date)
        if preds is not None:
            dfs[get_model_display(model_type)] = preds.set_index("title")["probability"]

    if not dfs:
        return None

    combined = pd.DataFrame(dfs)
    combined = combined.sort_values(combined.columns[0], ascending=False)
    return combined


# ============================================================================
# Reporting
# ============================================================================


def format_comparison_table(averaged: pd.DataFrame) -> str:
    """Format the averaged metrics as a markdown table."""
    lines = []
    lines.append("| Model | Accuracy | Top-3 | MRR | Brier | Log Loss | Winner Prob | Prob Sum |")
    lines.append("|-------|----------|-------|-----|-------|----------|-------------|---------|")

    for model_type in MODEL_TYPES:
        if model_type not in averaged.index:
            continue
        row = averaged.loc[model_type]
        display = get_model_display(model_type)
        lines.append(
            f"| {display} | {row['accuracy']:.1%} | {row['top_3_accuracy']:.1%} | "
            f"{row['mrr']:.3f} | {row['brier_score']:.4f} | {row['log_loss']:.4f} | "
            f"{row['mean_winner_prob']:.3f} | "
            f"{row['prob_sum_mean']:.2f} +/- {row['prob_sum_std']:.2f} |"
        )

    return "\n".join(lines)


def format_normalized_table(norm_df: pd.DataFrame) -> str:
    """Format the normalization comparison as a markdown table."""
    if norm_df.empty:
        return "(no data)"

    lines = []
    lines.append("| Model | Accuracy | Top-3 | MRR | Brier | Log Loss | Winner Prob | Prob Sum |")
    lines.append("|-------|----------|-------|-----|-------|----------|-------------|---------|")

    for model_type in norm_df.index:
        row = norm_df.loc[model_type]
        lines.append(
            f"| {model_type} | {row['accuracy']:.1%} | {row['top_3_accuracy']:.1%} | "
            f"{row['mrr']:.3f} | {row['brier_score']:.4f} | {row['log_loss']:.4f} | "
            f"{row['mean_winner_prob']:.3f} | {row['prob_sum_mean']:.2f} |"
        )

    return "\n".join(lines)


def write_summary(
    averaged: pd.DataFrame,
    norm_df: pd.DataFrame,
    per_snapshot: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write a text summary of the analysis."""
    lines = []
    lines.append("=" * 70)
    lines.append("MULTINOMIAL MODELING -- CV ANALYSIS SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    lines.append("## Model Comparison (averaged across snapshot dates)")
    lines.append("")
    lines.append(format_comparison_table(averaged))
    lines.append("")

    lines.append("## Binary Models: Raw vs. Normalized")
    lines.append("")
    lines.append(format_normalized_table(norm_df))
    lines.append("")

    lines.append("## Per-Snapshot Detail")
    lines.append("")
    if not per_snapshot.empty:
        for snap_date in sorted(per_snapshot["snapshot_date"].unique()):
            snap_data = per_snapshot[per_snapshot["snapshot_date"] == snap_date]
            lines.append(f"### {snap_date}")
            for _, row in snap_data.iterrows():
                model = MODEL_DISPLAY.get(str(row["model_type"]), str(row["model_type"]))
                lines.append(
                    f"  {model:20s}  acc={row['accuracy']:.1%}  "
                    f"brier={row['brier_score']:.4f}  "
                    f"prob_sum={row['prob_sum_mean']:.2f}"
                )
            lines.append("")

    summary_text = "\n".join(lines)
    (output_dir / "summary.txt").write_text(summary_text)
    print(summary_text)

    if not per_snapshot.empty:
        per_snapshot.to_csv(output_dir / "per_snapshot_metrics.csv", index=False)
    if not averaged.empty:
        averaged.to_csv(output_dir / "averaged_metrics.csv")
    if not norm_df.empty:
        norm_df.to_csv(output_dir / "normalized_comparison.csv")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CV results for multinomial comparison")
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="storage/d20260217_multinomial_modeling",
        help="Experiment directory with models/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="storage/d20260217_multinomial_modeling/cv_analysis",
        help="Output directory for analysis artifacts",
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Multinomial Modeling -- CV Analysis")
    print("=" * 70)
    print(f"  Experiment dir: {exp_dir}")
    print(f"  Output dir: {output_dir}")
    print()

    snapshot_dates = get_available_snapshots(exp_dir)
    if not snapshot_dates:
        print("ERROR: No snapshot dates found")
        return
    print(f"  Found {len(snapshot_dates)} snapshot dates")

    for mt in MODEL_TYPES:
        model_dir = exp_dir / "models" / mt
        n_dates = len(list(model_dir.iterdir())) if model_dir.exists() else 0
        print(f"  {get_model_display(mt)}: {n_dates} snapshots")
    print()

    # Build comparison tables
    print("Building comparison tables...")
    per_snapshot, averaged = build_comparison_table(exp_dir, snapshot_dates)
    if averaged.empty:
        print("ERROR: No metrics found")
        return

    # Build normalization comparison
    print("Computing post-hoc normalization...")
    norm_df = build_normalized_comparison(exp_dir, snapshot_dates)

    # Write summary
    print()
    write_summary(averaged, norm_df, per_snapshot, output_dir)

    # Generate plots
    print("\nGenerating plots...")
    if not per_snapshot.empty:
        plot_metrics_over_time(
            per_snapshot, MODEL_TYPES, output_path=output_dir / "metrics_over_time.png"
        )

        # Build prob sum data for plot_prob_sum_distribution
        cv_prob_sums: dict[str, list[float]] = {}
        for model_type in MODEL_TYPES:
            sums: list[float] = []
            for snap_date in snapshot_dates:
                preds = load_cv_predictions(exp_dir, model_type, snap_date)
                if preds is None:
                    continue
                ceremony_sums = preds.groupby("ceremony")["probability"].sum()
                sums.extend(ceremony_sums.tolist())
            cv_prob_sums[model_type] = sums
        plot_prob_sum_distribution(
            cv_prob_sums, output_path=output_dir / "prob_sum_distribution.png"
        )

    # Final predictions comparison for latest snapshot
    latest_snapshot = snapshot_dates[-1]
    print(f"\n  Final predictions comparison for {latest_snapshot}:")
    combined = build_final_predictions_table(exp_dir, latest_snapshot)
    if combined is not None:
        print(combined.to_string(float_format=lambda x: f"{x:.3f}"))
        combined.to_csv(output_dir / f"predictions_{latest_snapshot}.csv")
        plot_final_predictions_comparison(
            combined, latest_snapshot, output_path=output_dir / f"predictions_{latest_snapshot}.png"
        )

    print(f"\nAnalysis complete. Results in: {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
