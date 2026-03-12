"""Collect results from temporal model snapshot runs into a single CSV.

Walks the model output directory tree, extracts test predictions, CV metrics,
and feature counts for each model/date combination.

Output: model_predictions_timeseries.csv with columns:
  snapshot_date, model_type, feature_count, cv_accuracy, cv_logloss, cv_brier,
  ceremony, year, film_id, title, probability, rank

Usage:
    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\
d20260211_temporal_model_snapshots.collect_results \
        --models-dir storage/d20260211_temporal_model_snapshots/models \
        --output storage/d20260211_temporal_model_snapshots/model_predictions_timeseries.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def collect_snapshot_results(models_dir: Path) -> pd.DataFrame:
    """Collect results from all model/date snapshots.

    Expected directory structure:
        models_dir/{model_type}/{date}/{model_type}_{date}/
            4_selected_cv/metrics.json     (CV metrics)
            5_final_predict/predictions_test.csv (test predictions)
            5_final_predict/config.json    (feature count)

    Args:
        models_dir: Root directory containing model outputs (e.g., storage/.../models)

    Returns:
        DataFrame with all predictions annotated with snapshot metadata
    """
    all_rows: list[dict] = []

    for model_dir in sorted(models_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_type = model_dir.name  # "lr" or "gbt"

        for date_dir in sorted(model_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            snapshot_date = date_dir.name  # "2025-11-30"

            # Find the run directory (model_type_date or just the date)
            run_name = f"{model_type}_{snapshot_date}"
            run_dir = date_dir / run_name

            if not run_dir.is_dir():
                # Try alternative naming
                candidates = [d for d in date_dir.iterdir() if d.is_dir()]
                if len(candidates) == 1:
                    run_dir = candidates[0]
                else:
                    print(f"  SKIP {model_type}/{snapshot_date}: no run directory found")
                    continue

            # Load CV metrics from selected-feature CV (step 4)
            cv_metrics_path = run_dir / "4_selected_cv" / "metrics.json"
            cv_accuracy = None
            cv_logloss = None
            cv_brier = None
            if cv_metrics_path.exists():
                with open(cv_metrics_path) as f:
                    metrics = json.load(f)
                # Metrics are nested under "micro" (pooled across all years)
                micro = metrics.get("micro", {})
                cv_accuracy = micro.get("accuracy")
                cv_logloss = micro.get("log_loss")
                cv_brier = micro.get("brier_score")

            # Load feature count from final predict config (step 5)
            config_path = run_dir / "5_final_predict" / "config.json"
            feature_count = None
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                feature_count = config.get("feature_count")

            # Load feature importance from final predict (step 5)
            fi_path = run_dir / "5_final_predict" / "feature_importance.csv"
            feature_importance: dict[str, float] = {}
            if fi_path.exists():
                fi_df = pd.read_csv(fi_path)
                # LR uses abs_coefficient, GBT uses importance
                importance_col = (
                    "abs_coefficient" if "abs_coefficient" in fi_df.columns else "importance"
                )
                for _, row in fi_df.iterrows():
                    feature_importance[row["feature"]] = float(row[importance_col])

            # Load test predictions (step 5)
            preds_path = run_dir / "5_final_predict" / "predictions_test.csv"
            if not preds_path.exists():
                print(f"  SKIP {model_type}/{snapshot_date}: no predictions_test.csv")
                continue

            preds_df = pd.read_csv(preds_path)

            for _, pred_row in preds_df.iterrows():
                row_dict = {
                    "snapshot_date": snapshot_date,
                    "model_type": model_type,
                    "feature_count": feature_count,
                    "cv_accuracy": cv_accuracy,
                    "cv_logloss": cv_logloss,
                    "cv_brier": cv_brier,
                    "ceremony": pred_row["ceremony"],
                    "year": pred_row["year"],
                    "film_id": pred_row.get("film_id", ""),
                    "title": pred_row["title"],
                    "probability": pred_row["probability"],
                    "rank": pred_row["rank"],
                    "is_actual_winner": pred_row.get("is_actual_winner", False),
                }
                all_rows.append(row_dict)

            # Store feature importance as a separate JSON for later analysis
            if feature_importance:
                fi_out = run_dir / "5_final_predict" / "feature_importance_dict.json"
                if not fi_out.exists():
                    with open(fi_out, "w") as f:
                        json.dump(feature_importance, f, indent=2)

            n_preds = len(preds_df)
            print(
                f"  {model_type}/{snapshot_date}: "
                f"{n_preds} predictions, {feature_count} features, "
                f"CV acc={cv_accuracy}"
            )

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values(["model_type", "snapshot_date", "rank"]).reset_index(drop=True)
    return df


def main() -> None:
    """Collect results and write timeseries CSV."""
    parser = argparse.ArgumentParser(
        description="Collect temporal model snapshot results into a single CSV"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        required=True,
        help="Root directory with model outputs (e.g., storage/.../models)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV path",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    output_path = Path(args.output)

    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    print(f"Collecting results from {models_dir}...")
    df = collect_snapshot_results(models_dir)

    if df.empty:
        print("WARNING: No results collected!")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Summary
    n_snapshots = df["snapshot_date"].nunique()
    n_models = df["model_type"].nunique()
    print(f"\nCollected {len(df)} rows: {n_snapshots} snapshots × {n_models} models")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
