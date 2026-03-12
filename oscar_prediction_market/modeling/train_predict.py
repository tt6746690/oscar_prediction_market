"""Train and/or predict for Oscar Best Picture prediction.

A unified script for training models and making predictions.

Modes:
- train: Train model, save to disk (no predictions)
- predict: Load trained model, make predictions
- both: Train model, save to disk, and make predictions on test years

Config Sources:
- Vanilla: Specify --model-config and --feature-config explicitly
- CV Output: Use --cv-output to load best config from evaluate_cv.py output
- Train Output: Use --train-output to load model from previous training (predict mode only)

Outputs (saved to output directory):
- model.pkl: Trained model artifact (train/both modes)
- config.json: Full configuration
- feature_importance.csv: Feature importance from trained model (train/both modes)
- predictions.csv: Predictions (predict/both modes)
- summary.txt: Human-readable report

Usage:
    # Train only
    uv run python -m oscar_prediction_market.modeling.train_predict \\
        --mode train \\
        --model-config configs/models/logistic_regression_default.json \\
        --feature-config configs/features/bp_lr_standard.json \\
        --train-years 2000-2025 \\
        --output-dir storage/model_2026-02-07

    # Train and predict
    uv run python -m oscar_prediction_market.modeling.train_predict \\
        --mode both \\
        --cv-output storage/gbt_loyo_tuning \\
        --train-years 2000-2025 \\
        --test-years 2026 \\
        --output-dir storage/model_from_cv

    # Predict only (from trained model)
    uv run python -m oscar_prediction_market.modeling.train_predict \\
        --mode predict \\
        --train-output storage/model_2026-02-07 \\
        --test-years 2026 \\
        --output-dir storage/predictions_2026
"""

import argparse
import json
import logging
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from oscar_prediction_market.data.awards_calendar import (
    CALENDARS,
    AwardsCalendar,
)
from oscar_prediction_market.modeling.data_loader import (
    filter_feature_set_by_availability,
    load_data,
    prepare_model_data,
    print_dataset_summary,
)
from oscar_prediction_market.modeling.evaluation import (
    YearPrediction,
    predictions_to_dataframe,
)
from oscar_prediction_market.modeling.feature_engineering import (
    FeatureSet,
)
from oscar_prediction_market.modeling.feature_engineering.types import (
    FeatureFamily,
)
from oscar_prediction_market.modeling.models import (
    BaggedClassifierModel,
    ModelConfig,
    ModelType,
    create_model,
    load_model,
    load_model_config,
    model_config_adapter,
    save_model,
    validate_model_feature_consistency,
)
from oscar_prediction_market.modeling.utils import (
    ceremony_to_year,
    parse_year_range,
    resolve_config_path,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CLI Parsing
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and/or predict for Oscar Best Picture prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train only
    python train_predict.py --mode train \\
        --model-config configs/models/logistic_regression_default.json \\
        --feature-config configs/features/bp_lr_standard.json \\
        --train-years 2000-2025 \\
        --output-dir storage/model

    # Train and predict
    python train_predict.py --mode both \\
        --cv-output storage/gbt_loyo_tuning \\
        --train-years 2000-2025 \\
        --test-years 2026 \\
        --output-dir storage/model_with_preds

    # Predict only
    python train_predict.py --mode predict \\
        --train-output storage/model \\
        --test-years 2026 \\
        --output-dir storage/predictions
        """,
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "predict", "both"],
        required=True,
        help="Mode: train (train only), predict (predict only), both (train and predict)",
    )

    # Config source - mutually exclusive options
    config_group = parser.add_argument_group(
        "config source",
        "Specify one of: --cv-output, --train-output, or (--model-config + --feature-config)",
    )
    config_group.add_argument(
        "--cv-output",
        type=str,
        default=None,
        help="Path to evaluate_cv.py output directory (loads best_config.json)",
    )
    config_group.add_argument(
        "--train-output",
        type=str,
        default=None,
        help="Path to previous train output directory (for predict mode)",
    )
    config_group.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to model configuration JSON file",
    )
    config_group.add_argument(
        "--feature-config",
        type=str,
        default=None,
        help="Path to feature set configuration JSON file",
    )

    # Year ranges
    parser.add_argument(
        "--train-years",
        type=str,
        default=None,
        help="Training year range, e.g., '2000-2025' (required for train/both modes)",
    )
    parser.add_argument(
        "--test-years",
        type=str,
        default=None,
        help="Test year(s) for prediction, e.g., '2026' or '2025-2026' (required for predict/both)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results",
    )

    # Optional arguments
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Date for feature availability filtering (YYYY-MM-DD). Overrides CV/train output.",
    )
    parser.add_argument(
        "--ceremony-year",
        type=int,
        default=None,
        help="Ceremony year for calendar resolution (required with --as-of-date)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        required=True,
        help="Path to raw dataset JSON file",
    )

    args = parser.parse_args()

    # Validate argument combinations
    mode = args.mode

    # train-output only valid for predict mode
    if args.train_output and mode != "predict":
        parser.error("--train-output can only be used with --mode predict")

    # For train/both modes, need train-years
    if mode in ("train", "both") and not args.train_years:
        parser.error(f"--train-years is required for --mode {mode}")

    # For predict/both modes, need test-years
    if mode in ("predict", "both") and not args.test_years:
        parser.error(f"--test-years is required for --mode {mode}")

    # Validate config source
    if args.train_output:
        if args.cv_output or args.model_config or args.feature_config:
            parser.error("--train-output cannot be used with other config options")
    elif args.cv_output:
        if args.model_config or args.feature_config:
            parser.error("--cv-output cannot be used with --model-config or --feature-config")
    else:
        # Neither train-output nor cv-output provided - need both model-config and feature-config
        if mode != "predict":
            if not args.model_config or not args.feature_config:
                parser.error(
                    "Either --cv-output OR both --model-config and --feature-config required"
                )

    return args


# ============================================================================
# Config Loading
# ============================================================================


def load_from_cv_output(cv_output_dir: Path) -> tuple[ModelConfig, FeatureSet, date | None]:
    """Load model config, feature set, and as_of_date from evaluate_cv.py output.

    Args:
        cv_output_dir: Path to evaluate_cv.py output directory

    Returns:
        (model_config, feature_set, as_of_date) tuple
    """
    best_config_path = cv_output_dir / "best_config.json"
    if not best_config_path.exists():
        raise FileNotFoundError(
            f"best_config.json not found in {cv_output_dir}. "
            "Make sure this is an evaluate_cv.py output directory."
        )

    with open(best_config_path) as f:
        best_config = json.load(f)

    # Load model config using Pydantic discriminated union (handles nested configs like bagged)
    model_config_dict = best_config["model_config"]
    model_config: ModelConfig = model_config_adapter.validate_python(model_config_dict)

    # Load feature set
    feature_set_dict = best_config["feature_set"]
    feature_set_dict.setdefault(
        "description", f"Feature set from CV output: {feature_set_dict['name']}"
    )
    feature_set = FeatureSet(**feature_set_dict)

    # Load as_of_date
    as_of_date: date | None = None
    if best_config.get("as_of_date"):
        as_of_date = date.fromisoformat(best_config["as_of_date"])

    return model_config, feature_set, as_of_date


def load_from_train_output(
    train_output_dir: Path,
) -> tuple[Path, FeatureSet, date | None, ModelType]:
    """Load model path, feature set, as_of_date and model type from previous train output.

    Args:
        train_output_dir: Path to train output directory

    Returns:
        (model_path, feature_set, as_of_date, model_type) tuple
    """
    model_path = train_output_dir / "model.pkl"
    config_path = train_output_dir / "config.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"model.pkl not found in {train_output_dir}. "
            "Make sure this is a train output directory."
        )
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.json not found in {train_output_dir}. "
            "Make sure this is a train output directory."
        )

    with open(config_path) as f:
        train_config = json.load(f)

    # Extract model_type from model_config to determine feature family
    model_config = train_config.get("model_config", {})
    model_type_str = model_config.get("model_type", "logistic_regression")
    model_type = ModelType(model_type_str)

    # Build feature set from train config — use feature_family instead of model_type
    feature_set = FeatureSet(
        name=train_config.get("feature_set", "from_train_output"),
        description=f"Feature set from train output: {train_output_dir.name}",
        features=train_config["features"],
        feature_family=model_type.feature_family,
    )

    # Load as_of_date
    as_of_date: date | None = None
    if train_config.get("as_of_date"):
        as_of_date = date.fromisoformat(train_config["as_of_date"])

    return model_path, feature_set, as_of_date, model_type


# ============================================================================
# Model Operations
# ============================================================================


def train_model(
    df: pd.DataFrame,
    feature_set: FeatureSet,
    model_config: ModelConfig,
    train_ceremonies: list[int],
) -> tuple:
    """Train model on specified ceremonies.

    Returns:
        (model, X_train, y_train) tuple
    """
    train_df = df[df["ceremony"].isin(train_ceremonies)].copy()
    X_train, y_train, meta_train = prepare_model_data(train_df, feature_set)

    model = create_model(model_config)
    model.fit(X_train, y_train, groups=np.asarray(meta_train["ceremony"].values))

    return model, X_train, y_train


def predict_year(
    model,
    df: pd.DataFrame,
    feature_set: FeatureSet,
    ceremony: int,
) -> YearPrediction | None:
    """Make predictions for a single ceremony year.

    Returns:
        YearPrediction or None if no data for that ceremony
    """
    test_df = df[df["ceremony"] == ceremony].copy()
    if len(test_df) == 0:
        return None

    X_test, y_test, metadata_test = prepare_model_data(test_df, feature_set)
    probabilities = model.predict_proba(X_test, groups=np.asarray(metadata_test["ceremony"].values))
    y_test_array = np.asarray(y_test.values)
    actual_winner_idx = int(np.argmax(y_test_array))

    return YearPrediction(
        ceremony=ceremony,
        film_ids=metadata_test["film_id"].tolist(),
        titles=metadata_test["title"].tolist(),
        probabilities=probabilities.tolist(),
        actual_winner_idx=actual_winner_idx,
        y_true=y_test_array.tolist(),
    )


# ============================================================================
# Output
# ============================================================================


def save_config(
    output_dir: Path,
    mode: str,
    model_config: ModelConfig | None,
    feature_set: FeatureSet,
    train_years: tuple[int, int] | None,
    test_years: tuple[int, int] | None,
    as_of_date: date | None,
    train_ceremonies: list[int],
    test_ceremonies: list[int],
) -> None:
    """Save configuration to JSON."""
    config_dict = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "model_config": model_config.model_dump() if model_config else None,
        "feature_set": feature_set.name,
        "feature_count": len(feature_set.features),
        "features": feature_set.features,
        "train_years": list(train_years) if train_years else None,
        "test_years": list(test_years) if test_years else None,
        "as_of_date": as_of_date.isoformat() if as_of_date else None,
        "train_ceremonies": [int(c) for c in train_ceremonies],
        "test_ceremonies": [int(c) for c in test_ceremonies],
        "num_train_ceremonies": len(train_ceremonies),
        "num_test_ceremonies": len(test_ceremonies),
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)


def save_feature_importance(model, feature_names: list[str], output_dir: Path) -> None:
    """Save feature importance to CSV."""
    fi_df = model.get_feature_importance(feature_names)
    fi_df.to_csv(output_dir / "feature_importance.csv", index=False)


def save_predictions(
    predictions: list[YearPrediction],
    output_dir: Path,
    prefix: str = "",
) -> None:
    """Save predictions to CSV."""
    df_preds = predictions_to_dataframe(predictions)
    filename = f"predictions_{prefix}.csv" if prefix else "predictions.csv"
    df_preds.to_csv(output_dir / filename, index=False)


def save_bag_distribution(
    model: BaggedClassifierModel,
    df: pd.DataFrame,
    feature_set: FeatureSet,
    ceremonies: list[int],
    output_dir: Path,
    prefix: str = "",
) -> None:
    """Save per-bag probability distribution for bagged models.

    For each nominee in each ceremony, saves per-bag probabilities along with
    summary statistics (mean, std, min, max, q25, q75). This enables
    uncertainty analysis on the ensemble's predictions.

    Output CSV columns: ceremony, year, title, prob_mean, prob_std, prob_min,
    prob_q25, prob_median, prob_q75, prob_max, bag_1, bag_2, ...
    """
    rows = []
    for ceremony in ceremonies:
        test_df = df[df["ceremony"] == ceremony].copy()
        if len(test_df) == 0:
            continue

        X_test, _, metadata_test = prepare_model_data(test_df, feature_set)
        dist = model.predict_proba_distribution(X_test)  # (n_samples, n_bags)

        for i in range(len(X_test)):
            bag_probs = dist[i]
            row: dict = {
                "ceremony": ceremony,
                "year": ceremony_to_year(ceremony),
                "title": metadata_test["title"].iloc[i],
                "prob_mean": float(np.mean(bag_probs)),
                "prob_std": float(np.std(bag_probs)),
                "prob_min": float(np.min(bag_probs)),
                "prob_q25": float(np.percentile(bag_probs, 25)),
                "prob_median": float(np.median(bag_probs)),
                "prob_q75": float(np.percentile(bag_probs, 75)),
                "prob_max": float(np.max(bag_probs)),
            }
            for b in range(dist.shape[1]):
                row[f"bag_{b + 1}"] = float(dist[i, b])
            rows.append(row)

    df_dist = pd.DataFrame(rows)
    filename = f"bag_distribution_{prefix}.csv" if prefix else "bag_distribution.csv"
    df_dist.to_csv(output_dir / filename, index=False)
    print(f"Saved bag distribution to: {output_dir / filename}")


def print_predictions(predictions: list[YearPrediction]) -> None:
    """Print predictions in a readable format."""
    print("\n" + "=" * 80)
    print("PREDICTIONS")
    print("=" * 80)

    for pred in predictions:
        print(f"\nCeremony {pred.ceremony} ({pred.year})")
        print("-" * 60)
        print(f"{'Rank':<6} {'Title':<40} {'Prob':<10} {'Winner'}")
        print("-" * 60)

        # Sort by rank
        sorted_indices = sorted(range(len(pred.titles)), key=lambda i: pred.predicted_ranks[i])
        for idx in sorted_indices:
            rank = pred.predicted_ranks[idx]
            title = pred.titles[idx][:38]
            prob = pred.probabilities[idx]
            is_winner = idx == pred.actual_winner_idx
            winner_mark = "★" if is_winner else ""
            print(f"{rank:<6} {title:<40} {prob:>8.1%}  {winner_mark}")

        print(f"\nPredicted winner: {pred.top_predicted_title}")
        print(f"Actual winner: {pred.winner_title}")
        if pred.is_correct:
            print("✓ Correct!")
        else:
            print(f"✗ Incorrect (actual winner ranked #{pred.winner_predicted_rank})")


def generate_summary_report(
    output_dir: Path,
    mode: str,
    model_config: ModelConfig | None,
    feature_set: FeatureSet,
    train_years: tuple[int, int] | None,
    test_years: tuple[int, int] | None,
    train_predictions: list[YearPrediction],
    test_predictions: list[YearPrediction],
    feature_importance: pd.DataFrame | None,
) -> str:
    """Generate human-readable summary report."""
    lines = [
        "=" * 80,
        "MODEL SUMMARY",
        "=" * 80,
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Mode: {mode}",
        f"Output directory: {output_dir}",
        "",
        "CONFIGURATION",
        "-" * 50,
    ]

    if model_config:
        lines.append(f"Model: {model_config.model_type}")
    lines.append(f"Feature set: {feature_set.name} ({len(feature_set.features)} features)")
    if train_years:
        lines.append(f"Train years: {train_years[0]}-{train_years[1]}")
    if test_years:
        lines.append(f"Test years: {test_years[0]}-{test_years[1]}")
    lines.append("")

    # Model parameters (if available)
    if model_config:
        lines.extend(["MODEL PARAMETERS", "-" * 50])
        for k, v in model_config.model_dump().items():
            if k != "model_type":
                lines.append(f"  {k}: {v}")
        lines.append("")

    # Training accuracy
    if train_predictions:
        train_correct = sum(1 for p in train_predictions if p.is_correct)
        train_total = len(train_predictions)
        lines.extend(
            [
                "TRAINING SET PERFORMANCE",
                "-" * 50,
                f"Accuracy: {train_correct}/{train_total} ({100 * train_correct / train_total:.1f}%)",
                "",
            ]
        )

    # Test predictions
    if test_predictions:
        test_correct = sum(1 for p in test_predictions if p.is_correct)
        test_total = len(test_predictions)
        lines.extend(
            [
                "TEST SET PREDICTIONS",
                "-" * 80,
                f"{'Year':<6} {'Winner':<30} {'Predicted':<30} {'Prob':<8}",
                "-" * 80,
            ]
        )
        for pred in test_predictions:
            status = "✓" if pred.is_correct else "✗"
            lines.append(
                f"{pred.year:<6} {pred.winner_title[:28]:<30} "
                f"{pred.top_predicted_title[:28]:<30} {pred.winner_probability:>6.1%} {status}"
            )
        if test_total > 0:
            lines.append(
                f"\nTest Accuracy: {test_correct}/{test_total} ({100 * test_correct / test_total:.1f}%)"
            )
        lines.append("")

    # Top features
    if feature_importance is not None and len(feature_importance) > 0:
        lines.extend(
            [
                "TOP 10 FEATURES",
                "-" * 50,
            ]
        )
        for idx, row in enumerate(feature_importance.head(10).itertuples()):
            imp = row.importance
            lines.append(f"{idx + 1:>2}. {row.feature:<40} {imp:.4f}")

    lines.extend(["", "=" * 80])

    report = "\n".join(lines)

    with open(output_dir / "summary.txt", "w") as f:
        f.write(report)

    return report


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s")

    mode = args.mode
    model_config: ModelConfig | None = None
    feature_set: FeatureSet
    as_of_date: date | None = None
    feature_family: FeatureFamily
    model = None

    # Load configuration based on source
    if args.train_output:
        # Predict mode with previous train output
        train_output_dir = Path(args.train_output)
        print(f"Loading from train output: {train_output_dir}")
        model_path, feature_set, train_as_of_date, loaded_model_type = load_from_train_output(
            train_output_dir
        )
        model = load_model(model_path)
        feature_family = loaded_model_type.feature_family

        # Use as_of_date from train output unless explicitly overridden
        as_of_date = date.fromisoformat(args.as_of_date) if args.as_of_date else train_as_of_date
        print(f"  Model: {model_path}")
        print(f"  Features: {len(feature_set.features)} features from {feature_set.name}")
        if as_of_date:
            print(f"  As-of date: {as_of_date}")

    elif args.cv_output:
        # Load from CV output
        cv_output_dir = Path(args.cv_output)
        print(f"Loading configuration from CV output: {cv_output_dir}")
        model_config, feature_set, cv_as_of_date = load_from_cv_output(cv_output_dir)

        # Use as_of_date from CV output unless explicitly overridden
        as_of_date = date.fromisoformat(args.as_of_date) if args.as_of_date else cv_as_of_date

        # Validate and infer feature family
        feature_family = validate_model_feature_consistency(
            [model_config], feature_set.feature_family
        )

        print(f"  Model: {model_config.model_type}")
        print(f"  Features: {len(feature_set.features)} features from {feature_set.name}")
        if as_of_date:
            print(f"  As-of date: {as_of_date}")

    else:
        # Vanilla mode - load from explicit config files
        model_config_path = resolve_config_path(args.model_config)  # type: ignore
        feature_config_path = resolve_config_path(args.feature_config)  # type: ignore

        logger.info(f"Loading model config: {model_config_path}")
        model_config = load_model_config(model_config_path)

        logger.info(f"Loading feature config: {feature_config_path}")
        with open(feature_config_path) as f:
            feature_config_dict = json.load(f)
        feature_set = FeatureSet(**feature_config_dict)

        # Parse as_of_date if provided
        if args.as_of_date:
            as_of_date = date.fromisoformat(args.as_of_date)
            logger.info(f"Feature availability filtering as of: {as_of_date}")

        # Validate and infer feature family
        feature_family = validate_model_feature_consistency(
            [model_config], feature_set.feature_family
        )

    # Filter feature set by availability if as_of_date is specified
    # Resolve calendar for availability filtering
    calendar: AwardsCalendar | None = None
    if args.ceremony_year:
        if args.ceremony_year not in CALENDARS:
            raise ValueError(
                f"Unknown ceremony year {args.ceremony_year}. Available: {sorted(CALENDARS.keys())}"
            )
        calendar = CALENDARS[args.ceremony_year]
    if as_of_date:
        if calendar is None:
            raise ValueError(
                "--ceremony-year is required when --as-of-date is set "
                "(needed for calendar-based feature availability resolution)"
            )
        feature_set, available, unavailable = filter_feature_set_by_availability(
            feature_set, as_of_date, calendar=calendar
        )
        if unavailable:
            print(f"\nFiltered {len(unavailable)} features not available as of {as_of_date}:")
            for feat in unavailable[:5]:  # Show first 5
                print(f"  - {feat}")
            if len(unavailable) > 5:
                print(f"  ... and {len(unavailable) - 5} more")
            print(f"Using {len(available)} available features")

    # Parse year ranges
    train_years = parse_year_range(args.train_years) if args.train_years else None
    test_years = parse_year_range(args.test_years) if args.test_years else None

    logger.info(f"Using {feature_family.upper()} feature engineering")

    # Load data
    print("Loading dataset with model-specific feature engineering...")
    df = load_data(
        raw_path=Path(args.raw_path),
        calendar=calendar,
    )
    print_dataset_summary(df)

    # Convert years to ceremonies
    train_ceremonies: list[int] = []
    if train_years:
        train_ceremonies = [
            c
            for c in df["ceremony"].unique()
            if train_years[0] <= ceremony_to_year(c) <= train_years[1]
        ]
        train_ceremonies = sorted(train_ceremonies)

    test_ceremonies: list[int] = []
    if test_years:
        test_ceremonies = [
            c
            for c in df["ceremony"].unique()
            if test_years[0] <= ceremony_to_year(c) <= test_years[1]
        ]
        test_ceremonies = sorted(test_ceremonies)

    if train_years:
        print(
            f"\nTrain years: {train_years[0]}-{train_years[1]} ({len(train_ceremonies)} ceremonies)"
        )
    if test_years:
        print(f"Test years: {test_years[0]}-{test_years[1]} ({len(test_ceremonies)} ceremonies)")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Execute based on mode
    train_predictions: list[YearPrediction] = []
    test_predictions: list[YearPrediction] = []
    fi_df: pd.DataFrame | None = None

    if mode in ("train", "both"):
        # Train model
        assert model_config is not None
        print("\n" + "=" * 70)
        print("Training model...")
        print("=" * 70)

        model, X_train, y_train = train_model(df, feature_set, model_config, train_ceremonies)
        print(f"Trained on {len(X_train)} samples")

        # Save model
        model_path = output_dir / "model.pkl"
        save_model(model, model_path)
        print(f"Saved model to: {model_path}")

        # Get feature importance
        feature_names = list(X_train.columns)
        fi_df = model.get_feature_importance(feature_names)
        save_feature_importance(model, feature_names, output_dir)

        # Generate train predictions
        print("\nGenerating predictions on training data...")
        for ceremony in train_ceremonies:
            pred = predict_year(model, df, feature_set, ceremony)
            if pred:
                train_predictions.append(pred)
        save_predictions(train_predictions, output_dir, prefix="train")

        # Save per-bag distribution for bagged models (training data)
        if isinstance(model, BaggedClassifierModel):
            save_bag_distribution(
                model, df, feature_set, train_ceremonies, output_dir, prefix="train"
            )

    if mode in ("predict", "both"):
        # Make predictions
        if model is None:
            raise ValueError("No model loaded for prediction")

        print("\n" + "=" * 70)
        print("Making predictions...")
        print("=" * 70)

        for ceremony in test_ceremonies:
            pred = predict_year(model, df, feature_set, ceremony)
            if pred:
                test_predictions.append(pred)

        if not test_predictions:
            raise ValueError(f"No predictions generated for years {test_years}")

        save_predictions(test_predictions, output_dir, prefix="test")

        # Save per-bag distribution for bagged models
        if isinstance(model, BaggedClassifierModel):
            save_bag_distribution(
                model, df, feature_set, test_ceremonies, output_dir, prefix="test"
            )

        print_predictions(test_predictions)

    # Save config
    save_config(
        output_dir,
        mode,
        model_config,
        feature_set,
        train_years,
        test_years,
        as_of_date,
        train_ceremonies,
        test_ceremonies,
    )

    # Generate summary
    report = generate_summary_report(
        output_dir,
        mode,
        model_config,
        feature_set,
        train_years,
        test_years,
        train_predictions,
        test_predictions,
        fi_df,
    )
    print("\n" + report)

    print(f"\nAll outputs saved to: {output_dir}")
    print("\nFiles created:")
    for output_file in sorted(output_dir.iterdir()):
        print(f"  - {output_file.name}")


if __name__ == "__main__":
    main()
