"""Cross-validation evaluation for Oscar Best Picture prediction.

Runs CV to evaluate model configurations and estimate performance.
Supports single configs or parameter grids (non-nested and nested CV).

Outputs (saved to experiment directory):
- config.json: Full experiment configuration (model config as object, not path)
- best_config.json: Best model configuration (for use with train.py --cv-output)
- predictions.csv: Detailed predictions per year
- metrics.json: Evaluation metrics
- summary.txt: Human-readable report
- nested_cv_result.json: (if nested mode) Full nested CV details

Usage:
    # Single config with LOYO CV
    uv run python -m oscar_prediction_market.modeling.evaluate_cv \
        --model-config configs/models/logistic_regression_default.json \
        --feature-config configs/features/bp_lr_standard.json \
        --cv-split configs/cv_splits/leave_one_year_out.json \
        --name lr_loyo \
        --output-dir storage/my_experiment

    # Grid search (non-nested)
    uv run python -m oscar_prediction_market.modeling.evaluate_cv \
        --model-config configs/param_grids/gbt_grid.json \
        --feature-config configs/features/bp_gbt_standard.json \
        --cv-split configs/cv_splits/leave_one_year_out.json \
        --cv-mode simple \
        --name gbt_tuning

    # Nested CV for unbiased hyperparameter evaluation
    uv run python -m oscar_prediction_market.modeling.evaluate_cv \
        --model-config configs/param_grids/lr_grid.json \
        --feature-config configs/features/bp_lr_standard.json \
        --cv-split configs/cv_splits/expanding_window.json \
        --cv-mode nested \
        --name lr_nested_tuning
"""

import argparse
import ast
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pydantic import BaseModel, Field

from oscar_prediction_market.constants import STORAGE_DIR
from oscar_prediction_market.data.awards_calendar import (
    CALENDARS,
    AwardsCalendar,
)
from oscar_prediction_market.modeling.cv_splitting import (
    CVSplitConfig,
    CVSplitter,
    LeaveOneYearOutSplitter,
    create_splitter,
)
from oscar_prediction_market.modeling.data_loader import (
    filter_feature_set_by_availability,
    get_ceremony_years,
    load_data,
    prepare_model_data,
    print_dataset_summary,
    save_engineered_features,
)
from oscar_prediction_market.modeling.evaluation import (
    EvaluationMetrics,
    YearPrediction,
    predictions_to_dataframe,
)
from oscar_prediction_market.modeling.feature_engineering import (
    FeatureSet,
)
from oscar_prediction_market.modeling.feature_engineering.types import (
    FeatureFamily,
)
from oscar_prediction_market.modeling.hyperparameter_tuning import (
    CombinedScore,
    NestedCVResult,
    run_cv_for_config,
    run_nested_cv,
    select_best_config,
)
from oscar_prediction_market.modeling.models import (
    ModelConfig,
    create_model,
    load_model_config_grid,
    model_config_adapter,
    validate_model_feature_consistency,
)
from oscar_prediction_market.modeling.utils import (
    ceremony_to_year,
    load_cv_split_config,
    parse_year_range,
    resolve_config_path,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Result Models
# ============================================================================


class CVResult(BaseModel):
    """Results from cross-validation evaluation.

    All fields required - no defaults.
    """

    cv_mode: Literal["simple", "nested"] = Field(..., description="CV mode used")
    model_config_used: dict = Field(..., description="Best model configuration as dict")
    feature_set_name: str = Field(..., description="Feature set name used")
    splitter_name: str = Field(..., description="CV splitter name")
    splitter_config: dict = Field(..., description="CV splitter configuration")
    predictions: list[YearPrediction] = Field(..., description="All year predictions")
    metrics: EvaluationMetrics = Field(..., description="Aggregated metrics")
    num_configs_evaluated: int = Field(..., description="Number of configs evaluated")
    all_config_results: list[dict] | None = Field(
        default=None, description="Results for all configs (if grid)"
    )
    nested_cv_result: NestedCVResult | None = Field(
        default=None, description="Full nested CV result (if cv_mode=nested)"
    )

    model_config = {"extra": "forbid"}


# ============================================================================
# CLI Parsing
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cross-validation evaluation for Oscar Best Picture prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single config evaluation
    python evaluate_cv.py \\
        --model-config configs/models/logistic_regression_default.json \\
        --feature-config configs/features/bp_lr_standard.json \\
        --cv-split configs/cv_splits/leave_one_year_out.json \\
        --name lr_loyo

    # Grid search (non-nested)
    python evaluate_cv.py \\
        --model-config configs/param_grids/gbt_grid.json \\
        --feature-config configs/features/bp_gbt_standard.json \\
        --cv-split configs/cv_splits/leave_one_year_out.json \\
        --name gbt_tuning

    # Nested CV with grid
    python evaluate_cv.py \\
        --model-config configs/param_grids/lr_grid.json \\
        --feature-config configs/features/bp_lr_standard.json \\
        --cv-split configs/cv_splits/expanding_window.json \\
        --cv-mode nested \\
        --name lr_nested
        """,
    )

    # Required arguments
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to model config (single) or param grid (multiple) JSON file",
    )
    parser.add_argument(
        "--feature-config",
        type=str,
        required=True,
        help="Path to feature set configuration JSON file",
    )
    parser.add_argument(
        "--cv-split",
        type=str,
        required=True,
        help="Path to CV split configuration JSON file",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Experiment name for identification",
    )

    # CV mode
    parser.add_argument(
        "--cv-mode",
        type=str,
        choices=["simple", "nested"],
        default="simple",
        help="CV mode: simple (non-nested) or nested (for unbiased hyperparameter eval)",
    )

    # Optional arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: storage/<timestamp>_<name>)",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Date for feature availability filtering (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--ceremony-year",
        type=int,
        default=None,
        help="Ceremony year for calendar resolution (required with --as-of-date)",
    )
    parser.add_argument(
        "--cv-years",
        type=str,
        default=None,
        help="Year range for CV, e.g., '2000-2025' to exclude 2026. Format: START-END",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for grid search (-1 for all CPUs)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--save-features",
        action="store_true",
        help="Save engineered features to JSON for inspection",
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        required=True,
        help="Path to raw dataset JSON file",
    )
    parser.add_argument(
        "--save-fold-importance",
        action="store_true",
        help="Save per-fold feature importance for LOYO Jaccard analysis",
    )

    return parser.parse_args()


# ============================================================================
# Experiment Directory and Output
# ============================================================================


def create_experiment_dir(name: str, output_dir: str | None) -> Path:
    """Create timestamped experiment directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_dir:
        exp_dir = Path(output_dir)
    else:
        exp_dir = STORAGE_DIR / f"{timestamp}_{name}"

    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_experiment_config(
    exp_dir: Path,
    name: str,
    cv_mode: str,
    model_configs: list[ModelConfig],
    feature_set: FeatureSet,
    cv_split_config: CVSplitConfig,
    df: pd.DataFrame,
    as_of_date: date | None,
    cv_years: tuple[int, int] | None,
    n_jobs: int,
) -> None:
    """Save experiment configuration to JSON."""
    ceremonies = get_ceremony_years(df)

    # Store model config as grid (always grid format)
    model_config_section = {
        "type": "grid",
        "model_type": model_configs[0].model_type,
        "num_configs": len(model_configs),
        "configs": [c.model_dump() for c in model_configs],
    }

    config_dict = {
        "experiment_name": name,
        "timestamp": datetime.now().isoformat(),
        "cv_mode": cv_mode,
        "model_config": model_config_section,
        "feature_set": feature_set.name,
        "feature_count": len(feature_set.features),
        "features": feature_set.features,
        "cv_split_config": cv_split_config.model_dump(),
        "data_summary": {
            "total_records": len(df),
            "ceremony_range": [min(ceremonies), max(ceremonies)],
            "year_range": [ceremony_to_year(min(ceremonies)), ceremony_to_year(max(ceremonies))],
            "num_ceremonies": len(ceremonies),
        },
        "as_of_date": as_of_date.isoformat() if as_of_date else None,
        "cv_years": list(cv_years) if cv_years else None,
        "n_jobs": n_jobs,
    }

    with open(exp_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)


def save_predictions_csv(exp_dir: Path, predictions: list[YearPrediction]) -> None:
    """Save predictions to CSV (flattened format)."""
    df_preds = predictions_to_dataframe(predictions)
    df_preds.to_csv(exp_dir / "predictions.csv", index=False)


def save_metrics(exp_dir: Path, result: CVResult) -> None:
    """Save evaluation metrics to JSON."""
    metrics_dict = result.metrics.model_dump()

    # Add year-by-year details
    metrics_dict["year_by_year"] = [
        {
            "ceremony": p.ceremony,
            "year": p.year,
            "is_correct": p.is_correct,
            "winner_rank": p.winner_predicted_rank,
            "winner_probability": p.winner_probability,
            "predicted_title": p.top_predicted_title,
            "actual_title": p.winner_title,
        }
        for p in result.predictions
    ]

    # Add grid results if present
    if result.all_config_results:
        metrics_dict["grid_results"] = result.all_config_results

    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)


def generate_summary_report(
    exp_dir: Path,
    name: str,
    cv_mode: str,
    best_config: ModelConfig,
    feature_set: FeatureSet,
    splitter: CVSplitter,
    result: CVResult,
) -> str:
    """Generate human-readable summary report."""
    m = result.metrics.micro
    mm = result.metrics.macro

    lines = [
        "=" * 80,
        f"CV EVALUATION SUMMARY: {name}",
        "=" * 80,
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Output directory: {exp_dir}",
        "",
        "CONFIGURATION",
        "-" * 50,
        f"CV Mode: {cv_mode}",
        f"Model: {best_config.model_type}",
        f"Configs evaluated: {result.num_configs_evaluated}",
        f"Feature set: {feature_set.name} ({len(feature_set.features)} features)",
        f"CV Splitter: {splitter.name}",
        "",
        "RESULTS (Micro = Pooled, Macro = Per-Year Averaged)",
        "-" * 50,
        f"Test years: {result.metrics.num_years} ceremonies",
        "",
        f"{'Metric':<25} {'Micro':<12} {'Macro':<12}",
        "-" * 50,
        f"{'Accuracy':<25} {m.accuracy:>8.1%}     {mm.accuracy:>8.1%}",
        f"{'Top-3 Accuracy':<25} {m.top_3_accuracy:>8.1%}     {mm.top_3_accuracy:>8.1%}",
        f"{'Top-5 Accuracy':<25} {m.top_5_accuracy:>8.1%}     {mm.top_5_accuracy:>8.1%}",
        f"{'MRR':<25} {m.mrr:>8.3f}     {mm.mrr:>8.3f}",
        f"{'AUC-ROC':<25} {m.auc_roc:>8.3f}     {mm.auc_roc:>8.3f}",
        f"{'Brier Score':<25} {m.brier_score:>8.4f}     {mm.brier_score:>8.4f}",
        f"{'Log Loss':<25} {m.log_loss:>8.4f}     {mm.log_loss:>8.4f}",
        f"{'Mean Winner Prob':<25} {m.mean_winner_prob:>8.1%}     {mm.mean_winner_prob:>8.1%}",
        "",
        f"Correct: {[f'{c} ({ceremony_to_year(c)})' for c in result.metrics.correct_ceremonies]}",
        f"Incorrect: {[f'{c} ({ceremony_to_year(c)})' for c in result.metrics.incorrect_ceremonies]}",
        "",
    ]

    # Add year-by-year results
    lines.extend(
        [
            "YEAR-BY-YEAR RESULTS",
            "-" * 100,
            f"{'Ceremony':<10} {'Year':<6} {'Winner':<30} {'Predicted':<30} {'Rank':<6} {'Prob':<8} {'✓/✗'}",
            "-" * 100,
        ]
    )

    for pred in result.predictions:
        status = "✓" if pred.is_correct else "✗"
        lines.append(
            f"{pred.ceremony:<10} {pred.year:<6} {pred.winner_title[:28]:<30} "
            f"{pred.top_predicted_title[:28]:<30} {pred.winner_predicted_rank:<6} "
            f"{pred.winner_probability:>6.1%}  {status}"
        )

    lines.append("")

    # Add best config details
    lines.extend(
        [
            "BEST MODEL CONFIGURATION",
            "-" * 50,
        ]
    )
    for k, v in result.model_config_used.items():
        lines.append(f"  {k}: {v}")

    lines.extend(["", "=" * 80])

    report = "\n".join(lines)

    # Save to file
    with open(exp_dir / "summary.txt", "w") as f:
        f.write(report)

    return report


# ============================================================================
# Core CV Execution
# ============================================================================


def evaluate_single_config(
    df: pd.DataFrame,
    feature_set: FeatureSet,
    model_config: ModelConfig,
    splitter: CVSplitter,
) -> tuple[ModelConfig, EvaluationMetrics, list[YearPrediction]]:
    """Evaluate a single model configuration with CV.

    Returns:
        (config, metrics, predictions) tuple
    """
    metrics, predictions = run_cv_for_config(df, feature_set, model_config, splitter)
    return model_config, metrics, predictions


def run_simple_cv_grid(
    df: pd.DataFrame,
    feature_set: FeatureSet,
    model_configs: list[ModelConfig],
    splitter: CVSplitter,
    n_jobs: int = 1,
    verbose: bool = True,
) -> CVResult:
    """Run simple (non-nested) CV for one or more configs.

    If multiple configs, runs grid search in parallel and selects best.
    """
    if verbose:
        logger.info(f"Running simple CV with {splitter.name}")
        logger.info(f"  Evaluating {len(model_configs)} configuration(s)")
        if n_jobs != 1:
            logger.info(f"  Parallel jobs: {n_jobs}")

    if len(model_configs) == 1:
        # Single config - run directly
        config, metrics, predictions = evaluate_single_config(
            df, feature_set, model_configs[0], splitter
        )
        return CVResult(
            cv_mode="simple",
            model_config_used=config.model_dump(),
            feature_set_name=feature_set.name,
            splitter_name=splitter.name,
            splitter_config=splitter.get_config().model_dump(),
            predictions=predictions,
            metrics=metrics,
            num_configs_evaluated=1,
            all_config_results=None,
            nested_cv_result=None,
        )

    # Grid search with parallelization
    results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(evaluate_single_config)(df, feature_set, config, splitter)
        for config in model_configs
    )

    # Collect results, keeping predictions keyed by config for later retrieval
    all_results: list[dict] = []
    config_metrics: list[tuple[ModelConfig, EvaluationMetrics]] = []
    config_predictions: dict[str, list[YearPrediction]] = {}

    for config, metrics, predictions in results:
        score = CombinedScore.compute(metrics)
        all_results.append(
            {
                "config": config.model_dump(),
                "accuracy": metrics.macro.accuracy,
                "log_loss": metrics.macro.log_loss,
                "combined_score": score.combined,
            }
        )
        config_metrics.append((config, metrics))
        # Key by JSON-serialized config for exact lookup
        config_predictions[json.dumps(config.model_dump(), sort_keys=True)] = predictions

    # Select best config
    best_config, best_metrics, _ = select_best_config(config_metrics)

    # Retrieve predictions from grid search (no redundant re-evaluation)
    best_key = json.dumps(best_config.model_dump(), sort_keys=True)
    best_predictions = config_predictions[best_key]

    if verbose:
        logger.info(f"\nBest config: accuracy={best_metrics.macro.accuracy:.3f}")

    return CVResult(
        cv_mode="simple",
        model_config_used=best_config.model_dump(),
        feature_set_name=feature_set.name,
        splitter_name=splitter.name,
        splitter_config=splitter.get_config().model_dump(),
        predictions=best_predictions,
        metrics=best_metrics,
        num_configs_evaluated=len(model_configs),
        all_config_results=all_results,
        nested_cv_result=None,
    )


def run_nested_cv_grid(
    df: pd.DataFrame,
    feature_set: FeatureSet,
    model_configs: list[ModelConfig],
    outer_splitter: CVSplitter,
    feature_family: FeatureFamily,
    verbose: bool = True,
) -> CVResult:
    """Run nested CV for unbiased hyperparameter evaluation."""
    if verbose:
        logger.info(f"Running nested CV with outer={outer_splitter.name}")
        logger.info(f"  Param grid size: {len(model_configs)}")

    # Inner splitter is always LOYO for maximum data usage
    inner_splitter = LeaveOneYearOutSplitter()

    # Run nested CV
    nested_result = run_nested_cv(
        df=df,
        feature_set=feature_set,
        param_grid=model_configs,
        outer_splitter=outer_splitter,
        inner_splitter=inner_splitter,
        feature_family=feature_family,
        verbose=verbose,
    )

    # Extract predictions
    predictions = []
    for fold_result in nested_result.fold_results:
        for pred_dict in fold_result.predictions:
            predictions.append(YearPrediction(**pred_dict))

    # Find most common selected config
    most_common_config_str = max(
        nested_result.config_selection_frequency,
        key=nested_result.config_selection_frequency.get,  # type: ignore
    )
    most_common_config_dict = ast.literal_eval(most_common_config_str)

    return CVResult(
        cv_mode="nested",
        model_config_used=most_common_config_dict,
        feature_set_name=feature_set.name,
        splitter_name=outer_splitter.name,
        splitter_config=outer_splitter.get_config().model_dump(),
        predictions=predictions,
        metrics=nested_result.aggregated_metrics,
        num_configs_evaluated=len(model_configs),
        all_config_results=None,
        nested_cv_result=nested_result,
    )


# ============================================================================
# Fold Importance (for LOYO Jaccard)
# ============================================================================


def save_fold_importance(
    df: pd.DataFrame,
    feature_set: FeatureSet,
    model_config: ModelConfig,
    splitter: CVSplitter,
    exp_dir: Path,
) -> None:
    """Save per-fold feature importance for LOYO Jaccard analysis.

    Re-runs the best config through each CV fold, trains a model, and saves
    the feature importance dict for each fold. This enables computing feature
    stability (Jaccard similarity) across LOYO folds.

    Output:
        {exp_dir}/fold_importance/fold_{year}.json
        Each file maps feature_name -> importance_value (abs coef for LR, importance for GBT).
    """
    fold_dir = exp_dir / "fold_importance"
    fold_dir.mkdir(parents=True, exist_ok=True)

    ceremony_years = get_ceremony_years(df)
    folds = splitter.generate_folds(ceremony_years)

    for fold in folds:
        train_df = df[df["ceremony"].isin(fold.train_ceremonies)].copy()
        X_train, y_train, meta_train = prepare_model_data(train_df, feature_set)

        model = create_model(model_config)
        model.fit(X_train, y_train, groups=np.asarray(meta_train["ceremony"].values))

        importance_df = model.get_feature_importance(list(X_train.columns))

        # All models now provide a unified 'importance' column
        importance_dict = dict(
            zip(
                importance_df["feature"],
                importance_df["importance"].astype(float),
                strict=True,
            )
        )

        for test_ceremony in fold.test_ceremonies:
            year = ceremony_to_year(test_ceremony)
            filepath = fold_dir / f"fold_{year}.json"
            with open(filepath, "w") as f:
                json.dump(importance_dict, f, indent=2)

    logger.info(f"Saved {len(list(fold_dir.glob('*.json')))} fold importance files to {fold_dir}")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s")

    # Load configs
    model_config_path = resolve_config_path(args.model_config)
    feature_config_path = resolve_config_path(args.feature_config)
    cv_split_config_path = resolve_config_path(args.cv_split)

    logger.info(f"Loading model config grid: {model_config_path}")
    model_configs = load_model_config_grid(model_config_path)
    logger.info(f"  Loaded {len(model_configs)} config(s)")

    logger.info(f"Loading feature config: {feature_config_path}")
    with open(feature_config_path) as f:
        feature_config_dict = json.load(f)
    feature_set = FeatureSet(**feature_config_dict)

    logger.info(f"Loading CV split config: {cv_split_config_path}")
    cv_split_config = load_cv_split_config(cv_split_config_path)
    splitter = create_splitter(cv_split_config)

    # Parse as_of_date if provided
    as_of_date: date | None = None
    if args.as_of_date:
        as_of_date = date.fromisoformat(args.as_of_date)
        logger.info(f"Feature availability filtering as of: {as_of_date}")

    # Resolve calendar for availability filtering
    calendar: AwardsCalendar | None = None
    if args.ceremony_year:
        if args.ceremony_year not in CALENDARS:
            raise ValueError(
                f"Unknown ceremony year {args.ceremony_year}. Available: {sorted(CALENDARS.keys())}"
            )
        calendar = CALENDARS[args.ceremony_year]
    if as_of_date and calendar is None:
        raise ValueError(
            "--ceremony-year is required when --as-of-date is set "
            "(needed for calendar-based feature availability resolution)"
        )

    # Parse cv_years if provided
    cv_years: tuple[int, int] | None = None
    if args.cv_years:
        cv_years = parse_year_range(args.cv_years)
        logger.info(f"CV year range: {cv_years[0]}-{cv_years[1]}")

    # Filter feature set by availability if as_of_date is specified
    if as_of_date:
        feature_set, available, unavailable = filter_feature_set_by_availability(
            feature_set, as_of_date, calendar=calendar
        )
        if unavailable:
            print(f"\nFiltered {len(unavailable)} features not available as of {as_of_date}:")
            for feat in unavailable:
                print(f"  - {feat}")
            print(f"Using {len(available)} available features")

    # Validate and infer model type for feature engineering
    feature_family: FeatureFamily = validate_model_feature_consistency(
        model_configs, feature_set.feature_family
    )
    logger.info(f"Using {feature_family.upper()} feature engineering")

    # Load data with model-specific feature engineering
    print("Loading dataset with model-specific feature engineering...")
    df = load_data(
        raw_path=Path(args.raw_path),
        calendar=calendar,
    )

    # Filter data by cv_years if specified
    if cv_years:
        start_ceremony = cv_years[0] - 1928  # year to ceremony
        end_ceremony = cv_years[1] - 1928
        original_count = len(df)
        df = df[(df["ceremony"] >= start_ceremony) & (df["ceremony"] <= end_ceremony)].copy()
        print(
            f"Filtered data to ceremonies {start_ceremony}-{end_ceremony} ({len(df)}/{original_count} records)"
        )

    print_dataset_summary(df)

    # Print experiment info
    print(f"\nExperiment: {args.name}")
    print(f"CV Mode: {args.cv_mode}")
    print(f"Model: {model_configs[0].model_type}")
    print(f"Configs to evaluate: {len(model_configs)}")
    print(f"Feature set: {feature_set.name} ({len(feature_set.features)} features)")
    print(f"CV Splitter: {splitter.name}")
    if as_of_date:
        print(f"Feature availability as of: {as_of_date}")
    if cv_years:
        print(f"CV year range: {cv_years[0]}-{cv_years[1]}")
    if args.n_jobs != 1:
        print(f"Parallel jobs: {args.n_jobs}")

    # Create experiment directory
    exp_dir = create_experiment_dir(args.name, args.output_dir)
    print(f"\nOutput directory: {exp_dir}")

    # Save engineered features if requested
    if args.save_features:
        features_path = save_engineered_features(df, feature_family, exp_dir)
        print(f"Saved engineered features to: {features_path}")

    # Run experiment
    print("\n" + "=" * 70)
    print(f"Running {args.cv_mode.upper()} CV")
    print("=" * 70)

    if args.cv_mode == "simple":
        result = run_simple_cv_grid(
            df=df,
            feature_set=feature_set,
            model_configs=model_configs,
            splitter=splitter,
            n_jobs=args.n_jobs,
            verbose=args.verbose,
        )
    else:
        # Nested CV — both LR and GBT families are supported
        _NESTED_CV_SUPPORTED = (FeatureFamily.LR, FeatureFamily.GBT)
        if feature_family not in _NESTED_CV_SUPPORTED:
            raise ValueError(
                f"Nested CV does not support feature family '{feature_family}'. "
                f"Supported: {[f.value for f in _NESTED_CV_SUPPORTED]}"
            )
        result = run_nested_cv_grid(
            df=df,
            feature_set=feature_set,
            model_configs=model_configs,
            outer_splitter=splitter,
            feature_family=feature_family,
            verbose=args.verbose,
        )

    # Save all outputs
    print("\nSaving outputs...")
    save_experiment_config(
        exp_dir,
        args.name,
        args.cv_mode,
        model_configs,
        feature_set,
        cv_split_config,
        df,
        as_of_date,
        cv_years,
        args.n_jobs,
    )
    save_predictions_csv(exp_dir, result.predictions)
    save_metrics(exp_dir, result)

    # Save nested CV specific output
    if result.nested_cv_result:
        with open(exp_dir / "nested_cv_result.json", "w") as f:
            json.dump(result.nested_cv_result.model_dump(), f, indent=2)

    # Save best config for downstream use (train.py --cv-output)
    best_config_output = {
        "model_config": result.model_config_used,
        "feature_set": {
            "name": feature_set.name,
            "features": feature_set.features,
            "feature_family": feature_set.feature_family,
        },
        "as_of_date": as_of_date.isoformat() if as_of_date else None,
        "cv_metrics": {
            "accuracy": result.metrics.micro.accuracy,
            "log_loss": result.metrics.micro.log_loss,
        },
    }
    with open(exp_dir / "best_config.json", "w") as f:
        json.dump(best_config_output, f, indent=2)

    # Reconstruct best config for report using TypeAdapter (handles all model types)
    best_config: ModelConfig = model_config_adapter.validate_python(result.model_config_used)

    # Generate and print summary report
    report = generate_summary_report(
        exp_dir, args.name, args.cv_mode, best_config, feature_set, splitter, result
    )
    print("\n" + report)

    # Save per-fold feature importance for LOYO Jaccard analysis
    if args.save_fold_importance:
        print("\nSaving per-fold feature importance...")
        save_fold_importance(df, feature_set, best_config, splitter, exp_dir)

    print(f"\nAll outputs saved to: {exp_dir}")
    print("\nFiles created:")
    for output_file in sorted(exp_dir.iterdir()):
        print(f"  - {output_file.name}")


if __name__ == "__main__":
    main()
