"""Build a single Oscar Best Picture prediction model end-to-end.

Consolidates the common experiment pattern into a single invocation:
  Step 1: Full-feature CV with grid search -> best hyperparams
  Step 2: Train full-feature model -> feature importance
  Step 3: Extract nonzero-importance features
  Step 4: CV with selected features -> honest performance estimate
  Step 5: Train final model + predict test year

Without --feature-selection, simplifies to:
  Step 1: CV with grid search -> best hyperparams
  Step 2: Train final model + predict test year

Each invocation builds ONE model. Use a bash script to loop over models/variants.
All paths are relative to the project root.

Usage:
    # Single model with feature selection
    uv run python -m oscar_prediction_market.modeling.build_model \
        --name lr_baseline \
        --param-grid configs/param_grids/lr_grid.json \
        --feature-config configs/features/lr_baseline.json \
        --cv-split configs/cv_splits/leave_one_year_out.json \
        --train-years 2000-2025 --test-years 2026 \
        --output-dir storage/my_experiment \
        --as-of-date 2026-02-07 --n-jobs 4 \
        --feature-selection

    # Without feature selection (pre-existing feature set)
    uv run python -m oscar_prediction_market.modeling.build_model \
        --name gbt_nonzero \
        --param-grid configs/param_grids/gbt_grid.json \
        --feature-config storage/previous_exp/gbt_nonzero.json \
        --cv-split configs/cv_splits/leave_one_year_out.json \
        --train-years 2000-2025 --test-years 2026 \
        --output-dir storage/my_experiment
"""

import argparse
import csv
import json
import logging
import subprocess
import sys
from datetime import date as date_cls
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from oscar_prediction_market.data.awards_calendar import (
    CALENDARS,
    AwardsCalendar,
)
from oscar_prediction_market.modeling.data_loader import (
    filter_feature_set_by_availability,
    load_data,
    prepare_model_data,
)
from oscar_prediction_market.modeling.feature_engineering import (
    FeatureSet,
)
from oscar_prediction_market.modeling.feature_engineering.types import (
    FeatureFamily,
)
from oscar_prediction_market.modeling.utils import (
    parse_year_range,
    year_to_ceremony,
)

logger = logging.getLogger(__name__)

MODULE_PREFIX = "oscar_prediction_market.modeling"


# ============================================================================
# Subprocess runners
# ============================================================================


def _run_command(cmd: list[str], step_name: str) -> None:
    """Run a subprocess command, raising on failure."""
    logger.info(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Step '{step_name}' failed with exit code {result.returncode}")


def _run_evaluate_cv(
    *,
    param_grid: str,
    feature_config: str,
    cv_split: str,
    train_years: str,
    name: str,
    output_dir: str,
    as_of_date: str | None,
    ceremony_year: int | None,
    raw_path: str,
    n_jobs: int,
    save_fold_importance: bool,
) -> None:
    """Run cross-validation evaluation. Skips if best_config.json already exists."""
    if Path(output_dir, "best_config.json").exists():
        logger.info(f"  SKIP (already done): {name}")
        return

    cmd = [
        sys.executable,
        "-m",
        f"{MODULE_PREFIX}.evaluate_cv",
        "--model-config",
        param_grid,
        "--feature-config",
        feature_config,
        "--cv-split",
        cv_split,
        "--cv-mode",
        "simple",
        "--cv-years",
        train_years,
        "--name",
        name,
        "--output-dir",
        output_dir,
        "--raw-path",
        raw_path,
    ]
    if as_of_date:
        cmd.extend(["--as-of-date", as_of_date])
    if ceremony_year is not None:
        cmd.extend(["--ceremony-year", str(ceremony_year)])
    if n_jobs != 1:
        cmd.extend(["--n-jobs", str(n_jobs)])
    if save_fold_importance:
        cmd.append("--save-fold-importance")

    _run_command(cmd, f"evaluate_cv:{name}")


def _run_train_predict(
    *,
    cv_output: str,
    train_years: str,
    test_years: str,
    output_dir: str,
    step_name: str,
    as_of_date: str | None,
    ceremony_year: int | None,
    raw_path: str,
    check_file: str,
) -> None:
    """Run training and prediction. Skips if check_file already exists."""
    if Path(output_dir, check_file).exists():
        logger.info(f"  SKIP (already done): {step_name}")
        return

    cmd = [
        sys.executable,
        "-m",
        f"{MODULE_PREFIX}.train_predict",
        "--mode",
        "both",
        "--cv-output",
        cv_output,
        "--train-years",
        train_years,
        "--test-years",
        test_years,
        "--output-dir",
        output_dir,
        "--raw-path",
        raw_path,
    ]
    if as_of_date:
        cmd.extend(["--as-of-date", as_of_date])
    if ceremony_year is not None:
        cmd.extend(["--ceremony-year", str(ceremony_year)])

    _run_command(cmd, f"train_predict:{step_name}")


# ============================================================================
# Feature extraction
# ============================================================================


def _compute_vif(X: pd.DataFrame, feature: str) -> float:
    """Compute Variance Inflation Factor for a single feature using sklearn.

    VIF = 1 / (1 - R²) where R² is from regressing the feature on all others.
    VIF > 5-10 indicates problematic multicollinearity.
    """
    others = [c for c in X.columns if c != feature]
    if not others:
        return 1.0
    lr = LinearRegression()
    y = np.asarray(X[feature].values)
    lr.fit(X[others].values, y)
    r_squared = float(lr.score(X[others].values, y))
    if r_squared >= 1.0:
        return float("inf")
    return 1.0 / (1.0 - r_squared)


def _filter_by_vif(
    features: list[str],
    raw_path: str,
    feature_family: str,
    as_of_date: str | None,
    ceremony_year: int | None,
    train_years: str,
    vif_threshold: float,
) -> list[str]:
    """Iteratively remove features with VIF > threshold.

    Loads the training data, computes VIF for each feature, and removes the
    highest-VIF feature until all are below threshold.

    Args:
        features: List of feature names to filter
        raw_path: Path to raw dataset
        feature_family: Feature family string (e.g., "lr" or "gbt")
        as_of_date: Optional as-of date for feature availability
        ceremony_year: Ceremony year for calendar resolution
        train_years: Training year range string
        vif_threshold: Maximum allowed VIF

    Returns:
        Filtered list of feature names
    """
    aod = None
    calendar: AwardsCalendar | None = None
    if as_of_date:
        aod = date_cls.fromisoformat(as_of_date)
    if ceremony_year is not None:
        calendar = CALENDARS[ceremony_year]

    df = load_data(Path(raw_path), aod, calendar=calendar)
    start_year, end_year = parse_year_range(train_years)
    start_cer = year_to_ceremony(start_year)
    end_cer = year_to_ceremony(end_year)
    train_df = df[(df["ceremony"] >= start_cer) & (df["ceremony"] <= end_cer)].copy()

    feature_set = FeatureSet(
        name="vif_check",
        description="Temporary feature set for VIF computation",
        features=features,
        feature_family=FeatureFamily(feature_family),
    )
    X, _, _ = prepare_model_data(train_df, feature_set)

    # Drop columns that are constant (VIF undefined)
    non_const = [c for c in X.columns if X[c].nunique() > 1]
    X = X[non_const]
    remaining = list(X.columns)

    dropped: list[tuple[str, float]] = []
    while len(remaining) > 1:
        vifs = {f: _compute_vif(X[remaining], f) for f in remaining}
        max_feat = max(vifs, key=lambda k: vifs[k])
        max_vif = vifs[max_feat]
        if max_vif <= vif_threshold:
            break
        remaining.remove(max_feat)
        dropped.append((max_feat, max_vif))
        logger.info(f"    VIF drop: {max_feat} (VIF={max_vif:.1f})")

    if dropped:
        logger.info(f"    VIF filtering: {len(dropped)} features dropped, {len(remaining)} remain")
    else:
        logger.info(
            f"    VIF filtering: all {len(remaining)} features below threshold {vif_threshold}"
        )

    return remaining


def _extract_selected_features(
    feature_importance_csv: Path,
    source_feature_config_path: Path,
    output_path: Path,
    run_name: str,
    importance_threshold: float | None,
    max_features: int | None,
) -> Path:
    """Extract important features from a trained model.

    Selection strategy (applied in order, most restrictive wins):
    1. Keep only features with importance > 0 (always applied)
    2. If importance_threshold set: keep features whose cumulative importance
       (sorted descending) reaches the threshold fraction of total importance
    3. If max_features set: cap at this many features (by importance rank)

    Args:
        feature_importance_csv: Path to feature_importance.csv from training
        source_feature_config_path: Original feature config (for feature_family)
        output_path: Where to write the selected feature config JSON
        run_name: Name for logging
        importance_threshold: Cumulative importance fraction (e.g., 0.95 = keep
            features explaining 95% of total importance). None = no threshold.
        max_features: Maximum number of features to keep. None = no cap.

    Returns:
        Path to the generated feature config.
    """
    if output_path.exists():
        logger.info(f"  SKIP (already done): {run_name} feature extraction")
        return output_path

    source_config = json.loads(source_feature_config_path.read_text())
    feature_family = source_config["feature_family"]

    # Read all features with their importance (unified column since refactor)
    feature_importances: list[tuple[str, float]] = []
    with open(feature_importance_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # All models now provide 'importance'; fallback to 'abs_coefficient' for old CSVs
            importance = float(row.get("importance", 0) or row.get("abs_coefficient", 0))
            if importance > 0:
                feature_importances.append((row["feature"], importance))

    # Sort by importance descending
    feature_importances.sort(key=lambda x: x[1], reverse=True)
    total_importance = sum(imp for _, imp in feature_importances)

    # Apply cumulative importance threshold
    if importance_threshold is not None and total_importance > 0:
        cumulative = 0.0
        threshold_features: list[tuple[str, float]] = []
        for feat, imp in feature_importances:
            cumulative += imp / total_importance
            threshold_features.append((feat, imp))
            if cumulative >= importance_threshold:
                break
        logger.info(
            f"  {run_name}: importance threshold {importance_threshold:.0%} "
            f"-> {len(threshold_features)}/{len(feature_importances)} features "
            f"(cumulative={cumulative:.1%})"
        )
        feature_importances = threshold_features

    # Apply max_features cap
    if max_features is not None and len(feature_importances) > max_features:
        logger.info(
            f"  {run_name}: max_features cap {max_features} "
            f"-> dropping {len(feature_importances) - max_features} features"
        )
        feature_importances = feature_importances[:max_features]

    features = [f for f, _ in feature_importances]

    selection_desc_parts = [f"{len(features)} features from {run_name}"]
    if importance_threshold is not None:
        selection_desc_parts.append(f"importance_threshold={importance_threshold}")
    if max_features is not None:
        selection_desc_parts.append(f"max_features={max_features}")

    selected_config = {
        "name": f"{run_name}_selected",
        "description": ", ".join(selection_desc_parts),
        "features": features,
        "feature_family": feature_family,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(selected_config, indent=2))
    logger.info(f"  {run_name}: selected {len(features)} features")

    return output_path


# ============================================================================
# Pipeline execution
# ============================================================================


def build_model(
    *,
    name: str,
    param_grid: str,
    feature_config: str,
    cv_split: str,
    train_years: str,
    test_years: str,
    output_dir: str,
    raw_path: str,
    as_of_date: str | None,
    ceremony_year: int | None,
    n_jobs: int,
    feature_selection: bool,
    importance_threshold: float | None,
    max_features: int | None,
    vif_filter: bool,
    vif_threshold: float,
    save_fold_importance: bool,
) -> None:
    """Execute the full pipeline for a single model.

    With feature_selection=True (5-6 steps):
        1. Cross-validate with full features + grid search
        2. Train full-feature model for feature importance
        3. Extract selected features (importance threshold + max cap)
        3.5. (Optional) VIF filtering to reduce multicollinearity
        4. Cross-validate with selected features (honest estimate)
        5. Train final model + predict test year

    With feature_selection=False (2 steps):
        1. Cross-validate with grid search
        2. Train final model + predict test year

    Output directories created under output_dir/{name}/:
        With feature_selection:
            1_full_cv/              Full-feature cross-validation
            2_full_train/           Full-feature train (importance extraction)
            3_selected_features.json  Selected feature config
            4_selected_cv/          Selected-feature cross-validation
            5_final_predict/        Final model + predictions
        Without feature_selection:
            1_cv/                   Cross-validation
            2_final_predict/        Final model + predictions
    """
    base = Path(output_dir) / name
    base.mkdir(parents=True, exist_ok=True)

    # Pre-check: skip early if temporal filtering eliminates all features
    calendar: AwardsCalendar | None = None
    if ceremony_year is not None:
        calendar = CALENDARS[ceremony_year]
    if as_of_date:
        feature_data = json.loads(Path(feature_config).read_text())
        fs = FeatureSet(**feature_data)
        try:
            filter_feature_set_by_availability(
                fs, date_cls.fromisoformat(as_of_date), calendar=calendar
            )
        except ValueError as e:
            logger.warning(f"[{name}] {e} Skipping entire pipeline.")
            return

    if feature_selection:
        # Step 1: Full-feature CV with grid search
        full_cv_dir = str(base / "1_full_cv")
        logger.info(f"[{name}] Step 1/5: Full-feature CV with grid search")
        _run_evaluate_cv(
            param_grid=param_grid,
            feature_config=feature_config,
            cv_split=cv_split,
            train_years=train_years,
            name=f"{name}_full_cv",
            output_dir=full_cv_dir,
            as_of_date=as_of_date,
            raw_path=raw_path,
            n_jobs=n_jobs,
            save_fold_importance=False,
            ceremony_year=ceremony_year,
        )

        # Step 2: Train full-feature model for feature importance
        full_train_dir = str(base / "2_full_train")
        logger.info(f"[{name}] Step 2/5: Train full-feature model (feature importance)")
        _run_train_predict(
            cv_output=full_cv_dir,
            train_years=train_years,
            test_years=test_years,
            output_dir=full_train_dir,
            step_name=f"{name}_full_train",
            as_of_date=as_of_date,
            raw_path=raw_path,
            check_file="feature_importance.csv",
            ceremony_year=ceremony_year,
        )

        # Step 3: Extract selected features (with importance threshold + max cap)
        logger.info(f"[{name}] Step 3/5: Extract selected features")
        nonzero_path = _extract_selected_features(
            Path(full_train_dir) / "feature_importance.csv",
            Path(feature_config),
            base / "3_selected_features.json",
            name,
            importance_threshold=importance_threshold,
            max_features=max_features,
        )

        # Step 3.5 (optional): VIF filtering for multicollinearity
        if vif_filter:
            logger.info(f"[{name}] Step 3.5: VIF filtering (threshold={vif_threshold})")
            selected_config = json.loads(Path(nonzero_path).read_text())
            original_features = selected_config["features"]
            filtered_features = _filter_by_vif(
                features=original_features,
                raw_path=raw_path,
                feature_family=selected_config["feature_family"],
                as_of_date=as_of_date,
                train_years=train_years,
                vif_threshold=vif_threshold,
                ceremony_year=ceremony_year,
            )
            if len(filtered_features) < len(original_features):
                selected_config["features"] = filtered_features
                dropped_count = len(original_features) - len(filtered_features)
                selected_config["description"] += (
                    f", VIF filtered (threshold={vif_threshold}, dropped {dropped_count})"
                )
                Path(nonzero_path).write_text(json.dumps(selected_config, indent=2))
                logger.info(
                    f"  {name}: VIF filtered {len(original_features)} -> "
                    f"{len(filtered_features)} features"
                )

        # Check if feature selection eliminated all features
        selected_config_check = json.loads(Path(nonzero_path).read_text())
        if not selected_config_check["features"]:
            logger.warning(
                f"[{name}] Feature selection eliminated ALL features. "
                "Skipping steps 4-5. Full CV results (step 1) are the final output."
            )
            return

        # Step 4: CV with selected features (honest performance estimate)
        selected_cv_dir = str(base / "4_selected_cv")
        logger.info(f"[{name}] Step 4/5: CV with selected features")
        _run_evaluate_cv(
            param_grid=param_grid,
            feature_config=str(nonzero_path),
            cv_split=cv_split,
            train_years=train_years,
            name=f"{name}_selected_cv",
            output_dir=selected_cv_dir,
            as_of_date=as_of_date,
            raw_path=raw_path,
            n_jobs=n_jobs,
            save_fold_importance=save_fold_importance,
            ceremony_year=ceremony_year,
        )

        # Step 5: Train final model + predict
        final_dir = str(base / "5_final_predict")
        logger.info(f"[{name}] Step 5/5: Train final model + predict {test_years}")
        _run_train_predict(
            cv_output=selected_cv_dir,
            train_years=train_years,
            test_years=test_years,
            output_dir=final_dir,
            step_name=f"{name}_final_predict",
            as_of_date=as_of_date,
            raw_path=raw_path,
            check_file="predictions.csv",
            ceremony_year=ceremony_year,
        )

    else:
        # Simple: CV + train_predict
        cv_dir = str(base / "1_cv")
        logger.info(f"[{name}] Step 1/2: CV with grid search")
        _run_evaluate_cv(
            param_grid=param_grid,
            feature_config=feature_config,
            cv_split=cv_split,
            train_years=train_years,
            name=f"{name}_cv",
            output_dir=cv_dir,
            as_of_date=as_of_date,
            raw_path=raw_path,
            n_jobs=n_jobs,
            save_fold_importance=False,
            ceremony_year=ceremony_year,
        )

        final_dir = str(base / "2_final_predict")
        logger.info(f"[{name}] Step 2/2: Train final model + predict {test_years}")
        _run_train_predict(
            cv_output=cv_dir,
            train_years=train_years,
            test_years=test_years,
            output_dir=final_dir,
            step_name=f"{name}_final_predict",
            as_of_date=as_of_date,
            raw_path=raw_path,
            check_file="predictions.csv",
            ceremony_year=ceremony_year,
        )


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build a single Oscar Best Picture prediction model (end-to-end)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Builds ONE model end-to-end: CV -> (optional feature selection) -> train -> predict.
Use a bash script to loop over multiple models/variants.

With --feature-selection (output under {output-dir}/{name}/):
  Step 1: Full-feature CV + grid search        -> 1_full_cv/
  Step 2: Train full-feature model (importance) -> 2_full_train/
  Step 3: Extract nonzero features              -> 3_selected_features.json
  Step 4: CV with selected features (honest)    -> 4_selected_cv/
  Step 5: Train final model + predict           -> 5_final_predict/

Without --feature-selection (output under {output-dir}/{name}/):
  Step 1: CV + grid search          -> 1_cv/
  Step 2: Train final + predict     -> 2_final_predict/

Example:
    uv run python -m oscar_prediction_market.modeling.build_model \\
        --name lr_baseline \\
        --param-grid configs/param_grids/lr_grid.json \\
        --feature-config configs/features/lr_baseline.json \\
        --cv-split configs/cv_splits/leave_one_year_out.json \\
        --train-years 2000-2025 --test-years 2026 \\
        --output-dir storage/my_experiment \\
        --feature-selection
        """,
    )

    # Required arguments
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Run name (used for output subdirectory naming)",
    )
    parser.add_argument(
        "--param-grid",
        type=str,
        required=True,
        help="Path to parameter grid JSON file",
    )
    parser.add_argument(
        "--feature-config",
        type=str,
        required=True,
        help="Path to feature config JSON file",
    )
    parser.add_argument(
        "--cv-split",
        type=str,
        required=True,
        help="Path to CV split config JSON file",
    )
    parser.add_argument(
        "--train-years",
        type=str,
        required=True,
        help="Training year range, e.g. '2000-2025'",
    )
    parser.add_argument(
        "--test-years",
        type=str,
        required=True,
        help="Test year(s), e.g. '2026'",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Base output directory (subdirs created per step)",
    )

    # Optional arguments
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Feature availability date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--ceremony-year",
        type=int,
        default=None,
        help="Ceremony year for calendar-based feature availability (e.g. 2025, 2026)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for grid search (default: 1)",
    )
    parser.add_argument(
        "--feature-selection",
        action="store_true",
        help="Enable 2-pass feature selection (full -> nonzero -> selected)",
    )
    parser.add_argument(
        "--importance-threshold",
        type=float,
        default=None,
        help="Cumulative importance threshold (0-1). Keep features until cumulative "
        "importance reaches this fraction. E.g., 0.95 = keep features explaining 95%% "
        "of total importance. (default: no threshold, keep all nonzero)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Maximum number of features to keep after importance ranking. (default: no cap)",
    )
    parser.add_argument(
        "--vif-filter",
        action="store_true",
        help="Enable VIF-based multicollinearity filtering after feature selection. "
        "Iteratively removes features with VIF > --vif-threshold.",
    )
    parser.add_argument(
        "--vif-threshold",
        type=float,
        default=5.0,
        help="Maximum allowed VIF when --vif-filter is enabled (default: 5.0)",
    )
    parser.add_argument(
        "--save-fold-importance",
        action="store_true",
        help="Save per-fold feature importance in CV step for LOYO Jaccard analysis",
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        required=True,
        help="Path to raw dataset JSON file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate paths
    for path, label in [
        (args.param_grid, "--param-grid"),
        (args.feature_config, "--feature-config"),
        (args.cv_split, "--cv-split"),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    logger.info(f"Building model: {args.name}")
    logger.info(f"  Output dir:         {args.output_dir}")
    logger.info(f"  Train years:        {args.train_years}")
    logger.info(f"  Test years:         {args.test_years}")
    logger.info(f"  Feature selection:  {args.feature_selection}")
    logger.info(f"  Importance thresh:  {args.importance_threshold or '(none)'}")
    logger.info(f"  Max features:       {args.max_features or '(none)'}")
    logger.info(f"  VIF filter:         {args.vif_filter} (threshold={args.vif_threshold})")
    logger.info(f"  As-of date:         {args.as_of_date or '(none)'}")
    logger.info(f"  Ceremony year:      {args.ceremony_year or '(none)'}")

    build_model(
        name=args.name,
        param_grid=args.param_grid,
        feature_config=args.feature_config,
        cv_split=args.cv_split,
        train_years=args.train_years,
        test_years=args.test_years,
        output_dir=args.output_dir,
        raw_path=args.raw_path,
        as_of_date=args.as_of_date,
        n_jobs=args.n_jobs,
        feature_selection=args.feature_selection,
        importance_threshold=args.importance_threshold,
        max_features=args.max_features,
        vif_filter=args.vif_filter,
        vif_threshold=args.vif_threshold,
        save_fold_importance=args.save_fold_importance,
        ceremony_year=args.ceremony_year,
    )

    logger.info(f"Done: {args.name}")


if __name__ == "__main__":
    main()
