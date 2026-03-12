"""Data loading and preparation for Oscar prediction.

This module orchestrates the data pipeline:
1. Load raw dataset (from data/processed/oscar_{category}_raw.json)
2. Apply feature engineering (all features from FEATURE_REGISTRY)
3. Filter features by availability date (for real-time predictions)
4. Prepare train/test splits

Usage:
    from modeling.data_loader import load_data, get_train_test_split_by_year

    # Load all features (backtesting)
    df = load_data(raw_path=Path("data/processed/oscar_best_picture_raw.json"))

    # Load with date-filtered features (real-time 2026 prediction)
    df = load_data(raw_path=Path(...), as_of_date=date(2026, 2, 3), calendar=CALENDARS[98])

    # Split for backtesting
    train_df, test_df = get_train_test_split_by_year(df, test_ceremony=97)
"""

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from oscar_prediction_market.data.awards_calendar import AwardsCalendar
from oscar_prediction_market.data.schema import (
    NominationDataset,
)
from oscar_prediction_market.modeling.feature_engineering import (
    FEATURE_REGISTRY,
    FeatureFamily,
    FeatureSet,
    get_feature_names,
    transform_dataset,
)

logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent


# ============================================================================
# Raw Data Loading
# ============================================================================


def load_raw_dataset(path: Path) -> NominationDataset:
    """Load raw dataset from JSON.

    Args:
        path: Path to raw dataset JSON file.

    Returns:
        Raw dataset with all records
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {path}\n"
            "Run: python -m oscar_prediction_market.data.build_dataset"
        )

    with open(path) as f:
        return NominationDataset(**json.load(f))


# ============================================================================
# Main Data Loading Pipeline
# ============================================================================


def load_data(
    raw_path: Path,
    as_of_date: date | None = None,
    calendar: AwardsCalendar | None = None,
) -> pd.DataFrame:
    """Load and engineer ALL features from FEATURE_REGISTRY for modeling.

    This is the main entry point for the data pipeline:
    1. Load raw dataset
    2. Generate ALL features from FEATURE_REGISTRY (not just model-specific ones)
    3. Filter features by availability (if as_of_date provided)

    We generate all features from FEATURE_REGISTRY regardless of model_type because
    downstream ablation configs may reference supplementary features (individual
    precursors, person career, animated, etc.). The feature config and
    prepare_features() handle selecting the right subset for each model.

    Features that don't apply to a category (e.g., person features for
    best_picture) will be NaN — the feature config simply won't include them.

    Args:
        raw_path: Path to raw dataset JSON file
        as_of_date: Date for feature availability filtering. None = all features (backtesting)
        calendar: Awards calendar for the target ceremony year. Required when
            as_of_date is set.

    Returns:
        DataFrame with all engineered features, ready for train/test split
    """
    # 1. Load raw data
    raw_dataset = load_raw_dataset(raw_path)

    # 2. Generate ALL features from registry (not just model-specific ones)
    all_features = list(FEATURE_REGISTRY.values())

    # 3. Apply feature engineering with availability filtering
    df = transform_dataset(raw_dataset, all_features, as_of_date, calendar)

    return df


def save_engineered_features(
    df: pd.DataFrame,
    feature_family: FeatureFamily,
    output_dir: Path,
) -> Path:
    """Save engineered features to JSON for inspection.

    Args:
        df: DataFrame with engineered features
        feature_family: Feature family used for engineering
        output_dir: Output directory for the features file

    Returns:
        Path to saved file
    """

    output_path = output_dir / f"oscar_features_{feature_family}.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    records = df.to_dict(orient="records")
    output = {
        "feature_family": feature_family,
        "record_count": len(records),
        "columns": list(df.columns),
        "records": records,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    return output_path


# ============================================================================
# Train/Test Splitting
# ============================================================================


def get_train_test_split_by_year(
    df: pd.DataFrame,
    test_ceremony: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data for leave-one-year-out cross-validation.

    Train on all ceremonies before test_ceremony, test on test_ceremony.
    This ensures no future data leakage.

    Args:
        df: Full dataset as DataFrame
        test_ceremony: The ceremony year to use as test set

    Returns:
        (train_df, test_df) tuple
    """
    train_df = df[df["ceremony"] < test_ceremony].copy()
    test_df = df[df["ceremony"] == test_ceremony].copy()
    return train_df, test_df


def get_ceremony_years(df: pd.DataFrame) -> list[int]:
    """Get sorted list of unique ceremony years in the dataset."""
    return sorted(df["ceremony"].unique().tolist())


# ============================================================================
# Availability Filtering
# ============================================================================


def _partition_features_by_availability(
    feature_names: list[str],
    as_of_date: date,
    calendar: AwardsCalendar,
) -> tuple[list[str], list[str]]:
    """Partition feature names into (available, unavailable) as of a date.

    Uses FEATURE_REGISTRY to resolve availability events. Features not in
    the registry are treated as unavailable. Features without an
    ``available_from`` are always available.

    Args:
        feature_names: Feature names to partition.
        as_of_date: Date for availability check.
        calendar: Awards calendar for resolving availability events.

    Returns:
        (available, unavailable) lists preserving input order.
    """
    available: list[str] = []
    unavailable: list[str] = []
    for feat_name in feature_names:
        feat_def = FEATURE_REGISTRY.get(feat_name)
        if feat_def is None:
            unavailable.append(feat_name)
        elif feat_def.available_from is not None:
            resolved_date = feat_def.available_from(calendar)
            if as_of_date < resolved_date:
                unavailable.append(feat_name)
            else:
                available.append(feat_name)
        else:
            available.append(feat_name)
    return available, unavailable


# ============================================================================
# Feature Preparation
# ============================================================================


def prepare_features(
    df: pd.DataFrame,
    feature_set: FeatureSet,
    fill_missing: bool = True,
    as_of_date: date | None = None,
    calendar: AwardsCalendar | None = None,
) -> pd.DataFrame:
    """Extract and preprocess features for modeling.

    Args:
        df: DataFrame with all columns
        feature_set: Feature set configuration with list of features
        fill_missing: Whether to fill missing values with 0
        as_of_date: Date for feature availability filtering. None = all features.
        calendar: Awards calendar for resolving availability events. Required
            when as_of_date is set.

    Returns:
        DataFrame with only the selected features, preprocessed
    """
    feature_cols = list(feature_set.features)

    # Filter by availability if date specified
    if as_of_date is not None:
        if calendar is None:
            raise ValueError(
                "calendar is required when as_of_date is set for feature availability filtering"
            )
        feature_cols, unavailable_cols = _partition_features_by_availability(
            feature_cols, as_of_date, calendar
        )
        if unavailable_cols:
            logger.info(
                f"Filtered {len(unavailable_cols)} features not available as of {as_of_date}: "
                f"{unavailable_cols}"
            )

    # Fail-fast: all requested features must exist in the DataFrame.
    # Missing features indicate a bug in load_data() (not generating all features)
    # or a mismatch between the feature config and the data pipeline.
    missing_cols = set(feature_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"{len(missing_cols)} features requested but not found in DataFrame: "
            f"{sorted(missing_cols)}. "
            f"Check that load_data() generates all features in FEATURE_REGISTRY."
        )

    X = df[feature_cols].copy()

    # Convert boolean columns to int
    bool_cols = X.select_dtypes(include=["bool"]).columns
    X[bool_cols] = X[bool_cols].astype(int)

    # Fill missing values.
    #
    # Most NaNs come from structurally absent precursor awards — awards that didn't
    # exist for a given ceremony year (e.g., BAFTA Animated before 2007, Critics Choice
    # Cinematography before 2009). fillna(0) treats "award didn't exist" the same as
    # "not nominated / not won."
    #
    # This is acceptable because:
    # 1. Missingness is year-correlated, not nominee-correlated — every nominee in a
    #    given year has the same NaN pattern, so within-year comparisons (the core
    #    prediction task for clogit / choice-level models) are unaffected.
    # 2. The strongest precursors (DGA, SAG, BAFTA, PGA for BP; Annie/ASC for other
    #    categories) have 100% coverage. Sparse features are secondary predictors.
    # 3. Aggregate features (precursor_wins_count) already skip None values — only
    #    confirmed True wins are counted, so missing awards don't inflate/deflate counts.
    # 4. For tree models, 0 falls naturally into the "no positive signal" branch.
    #
    # Alternatives considered (Feb 2026): XGBoost native NaN handling, missingness
    # indicator features, fraction-based aggregates, dropping sparse features. None
    # warranted given the structural nature of the gaps and dominance of full-coverage
    # precursors in ablation experiments.
    if fill_missing:
        X = X.infer_objects(copy=False)
        X = X.fillna(0)

    return X


def prepare_target(df: pd.DataFrame) -> pd.Series:
    """Extract the target variable (category_winner)."""
    return df["category_winner"].astype(int)


def prepare_model_data(
    df: pd.DataFrame,
    feature_set: FeatureSet,
    as_of_date: date | None = None,
    calendar: AwardsCalendar | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepare X, y, and metadata for modeling.

    Args:
        df: DataFrame with all columns
        feature_set: Feature set configuration
        as_of_date: Date for feature availability filtering. None = all features.
        calendar: Awards calendar for resolving availability events.

    Returns:
        (X, y, metadata) where metadata contains film_id, title, ceremony
    """
    X = prepare_features(df, feature_set, as_of_date=as_of_date, calendar=calendar)
    y = prepare_target(df)
    metadata = df[["film_id", "title", "ceremony"]].copy()
    return X, y, metadata


def filter_feature_set_by_availability(
    feature_set: FeatureSet,
    as_of_date: date,
    calendar: AwardsCalendar | None = None,
) -> tuple[FeatureSet, list[str], list[str]]:
    """Filter a FeatureSet to only include features available as of a date.

    Args:
        feature_set: Original feature set
        as_of_date: Date for availability filtering
        calendar: Awards calendar for resolving availability events. Required
            when features have available_from set.

    Returns:
        Tuple of (filtered_feature_set, available_features, unavailable_features)
    """
    if not feature_set.feature_family:
        # No feature family, can't determine availability
        return feature_set, list(feature_set.features), []

    if calendar is None:
        raise ValueError("calendar is required to resolve feature availability events")

    available, unavailable = _partition_features_by_availability(
        list(feature_set.features), as_of_date, calendar
    )

    if not available:
        raise ValueError(
            f"No features from feature set '{feature_set.name}' are available "
            f"as of {as_of_date}. All {len(unavailable)} features are unavailable."
        )

    # Create new feature set
    filtered = FeatureSet(
        name=f"{feature_set.name}_as_of_{as_of_date.isoformat()}",
        description=f"{feature_set.description} (filtered for {as_of_date})",
        features=available,
        feature_family=feature_set.feature_family,
    )

    return filtered, available, unavailable


# ============================================================================
# Data Summary
# ============================================================================


def summarize_dataset(df: pd.DataFrame) -> dict:
    """Generate summary statistics for the dataset."""
    ceremonies = get_ceremony_years(df)

    # Get feature columns (exclude ID and target)
    id_cols = {"film_id", "title", "ceremony", "year_film", "category_winner"}
    feature_cols = [c for c in df.columns if c not in id_cols]

    return {
        "total_records": len(df),
        "ceremony_range": (min(ceremonies), max(ceremonies)),
        "num_ceremonies": len(ceremonies),
        "winners": int(df["category_winner"].sum()),
        "num_features": len(feature_cols),
        "feature_columns": feature_cols,
        "nominees_per_year": df.groupby("ceremony").size().describe().to_dict(),
    }


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print a summary of the dataset."""
    summary = summarize_dataset(df)
    print("=" * 60)
    print("Oscar Best Picture Dataset Summary")
    print("=" * 60)
    print(f"Total records: {summary['total_records']}")
    print(f"Ceremony range: {summary['ceremony_range'][0]} - {summary['ceremony_range'][1]}")
    print(f"Number of ceremonies: {summary['num_ceremonies']}")
    print(f"Total winners: {summary['winners']}")  # category_winner count
    print(f"Avg nominees per year: {summary['nominees_per_year']['mean']:.1f}")
    print(f"Number of features: {summary['num_features']}")
    print("=" * 60)


# ============================================================================
# CLI for Testing
# ============================================================================


if __name__ == "__main__":
    import argparse

    from oscar_prediction_market.data.awards_calendar import CALENDARS

    parser = argparse.ArgumentParser(description="Test data loading pipeline")
    parser.add_argument(
        "--feature-family",
        type=FeatureFamily,
        choices=list(FeatureFamily),
        default=FeatureFamily.LR,
        help="Feature family for engineering (lr or gbt)",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Date for feature filtering (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--ceremony-year",
        type=int,
        default=None,
        help="Ceremony year for calendar resolution (required with --as-of-date)",
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        required=True,
        help="Path to raw dataset JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for saved features (required with --save)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save engineered features to JSON",
    )
    args = parser.parse_args()

    # Parse date and calendar
    as_of = date.fromisoformat(args.as_of_date) if args.as_of_date else None
    cal = CALENDARS[args.ceremony_year] if args.ceremony_year else None
    if as_of and not cal:
        parser.error("--ceremony-year is required when --as-of-date is set")

    # Load data
    print(f"\nLoading data for feature_family={args.feature_family}, as_of_date={as_of}")
    df = load_data(
        raw_path=Path(args.raw_path),
        as_of_date=as_of,
        calendar=cal,
    )

    # Print summary
    print_dataset_summary(df)

    # Show features
    feature_cols = get_feature_names(list(FEATURE_REGISTRY.values()))
    print(f"\nFeature columns for {args.feature_family}:")
    for col in feature_cols:
        in_df = "✓" if col in df.columns else "✗"
        print(f"  {in_df} {col}")

    # Save if requested
    if args.save:
        if not args.output_dir:
            parser.error("--output-dir is required when using --save")
        path = save_engineered_features(df, args.feature_family, Path(args.output_dir))
        print(f"\nSaved engineered features to: {path}")

    # Test split
    train_df, test_df = get_train_test_split_by_year(df, test_ceremony=97)
    print("\nSplit for ceremony 97:")
    print(
        f"  Train: {len(train_df)} records "
        f"(ceremonies {train_df['ceremony'].min()}-{train_df['ceremony'].max()})"
    )
    print(f"  Test: {len(test_df)} records")
