"""Transform engine — converts raw NominationDataset to feature DataFrame.

Contains:
- Feature availability filtering
- Transform context building
- Main transform_dataset() function
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from oscar_prediction_market.data.awards_calendar import AwardsCalendar
from oscar_prediction_market.data.schema import (
    NominationDataset,
    NominationRecord,
)
from oscar_prediction_market.modeling.feature_engineering.types import (
    FeatureDefinition,
    TransformContext,
)

# ============================================================================
# Feature Availability Filtering
# ============================================================================


def filter_features_by_availability(
    features: list[FeatureDefinition],
    as_of_date: date | None,
    calendar: AwardsCalendar | None = None,
) -> list[FeatureDefinition]:
    """Filter to features available at as_of_date.

    Args:
        features: List of feature definitions
        as_of_date: Date for availability check. None = all features (backtesting).
        calendar: Awards calendar for resolving availability dates.
            Required when as_of_date is not None and features have available_from set.

    Returns:
        Features available at as_of_date
    """
    if as_of_date is None:
        return features
    result = []
    for f in features:
        if f.available_from is None:
            result.append(f)
        else:
            if calendar is None:
                raise ValueError(
                    f"calendar is required to resolve availability for feature {f.name!r}"
                )
            resolved_date = f.available_from(calendar)
            if as_of_date >= resolved_date:
                result.append(f)
    return result


def get_unavailable_features(
    features: list[FeatureDefinition],
    as_of_date: date,
    calendar: AwardsCalendar,
) -> list[tuple[str, date]]:
    """Get features not available at as_of_date with their availability dates.

    Args:
        features: List of feature definitions
        as_of_date: Date to check
        calendar: Awards calendar for resolving availability events.

    Returns:
        List of (feature_name, resolved_date) tuples for unavailable features
    """
    unavailable = []
    for f in features:
        if f.available_from is not None:
            resolved_date = f.available_from(calendar)
            if as_of_date < resolved_date:
                unavailable.append((f.name, resolved_date))
    return unavailable


# ============================================================================
# Transform Context Building
# ============================================================================


def build_transform_context(
    dataset: NominationDataset,
    as_of_date: date | None = None,
    calendar: AwardsCalendar | None = None,
) -> TransformContext:
    """Build transform context from raw dataset.

    Used for year-relative features like nominations_percentile_in_year
    and incremental aggregate features.

    Args:
        dataset: Raw dataset with all records
        as_of_date: Date for incremental feature availability. None = backtesting.
        calendar: Awards calendar for the target ceremony year. Required when
            as_of_date is set (needed by incremental aggregate features).

    Returns:
        TransformContext with records grouped by ceremony, as_of_date, and calendar
    """
    records_by_ceremony: dict[int, list[NominationRecord]] = {}
    for record in dataset.records:
        ceremony = record.ceremony
        if ceremony not in records_by_ceremony:
            records_by_ceremony[ceremony] = []
        records_by_ceremony[ceremony].append(record)
    return TransformContext(
        records_by_ceremony=records_by_ceremony,
        as_of_date=as_of_date,
        calendar=calendar,
    )


# ============================================================================
# Main Transform Function
# ============================================================================


def _resolve_feature_input(
    features: list[FeatureDefinition] | list[str],
) -> list[FeatureDefinition]:
    """Resolve feature input to a list of FeatureDefinitions.

    Accepts either:
    - list[FeatureDefinition]: returned as-is
    - list[str]: resolved via FEATURE_REGISTRY
    """
    if not features:
        return []
    if isinstance(features[0], str):
        from oscar_prediction_market.modeling.feature_engineering.registry import (
            resolve_features,
        )

        return resolve_features(features)  # type: ignore[arg-type]
    return features  # type: ignore[return-value]


def transform_dataset(
    raw_dataset: NominationDataset,
    features: list[FeatureDefinition] | list[str],
    as_of_date: date | None = None,
    calendar: AwardsCalendar | None = None,
) -> pd.DataFrame:
    """Transform raw dataset to feature DataFrame.

    Args:
        raw_dataset: Raw dataset with all records
        features: Feature definitions or feature names to apply. If names (list[str]),
            they are resolved via FEATURE_REGISTRY.
        as_of_date: Filter features by availability. None = all features (backtesting).
        calendar: Awards calendar for the target ceremony year. Required when
            as_of_date is set.

    Returns:
        DataFrame with ID columns + engineered features
    """
    # Resolve feature names to definitions if needed
    resolved_features = _resolve_feature_input(features)

    # Filter to available features
    available_features = filter_features_by_availability(resolved_features, as_of_date, calendar)

    # Build context for year-relative and incremental features
    ctx = build_transform_context(raw_dataset, as_of_date, calendar)

    # Transform each record
    rows = []
    for record in raw_dataset.records:
        row = {
            # ID columns (always included)
            "film_id": record.film.film_id,
            "title": record.film.title,
            "ceremony": record.ceremony,
            "year_film": record.year_film,
            "category_winner": record.category_winner,
        }
        # Apply feature transforms
        for feat in available_features:
            row[feat.name] = feat.transform(record, ctx)
        rows.append(row)

    return pd.DataFrame(rows)
