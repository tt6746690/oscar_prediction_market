"""Pydantic models and type aliases for feature engineering.

Contains:
- FeatureFamily enum (LR vs GBT feature engineering type)
- Availability lambda factories (available_after_oscar_noms, etc.)
- FeatureSet (JSON-loadable feature set config)
- TransformContext (per-ceremony context for transforms)
- FeatureDefinition (single feature with colocated transform)
- Oscar category grouping constants
- Type aliases: TransformFn, FieldExtractor
"""

from collections.abc import Callable
from datetime import date
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from oscar_prediction_market.data.awards_calendar import (
    PRECURSOR_ORGS,
    AwardOrg,
    AwardsCalendar,
    EventPhase,
)
from oscar_prediction_market.data.schema import (
    NominationRecord,
    OscarCategory,
)

# ============================================================================
# Feature Family
# ============================================================================


class FeatureFamily(StrEnum):
    """Feature engineering family — determines which transforms are applied.

    LR (linear model) family:
        Log-transformed commercial features, percentile-based normalization,
        cyclical encoding, and interaction terms. Used by logistic regression
        and conditional logit models.

    GBT (tree-based model) family:
        Raw features with minimal transforms. Trees handle non-linearities
        natively. Used by gradient boosting, XGBoost, softmax GBT, and
        calibrated softmax GBT models.
    """

    LR = "lr"
    GBT = "gbt"


# ============================================================================
# Feature Set Configuration
# ============================================================================


class FeatureSet(BaseModel):
    """Configuration for a feature set (loaded from JSON or defined programmatically).

    Feature sets can be family-specific:
    - lr: Features with log transforms, cyclical encoding, interactions (linear models)
    - gbt: Raw features, minimal transforms (tree-based models)

    Feature names are validated against FEATURE_REGISTRY at construction time
    to catch typos in configs early.
    """

    name: str = Field(..., description="Feature set name", min_length=1)
    description: str = Field(..., description="Human-readable description")
    features: list[str] = Field(..., description="List of feature column names", min_length=1)
    feature_family: FeatureFamily | None = Field(
        default=None,
        description="Feature engineering family (lr or gbt). None means universal.",
    )

    model_config = {"extra": "forbid"}

    @field_validator("features")
    @classmethod
    def _validate_feature_names(cls, features: list[str]) -> list[str]:
        """Validate that all feature names exist in FEATURE_REGISTRY."""
        # Lazy import to avoid circular dependency (registry -> features -> types)
        from oscar_prediction_market.modeling.feature_engineering.registry import (
            FEATURE_REGISTRY,
        )

        unknown = [f for f in features if f not in FEATURE_REGISTRY]
        if unknown:
            raise ValueError(
                f"{len(unknown)} unknown feature name(s): {unknown}. "
                f"Check spelling against FEATURE_REGISTRY keys."
            )
        return features


# ============================================================================
# Transform Context
# ============================================================================


class TransformContext(BaseModel):
    """Context passed to transform functions.

    Attributes:
        records_by_ceremony: Mapping from ceremony number to records for that ceremony.
            Used for within-year relative features.
        as_of_date: Date for incremental feature availability. None = all data available
            (backtesting). Used by aggregate features that count constituents incrementally.
        calendar: Awards calendar for the target ceremony year. Required for incremental
            aggregate features (precursor counts) that need to know the target ceremony
            and precursor announcement dates.
    """

    model_config = {"extra": "forbid"}

    records_by_ceremony: dict[int, list[NominationRecord]]
    as_of_date: date | None = None
    calendar: AwardsCalendar | None = None


# Transform function signature: (record, context) -> feature value
TransformFn = Callable[[NominationRecord, TransformContext], Any]

# Field extractor type: extracts a numeric value from a record
FieldExtractor = Callable[[NominationRecord], float | int | None]


# ============================================================================
# Oscar Category Groups
# ============================================================================

ACTING_CATEGORIES: frozenset[OscarCategory] = frozenset(
    {
        OscarCategory.ACTOR_LEADING,
        OscarCategory.ACTOR_SUPPORTING,
        OscarCategory.ACTRESS_LEADING,
        OscarCategory.ACTRESS_SUPPORTING,
    }
)

SCREENPLAY_CATEGORIES: frozenset[OscarCategory] = frozenset(
    {
        OscarCategory.ADAPTED_SCREENPLAY,
        OscarCategory.ORIGINAL_SCREENPLAY,
    }
)

BIG_5_CATEGORIES: frozenset[OscarCategory] = frozenset(
    {
        OscarCategory.BEST_PICTURE,
        OscarCategory.DIRECTING,
        OscarCategory.ACTOR_LEADING,
        OscarCategory.ACTRESS_LEADING,
        OscarCategory.ADAPTED_SCREENPLAY,
        OscarCategory.ORIGINAL_SCREENPLAY,
    }
)


# ============================================================================
# Feature Definition
# ============================================================================


class FeatureDefinition(BaseModel):
    """Single feature definition with colocated transform.

    Attributes:
        name: Output column name in the feature DataFrame
        transform: Function to compute feature value from raw record and context
        available_from: Callable that resolves an AwardsCalendar to the local date
            when this feature becomes available. None = always available.
        description: Optional human-readable description
    """

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    name: str
    transform: TransformFn
    available_from: Callable[[AwardsCalendar], date] | None = None
    description: str = Field(..., description="Human-readable description (always required)")


# ============================================================================
# Voting Strategy Constants
# ============================================================================

# Best Picture winner selection methodology changed over time:
#
# | Period    | Ceremonies | Winner Selection                    | Nominees              |
# |-----------|------------|-------------------------------------|-----------------------|
# | Pre-2009  | 1-81       | Plurality (first-past-the-post)     | 5 (from #17 onward)   |
# | 2009-2010 | 82-83      | IRV (instant-runoff / preferential) | Fixed 10              |
# | 2011-2020 | 84-93      | IRV                                 | Variable 5-10 (>=5%)  |
# | 2021+     | 94+        | IRV                                 | Fixed 10              |
#
# Key facts:
# - Switch announced June 24, 2009; AMPAS press release August 31, 2009.
# - Triggered by backlash over The Dark Knight / WALL-E not nominated (2008).
# - IRV favors consensus picks over polarizing frontrunners — a film needs
#   broad 2nd/3rd-choice support, not just the largest voting bloc.
# - Nominations have always used single transferable vote (STV).
# - Only Best Picture uses IRV for final ballot; all other categories use plurality.
# - Source: Wikipedia "Academy Award for Best Picture", oscars.org rules.
#
# The first ceremony under IRV is 82 (2009 awards, held Feb 2010).
IRV_FIRST_CEREMONY = 82


# ============================================================================
# Availability Lambda Factories
# ============================================================================


def available_after_oscar_noms() -> Callable[[AwardsCalendar], date]:
    """Feature available after Oscar nominations announcement."""
    return lambda cal: cal.get_local_date(AwardOrg.OSCAR, EventPhase.NOMINATION)


def available_after_winner(org: AwardOrg) -> Callable[[AwardsCalendar], date]:
    """Feature available after a specific organization announces winners."""
    return lambda cal: cal.get_local_date(org, EventPhase.WINNER)


def available_after_nomination(org: AwardOrg) -> Callable[[AwardsCalendar], date]:
    """Feature available after a specific organization announces nominations."""
    return lambda cal: cal.get_local_date(org, EventPhase.NOMINATION)


def available_after_earliest_winner() -> Callable[[AwardsCalendar], date]:
    """Feature available after the earliest precursor winner announcement."""
    return lambda cal: min(cal.get_local_date(org, EventPhase.WINNER) for org in PRECURSOR_ORGS)


def available_after_earliest_nomination() -> Callable[[AwardsCalendar], date]:
    """Feature available after the earliest precursor nomination announcement."""
    return lambda cal: min(cal.get_local_date(org, EventPhase.NOMINATION) for org in PRECURSOR_ORGS)
