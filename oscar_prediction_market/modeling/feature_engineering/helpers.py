"""Pure utility functions for feature transforms.

Contains:
- Math helpers: safe_log10, compute_percentile_rank, compute_zscore
- Composite logic: _any_true
- Factory functions for within-year normalization transforms
- Field extractors for numeric record fields
"""

import math
from collections.abc import Sequence

from oscar_prediction_market.data.schema import NominationRecord
from oscar_prediction_market.modeling.feature_engineering.types import (
    FieldExtractor,
    TransformContext,
    TransformFn,
)

# ============================================================================
# Math Helpers
# ============================================================================


def safe_log10(value: int | float | None) -> float | None:
    """Compute log10, returning None for invalid inputs."""
    if value is None or value <= 0:
        return None
    return math.log10(value)


def compute_percentile_rank(value: float | int, all_values: Sequence[float | int]) -> float:
    """Compute percentile rank of value within a list. Returns 0.0-1.0."""
    if not all_values:
        return 0.5
    sorted_values = sorted(all_values)
    rank = sum(1 for v in sorted_values if v < value)
    return rank / len(sorted_values)


def compute_zscore(value: float | int, all_values: Sequence[float | int]) -> float:
    """Compute z-score of value within a list. Returns 0.0 if std is 0."""
    if len(all_values) < 2:
        return 0.0
    mean = sum(all_values) / len(all_values)
    variance = sum((v - mean) ** 2 for v in all_values) / len(all_values)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    return (value - mean) / std


def _any_true(*values: bool | None) -> bool | None:
    """Check if any value is True, handling None for unavailable data.

    Used for combining split precursor fields (e.g., golden_globe_drama + golden_globe_musical).

    Returns:
        True: At least one value is True
        False: No values are True, and at least one value has data (not None)
        None: All values are None (no data available)
    """
    if any(v is True for v in values):
        return True
    if any(v is not None for v in values):
        return False
    return None


# ============================================================================
# Within-Year Normalization Factories
# ============================================================================


def _make_within_year_percentile(
    extractor: FieldExtractor,
) -> TransformFn:
    """Factory: create a TransformFn that computes percentile rank within ceremony year.

    Uses the extractor to pull a numeric value from each record in the same ceremony,
    then computes the percentile rank of the current record's value.
    Returns 0.5 if the value is None (neutral default for missing data).
    """

    def _transform(r: NominationRecord, ctx: TransformContext) -> float:  # noqa: F821
        value = extractor(r)
        if value is None:
            return 0.5
        ceremony_records = ctx.records_by_ceremony.get(r.ceremony, [])
        all_values = [v for rec in ceremony_records if (v := extractor(rec)) is not None]
        return compute_percentile_rank(value, all_values)

    return _transform


def _make_within_year_zscore(
    extractor: FieldExtractor,
) -> TransformFn:
    """Factory: create a TransformFn that computes z-score within ceremony year.

    Uses the extractor to pull a numeric value from each record in the same ceremony,
    then computes the z-score of the current record's value.
    Returns 0.0 if the value is None (neutral default for missing data).
    """

    def _transform(r: NominationRecord, ctx: TransformContext) -> float:  # noqa: F821
        value = extractor(r)
        if value is None:
            return 0.0
        ceremony_records = ctx.records_by_ceremony.get(r.ceremony, [])
        all_values = [v for rec in ceremony_records if (v := extractor(rec)) is not None]
        return compute_zscore(value, all_values)

    return _transform


# ============================================================================
# Field Extractors for Within-Year Normalization
# ============================================================================


def _extract_metacritic(r: NominationRecord) -> int | None:
    return r.film.metadata.metacritic if r.film.metadata else None


def _extract_rotten_tomatoes(r: NominationRecord) -> int | None:
    return r.film.metadata.rotten_tomatoes if r.film.metadata else None


def _extract_imdb_rating(r: NominationRecord) -> float | None:
    return r.film.metadata.imdb_rating if r.film.metadata else None


def _extract_box_office_worldwide(r: NominationRecord) -> int | None:
    return r.film.metadata.box_office_worldwide if r.film.metadata else None


def _extract_box_office_domestic(r: NominationRecord) -> int | None:
    return r.film.metadata.box_office_domestic if r.film.metadata else None


def _extract_budget(r: NominationRecord) -> int | None:
    return r.film.metadata.budget if r.film.metadata else None


def _extract_runtime(r: NominationRecord) -> int | None:
    return r.film.metadata.runtime_minutes if r.film.metadata else None
