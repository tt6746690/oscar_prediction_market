"""Precursor award specifications and category mappings.

Single source of truth for how precursor awards map to Oscar categories.
Used by:
- data/build_dataset.py: key → storage in flat precursors dict + date filtering
- modeling/feature_engineering.py: key → feature names ({key}_winner/{key}_nominee)
- data/dataset_report.py: key → completeness checking

Design notes:
- PrecursorSpec.key is both the storage key in NominationRecord.precursors (a flat
  dict[str, AwardResult]) AND the fetch-layer key in PrecursorAwardsFile.awards AND
  the feature name prefix. One key, used everywhere.
- The nested PrecursorAwardsComplete model was removed in favor of a flat dict to
  eliminate ~200 lines of org model boilerplate and remove null bloat in serialized
  JSON (previously every record carried ~30 sub-award slots, most null).
- Composite features (e.g., golden_globe_any = Drama OR Musical) are purely
  feature-engineering concerns, defined in modeling/feature_engineering.py.
"""

from pydantic import BaseModel, Field

from oscar_prediction_market.data.schema import (
    OscarCategory,
    PrecursorAward,
    PrecursorKey,
)


class PrecursorSpec(BaseModel):
    """Single precursor award specification.

    Maps an Oscar category to a precursor award for both data storage and
    feature generation. The ``key`` field is used everywhere:
    - Key in ``PrecursorAwardsFile.awards`` (fetch layer)
    - Key in ``NominationRecord.precursors`` (flat ``dict[str, AwardResult]``)
    - Feature name prefix: ``{key}_winner``, ``{key}_nominee``

    Example::

        >>> spec = PrecursorSpec(key=PrecursorKey.PGA_BP, award=PrecursorAward.PGA)
        >>> # Fetch layer: precursor_awards.awards["pga_bp"]
        >>> # Storage: nomination.precursors["pga_bp"].winner
        >>> # Features: pga_bp_winner, pga_bp_nominee
    """

    model_config = {"extra": "forbid"}

    key: PrecursorKey = Field(..., description="Universal key: fetch, storage, and feature prefix")
    award: PrecursorAward = Field(..., description="Award org for calendar date lookup")


# ============================================================================
# Category → Precursor Mappings (single source of truth)
# ============================================================================

CATEGORY_PRECURSORS: dict[OscarCategory, list[PrecursorSpec]] = {
    OscarCategory.BEST_PICTURE: [
        PrecursorSpec(key=PrecursorKey.PGA_BP, award=PrecursorAward.PGA),
        PrecursorSpec(key=PrecursorKey.DGA_DIRECTING, award=PrecursorAward.DGA),
        PrecursorSpec(key=PrecursorKey.SAG_ENSEMBLE, award=PrecursorAward.SAG),
        PrecursorSpec(key=PrecursorKey.BAFTA_FILM, award=PrecursorAward.BAFTA),
        PrecursorSpec(key=PrecursorKey.GOLDEN_GLOBE_DRAMA, award=PrecursorAward.GOLDEN_GLOBE),
        PrecursorSpec(key=PrecursorKey.GOLDEN_GLOBE_MUSICAL, award=PrecursorAward.GOLDEN_GLOBE),
        PrecursorSpec(key=PrecursorKey.CRITICS_CHOICE_PICTURE, award=PrecursorAward.CRITICS_CHOICE),
    ],
    OscarCategory.DIRECTING: [
        PrecursorSpec(key=PrecursorKey.DGA_DIRECTING, award=PrecursorAward.DGA),
        PrecursorSpec(key=PrecursorKey.BAFTA_DIRECTOR, award=PrecursorAward.BAFTA),
        PrecursorSpec(key=PrecursorKey.GOLDEN_GLOBE_DIRECTOR, award=PrecursorAward.GOLDEN_GLOBE),
        PrecursorSpec(
            key=PrecursorKey.CRITICS_CHOICE_DIRECTOR, award=PrecursorAward.CRITICS_CHOICE
        ),
    ],
    OscarCategory.ACTOR_LEADING: [
        PrecursorSpec(key=PrecursorKey.SAG_LEAD_ACTOR, award=PrecursorAward.SAG),
        PrecursorSpec(key=PrecursorKey.BAFTA_LEAD_ACTOR, award=PrecursorAward.BAFTA),
        PrecursorSpec(key=PrecursorKey.GOLDEN_GLOBE_ACTOR_DRAMA, award=PrecursorAward.GOLDEN_GLOBE),
        PrecursorSpec(
            key=PrecursorKey.GOLDEN_GLOBE_ACTOR_MUSICAL, award=PrecursorAward.GOLDEN_GLOBE
        ),
        PrecursorSpec(key=PrecursorKey.CRITICS_CHOICE_ACTOR, award=PrecursorAward.CRITICS_CHOICE),
    ],
    OscarCategory.ACTRESS_LEADING: [
        PrecursorSpec(key=PrecursorKey.SAG_LEAD_ACTRESS, award=PrecursorAward.SAG),
        PrecursorSpec(key=PrecursorKey.BAFTA_LEAD_ACTRESS, award=PrecursorAward.BAFTA),
        PrecursorSpec(
            key=PrecursorKey.GOLDEN_GLOBE_ACTRESS_DRAMA, award=PrecursorAward.GOLDEN_GLOBE
        ),
        PrecursorSpec(
            key=PrecursorKey.GOLDEN_GLOBE_ACTRESS_MUSICAL, award=PrecursorAward.GOLDEN_GLOBE
        ),
        PrecursorSpec(key=PrecursorKey.CRITICS_CHOICE_ACTRESS, award=PrecursorAward.CRITICS_CHOICE),
    ],
    OscarCategory.ACTOR_SUPPORTING: [
        PrecursorSpec(key=PrecursorKey.SAG_SUPPORTING_ACTOR, award=PrecursorAward.SAG),
        PrecursorSpec(key=PrecursorKey.BAFTA_SUPPORTING_ACTOR, award=PrecursorAward.BAFTA),
        PrecursorSpec(
            key=PrecursorKey.GOLDEN_GLOBE_SUPPORTING_ACTOR, award=PrecursorAward.GOLDEN_GLOBE
        ),
        PrecursorSpec(
            key=PrecursorKey.CRITICS_CHOICE_SUPPORTING_ACTOR, award=PrecursorAward.CRITICS_CHOICE
        ),
    ],
    OscarCategory.ACTRESS_SUPPORTING: [
        PrecursorSpec(key=PrecursorKey.SAG_SUPPORTING_ACTRESS, award=PrecursorAward.SAG),
        PrecursorSpec(key=PrecursorKey.BAFTA_SUPPORTING_ACTRESS, award=PrecursorAward.BAFTA),
        PrecursorSpec(
            key=PrecursorKey.GOLDEN_GLOBE_SUPPORTING_ACTRESS, award=PrecursorAward.GOLDEN_GLOBE
        ),
        PrecursorSpec(
            key=PrecursorKey.CRITICS_CHOICE_SUPPORTING_ACTRESS, award=PrecursorAward.CRITICS_CHOICE
        ),
    ],
    OscarCategory.ORIGINAL_SCREENPLAY: [
        PrecursorSpec(key=PrecursorKey.WGA_ORIGINAL, award=PrecursorAward.WGA),
        PrecursorSpec(key=PrecursorKey.BAFTA_ORIGINAL_SCREENPLAY, award=PrecursorAward.BAFTA),
        PrecursorSpec(
            key=PrecursorKey.CRITICS_CHOICE_ORIGINAL_SCREENPLAY,
            award=PrecursorAward.CRITICS_CHOICE,
        ),
        PrecursorSpec(key=PrecursorKey.GOLDEN_GLOBE_SCREENPLAY, award=PrecursorAward.GOLDEN_GLOBE),
    ],
    OscarCategory.CINEMATOGRAPHY: [
        PrecursorSpec(key=PrecursorKey.ASC_CINEMATOGRAPHY, award=PrecursorAward.ASC),
        PrecursorSpec(key=PrecursorKey.BAFTA_CINEMATOGRAPHY, award=PrecursorAward.BAFTA),
        PrecursorSpec(
            key=PrecursorKey.CRITICS_CHOICE_CINEMATOGRAPHY,
            award=PrecursorAward.CRITICS_CHOICE,
        ),
    ],
    OscarCategory.ANIMATED_FEATURE: [
        PrecursorSpec(key=PrecursorKey.ANNIE_FEATURE, award=PrecursorAward.ANNIE),
        PrecursorSpec(key=PrecursorKey.BAFTA_ANIMATED, award=PrecursorAward.BAFTA),
        PrecursorSpec(key=PrecursorKey.PGA_ANIMATED, award=PrecursorAward.PGA),
        PrecursorSpec(key=PrecursorKey.GOLDEN_GLOBE_ANIMATED, award=PrecursorAward.GOLDEN_GLOBE),
        PrecursorSpec(
            key=PrecursorKey.CRITICS_CHOICE_ANIMATED, award=PrecursorAward.CRITICS_CHOICE
        ),
    ],
}


def get_precursor_specs(category: OscarCategory) -> list[PrecursorSpec]:
    """Get precursor specs for a category.

    Raises ValueError if category has no precursor mapping defined.
    """
    if category not in CATEGORY_PRECURSORS:
        raise ValueError(
            f"No precursor mapping for {category}. "
            f"Supported: {sorted(c.name for c in CATEGORY_PRECURSORS)}"
        )
    return CATEGORY_PRECURSORS[category]
