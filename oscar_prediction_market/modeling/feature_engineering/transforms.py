"""Transform functions for feature engineering.

Contains all named transform functions, factory functions, and constants used by
feature definitions. Transform functions have signature:
    (NominationRecord, TransformContext) -> value

Organized by domain:
- Oscar nomination transforms
- Precursor aggregate transforms (incremental, category-aware)
- Critic score transforms
- Release timing transforms
- Person career transforms
- Animated feature transforms (studio, sequel)
- Precursor individual transforms (factory functions: winner/nominee/GG composite)
- BP interaction transforms
"""

import math

from oscar_prediction_market.data.awards_calendar import (
    AwardOrg,
    EventPhase,
)
from oscar_prediction_market.data.precursor_mappings import (
    CATEGORY_PRECURSORS,
)
from oscar_prediction_market.data.schema import (
    AwardResult,
    NominationRecord,
    OscarCategory,
    PrecursorAward,
)
from oscar_prediction_market.modeling.feature_engineering.helpers import (
    _any_true,
    compute_percentile_rank,
)
from oscar_prediction_market.modeling.feature_engineering.types import (
    ACTING_CATEGORIES,
    BIG_5_CATEGORIES,
    TransformContext,
    TransformFn,
)

# ============================================================================
# Oscar Nomination Transforms
# ============================================================================


def _acting_nomination_count(r: NominationRecord, _ctx: TransformContext) -> int:
    """Count acting category nominations."""
    cats = r.film.oscar_noms.oscar_nominations_by_category
    return sum(cats.get(cat.value, 0) for cat in ACTING_CATEGORIES)


def _major_category_count(r: NominationRecord, _ctx: TransformContext) -> int:
    """Count Big 5 category nominations."""
    cats = r.film.oscar_noms.oscar_nominations_by_category
    return sum(1 for cat in BIG_5_CATEGORIES if cat.value in cats)


def _nominations_percentile_in_year(r: NominationRecord, ctx: TransformContext) -> float:
    """Percentile rank of nominations within same ceremony year."""
    ceremony_records = ctx.records_by_ceremony.get(r.ceremony, [])
    all_noms = [rec.film.oscar_noms.oscar_total_nominations for rec in ceremony_records]
    return compute_percentile_rank(r.film.oscar_noms.oscar_total_nominations, all_noms)


# ============================================================================
# Precursor Aggregate Transforms (incremental, category-aware)
# ============================================================================


def _precursor_wins_count(r: NominationRecord, ctx: TransformContext) -> int:
    """Count precursor award wins for this record's category, incrementally based on as_of_date.

    Category-aware: uses CATEGORY_PRECURSORS to determine which precursor
    fields are relevant. Groups by PrecursorAward org so GG drama + musical count as
    ONE win (not two).

    For historical ceremonies (not the target ceremony), counts all wins.
    For the target ceremony, only counts wins whose announcement date <= as_of_date.
    """
    specs = CATEGORY_PRECURSORS.get(r.category, [])
    if not specs:
        return 0

    # Determine target ceremony from calendar (if available)
    target_ceremony = ctx.calendar.ceremony_number if ctx.calendar else None

    # Group specs by org to avoid double-counting GG
    by_org: dict[PrecursorAward, list[str]] = {}
    for spec in specs:
        by_org.setdefault(spec.award, []).append(spec.key)

    count = 0
    for award, keys in by_org.items():
        # Check if any sub-award for this org is a winner
        values = [r.precursors.get(k, AwardResult()).winner for k in keys]
        if not any(v is True for v in values):
            continue

        # Incremental availability check
        if ctx.as_of_date is None or target_ceremony is None or r.ceremony != target_ceremony:
            count += 1
        elif ctx.calendar is not None and ctx.as_of_date >= ctx.calendar.get_local_date(
            AwardOrg(award.value), EventPhase.WINNER
        ):
            count += 1

    return count


def _precursor_nominations_count(r: NominationRecord, ctx: TransformContext) -> int:
    """Count precursor award nominations for this record's category, incrementally.

    Category-aware: uses CATEGORY_PRECURSORS to determine which precursor
    fields are relevant. Groups by PrecursorAward org so GG drama + musical count as
    ONE nomination (not two).

    For historical ceremonies, counts all nominations.
    For the target ceremony, only counts noms whose announcement date <= as_of_date.
    """
    specs = CATEGORY_PRECURSORS.get(r.category, [])
    if not specs:
        return 0

    target_ceremony = ctx.calendar.ceremony_number if ctx.calendar else None

    # Group specs by org
    by_org: dict[PrecursorAward, list[str]] = {}
    for spec in specs:
        by_org.setdefault(spec.award, []).append(spec.key)

    count = 0
    for award, keys in by_org.items():
        values = [r.precursors.get(k, AwardResult()).nominee for k in keys]
        if not any(v is True for v in values):
            continue

        if ctx.as_of_date is None or target_ceremony is None or r.ceremony != target_ceremony:
            count += 1
        elif ctx.calendar is not None and ctx.as_of_date >= ctx.calendar.get_local_date(
            AwardOrg(award.value), EventPhase.NOMINATION
        ):
            count += 1

    return count


# ============================================================================
# Critic Score Transforms
# ============================================================================


def _critics_consensus_score(r: NominationRecord, _ctx: TransformContext) -> float | None:
    """Average of RT and Metacritic scores."""
    if not r.film.metadata:
        return None
    scores = [
        s for s in [r.film.metadata.rotten_tomatoes, r.film.metadata.metacritic] if s is not None
    ]
    return sum(scores) / len(scores) if scores else None


def _critics_audience_gap(r: NominationRecord, _ctx: TransformContext) -> float | None:
    """Gap between critics (RT) and audience (IMDb) scores."""
    if not r.film.metadata:
        return None
    if r.film.metadata.rotten_tomatoes is None or r.film.metadata.imdb_rating is None:
        return None
    return r.film.metadata.rotten_tomatoes - (r.film.metadata.imdb_rating * 10)


# ============================================================================
# Release Timing Transforms
# ============================================================================


def _release_month_sin(r: NominationRecord, _ctx: TransformContext) -> float | None:
    """Cyclical encoding of release month (sin component)."""
    if not r.film.metadata or not r.film.metadata.released:
        return None
    month = r.film.metadata.released.month
    return math.sin(2 * math.pi * month / 12)


def _release_month_cos(r: NominationRecord, _ctx: TransformContext) -> float | None:
    """Cyclical encoding of release month (cos component)."""
    if not r.film.metadata or not r.film.metadata.released:
        return None
    month = r.film.metadata.released.month
    return math.cos(2 * math.pi * month / 12)


# ============================================================================
# Person Career Transforms
# ============================================================================


def _person_prev_noms_same(r: NominationRecord, _ctx: TransformContext) -> int | None:
    """Prior Oscar nominations in the same category."""
    return r.person.prev_noms_same_category if r.person else None


def _person_prev_noms_any(r: NominationRecord, _ctx: TransformContext) -> int | None:
    """Prior Oscar nominations in any category."""
    return r.person.prev_noms_any_category if r.person else None


def _person_prev_wins_same(r: NominationRecord, _ctx: TransformContext) -> int | None:
    """Prior Oscar wins in the same category."""
    return r.person.prev_wins_same_category if r.person else None


def _person_prev_wins_any(r: NominationRecord, _ctx: TransformContext) -> int | None:
    """Prior Oscar wins in any category."""
    return r.person.prev_wins_any_category if r.person else None


def _person_is_overdue(r: NominationRecord, _ctx: TransformContext) -> bool | None:
    """Person has 3+ nominations in same category but zero wins.

    The 'overdue narrative' is a known predictor of Oscar wins — voters may give
    a sympathy/legacy vote to long-nominated actors who haven't won yet.
    """
    if r.person is None:
        return None
    return r.person.prev_noms_same_category >= 3 and r.person.prev_wins_same_category == 0


def _person_age_at_ceremony(r: NominationRecord, _ctx: TransformContext) -> int | None:
    """Person's age at the Oscar ceremony.

    Age is a known predictor for acting categories — very young and very old
    winners are disproportionately common (prodigy/legacy effects).
    """
    if r.person is None or r.person.birth_date is None:
        return None
    # Ceremony is typically in Feb/Mar of year_film + 1
    ceremony_year = r.year_film + 1
    age = ceremony_year - r.person.birth_date.year
    # Adjust if birthday hasn't happened yet (approximate using March 1)
    if r.person.birth_date.month > 3:
        age -= 1
    return age


def _person_total_film_credits(r: NominationRecord, _ctx: TransformContext) -> int | None:
    """Total film credits from TMDb (cast + crew, deduplicated)."""
    return r.person.total_film_credits if r.person else None


def _person_tmdb_popularity(r: NominationRecord, _ctx: TransformContext) -> float | None:
    """TMDb popularity score."""
    return r.person.tmdb_popularity if r.person else None


def _film_also_bp_nominated(r: NominationRecord, _ctx: TransformContext) -> bool:
    """Film is also nominated for Best Picture.

    Prestige signal for acting/directing categories — being in a BP nominee
    correlates with individual wins.
    """
    return OscarCategory.BEST_PICTURE.value in r.film.oscar_noms.oscar_nominations_by_category


# ============================================================================
# Animated Feature Transforms
# ============================================================================

_SEQUEL_PATTERNS = [
    " 2",
    " 3",
    " 4",
    " 5",
    " II",
    " III",
    " IV",
    ": Part",
    " Part ",
    " Chapter ",
]


def _is_sequel(r: NominationRecord, _ctx: TransformContext) -> bool:
    """Detect sequel from title heuristics.

    Sequels in animated features have a mixed Oscar track record — some
    (Toy Story 3) win, but originals are generally favored.
    """
    title = r.film.title.upper()
    return any(pat.upper() in title for pat in _SEQUEL_PATTERNS)


# Major animation studio name fragments for production company matching.
# Each tuple: (feature_suffix, company_name_fragments).
_ANIMATION_STUDIOS: list[tuple[str, list[str]]] = [
    ("disney_pixar", ["walt disney", "pixar"]),
    ("dreamworks", ["dreamworks animation"]),
    ("illumination", ["illumination"]),
    ("sony_animation", ["sony pictures animation"]),
    ("laika", ["laika"]),
]


def _make_studio_feature(company_fragments: list[str]) -> TransformFn:
    """Factory: create a feature that checks if film is from a specific studio."""
    lower_fragments = [f.lower() for f in company_fragments]

    def _transform(r: NominationRecord, _ctx: TransformContext) -> bool:
        if not r.film.metadata:
            return False
        companies_lower = [c.lower() for c in r.film.metadata.production_companies]
        return any(frag in co for co in companies_lower for frag in lower_fragments)

    return _transform


# ============================================================================
# Precursor Individual Transforms (factory functions)
# ============================================================================


def _precursor_winner(key: str) -> TransformFn:
    def _fn(r: NominationRecord, _ctx: TransformContext) -> bool | None:
        return r.precursors.get(key, AwardResult()).winner

    return _fn


def _precursor_nominee(key: str) -> TransformFn:
    def _fn(r: NominationRecord, _ctx: TransformContext) -> bool | None:
        return r.precursors.get(key, AwardResult()).nominee

    return _fn


def _gg_any(field: str, k1: str, k2: str) -> TransformFn:
    """Build extractor: True if either Golden Globe sub-award field is True."""

    def _extract(r: NominationRecord, _ctx: TransformContext) -> bool | None:
        return _any_true(
            getattr(r.precursors.get(k1, AwardResult()), field),
            getattr(r.precursors.get(k2, AwardResult()), field),
        )

    return _extract
