"""All FeatureDefinition instances, organized by domain group.

Feature lists are organized to match the domain groups in groups.py:

1. OSCAR_NOMINATION_FEATURES  — cross-category Oscar nomination profile
2. PRECURSOR_FEATURES         — individual precursor winner/nominee flags (auto-generated)
3. GOLDEN_GLOBE_COMPOSITE_FEATURES — OR-combined genre-split GG features
4. BP_INTERACTION_FEATURES    — Best Picture-specific interaction features
5. PERSON_CAREER_FEATURES     — Oscar career history for person categories
6. PERSON_ENRICHMENT_FEATURES — TMDb person data (age, credits, popularity)
7. ANIMATED_FEATURES          — studio identity, sequel detection
8. VOTING_SYSTEM_FEATURES     — BP voting methodology (IRV era, nominee count)
9. CRITIC_SCORE_FEATURES      — review aggregator scores (LR and GBT variants)
10. COMMERCIAL_FEATURES       — budget and box office (LR and GBT variants)
11. TIMING_FEATURES           — release timing and runtime (LR and GBT variants)
12. FILM_METADATA_FEATURES    — genre indicators and MPAA rating

Each list contains FeatureDefinition instances. Features that differ between
LR and GBT (percentile vs z-score, log vs raw) are in separate lists
prefixed with the domain group name.
"""

from oscar_prediction_market.data.awards_calendar import AwardOrg
from oscar_prediction_market.data.precursor_mappings import (
    CATEGORY_PRECURSORS,
)
from oscar_prediction_market.data.schema import (
    AwardResult,
    OscarCategory,
    PrecursorKey,
)
from oscar_prediction_market.modeling.feature_engineering.helpers import (
    _extract_box_office_domestic,
    _extract_box_office_worldwide,
    _extract_budget,
    _extract_imdb_rating,
    _extract_metacritic,
    _extract_rotten_tomatoes,
    _extract_runtime,
    _make_within_year_percentile,
    _make_within_year_zscore,
    safe_log10,
)
from oscar_prediction_market.modeling.feature_engineering.transforms import (
    _ANIMATION_STUDIOS,
    _acting_nomination_count,
    _critics_audience_gap,
    _critics_consensus_score,
    _film_also_bp_nominated,
    _gg_any,
    _is_sequel,
    _major_category_count,
    _make_studio_feature,
    _nominations_percentile_in_year,
    _person_age_at_ceremony,
    _person_is_overdue,
    _person_prev_noms_any,
    _person_prev_noms_same,
    _person_prev_wins_any,
    _person_prev_wins_same,
    _person_tmdb_popularity,
    _person_total_film_credits,
    _precursor_nominations_count,
    _precursor_nominee,
    _precursor_winner,
    _precursor_wins_count,
    _release_month_cos,
    _release_month_sin,
)
from oscar_prediction_market.modeling.feature_engineering.types import (
    ACTING_CATEGORIES,
    IRV_FIRST_CEREMONY,
    SCREENPLAY_CATEGORIES,
    FeatureDefinition,
    available_after_earliest_nomination,
    available_after_earliest_winner,
    available_after_nomination,
    available_after_oscar_noms,
    available_after_winner,
)

# ============================================================================
# 1. OSCAR NOMINATION FEATURES
# ============================================================================

OSCAR_NOMINATION_FEATURES: list[FeatureDefinition] = [
    FeatureDefinition(
        name="oscar_total_nominations",
        transform=lambda r, _: r.film.oscar_noms.oscar_total_nominations,
        available_from=available_after_oscar_noms(),
        description="Total Oscar nominations - strong BP predictor",
    ),
    FeatureDefinition(
        name="has_director_nomination",
        transform=lambda r, _: (
            OscarCategory.DIRECTING.value in r.film.oscar_noms.oscar_nominations_by_category
        ),
        available_from=available_after_oscar_noms(),
        description="Has directing nomination - 'no director nom, no win' rule",
    ),
    FeatureDefinition(
        name="has_editing_nomination",
        transform=lambda r, _: (
            OscarCategory.FILM_EDITING.value in r.film.oscar_noms.oscar_nominations_by_category
        ),
        available_from=available_after_oscar_noms(),
        description="Has editing nomination - the editing rule",
    ),
    FeatureDefinition(
        name="has_acting_nomination",
        transform=lambda r, _: any(
            cat.value in r.film.oscar_noms.oscar_nominations_by_category
            for cat in ACTING_CATEGORIES
        ),
        available_from=available_after_oscar_noms(),
        description="Has any acting nomination",
    ),
    FeatureDefinition(
        name="acting_nomination_count",
        transform=_acting_nomination_count,
        available_from=available_after_oscar_noms(),
        description="Count of acting nominations (0-4)",
    ),
    FeatureDefinition(
        name="has_screenplay_nomination",
        transform=lambda r, _: any(
            cat.value in r.film.oscar_noms.oscar_nominations_by_category
            for cat in SCREENPLAY_CATEGORIES
        ),
        available_from=available_after_oscar_noms(),
        description="Has screenplay nomination",
    ),
    FeatureDefinition(
        name="major_category_count",
        transform=_major_category_count,
        available_from=available_after_oscar_noms(),
        description="Count of Big 5 category nominations",
    ),
    # LR-specific: percentile normalization of nominations within year
    FeatureDefinition(
        name="nominations_percentile_in_year",
        transform=_nominations_percentile_in_year,
        available_from=available_after_oscar_noms(),
        description="Percentile rank of nominations within same year",
    ),
]


# ============================================================================
# 2. PRECURSOR FEATURES (auto-generated from CATEGORY_PRECURSORS)
# ============================================================================

PRECURSOR_FEATURES: list[FeatureDefinition] = []
_seen_precursor_keys: set[str] = set()

for _cat_specs in CATEGORY_PRECURSORS.values():
    for _spec in _cat_specs:
        if _spec.key in _seen_precursor_keys:
            continue
        _seen_precursor_keys.add(_spec.key)
        _k, _award = _spec.key, _spec.award
        _label = _k.replace("_", " ").title()
        PRECURSOR_FEATURES += [
            FeatureDefinition(
                name=f"{_k}_winner",
                transform=_precursor_winner(_k),
                available_from=available_after_winner(AwardOrg(_award.value)),
                description=f"Won {_label}",
            ),
            FeatureDefinition(
                name=f"{_k}_nominee",
                transform=_precursor_nominee(_k),
                available_from=available_after_nomination(AwardOrg(_award.value)),
                description=f"Nominated for {_label}",
            ),
        ]

# Precursor aggregate features (used by both LR and GBT)
PRECURSOR_AGGREGATE_FEATURES: list[FeatureDefinition] = [
    FeatureDefinition(
        name="precursor_wins_count",
        transform=_precursor_wins_count,
        available_from=available_after_earliest_winner(),
        description="Count of precursor wins for this category (incremental, grouped by org)",
    ),
    FeatureDefinition(
        name="precursor_nominations_count",
        transform=_precursor_nominations_count,
        available_from=available_after_earliest_nomination(),
        description="Count of precursor nominations for this category (incremental, grouped by org)",
    ),
]


# ============================================================================
# 3. GOLDEN GLOBE COMPOSITE FEATURES
# ============================================================================

GOLDEN_GLOBE_COMPOSITE_FEATURES: list[FeatureDefinition] = [
    # BP: golden_globe_any = Drama OR Musical
    FeatureDefinition(
        name="golden_globe_any_winner",
        transform=_gg_any(
            "winner", PrecursorKey.GOLDEN_GLOBE_DRAMA, PrecursorKey.GOLDEN_GLOBE_MUSICAL
        ),
        available_from=available_after_winner(AwardOrg.GOLDEN_GLOBE),
        description="Won any Golden Globe (Drama or Musical/Comedy)",
    ),
    FeatureDefinition(
        name="golden_globe_any_nominee",
        transform=_gg_any(
            "nominee", PrecursorKey.GOLDEN_GLOBE_DRAMA, PrecursorKey.GOLDEN_GLOBE_MUSICAL
        ),
        available_from=available_after_nomination(AwardOrg.GOLDEN_GLOBE),
        description="Nominated for any Golden Globe (Drama or Musical/Comedy)",
    ),
    # Actor: golden_globe_actor_any = Drama OR Musical
    FeatureDefinition(
        name="golden_globe_actor_any_winner",
        transform=_gg_any(
            "winner",
            PrecursorKey.GOLDEN_GLOBE_ACTOR_DRAMA,
            PrecursorKey.GOLDEN_GLOBE_ACTOR_MUSICAL,
        ),
        available_from=available_after_winner(AwardOrg.GOLDEN_GLOBE),
        description="Won any Golden Globe Actor (Drama or Musical/Comedy)",
    ),
    FeatureDefinition(
        name="golden_globe_actor_any_nominee",
        transform=_gg_any(
            "nominee",
            PrecursorKey.GOLDEN_GLOBE_ACTOR_DRAMA,
            PrecursorKey.GOLDEN_GLOBE_ACTOR_MUSICAL,
        ),
        available_from=available_after_nomination(AwardOrg.GOLDEN_GLOBE),
        description="Nominated for any Golden Globe Actor (Drama or Musical/Comedy)",
    ),
    # Actress: golden_globe_actress_any = Drama OR Musical
    FeatureDefinition(
        name="golden_globe_actress_any_winner",
        transform=_gg_any(
            "winner",
            PrecursorKey.GOLDEN_GLOBE_ACTRESS_DRAMA,
            PrecursorKey.GOLDEN_GLOBE_ACTRESS_MUSICAL,
        ),
        available_from=available_after_winner(AwardOrg.GOLDEN_GLOBE),
        description="Won any Golden Globe Actress (Drama or Musical/Comedy)",
    ),
    FeatureDefinition(
        name="golden_globe_actress_any_nominee",
        transform=_gg_any(
            "nominee",
            PrecursorKey.GOLDEN_GLOBE_ACTRESS_DRAMA,
            PrecursorKey.GOLDEN_GLOBE_ACTRESS_MUSICAL,
        ),
        available_from=available_after_nomination(AwardOrg.GOLDEN_GLOBE),
        description="Nominated for any Golden Globe Actress (Drama or Musical/Comedy)",
    ),
]


# ============================================================================
# 4. BP INTERACTION FEATURES
# ============================================================================

BP_INTERACTION_FEATURES: list[FeatureDefinition] = [
    FeatureDefinition(
        name="has_pga_dga_combo",
        transform=lambda r, _: (
            (r.precursors.get(PrecursorKey.PGA_BP, AwardResult()).winner is True)
            and (r.precursors.get(PrecursorKey.DGA_DIRECTING, AwardResult()).winner is True)
        ),
        available_from=available_after_winner(AwardOrg.PGA),
        description="Won both PGA BP and DGA Directing (~90% Oscar correlation). BP-specific.",
    ),
    FeatureDefinition(
        name="has_pga_dga_nomination_combo",
        transform=lambda r, _: (
            (r.precursors.get(PrecursorKey.PGA_BP, AwardResult()).nominee is True)
            and (r.precursors.get(PrecursorKey.DGA_DIRECTING, AwardResult()).nominee is True)
        ),
        available_from=available_after_nomination(AwardOrg.PGA),
        description="Nominated for both PGA BP and DGA Directing. BP-specific.",
    ),
]


# ============================================================================
# 5. PERSON CAREER FEATURES
# ============================================================================

PERSON_CAREER_FEATURES: list[FeatureDefinition] = [
    FeatureDefinition(
        name="person_prev_noms_same_category",
        transform=_person_prev_noms_same,
        description="Prior Oscar nominations in the same category",
    ),
    FeatureDefinition(
        name="person_prev_noms_any_category",
        transform=_person_prev_noms_any,
        description="Prior Oscar nominations in any category",
    ),
    FeatureDefinition(
        name="person_prev_wins_same_category",
        transform=_person_prev_wins_same,
        description="Prior Oscar wins in the same category",
    ),
    FeatureDefinition(
        name="person_prev_wins_any_category",
        transform=_person_prev_wins_any,
        description="Prior Oscar wins in any category",
    ),
    FeatureDefinition(
        name="person_is_overdue",
        transform=_person_is_overdue,
        description="3+ noms in same category with 0 wins (legacy/sympathy effect)",
    ),
    FeatureDefinition(
        name="film_also_bp_nominated",
        transform=_film_also_bp_nominated,
        available_from=available_after_oscar_noms(),
        description="Film is also nominated for Best Picture (prestige signal)",
    ),
]


# ============================================================================
# 6. PERSON ENRICHMENT FEATURES
# ============================================================================

PERSON_ENRICHMENT_FEATURES: list[FeatureDefinition] = [
    FeatureDefinition(
        name="person_age_at_ceremony",
        transform=_person_age_at_ceremony,
        description="Person's age at the Oscar ceremony",
    ),
    FeatureDefinition(
        name="person_total_film_credits",
        transform=_person_total_film_credits,
        description="Total filmography size from TMDb",
    ),
    FeatureDefinition(
        name="person_tmdb_popularity",
        transform=_person_tmdb_popularity,
        description="TMDb popularity score",
    ),
]


# ============================================================================
# 7. ANIMATED FEATURES
# ============================================================================

ANIMATED_FEATURES: list[FeatureDefinition] = [
    FeatureDefinition(
        name="is_sequel",
        transform=_is_sequel,
        description="Film title suggests it is a sequel",
    ),
]

# Generate studio identity features
for _studio_suffix, _fragments in _ANIMATION_STUDIOS:
    ANIMATED_FEATURES.append(
        FeatureDefinition(
            name=f"studio_{_studio_suffix}",
            transform=_make_studio_feature(_fragments),
            description=f"Produced by {_studio_suffix.replace('_', ' ').title()}",
        )
    )


# ============================================================================
# 8. VOTING SYSTEM FEATURES
# ============================================================================

VOTING_SYSTEM_FEATURES: list[FeatureDefinition] = [
    FeatureDefinition(
        name="is_irv_era",
        transform=lambda r, _: r.ceremony >= IRV_FIRST_CEREMONY,
        description="Best Picture winner chosen by IRV (ceremony >= 82, i.e. 2009+)",
    ),
    FeatureDefinition(
        name="nominees_in_year",
        transform=lambda r, ctx: len(ctx.records_by_ceremony.get(r.ceremony, [])),
        available_from=available_after_oscar_noms(),
        description="Number of nominees in this ceremony year (5 or 6-10)",
    ),
]


# ============================================================================
# 9. CRITIC SCORE FEATURES
# ============================================================================

# LR variant: consensus/gap metrics + percentile normalization
CRITIC_SCORE_LR_FEATURES: list[FeatureDefinition] = [
    FeatureDefinition(
        name="critics_consensus_score",
        transform=_critics_consensus_score,
        description="Average of RT and Metacritic",
    ),
    FeatureDefinition(
        name="critics_audience_gap",
        transform=_critics_audience_gap,
        description="RT - (IMDb*10); divisive films rarely win",
    ),
    FeatureDefinition(
        name="metacritic_percentile_in_year",
        transform=_make_within_year_percentile(_extract_metacritic),
        available_from=available_after_oscar_noms(),
        description="Percentile rank of Metacritic score within same year's nominees",
    ),
    FeatureDefinition(
        name="rt_percentile_in_year",
        transform=_make_within_year_percentile(_extract_rotten_tomatoes),
        available_from=available_after_oscar_noms(),
        description="Percentile rank of RT score within same year's nominees",
    ),
    FeatureDefinition(
        name="imdb_percentile_in_year",
        transform=_make_within_year_percentile(_extract_imdb_rating),
        available_from=available_after_oscar_noms(),
        description="Percentile rank of IMDb rating within same year's nominees",
    ),
]

# GBT variant: raw scores + z-score normalization
CRITIC_SCORE_GBT_FEATURES: list[FeatureDefinition] = [
    FeatureDefinition(
        name="imdb_rating",
        transform=lambda r, _: r.film.metadata.imdb_rating if r.film.metadata else None,
        description="IMDb rating (0-10)",
    ),
    FeatureDefinition(
        name="rotten_tomatoes",
        transform=lambda r, _: r.film.metadata.rotten_tomatoes if r.film.metadata else None,
        description="Rotten Tomatoes score (0-100)",
    ),
    FeatureDefinition(
        name="metacritic",
        transform=lambda r, _: r.film.metadata.metacritic if r.film.metadata else None,
        description="Metacritic score (0-100)",
    ),
    FeatureDefinition(
        name="metacritic_zscore_in_year",
        transform=_make_within_year_zscore(_extract_metacritic),
        available_from=available_after_oscar_noms(),
        description="Z-score of Metacritic score within same year's nominees",
    ),
    FeatureDefinition(
        name="rt_zscore_in_year",
        transform=_make_within_year_zscore(_extract_rotten_tomatoes),
        available_from=available_after_oscar_noms(),
        description="Z-score of RT score within same year's nominees",
    ),
    FeatureDefinition(
        name="imdb_zscore_in_year",
        transform=_make_within_year_zscore(_extract_imdb_rating),
        available_from=available_after_oscar_noms(),
        description="Z-score of IMDb rating within same year's nominees",
    ),
]


# ============================================================================
# 10. COMMERCIAL FEATURES
# ============================================================================

# LR variant: log transforms + percentile normalization
COMMERCIAL_LR_FEATURES: list[FeatureDefinition] = [
    FeatureDefinition(
        name="log_budget",
        transform=lambda r, _: safe_log10(r.film.metadata.budget) if r.film.metadata else None,
        description="Log-transformed budget",
    ),
    FeatureDefinition(
        name="log_box_office_worldwide",
        transform=lambda r, _: (
            safe_log10(r.film.metadata.box_office_worldwide) if r.film.metadata else None
        ),
        description="Log-transformed worldwide box office",
    ),
    FeatureDefinition(
        name="box_office_worldwide_percentile_in_year",
        transform=_make_within_year_percentile(_extract_box_office_worldwide),
        available_from=available_after_oscar_noms(),
        description="Percentile rank of worldwide box office within same year's nominees",
    ),
    FeatureDefinition(
        name="box_office_domestic_percentile_in_year",
        transform=_make_within_year_percentile(_extract_box_office_domestic),
        available_from=available_after_oscar_noms(),
        description="Percentile rank of domestic box office within same year's nominees",
    ),
    FeatureDefinition(
        name="budget_percentile_in_year",
        transform=_make_within_year_percentile(_extract_budget),
        available_from=available_after_oscar_noms(),
        description="Percentile rank of budget within same year's nominees",
    ),
]

# GBT variant: raw values + z-score normalization
COMMERCIAL_GBT_FEATURES: list[FeatureDefinition] = [
    FeatureDefinition(
        name="budget",
        transform=lambda r, _: r.film.metadata.budget if r.film.metadata else None,
        description="Production budget ($)",
    ),
    FeatureDefinition(
        name="box_office_domestic",
        transform=lambda r, _: r.film.metadata.box_office_domestic if r.film.metadata else None,
        description="US domestic box office ($)",
    ),
    FeatureDefinition(
        name="box_office_worldwide",
        transform=lambda r, _: r.film.metadata.box_office_worldwide if r.film.metadata else None,
        description="Worldwide box office ($)",
    ),
    FeatureDefinition(
        name="box_office_worldwide_zscore_in_year",
        transform=_make_within_year_zscore(_extract_box_office_worldwide),
        available_from=available_after_oscar_noms(),
        description="Z-score of worldwide box office within same year's nominees",
    ),
    FeatureDefinition(
        name="box_office_domestic_zscore_in_year",
        transform=_make_within_year_zscore(_extract_box_office_domestic),
        available_from=available_after_oscar_noms(),
        description="Z-score of domestic box office within same year's nominees",
    ),
    FeatureDefinition(
        name="budget_zscore_in_year",
        transform=_make_within_year_zscore(_extract_budget),
        available_from=available_after_oscar_noms(),
        description="Z-score of budget within same year's nominees",
    ),
]


# ============================================================================
# 11. TIMING FEATURES
# ============================================================================

# Shared timing features (both LR and GBT)
TIMING_FEATURES: list[FeatureDefinition] = [
    FeatureDefinition(
        name="release_month",
        transform=lambda r, _: (
            r.film.metadata.released.month if r.film.metadata and r.film.metadata.released else None
        ),
        description="Release month (1-12)",
    ),
    FeatureDefinition(
        name="is_awards_season_release",
        transform=lambda r, _: (
            r.film.metadata.released.month in [10, 11, 12]
            if r.film.metadata and r.film.metadata.released
            else False
        ),
        description="Released Oct-Dec (awards season)",
    ),
    FeatureDefinition(
        name="runtime_minutes",
        transform=lambda r, _: r.film.metadata.runtime_minutes if r.film.metadata else None,
        description="Film runtime in minutes",
    ),
]

# LR-specific: cyclical encoding + percentile
TIMING_LR_FEATURES: list[FeatureDefinition] = [
    FeatureDefinition(
        name="release_month_sin",
        transform=_release_month_sin,
        description="Cyclical month encoding (sin)",
    ),
    FeatureDefinition(
        name="release_month_cos",
        transform=_release_month_cos,
        description="Cyclical month encoding (cos)",
    ),
    FeatureDefinition(
        name="runtime_percentile_in_year",
        transform=_make_within_year_percentile(_extract_runtime),
        available_from=available_after_oscar_noms(),
        description="Percentile rank of runtime within same year's nominees",
    ),
]

# GBT-specific: z-score
TIMING_GBT_FEATURES: list[FeatureDefinition] = [
    FeatureDefinition(
        name="runtime_zscore_in_year",
        transform=_make_within_year_zscore(_extract_runtime),
        available_from=available_after_oscar_noms(),
        description="Z-score of runtime within same year's nominees",
    ),
]


# ============================================================================
# 12. FILM METADATA FEATURES
# ============================================================================

FILM_METADATA_FEATURES: list[FeatureDefinition] = [
    FeatureDefinition(
        name="genre_drama",
        transform=lambda r, _: (
            "drama" in {g.lower() for g in (r.film.metadata.genres if r.film.metadata else [])}
        ),
        description="Film is categorized as drama",
    ),
    FeatureDefinition(
        name="genre_biography",
        transform=lambda r, _: (
            "biography" in {g.lower() for g in (r.film.metadata.genres if r.film.metadata else [])}
        ),
        description="Film is categorized as biography",
    ),
    FeatureDefinition(
        name="genre_war",
        transform=lambda r, _: (
            "war" in {g.lower() for g in (r.film.metadata.genres if r.film.metadata else [])}
        ),
        description="Film is categorized as war",
    ),
    FeatureDefinition(
        name="genre_musical",
        transform=lambda r, _: any(
            g in {genre.lower() for genre in (r.film.metadata.genres if r.film.metadata else [])}
            for g in ["musical", "music"]
        ),
        description="Film is categorized as musical",
    ),
    FeatureDefinition(
        name="rated_r",
        transform=lambda r, _: r.film.metadata.rated == "R" if r.film.metadata else False,
        description="Film has R rating",
    ),
]


# ============================================================================
# ALL FEATURES (flat list for FEATURE_REGISTRY population)
# ============================================================================

ALL_FEATURE_LISTS: list[list[FeatureDefinition]] = [
    OSCAR_NOMINATION_FEATURES,
    PRECURSOR_FEATURES,
    PRECURSOR_AGGREGATE_FEATURES,
    GOLDEN_GLOBE_COMPOSITE_FEATURES,
    BP_INTERACTION_FEATURES,
    PERSON_CAREER_FEATURES,
    PERSON_ENRICHMENT_FEATURES,
    ANIMATED_FEATURES,
    VOTING_SYSTEM_FEATURES,
    CRITIC_SCORE_LR_FEATURES,
    CRITIC_SCORE_GBT_FEATURES,
    COMMERCIAL_LR_FEATURES,
    COMMERCIAL_GBT_FEATURES,
    TIMING_FEATURES,
    TIMING_LR_FEATURES,
    TIMING_GBT_FEATURES,
    FILM_METADATA_FEATURES,
]
