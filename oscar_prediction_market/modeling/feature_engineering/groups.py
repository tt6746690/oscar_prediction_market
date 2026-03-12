"""Feature groups — semantic groupings of features for ablation and config generation.

This is the single source of truth for which features belong to which group,
and which features are used for each model_type × category combination.

Feature Groups (universal, all categories):
    precursor_winners   — Category-specific precursor winner flags + composites + aggregates
    precursor_noms      — Category-specific precursor nomination flags + composites + aggregates
    oscar_nominations   — Cross-category Oscar nomination profile (minus constant features)
    critic_scores       — Review aggregator scores (percentile for LR, z-score for GBT)
    commercial          — Budget and box office (log for LR, raw for GBT)
    timing              — Release timing and runtime
    film_metadata       — Genre indicators and MPAA rating

Feature Groups (conditional):
    person_career       — Oscar history: prior noms/wins, overdue, BP-nominated film
                          (PERSON_CATEGORIES: acting, directing, screenplay, cinematography)
    person_enrichment   — TMDb person data: age, credits, popularity
                          (acting + directing only; insufficient coverage for screenplay/cinematography)
    animated_specific   — Studio identity + sequel detection (Animated Feature only)
    voting_system       — IRV era + nominee count (Best Picture only)
"""

from pydantic import BaseModel, field_validator

from oscar_prediction_market.data.precursor_mappings import (
    CATEGORY_PRECURSORS,
)
from oscar_prediction_market.data.schema import (
    PERSON_CATEGORIES,
    OscarCategory,
)
from oscar_prediction_market.modeling.feature_engineering.types import (
    FeatureFamily,
)

# ============================================================================
# Category Constants
# ============================================================================

ABLATION_CATEGORIES: list[OscarCategory] = [
    OscarCategory.BEST_PICTURE,
    OscarCategory.DIRECTING,
    OscarCategory.ACTOR_LEADING,
    OscarCategory.ACTRESS_LEADING,
    OscarCategory.ACTOR_SUPPORTING,
    OscarCategory.ACTRESS_SUPPORTING,
    OscarCategory.ORIGINAL_SCREENPLAY,
    OscarCategory.CINEMATOGRAPHY,
    OscarCategory.ANIMATED_FEATURE,
]

# Person categories with sufficient TMDb enrichment coverage (acting + directing).
# Excludes Original Screenplay (4% TMDb) and Cinematography (53% TMDb).
PERSON_ENRICHMENT_CATEGORIES: frozenset[OscarCategory] = frozenset(
    {
        OscarCategory.ACTOR_LEADING,
        OscarCategory.ACTRESS_LEADING,
        OscarCategory.ACTOR_SUPPORTING,
        OscarCategory.ACTRESS_SUPPORTING,
        OscarCategory.DIRECTING,
    }
)


# ============================================================================
# Golden Globe Composite & Constant Feature Mappings
# ============================================================================

# Categories with GG genre splits need composite features (Drama OR Musical).
# Other categories have a single GG precursor → no composite needed.
_GG_COMPOSITE_WINNERS: dict[OscarCategory, str] = {
    OscarCategory.BEST_PICTURE: "golden_globe_any_winner",
    OscarCategory.ACTOR_LEADING: "golden_globe_actor_any_winner",
    OscarCategory.ACTRESS_LEADING: "golden_globe_actress_any_winner",
}

_GG_COMPOSITE_NOMINEES: dict[OscarCategory, str] = {
    OscarCategory.BEST_PICTURE: "golden_globe_any_nominee",
    OscarCategory.ACTOR_LEADING: "golden_globe_actor_any_nominee",
    OscarCategory.ACTRESS_LEADING: "golden_globe_actress_any_nominee",
}

# Oscar nomination features that are constant (always True) for specific categories.
# These produce zero-variance columns and must be excluded.
_CONSTANT_OSCAR_FEATURES: dict[OscarCategory, list[str]] = {
    OscarCategory.DIRECTING: ["has_director_nomination"],
    OscarCategory.ACTOR_LEADING: ["has_acting_nomination"],
    OscarCategory.ACTRESS_LEADING: ["has_acting_nomination"],
    OscarCategory.ACTOR_SUPPORTING: ["has_acting_nomination"],
    OscarCategory.ACTRESS_SUPPORTING: ["has_acting_nomination"],
    OscarCategory.ORIGINAL_SCREENPLAY: ["has_screenplay_nomination"],
}


# ============================================================================
# Feature Group Model
# ============================================================================


class FeatureGroup(BaseModel):
    """A semantic group of related features.

    lr_features and gbt_features differ because LR uses engineered features
    (percentile, log, cyclical) while GBT uses raw values + z-scores.
    Precursor and categorical features are typically identical for both.

    Feature names are validated against FEATURE_REGISTRY at construction time.
    """

    model_config = {"extra": "forbid"}

    name: str
    description: str
    lr_features: list[str]
    gbt_features: list[str]

    @field_validator("lr_features", "gbt_features")
    @classmethod
    def _validate_feature_names(cls, features: list[str]) -> list[str]:
        """Validate that all feature names exist in FEATURE_REGISTRY."""
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
# Feature Group Builders (category-aware)
# ============================================================================


def _precursor_winners_group(category: OscarCategory) -> FeatureGroup:
    """Precursor award winner flags + GG composites + BP interactions + aggregate.

    Includes:
    - Individual {key}_winner for each precursor in CATEGORY_PRECURSORS
    - GG composite winner for categories with genre-split GG (BP, actor, actress leading)
    - has_pga_dga_combo for BP
    - precursor_wins_count aggregate
    """
    specs = CATEGORY_PRECURSORS[category]
    features = [f"{spec.key}_winner" for spec in specs]

    if category in _GG_COMPOSITE_WINNERS:
        features.append(_GG_COMPOSITE_WINNERS[category])

    if category == OscarCategory.BEST_PICTURE:
        features.append("has_pga_dga_combo")

    features.append("precursor_wins_count")

    return FeatureGroup(
        name="precursor_winners",
        description=f"Precursor award winners for {category.name}",
        lr_features=features,
        gbt_features=features,
    )


def _precursor_noms_group(category: OscarCategory) -> FeatureGroup:
    """Precursor award nomination flags + GG composites + BP interactions + aggregate."""
    specs = CATEGORY_PRECURSORS[category]
    features = [f"{spec.key}_nominee" for spec in specs]

    if category in _GG_COMPOSITE_NOMINEES:
        features.append(_GG_COMPOSITE_NOMINEES[category])

    if category == OscarCategory.BEST_PICTURE:
        features.append("has_pga_dga_nomination_combo")

    features.append("precursor_nominations_count")

    return FeatureGroup(
        name="precursor_noms",
        description=f"Precursor award nominations for {category.name}",
        lr_features=features,
        gbt_features=features,
    )


def _oscar_nominations_group(category: OscarCategory) -> FeatureGroup:
    """Cross-category Oscar nomination profile.

    Removes features that are trivially constant for the given category:
    - has_director_nomination for Directing (always True)
    - has_acting_nomination for all 4 acting categories (always True)
    - has_screenplay_nomination for Original Screenplay (always True)
    """
    base = [
        "oscar_total_nominations",
        "has_director_nomination",
        "has_editing_nomination",
        "has_acting_nomination",
        "acting_nomination_count",
        "has_screenplay_nomination",
        "major_category_count",
    ]

    constants = set(_CONSTANT_OSCAR_FEATURES.get(category, []))
    base = [f for f in base if f not in constants]

    lr_features = base + ["nominations_percentile_in_year"]
    gbt_features = base.copy()

    return FeatureGroup(
        name="oscar_nominations",
        description="Cross-category Oscar nomination profile",
        lr_features=lr_features,
        gbt_features=gbt_features,
    )


def _person_career_group() -> FeatureGroup:
    """Oscar career history for person categories.

    Prior nominations/wins (same and any category), overdue flag, and whether
    the person's film is also nominated for Best Picture (prestige signal).
    """
    features = [
        "person_prev_noms_same_category",
        "person_prev_noms_any_category",
        "person_prev_wins_same_category",
        "person_prev_wins_any_category",
        "person_is_overdue",
        "film_also_bp_nominated",
    ]
    return FeatureGroup(
        name="person_career",
        description="Oscar career history (prior noms/wins, overdue, BP-nominated film)",
        lr_features=features,
        gbt_features=features,
    )


def _person_enrichment_group() -> FeatureGroup:
    """TMDb person enrichment data.

    Only for categories with good TMDb coverage (acting 4 + directing: ~92%).
    Excluded for screenplay (4%) and cinematography (53%).
    """
    features = [
        "person_age_at_ceremony",
        "person_total_film_credits",
        "person_tmdb_popularity",
    ]
    return FeatureGroup(
        name="person_enrichment",
        description="TMDb person data (age, credits, popularity)",
        lr_features=features,
        gbt_features=features,
    )


def _animated_specific_group() -> FeatureGroup:
    """Animation-specific features: studio identity and sequel detection."""
    features = [
        "is_sequel",
        "studio_disney_pixar",
        "studio_dreamworks",
        "studio_illumination",
        "studio_sony_animation",
        "studio_laika",
    ]
    return FeatureGroup(
        name="animated_specific",
        description="Animation studio identity and sequel detection",
        lr_features=features,
        gbt_features=features,
    )


def _voting_system_group() -> FeatureGroup:
    """BP-only voting methodology features.

    IRV era (post-2009) and variable nominee count (5 vs 8-10).
    Other categories always have 5 nominees and use plurality voting.
    """
    features = [
        "is_irv_era",
        "nominees_in_year",
    ]
    return FeatureGroup(
        name="voting_system",
        description="BP voting methodology (plurality vs IRV) and nominee count",
        lr_features=features,
        gbt_features=features,
    )


def _critic_scores_group() -> FeatureGroup:
    """Review aggregator scores.

    LR uses consensus/gap metrics + percentile-in-year normalization.
    GBT uses raw scores + z-score-in-year normalization.
    """
    return FeatureGroup(
        name="critic_scores",
        description="Review aggregator scores and derived consensus metrics",
        lr_features=[
            "critics_consensus_score",
            "critics_audience_gap",
            "metacritic_percentile_in_year",
            "rt_percentile_in_year",
            "imdb_percentile_in_year",
        ],
        gbt_features=[
            "imdb_rating",
            "rotten_tomatoes",
            "metacritic",
            "metacritic_zscore_in_year",
            "rt_zscore_in_year",
            "imdb_zscore_in_year",
        ],
    )


def _commercial_group() -> FeatureGroup:
    """Budget and box office performance.

    LR uses log transforms + percentile-in-year.
    GBT uses raw values + z-score-in-year.
    """
    return FeatureGroup(
        name="commercial",
        description="Budget and box office performance",
        lr_features=[
            "log_budget",
            "log_box_office_worldwide",
            "box_office_worldwide_percentile_in_year",
            "box_office_domestic_percentile_in_year",
            "budget_percentile_in_year",
        ],
        gbt_features=[
            "budget",
            "box_office_domestic",
            "box_office_worldwide",
            "box_office_worldwide_zscore_in_year",
            "box_office_domestic_zscore_in_year",
            "budget_zscore_in_year",
        ],
    )


def _timing_group() -> FeatureGroup:
    """Release timing and runtime.

    LR uses cyclical encoding (sin/cos) for release month.
    GBT uses raw month (trees handle non-linear naturally).
    """
    return FeatureGroup(
        name="timing",
        description="Release timing and runtime features",
        lr_features=[
            "release_month",
            "release_month_sin",
            "release_month_cos",
            "is_awards_season_release",
            "runtime_minutes",
            "runtime_percentile_in_year",
        ],
        gbt_features=[
            "release_month",
            "is_awards_season_release",
            "runtime_minutes",
            "runtime_zscore_in_year",
        ],
    )


def _film_metadata_group() -> FeatureGroup:
    """Genre indicators and MPAA rating. Same for LR and GBT."""
    features = [
        "genre_drama",
        "genre_biography",
        "genre_war",
        "genre_musical",
        "rated_r",
    ]
    return FeatureGroup(
        name="film_metadata",
        description="Genre indicators and MPAA rating",
        lr_features=features,
        gbt_features=features,
    )


def get_feature_groups(category: OscarCategory) -> list[FeatureGroup]:
    """Return ordered feature groups for a category.

    Group order determines additive ablation priority (most predictive first):
    1. precursor_winners     (strongest signal)
    2. precursor_noms        (available earlier)
    3. oscar_nominations     (strong baseline)
    4. [conditional groups]  (person_career, person_enrichment, animated, voting_system)
    5. critic_scores         (supplementary)
    6. commercial            (supplementary)
    7. timing                (supplementary)
    8. film_metadata         (supplementary)
    """
    groups = [
        _precursor_winners_group(category),
        _precursor_noms_group(category),
        _oscar_nominations_group(category),
    ]

    # Conditional groups inserted between core and supplementary
    if category in PERSON_CATEGORIES:
        groups.append(_person_career_group())
    if category in PERSON_ENRICHMENT_CATEGORIES:
        groups.append(_person_enrichment_group())
    if category == OscarCategory.ANIMATED_FEATURE:
        groups.append(_animated_specific_group())
    if category == OscarCategory.BEST_PICTURE:
        groups.append(_voting_system_group())

    # Supplementary groups (tail)
    groups.extend(
        [
            _critic_scores_group(),
            _commercial_group(),
            _timing_group(),
            _film_metadata_group(),
        ]
    )

    return groups


def get_all_features(feature_family: FeatureFamily, category: OscarCategory) -> list[str]:
    """Get all features for a feature family and category.

    This is the canonical way to get the full feature list for a feature_family × category
    combination. Replaces the old category-agnostic get_features_for_model().
    """
    groups = get_feature_groups(category)
    features = []
    for group in groups:
        if feature_family == FeatureFamily.LR:
            features.extend(group.lr_features)
        else:
            features.extend(group.gbt_features)
    return features
