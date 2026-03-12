"""Regression tests for feature engineering.

Tests transform_dataset() end-to-end with synthetic NominationRecords.
Hand-computed expected values ensure the refactoring doesn't change any
feature computation. Also tests feature groups, registry consistency,
and availability filtering.

These tests run against the PUBLIC API of the feature_engineering module.
Any internal restructuring that preserves the same outputs should pass.
"""

import math

import pytest

from oscar_prediction_market.data.awards_calendar import AwardsCalendar
from oscar_prediction_market.data.schema import (
    NominationDataset,
)
from oscar_prediction_market.modeling.feature_engineering import (
    FEATURE_REGISTRY,
    filter_features_by_availability,
    get_feature_names,
    resolve_features,
    transform_dataset,
)

# ============================================================================
# Test: transform_dataset produces correct values for known BP records
# ============================================================================


class TestTransformDatasetBP:
    """End-to-end tests: synthetic BP records → transform_dataset → expected values.

    Film A: Frontrunner (10 noms, won PGA+DGA+BAFTA+GG drama, metacritic=90)
    Film B: Contender (6 noms, won Critics Choice, metacritic=80)
    Film C: Dark horse (3 noms, no precursor wins, metacritic=70)
    """

    def test_oscar_nomination_features(self, bp_dataset_3_nominees: NominationDataset) -> None:
        """Oscar nomination counts and flags are extracted correctly."""
        all_features = list(FEATURE_REGISTRY.values())
        df = transform_dataset(bp_dataset_3_nominees, all_features)

        a = df[df["title"] == "Film A"].iloc[0]
        assert a["oscar_total_nominations"] == 10
        assert a["has_director_nomination"]
        assert a["has_editing_nomination"]
        assert a["has_acting_nomination"]
        assert a["acting_nomination_count"] == 1
        assert a["has_screenplay_nomination"]  # adapted_screenplay
        assert a["major_category_count"] == 4  # BP, directing, actor_leading, adapted_screenplay

        c = df[df["title"] == "Film C"].iloc[0]
        assert c["oscar_total_nominations"] == 3
        assert not c["has_director_nomination"]
        assert not c["has_editing_nomination"]
        assert not c["has_acting_nomination"]
        assert c["acting_nomination_count"] == 0
        assert c["has_screenplay_nomination"]  # original_screenplay
        assert c["major_category_count"] == 2  # BP, original_screenplay

    def test_precursor_winner_flags(self, bp_dataset_3_nominees: NominationDataset) -> None:
        """Individual precursor winner/nominee flags extracted from record."""
        all_features = list(FEATURE_REGISTRY.values())
        df = transform_dataset(bp_dataset_3_nominees, all_features)

        a = df[df["title"] == "Film A"].iloc[0]
        assert a["pga_bp_winner"]
        assert a["dga_directing_winner"]
        assert a["golden_globe_drama_winner"]
        assert not a["sag_ensemble_winner"]
        assert not a["critics_choice_picture_winner"]

        b = df[df["title"] == "Film B"].iloc[0]
        assert b["critics_choice_picture_winner"]
        assert not b["pga_bp_winner"]

    def test_golden_globe_composite(self, bp_dataset_3_nominees: NominationDataset) -> None:
        """GG composite: True if Drama OR Musical == True."""
        all_features = list(FEATURE_REGISTRY.values())
        df = transform_dataset(bp_dataset_3_nominees, all_features)

        # Film A: GG drama winner=True, musical winner=False → any=True
        a = df[df["title"] == "Film A"].iloc[0]
        assert a["golden_globe_any_winner"]

        # Film B: GG drama winner=False, musical winner=False → any=False
        b = df[df["title"] == "Film B"].iloc[0]
        assert not b["golden_globe_any_winner"]

    def test_pga_dga_combo(self, bp_dataset_3_nominees: NominationDataset) -> None:
        """PGA+DGA combo feature: True only if both are True."""
        all_features = list(FEATURE_REGISTRY.values())
        df = transform_dataset(bp_dataset_3_nominees, all_features)

        a = df[df["title"] == "Film A"].iloc[0]
        assert a["has_pga_dga_combo"]

        b = df[df["title"] == "Film B"].iloc[0]
        assert not b["has_pga_dga_combo"]

    def test_precursor_aggregate_counts(self, bp_dataset_3_nominees: NominationDataset) -> None:
        """precursor_wins_count and precursor_nominations_count.

        Film A: Won PGA, DGA, BAFTA, GG (4 distinct orgs) → count=4
        Film B: Won Critics Choice only → count=1
        Film C: No wins → count=0
        """
        all_features = list(FEATURE_REGISTRY.values())
        df = transform_dataset(bp_dataset_3_nominees, all_features)

        a = df[df["title"] == "Film A"].iloc[0]
        assert a["precursor_wins_count"] == 4  # pga, dga, bafta, gg (grouped by org)

        b = df[df["title"] == "Film B"].iloc[0]
        assert b["precursor_wins_count"] == 1  # critics_choice only

        c = df[df["title"] == "Film C"].iloc[0]
        assert c["precursor_wins_count"] == 0

    def test_within_year_percentile(self, bp_dataset_3_nominees: NominationDataset) -> None:
        """Percentile rank of metacritic within same year's nominees.

        Values: 90, 80, 70 → sorted: [70, 80, 90]
        Film A (90): 2 values below → 2/3 ≈ 0.667
        Film B (80): 1 value below → 1/3 ≈ 0.333
        Film C (70): 0 values below → 0/3 = 0.0
        """
        all_features = list(FEATURE_REGISTRY.values())
        df = transform_dataset(bp_dataset_3_nominees, all_features)

        a = df[df["title"] == "Film A"].iloc[0]
        assert a["metacritic_percentile_in_year"] == pytest.approx(2 / 3)

        b = df[df["title"] == "Film B"].iloc[0]
        assert b["metacritic_percentile_in_year"] == pytest.approx(1 / 3)

        c = df[df["title"] == "Film C"].iloc[0]
        assert c["metacritic_percentile_in_year"] == pytest.approx(0.0)

    def test_within_year_zscore(self, bp_dataset_3_nominees: NominationDataset) -> None:
        """Z-score of metacritic within same year's nominees.

        Values: 90, 80, 70 → mean=80, std=sqrt(200/3)≈8.165
        Film A (90): (90-80)/8.165 ≈ 1.2247
        Film B (80): (80-80)/8.165 = 0.0
        Film C (70): (70-80)/8.165 ≈ -1.2247
        """
        all_features = list(FEATURE_REGISTRY.values())
        df = transform_dataset(bp_dataset_3_nominees, all_features)

        std = math.sqrt(200 / 3)
        a = df[df["title"] == "Film A"].iloc[0]
        assert a["metacritic_zscore_in_year"] == pytest.approx(10 / std, abs=1e-4)

        b = df[df["title"] == "Film B"].iloc[0]
        assert b["metacritic_zscore_in_year"] == pytest.approx(0.0, abs=1e-6)

    def test_log_transforms(self, bp_dataset_3_nominees: NominationDataset) -> None:
        """Log10 transforms for LR features."""
        all_features = list(FEATURE_REGISTRY.values())
        df = transform_dataset(bp_dataset_3_nominees, all_features)

        a = df[df["title"] == "Film A"].iloc[0]
        assert a["log_budget"] == pytest.approx(math.log10(80_000_000))
        assert a["log_box_office_worldwide"] == pytest.approx(math.log10(300_000_000))

    def test_cyclical_release_month(self, bp_dataset_3_nominees: NominationDataset) -> None:
        """Cyclical encoding of release month (sin/cos)."""
        all_features = list(FEATURE_REGISTRY.values())
        df = transform_dataset(bp_dataset_3_nominees, all_features)

        # Film A: released 2024-11-15 → month=11
        a = df[df["title"] == "Film A"].iloc[0]
        assert a["release_month_sin"] == pytest.approx(math.sin(2 * math.pi * 11 / 12))
        assert a["release_month_cos"] == pytest.approx(math.cos(2 * math.pi * 11 / 12))

    def test_genre_flags(self, bp_dataset_3_nominees: NominationDataset) -> None:
        """Genre boolean features."""
        all_features = list(FEATURE_REGISTRY.values())
        df = transform_dataset(bp_dataset_3_nominees, all_features)

        a = df[df["title"] == "Film A"].iloc[0]
        assert a["genre_drama"]
        assert a["genre_biography"]
        assert not a["genre_war"]
        assert a["rated_r"]

        c = df[df["title"] == "Film C"].iloc[0]
        assert c["genre_war"]

    def test_voting_strategy(self, bp_dataset_3_nominees: NominationDataset) -> None:
        """IRV era and nominees_in_year."""
        all_features = list(FEATURE_REGISTRY.values())
        df = transform_dataset(bp_dataset_3_nominees, all_features)

        a = df[df["title"] == "Film A"].iloc[0]
        assert a["is_irv_era"]  # ceremony 97 >= 82
        assert a["nominees_in_year"] == 3

    def test_critics_features(self, bp_dataset_3_nominees: NominationDataset) -> None:
        """Critic consensus and audience gap computations."""
        all_features = list(FEATURE_REGISTRY.values())
        df = transform_dataset(bp_dataset_3_nominees, all_features)

        # Film A: consensus = (95+90)/2 = 92.5, gap = 95 - 8.5*10 = 10.0
        a = df[df["title"] == "Film A"].iloc[0]
        assert a["critics_consensus_score"] == pytest.approx(92.5)
        assert a["critics_audience_gap"] == pytest.approx(10.0)

    def test_nominations_percentile_in_year(self, bp_dataset_3_nominees: NominationDataset) -> None:
        """Percentile rank of total nominations within year.

        Values: 10, 6, 3 → sorted: [3, 6, 10]
        Film A (10): 2 below → 2/3
        Film B (6):  1 below → 1/3
        Film C (3):  0 below → 0/3
        """
        all_features = list(FEATURE_REGISTRY.values())
        df = transform_dataset(bp_dataset_3_nominees, all_features)

        a = df[df["title"] == "Film A"].iloc[0]
        assert a["nominations_percentile_in_year"] == pytest.approx(2 / 3)


# ============================================================================
# Test: Person features (actor dataset)
# ============================================================================


class TestTransformDatasetActor:
    """Tests for person-level features using actor dataset."""

    def test_person_career_features(self, actor_dataset_3_nominees: NominationDataset) -> None:
        """Person career features extracted from PersonData."""
        all_features = list(FEATURE_REGISTRY.values())
        df = transform_dataset(actor_dataset_3_nominees, all_features)

        x = df[df["title"] == "Film A"].iloc[0]
        assert x["person_prev_noms_same_category"] == 4
        assert x["person_prev_wins_same_category"] == 0
        assert x["person_is_overdue"]  # 4 noms, 0 wins

        z = df[df["title"] == "Film D"].iloc[0]
        assert z["person_prev_noms_same_category"] == 2
        assert z["person_prev_wins_same_category"] == 1
        assert not z["person_is_overdue"]  # only 2 noms

    def test_person_age_at_ceremony(self, actor_dataset_3_nominees: NominationDataset) -> None:
        """Age computed from birth_date and ceremony year.

        Actor X: born 1970-05-15, ceremony year 2025 → age 54 (birthday after March)
        Actor Y: born 1995-08-20, ceremony year 2025 → age 29 (birthday after March)
        Actor Z: born 1960-01-10, ceremony year 2025 → age 65 (birthday before March)
        """
        all_features = list(FEATURE_REGISTRY.values())
        df = transform_dataset(actor_dataset_3_nominees, all_features)

        x = df[df["title"] == "Film A"].iloc[0]
        assert x["person_age_at_ceremony"] == 54  # 2025-1970=55, month>3 → -1 = 54

        z = df[df["title"] == "Film D"].iloc[0]
        assert z["person_age_at_ceremony"] == 65  # 2025-1960=65, month<=3 → 65

    def test_film_also_bp_nominated(self, actor_dataset_3_nominees: NominationDataset) -> None:
        """film_also_bp_nominated checks oscar_nominations_by_category."""
        all_features = list(FEATURE_REGISTRY.values())
        df = transform_dataset(actor_dataset_3_nominees, all_features)

        x = df[df["title"] == "Film A"].iloc[0]
        assert x["film_also_bp_nominated"]

        z = df[df["title"] == "Film D"].iloc[0]
        assert not z["film_also_bp_nominated"]

    def test_golden_globe_actor_composite(
        self, actor_dataset_3_nominees: NominationDataset
    ) -> None:
        """GG actor composite: Drama OR Musical winner."""
        all_features = list(FEATURE_REGISTRY.values())
        df = transform_dataset(actor_dataset_3_nominees, all_features)

        # Actor X: GG drama winner=True → any=True
        x = df[df["title"] == "Film A"].iloc[0]
        assert x["golden_globe_actor_any_winner"]

        # Actor Y: GG drama winner=False, musical winner=False → any=False
        y = df[df["title"] == "Film B"].iloc[0]
        assert not y["golden_globe_actor_any_winner"]


# ============================================================================
# Test: Feature availability filtering
# ============================================================================


class TestFeatureAvailability:
    """Tests for filtering features by as_of_date."""

    def test_all_available_when_no_date(self, bp_dataset_3_nominees: NominationDataset) -> None:
        """No as_of_date → all features available."""
        all_features = list(FEATURE_REGISTRY.values())
        df = transform_dataset(bp_dataset_3_nominees, all_features, as_of_date=None)
        assert "pga_bp_winner" in df.columns
        assert "oscar_total_nominations" in df.columns

    def test_precursor_winners_unavailable_before_date(
        self,
        bp_dataset_3_nominees: NominationDataset,
        test_calendar: AwardsCalendar,
    ) -> None:
        """Precursor winner features unavailable before their announcement date.

        GG winners announced 2025-01-05. PGA winners announced 2025-02-08.
        On 2025-01-10: GG winner available, PGA winner not available.
        """
        from datetime import date

        all_features = list(FEATURE_REGISTRY.values())
        df = transform_dataset(
            bp_dataset_3_nominees,
            all_features,
            as_of_date=date(2025, 1, 10),
            calendar=test_calendar,
        )
        assert "golden_globe_drama_winner" in df.columns  # GG announced 2025-01-05
        assert "pga_bp_winner" not in df.columns  # PGA announced 2025-02-08

    def test_filter_features_by_availability(self, test_calendar: AwardsCalendar) -> None:
        """filter_features_by_availability returns only available features."""
        from datetime import date

        features = resolve_features(["golden_globe_drama_winner", "pga_bp_winner", "genre_drama"])
        available = filter_features_by_availability(
            features,
            as_of_date=date(2025, 1, 10),
            calendar=test_calendar,
        )
        names = get_feature_names(available)
        assert "golden_globe_drama_winner" in names
        assert "genre_drama" in names  # always available
        assert "pga_bp_winner" not in names


# ============================================================================
# Test: Feature registry consistency
# ============================================================================


class TestFeatureRegistry:
    """Tests for FEATURE_REGISTRY consistency."""

    def test_registry_not_empty(self) -> None:
        """Registry should contain a substantial number of features."""
        assert len(FEATURE_REGISTRY) > 100

    def test_all_features_have_descriptions(self) -> None:
        """Every feature definition must have a description."""
        for name, feat in FEATURE_REGISTRY.items():
            assert feat.description, f"Feature {name!r} has no description"

    def test_feature_names_match_keys(self) -> None:
        """Registry keys must match FeatureDefinition.name."""
        for key, feat in FEATURE_REGISTRY.items():
            assert key == feat.name, f"Key {key!r} != feature name {feat.name!r}"

    def test_resolve_features_round_trip(self) -> None:
        """resolve_features(names) returns definitions with those names."""
        names = ["oscar_total_nominations", "genre_drama", "pga_bp_winner"]
        defs = resolve_features(names)
        assert [d.name for d in defs] == names

    def test_resolve_features_raises_on_unknown(self) -> None:
        """resolve_features raises ValueError for unknown feature names."""
        with pytest.raises(ValueError, match="Unknown features"):
            resolve_features(["nonexistent_feature_xyz"])


# ============================================================================
# Test: Feature groups (from generate_feature_ablation_configs)
# ============================================================================


class TestFeatureGroupConsistency:
    """Tests that feature groups are consistent with the registry.

    Feature groups live in feature_engineering.groups and reference features
    by string name. These tests verify all referenced names exist.
    """

    def test_all_group_features_exist_in_registry(self) -> None:
        """Every feature name referenced by any group must exist in FEATURE_REGISTRY."""
        from oscar_prediction_market.modeling.feature_engineering import (
            ABLATION_CATEGORIES,
            get_feature_groups,
        )

        missing = []
        for category in ABLATION_CATEGORIES:
            groups = get_feature_groups(category)
            for group in groups:
                for feat_name in group.lr_features:
                    if feat_name not in FEATURE_REGISTRY:
                        missing.append((category.name, group.name, feat_name, "lr"))
                for feat_name in group.gbt_features:
                    if feat_name not in FEATURE_REGISTRY:
                        missing.append((category.name, group.name, feat_name, "gbt"))
        assert missing == [], f"Features in groups but not in FEATURE_REGISTRY: {missing}"

    def test_no_duplicate_features_within_category(self) -> None:
        """No feature appears twice when combining all groups for a category."""
        from oscar_prediction_market.modeling.feature_engineering import (
            ABLATION_CATEGORIES,
            FeatureFamily,
            get_all_features,
        )

        for category in ABLATION_CATEGORIES:
            for ff in [FeatureFamily.LR, FeatureFamily.GBT]:
                features = get_all_features(ff, category)
                duplicates = [f for f in features if features.count(f) > 1]
                assert duplicates == [], (
                    f"Duplicate features for {category.name}/{ff.value}: {set(duplicates)}"
                )
