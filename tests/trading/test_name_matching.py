"""Tests for name_matching — nominee name normalization and matching.

Verifies the multi-stage matching pipeline:
1. normalize_name strips accents, prefixes, articles, punctuation
2. match_nominees applies exact → fuzzy → film-title fallback in order
"""

from oscar_prediction_market.data.schema import OscarCategory
from oscar_prediction_market.trading.name_matching import (
    match_nominees,
    normalize_name,
)

# ---------------------------------------------------------------------------
# normalize_name
# ---------------------------------------------------------------------------


class TestNormalizeName:
    """Name normalization: accents, prefixes, articles, punctuation."""

    def test_strips_accents(self) -> None:
        assert normalize_name("Timothée Chalamet") == "timothee chalamet"

    def test_strips_the_prefix(self) -> None:
        assert normalize_name("The Brutalist") == "brutalist"

    def test_strips_written_by_prefix(self) -> None:
        assert normalize_name("Written by Sean Baker") == "sean baker"

    def test_strips_screenplay_by_prefix(self) -> None:
        assert normalize_name("Screenplay by Jesse Eisenberg") == "jesse eisenberg"

    def test_removes_punctuation(self) -> None:
        # Forward slash, apostrophe — all removed
        assert normalize_name("Wo/Men") == "women"
        assert normalize_name("It's a Wonderful Life") == "its a wonderful life"

    def test_collapses_whitespace(self) -> None:
        assert normalize_name("  Adrien   Brody  ") == "adrien brody"

    def test_already_normalized_is_idempotent(self) -> None:
        assert normalize_name("anora") == "anora"


# ---------------------------------------------------------------------------
# match_nominees
# ---------------------------------------------------------------------------


class TestMatchNominees:
    """Multi-stage matching: exact → fuzzy → film-title fallback."""

    def test_exact_match_after_normalization(self) -> None:
        """Accent-only differences → exact match on normalized form.

        Example: "Timothée Chalamet" normalizes the same as "Timothee Chalamet".
        """
        result = match_nominees(
            model_names=["Timothée Chalamet", "Adrien Brody"],
            kalshi_names=["Timothee Chalamet", "Adrien Brody"],
            category=OscarCategory.ACTOR_LEADING,
            ceremony_year=2025,
        )
        assert result == {
            "Timothée Chalamet": "Timothee Chalamet",
            "Adrien Brody": "Adrien Brody",
        }

    def test_fuzzy_match_above_threshold(self) -> None:
        """Minor spelling variation → fuzzy match kicks in.

        "Ralph Feinnes" vs "Ralph Fiennes" — Levenshtein ratio ~91, above 85 default.
        """
        result = match_nominees(
            model_names=["Ralph Feinnes"],
            kalshi_names=["Ralph Fiennes"],
            category=OscarCategory.ACTOR_LEADING,
            ceremony_year=2025,
        )
        assert "Ralph Feinnes" in result
        assert result["Ralph Feinnes"] == "Ralph Fiennes"

    def test_no_match_below_threshold(self) -> None:
        """Completely different names → no match."""
        result = match_nominees(
            model_names=["John Smith"],
            kalshi_names=["Jane Doe"],
            category=OscarCategory.ACTOR_LEADING,
            ceremony_year=2025,
        )
        assert result == {}

    def test_film_title_fallback_for_screenplay(self) -> None:
        """Screenplay: model has writer name, Kalshi has film title.

        model_film_titles maps writer → film, which matches Kalshi.
        """
        result = match_nominees(
            model_names=["Sean Baker"],
            kalshi_names=["Anora"],
            category=OscarCategory.ORIGINAL_SCREENPLAY,
            ceremony_year=2025,
            model_film_titles={"Sean Baker": "Anora"},
        )
        assert result == {"Sean Baker": "Anora"}

    def test_empty_inputs(self) -> None:
        result = match_nominees(
            model_names=[],
            kalshi_names=["Anora"],
            category=OscarCategory.BEST_PICTURE,
            ceremony_year=2025,
        )
        assert result == {}

    def test_the_prefix_matching(self) -> None:
        """'The Brutalist' should match 'Brutalist' after normalization.

        Both normalize to 'brutalist'.
        """
        result = match_nominees(
            model_names=["The Brutalist"],
            kalshi_names=["Brutalist"],
            category=OscarCategory.BEST_PICTURE,
            ceremony_year=2025,
        )
        assert result == {"The Brutalist": "Brutalist"}
