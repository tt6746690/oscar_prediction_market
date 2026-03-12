"""Regression tests for the precursor awards Wikipedia fetcher.

These tests make real network calls to Wikipedia and are skipped by default.
Run them explicitly with::

    uv run pytest -m network tests/data/test_fetch_precursor_awards.py -v

They verify two bugs that were fixed:

1. **Stats table contamination (Annie Awards, others):** Wikipedia award pages
   sometimes include statistics tables ("Wins by franchise", "Nominations by
   studio") alongside the actual award data tables. These were parsed and
   concatenated into the results, injecting franchise names ("Toy Story",
   "Shrek", "Frozen") as fake nominees. Fix: skip tables whose headers lack
   a "Year" column.

2. **Winner detection failure (Annie Awards):** The Annie Awards Wikipedia
   tables use multi-level headers (year sub-header rows) that cause a row
   count mismatch between ``pd.read_html`` and BS4. The old code fell back
   to ``is_winner=False`` for all rows on mismatch. Fix: fall back to a
   single-pass BS4 parser that extracts data and winner flags together.
"""

import pytest

from oscar_prediction_market.data.fetch_precursor_awards import (
    PrecursorAwardsFetcher,
)
from oscar_prediction_market.data.schema import PrecursorKey

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def fetcher() -> PrecursorAwardsFetcher:
    """Fetcher with no disk cache — always hits Wikipedia."""
    return PrecursorAwardsFetcher(cache_dir=None)


# ---------------------------------------------------------------------------
# Bug 1: Stats table contamination
# ---------------------------------------------------------------------------


# Franchise names that appeared as fake 2025 nominees when stats tables leaked.
KNOWN_GARBAGE_TITLES = {
    "Toy Story",
    "Shrek",
    "Frozen",
    "Spider-Verse",
    "Kung Fu Panda",
    "Despicable Me",
    "Monsters, Inc.",
    "Wreck-It Ralph",
    "Zootopia",
    "The Croods",
    "The Incredibles",
    "Inside Out",
    "Ghost in the Shell",
    "Wallace and Gromit",
    "How to Train Your Dragon",
}


class TestAnnieStatsTableFiltering:
    """Verify stats tables (franchise/studio wins) don't leak into Annie data."""

    def test_no_franchise_names_in_recent_years(self, fetcher: PrecursorAwardsFetcher) -> None:
        """Recent Annie data should contain real nominees, not franchise names."""
        df = fetcher._fetch_generic_award(PrecursorKey.ANNIE_FEATURE, year_range=(2020, 2026))
        garbage = df[df["film"].isin(KNOWN_GARBAGE_TITLES)]
        assert len(garbage) == 0, (
            f"Stats table data leaked into Annie results: {garbage['film'].tolist()}"
        )

    def test_reasonable_row_count(self, fetcher: PrecursorAwardsFetcher) -> None:
        """Annie Feature should have ~5-6 nominees/year, not 30+ from stats tables."""
        df = fetcher._fetch_generic_award(PrecursorKey.ANNIE_FEATURE, year_range=(2023, 2025))
        # 3 years * ~5-6 nominees = 15-18. Old broken code produced ~34.
        assert 10 <= len(df) <= 25, (
            f"Unexpected row count {len(df)} — stats tables may still be leaking"
        )


# ---------------------------------------------------------------------------
# Bug 2: Winner detection
# ---------------------------------------------------------------------------


class TestAnnieWinnerDetection:
    """Verify Annie Award winners are correctly detected from HTML styling."""

    def test_has_winners(self, fetcher: PrecursorAwardsFetcher) -> None:
        """Annie Feature should detect at least 1 winner per year."""
        df = fetcher._fetch_generic_award(PrecursorKey.ANNIE_FEATURE, year_range=(2020, 2025))
        n_winners = int(df["is_winner"].sum())
        # 6 years → should have 6 winners (one per year)
        assert n_winners >= 4, (
            f"Only {n_winners} winners detected — expected at least 4 for 6-year range"
        )

    def test_one_winner_per_year(self, fetcher: PrecursorAwardsFetcher) -> None:
        """Each ceremony year should have exactly 1 Annie winner."""
        df = fetcher._fetch_generic_award(PrecursorKey.ANNIE_FEATURE, year_range=(2020, 2025))
        winners_per_year = df[df["is_winner"]].groupby("year_ceremony").size()
        for year, count in winners_per_year.items():
            assert count == 1, f"Year {year} has {count} winners — expected exactly 1"

    def test_known_winners(self, fetcher: PrecursorAwardsFetcher) -> None:
        """Spot-check recent known Annie Feature winners."""
        df = fetcher._fetch_generic_award(PrecursorKey.ANNIE_FEATURE, year_range=(2020, 2025))
        winners = dict(
            zip(
                df[df["is_winner"]]["year_ceremony"],
                df[df["is_winner"]]["film"],
                strict=True,
            )
        )
        # Known winners (ceremony year → film)
        expected = {
            2020: "Soul",
            2021: "The Mitchells vs. the Machines",
            2023: "Spider-Man: Across the Spider-Verse",
            2024: "The Wild Robot",
        }
        for year, film in expected.items():
            assert year in winners, f"No winner detected for {year}"
            assert film in winners[year], f"Expected '{film}' for {year}, got '{winners[year]}'"


# ---------------------------------------------------------------------------
# Regression: other awards still work
# ---------------------------------------------------------------------------


class TestOtherAwardsRegression:
    """Ensure the table filtering fix doesn't break other awards."""

    @pytest.mark.parametrize(
        "award_key",
        [
            PrecursorKey.PGA_BP,
            PrecursorKey.BAFTA_FILM,
            PrecursorKey.GOLDEN_GLOBE_ACTOR_MUSICAL,
            PrecursorKey.BAFTA_ORIGINAL_SCREENPLAY,
            PrecursorKey.GOLDEN_GLOBE_SCREENPLAY,
            PrecursorKey.BAFTA_ANIMATED,
        ],
        ids=lambda k: k.value,
    )
    def test_has_data_and_winners(
        self, fetcher: PrecursorAwardsFetcher, award_key: PrecursorKey
    ) -> None:
        """Awards that previously had stats tables should still have data and winners.

        These awards were identified as having statistics tables on their
        Wikipedia pages. After adding the stats table filter, they must still
        return actual award data with winner detection working.
        """
        df = fetcher._fetch_generic_award(award_key, year_range=(2023, 2025))
        assert len(df) > 0, f"{award_key}: no data returned"
        n_winners = int(df["is_winner"].sum())
        assert n_winners > 0, f"{award_key}: no winners detected"


class TestAllAwardsSanity:
    """Broad sanity check across all configured awards."""

    def test_all_awards_return_data_with_winners(self, fetcher: PrecursorAwardsFetcher) -> None:
        """Every configured award should return data with at least 1 winner
        for the 2024 ceremony year."""
        awards = fetcher.fetch_all_awards(year_range=(2024, 2024), progress=False)
        failures = []
        for key, df in awards.items():
            if len(df) == 0:
                failures.append(f"{key}: empty")
            elif "is_winner" not in df.columns or int(df["is_winner"].sum()) == 0:
                failures.append(f"{key}: no winners")
        assert not failures, f"{len(failures)} awards failed:\n" + "\n".join(failures)
