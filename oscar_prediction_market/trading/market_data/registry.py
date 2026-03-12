"""Oscar market metadata registry — loaded from ``ticker_inventory.json``.

Transforms the raw Kalshi ticker inventory (produced by ``discover_tickers.py``)
into typed, validated Pydantic models for category × ceremony-year lookups.

The registry is loaded once at import time and exposed as the ``OSCAR_MARKETS``
singleton.  Consumer code should import from ``market_data`` (the package), not
from this module directly::

    from oscar_prediction_market.trading.market_data import (
        OSCAR_MARKETS,
        CategoryCeremonyData,
    )

This module intentionally does **not** import ``OscarMarket`` — that breaks the
one-way dependency: ``oscar_market`` → ``market_data``, never the reverse.
Callers that need an ``OscarMarket`` do a two-step construction::

    data = OSCAR_MARKETS.get_category_data(OscarCategory.BEST_PICTURE, 2026)
    mkt = OscarMarket(event_ticker=data.event_ticker, nominee_tickers=data.nominee_tickers)
"""

import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel, Field, computed_field

from oscar_prediction_market.data.schema import OscarCategory

logger = logging.getLogger(__name__)

_INVENTORY_PATH = Path(__file__).parent / "ticker_inventory.json"

# Maps Kalshi series tickers to OscarCategory enum names (lowercased).
# Update this when Kalshi adds new Oscar category markets.
_SERIES_TO_SLUG: dict[str, str] = {
    "KXOSCARPIC": "best_picture",
    "KXOSCARDIR": "directing",
    "KXOSCARACTO": "actor_leading",
    "KXOSCARACTR": "actress_leading",
    "KXOSCARSUPACTO": "actor_supporting",
    "KXOSCARSUPACTR": "actress_supporting",
    "KXOSCARSPLAY": "original_screenplay",
    "KXOSCARASPLAY": "adapted_screenplay",
    "KXOSCARANIMATED": "animated_feature",
    "KXOSCARCINE": "cinematography",
    "KXOSCARCOSTUME": "costume_design",
    "KXOSCARPROD": "art_direction",
    "KXOSCARSOUND": "sound",
    "KXOSCARSONG": "original_song",
    "KXOSCARSCORE": "original_score",
    "KXOSCAREDIT": "film_editing",
    "KXOSCARINTLFILM": "international_feature",
}

_SLUG_TO_SERIES: dict[str, str] = {v: k for k, v in _SERIES_TO_SLUG.items()}

# Raw inventory type: {series_ticker: {event_ticker: [{ticker, nominee, status}, ...]}}
RawInventory = dict[str, dict[str, list[dict[str, str]]]]


# ============================================================================
# Pydantic models
# ============================================================================


class CategoryCeremonyData(BaseModel):
    """Market data for one category in one ceremony year.

    Stores the event ticker and nominee→ticker mapping for a specific
    (category, year) pair.  The reverse mapping is computed automatically.

    Example::

        data = CategoryCeremonyData(
            event_ticker="KXOSCARPIC-26",
            nominee_tickers={"Sinners": "KXOSCARPIC-26-SIN", ...},
        )
        data.ticker_to_nominee["KXOSCARPIC-26-SIN"]  # "Sinners"
    """

    model_config = {"extra": "forbid"}

    event_ticker: str
    nominee_tickers: dict[str, str]

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def ticker_to_nominee(self) -> dict[str, str]:
        """Reverse mapping: ticker → nominee name."""
        return {v: k for k, v in self.nominee_tickers.items()}


class OscarMarketsRegistry(BaseModel):
    """Registry of all Oscar market tickers, loaded from the raw ticker inventory.

    Provides typed lookups by :class:`OscarCategory` and ceremony year.
    Category keys use lowercase enum names (e.g. ``"best_picture"``).

    Example::

        registry = OSCAR_MARKETS
        series = registry.get_series_ticker(OscarCategory.BEST_PICTURE)
        data = registry.get_category_data(OscarCategory.BEST_PICTURE, 2026)
    """

    model_config = {"extra": "forbid"}

    series_tickers: dict[str, str] = Field(
        ..., description="category_slug → year-independent series ticker"
    )
    ceremonies: dict[str, dict[str, CategoryCeremonyData]] = Field(
        ..., description="ceremony_year_str → category_slug → market data"
    )

    @staticmethod
    def _slug(category: OscarCategory) -> str:
        """Convert OscarCategory enum to registry key (lowercase enum name)."""
        return category.slug

    def get_series_ticker(self, category: OscarCategory) -> str:
        """Get the year-independent series ticker for a category.

        Raises ``KeyError`` if the category is not in the registry.
        """
        slug = self._slug(category)
        if slug not in self.series_tickers:
            raise KeyError(f"No series ticker for category {category!r}")
        return self.series_tickers[slug]

    def get_category_data(
        self, category: OscarCategory, ceremony_year: int
    ) -> CategoryCeremonyData:
        """Get full market data for a category in a specific ceremony year.

        Raises ``KeyError`` if the year or category is not in the registry.
        """
        year_str = str(ceremony_year)
        slug = self._slug(category)
        if year_str not in self.ceremonies:
            raise KeyError(f"No ceremony data for year {ceremony_year}")
        year_data = self.ceremonies[year_str]
        if slug not in year_data:
            raise KeyError(f"No data for category {category!r} in year {ceremony_year}")
        return year_data[slug]

    def get_event_ticker(self, category: OscarCategory, ceremony_year: int) -> str:
        """Get the event ticker for a category in a specific ceremony year."""
        return self.get_category_data(category, ceremony_year).event_ticker

    def get_nominee_tickers(self, category: OscarCategory, ceremony_year: int) -> dict[str, str]:
        """Get nominee → ticker mapping for a category in a ceremony year."""
        return self.get_category_data(category, ceremony_year).nominee_tickers

    def categories_for_year(self, ceremony_year: int) -> list[OscarCategory]:
        """List categories that have market data for a ceremony year."""
        year_str = str(ceremony_year)
        if year_str not in self.ceremonies:
            return []
        slugs = self.ceremonies[year_str].keys()
        result = []
        for cat in OscarCategory:
            if cat.slug in slugs:
                result.append(cat)
        return result


# ============================================================================
# Loading logic — transforms raw inventory format at import time
# ============================================================================


def _extract_ceremony_year(event_ticker: str) -> int | None:
    """Extract the ceremony year from an event ticker.

    Patterns::

        KXOSCARPIC-25        → 2025
        OSCARPIC-22          → 2022
        KXOSCARANIMATED-26B  → 2026
        OSCARASPLAY          → None (no year suffix, skip)
    """
    match = re.search(r"-(\d{2})B?$", event_ticker)
    if not match:
        return None
    return 2000 + int(match.group(1))


def _is_junk_nominee(nominee: str) -> bool:
    """Return True for nominee names that are junk data from early Kalshi markets.

    The 2022 markets had nominee names stored as ``"{}"`` (broken JSON
    extraction).  Filter these out so they don't pollute the registry.
    """
    stripped = nominee.strip()
    return stripped in ("", "{}", "None")


def _load_from_inventory(path: Path) -> OscarMarketsRegistry:
    """Load and transform the raw ticker inventory into an OscarMarketsRegistry.

    The raw inventory has the shape::

        {series_ticker: {event_ticker: [{ticker, nominee, status}, ...]}}

    This function:
    1. Maps series tickers to category slugs
    2. Extracts ceremony years from event tickers
    3. Builds nominee→ticker mappings, filtering out "Tie" and junk entries
    4. Constructs the typed registry
    """
    raw: RawInventory = json.loads(path.read_text())

    series_tickers: dict[str, str] = {}
    ceremonies: dict[str, dict[str, CategoryCeremonyData]] = {}

    for series_ticker, events_data in raw.items():
        slug = _SERIES_TO_SLUG.get(series_ticker)
        if slug is None:
            logger.debug("Unknown series ticker %s, skipping", series_ticker)
            continue

        series_tickers[slug] = series_ticker

        for event_ticker, markets in events_data.items():
            year = _extract_ceremony_year(event_ticker)
            if year is None:
                logger.debug("Cannot extract year from %s, skipping", event_ticker)
                continue

            # Build nominee_tickers, excluding "Tie" and junk entries
            nominee_tickers: dict[str, str] = {}
            for market in markets:
                nominee = market["nominee"]
                if nominee.lower().strip() == "tie":
                    continue
                if _is_junk_nominee(nominee):
                    continue
                nominee_tickers[nominee] = market["ticker"]

            year_str = str(year)
            if year_str not in ceremonies:
                ceremonies[year_str] = {}

            ceremonies[year_str][slug] = CategoryCeremonyData(
                event_ticker=event_ticker,
                nominee_tickers=nominee_tickers,
            )

    return OscarMarketsRegistry(
        series_tickers=series_tickers,
        ceremonies=ceremonies,
    )


#: Singleton registry instance, loaded at import time from ticker_inventory.json.
OSCAR_MARKETS = _load_from_inventory(_INVENTORY_PATH)
