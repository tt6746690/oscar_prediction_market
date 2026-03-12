"""
Metadata fetcher module for Oscar film data.

Fetches film metadata from OMDb API using IMDb IDs.
Includes caching to avoid repeated API calls.

Usage:
    from fetch_metadata import OMDbFetcher

    fetcher = OMDbFetcher(api_key="your_key")  # or use default demo key
    metadata = fetcher.fetch("tt0137523")  # Fight Club
"""

import json
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import requests
from diskcache import Cache
from pydantic import BaseModel, Field

from oscar_prediction_market.constants import OMDB_API_KEY, OMDB_CACHE_DIR


def parse_release_date(date_str: str | None) -> date | None:
    """Parse OMDb release date string to date object.

    Args:
        date_str: Release date string from OMDb (e.g., "01 Oct 1999", "N/A")

    Returns:
        Parsed date object, or None if parsing fails
    """
    if not date_str or date_str == "N/A":
        return None

    formats = [
        "%d %b %Y",  # "01 Oct 1999" - most common OMDb format
        "%Y-%m-%d",  # "2023-07-19" - ISO format
        "%B %d, %Y",  # "October 1, 1999"
        "%m/%d/%Y",  # "10/01/1999"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


# Default cache directory


class OMDbParsedAwards(BaseModel):
    """Parsed awards counts from OMDb awards text."""

    model_config = {"extra": "forbid"}

    oscar_wins: int = Field(default=0, ge=0)
    oscar_nominations: int = Field(default=0, ge=0)
    total_wins: int = Field(default=0, ge=0)
    total_nominations: int = Field(default=0, ge=0)


class OMDbResult(BaseModel):
    """Parsed metadata from OMDb API response."""

    model_config = {"extra": "forbid"}

    imdb_id: str | None = None
    title: str | None = None
    year: str | None = None
    rated: str | None = None
    released: date | None = None
    runtime_minutes: int | None = None
    genres: list[str] = Field(default_factory=list)
    director: str | None = None
    writer: str | None = None
    actors: list[str] = Field(default_factory=list)
    plot: str | None = None
    language: str | None = None
    country: str | None = None
    box_office_domestic: int | None = Field(default=None, ge=0)
    imdb_rating: float | None = Field(default=None, ge=0, le=10)
    imdb_votes: int | None = Field(default=None, ge=0)
    rotten_tomatoes: int | None = Field(default=None, ge=0, le=100)
    metacritic: int | None = Field(default=None, ge=0, le=100)
    awards_text: str | None = None
    awards_parsed: OMDbParsedAwards = Field(default_factory=OMDbParsedAwards)
    poster_url: str | None = None


class OMDbFetcher:
    """Fetch film metadata from OMDb API with caching."""

    BASE_URL = "http://www.omdbapi.com/"

    def __init__(
        self,
        api_key: str = OMDB_API_KEY,
        cache_dir: Path | str | None = OMDB_CACHE_DIR,
        rate_limit_delay: float = 0.25,  # Seconds between
    ):
        """
        Initialize the OMDb fetcher.

        Args:
            api_key: OMDb API key. Default is demo key (1000 requests/day).
            cache_dir: Directory for disk cache. None to disable caching.
            rate_limit_delay: Seconds to wait between API calls.
        """
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0.0

        if cache_dir:
            self.cache = Cache(str(cache_dir))
        else:
            self.cache = None

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def fetch_raw(self, imdb_id: str) -> dict[str, Any]:
        """
        Fetch raw OMDb response for a film.

        Args:
            imdb_id: IMDb ID (e.g., "tt0137523")

        Returns:
            Raw OMDb API response as dict.
        """
        # Check cache first
        cache_key = f"omdb:{imdb_id}"
        if self.cache is not None and cache_key in self.cache:
            return self.cache[cache_key]  # Return immediately, no rate limiting

        # Rate limit only for actual API calls
        self._rate_limit()

        # Make API request
        params = {
            "i": imdb_id,
            "apikey": self.api_key,
            "plot": "short",
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            return {"Response": "False", "Error": str(e)}

        # Cache only successful responses with valid data (Response=True and Title exists)
        # This prevents caching partial/malformed responses
        if self.cache is not None and data.get("Response") == "True" and data.get("Title"):
            self.cache[cache_key] = data

        return data

    def fetch(self, imdb_id: str) -> OMDbResult | None:
        """
        Fetch and parse film metadata.

        Args:
            imdb_id: IMDb ID (e.g., "tt0137523")

        Returns:
            Parsed OMDbResult, or None if not found.
        """
        raw = self.fetch_raw(imdb_id)

        if raw.get("Response") != "True":
            return None

        return self._parse_response(raw)

    def _parse_response(self, raw: dict[str, Any]) -> OMDbResult:
        """Parse raw OMDb response into structured metadata."""

        # Extract ratings
        imdb_rating: float | None = None
        rotten_tomatoes: int | None = None
        metacritic: int | None = None

        for rating in raw.get("Ratings", []):
            source = rating.get("Source", "")
            value = rating.get("Value", "")

            if "Internet Movie Database" in source:
                # "8.8/10" -> 8.8
                try:
                    imdb_rating = float(value.split("/")[0])
                except (ValueError, IndexError):
                    pass
            elif "Rotten Tomatoes" in source:
                # "93%" -> 93
                try:
                    rotten_tomatoes = int(value.replace("%", ""))
                except ValueError:
                    pass
            elif "Metacritic" in source:
                # "81/100" -> 81
                try:
                    metacritic = int(value.split("/")[0])
                except (ValueError, IndexError):
                    pass

        # Also get metascore directly if available
        if "Metascore" in raw and raw["Metascore"] != "N/A":
            try:
                metacritic = int(raw["Metascore"])
            except ValueError:
                pass

        # Parse box office
        box_office = None
        if raw.get("BoxOffice") and raw["BoxOffice"] != "N/A":
            try:
                # "$37,030,102" -> 37030102
                box_office = int(raw["BoxOffice"].replace("$", "").replace(",", ""))
            except ValueError:
                pass

        # Parse runtime
        runtime = None
        if raw.get("Runtime") and raw["Runtime"] != "N/A":
            try:
                # "139 min" -> 139
                runtime = int(raw["Runtime"].replace(" min", ""))
            except ValueError:
                pass

        # Parse awards string
        awards_parsed = self._parse_awards(raw.get("Awards", ""))

        # Parse imdb votes
        imdb_votes = None
        if raw.get("imdbVotes") and raw["imdbVotes"] != "N/A":
            try:
                imdb_votes = int(raw["imdbVotes"].replace(",", ""))
            except ValueError:
                pass

        return OMDbResult(
            imdb_id=raw.get("imdbID"),
            title=raw.get("Title"),
            year=raw.get("Year"),
            rated=raw.get("Rated"),
            released=parse_release_date(raw.get("Released")),
            runtime_minutes=runtime,
            genres=[g.strip() for g in raw.get("Genre", "").split(",") if g.strip()],
            director=raw.get("Director"),
            writer=raw.get("Writer"),
            actors=[a.strip() for a in raw.get("Actors", "").split(",") if a.strip()],
            plot=raw.get("Plot"),
            language=raw.get("Language"),
            country=raw.get("Country"),
            box_office_domestic=box_office,
            imdb_rating=imdb_rating,
            imdb_votes=imdb_votes,
            rotten_tomatoes=rotten_tomatoes,
            metacritic=metacritic,
            awards_text=raw.get("Awards"),
            awards_parsed=awards_parsed,
            poster_url=raw.get("Poster"),
        )

    def _parse_awards(self, awards_text: str) -> OMDbParsedAwards:
        """
        Parse awards string like "Won 7 Oscars. 364 wins & 373 nominations total"

        Returns:
            OMDbParsedAwards with oscar_wins, oscar_noms, total_wins, total_nominations
        """
        result = {
            "oscar_wins": 0,
            "oscar_nominations": 0,
            "total_wins": 0,
            "total_nominations": 0,
        }

        if not awards_text or awards_text == "N/A":
            return OMDbParsedAwards(**result)

        # Look for Oscar wins: "Won 7 Oscars" or "Won 1 Oscar"
        oscar_win_match = re.search(r"Won (\d+) Oscar", awards_text)
        if oscar_win_match:
            result["oscar_wins"] = int(oscar_win_match.group(1))

        # Look for Oscar nominations: "Nominated for 7 Oscars"
        oscar_nom_match = re.search(r"Nominated for (\d+) Oscar", awards_text)
        if oscar_nom_match:
            result["oscar_nominations"] = int(oscar_nom_match.group(1))

        # Look for total wins & nominations: "364 wins & 373 nominations total"
        totals_match = re.search(r"(\d+) wins? & (\d+) nominations? total", awards_text)
        if totals_match:
            result["total_wins"] = int(totals_match.group(1))
            result["total_nominations"] = int(totals_match.group(2))

        return OMDbParsedAwards(**result)

    def fetch_batch(
        self, imdb_ids: list[str], progress: bool = True
    ) -> dict[str, OMDbResult | None]:
        """
        Fetch metadata for multiple films.

        Args:
            imdb_ids: List of IMDb IDs
            progress: Whether to print progress

        Returns:
            Dict mapping IMDb ID to metadata (or None if not found)
        """
        results = {}
        total = len(imdb_ids)

        for i, imdb_id in enumerate(imdb_ids):
            if progress:
                print(f"Fetching {i + 1}/{total}: {imdb_id}")

            results[imdb_id] = self.fetch(imdb_id)

        return results


def test_fetcher():
    """Test the fetcher with sample films."""
    fetcher = OMDbFetcher()

    test_ids = [
        "tt0137523",  # Fight Club
        "tt31193180",  # Sinners (2025)
        "tt6710474",  # Everything Everywhere All at Once
    ]

    for imdb_id in test_ids:
        print(f"\n{'=' * 60}")
        print(f"Testing: {imdb_id}")
        print("=" * 60)

        metadata = fetcher.fetch(imdb_id)
        if metadata:
            print(json.dumps(metadata, indent=2, default=str))
        else:
            print("NOT FOUND")


if __name__ == "__main__":
    test_fetcher()
