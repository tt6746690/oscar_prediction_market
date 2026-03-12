"""
TMDb API fetcher for enhanced film and person metadata.

Film metadata (existing):
- Production budget, worldwide revenue, production companies, release info

Person metadata (new, for acting/directing categories):
- Birth date (for computing age at ceremony)
- TMDb popularity score
- Total movie credits (filmography size)

Uses disk caching to avoid repeated API calls.

Usage:
    from fetch_tmdb import TMDbFetcher

    fetcher = TMDbFetcher(api_key="your_key")
    metadata = fetcher.fetch_by_imdb_id("tt0137523")  # Fight Club

    person = fetcher.fetch_person_data("Timothée Chalamet", known_film_title="Dune")
"""

import logging
import time
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any

import requests
from diskcache import Cache
from pydantic import BaseModel, Field

from oscar_prediction_market.constants import TMDB_CACHE_DIR, TMDB_PERSON_CACHE_DIR

logger = logging.getLogger(__name__)


class TMDbResult(BaseModel):
    """Parsed metadata from TMDb API response."""

    model_config = {"extra": "forbid"}

    tmdb_id: int | None = None
    imdb_id: str | None = None
    title: str | None = None
    budget: int | None = Field(default=None, ge=0)
    revenue: int | None = Field(default=None, ge=0)
    roi: float | None = None
    production_companies: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    release_date: str | None = None
    runtime: int | None = None
    vote_average: float | None = None
    vote_count: int | None = None
    popularity: float | None = None
    tagline: str | None = None
    overview: str | None = None


class TMDbPersonResult(BaseModel):
    """Parsed person metadata from TMDb API.

    Used to enrich PersonData with birth_date, popularity, and filmography size.
    TMDb person search → person details + movie credits.
    """

    model_config = {"extra": "forbid"}

    tmdb_person_id: int = Field(..., description="TMDb person ID")
    name: str = Field(..., description="Person name as listed on TMDb")
    birth_date: date | None = Field(default=None, description="Birth date")
    tmdb_popularity: float | None = Field(default=None, description="TMDb popularity score")
    total_film_credits: int = Field(
        ..., ge=0, description="Total movie credits (cast + crew, deduplicated)"
    )


class TMDbFetcher:
    """Fetch enhanced film metadata from TMDb API with caching."""

    BASE_URL = "https://api.themoviedb.org/3"

    def __init__(
        self,
        api_key: str,
        cache_dir: Path | str | None = TMDB_CACHE_DIR,
        person_cache_dir: Path | str | None = TMDB_PERSON_CACHE_DIR,
        rate_limit_delay: float = 0.3,  # 40 requests/10s = ~0.25s, use 0.3s to be safe
        refresh_cache: bool = False,
    ):
        """
        Initialize the TMDb fetcher.

        Args:
            api_key: TMDb API key (required, get from themoviedb.org).
            cache_dir: Directory for film disk cache. None to disable caching.
            person_cache_dir: Directory for person disk cache. None to disable caching.
            rate_limit_delay: Seconds to wait between API calls.
            refresh_cache: If True, ignore cached data and fetch fresh.
        """
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.refresh_cache = refresh_cache
        self._last_request_time = 0.0

        if cache_dir:
            self.cache = Cache(str(cache_dir))
        else:
            self.cache = None

        if person_cache_dir:
            self.person_cache = Cache(str(person_cache_dir))
        else:
            self.person_cache = None

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make a GET request to TMDb API."""
        self._rate_limit()

        if params is None:
            params = {}
        params["api_key"] = self.api_key

        url = f"{self.BASE_URL}{endpoint}"

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}

    def find_by_imdb_id(self, imdb_id: str) -> dict[str, Any] | None:
        """
        Find TMDb ID using IMDb ID.

        Args:
            imdb_id: IMDb ID (e.g., "tt0137523")

        Returns:
            TMDb movie data or None if not found.
        """
        # Check cache first
        cache_key = f"tmdb_find:{imdb_id}"
        if self.cache is not None and not self.refresh_cache and cache_key in self.cache:
            return self.cache[cache_key]  # Return immediately, no rate limiting

        # Make API request (rate limit will be applied in _get)
        endpoint = f"/find/{imdb_id}"
        params = {"external_source": "imdb_id"}

        data = self._get(endpoint, params)

        if data.get("success") is False:
            return None

        # Extract movie results
        movie_results = data.get("movie_results", [])
        if not movie_results:
            return None

        result = movie_results[0]  # Take first match

        # Cache only valid results with required fields (id and title)
        if self.cache is not None and result.get("id") and result.get("title"):
            self.cache[cache_key] = result

        return result

    def fetch_movie_details(self, tmdb_id: int) -> dict[str, Any] | None:
        """
        Fetch detailed movie information by TMDb ID.

        Args:
            tmdb_id: TMDb movie ID

        Returns:
            Full movie details including budget, revenue, production companies.
        """
        # Check cache first
        cache_key = f"tmdb_movie:{tmdb_id}"
        if self.cache is not None and not self.refresh_cache and cache_key in self.cache:
            return self.cache[cache_key]  # Return immediately, no rate limiting

        # Make API request (rate limit will be applied in _get)
        endpoint = f"/movie/{tmdb_id}"

        data = self._get(endpoint)

        if data.get("success") is False:
            return None

        # Cache only valid results with required fields (id and title)
        if self.cache is not None and data.get("id") and data.get("title"):
            self.cache[cache_key] = data

        return data

    def fetch_by_imdb_id(self, imdb_id: str) -> TMDbResult | None:
        """
        Fetch full movie details using IMDb ID (convenience method).

        Args:
            imdb_id: IMDb ID (e.g., "tt0137523")

        Returns:
            Parsed TMDbResult or None if not found.
        """
        # First, find TMDb ID
        find_result = self.find_by_imdb_id(imdb_id)
        if not find_result:
            return None

        tmdb_id = find_result.get("id")
        if not tmdb_id:
            return None

        # Then, fetch full details
        details = self.fetch_movie_details(tmdb_id)
        if not details:
            return None

        return self._parse_response(details)

    def _parse_response(self, raw: dict[str, Any]) -> TMDbResult:
        """Parse raw TMDb response into structured metadata."""

        # Extract production companies
        production_companies = []
        for company in raw.get("production_companies", []):
            name = company.get("name")
            if name:
                production_companies.append(name)

        # Extract genres
        genres = []
        for genre in raw.get("genres", []):
            name = genre.get("name")
            if name:
                genres.append(name)

        # Calculate ROI if we have budget and revenue
        budget = raw.get("budget", 0)
        revenue = raw.get("revenue", 0)
        roi = None
        if budget and budget > 0 and revenue:
            roi = (revenue - budget) / budget

        return TMDbResult(
            tmdb_id=raw.get("id"),
            imdb_id=raw.get("imdb_id"),
            title=raw.get("title"),
            budget=budget if budget > 0 else None,
            revenue=revenue if revenue > 0 else None,
            roi=roi,
            production_companies=production_companies,
            genres=genres,
            release_date=raw.get("release_date"),
            runtime=raw.get("runtime"),
            vote_average=raw.get("vote_average"),
            vote_count=raw.get("vote_count"),
            popularity=raw.get("popularity"),
            tagline=raw.get("tagline"),
            overview=raw.get("overview"),
        )

    def fetch_batch(
        self,
        imdb_ids: list[str],
        progress: bool = True,
    ) -> dict[str, TMDbResult | None]:
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
                cache_hit = (
                    self.cache and not self.refresh_cache and f"tmdb_find:{imdb_id}" in self.cache
                )
                status = "cached" if cache_hit else "fetching"
                print(f"[{i + 1}/{total}] {imdb_id} ({status})")

            results[imdb_id] = self.fetch_by_imdb_id(imdb_id)

        return results

    # ========================================================================
    # Person API (for acting/directing categories)
    # ========================================================================

    @staticmethod
    def _normalize_for_comparison(name: str) -> str:
        """Normalize a person name for comparison: strip accents, lowercase, collapse whitespace.

        Example::

            >>> TMDbFetcher._normalize_for_comparison("Timothée Chalamet")
            'timothee chalamet'
            >>> TMDbFetcher._normalize_for_comparison("Renée Zellweger")
            'renee zellweger'
        """
        # NFD decomposition splits accented chars into base + combining mark
        nfkd = unicodedata.normalize("NFKD", name)
        # Strip combining marks (accents)
        stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
        # Lowercase + collapse whitespace
        return " ".join(stripped.lower().split())

    def search_person(self, name: str) -> list[dict[str, Any]]:
        """Search TMDb for a person by name.

        Returns list of search results (up to 20), each containing:
        - id, name, popularity, known_for_department
        - known_for: list of movies/shows they're known for

        Args:
            name: Person name to search for.
        """
        cache_key = f"tmdb_person_search:{name}"
        if (
            self.person_cache is not None
            and not self.refresh_cache
            and cache_key in self.person_cache
        ):
            return self.person_cache[cache_key]

        data = self._get("/search/person", params={"query": name})
        results = data.get("results", [])

        if self.person_cache is not None and results:
            self.person_cache[cache_key] = results

        return results

    def _disambiguate_person(
        self,
        name: str,
        search_results: list[dict[str, Any]],
        known_film_title: str | None = None,
    ) -> dict[str, Any] | None:
        """Pick the best match from TMDb search results.

        Disambiguation strategy (in order):
        1. If known_film_title provided, check each result's known_for list
        2. Otherwise, pick highest-popularity exact name match
        3. Fall back to first result if name matches

        Args:
            name: Expected person name.
            search_results: TMDb search results.
            known_film_title: A film title this person is known for (e.g., their
                Oscar-nominated film). Used to disambiguate common names.

        Returns:
            Best matching search result dict, or None.
        """
        if not search_results:
            return None

        norm_name = self._normalize_for_comparison(name)

        # Strategy 1: Match by known_film_title
        if known_film_title:
            norm_film = self._normalize_for_comparison(known_film_title)
            for result in search_results:
                for kf in result.get("known_for", []):
                    kf_title = kf.get("title") or kf.get("name") or ""
                    if self._normalize_for_comparison(kf_title) == norm_film:
                        return result

        # Strategy 2: Exact name match (normalized), highest popularity
        name_matches = [
            r
            for r in search_results
            if self._normalize_for_comparison(r.get("name", "")) == norm_name
        ]
        if name_matches:
            return max(name_matches, key=lambda r: r.get("popularity", 0))

        # Strategy 3: First result if only one
        if len(search_results) == 1:
            return search_results[0]

        # No confident match
        logger.debug(
            "No confident TMDb person match for '%s' (%d results)", name, len(search_results)
        )
        return None

    def fetch_person_details(self, person_id: int) -> dict[str, Any] | None:
        """Fetch TMDb person details (birthday, popularity, biography, etc.).

        Args:
            person_id: TMDb person ID.
        """
        cache_key = f"tmdb_person_details:{person_id}"
        if (
            self.person_cache is not None
            and not self.refresh_cache
            and cache_key in self.person_cache
        ):
            return self.person_cache[cache_key]

        data = self._get(f"/person/{person_id}")
        if data.get("success") is False:
            return None

        if self.person_cache is not None and data.get("id"):
            self.person_cache[cache_key] = data

        return data

    def fetch_person_movie_credits(self, person_id: int) -> dict[str, Any] | None:
        """Fetch a person's movie credits (cast + crew).

        Args:
            person_id: TMDb person ID.
        """
        cache_key = f"tmdb_person_credits:{person_id}"
        if (
            self.person_cache is not None
            and not self.refresh_cache
            and cache_key in self.person_cache
        ):
            return self.person_cache[cache_key]

        data = self._get(f"/person/{person_id}/movie_credits")
        if data.get("success") is False:
            return None

        if self.person_cache is not None and data.get("id"):
            self.person_cache[cache_key] = data

        return data

    def fetch_person_data(
        self,
        name: str,
        known_film_title: str | None = None,
    ) -> TMDbPersonResult | None:
        """Fetch person metadata: birth_date, popularity, total_film_credits.

        Searches TMDb for the person, disambiguates using known_film_title if
        provided, then fetches details and movie credits.

        Args:
            name: Person's name (e.g., "Timothée Chalamet").
            known_film_title: A film they're in, for disambiguation.

        Returns:
            TMDbPersonResult with enriched data, or None if person not found.

        Example::

            >>> fetcher = TMDbFetcher(api_key="...")
            >>> result = fetcher.fetch_person_data("Cillian Murphy", "Oppenheimer")
            >>> result.birth_date
            datetime.date(1976, 5, 25)
        """
        # Check assembled-result cache first (avoids 3 API calls for repeat lookups)
        assembled_key = f"tmdb_person_assembled:{name}:{known_film_title}"
        if (
            self.person_cache is not None
            and not self.refresh_cache
            and assembled_key in self.person_cache
        ):
            cached = self.person_cache[assembled_key]
            if cached is None:
                return None
            return TMDbPersonResult.model_validate(cached)

        result = self._fetch_person_data_uncached(name, known_film_title)

        # Cache the assembled result (including None for "not found")
        if self.person_cache is not None:
            self.person_cache[assembled_key] = result.model_dump() if result else None

        return result

    def _fetch_person_data_uncached(
        self,
        name: str,
        known_film_title: str | None = None,
    ) -> TMDbPersonResult | None:
        """Internal: fetch person data without checking assembled cache."""
        # Step 1: Search
        search_results = self.search_person(name)
        if not search_results:
            logger.debug("TMDb person search returned no results for '%s'", name)
            return None

        # Step 2: Disambiguate
        best = self._disambiguate_person(name, search_results, known_film_title)
        if not best:
            return None

        person_id = best["id"]

        # Step 3: Fetch details (birthday, popularity)
        details = self.fetch_person_details(person_id)
        birth_date: date | None = None
        tmdb_popularity: float | None = None

        if details:
            bd_str = details.get("birthday")
            if bd_str:
                try:
                    birth_date = date.fromisoformat(bd_str)
                except ValueError:
                    pass
            tmdb_popularity = details.get("popularity")

        # Step 4: Fetch movie credits (count unique movies)
        credits_data = self.fetch_person_movie_credits(person_id)
        total_film_credits = 0
        if credits_data:
            # Deduplicate by movie ID across cast and crew
            movie_ids: set[int] = set()
            for entry in credits_data.get("cast", []):
                mid = entry.get("id")
                if mid is not None:
                    movie_ids.add(mid)
            for entry in credits_data.get("crew", []):
                mid = entry.get("id")
                if mid is not None:
                    movie_ids.add(mid)
            total_film_credits = len(movie_ids)

        return TMDbPersonResult(
            tmdb_person_id=person_id,
            name=best.get("name", name),
            birth_date=birth_date,
            tmdb_popularity=tmdb_popularity,
            total_film_credits=total_film_credits,
        )


def test_fetcher():
    """Test the fetcher with sample films."""
    import json

    from oscar_prediction_market.constants import TMDB_API_KEY

    API_KEY = TMDB_API_KEY

    fetcher = TMDbFetcher(api_key=API_KEY)

    test_ids = [
        "tt15398776",  # Oppenheimer
        "tt0137523",  # Fight Club
        "tt31193180",  # Sinners (2025)
    ]

    for imdb_id in test_ids:
        print(f"\n{'=' * 60}")
        print(f"Testing: {imdb_id}")
        print("=" * 60)

        metadata = fetcher.fetch_by_imdb_id(imdb_id)
        if metadata:
            print(json.dumps(metadata, indent=2, default=str))
        else:
            print("NOT FOUND")


if __name__ == "__main__":
    test_fetcher()
