"""
Precursor awards fetcher — scrapes Wikipedia for guild awards data.

Fetches historical winners/nominees for all awards in PrecursorKey, including PGA,
DGA, SAG, BAFTA, Golden Globe, Critics Choice, WGA, ASC, and Annie awards.

Uses disk caching and fuzzy matching to match films to Oscar nominees.

Known Wikipedia table quirks (documented here to explain non-obvious parsing logic):

- **BAFTA Supporting Actor/Actress**: The Wikipedia tables have "Role(s)" and "Film"
  columns SWAPPED — "Role(s)" contains the film title, "Film" contains the character
  name. We use film_col_hint="Role" for these awards to get the correct column.
  Example: Row has Actor='Ian Holm', Role(s)='The Bofors Gun', Film='Flynn' where
  'The Bofors Gun' is the film and 'Flynn' is the character.

- **BAFTA Lead Actress**: Uses gender-neutral "Actor" column header (not "Actress").
  We use person_col_hint="Actor" instead of "Actress".

- **Annie Feature**: Studio/production company names sometimes appear in the film
  column (e.g., "Walt Disney Pictures, Pixar Animation Studios"). Filtered by
  ``_is_studio_name()``.

- **BAFTA Animated (2025)**: Concatenated nominee lists appear as single entries
  (e.g., "Cars, Ratatouille, WALL-E, ..."). Filtered by ``_is_concatenated_film_list()``.

- **Empty person names**: Some person-level awards (BAFTA acting categories) have
  empty person fields due to Wikipedia table formatting. These are set to None
  (preserving the row for film-level matching) rather than dropped.

Usage:
    from fetch_precursor_awards import PrecursorAwardsFetcher

    fetcher = PrecursorAwardsFetcher()
    awards = fetcher.fetch_all_awards(year_range=(2000, 2025))
"""

import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from diskcache import Cache
from thefuzz import fuzz

from oscar_prediction_market.constants import PRECURSOR_AWARDS_CACHE_DIR
from oscar_prediction_market.data.schema import PrecursorKey

# Wikipedia URLs for each precursor award.
# Keys are PrecursorKey enum members (StrEnum, so also usable as strings).
AWARD_URLS: dict[PrecursorKey, str] = {
    # --- Best Picture precursors (film-level) ---
    PrecursorKey.PGA_BP: "https://en.wikipedia.org/wiki/Producers_Guild_of_America_Award_for_Best_Theatrical_Motion_Picture",
    PrecursorKey.DGA_DIRECTING: "https://en.wikipedia.org/wiki/Directors_Guild_of_America_Award_for_Outstanding_Directing_%E2%80%93_Feature_Film",
    PrecursorKey.SAG_ENSEMBLE: "https://en.wikipedia.org/wiki/Screen_Actors_Guild_Award_for_Outstanding_Performance_by_a_Cast_in_a_Motion_Picture",
    PrecursorKey.BAFTA_FILM: "https://en.wikipedia.org/wiki/BAFTA_Award_for_Best_Film",
    PrecursorKey.GOLDEN_GLOBE_DRAMA: "https://en.wikipedia.org/wiki/Golden_Globe_Award_for_Best_Motion_Picture_%E2%80%93_Drama",
    PrecursorKey.GOLDEN_GLOBE_MUSICAL: "https://en.wikipedia.org/wiki/Golden_Globe_Award_for_Best_Motion_Picture_%E2%80%93_Musical_or_Comedy",
    PrecursorKey.CRITICS_CHOICE_PICTURE: "https://en.wikipedia.org/wiki/Critics%27_Choice_Movie_Award_for_Best_Picture",
    # --- Director precursors ---
    PrecursorKey.BAFTA_DIRECTOR: "https://en.wikipedia.org/wiki/BAFTA_Award_for_Best_Direction",
    PrecursorKey.GOLDEN_GLOBE_DIRECTOR: "https://en.wikipedia.org/wiki/Golden_Globe_Award_for_Best_Director",
    PrecursorKey.CRITICS_CHOICE_DIRECTOR: "https://en.wikipedia.org/wiki/Critics%27_Choice_Movie_Award_for_Best_Director",
    # --- Lead Actor precursors ---
    PrecursorKey.SAG_LEAD_ACTOR: "https://en.wikipedia.org/wiki/Screen_Actors_Guild_Award_for_Outstanding_Performance_by_a_Male_Actor_in_a_Leading_Role",
    PrecursorKey.BAFTA_LEAD_ACTOR: "https://en.wikipedia.org/wiki/BAFTA_Award_for_Best_Actor_in_a_Leading_Role",
    PrecursorKey.GOLDEN_GLOBE_ACTOR_DRAMA: "https://en.wikipedia.org/wiki/Golden_Globe_Award_for_Best_Actor_%E2%80%93_Motion_Picture_Drama",
    PrecursorKey.GOLDEN_GLOBE_ACTOR_MUSICAL: "https://en.wikipedia.org/wiki/Golden_Globe_Award_for_Best_Actor_%E2%80%93_Motion_Picture_Musical_or_Comedy",
    PrecursorKey.CRITICS_CHOICE_ACTOR: "https://en.wikipedia.org/wiki/Critics%27_Choice_Movie_Award_for_Best_Actor",
    # --- Lead Actress precursors ---
    PrecursorKey.SAG_LEAD_ACTRESS: "https://en.wikipedia.org/wiki/Screen_Actors_Guild_Award_for_Outstanding_Performance_by_a_Female_Actor_in_a_Leading_Role",
    PrecursorKey.BAFTA_LEAD_ACTRESS: "https://en.wikipedia.org/wiki/BAFTA_Award_for_Best_Actress_in_a_Leading_Role",
    PrecursorKey.GOLDEN_GLOBE_ACTRESS_DRAMA: "https://en.wikipedia.org/wiki/Golden_Globe_Award_for_Best_Actress_%E2%80%93_Motion_Picture_Drama",
    PrecursorKey.GOLDEN_GLOBE_ACTRESS_MUSICAL: "https://en.wikipedia.org/wiki/Golden_Globe_Award_for_Best_Actress_%E2%80%93_Motion_Picture_Musical_or_Comedy",
    PrecursorKey.CRITICS_CHOICE_ACTRESS: "https://en.wikipedia.org/wiki/Critics%27_Choice_Movie_Award_for_Best_Actress",
    # --- Supporting Actor precursors ---
    PrecursorKey.SAG_SUPPORTING_ACTOR: "https://en.wikipedia.org/wiki/Screen_Actors_Guild_Award_for_Outstanding_Performance_by_a_Male_Actor_in_a_Supporting_Role",
    PrecursorKey.BAFTA_SUPPORTING_ACTOR: "https://en.wikipedia.org/wiki/BAFTA_Award_for_Best_Actor_in_a_Supporting_Role",
    PrecursorKey.GOLDEN_GLOBE_SUPPORTING_ACTOR: "https://en.wikipedia.org/wiki/Golden_Globe_Award_for_Best_Supporting_Actor_%E2%80%93_Motion_Picture",
    PrecursorKey.CRITICS_CHOICE_SUPPORTING_ACTOR: "https://en.wikipedia.org/wiki/Critics%27_Choice_Movie_Award_for_Best_Supporting_Actor",
    # --- Supporting Actress precursors ---
    PrecursorKey.SAG_SUPPORTING_ACTRESS: "https://en.wikipedia.org/wiki/Screen_Actors_Guild_Award_for_Outstanding_Performance_by_a_Female_Actor_in_a_Supporting_Role",
    PrecursorKey.BAFTA_SUPPORTING_ACTRESS: "https://en.wikipedia.org/wiki/BAFTA_Award_for_Best_Actress_in_a_Supporting_Role",
    PrecursorKey.GOLDEN_GLOBE_SUPPORTING_ACTRESS: "https://en.wikipedia.org/wiki/Golden_Globe_Award_for_Best_Supporting_Actress_%E2%80%93_Motion_Picture",
    PrecursorKey.CRITICS_CHOICE_SUPPORTING_ACTRESS: "https://en.wikipedia.org/wiki/Critics%27_Choice_Movie_Award_for_Best_Supporting_Actress",
    # --- Screenplay precursors ---
    PrecursorKey.WGA_ORIGINAL: "https://en.wikipedia.org/wiki/Writers_Guild_of_America_Award_for_Best_Original_Screenplay",
    PrecursorKey.BAFTA_ORIGINAL_SCREENPLAY: "https://en.wikipedia.org/wiki/BAFTA_Award_for_Best_Original_Screenplay",
    PrecursorKey.CRITICS_CHOICE_ORIGINAL_SCREENPLAY: "https://en.wikipedia.org/wiki/Critics%27_Choice_Movie_Award_for_Best_Original_Screenplay",
    # --- Cinematography precursors ---
    PrecursorKey.ASC_CINEMATOGRAPHY: "https://en.wikipedia.org/wiki/American_Society_of_Cinematographers_Award_for_Outstanding_Achievement_in_Cinematography_in_Theatrical_Releases",
    PrecursorKey.BAFTA_CINEMATOGRAPHY: "https://en.wikipedia.org/wiki/BAFTA_Award_for_Best_Cinematography",
    # --- Animated Feature precursors ---
    PrecursorKey.ANNIE_FEATURE: "https://en.wikipedia.org/wiki/Annie_Award_for_Best_Animated_Feature",
    PrecursorKey.BAFTA_ANIMATED: "https://en.wikipedia.org/wiki/BAFTA_Award_for_Best_Animated_Film",
    PrecursorKey.PGA_ANIMATED: "https://en.wikipedia.org/wiki/Producers_Guild_of_America_Award_for_Best_Animated_Motion_Picture",
    PrecursorKey.GOLDEN_GLOBE_ANIMATED: "https://en.wikipedia.org/wiki/Golden_Globe_Award_for_Best_Animated_Feature_Film",
    PrecursorKey.CRITICS_CHOICE_ANIMATED: "https://en.wikipedia.org/wiki/Critics%27_Choice_Movie_Award_for_Best_Animated_Feature",
    # --- New cross-category precursors ---
    PrecursorKey.CRITICS_CHOICE_CINEMATOGRAPHY: "https://en.wikipedia.org/wiki/Critics%27_Choice_Movie_Award_for_Best_Cinematography",
    PrecursorKey.GOLDEN_GLOBE_SCREENPLAY: "https://en.wikipedia.org/wiki/Golden_Globe_Award_for_Best_Screenplay",
}

# Per-award table parsing configuration.
# Maps each precursor award to (film_col_hint, person_col_hint).
# Hints are partial column name matches used to identify the correct column
# in Wikipedia tables that may have been concatenated from multiple era-tables.
# person_col_hint=None for film-level-only awards.
AWARD_TABLE_CONFIG: dict[PrecursorKey, tuple[str, str | None]] = {
    # --- Best Picture precursors (film-level) ---
    PrecursorKey.PGA_BP: ("Film", None),
    PrecursorKey.DGA_DIRECTING: ("Film", "Director"),
    PrecursorKey.SAG_ENSEMBLE: ("Film", None),
    PrecursorKey.BAFTA_FILM: ("Film", None),
    PrecursorKey.GOLDEN_GLOBE_DRAMA: ("Film", None),
    PrecursorKey.GOLDEN_GLOBE_MUSICAL: ("Film", None),
    PrecursorKey.CRITICS_CHOICE_PICTURE: ("Film", None),
    # --- Director precursors ---
    PrecursorKey.BAFTA_DIRECTOR: ("Film", "Director"),
    PrecursorKey.GOLDEN_GLOBE_DIRECTOR: ("Film", "Name"),  # GG Director uses "Name" header
    PrecursorKey.CRITICS_CHOICE_DIRECTOR: ("Film", "Director"),
    # --- Lead Actor precursors ---
    PrecursorKey.SAG_LEAD_ACTOR: ("Film", "Actor"),
    PrecursorKey.BAFTA_LEAD_ACTOR: ("Film", "Actor"),
    PrecursorKey.GOLDEN_GLOBE_ACTOR_DRAMA: ("Film", "Actor"),
    PrecursorKey.GOLDEN_GLOBE_ACTOR_MUSICAL: ("Film", "Actor"),
    PrecursorKey.CRITICS_CHOICE_ACTOR: ("Film", "Actor"),
    # --- Lead Actress precursors ---
    PrecursorKey.SAG_LEAD_ACTRESS: ("Film", "Actress"),
    PrecursorKey.BAFTA_LEAD_ACTRESS: ("Film", "Actor"),  # BAFTA uses gender-neutral "Actor" header
    PrecursorKey.GOLDEN_GLOBE_ACTRESS_DRAMA: ("Film", "Actress"),
    PrecursorKey.GOLDEN_GLOBE_ACTRESS_MUSICAL: ("Film", "Actress"),
    PrecursorKey.CRITICS_CHOICE_ACTRESS: ("Work", "Actress"),  # CC Actress uses "Work" not "Film"
    # --- Supporting Actor precursors ---
    PrecursorKey.SAG_SUPPORTING_ACTOR: ("Film", "Actor"),
    # BAFTA supporting: Wikipedia swaps "Role(s)" (= film title) and "Film" (= character).
    # Use "Role" hint to get actual film titles from the "Role(s)" column.
    PrecursorKey.BAFTA_SUPPORTING_ACTOR: ("Role", "Actor"),
    PrecursorKey.GOLDEN_GLOBE_SUPPORTING_ACTOR: ("Film", "Actor"),
    PrecursorKey.CRITICS_CHOICE_SUPPORTING_ACTOR: ("Film", "Actor"),
    # --- Supporting Actress precursors ---
    PrecursorKey.SAG_SUPPORTING_ACTRESS: ("Film", "Actress"),
    # BAFTA supporting actress: same Role(s)/Film swap + gender-neutral "Actor" header.
    PrecursorKey.BAFTA_SUPPORTING_ACTRESS: ("Role", "Actor"),
    PrecursorKey.GOLDEN_GLOBE_SUPPORTING_ACTRESS: ("Film", "Actress"),
    PrecursorKey.CRITICS_CHOICE_SUPPORTING_ACTRESS: (
        "Film",
        "Actor",
    ),  # CC uses "Actor" header for both genders
    # --- Screenplay precursors (Year, Film, Writer) ---
    PrecursorKey.WGA_ORIGINAL: ("Film", "Writer"),
    PrecursorKey.BAFTA_ORIGINAL_SCREENPLAY: ("Film", "Writer"),
    PrecursorKey.CRITICS_CHOICE_ORIGINAL_SCREENPLAY: ("Film", None),
    # --- Cinematography precursors (Year, Film, Cinematographer) ---
    PrecursorKey.ASC_CINEMATOGRAPHY: ("Film", "Cinematographer"),
    PrecursorKey.BAFTA_CINEMATOGRAPHY: ("Film", "Cinematographer"),
    # --- Animated Feature precursors (film-level) ---
    PrecursorKey.ANNIE_FEATURE: ("Film", None),
    PrecursorKey.BAFTA_ANIMATED: ("Film", None),
    PrecursorKey.PGA_ANIMATED: ("Film", None),
    PrecursorKey.GOLDEN_GLOBE_ANIMATED: ("Film", None),
    PrecursorKey.CRITICS_CHOICE_ANIMATED: ("Winner", None),
    # --- New cross-category precursors ---
    PrecursorKey.CRITICS_CHOICE_CINEMATOGRAPHY: ("Film", "Nominee"),
    PrecursorKey.GOLDEN_GLOBE_SCREENPLAY: ("Film", "Writer"),
}

# First ceremony year each precursor award was given.
# Used to distinguish "award didn't exist" (→ None) from "not nominated" (→ False).
PRECURSOR_YEAR_INTRODUCED: dict[PrecursorKey, int] = {
    # Best Picture
    PrecursorKey.PGA_BP: 1990,
    PrecursorKey.DGA_DIRECTING: 1949,
    PrecursorKey.SAG_ENSEMBLE: 1995,
    PrecursorKey.BAFTA_FILM: 1948,
    PrecursorKey.GOLDEN_GLOBE_DRAMA: 1944,
    PrecursorKey.GOLDEN_GLOBE_MUSICAL: 1952,
    PrecursorKey.CRITICS_CHOICE_PICTURE: 1996,
    # Director
    PrecursorKey.BAFTA_DIRECTOR: 1969,
    PrecursorKey.GOLDEN_GLOBE_DIRECTOR: 1944,
    PrecursorKey.CRITICS_CHOICE_DIRECTOR: 1996,
    # Lead Actor
    PrecursorKey.SAG_LEAD_ACTOR: 1995,
    PrecursorKey.BAFTA_LEAD_ACTOR: 1953,
    PrecursorKey.GOLDEN_GLOBE_ACTOR_DRAMA: 1944,
    PrecursorKey.GOLDEN_GLOBE_ACTOR_MUSICAL: 1952,
    PrecursorKey.CRITICS_CHOICE_ACTOR: 1996,
    # Lead Actress
    PrecursorKey.SAG_LEAD_ACTRESS: 1995,
    PrecursorKey.BAFTA_LEAD_ACTRESS: 1953,
    PrecursorKey.GOLDEN_GLOBE_ACTRESS_DRAMA: 1944,
    PrecursorKey.GOLDEN_GLOBE_ACTRESS_MUSICAL: 1952,
    PrecursorKey.CRITICS_CHOICE_ACTRESS: 1996,
    # Supporting Actor
    PrecursorKey.SAG_SUPPORTING_ACTOR: 1995,
    PrecursorKey.BAFTA_SUPPORTING_ACTOR: 1969,
    PrecursorKey.GOLDEN_GLOBE_SUPPORTING_ACTOR: 1944,
    PrecursorKey.CRITICS_CHOICE_SUPPORTING_ACTOR: 2001,
    # Supporting Actress
    PrecursorKey.SAG_SUPPORTING_ACTRESS: 1995,
    PrecursorKey.BAFTA_SUPPORTING_ACTRESS: 1969,
    PrecursorKey.GOLDEN_GLOBE_SUPPORTING_ACTRESS: 1944,
    PrecursorKey.CRITICS_CHOICE_SUPPORTING_ACTRESS: 2001,
    # Screenplay
    PrecursorKey.WGA_ORIGINAL: 1969,
    PrecursorKey.BAFTA_ORIGINAL_SCREENPLAY: 1984,
    PrecursorKey.CRITICS_CHOICE_ORIGINAL_SCREENPLAY: 2009,
    # Cinematography
    PrecursorKey.ASC_CINEMATOGRAPHY: 1987,
    PrecursorKey.BAFTA_CINEMATOGRAPHY: 1964,
    # Animated Feature
    PrecursorKey.ANNIE_FEATURE: 1992,
    PrecursorKey.BAFTA_ANIMATED: 2007,
    PrecursorKey.PGA_ANIMATED: 2007,
    PrecursorKey.GOLDEN_GLOBE_ANIMATED: 2006,
    PrecursorKey.CRITICS_CHOICE_ANIMATED: 1998,
    # New cross-category
    PrecursorKey.CRITICS_CHOICE_CINEMATOGRAPHY: 2009,
    PrecursorKey.GOLDEN_GLOBE_SCREENPLAY: 1947,
}


class PrecursorAwardsFetcher:
    """Fetch precursor awards data from Wikipedia with caching and fuzzy matching."""

    def __init__(
        self,
        cache_dir: Path | str | None = PRECURSOR_AWARDS_CACHE_DIR,
        refresh_cache: bool = False,
        fuzzy_threshold: int = 90,  # Match score >= 90 (out of 100) - stricter
    ):
        """
        Initialize the precursor awards fetcher.

        Args:
            cache_dir: Directory for disk cache. None to disable caching.
            refresh_cache: If True, ignore cached data and fetch fresh.
            fuzzy_threshold: Minimum fuzzy match score (0-100) to consider a match.
        """
        self.refresh_cache = refresh_cache
        self.fuzzy_threshold = fuzzy_threshold

        if cache_dir:
            self.cache = Cache(str(cache_dir))
        else:
            self.cache = None

        # User agent for Wikipedia requests
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }

    def _is_winner_row(self, row: Tag) -> bool:
        """Check if a table row represents a winner based on Wikipedia styling.

        Wikipedia uses different background colors to highlight winners:
        - #FAEB86 (gold/yellow): PGA, DGA, SAG, BAFTA
        - #B0C4DE (light blue): Golden Globe, Critics Choice

        Both colors indicate a winner row.
        """
        # Winner highlight colors used by Wikipedia award tables
        winner_colors = ["FAEB86", "B0C4DE"]

        # Check row style
        row_style = row.get("style", "") or ""
        if isinstance(row_style, list):
            row_style = " ".join(row_style)
        row_style_upper = row_style.upper()
        for color in winner_colors:
            if color in row_style_upper:
                return True

        # Check cell styles within the row
        for cell in row.find_all(["td", "th"]):
            cell_style = cell.get("style", "") or ""
            if isinstance(cell_style, list):
                cell_style = " ".join(cell_style)
            cell_style_upper = cell_style.upper()
            for color in winner_colors:
                if color in cell_style_upper:
                    return True

        return False

    @staticmethod
    def _is_data_table(table: Tag) -> bool:
        """Check if a Wikipedia table is award data (not statistics/records).

        Award data tables have a "Year" column. Statistics tables (wins by
        franchise, nominations by studio, etc.) have "Wins" or "Nominations"
        as column headers instead. Parsing stats tables pollutes the dataset
        with franchise/studio names misidentified as film nominees.
        """
        header_row = table.find("tr")
        if not header_row:
            return False
        header_texts = [
            c.get_text(strip=True).lower()
            for c in header_row.find_all(["th"])  # type: ignore[union-attr]
        ]
        return any("year" in h for h in header_texts)

    def _parse_table_with_bs4(self, table: Tag, award_name: str) -> pd.DataFrame | None:
        """Parse a Wikipedia table with BeautifulSoup, extracting data and winner
        flags in a single pass.

        This is the primary table parser for all awards. It avoids the row-count
        alignment issues that arise when using ``pd.read_html`` separately from
        BS4's HTML traversal (which was the root cause of the Annie Awards winner
        detection bug).

        Handles:
        - Rowspan/colspan (merged year cells across nominees)
        - Year sub-header rows: single ``<th>`` rows like "2020(48th)[36]" that
          act as group headers, with subsequent data rows having one fewer column
        - Winner detection via ``_is_winner_row()`` background color checks

        Returns:
            DataFrame including ``_is_winner_from_html`` column, or None if the
            table could not be parsed.
        """
        header_row = table.find("tr")
        headers: list[str] = []
        if header_row:
            headers = [c.get_text(strip=True) for c in header_row.find_all(["th"])]  # type: ignore[union-attr]
        num_cols = len(headers) if headers else 0

        data_rows: list[list[str]] = []
        winner_flags: list[bool] = []
        # Track cells spanning multiple rows: col_idx -> (text, rows_left)
        span_buf: dict[int, tuple[str, int]] = {}
        # Current year group header (for tables where year is a sub-header row,
        # e.g., Annie Awards: a single <th> row like "2020(48th)[36]" followed
        # by data rows with only Film + Production Company columns).
        current_year_header: str | None = None

        for tr in table.find_all("tr")[1:]:
            cells = tr.find_all(["td", "th"])
            if not cells:
                continue

            # Detect year sub-header rows: a single <th> cell that isn't a normal
            # data row. These are year group headers (e.g., "2020(48th)[36][37]").
            if len(cells) == 1 and cells[0].name == "th":
                current_year_header = cells[0].get_text(strip=True)
                continue

            row: list[str] = []
            col_idx = 0
            cell_idx = 0

            # If data rows have fewer columns than headers and we have a year
            # sub-header, prepend the year as the first column.
            active_spans = sum(1 for v in span_buf.values() if v[1] > 0)
            actual_data_cols = len(cells) + active_spans
            if current_year_header and actual_data_cols < num_cols:
                row.append(current_year_header)
                col_idx = 1

            # Walk columns left-to-right, filling from span_buf or the next
            # HTML cell.  Stop after consuming all cells and pending spans,
            # but never exceed num_cols when we know the table width.
            while cell_idx < len(cells) or any(
                span_buf.get(c, (None, 0))[1] > 0  # type: ignore[comparison-overlap]
                for c in range(col_idx, num_cols if num_cols else col_idx)
            ):
                # Respect table width — don't overshoot header count.
                if num_cols and col_idx >= num_cols:
                    break
                if col_idx in span_buf and span_buf[col_idx][1] > 0:
                    text, remaining = span_buf[col_idx]
                    row.append(text)
                    span_buf[col_idx] = (text, remaining - 1)
                elif cell_idx < len(cells):
                    cell = cells[cell_idx]
                    text = cell.get_text(strip=True)
                    try:
                        rowspan = int(cell.get("rowspan") or 1)  # type: ignore[arg-type]  # BS4 returns str|None
                    except (ValueError, TypeError):
                        rowspan = 1
                    try:
                        colspan = int(cell.get("colspan") or 1)  # type: ignore[arg-type]  # BS4 returns str|None
                    except (ValueError, TypeError):
                        colspan = 1
                    if rowspan > 1:
                        # Register span for this column (and any colspan'd
                        # extra columns) so subsequent rows inherit the value.
                        for span_c in range(colspan):
                            span_buf[col_idx + span_c] = (text, rowspan - 1)
                    row.append(text)
                    # colspan > 1: fill extra logical columns with the same text.
                    for _ in range(1, colspan):
                        col_idx += 1
                        if num_cols and col_idx >= num_cols:
                            break
                        row.append(text)
                    cell_idx += 1
                else:
                    break
                col_idx += 1

            # Normalise row width: pad short rows, truncate long ones.
            if num_cols:
                if len(row) < num_cols:
                    row.extend("" for _ in range(num_cols - len(row)))
                elif len(row) > num_cols:
                    row = row[:num_cols]

            if row:
                data_rows.append(row)
                winner_flags.append(self._is_winner_row(tr))  # type: ignore[arg-type]

        if not data_rows:
            return None

        # All rows are now exactly num_cols wide, so named columns always work.
        # Deduplicate header names (e.g., Golden Globe Musical 1958-62 table has
        # side-by-side Comedy/Musical sections with duplicate "Director"/"Producer").
        if headers and num_cols:
            seen: dict[str, int] = {}
            unique_headers: list[str] = []
            for h in headers:
                if h in seen:
                    seen[h] += 1
                    unique_headers.append(f"{h}_{seen[h]}")
                else:
                    seen[h] = 0
                    unique_headers.append(h)
            df = pd.DataFrame(data_rows, columns=unique_headers)
        else:
            df = pd.DataFrame(data_rows)

        df["_is_winner_from_html"] = winner_flags
        return df

    def _fetch_wikipedia_table_with_winners(self, url: str, award_name: str) -> pd.DataFrame | None:
        """
        Fetch and parse Wikipedia table for an award, detecting winners from HTML styling.

        Wikipedia uses background:#FAEB86 (gold/yellow) to highlight winners.
        This method preserves that information in an 'is_winner' column.

        Args:
            url: Wikipedia URL
            award_name: Award identifier (for caching)

        Returns:
            DataFrame with award data including 'is_winner' column, or None if failed.
        """
        # Check cache first
        cache_key = f"wiki_table_v4:{award_name}"  # v4: BS4-only parser (removed pd.read_html)
        if self.cache and not self.refresh_cache and cache_key in self.cache:
            cached_data = self.cache[cache_key]
            return pd.DataFrame(cached_data)

        try:
            time.sleep(0.5)  # Be nice to Wikipedia

            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            tables = soup.find_all("table", class_="wikitable")

            if not tables:
                print(f"⚠ Warning: No tables found for {award_name}")
                return None

            # Parse all tables and combine (some awards split by era)
            all_dfs = []
            for table in tables:
                # Skip statistics/records tables (wins by franchise, studio, etc.)
                # that would pollute the dataset with non-nominee entries.
                if not self._is_data_table(table):
                    continue

                try:
                    df = self._parse_table_with_bs4(table, award_name)
                    if df is not None:
                        all_dfs.append(df)
                except Exception as e:
                    print(f"⚠ Warning: Could not parse table for {award_name}: {e}")

            if not all_dfs:
                return None

            # Combine all dataframes
            combined_df = pd.concat(all_dfs, ignore_index=True)

            # Cache the result
            if self.cache:
                self.cache[cache_key] = combined_df.to_dict("records")

            return combined_df

        except Exception as e:
            print(f"✗ Error fetching {award_name}: {e}")
            return None

    def _parse_award_year(self, year_str: str) -> int | None:
        """
        Parse year from various formats like '2023', '2022–23', '1989 (1st)'.

        Returns the ceremony year (e.g., 2023 for films from 2022).
        """
        if pd.isna(year_str):
            return None

        year_str = str(year_str).strip()

        # Extract first 4-digit year
        import re

        match = re.search(r"\b(19\d{2}|20\d{2})\b", year_str)
        if match:
            return int(match.group(1))

        return None

    def _clean_film_title(self, title: str) -> str:
        """Clean film title for better matching.

        Removes:
        - Year annotations like (2023)
        - Reference annotations like [1]
        - Wikipedia Oscar markers: † (winner), ‡ (multiple awards)
        - Normalizes ampersand to "and"
        """
        if pd.isna(title):
            return ""

        title = str(title).strip()

        # Remove Wikipedia Oscar markers († = won Oscar, ‡ = multiple wins)
        title = title.replace("†", "").replace("‡", "").strip()

        # Remove common annotations
        title = title.split("(")[0].strip()  # Remove (2023) etc.
        title = title.split("[")[0].strip()  # Remove [1] etc.

        # Normalize ampersand to "and" for consistent matching
        title = title.replace(" & ", " and ")

        return title

    def _clean_person_name(self, name: str) -> str:
        """Clean person name from Wikipedia formatting.

        Removes reference markers [1], Oscar markers (†, ‡), and extra whitespace.
        """
        if pd.isna(name):
            return ""

        name = str(name).strip()
        name = name.replace("†", "").replace("‡", "").strip()
        name = name.split("[")[0].strip()  # Remove [1] etc.
        name = name.split("(")[0].strip()  # Remove (posthumous) etc.

        return name

    @staticmethod
    def _is_studio_name(title: str) -> bool:
        """Detect studio/production company names mistakenly parsed as film titles.

        Known issue: Annie Award Wikipedia tables sometimes put studio names in the
        film column (e.g., "Pixar Animation Studios, Walt Disney Pictures").
        """
        studio_indicators = [
            "Studios",
            "Pictures",
            "Productions",
            "Entertainment",
            "Animation",
            "Motion Pictures",
        ]
        return any(indicator.lower() in title.lower() for indicator in studio_indicators)

    @staticmethod
    def _is_concatenated_film_list(title: str) -> bool:
        """Detect concatenated film lists mistakenly parsed as a single title.

        Known issue: BAFTA Animated 2025 Wikipedia table has concatenated nominees
        (e.g., "Flow The Wild Robot Wallace and Gromit: Vengeance Most Fowl").
        Heuristic: film titles are generally < 80 chars. Concatenated lists are longer.
        """
        return len(title) > 80

    def _find_column_by_hint(self, df: pd.DataFrame, hint: str) -> pd.Series:
        """Find a DataFrame column by partial name match.

        Handles Wikipedia tables where column names vary across eras
        (e.g., "Cinematographer" vs "Cinematographer(s)", "Film" vs "Films").
        When multiple columns match, combines them with fillna to merge data
        from different era-tables.

        Args:
            df: DataFrame from concatenated Wikipedia tables.
            hint: Partial column name to search for (case-insensitive).

        Returns:
            Series with combined values from matching columns.

        Raises:
            ValueError: If no column matches the hint.
        """
        hint_lower = hint.lower()

        # Find ALL matching columns (partial, case-insensitive)
        matches = [c for c in df.columns if isinstance(c, str) and hint_lower in c.lower()]
        if not matches:
            raise ValueError(f"No column matching '{hint}' in columns: {list(df.columns)}")

        # Combine matching columns (fillna across eras)
        result = df[matches[0]]
        for m in matches[1:]:
            result = result.fillna(df[m])
        return result

    def _fetch_generic_award(
        self,
        award_key: PrecursorKey,
        year_range: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        """Fetch any precursor award using generalized table parsing.

        Uses AWARD_TABLE_CONFIG column name hints to identify the correct columns
        in concatenated Wikipedia tables. This handles pages with multiple tables
        (e.g., nominees + statistics) that have different column structures.

        Args:
            award_key: PrecursorKey for the award to fetch.
            year_range: Optional (min_year, max_year) filter.

        Returns:
            DataFrame with columns [year_ceremony, film, is_winner] for film-only awards,
            or [year_ceremony, film, is_winner, person] for person-level awards.
        """
        if award_key not in AWARD_TABLE_CONFIG:
            raise ValueError(f"No table config for award: {award_key}")

        film_hint, person_hint = AWARD_TABLE_CONFIG[award_key]
        url = AWARD_URLS[award_key]

        df = self._fetch_wikipedia_table_with_winners(url, award_key)
        if df is None:
            return pd.DataFrame()

        # Find year column — look for "Year" name, fall back to first column
        try:
            year_series = self._find_column_by_hint(df, "Year")
        except ValueError:
            year_series = df.iloc[:, 0]

        df["year_ceremony"] = year_series.apply(self._parse_award_year)

        # Forward-fill years for tables with merged year cells (rowspan).
        # Wikipedia award tables commonly span the year cell across all nominees
        # for that year. The BS4 parser handles year sub-headers separately,
        # but for standard tables with rowspan year cells, ffill is still needed.
        df["year_ceremony"] = df["year_ceremony"].ffill()

        # Find film column by name hint
        try:
            film_series = self._find_column_by_hint(df, film_hint)
        except ValueError:
            print(f"⚠ Warning: {award_key} — no column matching '{film_hint}'")
            return pd.DataFrame()

        df["film"] = film_series.apply(self._clean_film_title)
        df["is_winner"] = df.get("_is_winner_from_html", False)

        # Find person column if this is a person-level award
        if person_hint is not None:
            try:
                person_series = self._find_column_by_hint(df, person_hint)
                df["person"] = person_series.apply(self._clean_person_name)
            except ValueError:
                print(f"⚠ Warning: {award_key} — no column matching '{person_hint}'")
                df["person"] = None

        # Filter empty films and rows without parseable years
        df = df[df["film"].str.strip().str.len() > 0].copy()
        df = df.dropna(subset=["year_ceremony"])

        # Filter out known data quality issues
        # 1. Studio names parsed as film titles (Annie Award tables)
        studio_mask = df["film"].apply(self._is_studio_name)
        if studio_mask.any():
            n_dropped = studio_mask.sum()
            print(f"  ⚠ {award_key}: dropped {n_dropped} rows with studio names as film titles")
            df = df[~studio_mask].copy()

        # 2. Concatenated film lists parsed as single title (BAFTA animated)
        concat_mask = df["film"].apply(self._is_concatenated_film_list)
        if concat_mask.any():
            n_dropped = concat_mask.sum()
            print(f"  ⚠ {award_key}: dropped {n_dropped} rows with concatenated film lists")
            df = df[~concat_mask].copy()

        # 3. Empty person names in person-level awards: set to None instead of dropping.
        # The row is still useful for film-level matching in build_dataset.py (the
        # matching logic auto-detects person vs film level and falls back to film-only
        # when all persons are None). Dropping would lose entire awards where Wikipedia
        # table formatting doesn't expose person names (e.g., BAFTA Lead Actress used
        # gender-neutral "Actor" header that didn't match the old "Actress" hint).
        if person_hint is not None and "person" in df.columns:
            empty_person_mask = df["person"].str.strip().str.len() == 0
            if empty_person_mask.any():
                n_empty = int(empty_person_mask.sum())
                print(f"  ⚠ {award_key}: {n_empty} rows with empty person names (set to None)")
                df.loc[empty_person_mask, "person"] = None

        if year_range:
            df = df[(df["year_ceremony"] >= year_range[0]) & (df["year_ceremony"] <= year_range[1])]

        output_cols = ["year_ceremony", "film", "is_winner"]
        if person_hint is not None:
            output_cols.append("person")

        return df[output_cols]

    def fetch_all_awards(
        self,
        year_range: tuple[int, int] | None = None,
        progress: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch all precursor awards across all categories.

        Uses the generalized parser for all awards, driven by AWARD_TABLE_CONFIG
        column hints for each award.

        Args:
            year_range: Tuple of (min_year, max_year) for ceremony years.
            progress: Whether to print progress.

        Returns:
            Dict mapping award key (str) to DataFrame.
        """
        awards: dict[str, pd.DataFrame] = {}

        for award_key in AWARD_URLS:
            if progress:
                print(f"Fetching {award_key.upper()}...")
            try:
                df = self._fetch_generic_award(award_key, year_range)
                awards[award_key] = df
                if progress:
                    person_info = ""
                    if "person" in df.columns and len(df) > 0:
                        person_info = " (with person data)"
                    print(f"  ✓ {len(df)} records{person_info}")
            except Exception as e:
                print(f"  ✗ Error fetching {award_key}: {e}")
                awards[award_key] = pd.DataFrame()

        return awards

    def match_film(
        self,
        oscar_film: str,
        precursor_films: list[str],
        year: int | None = None,
    ) -> tuple[str | None, int]:
        """
        Find best fuzzy match for Oscar film in precursor awards list.

        Args:
            oscar_film: Film title from Oscar nominations
            precursor_films: List of film titles from precursor award
            year: Optional year to narrow search

        Returns:
            Tuple of (matched_film, match_score) or (None, 0) if no good match.
        """
        oscar_clean = self._clean_film_title(oscar_film).lower()

        # Remove articles for comparison
        oscar_no_article = oscar_clean.replace("the ", "").replace("a ", "").strip()

        best_match = None
        best_score = 0

        for precursor_film in precursor_films:
            precursor_clean = self._clean_film_title(precursor_film).lower()
            precursor_no_article = precursor_clean.replace("the ", "").replace("a ", "").strip()

            # Try multiple fuzzy matching strategies
            ratio = fuzz.ratio(oscar_clean, precursor_clean)
            token_sort = fuzz.token_sort_ratio(oscar_clean, precursor_clean)
            partial = fuzz.partial_ratio(oscar_clean, precursor_clean)

            # Also try without articles
            ratio_no_article = fuzz.ratio(oscar_no_article, precursor_no_article)

            # Use best score from all strategies
            score = max(ratio, token_sort, partial, ratio_no_article)

            # Additional validation: length check to avoid substring matches
            # E.g., "Lincoln" shouldn't match "Abraham Lincoln: Vampire Hunter"
            len_ratio = min(len(oscar_clean), len(precursor_clean)) / max(
                len(oscar_clean), len(precursor_clean)
            )

            # If one title is much shorter, penalize the score
            if len_ratio < 0.6:
                score = int(score * len_ratio * 1.5)  # Penalize short matches

            if score > best_score:
                best_score = score
                best_match = precursor_film

        if best_score >= self.fuzzy_threshold:
            return best_match, best_score

        return None, 0


def test_fetcher():
    """Test the precursor awards fetcher."""
    fetcher = PrecursorAwardsFetcher()

    # Test fetching a specific award via generic parser
    print("\n=== Testing PGA Fetching ===")
    pga_df = fetcher._fetch_generic_award(PrecursorKey.PGA_BP, year_range=(2023, 2025))
    print(pga_df)

    # Test fuzzy matching
    print("\n=== Testing Fuzzy Matching ===")
    test_cases = [
        ("Oppenheimer", ["Oppenheimer", "The Holdovers", "American Fiction"]),
        ("One Battle After Another", ["One Battle after Another", "Sinners"]),
        ("Everything Everywhere All at Once", ["Everything Everywhere All At Once"]),
    ]

    for oscar_film, precursor_list in test_cases:
        match, score = fetcher.match_film(oscar_film, precursor_list)
        print(f"\nOscar: '{oscar_film}'")
        print(f"Best match: '{match}' (score: {score})")


if __name__ == "__main__":
    test_fetcher()
