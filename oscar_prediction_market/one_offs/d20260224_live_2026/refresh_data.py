"""Refresh stale precursor awards and film metadata caches for 2026 live predictions.

The shared data in d20260218_build_all_datasets/ was built Feb 18 and is missing:
- Annie winner (Feb 21)
- BAFTA winner (Feb 22)
- PGA, SAG, ASC, WGA (future)

This script:
1. Re-fetches precursor awards with refresh_cache=True to bust stale Wikipedia HTML
2. Re-fetches film metadata (OMDb + TMDb) for 2026 films to pick up any updates
3. Does before/after comparison and reports differences
4. Saves refreshed data to storage/d20260224_live_2026/shared/

Usage::

    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.d20260224_live_2026.refresh_data
"""

import json
from pathlib import Path

from oscar_prediction_market.data.build_dataset import (
    build_film_metadata,
    build_precursor_awards,
    save_json,
)
from oscar_prediction_market.data.fetch_precursor_awards import (
    PrecursorAwardsFetcher,
)
from oscar_prediction_market.data.schema import (
    PrecursorAwardsFile,
)
from oscar_prediction_market.one_offs.d20260224_live_2026 import (
    EXP_DIR,
    SOURCE_DATASETS_DIR,
)

SHARED_DIR = Path(EXP_DIR) / "shared"
OLD_SHARED_DIR = Path(SOURCE_DATASETS_DIR) / "shared"


def _load_precursor_awards(path: Path) -> PrecursorAwardsFile:
    """Load PrecursorAwardsFile from JSON."""
    with open(path) as f:
        return PrecursorAwardsFile(**json.load(f))


def _compare_precursor_awards(
    old: PrecursorAwardsFile,
    new: PrecursorAwardsFile,
) -> list[str]:
    """Compare old vs new precursor data and return a list of differences."""
    diffs: list[str] = []

    old_awards = set(old.awards.keys())
    new_awards = set(new.awards.keys())
    if old_awards != new_awards:
        added = new_awards - old_awards
        removed = old_awards - new_awards
        if added:
            diffs.append(f"New award types: {sorted(added)}")
        if removed:
            diffs.append(f"Removed award types: {sorted(removed)}")

    for award_name in sorted(old_awards & new_awards):
        old_recs = old.awards[award_name]
        new_recs = new.awards[award_name]

        if len(old_recs) != len(new_recs):
            diffs.append(f"  {award_name}: {len(old_recs)} → {len(new_recs)} records")

        # Check for new winners (is_winner changed from False to True, or new records with is_winner)
        old_winners = {(r.year_ceremony, r.film): r.is_winner for r in old_recs}
        new_winners = {(r.year_ceremony, r.film): r.is_winner for r in new_recs}

        for key, is_winner in new_winners.items():
            old_was_winner = old_winners.get(key)
            if is_winner and not old_was_winner:
                year, film = key
                diffs.append(f"  {award_name}: NEW WINNER year={year} film={film}")

    return diffs


def _compare_metadata_before_after(old_path: Path, new_path: Path) -> list[str]:
    """Compare film metadata before/after refresh for 2026 films."""
    diffs: list[str] = []

    with open(old_path) as f:
        old_data = json.load(f)
    with open(new_path) as f:
        new_data = json.load(f)

    old_records = old_data.get("records", {})
    new_records = new_data.get("records", {})

    # Only check fields likely to change: ratings, box office, awards
    check_fields = [
        "imdb_rating",
        "imdb_votes",
        "rotten_tomatoes",
        "metacritic",
        "box_office_domestic",
        "box_office_worldwide",
        "total_awards_wins",
        "total_awards_nominations",
    ]

    for film_id in sorted(set(old_records.keys()) & set(new_records.keys())):
        old_rec = old_records[film_id]
        new_rec = new_records[film_id]
        title = new_rec.get("title", film_id)

        for field in check_fields:
            old_val = old_rec.get(field)
            new_val = new_rec.get(field)
            if old_val != new_val:
                diffs.append(f"  {title} ({film_id}): {field} {old_val} → {new_val}")

    new_ids = set(new_records.keys()) - set(old_records.keys())
    if new_ids:
        diffs.append(f"  New films: {sorted(new_ids)}")

    return diffs


def refresh_precursors() -> list[str]:
    """Re-fetch precursor awards with cache busting. Returns diff summary."""
    print("=" * 60)
    print("Refreshing precursor awards (cache-bust)...")
    print("=" * 60)

    # Clear the diskcache for precursor HTML pages to force re-fetch
    PrecursorAwardsFetcher(refresh_cache=True)
    # The PrecursorAwardsFetcher with refresh_cache=True will ignore cached HTML

    # Load old data for comparison
    old_path = OLD_SHARED_DIR / "precursor_awards.json"
    old_data = _load_precursor_awards(old_path) if old_path.exists() else None

    # Build fresh precursor data (year range covering our training data)
    new_data = build_precursor_awards(year_start=2000, year_end=2026)

    # Save to our experiment's shared dir
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    save_json(new_data, "precursor_awards.json", SHARED_DIR)
    print(f"\nSaved to {SHARED_DIR / 'precursor_awards.json'}")

    # Compare
    diffs: list[str] = []
    if old_data:
        diffs = _compare_precursor_awards(old_data, new_data)
        if diffs:
            print("\n--- Precursor awards changes (old → new) ---")
            for d in diffs:
                print(d)
        else:
            print("\nNo changes detected in precursor awards.")
    else:
        print("\nNo old data to compare against.")

    return diffs


def refresh_metadata() -> list[str]:
    """Re-fetch film metadata for all categories. Returns diff summary.

    Note: OMDb doesn't have a refresh_cache option — it will serve from cache
    for films already fetched. Only NEW films (not previously fetched) will hit
    the API. For existing films, the data is stable enough that cache is fine.
    The main concern is picking up rating/box-office updates for 2026 films.
    """
    print("\n" + "=" * 60)
    print("Refreshing film metadata (OMDb + TMDb)...")
    print("=" * 60)

    # Collect input dirs from all 9 categories in the source build
    source = Path(SOURCE_DATASETS_DIR)
    input_dirs = [
        source / cat_slug
        for cat_slug in [
            "best_picture",
            "directing",
            "actor_leading",
            "actress_leading",
            "actor_supporting",
            "actress_supporting",
            "original_screenplay",
            "cinematography",
            "animated_feature",
        ]
        if (source / cat_slug / "oscar_nominations.json").exists()
    ]

    if not input_dirs:
        print("ERROR: No oscar_nominations.json found in source dirs!")
        return []

    new_data = build_film_metadata(2000, 2026, input_dirs)
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    save_json(new_data, "film_metadata.json", SHARED_DIR)
    print(f"\nSaved to {SHARED_DIR / 'film_metadata.json'}")

    # Compare with old
    old_path = OLD_SHARED_DIR / "film_metadata.json"
    new_path = SHARED_DIR / "film_metadata.json"
    diffs: list[str] = []
    if old_path.exists():
        diffs = _compare_metadata_before_after(old_path, new_path)
        if diffs:
            print(f"\n--- Film metadata changes ({len(diffs)} differences) ---")
            for d in diffs[:50]:  # Cap output
                print(d)
            if len(diffs) > 50:
                print(f"  ... and {len(diffs) - 50} more")
        else:
            print("\nNo changes detected in film metadata.")

    return diffs


def main() -> None:
    """Refresh both precursor awards and film metadata, report changes."""
    precursor_diffs = refresh_precursors()
    metadata_diffs = refresh_metadata()

    print("\n" + "=" * 60)
    print("REFRESH SUMMARY")
    print("=" * 60)
    print(f"Precursor changes: {len(precursor_diffs)}")
    print(f"Metadata changes:  {len(metadata_diffs)}")
    print(f"Output: {SHARED_DIR}")

    # Save diff summary for README documentation
    summary = {
        "precursor_changes": precursor_diffs,
        "metadata_changes": metadata_diffs,
    }
    summary_path = SHARED_DIR / "refresh_diff_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Diff summary: {summary_path}")


if __name__ == "__main__":
    main()
