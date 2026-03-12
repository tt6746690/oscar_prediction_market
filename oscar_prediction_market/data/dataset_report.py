"""Dataset report — single-file informational summary for any Oscar category dataset.

Prints completeness, coverage, and data quality diagnostics for one raw dataset JSON.
Purely informational — no pass/fail or sys.exit. Consolidates the reporting logic
from the former data_completeness_report.py, dataset_stats.py, and validate_datasets.py.

Usage:
    uv run python -m oscar_prediction_market.data.dataset_report \
        path/to/oscar_best_picture_raw.json

    # With per-year breakdown
    uv run python -m ...dataset_report path/to/oscar_best_picture_raw.json --per-year

    # Sample records
    uv run python -m ...dataset_report path/to/oscar_best_picture_raw.json --samples 3
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from oscar_prediction_market.data.precursor_mappings import (
    get_precursor_specs,
)
from oscar_prediction_market.data.schema import (
    PERSON_CATEGORIES,
    AwardResult,
    NominationDataset,
)
from oscar_prediction_market.modeling.utils import ceremony_to_year


def _pct(n: int, total: int) -> str:
    """Format count/total as percentage string."""
    if total == 0:
        return "0/0 (0.0%)"
    return f"{n}/{total} ({100 * n / total:.1f}%)"


def print_overview(data: NominationDataset) -> None:
    """Print high-level dataset overview."""
    print("=" * 70)
    print(f"Dataset Report — {data.category.value}")
    print("=" * 70)

    winners = sum(1 for r in data.records if r.category_winner)
    ceremonies = sorted({r.ceremony for r in data.records})
    year_range = f"{ceremony_to_year(ceremonies[0])}–{ceremony_to_year(ceremonies[-1])}"

    print(f"  Category:       {data.category.value}")
    print(f"  Records:        {len(data.records)}")
    print(f"  Winners:        {winners}")
    print(f"  Ceremonies:     {len(ceremonies)} ({year_range})")
    print(f"  Avg/ceremony:   {len(data.records) / len(ceremonies):.1f}")
    is_person = data.category in PERSON_CATEGORIES
    print(f"  Person-level:   {'yes' if is_person else 'no'}")


def print_metadata_completeness(data: NominationDataset) -> None:
    """Print metadata field completeness."""
    print("\n" + "-" * 70)
    print("Metadata Completeness")
    print("-" * 70)

    fields = [
        "imdb_rating",
        "imdb_votes",
        "rotten_tomatoes",
        "metacritic",
        "box_office_domestic",
        "box_office_worldwide",
        "budget",
        "runtime_minutes",
    ]
    total = len(data.records)
    for field in fields:
        n = sum(
            1
            for r in data.records
            if r.film.metadata and getattr(r.film.metadata, field) is not None
        )
        pct = 100 * n / total if total else 0
        flag = "" if pct >= 95 else "  ⚠" if pct >= 70 else "  ✗"
        print(f"  {field:<28s} {_pct(n, total)}{flag}")

    # Has metadata at all
    n_meta = sum(1 for r in data.records if r.film.metadata is not None)
    print(f"  {'(has metadata)':<28s} {_pct(n_meta, total)}")


def print_precursor_coverage(data: NominationDataset) -> None:
    """Print precursor award coverage per mapped award."""
    print("\n" + "-" * 70)
    print("Precursor Coverage")
    print("-" * 70)

    specs = get_precursor_specs(data.category)
    total = len(data.records)

    for spec in specs:
        n_nom = sum(
            1 for r in data.records if r.precursors.get(spec.key, AwardResult()).nominee is not None
        )
        n_win = sum(
            1 for r in data.records if r.precursors.get(spec.key, AwardResult()).winner is not None
        )
        pct_nom = 100 * n_nom / total if total else 0
        flag = "" if pct_nom >= 70 else "  ⚠" if pct_nom >= 30 else "  ✗"
        print(f"  {spec.key:<36s} nom={_pct(n_nom, total)}  win={_pct(n_win, total)}{flag}")


def print_person_data(data: NominationDataset) -> None:
    """Print person data completeness (only for person-level categories)."""
    if data.category not in PERSON_CATEGORIES:
        return

    print("\n" + "-" * 70)
    print("Person Data")
    print("-" * 70)

    total = len(data.records)
    n_person = sum(1 for r in data.records if r.person is not None)
    print(f"  {'Has PersonData':<28s} {_pct(n_person, total)}")

    # TMDb enrichment fields
    tmdb_fields = [
        "birth_date",
        "tmdb_popularity",
        "total_film_credits",
    ]
    for field in tmdb_fields:
        n = sum(1 for r in data.records if r.person and getattr(r.person, field) is not None)
        print(f"  {field:<28s} {_pct(n, total)}")


def print_per_year_breakdown(data: NominationDataset) -> None:
    """Print per-ceremony-year breakdown: nominees, winners, precursor coverage."""
    print("\n" + "-" * 70)
    print("Per-Year Breakdown")
    print("-" * 70)

    specs = get_precursor_specs(data.category)
    by_year: dict[int, list] = defaultdict(list)
    for r in data.records:
        by_year[r.ceremony].append(r)

    # Header
    header = f"  {'Year':<6s} {'N':>3s} {'W':>3s}"
    for spec in specs:
        # Abbreviate key for column header
        short = spec.key[:12]
        header += f" {short:>12s}"
    print(header)

    for ceremony in sorted(by_year):
        records = by_year[ceremony]
        year = ceremony_to_year(ceremony)
        n_records = len(records)
        n_winners = sum(1 for r in records if r.category_winner)

        row = f"  {year:<6d} {n_records:>3d} {n_winners:>3d}"
        for spec in specs:
            n_avail = sum(
                1 for r in records if r.precursors.get(spec.key, AwardResult()).nominee is not None
            )
            pct = 100 * n_avail / n_records if n_records else 0
            cell = f"{pct:.0f}%" if pct > 0 else "—"
            row += f" {cell:>12s}"
        print(row)


def print_sample_records(data: NominationDataset, n_samples: int = 3) -> None:
    """Print sample records for manual inspection."""
    print("\n" + "-" * 70)
    print(f"Sample Records (first {n_samples})")
    print("-" * 70)

    for r in data.records[:n_samples]:
        year = ceremony_to_year(r.ceremony)
        winner = "★" if r.category_winner else " "
        person_str = f" — {r.nominee_name}" if r.nominee_name else ""
        print(f"\n  [{winner}] {r.film.title} ({year}){person_str}")

        # Metadata snippet
        if r.film.metadata:
            m = r.film.metadata
            parts = []
            if m.imdb_rating is not None:
                parts.append(f"IMDb={m.imdb_rating}")
            if m.rotten_tomatoes is not None:
                parts.append(f"RT={m.rotten_tomatoes}%")
            if m.metacritic is not None:
                parts.append(f"MC={m.metacritic}")
            if parts:
                print(f"      metadata: {', '.join(parts)}")

        # Precursors
        if r.precursors:
            prec_parts = []
            for key, ar in sorted(r.precursors.items()):
                if ar.winner is True:
                    prec_parts.append(f"{key}=W")
                elif ar.nominee is True:
                    prec_parts.append(f"{key}=N")
                elif ar.nominee is False:
                    prec_parts.append(f"{key}=✗")
            if prec_parts:
                print(f"      precursors: {', '.join(prec_parts)}")

        # Oscar profile
        noms = r.film.oscar_noms
        if noms:
            print(f"      oscar: {noms.oscar_total_nominations} noms, {noms.oscar_total_wins} wins")


def print_duplicate_check(data: NominationDataset) -> None:
    """Check for duplicate (ceremony, film) pairs.

    In supporting actor/actress categories, the same film may appear multiple times
    with different nominees (e.g., two actors from "The Irishman" both nominated).
    This is expected for person-level categories and flagged as informational.

    For film-level categories (BP, Animated), duplicates indicate a data issue.
    """
    print("\n" + "-" * 70)
    print("Duplicate Check")
    print("-" * 70)

    is_person_cat = data.category in PERSON_CATEGORIES

    # Check (ceremony, film) duplicates
    film_pairs = Counter((r.ceremony, r.film.title) for r in data.records)
    dupes = {k: v for k, v in film_pairs.items() if v > 1}

    if not dupes:
        print("  No duplicate (ceremony, film) pairs.")
        return

    label = "expected (person-level category)" if is_person_cat else "⚠ unexpected"
    print(f"  {len(dupes)} duplicate (ceremony, film) pairs ({label}):")
    for (ceremony, film), count in sorted(dupes.items()):
        year = ceremony_to_year(ceremony)
        print(f"    {year}: '{film}' x{count}")

    # For person-level, also check (ceremony, film, person) duplicates — those are real bugs
    if is_person_cat:
        person_pairs = Counter((r.ceremony, r.film.title, r.nominee_name) for r in data.records)
        person_dupes = {k: v for k, v in person_pairs.items() if v > 1}
        if person_dupes:
            print(f"\n  ⚠ {len(person_dupes)} duplicate (ceremony, film, person) — data bug:")
            for (ceremony, film, name), count in sorted(person_dupes.items()):
                year = ceremony_to_year(ceremony)
                print(f"    {year}: '{film}' / '{name}' x{count}")
        else:
            print("  No duplicate (ceremony, film, person) pairs — all expected.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print informational report for an Oscar raw dataset JSON."
    )
    parser.add_argument("dataset_path", type=Path, help="Path to oscar_*_raw.json")
    parser.add_argument("--per-year", action="store_true", help="Include per-year breakdown table")
    parser.add_argument("--samples", type=int, default=0, help="Number of sample records to show")
    args = parser.parse_args()

    path: Path = args.dataset_path
    if not path.exists():
        print(f"Error: file not found: {path}")
        return

    with open(path) as f:
        raw = json.load(f)
    data = NominationDataset.model_validate(raw)

    print_overview(data)
    print_metadata_completeness(data)
    print_precursor_coverage(data)
    print_person_data(data)
    print_duplicate_check(data)

    if args.per_year:
        print_per_year_breakdown(data)

    if args.samples > 0:
        print_sample_records(data, args.samples)

    print()


if __name__ == "__main__":
    main()
