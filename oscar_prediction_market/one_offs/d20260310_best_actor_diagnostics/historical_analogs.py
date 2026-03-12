"""Historical analog analysis for Best Actor nominees.

For each 2026 Best Actor nominee, finds the most similar historical nominees
by precursor award profile (Hamming distance on winner/nomination vectors).

This answers questions like:
- "Has anyone with Chalamet's exact precursor profile ever won?"
- "What's the base rate for nominees with N precursor wins?"
- "Does nomination breadth matter even without wins?"

Usage::

    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\
d20260310_best_actor_diagnostics.historical_analogs
"""

from pathlib import Path

from oscar_prediction_market.data.schema import (
    AwardResult,
    NominationDataset,
    NominationRecord,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_PATH = Path(
    "storage/d20260224_live_2026/datasets/actor_leading/2026-03-08_wga/oscar_actor_leading_raw.json"
)

CEREMONY_YEAR_2026 = 2026
CEREMONY_NUMBER_2026 = CEREMONY_YEAR_2026 - 1928  # 98

# Precursor keys relevant to ACTOR_LEADING, in display order.
PRECURSOR_KEYS: list[str] = [
    "sag_lead_actor",
    "bafta_lead_actor",
    "golden_globe_actor_drama",
    "golden_globe_actor_musical",
    "critics_choice_actor",
]

# Short display names for table headers.
PRECURSOR_SHORT: dict[str, str] = {
    "sag_lead_actor": "SAG",
    "bafta_lead_actor": "BAFTA",
    "golden_globe_actor_drama": "GG-Dra",
    "golden_globe_actor_musical": "GG-Mus",
    "critics_choice_actor": "CC",
}

TOP_K = 5  # Number of analogs to show per nominee.

# Nominees to highlight (printed first); the rest follow alphabetically.
FOCUS_NOMINEES: list[str] = ["Timothée Chalamet", "Michael B. Jordan"]


# ---------------------------------------------------------------------------
# Profile extraction
# ---------------------------------------------------------------------------


def _get_award(precursors: dict[str, AwardResult], key: str) -> AwardResult:
    """Return AwardResult for a key, defaulting to empty (None/None)."""
    return precursors.get(key, AwardResult(nominee=None, winner=None))


def winner_vector(record: NominationRecord) -> list[int]:
    """Binary vector: 1 if won precursor, 0 otherwise (None → 0)."""
    return [int(bool(_get_award(record.precursors, k).winner)) for k in PRECURSOR_KEYS]


def nominee_vector(record: NominationRecord) -> list[int]:
    """Binary vector: 1 if nominated for precursor, 0 otherwise (None → 0)."""
    return [int(bool(_get_award(record.precursors, k).nominee)) for k in PRECURSOR_KEYS]


def hamming_distance(a: list[int], b: list[int]) -> int:
    """Number of positions where two equal-length binary vectors differ."""
    return sum(x != y for x, y in zip(a, b, strict=True))


# ---------------------------------------------------------------------------
# Analog search
# ---------------------------------------------------------------------------


def find_winner_analogs(
    target: NominationRecord,
    pool: list[NominationRecord],
    top_k: int = TOP_K,
) -> list[tuple[NominationRecord, int]]:
    """Find top-k most similar historical nominees by winner-vector Hamming distance."""
    target_vec = winner_vector(target)
    scored: list[tuple[NominationRecord, int]] = []
    for rec in pool:
        dist = hamming_distance(target_vec, winner_vector(rec))
        scored.append((rec, dist))
    scored.sort(key=lambda x: (x[1], -(x[0].ceremony)))
    return scored[:top_k]


def find_nominee_analogs(
    target: NominationRecord,
    pool: list[NominationRecord],
    top_k: int = TOP_K,
) -> list[tuple[NominationRecord, int]]:
    """Find top-k most similar historical nominees by nominee-vector Hamming distance."""
    target_vec = nominee_vector(target)
    scored: list[tuple[NominationRecord, int]] = []
    for rec in pool:
        dist = hamming_distance(target_vec, nominee_vector(rec))
        scored.append((rec, dist))
    scored.sort(key=lambda x: (x[1], -(x[0].ceremony)))
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _ceremony_year(record: NominationRecord) -> int:
    return record.ceremony + 1928


def _oscar_icon(record: NominationRecord) -> str:
    return "WIN" if record.category_winner else "---"


def _profile_str(vec: list[int]) -> str:
    """Compact display: e.g. '1 0 1 0 1'."""
    return " ".join(str(v) for v in vec)


def _precursor_detail_str(record: NominationRecord) -> str:
    """Show nomination status per precursor: W=won, N=nominated, .=none."""
    parts: list[str] = []
    for key in PRECURSOR_KEYS:
        ar = _get_award(record.precursors, key)
        if ar.winner:
            parts.append("W")
        elif ar.nominee:
            parts.append("N")
        else:
            parts.append(".")
    return " ".join(parts)


def _format_profile_long(record: NominationRecord) -> str:
    """Human-readable like ``SAG(W), BAFTA(N), GG-Dra(-), ...``."""
    parts: list[str] = []
    for key in PRECURSOR_KEYS:
        label = PRECURSOR_SHORT[key]
        ar = _get_award(record.precursors, key)
        if ar.winner:
            parts.append(f"{label}(W)")
        elif ar.nominee:
            parts.append(f"{label}(N)")
        else:
            parts.append(f"{label}(-)")
    return ", ".join(parts)


def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def _order_nominees(
    nominees: list[NominationRecord],
) -> list[NominationRecord]:
    """Order: focus nominees first, then rest alphabetically."""
    focus_set = set(FOCUS_NOMINEES)
    ordered: list[NominationRecord] = []
    for name in FOCUS_NOMINEES:
        match = [r for r in nominees if r.nominee_name == name]
        if match:
            ordered.append(match[0])
    rest = sorted(
        [r for r in nominees if r.nominee_name not in focus_set],
        key=lambda r: r.nominee_name or "",
    )
    ordered.extend(rest)
    return ordered


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def run_analog_analysis(dataset: NominationDataset) -> None:
    """Run the full historical analog analysis and print results."""

    # Partition into 2026 nominees and historical pool.
    nominees_2026 = [r for r in dataset.records if r.ceremony == CEREMONY_NUMBER_2026]
    historical = [r for r in dataset.records if r.ceremony < CEREMONY_NUMBER_2026]

    if not nominees_2026:
        print(f"ERROR: No 2026 nominees found (ceremony={CEREMONY_NUMBER_2026})")
        return

    ordered_2026 = _order_nominees(nominees_2026)

    # -----------------------------------------------------------------------
    # Section 1: 2026 nominee profiles
    # -----------------------------------------------------------------------
    print_header("2026 BEST ACTOR NOMINEE PROFILES")

    hdr_keys = "  ".join(f"{PRECURSOR_SHORT[k]:>6}" for k in PRECURSOR_KEYS)
    print(f"\n{'Nominee':<30} {hdr_keys}  {'Wins':>5}  {'Noms':>5}")
    print("-" * 80)

    for rec in ordered_2026:
        wvec = winner_vector(rec)
        nvec = nominee_vector(rec)
        wins = sum(wvec)
        noms = sum(nvec)
        detail = ""
        for key in PRECURSOR_KEYS:
            ar = _get_award(rec.precursors, key)
            if ar.winner:
                detail += f"{'W':>8}"
            elif ar.nominee:
                detail += f"{'N':>8}"
            else:
                detail += f"{'.':>8}"
        name_film = f"{rec.nominee_name} ({rec.film.title})"
        if len(name_film) > 30:
            name_film = name_film[:27] + "..."
        print(f"{name_film:<30}{detail}  {wins:>5}  {noms:>5}")

    # -----------------------------------------------------------------------
    # Section 2: Historical analogs per 2026 nominee (winner vector)
    # -----------------------------------------------------------------------
    print_header("HISTORICAL ANALOGS (by precursor WINNER profile)")

    for rec in ordered_2026:
        target_wvec = winner_vector(rec)
        target_wins = sum(target_wvec)
        print(f"\n--- {rec.nominee_name} ({rec.film.title}) ---")
        print(f"    Winner profile: [{_profile_str(target_wvec)}]  ({target_wins} wins)")
        print(f"    Detail: {_format_profile_long(rec)}")
        print(f"    {'Year':<6} {'Nominee':<28} {'Film':<24} {'Profile':<14} {'Dist':>4}  Oscar")
        print(f"    {'-' * 86}")

        analogs = find_winner_analogs(rec, historical, top_k=TOP_K)
        for analog_rec, dist in analogs:
            yr = _ceremony_year(analog_rec)
            awvec = winner_vector(analog_rec)
            prof = _profile_str(awvec)
            oscar = _oscar_icon(analog_rec)
            nominee = analog_rec.nominee_name or "?"
            film = analog_rec.film.title or "?"
            print(f"    {yr:<6} {nominee:<28} {film:<24} [{prof}]  {dist:>4}  {oscar}")

        analog_wins = sum(1 for a, _ in analogs if a.category_winner)
        print(f"    → {analog_wins}/{len(analogs)} analogs won the Oscar")

    # -----------------------------------------------------------------------
    # Section 3: Historical analogs per 2026 nominee (nomination vector)
    # -----------------------------------------------------------------------
    print_header("HISTORICAL ANALOGS (by precursor NOMINATION profile)")

    for rec in ordered_2026:
        target_nvec = nominee_vector(rec)
        target_noms = sum(target_nvec)
        print(f"\n--- {rec.nominee_name} ({rec.film.title}) ---")
        print(f"    Nomination profile: [{_profile_str(target_nvec)}]  ({target_noms} nominations)")
        print(f"    Detail: {_format_profile_long(rec)}")
        print(
            f"    {'Year':<6} {'Nominee':<28} {'Film':<24} "
            f"{'NomProf':<14} {'Detail':<14} {'Dist':>4}  Oscar"
        )
        print(f"    {'-' * 100}")

        analogs_nom = find_nominee_analogs(rec, historical, top_k=TOP_K)
        for analog_rec, dist in analogs_nom:
            yr = _ceremony_year(analog_rec)
            anvec = nominee_vector(analog_rec)
            nom_prof = _profile_str(anvec)
            detail = _precursor_detail_str(analog_rec)
            oscar = _oscar_icon(analog_rec)
            nominee = analog_rec.nominee_name or "?"
            film = analog_rec.film.title or "?"
            print(
                f"    {yr:<6} {nominee:<28} {film:<24} [{nom_prof}]  [{detail}]  {dist:>4}  {oscar}"
            )

        analog_wins = sum(1 for a, _ in analogs_nom if a.category_winner)
        print(f"    → {analog_wins}/{len(analogs_nom)} analogs won the Oscar")

    # -----------------------------------------------------------------------
    # Section 4: Base rate table — Oscar win rate by # precursor wins
    # -----------------------------------------------------------------------
    print_header("BASE RATE TABLE: Oscar win rate by # precursor wins")

    max_wins = len(PRECURSOR_KEYS)
    counts: dict[int, int] = dict.fromkeys(range(max_wins + 1), 0)
    oscar_wins: dict[int, int] = dict.fromkeys(range(max_wins + 1), 0)

    for rec in historical:
        n_wins = sum(winner_vector(rec))
        counts[n_wins] += 1
        if rec.category_winner:
            oscar_wins[n_wins] += 1

    print(f"\n{'Precursor Wins':>15}  {'Nominees':>8}  {'Oscar Wins':>10}  {'Win Rate':>8}")
    print("-" * 50)
    for n in range(max_wins + 1):
        total = counts[n]
        wins = oscar_wins[n]
        rate = f"{wins / total:.1%}" if total > 0 else "N/A"
        print(f"{n:>15}  {total:>8}  {wins:>10}  {rate:>8}")

    total_all = sum(counts.values())
    wins_all = sum(oscar_wins.values())
    rate_all = f"{wins_all / total_all:.1%}" if total_all > 0 else "N/A"
    print("-" * 50)
    print(f"{'TOTAL':>15}  {total_all:>8}  {wins_all:>10}  {rate_all:>8}")

    # -----------------------------------------------------------------------
    # Section 5: Base rate by # precursor nominations
    # -----------------------------------------------------------------------
    print_header("BASE RATE TABLE: Oscar win rate by # precursor nominations")

    nom_counts: dict[int, int] = dict.fromkeys(range(max_wins + 1), 0)
    nom_oscar_wins: dict[int, int] = dict.fromkeys(range(max_wins + 1), 0)

    for rec in historical:
        n_noms = sum(nominee_vector(rec))
        nom_counts[n_noms] += 1
        if rec.category_winner:
            nom_oscar_wins[n_noms] += 1

    print(f"\n{'Precursor Noms':>15}  {'Nominees':>8}  {'Oscar Wins':>10}  {'Win Rate':>8}")
    print("-" * 50)
    for n in range(max_wins + 1):
        total = nom_counts[n]
        wins = nom_oscar_wins[n]
        rate = f"{wins / total:.1%}" if total > 0 else "N/A"
        print(f"{n:>15}  {total:>8}  {wins:>10}  {rate:>8}")

    total_all = sum(nom_counts.values())
    wins_all = sum(nom_oscar_wins.values())
    rate_all = f"{wins_all / total_all:.1%}" if total_all > 0 else "N/A"
    print("-" * 50)
    print(f"{'TOTAL':>15}  {total_all:>8}  {wins_all:>10}  {rate_all:>8}")

    # -----------------------------------------------------------------------
    # Section 6: Cross-tab — wins vs nominations (the "nominated but lost" gap)
    # -----------------------------------------------------------------------
    print_header("CROSS-TAB: Precursor Wins vs Nominations → Oscar win rate")
    print("\nRows = # precursor nominations, Cols = # precursor wins")
    print("Cell = Oscar wins / total nominees (win rate)\n")

    # Build cross-tab: (n_noms, n_wins) → (oscar_wins, total)
    cross: dict[tuple[int, int], tuple[int, int]] = {}
    for rec in historical:
        n_wins_val = sum(winner_vector(rec))
        n_noms_val = sum(nominee_vector(rec))
        key = (n_noms_val, n_wins_val)
        prev_ow, prev_tot = cross.get(key, (0, 0))
        cross[key] = (prev_ow + int(rec.category_winner), prev_tot + 1)

    col_header = f"{'Noms\\Wins':>10}"
    for w in range(max_wins + 1):
        col_header += f"  {w:>12}"
    print(col_header)
    print("-" * (12 + 14 * (max_wins + 1)))

    for n in range(max_wins + 1):
        row = f"{n:>10}"
        for w in range(max_wins + 1):
            if w > n:
                # Can't win more than nominated for
                row += f"  {'':>12}"
            else:
                ow, tot = cross.get((n, w), (0, 0))
                if tot == 0:
                    row += f"  {'--':>12}"
                else:
                    rate_str = f"{ow}/{tot} ({ow / tot:.0%})"
                    row += f"  {rate_str:>12}"
        print(row)

    # -----------------------------------------------------------------------
    # Section 7: Where 2026 nominees sit in the base rate tables
    # -----------------------------------------------------------------------
    print_header("2026 NOMINEES IN CONTEXT")
    print()
    for rec in ordered_2026:
        wvec = winner_vector(rec)
        nvec = nominee_vector(rec)
        n_wins = sum(wvec)
        n_noms = sum(nvec)

        # Historical win rate for this bucket.
        hist_total = counts.get(n_wins, 0)
        hist_oscar = oscar_wins.get(n_wins, 0)
        rate_w = (
            f"{hist_oscar}/{hist_total} ({hist_oscar / hist_total:.0%})" if hist_total else "N/A"
        )

        hist_total_n = nom_counts.get(n_noms, 0)
        hist_oscar_n = nom_oscar_wins.get(n_noms, 0)
        rate_n = (
            f"{hist_oscar_n}/{hist_total_n} ({hist_oscar_n / hist_total_n:.0%})"
            if hist_total_n
            else "N/A"
        )

        name_film = f"{rec.nominee_name} ({rec.film.title})"
        print(f"  {name_film}")
        print(
            f"    Precursor wins:  {n_wins}/{len(PRECURSOR_KEYS)}  "
            f"→ historical Oscar rate: {rate_w}"
        )
        print(
            f"    Precursor noms:  {n_noms}/{len(PRECURSOR_KEYS)}  "
            f"→ historical Oscar rate: {rate_n}"
        )
        print(f"    Detail: [{_precursor_detail_str(rec)}]")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"Loading dataset from: {DATASET_PATH}")
    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        return
    dataset = NominationDataset.model_validate_json(DATASET_PATH.read_text())
    print(
        f"Loaded {dataset.record_count} records, ceremonies {dataset.year_start}–{dataset.year_end}"
    )
    run_analog_analysis(dataset)


if __name__ == "__main__":
    main()
