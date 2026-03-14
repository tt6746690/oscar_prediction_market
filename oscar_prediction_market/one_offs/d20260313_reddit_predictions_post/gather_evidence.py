"""Gather precursor, model, and historical evidence for all 9 Oscar categories.

Produces a structured text summary for each category with:
1. Model ensemble probabilities (all 4 models + avg_ensemble)
2. Precursor award breakdown per nominee
3. Historical base rates (win rate by # precursor wins)
4. Historical analogs (top-5 most similar past nominees by winner profile)
5. Model agreement/disagreement analysis

Usage::

    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.d20260313_reddit_predictions_post.gather_evidence
"""

from pathlib import Path

from oscar_prediction_market.data.precursor_mappings import get_precursor_specs
from oscar_prediction_market.data.schema import (
    AwardResult,
    NominationDataset,
    NominationRecord,
    OscarCategory,
)
from oscar_prediction_market.modeling.models import ModelType
from oscar_prediction_market.modeling.prediction_io import load_predictions

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASETS_DIR = Path("storage/d20260224_live_2026/datasets")
MODELS_DIR = Path("storage/d20260224_live_2026/models")
SNAPSHOT_KEY = "2026-03-08_wga"
CEREMONY_YEAR = 2026

CATEGORIES = [
    OscarCategory.BEST_PICTURE,
    OscarCategory.DIRECTING,
    OscarCategory.ACTOR_LEADING,
    OscarCategory.ACTRESS_LEADING,
    OscarCategory.ACTOR_SUPPORTING,
    OscarCategory.ACTRESS_SUPPORTING,
    OscarCategory.ORIGINAL_SCREENPLAY,
    OscarCategory.CINEMATOGRAPHY,
    OscarCategory.ANIMATED_FEATURE,
]

MODEL_TYPES = [
    ModelType.CONDITIONAL_LOGIT,
    ModelType.LOGISTIC_REGRESSION,
    ModelType.GRADIENT_BOOSTING,
    ModelType.CALIBRATED_SOFTMAX_GBT,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_award(precursors: dict[str, AwardResult], key: str) -> AwardResult:
    return precursors.get(key, AwardResult(nominee=None, winner=None))


def _fmt_bool(val: bool | None) -> str:
    if val is None:
        return "—"
    return "W" if val else ("N" if val is False else "—")


def winner_vector(record: NominationRecord, precursor_keys: list[str]) -> list[int]:
    return [int(bool(_get_award(record.precursors, k).winner)) for k in precursor_keys]


def nominee_vector(record: NominationRecord, precursor_keys: list[str]) -> list[int]:
    return [int(bool(_get_award(record.precursors, k).nominee)) for k in precursor_keys]


def hamming_distance(a: list[int], b: list[int]) -> int:
    return sum(x != y for x, y in zip(a, b, strict=True))


def _nominee_display(record: NominationRecord) -> str:
    """Display name: person for acting, film title for others."""
    if record.nominee_name:
        return f"{record.nominee_name} ({record.film.title})"
    return record.film.title


def _short_display(record: NominationRecord) -> str:
    if record.nominee_name:
        return record.nominee_name
    return record.film.title


# ---------------------------------------------------------------------------
# Per-category analysis
# ---------------------------------------------------------------------------


def analyze_category(category: OscarCategory) -> str:
    """Run full analysis for one category, returning formatted text."""
    lines: list[str] = []
    cat_slug = category.slug
    dataset_path = DATASETS_DIR / cat_slug / SNAPSHOT_KEY / f"oscar_{cat_slug}_raw.json"

    if not dataset_path.exists():
        return f"=== {category.value} — DATASET NOT FOUND ({dataset_path}) ==="

    raw = NominationDataset.model_validate_json(dataset_path.read_text())
    ceremony_number = CEREMONY_YEAR - 1928
    nominees_2026 = [r for r in raw.records if r.ceremony == ceremony_number]
    historical = [r for r in raw.records if r.ceremony < ceremony_number]

    specs = get_precursor_specs(category)
    precursor_keys = [s.key for s in specs]
    # Derive short display names from keys (e.g. "sag_lead_actor" -> "SAG")
    short_names = {s.key: s.key.split("_")[0].upper() for s in specs}

    if not nominees_2026:
        return f"=== {category.value} — NO 2026 NOMINEES FOUND ==="

    lines.append(f"{'=' * 80}")
    lines.append(f"  {category.value}")
    lines.append(f"{'=' * 80}")
    lines.append("")

    # --- 1) Model probabilities ---
    lines.append("--- MODEL PROBABILITIES ---")
    model_preds: dict[str, dict[str, float]] = {}
    for mt in MODEL_TYPES:
        try:
            preds = load_predictions(category, mt, SNAPSHOT_KEY, MODELS_DIR)
            model_preds[mt.short_name] = preds
        except FileNotFoundError:
            pass

    # Compute ensemble average
    all_titles = set()
    for preds in model_preds.values():
        all_titles.update(preds.keys())

    ensemble: dict[str, float] = {}
    for title in all_titles:
        probs = [preds[title] for preds in model_preds.values() if title in preds]
        if probs:
            ensemble[title] = sum(probs) / len(probs)

    # Sort by ensemble prob descending
    sorted_titles = sorted(ensemble.keys(), key=lambda t: -ensemble[t])

    # Map film titles to nominee display names
    title_to_display: dict[str, str] = {}
    for rec in nominees_2026:
        title_to_display[rec.film.title] = _short_display(rec)

    model_names = list(model_preds.keys())
    header = f"{'Nominee':<30} {'Ensemble':>8}"
    for mn in model_names:
        header += f" {mn:>8}"
    lines.append(header)
    lines.append("-" * len(header))

    for title in sorted_titles:
        display = title_to_display.get(title, title)
        if len(display) > 28:
            display = display[:25] + "..."
        row = f"{display:<30} {ensemble[title]:>7.1%}"
        for mn in model_names:
            prob = model_preds[mn].get(title, 0.0)
            row += f" {prob:>7.1%}"
        lines.append(row)

    # Model agreement analysis
    lines.append("")
    lines.append("--- MODEL AGREEMENT ---")
    for title in sorted_titles:
        probs = [model_preds[mn].get(title, 0.0) for mn in model_names]
        spread = max(probs) - min(probs)
        display = title_to_display.get(title, title)
        if spread > 0.15:
            lines.append(f"  HIGH DISAGREEMENT: {display} — spread {spread:.1%} "
                         f"(range {min(probs):.1%}–{max(probs):.1%})")
        elif spread > 0.10:
            lines.append(f"  Moderate spread: {display} — {spread:.1%} "
                         f"({min(probs):.1%}–{max(probs):.1%})")

    # --- 2) Precursor breakdown ---
    lines.append("")
    lines.append("--- PRECURSOR BREAKDOWN ---")
    short_header = "  ".join(f"{short_names[k]:>6}" for k in precursor_keys)
    lines.append(f"{'Nominee':<30} {short_header}  {'Wins':>4}  {'Noms':>4}")
    lines.append("-" * 80)

    # Sort nominees by ensemble prob
    nominees_sorted = sorted(
        nominees_2026,
        key=lambda r: -ensemble.get(r.film.title, 0.0),
    )

    for rec in nominees_sorted:
        display = _short_display(rec)
        if len(display) > 28:
            display = display[:25] + "..."
        wvec = winner_vector(rec, precursor_keys)
        nvec = nominee_vector(rec, precursor_keys)
        detail = ""
        for key in precursor_keys:
            ar = _get_award(rec.precursors, key)
            if ar.winner:
                detail += f"{'W':>8}"
            elif ar.nominee:
                detail += f"{'N':>8}"
            else:
                detail += f"{'.':>8}"
        lines.append(f"{display:<30}{detail}  {sum(wvec):>4}  {sum(nvec):>4}")

    # --- 3) Historical base rates ---
    lines.append("")
    lines.append("--- BASE RATES: Oscar win rate by # precursor wins ---")
    max_wins = len(precursor_keys)
    counts: dict[int, int] = dict.fromkeys(range(max_wins + 1), 0)
    oscar_wins: dict[int, int] = dict.fromkeys(range(max_wins + 1), 0)

    for rec in historical:
        n_wins = sum(winner_vector(rec, precursor_keys))
        counts[n_wins] += 1
        if rec.category_winner:
            oscar_wins[n_wins] += 1

    lines.append(f"{'Precursor Wins':>15}  {'Nominees':>8}  {'Oscar Wins':>10}  {'Win Rate':>8}")
    lines.append("-" * 50)
    for n in range(max_wins + 1):
        total = counts[n]
        wins = oscar_wins[n]
        rate = f"{wins / total:.1%}" if total > 0 else "N/A"
        lines.append(f"{n:>15}  {total:>8}  {wins:>10}  {rate:>8}")

    # --- 4) Historical analogs for top 3 nominees ---
    lines.append("")
    lines.append("--- HISTORICAL ANALOGS (top 3 nominees, by winner profile) ---")

    for rec in nominees_sorted[:3]:
        target_wvec = winner_vector(rec, precursor_keys)
        display = _nominee_display(rec)
        lines.append(f"")
        lines.append(f"  {display} — winner profile: {target_wvec} ({sum(target_wvec)} wins)")

        scored: list[tuple[NominationRecord, int]] = []
        for hist in historical:
            dist = hamming_distance(target_wvec, winner_vector(hist, precursor_keys))
            scored.append((hist, dist))
        scored.sort(key=lambda x: (x[1], -(x[0].ceremony)))

        # Show exact matches (dist=0) count and top 5
        exact = [s for s in scored if s[1] == 0]
        exact_winners = sum(1 for s, _ in exact if s.category_winner)
        lines.append(f"  Exact profile matches: {len(exact)} nominees, "
                      f"{exact_winners} Oscar winners ({exact_winners}/{len(exact)})")

        lines.append(f"  Top 5 closest analogs:")
        for analog, dist in scored[:5]:
            yr = analog.ceremony + 1928
            name = _nominee_display(analog)
            oscar = "WIN" if analog.category_winner else "---"
            awvec = winner_vector(analog, precursor_keys)
            lines.append(f"    {yr} {name:<45} {awvec} dist={dist}  {oscar}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    output_dir = Path("storage/d20260313_reddit_predictions_post")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_output: list[str] = []
    all_output.append(f"2026 Oscar Predictions — Evidence Gathering")
    all_output.append(f"Generated from model snapshot: {SNAPSHOT_KEY}")
    all_output.append(f"Ceremony year: {CEREMONY_YEAR}")
    all_output.append("")

    for category in CATEGORIES:
        print(f"Analyzing {category.value}...")
        result = analyze_category(category)
        all_output.append(result)
        print(result[:200] + "...")
        print()

    output_path = output_dir / "all_categories_evidence.txt"
    output_path.write_text("\n".join(all_output))
    print(f"\nFull output written to {output_path}")


if __name__ == "__main__":
    main()
