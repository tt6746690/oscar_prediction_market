"""Show raw precursor award data for all 2026 Best Actor nominees.

Diagnostic script to verify what the model sees for each nominee. Useful for
understanding why model probabilities diverge from market prices — especially
for Timothée Chalamet (Marty Supreme) vs Michael B. Jordan (Sinners).

Usage:
    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.d20260310_best_actor_diagnostics.precursor_breakdown
"""

from pathlib import Path

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

DATASET_PATH = Path(
    "storage/d20260224_live_2026/datasets/actor_leading/2026-03-08_wga/oscar_actor_leading_raw.json"
)
MODELS_DIR = Path("storage/d20260224_live_2026/models")
SNAPSHOT_KEY = "2026-03-08_wga"
CEREMONY_YEAR = 2026
CATEGORY = OscarCategory.ACTOR_LEADING

# The precursor keys relevant to ACTOR_LEADING, in display order.
PRECURSOR_KEYS = [
    "sag_lead_actor",
    "bafta_lead_actor",
    "golden_globe_actor_drama",
    "golden_globe_actor_musical",
    "critics_choice_actor",
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


def _fmt_bool(val: bool | None) -> str:
    """Format a boolean as ✓/✗/— for display."""
    if val is None:
        return "—"
    return "✓" if val else "✗"


def _print_precursor_detail(record: NominationRecord) -> int:
    """Print precursor award detail for one nominee. Returns total wins."""
    total_wins = 0
    total_available = 0
    key_width = max(len(k) for k in PRECURSOR_KEYS)

    for key in PRECURSOR_KEYS:
        result: AwardResult | None = record.precursors.get(key)
        if result is None:
            print(f"  {key:<{key_width}}  (not in dataset)")
            continue
        nom_str = f"Nominee {_fmt_bool(result.nominee)}"
        win_str = f"Winner {_fmt_bool(result.winner)}"
        print(f"  {key:<{key_width}}  {nom_str}  {win_str}")
        if result.winner is not None:
            total_available += 1
            if result.winner:
                total_wins += 1

    print(f"  → Total precursor wins: {total_wins}/{total_available}")
    return total_wins


def _load_model_predictions() -> dict[ModelType, dict[str, float]]:
    """Load predictions from each model type, keyed by film title."""
    preds: dict[ModelType, dict[str, float]] = {}
    for mt in MODEL_TYPES:
        try:
            preds[mt] = load_predictions(CATEGORY, mt, SNAPSHOT_KEY, MODELS_DIR)
        except FileNotFoundError as exc:
            print(f"  WARNING: could not load {mt.short_name}: {exc}")
    return preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Load dataset
    raw = NominationDataset.model_validate_json(DATASET_PATH.read_text())
    records = raw.records_for_ceremony_year(CEREMONY_YEAR)

    if not records:
        print(f"No records found for ceremony year {CEREMONY_YEAR}.")
        return

    print(f"=== {CEREMONY_YEAR} Best Actor — Precursor Awards Breakdown ===")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Nominees: {len(records)}")
    print()

    # --- Section 1: Per-nominee precursor detail ---
    wins_by_title: dict[str, int] = {}
    for record in records:
        person = record.nominee_name or "(unknown)"
        title = record.film.title
        print(f"Nominee: {person} ({title})")
        wins = _print_precursor_detail(record)
        wins_by_title[title] = wins
        print()

    # --- Section 2: Precursor wins scorecard ---
    print("=" * 60)
    print("Precursor Wins Scorecard")
    print("=" * 60)
    name_width = max(len(f"{r.nominee_name or '?'} ({r.film.title})") for r in records)
    for record in sorted(records, key=lambda r: wins_by_title[r.film.title], reverse=True):
        person = record.nominee_name or "?"
        label = f"{person} ({record.film.title})"
        wins = wins_by_title[record.film.title]
        print(f"  {label:<{name_width}}  {wins} wins")
    print()

    # --- Section 3: Model predictions comparison ---
    print("=" * 60)
    print("Model Predictions Comparison")
    print("=" * 60)

    preds_by_model = _load_model_predictions()
    if not preds_by_model:
        print("  No model predictions available.")
        return

    # Header
    short_names = [mt.short_name for mt in preds_by_model]
    header_cols = "  ".join(f"{sn:>8}" for sn in short_names)
    name_col_width = max(len(f"{r.nominee_name or '?'} ({r.film.title})") for r in records)
    print(f"  {'Nominee':<{name_col_width}}  {header_cols}")
    print(f"  {'-' * name_col_width}  {'  '.join('-' * 8 for _ in short_names)}")

    # Rows — sort by average model probability descending
    def _avg_prob(record: NominationRecord) -> float:
        title = record.film.title
        probs = [p[title] for p in preds_by_model.values() if title in p]
        return sum(probs) / len(probs) if probs else 0.0

    for record in sorted(records, key=_avg_prob, reverse=True):
        person = record.nominee_name or "?"
        title = record.film.title
        label = f"{person} ({title})"
        prob_strs: list[str] = []
        for mt in preds_by_model:
            prob = preds_by_model[mt].get(title)
            if prob is not None:
                prob_strs.append(f"{prob:8.1%}")
            else:
                prob_strs.append(f"{'N/A':>8}")
        print(f"  {label:<{name_col_width}}  {'  '.join(prob_strs)}")

    print()


if __name__ == "__main__":
    main()
