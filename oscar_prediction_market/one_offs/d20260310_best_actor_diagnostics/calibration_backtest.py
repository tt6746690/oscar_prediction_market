"""Best Actor calibration and upset analysis.

Part A: Category-specific historical calibration — ECE/MCE, reliability, per-year accuracy.
Part B: Model vs precursor-favorite upset analysis — when the model disagrees with the
        consensus precursor leader, who turns out to be right?
"""

import csv
from collections import defaultdict
from pathlib import Path

from oscar_prediction_market.data.schema import NominationDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_SHORT_NAMES: list[str] = ["clogit", "lr", "gbt", "cal_sgbt"]

PREDICTIONS_TEMPLATE: str = (
    "storage/d20260224_live_2026/models/actor_leading/{short_name}"
    "/2026-03-08_wga/{short_name}_2026-03-08_wga/4_selected_cv/predictions.csv"
)

DATASET_PATH: str = (
    "storage/d20260224_live_2026/datasets/actor_leading/2026-03-08_wga/oscar_actor_leading_raw.json"
)

# Precursor keys relevant to Best Actor
ACTOR_PRECURSOR_KEYS: list[str] = [
    "sag_lead_actor",
    "bafta_lead_actor",
    "golden_globe_actor_drama",
    "golden_globe_actor_musical",
    "critics_choice_actor",
]

NUM_BINS: int = 10


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_cv_predictions(short_name: str) -> list[dict[str, str]]:
    """Load CV prediction rows for a single model, returning raw string dicts."""
    path = Path(PREDICTIONS_TEMPLATE.format(short_name=short_name))
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def parse_bool(val: str) -> bool:
    return val.strip() == "True"


# ---------------------------------------------------------------------------
# Part A — Calibration metrics
# ---------------------------------------------------------------------------


def compute_accuracy(rows: list[dict[str, str]]) -> tuple[int, int]:
    """Return (correct, total_ceremonies).

    A ceremony is correct when the rank-1 nominee is the actual winner.
    """
    # Group by ceremony
    by_ceremony: dict[int, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_ceremony[int(r["ceremony"])].append(r)

    correct = 0
    total = 0
    for _ceremony, nominees in sorted(by_ceremony.items()):
        total += 1
        for n in nominees:
            if int(n["rank"]) == 1 and parse_bool(n["is_actual_winner"]):
                correct += 1
                break
    return correct, total


def compute_mean_winner_prob(rows: list[dict[str, str]]) -> float:
    """Average model probability assigned to the actual winner across ceremonies."""
    probs: list[float] = []
    for r in rows:
        if parse_bool(r["is_actual_winner"]):
            probs.append(float(r["probability"]))
    return sum(probs) / len(probs) if probs else 0.0


def compute_brier_score(rows: list[dict[str, str]]) -> float:
    """Mean squared error of predicted probability vs binary outcome."""
    total = 0.0
    n = 0
    for r in rows:
        p = float(r["probability"])
        y = 1.0 if parse_bool(r["is_actual_winner"]) else 0.0
        total += (p - y) ** 2
        n += 1
    return total / n if n else 0.0


def compute_calibration(
    rows: list[dict[str, str]], num_bins: int = NUM_BINS
) -> tuple[float, float, list[tuple[float, float, float, int]]]:
    """ECE, MCE, and per-bin reliability data.

    Returns (ece, mce, bins) where each bin is
    (bin_center, avg_predicted, fraction_actually_won, count).
    """
    # Bin predictions into equal-width buckets [0, 0.1), [0.1, 0.2), ... [0.9, 1.0]
    bin_sums: list[float] = [0.0] * num_bins
    bin_true: list[float] = [0.0] * num_bins
    bin_counts: list[int] = [0] * num_bins

    for r in rows:
        p = float(r["probability"])
        y = 1.0 if parse_bool(r["is_actual_winner"]) else 0.0
        idx = min(int(p * num_bins), num_bins - 1)
        bin_sums[idx] += p
        bin_true[idx] += y
        bin_counts[idx] += 1

    ece = 0.0
    mce = 0.0
    total_n = sum(bin_counts)
    bin_data: list[tuple[float, float, float, int]] = []

    for i in range(num_bins):
        center = (i + 0.5) / num_bins
        if bin_counts[i] > 0:
            avg_pred = bin_sums[i] / bin_counts[i]
            frac_win = bin_true[i] / bin_counts[i]
            gap = abs(avg_pred - frac_win)
            ece += gap * bin_counts[i] / total_n
            mce = max(mce, gap)
            bin_data.append((center, avg_pred, frac_win, bin_counts[i]))
        else:
            bin_data.append((center, 0.0, 0.0, 0))

    return ece, mce, bin_data


def per_year_accuracy(rows: list[dict[str, str]]) -> dict[int, bool]:
    """Return {year: True/False} indicating if rank-1 was the actual winner."""
    by_year: dict[int, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_year[int(r["year"])].append(r)

    result: dict[int, bool] = {}
    for year, nominees in sorted(by_year.items()):
        for n in nominees:
            if int(n["rank"]) == 1:
                result[year] = parse_bool(n["is_actual_winner"])
                break
    return result


# ---------------------------------------------------------------------------
# Part B — Upset analysis helpers
# ---------------------------------------------------------------------------


def get_model_top_pick_per_year(
    rows: list[dict[str, str]],
) -> dict[int, tuple[str, str]]:
    """Return {year: (title, nominee_name_or_title)} for the model's rank-1 pick.

    Since the CSV has 'title' (film title) but not person name, we use title as
    the display name. We'll join with dataset records later for person names.
    """
    by_year: dict[int, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_year[int(r["year"])].append(r)

    result: dict[int, tuple[str, str]] = {}
    for year, nominees in sorted(by_year.items()):
        for n in nominees:
            if int(n["rank"]) == 1:
                result[year] = (n["film_id"], n["title"])
                break
    return result


def get_precursor_favorites(
    dataset: NominationDataset,
) -> dict[int, tuple[str, str, int]]:
    """Return {ceremony_year: (film_id, display_name, win_count)} for the precursor favorite.

    The precursor favorite is the nominee with the most precursor wins among
    ACTOR_PRECURSOR_KEYS. Ties broken by first encountered (arbitrary but stable).
    """
    # Group records by ceremony year
    by_year: dict[int, list] = defaultdict(list)
    for record in dataset.records:
        ceremony_year = record.ceremony + 1928
        by_year[ceremony_year].append(record)

    result: dict[int, tuple[str, str, int]] = {}
    for year, nominees in sorted(by_year.items()):
        best_film_id = ""
        best_name = ""
        best_wins = -1
        for nom in nominees:
            wins = 0
            for key in ACTOR_PRECURSOR_KEYS:
                award = nom.precursors.get(key)
                if award is not None and award.winner is True:
                    wins += 1
            if wins > best_wins:
                best_wins = wins
                best_film_id = nom.film.film_id
                best_name = nom.nominee_name or nom.film.title
        if best_wins >= 0:
            result[year] = (best_film_id, best_name, best_wins)
    return result


def get_actual_winners(
    dataset: NominationDataset,
) -> dict[int, tuple[str, str]]:
    """Return {ceremony_year: (film_id, display_name)} for actual Oscar winners."""
    result: dict[int, tuple[str, str]] = {}
    for record in dataset.records:
        if record.category_winner:
            ceremony_year = record.ceremony + 1928
            result[ceremony_year] = (
                record.film.film_id,
                record.nominee_name or record.film.title,
            )
    return result


def get_film_id_to_person(dataset: NominationDataset) -> dict[tuple[int, str], str]:
    """Return {(ceremony_year, film_id): person_name} mapping."""
    result: dict[tuple[int, str], str] = {}
    for record in dataset.records:
        ceremony_year = record.ceremony + 1928
        if record.nominee_name:
            result[(ceremony_year, record.film.film_id)] = record.nominee_name
    return result


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_calibration_analysis() -> None:
    """Print Part A — calibration metrics, reliability, and per-year accuracy."""
    print("=" * 60)
    print("  Best Actor — Calibration Analysis")
    print("=" * 60)

    all_year_acc: dict[str, dict[int, bool]] = {}
    all_years: set[int] = set()

    for short_name in MODEL_SHORT_NAMES:
        rows = load_cv_predictions(short_name)
        correct, total = compute_accuracy(rows)
        mean_wp = compute_mean_winner_prob(rows)
        brier = compute_brier_score(rows)
        ece, mce, bin_data = compute_calibration(rows)

        pct = 100.0 * correct / total if total else 0.0
        print(f"\nModel: {short_name}")
        print(f"  Accuracy:             {pct:.1f}% ({correct}/{total} correct)")
        print(f"  Mean winner prob:     {mean_wp:.3f}")
        print(f"  Brier score:          {brier:.4f}")
        print(f"  ECE ({NUM_BINS} bins):        {ece:.4f}")
        print(f"  MCE ({NUM_BINS} bins):        {mce:.4f}")

        print(f"\n  Reliability ({short_name}):")
        for center, avg_pred, frac_win, count in bin_data:
            lo = center - 0.5 / NUM_BINS
            hi = center + 0.5 / NUM_BINS
            if count > 0:
                print(
                    f"    Bin [{lo:.1f}, {hi:.1f}): "
                    f"avg_pred={avg_pred:.3f}, actual_win={100 * frac_win:.1f}%, n={count}"
                )
            else:
                print(f"    Bin [{lo:.1f}, {hi:.1f}): (empty)")

        ya = per_year_accuracy(rows)
        all_year_acc[short_name] = ya
        all_years.update(ya.keys())

    # Per-year accuracy table
    print("\n" + "-" * 60)
    print("  Per-Year Accuracy")
    print("-" * 60)
    header = "Year | " + " | ".join(f"{m:>8s}" for m in MODEL_SHORT_NAMES)
    print(header)
    print("-" * len(header))
    for year in sorted(all_years):
        cells: list[str] = []
        for m in MODEL_SHORT_NAMES:
            hit = all_year_acc.get(m, {}).get(year)
            if hit is None:
                cells.append(f"{'—':>8s}")
            elif hit:
                cells.append(f"{'✓':>8s}")
            else:
                cells.append(f"{'✗':>8s}")
        print(f"{year} | " + " | ".join(cells))


def print_upset_analysis() -> None:
    """Print Part B — model vs precursor favorite upset analysis."""
    print("\n" + "=" * 60)
    print("  Model vs Precursor Favorite — Upset Analysis")
    print("=" * 60)

    dataset = NominationDataset.model_validate_json(Path(DATASET_PATH).read_text())
    precursor_favs = get_precursor_favorites(dataset)
    actual_winners = get_actual_winners(dataset)
    film_person = get_film_id_to_person(dataset)

    # Use the first model (clogit) as the representative model for upset analysis,
    # then show a summary for all models.
    for short_name in MODEL_SHORT_NAMES:
        rows = load_cv_predictions(short_name)
        model_picks = get_model_top_pick_per_year(rows)

        print(f"\nModel: {short_name}")
        print(
            f"  {'Year':<6s} {'Model pick':<25s} {'Precursor fav':<25s} {'Winner':<25s} {'Result'}"
        )
        print("  " + "-" * 110)

        upset_correct = 0
        upset_total = 0

        for year in sorted(model_picks.keys()):
            model_film_id, model_title = model_picks[year]
            prec = precursor_favs.get(year)
            winner = actual_winners.get(year)

            if prec is None or winner is None:
                continue

            prec_film_id, prec_name, prec_win_count = prec
            winner_film_id, winner_name = winner

            # Resolve person name for model pick
            model_name = film_person.get((year, model_film_id), model_title)

            # Only show disagreements
            if model_film_id == prec_film_id:
                continue

            upset_total += 1
            model_right = model_film_id == winner_film_id
            prec_right = prec_film_id == winner_film_id
            if model_right:
                upset_correct += 1
                tag = "MODEL correct"
            elif prec_right:
                tag = "PRECURSOR correct"
            else:
                tag = "BOTH wrong"

            # Truncate long names for display
            def trunc(s: str, maxlen: int = 23) -> str:
                return s if len(s) <= maxlen else s[: maxlen - 2] + ".."

            print(
                f"  {year:<6d} {trunc(model_name):<25s} {trunc(prec_name):<25s} "
                f"{trunc(winner_name):<25s} {tag}"
            )

        if upset_total > 0:
            pct = 100.0 * upset_correct / upset_total
            print(f"\n  Model correct on upsets: {upset_correct}/{upset_total} ({pct:.1f}%)")
        else:
            print("\n  No upsets found (model always agreed with precursor favorite).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print_calibration_analysis()
    print_upset_analysis()


if __name__ == "__main__":
    main()
