"""Feature importance and per-nominee log-odds decomposition for Best Actor models.

Shows:
(a) Feature importance rankings for all 4 model types (clogit, lr, gbt, cal_sgbt).
(b) Per-nominee feature contributions (coefficient × feature_value) for interpretable
    models (clogit, lr) to explain WHY the model ranks nominees the way it does.
(c) Cross-model comparison of top-5 features — agreement/disagreement across models.
(d) Verification that the manual decomposition matches stored model predictions.

Usage:
    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\
d20260310_best_actor_diagnostics.feature_importance
"""

import csv
import math
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

STORAGE_BASE = Path("storage/d20260224_live_2026")
DATASET_PATH = STORAGE_BASE / "datasets/actor_leading/2026-03-08_wga/oscar_actor_leading_raw.json"
MODELS_DIR = STORAGE_BASE / "models"
ACTOR_MODELS_DIR = MODELS_DIR / "actor_leading"
SNAPSHOT_KEY = "2026-03-08_wga"
CEREMONY_YEAR = 2026

# Models to analyse: short_name → (display name, ModelType)
MODEL_INFO: dict[str, tuple[str, ModelType]] = {
    "clogit": ("Conditional Logit (clogit)", ModelType.CONDITIONAL_LOGIT),
    "lr": ("Logistic Regression (lr)", ModelType.LOGISTIC_REGRESSION),
    "gbt": ("Gradient Boosting (gbt)", ModelType.GRADIENT_BOOSTING),
    "cal_sgbt": ("Calibrated Softmax GBT (cal_sgbt)", ModelType.CALIBRATED_SOFTMAX_GBT),
}

# Organisation prefixes used to collapse sub-awards into parent orgs
# (e.g. golden_globe_actor_drama → golden_globe).  Longest prefixes first
# so "critics_choice" matches before a hypothetical "critics".
ORG_PREFIXES: list[str] = [
    "golden_globe",
    "critics_choice",
    "sag",
    "bafta",
    "pga",
    "dga",
    "wga",
    "asc",
    "annie",
]

# Golden Globe "any" composite keys — union of drama + musical sub-awards
GOLDEN_GLOBE_ACTOR_KEYS: list[str] = [
    "golden_globe_actor_drama",
    "golden_globe_actor_musical",
]
GOLDEN_GLOBE_ANY_KEYS: list[str] = [
    "golden_globe_drama",
    "golden_globe_musical",
    "golden_globe_actor_drama",
    "golden_globe_actor_musical",
    "golden_globe_actress_drama",
    "golden_globe_actress_musical",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class FeatureImportanceRow:
    """One row from feature_importance.csv."""

    __slots__ = ("feature", "coefficient", "abs_coefficient", "importance")

    def __init__(
        self,
        feature: str,
        coefficient: float | None,
        abs_coefficient: float | None,
        importance: float,
    ) -> None:
        self.feature = feature
        self.coefficient = coefficient
        self.abs_coefficient = abs_coefficient
        self.importance = importance


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _final_predict_dir(short_name: str) -> Path:
    """Path to a model's ``5_final_predict`` directory."""
    return (
        ACTOR_MODELS_DIR
        / short_name
        / SNAPSHOT_KEY
        / f"{short_name}_{SNAPSHOT_KEY}"
        / "5_final_predict"
    )


def load_feature_importance(short_name: str) -> list[FeatureImportanceRow]:
    """Load ``feature_importance.csv`` for a given model.

    CSV columns: feature, coefficient (optional), abs_coefficient (optional), importance.
    """
    path = _final_predict_dir(short_name) / "feature_importance.csv"
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return []
    rows: list[FeatureImportanceRow] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                FeatureImportanceRow(
                    feature=row["feature"],
                    coefficient=(float(row["coefficient"]) if row.get("coefficient") else None),
                    abs_coefficient=(
                        float(row["abs_coefficient"]) if row.get("abs_coefficient") else None
                    ),
                    importance=float(row["importance"]),
                )
            )
    return rows


def load_raw_dataset() -> NominationDataset:
    """Load the raw Oscar actor_leading dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    return NominationDataset.model_validate_json(DATASET_PATH.read_text())


# ---------------------------------------------------------------------------
# Feature value extraction from raw NominationRecord
# ---------------------------------------------------------------------------


def _precursor_key_to_org(key: str) -> str:
    """Map a precursor key to its parent organisation.

    e.g. ``"sag_lead_actor"`` → ``"sag"``,
    ``"golden_globe_actor_drama"`` → ``"golden_globe"``.
    """
    for prefix in ORG_PREFIXES:
        if key.startswith(prefix):
            return prefix
    return key


def _award_field(record: NominationRecord, key: str, field: str) -> bool:
    """Look up ``record.precursors[key].<field>`` safely, defaulting to False."""
    award: AwardResult = record.precursors.get(key, AwardResult())
    val = getattr(award, field, None)
    return bool(val) if val is not None else False


def _any_precursor_field(
    record: NominationRecord,
    keys: list[str],
    field: str,
) -> bool:
    """Return True if *any* of ``keys`` has ``field`` == True."""
    return any(_award_field(record, k, field) for k in keys)


def extract_feature_value(record: NominationRecord, feature_name: str) -> float | None:
    """Extract a single feature value from a raw ``NominationRecord``.

    Returns ``None`` for features we cannot map (caller should skip / note them).
    Otherwise returns a numeric value (int cast to float for booleans).

    Mapping rules (mirror feature_engineering):
      - ``{key}_winner``                → precursors[key].winner  (bool → int)
      - ``{key}_nominee``               → precursors[key].nominee (bool → int)
      - ``precursor_wins_count``        → count of keys where winner is True
      - ``precursor_nominations_count`` → count of keys where nominee is True
      - ``person_prev_noms_same_category`` etc. → from PersonData
      - ``person_is_overdue``           → noms_same >= 2 and wins_same == 0
      - ``golden_globe_any_winner/nominee`` → union of drama + musical sub-awards
      - ``golden_globe_actor_any_winner/nominee`` → union of actor drama + musical
    """
    # --- precursor winner / nominee booleans ---
    if feature_name.endswith("_winner"):
        precursor_key = feature_name[: -len("_winner")]
        # Check it's a known precursor key (exists in record or at least looks valid)
        return float(_award_field(record, precursor_key, "winner"))
    if feature_name.endswith("_nominee"):
        precursor_key = feature_name[: -len("_nominee")]
        return float(_award_field(record, precursor_key, "nominee"))

    # --- aggregate counts ---
    if feature_name == "precursor_wins_count":
        return float(sum(1 for a in record.precursors.values() if a.winner))

    if feature_name == "precursor_nominations_count":
        return float(sum(1 for a in record.precursors.values() if a.nominee))

    # --- person career features ---
    person = record.person
    person_features: dict[str, float] = {}
    if person is not None:
        person_features = {
            "person_prev_noms_same_category": float(person.prev_noms_same_category),
            "person_prev_wins_same_category": float(person.prev_wins_same_category),
            "person_prev_noms_any_category": float(person.prev_noms_any_category),
            "person_prev_wins_any_category": float(person.prev_wins_any_category),
            "person_is_overdue": float(
                person.prev_noms_same_category >= 2 and person.prev_wins_same_category == 0
            ),
        }
        if person.tmdb_popularity is not None:
            person_features["person_tmdb_popularity"] = person.tmdb_popularity
    if feature_name in person_features:
        return person_features[feature_name]

    # --- golden globe composite booleans ---
    if feature_name == "golden_globe_any_winner":
        return float(_any_precursor_field(record, GOLDEN_GLOBE_ANY_KEYS, "winner"))
    if feature_name == "golden_globe_any_nominee":
        return float(_any_precursor_field(record, GOLDEN_GLOBE_ANY_KEYS, "nominee"))
    if feature_name in ("golden_globe_actor_any_winner",):
        return float(_any_precursor_field(record, GOLDEN_GLOBE_ACTOR_KEYS, "winner"))
    if feature_name in ("golden_globe_actor_any_nominee",):
        return float(_any_precursor_field(record, GOLDEN_GLOBE_ACTOR_KEYS, "nominee"))

    # --- unmapped ---
    return None


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def softmax(log_odds: list[float]) -> list[float]:
    """Softmax: ``P(i) = exp(logodds_i) / sum_j exp(logodds_j)``.

    Numerically stable — subtracts max before exp to prevent overflow.
    """
    max_val = max(log_odds)
    exps = [math.exp(x - max_val) for x in log_odds]
    total = sum(exps)
    return [e / total for e in exps]


# ---------------------------------------------------------------------------
# Display: feature importances per model
# ---------------------------------------------------------------------------


def display_feature_importance(short_name: str, rows: list[FeatureImportanceRow]) -> None:
    """Print feature importance table for a single model."""
    display_name = MODEL_INFO[short_name][0]
    print(f"\n--- {display_name} ---")
    if not rows:
        print("  (no features)")
        return

    has_coeff = rows[0].coefficient is not None
    for i, row in enumerate(rows, 1):
        if has_coeff:
            sign = "+" if (row.coefficient is not None and row.coefficient >= 0) else ""
            print(f"  {i}. {row.feature:45s} coeff = {sign}{row.coefficient:.6f}")
        else:
            print(f"  {i}. {row.feature:45s} importance = {row.importance:.6f}")


# ---------------------------------------------------------------------------
# Display: per-nominee log-odds decomposition (linear models only)
# ---------------------------------------------------------------------------


def display_nominee_decomposition(
    short_name: str,
    rows: list[FeatureImportanceRow],
    nominees: list[NominationRecord],
) -> None:
    """Print per-nominee feature contribution decomposition for linear models.

    For each nominee: feature_value × coefficient = contribution.
    Then converts total log-odds to probability via softmax across nominees.
    """
    display_name = MODEL_INFO[short_name][0]
    has_coeff = rows[0].coefficient is not None if rows else False
    if not has_coeff:
        print(f"\n--- Per-Nominee Decomposition ({display_name}) ---")
        print("  (Not applicable — model has no coefficients)")
        return

    features_with_coeff: list[tuple[str, float]] = [
        (r.feature, r.coefficient) for r in rows if r.coefficient is not None
    ]

    print(f"\n--- Per-Nominee Decomposition ({display_name}) ---")

    # Collect per-nominee data
    skipped_features: set[str] = set()
    nominee_data: list[tuple[str, dict[str, tuple[float, float]], float]] = []
    #                   name   {feat: (value, contribution)}           log-odds

    for record in nominees:
        name = record.nominee_name or record.film.title
        contribs: dict[str, tuple[float, float]] = {}
        total_log_odds = 0.0
        for feat_name, coeff in features_with_coeff:
            value = extract_feature_value(record, feat_name)
            if value is None:
                skipped_features.add(feat_name)
                continue
            contribution = value * coeff
            contribs[feat_name] = (value, contribution)
            total_log_odds += contribution
        nominee_data.append((name, contribs, total_log_odds))

    if skipped_features:
        print(f"  (skipped unmapped features: {', '.join(sorted(skipped_features))})")

    # Softmax to get probabilities
    all_log_odds = [lo for _, _, lo in nominee_data]
    probs = softmax(all_log_odds)

    # Sort by probability descending
    indexed = sorted(
        zip(nominee_data, probs, strict=True),
        key=lambda x: x[1],
        reverse=True,
    )

    # Step-by-step table per nominee
    for (name, contribs, total_lo), prob in indexed:
        print(f"\n  {name}  (log-odds = {total_lo:+.4f}, prob = {prob:.1%})")
        # Sort contributions by absolute value descending
        sorted_contribs = sorted(
            contribs.items(),
            key=lambda x: abs(x[1][1]),
            reverse=True,
        )
        for feat_name, (value, contribution) in sorted_contribs:
            if abs(contribution) < 1e-8:
                continue
            coeff = contribution / value if value != 0 else 0.0
            print(
                f"    {feat_name:45s}  val={value:5.1f}  × coeff={coeff:+.6f}"
                f"  = {contribution:+.6f}"
            )

    # Pairwise comparison: Chalamet vs MBJ (if both present)
    chalamet = next(((n, c, lo) for n, c, lo in nominee_data if "Chalamet" in n), None)
    mbj = next(((n, c, lo) for n, c, lo in nominee_data if "Jordan" in n), None)

    if chalamet and mbj:
        print(f"\n  Pairwise: {chalamet[0]} vs {mbj[0]}")
        lo_diff = chalamet[2] - mbj[2]
        print(f"    Log-odds difference: {lo_diff:+.4f}")
        # Per-feature diff
        all_feats = sorted(
            set(chalamet[1].keys()) | set(mbj[1].keys()),
            key=lambda f: abs(chalamet[1].get(f, (0, 0))[1] - mbj[1].get(f, (0, 0))[1]),
            reverse=True,
        )
        for feat in all_feats:
            c_val, c_con = chalamet[1].get(feat, (0.0, 0.0))
            m_val, m_con = mbj[1].get(feat, (0.0, 0.0))
            diff = c_con - m_con
            if abs(diff) < 1e-8:
                continue
            short_feat = feat.replace("_winner", " (W)").replace("_nominee", " (N)")
            print(
                f"      {short_feat:45s}  {chalamet[0][:5]}={c_val:.0f} vs"
                f" {mbj[0][:5]}={m_val:.0f}  → Δ = {diff:+.6f}"
            )


# ---------------------------------------------------------------------------
# Display: cross-model comparison (top-5)
# ---------------------------------------------------------------------------


def display_cross_model_comparison(
    all_importances: dict[str, list[FeatureImportanceRow]],
) -> None:
    """Show top-5 features from each model and note agreement / disagreement."""
    n_top = 5
    print("\n" + "=" * 80)
    print("=== Cross-Model Feature Comparison (top-5) ===")
    print("=" * 80)

    # Per-model top-N
    for short_name, (display_name, _) in MODEL_INFO.items():
        rows = all_importances.get(short_name, [])
        top = rows[:n_top]
        label = display_name.split("(")[0].strip()
        print(f"\n  {label}:")
        for i, r in enumerate(top, 1):
            extra = f"  coeff={r.coefficient:+.6f}" if r.coefficient is not None else ""
            print(f"    {i}. {r.feature:40s}  imp={r.importance:.6f}{extra}")

    # Agreement matrix: how many models rank each feature in their top-N
    print(f"\n  Feature frequency in top-{n_top} across {len(MODEL_INFO)} models:")
    feature_counts: dict[str, list[str]] = {}
    for short_name in MODEL_INFO:
        rows = all_importances.get(short_name, [])
        for r in rows[:n_top]:
            feature_counts.setdefault(r.feature, []).append(short_name)

    for feat, models in sorted(feature_counts.items(), key=lambda x: -len(x[1])):
        count = len(models)
        bar = "█" * count + "░" * (len(MODEL_INFO) - count)
        model_list = ", ".join(models)
        print(f"    {feat:40s} {bar} ({count}/{len(MODEL_INFO)})  [{model_list}]")

    # Disagreements: features in only 1 model's top-N
    unique = {f: ms for f, ms in feature_counts.items() if len(ms) == 1}
    if unique:
        print(f"\n  Features unique to a single model's top-{n_top}:")
        for feat, ms in unique.items():
            print(f"    {feat:40s} → only in {ms[0]}")


# ---------------------------------------------------------------------------
# Display: model predictions with verification
# ---------------------------------------------------------------------------


def display_model_predictions(
    raw: NominationDataset,
    nominees: list[NominationRecord],
) -> None:
    """Load predictions via ``load_predictions`` and show them side-by-side.

    Also verifies that the clogit/lr manual decomposition probabilities roughly
    match the stored predictions (sanity check).
    """
    title_to_person = raw.build_title_to_person_map(CEREMONY_YEAR)

    # Header
    print(f"\n  {'Nominee':25s}", end="")
    for short_name, (_display_name, _) in MODEL_INFO.items():
        label = short_name
        print(f"  {label:>12s}", end="")
    print()
    print("  " + "-" * (25 + 14 * len(MODEL_INFO)))

    # Load predictions from each model
    model_preds_by_person: dict[str, dict[str, float]] = {}
    for short_name, (_, model_type) in MODEL_INFO.items():
        try:
            preds_by_title = load_predictions(
                OscarCategory.ACTOR_LEADING,
                model_type,
                SNAPSHOT_KEY,
                MODELS_DIR,
            )
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            continue

        preds_by_person: dict[str, float] = {}
        for title, prob in preds_by_title.items():
            person = title_to_person.get(title, title)
            preds_by_person[person] = prob
        model_preds_by_person[short_name] = preds_by_person

    # Collect all person names, sorted by average probability
    all_names = [r.nominee_name or r.film.title for r in nominees]
    avg_probs = {
        name: (
            sum(model_preds_by_person.get(m, {}).get(name, 0.0) for m in MODEL_INFO)
            / len(MODEL_INFO)
        )
        for name in all_names
    }
    all_names.sort(key=lambda n: avg_probs.get(n, 0.0), reverse=True)

    for name in all_names:
        print(f"  {name:25s}", end="")
        for short_name in MODEL_INFO:
            prob = model_preds_by_person.get(short_name, {}).get(name, 0.0)
            print(f"  {prob:>11.1%}", end="")
        print()


def verify_decomposition(
    short_name: str,
    rows: list[FeatureImportanceRow],
    nominees: list[NominationRecord],
    raw: NominationDataset,
) -> None:
    """Compare manual log-odds → softmax probabilities against stored predictions.

    Prints per-nominee comparison and highlights discrepancies > 1 pp.
    """
    _, model_type = MODEL_INFO[short_name]
    display_name = MODEL_INFO[short_name][0]

    has_coeff = rows[0].coefficient is not None if rows else False
    if not has_coeff:
        return

    # Manual decomposition
    features_with_coeff: list[tuple[str, float]] = [
        (r.feature, r.coefficient) for r in rows if r.coefficient is not None
    ]
    manual_logodds: dict[str, float] = {}
    for record in nominees:
        name = record.nominee_name or record.film.title
        lo = 0.0
        for feat_name, coeff in features_with_coeff:
            value = extract_feature_value(record, feat_name)
            if value is not None:
                lo += value * coeff
        manual_logodds[name] = lo

    names = list(manual_logodds.keys())
    manual_probs_list = softmax([manual_logodds[n] for n in names])
    manual_probs = dict(zip(names, manual_probs_list, strict=True))

    # Stored predictions
    title_to_person = raw.build_title_to_person_map(CEREMONY_YEAR)
    try:
        preds_by_title = load_predictions(
            OscarCategory.ACTOR_LEADING, model_type, SNAPSHOT_KEY, MODELS_DIR
        )
    except FileNotFoundError:
        print(f"\n  Cannot verify {display_name} — predictions file missing")
        return

    stored_probs: dict[str, float] = {}
    for title, prob in preds_by_title.items():
        person = title_to_person.get(title, title)
        stored_probs[person] = prob

    print(f"\n--- Verification: {display_name} ---")
    print(f"  {'Nominee':25s}  {'manual':>8s}  {'stored':>8s}  {'diff':>8s}")
    print("  " + "-" * 55)
    max_diff = 0.0
    for name in names:
        mp = manual_probs.get(name, 0.0)
        sp = stored_probs.get(name, 0.0)
        diff = mp - sp
        max_diff = max(max_diff, abs(diff))
        flag = " ⚠" if abs(diff) > 0.01 else ""
        print(f"  {name:25s}  {mp:>7.1%}  {sp:>7.1%}  {diff:>+7.2%}{flag}")
    if max_diff < 0.01:
        print("  ✓ Decomposition matches stored predictions (all diffs < 1 pp)")
    else:
        print(f"  ⚠ Max discrepancy: {max_diff:.2%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 80)
    print("=== Feature Importance — Best Actor (2026-03-08_wga snapshot) ===")
    print("=" * 80)

    # 1. Load feature importances for all models
    all_importances: dict[str, list[FeatureImportanceRow]] = {}
    for short_name in MODEL_INFO:
        rows = load_feature_importance(short_name)
        all_importances[short_name] = rows
        display_feature_importance(short_name, rows)

    # 2. Load raw dataset and 2026 nominees
    raw = load_raw_dataset()
    nominees = raw.records_for_ceremony_year(CEREMONY_YEAR)
    print(f"\n2026 Best Actor nominees ({len(nominees)}):")
    for record in nominees:
        name = record.nominee_name or record.film.title
        precursor_wins = sum(1 for a in record.precursors.values() if a.winner)
        precursor_noms = sum(1 for a in record.precursors.values() if a.nominee)
        person_info = ""
        if record.person:
            p = record.person
            person_info = (
                f"  [noms_same={p.prev_noms_same_category}, wins_same={p.prev_wins_same_category}]"
            )
        print(
            f"  - {name} ({record.film.title})"
            f"  precursor W/N={precursor_wins}/{precursor_noms}{person_info}"
        )

    # 3. Per-nominee log-odds decomposition for interpretable models (clogit, lr)
    print("\n" + "=" * 80)
    print("=== Per-Nominee Log-Odds Decomposition ===")
    print("=" * 80)

    for short_name in ["clogit", "lr"]:
        rows = all_importances.get(short_name, [])
        if rows:
            display_nominee_decomposition(short_name, rows, nominees)

    # 4. Cross-model comparison (top-5)
    display_cross_model_comparison(all_importances)

    # 5. Model predictions side-by-side
    print("\n" + "=" * 80)
    print("=== Model Predictions (via load_predictions) ===")
    print("=" * 80)
    display_model_predictions(raw, nominees)

    # 6. Verify decomposition matches stored predictions
    print("\n" + "=" * 80)
    print("=== Decomposition Verification ===")
    print("=" * 80)
    for short_name in ["clogit", "lr"]:
        rows = all_importances.get(short_name, [])
        if rows:
            verify_decomposition(short_name, rows, nominees, raw)


if __name__ == "__main__":
    main()
