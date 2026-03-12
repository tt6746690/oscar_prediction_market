"""Counterfactual sensitivity analysis for Best Actor 2026 (clogit model).

What would happen to model predictions if we flip individual precursor awards
for MBJ and Chalamet?

Since clogit is linear in log-odds, we can compute counterfactuals analytically:
    log_odds(nominee) = sum(coeff_k * feature_k)
    P(nominee) = softmax(log_odds) across the nominee slate

Flipping a binary feature changes a nominee's log-odds by exactly ±coeff_k,
and softmax redistributes probabilities across the slate.
"""

import csv
import math
from pathlib import Path

from oscar_prediction_market.data.schema import (
    AwardResult,
    NominationDataset,
    NominationRecord,
)

DATASET_PATH = Path(
    "storage/d20260224_live_2026/datasets/actor_leading/2026-03-08_wga/oscar_actor_leading_raw.json"
)
FEATURE_IMPORTANCE_PATH = Path(
    "storage/d20260224_live_2026/models/actor_leading/clogit/"
    "2026-03-08_wga/clogit_2026-03-08_wga/5_final_predict/feature_importance.csv"
)

# The 5 major acting precursors. The clogit model may only use a subset,
# but we enumerate all for "sweep" scenarios.
ACTING_PRECURSOR_KEYS = [
    "sag_lead_actor",
    "bafta_lead_actor",
    "critics_choice_actor",
    "golden_globe_actor_drama",
    "golden_globe_actor_musical",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_coefficients(path: Path) -> dict[str, float]:
    """Load {feature_name: coefficient} from feature_importance.csv."""
    coefficients: dict[str, float] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            coefficients[row["feature"]] = float(row["coefficient"])
    return coefficients


def get_feature_value(record: NominationRecord, feature_name: str) -> float:
    """Extract a numeric feature value from a NominationRecord.

    Convention:
      - ``{precursor_key}_winner`` → 1.0 if winner, else 0.0
      - ``{precursor_key}_nominee`` → 1.0 if nominee, else 0.0
      - ``precursor_wins_count`` / ``precursor_nominations_count`` → aggregates
      - ``person_prev_noms_*`` / ``person_prev_wins_*`` → person career stats
    """
    if feature_name.endswith("_winner"):
        key = feature_name[: -len("_winner")]
        award = record.precursors.get(key, AwardResult())
        return 1.0 if award.winner else 0.0
    elif feature_name.endswith("_nominee"):
        key = feature_name[: -len("_nominee")]
        award = record.precursors.get(key, AwardResult())
        return 1.0 if award.nominee else 0.0

    if feature_name == "precursor_wins_count":
        return float(sum(1 for a in record.precursors.values() if a.winner))
    if feature_name == "precursor_nominations_count":
        return float(sum(1 for a in record.precursors.values() if a.nominee))

    if record.person:
        person_fields: dict[str, float] = {
            "person_prev_noms_same_category": float(record.person.prev_noms_same_category),
            "person_prev_wins_same_category": float(record.person.prev_wins_same_category),
            "person_prev_noms_any_category": float(record.person.prev_noms_any_category),
            "person_prev_wins_any_category": float(record.person.prev_wins_any_category),
        }
        if feature_name in person_fields:
            return person_fields[feature_name]

    return 0.0  # Unknown feature → 0


def compute_log_odds(
    records: list[NominationRecord],
    coefficients: dict[str, float],
) -> dict[str, float]:
    """Compute log-odds for each nominee: sum(coeff * feature_value)."""
    log_odds: dict[str, float] = {}
    for record in records:
        name = record.nominee_name or record.film.title
        lo = sum(coeff * get_feature_value(record, feat) for feat, coeff in coefficients.items())
        log_odds[name] = lo
    return log_odds


def softmax(log_odds: dict[str, float]) -> dict[str, float]:
    """Softmax over a dict of {name: log_odds} → {name: probability}."""
    max_lo = max(log_odds.values())
    exps = {k: math.exp(v - max_lo) for k, v in log_odds.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def format_pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def format_delta(old: float, new: float) -> str:
    delta_pp = (new - old) * 100
    sign = "+" if delta_pp >= 0 else ""
    return f"{sign}{delta_pp:.1f}pp"


# ---------------------------------------------------------------------------
# Counterfactual engine
# ---------------------------------------------------------------------------


class FeatureOverride:
    """One feature override: set a specific feature to a new value for a nominee."""

    def __init__(self, nominee_name: str, feature_name: str, new_value: float) -> None:
        self.nominee_name = nominee_name
        self.feature_name = feature_name
        self.new_value = new_value


def compute_counterfactual(
    records: list[NominationRecord],
    coefficients: dict[str, float],
    overrides: list[FeatureOverride],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute counterfactual log-odds and probabilities.

    Returns (log_odds, probs) with overrides applied.
    """
    log_odds: dict[str, float] = {}
    for record in records:
        name = record.nominee_name or record.film.title
        lo = 0.0
        for feat, coeff in coefficients.items():
            # Check if there's an override for this nominee + feature
            override_val = None
            for ov in overrides:
                if ov.nominee_name == name and ov.feature_name == feat:
                    override_val = ov.new_value
                    break
            val = override_val if override_val is not None else get_feature_value(record, feat)
            lo += coeff * val
        log_odds[name] = lo
    probs = softmax(log_odds)
    return log_odds, probs


def describe_overrides(
    records: list[NominationRecord],
    overrides: list[FeatureOverride],
) -> list[str]:
    """Return human-readable description of each override."""
    lines: list[str] = []
    record_by_name = {(r.nominee_name or r.film.title): r for r in records}
    for ov in overrides:
        rec = record_by_name.get(ov.nominee_name)
        if rec:
            old_val = get_feature_value(rec, ov.feature_name)
            lines.append(
                f"  Changed: {ov.feature_name}: {int(old_val)}→{int(ov.new_value)} "
                f"for {ov.nominee_name}"
            )
    return lines


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------


def build_scenarios(
    coefficients: dict[str, float],
) -> list[tuple[str, list[FeatureOverride]]]:
    """Build all counterfactual scenarios to test.

    Only includes overrides for features that the model actually uses
    (i.e., present in coefficients). Features the model ignores don't
    change predictions.
    """
    model_features = set(coefficients.keys())

    def _winner_feat(precursor_key: str) -> str:
        return f"{precursor_key}_winner"

    def _is_model_feature(precursor_key: str) -> bool:
        return _winner_feat(precursor_key) in model_features

    scenarios: list[tuple[str, list[FeatureOverride]]] = []

    # --- MBJ gains ---
    if _is_model_feature("bafta_lead_actor"):
        scenarios.append(
            (
                "What if MBJ had won BAFTA?",
                [FeatureOverride("Michael B. Jordan", "bafta_lead_actor_winner", 1.0)],
            )
        )

    if _is_model_feature("critics_choice_actor"):
        scenarios.append(
            (
                "What if MBJ had won Critics Choice?",
                [FeatureOverride("Michael B. Jordan", "critics_choice_actor_winner", 1.0)],
            )
        )

    if _is_model_feature("bafta_lead_actor") and _is_model_feature("critics_choice_actor"):
        scenarios.append(
            (
                "What if MBJ had won BAFTA + CC?",
                [
                    FeatureOverride("Michael B. Jordan", "bafta_lead_actor_winner", 1.0),
                    FeatureOverride("Michael B. Jordan", "critics_choice_actor_winner", 1.0),
                ],
            )
        )

    # --- Chalamet loses ---
    # Only include "NOT won" scenarios for precursors Chalamet actually won,
    # otherwise the override is a no-op (already 0→0).
    if _is_model_feature("sag_lead_actor"):
        scenarios.append(
            (
                "What if Chalamet had NOT won SAG?",
                [FeatureOverride("Timothée Chalamet", "sag_lead_actor_winner", 0.0)],
            )
        )

    if _is_model_feature("bafta_lead_actor"):
        scenarios.append(
            (
                "What if Chalamet had NOT won BAFTA?",
                [FeatureOverride("Timothée Chalamet", "bafta_lead_actor_winner", 0.0)],
            )
        )

    if _is_model_feature("critics_choice_actor"):
        scenarios.append(
            (
                "What if Chalamet had NOT won CC?",
                [FeatureOverride("Timothée Chalamet", "critics_choice_actor_winner", 0.0)],
            )
        )

    # --- Sweep scenarios ---
    # "All 5 precursors" in domain terms, but only model features matter for predictions
    model_precursor_keys = [k for k in ACTING_PRECURSOR_KEYS if _is_model_feature(k)]
    non_model_precursors = [k for k in ACTING_PRECURSOR_KEYS if not _is_model_feature(k)]

    scenarios.append(
        (
            f"What if MBJ had swept all {len(ACTING_PRECURSOR_KEYS)} precursors?"
            + (f" (model only uses {len(model_precursor_keys)})" if non_model_precursors else ""),
            [
                FeatureOverride("Michael B. Jordan", _winner_feat(k), 1.0)
                for k in model_precursor_keys
            ],
        )
    )

    scenarios.append(
        (
            f"What if Chalamet had swept all {len(ACTING_PRECURSOR_KEYS)} precursors?"
            + (f" (model only uses {len(model_precursor_keys)})" if non_model_precursors else ""),
            [
                FeatureOverride("Timothée Chalamet", _winner_feat(k), 1.0)
                for k in model_precursor_keys
            ],
        )
    )

    return scenarios


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_baseline(
    records: list[NominationRecord],
    coefficients: dict[str, float],
    log_odds: dict[str, float],
    probs: dict[str, float],
) -> None:
    print("Baseline (actual 2026 precursors):")
    print(f"  {'Nominee':<25s} {'log_odds':>10s}  {'probability':>11s}  features")
    for name in sorted(probs, key=lambda n: -probs[n]):
        rec = next(r for r in records if (r.nominee_name or r.film.title) == name)
        feat_strs = []
        for feat in sorted(coefficients):
            val = get_feature_value(rec, feat)
            if val > 0:
                feat_strs.append(f"{feat}={int(val)}")
        feat_display = ", ".join(feat_strs) if feat_strs else "(none)"
        print(
            f"  {name:<25s} {log_odds[name]:>10.4f}  {format_pct(probs[name]):>11s}  {feat_display}"
        )
    print()


def print_scenario(
    scenario_name: str,
    records: list[NominationRecord],
    overrides: list[FeatureOverride],
    baseline_probs: dict[str, float],
    cf_log_odds: dict[str, float],
    cf_probs: dict[str, float],
) -> None:
    print(f'--- Scenario: "{scenario_name}" ---')
    for line in describe_overrides(records, overrides):
        print(line)
    # Show nominees sorted by counterfactual probability, but only the movers
    affected_names = {ov.nominee_name for ov in overrides}
    # Always show affected nominees + top 2
    top_names = sorted(cf_probs, key=lambda n: -cf_probs[n])[:2]
    show_names = list(dict.fromkeys(top_names + sorted(affected_names)))

    for name in sorted(show_names, key=lambda n: -cf_probs[n]):
        bp = baseline_probs[name]
        cp = cf_probs[name]
        delta = format_delta(bp, cp)
        note = ""
        if name not in affected_names:
            note = "  [features unchanged, softmax rebalance]"
        print(f"  {name:<25s} {format_pct(bp):>6s} → {format_pct(cp):>6s} ({delta}){note}")
    print()


def print_summary(
    baseline_probs: dict[str, float],
    scenarios: list[tuple[str, list[FeatureOverride]]],
    records: list[NominationRecord],
    coefficients: dict[str, float],
) -> None:
    """Print summary: what would it take for MBJ to be the model favorite?"""
    print("=== Summary: What would it take for MBJ to be the model favorite? ===")
    print()

    mbj = "Michael B. Jordan"
    chalamet = "Timothée Chalamet"
    bp_mbj = baseline_probs[mbj]
    bp_chal = baseline_probs[chalamet]
    print(f"  Baseline: {mbj} = {format_pct(bp_mbj)}, {chalamet} = {format_pct(bp_chal)}")
    print()

    for name, overrides in scenarios:
        _, cf_probs = compute_counterfactual(records, coefficients, overrides)
        # Only show MBJ-relevant scenarios
        affected = {ov.nominee_name for ov in overrides}
        if mbj in affected:
            is_favorite = cf_probs[mbj] == max(cf_probs.values())
            marker = " ★ FAVORITE" if is_favorite else ""
            print(f"  {name:<65s} → MBJ {format_pct(cf_probs[mbj]):>6s}{marker}")

    print()
    print("  Coefficient magnitudes (what each precursor win is worth in log-odds):")
    for feat in sorted(coefficients, key=lambda f: -abs(coefficients[f])):
        print(f"    {feat}: {coefficients[feat]:+.4f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Load data
    dataset = NominationDataset.model_validate_json(DATASET_PATH.read_text())
    records = dataset.records_for_ceremony_year(2026)
    coefficients = load_coefficients(FEATURE_IMPORTANCE_PATH)

    print("=== Counterfactual Sensitivity — Best Actor (clogit) ===")
    print(f"Model features: {sorted(coefficients.keys())}")
    print(f"Nominees: {[r.nominee_name for r in records]}")
    print()

    # Baseline
    baseline_lo = compute_log_odds(records, coefficients)
    baseline_probs = softmax(baseline_lo)
    print_baseline(records, coefficients, baseline_lo, baseline_probs)

    # Scenarios
    scenarios = build_scenarios(coefficients)
    for scenario_name, overrides in scenarios:
        cf_lo, cf_probs = compute_counterfactual(records, coefficients, overrides)
        print_scenario(scenario_name, records, overrides, baseline_probs, cf_lo, cf_probs)

    # Summary
    print_summary(baseline_probs, scenarios, records, coefficients)


if __name__ == "__main__":
    main()
