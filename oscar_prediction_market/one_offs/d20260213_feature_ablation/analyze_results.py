"""Analyze feature ablation & stability experiment results.

Handles three experiment types:
1. Single-date: Compare model configs at one snapshot (accuracy, Brier, features, predictions)
2. Temporal: Prediction jitter and feature stability across snapshot dates
3. Group ablation: Leave-one-group-out, additive, and single-group importance

Also computes LOYO Jaccard from saved per-fold feature importance files.

Usage:
    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\
d20260213_feature_ablation.analyze_results \
        --exp-dir storage/d20260213_feature_ablation
"""

import argparse
import csv
import json
import logging
from itertools import combinations
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

KEY_NOMINEES = [
    "One Battle after Another",
    "Sinners",
    "Hamnet",
    "Marty Supreme",
    "Frankenstein",
]


# ============================================================================
# Helper: read model run results
# ============================================================================


def _read_run(run_dir: Path) -> dict | None:
    """Read results from a single build_model output directory."""
    if not run_dir.is_dir():
        return None

    name = run_dir.name
    result: dict = {"name": name, "dir": run_dir}

    # Feature count + list (from 3_selected_features.json)
    feat_file = run_dir / "3_selected_features.json"
    if feat_file.exists():
        data = json.loads(feat_file.read_text())
        result["features"] = data["features"]
        result["n_features"] = len(data["features"])
    else:
        result["features"] = []
        result["n_features"] = None

    # CV metrics — prefer selected-feature CV, fall back to full/simple
    result["brier"] = None
    result["accuracy"] = None
    for step in ["4_selected_cv", "1_cv", "1_full_cv"]:
        mf = run_dir / step / "metrics.json"
        if mf.exists():
            m = json.loads(mf.read_text())
            result["brier"] = m["micro"]["brier_score"]
            result["accuracy"] = m["micro"]["accuracy"]
            break

    # Test predictions
    result["predictions"] = {}
    for step in ["5_final_predict", "2_final_predict"]:
        pf = run_dir / step / "predictions_test.csv"
        if pf.exists():
            df = pd.read_csv(pf)
            for _, row in df.iterrows():
                result["predictions"][str(row["title"])] = float(row["probability"])
            break

    # Fold importance (for LOYO Jaccard)
    fold_dir = None
    for step in ["4_selected_cv", "1_cv"]:
        candidate = run_dir / step / "fold_importance"
        if candidate.is_dir():
            fold_dir = candidate
            break
    result["fold_importance_dir"] = fold_dir

    return result


# ============================================================================
# LOYO Jaccard
# ============================================================================


def compute_loyo_jaccard(fold_dir: Path, threshold: float = 0.0) -> dict:
    """Compute pairwise Jaccard similarity of nonzero feature sets across LOYO folds.

    Args:
        fold_dir: Directory with fold_{year}.json files
        threshold: Minimum importance to count as "selected" (default 0 = nonzero)

    Returns:
        Dict with mean_jaccard, min_jaccard, max_jaccard, n_folds, per_fold_features
    """
    fold_files = sorted(fold_dir.glob("fold_*.json"))
    if len(fold_files) < 2:
        return {"mean_jaccard": None, "n_folds": len(fold_files)}

    fold_feature_sets: list[tuple[str, set[str]]] = []
    for ff in fold_files:
        data = json.loads(ff.read_text())
        nonzero = {feat for feat, imp in data.items() if imp > threshold}
        fold_feature_sets.append((ff.stem, nonzero))

    jaccards = []
    for (_, s1), (_, s2) in combinations(fold_feature_sets, 2):
        union = s1 | s2
        if not union:
            continue
        jaccards.append(len(s1 & s2) / len(union))

    return {
        "mean_jaccard": sum(jaccards) / len(jaccards) if jaccards else None,
        "min_jaccard": min(jaccards) if jaccards else None,
        "max_jaccard": max(jaccards) if jaccards else None,
        "n_folds": len(fold_feature_sets),
        "per_fold_n_features": [len(s) for _, s in fold_feature_sets],
    }


# ============================================================================
# Single-date analysis
# ============================================================================


def analyze_single_date(exp_dir: Path) -> list[dict]:
    """Analyze single-date experiment results."""
    single_dir = exp_dir / "single_date"
    if not single_dir.is_dir():
        return []

    runs = []
    for d in sorted(single_dir.iterdir()):
        if not d.is_dir():
            continue
        # build_model creates output_dir/name/ — check for numbered step dirs
        has_steps = (d / "1_full_cv").is_dir() or (d / "1_cv").is_dir()
        if has_steps:
            result = _read_run(d)
            if result:
                runs.append(result)
        else:
            # Check subdirectories (build_model nesting)
            for sub in sorted(d.iterdir()):
                if sub.is_dir() and ((sub / "1_full_cv").is_dir() or (sub / "1_cv").is_dir()):
                    result = _read_run(sub)
                    if result:
                        runs.append(result)

    return runs


def print_single_date_table(runs: list[dict]) -> None:
    """Print single-date comparison table."""
    if not runs:
        print("No single-date results found.")
        return

    short = [n[:12] for n in KEY_NOMINEES]
    header = f"{'Config':<35s} feat  {'Brier':>8s}  {'Acc':>6s}  " + "".join(
        f"{s:>14s}" for s in short
    )
    print("\n" + "=" * len(header))
    print("SINGLE-DATE COMPARISON (as-of 2026-02-07)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in runs:
        n_feat = str(r["n_features"]) if r["n_features"] else "?"
        brier = f"{r['brier']:.4f}" if r["brier"] is not None else "?"
        acc = f"{r['accuracy'] * 100:.1f}%" if r["accuracy"] is not None else "?"
        line = f"{r['name']:<35s} {n_feat:>4s}  {brier:>8s}  {acc:>6s}  "
        for nom in KEY_NOMINEES:
            p = r["predictions"].get(nom)
            if p is not None:
                line += f"{p * 100:13.1f}%"
            else:
                line += f"{'n/a':>14s}"
        print(line)


# ============================================================================
# Temporal analysis
# ============================================================================


def analyze_temporal(exp_dir: Path) -> dict[str, list[dict]]:
    """Analyze temporal experiment results. Returns {config: [run_per_date]}."""
    temporal_dir = exp_dir / "temporal"
    if not temporal_dir.is_dir():
        return {}

    configs: dict[str, list[dict]] = {}
    for config_dir in sorted(temporal_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        config_name = config_dir.name
        date_runs = []
        for date_dir in sorted(config_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            # Check date_dir itself
            has_steps = (date_dir / "5_final_predict").is_dir() or (
                date_dir / "2_final_predict"
            ).is_dir()
            if has_steps:
                result = _read_run(date_dir)
                if result:
                    result["snapshot_date"] = date_dir.name
                    date_runs.append(result)
            else:
                # Check subdirectories
                for sub in sorted(date_dir.iterdir()):
                    if sub.is_dir() and (
                        (sub / "5_final_predict").is_dir() or (sub / "2_final_predict").is_dir()
                    ):
                        result = _read_run(sub)
                        if result:
                            result["snapshot_date"] = date_dir.name
                            date_runs.append(result)
                            break

        if date_runs:
            configs[config_name] = sorted(date_runs, key=lambda r: r["snapshot_date"])

    return configs


def compute_jitter(temporal_configs: dict[str, list[dict]]) -> list[dict]:
    """Compute prediction jitter for each config across temporal snapshots."""
    jitter_rows = []

    for config_name, runs in temporal_configs.items():
        if len(runs) < 2:
            continue

        # Collect all nominees across all snapshots
        all_nominees: set[str] = set()
        for r in runs:
            all_nominees.update(r["predictions"].keys())

        # Compute per-nominee jitter
        nominee_jitters: dict[str, float] = {}
        for nominee in all_nominees:
            diffs = []
            for i in range(1, len(runs)):
                p_prev = runs[i - 1]["predictions"].get(nominee)
                p_curr = runs[i]["predictions"].get(nominee)
                if p_prev is not None and p_curr is not None:
                    diffs.append(abs(p_curr - p_prev))
            if diffs:
                nominee_jitters[nominee] = sum(diffs) / len(diffs)

        # Feature count variance
        feat_counts = [r["n_features"] for r in runs if r["n_features"] is not None]
        feat_count_var = None
        if feat_counts:
            mean_fc = sum(feat_counts) / len(feat_counts)
            feat_count_var = sum((fc - mean_fc) ** 2 for fc in feat_counts) / len(feat_counts)

        aggregate_jitter = (
            sum(nominee_jitters.values()) / len(nominee_jitters) if nominee_jitters else None
        )

        jitter_rows.append(
            {
                "config": config_name,
                "n_snapshots": len(runs),
                "aggregate_jitter": aggregate_jitter,
                "feat_count_mean": (sum(feat_counts) / len(feat_counts) if feat_counts else None),
                "feat_count_var": feat_count_var,
                "nominee_jitters": {
                    nom: nominee_jitters.get(nom) for nom in KEY_NOMINEES if nom in nominee_jitters
                },
            }
        )

    return jitter_rows


def print_temporal_table(jitter_rows: list[dict]) -> None:
    """Print temporal stability table."""
    if not jitter_rows:
        print("No temporal results found.")
        return

    short = [n[:12] for n in KEY_NOMINEES]
    header = f"{'Config':<25s} snaps  {'Jitter':>8s}  {'Feat':>8s}  " + "".join(
        f"{s:>14s}" for s in short
    )
    print("\n" + "=" * len(header))
    print("TEMPORAL STABILITY (jitter = mean |delta P| between consecutive snapshots)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in jitter_rows:
        jitter = f"{r['aggregate_jitter']:.4f}" if r["aggregate_jitter"] is not None else "?"
        feat_info = f"{r['feat_count_mean']:.0f}" if r["feat_count_mean"] is not None else "?"
        line = f"{r['config']:<25s} {r['n_snapshots']:>5d}  {jitter:>8s}  {feat_info:>8s}  "
        for nom in KEY_NOMINEES:
            j = r["nominee_jitters"].get(nom)
            if j is not None:
                line += f"{j:13.4f} "
            else:
                line += f"{'n/a':>14s}"
        print(line)


# ============================================================================
# Group ablation analysis
# ============================================================================


def analyze_group_ablation(exp_dir: Path) -> dict[str, list[dict]]:
    """Analyze group ablation results. Returns {mode: [runs]}."""
    ablation_dir = exp_dir / "group_ablation"
    if not ablation_dir.is_dir():
        return {}

    modes: dict[str, list[dict]] = {}
    for mode_dir in sorted(ablation_dir.iterdir()):
        if not mode_dir.is_dir():
            continue
        mode_name = mode_dir.name
        runs = []
        for config_dir in sorted(mode_dir.iterdir()):
            if not config_dir.is_dir():
                continue
            has_steps = (config_dir / "1_full_cv").is_dir() or (config_dir / "1_cv").is_dir()
            if has_steps:
                result = _read_run(config_dir)
                if result:
                    runs.append(result)
            else:
                for sub in sorted(config_dir.iterdir()):
                    if sub.is_dir() and ((sub / "1_full_cv").is_dir() or (sub / "1_cv").is_dir()):
                        result = _read_run(sub)
                        if result:
                            runs.append(result)
                            break

        if runs:
            modes[mode_name] = runs

    return modes


def print_group_ablation_table(mode_name: str, runs: list[dict]) -> None:
    """Print group ablation table for one mode."""
    if not runs:
        return

    # Find baseline (lr_full or gbt_full)
    baseline = None
    for r in runs:
        if "_full" in r["name"] and "without" not in r["name"]:
            baseline = r
            break

    header = f"{'Config':<45s} feat  {'Brier':>8s}  {'Acc':>7s}  {'dBrier':>8s}  {'dAcc':>7s}"
    print(f"\n{'=' * len(header)}")
    print(f"GROUP ABLATION: {mode_name}")
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))

    for r in runs:
        n_feat = str(r["n_features"]) if r["n_features"] else "?"
        brier = f"{r['brier']:.4f}" if r["brier"] is not None else "?"
        acc = f"{r['accuracy'] * 100:.1f}%" if r["accuracy"] is not None else "?"

        delta_brier = ""
        delta_acc = ""
        if baseline and r["brier"] is not None and baseline["brier"] is not None:
            db = r["brier"] - baseline["brier"]
            delta_brier = f"{db:+.4f}"
            if r["accuracy"] is not None and baseline["accuracy"] is not None:
                da = (r["accuracy"] - baseline["accuracy"]) * 100
                delta_acc = f"{da:+.1f}%"

        print(
            f"{r['name']:<45s} {n_feat:>4s}  {brier:>8s}  {acc:>7s}"
            f"  {delta_brier:>8s}  {delta_acc:>7s}"
        )


# ============================================================================
# Stability metrics CSV
# ============================================================================


def write_stability_metrics(
    exp_dir: Path,
    single_date_runs: list[dict],
    jitter_rows: list[dict],
    group_ablation_modes: dict[str, list[dict]],
) -> None:
    """Write combined stability metrics CSV."""
    rows = []

    # Single-date runs
    for r in single_date_runs:
        row = {
            "experiment": "single_date",
            "config": r["name"],
            "n_features": r["n_features"],
            "brier": r["brier"],
            "accuracy": r["accuracy"],
            "aggregate_jitter": None,
            "loyo_jaccard": None,
        }

        # LOYO Jaccard if available
        if r.get("fold_importance_dir"):
            jaccard_result = compute_loyo_jaccard(r["fold_importance_dir"])
            row["loyo_jaccard"] = jaccard_result.get("mean_jaccard")

        rows.append(row)

    # Temporal runs
    for jr in jitter_rows:
        rows.append(
            {
                "experiment": "temporal",
                "config": jr["config"],
                "n_features": jr.get("feat_count_mean"),
                "brier": None,
                "accuracy": None,
                "aggregate_jitter": jr["aggregate_jitter"],
                "loyo_jaccard": None,
            }
        )

    # Group ablation runs
    for mode_name, mode_runs in group_ablation_modes.items():
        for r in mode_runs:
            rows.append(
                {
                    "experiment": f"group_ablation_{mode_name}",
                    "config": r["name"],
                    "n_features": r["n_features"],
                    "brier": r["brier"],
                    "accuracy": r["accuracy"],
                    "aggregate_jitter": None,
                    "loyo_jaccard": None,
                }
            )

    if rows:
        output_path = exp_dir / "stability_metrics.csv"
        fieldnames = [
            "experiment",
            "config",
            "n_features",
            "brier",
            "accuracy",
            "aggregate_jitter",
            "loyo_jaccard",
        ]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {output_path}")


# ============================================================================
# Full predictions table
# ============================================================================


def print_full_predictions(runs: list[dict]) -> None:
    """Print full predictions for all nominees across all configs."""
    all_nominees: set[str] = set()
    for r in runs:
        all_nominees.update(r["predictions"].keys())

    if not all_nominees:
        return

    config_names = [r["name"][:20] for r in runs]
    header = f"{'Nominee':<35s}" + "".join(f"{cn:>22s}" for cn in config_names)
    print(f"\n{'=' * len(header)}")
    print("FULL PREDICTIONS (all nominees)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for nominee in sorted(all_nominees):
        line = f"{nominee[:33]:<35s}"
        for r in runs:
            p = r["predictions"].get(nominee)
            if p is not None:
                line += f"{p * 100:21.1f}%"
            else:
                line += f"{'n/a':>22s}"
        print(line)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze feature ablation experiment results")
    parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Path to experiment directory",
    )
    parser.add_argument(
        "--section",
        type=str,
        nargs="*",
        choices=["single_date", "temporal", "group_ablation", "all"],
        default=["all"],
        help="Which sections to analyze (default: all)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    exp_dir = Path(args.exp_dir)

    sections = (
        args.section if "all" not in args.section else ["single_date", "temporal", "group_ablation"]
    )

    # Single-date
    single_date_runs: list[dict] = []
    if "single_date" in sections:
        single_date_runs = analyze_single_date(exp_dir)
        print_single_date_table(single_date_runs)
        if single_date_runs:
            print_full_predictions(single_date_runs)

    # Temporal
    jitter_rows: list[dict] = []
    if "temporal" in sections:
        temporal_configs = analyze_temporal(exp_dir)
        jitter_rows = compute_jitter(temporal_configs)
        print_temporal_table(jitter_rows)

    # Group ablation
    group_ablation_modes: dict[str, list[dict]] = {}
    if "group_ablation" in sections:
        group_ablation_modes = analyze_group_ablation(exp_dir)
        for mode_name, mode_runs in group_ablation_modes.items():
            print_group_ablation_table(mode_name, mode_runs)

    # Write stability metrics CSV
    write_stability_metrics(exp_dir, single_date_runs, jitter_rows, group_ablation_modes)

    print("\nDone.")


if __name__ == "__main__":
    main()
