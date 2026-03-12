"""Analyze Phase 3 feature selection ablation results.

Reads results from the fs_ablation/ subdirectory across 9 categories × 2 models
(clogit, cal_sgbt). Produces:
- Threshold sweep: Brier/accuracy by threshold per category × model
- Best default threshold: single threshold that works best across categories
- Max features cap effect at threshold 0.80
- Gap analysis: single default vs per-category best
- Plots: threshold curves, heatmaps, bar charts

Directory structure expected:
    storage/d20260220_feature_ablation/
      {category}/
        fs_ablation/
          {model}_full_nofs/        # No feature selection baseline
          {model}_full_t050/        # threshold=0.50
          {model}_full_t060/        # threshold=0.60
          ...
          {model}_full_t100/        # threshold=1.00 (all nonzero)
          {model}_full_t080_m3/     # threshold=0.80, max_features=3
          {model}_full_t080_m5/     # threshold=0.80, max_features=5

Usage:
    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\
d20260220_feature_ablation.analyze_phase3 \
        --exp-dir storage/d20260220_feature_ablation

    # Specific sections:
    uv run python -m ... --section threshold_sweep best_default plots
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oscar_prediction_market.one_offs.analysis_utils.style import (
    CATEGORIES,
    CATEGORY_SHORT,
    get_model_color,
)

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

MODELS = ["clogit", "cal_sgbt"]

# Local display names (shorter than shared — no "Binary" prefix needed here)
MODEL_DISPLAY = {
    "clogit": "Clogit",
    "cal_sgbt": "Cal-SGBT",
}

# Ordered thresholds for display
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
THRESHOLD_LABELS = {t: f"t{int(t * 100):03d}" for t in THRESHOLDS}

MAX_FEATURES_CAPS = [3, 5]


# ============================================================================
# Helper: read results
# ============================================================================


def _safe_notna(val: object) -> bool:
    """Scalar-safe check for not-NA."""
    if val is None:
        return False
    try:
        return not np.isnan(float(val))  # type: ignore[arg-type]  # val is checked for None above
    except (TypeError, ValueError):
        return True


def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE).

    Bins predictions by predicted probability, then for each bin computes
    |avg_predicted_prob - fraction_actually_positive|, weighted by bin count.

    ECE = sum_b (|bin_b| / N) * |acc(b) - conf(b)|

    Lower is better. 0 means perfectly calibrated.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = len(probs)
    if n_total == 0:
        return 0.0
    for i in range(n_bins):
        mask = (probs > bin_edges[i]) & (probs <= bin_edges[i + 1])
        if i == 0:  # Include 0.0 in first bin
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_conf = probs[mask].mean()
        avg_acc = labels[mask].mean()
        ece += (n_bin / n_total) * abs(avg_acc - avg_conf)
    return float(ece)


def _compute_mce(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Maximum Calibration Error (MCE).

    Maximum |acc(b) - conf(b)| across non-empty bins.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        mask = (probs > bin_edges[i]) & (probs <= bin_edges[i + 1])
        if i == 0:
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_conf = probs[mask].mean()
        avg_acc = labels[mask].mean()
        mce = max(mce, abs(avg_acc - avg_conf))
    return float(mce)


def _read_run(run_dir: Path) -> dict | None:
    """Read results from a build_model output directory."""
    if not run_dir.is_dir():
        return None

    result: dict = {"name": run_dir.name, "dir": str(run_dir)}

    # Feature count from selected features
    feat_file = run_dir / "3_selected_features.json"
    if feat_file.exists():
        data = json.loads(feat_file.read_text())
        result["n_features"] = len(data["features"])
        result["selected_features"] = data["features"]
    else:
        result["n_features"] = None
        result["selected_features"] = None

    # CV metrics — prefer 4_selected_cv, fall back to 1_cv, 1_full_cv
    result["brier"] = None
    result["accuracy"] = None
    result["correct_ceremonies"] = None
    result["total_ceremonies"] = None

    for step in ["4_selected_cv", "1_cv", "1_full_cv"]:
        mf = run_dir / step / "metrics.json"
        if mf.exists():
            m = json.loads(mf.read_text())
            result["brier"] = m["micro"]["brier_score"]
            result["accuracy"] = m["micro"]["accuracy"]
            result["correct_ceremonies"] = m.get("correct_ceremonies")
            result["total_ceremonies"] = m.get("num_years")
            break

    # Read best hyperparams for alpha analysis
    result["best_alpha"] = None
    for step in ["4_selected_cv", "1_cv", "1_full_cv"]:
        bc = run_dir / step / "best_config.json"
        if bc.exists():
            config = json.loads(bc.read_text())
            # best_config.json nests params under "model_config"
            model_config = config.get("model_config", config.get("params", {}))
            if "alpha" in model_config:
                result["best_alpha"] = model_config["alpha"]
            break

    # For feature selection runs, also get full-feature CV baseline
    result["brier_full_cv"] = None
    result["accuracy_full_cv"] = None
    full_cv = run_dir / "1_full_cv" / "metrics.json"
    if full_cv.exists():
        fm = json.loads(full_cv.read_text())
        result["brier_full_cv"] = fm["micro"]["brier_score"]
        result["accuracy_full_cv"] = fm["micro"]["accuracy"]

    # Calibration metrics from predictions CSV
    result["ece"] = None
    result["mce"] = None
    result["mean_winner_prob"] = None
    for step in ["4_selected_cv", "1_cv", "1_full_cv"]:
        pred_file = run_dir / step / "predictions.csv"
        if pred_file.exists():
            try:
                preds = pd.read_csv(pred_file)
                probs = np.asarray(preds["probability"].values)
                labels = np.asarray(preds["is_actual_winner"].astype(int).values)
                result["ece"] = _compute_ece(probs, labels)
                result["mce"] = _compute_mce(probs, labels)
                # Mean probability assigned to actual winners
                winners = preds[preds["is_actual_winner"]]
                if not winners.empty:
                    result["mean_winner_prob"] = float(winners["probability"].mean())
            except Exception as e:
                logger.warning("Failed to read predictions from %s: %s", pred_file, e)
            break

    return result


def _parse_fs_config_name(name: str) -> dict:
    """Parse fs_ablation config name into components.

    Examples:
        clogit_full_nofs     -> {model: "clogit", threshold: None, max_features: None, is_nofs: True}
        clogit_full_t080     -> {model: "clogit", threshold: 0.80, max_features: None, is_nofs: False}
        cal_sgbt_full_t080_m5 -> {model: "cal_sgbt", threshold: 0.80, max_features: 5, is_nofs: False}
    """
    result: dict = {"model": "", "threshold": None, "max_features": None, "is_nofs": False}

    for prefix in MODELS:
        if name.startswith(prefix + "_"):
            result["model"] = prefix
            rest = name[len(prefix) + 1 :]
            break
    else:
        result["model"] = name.split("_")[0]
        rest = "_".join(name.split("_")[1:])

    if rest.endswith("_nofs") or rest == "full_nofs":
        result["is_nofs"] = True
    elif "_t" in rest:
        # Parse threshold: full_t080 or full_t080_m5
        parts = rest.split("_")
        for p in parts:
            if p.startswith("t") and len(p) == 4 and p[1:].isdigit():
                result["threshold"] = int(p[1:]) / 100.0
            elif p.startswith("m") and p[1:].isdigit():
                result["max_features"] = int(p[1:])

    return result


# ============================================================================
# Data collection
# ============================================================================


def collect_fs_ablation_results(exp_dir: Path) -> pd.DataFrame:
    """Collect all fs_ablation results into a DataFrame."""
    rows = []

    for category in CATEGORIES:
        fs_dir = exp_dir / category / "fs_ablation"
        if not fs_dir.is_dir():
            continue

        for config_dir in sorted(fs_dir.iterdir()):
            if not config_dir.is_dir():
                continue

            result = _read_run(config_dir)
            if result is None or result["brier"] is None:
                continue

            parsed = _parse_fs_config_name(result["name"])

            rows.append(
                {
                    "category": category,
                    "config_name": result["name"],
                    "model": parsed["model"],
                    "threshold": parsed["threshold"],
                    "max_features": parsed["max_features"],
                    "is_nofs": parsed["is_nofs"],
                    "brier": result["brier"],
                    "accuracy": result["accuracy"],
                    "correct_ceremonies": result["correct_ceremonies"],
                    "total_ceremonies": result["total_ceremonies"],
                    "n_features": result["n_features"],
                    "best_alpha": result["best_alpha"],
                    "brier_full_cv": result["brier_full_cv"],
                    "accuracy_full_cv": result["accuracy_full_cv"],
                    "selected_features": result["selected_features"],
                    "ece": result["ece"],
                    "mce": result["mce"],
                    "mean_winner_prob": result["mean_winner_prob"],
                }
            )

    return pd.DataFrame(rows)


# ============================================================================
# Analysis functions
# ============================================================================


def print_threshold_sweep(df: pd.DataFrame) -> None:
    """Print Brier/accuracy by threshold for each category × model."""
    # Filter to threshold sweep configs (no max_features cap)
    sweep_df = df[(df["max_features"].isna()) | (df["is_nofs"])]

    print("\n" + "=" * 120)
    print("THRESHOLD SWEEP — Brier (Accuracy%) by Category × Model")
    print("Thresholds: nofs, " + ", ".join(f"{t:.2f}" for t in THRESHOLDS))
    print("=" * 120)

    for model in MODELS:
        model_df = sweep_df[sweep_df["model"] == model]
        if model_df.empty:
            continue

        print(f"\n--- {MODEL_DISPLAY[model]} ---")
        header = f"{'Category':<15s}  {'nofs':>14s}"
        for t in THRESHOLDS:
            header += f"  {t:.2f}{'':>8s}"
        print(header)
        print("-" * (15 + 16 + len(THRESHOLDS) * 16))

        for category in CATEGORIES:
            cat_df = model_df[model_df["category"] == category]
            if cat_df.empty:
                continue

            line = f"{CATEGORY_SHORT.get(category, category):<15s}"

            # No-fs baseline
            nofs = cat_df[cat_df["is_nofs"]]
            if not nofs.empty:
                b = nofs.iloc[0]["brier"]
                a = nofs.iloc[0]["accuracy"]
                line += f"  {b:.4f} ({a * 100:4.1f}%)"
            else:
                line += f"  {'—':>14s}"

            # Thresholds
            best_brier = float("inf")
            for t in THRESHOLDS:
                t_df = cat_df[(cat_df["threshold"] == t) & (cat_df["max_features"].isna())]
                if not t_df.empty:
                    b = t_df.iloc[0]["brier"]
                    a = t_df.iloc[0]["accuracy"]
                    best_brier = min(best_brier, b)
                    line += f"  {b:.4f} ({a * 100:4.1f}%)"
                else:
                    line += f"  {'—':>14s}"

            print(line)

        # Print best threshold per category for this model
        print()
        best_line = f"{'BEST THRESH':<15s}  {'':>14s}"
        for t in THRESHOLDS:
            count = 0
            for category in CATEGORIES:
                cat_model = model_df[model_df["category"] == category]
                thresh_only = cat_model[
                    (cat_model["max_features"].isna()) & (~cat_model["is_nofs"])
                ]
                if not thresh_only.empty:
                    best_idx = thresh_only["brier"].idxmin()
                    best_t = thresh_only.loc[best_idx, "threshold"]
                    if best_t == t:
                        count += 1
            best_line += f"  {'wins=' + str(count):>14s}"
        print(best_line)


def print_max_features_effect(df: pd.DataFrame) -> None:
    """Print max features cap effect at threshold 0.80."""
    print("\n" + "=" * 120)
    print("MAX FEATURES CAP EFFECT (at threshold 0.80)")
    print("=" * 120)

    # Get threshold 0.80 results: no cap, cap=3, cap=5
    for model in MODELS:
        model_df = df[df["model"] == model]
        if model_df.empty:
            continue

        print(f"\n--- {MODEL_DISPLAY[model]} ---")
        header = (
            f"{'Category':<15s}  {'t=0.80':>14s}  {'t=0.80 m=3':>14s}  "
            f"{'t=0.80 m=5':>14s}  {'nofs':>14s}  {'Feat(t80)':>9s}  "
            f"{'Feat(m3)':>8s}  {'Feat(m5)':>8s}"
        )
        print(header)
        print("-" * 120)

        for category in CATEGORIES:
            cat_df = model_df[model_df["category"] == category]
            if cat_df.empty:
                continue

            line = f"{CATEGORY_SHORT.get(category, category):<15s}"

            # t=0.80 no cap
            t80 = cat_df[(cat_df["threshold"] == 0.80) & (cat_df["max_features"].isna())]
            if not t80.empty:
                b = t80.iloc[0]["brier"]
                a = t80.iloc[0]["accuracy"]
                nf = t80.iloc[0]["n_features"]
                line += f"  {b:.4f} ({a * 100:4.1f}%)"
                nf_t80 = str(int(nf)) if _safe_notna(nf) else "?"
            else:
                line += f"  {'—':>14s}"
                nf_t80 = "?"

            # t=0.80 m=3
            m3 = cat_df[(cat_df["threshold"] == 0.80) & (cat_df["max_features"] == 3)]
            if not m3.empty:
                b = m3.iloc[0]["brier"]
                a = m3.iloc[0]["accuracy"]
                nf = m3.iloc[0]["n_features"]
                line += f"  {b:.4f} ({a * 100:4.1f}%)"
                nf_m3 = str(int(nf)) if _safe_notna(nf) else "?"
            else:
                line += f"  {'—':>14s}"
                nf_m3 = "?"

            # t=0.80 m=5
            m5 = cat_df[(cat_df["threshold"] == 0.80) & (cat_df["max_features"] == 5)]
            if not m5.empty:
                b = m5.iloc[0]["brier"]
                a = m5.iloc[0]["accuracy"]
                nf = m5.iloc[0]["n_features"]
                line += f"  {b:.4f} ({a * 100:4.1f}%)"
                nf_m5 = str(int(nf)) if _safe_notna(nf) else "?"
            else:
                line += f"  {'—':>14s}"
                nf_m5 = "?"

            # nofs baseline
            nofs = cat_df[cat_df["is_nofs"]]
            if not nofs.empty:
                b = nofs.iloc[0]["brier"]
                a = nofs.iloc[0]["accuracy"]
                line += f"  {b:.4f} ({a * 100:4.1f}%)"
            else:
                line += f"  {'—':>14s}"

            line += f"  {nf_t80:>9s}  {nf_m3:>8s}  {nf_m5:>8s}"
            print(line)


def print_best_default(df: pd.DataFrame) -> None:
    """Find single best default threshold across categories.

    For each candidate threshold, compute average Brier across categories × models,
    then compare to per-category best.
    """
    # Only threshold sweep runs (no max_features)
    sweep_df = df[(df["max_features"].isna())]

    print("\n" + "=" * 120)
    print("BEST DEFAULT THRESHOLD — Average Brier across categories")
    print("=" * 120)

    candidates = ["nofs"] + [f"{t:.2f}" for t in THRESHOLDS]

    for model in MODELS:
        model_df = sweep_df[sweep_df["model"] == model]
        if model_df.empty:
            continue

        print(f"\n--- {MODEL_DISPLAY[model]} ---")

        # For each threshold candidate, compute avg Brier
        thresh_avg: dict[str, float] = {}
        thresh_missing: dict[str, int] = {}

        for label in candidates:
            briers = []
            missing = 0
            for category in CATEGORIES:
                cat_df = model_df[model_df["category"] == category]
                if label == "nofs":
                    subset = cat_df[cat_df["is_nofs"]]
                else:
                    t = float(label)
                    subset = cat_df[(cat_df["threshold"] == t) & (~cat_df["is_nofs"])]
                if not subset.empty:
                    briers.append(subset.iloc[0]["brier"])
                else:
                    missing += 1

            if briers:
                thresh_avg[label] = float(np.mean(briers))
                thresh_missing[label] = missing

        # Print sorted by avg Brier
        print(f"  {'Threshold':<12s}  {'Avg Brier':>10s}  {'Missing':>8s}  {'Rank':>5s}")
        print("  " + "-" * 45)
        sorted_items = sorted(thresh_avg.items(), key=lambda x: x[1])
        for rank, (label, avg) in enumerate(sorted_items, 1):
            marker = " <-- BEST" if rank == 1 else ""
            print(f"  {label:<12s}  {avg:>10.4f}  {thresh_missing[label]:>8d}  {rank:>5d}{marker}")


def print_gap_analysis(df: pd.DataFrame) -> None:
    """Show gap between single universal default and per-category best threshold."""
    sweep_df = df[(df["max_features"].isna())]

    print("\n" + "=" * 120)
    print("GAP ANALYSIS — Universal default vs per-category best threshold")
    print("=" * 120)

    for model in MODELS:
        model_df = sweep_df[sweep_df["model"] == model]
        if model_df.empty:
            continue

        # Find overall best default threshold (lowest avg Brier across categories)
        thresh_avg: dict[float, list[float]] = {}
        for t in THRESHOLDS:
            briers = []
            for category in CATEGORIES:
                cat_df = model_df[model_df["category"] == category]
                subset = cat_df[(cat_df["threshold"] == t) & (~cat_df["is_nofs"])]
                if not subset.empty:
                    briers.append(subset.iloc[0]["brier"])
            if briers:
                thresh_avg[t] = briers

        if not thresh_avg:
            continue

        best_default_t = min(thresh_avg, key=lambda t: float(np.mean(thresh_avg[t])))
        best_default_avg = float(np.mean(thresh_avg[best_default_t]))

        print(
            f"\n--- {MODEL_DISPLAY[model]} (best default: threshold={best_default_t:.2f}, "
            f"avg Brier={best_default_avg:.4f}) ---"
        )
        header = (
            f"{'Category':<15s}  {'Default':>14s}  {'Best':>14s}  "
            f"{'Best Thresh':>11s}  {'Gap':>8s}  {'Feat(def)':>9s}  {'Feat(best)':>10s}"
        )
        print(header)
        print("-" * 100)

        total_gap = 0.0
        n_cats = 0

        for category in CATEGORIES:
            cat_df = model_df[model_df["category"] == category]
            if cat_df.empty:
                continue

            # Default threshold result
            default = cat_df[(cat_df["threshold"] == best_default_t) & (~cat_df["is_nofs"])]

            # Per-category best threshold (include nofs)
            all_cat = cat_df.copy()
            best_idx = all_cat["brier"].idxmin()
            best_row = all_cat.loc[best_idx]

            if default.empty:
                continue

            d_brier = default.iloc[0]["brier"]
            d_acc = default.iloc[0]["accuracy"]
            d_nf = default.iloc[0]["n_features"]
            b_brier = best_row["brier"]
            b_acc = best_row["accuracy"]
            b_nf = best_row["n_features"]

            if best_row["is_nofs"]:
                b_thresh = "nofs"
            else:
                b_thresh = f"{best_row['threshold']:.2f}"

            gap = d_brier - b_brier
            total_gap += gap
            n_cats += 1

            d_nf_str = str(int(d_nf)) if _safe_notna(d_nf) else "all"
            b_nf_str = str(int(b_nf)) if _safe_notna(b_nf) else "all"

            line = (
                f"{CATEGORY_SHORT.get(category, category):<15s}"
                f"  {d_brier:.4f} ({d_acc * 100:4.1f}%)"
                f"  {b_brier:.4f} ({b_acc * 100:4.1f}%)"
                f"  {b_thresh:>11s}"
                f"  {gap:>+8.4f}"
                f"  {d_nf_str:>9s}"
                f"  {b_nf_str:>10s}"
            )
            print(line)

        if n_cats > 0:
            avg_gap = total_gap / n_cats
            print(f"\n  Average gap: {avg_gap:+.4f} Brier (across {n_cats} categories)")

            # Find max gap category
            max_gap_cat = ""
            max_gap_val = 0.0
            for cat in CATEGORIES:
                cat_sub = model_df[model_df["category"] == cat]
                default_sub = cat_sub[
                    (cat_sub["threshold"] == best_default_t) & (~cat_sub["is_nofs"])
                ]
                if default_sub.empty:
                    continue
                gap_val = default_sub.iloc[0]["brier"] - cat_sub["brier"].min()
                if gap_val > max_gap_val:
                    max_gap_val = gap_val
                    max_gap_cat = cat
            if max_gap_cat:
                print(f"  Max gap category: {max_gap_cat} (+{max_gap_val:.4f})")


def print_alpha_analysis(df: pd.DataFrame) -> None:
    """Check if the expanded alpha grid (0.001) is being selected."""
    print("\n" + "=" * 120)
    print("CLOGIT ALPHA GRID ANALYSIS — Selected alpha by category")
    print("=" * 120)

    clogit_df = df[df["model"] == "clogit"]
    if clogit_df.empty:
        print("No clogit results found.")
        return

    # For each category, show what alpha was selected for the nofs and best threshold
    header = f"{'Category':<15s}  {'Config':<25s}  {'Alpha':>8s}  {'Brier':>8s}"
    print(header)
    print("-" * 65)

    for category in CATEGORIES:
        cat_df = clogit_df[clogit_df["category"] == category]
        if cat_df.empty:
            continue

        # Show nofs and best-threshold alpha
        for label, subset in [
            ("nofs", cat_df[cat_df["is_nofs"]]),
            ("t=0.80", cat_df[(cat_df["threshold"] == 0.80) & (cat_df["max_features"].isna())]),
        ]:
            if subset.empty:
                continue
            row = subset.iloc[0]
            alpha_str = str(row["best_alpha"]) if _safe_notna(row["best_alpha"]) else "?"
            print(
                f"{CATEGORY_SHORT.get(category, category):<15s}"
                f"  {label:<25s}"
                f"  {alpha_str:>8s}"
                f"  {row['brier']:>8.4f}"
            )

    # Count alpha value frequencies
    alphas = clogit_df["best_alpha"].dropna()
    if not alphas.empty:
        print("\nAlpha frequency (all clogit runs):")
        for val, count in alphas.value_counts().sort_index().items():
            print(f"  alpha={val}: {count} runs")


def print_selected_features_summary(df: pd.DataFrame) -> None:
    """Show which features are selected at each threshold for each category."""
    print("\n" + "=" * 120)
    print("SELECTED FEATURES SUMMARY — Feature count by threshold")
    print("=" * 120)

    sweep_df = df[(df["max_features"].isna()) & (~df["is_nofs"])]

    for model in MODELS:
        model_df = sweep_df[sweep_df["model"] == model]
        if model_df.empty:
            continue

        print(f"\n--- {MODEL_DISPLAY[model]} ---")
        header = f"{'Category':<15s}" + "".join(f"  {t:>8.2f}" for t in THRESHOLDS)
        print(header)
        print("-" * (15 + len(THRESHOLDS) * 10))

        for category in CATEGORIES:
            cat_df = model_df[model_df["category"] == category]
            line = f"{CATEGORY_SHORT.get(category, category):<15s}"
            for t in THRESHOLDS:
                t_df = cat_df[cat_df["threshold"] == t]
                if not t_df.empty:
                    nf = t_df.iloc[0]["n_features"]
                    nf_str = str(int(nf)) if _safe_notna(nf) else "?"
                    line += f"  {nf_str:>8s}"
                else:
                    line += f"  {'—':>8s}"
            print(line)


def print_unified_threshold_comparison(df: pd.DataFrame) -> None:
    """Compare t=0.90 (recommended) vs t=0.80 (previous default) and t=0.95.

    Shows both Brier and accuracy averages, and per-category deltas.
    """
    sweep_df = df[df["max_features"].isna()].copy()

    print("\n" + "=" * 120)
    print("UNIFIED THRESHOLD COMPARISON — t=0.90 (recommended) vs t=0.80 (previous) vs t=0.95")
    print("=" * 120)

    # Cross-model averages
    print("\n--- Cross-model averages (both models × 9 categories) ---")
    for label, selector in [
        ("nofs", lambda d: d[d["is_nofs"]]),
        ("t=0.80", lambda d: d[d["threshold"] == 0.80]),
        ("t=0.90", lambda d: d[d["threshold"] == 0.90]),
        ("t=0.95", lambda d: d[d["threshold"] == 0.95]),
    ]:
        rows = selector(sweep_df)
        if not rows.empty:
            avg_b = rows["brier"].mean()
            avg_a = rows["accuracy"].mean()
            print(f"  {label:<8s}: avg Brier = {avg_b:.4f}, avg Acc = {avg_a * 100:.1f}%")

    # Per-model tables
    for model in MODELS:
        mdf = sweep_df[sweep_df["model"] == model]
        if mdf.empty:
            continue

        print(f"\n--- {MODEL_DISPLAY[model]}: t=0.90 vs t=0.80 per category ---")
        header = (
            f"{'Category':<15s}  {'nofs':>14s}  {'t=0.80':>14s}  "
            f"{'t=0.90':>14s}  {'t=0.95':>14s}  {'Δ(90-80)':>10s}"
        )
        print(header)
        print("-" * 90)

        briers_080: list[float] = []
        briers_090: list[float] = []
        briers_095: list[float] = []
        briers_nofs: list[float] = []
        accs_080: list[float] = []
        accs_090: list[float] = []
        accs_095: list[float] = []
        accs_nofs: list[float] = []

        for category in CATEGORIES:
            cat_df = mdf[mdf["category"] == category]
            line = f"{CATEGORY_SHORT.get(category, category):<15s}"

            for _label, rows, b_list, a_list in [
                ("nofs", cat_df[cat_df["is_nofs"]], briers_nofs, accs_nofs),
                ("t080", cat_df[cat_df["threshold"] == 0.80], briers_080, accs_080),
                ("t090", cat_df[cat_df["threshold"] == 0.90], briers_090, accs_090),
                ("t095", cat_df[cat_df["threshold"] == 0.95], briers_095, accs_095),
            ]:
                if not rows.empty:
                    b = rows.iloc[0]["brier"]
                    a = rows.iloc[0]["accuracy"]
                    b_list.append(b)
                    a_list.append(a)
                    line += f"  {b:.4f} ({a * 100:4.1f}%)"
                else:
                    line += f"  {'—':>14s}"

            # Delta t=0.90 - t=0.80
            t080 = cat_df[cat_df["threshold"] == 0.80]
            t090 = cat_df[cat_df["threshold"] == 0.90]
            if not t080.empty and not t090.empty:
                delta = t090.iloc[0]["brier"] - t080.iloc[0]["brier"]
                line += f"  {delta:>+10.4f}"
            else:
                line += f"  {'—':>10s}"

            print(line)

        # Averages
        def _avg(lst: list[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        print(
            f"{'AVERAGE':<15s}"
            f"  {_avg(briers_nofs):.4f} ({_avg(accs_nofs) * 100:4.1f}%)"
            f"  {_avg(briers_080):.4f} ({_avg(accs_080) * 100:4.1f}%)"
            f"  {_avg(briers_090):.4f} ({_avg(accs_090) * 100:4.1f}%)"
            f"  {_avg(briers_095):.4f} ({_avg(accs_095) * 100:4.1f}%)"
            f"  {_avg(briers_090) - _avg(briers_080):>+10.4f}"
        )


def print_calibration_analysis(df: pd.DataFrame) -> None:
    """Show calibration metrics (ECE, MCE, mean winner prob) by threshold."""
    sweep_df = df[df["max_features"].isna()].copy()

    print("\n" + "=" * 120)
    print("CALIBRATION ANALYSIS — ECE, MCE, Mean Winner Prob by threshold")
    print("=" * 120)

    # Average calibration by threshold × model
    print("\n--- Average calibration metrics across 9 categories ---")
    for model in MODELS:
        mdf = sweep_df[sweep_df["model"] == model]
        if mdf.empty:
            continue

        print(f"\n  {MODEL_DISPLAY[model]}:")
        print(
            f"  {'Setting':<10s}  {'Avg ECE':>8s}  {'Avg MCE':>8s}  {'Avg WinP':>8s}  {'Avg Brier':>10s}"
        )
        print("  " + "-" * 55)

        for label in ["nofs"] + [f"{t:.2f}" for t in THRESHOLDS]:
            if label == "nofs":
                rows = mdf[mdf["is_nofs"]]
            else:
                t = float(label)
                rows = mdf[(mdf["threshold"] == t) & (~mdf["is_nofs"])]
            if rows.empty:
                continue
            ece_vals = rows["ece"].dropna()
            mce_vals = rows["mce"].dropna()
            wp_vals = rows["mean_winner_prob"].dropna()
            avg_b = rows["brier"].mean()
            ece_str = f"{ece_vals.mean():.4f}" if not ece_vals.empty else "?"
            mce_str = f"{mce_vals.mean():.4f}" if not mce_vals.empty else "?"
            wp_str = f"{wp_vals.mean():.2f}" if not wp_vals.empty else "?"
            marker = " *" if label == "0.90" else ""
            print(
                f"  {label:<10s}  {ece_str:>8s}  {mce_str:>8s}  {wp_str:>8s}  {avg_b:>10.4f}{marker}"
            )

    # Per-category ECE at t=0.90 vs nofs
    print("\n--- Per-category ECE: t=0.90 vs nofs ---")
    print(
        f"{'Category':<15s}  {'Clogit nofs':>12s}  {'Clogit t90':>12s}  "
        f"{'CalSGBT nofs':>12s}  {'CalSGBT t90':>12s}"
    )
    print("-" * 75)

    for category in CATEGORIES:
        line = f"{CATEGORY_SHORT.get(category, category):<15s}"
        for model in MODELS:
            for is_nofs, thresh in [(True, None), (False, 0.90)]:
                cat_df = sweep_df[(sweep_df["category"] == category) & (sweep_df["model"] == model)]
                if is_nofs:
                    rows = cat_df[cat_df["is_nofs"]]
                else:
                    rows = cat_df[cat_df["threshold"] == thresh]
                if not rows.empty and _safe_notna(rows.iloc[0]["ece"]):
                    line += f"  {rows.iloc[0]['ece']:>12.4f}"
                else:
                    line += f"  {'?':>12s}"
        print(line)


def print_threshold_policy(df: pd.DataFrame) -> None:
    """Analyze: single universal threshold vs per-model vs per-(model,category)."""
    sweep_df = df[(df["max_features"].isna()) & (~df["is_nofs"])].copy()

    print("\n" + "=" * 120)
    print("THRESHOLD POLICY — Universal vs per-model vs per-(model,category)")
    print("=" * 120)

    # Option A: single threshold for both models
    print("\n--- Option A: Single universal threshold ---")
    for t in [0.80, 0.90, 0.95]:
        briers = []
        accs = []
        for model in MODELS:
            mdf = sweep_df[(sweep_df["model"] == model) & (sweep_df["threshold"] == t)]
            briers.extend(mdf["brier"].tolist())
            accs.extend(mdf["accuracy"].tolist())
        if briers:
            print(
                f"  t={t:.2f}: avg Brier = {np.mean(briers):.4f}, "
                f"avg Acc = {np.mean(accs) * 100:.1f}% (n={len(briers)})"
            )

    # Option B: per-model threshold
    print("\n--- Option B: Per-model threshold ---")
    for model in MODELS:
        mdf = sweep_df[sweep_df["model"] == model]
        best_t = None
        best_avg = float("inf")
        for t in THRESHOLDS:
            t_df = mdf[mdf["threshold"] == t]
            if not t_df.empty:
                avg = t_df["brier"].mean()
                if avg < best_avg:
                    best_avg = avg
                    best_t = t
        if best_t is not None:
            t_df = mdf[mdf["threshold"] == best_t]
            print(
                f"  {MODEL_DISPLAY[model]}: best t={best_t:.2f}, "
                f"avg Brier = {best_avg:.4f}, avg Acc = {t_df['accuracy'].mean() * 100:.1f}%"
            )

    # Option C: per-(model, category)
    print("\n--- Option C: Per-(model, category) threshold ---")
    total_brier: list[float] = []
    total_acc: list[float] = []
    for model in MODELS:
        mdf = sweep_df[sweep_df["model"] == model]
        for cat in CATEGORIES:
            cat_df = mdf[mdf["category"] == cat]
            if not cat_df.empty:
                best_idx = cat_df["brier"].idxmin()
                total_brier.append(float(cat_df.loc[best_idx, "brier"]))  # type: ignore[arg-type]
                total_acc.append(float(cat_df.loc[best_idx, "accuracy"]))  # type: ignore[arg-type]
    if total_brier:
        print(
            f"  avg Brier = {np.mean(total_brier):.4f}, "
            f"avg Acc = {np.mean(total_acc) * 100:.1f}% (n={len(total_brier)})"
        )

    # Gap summary
    print("\n--- Gap from per-(model,category) best ---")
    # Get per-(model,cat) bests
    best_by_mc: dict[tuple[str, str], float] = {}
    for model in MODELS:
        mdf = sweep_df[sweep_df["model"] == model]
        for cat in CATEGORIES:
            cat_df = mdf[mdf["category"] == cat]
            if not cat_df.empty:
                best_by_mc[(model, cat)] = float(cat_df["brier"].min())

    for label, threshold_map in [
        ("Universal t=0.90", dict.fromkeys(MODELS, 0.9)),
        ("Per-model (clogit=0.95, sgbt=0.90)", {"clogit": 0.95, "cal_sgbt": 0.90}),
    ]:
        gaps: list[float] = []
        for model in MODELS:
            t = threshold_map[model]
            mdf = sweep_df[(sweep_df["model"] == model) & (sweep_df["threshold"] == t)]
            for _, row in mdf.iterrows():
                key = (model, row["category"])
                if key in best_by_mc:
                    gaps.append(float(row["brier"]) - best_by_mc[key])
        if gaps:
            print(f"  {label}: avg gap = +{np.mean(gaps):.4f} Brier")


# ============================================================================
# Plotting functions
# ============================================================================


def plot_threshold_curves(df: pd.DataFrame, output_dir: Path) -> Path:
    """Line plot: Brier vs threshold for each category, one subplot per model.

    3×3 grid of categories, with separate figures for clogit and cal_sgbt.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_df = df[(df["max_features"].isna())]
    paths = []

    for model in MODELS:
        model_df = sweep_df[sweep_df["model"] == model]
        if model_df.empty:
            continue

        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        axes_flat = axes.flatten()

        for idx, category in enumerate(CATEGORIES):
            ax = axes_flat[idx]
            cat_df = model_df[model_df["category"] == category]
            if cat_df.empty:
                ax.set_visible(False)
                continue

            # Threshold results
            thresh_data = cat_df[(~cat_df["is_nofs"]) & (cat_df["threshold"].notna())]
            thresh_data = thresh_data.sort_values("threshold")

            if not thresh_data.empty:
                ax.plot(
                    thresh_data["threshold"],
                    thresh_data["brier"],
                    "o-",
                    color=get_model_color(model),
                    markersize=5,
                    linewidth=1.5,
                    label="Brier",
                )

                # Mark the best threshold
                best_idx = thresh_data["brier"].idxmin()
                best_row = thresh_data.loc[best_idx]
                ax.plot(
                    best_row["threshold"],
                    best_row["brier"],
                    "*",
                    color="gold",
                    markersize=14,
                    zorder=5,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                )

                # Feature count on secondary axis
                ax2 = ax.twinx()
                nf_data = thresh_data[thresh_data["n_features"].notna()]
                if not nf_data.empty:
                    ax2.plot(
                        nf_data["threshold"],
                        nf_data["n_features"],
                        "s--",
                        color="gray",
                        markersize=3,
                        linewidth=1,
                        alpha=0.5,
                        label="# Features",
                    )
                    ax2.set_ylabel("# Features", fontsize=8, color="gray")
                    ax2.tick_params(axis="y", labelsize=7, colors="gray")

            # No-fs baseline as horizontal line
            nofs = cat_df[cat_df["is_nofs"]]
            if not nofs.empty:
                nofs_brier = nofs.iloc[0]["brier"]
                ax.axhline(
                    y=nofs_brier,
                    color="black",
                    linestyle=":",
                    alpha=0.5,
                    label=f"no_fs ({nofs_brier:.4f})",
                )

            ax.set_title(CATEGORY_SHORT.get(category, category), fontsize=11, fontweight="bold")
            ax.set_xlabel("Importance Threshold", fontsize=8)
            ax.set_ylabel("Brier Score", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

            if idx == 0:
                ax.legend(fontsize=7, loc="upper right")

        fig.suptitle(
            f"{MODEL_DISPLAY[model]} — Feature Selection Threshold Sweep\n"
            f"(★ = best threshold, dotted = no feature selection baseline)",
            fontsize=13,
            y=1.02,
        )
        plt.tight_layout()
        out_path = output_dir / f"fs_threshold_curves_{model}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")
        paths.append(out_path)

    return paths[0] if paths else output_dir / "fs_threshold_curves.png"


def plot_threshold_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    """Heatmap: Brier score by category × threshold, one per model.

    Cells colored by Brier (lower = greener). Star marks the best threshold per category.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_df = df[(df["max_features"].isna()) & (~df["is_nofs"])]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax_idx, model in enumerate(MODELS):
        ax = axes[ax_idx]
        model_df = sweep_df[sweep_df["model"] == model]
        if model_df.empty:
            ax.set_visible(False)
            continue

        # Build matrix: categories × thresholds
        matrix = np.full((len(CATEGORIES), len(THRESHOLDS)), np.nan)
        for i, category in enumerate(CATEGORIES):
            cat_df = model_df[model_df["category"] == category]
            for j, t in enumerate(THRESHOLDS):
                t_df = cat_df[cat_df["threshold"] == t]
                if not t_df.empty:
                    matrix[i, j] = t_df.iloc[0]["brier"]

        im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")

        ax.set_xticks(range(len(THRESHOLDS)))
        ax.set_xticklabels([f"{t:.2f}" for t in THRESHOLDS])
        ax.set_yticks(range(len(CATEGORIES)))
        ax.set_yticklabels([CATEGORY_SHORT.get(c, c) for c in CATEGORIES])

        # Annotate cells and mark best per row
        for i in range(len(CATEGORIES)):
            row_vals = matrix[i, :]
            valid = ~np.isnan(row_vals)
            if valid.any():
                best_j = int(np.nanargmin(row_vals))

            for j in range(len(THRESHOLDS)):
                val = matrix[i, j]
                if np.isnan(val):
                    continue
                is_best = valid.any() and j == best_j
                fontweight = "bold" if is_best else "normal"
                text = f"{val:.4f}" if not is_best else f"★{val:.4f}"
                text_color = "white" if val > np.nanmedian(matrix) else "black"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=text_color,
                    fontweight=fontweight,
                )

        ax.set_title(f"{MODEL_DISPLAY[model]}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Importance Threshold")
        plt.colorbar(im, ax=ax, label="Brier Score", shrink=0.8)

    fig.suptitle(
        "Feature Selection Threshold Heatmap — Brier Score\n(★ = best threshold per category)",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    out_path = output_dir / "fs_threshold_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


def plot_default_vs_best(df: pd.DataFrame, output_dir: Path) -> Path:
    """Bar chart: gap between universal default threshold and per-category best.

    Shows the Brier penalty of using a single default vs optimal per-category.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_df = df[(df["max_features"].isna())]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, model in enumerate(MODELS):
        ax = axes[ax_idx]
        model_df = sweep_df[sweep_df["model"] == model]
        if model_df.empty:
            ax.set_visible(False)
            continue

        # Find best default threshold
        thresh_avg: dict[float, list[float]] = {}
        for t in THRESHOLDS:
            briers = []
            for category in CATEGORIES:
                cat_df = model_df[model_df["category"] == category]
                subset = cat_df[(cat_df["threshold"] == t) & (~cat_df["is_nofs"])]
                if not subset.empty:
                    briers.append(subset.iloc[0]["brier"])
            if briers:
                thresh_avg[t] = briers

        if not thresh_avg:
            continue

        best_default_t = min(thresh_avg, key=lambda t: float(np.mean(thresh_avg[t])))

        # Compute gaps
        gaps = []
        cat_labels = []
        default_briers = []
        best_briers = []
        best_thresholds = []

        for category in CATEGORIES:
            cat_df = model_df[model_df["category"] == category]
            default = cat_df[(cat_df["threshold"] == best_default_t) & (~cat_df["is_nofs"])]
            if default.empty:
                continue

            best_row = cat_df.loc[cat_df["brier"].idxmin()]
            gap = default.iloc[0]["brier"] - best_row["brier"]
            gaps.append(gap)
            cat_labels.append(CATEGORY_SHORT.get(category, category))
            default_briers.append(default.iloc[0]["brier"])
            best_briers.append(best_row["brier"])
            if best_row["is_nofs"]:
                best_thresholds.append("nofs")
            else:
                best_thresholds.append(f"{best_row['threshold']:.2f}")

        x = np.arange(len(cat_labels))
        width = 0.35

        ax.bar(
            x - width / 2,
            default_briers,
            width,
            label=f"Default (t={best_default_t:.2f})",
            color=get_model_color(model),
            alpha=0.8,
        )
        ax.bar(
            x + width / 2,
            best_briers,
            width,
            label="Per-category best",
            color=get_model_color(model),
            alpha=0.4,
            hatch="//",
        )

        # Annotate gaps
        for i, (gap, bt) in enumerate(zip(gaps, best_thresholds, strict=True)):
            if gap > 0.0005:
                ax.annotate(
                    f"+{gap:.4f}\n(best: {bt})",
                    xy=(x[i], max(default_briers[i], best_briers[i])),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=6,
                    color="red",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(cat_labels, rotation=30, ha="right")
        ax.set_ylabel("Brier Score")
        ax.set_title(
            f"{MODEL_DISPLAY[model]} — Default (t={best_default_t:.2f}) vs Per-Category Best",
            fontsize=11,
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = output_dir / "fs_default_vs_best.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


def plot_max_features_comparison(df: pd.DataFrame, output_dir: Path) -> Path:
    """Grouped bar chart: Brier for t=0.80 with no cap, cap=3, cap=5."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, model in enumerate(MODELS):
        ax = axes[ax_idx]
        model_df = df[df["model"] == model]
        if model_df.empty:
            ax.set_visible(False)
            continue

        cat_labels = []
        no_cap_b = []
        m3_b = []
        m5_b = []

        for category in CATEGORIES:
            cat_df = model_df[model_df["category"] == category]
            t80 = cat_df[(cat_df["threshold"] == 0.80) & (cat_df["max_features"].isna())]
            t80_m3 = cat_df[(cat_df["threshold"] == 0.80) & (cat_df["max_features"] == 3)]
            t80_m5 = cat_df[(cat_df["threshold"] == 0.80) & (cat_df["max_features"] == 5)]

            if t80.empty:
                continue

            cat_labels.append(CATEGORY_SHORT.get(category, category))
            no_cap_b.append(t80.iloc[0]["brier"])
            m3_b.append(t80_m3.iloc[0]["brier"] if not t80_m3.empty else np.nan)
            m5_b.append(t80_m5.iloc[0]["brier"] if not t80_m5.empty else np.nan)

        x = np.arange(len(cat_labels))
        width = 0.25

        ax.bar(x - width, no_cap_b, width, label="No cap", color=get_model_color(model), alpha=0.8)
        ax.bar(x, m5_b, width, label="Max 5", color=get_model_color(model), alpha=0.5)
        ax.bar(x + width, m3_b, width, label="Max 3", color=get_model_color(model), alpha=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels(cat_labels, rotation=30, ha="right")
        ax.set_ylabel("Brier Score")
        ax.set_title(f"{MODEL_DISPLAY[model]} — Max Features Cap (at threshold=0.80)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = output_dir / "fs_max_features.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


def plot_nofs_vs_fs_improvement(df: pd.DataFrame, output_dir: Path) -> Path:
    """Bar chart showing Brier improvement from best threshold vs nofs, per category."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_df = df[(df["max_features"].isna())]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, model in enumerate(MODELS):
        ax = axes[ax_idx]
        model_df = sweep_df[sweep_df["model"] == model]
        if model_df.empty:
            ax.set_visible(False)
            continue

        cat_labels = []
        improvements = []
        best_thresholds = []

        for category in CATEGORIES:
            cat_df = model_df[model_df["category"] == category]
            nofs = cat_df[cat_df["is_nofs"]]
            thresh_only = cat_df[~cat_df["is_nofs"]]

            if nofs.empty or thresh_only.empty:
                continue

            nofs_brier = nofs.iloc[0]["brier"]
            best_thresh = thresh_only.loc[thresh_only["brier"].idxmin()]
            improvement = nofs_brier - best_thresh["brier"]

            cat_labels.append(CATEGORY_SHORT.get(category, category))
            improvements.append(improvement)
            best_thresholds.append(best_thresh["threshold"])

        x = np.arange(len(cat_labels))
        colors = ["green" if imp > 0 else "red" for imp in improvements]
        ax.bar(x, improvements, color=colors, alpha=0.7)

        # Annotate with best threshold
        for i, (imp, bt) in enumerate(zip(improvements, best_thresholds, strict=True)):
            ax.text(
                x[i],
                imp + (0.001 if imp > 0 else -0.002),
                f"t={bt:.2f}",
                ha="center",
                va="bottom" if imp > 0 else "top",
                fontsize=7,
                fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(cat_labels, rotation=30, ha="right")
        ax.set_ylabel("Brier Improvement (positive = FS helps)")
        ax.set_title(f"{MODEL_DISPLAY[model]} — Best FS Threshold vs No Feature Selection")
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = output_dir / "fs_improvement_vs_nofs.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


def generate_all_plots(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Generate all Phase 3 plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating Phase 3 plots in {output_dir}/...")

    paths = []
    paths.append(plot_threshold_curves(df, output_dir))
    paths.append(plot_threshold_heatmap(df, output_dir))
    paths.append(plot_default_vs_best(df, output_dir))
    paths.append(plot_max_features_comparison(df, output_dir))
    paths.append(plot_nofs_vs_fs_improvement(df, output_dir))

    print(f"  Generated {len(paths)} plots")
    return paths


# ============================================================================
# Summary CSV
# ============================================================================


def write_summary(df: pd.DataFrame, output_path: Path) -> None:
    """Write Phase 3 summary to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Drop selected_features column (too verbose for CSV)
    export_df = df.drop(columns=["selected_features"], errors="ignore")
    export_df.to_csv(output_path, index=False)
    print(f"\nWrote {len(export_df)} rows to {output_path}")

    json_path = output_path.with_suffix(".json")
    export_df.to_json(json_path, orient="records", indent=2)
    print(f"Wrote {json_path}")


# ============================================================================
# Main
# ============================================================================

ALL_SECTIONS = [
    "threshold_sweep",
    "max_features",
    "best_default",
    "gap_analysis",
    "unified_comparison",
    "calibration",
    "threshold_policy",
    "alpha",
    "features_summary",
    "plots",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Phase 3 feature selection ablation results"
    )
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
        choices=ALL_SECTIONS + ["all"],
        default=["all"],
        help="Which sections to show",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory for plots (default: {exp-dir}/plots)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    exp_dir = Path(args.exp_dir)
    plot_dir = Path(args.plot_dir) if args.plot_dir else exp_dir / "plots"

    sections = args.section if "all" not in args.section else ALL_SECTIONS

    print("Collecting Phase 3 (fs_ablation) results...")
    df = collect_fs_ablation_results(exp_dir)
    if df.empty:
        print("No fs_ablation results found.")
        return

    n_cats = df["category"].nunique()
    n_models = df["model"].nunique()
    n_configs = df["config_name"].nunique()
    print(f"Found {len(df)} results: {n_cats} categories, {n_models} models, {n_configs} configs")

    if "threshold_sweep" in sections:
        print_threshold_sweep(df)

    if "max_features" in sections:
        print_max_features_effect(df)

    if "best_default" in sections:
        print_best_default(df)

    if "gap_analysis" in sections:
        print_gap_analysis(df)

    if "unified_comparison" in sections:
        print_unified_threshold_comparison(df)

    if "calibration" in sections:
        print_calibration_analysis(df)

    if "threshold_policy" in sections:
        print_threshold_policy(df)

    if "alpha" in sections:
        print_alpha_analysis(df)

    if "features_summary" in sections:
        print_selected_features_summary(df)

    if "plots" in sections:
        generate_all_plots(df, plot_dir)

    # Always write summary
    summary_dir = exp_dir / "summary"
    write_summary(df, summary_dir / "phase3_results.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
