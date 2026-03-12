"""Analyze multi-category feature ablation results with plots.

Reads results from all 9 categories × 4 model types × 2 modes × 3 ablation
strategies. Produces:
- Per-category tables: best model, best feature group combination (Brier + accuracy)
- Cross-category comparison: feature group importance ranking
- Feature selection comparison (no_fs vs with_fs)
- Additive ablation & leave-one-out analysis
- Plots saved to storage assets directory
- Summary CSV for further analysis

Directory structure expected:
    storage/d20260220_feature_ablation/
      {category}/
        {mode}/          # no_fs, with_fs
          {config_name}/
            1_cv/ or 4_selected_cv/   # CV metrics
            2_final_predict/ or 5_final_predict/  # Predictions

Usage:
    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\
d20260220_feature_ablation.analyze_results \
        --exp-dir storage/d20260220_feature_ablation

    # Specific sections only:
    uv run python -m ... --section summary plots

    # Skip plots:
    uv run python -m ... --section summary model_comparison additive
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

matplotlib.use("Agg")  # Non-interactive backend for headless/CI environments

logger = logging.getLogger(__name__)

MODEL_PREFIXES = ["lr", "clogit", "gbt", "cal_sgbt"]

# Local display names (shorter than shared — no "Binary" prefix needed here)
MODEL_DISPLAY = {
    "lr": "LR",
    "clogit": "Clogit",
    "gbt": "GBT",
    "cal_sgbt": "Cal-SGBT",
}

MODES = ["no_fs", "with_fs"]


# ============================================================================
# Helper: read model run results
# ============================================================================


def _read_run(run_dir: Path) -> dict | None:
    """Read results from a single build_model output directory.

    Extracts CV metrics (Brier score, accuracy, correct/total ceremonies) and
    feature count. For with_fs runs, prefers selected-feature CV (step 4) and
    also reads full-feature CV (step 1) for comparison.
    """
    if not run_dir.is_dir():
        return None

    name = run_dir.name
    result: dict = {"name": name, "dir": str(run_dir)}

    # Feature count (from 3_selected_features.json if feature selection was used)
    feat_file = run_dir / "3_selected_features.json"
    if feat_file.exists():
        data = json.loads(feat_file.read_text())
        result["n_features"] = len(data["features"])
        result["selected_features"] = data["features"]
    else:
        result["n_features"] = None
        result["selected_features"] = None

    # CV metrics — prefer selected-feature CV, fall back to full/simple
    result["brier"] = None
    result["accuracy"] = None
    result["correct_ceremonies"] = None
    result["total_ceremonies"] = None
    result["log_loss"] = None
    result["mean_winner_prob"] = None

    for step in ["4_selected_cv", "1_cv", "1_full_cv"]:
        mf = run_dir / step / "metrics.json"
        if mf.exists():
            m = json.loads(mf.read_text())
            result["brier"] = m["micro"]["brier_score"]
            result["accuracy"] = m["micro"]["accuracy"]
            result["log_loss"] = m["micro"]["log_loss"]
            result["mean_winner_prob"] = m["micro"]["mean_winner_prob"]
            result["correct_ceremonies"] = m.get("correct_ceremonies")
            result["total_ceremonies"] = m.get("num_years")
            break

    # For with_fs runs, also read the full-feature CV for comparison
    result["brier_full_cv"] = None
    result["accuracy_full_cv"] = None
    full_cv = run_dir / "1_full_cv" / "metrics.json"
    if full_cv.exists():
        fm = json.loads(full_cv.read_text())
        result["brier_full_cv"] = fm["micro"]["brier_score"]
        result["accuracy_full_cv"] = fm["micro"]["accuracy"]

    return result


def _parse_config_name(name: str) -> dict:
    """Parse a config name into its components.

    Examples:
        lr_full -> {model: "lr", strategy: "full", groups: "all"}
        clogit_additive_3_oscar_nominations -> {model: "clogit", strategy: "additive",
            step: 3, group: "oscar_nominations"}
        gbt_without_critic_scores -> {model: "gbt", strategy: "without",
            group: "critic_scores"}
        cal_sgbt_only_precursor_winners -> {model: "cal_sgbt", strategy: "only",
            group: "precursor_winners"}
    """
    result: dict = {"model": "", "strategy": "", "group": "", "step": None}

    # Find model prefix
    for prefix in MODEL_PREFIXES:
        if name.startswith(prefix + "_"):
            result["model"] = prefix
            rest = name[len(prefix) + 1 :]
            break
    else:
        result["model"] = name.split("_")[0]
        rest = "_".join(name.split("_")[1:])

    # Parse strategy
    if rest == "full":
        result["strategy"] = "full"
        result["group"] = "all"
    elif rest.startswith("without_"):
        result["strategy"] = "without"
        result["group"] = rest[len("without_") :]
    elif rest.startswith("only_"):
        result["strategy"] = "only"
        result["group"] = rest[len("only_") :]
    elif rest.startswith("additive_"):
        result["strategy"] = "additive"
        parts = rest[len("additive_") :].split("_", 1)
        if len(parts) >= 2:
            result["step"] = int(parts[0])
            result["group"] = parts[1]
        else:
            result["group"] = parts[0]
    else:
        result["strategy"] = rest

    return result


# ============================================================================
# Data collection
# ============================================================================


def collect_all_results(exp_dir: Path) -> pd.DataFrame:
    """Collect all ablation results into a single DataFrame.

    Returns DataFrame with columns:
        category, mode, config_name, model, strategy, group, step,
        brier, accuracy, correct_ceremonies, total_ceremonies,
        log_loss, mean_winner_prob, n_features,
        brier_full_cv, accuracy_full_cv
    """
    rows = []

    for category in CATEGORIES:
        cat_dir = exp_dir / category
        if not cat_dir.is_dir():
            continue

        for mode in MODES:
            mode_dir = cat_dir / mode
            if not mode_dir.is_dir():
                continue

            for config_dir in sorted(mode_dir.iterdir()):
                if not config_dir.is_dir():
                    continue

                result = _read_run(config_dir)
                if result is None or result["brier"] is None:
                    continue

                parsed = _parse_config_name(result["name"])

                rows.append(
                    {
                        "category": category,
                        "mode": mode,
                        "config_name": result["name"],
                        "model": parsed["model"],
                        "strategy": parsed["strategy"],
                        "group": parsed["group"],
                        "step": parsed["step"],
                        "brier": result["brier"],
                        "accuracy": result["accuracy"],
                        "correct_ceremonies": result["correct_ceremonies"],
                        "total_ceremonies": result["total_ceremonies"],
                        "log_loss": result["log_loss"],
                        "mean_winner_prob": result["mean_winner_prob"],
                        "n_features": result["n_features"],
                        "brier_full_cv": result["brier_full_cv"],
                        "accuracy_full_cv": result["accuracy_full_cv"],
                    }
                )

    return pd.DataFrame(rows)


# ============================================================================
# Analysis functions
# ============================================================================


def _fmt_correct(row: pd.Series) -> str:
    """Format correct/total ceremonies as 'X/Y'."""
    c = row.get("correct_ceremonies")
    t = row.get("total_ceremonies")
    try:
        if c is not None and t is not None and not np.isnan(c) and not np.isnan(t):
            return f"{int(c)}/{int(t)}"
    except (TypeError, ValueError):
        pass
    return "?"


def _fmt_acc(val: object) -> str:
    """Format accuracy as percentage."""
    try:
        f = float(str(val))  # type: ignore[arg-type]  # val checked at runtime
        if val is not None and not np.isnan(f):
            return f"{f * 100:.1f}%"
    except (TypeError, ValueError):
        pass
    return "?"


def _safe_notna(val: object) -> bool:
    """Scalar-safe check for not-NA. Works on numpy scalars, Python scalars, etc."""
    if val is None:
        return False
    try:
        return not np.isnan(float(str(val)))  # type: ignore[arg-type]  # runtime check
    except (TypeError, ValueError):
        return True  # non-numeric, non-None → present


def _fmt_acc_short(val: object) -> str:
    """Format accuracy as short percentage (no decimal), e.g. '62%'."""
    if _safe_notna(val):
        f = float(str(val))  # type: ignore[arg-type]  # guarded by _safe_notna
        return f"{f * 100:.0f}%"
    return "?"


def print_per_category_summary(df: pd.DataFrame) -> None:
    """Print per-category summary: best model + config for each category, with accuracy."""
    print("\n" + "=" * 120)
    print("PER-CATEGORY BEST CONFIGS (lowest Brier score)")
    print("=" * 120)

    header = (
        f"{'Category':<22s} {'Mode':<8s} {'Model':<10s} "
        f"{'Config':<40s} {'Brier':>8s} {'Acc':>7s} {'Correct':>8s} {'Feat':>5s}"
    )
    print(header)
    print("-" * 120)

    for category in CATEGORIES:
        cat_df = df[df["category"] == category]
        if cat_df.empty:
            print(f"{category:<22s} (no results)")
            continue

        # Best per mode
        for mode in MODES:
            mode_df = cat_df[cat_df["mode"] == mode]
            if mode_df.empty:
                continue
            best_idx = mode_df["brier"].idxmin()
            best: pd.Series = mode_df.loc[best_idx]  # type: ignore[assignment]  # scalar idx → Series

            feat_str = str(int(best["n_features"])) if _safe_notna(best["n_features"]) else "—"
            print(
                f"{category:<22s} {best['mode']:<8s} {best['model']:<10s} "
                f"{best['config_name']:<40s} {best['brier']:>8.4f} "
                f"{_fmt_acc(best['accuracy']):>7s} {_fmt_correct(best):>8s} {feat_str:>5s}"
            )


def print_model_comparison(df: pd.DataFrame) -> None:
    """Print cross-model comparison using full-feature baseline configs."""
    full_df = df[df["strategy"] == "full"]
    if full_df.empty:
        print("\nNo full-feature baselines found.")
        return

    print("\n" + "=" * 120)
    print("MODEL COMPARISON — FULL BASELINE (Brier / Accuracy)")
    print("=" * 120)

    for mode in MODES:
        mode_df = full_df[full_df["mode"] == mode]
        if mode_df.empty:
            continue

        print(f"\n--- {mode} ---")
        header = f"{'Category':<22s}" + "".join(f"  {m:>16s}" for m in MODEL_PREFIXES)
        print(header)
        print("-" * (22 + 18 * len(MODEL_PREFIXES)))

        for category in CATEGORIES:
            cat_df = mode_df[mode_df["category"] == category]
            line = f"{category:<22s}"
            for model in MODEL_PREFIXES:
                model_df = cat_df[cat_df["model"] == model]
                if not model_df.empty:
                    b = model_df.iloc[0]["brier"]
                    a = model_df.iloc[0]["accuracy"]
                    a_str = f"{a * 100:.0f}%" if _safe_notna(a) else "?"
                    line += f"  {b:.4f} ({a_str:>4s})"
                else:
                    line += f"  {'—':>16s}"
            print(line)


def print_additive_ablation(df: pd.DataFrame) -> None:
    """Print additive ablation results per category/model (Brier + accuracy)."""
    additive_df = df[df["strategy"] == "additive"].sort_values(
        ["category", "mode", "model", "step"]
    )
    full_df = df[df["strategy"] == "full"]

    if additive_df.empty:
        print("\nNo additive ablation results found.")
        return

    print("\n" + "=" * 120)
    print("ADDITIVE ABLATION (cumulative group addition) — Brier (Acc%)")
    print("=" * 120)

    for category in CATEGORIES:
        cat_add = additive_df[additive_df["category"] == category]
        cat_full = full_df[full_df["category"] == category]
        if cat_add.empty:
            continue

        for mode in MODES:
            mode_add = cat_add[cat_add["mode"] == mode]
            mode_full = cat_full[cat_full["mode"] == mode]
            if mode_add.empty:
                continue

            # Determine which models have data
            models_present = [m for m in MODEL_PREFIXES if m in mode_add["model"].values]

            print(f"\n  {category} [{mode}]")
            header = f"  {'Step':<4s} {'Group':<25s}" + "".join(
                f"  {MODEL_DISPLAY.get(m, m):>16s}" for m in models_present
            )
            print(header)

            steps = sorted(mode_add["step"].dropna().unique())
            for step in steps:
                step_df = mode_add[mode_add["step"] == step]
                if step_df.empty:
                    continue
                group = step_df.iloc[0]["group"]
                line = f"  {int(step):<4d} {group:<25s}"
                for model in models_present:
                    model_df = step_df[step_df["model"] == model]
                    if not model_df.empty:
                        b = model_df.iloc[0]["brier"]
                        a = model_df.iloc[0]["accuracy"]
                        a_str = f"{a * 100:.0f}" if _safe_notna(a) else "?"
                        line += f"  {b:.4f} ({a_str:>3s}%)"
                    else:
                        line += f"  {'—':>16s}"
                print(line)

            # Print "full" row
            if not mode_full.empty:
                line = f"  {'—':<4s} {'full':<25s}"
                for model in models_present:
                    model_df = mode_full[
                        (mode_full["model"] == model) & (mode_full["mode"] == mode)
                    ]
                    if not model_df.empty:
                        b = model_df.iloc[0]["brier"]
                        a = model_df.iloc[0]["accuracy"]
                        a_str = f"{a * 100:.0f}" if _safe_notna(a) else "?"
                        line += f"  {b:.4f} ({a_str:>3s}%)"
                    else:
                        line += f"  {'—':>16s}"
                print(line)


def print_leave_one_out_impact(df: pd.DataFrame) -> None:
    """Print leave-one-out impact: how much Brier increases when removing each group."""
    without_df = df[df["strategy"] == "without"]
    full_df = df[df["strategy"] == "full"]

    if without_df.empty or full_df.empty:
        print("\nNo leave-one-out results found.")
        return

    print("\n" + "=" * 120)
    print("LEAVE-ONE-OUT IMPACT (Brier delta; positive = removing group hurts = group useful)")
    print("=" * 120)

    for mode in MODES:
        mode_without = without_df[without_df["mode"] == mode]
        mode_full = full_df[full_df["mode"] == mode]
        if mode_without.empty:
            continue

        print(f"\n--- {mode} ---")

        for category in CATEGORIES:
            cat_without = mode_without[mode_without["category"] == category]
            cat_full = mode_full[mode_full["category"] == category]
            if cat_without.empty:
                continue

            models_present = sorted(
                set(cat_without["model"].unique()) & set(cat_full["model"].unique())
            )
            if not models_present:
                continue

            print(f"\n  {category}:")
            header = f"  {'Removed Group':<25s}" + "".join(
                f"  {MODEL_DISPLAY.get(m, m):>12s}" for m in models_present
            )
            print(header)

            groups = sorted(cat_without["group"].unique())
            for group in groups:
                group_df = cat_without[cat_without["group"] == group]
                line = f"  {group:<25s}"
                for model in models_present:
                    model_df = group_df[group_df["model"] == model]
                    full_model = cat_full[
                        (cat_full["category"] == category) & (cat_full["model"] == model)
                    ]
                    if not model_df.empty and not full_model.empty:
                        delta = model_df.iloc[0]["brier"] - full_model.iloc[0]["brier"]
                        line += f"  {delta:>+12.4f}"
                    else:
                        line += f"  {'—':>12s}"
                print(line)


def print_feature_selection_comparison(df: pd.DataFrame) -> None:
    """Compare no_fs vs with_fs: best result per category × model."""
    print("\n" + "=" * 120)
    print("FEATURE SELECTION: BEST no_fs vs BEST with_fs PER CATEGORY × MODEL")
    print("=" * 120)

    header = (
        f"{'Category':<22s} {'Model':<10s} "
        f"{'no_fs Brier':>11s} {'(Acc)':>7s} {'Config':<30s} "
        f"{'with_fs Brier':>12s} {'(Acc)':>7s} {'Config':<30s} "
        f"{'Δ Brier':>8s}"
    )
    print(header)
    print("-" * 150)

    for category in CATEGORIES:
        cat_df = df[df["category"] == category]
        for model in MODEL_PREFIXES:
            model_df = cat_df[cat_df["model"] == model]
            no_fs = model_df[model_df["mode"] == "no_fs"]
            with_fs = model_df[model_df["mode"] == "with_fs"]

            if no_fs.empty and with_fs.empty:
                continue

            nf_str = wf_str = delta_str = "—"
            nf_acc = wf_acc = "—"
            nf_config = wf_config = ""

            if not no_fs.empty:
                best_nf: pd.Series = no_fs.loc[no_fs["brier"].idxmin()]  # type: ignore[assignment]
                nf_str = f"{best_nf['brier']:.4f}"
                nf_acc = _fmt_acc(best_nf["accuracy"])
                nf_config = str(best_nf["config_name"])

            if not with_fs.empty:
                best_wf: pd.Series = with_fs.loc[with_fs["brier"].idxmin()]  # type: ignore[assignment]
                wf_str = f"{best_wf['brier']:.4f}"
                wf_acc = _fmt_acc(best_wf["accuracy"])
                wf_config = str(best_wf["config_name"])

            if not no_fs.empty and not with_fs.empty:
                d = best_wf["brier"] - best_nf["brier"]
                delta_str = f"{d:+.4f}"

            print(
                f"{category:<22s} {model:<10s} "
                f"{nf_str:>11s} {nf_acc:>7s} {nf_config:<30s} "
                f"{wf_str:>12s} {wf_acc:>7s} {wf_config:<30s} "
                f"{delta_str:>8s}"
            )


def print_single_group_ranking(df: pd.DataFrame) -> None:
    """Print single-group standalone ranking per category."""
    only_df = df[df["strategy"] == "only"]
    if only_df.empty:
        print("\nNo single-group results found.")
        return

    print("\n" + "=" * 120)
    print("SINGLE-GROUP STANDALONE RANKING (lower Brier = more useful alone)")
    print("=" * 120)

    for mode in MODES:
        mode_df = only_df[only_df["mode"] == mode]
        if mode_df.empty:
            continue

        print(f"\n--- {mode} ---")

        for category in CATEGORIES:
            cat_df = mode_df[mode_df["category"] == category]
            if cat_df.empty:
                continue

            models_present = sorted(cat_df["model"].unique())

            print(f"\n  {category}:")
            header = f"  {'Group':<25s}" + "".join(
                f"  {MODEL_DISPLAY.get(m, m):>16s}" for m in models_present
            )
            print(header)

            # Sort groups by average Brier across models
            group_avg = cat_df.groupby("group")["brier"].mean().sort_values()
            for group in group_avg.index:
                group_df = cat_df[cat_df["group"] == group]
                line = f"  {group:<25s}"
                for model in models_present:
                    model_df = group_df[group_df["model"] == model]
                    if not model_df.empty:
                        b = model_df.iloc[0]["brier"]
                        a = model_df.iloc[0]["accuracy"]
                        a_str = f"{a * 100:.0f}%" if _safe_notna(a) else "?"
                        line += f"  {b:.4f} ({a_str:>4s})"
                    else:
                        line += f"  {'—':>16s}"
                print(line)


def print_feature_group_importance_summary(df: pd.DataFrame) -> None:
    """Print cross-category feature group importance ranking.

    Uses leave-one-out Brier delta averaged across models as the importance metric.
    Positive delta = removing the group hurts (important).
    """
    without_df = df[(df["strategy"] == "without") & (df["mode"] == "no_fs")]
    full_df = df[(df["strategy"] == "full") & (df["mode"] == "no_fs")]

    if without_df.empty or full_df.empty:
        print("\nInsufficient data for importance summary.")
        return

    print("\n" + "=" * 100)
    print("CROSS-CATEGORY FEATURE GROUP IMPORTANCE")
    print("(LOO Brier delta averaged across models; higher = more important)")
    print("=" * 100)

    # Compute delta for each category × model × group
    deltas = []
    for _, row in without_df.iterrows():
        full_match = full_df[
            (full_df["category"] == row["category"])
            & (full_df["model"] == row["model"])
            & (full_df["mode"] == row["mode"])
        ]
        if not full_match.empty:
            delta = row["brier"] - full_match.iloc[0]["brier"]
            deltas.append(
                {
                    "category": row["category"],
                    "model": row["model"],
                    "group": row["group"],
                    "delta_brier": delta,
                }
            )

    if not deltas:
        print("No delta data computed.")
        return

    delta_df = pd.DataFrame(deltas)

    # Average across models per category × group
    cat_group_avg = delta_df.groupby(["category", "group"])["delta_brier"].mean().reset_index()

    # Pivot: rows = groups, columns = categories
    pivot = cat_group_avg.pivot(index="group", columns="category", values="delta_brier")
    pivot = pivot.reindex(columns=[c for c in CATEGORIES if c in pivot.columns])

    # Add mean column and sort by it
    pivot["MEAN"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("MEAN", ascending=False)

    # Print
    short_cols = [CATEGORY_SHORT.get(c, c[:8]) for c in pivot.columns[:-1]] + ["MEAN"]
    header = f"{'Group':<25s}" + "".join(f"  {c:>10s}" for c in short_cols)
    print(header)
    print("-" * len(header))

    for group, row in pivot.iterrows():
        line = f"{group:<25s}"
        for val in row:
            if _safe_notna(val):
                line += f"  {val:>+10.4f}"
            else:
                line += f"  {'—':>10s}"
        print(line)


# ============================================================================
# Plotting functions
# ============================================================================


def plot_category_difficulty(df: pd.DataFrame, output_dir: Path) -> Path:
    """Bar chart: best Brier + accuracy per category, colored by winning model.

    Shows both metrics side by side for each category.
    """
    # Use best result across all modes and models
    best_rows = []
    for cat in CATEGORIES:
        cat_df = df[df["category"] == cat]
        if cat_df.empty:
            continue
        best_idx = cat_df["brier"].idxmin()
        best_rows.append(cat_df.loc[best_idx])
    if not best_rows:
        return output_dir / "category_difficulty.png"

    best_df = pd.DataFrame(best_rows).sort_values("brier")

    fig, ax1 = plt.subplots(figsize=(12, 5))

    cats = [CATEGORY_SHORT.get(c, c) for c in best_df["category"]]
    briers: np.ndarray = np.asarray(best_df["brier"].values)
    accs: np.ndarray = np.asarray(best_df["accuracy"].values) * 100
    colors = [get_model_color(m) for m in best_df["model"]]

    x = np.arange(len(cats))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, briers, width, color=colors, alpha=0.85, label="Brier Score")
    ax1.set_ylabel("Brier Score (lower = better)", fontsize=11)
    ax1.set_ylim(0, max(briers) * 1.3)

    # Add value labels on bars
    for bar, b in zip(bars1, briers, strict=True):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{b:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        x + width / 2, accs, width, color=colors, alpha=0.4, hatch="//", label="Accuracy %"
    )
    ax2.set_ylabel("Accuracy % (higher = better)", fontsize=11)
    ax2.set_ylim(0, 105)

    for bar, a in zip(bars2, accs, strict=True):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{a:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(cats, rotation=30, ha="right")
    ax1.set_title("Category Difficulty — Best Brier & Accuracy (any model/mode)", fontsize=13)

    # Legend for models
    from matplotlib.patches import Patch

    legend_patches = [
        Patch(facecolor=get_model_color(m), label=MODEL_DISPLAY[m])
        for m in MODEL_PREFIXES
        if m in best_df["model"].values
    ]
    ax1.legend(handles=legend_patches, loc="upper left", title="Best Model")

    plt.tight_layout()
    out_path = output_dir / "category_difficulty.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


def plot_additive_ablation(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Line plots: Brier score vs additive step for each category (3×3 grid)."""
    additive_df = df[(df["strategy"] == "additive") & (df["mode"] == "no_fs")]
    full_df = df[(df["strategy"] == "full") & (df["mode"] == "no_fs")]

    if additive_df.empty:
        return []

    paths = []
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes_flat = axes.flatten()

    for idx, category in enumerate(CATEGORIES):
        ax = axes_flat[idx]
        cat_add = additive_df[additive_df["category"] == category]
        cat_full = full_df[full_df["category"] == category]

        if cat_add.empty:
            ax.set_visible(False)
            continue

        models_in_cat = sorted(cat_add["model"].unique())
        max_step = int(cat_add["step"].max())

        for model in models_in_cat:
            model_add = cat_add[cat_add["model"] == model].sort_values("step")
            steps = model_add["step"].values
            briers = model_add["brier"].values

            color = get_model_color(model)
            label = MODEL_DISPLAY.get(model, model)
            ax.plot(steps, briers, "o-", color=color, label=label, markersize=4, linewidth=1.5)

            # Add full model point
            model_full = cat_full[(cat_full["category"] == category) & (cat_full["model"] == model)]
            if not model_full.empty:
                fb = model_full.iloc[0]["brier"]
                ax.plot(max_step + 1, fb, "s", color=color, markersize=6, alpha=0.6)

        ax.set_title(CATEGORY_SHORT.get(category, category), fontsize=11, fontweight="bold")
        ax.set_xlabel("Additive Step")
        ax.set_ylabel("Brier Score")
        ax.grid(True, alpha=0.3)

        xticks = list(range(1, max_step + 1)) + [max_step + 1]
        xticklabels = [str(i) for i in range(1, max_step + 1)] + ["full"]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=8)

        if idx == 0:
            ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("Additive Feature Ablation — Brier Score by Step (no_fs)", fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = output_dir / "additive_ablation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")
    paths.append(out_path)
    return paths


def plot_feature_selection_effect(df: pd.DataFrame, output_dir: Path) -> Path:
    """Grouped bar chart: best no_fs vs best with_fs Brier per category × model."""
    cats_with_both = []
    for cat in CATEGORIES:
        cat_df = df[df["category"] == cat]
        has_nfs = not cat_df[cat_df["mode"] == "no_fs"].empty
        has_wfs = not cat_df[cat_df["mode"] == "with_fs"].empty
        if has_nfs and has_wfs:
            cats_with_both.append(cat)

    if not cats_with_both:
        return output_dir / "feature_selection_effect.png"

    # Collect best per category × model × mode
    rows = []
    for cat in cats_with_both:
        cat_df = df[df["category"] == cat]
        for model in MODEL_PREFIXES:
            for mode in MODES:
                subset = cat_df[(cat_df["model"] == model) & (cat_df["mode"] == mode)]
                if subset.empty:
                    continue
                best = subset.loc[subset["brier"].idxmin()]
                rows.append(
                    {
                        "category": cat,
                        "model": model,
                        "mode": mode,
                        "brier": best["brier"],
                        "accuracy": best["accuracy"],
                    }
                )

    plot_df = pd.DataFrame(rows)

    # Focus on clogit and cal_sgbt (most data)
    focus_models = [m for m in ["clogit", "cal_sgbt"] if m in plot_df["model"].values]

    fig, axes = plt.subplots(1, len(focus_models), figsize=(7 * len(focus_models), 5))
    if len(focus_models) == 1:
        axes = [axes]

    for ax, model in zip(axes, focus_models, strict=True):
        model_df = plot_df[plot_df["model"] == model]
        cats_present = [c for c in cats_with_both if c in model_df["category"].values]

        x = np.arange(len(cats_present))
        width = 0.35

        nfs_briers = []
        wfs_briers = []
        for cat in cats_present:
            nfs = model_df[(model_df["category"] == cat) & (model_df["mode"] == "no_fs")]
            wfs = model_df[(model_df["category"] == cat) & (model_df["mode"] == "with_fs")]
            nfs_briers.append(nfs.iloc[0]["brier"] if not nfs.empty else 0)
            wfs_briers.append(wfs.iloc[0]["brier"] if not wfs.empty else 0)

        nfs_arr = np.array(nfs_briers)
        wfs_arr = np.array(wfs_briers)

        ax.bar(x - width / 2, nfs_arr, width, label="no_fs", color="#4c72b0", alpha=0.8)
        ax.bar(x + width / 2, wfs_arr, width, label="with_fs", color="#dd8452", alpha=0.8)

        # Add delta labels
        for i, (n, w) in enumerate(zip(nfs_arr, wfs_arr, strict=True)):
            if n > 0 and w > 0:
                delta = w - n
                color = "green" if delta < 0 else "red"
                ax.text(
                    x[i] + width / 2,
                    w + 0.001,
                    f"{delta:+.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=color,
                    fontweight="bold",
                )

        cat_labels = [CATEGORY_SHORT.get(c, c) for c in cats_present]
        ax.set_xticks(x)
        ax.set_xticklabels(cat_labels, rotation=30, ha="right")
        ax.set_ylabel("Brier Score")
        ax.set_title(f"{MODEL_DISPLAY.get(model, model)} — Feature Selection Effect", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = output_dir / "feature_selection_effect.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


def plot_loo_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    """Heatmap: LOO Brier delta for precursor_winners across categories × models."""
    without_df = df[(df["strategy"] == "without") & (df["mode"] == "no_fs")]
    full_df = df[(df["strategy"] == "full") & (df["mode"] == "no_fs")]

    if without_df.empty or full_df.empty:
        return output_dir / "loo_precursor_winners.png"

    # Compute deltas for all groups
    deltas = []
    for _, row in without_df.iterrows():
        full_match = full_df[
            (full_df["category"] == row["category"]) & (full_df["model"] == row["model"])
        ]
        if not full_match.empty:
            delta = row["brier"] - full_match.iloc[0]["brier"]
            deltas.append(
                {
                    "category": row["category"],
                    "model": row["model"],
                    "group": row["group"],
                    "delta": delta,
                }
            )

    if not deltas:
        return output_dir / "loo_precursor_winners.png"

    delta_df = pd.DataFrame(deltas)

    # Focus on precursor_winners (the dominant group)
    pw_df = delta_df[delta_df["group"] == "precursor_winners"]
    if pw_df.empty:
        return output_dir / "loo_precursor_winners.png"

    pivot = pw_df.pivot(index="category", columns="model", values="delta")
    # Reorder
    cat_order = [c for c in CATEGORIES if c in pivot.index]
    model_order = [m for m in MODEL_PREFIXES if m in pivot.columns]
    pivot = pivot.reindex(index=cat_order, columns=model_order)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in model_order])
    ax.set_yticks(range(len(cat_order)))
    ax.set_yticklabels([CATEGORY_SHORT.get(c, c) for c in cat_order])

    # Annotate cells
    for i in range(len(cat_order)):
        for j in range(len(model_order)):
            val = pivot.values[i, j]
            if np.isnan(val):
                continue
            text_color = "white" if val > 0.06 else "black"
            ax.text(
                j,
                i,
                f"{val:+.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                fontweight="bold",
            )

    ax.set_title(
        "LOO Impact of Removing precursor_winners\n(Brier increase; larger = more important)",
        fontsize=12,
    )
    plt.colorbar(im, ax=ax, label="Brier Δ")
    plt.tight_layout()

    out_path = output_dir / "loo_precursor_winners.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


def plot_brier_accuracy_scatter(df: pd.DataFrame, output_dir: Path) -> Path:
    """Scatter plot: Brier vs accuracy across all runs, colored by model."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model in MODEL_PREFIXES:
        model_df = df[(df["model"] == model) & (df["mode"] == "no_fs")]
        if model_df.empty:
            continue
        ax.scatter(
            model_df["brier"],
            model_df["accuracy"] * 100,
            c=get_model_color(model),
            label=MODEL_DISPLAY.get(model, model),
            alpha=0.4,
            s=20,
            edgecolors="none",
        )

    ax.set_xlabel("Brier Score (lower = better)", fontsize=11)
    ax.set_ylabel("Accuracy % (higher = better)", fontsize=11)
    ax.set_title("Brier vs Accuracy — All Runs (no_fs)", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "brier_vs_accuracy.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


def generate_all_plots(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Generate all plots and return paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating plots in {output_dir}/...")

    paths = []
    paths.append(plot_category_difficulty(df, output_dir))
    paths.extend(plot_additive_ablation(df, output_dir))
    paths.append(plot_feature_selection_effect(df, output_dir))
    paths.append(plot_loo_heatmap(df, output_dir))
    paths.append(plot_brier_accuracy_scatter(df, output_dir))

    print(f"  Generated {len(paths)} plots")
    return paths


# ============================================================================
# Summary CSV
# ============================================================================


def write_summary(df: pd.DataFrame, output_path: Path) -> None:
    """Write all results to CSV and JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nWrote {len(df)} rows to {output_path}")

    json_path = output_path.with_suffix(".json")
    df.to_json(json_path, orient="records", indent=2)
    print(f"Wrote {json_path}")


# ============================================================================
# Main
# ============================================================================

ALL_SECTIONS = [
    "summary",
    "model_comparison",
    "additive",
    "leave_one_out",
    "single_group",
    "importance",
    "feature_selection",
    "plots",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze multi-category feature ablation results")
    parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Path to experiment directory (storage/d20260220_feature_ablation)",
    )
    parser.add_argument(
        "--section",
        type=str,
        nargs="*",
        choices=ALL_SECTIONS + ["all"],
        default=["all"],
        help="Which analysis sections to show (default: all)",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory for plot output (default: {exp-dir}/plots)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    exp_dir = Path(args.exp_dir)
    plot_dir = Path(args.plot_dir) if args.plot_dir else exp_dir / "plots"

    sections = args.section if "all" not in args.section else ALL_SECTIONS

    # Collect all results
    print("Collecting results...")
    df = collect_all_results(exp_dir)
    if df.empty:
        print("No results found.")
        return

    n_cats = df["category"].nunique()
    n_modes = df["mode"].nunique()
    n_models = df["model"].nunique()
    print(f"Found {len(df)} results across {n_cats} categories, {n_modes} modes, {n_models} models")

    # Run requested sections
    if "summary" in sections:
        print_per_category_summary(df)

    if "model_comparison" in sections:
        print_model_comparison(df)

    if "additive" in sections:
        print_additive_ablation(df)

    if "leave_one_out" in sections:
        print_leave_one_out_impact(df)

    if "single_group" in sections:
        print_single_group_ranking(df)

    if "importance" in sections:
        print_feature_group_importance_summary(df)

    if "feature_selection" in sections:
        print_feature_selection_comparison(df)

    if "plots" in sections:
        generate_all_plots(df, plot_dir)

    # Always write summary
    summary_dir = exp_dir / "summary"
    write_summary(df, summary_dir / "phase1_results.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
