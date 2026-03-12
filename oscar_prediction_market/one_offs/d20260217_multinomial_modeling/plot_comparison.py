"""Model comparison visualisation.

Plots that compare predictions or probability distributions across model
types, including side-by-side bar charts and probability anatomy panels.

- :func:`plot_final_predictions_comparison` — grouped bar chart of test-year
  predictions across models.
- :func:`plot_prob_distribution_anatomy` — 3-panel (prob-sum, top-1 share,
  entropy) across snapshot dates.
- :func:`plot_binary_vs_multinomial` — side-by-side horizontal bars for two
  model types over the last 4 snapshots.

Usage::

    from oscar_prediction_market.one_offs.d20260217_multinomial_modeling.plot_comparison import (
        plot_final_predictions_comparison,
        plot_prob_distribution_anatomy,
        plot_binary_vs_multinomial,
    )
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oscar_prediction_market.one_offs.analysis_utils.style import (
    AWARDS_SEASON_EVENTS,
    apply_style,
    get_model_color,
    get_model_display,
)

apply_style()


# ============================================================================
# Final predictions comparison
# ============================================================================


def plot_final_predictions_comparison(
    combined: pd.DataFrame,
    snapshot_date: str,
    output_path: Path | None = None,
) -> plt.Figure:
    """Bar chart comparing test-year predictions across model types.

    Args:
        combined: DataFrame with columns = model display names,
            index = nominee titles.
        snapshot_date: For the plot title.
        output_path: If provided, save figure.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(combined))
    width = 0.8 / len(combined.columns)

    for i, col in enumerate(combined.columns):
        ax.bar(x + i * width, np.asarray(combined[col].values), width, label=col, alpha=0.85)

    ax.set_xticks(x + width * (len(combined.columns) - 1) / 2)
    ax.set_xticklabels(combined.index, rotation=45, ha="right")
    ax.set_ylabel("Predicted Probability")
    ax.set_title(f"2026 Best Picture Predictions (Snapshot: {snapshot_date})", fontsize=14)
    ax.legend(fontsize=9)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


# ============================================================================
# Probability distribution anatomy
# ============================================================================


def plot_prob_distribution_anatomy(
    preds_df: pd.DataFrame,
    model_types: list[str],
    test_year: int = 2026,
    output_path: Path | None = None,
) -> plt.Figure:
    """3-panel: prob-sum, top-1 share, entropy over time per model type.

    Args:
        preds_df: All test predictions with columns ``year``,
            ``model_type``, ``snapshot_date``, ``probability``.
        model_types: Model types to include.
        test_year: Which year to analyse.
        output_path: If provided, save figure.

    Returns:
        matplotlib Figure.
    """
    test = preds_df[preds_df["year"] == test_year].copy()
    if test.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return fig

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for mt in model_types:
        mt_data = test[test["model_type"] == mt]
        dates = sorted(mt_data["snapshot_date"].unique())

        prob_sums: list[float] = []
        top1_shares: list[float] = []
        entropies: list[float] = []

        for d in dates:
            snap = mt_data[mt_data["snapshot_date"] == d]
            probs = snap["probability"].values
            psum = float(probs.sum())
            prob_sums.append(psum)

            p_norm = probs / psum if psum > 0 else probs
            top1_shares.append(float(p_norm.max()))

            p_pos = p_norm[p_norm > 0]
            entropy = float(-np.sum(p_pos * np.log2(p_pos)))
            entropies.append(entropy)

        color = get_model_color(mt)
        display = get_model_display(mt)
        axes[0].plot(dates, prob_sums, "o-", color=color, label=display)
        axes[1].plot(dates, top1_shares, "o-", color=color, label=display)
        axes[2].plot(dates, entropies, "o-", color=color, label=display)

    axes[0].axhline(1.0, color="red", linestyle="--", alpha=0.5, label="Sum=1.0")
    axes[0].set_ylabel("Probability Sum")
    axes[0].set_title("(a) Probability Sum Over Time", fontsize=13)
    axes[0].legend(fontsize=9)
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].set_ylabel("Top-1 Share")
    axes[1].set_title("(b) Frontrunner Concentration Over Time", fontsize=13)
    axes[1].legend(fontsize=9)
    axes[1].tick_params(axis="x", rotation=45)

    axes[2].set_ylabel("Entropy (bits)")
    axes[2].set_title("(c) Prediction Entropy Over Time", fontsize=13)
    axes[2].legend(fontsize=9)
    axes[2].tick_params(axis="x", rotation=45)

    fig.suptitle("Probability Distribution Anatomy", fontsize=15)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


# ============================================================================
# Binary vs multinomial comparison
# ============================================================================


def plot_binary_vs_multinomial(
    preds_df: pd.DataFrame,
    model_a: str,
    model_b: str,
    test_year: int = 2026,
    output_path: Path | None = None,
) -> plt.Figure:
    """Side-by-side comparison of two model types' probability assignments.

    Shows the last 4 snapshots as horizontal bar charts.

    Args:
        preds_df: All test predictions.
        model_a: First model type (e.g. ``"lr"``).
        model_b: Second model type (e.g. ``"conditional_logit"``).
        test_year: Which year to analyse.
        output_path: If provided, save figure.

    Returns:
        matplotlib Figure.
    """
    test = preds_df[preds_df["year"] == test_year].copy()
    if test.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return fig

    dates = sorted(test["snapshot_date"].unique())
    show_dates = dates[-4:]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat = axes.flatten()

    for idx, d in enumerate(show_dates):
        ax = axes_flat[idx]
        a_df = test[(test["model_type"] == model_a) & (test["snapshot_date"] == d)].sort_values(
            "probability", ascending=False
        )
        b_df = test[(test["model_type"] == model_b) & (test["snapshot_date"] == d)].sort_values(
            "probability", ascending=False
        )

        if a_df.empty or b_df.empty:
            continue

        merged = (
            a_df[["title", "probability"]]
            .merge(b_df[["title", "probability"]], on="title", suffixes=("_a", "_b"))
            .sort_values("probability_a", ascending=False)
        )

        nominees = [t[:18] for t in merged["title"]]
        x = np.arange(len(nominees))
        width = 0.35

        color_a = get_model_color(model_a)
        color_b = get_model_color(model_b)
        display_a = get_model_display(model_a)
        display_b = get_model_display(model_b)

        ax.barh(
            x - width / 2,
            merged["probability_a"] * 100,
            width,
            color=color_a,
            alpha=0.8,
            label=display_a,
        )
        ax.barh(
            x + width / 2,
            merged["probability_b"] * 100,
            width,
            color=color_b,
            alpha=0.8,
            label=display_b,
        )

        ax.set_yticks(x)
        ax.set_yticklabels(nominees)
        ax.set_xlabel("Probability (%)")
        event = AWARDS_SEASON_EVENTS.get(d, "")
        ax.set_title(f"{d} ({event})", fontsize=12)
        ax.legend(fontsize=9)
        ax.invert_yaxis()

        # Annotate prob sums
        a_sum = merged["probability_a"].sum()
        b_sum = merged["probability_b"].sum()
        ax.text(
            0.98,
            0.02,
            f"{display_a} sum={a_sum:.2f}\n{display_b} sum={b_sum:.2f}",
            transform=ax.transAxes,
            fontsize=9,
            ha="right",
            va="bottom",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

    display_a = get_model_display(model_a)
    display_b = get_model_display(model_b)
    fig.suptitle(f"{display_a} vs {display_b}: Probability Assignments", fontsize=15)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig
