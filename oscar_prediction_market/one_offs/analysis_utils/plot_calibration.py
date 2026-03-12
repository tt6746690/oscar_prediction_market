"""Calibration and probability sum visualisation.

- :func:`plot_reliability_diagrams` — calibration reliability bar chart,
  one panel per model type.
- :func:`plot_prob_sum_distribution` — histogram of per-ceremony probability
  sums (should cluster around 1.0 for well-calibrated models).

Usage::

    from oscar_prediction_market.one_offs.analysis_utils.plot_calibration import (
        plot_reliability_diagrams,
        plot_prob_sum_distribution,
    )
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oscar_prediction_market.one_offs.analysis_utils.style import (
    apply_style,
    get_model_color,
    get_model_display,
)

apply_style()


# ============================================================================
# Reliability diagrams
# ============================================================================


def plot_reliability_diagrams(
    cv_predictions: dict[str, pd.DataFrame],
    output_path: Path | None = None,
    n_bins: int = 10,
) -> plt.Figure:
    """Calibration reliability diagram from CV predictions, one panel per model.

    Args:
        cv_predictions: ``{model_type: DataFrame}`` where each DataFrame has
            columns ``probability`` and ``is_actual_winner``.
        output_path: If provided, save figure to this path.
        n_bins: Number of probability bins.

    Returns:
        matplotlib Figure.
    """
    n_models = len(cv_predictions)
    ncols = min(n_models, 3)
    nrows = math.ceil(n_models / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
    if n_models == 1:
        axes_flat = [axes]
    elif nrows == 1:
        axes_flat = list(axes)
    else:
        axes_flat = axes.flatten()

    for idx, (model_type, cv_df) in enumerate(cv_predictions.items()):
        ax = axes_flat[idx]
        probs = cv_df["probability"].values
        actuals = cv_df["is_actual_winner"].astype(int).values

        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers: list[float] = []
        bin_freqs: list[float] = []
        bin_counts: list[int] = []

        for i in range(n_bins):
            if i == n_bins - 1:
                mask = (probs >= bins[i]) & (probs <= bins[i + 1])
            else:
                mask = (probs >= bins[i]) & (probs < bins[i + 1])
            count = int(mask.sum())
            if count > 0:
                bin_centers.append(float((bins[i] + bins[i + 1]) / 2))
                bin_freqs.append(float(actuals[mask].mean()))
                bin_counts.append(count)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        color = get_model_color(model_type)
        ax.bar(
            bin_centers,
            bin_freqs,
            width=1 / n_bins * 0.8,
            alpha=0.6,
            color=color,
            edgecolor="white",
        )
        ax.scatter(bin_centers, bin_freqs, color="red", zorder=3, s=25)

        # Annotate counts on bars
        for c, f, n in zip(bin_centers, bin_freqs, bin_counts, strict=True):
            if n > 0:
                ax.text(c, f + 0.03, str(n), ha="center", fontsize=8, alpha=0.7)

        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Frequency")
        display = get_model_display(model_type)
        ax.set_title(f"{display} (n={len(probs)})")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Hide unused axes
    for j in range(len(cv_predictions), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Calibration Reliability Diagrams", fontsize=15, y=1.01)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


# ============================================================================
# Probability sum distribution
# ============================================================================


def plot_prob_sum_distribution(
    cv_prob_sums: dict[str, list[float]],
    output_path: Path | None = None,
) -> plt.Figure:
    """Histogram of per-ceremony probability sums per model type.

    Args:
        cv_prob_sums: ``{model_type: [prob_sum_per_ceremony]}``.
        output_path: If provided, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    n_models = len(cv_prob_sums)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (model_type, sums) in zip(axes, cv_prob_sums.items(), strict=True):
        color = get_model_color(model_type)
        display = get_model_display(model_type)

        if sums:
            data_range = max(sums) - min(sums) if len(sums) > 1 else 0
            n_hist_bins = min(20, max(1, int(data_range / 0.01))) if data_range > 0 else 1
            ax.hist(sums, bins=n_hist_bins, color=color, alpha=0.7, edgecolor="white")
            ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.8, label="Sum = 1.0")
            ax.axvline(
                x=float(np.mean(sums)),
                color="navy",
                linestyle="--",
                alpha=0.8,
                label=f"Mean = {np.mean(sums):.2f}",
            )

        ax.set_title(display)
        ax.set_xlabel("Probability Sum")
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Count")  # type: ignore[union-attr]
    fig.suptitle("Per-Ceremony Probability Sum Distribution (CV Predictions)", fontsize=14)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig
