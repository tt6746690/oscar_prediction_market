"""Temporal metrics and feature importance visualisation.

Plots that show how model performance and feature importances evolve
across snapshot dates during the awards season.

- :func:`plot_metrics_over_time` — 5-panel: accuracy, Brier, log-loss,
  prob-sum, winner-prob over snapshot dates.
- :func:`plot_metrics_over_time_detailed` — 4-panel (from pre-computed
  metrics dict).
- :func:`plot_feature_importance_heatmap` — feature × date heatmap.

Usage::

    from oscar_prediction_market.one_offs.analysis_utils.plot_temporal import (
        plot_metrics_over_time,
        plot_feature_importance_heatmap,
    )
"""

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
# Metrics over time
# ============================================================================


def plot_metrics_over_time(
    per_snapshot: pd.DataFrame,
    model_types: list[str],
    output_path: Path | None = None,
) -> plt.Figure:
    """Multi-panel plot of CV metrics across snapshot dates.

    Args:
        per_snapshot: DataFrame with columns ``model_type``,
            ``snapshot_date``, ``accuracy``, ``brier_score``, ``log_loss``,
            ``prob_sum_mean``, ``mean_winner_prob``.
        model_types: Model types to include.
        output_path: If provided, save figure.

    Returns:
        matplotlib Figure.
    """
    metrics_to_plot = [
        ("accuracy", "Accuracy", True),
        ("brier_score", "Brier Score", False),
        ("log_loss", "Log Loss", False),
        ("prob_sum_mean", "Mean Prob Sum", None),
        ("mean_winner_prob", "Mean Winner Prob", True),
    ]

    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(14, 4.5 * len(metrics_to_plot)))

    for ax, (metric, label, _higher_better) in zip(axes, metrics_to_plot, strict=True):
        for model_type in model_types:
            model_data = per_snapshot[per_snapshot["model_type"] == model_type]
            if model_data.empty:
                continue
            color = get_model_color(model_type)
            display = get_model_display(model_type)
            ax.plot(
                model_data["snapshot_date"],
                model_data[metric],
                marker="o",
                label=display,
                color=color,
            )

        if metric == "prob_sum_mean":
            ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="Ideal (1.0)")

        ax.set_ylabel(label)
        ax.legend(fontsize=9)
        ax.tick_params(axis="x", rotation=45)

    axes[0].set_title("CV Metrics Across Snapshot Dates", fontsize=14)
    axes[-1].set_xlabel("Snapshot Date")

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


def plot_metrics_over_time_detailed(
    all_cv_metrics: dict[str, dict[str, dict[str, float]]],
    model_types: list[str],
    output_path: Path | None = None,
) -> plt.Figure:
    """4-panel plot: accuracy, Brier, log-loss, winner_prob over time.

    Args:
        all_cv_metrics: ``{model_type: {date_str: {metric_name: value}}}``.
        model_types: Model types to include.
        output_path: If provided, save figure.

    Returns:
        matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    metric_configs = [
        ("accuracy", "CV Accuracy (%)", lambda x: x * 100),
        ("brier", "CV Brier Score", lambda x: x),
        ("log_loss", "CV Log-Loss", lambda x: x),
        ("winner_prob", "Winner Probability", lambda x: x),
    ]

    for panel_idx, (metric, title, transform) in enumerate(metric_configs):
        ax = axes.flatten()[panel_idx]
        for mt in model_types:
            if mt not in all_cv_metrics:
                continue
            dates = sorted(all_cv_metrics[mt].keys())
            values = [transform(all_cv_metrics[mt][d].get(metric, 0)) for d in dates]
            color = get_model_color(mt)
            display = get_model_display(mt)
            ax.plot(dates, values, "o-", color=color, label=display)

        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("CV Metrics Evolution Across Awards Season", fontsize=15)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


# ============================================================================
# Feature importance heatmap
# ============================================================================


def plot_feature_importance_heatmap(
    importances: dict[str, dict[str, float]],
    model_display_name: str,
    top_n: int = 15,
    output_path: Path | None = None,
) -> plt.Figure:
    """Heatmap of top feature importances evolving across snapshots.

    Args:
        importances: ``{date_str: {feature_name: importance_value}}``.
        model_display_name: Name for the plot title.
        top_n: Number of top features to show.
        output_path: If provided, save figure.

    Returns:
        matplotlib Figure.
    """
    if not importances:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return fig

    # Find top features by max importance across all dates
    all_feats: dict[str, float] = {}
    for imps in importances.values():
        for feat, imp in imps.items():
            all_feats[feat] = max(all_feats.get(feat, 0), imp)

    top_features = sorted(all_feats, key=lambda f: all_feats[f], reverse=True)[:top_n]
    dates = sorted(importances.keys())

    matrix = np.zeros((len(top_features), len(dates)))
    for j, d in enumerate(dates):
        for i, feat in enumerate(top_features):
            matrix[i, j] = importances[d].get(feat, 0)

    fig, ax = plt.subplots(figsize=(14, max(6, len(top_features) * 0.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=45, ha="right")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_title(f"Feature Importance Evolution — {model_display_name}", fontsize=14)
    plt.colorbar(im, ax=ax, label="Importance", shrink=0.8)

    # Annotate cells
    for i in range(len(top_features)):
        for j in range(len(dates)):
            val = matrix[i, j]
            if val > 0:
                color = "white" if val > matrix.max() * 0.6 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7, color=color)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig
