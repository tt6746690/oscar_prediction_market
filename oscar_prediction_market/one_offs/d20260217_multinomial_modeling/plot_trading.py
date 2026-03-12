"""Trading backtest visualisation.

Plots for analysing backtest results: wealth curves, settlement heatmaps,
position evolution, edge distributions, and model-vs-market comparisons.

All functions are stateless — they take DataFrames / dicts and return
``Figure`` objects.  No data loading or market queries inside.

Usage::

    from oscar_prediction_market.one_offs.d20260217_multinomial_modeling.plot_trading import (
        plot_model_vs_market,
        plot_divergence_heatmaps,
        plot_wealth_curves,
        plot_settlement_heatmap,
        plot_edge_distributions,
    )
"""

import math
from pathlib import Path

import matplotlib
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
# Model vs Market
# ============================================================================


def plot_model_vs_market(
    mm_df: pd.DataFrame,
    model_types: list[str],
    output_path: Path | None = None,
) -> plt.Figure:
    """Per-nominee subplot: model prob vs market price, all model types overlaid.

    Args:
        mm_df: DataFrame with columns: title, snapshot_date, model_type,
            model_prob, market_prob
        model_types: List of model types to plot.
        output_path: If provided, save figure.

    Returns:
        matplotlib Figure
    """
    valid = mm_df.dropna(subset=["market_prob"])
    if valid.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No market data", transform=ax.transAxes, ha="center")
        return fig

    nominees = sorted(valid["title"].unique())
    ncols = 3
    nrows = math.ceil(len(nominees) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.5 * nrows), sharex=True)
    if nrows == 1 and ncols == 1:
        axes_flat = [axes]
    elif nrows == 1:
        axes_flat = list(axes)
    else:
        axes_flat = axes.flatten()

    for i, nominee in enumerate(nominees):
        ax = axes_flat[i]
        nom_data = valid[valid["title"] == nominee]

        # Market prices (use first model type for dates)
        mkt = nom_data[nom_data["model_type"] == model_types[0]].sort_values("snapshot_date")
        if not mkt.empty:
            dates = mkt["snapshot_date"].tolist()
            mkt_probs = mkt["market_prob"].values * 100
            ax.plot(dates, mkt_probs, "k--", alpha=0.6, linewidth=2.5, label="Market")

        for mt in model_types:
            mt_data = nom_data[nom_data["model_type"] == mt].sort_values("snapshot_date")
            if mt_data.empty:
                continue
            color = get_model_color(mt)
            display = get_model_display(mt)
            ax.plot(
                mt_data["snapshot_date"].tolist(),
                mt_data["model_prob"].values * 100,
                "o-",
                color=color,
                markersize=4,
                label=display,
            )

        short = nominee[:22] + "..." if len(nominee) > 22 else nominee
        ax.set_title(short, fontsize=11)
        ax.tick_params(axis="x", rotation=45)
        if i % ncols == 0:
            ax.set_ylabel("Probability (%)")
        if i == 0:
            ax.legend(fontsize=8, loc="upper left")

    for j in range(len(nominees), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Model Probability vs Market Price per Nominee", fontsize=15)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


def plot_divergence_heatmaps(
    mm_df: pd.DataFrame,
    model_types: list[str],
    output_path: Path | None = None,
) -> plt.Figure:
    """Nominee x date heatmap of model-market divergence (pp) per model type.

    Args:
        mm_df: DataFrame with columns: title, snapshot_date, model_type, divergence
        model_types: List of model types (one panel per type).
        output_path: If provided, save figure.

    Returns:
        matplotlib Figure
    """
    valid = mm_df.dropna(subset=["market_prob"])
    if valid.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No market data", transform=ax.transAxes, ha="center")
        return fig

    n_types = len(model_types)
    ncols = min(n_types, 2)
    nrows = math.ceil(n_types / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(11 * ncols, 8 * nrows))
    if n_types == 1:
        axes_flat = [axes]
    elif nrows == 1:
        axes_flat = list(axes) if ncols > 1 else [axes]
    else:
        axes_flat = axes.flatten()

    for idx, model_type in enumerate(model_types):
        ax = axes_flat[idx]
        model_data = valid[valid["model_type"] == model_type]
        nominees = sorted(model_data["title"].unique())
        dates = sorted(model_data["snapshot_date"].unique())

        matrix = np.full((len(nominees), len(dates)), np.nan)
        for i, nominee in enumerate(nominees):
            for j, d in enumerate(dates):
                sub = model_data[
                    (model_data["title"] == nominee) & (model_data["snapshot_date"] == d)
                ]
                if not sub.empty:
                    matrix[i, j] = sub.iloc[0]["divergence"] * 100

        vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 1)
        im = ax.imshow(
            matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest"
        )
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels(dates, rotation=45, ha="right")
        ax.set_yticks(range(len(nominees)))
        ax.set_yticklabels([n[:20] for n in nominees])
        display = get_model_display(model_type)
        ax.set_title(f"{display} (model − market, pp)", fontsize=12)
        plt.colorbar(im, ax=ax, label="pp", shrink=0.8)

        for i in range(len(nominees)):
            for j in range(len(dates)):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > vmax * 0.6 else "black"
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7, color=color)

    for j in range(len(model_types), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Divergence Heatmaps: Model − Market (pp)", fontsize=15)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


# ============================================================================
# Trading backtest plots
# ============================================================================


def plot_wealth_curves(
    bt_results: dict,
    model_types: list[str],
    output_path: Path | None = None,
) -> plt.Figure:
    """Wealth curves for all models with trade count annotations.

    Args:
        bt_results: Full backtest results dict (with config, backtests keys).
        model_types: Model types to include.
        output_path: If provided, save figure.

    Returns:
        matplotlib Figure
    """
    backtests = bt_results.get("backtests", {})
    config = bt_results.get("config", {})
    initial = config.get("bankroll_dollars", 1000)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax_idx, mode in enumerate(["fixed", "dynamic"]):
        ax = axes[ax_idx]
        for mt in model_types + ["average"]:
            rk = f"{mt}_{mode}"
            if rk not in backtests:
                continue
            snaps = backtests[rk]["snapshots"]
            dates = [s["snapshot_date"] for s in snaps]
            wealths = [s["total_wealth"] for s in snaps]
            color = get_model_color(mt)
            label = get_model_display(mt)
            ax.plot(dates, wealths, "o-", color=color, label=label)

            # Annotate buy/sell counts
            for s in snaps:
                n_buy = s.get("n_buy_signals", 0)
                n_sell = s.get("n_sell_signals", 0)
                if n_buy + n_sell > 0:
                    ax.annotate(
                        f"+{n_buy}/-{n_sell}",
                        (s["snapshot_date"], s["total_wealth"]),
                        fontsize=7,
                        alpha=0.6,
                        textcoords="offset points",
                        xytext=(0, 8),
                        ha="center",
                    )

        ax.axhline(initial, color="gray", linestyle="--", alpha=0.5, label=f"Initial ${initial}")
        ax.set_ylabel("Total Wealth ($)")
        ax.set_title(f"Wealth Curves — {mode.title()} Bankroll", fontsize=13)
        ax.legend(fontsize=9, loc="upper left")
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Trading Backtest: Wealth Over Time", fontsize=15)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


def plot_settlement_heatmap(
    bt_results: dict,
    output_path: Path | None = None,
) -> plt.Figure:
    """Settlement return % heatmap: models x possible winners.

    Args:
        bt_results: Full backtest results dict.
        output_path: If provided, save figure.

    Returns:
        matplotlib Figure
    """
    backtests = bt_results.get("backtests", {})
    dynamic_runs = {k: v for k, v in backtests.items() if k.endswith("_dynamic")}
    if not dynamic_runs:
        dynamic_runs = backtests

    all_winners: set[str] = set()
    for run_data in dynamic_runs.values():
        all_winners.update(run_data.get("settlements", {}).keys())

    winners = sorted(all_winners)
    run_keys = sorted(dynamic_runs.keys())

    if not winners or not run_keys:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No settlement data", transform=ax.transAxes, ha="center")
        return fig

    matrix = np.full((len(winners), len(run_keys)), np.nan)
    for j, rk in enumerate(run_keys):
        settlements = dynamic_runs[rk].get("settlements", {})
        for i, winner in enumerate(winners):
            if winner in settlements:
                matrix[i, j] = settlements[winner]["return_pct"]

    fig, ax = plt.subplots(figsize=(max(12, len(run_keys) * 2.5), max(8, len(winners) * 0.55)))
    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 1)
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(run_keys)))
    ax.set_xticklabels(
        [rk.replace("_dynamic", " (dyn)").replace("_fixed", " (fix)") for rk in run_keys],
        rotation=45,
        ha="right",
    )
    ax.set_yticks(range(len(winners)))
    ax.set_yticklabels([w[:25] for w in winners])
    ax.set_title("Settlement Return (%) by Winner × Model", fontsize=14)
    plt.colorbar(im, ax=ax, label="Return %", shrink=0.8)

    for i in range(len(winners)):
        for j in range(len(run_keys)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.4 else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=8, color=color)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


def plot_position_evolution(
    bt_results: dict,
    model_type: str,
    output_path: Path | None = None,
) -> plt.Figure:
    """Stacked bar chart of positions held for a single model (dynamic bankroll).

    Args:
        bt_results: Full backtest results dict.
        model_type: Which model's positions to show.
        output_path: If provided, save figure.

    Returns:
        matplotlib Figure
    """
    backtests = bt_results.get("backtests", {})
    rk = f"{model_type}_dynamic"
    if rk not in backtests:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No data for {model_type}", transform=ax.transAxes, ha="center")
        return fig

    snaps = backtests[rk]["snapshots"]
    if not snaps:
        fig, ax = plt.subplots()
        return fig

    all_nominees: set[str] = set()
    for s in snaps:
        all_nominees.update(s["positions"].keys())

    nominees = sorted(all_nominees)
    if not nominees:
        fig, ax = plt.subplots()
        return fig

    dates = [s["snapshot_date"] for s in snaps]
    data = {nom: [0.0] * len(snaps) for nom in nominees}

    for i, s in enumerate(snaps):
        for nom, pos in s["positions"].items():
            avg_cost = pos.get("avg_cost", 0)
            data[nom][i] = pos["contracts"] * avg_cost

    fig, ax = plt.subplots(figsize=(14, 7))
    bottom = np.zeros(len(snaps))
    cmap = matplotlib.colormaps["tab10"]

    for j, nom in enumerate(nominees):
        values = np.array(data[nom])
        ax.bar(
            range(len(dates)),
            values,
            bottom=bottom,
            label=nom[:20],
            color=cmap(j % 10),
            alpha=0.85,
        )
        bottom += values

    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=45, ha="right")
    ax.set_ylabel("Position Value ($)")
    display = get_model_display(model_type)
    ax.set_title(f"Position Evolution — {display} (Dynamic Bankroll)", fontsize=13)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1, 1))

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


def plot_edge_distributions(
    mm_df: pd.DataFrame,
    model_types: list[str],
    market_blend_alpha: float = 0.15,
    output_path: Path | None = None,
) -> plt.Figure:
    """Edge histograms per model type (blended model prob - market price).

    Args:
        mm_df: DataFrame with columns: model_type, model_prob, market_prob
        model_types: Model types to include.
        market_blend_alpha: Blend factor for edge computation.
        output_path: If provided, save figure.

    Returns:
        matplotlib Figure
    """
    valid = mm_df.dropna(subset=["market_prob"]).copy()
    if valid.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No market data", transform=ax.transAxes, ha="center")
        return fig

    valid["blended_prob"] = (
        market_blend_alpha * valid["market_prob"] + (1 - market_blend_alpha) * valid["model_prob"]
    )
    valid["edge"] = valid["blended_prob"] - valid["market_prob"]

    n_types = len(model_types)
    ncols = min(n_types, 3)
    nrows = math.ceil(n_types / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if n_types == 1:
        axes_flat = [axes]
    elif nrows == 1:
        axes_flat = list(axes) if ncols > 1 else [axes]
    else:
        axes_flat = axes.flatten()

    for idx, mt in enumerate(model_types):
        ax = axes_flat[idx]
        mt_data = valid[valid["model_type"] == mt]
        edges: np.ndarray = mt_data["edge"].to_numpy() * 100

        n_bins = min(30, max(5, len(edges) // 3))
        color = get_model_color(mt)
        ax.hist(edges, bins=n_bins, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", alpha=0.5)
        ax.axvline(5, color="green", linestyle=":", alpha=0.5, label="5pp threshold")
        ax.axvline(-3, color="orange", linestyle=":", alpha=0.5, label="-3pp sell")

        mean_edge = np.mean(edges)
        actionable = np.sum(np.abs(edges) >= 5)
        display = get_model_display(mt)
        ax.set_title(f"{display} (mean={mean_edge:.1f}pp, |edge|≥5pp: {actionable})", fontsize=11)
        ax.set_xlabel("Edge (pp)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    for j in range(len(model_types), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Edge Distribution (α={market_blend_alpha} blend)", fontsize=15)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig
