"""Scenario-based analysis plots for buy-and-hold backtest.

Generates plots based on the EV + bounded worst-case scoring framework:

1. Pareto frontier: EV vs worst-case across configs
2. Temporal EV evolution: EV by entry point (days before ceremony)
3. Config heatmap: edge threshold × KF, colored by EV
4. Cross-year EV scatter: 2024 EV vs 2025 EV
5. Portfolio worst-case distribution: histogram across configs
6. EV decomposition: model vs market vs blend comparison
7. Category-level scenario heatmap

Usage::

    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\\
d20260225_buy_hold_backtest.analyze_scenario_plots --ceremony-year 2025
    # Cross-year:
    uv run python -m oscar_prediction_market.one_offs.\\
d20260225_buy_hold_backtest.analyze_scenario_plots --cross-year
"""

import argparse
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
from oscar_prediction_market.one_offs.d20260225_buy_hold_backtest.scenario_scoring import (
    CONFIG_COLS,
    CROSS_YEAR_LOSS_BOUNDS,
    DEFAULT_LOSS_BOUNDS,
    compute_cross_year_pareto,
    compute_cross_year_scores,
    compute_pareto_frontier,
    compute_portfolio_scores,
    compute_temporal_scores,
    load_entry_pnl,
    load_scenario_pnl,
)
from oscar_prediction_market.one_offs.d20260225_buy_hold_backtest.year_config import (
    BUY_HOLD_EXP_DIR,
    YEAR_CONFIGS,
)

apply_style()

BANKROLL = 1000.0

MODEL_ORDER = ["clogit", "lr", "gbt", "cal_sgbt", "avg_ensemble", "clogit_cal_sgbt_ensemble"]


def _save_fig(fig: matplotlib.figure.Figure, path: Path, *, dpi: int = 150) -> None:
    """Save figure and close."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ============================================================================
# Per-year plots
# ============================================================================


def plot_pareto_frontier(
    portfolio_df: pd.DataFrame,
    plots_dir: Path,
    year_label: str,
) -> None:
    """Plot EV vs worst-case with Pareto frontier highlighted.

    Each dot is a config. The frontier shows the best EV at each worst-case level.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    ev_cols = [
        ("ev_pnl_model", "Model Probs"),
        ("ev_pnl_market", "Market Probs"),
        ("ev_pnl_blend", "Blend (avg)"),
    ]

    for ax, (ev_col, title) in zip(axes, ev_cols, strict=True):
        # Scatter all configs
        models = sorted(portfolio_df["model_type"].unique())
        for model in models:
            mask = portfolio_df["model_type"] == model
            subset = portfolio_df[mask]
            ax.scatter(
                subset["worst_pnl"],
                subset[ev_col],
                c=get_model_color(model),
                label=get_model_display(model),
                alpha=0.3,
                s=10,
            )

        # Overlay Pareto frontier
        frontier = compute_pareto_frontier(portfolio_df, ev_column=ev_col)
        feasible = frontier[frontier["n_feasible"] > 0]
        if not feasible.empty:
            ax.plot(
                feasible["best_risk"],
                feasible["best_ev"],
                "k-o",
                linewidth=2,
                markersize=6,
                zorder=10,
                label="Pareto frontier",
            )
            # Annotate L values on frontier
            for _, frow in feasible.iterrows():
                ax.annotate(
                    f"L={frow['loss_bound_pct']:.0f}%",
                    (frow["best_risk"], frow["best_ev"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=6,
                    alpha=0.8,
                )

        ax.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=0.5)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5, linewidth=0.5)
        ax.set_xlabel("Portfolio Worst-Case PnL ($)")
        ax.set_title(f"EV: {title}")
        if ax == axes[0]:
            ax.set_ylabel("Expected PnL ($)")

    axes[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"EV vs Worst-Case — {year_label}", fontsize=14, y=1.02)

    _save_fig(fig, plots_dir / "pareto_frontier.png")


def plot_temporal_ev(
    entry_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    plots_dir: Path,
    year_label: str,
) -> None:
    """Plot EV evolution over entry points for top configs.

    Shows how EV accumulates as more precursor information becomes available.
    """
    # Pick top 5 configs by portfolio EV blend
    top_configs = portfolio_df.head(5)["config_label"].tolist()
    top_models = portfolio_df.head(5)["model_type"].tolist()

    temporal = compute_temporal_scores(entry_df)

    fig, ax = plt.subplots(figsize=(12, 6))

    snap_order = sorted(entry_df["entry_snapshot"].unique())

    for cfg_label, model in zip(top_configs, top_models, strict=True):
        mask = (temporal["config_label"] == cfg_label) & (temporal["model_type"] == model)
        subset = temporal[mask].copy()
        if subset.empty:
            continue

        # Order by snapshot
        subset = subset.set_index("entry_snapshot").reindex(snap_order).reset_index()
        subset["cum_ev"] = subset["ev_pnl_blend"].cumsum()

        short_label = f"{model[:8]}|{cfg_label.split('_bet=')[1][:12] if '_bet=' in cfg_label else cfg_label[:16]}"
        ax.plot(
            range(len(subset)),
            subset["cum_ev"],
            marker="o",
            markersize=4,
            label=short_label,
            linewidth=1.5,
        )

    ax.set_xticks(range(len(snap_order)))
    ax.set_xticklabels(
        [s.split("_", 1)[1] if "_" in s else s for s in snap_order],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Entry Point")
    ax.set_ylabel("Cumulative EV PnL ($)")
    ax.set_title(f"Temporal EV Evolution — Top 5 Configs — {year_label}")
    ax.legend(fontsize=7, loc="upper left")

    _save_fig(fig, plots_dir / "temporal_ev.png")


def plot_temporal_marginal_ev(
    entry_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    plots_dir: Path,
) -> None:
    """Grouped bar chart of marginal (non-cumulative) EV per entry point for top 5 configs.

    Also shows the mean EV across all configs as a dashed line.
    """
    top5 = portfolio_df.nlargest(5, "ev_pnl_blend")
    top_configs = top5["config_label"].tolist()
    top_models = top5["model_type"].tolist()

    temporal = compute_temporal_scores(entry_df)
    snap_order = sorted(entry_df["entry_snapshot"].unique())
    snap_labels = [s.split("_", 1)[1] if "_" in s else s for s in snap_order]

    n_snaps = len(snap_order)
    n_configs = len(top_configs)
    bar_width = 0.8 / max(n_configs, 1)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_snaps)

    for idx, (cfg_label, model) in enumerate(zip(top_configs, top_models, strict=True)):
        mask = (temporal["config_label"] == cfg_label) & (temporal["model_type"] == model)
        subset = temporal[mask].copy()
        if subset.empty:
            continue
        subset = subset.set_index("entry_snapshot").reindex(snap_order).reset_index()
        ev_vals = subset["ev_pnl_blend"].fillna(0).values

        short_label = (
            f"{model[:8]}|"
            f"{cfg_label.split('_bet=')[1][:12] if '_bet=' in cfg_label else cfg_label[:16]}"
        )
        ax.bar(x + idx * bar_width, ev_vals, bar_width, label=short_label, alpha=0.8)

    # Mean EV across ALL configs per entry point
    mean_ev = (
        temporal.groupby("entry_snapshot")["ev_pnl_blend"]
        .mean()
        .reindex(snap_order)
        .fillna(0)
        .to_numpy()
    )
    ax.plot(x + (n_configs - 1) * bar_width / 2, mean_ev, "k--", linewidth=1.5, label="Mean (all)")

    ax.set_xticks(x + (n_configs - 1) * bar_width / 2)
    ax.set_xticklabels(snap_labels, rotation=45, ha="right", fontsize=7)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Entry Point")
    ax.set_ylabel("Marginal EV PnL ($)")
    ax.set_title("Marginal EV per Entry Point (Top 5 Configs)")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()

    _save_fig(fig, plots_dir / "temporal_marginal_ev.png")


def plot_temporal_envelope(
    entry_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    plots_dir: Path,
) -> None:
    """1×3 subplot panel showing cumulative EV envelope for Pareto-optimal configs.

    Picks configs at L=10%, L=30%, L=50% from the worst-case Pareto frontier.
    Each panel shows cumulative ev_pnl_blend (solid), cumulative total_pnl (dashed),
    and a filled band from cumulative worst_pnl to cumulative best_pnl.
    """
    # Pick representative configs from the Pareto frontier at L=10/30/50
    frontier = compute_pareto_frontier(portfolio_df, ev_column="ev_pnl_blend")
    target_ls = [10.0, 30.0, 50.0]
    picks = []
    for target_l in target_ls:
        row = frontier[frontier["loss_bound_pct"] == target_l]
        if not row.empty and row.iloc[0]["n_feasible"] > 0:
            picks.append((row.iloc[0]["best_config"], row.iloc[0]["best_model"]))
    if not picks:
        # Fallback to top 3 by EV
        top3 = portfolio_df.nlargest(3, "ev_pnl_blend")
        picks = list(zip(top3["config_label"].tolist(), top3["model_type"].tolist(), strict=True))
    top_configs = [p[0] for p in picks]
    top_models = [p[1] for p in picks]

    temporal = compute_temporal_scores(entry_df)
    snap_order = sorted(entry_df["entry_snapshot"].unique())
    snap_labels = [s.split("_", 1)[1] if "_" in s else s for s in snap_order]
    n_snaps = len(snap_order)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for idx, (cfg_label, model) in enumerate(zip(top_configs, top_models, strict=True)):
        ax = axes[idx]
        mask = (temporal["config_label"] == cfg_label) & (temporal["model_type"] == model)
        subset = temporal[mask].copy()
        if subset.empty:
            ax.set_visible(False)
            continue

        subset = subset.set_index("entry_snapshot").reindex(snap_order).reset_index()
        x = np.arange(n_snaps)

        cum_ev = subset["ev_pnl_blend"].fillna(0).cumsum().values
        cum_worst = subset["worst_pnl"].fillna(0).cumsum().values
        cum_best = subset["best_pnl"].fillna(0).cumsum().values
        cum_actual = subset["total_pnl"].fillna(0).cumsum().values

        color = colors[idx % len(colors)]
        ax.fill_between(x, cum_worst, cum_best, alpha=0.15, color=color, label="Worst–Best")
        ax.plot(x, cum_ev, "-o", color=color, markersize=3, linewidth=1.5, label="EV (blend)")
        ax.plot(
            x,
            cum_actual,
            "--s",
            color=color,
            markersize=3,
            linewidth=1.2,
            alpha=0.7,
            label="Actual",
        )

        ax.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(snap_labels, rotation=45, ha="right", fontsize=6)
        ax.set_xlabel("Entry Point")

        short_label = (
            f"{model[:8]}|"
            f"{cfg_label.split('_bet=')[1][:12] if '_bet=' in cfg_label else cfg_label[:16]}"
        )
        ax.set_title(short_label, fontsize=9)
        ax.legend(fontsize=6, loc="upper left")

    axes[0].set_ylabel("Cumulative $ PnL")
    fig.suptitle("Temporal EV Envelope — Pareto Picks (L=10/30/50%)", fontsize=13, y=1.02)
    fig.tight_layout()

    _save_fig(fig, plots_dir / "temporal_envelope.png")


def plot_per_entry_violin(
    entry_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    plots_dir: Path,
    ceremony_year: int,
) -> None:
    """1×3 subplot panel showing MC P&L distribution violins at each entry point.

    Picks 3 representative configs from the CVaR-5% Pareto frontier at L=10%,
    L=20%, L=30%.  For each config × entry snapshot, runs 10,000 Monte Carlo
    simulations drawing category winners from blend probabilities, producing
    a portfolio-level P&L distribution.  Each entry point gets a violin plot;
    actual realised P&L is overlaid as a red dot.
    """
    # --- 1. Pick 3 representative configs from CVaR-5% Pareto frontier ---
    frontier = compute_pareto_frontier(portfolio_df, ev_column="ev_pnl_blend", risk_col="cvar_5")
    target_ls = [10.0, 20.0, 30.0]
    picks: list[tuple[str, str]] = []
    for target_l in target_ls:
        row = frontier[frontier["loss_bound_pct"] == target_l]
        if not row.empty and row.iloc[0]["n_feasible"] > 0:
            picks.append((row.iloc[0]["best_config"], row.iloc[0]["best_model"]))
    if not picks:
        # Fallback to top 3 by EV
        top3 = portfolio_df.nlargest(3, "ev_pnl_blend")
        picks = list(zip(top3["config_label"].tolist(), top3["model_type"].tolist(), strict=True))
    if not picks:
        print("  Skipping per-entry violin — no configs found.")
        return

    # --- 2. Load scenario_pnl.csv ---
    try:
        scenario_df = load_scenario_pnl(ceremony_year)
    except FileNotFoundError as exc:
        print(f"  Skipping per-entry violin — {exc}")
        return

    snap_order = sorted(entry_df["entry_snapshot"].unique())
    snap_labels = [s.split("_", 1)[1] if "_" in s else s for s in snap_order]
    n_snaps = len(snap_order)

    # Precompute actual PnL per (entry, config, model)
    actual_by_entry = entry_df.groupby(["entry_snapshot", "model_type", "config_label"])[
        "total_pnl"
    ].sum()

    # --- 3 & 4. MC simulation + plotting ---
    rng = np.random.default_rng(42)
    n_samples = 10_000

    n_panels = len(picks)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    panel_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, (cfg_label, model) in enumerate(picks):
        ax = axes[idx]
        color = panel_colors[idx % len(panel_colors)]

        # Build MC distributions per entry snapshot
        grouped = scenario_df[
            (scenario_df["model_type"] == model) & (scenario_df["config_label"] == cfg_label)
        ].groupby(["entry_snapshot", "category"])

        entry_cats: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        for (entry, category), grp in grouped:
            entry_str = str(entry)
            if entry_str not in entry_cats:
                entry_cats[entry_str] = {}
            pnls = np.asarray(grp["pnl"].values)
            blend = np.asarray(
                (grp["model_prob"].values + grp["market_prob"].values) / 2.0  # type: ignore[operator]
            )
            total = blend.sum()
            if total > 0:
                blend = blend / total
            else:
                blend = np.ones(len(blend)) / len(blend)
            entry_cats[entry_str][str(category)] = (pnls, blend)

        mc_data: list[np.ndarray] = []
        actual_vals: list[float] = []
        positions: list[int] = []

        for snap_idx, snap in enumerate(snap_order):
            if snap not in entry_cats:
                continue
            cats = entry_cats[snap]
            portfolio_pnl = np.zeros(n_samples)
            for _cat, (pnls, probs) in cats.items():
                indices = rng.choice(len(pnls), size=n_samples, p=probs)
                portfolio_pnl += pnls[indices]

            mc_data.append(portfolio_pnl)
            positions.append(snap_idx)

            # Actual PnL
            key = (snap, model, cfg_label)
            if key in actual_by_entry.index:
                actual_vals.append(float(actual_by_entry.loc[key]))
            else:
                actual_vals.append(np.nan)

        if not mc_data:
            ax.set_visible(False)
            continue

        # Plot violins
        parts = ax.violinplot(
            mc_data,
            positions=positions,
            showmeans=True,
            showmedians=True,
            widths=0.7,
        )
        # Style violin bodies
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.3)
        for partname in ("cmeans", "cmedians", "cmins", "cmaxes", "cbars"):
            if partname in parts:
                parts[partname].set_color(color)
                parts[partname].set_alpha(0.7)

        # Overlay actual PnL as red dots
        ax.scatter(
            positions,
            actual_vals,
            color="red",
            s=40,
            zorder=5,
            label="Actual",
            edgecolors="darkred",
            linewidths=0.5,
        )

        ax.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=0.5)
        ax.set_xticks(range(n_snaps))
        ax.set_xticklabels(snap_labels, rotation=45, ha="right", fontsize=6)
        ax.set_xlabel("Entry Point")

        short_label = (
            f"{model[:8]}|"
            f"{cfg_label.split('_bet=')[1][:12] if '_bet=' in cfg_label else cfg_label[:16]}"
        )
        ax.set_title(short_label, fontsize=9)
        ax.legend(fontsize=6, loc="upper left")

    axes[0].set_ylabel("Portfolio PnL ($)")
    fig.suptitle(
        "Per-Entry MC P&L Distribution — CVaR-5% Pareto Picks (L=10/20/30%)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()

    _save_fig(fig, plots_dir / "per_entry_violin.png")


def plot_config_heatmap(
    portfolio_df: pd.DataFrame,
    plots_dir: Path,
    year_label: str,
) -> None:
    """Heatmap of EV by edge threshold × kelly_fraction for each model.

    Faceted by model type. Multi-outcome configs are collapsed (take first
    since KF doesn't affect multi-outcome Kelly).
    """
    models = sorted(portfolio_df["model_type"].unique())
    n_models = len(models)
    ncols = min(3, n_models)
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_2d(axes)

    for idx, model in enumerate(models):
        ax = axes[idx // ncols, idx % ncols]
        mdf = portfolio_df[
            (portfolio_df["model_type"] == model)
            & (portfolio_df["fee_type"] == "maker")
            & (portfolio_df["allowed_directions"] == "yes")
        ]

        if mdf.empty:
            ax.set_visible(False)
            continue

        # Pivot: edge × kf → EV
        for km, _marker in [("independent", "o"), ("multi_outcome", "s")]:
            km_df = mdf[mdf["kelly_mode"] == km]
            if km_df.empty:
                continue

            pivot = km_df.pivot_table(
                index="buy_edge_threshold",
                columns="kelly_fraction",
                values="ev_pnl_blend",
                aggfunc="first",
            )

            if km == "independent":
                im = ax.imshow(
                    pivot.values,
                    aspect="auto",
                    cmap="RdYlGn",
                    origin="lower",
                )
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns], fontsize=7)
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels([f"{v:.2f}" for v in pivot.index], fontsize=7)
                fig.colorbar(im, ax=ax, shrink=0.8, label="EV PnL ($)")

        ax.set_xlabel("Kelly Fraction")
        ax.set_ylabel("Edge Threshold")
        ax.set_title(f"{get_model_display(model)}", fontsize=10)

    # Hide unused axes
    for idx in range(n_models, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(f"EV Heatmap (maker/YES/independent) — {year_label}", fontsize=13, y=1.02)
    fig.tight_layout()
    _save_fig(fig, plots_dir / "ev_config_heatmap.png")


def plot_worst_case_distribution(
    portfolio_df: pd.DataFrame,
    plots_dir: Path,
    year_label: str,
) -> None:
    """Histogram of portfolio worst-case PnL across all configs."""
    fig, ax = plt.subplots(figsize=(10, 5))

    models = sorted(portfolio_df["model_type"].unique())
    for model in models:
        mask = portfolio_df["model_type"] == model
        ax.hist(
            portfolio_df.loc[mask, "worst_pnl"],
            bins=50,
            alpha=0.4,
            label=get_model_display(model),
            color=get_model_color(model),
        )

    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    total_bankroll = portfolio_df["total_bankroll_deployed"].iloc[0]
    for pct in [0.10, 0.20, 0.30, 0.50]:
        ax.axvline(
            -pct * total_bankroll,
            color="red",
            linestyle=":",
            alpha=0.3,
        )
        ax.text(
            -pct * total_bankroll,
            ax.get_ylim()[1] * 0.95,
            f"-{pct:.0%}",
            fontsize=7,
            color="red",
            ha="center",
        )

    ax.set_xlabel("Portfolio Worst-Case PnL ($)")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Portfolio Worst-Case PnL — {year_label}")
    ax.legend(fontsize=7)

    _save_fig(fig, plots_dir / "worst_case_distribution.png")


def plot_ev_comparison(
    portfolio_df: pd.DataFrame,
    plots_dir: Path,
    year_label: str,
) -> None:
    """Scatter: model EV vs market EV, colored by model type."""
    fig, ax = plt.subplots(figsize=(8, 8))

    models = sorted(portfolio_df["model_type"].unique())
    for model in models:
        mask = portfolio_df["model_type"] == model
        subset = portfolio_df[mask]
        ax.scatter(
            subset["ev_pnl_market"],
            subset["ev_pnl_model"],
            c=get_model_color(model),
            label=get_model_display(model),
            alpha=0.3,
            s=12,
        )

    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=0.5)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)

    ax.set_xlabel("EV PnL (Market Probs, $)")
    ax.set_ylabel("EV PnL (Model Probs, $)")
    ax.set_title(f"Model EV vs Market EV — {year_label}")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")

    _save_fig(fig, plots_dir / "ev_model_vs_market.png")


def plot_actual_vs_ev(
    portfolio_df: pd.DataFrame,
    plots_dir: Path,
    year_label: str,
) -> None:
    """Scatter: EV (blend) vs actual realized PnL."""
    fig, ax = plt.subplots(figsize=(8, 8))

    models = sorted(portfolio_df["model_type"].unique())
    for model in models:
        mask = portfolio_df["model_type"] == model
        subset = portfolio_df[mask]
        ax.scatter(
            subset["ev_pnl_blend"],
            subset["total_pnl"],
            c=get_model_color(model),
            label=get_model_display(model),
            alpha=0.3,
            s=12,
        )

    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=0.5)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)

    ax.set_xlabel("Expected PnL (Blend, $)")
    ax.set_ylabel("Actual Realized PnL ($)")
    ax.set_title(f"EV vs Actual — {year_label}")
    ax.legend(fontsize=7)

    _save_fig(fig, plots_dir / "ev_vs_actual.png")


# ============================================================================
# CVaR plots (per-year)
# ============================================================================


def plot_cvar_pareto(
    portfolio_df: pd.DataFrame,
    plots_dir: Path,
    year_label: str,
    risk_col: str = "cvar_5",
    risk_label: str = "CVaR-5%",
) -> None:
    """CVaR Pareto frontier: EV vs risk constraint, per EV variant.

    Creates a 1×3 subplot (model / market / blend), same layout as the
    worst-case Pareto plot but using the specified risk_col on the x-axis.
    Saves as ``cvar_pareto.png``.

    Args:
        portfolio_df: Portfolio-level scores with CVaR columns already merged.
        plots_dir: Output directory for plots.
        year_label: Label for title.
        risk_col: Column for x-axis risk metric (e.g. "cvar_5", "cvar_10").
        risk_label: Display label for the risk metric.
    """
    if risk_col not in portfolio_df.columns:
        print(f"  Skipping CVaR Pareto — {risk_col} not in portfolio_df.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    ev_cols = [
        ("ev_pnl_model", "Model Probs"),
        ("ev_pnl_market", "Market Probs"),
        ("ev_pnl_blend", "Blend (avg)"),
    ]

    for ax, (ev_col, title) in zip(axes, ev_cols, strict=True):
        models = sorted(portfolio_df["model_type"].unique())
        for model in models:
            mask = portfolio_df["model_type"] == model
            subset = portfolio_df[mask]
            ax.scatter(
                subset[risk_col],
                subset[ev_col],
                c=get_model_color(model),
                label=get_model_display(model),
                alpha=0.3,
                s=10,
            )

        # CVaR Pareto frontier
        frontier = compute_pareto_frontier(portfolio_df, ev_column=ev_col, risk_col=risk_col)
        feasible = frontier[frontier["n_feasible"] > 0]
        if not feasible.empty:
            ax.plot(
                feasible["best_risk"],
                feasible["best_ev"],
                "r--^",
                linewidth=2,
                markersize=6,
                zorder=10,
                label=f"{risk_label} Pareto frontier",
            )
            # Annotate L values on frontier
            for _, frow in feasible.iterrows():
                ax.annotate(
                    f"L={frow['loss_bound_pct']:.0f}%",
                    (frow["best_risk"], frow["best_ev"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=6,
                    alpha=0.8,
                )

        ax.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=0.5)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5, linewidth=0.5)
        ax.set_xlabel(f"Portfolio {risk_label} ($)")
        ax.set_title(f"EV: {title}")
        if ax == axes[0]:
            ax.set_ylabel("Expected PnL ($)")

    axes[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"EV vs {risk_label} — {year_label}", fontsize=14, y=1.02)

    _save_fig(fig, plots_dir / "cvar_pareto.png")


def plot_pareto_comparison(
    portfolio_df: pd.DataFrame,
    plots_dir: Path,
    year_label: str,
) -> None:
    """Compare worst-case vs CVaR-5% Pareto frontiers on the same axes.

    x-axis: Loss Bound (% of bankroll), y-axis: Best EV ($).
    Two lines: worst-case (black) and CVaR-5% (red).
    Saves as ``pareto_comparison.png``.

    Args:
        portfolio_df: Portfolio-level scores with CVaR columns already merged.
    """
    has_cvar5 = "cvar_5" in portfolio_df.columns

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    ev_cols = [
        ("ev_pnl_model", "Model Probs"),
        ("ev_pnl_market", "Market Probs"),
        ("ev_pnl_blend", "Blend (avg)"),
    ]

    loss_bounds_pct = [L * 100 for L in DEFAULT_LOSS_BOUNDS]

    for ax, (ev_col, title) in zip(axes, ev_cols, strict=True):
        # Worst-case frontier (cvar_0)
        wc_frontier = compute_pareto_frontier(portfolio_df, ev_column=ev_col, risk_col="cvar_0")
        wc_feasible = wc_frontier[wc_frontier["n_feasible"] > 0]
        if not wc_feasible.empty:
            ax.plot(
                wc_feasible["loss_bound_pct"],
                wc_feasible["best_ev"],
                "ko-",
                linewidth=2,
                markersize=5,
                label="Worst-case frontier",
            )

        # CVaR-5% frontier
        if has_cvar5:
            cvar_frontier = compute_pareto_frontier(
                portfolio_df, ev_column=ev_col, risk_col="cvar_5"
            )
            cvar_feasible = cvar_frontier[cvar_frontier["n_feasible"] > 0]
            if not cvar_feasible.empty:
                ax.plot(
                    cvar_feasible["loss_bound_pct"],
                    cvar_feasible["best_ev"],
                    "r--^",
                    linewidth=2,
                    markersize=5,
                    label="CVaR-5% frontier",
                )

        ax.set_xlabel("Loss Bound (%)")
        ax.set_title(f"EV: {title}")
        if ax == axes[0]:
            ax.set_ylabel("Best EV ($)")
        ax.set_xticks(loss_bounds_pct)
        ax.tick_params(axis="x", rotation=45)

    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Pareto Comparison: Worst-Case vs CVaR-5% — {year_label}", fontsize=14, y=1.02)

    _save_fig(fig, plots_dir / "pareto_comparison.png")


def plot_mc_convergence(
    mc_cal_df: pd.DataFrame,
    plots_dir: Path,
    year_label: str,
) -> None:
    """MC convergence: CVaR vs sample size for representative configs.

    For each of 3 configs, plots sample_size (x, log) vs cvar_value (y)
    with min/max error bands across trials.
    Saves as ``mc_convergence.png``.
    """
    configs = mc_cal_df.groupby(["config_label", "model_type"]).ngroups
    fig, axes = plt.subplots(1, min(configs, 3), figsize=(6 * min(configs, 3), 5), sharey=True)
    if configs == 1:
        axes = [axes]
    else:
        axes = list(axes)

    config_keys = (
        mc_cal_df.groupby(["model_type", "config_label"])
        .size()
        .reset_index()[["model_type", "config_label"]]
        .values.tolist()
    )

    for idx, (model_type, config_label) in enumerate(config_keys[:3]):
        ax = axes[idx]
        mask = (mc_cal_df["model_type"] == model_type) & (mc_cal_df["config_label"] == config_label)
        cfg_df = mc_cal_df[mask]

        agg = cfg_df.groupby("sample_size")["cvar_value"].agg(["mean", "min", "max"])
        agg = agg.sort_index()

        ax.plot(agg.index, agg["mean"], "o-", color=get_model_color(model_type), linewidth=1.5)
        ax.fill_between(
            agg.index,
            agg["min"],
            agg["max"],
            alpha=0.2,
            color=get_model_color(model_type),
        )
        ax.set_xscale("log")
        ax.set_xlabel("Sample Size")
        if idx == 0:
            ax.set_ylabel("CVaR-5% ($)")
        short_label = f"{model_type[:8]}|{config_label.split('_bet=')[1][:12] if '_bet=' in config_label else config_label[:16]}"
        ax.set_title(short_label, fontsize=9)

    fig.suptitle(f"MC Convergence — {year_label}", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, plots_dir / "mc_convergence.png")


# ============================================================================
# Cross-year plots
# ============================================================================


def plot_cross_year_scatter(
    cross_year_df: pd.DataFrame,
    plots_dir: Path,
) -> None:
    """Scatter: 2024 EV vs 2025 EV, colored by model."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (col_suffix, title) in zip(
        axes,
        [
            ("ev_pnl_model", "Model Probs"),
            ("ev_pnl_market", "Market Probs"),
            ("ev_pnl_blend", "Blend"),
        ],
        strict=True,
    ):
        c24 = f"{col_suffix}_2024"
        c25 = f"{col_suffix}_2025"
        models = sorted(cross_year_df["model_type"].unique())

        for model in models:
            mask = cross_year_df["model_type"] == model
            subset = cross_year_df[mask]
            ax.scatter(
                subset[c24],
                subset[c25],
                c=get_model_color(model),
                label=get_model_display(model),
                alpha=0.3,
                s=10,
            )

        ax.axhline(0, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, "k--", alpha=0.3, linewidth=0.5)
        ax.set_xlabel("2024 EV PnL ($)")
        ax.set_ylabel("2025 EV PnL ($)")
        ax.set_title(f"EV: {title}")

    axes[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Cross-Year EV Comparison", fontsize=14, y=1.02)

    _save_fig(fig, plots_dir / "cross_year_ev_scatter.png")


def plot_cross_year_pareto(
    cross_year_df: pd.DataFrame,
    plots_dir: Path,
) -> None:
    """Cross-year Pareto frontier: avg EV vs min-worst-case."""
    fig, ax = plt.subplots(figsize=(10, 7))

    models = sorted(cross_year_df["model_type"].unique())
    for model in models:
        mask = cross_year_df["model_type"] == model
        subset = cross_year_df[mask]
        ax.scatter(
            subset["min_worst_pnl"],
            subset["avg_ev_pnl_blend"],
            c=get_model_color(model),
            label=get_model_display(model),
            alpha=0.3,
            s=10,
        )

    # Overlay Pareto frontier
    frontier = compute_cross_year_pareto(cross_year_df, ev_column="avg_ev_pnl_blend")
    feasible = frontier[frontier["n_feasible"] > 0]
    if not feasible.empty:
        x_vals = feasible.apply(
            lambda r: min(r["best_risk_2024"], r["best_risk_2025"]),
            axis=1,
        )
        ax.plot(
            x_vals,
            feasible["best_avg_ev"],
            "k-o",
            linewidth=2,
            markersize=6,
            zorder=10,
            label="Pareto frontier",
        )
        # Annotate L values on frontier
        for (_, frow), x_val in zip(feasible.iterrows(), x_vals, strict=True):
            ax.annotate(
                f"L={frow['loss_bound_pct']:.0f}%",
                (x_val, frow["best_avg_ev"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=6,
                alpha=0.8,
            )

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5, linewidth=0.5)
    ax.set_xlabel("Min Worst-Case PnL (worst of 2024, 2025) ($)")
    ax.set_ylabel("Average EV PnL (Blend, $)")
    ax.set_title("Cross-Year Pareto Frontier")
    ax.legend(fontsize=7, loc="lower right")

    _save_fig(fig, plots_dir / "cross_year_pareto.png")


def plot_cross_year_cvar_pareto(
    cross_year_df: pd.DataFrame,
    plots_dir: Path,
) -> None:
    """Cross-year CVaR Pareto frontier: avg EV vs min-avg-CVaR."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Use avg_cvar_5 for the scatter x-axis
    scatter_col = "avg_cvar_5"
    if scatter_col not in cross_year_df.columns:
        # Try to derive from per-year columns
        if "cvar_5_2024" in cross_year_df.columns and "cvar_5_2025" in cross_year_df.columns:
            cross_year_df = cross_year_df.copy()
            cross_year_df[scatter_col] = cross_year_df[["cvar_5_2024", "cvar_5_2025"]].min(axis=1)
        else:
            print("  Skipping cross_year_cvar_pareto — missing cvar_5 columns.")
            plt.close(fig)
            return

    models = sorted(cross_year_df["model_type"].unique())
    for model in models:
        mask = cross_year_df["model_type"] == model
        subset = cross_year_df[mask]
        ax.scatter(
            subset[scatter_col],
            subset["avg_ev_pnl_blend"],
            c=get_model_color(model),
            label=get_model_display(model),
            alpha=0.3,
            s=10,
        )

    # Overlay CVaR Pareto frontier using unified compute_cross_year_pareto
    frontier = compute_cross_year_pareto(
        cross_year_df,
        ev_column="avg_ev_pnl_blend",
        risk_col="cvar_5",
        loss_bounds=CROSS_YEAR_LOSS_BOUNDS,
    )
    feasible = frontier[frontier["n_feasible"] > 0]
    if not feasible.empty:
        # x = min CVaR across years for the best config at each loss bound
        x_vals = feasible.apply(
            lambda r: min(r["best_risk_2024"], r["best_risk_2025"]),
            axis=1,
        )
        ax.plot(
            x_vals,
            feasible["best_avg_ev"],
            "r--^",
            linewidth=2,
            markersize=6,
            zorder=10,
            label="CVaR-5% Pareto frontier",
        )
        # Annotate L values on frontier
        for (_, frow), x_val in zip(feasible.iterrows(), x_vals, strict=True):
            ax.annotate(
                f"L={frow['loss_bound_pct']:.0f}%",
                (x_val, frow["best_avg_ev"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=6,
                alpha=0.8,
            )

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5, linewidth=0.5)
    ax.set_xlabel("Min Avg CVaR-5% (worst of 2024, 2025) ($)")
    ax.set_ylabel("Average EV PnL (Blend, $)")
    ax.set_title("Cross-Year CVaR Pareto Frontier")
    ax.legend(fontsize=7, loc="lower right")

    _save_fig(fig, plots_dir / "cross_year_cvar_pareto.png")


def plot_cross_year_actual_vs_ev(
    cross_year_df: pd.DataFrame,
    plots_dir: Path,
) -> None:
    """Scatter: avg EV vs avg actual PnL across years."""
    fig, ax = plt.subplots(figsize=(8, 8))

    models = sorted(cross_year_df["model_type"].unique())
    for model in models:
        mask = cross_year_df["model_type"] == model
        subset = cross_year_df[mask]
        ax.scatter(
            subset["avg_ev_pnl_blend"],
            subset["avg_total_pnl"],
            c=get_model_color(model),
            label=get_model_display(model),
            alpha=0.3,
            s=12,
        )

    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=0.5)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
    ax.set_xlabel("Average EV PnL (Blend, $)")
    ax.set_ylabel("Average Actual PnL ($)")
    ax.set_title("Cross-Year: EV vs Actual")
    ax.legend(fontsize=7)

    _save_fig(fig, plots_dir / "cross_year_ev_vs_actual.png")


def plot_edge_sweep_pareto_overlay(
    cross_year_df: pd.DataFrame,
    plots_dir: Path,
) -> None:
    """Overlay edge-threshold sweep on the CVaR-5% Pareto frontier.

    Shows that for clogit/maker/multi_outcome/all, varying edge threshold
    traces a path along the Pareto frontier — a single risk dial.

    The 7 edge configs (best KF per edge) are plotted as colored markers
    on top of the full config scatter and the L-labeled Pareto frontier.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # X-axis: min CVaR-5% across years (most negative = worst)
    scatter_col = "min_cvar_5"
    cross_year_df = cross_year_df.copy()
    if scatter_col not in cross_year_df.columns:
        if "cvar_5_2024" in cross_year_df.columns and "cvar_5_2025" in cross_year_df.columns:
            cross_year_df[scatter_col] = cross_year_df[["cvar_5_2024", "cvar_5_2025"]].min(axis=1)
        else:
            print("  Skipping edge_sweep_pareto_overlay — missing cvar_5 columns.")
            plt.close(fig)
            return

    # 1. Gray scatter for all configs
    ax.scatter(
        cross_year_df[scatter_col],
        cross_year_df["avg_ev_pnl_blend"],
        c="#cccccc",
        alpha=0.3,
        s=8,
        zorder=1,
        label=f"All configs (n={len(cross_year_df)})",
    )

    # 2. Black Pareto frontier line
    pareto_path = BUY_HOLD_EXP_DIR / "cross_year_pareto_cvar05_blend.csv"
    if pareto_path.exists():
        pareto_df = pd.read_csv(pareto_path)
        pareto_feasible = pareto_df[pareto_df["n_feasible"] > 0]
        if not pareto_feasible.empty:
            pareto_x = pareto_feasible.apply(
                lambda r: min(r["best_risk_2024"], r["best_risk_2025"]),
                axis=1,
            )
            ax.plot(
                pareto_x,
                pareto_feasible["best_avg_ev"],
                "ko-",
                linewidth=2,
                markersize=8,
                zorder=5,
                label="CVaR-5% Pareto frontier",
            )
            for (_, frow), x_val in zip(pareto_feasible.iterrows(), pareto_x, strict=True):
                ax.annotate(
                    f"L={frow['loss_bound_pct']:.0f}%",
                    (x_val, frow["best_avg_ev"]),
                    textcoords="offset points",
                    xytext=(8, -3),
                    fontsize=8,
                    fontweight="bold",
                )

    # 3. Edge sweep: filter clogit/maker/multi_outcome/all, best KF per edge
    edge_mask = (
        (cross_year_df["model_type"] == "clogit")
        & (cross_year_df["fee_type"] == "maker")
        & (cross_year_df["kelly_mode"] == "multi_outcome")
        & (cross_year_df["allowed_directions"] == "all")
    )
    edge_sub = cross_year_df[edge_mask].copy()

    if edge_sub.empty:
        print("  Skipping edge sweep overlay — no clogit/maker/multi/all configs found.")
        plt.close(fig)
        return

    best_kf = edge_sub.loc[
        edge_sub.groupby("buy_edge_threshold")["avg_ev_pnl_blend"].idxmax()
    ].sort_values("buy_edge_threshold")

    # Colormap: blue (low edge) to red (high edge)
    edge_vals = np.asarray(best_kf["buy_edge_threshold"].values)
    norm = plt.Normalize(vmin=edge_vals.min(), vmax=edge_vals.max())  # type: ignore[attr-defined]
    cmap = plt.get_cmap("coolwarm")

    # Draw connecting line
    ax.plot(
        np.asarray(best_kf[scatter_col].values),
        np.asarray(best_kf["avg_ev_pnl_blend"].values),
        color="#555555",
        linewidth=1.5,
        linestyle="--",
        alpha=0.7,
        zorder=8,
    )

    # Draw colored markers with labels
    for _, row in best_kf.iterrows():
        color = cmap(norm(row["buy_edge_threshold"]))
        ax.scatter(
            row[scatter_col],
            row["avg_ev_pnl_blend"],
            c=[color],
            s=100,
            edgecolors="black",
            linewidths=0.8,
            zorder=10,
        )
        ax.annotate(
            f"e={row['buy_edge_threshold']:.2f}",
            (row[scatter_col], row["avg_ev_pnl_blend"]),
            textcoords="offset points",
            xytext=(-5, 8),
            fontsize=7,
            color=color,
            fontweight="bold",
        )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  # type: ignore[attr-defined]
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Edge Threshold", fontsize=9)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5, linewidth=0.5)
    ax.set_xlabel("Min CVaR-5% Across Years (worst of 2024, 2025) ($)")
    ax.set_ylabel("Average EV PnL (Blend, $)")
    ax.set_title(
        "Edge Threshold Sweep on CVaR-5% Pareto Frontier\n(clogit / maker / multi_outcome / all)"
    )
    ax.legend(fontsize=7, loc="lower right")

    _save_fig(fig, plots_dir / "edge_sweep_pareto_overlay.png")


# ============================================================================
# CLI
# ============================================================================


def run_per_year(ceremony_year: int) -> None:
    """Generate all per-year scenario plots."""
    print(f"\n{'=' * 70}")
    print(f"Scenario Plots: {ceremony_year}")
    print(f"{'=' * 70}")

    entry_df = load_entry_pnl(ceremony_year)
    portfolio_df = compute_portfolio_scores(entry_df)
    plots_dir = YEAR_CONFIGS[ceremony_year].plots_dir / "scenario"

    # Build merged portfolio with unified CVaR columns (cvar_0 = worst_pnl)
    merged_portfolio = portfolio_df.copy()
    merged_portfolio["cvar_0"] = merged_portfolio["worst_pnl"]

    results_dir = YEAR_CONFIGS[ceremony_year].results_dir
    cvar_path = results_dir / "portfolio_cvar.csv"
    mc_path = results_dir / "mc_calibration.csv"

    if cvar_path.exists():
        cvar_df = pd.read_csv(cvar_path)
        merged_portfolio = merged_portfolio.merge(cvar_df, on=CONFIG_COLS)
    else:
        cvar_df = None

    # Worst-case Pareto (uses cvar_0 by default)
    plot_pareto_frontier(merged_portfolio, plots_dir, str(ceremony_year))
    plot_temporal_ev(entry_df, merged_portfolio, plots_dir, str(ceremony_year))
    plot_temporal_marginal_ev(entry_df, merged_portfolio, plots_dir)
    plot_per_entry_violin(entry_df, merged_portfolio, plots_dir, ceremony_year)
    plot_config_heatmap(merged_portfolio, plots_dir, str(ceremony_year))
    plot_worst_case_distribution(merged_portfolio, plots_dir, str(ceremony_year))
    plot_ev_comparison(merged_portfolio, plots_dir, str(ceremony_year))
    plot_actual_vs_ev(merged_portfolio, plots_dir, str(ceremony_year))

    # CVaR plots — graceful fallback if CVaR data not available
    if cvar_df is not None:
        plot_cvar_pareto(merged_portfolio, plots_dir, str(ceremony_year))
        plot_pareto_comparison(merged_portfolio, plots_dir, str(ceremony_year))
    else:
        print(f"  Skipping CVaR plots — {cvar_path} not found.")

    if mc_path.exists():
        mc_cal_df = pd.read_csv(mc_path)
        plot_mc_convergence(mc_cal_df, plots_dir, str(ceremony_year))
    else:
        print(f"  Skipping MC convergence plot — {mc_path} not found.")


def run_cross_year() -> None:
    """Generate cross-year scenario plots."""
    print(f"\n{'=' * 70}")
    print("Cross-Year Scenario Plots")
    print(f"{'=' * 70}")

    entry_2024 = load_entry_pnl(2024)
    entry_2025 = load_entry_pnl(2025)

    portfolio_2024 = compute_portfolio_scores(entry_2024)
    portfolio_2025 = compute_portfolio_scores(entry_2025)

    # Load per-year CVaR if available
    cvar_2024: pd.DataFrame | None = None
    cvar_2025: pd.DataFrame | None = None
    cvar_path_2024 = YEAR_CONFIGS[2024].results_dir / "portfolio_cvar.csv"
    cvar_path_2025 = YEAR_CONFIGS[2025].results_dir / "portfolio_cvar.csv"
    if cvar_path_2024.exists():
        cvar_2024 = pd.read_csv(cvar_path_2024)
    if cvar_path_2025.exists():
        cvar_2025 = pd.read_csv(cvar_path_2025)

    cross_year = compute_cross_year_scores(
        portfolio_2024, portfolio_2025, cvar_2024=cvar_2024, cvar_2025=cvar_2025
    )
    plots_dir = BUY_HOLD_EXP_DIR / "cross_year_plots" / "scenario"

    plot_cross_year_scatter(cross_year, plots_dir)
    plot_cross_year_pareto(cross_year, plots_dir)
    plot_cross_year_actual_vs_ev(cross_year, plots_dir)

    # Cross-year CVaR Pareto — graceful fallback
    if cvar_2024 is not None and cvar_2025 is not None:
        plot_cross_year_cvar_pareto(cross_year, plots_dir)
        plot_edge_sweep_pareto_overlay(cross_year, plots_dir)
    else:
        print("  Skipping cross-year CVaR Pareto — CVaR data not available for both years.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scenario-based analysis plots")
    parser.add_argument(
        "--ceremony-year",
        type=int,
        choices=sorted(YEAR_CONFIGS.keys()),
        help="Generate plots for a single year.",
    )
    parser.add_argument(
        "--cross-year",
        action="store_true",
        help="Generate cross-year plots.",
    )
    args = parser.parse_args()

    if args.ceremony_year:
        run_per_year(args.ceremony_year)

    if args.cross_year:
        run_cross_year()

    if not args.ceremony_year and not args.cross_year:
        for year in [2024, 2025]:
            run_per_year(year)
        run_cross_year()


if __name__ == "__main__":
    main()
