"""Visualization for counterfactual analysis results.

Reads counterfactual_summary.json and generates publication-quality figures:
1. Grouped bar chart: baseline vs each DGA-winner scenario
2. Waterfall/delta chart: probability shifts for DGA winners
3. Heatmap: nominee × scenario probability matrix
4. Dot plot with market reference: trading edge visualization
5. Intraday DGA reaction: hourly candles around announcement

Usage:
    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\
d20260212_counterfactual_analysis.analyze_counterfactual \
        --input-dir storage/d20260213_lr_feature_ablation/dga_winner \
        --output-dir storage/d20260213_lr_feature_ablation/dga_winner
"""

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from oscar_prediction_market.one_offs.analysis_utils.style import (
    get_model_color,
    get_model_display,
)

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# Short names for nominees (for axis labels)
SHORT_NAMES = {
    "One Battle after Another": "One Battle",
    "Marty Supreme": "Marty",
    "Frankenstein": "Frankenstein",
    "Hamnet": "Hamnet",
    "Sinners": "Sinners",
    "The Secret Agent": "Secret Agent",
    "Sentimental Value": "Sentimental",
    "Bugonia": "Bugonia",
    "Wicked: For Good": "Wicked",
    "Train Dreams": "Train Dreams",
    "F1": "F1",
}


def _short(name: str) -> str:
    return SHORT_NAMES.get(name, name[:12])


def load_summary(input_dir: Path) -> dict:
    """Load counterfactual summary JSON."""
    path = input_dir / "counterfactual_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Summary not found: {path}")
    with open(path) as f:
        return json.load(f)


# ============================================================================
# Plot 1: Grouped bar chart — baseline vs scenarios
# ============================================================================


def plot_grouped_bars(summary: dict, output_dir: Path) -> Path:
    """Grouped bar chart: for each DGA nominee, baseline + scenario probabilities."""
    eligible = summary["event"]["eligible_nominees"]
    models = sorted(summary["baselines"].keys())

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6), sharey=True)
    if len(models) == 1:
        axes = [axes]

    scenario_labels = ["Baseline"] + [f"If {_short(n)} wins" for n in eligible]
    colors = plt.cm.Set2(np.linspace(0, 1, len(scenario_labels)))  # type: ignore[attr-defined]

    for ax, model in zip(axes, models, strict=True):
        x = np.arange(len(eligible))
        width = 0.13
        n_bars = len(scenario_labels)
        offsets = np.arange(n_bars) * width - (n_bars - 1) * width / 2

        for i, (label, color) in enumerate(zip(scenario_labels, colors, strict=True)):
            vals = []
            for nominee in eligible:
                if i == 0:
                    # Baseline
                    p = summary["baselines"][model].get(nominee, {}).get("probability", 0)
                else:
                    # Scenario: i-1 is the DGA winner index
                    scenario_winner = eligible[i - 1]
                    key = f"{model}_{scenario_winner}"
                    preds = summary["scenarios"].get(key, {}).get("predictions", {})
                    p = preds.get(nominee, {}).get("probability", 0)
                vals.append(p * 100)
            ax.bar(x + offsets[i], vals, width, label=label, color=color, edgecolor="white")

        ax.set_xlabel("Nominee")
        ax.set_ylabel("P(Oscar Win) %")
        ax.set_title(f"{get_model_display(model)} — Baseline vs Scenarios")
        ax.set_xticks(x)
        ax.set_xticklabels([_short(n) for n in eligible], rotation=45, ha="right")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    path = output_dir / "counterfactual_grouped_bars.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


# ============================================================================
# Plot 2: Waterfall/delta chart
# ============================================================================


def plot_waterfall(summary: dict, output_dir: Path) -> Path:
    """Waterfall chart: baseline → scenario delta for each DGA winner candidate."""
    eligible = summary["event"]["eligible_nominees"]
    models = sorted(summary["baselines"].keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    y_positions = np.arange(len(eligible))
    bar_height = 0.25

    for i, model in enumerate(models):
        deltas = []
        bases = []
        for nominee in eligible:
            base_p = summary["baselines"][model].get(nominee, {}).get("probability", 0)
            key = f"{model}_{nominee}"
            scen_p = (
                summary["scenarios"]
                .get(key, {})
                .get("predictions", {})
                .get(nominee, {})
                .get("probability", 0)
            )
            deltas.append((scen_p - base_p) * 100)
            bases.append(base_p * 100)

        # Horizontal waterfall: start at baseline, extend by delta
        ax.barh(
            y_positions + i * bar_height - bar_height,
            deltas,
            bar_height,
            left=bases,
            color=get_model_color(model),
            label=f"{get_model_display(model)} delta",
            alpha=0.8,
        )
        # Mark baseline with a thin line
        for y, b in zip(y_positions + i * bar_height - bar_height, bases, strict=True):
            ax.plot(b, y, "|", color="black", markersize=12, markeredgewidth=2)

    # Market prices as vertical dashed lines
    for _j, nominee in enumerate(eligible):
        market_p = summary["market_prices"].get(nominee, 0)
        if market_p > 0:
            ax.axvline(market_p * 100, color="red", linestyle="--", alpha=0.3, linewidth=0.8)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([_short(n) for n in eligible])
    ax.set_xlabel("P(Oscar Win) %")
    ax.set_title("DGA Winner Counterfactual: Probability Shift (baseline → scenario)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = output_dir / "counterfactual_waterfall.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


# ============================================================================
# Plot 3: Heatmap — nominee × scenario
# ============================================================================


def plot_heatmap(summary: dict, output_dir: Path) -> Path:
    """Heatmap: rows=nominees, columns=DGA winner scenarios, cells=P(Oscar)."""
    eligible = summary["event"]["eligible_nominees"]
    models = sorted(summary["baselines"].keys())

    # Average across models for a clean single heatmap
    all_nominees = list(summary["baselines"][models[0]].keys())

    matrix = np.zeros((len(all_nominees), len(eligible)))
    for j, scenario_winner in enumerate(eligible):
        for i, nominee in enumerate(all_nominees):
            probs = []
            for model in models:
                key = f"{model}_{scenario_winner}"
                p = (
                    summary["scenarios"]
                    .get(key, {})
                    .get("predictions", {})
                    .get(nominee, {})
                    .get("probability", 0)
                )
                probs.append(p)
            matrix[i, j] = np.mean(probs) * 100

    fig, ax = plt.subplots(figsize=(8, max(6, len(all_nominees) * 0.5)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=80)

    ax.set_xticks(range(len(eligible)))
    ax.set_xticklabels([f"If {_short(n)}\nwins DGA" for n in eligible], fontsize=8)
    ax.set_yticks(range(len(all_nominees)))
    ax.set_yticklabels([_short(n) for n in all_nominees], fontsize=9)

    # Annotate cells
    for i in range(len(all_nominees)):
        for j in range(len(eligible)):
            val = matrix[i, j]
            color = "white" if val > 40 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=7, color=color)

    # Highlight diagonal (DGA winner = Oscar candidate)
    for j, nominee in enumerate(eligible):
        if nominee in all_nominees:
            i = all_nominees.index(nominee)
            ax.add_patch(
                plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="blue", linewidth=2)
            )

    ax.set_title("Mean P(Oscar Win) by DGA Winner Scenario (avg across models)")
    fig.colorbar(im, ax=ax, label="P(Oscar Win) %", shrink=0.8)
    plt.tight_layout()

    path = output_dir / "counterfactual_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


# ============================================================================
# Plot 4: Dot plot with market reference
# ============================================================================


def plot_dot_market(summary: dict, output_dir: Path) -> Path:
    """Dot plot: model mean P(win) for DGA winner vs Kalshi market price."""
    eligible = summary["event"]["eligible_nominees"]
    models = sorted(summary["baselines"].keys())

    fig, ax = plt.subplots(figsize=(10, 5))

    y_positions = np.arange(len(eligible))

    for nominee_idx, nominee in enumerate(eligible):
        # Market price
        market_p = summary["market_prices"].get(nominee, 0) * 100
        ax.plot(
            market_p,
            nominee_idx,
            "D",
            color="red",
            markersize=10,
            zorder=5,
            label="Kalshi" if nominee_idx == 0 else None,
        )

        # Model predictions
        for model_idx, model in enumerate(models):
            key = f"{model}_{nominee}"
            scen_p = (
                summary["scenarios"]
                .get(key, {})
                .get("predictions", {})
                .get(nominee, {})
                .get("probability", 0)
            ) * 100

            ax.plot(
                scen_p,
                nominee_idx + (model_idx - 1) * 0.12,
                "o",
                color=get_model_color(model),
                markersize=8,
                label=get_model_display(model) if nominee_idx == 0 else None,
                zorder=4,
            )

            # Arrow from market to model
            if abs(scen_p - market_p) > 2:
                ax.annotate(
                    "",
                    xy=(scen_p, nominee_idx + (model_idx - 1) * 0.12),
                    xytext=(market_p, nominee_idx),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": get_model_color(model),
                        "alpha": 0.3,
                        "lw": 1.5,
                    },
                )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([_short(n) for n in eligible])
    ax.set_xlabel("P(Oscar Win) %")
    ax.set_title("If Nominee Wins DGA: Model Prediction vs Market Price")
    ax.legend(loc="upper right")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(-5, 100)
    plt.tight_layout()

    path = output_dir / "counterfactual_dot_market.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


# ============================================================================
# Plot 5: Intraday DGA reaction
# ============================================================================


def plot_intraday_reaction(output_dir: Path) -> Path | None:
    """Plot hourly candle chart around DGA announcement.

    Fetches 1-hour candles for the top nominees around the DGA announcement
    (Feb 7, ~10:30pm ET) and plots price trajectories.
    """
    from oscar_prediction_market.data.schema import OscarCategory
    from oscar_prediction_market.trading.market_data import OSCAR_MARKETS
    from oscar_prediction_market.trading.oscar_market import OscarMarket

    bp_data = OSCAR_MARKETS.get_category_data(OscarCategory.BEST_PICTURE, 2026)
    bp_nominee_tickers = bp_data.nominee_tickers

    # DGA 2026 announcement: ~10:30pm ET on Feb 7
    event_date = date(2026, 2, 7)
    event_time = "22:30"

    # Focus on top 5 nominees with most trading activity
    top_tickers = [
        bp_nominee_tickers["One Battle after Another"],
        bp_nominee_tickers["Sinners"],
        bp_nominee_tickers["Hamnet"],
        bp_nominee_tickers["Marty Supreme"],
        bp_nominee_tickers["Frankenstein"],
    ]

    mkt = OscarMarket(event_ticker=bp_data.event_ticker, nominee_tickers=bp_nominee_tickers)
    try:
        df = mkt.fetch_intraday_prices(
            event_date=event_date,
            event_time_et=event_time,
            hours_before=6,
            hours_after=12,
            period_interval=60,
            tickers=top_tickers,
        )
    except Exception as e:
        logger.warning(f"Failed to fetch intraday prices: {e}")
        return None

    if df.empty:
        logger.warning("No intraday price data available")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    nominees = df["nominee"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(nominees)))  # type: ignore[attr-defined]

    for nominee, color in zip(nominees, colors, strict=True):
        ndf = df[df["nominee"] == nominee].sort_values("time_relative_hours")
        if ndf.empty:
            continue
        ax.plot(
            ndf["time_relative_hours"],
            ndf["close"],
            "-o",
            color=color,
            label=_short(nominee),
            markersize=3,
            linewidth=1.5,
        )

    # Mark event time
    ax.axvline(0, color="red", linestyle="--", linewidth=2, alpha=0.7, label="DGA announcement")
    ax.axvspan(-0.5, 0.5, alpha=0.1, color="red")

    ax.set_xlabel("Hours relative to DGA announcement")
    ax.set_ylabel("Price (cents)")
    ax.set_title("Kalshi Oscar BP Prices Around DGA Winner Announcement (Feb 7, 2026)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = output_dir / "intraday_dga_reaction.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate counterfactual analysis visualizations")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory with counterfactual_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for output plots",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(input_dir)

    print(f"\nGenerating counterfactual visualizations from: {input_dir}")
    print(f"Output directory: {output_dir}")

    plots: list[Path | None] = []
    plots.append(plot_grouped_bars(summary, output_dir))
    plots.append(plot_waterfall(summary, output_dir))
    plots.append(plot_heatmap(summary, output_dir))
    plots.append(plot_dot_market(summary, output_dir))
    plots.append(plot_intraday_reaction(output_dir))

    print(f"\nGenerated {sum(1 for p in plots if p is not None)} plots:")
    for p in plots:
        if p is not None:
            print(f"  {p}")


if __name__ == "__main__":
    main()
