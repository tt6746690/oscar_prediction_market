"""Consistent plot styling for Oscar prediction analysis.

Provides a unified visual style across all analysis modules:

- Color palettes for model types (both long and short naming conventions)
- Display names for model types and Oscar categories
- Awards season event labels for temporal analysis
- matplotlib rcParams for consistent appearance

Two naming conventions coexist because different experiments use different
model config names:

- **Long names** (``conditional_logit``, ``calibrated_softmax_gbt``) — used by
  ``d20260217_multinomial_modeling`` and earlier temporal snapshot experiments.
- **Short names** (``clogit``, ``cal_sgbt``) — used by ``d20260220_feature_ablation``
  and multi-category experiments.

Both are supported: :func:`get_model_color` and :func:`get_model_display`
accept either convention.

Usage::

    from oscar_prediction_market.one_offs.analysis_utils.style import (
        apply_style,
        get_model_color,
        get_model_display,
        AWARDS_SEASON_EVENTS,
    )
    apply_style()
"""

import matplotlib.pyplot as plt

# ============================================================================
# Model type display configuration
# ============================================================================

# Canonical color palette — covers both long and short model type names.
# When a model type isn't found here, ``get_model_color()`` auto-assigns
# a colour from a 10-colour qualitative palette.
MODEL_COLORS: dict[str, str] = {
    # Binary models
    "lr": "#3274a1",
    "gbt": "#e1812c",
    # Multinomial / ensemble — long names
    "conditional_logit": "#3a923a",
    "softmax_gbt": "#c03d3e",
    "calibrated_softmax_gbt": "#9372b2",
    "average": "#7f7f7f",
    # Multinomial / ensemble — short names (same colours as long equivalents)
    "clogit": "#3a923a",
    "cal_sgbt": "#9372b2",
    # Additional model types
    "market_blend": "#8c564b",
    "avg": "#7f7f7f",
}

MODEL_DISPLAY: dict[str, str] = {
    # Long names
    "lr": "Binary LR",
    "gbt": "Binary GBT",
    "conditional_logit": "Conditional Logit",
    "softmax_gbt": "Softmax GBT",
    "calibrated_softmax_gbt": "Cal. Softmax GBT",
    "average": "Average",
    # Short names
    "clogit": "Clogit",
    "cal_sgbt": "Cal-SGBT",
    "market_blend": "Market Blend",
    "avg": "Average",
}

# Fallback palette for previously-unseen model types (qualitative tab10).
_FALLBACK_COLORS = [
    "#17becf",
    "#bcbd22",
    "#e377c2",
    "#d62728",
    "#ff7f0e",
    "#2ca02c",
    "#1f77b4",
    "#9467bd",
    "#8c564b",
    "#7f7f7f",
]
_auto_color_index = 0


def get_model_color(model_type: str) -> str:
    """Return the canonical colour for *model_type*, auto-assigning if unknown."""
    global _auto_color_index  # noqa: PLW0603
    if model_type in MODEL_COLORS:
        return MODEL_COLORS[model_type]
    color = _FALLBACK_COLORS[_auto_color_index % len(_FALLBACK_COLORS)]
    MODEL_COLORS[model_type] = color
    _auto_color_index += 1
    return color


def get_model_display(model_type: str) -> str:
    """Return the human-readable display name for *model_type*."""
    return MODEL_DISPLAY.get(model_type, model_type)


# ============================================================================
# Oscar category display configuration
# ============================================================================

CATEGORIES: list[str] = [
    "best_picture",
    "directing",
    "actor_leading",
    "actress_leading",
    "actor_supporting",
    "actress_supporting",
    "original_screenplay",
    "cinematography",
    "animated_feature",
]

CATEGORY_SHORT: dict[str, str] = {
    "best_picture": "BP",
    "directing": "Dir",
    "actor_leading": "Act-L",
    "actress_leading": "Actr-L",
    "actor_supporting": "Act-S",
    "actress_supporting": "Actr-S",
    "original_screenplay": "Orig-SP",
    "cinematography": "Cine",
    "animated_feature": "Anim",
}

CATEGORY_DISPLAY: dict[str, str] = {
    "best_picture": "Best Picture",
    "directing": "Directing",
    "actor_leading": "Lead Actor",
    "actress_leading": "Lead Actress",
    "actor_supporting": "Supp. Actor",
    "actress_supporting": "Supp. Actress",
    "original_screenplay": "Original Screenplay",
    "cinematography": "Cinematography",
    "animated_feature": "Animated Feature",
}


# ============================================================================
# Awards season events (display-only constant)
# ============================================================================

AWARDS_SEASON_EVENTS: dict[str, str] = {
    "2025-11-30": "Pre-season baseline",
    "2025-12-05": "Critics Choice noms",
    "2025-12-08": "Golden Globe noms",
    "2026-01-04": "Critics Choice winner",
    "2026-01-07": "SAG noms",
    "2026-01-08": "DGA noms",
    "2026-01-09": "PGA noms",
    "2026-01-11": "Golden Globe winner",
    "2026-01-22": "Oscar noms",
    "2026-01-27": "BAFTA noms",
    "2026-02-07": "DGA winner",
}


# ============================================================================
# Plot style
# ============================================================================


def apply_style() -> None:
    """Apply consistent matplotlib style for all analysis plots.

    Call once at the start of any analysis script.  Sets rcParams that
    produce clean, readable figures with larger text for presentations.
    """
    plt.rcParams.update(
        {
            # Figure
            "figure.dpi": 150,
            "figure.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 150,
            # Font sizes — larger than defaults for readability
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            # Grid
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            # Lines
            "lines.linewidth": 2,
            "lines.markersize": 5,
            # Axes
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": plt.cycler(
                color=[
                    MODEL_COLORS["lr"],
                    MODEL_COLORS["gbt"],
                    MODEL_COLORS["conditional_logit"],
                    MODEL_COLORS["softmax_gbt"],
                    MODEL_COLORS["calibrated_softmax_gbt"],
                    MODEL_COLORS["average"],
                ]
            ),
        }
    )
