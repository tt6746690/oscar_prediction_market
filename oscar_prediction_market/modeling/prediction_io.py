"""Load raw model predictions from disk for backtesting and analysis.

These functions are strict raw I/O helpers. They read the canonical
``build_model`` output layout, return exactly what is stored on disk, and
do not perform Oscar-specific remapping into Kalshi namespaces.
"""

import logging
from pathlib import Path

import pandas as pd

from oscar_prediction_market.data.schema import OscarCategory
from oscar_prediction_market.modeling.models import ModelType

logger = logging.getLogger(__name__)


def load_predictions(
    category: OscarCategory,
    model_type: ModelType,
    snapshot_key: str,
    models_dir: Path,
) -> dict[str, float]:
    """Load raw model predictions for a (category, model_type, snapshot_key).

    Uses the canonical on-disk layout and reads only the latest
    ``5_final_predict/predictions_test.csv`` artifact. Missing files raise
    immediately so stale or non-standard experiment outputs are surfaced
    rather than silently skipped.

    Args:
        category: Oscar category.
        model_type: Model type to load.
        snapshot_key: Snapshot key dir_name (e.g. ``"2025-02-07_critics_choice"``).
        models_dir: Root models directory (e.g. ``storage/.../models``).

    Returns:
        ``{title: probability}`` exactly as stored in the prediction CSV.
    """
    cat_slug = category.slug
    short_name = model_type.short_name
    snap_dir = models_dir / cat_slug / short_name / snapshot_key
    run_name = f"{short_name}_{snapshot_key}"
    model_dir = snap_dir / run_name
    pred_path = model_dir / "5_final_predict" / "predictions_test.csv"
    logger.info("Loading raw predictions from %s", pred_path)
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Missing canonical prediction file for {category.slug}/{short_name}/{snapshot_key}: "
            f"{pred_path}"
        )

    df = pd.read_csv(pred_path)
    return dict(zip(df["title"], df["probability"], strict=True))


def load_all_snapshot_predictions(
    category: OscarCategory,
    model_type: ModelType,
    snapshot_keys: list[str],
    models_dir: Path,
) -> dict[str, dict[str, float]]:
    """Load raw predictions across snapshots.

    This stays intentionally dumb: it just iterates the canonical on-disk
    layout and returns the raw model namespace. Oscar/Kalshi name remapping
    happens later in ``trading.oscar_prediction_source`` so this module can
    remain reusable by non-trading analysis code.

    Args:
        category: Oscar category.
        model_type: Model type to load.
        snapshot_keys: List of snapshot key dir_names.
        models_dir: Root models directory.
    Returns:
        ``{snapshot_key: {title: prob}}`` for every requested snapshot key.
    """
    result: dict[str, dict[str, float]] = {}
    for snap_key in snapshot_keys:
        result[snap_key] = load_predictions(category, model_type, snap_key, models_dir)
    return result


def load_ensemble_predictions(
    category: OscarCategory,
    model_types: list[ModelType],
    snapshot_keys: list[str],
    models_dir: Path,
) -> dict[str, dict[str, float]]:
    """Load raw predictions from multiple models and average them per snapshot.

    For each snapshot key, loads predictions from every model type, finds the
    common set of nominees across models, and returns the equal-weight average
    probability. Probabilities are renormalized to sum to 1.0 after averaging.

    The equal-weight ensemble is kept here as a raw I/O convenience because it
    is still just "combine model artifacts that already exist on disk." Oscar-
    specific remapping into Kalshi namespaces still happens one layer later.

    Args:
        category: Oscar category.
        model_types: Models to ensemble (e.g. all 4 backtest models).
        snapshot_keys: Snapshot key strings (dir_names).
        models_dir: Root models directory.

    Returns:
        ``{snapshot_key: {title: avg_prob}}`` with probabilities summing to 1.
    """
    result: dict[str, dict[str, float]] = {}

    for snap_key in snapshot_keys:
        all_preds: list[dict[str, float]] = []
        for mt in model_types:
            all_preds.append(load_predictions(category, mt, snap_key, models_dir))

        # Find common nominees across all models that produced predictions
        common_names = set(all_preds[0].keys())
        for p in all_preds[1:]:
            common_names &= set(p.keys())

        if not common_names:
            raise ValueError(
                f"No common prediction names across {len(all_preds)} models for "
                f"{category.slug}/{snap_key}"
            )

        # Average probabilities
        avg: dict[str, float] = {}
        for name in common_names:
            avg[name] = sum(p[name] for p in all_preds) / len(all_preds)

        # Renormalize to sum to 1
        total = sum(avg.values())
        if total > 0:
            avg = {k: v / total for k, v in avg.items()}

        result[snap_key] = avg

    return result
