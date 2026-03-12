"""Oscar-specific prediction loading, namespace remapping, and source construction.

This module exists to keep a sharp boundary between raw model artifacts on
disk and the trading pipeline. ``modeling.prediction_io`` is intentionally
strict but generic; this module adds the Oscar/Kalshi-specific interpretation
layer needed for trading:

- load the exact nomination dataset that defines the Oscar-side namespace
- remap model outputs into the Kalshi-facing namespace for each category
- resolve nominee matching once per source
- construct ``SnapshotModel`` / ``EnsembleModel`` objects ready for backtests

Keeping those concerns together avoids scattering dict remaps and matching
rules across every one-off runner.
"""

import logging
from datetime import datetime
from pathlib import Path

from oscar_prediction_market.data.oscar_winners import (
    KALSHI_PERSON_NAME_CATEGORIES,
)
from oscar_prediction_market.data.schema import (
    NominationDataset,
    OscarCategory,
)
from oscar_prediction_market.modeling.models import ModelType
from oscar_prediction_market.modeling.prediction_io import (
    load_all_snapshot_predictions as load_raw_snapshot_predictions,
)
from oscar_prediction_market.modeling.prediction_io import (
    load_ensemble_predictions as load_raw_ensemble_predictions,
)
from oscar_prediction_market.trading.name_matching import match_nominees
from oscar_prediction_market.trading.oscar_market import OscarMarket
from oscar_prediction_market.trading.temporal_model import (
    EnsembleModel,
    SnapshotModel,
    TemporalModel,
)

logger = logging.getLogger(__name__)

_REMAP_BUILDERS = {
    **dict.fromkeys(
        KALSHI_PERSON_NAME_CATEGORIES,
        (
            NominationDataset.build_title_to_person_map,
            "title->person",
        ),
    ),
    # NOTE: OscarCategory.ORIGINAL_SCREENPLAY is intentionally omitted.
    # Model predictions use the ``title`` column from the CSV, which already
    # contains film titles — the same namespace Kalshi uses. The
    # ``build_nominee_to_film_title_map`` helper maps writer_name → film_title,
    # which would fail because prediction keys are already film titles.
}


def load_nomination_dataset(
    datasets_dir: Path,
    category: OscarCategory,
    snapshot_key: str,
) -> NominationDataset:
    """Load the exact nomination dataset for a category snapshot.

    Trading uses the nomination dataset as the ground-truth Oscar namespace
    for a snapshot. Failing loudly here is important: if the dataset is not
    exactly where the one-off expects it, we want the pipeline to stop rather
    than silently drift onto the wrong remap/matching basis.
    """
    cat_slug = category.slug
    target = datasets_dir / cat_slug / snapshot_key / f"oscar_{cat_slug}_raw.json"
    if not target.exists():
        raise FileNotFoundError(
            f"Missing canonical nomination dataset for {cat_slug}/{snapshot_key}: {target}"
        )
    logger.info("Loading nomination dataset from %s", target)
    return NominationDataset.model_validate_json(target.read_text())


def _build_prediction_remap(
    category: OscarCategory,
    dataset: NominationDataset,
    ceremony_year: int,
) -> tuple[dict[str, str], str | None]:
    """Build the raw-model-name -> Kalshi-name remap for one category.

    The policy is table-driven so category-specific naming behavior is
    declared in one place rather than hidden in several conditional branches.
    Categories not listed here are treated as identity mappings.
    """
    builder_info = _REMAP_BUILDERS.get(category)
    if builder_info is None:
        return {}, None

    builder, label = builder_info
    return builder(dataset, ceremony_year), label


def _remap_predictions_to_kalshi_names(
    category: OscarCategory,
    predictions: dict[str, float],
    dataset: NominationDataset,
    ceremony_year: int,
) -> dict[str, float]:
    """Convert raw prediction keys into the namespace used by Kalshi.

    Trading eventually has to compare predictions to Kalshi contracts, so we
    normalize names as early as possible in the trading path. Missing remap
    entries are treated as hard errors because silent partial remaps would
    create very misleading matching and backtest results.
    """
    remap, label = _build_prediction_remap(category, dataset, ceremony_year)
    if not remap:
        return dict(predictions)

    missing = sorted(name for name in predictions if name not in remap)
    if missing:
        raise KeyError(f"Missing {label} remap entries for {category.slug}: {missing}")
    return {remap[name]: prob for name, prob in predictions.items()}


def _remap_all_snapshots(
    category: OscarCategory,
    raw_predictions: dict[str, dict[str, float]],
    dataset: NominationDataset,
    ceremony_year: int,
) -> dict[str, dict[str, float]]:
    """Remap every snapshot's prediction keys into Kalshi namespace."""
    return {
        snapshot_key: _remap_predictions_to_kalshi_names(
            category=category,
            predictions=predictions,
            dataset=dataset,
            ceremony_year=ceremony_year,
        )
        for snapshot_key, predictions in raw_predictions.items()
    }


def translate_predictions(
    preds_by_snapshot: dict[str, dict[str, float]],
    nominee_map: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Translate prediction keys from remapped model names to exact Kalshi names.

    After ``_remap_all_snapshots`` converts raw model keys into a Kalshi-ish
    namespace, there may still be residual differences (case, accents) between
    the remapped names and the actual Kalshi registry names.  ``match_nominees``
    resolves those into ``nominee_map: {remapped_name: kalshi_name}``.

    This function applies that final translation so all downstream code
    (market prices, trading moments, settlement) works in a single
    kalshi-native namespace without repeated dict remapping.

    Only keys present in ``nominee_map`` are kept; unmatched keys are dropped.
    """
    return {
        snap_key: {nominee_map[name]: prob for name, prob in preds.items() if name in nominee_map}
        for snap_key, preds in preds_by_snapshot.items()
    }


def _resolve_snapshot_source(
    *,
    category: OscarCategory,
    market: OscarMarket,
    predictions_by_key: dict[str, dict[str, float]],
    snapshot_availability: dict[str, datetime],
    source_name: str,
    error_label: str,
    ceremony_year: int,
) -> tuple[TemporalModel, dict[str, str]]:
    """Build a snapshot-backed TemporalModel and the corresponding nominee map.

    The first remapped snapshot is used to establish the stable model-name ->
    Kalshi-name mapping for the source.  Predictions are then translated to
    exact Kalshi registry names so that ``SnapshotModel.get_predictions()``
    returns kalshi-native keys — matching ``build_market_prices()`` output
    and eliminating repeated model↔kalshi translation downstream.

    Raises ``ValueError`` if any model name fails to match a Kalshi contract.
    A partial match would silently distort probabilities (because the
    unmatched outcome's probability mass disappears), so we require 100%
    matching.
    """
    first_snap_preds = next(iter(predictions_by_key.values()))
    model_names = list(first_snap_preds.keys())
    kalshi_names = list(market.nominee_tickers.keys())
    nominee_map = match_nominees(
        model_names=model_names,
        kalshi_names=kalshi_names,
        category=category,
        ceremony_year=ceremony_year,
    )
    if not nominee_map:
        raise ValueError(f"No name matches for {error_label}/{category.slug}")

    unmatched = sorted(set(model_names) - set(nominee_map.keys()))
    if unmatched:
        raise ValueError(
            f"Incomplete nominee matching for {error_label}/{category.slug}: "
            f"{len(nominee_map)}/{len(model_names)} matched, "
            f"unmatched model names: {unmatched}"
        )

    # Translate predictions from remapped model names to exact Kalshi names
    kalshi_predictions = translate_predictions(predictions_by_key, nominee_map)

    return (
        SnapshotModel(
            predictions_by_key=kalshi_predictions,
            snapshot_availability=snapshot_availability,
            name=source_name,
        ),
        nominee_map,
    )


def load_all_snapshot_predictions(
    category: OscarCategory,
    model_type: ModelType,
    snapshot_keys: list[str],
    models_dir: Path,
    dataset: NominationDataset,
    ceremony_year: int,
) -> dict[str, dict[str, float]]:
    """Load and remap predictions across snapshots into Kalshi namespace."""
    raw = load_raw_snapshot_predictions(category, model_type, snapshot_keys, models_dir)
    return _remap_all_snapshots(category, raw, dataset, ceremony_year)


def load_ensemble_predictions(
    category: OscarCategory,
    model_types: list[ModelType],
    snapshot_keys: list[str],
    models_dir: Path,
    dataset: NominationDataset,
    ceremony_year: int,
) -> dict[str, dict[str, float]]:
    """Load equal-weight ensemble predictions remapped into Kalshi namespace."""
    raw = load_raw_ensemble_predictions(category, model_types, snapshot_keys, models_dir)
    return _remap_all_snapshots(category, raw, dataset, ceremony_year)


def build_model_source(
    category: OscarCategory,
    model_type: ModelType,
    market: OscarMarket,
    snapshot_key_strs: list[str],
    snapshot_availability: dict[str, datetime],
    models_dir: Path,
    dataset: NominationDataset,
    ceremony_year: int,
) -> tuple[TemporalModel, dict[str, str]]:
    """Build a ``SnapshotModel`` from Oscar predictions on disk.

    This is the standard path for a single trained model: load raw artifacts,
    remap them into the trading namespace, then package them as a temporal
    source plus the resolved nominee map used by the rest of the backtest
    pipeline.
    """
    raw = load_raw_snapshot_predictions(category, model_type, snapshot_key_strs, models_dir)
    predictions_by_key = _remap_all_snapshots(category, raw, dataset, ceremony_year)
    return _resolve_snapshot_source(
        category=category,
        market=market,
        predictions_by_key=predictions_by_key,
        snapshot_availability=snapshot_availability,
        source_name=model_type.short_name,
        error_label=model_type.short_name,
        ceremony_year=ceremony_year,
    )


def build_ensemble_source(
    category: OscarCategory,
    model_types: list[ModelType],
    market: OscarMarket,
    snapshot_key_strs: list[str],
    snapshot_availability: dict[str, datetime],
    models_dir: Path,
    dataset: NominationDataset,
    ceremony_year: int,
    ensemble_name: str = "avg_ensemble",
    weights: list[float] | None = None,
) -> tuple[TemporalModel, dict[str, str]]:
    """Build an ensemble ``TemporalModel`` from Oscar predictions on disk.

    The equal-weight path averages raw predictions first and then resolves one
    shared nominee map. The weighted path instead builds the component sources
    separately so the downstream ``EnsembleModel`` can combine them at query
    time with explicit weights.
    """
    if weights is None:
        raw = load_raw_ensemble_predictions(category, model_types, snapshot_key_strs, models_dir)
        predictions_by_key = _remap_all_snapshots(category, raw, dataset, ceremony_year)
        return _resolve_snapshot_source(
            category=category,
            market=market,
            predictions_by_key=predictions_by_key,
            snapshot_availability=snapshot_availability,
            source_name=ensemble_name,
            error_label="ensemble",
            ceremony_year=ceremony_year,
        )

    weighted_sources: list[SnapshotModel] = []
    nominee_map: dict[str, str] | None = None
    for model_type in model_types:
        source, model_nominee_map = build_model_source(
            category=category,
            model_type=model_type,
            market=market,
            snapshot_key_strs=snapshot_key_strs,
            snapshot_availability=snapshot_availability,
            models_dir=models_dir,
            dataset=dataset,
            ceremony_year=ceremony_year,
        )
        if nominee_map is None:
            nominee_map = model_nominee_map
        weighted_sources.append(source)  # type: ignore[arg-type]

    assert nominee_map is not None
    return (
        EnsembleModel(models=weighted_sources, weights=weights, name=ensemble_name),
        nominee_map,
    )
