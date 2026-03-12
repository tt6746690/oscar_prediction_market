"""Temporal prediction models for backtesting.

Defines a clean interface for anything that produces probability predictions
at a point in time: individual snapshot-based models, ensembles, or any
combination thereof.

The key abstraction is ``TemporalModel`` — a Protocol for "a model you can
query with a UTC timestamp."  Implementations handle snapshot resolution
internally: given a timestamp, they determine which underlying model snapshot
was available at that point in time and return its predictions.

The **delay model stays external** — callers compute ``snapshot_availability``
as ``{s.dir_name: s.event_datetime_utc + timedelta(hours=lag) for s in snapshots}``
and pass the resulting ``{dir_name: datetime}`` mapping into the model constructor.
To change the lag, you change the availability mapping, not the model.

Example::

    from datetime import datetime, UTC
    from oscar_prediction_market.trading.temporal_model import (
        SnapshotModel, EnsembleModel,
    )

    preds = {"2026-02-01_oscar_noms": {"Anora": 0.45, "The Brutalist": 0.30, "Conclave": 0.25}}
    availability = {"2026-02-01_oscar_noms": datetime(2026, 2, 1, 14, 30, tzinfo=UTC)}

    model = SnapshotModel(
        predictions_by_key=preds,
        snapshot_availability=availability,
        name="lr",
    )

    # Query at a specific timestamp — resolves to the correct snapshot
    probs = model.get_predictions(datetime(2026, 2, 3, 16, 0, tzinfo=UTC))
    # => {"Anora": 0.45, "The Brutalist": 0.30, "Conclave": 0.25}
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from oscar_prediction_market.data.awards_calendar import (
    PRECURSOR_ORGS,
    AwardOrg,
    AwardsCalendar,
    EventPhase,
)

# ============================================================================
# Snapshot Info
# ============================================================================


class SnapshotInfo(BaseModel):
    """Pre-resolved snapshot metadata for temporal model and backtesting.

    Each snapshot corresponds to an awards-season event (Oscar nominations
    or a precursor winner) after which new information is available for
    predictions.  The dir_name embeds the local date + event suffix
    (e.g., "2025-02-08_dga") and serves as the on-disk directory name
    and dictionary key throughout the system.

    Produced by get_snapshot_sequence(), consumed by backtesting scripts
    and SnapshotModel.
    """

    model_config = {"extra": "forbid", "frozen": True}

    dir_name: str = Field(..., description="On-disk directory name: YYYY-MM-DD_<suffix>")
    event_datetime_utc: datetime = Field(
        ...,
        description="UTC datetime when the event results become public",
    )

    def __hash__(self) -> int:
        return hash(self.dir_name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SnapshotInfo):
            return self.dir_name == other.dir_name
        return NotImplemented

    def __str__(self) -> str:
        return self.dir_name


def get_snapshot_sequence(
    calendar: AwardsCalendar,
) -> list[SnapshotInfo]:
    """Return chronological sequence of per-event snapshot infos.

    Starts with Oscar nominations, then each post-nomination precursor
    winner event ordered by UTC datetime. Pre-nomination events (precursors
    whose local winner date <= oscar nominations local date) are excluded —
    their results are baked into the nominations snapshot.

    Each snapshot represents a point where new information becomes
    available for feature engineering. Same-day events produce
    separate snapshots ordered by datetime.

    Args:
        calendar: Awards calendar for the ceremony year.

    Returns:
        Sorted list of SnapshotInfo from nominations through
        the last pre-ceremony precursor event.
    """
    nom_date_local = calendar.oscar_nominations_date_local
    ceremony_date_local = calendar.oscar_ceremony_date_local
    nom_dt = calendar.get_event_datetime(AwardOrg.OSCAR, EventPhase.NOMINATION)

    snapshots: list[SnapshotInfo] = [
        SnapshotInfo(
            dir_name=f"{nom_date_local.isoformat()}_oscar_noms",
            event_datetime_utc=nom_dt,
        )
    ]

    # Collect post-nomination, pre-ceremony precursor winner events
    post_nom: list[tuple[datetime, AwardOrg]] = []
    for org in PRECURSOR_ORGS:
        key = (org, EventPhase.WINNER)
        if key not in calendar.events:
            continue
        winner_date_local = calendar.get_local_date(org, EventPhase.WINNER)
        if winner_date_local > nom_date_local and winner_date_local < ceremony_date_local:
            winner_dt = calendar.get_event_datetime(org, EventPhase.WINNER)
            post_nom.append((winner_dt, org))

    # Sort by UTC datetime; tiebreak by org name for deterministic order
    # when two events share the same datetime (e.g., ASC + WGA same night)
    post_nom.sort(key=lambda x: (x[0], x[1].value))

    for winner_dt, org in post_nom:
        winner_date_local = calendar.get_local_date(org, EventPhase.WINNER)
        snapshots.append(
            SnapshotInfo(
                dir_name=f"{winner_date_local.isoformat()}_{org.value}",
                event_datetime_utc=winner_dt,
            )
        )

    return snapshots


def get_trading_dates(
    calendar: AwardsCalendar,
) -> list[date]:
    """Get all trading dates from Oscar nominations through day before ceremony.

    Returns all days (Kalshi trades on weekends too for event markets).

    Args:
        calendar: Awards calendar.

    Returns:
        Sorted list of trading dates.
    """
    start = calendar.oscar_nominations_date_local
    # Day before ceremony
    end = calendar.oscar_ceremony_date_local - timedelta(days=1)

    dates = []
    current = start
    while current <= end:
        # Include all days (Kalshi trades on weekends too for events)
        dates.append(current)
        current += timedelta(days=1)

    return dates


def get_post_nomination_snapshot_dates(
    calendar: AwardsCalendar,
) -> list[tuple[date, list[str]]]:
    """Derive post-nomination snapshot dates from an awards calendar, grouping same-day events.

    Returns:
        List of ``(as_of_date, [event_dir_names])`` sorted chronologically.
        First entry is always ``(nominations_date, ["<date>_oscar_noms"])``.
    """
    snapshots = get_snapshot_sequence(calendar)
    # Group by date (UTC date from event_datetime_utc)
    events_by_date: dict[date, list[str]] = {}
    for snap in snapshots:
        snap_date = snap.event_datetime_utc.date()
        events_by_date.setdefault(snap_date, []).append(snap.dir_name)
    return [(d, events_by_date[d]) for d in sorted(events_by_date)]


# ============================================================================
# Snapshot Resolution
# ============================================================================


def _get_active_snapshot(
    timestamp: datetime,
    snapshot_keys: list[str],
    snapshot_availability: dict[str, datetime],
) -> str | None:
    """Get the most recent snapshot key available at the given timestamp.

    Iterates through snapshot keys chronologically and returns the latest
    one whose ``available_at <= timestamp``.

    Args:
        timestamp: UTC datetime representing the current moment.
        snapshot_keys: Sorted list of snapshot dir_names (must be chronological).
        snapshot_availability: Mapping from dir_name to available_at (UTC).

    Returns:
        The active snapshot dir_name, or None if no snapshot is available yet.
    """
    active: str | None = None
    for key in snapshot_keys:
        available_at = snapshot_availability.get(key)
        if available_at is None:
            continue
        if available_at <= timestamp:
            active = key
        else:
            break
    return active


@runtime_checkable
class TemporalModel(Protocol):
    """Anything that produces ``{outcome: probability}`` predictions at a point in time.

    Implementations must provide:

    ``name``
        Human-readable label (used in logging, CSV columns, etc.).
    ``get_predictions(timestamp)``
        Return ``{outcome: probability}`` for the given UTC timestamp.
        Internally resolves to the appropriate model snapshot.
        Must raise ``KeyError`` if no predictions are available at that time.
    ``available_snapshots()``
        Return sorted list of snapshot keys backing this model.
    """

    @property
    def name(self) -> str: ...

    def get_predictions(self, timestamp: datetime) -> dict[str, float]: ...

    def available_snapshots(self) -> list[str]: ...


# ============================================================================
# SnapshotModel
# ============================================================================


class SnapshotModel:
    """Snapshot-based temporal model.

    Backed by discrete model snapshots (one per snapshot key), each with an
    ``available_at`` UTC timestamp.  Given a query timestamp, resolves to the
    latest snapshot whose ``available_at <= timestamp`` and returns its
    predictions.

    Predictions are keyed by snapshot dir_name internally (e.g.,
    ``"2025-02-08_dga"``). Snapshot resolution uses string keys
    from the snapshot availability dict.

    Always returns *copies* of prediction dicts so callers can filter or
    mutate without affecting other consumers of the same model.

    Args:
        predictions_by_key: ``{"dir_name": {"outcome": probability, ...}, ...}``
        snapshot_availability: ``{dir_name: available_at_utc}`` mapping computed
            as ``{s.dir_name: s.event_datetime_utc + timedelta(hours=lag)}``.
            Determines when each snapshot's predictions become queryable.
        name: Human-readable label for this model (e.g. ``"lr"``, ``"cal_sgbt"``).
    """

    def __init__(
        self,
        predictions_by_key: dict[str, dict[str, float]],
        snapshot_availability: dict[str, datetime],
        name: str,
    ) -> None:
        self._predictions = predictions_by_key
        self._snapshot_availability = snapshot_availability
        # Sort keys by their available_at datetime for chronological iteration
        self._snapshot_keys = sorted(
            snapshot_availability.keys(), key=lambda k: snapshot_availability[k]
        )
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def snapshot_availability(self) -> dict[str, datetime]:
        """Snapshot key → available_at (UTC) mapping."""
        return dict(self._snapshot_availability)

    def get_predictions(self, timestamp: datetime) -> dict[str, float]:
        """Return ``{outcome: probability}`` for the given UTC timestamp.

        Resolves to the latest snapshot available at ``timestamp``.

        Raises:
            KeyError: If no snapshot is available at ``timestamp``.
        """
        active = _get_active_snapshot(timestamp, self._snapshot_keys, self._snapshot_availability)
        if active is None:
            raise KeyError(
                f"No snapshot available at {timestamp.isoformat()} in model '{self._name}'"
            )
        return self.get_predictions_for_key(active)

    def get_predictions_for_key(self, key: str) -> dict[str, float]:
        """Return predictions for a specific snapshot key (bypassing temporal resolution).

        Useful for analysis code that needs raw per-snapshot predictions
        (e.g. model-vs-model comparisons, accuracy evaluation at each snapshot).

        Raises:
            KeyError: If no predictions exist for ``key``.
        """
        if key not in self._predictions:
            raise KeyError(f"No predictions for snapshot {key!r} in model '{self._name}'")
        return dict(self._predictions[key])  # copy

    def available_snapshots(self) -> list[str]:
        """Return sorted list of snapshot keys."""
        return list(self._snapshot_keys)

    def __repr__(self) -> str:
        return f"SnapshotModel(name={self._name!r}, n_snapshots={len(self._predictions)})"


# ============================================================================
# EnsembleModel
# ============================================================================


class EnsembleModel:
    """Averages predictions from multiple TemporalModel instances.

    For each query timestamp, predictions are combined by:

    1. **Outcome intersection** — only outcomes present in *all* models are kept.
       This avoids introducing bias from models trained on different candidate sets.
    2. **Weighted average** — if ``weights`` is provided, applies those weights
       (normalized to sum to 1.0 internally).  Default is equal weight.
    3. **Renormalization** — the averaged probabilities are rescaled to sum to 1.0.

    Available snapshots are the **intersection** across all child models.

    Args:
        models: One or more TemporalModel instances.
        weights: Per-model weights (must be positive).  ``None`` = equal weight.
        name: Custom name.  Default: ``"ensemble_{n_models}"``.

    Raises:
        ValueError: If ``models`` is empty, ``weights`` length doesn't match,
            or any weight is ≤ 0.
    """

    def __init__(
        self,
        models: Sequence[TemporalModel],
        weights: list[float] | None = None,
        name: str | None = None,
    ) -> None:
        if not models:
            raise ValueError("EnsembleModel requires at least one model")

        if weights is not None:
            if len(weights) != len(models):
                raise ValueError(
                    f"weights length ({len(weights)}) must match models length ({len(models)})"
                )
            if any(w <= 0 for w in weights):
                raise ValueError("All weights must be positive (> 0)")
            total = sum(weights)
            self._weights = [w / total for w in weights]
        else:
            n = len(models)
            self._weights = [1.0 / n] * n

        self._models = list(models)
        self._name = name or f"ensemble_{len(models)}"

    @property
    def name(self) -> str:
        return self._name

    def get_predictions(self, timestamp: datetime) -> dict[str, float]:
        """Return weighted-average predictions for the given timestamp.

        Raises:
            KeyError: If any child model lacks predictions at ``timestamp``.
        """
        all_preds = [m.get_predictions(timestamp) for m in self._models]

        # Find common outcomes
        common_outcomes = set(all_preds[0].keys())
        for preds in all_preds[1:]:
            common_outcomes &= set(preds.keys())

        if not common_outcomes:
            raise KeyError(
                f"No common outcomes across {len(self._models)} models at {timestamp.isoformat()}"
            )

        # Weighted average
        result: dict[str, float] = {}
        for outcome in common_outcomes:
            result[outcome] = sum(
                w * preds[outcome] for w, preds in zip(self._weights, all_preds, strict=True)
            )

        # Renormalize to sum = 1
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}

        return result

    def available_snapshots(self) -> list[str]:
        """Return sorted intersection of snapshots across all models."""
        key_sets = [set(m.available_snapshots()) for m in self._models]
        common = key_sets[0]
        for ks in key_sets[1:]:
            common &= ks
        return sorted(common)

    def __repr__(self) -> str:
        model_names = [m.name for m in self._models]
        return (
            f"EnsembleModel(name={self._name!r}, "
            f"models={model_names}, "
            f"weights={[round(w, 3) for w in self._weights]})"
        )
