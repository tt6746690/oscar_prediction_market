"""Tests for temporal_model.py — TemporalModel protocol, SnapshotModel, EnsembleModel.

Also includes regression tests for get_snapshot_sequence() and SnapshotInfo
added during the AwardsCalendar flat-events-dict refactor.
"""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from oscar_prediction_market.data.awards_calendar import (
    CALENDARS,
    AwardOrg,
    AwardsCalendar,
    EventPhase,
)
from oscar_prediction_market.trading.temporal_model import (
    EnsembleModel,
    SnapshotInfo,
    SnapshotModel,
    TemporalModel,
    get_snapshot_sequence,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def two_snapshot_preds() -> dict[str, dict[str, float]]:
    """Predictions for two snapshot keys."""
    return {
        "2026-02-01_oscar_noms": {"Anora": 0.50, "The Brutalist": 0.30, "Conclave": 0.20},
        "2026-02-08_dga": {"Anora": 0.60, "The Brutalist": 0.25, "Conclave": 0.15},
    }


@pytest.fixture()
def snapshot_availability() -> dict[str, datetime]:
    """Snapshot availability: each available at 2 PM UTC on its date."""
    return {
        "2026-02-01_oscar_noms": datetime(2026, 2, 1, 14, 0, tzinfo=UTC),
        "2026-02-08_dga": datetime(2026, 2, 8, 14, 0, tzinfo=UTC),
    }


@pytest.fixture()
def lr_model(
    two_snapshot_preds: dict[str, dict[str, float]],
    snapshot_availability: dict[str, datetime],
) -> SnapshotModel:
    return SnapshotModel(
        predictions_by_key=two_snapshot_preds,
        snapshot_availability=snapshot_availability,
        name="lr",
    )


# ============================================================================
# SnapshotModel: basic functionality
# ============================================================================


class TestSnapshotModel:
    """SnapshotModel construction and basic querying."""

    def test_protocol_conformance(self, lr_model: SnapshotModel) -> None:
        """SnapshotModel is a TemporalModel."""
        assert isinstance(lr_model, TemporalModel)

    def test_name(self, lr_model: SnapshotModel) -> None:
        assert lr_model.name == "lr"

    def test_available_snapshots(self, lr_model: SnapshotModel) -> None:
        assert lr_model.available_snapshots() == [
            "2026-02-01_oscar_noms",
            "2026-02-08_dga",
        ]

    def test_snapshot_availability_property(self, lr_model: SnapshotModel) -> None:
        avail = lr_model.snapshot_availability
        assert len(avail) == 2
        assert "2026-02-01_oscar_noms" in avail

    def test_repr(self, lr_model: SnapshotModel) -> None:
        r = repr(lr_model)
        assert "SnapshotModel" in r
        assert "lr" in r


# ============================================================================
# SnapshotModel: temporal resolution
# ============================================================================


class TestSnapshotModelTemporalResolution:
    """Querying a SnapshotModel at various timestamps resolves to the correct snapshot.

    Timeline:
        Feb 1 14:00 UTC  — snapshot 1 becomes available
        Feb 8 14:00 UTC  — snapshot 2 becomes available

    Queries before Feb 1 14:00 should raise KeyError.
    Queries between Feb 1 14:00 and Feb 8 14:00 should return snapshot 1.
    Queries after Feb 8 14:00 should return snapshot 2 (latest available).
    """

    def test_before_any_snapshot_raises(self, lr_model: SnapshotModel) -> None:
        """Query before first availability → KeyError."""
        ts = datetime(2026, 2, 1, 13, 0, tzinfo=UTC)  # 1h before first avail
        with pytest.raises(KeyError, match="No snapshot available"):
            lr_model.get_predictions(ts)

    def test_exactly_at_first_availability(self, lr_model: SnapshotModel) -> None:
        """Query exactly at first snapshot's available_at → returns snapshot 1."""
        ts = datetime(2026, 2, 1, 14, 0, tzinfo=UTC)
        preds = lr_model.get_predictions(ts)
        assert preds["Anora"] == pytest.approx(0.50)

    def test_between_snapshots(self, lr_model: SnapshotModel) -> None:
        """Query between snapshot 1 and 2 availability → returns snapshot 1."""
        ts = datetime(2026, 2, 5, 16, 0, tzinfo=UTC)
        preds = lr_model.get_predictions(ts)
        assert preds["Anora"] == pytest.approx(0.50)  # still snapshot 1

    def test_after_second_snapshot(self, lr_model: SnapshotModel) -> None:
        """Query after snapshot 2 availability → returns snapshot 2."""
        ts = datetime(2026, 2, 10, 16, 0, tzinfo=UTC)
        preds = lr_model.get_predictions(ts)
        assert preds["Anora"] == pytest.approx(0.60)  # snapshot 2

    def test_exactly_at_second_availability(self, lr_model: SnapshotModel) -> None:
        """Query exactly at snapshot 2's available_at → returns snapshot 2."""
        ts = datetime(2026, 2, 8, 14, 0, tzinfo=UTC)
        preds = lr_model.get_predictions(ts)
        assert preds["Anora"] == pytest.approx(0.60)

    def test_returns_copy(self, lr_model: SnapshotModel) -> None:
        """Mutations by the caller don't affect the model's internal state."""
        ts = datetime(2026, 2, 5, 16, 0, tzinfo=UTC)
        preds = lr_model.get_predictions(ts)
        preds["Anora"] = 999.0
        fresh = lr_model.get_predictions(ts)
        assert fresh["Anora"] == pytest.approx(0.50)


# ============================================================================
# SnapshotModel: get_predictions_for_snapshot
# ============================================================================


class TestSnapshotModelDirectAccess:
    """Bypass temporal resolution with get_predictions_for_key."""

    def test_direct_access(self, lr_model: SnapshotModel) -> None:
        preds = lr_model.get_predictions_for_key("2026-02-01_oscar_noms")
        assert preds["Anora"] == pytest.approx(0.50)

    def test_direct_access_second_snapshot(self, lr_model: SnapshotModel) -> None:
        preds = lr_model.get_predictions_for_key("2026-02-08_dga")
        assert preds["Anora"] == pytest.approx(0.60)

    def test_direct_access_missing_key_raises(self, lr_model: SnapshotModel) -> None:
        with pytest.raises(KeyError, match="No predictions for snapshot"):
            lr_model.get_predictions_for_key("2026-03-01_bafta")


# ============================================================================
# EnsembleModel: basic functionality
# ============================================================================


class TestEnsembleModel:
    """EnsembleModel construction and weighted averaging."""

    @pytest.fixture()
    def gbt_model(self, snapshot_availability: dict[str, datetime]) -> SnapshotModel:
        return SnapshotModel(
            predictions_by_key={
                "2026-02-01_oscar_noms": {"Anora": 0.40, "The Brutalist": 0.35, "Conclave": 0.25},
                "2026-02-08_dga": {"Anora": 0.55, "The Brutalist": 0.30, "Conclave": 0.15},
            },
            snapshot_availability=snapshot_availability,
            name="gbt",
        )

    def test_protocol_conformance(self, lr_model: SnapshotModel, gbt_model: SnapshotModel) -> None:
        ensemble = EnsembleModel(models=[lr_model, gbt_model])
        assert isinstance(ensemble, TemporalModel)

    def test_equal_weight_averaging(
        self, lr_model: SnapshotModel, gbt_model: SnapshotModel
    ) -> None:
        """Equal-weight ensemble of two models.

        For snapshot 1 (Feb 1):
            LR:  Anora=0.50, Brutalist=0.30, Conclave=0.20
            GBT: Anora=0.40, Brutalist=0.35, Conclave=0.25

        Raw average: Anora=0.45, Brutalist=0.325, Conclave=0.225
        Sum = 1.00 → no rescaling needed.
        """
        ensemble = EnsembleModel(models=[lr_model, gbt_model], name="avg")
        ts = datetime(2026, 2, 5, 16, 0, tzinfo=UTC)  # after snap 1, before snap 2
        preds = ensemble.get_predictions(ts)
        assert preds["Anora"] == pytest.approx(0.45)
        assert preds["The Brutalist"] == pytest.approx(0.325)

    def test_weighted_averaging(self, lr_model: SnapshotModel, gbt_model: SnapshotModel) -> None:
        """Weighted ensemble: 75% LR, 25% GBT.

        For snapshot 1:
            Raw weighted: Anora = 0.75*0.50 + 0.25*0.40 = 0.475
            Brutalist    = 0.75*0.30 + 0.25*0.35 = 0.3125
            Conclave     = 0.75*0.20 + 0.25*0.25 = 0.2125
            Sum = 1.00 → no rescaling.
        """
        ensemble = EnsembleModel(models=[lr_model, gbt_model], weights=[3.0, 1.0], name="w_ens")
        ts = datetime(2026, 2, 5, 16, 0, tzinfo=UTC)
        preds = ensemble.get_predictions(ts)
        assert preds["Anora"] == pytest.approx(0.475)

    def test_resolves_to_correct_snapshot(
        self, lr_model: SnapshotModel, gbt_model: SnapshotModel
    ) -> None:
        """Ensemble after snapshot 2 uses snapshot 2 predictions."""
        ensemble = EnsembleModel(models=[lr_model, gbt_model])
        ts = datetime(2026, 2, 10, 16, 0, tzinfo=UTC)
        preds = ensemble.get_predictions(ts)
        # LR snap2: Anora=0.60, GBT snap2: Anora=0.55 → avg=0.575
        assert preds["Anora"] == pytest.approx(0.575)

    def test_before_any_snapshot_raises(
        self, lr_model: SnapshotModel, gbt_model: SnapshotModel
    ) -> None:
        ensemble = EnsembleModel(models=[lr_model, gbt_model])
        ts = datetime(2026, 1, 31, 16, 0, tzinfo=UTC)
        with pytest.raises(KeyError):
            ensemble.get_predictions(ts)

    def test_available_snapshots_intersection(
        self,
        snapshot_availability: dict[str, datetime],
    ) -> None:
        """Ensemble available_snapshots is the intersection of child models."""
        # Model A has both snapshots, Model B only has the second
        avail_b = {"2026-02-08_dga": snapshot_availability["2026-02-08_dga"]}
        model_a = SnapshotModel(
            predictions_by_key={
                "2026-02-01_oscar_noms": {"X": 0.5, "Y": 0.5},
                "2026-02-08_dga": {"X": 0.6, "Y": 0.4},
            },
            snapshot_availability=snapshot_availability,
            name="a",
        )
        model_b = SnapshotModel(
            predictions_by_key={"2026-02-08_dga": {"X": 0.7, "Y": 0.3}},
            snapshot_availability=avail_b,
            name="b",
        )
        ensemble = EnsembleModel(models=[model_a, model_b])
        assert ensemble.available_snapshots() == ["2026-02-08_dga"]

    def test_default_name(self, lr_model: SnapshotModel, gbt_model: SnapshotModel) -> None:
        ensemble = EnsembleModel(models=[lr_model, gbt_model])
        assert ensemble.name == "ensemble_2"

    def test_custom_name(self, lr_model: SnapshotModel, gbt_model: SnapshotModel) -> None:
        ensemble = EnsembleModel(models=[lr_model, gbt_model], name="my_ensemble")
        assert ensemble.name == "my_ensemble"

    def test_repr(self, lr_model: SnapshotModel, gbt_model: SnapshotModel) -> None:
        ensemble = EnsembleModel(models=[lr_model, gbt_model])
        r = repr(ensemble)
        assert "EnsembleModel" in r
        assert "lr" in r
        assert "gbt" in r


# ============================================================================
# EnsembleModel: validation
# ============================================================================


class TestEnsembleValidation:
    """EnsembleModel rejects invalid inputs."""

    def test_empty_models_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one model"):
            EnsembleModel(models=[])

    def test_weights_length_mismatch_raises(self, lr_model: SnapshotModel) -> None:
        with pytest.raises(ValueError, match="weights length"):
            EnsembleModel(models=[lr_model], weights=[1.0, 2.0])

    def test_zero_weight_raises(self, lr_model: SnapshotModel) -> None:
        with pytest.raises(ValueError, match="positive"):
            EnsembleModel(models=[lr_model], weights=[0.0])

    def test_negative_weight_raises(self, lr_model: SnapshotModel) -> None:
        with pytest.raises(ValueError, match="positive"):
            EnsembleModel(models=[lr_model], weights=[-1.0])


# ============================================================================
# EnsembleModel: outcome intersection
# ============================================================================


class TestEnsembleOutcomeIntersection:
    """Ensemble only includes outcomes present in ALL child models."""

    def test_extra_outcome_in_one_model_excluded(
        self, snapshot_availability: dict[str, datetime]
    ) -> None:
        """Model A has {X, Y, Z}, Model B has {X, Y} → ensemble returns {X, Y}.

        Model A: X=0.40, Y=0.30, Z=0.30
        Model B: X=0.60, Y=0.40

        Common outcomes: {X, Y}
        Raw avg: X = (0.40+0.60)/2 = 0.50, Y = (0.30+0.40)/2 = 0.35
        Total = 0.85 → renormalized: X = 0.50/0.85 ≈ 0.5882, Y = 0.35/0.85 ≈ 0.4118
        """
        model_a = SnapshotModel(
            predictions_by_key={"2026-02-01_oscar_noms": {"X": 0.40, "Y": 0.30, "Z": 0.30}},
            snapshot_availability=snapshot_availability,
            name="a",
        )
        model_b = SnapshotModel(
            predictions_by_key={"2026-02-01_oscar_noms": {"X": 0.60, "Y": 0.40}},
            snapshot_availability=snapshot_availability,
            name="b",
        )
        ensemble = EnsembleModel(models=[model_a, model_b])
        ts = datetime(2026, 2, 5, 16, 0, tzinfo=UTC)
        preds = ensemble.get_predictions(ts)
        assert "Z" not in preds
        assert preds["X"] == pytest.approx(0.50 / 0.85, abs=1e-4)
        assert abs(sum(preds.values()) - 1.0) < 1e-9

    def test_no_common_outcomes_raises(self, snapshot_availability: dict[str, datetime]) -> None:
        model_a = SnapshotModel(
            predictions_by_key={"2026-02-01_oscar_noms": {"X": 1.0}},
            snapshot_availability=snapshot_availability,
            name="a",
        )
        model_b = SnapshotModel(
            predictions_by_key={"2026-02-01_oscar_noms": {"Y": 1.0}},
            snapshot_availability=snapshot_availability,
            name="b",
        )
        ensemble = EnsembleModel(models=[model_a, model_b])
        ts = datetime(2026, 2, 5, 16, 0, tzinfo=UTC)
        with pytest.raises(KeyError, match="No common outcomes"):
            ensemble.get_predictions(ts)


# ============================================================================
# SnapshotModel: different availability windows
# ============================================================================


class TestDifferentAvailabilityWindows:
    """Models with different delay configs have different snapshot_availability,
    leading to different temporal resolution behavior even with the same predictions.

    This demonstrates the design: the delay model stays external.
    """

    def test_same_preds_different_delay(self) -> None:
        """Same predictions, but one model has a 1-day delay.

        Model A: snapshot available same day at 14:00
        Model B: same snapshot available NEXT day at 14:00

        Query at Feb 1 16:00:
          Model A → resolves to Feb 1 snapshot (available Feb 1 14:00)
          Model B → KeyError (not available until Feb 2 14:00)
        Query at Feb 2 16:00:
          Model A → Feb 1 snapshot
          Model B → Feb 1 snapshot (now available)
        """
        preds = {"2026-02-01_oscar_noms": {"X": 0.6, "Y": 0.4}}

        model_a = SnapshotModel(
            predictions_by_key=preds,
            snapshot_availability={
                "2026-02-01_oscar_noms": datetime(2026, 2, 1, 14, 0, tzinfo=UTC),
            },
            name="instant",
        )
        model_b = SnapshotModel(
            predictions_by_key=preds,
            snapshot_availability={
                "2026-02-01_oscar_noms": datetime(2026, 2, 2, 14, 0, tzinfo=UTC),
            },
            name="delayed",
        )

        ts_day1 = datetime(2026, 2, 1, 16, 0, tzinfo=UTC)
        ts_day2 = datetime(2026, 2, 2, 16, 0, tzinfo=UTC)

        # Model A has it on day 1, Model B doesn't
        assert model_a.get_predictions(ts_day1)["X"] == pytest.approx(0.6)
        with pytest.raises(KeyError):
            model_b.get_predictions(ts_day1)

        # Both have it on day 2
        assert model_a.get_predictions(ts_day2)["X"] == pytest.approx(0.6)
        assert model_b.get_predictions(ts_day2)["X"] == pytest.approx(0.6)


# ============================================================================
# Single-model ensemble
# ============================================================================


class TestSingleModelEnsemble:
    """Ensemble with a single model should behave identically to the model."""

    def test_single_model_passthrough(self, lr_model: SnapshotModel) -> None:
        ensemble = EnsembleModel(models=[lr_model], name="solo")
        ts = datetime(2026, 2, 5, 16, 0, tzinfo=UTC)
        direct = lr_model.get_predictions(ts)
        via_ensemble = ensemble.get_predictions(ts)
        for k in direct:
            assert via_ensemble[k] == pytest.approx(direct[k])


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_predictions_dict(self) -> None:
        """Model with no predictions — available_snapshots returns keys but
        get_predictions raises for those keys."""
        avail = {"2026-02-01_oscar_noms": datetime(2026, 2, 1, 14, 0, tzinfo=UTC)}
        model = SnapshotModel(
            predictions_by_key={},
            snapshot_availability=avail,
            name="empty",
        )
        assert model.available_snapshots() == ["2026-02-01_oscar_noms"]
        # Resolves to the snapshot key, but no predictions stored for it
        with pytest.raises(KeyError, match="No predictions for snapshot"):
            model.get_predictions(datetime(2026, 2, 2, 16, 0, tzinfo=UTC))

    def test_many_snapshots_resolves_latest(self) -> None:
        """With 5 snapshots, query at the end resolves to the last."""
        from datetime import date as date_cls

        base = date_cls(2026, 1, 1)
        preds = {}
        avail = {}
        for i in range(5):
            d = base + timedelta(days=7 * i)
            key = f"{d.isoformat()}_oscar_noms"
            preds[key] = {"X": 0.1 * (i + 1), "Y": 1.0 - 0.1 * (i + 1)}
            avail[key] = datetime(d.year, d.month, d.day, 14, 0, tzinfo=UTC)

        model = SnapshotModel(predictions_by_key=preds, snapshot_availability=avail, name="multi")
        # Query well after all snapshots
        ts = datetime(2026, 3, 1, 16, 0, tzinfo=UTC)
        result = model.get_predictions(ts)
        # Last snapshot (i=4): X = 0.5
        assert result["X"] == pytest.approx(0.5)


# ============================================================================
# Snapshot Sequence (production calendars)
# ============================================================================


class TestSnapshotSequenceProduction:
    """Test get_snapshot_sequence against production calendar constants.

    These tests validate the bridge between calendar data and the
    trading/backtesting system. Wrong snapshot sequences → wrong model
    predictions at wrong times.
    """

    @pytest.mark.parametrize("year", sorted(CALENDARS.keys()))
    def test_starts_with_oscar_noms(self, year: int) -> None:
        """First snapshot is always Oscar nominations."""
        snaps = get_snapshot_sequence(CALENDARS[year])
        assert snaps[0].dir_name.endswith("_oscar_noms"), (
            f"{year}: first snapshot is {snaps[0].dir_name}, expected *_oscar_noms"
        )

    @pytest.mark.parametrize("year", sorted(CALENDARS.keys()))
    def test_chronological_order(self, year: int) -> None:
        """Snapshots are in non-decreasing UTC datetime order.

        Same-datetime events (e.g., ASC + WGA on same night) are allowed —
        they should be ordered deterministically by org name as tiebreaker.
        """
        snaps = get_snapshot_sequence(CALENDARS[year])
        for i in range(1, len(snaps)):
            assert snaps[i].event_datetime_utc >= snaps[i - 1].event_datetime_utc, (
                f"{year}: snapshot {snaps[i].dir_name} "
                f"({snaps[i].event_datetime_utc}) not >= "
                f"{snaps[i - 1].dir_name} ({snaps[i - 1].event_datetime_utc})"
            )

    @pytest.mark.parametrize("year", sorted(CALENDARS.keys()))
    def test_same_datetime_tiebreak_by_org(self, year: int) -> None:
        """When two snapshots share the same UTC datetime, they are sorted by org name.

        This ensures deterministic ordering for same-night events
        (e.g., ASC and WGA often share a ceremony night).
        """
        snaps = get_snapshot_sequence(CALENDARS[year])
        for i in range(1, len(snaps)):
            if snaps[i].event_datetime_utc == snaps[i - 1].event_datetime_utc:
                # Both are precursor winners (not oscar_noms), so dir_name is YYYY-MM-DD_<org>
                prev_suffix = snaps[i - 1].dir_name.split("_", maxsplit=1)[1]
                curr_suffix = snaps[i].dir_name.split("_", maxsplit=1)[1]
                assert prev_suffix <= curr_suffix, (
                    f"{year}: same-datetime snapshots not sorted by org: "
                    f"{snaps[i - 1].dir_name} vs {snaps[i].dir_name}"
                )

    @pytest.mark.parametrize("year", sorted(CALENDARS.keys()))
    def test_dir_name_uses_local_date(self, year: int) -> None:
        """dir_name date component matches the local date, not the UTC date.

        For evening LA galas: UTC date is often +1 from local date.
        For BAFTA: UTC date = local date (GMT in winter).
        """
        snaps = get_snapshot_sequence(CALENDARS[year])
        cal = CALENDARS[year]
        for snap in snaps:
            if snap.dir_name.endswith("_oscar_noms"):
                expected_date = cal.oscar_nominations_date_local
                expected_name = f"{expected_date.isoformat()}_oscar_noms"
            else:
                # Extract org from dir_name: "YYYY-MM-DD_<org_value>"
                org_value = snap.dir_name.split("_", maxsplit=1)[1]
                org = AwardOrg(org_value)
                expected_date = cal.get_local_date(org, EventPhase.WINNER)
                expected_name = f"{expected_date.isoformat()}_{org_value}"
            assert snap.dir_name == expected_name, f"{year}: {snap.dir_name} != {expected_name}"

    @pytest.mark.parametrize("year", sorted(CALENDARS.keys()))
    def test_excludes_pre_nomination_winners(self, year: int) -> None:
        """Precursor winners announced before Oscar nominations are excluded.

        Their results are already baked into the nominations snapshot.
        """
        from oscar_prediction_market.data.awards_calendar import PRECURSOR_ORGS

        cal = CALENDARS[year]
        nom_date_local = cal.oscar_nominations_date_local
        snaps = get_snapshot_sequence(cal)
        dir_names = {s.dir_name for s in snaps}

        for org in PRECURSOR_ORGS:
            winner_date_local = cal.get_local_date(org, EventPhase.WINNER)
            expected_name = f"{winner_date_local.isoformat()}_{org.value}"
            if winner_date_local <= nom_date_local:
                assert expected_name not in dir_names, (
                    f"{year}: {org} winner ({winner_date_local}) is on or before "
                    f"noms ({nom_date_local}) but appears in snapshots"
                )

    @pytest.mark.parametrize("year", sorted(CALENDARS.keys()))
    def test_excludes_post_ceremony_winners(self, year: int) -> None:
        """Precursor winners after Oscar ceremony are excluded.

        Example: 2024 WGA winners (Apr 14) are after Oscars (Mar 10).
        """
        from oscar_prediction_market.data.awards_calendar import PRECURSOR_ORGS

        cal = CALENDARS[year]
        ceremony_date_local = cal.oscar_ceremony_date_local
        snaps = get_snapshot_sequence(cal)
        dir_names = {s.dir_name for s in snaps}

        for org in PRECURSOR_ORGS:
            winner_date_local = cal.get_local_date(org, EventPhase.WINNER)
            expected_name = f"{winner_date_local.isoformat()}_{org.value}"
            if winner_date_local >= ceremony_date_local:
                assert expected_name not in dir_names, (
                    f"{year}: {org} winner ({winner_date_local}) is on or after "
                    f"ceremony ({ceremony_date_local}) but appears in snapshots"
                )

    def test_2024_excludes_wga(self) -> None:
        """2024 WGA winners (Apr 14) were after Oscars (Mar 10) due to strike."""
        snaps = get_snapshot_sequence(CALENDARS[2024])
        dir_names = [s.dir_name for s in snaps]
        assert not any("wga" in d for d in dir_names)


# ============================================================================
# Snapshot Sequence (partial/edge-case calendars)
# ============================================================================


class TestSnapshotSequenceEdgeCases:
    """Test get_snapshot_sequence with non-production calendars."""

    def test_oscar_only_calendar(self) -> None:
        """Calendar with only Oscar events → just the nominations snapshot."""
        cal = AwardsCalendar(
            ceremony_year=2025,
            events={
                (AwardOrg.OSCAR, EventPhase.NOMINATION): datetime(2025, 1, 23, 13, 30, tzinfo=UTC),
                (AwardOrg.OSCAR, EventPhase.WINNER): datetime(2025, 3, 3, 6, 0, tzinfo=UTC),
            },
        )
        snaps = get_snapshot_sequence(cal)
        assert len(snaps) == 1
        assert snaps[0].dir_name == "2025-01-23_oscar_noms"

    def test_single_precursor_calendar(self) -> None:
        """Calendar with Oscar + 1 precursor → noms + 1 winner snapshot."""
        cal = AwardsCalendar(
            ceremony_year=2025,
            events={
                (AwardOrg.OSCAR, EventPhase.NOMINATION): datetime(2025, 1, 23, 13, 30, tzinfo=UTC),
                (AwardOrg.OSCAR, EventPhase.WINNER): datetime(2025, 3, 3, 6, 0, tzinfo=UTC),
                (AwardOrg.DGA, EventPhase.NOMINATION): datetime(2025, 1, 8, 16, 0, tzinfo=UTC),
                (AwardOrg.DGA, EventPhase.WINNER): datetime(2025, 2, 9, 6, 30, tzinfo=UTC),
            },
        )
        snaps = get_snapshot_sequence(cal)
        assert len(snaps) == 2
        assert snaps[0].dir_name == "2025-01-23_oscar_noms"
        assert snaps[1].dir_name == "2025-02-08_dga"

    def test_all_precursors_before_noms(self) -> None:
        """If all precursor winners are before nominations → only noms snapshot."""
        cal = AwardsCalendar(
            ceremony_year=2025,
            events={
                (AwardOrg.OSCAR, EventPhase.NOMINATION): datetime(2025, 3, 1, 13, 30, tzinfo=UTC),
                (AwardOrg.OSCAR, EventPhase.WINNER): datetime(2025, 4, 1, 6, 0, tzinfo=UTC),
                (AwardOrg.DGA, EventPhase.NOMINATION): datetime(2025, 1, 1, 16, 0, tzinfo=UTC),
                (AwardOrg.DGA, EventPhase.WINNER): datetime(2025, 1, 16, 6, 30, tzinfo=UTC),
            },
        )
        snaps = get_snapshot_sequence(cal)
        assert len(snaps) == 1
        assert snaps[0].dir_name.endswith("_oscar_noms")


# ============================================================================
# SnapshotInfo model behavior
# ============================================================================


class TestSnapshotInfoBehavior:
    """Test SnapshotInfo frozen model: equality, hashing, immutability.

    SnapshotInfo is used as dict keys and in sets throughout backtesting.
    These tests validate the custom __eq__ and __hash__ behavior.
    """

    def test_equality_by_dir_name(self) -> None:
        """Two SnapshotInfos with same dir_name are equal, even if datetimes differ."""
        s1 = SnapshotInfo(
            dir_name="2025-02-08_dga",
            event_datetime_utc=datetime(2025, 2, 9, 6, 30, tzinfo=UTC),
        )
        s2 = SnapshotInfo(
            dir_name="2025-02-08_dga",
            event_datetime_utc=datetime(2025, 3, 1, 0, 0, tzinfo=UTC),
        )
        assert s1 == s2

    def test_inequality_by_dir_name(self) -> None:
        """Different dir_names → not equal, even if datetimes match."""
        s1 = SnapshotInfo(
            dir_name="2025-02-08_dga",
            event_datetime_utc=datetime(2025, 2, 9, 6, 30, tzinfo=UTC),
        )
        s2 = SnapshotInfo(
            dir_name="2025-02-22_bafta",
            event_datetime_utc=datetime(2025, 2, 9, 6, 30, tzinfo=UTC),
        )
        assert s1 != s2

    def test_hashable_and_usable_as_dict_key(self) -> None:
        """SnapshotInfo can be used as dict key and in sets."""
        s = SnapshotInfo(
            dir_name="2025-02-08_dga",
            event_datetime_utc=datetime(2025, 2, 9, 6, 30, tzinfo=UTC),
        )
        d = {s: "value"}
        assert d[s] == "value"
        assert s in {s}

    def test_hash_consistent_with_equality(self) -> None:
        """Equal SnapshotInfos have the same hash."""
        s1 = SnapshotInfo(
            dir_name="2025-02-08_dga",
            event_datetime_utc=datetime(2025, 2, 9, 6, 30, tzinfo=UTC),
        )
        s2 = SnapshotInfo(
            dir_name="2025-02-08_dga",
            event_datetime_utc=datetime(2025, 3, 1, 0, 0, tzinfo=UTC),
        )
        assert hash(s1) == hash(s2)

    def test_frozen_immutable(self) -> None:
        """SnapshotInfo is frozen — attribute assignment raises ValidationError."""
        s = SnapshotInfo(
            dir_name="2025-02-08_dga",
            event_datetime_utc=datetime(2025, 2, 9, 6, 30, tzinfo=UTC),
        )
        with pytest.raises(ValidationError):
            s.dir_name = "new_name"  # type: ignore[misc]

    def test_str_is_dir_name(self) -> None:
        """str(SnapshotInfo) returns the dir_name for readable logging."""
        s = SnapshotInfo(
            dir_name="2025-02-08_dga",
            event_datetime_utc=datetime(2025, 2, 9, 6, 30, tzinfo=UTC),
        )
        assert str(s) == "2025-02-08_dga"
