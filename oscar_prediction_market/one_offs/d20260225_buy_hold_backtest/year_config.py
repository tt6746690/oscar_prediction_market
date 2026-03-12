"""Per-year configuration for buy-and-hold backtests.

Bundles all year-specific parameters (calendar, winners, paths, available
categories) into a single ``YearConfig`` so the backtest runner can operate
on any supported ceremony year without hardcoded constants.

The key insight: feature configs and param grids are year-agnostic (they live
in ``storage/d20260220_backtest_strategies/configs/``), but datasets, models,
market data, and winners differ per year.

Supported ceremony years: 2024, 2025.
"""

from pathlib import Path

from pydantic import BaseModel, Field, computed_field

from oscar_prediction_market.data.awards_calendar import (
    CALENDARS,
    AwardsCalendar,
)
from oscar_prediction_market.data.oscar_winners import (
    WINNERS_BY_YEAR,
)
from oscar_prediction_market.data.schema import (
    OscarCategory,
)
from oscar_prediction_market.trading.temporal_model import (
    SnapshotInfo,
    get_snapshot_sequence,
)

# ============================================================================
# Categories with Kalshi market data by year
# ============================================================================

#: Categories that have Kalshi prediction markets per ceremony year.
#: 2024 has no Cinematography market; 2025 has all 9.
CATEGORIES_BY_YEAR: dict[int, list[OscarCategory]] = {
    2024: [
        OscarCategory.BEST_PICTURE,
        OscarCategory.DIRECTING,
        OscarCategory.ACTOR_LEADING,
        OscarCategory.ACTRESS_LEADING,
        OscarCategory.ACTOR_SUPPORTING,
        OscarCategory.ACTRESS_SUPPORTING,
        OscarCategory.ORIGINAL_SCREENPLAY,
        OscarCategory.ANIMATED_FEATURE,
        # No Cinematography for 2024
    ],
    2025: [
        OscarCategory.BEST_PICTURE,
        OscarCategory.DIRECTING,
        OscarCategory.ACTOR_LEADING,
        OscarCategory.ACTRESS_LEADING,
        OscarCategory.ACTOR_SUPPORTING,
        OscarCategory.ACTRESS_SUPPORTING,
        OscarCategory.ORIGINAL_SCREENPLAY,
        OscarCategory.ANIMATED_FEATURE,
        OscarCategory.CINEMATOGRAPHY,
    ],
    2026: [
        OscarCategory.BEST_PICTURE,
        OscarCategory.DIRECTING,
        OscarCategory.ACTOR_LEADING,
        OscarCategory.ACTRESS_LEADING,
        OscarCategory.ACTOR_SUPPORTING,
        OscarCategory.ACTRESS_SUPPORTING,
        OscarCategory.ORIGINAL_SCREENPLAY,
        OscarCategory.ANIMATED_FEATURE,
        OscarCategory.CINEMATOGRAPHY,
    ],
}


# ============================================================================
# Year config model
# ============================================================================

# Shared experiment root (models, datasets, configs, market data)
BACKTEST_EXP_DIR = Path("storage/d20260220_backtest_strategies")

# Buy-hold experiment root (results, plots, analysis)
BUY_HOLD_EXP_DIR = Path("storage/d20260225_buy_hold_backtest")


class YearConfig(BaseModel):
    """All year-specific parameters for a buy-and-hold backtest.

    Bundles calendar, winners, storage paths, and available categories.
    The backtest runner takes one of these instead of importing hardcoded
    module-level constants from data_prep.
    """

    model_config = {"extra": "forbid"}

    ceremony_year: int = Field(...)

    #: Override storage root for models/datasets.  When None, uses the
    #: shared backtest experiment directory.  Set for 2026 live mode where
    #: models live under storage/d20260224_live_2026/ instead.
    custom_models_root: Path | None = Field(default=None)
    custom_datasets_root: Path | None = Field(default=None)

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def calendar(self) -> AwardsCalendar:
        return CALENDARS[self.ceremony_year]

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def winners(self) -> dict[OscarCategory, str] | None:
        """Known winners, or None if ceremony hasn't happened yet (live mode)."""
        return WINNERS_BY_YEAR.get(self.ceremony_year)

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def categories(self) -> list[OscarCategory]:
        return CATEGORIES_BY_YEAR[self.ceremony_year]

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def models_dir(self) -> Path:
        if self.custom_models_root is not None:
            return self.custom_models_root
        return BACKTEST_EXP_DIR / str(self.ceremony_year) / "models"

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def datasets_dir(self) -> Path:
        if self.custom_datasets_root is not None:
            return self.custom_datasets_root
        return BACKTEST_EXP_DIR / str(self.ceremony_year) / "datasets"

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def results_dir(self) -> Path:
        return BUY_HOLD_EXP_DIR / str(self.ceremony_year) / "results"

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def plots_dir(self) -> Path:
        return BUY_HOLD_EXP_DIR / str(self.ceremony_year) / "plots"

    @property
    def market_data_dir(self) -> Path:
        return BACKTEST_EXP_DIR / "market_data"

    @property
    def configs_dir(self) -> Path:
        """Feature configs and param grids — shared across years."""
        return BACKTEST_EXP_DIR / "configs"

    @property
    def train_years(self) -> str:
        """Training year range: 2000 to year before ceremony."""
        return f"2000-{self.ceremony_year - 1}"

    @property
    def test_years(self) -> str:
        return str(self.ceremony_year)

    def snapshot_keys(self) -> list[SnapshotInfo]:
        """Per-event snapshot keys, pre-ceremony only.

        Returns one SnapshotInfo per post-nomination precursor event ordered
        by datetime.  For 2024, WGA (Apr 14) is after ceremony (Mar 10)
        and is excluded.
        """
        ceremony_date = self.calendar.oscar_ceremony_date_local
        return [
            k
            for k in get_snapshot_sequence(self.calendar)
            if k.event_datetime_utc.date() < ceremony_date
        ]


# Pre-built configs for supported years
YEAR_CONFIGS: dict[int, YearConfig] = {
    2024: YearConfig(ceremony_year=2024),
    2025: YearConfig(ceremony_year=2025),
    2026: YearConfig(
        ceremony_year=2026,
        custom_models_root=Path("storage/d20260224_live_2026/models"),
        custom_datasets_root=Path("storage/d20260224_live_2026/datasets"),
    ),
}
