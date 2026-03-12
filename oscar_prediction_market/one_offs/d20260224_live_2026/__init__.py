"""2026 live Oscar predictions — pre-ceremony analysis and paper trading.

Constants and shared utilities for the 2026 live prediction experiment.
Uses reusable trading/ modules for temporal model, snapshot sequences,
and data loading.
"""

from datetime import date

from oscar_prediction_market.data.awards_calendar import (
    CALENDARS,
)
from oscar_prediction_market.data.schema import OscarCategory
from oscar_prediction_market.modeling.models import ModelType
from oscar_prediction_market.trading.temporal_model import (
    SnapshotInfo,
    get_post_nomination_snapshot_dates,
    get_snapshot_sequence,
)

# ============================================================================
# Experiment-specific model/category lists
# ============================================================================

#: The 4 model types used in backtesting.
BACKTEST_MODEL_TYPES: list[ModelType] = [
    ModelType.LOGISTIC_REGRESSION,
    ModelType.CONDITIONAL_LOGIT,
    ModelType.GRADIENT_BOOSTING,
    ModelType.CALIBRATED_SOFTMAX_GBT,
]

#: All 9 modeled categories.
MODELED_CATEGORIES: list[OscarCategory] = [
    OscarCategory.BEST_PICTURE,
    OscarCategory.DIRECTING,
    OscarCategory.ACTOR_LEADING,
    OscarCategory.ACTRESS_LEADING,
    OscarCategory.ACTOR_SUPPORTING,
    OscarCategory.ACTRESS_SUPPORTING,
    OscarCategory.ORIGINAL_SCREENPLAY,
    OscarCategory.ANIMATED_FEATURE,
    OscarCategory.CINEMATOGRAPHY,
]

# ============================================================================
# 2026-specific constants
# ============================================================================

CEREMONY_YEAR = 2026
CALENDAR = CALENDARS[CEREMONY_YEAR]

#: Post-nomination snapshot dates for 2026 (derived from CALENDAR_2026).
#: As of Feb 24, 2026: 4 snapshots available (nominations + DGA + Annie + BAFTA).
#: Future: PGA (Feb 28), SAG (Mar 1), ASC+WGA (Mar 8).
SNAPSHOT_INFO_2026 = get_post_nomination_snapshot_dates(CALENDAR)
SNAPSHOT_DATES_2026 = [d for d, _ in SNAPSHOT_INFO_2026]
SNAPSHOT_DATE_STRS_2026 = [str(d) for d in SNAPSHOT_DATES_2026]

#: Per-event snapshot keys for 2026.
SNAPSHOT_KEYS_2026: list[SnapshotInfo] = get_snapshot_sequence(CALENDAR)
SNAPSHOT_KEY_STRS_2026 = [k.dir_name for k in SNAPSHOT_KEYS_2026]


def available_snapshots(as_of: date | None = None) -> list[SnapshotInfo]:
    """Snapshot keys available as of a given date (default: today)."""
    cutoff = as_of or date.today()
    all_snapshots = get_snapshot_sequence(CALENDAR)
    return [s for s in all_snapshots if s.event_datetime_utc.date() <= cutoff]


#: Snapshots available as of Mar 9, 2026.
#: All precursors resolved (last: ASC+WGA Mar 8).  Final pre-ceremony snapshot.
_TODAY = date(2026, 3, 9)
AVAILABLE_SNAPSHOT_KEYS: list[SnapshotInfo] = available_snapshots(_TODAY)
AVAILABLE_SNAPSHOT_KEY_STRS = [k.dir_name for k in AVAILABLE_SNAPSHOT_KEYS]

# ============================================================================
# Storage paths (relative to repo root)
# ============================================================================

EXP_DIR = "storage/d20260224_live_2026"
SOURCE_DATASETS_DIR = "storage/d20260218_build_all_datasets"

# Training years: 2000-2025 (includes 2025 results as training data)
TRAIN_YEARS = "2000-2025"
TEST_YEARS = "2026"
