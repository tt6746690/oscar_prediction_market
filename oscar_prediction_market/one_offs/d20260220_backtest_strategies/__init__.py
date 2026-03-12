"""Shared constants for the multi-category backtest strategies experiment.

Experiment-specific constants that define the scope of the backtest
(which categories, which models, ensemble name). Data loading, temporal
logic, and market data functions live in ``trading/``.
"""

from oscar_prediction_market.data.schema import OscarCategory
from oscar_prediction_market.modeling.models import ModelType

# ============================================================================
# Category helpers
# ============================================================================

#: All 9 modeled categories
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
# Model type constants
# ============================================================================

#: The 4 model types used in this experiment
BACKTEST_MODEL_TYPES: list[ModelType] = [
    ModelType.LOGISTIC_REGRESSION,
    ModelType.CONDITIONAL_LOGIT,
    ModelType.GRADIENT_BOOSTING,
    ModelType.CALIBRATED_SOFTMAX_GBT,
]

#: Short name for the equal-weight ensemble of all 4 models.
#: Not a ModelType — handled as a special case in the backtest runner.
ENSEMBLE_SHORT_NAME = "avg_ensemble"
