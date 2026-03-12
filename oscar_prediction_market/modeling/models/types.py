"""Model type enumeration and feature family mappings.

ModelType is the central enum identifying each supported model type.
Each entry has:
- short_name: for filenames, directory names, and display (e.g. "lr", "gbt", "clogit")
- feature_family: which feature engineering family this model uses (LR or GBT)
"""

from enum import StrEnum
from typing import Self

from oscar_prediction_market.modeling.feature_engineering.types import (
    FeatureFamily,
)

# Populated after ModelType definition below
_SHORT_NAMES: dict["ModelType", str] = {}
_FEATURE_FAMILIES: dict["ModelType", FeatureFamily] = {}


class ModelType(StrEnum):
    """Supported model types.

    Use these values consistently throughout the codebase and in JSON configs.
    """

    LOGISTIC_REGRESSION = "logistic_regression"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    BAGGED_CLASSIFIER = "bagged_classifier"
    CONDITIONAL_LOGIT = "conditional_logit"
    SOFTMAX_GBT = "softmax_gbt"
    CALIBRATED_SOFTMAX_GBT = "calibrated_softmax_gbt"

    @property
    def short_name(self) -> str:
        """Short name for filenames, directory names, and display."""
        return _SHORT_NAMES[self]

    @classmethod
    def from_short_name(cls, short_name: str) -> Self:
        """Look up ModelType from its short name (e.g., 'lr' → LOGISTIC_REGRESSION)."""
        for member in cls:
            if member.short_name == short_name:
                return member
        raise ValueError(f"No ModelType with short_name={short_name!r}")

    @property
    def feature_family(self) -> FeatureFamily:
        """Feature engineering family this model type uses.

        Raises ValueError for BAGGED_CLASSIFIER — use
        BaggedClassifierConfig.base_model_type to determine the family.
        """
        if self not in _FEATURE_FAMILIES:
            raise ValueError(
                f"{self.value} has no intrinsic feature family. "
                "Use the base model config's model_type.feature_family instead."
            )
        return _FEATURE_FAMILIES[self]


# Populate after class is defined
_SHORT_NAMES.update(
    {
        ModelType.LOGISTIC_REGRESSION: "lr",
        ModelType.GRADIENT_BOOSTING: "gbt",
        ModelType.XGBOOST: "xgb",
        ModelType.BAGGED_CLASSIFIER: "bag",
        ModelType.CONDITIONAL_LOGIT: "clogit",
        ModelType.SOFTMAX_GBT: "sgbt",
        ModelType.CALIBRATED_SOFTMAX_GBT: "cal_sgbt",
    }
)

_FEATURE_FAMILIES.update(
    {
        ModelType.LOGISTIC_REGRESSION: FeatureFamily.LR,
        ModelType.GRADIENT_BOOSTING: FeatureFamily.GBT,
        ModelType.XGBOOST: FeatureFamily.GBT,
        ModelType.CONDITIONAL_LOGIT: FeatureFamily.LR,
        ModelType.SOFTMAX_GBT: FeatureFamily.GBT,
        ModelType.CALIBRATED_SOFTMAX_GBT: FeatureFamily.GBT,
        # BAGGED_CLASSIFIER intentionally omitted — delegates to base model
    }
)
