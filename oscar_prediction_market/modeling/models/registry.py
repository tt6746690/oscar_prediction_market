"""Model registry — maps ModelType to config and implementation classes.

Provides the ``create_model`` factory and the consolidated ``MODEL_INFO``
dictionary used by config generators and tuning scripts.
"""

from pydantic import BaseModel

from oscar_prediction_market.modeling.models.bagged import (
    BaggedClassifierModel,
)
from oscar_prediction_market.modeling.models.base import PredictionModel
from oscar_prediction_market.modeling.models.configs import (
    BaggedClassifierConfig,
    CalibratedSoftmaxGBTConfig,
    ConditionalLogitConfig,
    GradientBoostingConfig,
    LogisticRegressionConfig,
    ModelConfig,
    SoftmaxGBTConfig,
    XGBoostConfig,
)
from oscar_prediction_market.modeling.models.gbt import (
    CalibratedSoftmaxGBTModel,
    GradientBoostingModel,
    SoftmaxGBTModel,
    XGBoostModel,
)
from oscar_prediction_market.modeling.models.lr import (
    ConditionalLogitModel,
    LogisticRegressionModel,
)
from oscar_prediction_market.modeling.models.types import ModelType


class ModelInfo:
    """Registry entry for a model type — consolidates config class and model class.

    Used by create_model() and by config generators that need to look up
    the config class for a given ModelType.
    """

    __slots__ = ("config_class", "model_class")

    def __init__(
        self,
        config_class: type[BaseModel],
        model_class: type[PredictionModel],
    ) -> None:
        self.config_class = config_class
        self.model_class = model_class


MODEL_INFO: dict[ModelType, ModelInfo] = {
    ModelType.LOGISTIC_REGRESSION: ModelInfo(LogisticRegressionConfig, LogisticRegressionModel),
    ModelType.GRADIENT_BOOSTING: ModelInfo(GradientBoostingConfig, GradientBoostingModel),
    ModelType.XGBOOST: ModelInfo(XGBoostConfig, XGBoostModel),
    ModelType.BAGGED_CLASSIFIER: ModelInfo(BaggedClassifierConfig, BaggedClassifierModel),
    ModelType.CONDITIONAL_LOGIT: ModelInfo(ConditionalLogitConfig, ConditionalLogitModel),
    ModelType.SOFTMAX_GBT: ModelInfo(SoftmaxGBTConfig, SoftmaxGBTModel),
    ModelType.CALIBRATED_SOFTMAX_GBT: ModelInfo(
        CalibratedSoftmaxGBTConfig, CalibratedSoftmaxGBTModel
    ),
}

# Tunable model types — excludes BAGGED_CLASSIFIER (wraps other models)
TUNABLE_MODEL_TYPES: list[ModelType] = [mt for mt in ModelType if mt != ModelType.BAGGED_CLASSIFIER]


def create_model(config: ModelConfig) -> PredictionModel:
    """Factory function to create a model from configuration."""
    model_type = ModelType(config.model_type)
    info = MODEL_INFO.get(model_type)
    if info is None:
        raise ValueError(f"Unknown model type: {model_type}")
    return info.model_class(config)  # type: ignore[call-arg]
