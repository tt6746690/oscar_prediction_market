"""Model definitions for Oscar prediction.

This package provides a unified interface for different model types,
their configurations, a model registry, and persistence utilities.

Submodules:
- types: ModelType enum with short_name and feature_family properties
- configs: Pydantic config classes and discriminated union ModelConfig
- base: PredictionModel abstract base class
- lr: Linear model implementations (LogisticRegression, ConditionalLogit)
- gbt: Tree-based implementations (GradientBoosting, XGBoost, SoftmaxGBT, CalibratedSoftmaxGBT)
- bagged: Bootstrap aggregation meta-model (BaggedClassifier)
- registry: MODEL_INFO dict, TUNABLE_MODEL_TYPES, create_model factory
- loading: Config file loading and validation
- persistence: save_model / load_model
"""

# Re-export public API so callers can continue importing from modeling.models
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
    ModelConfigGrid,
    ModelConfigList,
    SoftmaxGBTConfig,
    XGBoostConfig,
    model_config_adapter,
    model_config_list_adapter,
)
from oscar_prediction_market.modeling.models.gbt import (
    CalibratedSoftmaxGBTModel,
    GradientBoostingModel,
    SoftmaxGBTModel,
    XGBoostModel,
)
from oscar_prediction_market.modeling.models.loading import (
    load_model_config,
    load_model_config_grid,
    validate_model_feature_consistency,
)
from oscar_prediction_market.modeling.models.lr import (
    ConditionalLogitModel,
    LogisticRegressionModel,
)
from oscar_prediction_market.modeling.models.persistence import (
    load_model,
    save_model,
)
from oscar_prediction_market.modeling.models.registry import (
    MODEL_INFO,
    TUNABLE_MODEL_TYPES,
    ModelInfo,
    create_model,
)
from oscar_prediction_market.modeling.models.types import ModelType

__all__ = [
    # Types
    "ModelType",
    # Configs
    "BaggedClassifierConfig",
    "CalibratedSoftmaxGBTConfig",
    "ConditionalLogitConfig",
    "GradientBoostingConfig",
    "LogisticRegressionConfig",
    "ModelConfig",
    "ModelConfigGrid",
    "ModelConfigList",
    "SoftmaxGBTConfig",
    "XGBoostConfig",
    "model_config_adapter",
    "model_config_list_adapter",
    # Base
    "PredictionModel",
    # Implementations
    "BaggedClassifierModel",
    "CalibratedSoftmaxGBTModel",
    "ConditionalLogitModel",
    "GradientBoostingModel",
    "LogisticRegressionModel",
    "SoftmaxGBTModel",
    "XGBoostModel",
    # Registry
    "MODEL_INFO",
    "TUNABLE_MODEL_TYPES",
    "ModelInfo",
    "create_model",
    # Loading
    "load_model_config",
    "load_model_config_grid",
    "validate_model_feature_consistency",
    # Persistence
    "load_model",
    "save_model",
]
