"""Feature engineering for Oscar prediction (any category).

Subpackage structure:
- types.py: Pydantic models (FeatureSet, TransformContext, FeatureDefinition),
  availability lambda factories
- helpers.py: Pure utility functions (math, normalization factories, field extractors)
- transforms.py: All transform functions (named functions and factory functions)
- features.py: All FeatureDefinition lists, organized by domain group
- registry.py: FEATURE_REGISTRY dict, resolve_features(), get_feature_names()
- engine.py: transform_dataset(), filter_features_by_availability()
- groups.py: FeatureGroup model, get_feature_groups(), get_all_features()
"""

# Engine
from oscar_prediction_market.modeling.feature_engineering.engine import (
    build_transform_context,
    filter_features_by_availability,
    get_unavailable_features,
    transform_dataset,
)

# Groups
from oscar_prediction_market.modeling.feature_engineering.groups import (
    ABLATION_CATEGORIES,
    FeatureGroup,
    get_all_features,
    get_feature_groups,
)

# Registry
from oscar_prediction_market.modeling.feature_engineering.registry import (
    FEATURE_REGISTRY,
    get_feature_names,
    resolve_features,
)

# Types
from oscar_prediction_market.modeling.feature_engineering.types import (
    FeatureDefinition,
    FeatureFamily,
    FeatureSet,
    TransformContext,
    available_after_earliest_nomination,
    available_after_earliest_winner,
    available_after_nomination,
    available_after_oscar_noms,
    available_after_winner,
)

__all__ = [
    # Types
    "FeatureDefinition",
    "FeatureFamily",
    "FeatureSet",
    "TransformContext",
    "available_after_earliest_nomination",
    "available_after_earliest_winner",
    "available_after_nomination",
    "available_after_oscar_noms",
    "available_after_winner",
    # Registry
    "FEATURE_REGISTRY",
    "get_feature_names",
    "resolve_features",
    # Engine
    "build_transform_context",
    "filter_features_by_availability",
    "get_unavailable_features",
    "transform_dataset",
    # Groups
    "ABLATION_CATEGORIES",
    "FeatureGroup",
    "get_all_features",
    "get_feature_groups",
]
