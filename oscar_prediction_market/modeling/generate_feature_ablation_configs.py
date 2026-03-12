"""Generate feature configuration files for category-aware ablation studies.

Organizes features into semantic groups, with groups dynamically constructed
per Oscar category from CATEGORY_PRECURSORS and category-specific conditions.
Supports systematic ablation analysis across all 9 Oscar trading categories.

Feature Groups (universal, all categories):
    precursor_winners   — Category-specific precursor winner flags + composites + aggregates
    precursor_noms      — Category-specific precursor nomination flags + composites + aggregates
    oscar_nominations   — Cross-category Oscar nomination profile (minus constant features)
    critic_scores       — Review aggregator scores (percentile for LR, z-score for GBT)
    commercial          — Budget and box office (log for LR, raw for GBT)
    timing              — Release timing and runtime
    film_metadata       — Genre indicators and MPAA rating

Feature Groups (conditional):
    person_career       — Oscar history: prior noms/wins, overdue, BP-nominated film
                          (PERSON_CATEGORIES: acting, directing, screenplay, cinematography)
    person_enrichment   — TMDb person data: age, credits, popularity
                          (acting + directing only; insufficient coverage for screenplay/cinematography)
    animated_specific   — Studio identity + sequel detection (Animated Feature only)
    voting_system       — IRV era + nominee count (Best Picture only)

Ablation Strategies:
    1. Leave-one-group-out: Remove one group at a time, measure impact
    2. Additive: Start minimal (most predictive group), add groups one at a time
    3. Single-group: Use only one group at a time

Usage:
    # Generate for one category
    uv run python -m oscar_prediction_market.modeling.generate_feature_ablation_configs \\
        --category best_picture \\
        --model-type logistic_regression \\
        --output-dir storage/d20260220_feature_ablation/configs/features/best_picture

    # Then run experiments via bash script looping over configs
"""

import argparse
import csv
import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from oscar_prediction_market.data.schema import OscarCategory
from oscar_prediction_market.modeling.feature_engineering.groups import (
    get_feature_groups,
)
from oscar_prediction_market.modeling.feature_engineering.types import (
    FeatureFamily,
)
from oscar_prediction_market.modeling.models import ModelType

logger = logging.getLogger(__name__)

MODELING_DIR = Path(__file__).parent


# ============================================================================
# Ablation Configuration
# ============================================================================


class AblationConfig(BaseModel):
    """Configuration for a feature ablation experiment.

    Features can be specified either via group names (included_groups) or
    directly (explicit_features). If explicit_features is set, it takes
    precedence over group-based feature resolution.
    """

    model_config = {"extra": "forbid"}

    name: str = Field(..., description="Ablation experiment name")
    description: str = Field(..., description="What this ablation tests")
    included_groups: list[str] = Field(
        default_factory=list, description="Feature groups to include"
    )
    explicit_features: list[str] | None = Field(
        default=None, description="Direct feature list (overrides group-based resolution)"
    )
    model_type: ModelType = Field(..., description="Model type for feature family selection")
    category: OscarCategory = Field(..., description="Oscar category for group resolution")

    def get_features(self) -> list[str]:
        """Get list of features for this ablation config.

        Resolves group names to feature lists based on category and model type.
        For LR/conditional_logit: uses lr_features from each group.
        For GBT/cal_sgbt/softmax_gbt: uses gbt_features from each group.
        """
        if self.explicit_features is not None:
            return self.explicit_features

        groups = get_feature_groups(self.category)
        group_by_name = {g.name: g for g in groups}

        features = []
        for group_name in self.included_groups:
            group = group_by_name[group_name]
            if self.model_type.feature_family == FeatureFamily.LR:
                features.extend(group.lr_features)
            else:
                features.extend(group.gbt_features)
        return features


# ============================================================================
# Ablation Generators
# ============================================================================


def generate_leave_one_out_ablations(
    model_type: ModelType,
    category: OscarCategory,
) -> list[AblationConfig]:
    """Generate leave-one-group-out ablation configs.

    For each group, create a config that includes all OTHER groups.
    This measures the marginal contribution of each group.
    """
    groups = get_feature_groups(category)
    all_group_names = [g.name for g in groups]
    short_name = model_type.short_name

    ablations = [
        # Full model (baseline)
        AblationConfig(
            name=f"{short_name}_full",
            description="All feature groups (baseline)",
            included_groups=all_group_names,
            model_type=model_type,
            category=category,
        )
    ]

    for excluded_group in groups:
        included = [n for n in all_group_names if n != excluded_group.name]
        ablations.append(
            AblationConfig(
                name=f"{short_name}_without_{excluded_group.name}",
                description=f"Exclude {excluded_group.name}: {excluded_group.description}",
                included_groups=included,
                model_type=model_type,
                category=category,
            )
        )

    return ablations


def generate_additive_ablations(
    model_type: ModelType,
    category: OscarCategory,
) -> list[AblationConfig]:
    """Generate additive ablation configs (cumulative feature addition).

    Start with most important group (precursor_winners), add groups one at a time
    in priority order. Shows cumulative benefit of adding each group.
    """
    groups = get_feature_groups(category)
    short_name = model_type.short_name

    ablations = []
    included_so_far: list[str] = []

    for i, group in enumerate(groups):
        included_so_far.append(group.name)
        ablations.append(
            AblationConfig(
                name=f"{short_name}_additive_{i + 1}_{group.name}",
                description=f"Add {group.name} (cumulative: {', '.join(included_so_far)})",
                included_groups=included_so_far.copy(),
                model_type=model_type,
                category=category,
            )
        )

    return ablations


def generate_single_group_ablations(
    model_type: ModelType,
    category: OscarCategory,
) -> list[AblationConfig]:
    """Generate single-group ablation configs.

    Each config uses only one feature group.
    Shows the standalone predictive power of each group.
    """
    groups = get_feature_groups(category)
    short_name = model_type.short_name

    ablations = []
    for group in groups:
        ablations.append(
            AblationConfig(
                name=f"{short_name}_only_{group.name}",
                description=f"Only {group.name}: {group.description}",
                included_groups=[group.name],
                model_type=model_type,
                category=category,
            )
        )

    return ablations


def generate_all_ablations(
    model_type: ModelType,
    category: OscarCategory,
) -> list[AblationConfig]:
    """Generate all ablation configs for comprehensive analysis."""
    ablations = []
    ablations.extend(generate_leave_one_out_ablations(model_type, category))
    ablations.extend(generate_additive_ablations(model_type, category))
    ablations.extend(generate_single_group_ablations(model_type, category))
    return ablations


# ============================================================================
# Utility Functions
# ============================================================================


def load_nonzero_features_from_csv(importance_csv: Path) -> list[str]:
    """Load features with non-zero importance from a feature importance CSV.

    All models now provide a unified 'importance' column. Falls back to
    'abs_coefficient' for reading old CSVs generated before the unification.

    Args:
        importance_csv: Path to feature_importance.csv

    Returns:
        List of feature names with non-zero importance, ordered by importance descending.
    """
    features = []
    with open(importance_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Unified 'importance' column; fallback to 'abs_coefficient' for old CSVs
            importance = float(row.get("importance", 0) or row.get("abs_coefficient", 0))
            if importance > 0:
                features.append(row["feature"])
    return features


def generate_nonzero_importance_configs(
    model_type: ModelType,
    category: OscarCategory,
    importance_dir: Path,
) -> list[AblationConfig]:
    """Generate config using only features with non-zero importance from a prior experiment.

    Looks for feature importance at:
        {importance_dir}/{short_name}_final_model/feature_importance.csv

    Args:
        model_type: Model type
        category: Oscar category
        importance_dir: Path to experiment directory with final model outputs

    Returns:
        List containing one AblationConfig with non-zero features.
    """
    short_name = model_type.short_name
    csv_path = importance_dir / f"{short_name}_final_model" / "feature_importance.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Feature importance CSV not found: {csv_path}")

    nonzero_features = load_nonzero_features_from_csv(csv_path)
    logger.info(
        f"  {short_name.upper()}: {len(nonzero_features)} non-zero features"
        f" from {csv_path.parent.name}"
    )

    return [
        AblationConfig(
            name=f"{short_name}_nonzero_importance",
            description=(
                f"Only features with non-zero importance from {importance_dir.name}. "
                f"{len(nonzero_features)} features."
            ),
            explicit_features=nonzero_features,
            model_type=model_type,
            category=category,
        )
    ]


# ============================================================================
# Config File Generation
# ============================================================================


def generate_configs(
    model_type: ModelType,
    category: OscarCategory,
    ablation_types: list[str],
    output_dir: Path,
    importance_dir: Path | None = None,
) -> list[Path]:
    """Generate ablation config JSON files for a category.

    Args:
        model_type: Model type (determines LR vs GBT feature family)
        category: Oscar category (determines which groups and features)
        ablation_types: Types to generate ("leave_one_out", "additive", "single_group",
            "all", "nonzero_importance")
        output_dir: Directory to write config files
        importance_dir: Experiment directory for nonzero_importance (required if that type requested)

    Returns:
        List of paths to created config files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []

    # Collect ablation configs based on requested types
    ablations: list[AblationConfig] = []
    if "all" in ablation_types or "leave_one_out" in ablation_types:
        ablations.extend(generate_leave_one_out_ablations(model_type, category))
    if "all" in ablation_types or "additive" in ablation_types:
        ablations.extend(generate_additive_ablations(model_type, category))
    if "all" in ablation_types or "single_group" in ablation_types:
        ablations.extend(generate_single_group_ablations(model_type, category))
    if "nonzero_importance" in ablation_types:
        if importance_dir is None:
            raise ValueError("--importance-dir required for nonzero_importance ablation type")
        ablations.extend(generate_nonzero_importance_configs(model_type, category, importance_dir))

    for ablation in ablations:
        config_dict = {
            "name": ablation.name,
            "description": ablation.description,
            "features": ablation.get_features(),
            "feature_family": ablation.model_type.feature_family.value,
        }

        output_path = output_dir / f"{ablation.name}.json"
        with open(output_path, "w") as f:
            json.dump(config_dict, f, indent=2)
            f.write("\n")

        created_files.append(output_path)
        logger.info(f"Created: {output_path.name} ({len(config_dict['features'])} features)")

    return created_files


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate feature configs for category-aware ablation studies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        choices=[cat.slug for cat in OscarCategory],
        help="Oscar category slug (e.g., best_picture, directing, actor_leading)",
    )
    parser.add_argument(
        "--model-type",
        type=ModelType,
        choices=[ModelType.LOGISTIC_REGRESSION, ModelType.GRADIENT_BOOSTING],
        required=True,
        help="Model type (determines LR vs GBT feature family)",
    )
    parser.add_argument(
        "--ablation-types",
        type=str,
        nargs="+",
        choices=["leave_one_out", "additive", "single_group", "nonzero_importance", "all"],
        default=["all"],
        help="Ablation types to generate (default: all). 'all' includes leave_one_out, additive,"
        " single_group but NOT nonzero_importance (which requires --importance-dir).",
    )
    parser.add_argument(
        "--importance-dir",
        type=str,
        default=None,
        help="Experiment directory with {model_type}_final_model/feature_importance.csv "
        "(required for nonzero_importance ablation type)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: configs/features/ablation/{category_slug})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s")

    category = OscarCategory.from_slug(args.category)

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = MODELING_DIR / "configs" / "features" / "ablation" / args.category

    # Resolve importance directory
    importance_dir = Path(args.importance_dir) if args.importance_dir else None
    if "nonzero_importance" in args.ablation_types and importance_dir is None:
        parser.error("--importance-dir is required when using nonzero_importance ablation type")

    slug = args.category
    groups = get_feature_groups(category)
    logger.info(f"\nCategory: {slug} ({category.value})")
    logger.info(f"  Groups ({len(groups)}): {', '.join(g.name for g in groups)}")
    logger.info(f"  Model type: {args.model_type.value}")
    logger.info(f"  Output: {output_dir}")

    created = generate_configs(
        args.model_type, category, args.ablation_types, output_dir, importance_dir
    )

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Created {len(created)} config files in {output_dir}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
