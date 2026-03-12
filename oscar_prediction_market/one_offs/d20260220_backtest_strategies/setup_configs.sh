#!/usr/bin/env bash
# Setup configs for multi-category backtest strategies experiment.
#
# 1. Generate feature configs for all 9 categories x 2 model families (lr, gbt)
# 2. Generate param grids for all 4 model types (LR, clogit, GBT, cal-SGBT)
# 3. Copy CV split config
#
# All configs are written to storage/d20260220_backtest_strategies/configs/
# for reproducibility. Source configs may change later; the storage copy is
# the ground truth for this experiment.
#
# Usage:
#   cd "$(git rev-parse --show-toplevel)"
#   bash oscar_prediction_market/one_offs/d20260220_backtest_strategies/setup_configs.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

EXP_DIR="storage/d20260220_backtest_strategies"
CONFIGS="$EXP_DIR/configs"
SRC_CONFIGS="oscar_prediction_market/modeling/configs"
MODULE_PREFIX="oscar_prediction_market"

echo "============================================================"
echo "Setup: Multi-Category Backtest Strategies"
echo "Output: $CONFIGS"
echo "============================================================"

# ============================================================================
# Step 1: Generate feature configs per category
# ============================================================================

echo ""
echo "--- Step 1: Generate feature configs ---"

mkdir -p "$CONFIGS/features"

CATEGORIES=(
    best_picture
    directing
    actor_leading
    actress_leading
    actor_supporting
    actress_supporting
    original_screenplay
    cinematography
    animated_feature
)

CATEGORY_ENUMS=(
    BEST_PICTURE
    DIRECTING
    ACTOR_LEADING
    ACTRESS_LEADING
    ACTOR_SUPPORTING
    ACTRESS_SUPPORTING
    ORIGINAL_SCREENPLAY
    CINEMATOGRAPHY
    ANIMATED_FEATURE
)

# Generate full feature configs programmatically using the ablation config generator
# which knows about category-specific feature groups
for i in "${!CATEGORIES[@]}"; do
    cat_slug="${CATEGORIES[$i]}"
    cat_enum="${CATEGORY_ENUMS[$i]}"
    FEATURES_DIR="$CONFIGS/features"

    lr_config="$FEATURES_DIR/${cat_slug}_lr_full.json"
    gbt_config="$FEATURES_DIR/${cat_slug}_gbt_full.json"

    if [[ -f "$lr_config" && -f "$gbt_config" ]]; then
        echo "  SKIP $cat_slug: configs already exist"
        continue
    fi

    echo "  Generating feature configs for $cat_slug..."

    # Generate LR full config
    if [[ ! -f "$lr_config" ]]; then
        uv run python -c "
import json
from pathlib import Path
from oscar_prediction_market.data.schema import OscarCategory
from oscar_prediction_market.modeling.models import ModelType
from oscar_prediction_market.modeling.generate_feature_ablation_configs import get_all_features

category = OscarCategory['${cat_enum}']
features = get_all_features(ModelType.LOGISTIC_REGRESSION, category)
config = {
    'name': '${cat_slug}_lr_full',
    'description': f'All {len(features)} LR features for ${cat_slug}. Feature selection prunes to nonzero-importance.',
    'model_type': 'logistic_regression',
    'features': features,
}
Path('$lr_config').parent.mkdir(parents=True, exist_ok=True)
Path('$lr_config').write_text(json.dumps(config, indent=2) + '\n')
print(f'    Created {\"$lr_config\"} ({len(features)} features)')
"
    fi

    # Generate GBT full config
    if [[ ! -f "$gbt_config" ]]; then
        uv run python -c "
import json
from pathlib import Path
from oscar_prediction_market.data.schema import OscarCategory
from oscar_prediction_market.modeling.models import ModelType
from oscar_prediction_market.modeling.generate_feature_ablation_configs import get_all_features

category = OscarCategory['${cat_enum}']
features = get_all_features(ModelType.GRADIENT_BOOSTING, category)
config = {
    'name': '${cat_slug}_gbt_full',
    'description': f'All {len(features)} GBT features for ${cat_slug}. Feature selection prunes to nonzero-importance.',
    'model_type': 'gradient_boosting',
    'features': features,
}
Path('$gbt_config').parent.mkdir(parents=True, exist_ok=True)
Path('$gbt_config').write_text(json.dumps(config, indent=2) + '\n')
print(f'    Created {\"$gbt_config\"} ({len(features)} features)')
"
    fi
done

echo ""
echo "  Feature config counts per category:"
for cat_slug in "${CATEGORIES[@]}"; do
    lr_n=$(python3 -c "import json; print(len(json.load(open('$CONFIGS/features/${cat_slug}_lr_full.json'))['features']))")
    gbt_n=$(python3 -c "import json; print(len(json.load(open('$CONFIGS/features/${cat_slug}_gbt_full.json'))['features']))")
    echo "    $cat_slug: LR=$lr_n, GBT=$gbt_n"
done

# ============================================================================
# Step 2: Generate param grids
# ============================================================================

echo ""
echo "--- Step 2: Generate param grids ---"

mkdir -p "$CONFIGS/param_grids"

TUNING_MODULE="$MODULE_PREFIX.modeling.generate_tuning_configs"

# Generate grids for 4 model types
for model_type in logistic_regression conditional_logit gradient_boosting calibrated_softmax_gbt; do
    grid_file="$CONFIGS/param_grids/${model_type}_grid.json"
    # generate_tuning_configs uses short names for output files, but we need to check
    echo "  Generating $model_type grid..."
done

# Use the Python generator (single source of truth)
uv run python -m "$TUNING_MODULE" --output-dir "$CONFIGS/param_grids"

# Verify all 4 required grids exist
for short_name in lr clogit gbt csmgbt; do
    grid="$CONFIGS/param_grids/${short_name}_grid.json"
    if [[ ! -f "$grid" ]]; then
        # The generator also creates sgbt and xgb grids - we only need 4
        echo "  WARNING: Missing $grid"
    else
        n=$(python3 -c "import json; print(len(json.load(open('$grid'))['grid']))")
        echo "  $grid: $n configs"
    fi
done

# ============================================================================
# Step 3: Copy CV split
# ============================================================================

echo ""
echo "--- Step 3: Copy CV split ---"

mkdir -p "$CONFIGS/cv_splits"
cp "$SRC_CONFIGS/cv_splits/leave_one_year_out.json" "$CONFIGS/cv_splits/"
echo "  Copied leave_one_year_out.json"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo "  Feature configs: $(find "$CONFIGS/features" -name '*.json' | wc -l | tr -d ' ') files"
echo "  Param grids:     $(find "$CONFIGS/param_grids" -name '*.json' | wc -l | tr -d ' ') files"
echo "  CV splits:       $(find "$CONFIGS/cv_splits" -name '*.json' | wc -l | tr -d ' ') files"
echo ""
ls -la "$CONFIGS/features/"
echo ""
ls -la "$CONFIGS/param_grids/"
