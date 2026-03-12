#!/bin/bash
# Generate all configs for the multi-category feature ablation experiment
#
# For each of 9 Oscar categories, generates LR and GBT ablation configs
# (leave-one-out, additive, single-group) and copies shared configs
# (param grids, CV splits) to the experiment storage directory.
#
# Config generation uses the refactored generate_feature_ablation_configs module
# which builds category-aware feature groups from CATEGORY_PRECURSORS.
#
# This script is idempotent — skips categories that already have configs.
# To regenerate, delete the target configs/ directory first.
#
# Output structure:
#   storage/d20260220_feature_ablation/configs/
#     param_grids/   — lr_grid.json, gbt_grid.json, xgb_grid.json,
#                      conditional_logit_grid.json, softmax_gbt_grid.json,
#                      calibrated_softmax_gbt_grid.json (generated from Python)
#     cv_splits/     — leave_one_year_out.json
#     features/
#       best_picture/   — lr_*.json, gbt_*.json (25 each = 50)
#       directing/      — lr_*.json, gbt_*.json (28 each = 56)
#       ...
#
# Usage:
#   cd "$(git rev-parse --show-toplevel)"
#   bash oscar_prediction_market/one_offs/d20260220_feature_ablation/run_generate_configs.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# ============================================================================
# Paths
# ============================================================================

SRC_CONFIGS="oscar_prediction_market/modeling/configs"
OUTPUT_DIR="storage/d20260220_feature_ablation"
CONFIGS="$OUTPUT_DIR/configs"
ABLATION_MODULE="oscar_prediction_market.modeling.generate_feature_ablation_configs"

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

# ============================================================================
# Step 1: Generate shared configs (param grids + CV splits)
# ============================================================================

echo "============================================================"
echo "Step 1: Generate shared configs"
echo "============================================================"

mkdir -p "$CONFIGS/param_grids" "$CONFIGS/cv_splits"

# Param grids — generated from Python (single source of truth)
TUNING_MODULE="oscar_prediction_market.modeling.generate_tuning_configs"
echo "  Generating param grids from Python..."
uv run python -m "$TUNING_MODULE" --output-dir "$CONFIGS/param_grids"

# CV split
cp -n "$SRC_CONFIGS/cv_splits/leave_one_year_out.json" "$CONFIGS/cv_splits/" 2>/dev/null || true
echo "  Copied CV split"
echo ""

# ============================================================================
# Step 2: Generate ablation feature configs per category
# ============================================================================

echo "============================================================"
echo "Step 2: Generate ablation feature configs"
echo "============================================================"

total_configs=0

for category in "${CATEGORIES[@]}"; do
    FEATURES_DIR="$CONFIGS/features/$category"
    mkdir -p "$FEATURES_DIR"

    # Check if configs already exist (skip if so)
    existing=$(find "$FEATURES_DIR" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$existing" -ge 25 ]]; then
        echo "  SKIP $category: $existing configs already exist"
        total_configs=$((total_configs + existing))
        continue
    fi

    echo ""
    echo "--- $category ---"

    # LR configs
    echo "  Generating LR configs..."
    uv run python -m "$ABLATION_MODULE" \
        --category "$category" \
        --model-type logistic_regression \
        --ablation-types all \
        --output-dir "$FEATURES_DIR"

    # GBT configs
    echo "  Generating GBT configs..."
    uv run python -m "$ABLATION_MODULE" \
        --category "$category" \
        --model-type gradient_boosting \
        --ablation-types all \
        --output-dir "$FEATURES_DIR"

    count=$(find "$FEATURES_DIR" -name '*.json' | wc -l | tr -d ' ')
    echo "  -> $category: $count configs"
    total_configs=$((total_configs + count))
done

echo ""

# ============================================================================
# Summary
# ============================================================================

echo "============================================================"
echo "Config generation complete"
echo "============================================================"
echo "  Output: $CONFIGS/"
echo ""
echo "  Param grids:  $(ls -1 "$CONFIGS/param_grids/" | wc -l | tr -d ' ') files"
echo "  CV splits:    $(ls -1 "$CONFIGS/cv_splits/" | wc -l | tr -d ' ') files"
echo "  Feature configs: $total_configs total"
for category in "${CATEGORIES[@]}"; do
    count=$(find "$CONFIGS/features/$category" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
    echo "    $category: $count"
done
