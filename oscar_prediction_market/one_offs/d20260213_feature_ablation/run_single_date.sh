#!/bin/bash
# Single-date feature ablation experiment
# Tests 16 LR + 2 GBT + 5 additive-subset configs at as-of 2026-02-07
#
# Ablation matrix:
#   Group 1: Grid comparison (no extra filters)
#     lr_wide          — lr_grid.json (C=[0.001..1.0]), all 47 features
#     lr_sparse        — lr_grid_v2.json (C=[0.01..0.05]), all 47 features
#     lr_moderate      — lr_grid_moderate.json (C=[0.02..0.2]), all 47 features
#
#   Group 2: Sparse grid + pipeline stages
#     lr_sparse_thresh95       — + importance-threshold 0.95
#     lr_sparse_thresh95_max15 — + max-features 15
#     lr_sparse_thresh95_vif   — + VIF filter
#     lr_sparse_full_pipeline  — all three
#
#   Group 3: Wide grid + individual filters
#     lr_wide_thresh{95,90,80} — importance thresholds
#     lr_wide_max{10,15,20}    — feature caps
#     lr_wide_vif{5,10}        — VIF thresholds
#     lr_wide_full_pipeline    — thresh=0.90 + max=15 + VIF=5
#
#   Group 4: GBT comparison
#     gbt_baseline     — 42 features (without new interaction features)
#     gbt_interactions — 45 features (with precursor_wins_count, precursor_nominations_count,
#                        has_pga_dga_combo)
#
#   Group 5: Additive subset validation (FULL vs BASE feature definitions)
#     lr_additive_3_full_wide_thresh80  — 25 features (FULL), wide grid + thresh 0.80
#     lr_additive_3_base_wide_thresh80  — 22 features (BASE), wide grid + thresh 0.80
#     gbt_additive_3_full  — 23 features (FULL)
#     gbt_additive_3_base  — 17 features (BASE)
#     gbt_additive_4_full  — 27 features (adds critic_scores to additive_3)
#
# Usage:
#   cd "$(git rev-parse --show-toplevel)"
#   bash oscar_prediction_market/one_offs/d20260213_feature_ablation/run_single_date.sh \
#       2>&1 | tee storage/d20260213_feature_ablation/single_date/run.log

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR="storage/d20260213_feature_ablation"
SINGLE_DATE_DIR="$OUTPUT_DIR/single_date"
mkdir -p "$SINGLE_DATE_DIR"

RAW_PATH="storage/d20260201_build_dataset/oscar_best_picture_raw.json"
MODULE="oscar_prediction_market.modeling.build_model"
SCRIPTS_DIR="oscar_prediction_market/one_offs/d20260213_feature_ablation"

TRAIN_YEARS="2000-2025"
TEST_YEARS="2026"
AS_OF_DATE="2026-02-07"
N_JOBS=4

# ============================================================================
# Generate configs (idempotent — skips if already present)
# ============================================================================

bash "$SCRIPTS_DIR/run_generate_configs.sh"

CONFIGS="$OUTPUT_DIR/configs"
CV_SPLIT="$CONFIGS/cv_splits/leave_one_year_out.json"
LR_WIDE="$CONFIGS/param_grids/lr_grid_wide.json"
LR_SPARSE="$CONFIGS/param_grids/lr_grid_sparse.json"
LR_MODERATE="$CONFIGS/param_grids/lr_grid_moderate.json"
GBT_GRID="$CONFIGS/param_grids/gbt_grid.json"
LR_FEATURES="$CONFIGS/features/lr_full.json"
GBT_FEATURES="$CONFIGS/features/gbt_full.json"
GBT_NO_INT="$CONFIGS/features/gbt_no_interactions.json"

# ============================================================================
# Validation
# ============================================================================

for f in "$RAW_PATH" "$LR_WIDE" "$LR_SPARSE" "$LR_MODERATE" "$GBT_GRID" \
         "$LR_FEATURES" "$GBT_FEATURES" "$GBT_NO_INT" "$CV_SPLIT"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing file: $f"
        exit 1
    fi
done
echo "All required files validated."
echo ""

# ============================================================================
# Helper function
# ============================================================================

run_model() {
    local name="$1"
    local param_grid="$2"
    local feature_config="$3"
    local extra_args="${4:-}"

    local run_dir="$SINGLE_DATE_DIR/$name"
    if [[ -d "$run_dir" ]]; then
        echo "  SKIP (exists): $name"
        return
    fi

    echo ""
    echo "--- $name ---"
    # shellcheck disable=SC2086
    uv run python -m "$MODULE" \
        --name "$name" \
        --param-grid "$param_grid" \
        --feature-config "$feature_config" \
        --cv-split "$CV_SPLIT" \
        --train-years "$TRAIN_YEARS" \
        --test-years "$TEST_YEARS" \
        --output-dir "$SINGLE_DATE_DIR" \
        --raw-path "$RAW_PATH" \
        --as-of-date "$AS_OF_DATE" \
        --n-jobs "$N_JOBS" \
        --feature-selection \
        --save-fold-importance \
        $extra_args
    echo "  -> $name complete"
}

# ============================================================================
# Group 1: Grid comparison (no extra filters)
# ============================================================================

echo "============================================================"
echo "Group 1: Grid comparison (3 configs)"
echo "============================================================"

run_model "lr_wide" "$LR_WIDE" "$LR_FEATURES" ""
run_model "lr_sparse" "$LR_SPARSE" "$LR_FEATURES" ""
run_model "lr_moderate" "$LR_MODERATE" "$LR_FEATURES" ""

# ============================================================================
# Group 2: Sparse grid + pipeline stages (4 configs)
# ============================================================================

echo ""
echo "============================================================"
echo "Group 2: Sparse grid + pipeline stages (4 configs)"
echo "============================================================"

run_model "lr_sparse_thresh95" "$LR_SPARSE" "$LR_FEATURES" \
    "--importance-threshold 0.95"
run_model "lr_sparse_thresh95_max15" "$LR_SPARSE" "$LR_FEATURES" \
    "--importance-threshold 0.95 --max-features 15"
run_model "lr_sparse_thresh95_vif" "$LR_SPARSE" "$LR_FEATURES" \
    "--importance-threshold 0.95 --vif-filter"
run_model "lr_sparse_full_pipeline" "$LR_SPARSE" "$LR_FEATURES" \
    "--importance-threshold 0.95 --max-features 15 --vif-filter"

# ============================================================================
# Group 3: Wide grid + individual filters (9 configs)
# ============================================================================

echo ""
echo "============================================================"
echo "Group 3: Wide grid + individual filters (9 configs)"
echo "============================================================"

run_model "lr_wide_thresh95" "$LR_WIDE" "$LR_FEATURES" \
    "--importance-threshold 0.95"
run_model "lr_wide_thresh90" "$LR_WIDE" "$LR_FEATURES" \
    "--importance-threshold 0.90"
run_model "lr_wide_thresh80" "$LR_WIDE" "$LR_FEATURES" \
    "--importance-threshold 0.80"
run_model "lr_wide_max10" "$LR_WIDE" "$LR_FEATURES" \
    "--max-features 10"
run_model "lr_wide_max15" "$LR_WIDE" "$LR_FEATURES" \
    "--max-features 15"
run_model "lr_wide_max20" "$LR_WIDE" "$LR_FEATURES" \
    "--max-features 20"
run_model "lr_wide_vif5" "$LR_WIDE" "$LR_FEATURES" \
    "--vif-filter --vif-threshold 5.0"
run_model "lr_wide_vif10" "$LR_WIDE" "$LR_FEATURES" \
    "--vif-filter --vif-threshold 10.0"
run_model "lr_wide_full_pipeline" "$LR_WIDE" "$LR_FEATURES" \
    "--importance-threshold 0.90 --max-features 15 --vif-filter"

# ============================================================================
# Group 4: GBT comparison (2 configs)
# ============================================================================

echo ""
echo "============================================================"
echo "Group 4: GBT comparison (2 configs)"
echo "============================================================"

run_model "gbt_baseline" "$GBT_GRID" "$GBT_NO_INT" ""
run_model "gbt_interactions" "$GBT_GRID" "$GBT_FEATURES" ""

# ============================================================================
# Group 5: Additive subset validation (5 configs)
# ============================================================================

echo ""
echo "============================================================"
echo "Group 5: Additive subset validation (5 configs)"
echo "============================================================"

LR_ADD3_FULL="$CONFIGS/features/lr_additive_3_oscar_nominations.json"
LR_ADD3_BASE="$CONFIGS/features/lr_additive_3_base.json"
GBT_ADD3_FULL="$CONFIGS/features/gbt_additive_3_oscar_nominations.json"
GBT_ADD3_BASE="$CONFIGS/features/gbt_additive_3_base.json"
GBT_ADD4_FULL="$CONFIGS/features/gbt_additive_4_critic_scores.json"

run_model "lr_additive_3_full_wide_thresh80" "$LR_WIDE" "$LR_ADD3_FULL" \
    "--importance-threshold 0.80"
run_model "lr_additive_3_base_wide_thresh80" "$LR_WIDE" "$LR_ADD3_BASE" \
    "--importance-threshold 0.80"
run_model "gbt_additive_3_full" "$GBT_GRID" "$GBT_ADD3_FULL" ""
run_model "gbt_additive_3_base" "$GBT_GRID" "$GBT_ADD3_BASE" ""
run_model "gbt_additive_4_full" "$GBT_GRID" "$GBT_ADD4_FULL" ""

echo ""
echo "============================================================"
echo "Single-date ablation complete (23 configs)."
echo "Results: $SINGLE_DATE_DIR/"
echo "============================================================"
