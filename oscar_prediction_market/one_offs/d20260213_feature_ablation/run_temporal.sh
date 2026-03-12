#!/bin/bash
# Temporal stability experiment
# Tests 10 configs × 11 snapshot dates for prediction stability & feature jitter
#
# Configs:
#   lr_wide_thresh80 — wide grid + importance-threshold 0.80
#   lr_wide_max10    — wide grid + max-features 10
#   lr_sparse        — sparse grid (no extra filters)
#   lr_moderate      — moderate grid (no extra filters)
#   gbt_baseline     — GBT without new interaction features (42 features)
#   gbt_interactions — GBT with new interaction features (45 features)
#   lr_additive_3_wide_thresh80  — additive_3 (25 feat), wide + thresh 0.80
#   lr_additive_3_moderate       — additive_3 (25 feat), moderate grid
#   gbt_additive_3  — additive_3 (23 feat)
#   gbt_additive_4  — additive_4 (27 feat, adds critic_scores)
#
# Snapshot dates (11, from d20260211_temporal_model_snapshots):
#   2025-11-30, 2025-12-05, 2025-12-08
#   2026-01-04, 2026-01-07, 2026-01-08, 2026-01-09, 2026-01-11
#   2026-01-22, 2026-01-27, 2026-02-07
#
# Total: 10 × 11 = 110 runs
#
# Prerequisite: run_single_date.sh (copies configs to experiment dir)
#
# Usage:
#   cd "$(git rev-parse --show-toplevel)"
#   bash oscar_prediction_market/one_offs/d20260213_feature_ablation/run_temporal.sh \
#       2>&1 | tee storage/d20260213_feature_ablation/temporal/run.log

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR="storage/d20260213_feature_ablation"
TEMPORAL_DIR="$OUTPUT_DIR/temporal"
mkdir -p "$TEMPORAL_DIR"

DATASET_BASE="storage/d20260211_temporal_model_snapshots/datasets"
MODULE="oscar_prediction_market.modeling.build_model"
SCRIPTS_DIR="oscar_prediction_market/one_offs/d20260213_feature_ablation"

TRAIN_YEARS="2000-2025"
TEST_YEARS="2026"
N_JOBS=4

# Generate configs (idempotent — skips if already present)
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

SNAPSHOT_DATES=(
    2025-11-30 2025-12-05 2025-12-08
    2026-01-04 2026-01-07 2026-01-08 2026-01-09 2026-01-11
    2026-01-22 2026-01-27 2026-02-07
)

# ============================================================================
# Validation
# ============================================================================

for f in "$CV_SPLIT" "$LR_WIDE" "$LR_SPARSE" "$LR_MODERATE" "$GBT_GRID" \
         "$LR_FEATURES" "$GBT_FEATURES" "$GBT_NO_INT"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing config: $f"
        echo "  (Run run_single_date.sh first to copy configs)"
        exit 1
    fi
done

for date in "${SNAPSHOT_DATES[@]}"; do
    raw="$DATASET_BASE/$date/oscar_best_picture_raw.json"
    if [[ ! -f "$raw" ]]; then
        echo "ERROR: Missing dataset: $raw"
        exit 1
    fi
done
echo "All configs and ${#SNAPSHOT_DATES[@]} snapshot datasets validated."
echo ""

# ============================================================================
# Helper function
# ============================================================================

run_temporal_config() {
    local config_name="$1"
    local param_grid="$2"
    local feature_config="$3"
    local extra_args="${4:-}"

    echo ""
    echo "============================================================"
    echo "Config: $config_name (${#SNAPSHOT_DATES[@]} snapshots)"
    echo "============================================================"

    for date in "${SNAPSHOT_DATES[@]}"; do
        local run_dir="$TEMPORAL_DIR/$config_name/$date"
        if [[ -d "$run_dir" ]]; then
            echo "  SKIP (exists): $config_name @ $date"
            continue
        fi

        local raw_path="$DATASET_BASE/$date/oscar_best_picture_raw.json"
        echo "  Running: $config_name @ $date"
        # shellcheck disable=SC2086
        uv run python -m "$MODULE" \
            --name "$date" \
            --param-grid "$param_grid" \
            --feature-config "$feature_config" \
            --cv-split "$CV_SPLIT" \
            --train-years "$TRAIN_YEARS" \
            --test-years "$TEST_YEARS" \
            --output-dir "$TEMPORAL_DIR/$config_name" \
            --raw-path "$raw_path" \
            --as-of-date "$date" \
            --n-jobs "$N_JOBS" \
            --feature-selection \
            $extra_args
        echo "  -> $config_name @ $date complete"
    done
}

# ============================================================================
# Run 10 configs × 11 dates = 110 runs
# ============================================================================

run_temporal_config "lr_wide_thresh80" "$LR_WIDE" "$LR_FEATURES" \
    "--importance-threshold 0.80"
run_temporal_config "lr_wide_max10" "$LR_WIDE" "$LR_FEATURES" \
    "--max-features 10"
run_temporal_config "lr_sparse" "$LR_SPARSE" "$LR_FEATURES" ""
run_temporal_config "lr_moderate" "$LR_MODERATE" "$LR_FEATURES" ""
run_temporal_config "gbt_baseline" "$GBT_GRID" "$GBT_NO_INT" ""
run_temporal_config "gbt_interactions" "$GBT_GRID" "$GBT_FEATURES" ""

# --- Additive subset configs ---

LR_ADD3="$CONFIGS/features/lr_additive_3_oscar_nominations.json"
GBT_ADD3="$CONFIGS/features/gbt_additive_3_oscar_nominations.json"
GBT_ADD4="$CONFIGS/features/gbt_additive_4_critic_scores.json"

run_temporal_config "lr_additive_3_wide_thresh80" "$LR_WIDE" "$LR_ADD3" \
    "--importance-threshold 0.80"
run_temporal_config "lr_additive_3_moderate" "$LR_MODERATE" "$LR_ADD3" ""
run_temporal_config "gbt_additive_3" "$GBT_GRID" "$GBT_ADD3" ""
run_temporal_config "gbt_additive_4" "$GBT_GRID" "$GBT_ADD4" ""

echo ""
echo "============================================================"
echo "Temporal experiment complete (10 × ${#SNAPSHOT_DATES[@]} = $((10 * ${#SNAPSHOT_DATES[@]})) runs)."
echo "Results: $TEMPORAL_DIR/"
echo "============================================================"
