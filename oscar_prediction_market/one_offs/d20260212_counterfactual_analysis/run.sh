#!/bin/bash
# Counterfactual analysis: DGA winner scenarios
#
# Trains baseline models (pre-DGA, as-of 2026-02-06) and scenario models
# (post-DGA, as-of 2026-02-07) for LR, GBT, and XGB. Then runs counterfactual
# analysis: for each DGA nominee, simulates "what if this film wins DGA?"
# and compares model predictions against Kalshi market prices.
#
# Usage:
#   cd "$(git rev-parse --show-toplevel)"
#   bash oscar_prediction_market/one_offs/d20260212_counterfactual_analysis/run.sh 2>&1 | tee storage/d20260212_counterfactual_analysis/run.log

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

OUTPUT_DIR="storage/d20260212_counterfactual_analysis"
mkdir -p "$OUTPUT_DIR/dga_winner"

CONFIGS="oscar_prediction_market/modeling/configs"
SCRIPT_DIR="oscar_prediction_market/one_offs/d20260212_counterfactual_analysis"

# Common args
TRAIN_YEARS="2000-2025"
TEST_YEARS="2026"
CV_SPLIT="$CONFIGS/cv_splits/leave_one_year_out.json"
N_JOBS=4

echo "============================================================"
echo "Step 1: Build baseline models (as-of-date 2026-02-06, no DGA winner feature)"
echo "============================================================"

for model in lr gbt xgb; do
    case $model in
        lr)
            param_grid="$CONFIGS/param_grids/lr_grid.json"
            feature_config="$CONFIGS/features/lr_standard.json"
            name="lr_baseline"
            ;;
        gbt)
            param_grid="$CONFIGS/param_grids/gbt_grid.json"
            feature_config="$CONFIGS/features/gbt_standard.json"
            name="gbt_baseline"
            ;;
        xgb)
            param_grid="$CONFIGS/param_grids/xgb_grid.json"
            feature_config="$CONFIGS/features/xgb_standard.json"
            name="xgb_baseline"
            ;;
    esac

    echo ""
    echo "--- Building $name ---"
    uv run python -m oscar_prediction_market.modeling.build_model \
        --name "$name" \
        --param-grid "$param_grid" \
        --feature-config "$feature_config" \
        --cv-split "$CV_SPLIT" \
        --train-years "$TRAIN_YEARS" \
        --test-years "$TEST_YEARS" \
        --output-dir "$OUTPUT_DIR/dga_winner" \
        --as-of-date "2026-02-06" \
        --n-jobs "$N_JOBS" \
        --feature-selection
done

echo ""
echo "============================================================"
echo "Step 2: Build scenario models (as-of-date 2026-02-07, DGA winner feature available)"
echo "============================================================"

for model in lr gbt xgb; do
    case $model in
        lr)
            param_grid="$CONFIGS/param_grids/lr_grid.json"
            feature_config="$CONFIGS/features/lr_standard.json"
            name="lr_scenario"
            ;;
        gbt)
            param_grid="$CONFIGS/param_grids/gbt_grid.json"
            feature_config="$CONFIGS/features/gbt_standard.json"
            name="gbt_scenario"
            ;;
        xgb)
            param_grid="$CONFIGS/param_grids/xgb_grid.json"
            feature_config="$CONFIGS/features/xgb_standard.json"
            name="xgb_scenario"
            ;;
    esac

    echo ""
    echo "--- Building $name ---"
    uv run python -m oscar_prediction_market.modeling.build_model \
        --name "$name" \
        --param-grid "$param_grid" \
        --feature-config "$feature_config" \
        --cv-split "$CV_SPLIT" \
        --train-years "$TRAIN_YEARS" \
        --test-years "$TEST_YEARS" \
        --output-dir "$OUTPUT_DIR/dga_winner" \
        --as-of-date "2026-02-07" \
        --n-jobs "$N_JOBS" \
        --feature-selection
done

echo ""
echo "============================================================"
echo "Step 3: Run counterfactual analysis"
echo "============================================================"

uv run python -m oscar_prediction_market.one_offs.d20260212_counterfactual_analysis.counterfactual_analysis \
    --output-dir "$OUTPUT_DIR/dga_winner"

echo ""
echo "============================================================"
echo "Done. Results in $OUTPUT_DIR/dga_winner/"
echo "============================================================"
