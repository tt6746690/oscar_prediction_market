#!/usr/bin/env bash
# Run temporal analysis on the d20260214 models (lr_standard + gbt_standard).
#
# This reuses the d20260211 temporal analysis infrastructure (collect_results +
# analysis) but runs against the models built with improved feature configs
# (additive_3 feature subset) in storage/d20260214_trade_signal_ablation/models/.
#
# Usage:
#   cd "$(git rev-parse --show-toplevel)"
#   bash oscar_prediction_market/one_offs/\
#       d20260214_trade_signal_ablation/analyze_temporal.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

EXP_DIR="storage/d20260214_trade_signal_ablation"
MODELS_DIR="${EXP_DIR}/models"
PREDICTIONS_CSV="${EXP_DIR}/model_predictions_timeseries.csv"

D211_MODULE="oscar_prediction_market.one_offs.d20260211_temporal_model_snapshots"

echo "============================================================"
echo "Temporal Analysis (Improved Feature Selection Models)"
echo "============================================================"
echo "  Models dir: ${MODELS_DIR}"
echo "  Output dir: ${EXP_DIR}"
echo ""

# Step 1: Collect results into timeseries CSV
echo "=== Step 1: Collect results ==="
uv run python -m "${D211_MODULE}.collect_results" \
    --models-dir "${MODELS_DIR}" \
    --output "${PREDICTIONS_CSV}"
echo ""

# Step 2: Run full temporal analysis
echo "=== Step 2: Run temporal analysis ==="
uv run python -m "${D211_MODULE}.analysis" \
    --predictions "${PREDICTIONS_CSV}" \
    --models-dir "${MODELS_DIR}" \
    --output-dir "${EXP_DIR}"
echo ""

echo "============================================================"
echo "Temporal analysis complete"
echo "  Predictions CSV: ${PREDICTIONS_CSV}"
echo "  Plots: ${EXP_DIR}/*.png"
echo "============================================================"
