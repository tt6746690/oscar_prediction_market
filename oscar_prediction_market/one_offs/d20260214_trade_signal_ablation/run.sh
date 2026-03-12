#!/usr/bin/env bash
# Trade signal backtest with parameter ablation.
#
# End-to-end pipeline:
#   1. Build temporal model snapshots with best configs
#   2. Generate ablation config grid (1728 main + α-blend + normalize configs)
#   3. Run all configs (parallelized)
#   4. Analyze results
#   5. Deep-dive analysis (best/worst configs, GBT vs LR, edge over time)
#
# Requires:
#   - Intermediate dataset files in storage/d20260201_build_dataset/
#   - Kalshi API access (for price data, cached after first fetch)
#
# Usage:
#   cd "$(git rev-parse --show-toplevel)"
#   bash oscar_prediction_market/one_offs/\
#       d20260214_trade_signal_ablation/run.sh \
#       2>&1 | tee storage/d20260214_trade_signal_ablation/run.log

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# ============================================================================
# Configuration
# ============================================================================

EXP_DIR="storage/d20260214_trade_signal_ablation"
SCRIPT_DIR="oscar_prediction_market/one_offs/d20260214_trade_signal_ablation"
MODULE_PREFIX="oscar_prediction_market.one_offs.d20260214_trade_signal_ablation"

N_WORKERS="${N_WORKERS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)}"

# ============================================================================
# Setup
# ============================================================================

mkdir -p "${EXP_DIR}"

echo "============================================================"
echo "Trade Signal Backtest — Parameter Ablation"
echo "============================================================"
echo "  Experiment dir: ${EXP_DIR}"
echo "  Workers: ${N_WORKERS}"
echo "  Started: $(date)"
echo ""

# ============================================================================
# Step 1: Build temporal model snapshots
# ============================================================================

echo "============================================================"
echo "Step 1: Build temporal model snapshots"
echo "============================================================"

# Check if models already exist
LR_MODELS_EXIST=$(ls "${EXP_DIR}/models/lr/"*/lr_*/5_final_predict/predictions_test.csv 2>/dev/null | wc -l)
GBT_MODELS_EXIST=$(ls "${EXP_DIR}/models/gbt/"*/gbt_*/5_final_predict/predictions_test.csv 2>/dev/null | wc -l)

if [[ "${LR_MODELS_EXIST}" -ge 10 && "${GBT_MODELS_EXIST}" -ge 10 ]]; then
    echo "  SKIP: Models already built (${LR_MODELS_EXIST} LR, ${GBT_MODELS_EXIST} GBT)"
else
    bash "${SCRIPT_DIR}/build_models.sh"
fi
echo ""

# ============================================================================
# Step 2: Generate ablation configs
# ============================================================================

echo "============================================================"
echo "Step 2: Generate ablation configs"
echo "============================================================"

CONFIGS_DIR="${EXP_DIR}/configs"

EXPECTED_CONFIGS=1742
if [[ -d "${CONFIGS_DIR}" ]] && [[ $(ls "${CONFIGS_DIR}"/*.json 2>/dev/null | wc -l) -ge ${EXPECTED_CONFIGS} ]]; then
    echo "  SKIP: Configs already generated ($(ls "${CONFIGS_DIR}"/*.json | wc -l) files)"
else
    uv run python -m "${MODULE_PREFIX}.generate_configs" \
        --output-dir "${CONFIGS_DIR}"
fi
echo ""

# ============================================================================
# Step 3: Run ablation grid
# ============================================================================

echo "============================================================"
echo "Step 3: Run ablation grid"
echo "============================================================"

RESULTS_DIR="${EXP_DIR}/results"

if [[ -f "${RESULTS_DIR}/ablation_results.json" ]]; then
    echo "  SKIP: Results already exist"
else
    uv run python -m "${MODULE_PREFIX}.run_ablation" \
        --configs-dir "${CONFIGS_DIR}" \
        --output-dir "${RESULTS_DIR}" \
        --snapshots-dir "${EXP_DIR}" \
        --n-workers "${N_WORKERS}"
fi
echo ""

# ============================================================================
# Step 4: Analyze results
# ============================================================================

echo "============================================================"
echo "Step 4: Analyze results"
echo "============================================================"

uv run python -m "${MODULE_PREFIX}.analyze_ablation" \
    --results-dir "${RESULTS_DIR}" \
    --output-dir "${EXP_DIR}"

echo ""

# ============================================================================
# Step 5: Deep-dive analysis
# ============================================================================

echo "============================================================"
echo "Step 5: Deep-dive analysis (best/worst configs, GBT vs LR, edge plots)"
echo "============================================================"

uv run python -m "${MODULE_PREFIX}.analyze_deep_dive" \
    --results-dir "${RESULTS_DIR}" \
    --models-dir "${EXP_DIR}/models" \
    --output-dir "${EXP_DIR}" \
    --old-predictions storage/d20260211_temporal_model_snapshots/model_predictions_timeseries.csv

echo ""

# ============================================================================
# Step 6: Temporal analysis (model vs market, α-blend, etc.)
# ============================================================================

echo "============================================================"
echo "Step 6: Temporal analysis"
echo "============================================================"

bash "${SCRIPT_DIR}/analyze_temporal.sh"

echo ""

# ============================================================================
# Step 7: Sync assets
# ============================================================================

echo "============================================================"
echo "Step 7: Sync assets to GitHub-renderable location"
echo "============================================================"

bash oscar_prediction_market/one_offs/sync_assets.sh

echo ""
echo "============================================================"
echo "EXPERIMENT COMPLETE: $(date)"
echo "Results in: ${EXP_DIR}"
echo "============================================================"
