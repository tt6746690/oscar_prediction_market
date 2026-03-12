#!/usr/bin/env bash
# Master script: run the full multi-category backtest pipeline.
#
# Steps:
#   0. Setup configs (feature configs, param grids, CV splits)
#   1. Build datasets for all snapshots (54 merge runs)
#   2. Train models for all snapshots (216 trainings)
#   3. Fetch market data + match nominees + run backtests
#   4. Analyze + plots
#
# Usage:
#   cd "$(git rev-parse --show-toplevel)"
#   bash oscar_prediction_market/one_offs/d20260220_backtest_strategies/run.sh 2>&1 \
#     | tee storage/d20260220_backtest_strategies/run.log
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

SCRIPT_DIR="oscar_prediction_market/one_offs/d20260220_backtest_strategies"
EXP_DIR="storage/d20260220_backtest_strategies"

echo "============================================================"
echo "Multi-Category Backtest Strategies"
echo "============================================================"
echo "Script dir: ${SCRIPT_DIR}"
echo "Storage dir: ${EXP_DIR}"
echo "Start time: $(date)"
echo ""

# Step 0: Setup configs
echo "=== Step 0: Setup Configs ==="
bash "${SCRIPT_DIR}/setup_configs.sh"
echo ""

# Step 1: Build datasets
echo "=== Step 1: Build Datasets ==="
bash "${SCRIPT_DIR}/build_datasets.sh"
echo ""

# Step 2: Train models
echo "=== Step 2: Train Models (216 trainings — this will take hours) ==="
bash "${SCRIPT_DIR}/train_models.sh"
echo ""

# Step 3: Run backtests
echo "=== Step 3: Run Backtests ==="
bash "${SCRIPT_DIR}/run_backtests.sh"
echo ""

# Step 4: Analyze
echo "=== Step 4: Analyze ==="
uv run python -m oscar_prediction_market.one_offs.d20260220_backtest_strategies.analyze
echo ""

echo "============================================================"
echo "Pipeline complete! End time: $(date)"
echo "Results: ${EXP_DIR}/2025/results/"
echo "Plots: ${EXP_DIR}/2025/plots/"
echo "============================================================"
