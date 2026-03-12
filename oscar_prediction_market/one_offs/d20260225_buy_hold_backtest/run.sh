#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

SCRIPT_DIR="oscar_prediction_market/one_offs/d20260225_buy_hold_backtest"
SHARED_DIR="storage/d20260220_backtest_strategies"
EXP_DIR="storage/d20260225_buy_hold_backtest"

echo "============================================================"
echo "Buy-Hold Backtest Pipeline"
echo "============================================================"
echo "Date:   $(date)"
echo "Commit: $(git rev-parse --short HEAD)"
echo ""
echo "Evaluates buy-and-hold strategies on Oscar prediction markets"
echo "using shared datasets/models from d20260220_backtest_strategies."
echo "Results: ${EXP_DIR}/"
echo "============================================================"
echo ""

# Check prerequisites (shared datasets + models)
for year in 2024 2025; do
    [[ -d "${SHARED_DIR}/${year}/models" ]] || {
        echo "ERROR: Shared datasets/models not found."
        echo "Run the d20260220_backtest_strategies pipeline first:"
        echo "  bash oscar_prediction_market/one_offs/d20260220_backtest_strategies/run.sh"
        exit 1
    }
done
[[ -d "${SHARED_DIR}/configs" ]] || {
    echo "ERROR: Shared datasets/models not found."
    echo "Run the d20260220_backtest_strategies pipeline first:"
    echo "  bash oscar_prediction_market/one_offs/d20260220_backtest_strategies/run.sh"
    exit 1
}

echo "============================================================"
echo "Step 1: Run Buy-Hold Backtests"
echo "============================================================"
bash "${SCRIPT_DIR}/run_backtests.sh"

echo ""
echo "============================================================"
echo "Step 2: Run EV + CVaR Scoring"
echo "============================================================"
bash "${SCRIPT_DIR}/run_scoring.sh"

echo ""
echo "============================================================"
echo "Step 3: Run Analysis (Plots, Tables, Asset Sync)"
echo "============================================================"
bash "${SCRIPT_DIR}/run_analysis.sh"

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "============================================================"
echo "Results: ${EXP_DIR}/"
echo "Finished at $(date)"
