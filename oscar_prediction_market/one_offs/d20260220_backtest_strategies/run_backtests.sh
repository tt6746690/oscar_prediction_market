#!/usr/bin/env bash
# Run multi-category backtests for 2025 ceremony.
#
# Fetches Kalshi market data, matches nominees, estimates spreads,
# runs backtests across all model types × trading configs, and
# settles against known winners.
#
# Prerequisites:
#   - Models trained (train_models.sh completed)
#   - ticker_inventory.json in storage dir
#
# Usage:
#   cd "$(git rev-parse --show-toplevel)"
#   bash oscar_prediction_market/one_offs/d20260220_backtest_strategies/run_backtests.sh 2>&1 | tee storage/d20260220_backtest_strategies/run_backtests.log
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

EXP_DIR="storage/d20260220_backtest_strategies"
RESULTS_DIR="${EXP_DIR}/2025/results"

echo "=== Multi-Category Backtest ==="
echo "Results will be in: ${RESULTS_DIR}"
echo ""

# Ensure results dir
mkdir -p "${RESULTS_DIR}"

# Run the backtest
uv run python -m oscar_prediction_market.one_offs.d20260220_backtest_strategies.run_backtests

echo ""
echo "=== Backtest complete ==="
echo "Results:"
ls -la "${RESULTS_DIR}/" 2>/dev/null || echo "(no result files yet)"
