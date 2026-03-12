#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODULE="oscar_prediction_market.one_offs.d20260225_buy_hold_backtest"
EXP_DIR="storage/d20260225_buy_hold_backtest"
SHARED_DIR="storage/d20260220_backtest_strategies"

echo "============================================================"
echo "Run Buy-Hold Backtests"
echo "============================================================"

# Validate prerequisites
[[ -d "${SHARED_DIR}/2024/models" ]] || { echo "ERROR: Missing ${SHARED_DIR}/2024/models/"; exit 1; }
[[ -d "${SHARED_DIR}/2025/models" ]] || { echo "ERROR: Missing ${SHARED_DIR}/2025/models/"; exit 1; }
[[ -d "${SHARED_DIR}/configs" ]]     || { echo "ERROR: Missing ${SHARED_DIR}/configs/"; exit 1; }

echo ""
echo "--- 2024 ceremony year ---"
uv run python -m "${MODULE}.run_backtests" --ceremony-year 2024

echo ""
echo "--- 2025 ceremony year ---"
uv run python -m "${MODULE}.run_backtests" --ceremony-year 2025

echo ""
echo "============================================================"
echo "Backtest outputs:"
echo "============================================================"
for year in 2024 2025; do
    echo "  ${EXP_DIR}/${year}/results/"
    ls -1 "${EXP_DIR}/${year}/results/"*.csv 2>/dev/null | sed 's/^/    /' || echo "    (no CSV files found)"
done
