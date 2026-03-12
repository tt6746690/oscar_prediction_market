#!/usr/bin/env bash
# Regression test: verify the d20260219_backtest_refactor doesn't change engine output.
#
# Runs compare.py against the 24 golden configs captured before the refactor
# and reports PASS/FAIL for each.
#
# Usage:
#   cd "$(git rev-parse --show-toplevel)"
#   bash one_offs/d20260219_backtest_regression/run.sh 2>&1 | tee storage/d20260219_backtest_regression/run.log
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

FIXTURE="storage/d20260219_backtest_regression/golden_fixture.json"
MODELS_DIR="storage/d20260214_trade_signal_ablation"
LOG="storage/d20260219_backtest_regression/run.log"

# Validate required inputs
[[ -f "${FIXTURE}" ]] || { echo "ERROR: Missing golden fixture: ${FIXTURE}"; exit 1; }
[[ -d "${MODELS_DIR}/models" ]] || { echo "ERROR: Missing models dir: ${MODELS_DIR}/models"; exit 1; }

echo "=== Backtest Regression Test ==="
echo "  Fixture:    ${FIXTURE}"
echo "  Models dir: ${MODELS_DIR}"
echo ""

uv run python -m oscar_prediction_market.one_offs.d20260219_backtest_regression.compare \
    --fixture "${FIXTURE}" \
    --models-dir "${MODELS_DIR}"
