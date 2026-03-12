#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODULE="oscar_prediction_market.one_offs.d20260225_buy_hold_backtest"
EXP_DIR="storage/d20260225_buy_hold_backtest"

echo "============================================================"
echo "Run EV + CVaR Scoring"
echo "============================================================"

# Validate prerequisites
[[ -f "${EXP_DIR}/2024/results/entry_pnl.csv" ]] || { echo "ERROR: Missing ${EXP_DIR}/2024/results/entry_pnl.csv — run run_backtests.sh first"; exit 1; }
[[ -f "${EXP_DIR}/2025/results/entry_pnl.csv" ]] || { echo "ERROR: Missing ${EXP_DIR}/2025/results/entry_pnl.csv — run run_backtests.sh first"; exit 1; }

echo ""
echo "--- 2024 ceremony year ---"
uv run python -m "${MODULE}.scenario_scoring" --ceremony-year 2024

echo ""
echo "--- 2025 ceremony year ---"
uv run python -m "${MODULE}.scenario_scoring" --ceremony-year 2025

echo ""
echo "--- Cross-year ---"
uv run python -m "${MODULE}.scenario_scoring" --cross-year

echo ""
echo "============================================================"
echo "Scoring outputs:"
echo "============================================================"
for year in 2024 2025; do
    echo "  ${EXP_DIR}/${year}/results/"
    ls -1 "${EXP_DIR}/${year}/results/"*.csv 2>/dev/null | sed 's/^/    /' || echo "    (no CSV files found)"
done
echo "  ${EXP_DIR}/cross_year/results/"
ls -1 "${EXP_DIR}/cross_year/results/"*.csv 2>/dev/null | sed 's/^/    /' || echo "    (no CSV files found)"
