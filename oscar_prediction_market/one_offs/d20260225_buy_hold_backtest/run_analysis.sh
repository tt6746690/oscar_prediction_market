#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODULE="oscar_prediction_market.one_offs.d20260225_buy_hold_backtest"
EXP_DIR="storage/d20260225_buy_hold_backtest"

echo "============================================================"
echo "Run Analysis (Plots, Tables, Asset Sync)"
echo "============================================================"

# Validate prerequisites
for year in 2024 2025; do
    [[ -d "${EXP_DIR}/${year}/results" ]] || { echo "ERROR: Missing ${EXP_DIR}/${year}/results/ — run earlier pipeline steps first"; exit 1; }
done

echo ""
echo "--- Per-year text analysis ---"
for year in 2024 2025; do
    echo "  ${year}..."
    uv run python -m "${MODULE}.analyze" --ceremony-year "${year}"
done

echo ""
echo "--- Per-year plots ---"
for year in 2024 2025; do
    echo "  ${year}..."
    uv run python -m "${MODULE}.analyze_plots" --ceremony-year "${year}"
done

echo ""
echo "--- Per-year scenario plots ---"
for year in 2024 2025; do
    echo "  ${year}..."
    uv run python -m "${MODULE}.analyze_scenario_plots" --ceremony-year "${year}"
done

echo ""
echo "--- Cross-year plots ---"
uv run python -m "${MODULE}.analyze_cross_year_plots"

echo ""
echo "--- Cross-year scenario plots ---"
uv run python -m "${MODULE}.analyze_scenario_plots" --mode cross-year

echo ""
echo "--- Per-year tables ---"
for year in 2024 2025; do
    echo "  ${year}..."
    uv run python -m "${MODULE}.generate_tables" --ceremony-year "${year}"
done

echo ""
echo "--- Cross-year tables ---"
uv run python -m "${MODULE}.generate_tables" --mode cross-year

echo ""
echo "--- Sync assets ---"
bash oscar_prediction_market/one_offs/sync_assets.sh

echo ""
echo "============================================================"
echo "Analysis complete. Outputs in:"
echo "============================================================"
for year in 2024 2025; do
    echo "  ${EXP_DIR}/${year}/results/"
    echo "  ${EXP_DIR}/${year}/plots/"
done
echo "  ${EXP_DIR}/cross_year/"
