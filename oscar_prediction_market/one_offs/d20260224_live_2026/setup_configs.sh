#!/usr/bin/env bash
# Copy configs from the 2025 backtest experiment to the 2026 live experiment.
#
# Feature configs, param grids, and CV splits are year-independent — they define
# *which* features/hyperparams to use, not *which year* of data.
#
# Usage:
#   cd "$(git rev-parse --show-toplevel)"
#   bash oscar_prediction_market/one_offs/d20260224_live_2026/setup_configs.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

EXP_DIR="storage/d20260224_live_2026"
SOURCE="storage/d20260220_backtest_strategies/configs"
DEST="$EXP_DIR/configs"

echo "============================================================"
echo "Setup: Copy configs from 2025 backtest to 2026 live"
echo "Source: $SOURCE"
echo "Dest:   $DEST"
echo "============================================================"

if [[ ! -d "$SOURCE" ]]; then
    echo "ERROR: Source configs not found at $SOURCE"
    echo "Run the 2025 backtest setup first."
    exit 1
fi

mkdir -p "$DEST"

# Copy features, param_grids, cv_splits
for subdir in features param_grids cv_splits; do
    if [[ -d "$DEST/$subdir" ]]; then
        echo "  SKIP $subdir: already exists"
    else
        cp -r "$SOURCE/$subdir" "$DEST/$subdir"
        n_files=$(find "$DEST/$subdir" -type f | wc -l | tr -d ' ')
        echo "  Copied $subdir/ ($n_files files)"
    fi
done

echo ""
echo "Configs ready at $DEST"
ls -la "$DEST/"
