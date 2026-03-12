#!/bin/bash
# Regenerate all processed datasets for Oscar Best Picture prediction.
#
# This script runs the full data pipeline:
# 1. Extract Oscar nominations from oscars.csv
# 2. Fetch film metadata from OMDb + TMDb (uses cache)
# 3. Fetch precursor awards from Wikipedia (uses cache)
# 4. Merge all sources into raw dataset
#
# By default, generates dataset with as-of date = TODAY (for real-time predictions).
# This filters precursor awards based on what's announced as of today.
#
# Usage:
#   bash one_offs/d20260201_build_dataset/build_dataset.sh                 # DEFAULT: today
#   bash one_offs/d20260201_build_dataset/build_dataset.sh 2026-02-04      # specific date
#   bash one_offs/d20260201_build_dataset/build_dataset.sh --all           # no date filtering
#
# Output goes to storage/d20260201_build_dataset/

set -e

# Change to project root (use git to resolve correctly even through symlinks)
cd "$(git rev-parse --show-toplevel)"

OUTPUT_DIR="storage/d20260201_build_dataset"

# Parse arguments
AS_OF_ARG=""
if [[ "$1" == "--all" ]]; then
    echo "Regenerating with all awards available (no date filtering)"
elif [[ -n "$1" ]]; then
    AS_OF_ARG="--as-of-date $1"
    echo "Regenerating with as-of date: $1"
else
    # Default to today
    TODAY=$(date +%Y-%m-%d)
    AS_OF_ARG="--as-of-date $TODAY"
    echo "Regenerating with as-of date: $TODAY (default: today)"
fi

echo ""
echo "============================================================"
echo "Oscar Best Picture Dataset Regeneration"
echo "Output: $OUTPUT_DIR"
echo "============================================================"

# Stage 1: Oscar nominations
echo ""
echo "[1/4] Extracting Oscar nominations from oscars.csv..."
uv run python -m oscar_prediction_market.data.build_dataset \
    --mode oscar --year-start 2000 --year-end 2026 \
    --output-dir "$OUTPUT_DIR"

# Stage 2: Film metadata (uses cache, so fast if already fetched)
echo ""
echo "[2/4] Fetching film metadata from OMDb + TMDb..."
uv run python -m oscar_prediction_market.data.build_dataset \
    --mode metadata --year-start 2000 --year-end 2026 \
    --output-dir "$OUTPUT_DIR"

# Stage 3: Precursor awards (uses cache)
echo ""
echo "[3/4] Fetching precursor awards from Wikipedia..."
uv run python -m oscar_prediction_market.data.build_dataset \
    --mode precursors --year-start 2000 --year-end 2026 \
    --output-dir "$OUTPUT_DIR"

# Stage 4: Merge all sources
echo ""
echo "[4/4] Merging into raw dataset..."
uv run python -m oscar_prediction_market.data.build_dataset \
    --mode merge --year-start 2000 --year-end 2026 \
    --output-dir "$OUTPUT_DIR" $AS_OF_ARG

echo ""
echo "============================================================"
echo "Done! Output files in $OUTPUT_DIR"
echo "============================================================"
echo ""
echo "Files generated:"
ls -la "$OUTPUT_DIR"/*.json
