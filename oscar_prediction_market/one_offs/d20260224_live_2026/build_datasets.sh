#!/usr/bin/env bash
# Build as-of-date datasets for 2026 temporal snapshots.
#
# For each available snapshot date x 9 categories, runs the merge step of
# build_dataset with --as-of-date. Uses refreshed shared data from our
# experiment directory.
#
# Key differences from 2025 build:
# - --year-end 2026 (includes 2026 nominees)
# - Uses refreshed precursor_awards.json (with Annie/BAFTA 2026 winners)
# - Shared data from storage/d20260224_live_2026/shared/
# - Oscar nominations from storage/d20260218_build_all_datasets/{cat}/
#
# Usage:
#   cd "$(git rev-parse --show-toplevel)"
#   bash oscar_prediction_market/one_offs/d20260224_live_2026/build_datasets.sh
#
# Output: storage/d20260224_live_2026/datasets/{category}/{as_of_date}/

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

EXP_DIR="storage/d20260224_live_2026"
SOURCE_DIR="storage/d20260218_build_all_datasets"
SHARED_DIR="$EXP_DIR/shared"
DATASETS_DIR="$EXP_DIR/datasets"
BUILD_MODULE="oscar_prediction_market.data.build_dataset"

# 2026 post-nomination snapshot keys (all precursors resolved as of Mar 9, 2026)
SNAPSHOT_KEYS=(
    "2026-01-22_oscar_noms"  # Oscar nominations (CC + GG pre-nom)
    "2026-02-07_dga"         # DGA winner
    "2026-02-21_annie"       # Annie winner
    "2026-02-22_bafta"       # BAFTA winner
    "2026-02-28_pga"         # PGA winner
    "2026-03-01_sag"         # SAG winner
    "2026-03-08_asc"         # ASC winner
    "2026-03-08_wga"         # WGA winner
)

CATEGORIES=(
    best_picture
    directing
    actor_leading
    actress_leading
    actor_supporting
    actress_supporting
    original_screenplay
    cinematography
    animated_feature
)

CATEGORY_ENUMS=(
    BEST_PICTURE
    DIRECTING
    ACTOR_LEADING
    ACTRESS_LEADING
    ACTOR_SUPPORTING
    ACTRESS_SUPPORTING
    ORIGINAL_SCREENPLAY
    CINEMATOGRAPHY
    ANIMATED_FEATURE
)

echo "============================================================"
echo "Build 2026 As-Of-Date Datasets"
echo "Source nominations: $SOURCE_DIR (per-category)"
echo "Shared (refreshed): $SHARED_DIR"
echo "Output:  $DATASETS_DIR"
echo "Snapshots: ${#SNAPSHOT_KEYS[@]}"
echo "Categories: ${#CATEGORIES[@]}"
echo "Total merges: $(( ${#SNAPSHOT_KEYS[@]} * ${#CATEGORIES[@]} ))"
echo "============================================================"

# Verify refreshed shared data exists
for required in "$SHARED_DIR/film_metadata.json" "$SHARED_DIR/precursor_awards.json"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: Missing required file: $required"
        echo "Run refresh_data.py first:"
        echo "  uv run python -m oscar_prediction_market.one_offs.d20260224_live_2026.refresh_data"
        exit 1
    fi
done

SUCCEEDED=0
SKIPPED=0
FAILED=0

for snap_key in "${SNAPSHOT_KEYS[@]}"; do
    as_of_date="${snap_key:0:10}"
    echo ""
    echo "=== Snapshot: $snap_key (as-of: $as_of_date) ==="

    for i in "${!CATEGORIES[@]}"; do
        cat_slug="${CATEGORIES[$i]}"
        cat_enum="${CATEGORY_ENUMS[$i]}"
        cat_dir="$SOURCE_DIR/$cat_slug"
        output_dir="$DATASETS_DIR/$cat_slug/$snap_key"
        output_file="$output_dir/oscar_${cat_slug}_raw.json"

        # Check source oscar_nominations exist
        if [[ ! -f "$cat_dir/oscar_nominations.json" ]]; then
            echo "  ERROR: Missing $cat_dir/oscar_nominations.json"
            FAILED=$((FAILED + 1))
            continue
        fi

        # Skip if already built
        if [[ -f "$output_file" ]]; then
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        mkdir -p "$output_dir"

        echo "  Merging: $cat_slug @ $snap_key"
        if uv run python -m "$BUILD_MODULE" \
            --mode merge \
            --category "$cat_enum" \
            --year-start 2000 --year-end 2026 \
            --output-dir "$output_dir" \
            --input-dir "$cat_dir" \
            --shared-dir "$SHARED_DIR" \
            --as-of-date "$as_of_date"; then
            SUCCEEDED=$((SUCCEEDED + 1))
        else
            echo "  FAILED: $cat_slug @ $snap_key"
            FAILED=$((FAILED + 1))
        fi
    done
done

TOTAL=$(( ${#SNAPSHOT_KEYS[@]} * ${#CATEGORIES[@]} ))
echo ""
echo "============================================================"
echo "Build Summary"
echo "============================================================"
echo "  Total:     $TOTAL"
echo "  Succeeded: $SUCCEEDED"
echo "  Skipped:   $SKIPPED"
echo "  Failed:    $FAILED"

if [[ $FAILED -gt 0 ]]; then
    echo "WARNING: $FAILED merges failed!"
    exit 1
fi

echo ""
echo "All 2026 datasets built successfully!"
