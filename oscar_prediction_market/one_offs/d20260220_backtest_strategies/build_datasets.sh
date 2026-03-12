#!/usr/bin/env bash
# Build as-of-date datasets for all temporal snapshots.
#
# For each snapshot date x category, runs the merge step of build_dataset with
# the appropriate --as-of-date. This gates features by the information available
# at each snapshot date.
#
# Reuses shared intermediates (film_metadata, precursor_awards, oscar_nominations)
# from storage/d20260218_build_all_datasets/. Only the merge step is re-run
# per snapshot date.
#
# Usage:
#   cd "$(git rev-parse --show-toplevel)"
#   # 2025 (default):
#   bash oscar_prediction_market/one_offs/d20260220_backtest_strategies/build_datasets.sh
#   # 2024:
#   bash oscar_prediction_market/one_offs/d20260220_backtest_strategies/build_datasets.sh 2024
#
# Output: storage/d20260220_backtest_strategies/{YEAR}/datasets/{category}/{as_of_date}/

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# ============================================================================
# Year selection (default: 2025)
# ============================================================================
CEREMONY_YEAR="${1:-2025}"

EXP_DIR="storage/d20260220_backtest_strategies"
SOURCE_DIR="storage/d20260218_build_all_datasets"
DATASETS_DIR="$EXP_DIR/$CEREMONY_YEAR/datasets"
BUILD_MODULE="oscar_prediction_market.data.build_dataset"

# Per-year snapshot dates and categories
if [[ "$CEREMONY_YEAR" == "2024" ]]; then
    # 2024 post-nomination snapshot keys (pre-ceremony only; WGA Apr 14 excluded)
    SNAPSHOT_KEYS=(
        "2024-01-23_oscar_noms"   # Oscar nominations (GG + CC already baked in)
        "2024-02-10_dga"          # DGA winner
        "2024-02-17_annie"        # Annie winner
        "2024-02-18_bafta"        # BAFTA winner
        "2024-02-24_sag"          # SAG winner
        "2024-02-25_pga"          # PGA winner
        "2024-03-03_asc"          # ASC winner
    )
    # 8 categories (no Cinematography market on Kalshi for 2024)
    CATEGORIES=(
        best_picture directing actor_leading actress_leading
        actor_supporting actress_supporting original_screenplay animated_feature
    )
    CATEGORY_ENUMS=(
        BEST_PICTURE DIRECTING ACTOR_LEADING ACTRESS_LEADING
        ACTOR_SUPPORTING ACTRESS_SUPPORTING ORIGINAL_SCREENPLAY ANIMATED_FEATURE
    )
    YEAR_END="2024"
elif [[ "$CEREMONY_YEAR" == "2025" ]]; then
    # 2025 post-nomination snapshot keys (per-event)
    SNAPSHOT_KEYS=(
        "2025-01-23_oscar_noms"      # Oscar nominations
        "2025-02-07_critics_choice"  # Critics Choice winner
        "2025-02-08_annie"           # Annie winner
        "2025-02-08_pga"             # PGA winner
        "2025-02-08_dga"             # DGA winner
        "2025-02-15_wga"             # WGA winner
        "2025-02-16_bafta"           # BAFTA winner
        "2025-02-23_sag"             # SAG winner
        "2025-02-23_asc"             # ASC winner
    )
    CATEGORIES=(
        best_picture directing actor_leading actress_leading
        actor_supporting actress_supporting original_screenplay
        cinematography animated_feature
    )
    CATEGORY_ENUMS=(
        BEST_PICTURE DIRECTING ACTOR_LEADING ACTRESS_LEADING
        ACTOR_SUPPORTING ACTRESS_SUPPORTING ORIGINAL_SCREENPLAY
        CINEMATOGRAPHY ANIMATED_FEATURE
    )
    YEAR_END="2025"
else
    echo "ERROR: Unsupported ceremony year: $CEREMONY_YEAR (supported: 2024, 2025)"
    exit 1
fi

echo "============================================================"
echo "Build As-Of-Date Datasets (Ceremony Year: $CEREMONY_YEAR)"
echo "Source:  $SOURCE_DIR (shared intermediates)"
echo "Output:  $DATASETS_DIR"
echo "Snapshots: ${#SNAPSHOT_KEYS[@]}"
echo "Categories: ${#CATEGORIES[@]}"
echo "Total merges: $(( ${#SNAPSHOT_KEYS[@]} * ${#CATEGORIES[@]} ))"
echo "============================================================"

# Verify shared intermediates exist
for required in "$SOURCE_DIR/shared/film_metadata.json" "$SOURCE_DIR/shared/precursor_awards.json"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: Missing required file: $required"
        echo "Run d20260218_build_all_datasets/build_all.sh first."
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
            --year-start 2000 --year-end "$YEAR_END" \
            --output-dir "$output_dir" \
            --input-dir "$cat_dir" \
            --shared-dir "$SOURCE_DIR/shared" \
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
echo "All datasets built successfully!"
