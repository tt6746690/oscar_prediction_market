#!/usr/bin/env zsh
# Build raw datasets for ALL 9 Oscar categories that have precursor mappings.
#
# Stages minimize duplication:
#   Stage 1: Build oscar_nominations.json per category (category-specific nominees)
#   Stage 2: Build ONE shared film_metadata.json covering ALL categories' films
#   Stage 3: Build ONE shared precursor_awards.json (same data for every category)
#   Stage 4: Merge each category using oscar_nominations from its own dir and
#            metadata/precursors from the shared dir.
#
# Usage:
#   zsh .../build_all.sh                      # DEFAULT: no date filtering (all awards)
#   zsh .../build_all.sh 2026-02-18           # specific as-of date
#   zsh .../build_all.sh --all                # explicit: no date filtering
#
# Output: storage/d20260218_build_all_datasets/
#   shared/                    -- film_metadata.json, precursor_awards.json
#   best_picture/              -- oscar_nominations.json, oscar_best_picture_raw.json
#   directing/                 -- oscar_nominations.json, oscar_directing_raw.json
#   ...

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

EXP_DIR="storage/d20260218_build_all_datasets"
SHARED_DIR="${EXP_DIR}/shared"

# All 9 categories with precursor mappings
CATEGORIES=(
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

# Parse arguments
AS_OF_ARG=""
if [[ "${1:-}" == "--all" ]] || [[ -z "${1:-}" ]]; then
    echo "Building with all awards available (no date filtering)"
elif [[ -n "${1:-}" ]]; then
    AS_OF_ARG="--as-of-date $1"
    echo "Building with as-of date: $1"
fi

echo ""
echo "============================================================"
echo "Multi-Category Oscar Dataset Build"
echo "Output:     $EXP_DIR"
echo "Shared dir: $SHARED_DIR"
echo "Categories: ${#CATEGORIES[@]}"
echo "============================================================"

# ------------------------------------------------------------------
# Stage 1: Build oscar_nominations.json for each category
# ------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Stage 1: Oscar nominations (per category)"
echo "============================================================"

for CAT in "${CATEGORIES[@]}"; do
    CAT_SLUG=$(echo "$CAT" | tr '[:upper:]' '[:lower:]')
    CAT_DIR="${EXP_DIR}/${CAT_SLUG}"
    OUTPUT="$CAT_DIR/oscar_nominations.json"

    if [[ -f "$OUTPUT" ]]; then
        echo "  SKIP $CAT: $OUTPUT already exists"
        continue
    fi

    echo "  Building nominations: $CAT"
    uv run python -m oscar_prediction_market.data.build_dataset \
        --mode oscar \
        --category "$CAT" \
        --year-start 2000 --year-end 2026 \
        --output-dir "$CAT_DIR"
done

# ------------------------------------------------------------------
# Stage 2: Build ONE shared film_metadata.json from all categories
# ------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Stage 2: Shared film metadata (union of all categories)"
echo "============================================================"

if [[ -f "$SHARED_DIR/film_metadata.json" ]]; then
    echo "  SKIP: $SHARED_DIR/film_metadata.json already exists"
else
    # Build metadata from all category dirs in one pass.
    # --input-dir points to the first category dir (required by CLI).
    # --extra-input-dirs lists the remaining dirs so all film IDs are included.
    FIRST_CAT_SLUG=$(echo "${CATEGORIES[1]}" | tr '[:upper:]' '[:lower:]')
    EXTRA_DIRS=()
    for CAT in "${CATEGORIES[@]:1}"; do
        CAT_SLUG=$(echo "$CAT" | tr '[:upper:]' '[:lower:]')
        EXTRA_DIRS+=("${EXP_DIR}/${CAT_SLUG}")
    done

    echo "  Building shared metadata from ${#CATEGORIES[@]} category dirs..."
    uv run python -m oscar_prediction_market.data.build_dataset \
        --mode metadata \
        --category "${CATEGORIES[1]}" \
        --year-start 2000 --year-end 2026 \
        --input-dir "${EXP_DIR}/${FIRST_CAT_SLUG}" \
        --output-dir "$SHARED_DIR" \
        --shared-dir "$SHARED_DIR" \
        --extra-input-dirs "${EXTRA_DIRS[@]}"
fi

# ------------------------------------------------------------------
# Stage 3: Build ONE shared precursor_awards.json
# ------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Stage 3: Shared precursor awards"
echo "============================================================"

if [[ -f "$SHARED_DIR/precursor_awards.json" ]]; then
    echo "  SKIP: $SHARED_DIR/precursor_awards.json already exists"
else
    echo "  Fetching precursor awards..."
    uv run python -m oscar_prediction_market.data.build_dataset \
        --mode precursors \
        --category BEST_PICTURE \
        --year-start 2000 --year-end 2026 \
        --output-dir "$SHARED_DIR" \
        --shared-dir "$SHARED_DIR"
fi

# ------------------------------------------------------------------
# Stage 4: Merge each category using shared metadata/precursors
# ------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Stage 4: Merge (per category, using shared files)"
echo "============================================================"

FAILED=()
SUCCEEDED=()

for CAT in "${CATEGORIES[@]}"; do
    CAT_SLUG=$(echo "$CAT" | tr '[:upper:]' '[:lower:]')
    CAT_DIR="${EXP_DIR}/${CAT_SLUG}"
    OUTPUT_FILE="${CAT_DIR}/oscar_${CAT_SLUG}_raw.json"

    if [[ -f "${OUTPUT_FILE}" ]]; then
        echo "  SKIP $CAT: $OUTPUT_FILE already exists"
        SUCCEEDED+=("${CAT}")
        continue
    fi

    echo ""
    echo "  Merging: $CAT"
    if uv run python -m oscar_prediction_market.data.build_dataset \
        --mode merge \
        --category "$CAT" \
        --year-start 2000 --year-end 2026 \
        --output-dir "$CAT_DIR" \
        --input-dir "$CAT_DIR" \
        --shared-dir "$SHARED_DIR" \
        ${AS_OF_ARG}; then
        SUCCEEDED+=("${CAT}")
    else
        FAILED+=("${CAT}")
        echo "  FAILED: $CAT"
    fi
done

echo ""
echo "============================================================"
echo "Build Summary"
echo "============================================================"
echo "Succeeded: ${#SUCCEEDED[@]}/${#CATEGORIES[@]}"
for CAT in "${SUCCEEDED[@]}"; do
    echo "  ok ${CAT}"
done
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed: ${#FAILED[@]}/${#CATEGORIES[@]}"
    for CAT in "${FAILED[@]}"; do
        echo "  FAIL ${CAT}"
    done
    exit 1
fi

echo ""
echo "All datasets built successfully!"
echo ""
echo "Shared files:"
ls -lh "${SHARED_DIR}/"*.json 2>/dev/null || true
echo ""
echo "Output files:"
for CAT in "${CATEGORIES[@]}"; do
    CAT_SLUG=$(echo "$CAT" | tr '[:upper:]' '[:lower:]')
    ls -lh "${EXP_DIR}/${CAT_SLUG}/oscar_${CAT_SLUG}_raw.json" 2>/dev/null || true
done
