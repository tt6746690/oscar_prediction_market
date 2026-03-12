#!/usr/bin/env bash
# Sync images referenced in one-off README.md files from storage/ to local assets/ dirs.
#
# Image convention in README.md:
#   ![storage/d20260214_trade_signal_backtest/wealth_curves.png](assets/wealth_curves.png)
#   - Alt text = source path relative to repo root (in storage/)
#   - Link target = local assets path (renders on GitHub)
#
# This script parses all README.md files under one_offs/, finds image references
# matching this convention, and copies the source files to the local assets/ dirs.
#
# Usage:
#   bash oscar_prediction_market/one_offs/sync_assets.sh
#   bash oscar_prediction_market/one_offs/sync_assets.sh path/to/specific/README.md

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

ONE_OFFS_DIR="oscar_prediction_market/one_offs"

if [[ $# -gt 0 ]]; then
    readmes=("$@")
else
    readmes=()
    while IFS= read -r f; do
        readmes+=("$f")
    done < <(find "$ONE_OFFS_DIR" -name "README.md" -type f | sort)
fi

total_copied=0
total_missing=0

for readme in "${readmes[@]}"; do
    dir=$(dirname "$readme")

    # Match ![storage/...](assets/...)
    # Use grep + sed for macOS compatibility (no -P flag)
    matches=$(grep -o '!\[storage/[^]]*\](assets/[^)]*)' "$readme" 2>/dev/null || true)
    [[ -z "$matches" ]] && continue

    echo "$(basename "$dir")/"

    while IFS= read -r match; do
        # Extract source path from alt text: ![SOURCE](...)
        src=$(echo "$match" | sed 's/^!\[//;s/\](.*//')
        # Extract destination path from link: ![...](DEST)
        dst=$(echo "$match" | sed 's/^.*](\([^)]*\))$/\1/')

        if [[ -f "$src" ]]; then
            mkdir -p "$(dirname "$dir/$dst")"
            cp "$src" "$dir/$dst"
            echo "  + $(basename "$dst")"
            total_copied=$((total_copied + 1))
        else
            echo "  ? $src (not found)"
            total_missing=$((total_missing + 1))
        fi
    done <<< "$matches"
done

echo ""
echo "Done: ${total_copied} copied, ${total_missing} missing"
