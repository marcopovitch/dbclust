#!/usr/bin/env bash
set -euo pipefail

PATCH_FILE="patch/obspy/obspy.io.nlloc.core.py.patch"

# Check that the patch file exists
if [[ ! -f "$PATCH_FILE" ]]; then
    echo "Error: patch file not found: $PATCH_FILE"
    exit 1
fi

# Locate the target file
TARGET_FILE=$(find .venv/lib -path '*/obspy/io/nlloc/core.py' -print -quit)

if [[ -z "$TARGET_FILE" ]]; then
    echo "Error: obspy core.py file not found"
    exit 1
fi

echo "Found target file: $TARGET_FILE"

# Apply the patch
patch --batch --forward "$TARGET_FILE" < "$PATCH_FILE"

echo "Patch applied successfully."
