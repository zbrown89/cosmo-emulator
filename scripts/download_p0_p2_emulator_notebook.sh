#!/usr/bin/env bash
set -euo pipefail

REPO="zbrown89/cosmo-emulator"
BRANCH="test-connection"
NOTEBOOK="04_train_p0_p2_pt_coefficient_emulator.ipynb"
DEST_DIR="${1:-notebooks}"

mkdir -p "$DEST_DIR"
URL="https://raw.githubusercontent.com/$REPO/$BRANCH/notebooks/$NOTEBOOK"

echo "Downloading $NOTEBOOK ..."
curl --fail --location --silent --show-error "$URL" -o "$DEST_DIR/$NOTEBOOK"
echo "Saved to: $DEST_DIR/$NOTEBOOK"
