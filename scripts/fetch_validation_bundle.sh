#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <dropbox_shared_url> <out_dir>"
  exit 1
fi

URL="$1"
OUT_DIR="$2"
ARCHIVE_NAME="${3:-bundle.tar.gz}"

mkdir -p "$OUT_DIR"

# Force Dropbox direct download by switching/adding dl=1.
if [[ "$URL" == *"dl="* ]]; then
  DL_URL="$(printf '%s' "$URL" | sed 's/dl=0/dl=1/g')"
else
  if [[ "$URL" == *"?"* ]]; then
    DL_URL="${URL}&dl=1"
  else
    DL_URL="${URL}?dl=1"
  fi
fi

ARCHIVE_PATH="$OUT_DIR/$ARCHIVE_NAME"

echo "Downloading bundle..."
wget -O "$ARCHIVE_PATH" "$DL_URL"

echo "Extracting..."
case "$ARCHIVE_PATH" in
  *.tar.gz|*.tgz)
    tar -xzf "$ARCHIVE_PATH" -C "$OUT_DIR"
    ;;
  *.zip)
    unzip -q "$ARCHIVE_PATH" -d "$OUT_DIR"
    ;;
  *)
    echo "Unsupported archive extension: $ARCHIVE_PATH"
    exit 1
    ;;
esac

echo "Done: $OUT_DIR"