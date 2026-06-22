#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="data/cmcl"
mkdir -p "$OUT_DIR"

curl -L -s \
  https://raw.githubusercontent.com/SPOClab-ca/cmcl-shared-task/main/data/provo.csv \
  -o "$OUT_DIR/provo.csv"

curl -L -s \
  https://raw.githubusercontent.com/SPOClab-ca/cmcl-shared-task/main/data/training_data/train_and_valid.csv \
  -o "$OUT_DIR/train_and_valid.csv"

curl -L -s \
  https://raw.githubusercontent.com/SPOClab-ca/cmcl-shared-task/main/data/training_data/train.csv \
  -o "$OUT_DIR/train.csv"

curl -L -s \
  https://raw.githubusercontent.com/SPOClab-ca/cmcl-shared-task/main/data/training_data/valid.csv \
  -o "$OUT_DIR/valid.csv"

echo "Saved CMCL data to $OUT_DIR"

