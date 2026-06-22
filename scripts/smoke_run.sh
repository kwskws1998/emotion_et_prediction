#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ARTIFACT_DIR="emotion_et_prediction/artifacts/smoke"
RUN_DIR="emotion_et_prediction/runs/smoke_tiny"

if [[ ! -f /tmp/cmcl_provo.csv || ! -f /tmp/cmcl_train_and_valid.csv ]]; then
  echo "Missing /tmp/cmcl_provo.csv or /tmp/cmcl_train_and_valid.csv."
  echo "Download CMCL data first, then rerun this smoke script."
  exit 1
fi

python -m emotion_et_prediction.emotion_et.preprocess_iitb \
  --fixation-csv data/iitb_sentiment_gaze_raw/extracted/v2/Eye-tracking_and_SA-II_released_dataset/Fixation_sequence.csv \
  --text-csv data/iitb_sentiment_gaze_raw/extracted/v2/Eye-tracking_and_SA-II_released_dataset/text_and_annorations.csv \
  --output-csv "$ARTIFACT_DIR/iitb_v2_cmcl_scaled.csv" \
  --raw-output-csv "$ARTIFACT_DIR/iitb_v2_raw_word_features.csv" \
  --stats-json "$ARTIFACT_DIR/iitb_v2_preprocess_stats.json"

python -m emotion_et_prediction.emotion_et.train_et \
  --backend tiny \
  --pretrain-csv /tmp/cmcl_provo.csv \
  --pretrain-csv /tmp/cmcl_train_and_valid.csv \
  --finetune-csv "$ARTIFACT_DIR/iitb_v2_cmcl_scaled.csv" \
  --output-dir "$RUN_DIR" \
  --pretrain-epochs 1 \
  --finetune-epochs 1 \
  --batch-size 4 \
  --lr 0.001 \
  --max-length 64 \
  --max-pretrain-sentences 8 \
  --max-finetune-train-sentences 8 \
  --max-valid-sentences 4
