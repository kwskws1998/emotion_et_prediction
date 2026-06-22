#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SMOKE_DATA_DIR="emotion_et_prediction/runs/smoke_data"
RUN_DIR="emotion_et_prediction/runs/smoke_roberta_base"
PRETRAIN_DATA_DIR="emotion_et_prediction/data/pretrain_data"

if [[ ! -f "$PRETRAIN_DATA_DIR/provo.csv" || ! -f "$PRETRAIN_DATA_DIR/train_and_valid.csv" ]]; then
  echo "Missing $PRETRAIN_DATA_DIR/provo.csv or $PRETRAIN_DATA_DIR/train_and_valid.csv."
  echo "Run emotion_et_prediction/scripts/download_cmcl_data.sh first, then rerun this smoke script."
  exit 1
fi

if [[ ! -f "$SMOKE_DATA_DIR/iitb_v2_cmcl_scaled.csv" ]]; then
  python -m emotion_et_prediction.emotion_et.preprocess_iitb \
    --fixation-csv data/iitb_sentiment_gaze_raw/extracted/v2/Eye-tracking_and_SA-II_released_dataset/Fixation_sequence.csv \
    --text-csv data/iitb_sentiment_gaze_raw/extracted/v2/Eye-tracking_and_SA-II_released_dataset/text_and_annorations.csv \
    --output-csv "$SMOKE_DATA_DIR/iitb_v2_cmcl_scaled.csv" \
    --raw-output-csv "$SMOKE_DATA_DIR/iitb_v2_raw_word_features.csv" \
    --stats-json "$SMOKE_DATA_DIR/iitb_v2_preprocess_stats.json"
fi

TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python -m emotion_et_prediction.emotion_et.train_et \
  --backend hf \
  --model-name roberta-base \
  --local-files-only \
  --pretrain-csv "$PRETRAIN_DATA_DIR/provo.csv" \
  --pretrain-csv "$PRETRAIN_DATA_DIR/train_and_valid.csv" \
  --finetune-csv "$SMOKE_DATA_DIR/iitb_v2_cmcl_scaled.csv" \
  --output-dir "$RUN_DIR" \
  --pretrain-epochs 1 \
  --finetune-epochs 1 \
  --batch-size 1 \
  --lr 0.00001 \
  --max-length 64 \
  --max-pretrain-sentences 2 \
  --max-finetune-train-sentences 2 \
  --max-valid-sentences 2
