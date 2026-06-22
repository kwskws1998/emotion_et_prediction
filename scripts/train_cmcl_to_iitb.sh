#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-roberta-base}"
IITB_CSV="${IITB_CSV:-emotion_et_prediction/data/finetune_data/iitb_v2_cmcl_scaled.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-emotion_et_prediction/runs/cmcl_to_iitb_roberta}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-100}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-5e-5}"
MAX_LENGTH="${MAX_LENGTH:-256}"
DEVICE="${DEVICE:-auto}"

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="$(dirname "$PACKAGE_DIR")"
cd "$WORK_DIR"

python -m emotion_et_prediction.emotion_et.train_et \
  --backend hf \
  --model-name "$MODEL_NAME" \
  --pretrain-csv emotion_et_prediction/data/pretrain_data/provo.csv \
  --pretrain-csv emotion_et_prediction/data/pretrain_data/train_and_valid.csv \
  --finetune-csv "$IITB_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --pretrain-epochs "$PRETRAIN_EPOCHS" \
  --finetune-epochs "$FINETUNE_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --max-length "$MAX_LENGTH" \
  --device "$DEVICE"

if [[ -n "${HF_MODEL_REPO:-}" ]]; then
  hf upload "$HF_MODEL_REPO" "$OUTPUT_DIR" . --type model \
    --commit-message "Add CMCL-pretrained IITB-finetuned ET predictor"
fi
