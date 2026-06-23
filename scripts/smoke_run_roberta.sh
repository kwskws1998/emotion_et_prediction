#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
PACKAGE_DIR="$ROOT_DIR/emotion_et_prediction"
export PYTHONPATH="$PACKAGE_DIR${PYTHONPATH:+:$PYTHONPATH}"

RUN_DIR="${RUN_DIR:-emotion_et_prediction/runs/smoke_roberta_base}"
PRETRAIN_DATA_DIR="${PRETRAIN_DATA_DIR:-emotion_et_prediction/data/pretrain_data}"
FINETUNE_CSV="${FINETUNE_CSV:-emotion_et_prediction/data/finetune_data/iitb_sa1_sa2_cmcl_scaled.csv}"
DEVICE="${DEVICE:-cpu}"

if [[ ! -f "$PRETRAIN_DATA_DIR/provo.csv" || ! -f "$PRETRAIN_DATA_DIR/train_and_valid.csv" ]]; then
  echo "Missing $PRETRAIN_DATA_DIR/provo.csv or $PRETRAIN_DATA_DIR/train_and_valid.csv."
  echo "Run emotion_et_prediction/scripts/download_cmcl_data.sh first, then rerun this smoke script."
  exit 1
fi

if [[ ! -f "$FINETUNE_CSV" ]]; then
  echo "Missing $FINETUNE_CSV."
  echo "Run emotion_et_prediction/emotion_et/preprocess_iitb_sa1.py or restore the processed final fine-tune CSV first."
  exit 1
fi

TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python -m emotion_et.train_et \
  --backend hf \
  --model-name roberta-base \
  --local-files-only \
  --pretrain-csv "$PRETRAIN_DATA_DIR/provo.csv" \
  --pretrain-csv "$PRETRAIN_DATA_DIR/train_and_valid.csv" \
  --finetune-csv "$FINETUNE_CSV" \
  --output-dir "$RUN_DIR" \
  --pretrain-epochs 1 \
  --finetune-epochs 1 \
  --batch-size 1 \
  --lr 0.00001 \
  --max-length 64 \
  --device "$DEVICE" \
  --max-pretrain-sentences 2 \
  --max-finetune-train-sentences 2 \
  --max-valid-sentences 2
