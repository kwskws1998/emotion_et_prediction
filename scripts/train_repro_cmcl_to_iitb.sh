#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-roberta-base}"
IITB_CSV="${IITB_CSV:-emotion_et_prediction/data/finetune_data/iitb_sa1_sa2_cmcl_scaled.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-emotion_et_prediction/runs/repro_cmcl_to_iitb_augmented_roberta}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-100}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-5e-5}"
REPRO_MAX_LENGTH="${REPRO_MAX_LENGTH:-512}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"
VALID_RATIO="${VALID_RATIO:-0.15}"
SPLIT_SEED="${SPLIT_SEED:-42}"
PRETRAIN_PROVO_CSV="${PRETRAIN_PROVO_CSV:-emotion_et_prediction/data/pretrain_data/provo.csv}"
PRETRAIN_ZUCO_CSV="${PRETRAIN_ZUCO_CSV:-emotion_et_prediction/data/pretrain_data/train_and_valid.csv}"

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="$(dirname "$PACKAGE_DIR")"
cd "$WORK_DIR"
export PYTHONPATH="$PACKAGE_DIR${PYTHONPATH:+:$PYTHONPATH}"

VALID_ARGS=(--valid-ratio "$VALID_RATIO" --split-seed "$SPLIT_SEED")
if [[ -n "${VALID_CSV:-}" ]]; then
  VALID_ARGS=(--valid-csv "$VALID_CSV")
fi

python -m emotion_et.reproduce_et2 \
  --seeds "$SEED" \
  --model-name "$MODEL_NAME" \
  --pretrain-csv "$PRETRAIN_PROVO_CSV" \
  --pretrain-csv "$PRETRAIN_ZUCO_CSV" \
  --finetune-csv "$IITB_CSV" \
  "${VALID_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --provo-epochs "$PRETRAIN_EPOCHS" \
  --task-epochs "$FINETUNE_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --max-length "$REPRO_MAX_LENGTH" \
  --device "$DEVICE" \
  --pretrain-label CMCL \
  --finetune-label IITB

if [[ -n "${HF_MODEL_REPO:-}" && "${UPLOAD_RAW_RUN:-0}" == "1" ]]; then
  hf upload "$HF_MODEL_REPO" "$OUTPUT_DIR" . --type model \
    --commit-message "Add raw notebook-compatible CMCL-to-IITB ET predictor run"
elif [[ -n "${HF_MODEL_REPO:-}" ]]; then
  echo "HF_MODEL_REPO is set, but raw run upload is disabled."
  echo "Use scripts/train_package_upload_augmented.sh or scripts/package_hf_model.sh to upload the Hugging Face bundle."
fi
