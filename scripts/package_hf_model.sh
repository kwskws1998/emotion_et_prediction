#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:-emotion_et_prediction/runs/repro_cmcl_to_iitb_augmented_roberta}"
OUTPUT_DIR="${OUTPUT_DIR:-emotion_et_prediction/hf_emotion_et_augmented}"
ZIP_PATH="${ZIP_PATH:-emotion_et_prediction/hf_emotion_et_augmented_upload.zip}"
MODEL_NAME="${MODEL_NAME:-roberta-base}"
WEIGHT_NAME="${WEIGHT_NAME:-et_predictor2_seed42.safetensors}"
LR_LABEL="${LR_LABEL:-5e-5}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
OVERWRITE="${OVERWRITE:-1}"

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="$(dirname "$PACKAGE_DIR")"
cd "$WORK_DIR"
export PYTHONPATH="$PACKAGE_DIR${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  --run-dir "$RUN_DIR"
  --output-dir "$OUTPUT_DIR"
  --zip-path "$ZIP_PATH"
  --model-name "$MODEL_NAME"
  --weight-name "$WEIGHT_NAME"
  --lr-label "$LR_LABEL"
)

if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  ARGS+=(--local-files-only)
fi

if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi

python -m emotion_et.package_hf_model "${ARGS[@]}"

if [[ -n "${HF_MODEL_REPO:-}" ]]; then
  hf upload "$HF_MODEL_REPO" "$OUTPUT_DIR" . --type model \
    --commit-message "Add augmented emotion ET predictor bundle"
fi
