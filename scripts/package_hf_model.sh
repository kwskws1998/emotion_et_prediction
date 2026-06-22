#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:-emotion_et_prediction/runs/repro_main_20260622_121533}"
OUTPUT_DIR="${OUTPUT_DIR:-emotion_et_prediction/hf_emotion_et_iitb_lr5e5}"
ZIP_PATH="${ZIP_PATH:-emotion_et_prediction/hf_emotion_et_iitb_lr5e5_upload.zip}"
MODEL_NAME="${MODEL_NAME:-roberta-base}"
WEIGHT_NAME="${WEIGHT_NAME:-et_predictor2_iitb_lr5e5_seed42.safetensors}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
OVERWRITE="${OVERWRITE:-1}"

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="$(dirname "$PACKAGE_DIR")"
cd "$WORK_DIR"

ARGS=(
  --run-dir "$RUN_DIR"
  --output-dir "$OUTPUT_DIR"
  --zip-path "$ZIP_PATH"
  --model-name "$MODEL_NAME"
  --weight-name "$WEIGHT_NAME"
)

if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  ARGS+=(--local-files-only)
fi

if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi

python -m emotion_et_prediction.emotion_et.package_hf_model "${ARGS[@]}"
