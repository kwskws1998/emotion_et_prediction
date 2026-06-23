#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN_DIR="${RUN_DIR:-${OUTPUT_DIR:-emotion_et_prediction/runs/repro_cmcl_to_iitb_augmented_roberta}}"
TARGET_HF_MODEL_REPO="${HF_MODEL_REPO:-}"

HF_MODEL_REPO="" OUTPUT_DIR="$RUN_DIR" bash "$PACKAGE_DIR/scripts/train_repro_cmcl_to_iitb_augmented.sh"

RUN_DIR="$RUN_DIR" HF_MODEL_REPO="$TARGET_HF_MODEL_REPO" bash "$PACKAGE_DIR/scripts/package_hf_model.sh"
