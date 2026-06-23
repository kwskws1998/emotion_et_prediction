#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export IITB_CSV="${IITB_CSV:-emotion_et_prediction/data/finetune_data/iitb_sa1_sa2_cmcl_scaled.csv}"
export OUTPUT_DIR="${OUTPUT_DIR:-emotion_et_prediction/runs/repro_cmcl_to_iitb_augmented_roberta}"

bash "$PACKAGE_DIR/scripts/train_repro_cmcl_to_iitb.sh"
