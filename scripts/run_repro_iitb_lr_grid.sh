#!/usr/bin/env bash
set -euo pipefail

LR_GRID="${LR_GRID:-1e-5 2e-5 5e-5}"
RUN_ROOT="${RUN_ROOT:-emotion_et_prediction/runs/repro_iitb_lr_grid}"
UPLOAD_GRID_RUNS="${UPLOAD_GRID_RUNS:-0}"

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="$(dirname "$PACKAGE_DIR")"
TRAIN_SCRIPT="$PACKAGE_DIR/scripts/train_repro_cmcl_to_iitb.sh"
cd "$WORK_DIR"

mkdir -p "$RUN_ROOT"
SUMMARY_PATH="$RUN_ROOT/summary.tsv"
printf "lr\tbest_epoch\tselected_metric\tselected_score\tnFix\tFFD\tGPT\tTRT\tfixProp\tall\toutput_dir\n" > "$SUMMARY_PATH"

for lr in $LR_GRID; do
  safe_lr="${lr//./p}"
  safe_lr="${safe_lr//-/_}"
  output_dir="$RUN_ROOT/lr_${safe_lr}"

  echo "Running notebook-compatible IITB LR=$lr -> $output_dir"

  if [[ "$UPLOAD_GRID_RUNS" == "1" ]]; then
    LR="$lr" OUTPUT_DIR="$output_dir" bash "$TRAIN_SCRIPT"
  else
    LR="$lr" OUTPUT_DIR="$output_dir" HF_MODEL_REPO="" bash "$TRAIN_SCRIPT"
  fi

  python -c '
import json
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
lr = sys.argv[2]
output_dir = sys.argv[3]
payload = json.loads(metrics_path.read_text())
mae = payload["valid_mae"]
fields = [
    lr,
    str(payload["epoch"]),
    payload["selected_metric"],
    f"{payload['selected_score']:.6f}",
    f"{mae['nFix']:.6f}",
    f"{mae['FFD']:.6f}",
    f"{mae['GPT']:.6f}",
    f"{mae['TRT']:.6f}",
    f"{mae['fixProp']:.6f}",
    f"{mae['all']:.6f}",
    output_dir,
]
print("\t".join(fields))
' "$output_dir/metrics_best.json" "$lr" "$output_dir" >> "$SUMMARY_PATH"
done

echo "Saved notebook-compatible IITB LR grid summary to $SUMMARY_PATH"
