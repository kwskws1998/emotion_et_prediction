#!/usr/bin/env bash
set -euo pipefail

RAW_IITB_CSV="${RAW_IITB_CSV:-emotion_et_prediction/data/finetune_data/iitb_v2_raw_word_features.csv}"
RUN_ROOT="${RUN_ROOT:-emotion_et_prediction/runs/repro_iitb_zero_scaling_ablation}"
GENERATED_DATA_DIR="${GENERATED_DATA_DIR:-$RUN_ROOT/data}"
RUN_TRAIN="${RUN_TRAIN:-1}"
UPLOAD_ABLATION_RUNS="${UPLOAD_ABLATION_RUNS:-0}"

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="$(dirname "$PACKAGE_DIR")"
TRAIN_SCRIPT="$PACKAGE_DIR/scripts/train_repro_cmcl_to_iitb.sh"
cd "$WORK_DIR"

mkdir -p "$RUN_ROOT" "$GENERATED_DATA_DIR"
DATA_SUMMARY_PATH="$RUN_ROOT/data_summary.tsv"

python -c '
import json
import sys
from pathlib import Path

import pandas as pd

from emotion_et_prediction.emotion_et.constants import (
    CMCL_NONZERO_TARGET_STATS,
    CMCL_TARGET_STATS,
    FEATURE_NAMES,
)
from emotion_et_prediction.emotion_et.preprocess_iitb import (
    scale_features_to_cmcl,
    summarize_features,
)

raw_csv = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
summary_path = Path(sys.argv[3])

raw_df = pd.read_csv(raw_csv)
variants = [
    ("preserve_zero_rows", True, CMCL_NONZERO_TARGET_STATS),
    ("scale_zero_rows", False, CMCL_TARGET_STATS),
]

summary_path.write_text(
    "variant\trows\tzero_rows\tnFix_mean\tFFD_mean\tGPT_mean\tTRT_mean\tfixProp_mean\tcsv\n",
    encoding="utf-8",
)

for variant, preserve_zero_rows, target_stats in variants:
    scaled_df = scale_features_to_cmcl(
        raw_df,
        target_stats=target_stats,
        preserve_zero_rows=preserve_zero_rows,
    )
    csv_path = out_dir / f"iitb_v2_{variant}.csv"
    stats_path = out_dir / f"iitb_v2_{variant}_stats.json"
    scaled_df.to_csv(csv_path, index=False)

    stats = summarize_features(
        raw_df,
        scaled_df,
        target_stats=target_stats,
        preserve_zero_rows=preserve_zero_rows,
    )
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    means = scaled_df[FEATURE_NAMES].mean()
    zero_rows = int(scaled_df[FEATURE_NAMES].eq(0.0).all(axis=1).sum())
    fields = [
        variant,
        str(len(scaled_df)),
        str(zero_rows),
        f"{means['nFix']:.6f}",
        f"{means['FFD']:.6f}",
        f"{means['GPT']:.6f}",
        f"{means['TRT']:.6f}",
        f"{means['fixProp']:.6f}",
        str(csv_path),
    ]
    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write("\t".join(fields) + "\n")

print(f"Saved zero-scaling data summary to {summary_path}")
' "$RAW_IITB_CSV" "$GENERATED_DATA_DIR" "$DATA_SUMMARY_PATH"

if [[ "$RUN_TRAIN" == "0" ]]; then
  echo "RUN_TRAIN=0, stopping after data generation."
  exit 0
fi

SUMMARY_PATH="$RUN_ROOT/summary.tsv"
printf "variant\tbest_epoch\tselected_metric\tselected_score\tnFix\tFFD\tGPT\tTRT\tfixProp\tall\toutput_dir\n" > "$SUMMARY_PATH"

for variant in preserve_zero_rows scale_zero_rows; do
  iitb_csv="$GENERATED_DATA_DIR/iitb_v2_${variant}.csv"
  output_dir="$RUN_ROOT/$variant"

  echo "Running notebook-compatible zero-scaling variant=$variant -> $output_dir"

  if [[ "$UPLOAD_ABLATION_RUNS" == "1" ]]; then
    IITB_CSV="$iitb_csv" OUTPUT_DIR="$output_dir" bash "$TRAIN_SCRIPT"
  else
    IITB_CSV="$iitb_csv" OUTPUT_DIR="$output_dir" HF_MODEL_REPO="" bash "$TRAIN_SCRIPT"
  fi

  python -c '
import json
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
variant = sys.argv[2]
output_dir = sys.argv[3]
payload = json.loads(metrics_path.read_text())
mae = payload["valid_mae"]
score = payload["selected_score"]
nfix = mae["nFix"]
ffd = mae["FFD"]
gpt = mae["GPT"]
trt = mae["TRT"]
fixprop = mae["fixProp"]
all_mae = mae["all"]
fields = [
    variant,
    str(payload["epoch"]),
    payload["selected_metric"],
    f"{score:.6f}",
    f"{nfix:.6f}",
    f"{ffd:.6f}",
    f"{gpt:.6f}",
    f"{trt:.6f}",
    f"{fixprop:.6f}",
    f"{all_mae:.6f}",
    output_dir,
]
print("\t".join(fields))
' "$output_dir/metrics_best.json" "$variant" "$output_dir" >> "$SUMMARY_PATH"
done

echo "Saved notebook-compatible zero-scaling ablation summary to $SUMMARY_PATH"
