#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-emotion_et_prediction/runs/cmcl_reproduction}"
TRAIN_CSV="${TRAIN_CSV:-emotion_et_prediction/data/pretrain_data/train.csv}"
VALID_CSV="${VALID_CSV:-emotion_et_prediction/data/pretrain_data/valid.csv}"
PROVO_CSV="${PROVO_CSV:-emotion_et_prediction/data/pretrain_data/provo.csv}"
MODEL_NAME="${MODEL_NAME:-roberta-base}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-100}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-5e-5}"
MAX_LENGTH="${MAX_LENGTH:-256}"
DEVICE="${DEVICE:-auto}"
BEST_METRIC="${BEST_METRIC:-all}"

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="$(dirname "$PACKAGE_DIR")"
cd "$WORK_DIR"

python -m emotion_et_prediction.emotion_et.train_et \
  --backend hf \
  --model-name "$MODEL_NAME" \
  --pretrain-csv "$PROVO_CSV" \
  --finetune-csv "$TRAIN_CSV" \
  --valid-csv "$VALID_CSV" \
  --output-dir "$RUN_ROOT" \
  --pretrain-epochs "$PRETRAIN_EPOCHS" \
  --finetune-epochs "$FINETUNE_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --max-length "$MAX_LENGTH" \
  --device "$DEVICE" \
  --best-metric "$BEST_METRIC"

python -c '
import json
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
payload = json.loads(metrics_path.read_text())
mae = payload["valid_mae"]
epoch = payload["epoch"]
selected_metric = payload["selected_metric"]
selected_score = payload["selected_score"]
print("CMCL reproduction best metrics")
print(f"epoch={epoch} selected_metric={selected_metric} selected_score={selected_score:.6f}")
for feature in ["nFix", "FFD", "GPT", "TRT", "fixProp", "all"]:
    print(f"{feature}={mae[feature]:.6f}")
' "$RUN_ROOT/metrics_best.json"
