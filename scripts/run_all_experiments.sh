#!/usr/bin/env bash
set -euo pipefail
set -o pipefail

MODEL_NAME="${MODEL_NAME:-roberta-base}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-100}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-256}"
DEVICE="${DEVICE:-cuda}"
BEST_METRIC="${BEST_METRIC:-all}"
SEED="${SEED:-42}"
CMCL_LR="${CMCL_LR:-5e-5}"
LR_GRID="${LR_GRID:-1e-5 2e-5 5e-5}"
EXP_ROOT="${EXP_ROOT:-emotion_et_prediction/runs/full_experiment_$(date +%Y%m%d_%H%M%S)}"
RUN_CMCL="${RUN_CMCL:-1}"
RUN_LR_GRID="${RUN_LR_GRID:-1}"
RUN_ZERO_ABLATION="${RUN_ZERO_ABLATION:-1}"
ZERO_ABLATION_LR="${ZERO_ABLATION_LR:-auto}"

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="$(dirname "$PACKAGE_DIR")"
SCRIPT_DIR="$PACKAGE_DIR/scripts"
cd "$WORK_DIR"

mkdir -p "$EXP_ROOT/logs"
LOG_DIR="$EXP_ROOT/logs"

COMMON_ENV=(
  "MODEL_NAME=$MODEL_NAME"
  "PRETRAIN_EPOCHS=$PRETRAIN_EPOCHS"
  "FINETUNE_EPOCHS=$FINETUNE_EPOCHS"
  "BATCH_SIZE=$BATCH_SIZE"
  "MAX_LENGTH=$MAX_LENGTH"
  "DEVICE=$DEVICE"
  "BEST_METRIC=$BEST_METRIC"
  "SEED=$SEED"
)

cat > "$EXP_ROOT/manifest.env" <<EOF
MODEL_NAME=$MODEL_NAME
PRETRAIN_EPOCHS=$PRETRAIN_EPOCHS
FINETUNE_EPOCHS=$FINETUNE_EPOCHS
BATCH_SIZE=$BATCH_SIZE
MAX_LENGTH=$MAX_LENGTH
DEVICE=$DEVICE
BEST_METRIC=$BEST_METRIC
SEED=$SEED
CMCL_LR=$CMCL_LR
LR_GRID=$LR_GRID
EXP_ROOT=$EXP_ROOT
RUN_CMCL=$RUN_CMCL
RUN_LR_GRID=$RUN_LR_GRID
RUN_ZERO_ABLATION=$RUN_ZERO_ABLATION
ZERO_ABLATION_LR=$ZERO_ABLATION_LR
EOF

run_logged() {
  local name="$1"
  shift
  local log_path="$LOG_DIR/${name}.log"

  echo "[$(date -Is)] START $name" | tee "$log_path"
  set +e
  "$@" 2>&1 | tee -a "$log_path"
  local status="${PIPESTATUS[0]}"
  set -e
  if [[ "$status" -ne 0 ]]; then
    echo "[$(date -Is)] FAILED $name status=$status" | tee -a "$log_path"
    exit "$status"
  fi
  echo "[$(date -Is)] DONE $name" | tee -a "$log_path"
}

if [[ "$RUN_CMCL" == "1" ]]; then
  run_logged cmcl_reproduction \
    env "${COMMON_ENV[@]}" LR="$CMCL_LR" RUN_ROOT="$EXP_ROOT/cmcl_reproduction" \
    bash "$SCRIPT_DIR/run_cmcl_reproduction.sh"
fi

if [[ "$RUN_LR_GRID" == "1" ]]; then
  run_logged lr_grid \
    env "${COMMON_ENV[@]}" LR_GRID="$LR_GRID" RUN_ROOT="$EXP_ROOT/lr_grid" \
    bash "$SCRIPT_DIR/run_lr_grid.sh"
fi

if [[ "$ZERO_ABLATION_LR" == "auto" ]]; then
  if [[ ! -f "$EXP_ROOT/lr_grid/summary.tsv" ]]; then
    echo "ZERO_ABLATION_LR=auto requires $EXP_ROOT/lr_grid/summary.tsv" >&2
    exit 1
  fi
  ZERO_ABLATION_LR="$(
    python - "$EXP_ROOT/lr_grid/summary.tsv" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
if not rows:
    raise SystemExit("empty LR grid summary")
best = min(rows, key=lambda row: float(row["selected_score"]))
print(best["lr"])
PY
  )"
fi

echo "$ZERO_ABLATION_LR" > "$EXP_ROOT/selected_zero_ablation_lr.txt"

if [[ "$RUN_ZERO_ABLATION" == "1" ]]; then
  run_logged zero_scaling_ablation \
    env "${COMMON_ENV[@]}" LR="$ZERO_ABLATION_LR" RUN_ROOT="$EXP_ROOT/zero_scaling_ablation" \
    bash "$SCRIPT_DIR/run_zero_scaling_ablation.sh"
fi

python - "$EXP_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
lines = [f"experiment_root={root}"]

cmcl_metrics = root / "cmcl_reproduction" / "metrics_best.json"
if cmcl_metrics.exists():
    payload = json.loads(cmcl_metrics.read_text())
    lines.append(
        "cmcl_reproduction="
        f"epoch:{payload['epoch']},"
        f"metric:{payload['selected_metric']},"
        f"score:{payload['selected_score']:.6f},"
        f"all:{payload['valid_mae']['all']:.6f}"
    )

lr_summary = root / "lr_grid" / "summary.tsv"
if lr_summary.exists():
    with lr_summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if rows:
        best = min(rows, key=lambda row: float(row["selected_score"]))
        lines.append(
            "lr_grid_best="
            f"lr:{best['lr']},"
            f"epoch:{best['best_epoch']},"
            f"score:{float(best['selected_score']):.6f},"
            f"all:{float(best['all']):.6f},"
            f"TRT:{float(best['TRT']):.6f}"
        )

zero_summary = root / "zero_scaling_ablation" / "summary.tsv"
if zero_summary.exists():
    with zero_summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if rows:
        best = min(rows, key=lambda row: float(row["selected_score"]))
        lines.append(
            "zero_scaling_best="
            f"variant:{best['variant']},"
            f"epoch:{best['best_epoch']},"
            f"score:{float(best['selected_score']):.6f},"
            f"all:{float(best['all']):.6f},"
            f"TRT:{float(best['TRT']):.6f}"
        )

(root / "final_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

echo "All experiments finished. Summary: $EXP_ROOT/final_summary.txt"
