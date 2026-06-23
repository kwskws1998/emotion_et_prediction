"""Collect best validation metrics from emotion ET experiment directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .constants import FEATURE_NAMES


def read_last_stage_loss(log_path: Path, stage: str) -> str:
    if not log_path.exists():
        return ""
    rows = json.loads(log_path.read_text(encoding="utf-8"))
    stage_rows = [row for row in rows if row.get("stage") == stage]
    if not stage_rows:
        return ""
    return f"{float(stage_rows[-1]['train_loss']):.6f}"


def collect_run_rows(runs_dir: Path, pattern: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run_dir in sorted(path for path in runs_dir.glob(pattern) if path.is_dir()):
        metrics_path = run_dir / "metrics_best.json"
        if not metrics_path.exists():
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        valid_mae = payload.get("valid_mae", {})
        row = {
            "run_dir": str(run_dir),
            "epoch": str(payload.get("epoch", "")),
            "selected_metric": str(payload.get("selected_metric", "")),
            "selected_score": f"{float(payload.get('selected_score', float('nan'))):.6f}",
            "pretrain_last_loss": read_last_stage_loss(run_dir / "train_log.json", "pretrain"),
            "finetune_last_loss": read_last_stage_loss(run_dir / "train_log.json", "finetune"),
        }
        for feature in [*FEATURE_NAMES, "all"]:
            value = valid_mae.get(feature)
            row[feature] = "" if value is None else f"{float(value):.6f}"
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--pattern", type=str, default="emotion_et_aug_*")
    parser.add_argument("--output", type=Path, default=Path("runs/summary.tsv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_run_rows(args.runs_dir, args.pattern)
    fieldnames = [
        "run_dir",
        "epoch",
        "selected_metric",
        "selected_score",
        *FEATURE_NAMES,
        "all",
        "pretrain_last_loss",
        "finetune_last_loss",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
