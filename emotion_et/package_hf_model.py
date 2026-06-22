"""Build a Hugging Face upload bundle for the emotion-specific ET predictor."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path

from transformers import AutoTokenizer, RobertaConfig

DEFAULT_RUN_DIR = Path("emotion_et_prediction/runs/repro_main_20260622_121533")
DEFAULT_OUTPUT_DIR = Path("emotion_et_prediction/hf_emotion_et_iitb_lr5e5")
DEFAULT_ZIP_PATH = Path("emotion_et_prediction/hf_emotion_et_iitb_lr5e5_upload.zip")
DEFAULT_WEIGHT_NAME = "et_predictor2_iitb_lr5e5_seed42.safetensors"
SOURCE_WEIGHT_NAME = "et_predictor2_seed42.safetensors"
FEATURE_NAMES = ["nFix", "FFD", "GPT", "TRT", "fixProp"]


def write_compat_tokenizer_files(output_dir: Path, model_name: str, local_files_only: bool) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_prefix_space=True,
        local_files_only=local_files_only,
    )
    tokenizer.save_pretrained(output_dir)

    config = RobertaConfig.from_pretrained(model_name, local_files_only=local_files_only)
    config.save_pretrained(output_dir)

    tokenizer_payload = json.loads((output_dir / "tokenizer.json").read_text())
    tokenizer_model = tokenizer_payload["model"]

    (output_dir / "vocab.json").write_text(
        json.dumps(tokenizer_model["vocab"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (output_dir / "merges.txt").open("w", encoding="utf-8") as handle:
        handle.write("#version: 0.2\n")
        for merge in tokenizer_model["merges"]:
            handle.write((" ".join(merge) if isinstance(merge, list) else str(merge)) + "\n")

    special_tokens = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "sep_token": "</s>",
        "pad_token": "<pad>",
        "cls_token": "<s>",
        "mask_token": "<mask>",
    }
    (output_dir / "special_tokens_map.json").write_text(
        json.dumps(special_tokens, indent=2),
        encoding="utf-8",
    )

    tokenizer_config_path = output_dir / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text())
    tokenizer_config["add_prefix_space"] = True
    tokenizer_config["model_max_length"] = 512
    tokenizer_config_path.write_text(json.dumps(tokenizer_config, indent=2), encoding="utf-8")


def read_lr_summary(summary_path: Path) -> dict[str, str]:
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"No LR summary rows found in {summary_path}")
    return min(rows, key=lambda row: float(row["selected_score"]))


def write_readme(output_dir: Path, best_row: dict[str, str], weight_name: str) -> None:
    readme = f"""---
library_name: transformers
base_model: roberta-base
tags:
- eye-tracking
- gaze
- roberta
- regression
- iitb
---

# Emotion-specific ET Predictor 2: CMCL -> IITB

This repository contains a RoBERTa-base token-level eye-tracking feature predictor.

## Model

- Encoder: `roberta-base`
- Head: linear regression layer, hidden size 768 -> 5
- Feature order: `[nFix, FFD, GPT, TRT, fixProp]`
- TRT index: `3`
- Weight file: `{weight_name}`

## Training

- Pretraining data: Provo + ZuCo train_and_valid from the CMCL-style ET Predictor 2 setup
- Fine-tuning data: IITB V2 sentiment gaze, CMCL-scaled
- Seed: 42
- Best LR: {best_row["lr"]}
- Max length: 512

## Validation MAE on IITB split

| Feature | MAE |
|---|---:|
| nFix | {float(best_row["nFix"]):.6f} |
| FFD | {float(best_row["FFD"]):.6f} |
| GPT | {float(best_row["GPT"]):.6f} |
| TRT | {float(best_row["TRT"]):.6f} |
| fixProp | {float(best_row["fixProp"]):.6f} |
| all | {float(best_row["all"]):.6f} |

## Usage

```python
from huggingface_hub import snapshot_download
from model import load_et_predictor, predict_word_trt

model_dir = snapshot_download("YOUR_NAMESPACE/YOUR_REPO")
model, tokenizer = load_et_predictor(model_dir)
words, trt = predict_word_trt("This sentence is emotionally intense.", model, tokenizer)
```

The exported `model.py` is a self-contained inference wrapper matching the training architecture.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def write_gitattributes(output_dir: Path) -> None:
    (output_dir / ".gitattributes").write_text(
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
        "*.bin filter=lfs diff=lfs merge=lfs -text\n"
        "*.pt filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )


def copy_required_files(run_dir: Path, output_dir: Path, weight_name: str) -> dict[str, str]:
    lr_summary_path = run_dir / "lr_grid" / "summary.tsv"
    best_row = read_lr_summary(lr_summary_path)
    summary_output_dir = Path(best_row["output_dir"])
    safe_lr = best_row["lr"].replace(".", "p").replace("-", "_")
    candidates = [
        summary_output_dir,
        Path.cwd() / summary_output_dir,
        run_dir / "lr_grid" / f"lr_{safe_lr}",
    ]
    best_dir = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])

    source_weight = best_dir / SOURCE_WEIGHT_NAME
    if not source_weight.exists():
        raise FileNotFoundError(f"Missing source weight: {source_weight}")

    shutil.copy2(source_weight, output_dir / weight_name)
    shutil.copy2(best_dir / "metrics_best.json", output_dir / "metrics_best.json")
    shutil.copy2(lr_summary_path, output_dir / "lr_grid_summary.tsv")
    shutil.copy2(run_dir / "manifest.env", output_dir / "manifest.env")
    return best_row


def copy_model_wrapper(output_dir: Path) -> None:
    source = Path(__file__).with_name("hf_model.py")
    shutil.copy2(source, output_dir / "model.py")


def zip_directory(output_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir))


def package_hf_model(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_compat_tokenizer_files(
        output_dir=output_dir,
        model_name=args.model_name,
        local_files_only=args.local_files_only,
    )
    best_row = copy_required_files(args.run_dir, output_dir, args.weight_name)
    copy_model_wrapper(output_dir)
    write_readme(output_dir, best_row, args.weight_name)
    write_gitattributes(output_dir)

    for path in output_dir.iterdir():
        if path.is_file():
            path.chmod(0o644)

    if args.zip_path is not None:
        zip_directory(output_dir, args.zip_path)
        print(f"Wrote zip: {args.zip_path}")
    print(f"Wrote HF bundle dir: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH)
    parser.add_argument("--weight-name", type=str, default=DEFAULT_WEIGHT_NAME)
    parser.add_argument("--model-name", type=str, default="roberta-base")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    package_hf_model(parse_args())


if __name__ == "__main__":
    main()
