# Emotion ET Prediction

This directory contains a modular version of the second ET prediction model from
`1st_and_2nd_ET_result_replication_Really_Final.ipynb`, with IITB V2
preprocessing added for emotion-domain ET fine-tuning.

Run commands from this directory unless noted otherwise.

## Data Roles

- General ET pretrain:
  - CMCL `data/pretrain_data/provo.csv`
  - CMCL `data/pretrain_data/train_and_valid.csv`
  - CMCL `data/pretrain_data/train.csv` and `data/pretrain_data/valid.csv` are also included for exact split inspection or alternate runs.
- Emotion-domain fine-tune:
  - `data/finetune_data/iitb_v2_cmcl_scaled.csv`
  - `data/finetune_data/iitb_sa1_sa2_cmcl_scaled.csv` is the default final fine-tune file. It adds non-duplicate IITB/CFILT SA-I snippet examples to IITB V2.

The model target contract is:

```text
sentence_id,word_id,word,nFix,FFD,GPT,TRT,fixProp
```

IITB V2 raw fixation durations are event-level milliseconds, so the converter
first aggregates participant-word features, then scales the resulting features to
the CMCL/ZuCo target statistics used by the CMCL Provo preprocessing notebook.
Words with no fixation from any participant are kept as all-zero rows by default.
Pass `--scale-zero-rows` only when you explicitly want ProvoProcess-style scaling
to move zero rows onto the z-scored scale.

## Convert IITB V2

If the CSVs are not already present, download CMCL pretraining CSVs into the
workspace:

```bash
bash scripts/download_cmcl_data.sh
```

```bash
python -m emotion_et.preprocess_iitb \
  --fixation-csv data/iitb_sentiment_gaze_raw/extracted/v2/Eye-tracking_and_SA-II_released_dataset/Fixation_sequence.csv \
  --text-csv data/iitb_sentiment_gaze_raw/extracted/v2/Eye-tracking_and_SA-II_released_dataset/text_and_annorations.csv \
  --output-csv data/finetune_data/iitb_v2_cmcl_scaled.csv \
  --raw-output-csv data/finetune_data/iitb_v2_raw_word_features.csv \
  --stats-json data/finetune_data/iitb_v2_preprocess_stats.json
```

## Convert IITB/CFILT SA-I

Keep the CFILT raw archive outside tracked files unless you have permission to
redistribute it. The converter reads `Eye-Tracking-Sentiment-Analysis.tar.gz`,
keeps binary sentiment snippets by default, preserves all-zero rows, scales to
CMCL nonzero target statistics, and can write a duplicate-filtered SA-I+SA-II
fine-tune CSV.

```bash
python -m emotion_et.preprocess_iitb_sa1 \
  --sa1-archive /path/to/Eye-Tracking-Sentiment-Analysis.tar.gz \
  --output-csv data/finetune_data/iitb_sa1_snippet_cmcl_scaled.csv \
  --raw-output-csv data/finetune_data/iitb_sa1_snippet_raw_word_features.csv \
  --stats-json data/finetune_data/iitb_sa1_snippet_preprocess_stats.json \
  --combined-with-csv data/finetune_data/iitb_v2_cmcl_scaled.csv \
  --combined-output-csv data/finetune_data/iitb_sa1_sa2_cmcl_scaled.csv \
  --combined-stats-json data/finetune_data/iitb_sa1_sa2_preprocess_stats.json
```

## Train

Use the Hugging Face backend for the real model:

```bash
python -m emotion_et.train_et \
  --backend hf \
  --model-name roberta-base \
  --pretrain-csv data/pretrain_data/provo.csv \
  --pretrain-csv data/pretrain_data/train_and_valid.csv \
  --finetune-csv data/finetune_data/iitb_sa1_sa2_cmcl_scaled.csv \
  --output-dir runs/cmcl_to_iitb_augmented_roberta \
  --pretrain-epochs 100 \
  --finetune-epochs 150 \
  --batch-size 16 \
  --lr 5e-5 \
  --best-metric all
```

For the notebook-compatible ET Predictor 2 export path, this is the default final training command:

```bash
bash scripts/train_repro_cmcl_to_iitb.sh
```

That script uses `data/finetune_data/iitb_sa1_sa2_cmcl_scaled.csv` by default.
The older SA-II-only file remains available for ablations by passing
`IITB_CSV=emotion_et_prediction/data/finetune_data/iitb_v2_cmcl_scaled.csv`.

The direct `train_et` command writes `checkpoint_best.pt`, `checkpoint_last.pt`,
and `checkpoint.pt`. `checkpoint.pt` is an alias of the best validation
checkpoint. Use `--best-metric TRT` when TRT MAE should choose the best
checkpoint instead of the mean `all` MAE.

The notebook-compatible script writes `et_predictor2_seed42.safetensors`,
`metrics_best.json`, predictions, and logs. Use this path for the Hugging Face
bundle below.

Use the tiny backend only for local smoke tests when `roberta-base` is not cached.

```bash
bash scripts/smoke_run.sh
```

After downloading `roberta-base`, run the real Hugging Face backend smoke:

```bash
hf download roberta-base
bash scripts/smoke_run_roberta.sh
```

## Package the Hugging Face Model

After the notebook-compatible augmented run finishes, package it into a
self-contained Hugging Face upload folder:

```bash
LOCAL_FILES_ONLY=1 \
bash scripts/package_hf_model.sh
```

For training, packaging, and uploading in one command, first authenticate with
`hf auth login`, then set the target repository:

```bash
HF_MODEL_REPO=<hf-user>/<repo-name> \
bash scripts/train_package_upload_augmented.sh
```

The package includes:

```text
.gitattributes
README.md
model.py
config.json
tokenizer.json
tokenizer_config.json
vocab.json
merges.txt
special_tokens_map.json
et_predictor2_seed42.safetensors
metrics_best.json
lr_grid_summary.tsv
manifest.env
```

Upload the contents of `OUTPUT_DIR` to the Hugging Face model repo. If using the
zip, unzip it first and upload the files inside it rather than uploading the zip
as a single model artifact.
