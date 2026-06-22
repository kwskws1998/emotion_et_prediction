# Emotion ET Prediction

This directory contains a modular version of the second ET prediction model from
`1st_and_2nd_ET_result_replication_Really_Final.ipynb`, with IITB V2
preprocessing added for emotion-domain ET fine-tuning.

## Data Roles

- General ET pretrain:
  - CMCL `data/pretrain_data/provo.csv`
  - CMCL `data/pretrain_data/train_and_valid.csv`
  - CMCL `data/pretrain_data/train.csv` and `data/pretrain_data/valid.csv` are also included for exact split inspection or alternate runs.
- Emotion-domain fine-tune:
  - `data/finetune_data/iitb_v2_cmcl_scaled.csv`

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
bash emotion_et_prediction/scripts/download_cmcl_data.sh
```

```bash
python -m emotion_et_prediction.emotion_et.preprocess_iitb \
  --fixation-csv data/iitb_sentiment_gaze_raw/extracted/v2/Eye-tracking_and_SA-II_released_dataset/Fixation_sequence.csv \
  --text-csv data/iitb_sentiment_gaze_raw/extracted/v2/Eye-tracking_and_SA-II_released_dataset/text_and_annorations.csv \
  --output-csv emotion_et_prediction/data/finetune_data/iitb_v2_cmcl_scaled.csv \
  --raw-output-csv emotion_et_prediction/data/finetune_data/iitb_v2_raw_word_features.csv \
  --stats-json emotion_et_prediction/data/finetune_data/iitb_v2_preprocess_stats.json
```

## Train

Use the Hugging Face backend for the real model:

```bash
python -m emotion_et_prediction.emotion_et.train_et \
  --backend hf \
  --model-name roberta-base \
  --pretrain-csv emotion_et_prediction/data/pretrain_data/provo.csv \
  --pretrain-csv emotion_et_prediction/data/pretrain_data/train_and_valid.csv \
  --finetune-csv emotion_et_prediction/data/finetune_data/iitb_v2_cmcl_scaled.csv \
  --output-dir emotion_et_prediction/runs/cmcl_to_iitb_roberta \
  --pretrain-epochs 100 \
  --finetune-epochs 150 \
  --batch-size 16 \
  --lr 5e-5 \
  --best-metric all
```

Training writes `checkpoint_best.pt`, `checkpoint_last.pt`, and `checkpoint.pt`.
`checkpoint.pt` is an alias of the best validation checkpoint. Use
`--best-metric TRT` when TRT MAE should choose the best checkpoint instead of
the mean `all` MAE.

Use the tiny backend only for local smoke tests when `roberta-base` is not cached.

```bash
bash emotion_et_prediction/scripts/smoke_run.sh
```

After downloading `roberta-base`, run the real Hugging Face backend smoke:

```bash
hf download roberta-base
bash emotion_et_prediction/scripts/smoke_run_roberta.sh
```

## Package the Hugging Face Model

After the notebook-compatible IITB run finishes, package the best LR run into a
self-contained Hugging Face upload folder:

```bash
RUN_DIR=emotion_et_prediction/runs/repro_main_20260622_121533 \
OUTPUT_DIR=emotion_et_prediction/hf_emotion_et_iitb_lr5e5 \
ZIP_PATH=emotion_et_prediction/hf_emotion_et_iitb_lr5e5_upload.zip \
LOCAL_FILES_ONLY=1 \
bash emotion_et_prediction/scripts/package_hf_model.sh
```

The package includes:

```text
model.py
config.json
tokenizer.json
tokenizer_config.json
vocab.json
merges.txt
special_tokens_map.json
et_predictor2_iitb_lr5e5_seed42.safetensors
metrics_best.json
lr_grid_summary.tsv
manifest.env
README.md
.gitattributes
```

Upload the contents of `OUTPUT_DIR` to the Hugging Face model repo. If using the
zip, unzip it first and upload the files inside it rather than uploading the zip
as a single model artifact.
