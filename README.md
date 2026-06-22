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
  --finetune-epochs 20 \
  --batch-size 16 \
  --lr 5e-5
```

Use the tiny backend only for local smoke tests when `roberta-base` is not cached.

```bash
bash emotion_et_prediction/scripts/smoke_run.sh
```

After downloading `roberta-base`, run the real Hugging Face backend smoke:

```bash
hf download roberta-base
bash emotion_et_prediction/scripts/smoke_run_roberta.sh
```
