# Cloud Server Training and Hub Upload

This guide assumes training runs on a normal cloud GPU server, then the same
server uploads the finished checkpoint to Hugging Face Hub.

Run commands from the parent directory of `emotion_et_prediction`.

## 1. Push Source Code to GitHub

From the local Desktop copy:

```bash
cd ~/Desktop/emotion_et_prediction
git init
git add .
git commit -m "Add emotion ET prediction training code"
git branch -M main
git remote add origin https://github.com/<github-user>/emotion_et_prediction.git
git push -u origin main
```

Do not commit `runs/` or raw datasets. This repository is configured to include
the small processed files needed to start training immediately:

```text
data/pretrain_data/provo.csv
data/pretrain_data/train_and_valid.csv
data/pretrain_data/train.csv
data/pretrain_data/valid.csv
data/finetune_data/iitb_sa1_sa2_cmcl_scaled.csv
data/finetune_data/iitb_sa1_sa2_preprocess_stats.json
data/finetune_data/iitb_v2_cmcl_scaled.csv
data/finetune_data/iitb_v2_raw_word_features.csv
data/finetune_data/iitb_v2_preprocess_stats.json
```

## 2. Prepare the Cloud GPU Server

On the cloud server:

```bash
git clone https://github.com/<github-user>/emotion_et_prediction.git
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r emotion_et_prediction/requirements.txt
```

If the server image already has PyTorch with CUDA installed, keep that version
and only install the remaining requirements if needed.

Check CUDA:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

## 3. Login to Hugging Face on the Server

Use a write token from https://huggingface.co/settings/tokens.

```bash
hf auth login
hf auth whoami
```

If the server is non-interactive:

```bash
export HF_TOKEN=<your-write-token>
hf auth whoami
```

Do not commit the token.

## 4. Prepare Data on the Server

CMCL Provo/ZuCo-style pretraining CSVs should already be in:

```text
emotion_et_prediction/data/pretrain_data/
```

The final processed IITB fine-tuning CSV should already be here:

```text
emotion_et_prediction/data/finetune_data/iitb_sa1_sa2_cmcl_scaled.csv
```

If any file is missing, copy it from local or download it again. For example:

```bash
mkdir -p emotion_et_prediction/data/finetune_data
scp <local-path>/iitb_sa1_sa2_cmcl_scaled.csv <server>:~/emotion_et_prediction/data/finetune_data/
```

or download it from a private Hugging Face dataset repo:

```bash
mkdir -p emotion_et_prediction/data/finetune_data
hf download <hf-user>/iitb-sa1-sa2-emotion-et-cmcl \
  iitb_sa1_sa2_cmcl_scaled.csv \
  --type dataset \
  --local-dir emotion_et_prediction/data/finetune_data
```

## 5. Create the Hugging Face Model Repo

Run this on the server once:

```bash
hf repos create <hf-user>/emotion-et-predictor-roberta \
  --type model \
  --private \
  --exist-ok
```

Remove `--private` if the model should be public.

## 6. Train and Upload From the Server

This runs CMCL pretraining first, then augmented IITB emotion-domain
fine-tuning, packages the Hugging Face bundle, then uploads the bundle to the
model repo.

```bash
PRETRAIN_EPOCHS=100 \
FINETUNE_EPOCHS=150 \
BATCH_SIZE=16 \
REPRO_MAX_LENGTH=512 \
DEVICE=cuda \
HF_MODEL_REPO=<hf-user>/emotion-et-predictor-roberta \
bash emotion_et_prediction/scripts/train_package_upload_augmented.sh
```

The run saves `et_predictor2_seed42.safetensors`, `metrics_best.json`, and the
packaged Hugging Face folder under `emotion_et_prediction/hf_emotion_et_augmented`.

For a cheap server smoke run:

```bash
PRETRAIN_EPOCHS=1 \
FINETUNE_EPOCHS=1 \
BATCH_SIZE=4 \
REPRO_MAX_LENGTH=128 \
DEVICE=cuda \
OUTPUT_DIR=emotion_et_prediction/runs/server_smoke \
bash emotion_et_prediction/scripts/train_repro_cmcl_to_iitb.sh
```

To package and upload an already finished run manually:

```bash
RUN_DIR=emotion_et_prediction/runs/repro_cmcl_to_iitb_augmented_roberta \
HF_MODEL_REPO=<hf-user>/emotion-et-predictor-roberta \
bash emotion_et_prediction/scripts/package_hf_model.sh
```
