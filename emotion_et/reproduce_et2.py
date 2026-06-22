"""Notebook-compatible reproduction for the second ET predictor."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers

from .constants import FEATURE_NAMES

try:
    from safetensors.torch import save_file as save_safetensors

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


class EyeTrackingCSV(torch.utils.data.Dataset):
    """Dataset behavior copied from the reproduction notebook."""

    def __init__(self, df: pd.DataFrame, model_name: str = "roberta-base", max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.df = df.copy()
        self.df = self.df.dropna(subset=["word"])
        self.df = self.df.dropna(subset=FEATURE_NAMES)
        self.df = self.df[self.df.word.str.strip() != ""]
        self.df.sentence_id = self.df.sentence_id - self.df.sentence_id.min()
        self.num_sentences = int(self.df.sentence_id.max() + 1)
        assert self.num_sentences == self.df.sentence_id.nunique()

        self.texts: list[list[str]] = []
        for sentence_id in range(self.num_sentences):
            rows = self.df[self.df.sentence_id == sentence_id]
            text = rows.word.tolist()
            text[-1] = text[-1].replace("<EOS>", "").strip()
            text = [word for word in text if word]
            self.texts.append(text)

        self.tokenizer = transformers.RobertaTokenizer.from_pretrained(
            model_name,
            add_prefix_space=True,
        )
        self.ids = self.tokenizer(
            self.texts,
            padding=True,
            is_split_into_words=True,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
        )

    def __len__(self) -> int:
        return self.num_sentences

    def __getitem__(self, index: int):
        input_ids = self.ids["input_ids"][index]
        attention_mask = self.ids["attention_mask"][index]
        input_tokens = [self.tokenizer.convert_ids_to_tokens(token_id) for token_id in input_ids]

        is_first_subword = [token[0] == "Ġ" for token in input_tokens]
        features = -torch.ones((len(input_ids), len(FEATURE_NAMES)))
        sentence_rows = self.df[self.df.sentence_id == index][FEATURE_NAMES].to_numpy()

        first_subword_positions = [position for position, value in enumerate(is_first_subword) if value]
        n_assign = min(len(first_subword_positions), len(sentence_rows))
        for row_index in range(n_assign):
            features[first_subword_positions[row_index]] = torch.tensor(
                sentence_rows[row_index],
                dtype=torch.float32,
            )

        return (
            input_tokens,
            torch.LongTensor(input_ids),
            torch.LongTensor(attention_mask),
            features,
        )


class RobertaRegressionModel(torch.nn.Module):
    """RoBERTa token regressor matching the notebook architecture."""

    def __init__(self, model_name: str = "roberta-base"):
        super().__init__()
        self.roberta = transformers.RobertaModel.from_pretrained(model_name)
        embed_size = 1024 if "large" in model_name else 768
        self.decoder = torch.nn.Linear(embed_size, len(FEATURE_NAMES))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        predict_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.roberta(input_ids, attention_mask=attention_mask).last_hidden_state
        predictions = self.decoder(hidden)
        mask = predict_mask.eq(0).unsqueeze(-1).expand_as(predictions).to(predictions.device)
        return predictions.masked_fill(mask, -1.0)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_mae(predict_df: pd.DataFrame, truth_df: pd.DataFrame) -> dict[str, float]:
    maes = {}
    for feature in FEATURE_NAMES:
        maes[feature] = float(np.abs(predict_df[feature].values - truth_df[feature].values).mean())
    maes["all"] = float(np.mean(list(maes.values())))
    return maes


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    mse_loss: torch.nn.Module,
    device: torch.device,
) -> float:
    total_loss = 0.0
    n_batches = 0
    for _, input_ids, attention_mask, targets in loader:
        optimizer.zero_grad()
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)
        predict_mask = torch.sum(targets, dim=2) >= 0
        predictions = model(input_ids, attention_mask, predict_mask)
        loss = mse_loss(targets, predictions)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def predict(
    model: torch.nn.Module,
    valid_df: pd.DataFrame,
    device: torch.device,
    model_name: str = "roberta-base",
    batch_size: int = 16,
    max_length: int = 512,
) -> pd.DataFrame:
    valid_data = EyeTrackingCSV(valid_df, model_name=model_name, max_length=max_length)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)

    predict_df = valid_df.copy()
    predict_df[FEATURE_NAMES] = 9999.0

    predictions = []
    model.eval()
    for _, input_ids, attention_mask, targets in valid_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        predict_mask = torch.sum(targets, dim=2) >= 0
        with torch.no_grad():
            predicted = model(input_ids, attention_mask, predict_mask).cpu().numpy()

        predict_mask_np = predict_mask.numpy()
        for batch_index in range(input_ids.shape[0]):
            for row_index in range(input_ids.shape[1]):
                if predict_mask_np[batch_index, row_index]:
                    token_pred = np.clip(predicted[batch_index, row_index], 0, None)
                    predictions.append(token_pred)

    predict_df[FEATURE_NAMES] = np.vstack(predictions)
    return predict_df


def train_stage(
    model: torch.nn.Module,
    train_df: pd.DataFrame,
    num_epochs: int,
    lr: float,
    batch_size: int,
    model_name: str,
    max_length: int,
    device: torch.device,
    stage_label: str,
    valid_df: pd.DataFrame | None = None,
    track_best: bool = False,
) -> tuple[torch.nn.Module, dict[str, list[object]], float | None]:
    train_data = EyeTrackingCSV(train_df, model_name=model_name, max_length=max_length)
    random.seed(12345)
    torch.manual_seed(12345)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    mse_loss = torch.nn.MSELoss()

    best_mae = float("inf")
    best_state = None
    log: dict[str, list[object]] = {
        "epoch": [],
        "train_loss": [],
        "val_mae": [],
        "val_mae_per_feat": [],
    }

    model.train()
    for epoch in range(1, num_epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, mse_loss, device)
        log["epoch"].append(epoch)
        log["train_loss"].append(loss)

        if valid_df is not None:
            pred_df = predict(
                model,
                valid_df,
                device=device,
                model_name=model_name,
                batch_size=batch_size,
                max_length=max_length,
            )
            maes = compute_mae(pred_df, valid_df)
            log["val_mae"].append(maes["all"])
            log["val_mae_per_feat"].append({key: value for key, value in maes.items() if key != "all"})
            if epoch % 10 == 0:
                print(
                    f"[{stage_label}] Epoch {epoch}/{num_epochs} | "
                    f"train_loss={loss:.4f} | val_MAE={maes['all']:.4f}",
                    flush=True,
                )
            if track_best and maes["all"] < best_mae:
                best_mae = maes["all"]
                best_state = {key: value.clone() for key, value in model.state_dict().items()}
        else:
            log["val_mae"].append(None)
            log["val_mae_per_feat"].append(None)
            if epoch % 10 == 0:
                print(f"[{stage_label}] Epoch {epoch}/{num_epochs} | train_loss={loss:.4f}", flush=True)

    if track_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"[{stage_label}] Best val MAE: {best_mae:.4f}", flush=True)

    return model, log, best_mae if np.isfinite(best_mae) else None


def write_metrics(path: Path, epoch: int, valid_mae: dict[str, float]) -> None:
    payload = {
        "stage": "finetune",
        "epoch": epoch,
        "selected_metric": "all",
        "selected_score": valid_mae["all"],
        "valid_mae": valid_mae,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_one_seed(
    seed: int,
    provo_df: pd.DataFrame,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    set_seed(seed)
    print(f"\n{'=' * 60}", flush=True)
    print(f"Training seed {seed}", flush=True)
    print(f"{'=' * 60}", flush=True)

    model = RobertaRegressionModel(model_name=args.model_name).to(device)
    log_provo: dict[str, list[object]] | None = None
    if args.provo_epochs > 0:
        print(f"\nStage 1: Provo pretraining ({args.provo_epochs} epochs)", flush=True)
        model, log_provo, _ = train_stage(
            model,
            provo_df,
            num_epochs=args.provo_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            model_name=args.model_name,
            max_length=args.max_length,
            device=device,
            stage_label="Provo",
            valid_df=None,
            track_best=False,
        )

    print(f"\nStage 2: ZuCo task finetuning ({args.task_epochs} epochs)", flush=True)
    model, log_zuco, best_mae = train_stage(
        model,
        train_df,
        num_epochs=args.task_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        model_name=args.model_name,
        max_length=args.max_length,
        device=device,
        stage_label="ZuCo",
        valid_df=valid_df,
        track_best=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / f"et_predictor2_seed{seed}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved (.pt): {ckpt_path}", flush=True)

    if HAS_SAFETENSORS:
        safetensors_path = args.output_dir / f"et_predictor2_seed{seed}.safetensors"
        save_safetensors(model.state_dict(), safetensors_path)
        print(f"Saved (.safetensors): {safetensors_path}", flush=True)

    log_path = args.output_dir / f"log_seed{seed}.json"
    log_path.write_text(
        json.dumps({"provo": log_provo if args.provo_epochs > 0 else {}, "zuco": log_zuco}, indent=2),
        encoding="utf-8",
    )

    pred_df = predict(
        model,
        valid_df,
        device=device,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    maes = compute_mae(pred_df, valid_df)
    print(f"Final val MAE (seed {seed}): {maes}", flush=True)
    pred_df.to_csv(args.output_dir / f"predictions_seed{seed}.csv", index=False)

    best_epoch = int(log_zuco["epoch"][int(np.argmin(log_zuco["val_mae"]))])
    write_metrics(args.output_dir / "metrics_best.json", epoch=best_epoch, valid_mae=maes)
    torch.save(model.state_dict(), args.output_dir / "checkpoint_best.pt")
    write_metrics(args.output_dir / "metrics.json", epoch=best_epoch, valid_mae=maes)
    torch.save(model.state_dict(), args.output_dir / "checkpoint.pt")

    if best_mae is not None:
        print(f"Best val MAE tracked during training: {best_mae:.4f}", flush=True)

    return maes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--provo-csv", type=Path, default=Path("./provo.csv"))
    parser.add_argument("--train-csv", type=Path, default=Path("./train.csv"))
    parser.add_argument("--valid-csv", type=Path, default=Path("./valid.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--model-name", type=str, default="roberta-base")
    parser.add_argument("--provo-epochs", type=int, default=100)
    parser.add_argument("--task-epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(seed.strip()) for seed in args.seeds.split(",")]
    device = get_device(args.device)

    provo_df = pd.read_csv(args.provo_csv)
    train_df = pd.read_csv(args.train_csv)
    valid_df = pd.read_csv(args.valid_csv)

    print(f"Train: {len(train_df)} words, Valid: {len(valid_df)} words, Provo: {len(provo_df)} words", flush=True)
    print(f"Seeds: {seeds}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Max length: {args.max_length}", flush=True)

    all_maes = {}
    for seed in seeds:
        all_maes[seed] = train_one_seed(seed, provo_df, train_df, valid_df, args, device)

    print(f"\n{'=' * 60}", flush=True)
    print("Summary across seeds:", flush=True)
    for seed, maes in all_maes.items():
        print(f"  seed {seed}: MAE={maes['all']:.4f}", flush=True)

    if len(seeds) > 1:
        all_values = [metrics["all"] for metrics in all_maes.values()]
        print(f"  mean={np.mean(all_values):.4f}, std={np.std(all_values):.4f}", flush=True)
        summary_path = args.output_dir / "training_summary.json"
        summary_path.write_text(
            json.dumps({str(seed): metrics for seed, metrics in all_maes.items()}, indent=2),
            encoding="utf-8",
        )
        print(f"Summary saved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
