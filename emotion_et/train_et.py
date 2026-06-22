"""Stage-wise training for general-to-emotion ET prediction."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .constants import FEATURE_NAMES
from .data import (
    HFEyeTrackingDataset,
    SimpleETDataset,
    SimpleVocab,
    collate_et_batch,
    limit_sentences,
    load_and_concat_csvs,
    renumber_sentences,
    split_by_sentence,
)
from .models import HFTokenRegressor, TinyTokenRegressor


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


def target_mask(features: torch.Tensor) -> torch.Tensor:
    return ~features.eq(-1.0).all(dim=-1)


def masked_mse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    mask = target_mask(targets)
    if not mask.any():
        return predictions.sum() * 0.0
    return torch.nn.functional.mse_loss(predictions[mask], targets[mask])


def compute_mae(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    values = {
        feature: float(np.abs(predictions[:, index] - targets[:, index]).mean())
        for index, feature in enumerate(FEATURE_NAMES)
    }
    values["all"] = float(np.mean([values[feature] for feature in FEATURE_NAMES]))
    return values


def serialize_args(args: argparse.Namespace) -> dict[str, object]:
    serialized: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        elif isinstance(value, list):
            serialized[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            serialized[key] = value
    return serialized


def metric_score(metrics: dict[str, float], metric_name: str) -> float:
    if metric_name not in metrics:
        raise ValueError(f"Metric '{metric_name}' is not available in {sorted(metrics)}.")
    return float(metrics[metric_name])


def checkpoint_payload(
    model: torch.nn.Module,
    vocab: SimpleVocab | None,
    args: argparse.Namespace,
    stage: str,
    epoch: int,
    valid_mae: dict[str, float],
    selected_metric: str,
) -> dict[str, object]:
    return {
        "state_dict": model.state_dict(),
        "backend": args.backend,
        "model_name": args.model_name,
        "feature_names": FEATURE_NAMES,
        "vocab": vocab.token_to_id if vocab is not None else None,
        "args": serialize_args(args),
        "stage": stage,
        "epoch": epoch,
        "valid_mae": valid_mae,
        "selected_metric": selected_metric,
        "selected_score": metric_score(valid_mae, selected_metric),
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    vocab: SimpleVocab | None,
    args: argparse.Namespace,
    stage: str,
    epoch: int,
    valid_mae: dict[str, float],
) -> None:
    torch.save(
        checkpoint_payload(
            model=model,
            vocab=vocab,
            args=args,
            stage=stage,
            epoch=epoch,
            valid_mae=valid_mae,
            selected_metric=args.best_metric,
        ),
        path,
    )


def write_metrics_json(
    path: Path,
    stage: str,
    epoch: int,
    valid_mae: dict[str, float],
    selected_metric: str,
) -> None:
    payload = {
        "stage": stage,
        "epoch": epoch,
        "selected_metric": selected_metric,
        "selected_score": metric_score(valid_mae, selected_metric),
        "valid_mae": valid_mae,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_loader(
    df: pd.DataFrame,
    backend: str,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    vocab: SimpleVocab | None = None,
    tokenizer=None,
) -> torch.utils.data.DataLoader:
    if backend == "tiny":
        if vocab is None:
            raise ValueError("Tiny backend requires a SimpleVocab.")
        dataset = SimpleETDataset(df, vocab=vocab, max_length=max_length)
        pad_id = vocab.pad_id
    elif backend == "hf":
        if tokenizer is None:
            raise ValueError("HF backend requires a tokenizer.")
        dataset = HFEyeTrackingDataset(df, tokenizer=tokenizer, max_length=max_length)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_et_batch(batch, pad_id=pad_id),
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["features"].to(device)
        predictions = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = masked_mse(predictions, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["features"].to(device)
        predictions = model(input_ids=input_ids, attention_mask=attention_mask)
        mask = target_mask(targets)
        if mask.any():
            all_predictions.append(predictions[mask].clamp_min(0.0).detach().cpu().numpy())
            all_targets.append(targets[mask].detach().cpu().numpy())

    if not all_predictions:
        return {feature: float("nan") for feature in [*FEATURE_NAMES, "all"]}
    return compute_mae(np.vstack(all_predictions), np.vstack(all_targets))


def run_stage(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    label: str,
    valid_loader: torch.utils.data.DataLoader | None = None,
    on_epoch_end=None,
) -> list[dict[str, object]]:
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
    )
    logs: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, valid_loader, device) if valid_loader is not None else None
        log_row = {
            "stage": label,
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_mae": metrics,
        }
        logs.append(log_row)
        if metrics is None:
            print(f"[{label}] epoch {epoch}/{epochs} train_loss={train_loss:.4f}")
        else:
            print(
                f"[{label}] epoch {epoch}/{epochs} "
                f"train_loss={train_loss:.4f} valid_mae={metrics['all']:.4f}"
            )
        if on_epoch_end is not None:
            on_epoch_end(log_row)
    return logs


def build_backend_objects(
    backend: str,
    model_name: str,
    train_frames: list[pd.DataFrame],
    freeze_encoder: bool,
    local_files_only: bool,
):
    if backend == "tiny":
        vocab = SimpleVocab.build(train_frames)
        model = TinyTokenRegressor(vocab_size=len(vocab.token_to_id), pad_id=vocab.pad_id)
        return model, vocab, None
    if backend == "hf":
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                add_prefix_space=True,
                local_files_only=local_files_only,
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            )
        model = HFTokenRegressor(
            model_name=model_name,
            freeze_encoder=freeze_encoder,
            local_files_only=local_files_only,
        )
        return model, None, tokenizer
    raise ValueError(f"Unsupported backend: {backend}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["hf", "tiny"], default="hf")
    parser.add_argument("--model-name", type=str, default="roberta-base")
    parser.add_argument("--pretrain-csv", type=Path, action="append", default=[])
    parser.add_argument("--finetune-csv", type=Path, required=True)
    parser.add_argument("--valid-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrain-epochs", type=int, default=0)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--best-metric", choices=[*FEATURE_NAMES, "all"], default="all")
    parser.add_argument("--max-pretrain-sentences", type=int, default=None)
    parser.add_argument("--max-finetune-train-sentences", type=int, default=None)
    parser.add_argument("--max-valid-sentences", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pretrain_df = None
    if args.pretrain_csv:
        pretrain_df = load_and_concat_csvs(args.pretrain_csv)
        pretrain_df = limit_sentences(pretrain_df, args.max_pretrain_sentences)

    finetune_df = renumber_sentences(pd.read_csv(args.finetune_csv))
    if args.valid_csv is not None:
        finetune_train_df = finetune_df
        valid_df = renumber_sentences(pd.read_csv(args.valid_csv))
    else:
        finetune_train_df, valid_df = split_by_sentence(
            finetune_df,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
        )
    finetune_train_df = limit_sentences(
        finetune_train_df,
        args.max_finetune_train_sentences,
    )
    valid_df = limit_sentences(valid_df, args.max_valid_sentences)

    backend_frames = [finetune_train_df, valid_df]
    if pretrain_df is not None:
        backend_frames.append(pretrain_df)
    model, vocab, tokenizer = build_backend_objects(
        backend=args.backend,
        model_name=args.model_name,
        train_frames=backend_frames,
        freeze_encoder=args.freeze_encoder,
        local_files_only=args.local_files_only,
    )
    model.to(device)

    valid_loader = make_loader(
        valid_df,
        backend=args.backend,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False,
        vocab=vocab,
        tokenizer=tokenizer,
    )
    logs: list[dict[str, object]] = []

    if pretrain_df is not None and args.pretrain_epochs > 0:
        pretrain_loader = make_loader(
            pretrain_df,
            backend=args.backend,
            batch_size=args.batch_size,
            max_length=args.max_length,
            shuffle=True,
            vocab=vocab,
            tokenizer=tokenizer,
        )
        logs.extend(
            run_stage(
                model=model,
                train_loader=pretrain_loader,
                device=device,
                epochs=args.pretrain_epochs,
                lr=args.lr,
                label="pretrain",
                valid_loader=None,
            )
        )

    finetune_loader = make_loader(
        finetune_train_df,
        backend=args.backend,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
        vocab=vocab,
        tokenizer=tokenizer,
    )
    best_state: dict[str, object] = {
        "score": float("inf"),
        "epoch": None,
        "metrics": None,
    }

    def save_best_if_needed(log_row: dict[str, object]) -> None:
        metrics = log_row["valid_mae"]
        if not isinstance(metrics, dict):
            return
        score = metric_score(metrics, args.best_metric)
        if not np.isfinite(score):
            return
        if score < float(best_state["score"]):
            epoch = int(log_row["epoch"])
            best_state["score"] = score
            best_state["epoch"] = epoch
            best_state["metrics"] = metrics
            save_checkpoint(
                args.output_dir / "checkpoint_best.pt",
                model=model,
                vocab=vocab,
                args=args,
                stage="finetune",
                epoch=epoch,
                valid_mae=metrics,
            )
            write_metrics_json(
                args.output_dir / "metrics_best.json",
                stage="finetune",
                epoch=epoch,
                valid_mae=metrics,
                selected_metric=args.best_metric,
            )
            print(f"[finetune] new best {args.best_metric}_mae={score:.4f} at epoch {epoch}")

    logs.extend(
        run_stage(
            model=model,
            train_loader=finetune_loader,
            device=device,
            epochs=args.finetune_epochs,
            lr=args.lr,
            label="finetune",
            valid_loader=valid_loader,
            on_epoch_end=save_best_if_needed,
        )
    )

    last_metrics = evaluate(model, valid_loader, device)
    save_checkpoint(
        args.output_dir / "checkpoint_last.pt",
        model=model,
        vocab=vocab,
        args=args,
        stage="finetune",
        epoch=args.finetune_epochs,
        valid_mae=last_metrics,
    )
    write_metrics_json(
        args.output_dir / "metrics_last.json",
        stage="finetune",
        epoch=args.finetune_epochs,
        valid_mae=last_metrics,
        selected_metric=args.best_metric,
    )

    if best_state["metrics"] is None:
        best_state["score"] = metric_score(last_metrics, args.best_metric)
        best_state["epoch"] = args.finetune_epochs
        best_state["metrics"] = last_metrics
        save_checkpoint(
            args.output_dir / "checkpoint_best.pt",
            model=model,
            vocab=vocab,
            args=args,
            stage="finetune",
            epoch=args.finetune_epochs,
            valid_mae=last_metrics,
        )
        write_metrics_json(
            args.output_dir / "metrics_best.json",
            stage="finetune",
            epoch=args.finetune_epochs,
            valid_mae=last_metrics,
            selected_metric=args.best_metric,
        )

    shutil.copy2(args.output_dir / "checkpoint_best.pt", args.output_dir / "checkpoint.pt")
    shutil.copy2(args.output_dir / "metrics_best.json", args.output_dir / "metrics.json")
    (args.output_dir / "train_log.json").write_text(
        json.dumps(logs, indent=2),
        encoding="utf-8",
    )
    print(f"Last valid MAE: {last_metrics}")
    print(
        f"Best valid MAE by {args.best_metric}: "
        f"{float(best_state['score']):.4f} at epoch {best_state['epoch']}"
    )
    print(f"Saved best checkpoint to {args.output_dir / 'checkpoint_best.pt'}")
    print(f"Saved last checkpoint to {args.output_dir / 'checkpoint_last.pt'}")


if __name__ == "__main__":
    main()
