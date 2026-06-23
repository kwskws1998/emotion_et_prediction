"""Self-contained Hugging Face inference wrapper for ET Predictor 2 exports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, RobertaConfig, RobertaModel

FEATURE_NAMES = ["nFix", "FFD", "GPT", "TRT", "fixProp"]
TRT_INDEX = 3
DEFAULT_WEIGHT = "et_predictor2_seed42.safetensors"


class RobertaRegressionModel(torch.nn.Module):
    """RoBERTa-base encoder with the ET Predictor 2 token regression head."""

    def __init__(self, config_path: str | Path = "."):
        super().__init__()
        config = RobertaConfig.from_pretrained(config_path)
        self.roberta = RobertaModel(config)
        self.decoder = torch.nn.Linear(config.hidden_size, len(FEATURE_NAMES))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.decoder(hidden)


def load_et_predictor(
    model_dir: str | Path,
    weight_name: str = DEFAULT_WEIGHT,
    device: str | torch.device | None = None,
) -> tuple[RobertaRegressionModel, AutoTokenizer]:
    """Load the exported ET predictor and tokenizer from a local or downloaded HF repo."""

    model_dir = Path(model_dir)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, add_prefix_space=True)
    model = RobertaRegressionModel(config_path=model_dir).to(device)
    state = load_file(str(model_dir / weight_name), device=str(device))
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def predict_word_features(
    text: str,
    model: RobertaRegressionModel,
    tokenizer,
    device: str | torch.device | None = None,
    max_length: int = 512,
) -> tuple[list[str], np.ndarray]:
    """Predict word-level ET features by taking the first RoBERTa subword per word."""

    device = torch.device(device or next(model.parameters()).device)
    words = text.strip().split()
    encoded = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    predictions = model(input_ids=input_ids, attention_mask=attention_mask)
    predictions = predictions.squeeze(0).clamp_min(0.0).cpu().numpy()

    word_ids = encoded.word_ids(batch_index=0)
    output = np.zeros((len(words), len(FEATURE_NAMES)), dtype=np.float32)
    seen: set[int] = set()
    for token_index, word_index in enumerate(word_ids):
        if word_index is None or word_index in seen or word_index >= len(words):
            continue
        output[word_index] = predictions[token_index]
        seen.add(word_index)
    return words, output


@torch.no_grad()
def predict_word_trt(
    text: str,
    model: RobertaRegressionModel,
    tokenizer,
    device: str | torch.device | None = None,
    max_length: int = 512,
) -> tuple[list[str], np.ndarray]:
    """Predict word-level TRT values only."""

    words, features = predict_word_features(text, model, tokenizer, device=device, max_length=max_length)
    return words, features[:, TRT_INDEX]
