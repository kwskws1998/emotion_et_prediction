"""Token-level regression models for ET feature prediction."""

from __future__ import annotations

import torch

from .constants import FEATURE_NAMES


class TinyTokenRegressor(torch.nn.Module):
    """Small local backend for smoke tests without external model downloads."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size, padding_idx=pad_id)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = torch.nn.Linear(hidden_size, len(FEATURE_NAMES))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        padding_mask = attention_mask.eq(0)
        hidden = self.encoder(embedded, src_key_padding_mask=padding_mask)
        return self.decoder(hidden)


class HFTokenRegressor(torch.nn.Module):
    """RoBERTa-style Hugging Face encoder with a token regression head."""

    def __init__(
        self,
        model_name: str = "roberta-base",
        freeze_encoder: bool = False,
        local_files_only: bool = False,
    ):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        hidden_size = self.encoder.config.hidden_size
        self.decoder = torch.nn.Linear(hidden_size, len(FEATURE_NAMES))
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.decoder(hidden)
