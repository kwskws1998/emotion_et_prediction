"""Datasets and batching for CMCL-style ET prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from .constants import FEATURE_NAMES


def _clean_word(value: object) -> str:
    return str(value).replace("<EOS>", "").strip()


def validate_et_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    missing = {"sentence_id", "word_id", "word", *FEATURE_NAMES} - set(df.columns)
    if missing:
        raise ValueError(f"Missing ET columns: {sorted(missing)}")

    clean = df.copy()
    clean = clean.dropna(subset=["sentence_id", "word_id", "word", *FEATURE_NAMES])
    clean["sentence_id"] = clean["sentence_id"].astype(int)
    clean["word_id"] = clean["word_id"].astype(int)
    clean["word"] = clean["word"].astype(str)
    for feature in FEATURE_NAMES:
        clean[feature] = pd.to_numeric(clean[feature], errors="coerce")
    clean = clean.dropna(subset=FEATURE_NAMES)
    clean = clean[clean["word"].map(_clean_word).ne("")]
    return clean.sort_values(["sentence_id", "word_id"]).reset_index(drop=True)


def renumber_sentences(df: pd.DataFrame, offset: int = 0) -> pd.DataFrame:
    clean = validate_et_dataframe(df)
    sentence_ids = clean["sentence_id"].drop_duplicates().tolist()
    id_map = {old_id: index + offset for index, old_id in enumerate(sentence_ids)}
    clean["sentence_id"] = clean["sentence_id"].map(id_map).astype(int)
    return clean


def load_and_concat_csvs(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    offset = 0
    for path in paths:
        frame = pd.read_csv(path)
        frame = renumber_sentences(frame, offset=offset)
        frames.append(frame)
        offset += frame["sentence_id"].nunique()
    if not frames:
        raise ValueError("No CSV paths were provided.")
    return pd.concat(frames, ignore_index=True)


def split_by_sentence(
    df: pd.DataFrame,
    valid_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = renumber_sentences(df)
    sentence_ids = clean["sentence_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(sentence_ids)
    valid_size = max(1, int(round(len(sentence_ids) * valid_ratio)))
    valid_ids = set(sentence_ids[:valid_size].tolist())
    train_df = clean[~clean["sentence_id"].isin(valid_ids)].reset_index(drop=True)
    valid_df = clean[clean["sentence_id"].isin(valid_ids)].reset_index(drop=True)
    train_df = renumber_sentences(train_df)
    valid_df = renumber_sentences(valid_df)
    return train_df, valid_df


def limit_sentences(df: pd.DataFrame, max_sentences: int | None) -> pd.DataFrame:
    clean = renumber_sentences(df)
    if max_sentences is None:
        return clean
    keep = set(clean["sentence_id"].drop_duplicates().head(max_sentences).tolist())
    return renumber_sentences(clean[clean["sentence_id"].isin(keep)])


@dataclass
class SimpleVocab:
    token_to_id: dict[str, int]

    @classmethod
    def build(cls, dataframes: Iterable[pd.DataFrame], min_freq: int = 1) -> "SimpleVocab":
        counts: dict[str, int] = {}
        for df in dataframes:
            for word in df["word"].map(_clean_word):
                counts[word] = counts.get(word, 0) + 1
        token_to_id = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
        for word, count in sorted(counts.items()):
            if count >= min_freq and word not in token_to_id:
                token_to_id[word] = len(token_to_id)
        return cls(token_to_id=token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    def encode(self, words: list[str], max_length: int) -> list[int]:
        body = [self.token_to_id.get(word, self.token_to_id["<unk>"]) for word in words]
        return [self.token_to_id["<s>"], *body[: max_length - 2], self.token_to_id["</s>"]]


class SimpleETDataset(torch.utils.data.Dataset):
    """Whitespace-token ET dataset used for offline smoke tests."""

    def __init__(self, df: pd.DataFrame, vocab: SimpleVocab, max_length: int = 128):
        self.df = renumber_sentences(df)
        self.vocab = vocab
        self.max_length = max_length
        self.sentence_ids = self.df["sentence_id"].drop_duplicates().tolist()

    def __len__(self) -> int:
        return len(self.sentence_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sentence_id = self.sentence_ids[index]
        rows = self.df[self.df["sentence_id"].eq(sentence_id)].sort_values("word_id")
        words = rows["word"].map(_clean_word).tolist()
        features_np = rows[FEATURE_NAMES].to_numpy(dtype=np.float32)
        input_ids = self.vocab.encode(words, max_length=self.max_length)
        attention_mask = [1] * len(input_ids)
        features = -np.ones((len(input_ids), len(FEATURE_NAMES)), dtype=np.float32)
        n_assign = min(len(words), self.max_length - 2)
        features[1 : 1 + n_assign] = features_np[:n_assign]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "features": torch.tensor(features, dtype=torch.float32),
        }


class HFEyeTrackingDataset(torch.utils.data.Dataset):
    """Hugging Face tokenizer dataset matching the second ET predictor contract."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512):
        self.df = renumber_sentences(df)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sentence_ids = self.df["sentence_id"].drop_duplicates().tolist()

    def __len__(self) -> int:
        return len(self.sentence_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sentence_id = self.sentence_ids[index]
        rows = self.df[self.df["sentence_id"].eq(sentence_id)].sort_values("word_id")
        words = rows["word"].map(_clean_word).tolist()
        features_np = rows[FEATURE_NAMES].to_numpy(dtype=np.float32)

        encoded = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        word_ids = encoded.word_ids()
        features = -np.ones((len(encoded["input_ids"]), len(FEATURE_NAMES)), dtype=np.float32)
        seen: set[int] = set()
        for token_index, word_index in enumerate(word_ids):
            if word_index is None or word_index in seen or word_index >= len(features_np):
                continue
            features[token_index] = features_np[word_index]
            seen.add(word_index)

        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "features": torch.tensor(features, dtype=torch.float32),
        }


def collate_et_batch(batch: list[dict[str, torch.Tensor]], pad_id: int = 0) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    batch_size = len(batch)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    features = -torch.ones((batch_size, max_len, len(FEATURE_NAMES)), dtype=torch.float32)

    for index, item in enumerate(batch):
        seq_len = item["input_ids"].shape[0]
        input_ids[index, :seq_len] = item["input_ids"]
        attention_mask[index, :seq_len] = item["attention_mask"]
        features[index, :seq_len] = item["features"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "features": features,
    }

