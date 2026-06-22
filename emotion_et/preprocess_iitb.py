"""Convert IITB V2 fixation sequences into CMCL-style word-level ET labels."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (
    CMCL_NONZERO_TARGET_STATS,
    CMCL_TARGET_STATS,
    DEFAULT_IITB_V2_DIR,
    FEATURE_NAMES,
)


@dataclass(frozen=True)
class AlignmentReport:
    total_texts: int
    texts_without_fixations: int
    texts_without_mapped_fixations: int
    unmapped_observed_tokens: int
    output_rows: int


def _strip_iitb_word(value: object) -> str:
    return str(value).strip()


def _tokenize_text(text: object) -> list[str]:
    return str(text).strip().split()


def _map_observed_words_to_text(
    observed: pd.DataFrame,
    text_tokens: list[str],
) -> tuple[dict[int, int], int]:
    mapping: dict[int, int] = {}
    cursor = 0
    unmapped = 0

    for word_id, word in zip(observed["Word_ID"].tolist(), observed["word_clean"].tolist()):
        next_cursor = cursor
        while next_cursor < len(text_tokens) and text_tokens[next_cursor] != word:
            next_cursor += 1
        if next_cursor < len(text_tokens):
            mapping[int(word_id)] = next_cursor + 1
            cursor = next_cursor + 1
        else:
            unmapped += 1

    return mapping, unmapped


def _participant_word_features(
    fixations: pd.DataFrame,
    participants: list[str],
    n_words: int,
) -> pd.DataFrame:
    if fixations.empty:
        base = pd.MultiIndex.from_product(
            [participants, range(1, n_words + 1)],
            names=["Participant_ID", "word_pos"],
        ).to_frame(index=False)
        for feature in ["nFix", "FFD", "GPT", "TRT"]:
            base[feature] = 0.0
        return base

    ordered = fixations.reset_index().sort_values(["Participant_ID", "index"])
    participant_word = (
        ordered.groupby(["Participant_ID", "word_pos"])
        .agg(
            nFix=("Fixation_Duration", "size"),
            FFD=("Fixation_Duration", "first"),
            TRT=("Fixation_Duration", "sum"),
        )
        .reset_index()
    )

    gpt_records: list[tuple[str, int, float]] = []
    for participant_id, participant_rows in ordered.groupby("Participant_ID"):
        scanpath = list(
            zip(
                participant_rows["word_pos"].astype(int).tolist(),
                participant_rows["Fixation_Duration"].astype(float).tolist(),
            )
        )
        for word_pos in sorted({pos for pos, _ in scanpath}):
            first_index = next(
                index for index, (pos, _) in enumerate(scanpath) if pos == word_pos
            )
            go_past = 0.0
            for pos, duration in scanpath[first_index:]:
                if pos > word_pos:
                    break
                go_past += duration
            gpt_records.append((str(participant_id), int(word_pos), float(go_past)))

    gpt = pd.DataFrame(gpt_records, columns=["Participant_ID", "word_pos", "GPT"])
    base = pd.MultiIndex.from_product(
        [participants, range(1, n_words + 1)],
        names=["Participant_ID", "word_pos"],
    ).to_frame(index=False)
    features = (
        base.merge(participant_word, on=["Participant_ID", "word_pos"], how="left")
        .merge(gpt, on=["Participant_ID", "word_pos"], how="left")
        .fillna({"nFix": 0.0, "FFD": 0.0, "GPT": 0.0, "TRT": 0.0})
    )
    return features


def build_iitb_v2_word_features(
    fixation_csv: Path,
    text_csv: Path,
    append_eos: bool = True,
) -> tuple[pd.DataFrame, AlignmentReport]:
    fixations = pd.read_csv(fixation_csv)
    texts = pd.read_csv(text_csv)

    required_fixation_cols = {
        "Participant_ID",
        "Text_ID",
        "Word_ID",
        "Word",
        "Fixation_Duration",
    }
    missing_fixation_cols = required_fixation_cols - set(fixations.columns)
    if missing_fixation_cols:
        raise ValueError(f"Missing IITB fixation columns: {sorted(missing_fixation_cols)}")
    if "Text_ID" not in texts.columns or "Text" not in texts.columns:
        raise ValueError("IITB text CSV must contain Text_ID and Text columns.")

    fixations = fixations.copy()
    fixations["word_clean"] = fixations["Word"].map(_strip_iitb_word)
    fixations["Fixation_Duration"] = pd.to_numeric(
        fixations["Fixation_Duration"], errors="coerce"
    ).fillna(0.0)

    participants = sorted(fixations["Participant_ID"].astype(str).unique().tolist())
    rows: list[dict[str, object]] = []
    text_id_to_sentence_id = {
        int(text_id): index
        for index, text_id in enumerate(texts["Text_ID"].astype(int).tolist())
    }

    texts_without_fixations = 0
    texts_without_mapped_fixations = 0
    unmapped_observed_tokens = 0

    for text_row in texts.itertuples(index=False):
        text_id = int(getattr(text_row, "Text_ID"))
        sentence_id = text_id_to_sentence_id[text_id]
        text_tokens = _tokenize_text(getattr(text_row, "Text"))
        text_fixations = fixations[fixations["Text_ID"].astype(int).eq(text_id)].copy()

        if text_fixations.empty:
            texts_without_fixations += 1
            mapped_fixations = text_fixations
        else:
            observed = (
                text_fixations[["Word_ID", "word_clean"]]
                .drop_duplicates()
                .sort_values("Word_ID")
            )
            observed = observed[~observed["word_clean"].str.startswith("Aspect--")]
            mapping, n_unmapped = _map_observed_words_to_text(observed, text_tokens)
            unmapped_observed_tokens += n_unmapped

            if mapping:
                mapped_fixations = text_fixations[
                    text_fixations["Word_ID"].astype(int).isin(mapping)
                ].copy()
                mapped_fixations["word_pos"] = (
                    mapped_fixations["Word_ID"].astype(int).map(mapping).astype(int)
                )
            else:
                texts_without_mapped_fixations += 1
                mapped_fixations = text_fixations.iloc[0:0].copy()
                mapped_fixations["word_pos"] = pd.Series(dtype=int)

        participant_features = _participant_word_features(
            mapped_fixations,
            participants=participants,
            n_words=len(text_tokens),
        )
        word_features = (
            participant_features.groupby("word_pos")
            .agg(
                nFix=("nFix", "mean"),
                FFD=("FFD", "mean"),
                GPT=("GPT", "mean"),
                TRT=("TRT", "mean"),
                fixProp=("nFix", lambda values: float((values > 0).mean())),
            )
            .reset_index()
        )

        for feature_row in word_features.itertuples(index=False):
            word_index = int(getattr(feature_row, "word_pos")) - 1
            word = text_tokens[word_index]
            if append_eos and word_index == len(text_tokens) - 1:
                word = f"{word}<EOS>"
            rows.append(
                {
                    "sentence_id": sentence_id,
                    "word_id": word_index,
                    "word": word,
                    "nFix": float(getattr(feature_row, "nFix")),
                    "FFD": float(getattr(feature_row, "FFD")),
                    "GPT": float(getattr(feature_row, "GPT")),
                    "TRT": float(getattr(feature_row, "TRT")),
                    "fixProp": float(getattr(feature_row, "fixProp")),
                }
            )

    output = pd.DataFrame(rows)
    output = output[["sentence_id", "word_id", "word", *FEATURE_NAMES]]
    report = AlignmentReport(
        total_texts=len(texts),
        texts_without_fixations=texts_without_fixations,
        texts_without_mapped_fixations=texts_without_mapped_fixations,
        unmapped_observed_tokens=unmapped_observed_tokens,
        output_rows=len(output),
    )
    return output, report


def scale_features_to_cmcl(
    raw_df: pd.DataFrame,
    target_stats: dict[str, dict[str, float]] = CMCL_TARGET_STATS,
    preserve_zero_rows: bool = True,
) -> pd.DataFrame:
    scaled = raw_df.copy()
    zero_mask = raw_df[FEATURE_NAMES].eq(0.0).all(axis=1)
    source = raw_df.loc[~zero_mask] if preserve_zero_rows else raw_df
    for feature in FEATURE_NAMES:
        source_mean = float(source[feature].mean())
        source_std = float(source[feature].std())
        if not np.isfinite(source_std) or source_std == 0.0:
            raise ValueError(f"Cannot scale {feature}: source std is {source_std}.")
        target_mean = target_stats[feature]["mean"]
        target_std = target_stats[feature]["std"]
        scaled.loc[source.index, feature] = target_mean + target_std * (
            (source[feature] - source_mean) / source_std
        )
    if preserve_zero_rows:
        scaled.loc[zero_mask, FEATURE_NAMES] = 0.0
    return scaled


def summarize_features(
    raw_df: pd.DataFrame,
    scaled_df: pd.DataFrame,
    target_stats: dict[str, dict[str, float]],
    preserve_zero_rows: bool,
) -> dict[str, object]:
    return {
        "raw": raw_df[FEATURE_NAMES].describe().to_dict(),
        "scaled": scaled_df[FEATURE_NAMES].describe().to_dict(),
        "target_stats": target_stats,
        "preserve_zero_rows": preserve_zero_rows,
    }


def convert_iitb_v2(
    fixation_csv: Path,
    text_csv: Path,
    output_csv: Path,
    raw_output_csv: Path | None = None,
    stats_json: Path | None = None,
    scale: bool = True,
    preserve_zero_rows: bool = True,
    append_eos: bool = True,
) -> pd.DataFrame:
    raw_df, report = build_iitb_v2_word_features(
        fixation_csv=fixation_csv,
        text_csv=text_csv,
        append_eos=append_eos,
    )
    target_stats = CMCL_NONZERO_TARGET_STATS if preserve_zero_rows else CMCL_TARGET_STATS
    output_df = (
        scale_features_to_cmcl(
            raw_df,
            target_stats=target_stats,
            preserve_zero_rows=preserve_zero_rows,
        )
        if scale
        else raw_df.copy()
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)

    if raw_output_csv is not None:
        raw_output_csv.parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(raw_output_csv, index=False)

    if stats_json is not None:
        stats_json.parent.mkdir(parents=True, exist_ok=True)
        stats = summarize_features(
            raw_df,
            output_df,
            target_stats=target_stats,
            preserve_zero_rows=preserve_zero_rows,
        )
        stats["alignment_report"] = report.__dict__
        stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return output_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixation-csv",
        type=Path,
        default=DEFAULT_IITB_V2_DIR / "Fixation_sequence.csv",
    )
    parser.add_argument(
        "--text-csv",
        type=Path,
        default=DEFAULT_IITB_V2_DIR / "text_and_annorations.csv",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--raw-output-csv", type=Path, default=None)
    parser.add_argument("--stats-json", type=Path, default=None)
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--scale-zero-rows", action="store_true")
    parser.add_argument("--no-append-eos", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_df = convert_iitb_v2(
        fixation_csv=args.fixation_csv,
        text_csv=args.text_csv,
        output_csv=args.output_csv,
        raw_output_csv=args.raw_output_csv,
        stats_json=args.stats_json,
        scale=not args.no_scale,
        preserve_zero_rows=not args.scale_zero_rows,
        append_eos=not args.no_append_eos,
    )
    print(f"Saved {len(output_df)} rows to {args.output_csv}")
    print(output_df[FEATURE_NAMES].describe().round(4).to_string())


if __name__ == "__main__":
    main()
