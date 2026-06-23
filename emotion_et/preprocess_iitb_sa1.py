"""Convert IITB/CFILT SA-I Translog XML into CMCL-style word-level ET labels."""

from __future__ import annotations

import argparse
import json
import re
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import CMCL_NONZERO_TARGET_STATS, CMCL_TARGET_STATS, FEATURE_NAMES
from .data import renumber_sentences
from .preprocess_iitb import scale_features_to_cmcl, summarize_features


DEFAULT_SA1_ARCHIVE = Path(
    "data/iitb_sentiment_gaze_raw/Eye-Tracking-Sentiment-Analysis.tar.gz"
)


@dataclass(frozen=True)
class SA1Report:
    total_annotations: int
    kept_annotations: int
    participants: int
    xml_files_seen: int
    xml_files_used: int
    xml_files_missing: int
    texts_without_mapped_fixations: int
    mapped_fixations: int
    unmapped_fixations: int
    output_rows: int


@dataclass(frozen=True)
class CharBox:
    cursor: int
    value: str
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class TokenSpan:
    word_id: int
    start: int
    end: int
    word: str


def _read_sa1_annotation(annotation_csv: Path, include_neutral: bool) -> pd.DataFrame:
    annotation = pd.read_csv(annotation_csv)
    required = {"ID", "Sentences", "Consensus"}
    missing = required - set(annotation.columns)
    if missing:
        raise ValueError(f"Missing SA-I annotation columns: {sorted(missing)}")

    clean = annotation.copy()
    clean["ID"] = pd.to_numeric(clean["ID"], errors="coerce")
    clean = clean.dropna(subset=["ID", "Sentences", "Consensus"])
    clean["ID"] = clean["ID"].astype(int)
    clean["Consensus"] = clean["Consensus"].astype(str).str.upper().str.strip()
    allowed = {"P", "N", "O"} if include_neutral else {"P", "N"}
    clean = clean[clean["Consensus"].isin(allowed)]
    clean = clean.sort_values("ID").reset_index(drop=True)
    return clean


def _extract_sa1_archive(sa1_archive: Path, extract_dir: Path) -> tuple[Path, Path]:
    with tarfile.open(sa1_archive, "r:gz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.name
            in {
                "Released/Snippet/Annotation.csv",
                "Released/Snippet/Translog-II.tar.gz",
            }
        ]
        if len(members) != 2:
            names = sorted(member.name for member in archive.getmembers())
            raise ValueError(
                "SA-I archive must contain Released/Snippet/Annotation.csv and "
                f"Released/Snippet/Translog-II.tar.gz. Found {len(names)} entries."
            )
        try:
            archive.extractall(extract_dir, members=members, filter="data")
        except TypeError:
            archive.extractall(extract_dir, members=members)

    return (
        extract_dir / "Released" / "Snippet" / "Annotation.csv",
        extract_dir / "Released" / "Snippet" / "Translog-II.tar.gz",
    )


def _token_spans(text: str) -> list[TokenSpan]:
    spans: list[TokenSpan] = []
    for index, match in enumerate(re.finditer(r"\S+", text), start=1):
        spans.append(
            TokenSpan(
                word_id=index,
                start=match.start(),
                end=match.end(),
                word=match.group(),
            )
        )
    return spans


def _cursor_to_word(cursor: int, spans: list[TokenSpan]) -> int | None:
    for span in spans:
        if span.start <= cursor < span.end:
            return span.word_id
    left = [span for span in spans if span.end <= cursor]
    right = [span for span in spans if span.start >= cursor]
    candidates: list[tuple[int, int]] = []
    if left:
        span = left[-1]
        candidates.append((cursor - (span.end - 1), span.word_id))
    if right:
        span = right[0]
        candidates.append((span.start - cursor, span.word_id))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item[0], -item[1]))[0][1]


def _float_attr(element: ET.Element, name: str) -> float | None:
    value = element.attrib.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _source_field_bounds(root: ET.Element) -> tuple[float, float, float, float] | None:
    source_field = root.find(".//SourceField")
    if source_field is None:
        return None
    values = []
    for name in ["X", "Y", "Width", "Height"]:
        value = _float_attr(source_field, name)
        if value is None:
            return None
        values.append(value)
    x, y, width, height = values
    return x, y, x + width, y + height


def _source_char_boxes(root: ET.Element, source_text: str) -> list[CharBox]:
    bounds = _source_field_bounds(root)
    boxes: dict[int, CharBox] = {}
    for element in root.iter("CharPos"):
        try:
            cursor = int(element.attrib["Cursor"])
            x = float(element.attrib["X"])
            y = float(element.attrib["Y"])
            width = float(element.attrib["Width"])
            height = float(element.attrib["Height"])
        except (KeyError, ValueError):
            continue
        if cursor < 0 or cursor >= len(source_text):
            continue
        value = element.attrib.get("Value", "")
        if value != source_text[cursor]:
            continue
        if bounds is not None:
            left, top, right, bottom = bounds
            if x < left or x > right or y < top or y >= bottom:
                continue
        boxes.setdefault(
            cursor,
            CharBox(cursor=cursor, value=value, x=x, y=y, width=width, height=height),
        )
    return [boxes[cursor] for cursor in sorted(boxes)]


def _nearest_cursor_from_xy(x: float, y: float, boxes: list[CharBox]) -> int | None:
    if not boxes:
        return None
    containing = [
        box
        for box in boxes
        if box.x <= x <= box.x + box.width and box.y <= y <= box.y + box.height
    ]
    if containing:
        return min(
            containing,
            key=lambda box: abs((box.x + box.width / 2.0) - x)
            + abs((box.y + box.height / 2.0) - y),
        ).cursor

    line_candidates = [
        box
        for box in boxes
        if abs((box.y + box.height / 2.0) - y) <= max(box.height, 1.0)
    ]
    candidates = line_candidates if line_candidates else boxes
    return min(
        candidates,
        key=lambda box: abs((box.x + box.width / 2.0) - x)
        + 4.0 * abs((box.y + box.height / 2.0) - y),
    ).cursor


def _fixation_word_id(
    fix: ET.Element,
    boxes: list[CharBox],
    spans: list[TokenSpan],
) -> int | None:
    cursor = None
    x = _float_attr(fix, "X")
    y = _float_attr(fix, "Y")
    if x is not None and y is not None:
        cursor = _nearest_cursor_from_xy(x, y, boxes)
    if cursor is None and "Cursor" in fix.attrib:
        try:
            cursor = int(fix.attrib["Cursor"])
        except ValueError:
            cursor = None
    if cursor is None:
        return None
    return _cursor_to_word(cursor, spans)


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

    ordered = fixations.sort_values(["Participant_ID", "order"])
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

    base = pd.MultiIndex.from_product(
        [participants, range(1, n_words + 1)],
        names=["Participant_ID", "word_pos"],
    ).to_frame(index=False)
    gpt = pd.DataFrame(gpt_records, columns=["Participant_ID", "word_pos", "GPT"])
    return (
        base.merge(participant_word, on=["Participant_ID", "word_pos"], how="left")
        .merge(gpt, on=["Participant_ID", "word_pos"], how="left")
        .fillna({"nFix": 0.0, "FFD": 0.0, "GPT": 0.0, "TRT": 0.0})
    )


def _parse_translog_xml(
    xml_bytes: bytes,
    participant_id: str,
) -> tuple[str, pd.DataFrame, int, int]:
    root = ET.fromstring(xml_bytes)
    source_text = root.findtext(".//SourceTextUTF8")
    if source_text is None:
        source_text = ""
    source_text = source_text.strip()
    spans = _token_spans(source_text)
    boxes = _source_char_boxes(root, source_text)
    records: list[dict[str, object]] = []
    unmapped = 0

    for order, fix in enumerate(root.iter("Fix")):
        if fix.attrib.get("Win") != "1" or "Dur" not in fix.attrib:
            continue
        duration = _float_attr(fix, "Dur")
        if duration is None or duration < 0:
            continue
        word_id = _fixation_word_id(fix, boxes, spans)
        if word_id is None:
            unmapped += 1
            continue
        records.append(
            {
                "Participant_ID": participant_id,
                "word_pos": int(word_id),
                "Fixation_Duration": float(duration),
                "order": int(order),
            }
        )

    return source_text, pd.DataFrame(records), len(records), unmapped


def build_iitb_sa1_word_features(
    annotation_csv: Path,
    translog_tar: Path,
    include_neutral: bool = False,
    append_eos: bool = True,
) -> tuple[pd.DataFrame, SA1Report]:
    annotation_all = pd.read_csv(annotation_csv)
    annotation = _read_sa1_annotation(annotation_csv, include_neutral=include_neutral)
    text_ids = annotation["ID"].astype(int).tolist()
    text_id_set = set(text_ids)
    text_frames: dict[int, list[pd.DataFrame]] = {text_id: [] for text_id in text_ids}
    xml_source_texts: dict[int, str] = {}
    participants: set[str] = set()
    xml_files_seen = 0
    xml_files_used = 0
    mapped_fixations = 0
    unmapped_fixations = 0

    with tarfile.open(translog_tar, "r:gz") as translog:
        for member in translog:
            if not member.isfile():
                continue
            match = re.match(r".*/(P\d+)_(\d+)\.xml$", member.name)
            if not match:
                continue
            xml_files_seen += 1
            participant_id = match.group(1)
            text_id = int(match.group(2))
            participants.add(participant_id)
            if text_id not in text_id_set:
                continue
            handle = translog.extractfile(member)
            if handle is None:
                continue
            xml_text, frame, mapped, unmapped = _parse_translog_xml(
                handle.read(),
                participant_id=participant_id,
            )
            if xml_text:
                xml_source_texts[text_id] = xml_text
            text_frames[text_id].append(frame)
            xml_files_used += 1
            mapped_fixations += mapped
            unmapped_fixations += unmapped

    participants_sorted = sorted(
        participants,
        key=lambda value: int(value[1:]) if value[1:].isdigit() else value,
    )
    missing_xml = len(participants_sorted) * len(text_ids) - xml_files_used
    rows: list[dict[str, object]] = []
    texts_without_mapped_fixations = 0

    for sentence_id, annotation_row in enumerate(annotation.itertuples(index=False)):
        text_id = int(getattr(annotation_row, "ID"))
        source_text = xml_source_texts.get(
            text_id,
            str(getattr(annotation_row, "Sentences")).strip(),
        )
        text_tokens = source_text.split()
        participant_frames = text_frames.get(text_id, [])
        if participant_frames:
            text_fixations = pd.concat(participant_frames, ignore_index=True)
        else:
            text_fixations = pd.DataFrame(
                columns=[
                    "Participant_ID",
                    "word_pos",
                    "Fixation_Duration",
                    "order",
                ]
            )
        if text_fixations.empty:
            texts_without_mapped_fixations += 1
        participant_features = _participant_word_features(
            text_fixations,
            participants=participants_sorted,
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
            if word_index >= len(text_tokens):
                continue
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
    report = SA1Report(
        total_annotations=len(annotation_all),
        kept_annotations=len(annotation),
        participants=len(participants_sorted),
        xml_files_seen=xml_files_seen,
        xml_files_used=xml_files_used,
        xml_files_missing=missing_xml,
        texts_without_mapped_fixations=texts_without_mapped_fixations,
        mapped_fixations=mapped_fixations,
        unmapped_fixations=unmapped_fixations,
        output_rows=len(output),
    )
    return output, report


def _sentence_texts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sentence_id, group in df.sort_values(["sentence_id", "word_id"]).groupby(
        "sentence_id"
    ):
        text = " ".join(
            str(word).replace("<EOS>", "").strip()
            for word in group["word"].tolist()
        )
        rows.append({"sentence_id": int(sentence_id), "text_norm": _normalize_text(text)})
    return pd.DataFrame(rows)


def _normalize_text(text: str) -> str:
    text = text.replace("\\'", "'").replace('\\"', '"')
    return re.sub(r"\s+", " ", text.strip().lower())


def combine_without_duplicate_sentences(
    base_df: pd.DataFrame,
    add_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    base_clean = renumber_sentences(base_df)
    add_clean = renumber_sentences(add_df)
    base_texts = set(_sentence_texts(base_clean)["text_norm"].tolist())
    add_texts = _sentence_texts(add_clean)
    keep_ids: list[int] = []
    seen_add: set[str] = set()
    duplicate_against_base = 0
    duplicate_within_add = 0
    for row in add_texts.itertuples(index=False):
        if row.text_norm in base_texts:
            duplicate_against_base += 1
            continue
        if row.text_norm in seen_add:
            duplicate_within_add += 1
            continue
        seen_add.add(row.text_norm)
        keep_ids.append(int(row.sentence_id))

    add_kept = add_clean[add_clean["sentence_id"].isin(keep_ids)].copy()
    add_kept = renumber_sentences(
        add_kept,
        offset=base_clean["sentence_id"].nunique(),
    )
    combined = pd.concat([base_clean, add_kept], ignore_index=True)
    return combined, {
        "base_sentences": int(base_clean["sentence_id"].nunique()),
        "candidate_sentences": int(add_clean["sentence_id"].nunique()),
        "added_sentences": int(add_kept["sentence_id"].nunique()),
        "duplicate_against_base": int(duplicate_against_base),
        "duplicate_within_add": int(duplicate_within_add),
        "combined_sentences": int(combined["sentence_id"].nunique()),
        "combined_rows": int(len(combined)),
    }


def convert_iitb_sa1(
    output_csv: Path,
    sa1_archive: Path | None = None,
    annotation_csv: Path | None = None,
    translog_tar: Path | None = None,
    raw_output_csv: Path | None = None,
    stats_json: Path | None = None,
    combined_with_csv: Path | None = None,
    combined_output_csv: Path | None = None,
    combined_stats_json: Path | None = None,
    include_neutral: bool = False,
    scale: bool = True,
    preserve_zero_rows: bool = True,
    append_eos: bool = True,
) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmpdir:
        if sa1_archive is not None:
            annotation_path, translog_path = _extract_sa1_archive(
                sa1_archive,
                Path(tmpdir),
            )
        else:
            if annotation_csv is None or translog_tar is None:
                raise ValueError(
                    "Provide either --sa1-archive or both --annotation-csv and --translog-tar."
                )
            annotation_path = annotation_csv
            translog_path = translog_tar

        raw_df, report = build_iitb_sa1_word_features(
            annotation_csv=annotation_path,
            translog_tar=translog_path,
            include_neutral=include_neutral,
            append_eos=append_eos,
        )

    target_stats = CMCL_NONZERO_TARGET_STATS if preserve_zero_rows else CMCL_TARGET_STATS
    if scale:
        scaled_df = scale_features_to_cmcl(
            raw_df,
            target_stats=target_stats,
            preserve_zero_rows=preserve_zero_rows,
        )
    else:
        scaled_df = raw_df.copy()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scaled_df.to_csv(output_csv, index=False)

    if raw_output_csv is not None:
        raw_output_csv.parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(raw_output_csv, index=False)

    stats: dict[str, object] | None = None
    if stats_json is not None:
        stats_json.parent.mkdir(parents=True, exist_ok=True)
        stats = summarize_features(
            raw_df,
            scaled_df,
            target_stats=target_stats,
            preserve_zero_rows=preserve_zero_rows,
        )
        stats["sa1_report"] = report.__dict__
        stats["include_neutral"] = include_neutral
        stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    if combined_with_csv is not None or combined_output_csv is not None:
        if combined_with_csv is None or combined_output_csv is None:
            raise ValueError(
                "--combined-with-csv and --combined-output-csv must be provided together."
            )
        base_df = pd.read_csv(combined_with_csv)
        combined_df, dedupe_report = combine_without_duplicate_sentences(
            base_df,
            scaled_df,
        )
        combined_output_csv.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(combined_output_csv, index=False)
        if combined_stats_json is not None:
            combined_stats_json.parent.mkdir(parents=True, exist_ok=True)
            combined_stats = {
                "dedupe_report": dedupe_report,
                "combined": combined_df[FEATURE_NAMES].describe().to_dict(),
                "base_csv": str(combined_with_csv),
                "added_csv": str(output_csv),
            }
            combined_stats_json.write_text(
                json.dumps(combined_stats, indent=2),
                encoding="utf-8",
            )
        elif stats_json is not None and stats is not None:
            stats["dedupe_report"] = dedupe_report
            stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return scaled_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sa1-archive", type=Path, default=DEFAULT_SA1_ARCHIVE)
    parser.add_argument("--annotation-csv", type=Path, default=None)
    parser.add_argument("--translog-tar", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--raw-output-csv", type=Path, default=None)
    parser.add_argument("--stats-json", type=Path, default=None)
    parser.add_argument("--combined-with-csv", type=Path, default=None)
    parser.add_argument("--combined-output-csv", type=Path, default=None)
    parser.add_argument("--combined-stats-json", type=Path, default=None)
    parser.add_argument("--include-neutral", action="store_true")
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--scale-zero-rows", action="store_true")
    parser.add_argument("--no-append-eos", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sa1_archive = args.sa1_archive
    if args.annotation_csv is not None or args.translog_tar is not None:
        sa1_archive = None
    output_df = convert_iitb_sa1(
        sa1_archive=sa1_archive,
        annotation_csv=args.annotation_csv,
        translog_tar=args.translog_tar,
        output_csv=args.output_csv,
        raw_output_csv=args.raw_output_csv,
        stats_json=args.stats_json,
        combined_with_csv=args.combined_with_csv,
        combined_output_csv=args.combined_output_csv,
        combined_stats_json=args.combined_stats_json,
        include_neutral=args.include_neutral,
        scale=not args.no_scale,
        preserve_zero_rows=not args.scale_zero_rows,
        append_eos=not args.no_append_eos,
    )
    print(f"Saved {len(output_df)} rows to {args.output_csv}")
    print(output_df[FEATURE_NAMES].describe().round(4).to_string())


if __name__ == "__main__":
    main()
