from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import spacy


def clean_text(text: object) -> str:
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^a-zA-Z0-9,.!?;:'\"\-()%$]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_list_items(text: object) -> str:
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return re.sub(r"(\s*-\s+)([a-z])", r". \2", text)


def split_sentences(text: object, nlp) -> list[str]:
    if isinstance(text, (list, np.ndarray)):
        text = " ".join(text) if len(text) else ""
    if pd.isna(text) or not isinstance(text, str):
        return []

    doc = nlp(preprocess_list_items(text))
    sentences = [sent.text.strip() for sent in doc.sents]

    # Preserve the original rule: fragments of <=2 words are attached to
    # the following/adjacent content rather than retained as standalone rows.
    merged: list[str] = []
    current = ""
    for sent in sentences:
        if len(sent.split()) <= 2:
            current += " " + sent
        else:
            if current:
                merged.append(current.strip())
                current = ""
            merged.append(sent)
    if current:
        merged.append(current.strip())
    return merged


def clean_repeated_symbols(text: object) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"([.,!?])\1{1,}", r"\1", text)


def has_valid_words(text: object) -> bool:
    return isinstance(text, str) and bool(re.search(r"\b[a-zA-Z]{2,}\b", text))


def run(input_file: Path, output_file: Path, text_column: str, output_column: str, spacy_model: str) -> None:
    nlp = spacy.load(spacy_model)
    df = pd.read_excel(input_file)
    if text_column not in df.columns:
        raise KeyError(f"Missing text column: {text_column}")

    work = df.copy()
    work[text_column] = work[text_column].apply(clean_text)
    work[output_column] = work[text_column].apply(lambda x: split_sentences(x, nlp))
    work = work.explode(output_column, ignore_index=True)
    work[output_column] = work[output_column].apply(clean_repeated_symbols)

    # The original preprocessing applied sentence segmentation a second time
    # after repeated-symbol cleanup. It is retained for reproducibility.
    work[output_column] = work[output_column].apply(lambda x: split_sentences(x, nlp))
    work = work.explode(output_column, ignore_index=True)
    work = work[work[output_column].apply(has_valid_words)].copy()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    work.to_excel(output_file, index=False)
    print(f"Saved {len(work):,} sentence rows to {output_file}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean and sentence-split hotel review text before Qwen information extraction.")
    p.add_argument("--input", type=Path, required=True, help="Raw Booking.com review workbook.")
    p.add_argument("--output", type=Path, required=True, help="Output .xlsx file containing sentence-level review rows.")
    p.add_argument("--text-column", default="neg_comments", help="Review text column to split.")
    p.add_argument("--output-column", default="split_text", help="Canonical sentence-text column used by downstream scripts.")
    p.add_argument("--spacy-model", default="en_core_web_sm")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output, args.text_column, args.output_column, args.spacy_model)
