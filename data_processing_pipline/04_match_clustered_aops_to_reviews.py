from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def norm_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def match_metadata(review_rows: pd.DataFrame, clustered_aops: pd.DataFrame, text_col: str, fallback_col: str) -> pd.DataFrame:
    """Attach original review metadata to clustered AOP rows.

    This preserves the matching logic used in the research script: a clustered
    sentence is first searched as a substring of the sentence-level review text;
    if no match is found, the positive-comment field is searched as fallback.
    The first matching review row is used, matching the original workflow.
    """
    a = review_rows.copy()
    b = clustered_aops.copy()
    for col in [text_col, fallback_col]:
        if col not in a.columns:
            raise KeyError(f"Missing review column: {col}")
        a[col] = norm_text(a[col])
    if text_col not in b.columns:
        raise KeyError(f"Missing clustered-AOP text column: {text_col}")
    b[text_col] = norm_text(b[text_col])

    matched_count = 0
    unmatched_count = 0
    for idx, row in b.iterrows():
        query = row[text_col]
        if not query:
            unmatched_count += 1
            continue

        matched = a[a[text_col].str.contains(re.escape(query), na=False)]
        if matched.empty:
            matched = a[a[fallback_col].str.contains(re.escape(query), na=False)]

        if matched.empty:
            unmatched_count += 1
            continue

        source = matched.iloc[0]
        for col in a.columns:
            if col != text_col:
                b.at[idx, col] = source[col]
        matched_count += 1

    print(f"Matched AOP rows: {matched_count:,}; unmatched: {unmatched_count:,}")
    return b


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Match clustered AOP rows back to their original review metadata.")
    p.add_argument("--reviews", type=Path, required=True, help="Sentence-level Qwen extraction output with review metadata.")
    p.add_argument("--clusters", type=Path, required=True, help="Final clustered AOP/service-element table.")
    p.add_argument("--cluster-sheet", default=0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--text-column", default="split_text")
    p.add_argument("--fallback-column", default="pos_comments")
    return p.parse_args()


def read_table(path: Path, sheet=0) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet)
    return pd.read_csv(path)


if __name__ == "__main__":
    args = parse_args()
    reviews = read_table(args.reviews)
    clusters = read_table(args.clusters, args.cluster_sheet)
    result = match_metadata(reviews, clusters, args.text_column, args.fallback_column)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_excel(args.output, index=False)
