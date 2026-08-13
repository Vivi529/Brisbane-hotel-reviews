from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

DEFAULT_GROUP_COLS = ["Name", "release time", "Room info", "total score", "pos_comments"]
AOP_DETAIL_COLS = {
    "a", "o", "c", "s", "r", "evidence_phrase", "evaluation_dimension",
    "D_Std", "D_Final", "cluster_text", "distance_to_medoid", "medoid_text",
    "original_topic", "sentiment", "Cluster", "Sub_Issue"
}


def sort_key(col: object):
    s = str(col)
    if s.startswith("ES_"):
        s = s[3:]
    if "_" in s:
        main, sub = s.split("_", 1)
        if main.isdigit() and sub.isdigit():
            return int(main), int(sub)
    if s.isdigit():
        return int(s), 0
    return float("inf"), float("inf")


def canonical_es_name(col: object) -> str:
    s = str(col)
    if s.startswith("ES_"):
        return s
    if s.replace("_", "").isdigit():
        return f"ES_{s}"
    return s


def build_matrix(df: pd.DataFrame, element_col: str, sentiment_col: str, group_cols: list[str]) -> pd.DataFrame:
    required = set(group_cols) | {element_col, sentiment_col}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    pivot = df.pivot_table(
        index=group_cols,
        columns=element_col,
        values=sentiment_col,
        aggfunc="mean",
    ).reset_index()
    pivot.columns.name = None

    element_cols = [c for c in pivot.columns if c not in group_cols]
    element_cols = sorted(element_cols, key=sort_key)
    pivot = pivot[group_cols + element_cols]
    pivot.columns = [canonical_es_name(c) if c not in group_cols else c for c in pivot.columns]

    # Retain review-level metadata only; exclude tuple-/cluster-level fields.
    metadata_cols = [
        c for c in df.columns
        if c not in AOP_DETAIL_COLS and c not in group_cols and not str(c).startswith("ES_")
    ]
    if metadata_cols:
        info = df.groupby(group_cols)[metadata_cols].first().reset_index()
        pivot = pd.merge(pivot, info, on=group_cols, how="left")
    return pivot


def parse_args():
    p = argparse.ArgumentParser(description="Convert finalized AOP rows into the review × service-element matrix used by SHAP and downstream models.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--element-column", default="Cluster", help="Final service-element identifier; use Cluster for ES-level analysis.")
    p.add_argument("--sentiment-column", default="sentiment")
    p.add_argument("--group-columns", nargs="+", default=DEFAULT_GROUP_COLS)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    df = pd.read_excel(a.input)
    out = build_matrix(df, a.element_column, a.sentiment_column, a.group_columns)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_excel(a.output, index=False)
    print(f"Saved review × service-element matrix: {out.shape[0]:,} reviews, {sum(str(c).startswith('ES_') for c in out.columns)} ES columns")
