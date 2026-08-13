from __future__ import annotations
    
from pathlib import Path
import argparse

import numpy as np
import pandas as pd


def run(
    aop_file: Path,
    label_file: Path,
    output_file: Path,
    key: str,
) -> None:
    """Merge finalized service element labels and construct sentiment scores."""

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    aop = pd.read_excel(aop_file)
    labels = pd.read_excel(label_file)

    if key not in aop.columns or key not in labels.columns:
        raise KeyError(
            f"Both inputs must contain merge key '{key}'."
        )

    # ------------------------------------------------------------------
    # 2. Merge finalized service-element labels
    # ------------------------------------------------------------------
    df = pd.merge(
        aop,
        labels,
        on=key,
        how="left",
        suffixes=("", "_label"),
        validate="many_to_one",
    )

    # ------------------------------------------------------------------
    # 3. Convert LLM sentiment from [0, 1] to [1, 10]
    # ------------------------------------------------------------------
    required_cols = {
        "s",
        "Name",
        "release time",
        "Room info",
        "total score",
        "pos_comments",
        "split_text",
    }
    missing_cols = required_cols.difference(df.columns)

    if missing_cols:
        raise KeyError(
            f"Missing required columns: {sorted(missing_cols)}"
        )

    df["s"] = pd.to_numeric(df["s"], errors="coerce")
    df["total score"] = pd.to_numeric(
        df["total score"],
        errors="coerce",
    )

    # Linear transformation:
    # s = 0   -> sentiment = 1
    # s = 0.5 -> sentiment = 5.5
    # s = 1   -> sentiment = 10
    df["sentiment"] = 1.0 + 9.0 * df["s"]

    # ------------------------------------------------------------------
    # 4. Correct single neutral service evaluations
    # ------------------------------------------------------------------
    # If a review contains only one extracted service evaluation and the
    # LLM assigns the neutral score s = 0.5, use the observed overall
    # review rating as the service-level sentiment score.
    group_cols = [
        "Name",
        "release time",
        "Room info",
        "total score",
        "pos_comments",
    ]

    group_size = (
        df.groupby(group_cols, dropna=False)["split_text"]
        .transform("size")
    )

    neutral_single_mask = (
        (group_size == 1)
        & np.isclose(df["s"], 0.5, equal_nan=False)
        & df["total score"].notna()
    )

    df.loc[
        neutral_single_mask,
        "sentiment",
    ] = df.loc[
        neutral_single_mask,
        "total score",
    ]

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    df.to_excel(output_file, index=False)

    print(f"Saved postprocessed AOCS table to: {output_file}")
    print(f"Rows: {len(df):,}")
    print(
        "Single neutral evaluations replaced by overall rating: "
        f"{int(neutral_single_mask.sum()):,}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge finalized service-element labels into extracted AOCS rows "
            "and convert LLM sentiment scores from [0, 1] to [1, 10]."
        )
    )

    parser.add_argument(
        "--aop-file",
        type=Path,
        required=True,
        help="Input AOCS-level Excel file matched to the original reviews.",
    )
    parser.add_argument(
        "--label-file",
        type=Path,
        required=True,
        help="Excel file containing finalized service-element labels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output Excel file containing finalized AOCS rows.",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="Sub_Issue",
        help="Column used to merge the AOCS table with finalized labels.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        aop_file=args.aop_file,
        label_file=args.label_file,
        output_file=args.output,
        key=args.key,
    )


if __name__ == "__main__":
    main()
