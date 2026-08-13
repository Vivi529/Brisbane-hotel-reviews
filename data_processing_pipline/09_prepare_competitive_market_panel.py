from __future__ import annotations

import argparse
import re
from pathlib import Path
import pandas as pd

PERIOD_BOUNDARIES = [
    "2020-09-30", "2020-12-31", "2021-03-31", "2021-06-30",
    "2021-09-30", "2021-12-31", "2022-03-31", "2022-06-30",
    "2022-09-30", "2022-12-31", "2023-03-31", "2023-06-30",
    "2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30",
    "2024-09-30", "2024-12-31", "2025-03-31",
]


def normalize_hotel_name(x: object) -> str:
    if not isinstance(x, str):
        return ""
    return re.sub(r"[^\w\s]", "", x).strip().lower()


def add_period_id(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    bins = pd.to_datetime(PERIOD_BOUNDARIES)
    labels = [f"{bins[i].date()} - {bins[i+1].date()}" for i in range(len(bins)-1)]
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out["time_range"] = pd.cut(out[date_col], bins=bins, labels=labels)
    out["time_group_id"] = out["time_range"].cat.codes + 1
    return out


def run(review_matrix: Path, competitors_file: Path, row_output: Path, panel_output: Path) -> None:
    df = pd.read_excel(review_matrix)
    comp = pd.read_csv(competitors_file) if competitors_file.suffix.lower() == ".csv" else pd.read_excel(competitors_file)
    if "Name" not in df.columns or "Name" not in comp.columns:
        raise KeyError("Both inputs must contain a Name column.")

    df["Name"] = df["Name"].apply(normalize_hotel_name)
    comp["Name"] = comp["Name"].apply(normalize_hotel_name)
    focal = df[df["Name"].isin(set(comp["Name"]))].copy()
    focal = add_period_id(focal, "release time")

    row_output.parent.mkdir(parents=True, exist_ok=True)
    focal.to_excel(row_output, index=False)

    es_cols = [c for c in focal.columns if str(c).startswith("ES_")]
    panel = focal.groupby(["Name", "time_group_id"], as_index=False)[es_cols].mean()
    review_counts = focal.groupby(["Name", "time_group_id"]).size().rename("new_review_count").reset_index()
    panel = pd.merge(panel, review_counts, on=["Name", "time_group_id"], how="left")
    panel.to_excel(panel_output, index=False)
    print(f"Competitive review rows: {len(focal):,}; hotel-period rows: {len(panel):,}")


def parse_args():
    p = argparse.ArgumentParser(description="Filter competitive hotels and aggregate service-element performance by quarterly period.")
    p.add_argument("--review-matrix", type=Path, required=True)
    p.add_argument("--competitors", type=Path, required=True)
    p.add_argument("--row-output", type=Path, required=True)
    p.add_argument("--panel-output", type=Path, required=True)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(a.review_matrix, a.competitors, a.row_output, a.panel_output)
