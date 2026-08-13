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
HOTEL_FEATURES = ["pagerank_score", "Num", "Star", "distance from centre (km)", "location", "suburbs"]


def normalize_hotel_name(x: object) -> str:
    if not isinstance(x, str):
        return ""
    return re.sub(r"[^\w\s]", "", x).strip().lower()


def run(review_matrix: Path, hotel_info: Path, output: Path) -> None:
    reviews = pd.read_excel(review_matrix)
    hotels = pd.read_excel(hotel_info)
    for frame in (reviews, hotels):
        if "Name" not in frame.columns:
            raise KeyError("Both files must contain Name.")
        frame["Name"] = frame["Name"].apply(normalize_hotel_name)

    missing = [c for c in HOTEL_FEATURES if c not in hotels.columns]
    if missing:
        raise KeyError(f"Missing hotel-feature columns: {missing}")
    hotel_agg = hotels.groupby("Name", as_index=False)[HOTEL_FEATURES].first()
    df = pd.merge(reviews, hotel_agg, on="Name", how="left")

    bins = pd.to_datetime(PERIOD_BOUNDARIES)
    labels = [f"{bins[i].date()} - {bins[i+1].date()}" for i in range(len(bins)-1)]
    df["release time"] = pd.to_datetime(df["release time"], errors="coerce")
    df["time_range"] = pd.cut(df["release time"], bins=bins, labels=labels)
    df["time_group_id"] = df["time_range"].cat.codes + 1

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output, index=False)
    print(f"Saved GRF-ready review-level data: {len(df):,} rows")


def parse_args():
    p = argparse.ArgumentParser(description="Attach hotel characteristics and quarterly period IDs for GRF/Kano estimation.")
    p.add_argument("--review-matrix", type=Path, required=True)
    p.add_argument("--hotel-info", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(a.review_matrix, a.hotel_info, a.output)
