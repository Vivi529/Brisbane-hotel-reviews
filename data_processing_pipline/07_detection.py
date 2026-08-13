from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

EPS = 1e-12


def entropy(pos_count: float, neg_count: float) -> float:
    total = pos_count + neg_count
    if total <= 0:
        return 0.0
    result = 0.0
    for count in (pos_count, neg_count):
        if count > 0:
            p = count / total
            result -= p * np.log2(p)
    return float(result)


def calculate_detection_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the current RPN Detection measure plus IG diagnostics.

    Required input columns: Cluster, Pos, Neg, Total.
    Detection follows the manuscript definition based on above-baseline
    negative-sentiment enrichment. Information gain is retained only as a
    diagnostic because it appeared in earlier analysis code.
    """
    data = df.copy()
    required = {"Cluster", "Pos", "Neg", "Total"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data[["Pos", "Neg", "Total"]] = data[["Pos", "Neg", "Total"]].apply(pd.to_numeric, errors="raise")
    if not np.allclose(data["Pos"] + data["Neg"], data["Total"]):
        raise ValueError("Some rows do not satisfy Pos + Neg = Total.")

    total_pos = float(data["Pos"].sum())
    total_neg = float(data["Neg"].sum())
    total_count = total_pos + total_neg
    if total_count <= 0:
        raise ValueError("Total AOCS count must be positive.")

    h0 = total_neg / total_count
    data["Negative_rate"] = np.where(data["Total"] > 0, data["Neg"] / data["Total"], 0.0)

    # Standard one-vs-rest information gain (diagnostic only).
    H = entropy(total_pos, total_neg)
    ig = []
    for _, row in data.iterrows():
        pp, pn = float(row["Pos"]), float(row["Neg"])
        pt = pp + pn
        ap, an = total_pos - pp, total_neg - pn
        at = ap + an
        cond = (pt / total_count) * entropy(pp, pn) + (at / total_count) * entropy(ap, an)
        ig.append(max(H - cond, 0.0))
    data["Information_gain"] = ig
    lo, hi = float(np.min(ig)), float(np.max(ig))
    data["Normalized_IG"] = 0.0 if hi - lo <= EPS else (data["Information_gain"] - lo) / (hi - lo)

    denom = max(1.0 - h0, EPS)
    data["Monitorability"] = ((data["Negative_rate"] - h0) / denom).clip(0.0, 1.0)
    data["Detection"] = 1.0 + 9.0 * (1.0 - data["Monitorability"])

    data.attrs["global_negative_rate"] = h0
    data.attrs["global_entropy"] = H
    return data


def parse_args():
    p = argparse.ArgumentParser(description="Calculate service-element Detection scores for the RPN.")
    p.add_argument("--input", type=Path, required=True, help="Excel/CSV table with Cluster, Pos, Neg, Total.")
    p.add_argument("--sheet", default=0)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def read_table(path: Path, sheet=0):
    return pd.read_excel(path, sheet_name=sheet) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)


if __name__ == "__main__":
    a = parse_args()
    result = calculate_detection_metrics(read_table(a.input, a.sheet))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.suffix.lower() == ".csv":
        result.to_csv(a.output, index=False)
    else:
        result.to_excel(a.output, index=False)
    print(result[["Cluster", "Negative_rate", "Monitorability", "Detection"]].to_string(index=False))
