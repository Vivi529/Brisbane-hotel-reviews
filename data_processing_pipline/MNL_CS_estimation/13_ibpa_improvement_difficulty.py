# -*- coding: utf-8 -*-
"""
IBPA-based service-element improvement difficulty
=================================================

This script merges the complete preprocessing and IBPA workflow:

AOCS-level sentiment records
    -> quarterly negative review rates NR_m^t
    -> corrective response rates CRR_m^{t+1}
    -> iterative Bayesian probability approach (IBPA)
    -> transition-specific calibrated improvement rates CM_{m,t}
    -> overall improvability CM_m
    -> implementation difficulty Cost_m = 1 / CM_m

Formal definitions
------------------
For service element m in period t:

    NR_m^t = N_neg,m,t / (N_pos,m,t + N_neg,m,t)

    CRR_m^{t+1} = max(NR_m^t - NR_m^{t+1}, 0)

The IBPA update is

    CM_{m,t}
        = (CRR_m^{t+1} + alpha_m)
          / (NR_m^t + alpha_m + beta_m)

where alpha_m and beta_m are iteratively estimated from the current
distribution of CM_{m,t} using Beta moment matching.

After convergence:

    CM_m = mean_t(CM_{m,t})
    Cost_m = 1 / CM_m

Notes
-----
1. The original research preprocessing filled a missing quarter × service-
   element negative rate with 0. The default --missing-rate-policy zero
   preserves that behavior. Use --missing-rate-policy exclude if only adjacent
   periods with observed positive/negative AOCS instances should be used.
2. The original IBPA notebook printed estimated_h[-1]. The formal manuscript
   definition uses the mean of the converged transition-specific values.
   Therefore this script uses CM = mean(CM_{m,t}) for Cost, while also exporting
   CM_last_legacy for comparison with the historical notebook.
3. Neutral/other sentiment labels are not included in the denominator of NR,
   matching the original positive/negative counting logic.

Example
-------
python 13_ibpa_improvement_difficulty.py \
    --input data/04_postprocessing/AOCS_finalized.xlsx \
    --output data/08_improvement_difficulty/ibpa_improvement_difficulty.xlsx
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# 1. DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_START_DATE = "2020-09-30"
DEFAULT_END_DATE = "2025-03-11"

DEFAULT_DATE_COLUMN = "release time"
DEFAULT_CLUSTER_COLUMN = "Cluster"
DEFAULT_SENTIMENT_COLUMN = "情感倾向"
DEFAULT_POSITIVE_LABEL = "正向"
DEFAULT_NEGATIVE_LABEL = "负向"

DEFAULT_TOL = 1e-6
DEFAULT_MAX_ITER = 1000

EPS = 1e-12


# =============================================================================
# 2. DATA STRUCTURE
# =============================================================================

@dataclass
class IBPAResult:
    es: str
    alpha: float
    beta: float
    cm_by_transition: np.ndarray
    raw_ratio: np.ndarray
    cm: float
    cost: float
    cm_last_legacy: float
    mse: float
    iterations: int
    converged: bool
    status: str


# =============================================================================
# 3. GENERAL HELPERS
# =============================================================================

def normalize_es_name(value: Any) -> str:
    """
    Normalize service-element identifiers.

    Examples
    --------
    1       -> ES_1
    "1"     -> ES_1
    "ES1"   -> ES_1
    "ES-1"  -> ES_1
    "ES_1"  -> ES_1

    Non-numeric labels are retained as strings.
    """
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if re.fullmatch(r"\d+(?:\.0+)?", text):
        return f"ES_{int(float(text))}"

    match = re.fullmatch(r"(?i)ES[\s_-]?(\d+)", text)
    if match:
        return f"ES_{int(match.group(1))}"

    return text


def es_sort_key(value: str) -> tuple[int, int | str]:
    """Sort ES_1, ES_2, ... numerically, followed by other labels."""
    match = re.fullmatch(r"ES_(\d+)", str(value))
    if match:
        return (0, int(match.group(1)))
    return (1, str(value))


def expected_quarters(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> list[str]:
    start_period = pd.Timestamp(start_date).to_period("Q")
    end_period = pd.Timestamp(end_date).to_period("Q")
    return [
        str(period)
        for period in pd.period_range(start_period, end_period, freq="Q")
    ]


def validate_probability_matrix(
    matrix: pd.DataFrame,
    *,
    name: str,
    allow_nan: bool,
) -> None:
    values = matrix.to_numpy(dtype=float)

    if not allow_nan and np.isnan(values).any():
        raise ValueError(f"{name} contains missing values.")

    finite = values[np.isfinite(values)]
    if finite.size and np.any((finite < -EPS) | (finite > 1.0 + EPS)):
        raise ValueError(f"{name} contains values outside [0, 1].")


# =============================================================================
# 4. QUARTERLY NEGATIVE REVIEW RATE
# =============================================================================

def prepare_quarterly_negative_rates(
    df: pd.DataFrame,
    *,
    date_column: str,
    cluster_column: str,
    sentiment_column: str,
    positive_label: str,
    negative_label: str,
    start_date: str,
    end_date: str,
    missing_rate_policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Construct quarter × service-element positive/negative counts and NR_m^t.

    Returns
    -------
    sentiment_counts_long
        Long table with positive_count, negative_count, total_count and
        negative_rate.
    negative_rate_matrix
        Quarter × ES matrix of NR_m^t.
    positive_count_matrix
        Quarter × ES matrix of positive counts.
    negative_count_matrix
        Quarter × ES matrix of negative counts.
    """
    required = {
        date_column,
        cluster_column,
        sentiment_column,
    }
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(
            f"Input data are missing required columns: {sorted(missing)}"
        )

    work = df.copy()

    work[date_column] = pd.to_datetime(
        work[date_column],
        errors="coerce",
    )
    invalid_dates = int(work[date_column].isna().sum())
    if invalid_dates:
        print(
            f"Warning: {invalid_dates:,} rows have invalid/missing "
            f"{date_column!r} and will be excluded."
        )

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    if end_ts < start_ts:
        raise ValueError("end_date must not be earlier than start_date.")

    work = work.loc[
        work[date_column].between(start_ts, end_ts, inclusive="both")
    ].copy()

    if work.empty:
        raise ValueError(
            f"No rows remain between {start_date} and {end_date}."
        )

    work["ES"] = work[cluster_column].map(normalize_es_name)
    work = work.loc[work["ES"] != ""].copy()

    if work.empty:
        raise ValueError("No valid service-element identifiers remain.")

    work["quarter"] = (
        work[date_column]
        .dt.to_period("Q")
        .astype(str)
    )

    expected = expected_quarters(start_date, end_date)
    es_order = sorted(work["ES"].unique().tolist(), key=es_sort_key)

    # Count all sentiment labels first so that non-positive/non-negative labels
    # can be transparently diagnosed.
    raw_counts = (
        work.groupby(
            ["quarter", "ES"],
            observed=False,
        )[sentiment_column]
        .value_counts()
        .unstack(fill_value=0)
    )

    if positive_label not in raw_counts.columns:
        raw_counts[positive_label] = 0
    if negative_label not in raw_counts.columns:
        raw_counts[negative_label] = 0

    positive_count = (
        raw_counts[positive_label]
        .rename("positive_count")
        .astype(float)
    )
    negative_count = (
        raw_counts[negative_label]
        .rename("negative_count")
        .astype(float)
    )

    counts = pd.concat(
        [positive_count, negative_count],
        axis=1,
    )
    counts["total_count"] = (
        counts["positive_count"]
        + counts["negative_count"]
    )

    counts["negative_rate"] = np.where(
        counts["total_count"] > 0,
        counts["negative_count"] / counts["total_count"],
        np.nan,
    )

    # Reindex to the complete quarter × ES grid so that every period and
    # service element is explicitly represented.
    full_index = pd.MultiIndex.from_product(
        [expected, es_order],
        names=["quarter", "ES"],
    )
    counts = counts.reindex(full_index)

    counts["positive_count"] = counts["positive_count"].fillna(0.0)
    counts["negative_count"] = counts["negative_count"].fillna(0.0)
    counts["total_count"] = (
        counts["positive_count"]
        + counts["negative_count"]
    )

    if missing_rate_policy == "zero":
        # Reproduces the original preprocessing:
        # positive_pivot.fillna(0), negative_pivot.fillna(0).
        counts["negative_rate"] = counts["negative_rate"].fillna(0.0)
    elif missing_rate_policy == "exclude":
        # Leave periods with no positive/negative evidence as NaN.
        counts.loc[
            counts["total_count"] <= 0,
            "negative_rate",
        ] = np.nan
    else:
        raise ValueError(
            "missing_rate_policy must be 'zero' or 'exclude'."
        )

    sentiment_counts_long = counts.reset_index()

    negative_rate_matrix = (
        sentiment_counts_long
        .pivot(
            index="quarter",
            columns="ES",
            values="negative_rate",
        )
        .reindex(index=expected, columns=es_order)
    )

    positive_count_matrix = (
        sentiment_counts_long
        .pivot(
            index="quarter",
            columns="ES",
            values="positive_count",
        )
        .reindex(index=expected, columns=es_order)
        .fillna(0.0)
    )

    negative_count_matrix = (
        sentiment_counts_long
        .pivot(
            index="quarter",
            columns="ES",
            values="negative_count",
        )
        .reindex(index=expected, columns=es_order)
        .fillna(0.0)
    )

    validate_probability_matrix(
        negative_rate_matrix,
        name="negative_rate_matrix",
        allow_nan=(missing_rate_policy == "exclude"),
    )

    # Diagnostic: labels other than the two used in NR.
    known_mask = work[sentiment_column].isin(
        [positive_label, negative_label]
    )
    ignored = int((~known_mask & work[sentiment_column].notna()).sum())
    if ignored:
        ignored_labels = (
            work.loc[
                ~known_mask & work[sentiment_column].notna(),
                sentiment_column,
            ]
            .astype(str)
            .value_counts()
            .to_dict()
        )
        print(
            "Note: sentiment labels other than the specified positive/negative "
            f"labels are excluded from NR denominators: {ignored_labels}"
        )

    return (
        sentiment_counts_long,
        negative_rate_matrix,
        positive_count_matrix,
        negative_count_matrix,
    )


# =============================================================================
# 5. CORRECTIVE RESPONSE RATE
# =============================================================================

def calculate_corrective_response_rates(
    negative_rate_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate

        CRR_m^{t+1} = max(NR_m^t - NR_m^{t+1}, 0)

    for every adjacent quarter.

    Returns
    -------
    crr_matrix
        Transition × ES matrix.
    transition_long
        Long-form table containing NR_t, NR_t1 and CRR_t1.
    """
    if len(negative_rate_matrix) < 2:
        raise ValueError(
            "At least two periods are required to calculate CRR."
        )

    quarter_names = negative_rate_matrix.index.astype(str).tolist()
    es_names = negative_rate_matrix.columns.astype(str).tolist()

    rows: list[dict[str, Any]] = []
    crr_values = np.full(
        (len(quarter_names) - 1, len(es_names)),
        np.nan,
        dtype=float,
    )

    for t in range(len(quarter_names) - 1):
        period_t = quarter_names[t]
        period_t1 = quarter_names[t + 1]

        nr_t = negative_rate_matrix.iloc[t].to_numpy(dtype=float)
        nr_t1 = negative_rate_matrix.iloc[t + 1].to_numpy(dtype=float)

        valid = np.isfinite(nr_t) & np.isfinite(nr_t1)

        crr = np.full(len(es_names), np.nan, dtype=float)
        crr[valid] = np.maximum(
            nr_t[valid] - nr_t1[valid],
            0.0,
        )
        crr_values[t, :] = crr

        for j, es in enumerate(es_names):
            rows.append(
                {
                    "ES": es,
                    "period_t": period_t,
                    "period_t1": period_t1,
                    "NR_t": nr_t[j],
                    "NR_t1": nr_t1[j],
                    "CRR_t1": crr[j],
                    "available_transition": bool(valid[j]),
                }
            )

    transition_labels = [
        f"{quarter_names[t]}->{quarter_names[t + 1]}"
        for t in range(len(quarter_names) - 1)
    ]

    crr_matrix = pd.DataFrame(
        crr_values,
        index=transition_labels,
        columns=es_names,
    )
    crr_matrix.index.name = "transition"

    transition_long = pd.DataFrame(rows)

    finite_crr = crr_values[np.isfinite(crr_values)]
    if finite_crr.size and np.any(
        (finite_crr < -EPS) | (finite_crr > 1.0 + EPS)
    ):
        raise ValueError("CRR contains values outside [0, 1].")

    return crr_matrix, transition_long


# =============================================================================
# 6. IBPA
# =============================================================================

def estimate_ibpa_for_element(
    es: str,
    nr_t: np.ndarray,
    crr_t1: np.ndarray,
    *,
    alpha_init: float = 0.0,
    beta_init: float = 0.0,
    tol: float = DEFAULT_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
) -> IBPAResult:
    """
    Apply the original iterative Beta moment-matching procedure to one ES.

    Initial values
    --------------
    h_t^(0) = CRR_m^{t+1} / NR_m^t, when NR_m^t != 0;
              0, otherwise.

    Iteration
    ---------
    mu    = mean(h)
    sigma = var(h)

    alpha = mu * [mu(1-mu)/sigma - 1]
    beta  = (1-mu) * [mu(1-mu)/sigma - 1]

    h_t = (CRR_m^{t+1} + alpha)
          / (NR_m^t + alpha + beta)

    Final
    -----
    CM_m   = mean_t(h_t)
    Cost_m = 1 / CM_m
    """
    nr_t = np.asarray(nr_t, dtype=float)
    crr_t1 = np.asarray(crr_t1, dtype=float)

    if nr_t.shape != crr_t1.shape:
        raise ValueError(
            f"{es}: NR_t and CRR_t1 must have identical shapes."
        )
    if nr_t.ndim != 1:
        raise ValueError(f"{es}: inputs must be one-dimensional.")

    valid = np.isfinite(nr_t) & np.isfinite(crr_t1)
    nr = nr_t[valid]
    crr = crr_t1[valid]

    if len(nr) == 0:
        return IBPAResult(
            es=es,
            alpha=np.nan,
            beta=np.nan,
            cm_by_transition=np.asarray([], dtype=float),
            raw_ratio=np.asarray([], dtype=float),
            cm=np.nan,
            cost=np.nan,
            cm_last_legacy=np.nan,
            mse=np.nan,
            iterations=0,
            converged=False,
            status="no_available_transitions",
        )

    if np.any((nr < -EPS) | (nr > 1.0 + EPS)):
        raise ValueError(f"{es}: NR_t contains values outside [0, 1].")
    if np.any((crr < -EPS) | (crr > 1.0 + EPS)):
        raise ValueError(f"{es}: CRR_t1 contains values outside [0, 1].")

    # Because CRR = max(NR_t - NR_t1, 0), CRR should not exceed NR_t
    # apart from floating-point tolerance.
    if np.any(crr > nr + 1e-10):
        bad = np.where(crr > nr + 1e-10)[0].tolist()
        raise ValueError(
            f"{es}: CRR_t1 exceeds NR_t for transitions {bad}."
        )

    raw_ratio = np.zeros_like(nr, dtype=float)
    nonzero = nr != 0.0
    raw_ratio[nonzero] = crr[nonzero] / nr[nonzero]

    h = raw_ratio.copy()

    alpha = float(alpha_init)
    beta = float(beta_init)

    converged = False
    status = "max_iter_reached"
    iterations = 0

    for iteration in range(1, max_iter + 1):
        iterations = iteration

        mu = float(np.mean(h))
        sigma = float(np.var(h, ddof=0))

        if not np.isfinite(mu) or not np.isfinite(sigma):
            status = "invalid_moments"
            break

        if sigma <= EPS:
            # This reproduces the original notebook's logical stopping point:
            # no cross-period variance is available for Beta moment matching.
            status = "zero_variance"
            break

        concentration = mu * (1.0 - mu) / sigma - 1.0

        # For a variable bounded in [0,1], the theoretical variance cannot
        # exceed mu(1-mu). A negative concentration therefore indicates a
        # numerical/invalid moment estimate.
        if not np.isfinite(concentration) or concentration < -1e-10:
            status = "invalid_beta_moments"
            break

        # Permit a tiny negative value caused only by floating-point error.
        concentration = max(concentration, 0.0)

        alpha_new = float(mu * concentration)
        beta_new = float((1.0 - mu) * concentration)

        denominator = nr + alpha_new + beta_new

        if np.any(np.abs(denominator) <= EPS):
            status = "zero_posterior_denominator"
            break

        h_new = (crr + alpha_new) / denominator

        if np.any(~np.isfinite(h_new)):
            status = "invalid_posterior"
            break

        if np.any((h_new < -1e-10) | (h_new > 1.0 + 1e-10)):
            status = "posterior_outside_unit_interval"
            break

        h_new = np.clip(h_new, 0.0, 1.0)

        delta = max(
            abs(alpha_new - alpha),
            abs(beta_new - beta),
        )

        # Update before the convergence check. The historical notebook checked
        # first, which could return the previous alpha/beta while h had already
        # been calculated using alpha_new/beta_new.
        alpha = alpha_new
        beta = beta_new
        h = h_new

        if delta < tol:
            converged = True
            status = "converged"
            break

    cm = float(np.mean(h)) if len(h) else np.nan
    cm_last_legacy = float(h[-1]) if len(h) else np.nan

    if np.isfinite(cm):
        cost = np.inf if cm <= 0.0 else float(1.0 / cm)
    else:
        cost = np.nan

    mse = (
        float(np.mean((raw_ratio - h) ** 2))
        if len(h)
        else np.nan
    )

    return IBPAResult(
        es=es,
        alpha=alpha,
        beta=beta,
        cm_by_transition=h,
        raw_ratio=raw_ratio,
        cm=cm,
        cost=cost,
        cm_last_legacy=cm_last_legacy,
        mse=mse,
        iterations=iterations,
        converged=converged,
        status=status,
    )


def run_ibpa_for_all_elements(
    transition_long: pd.DataFrame,
    *,
    tol: float,
    max_iter: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate IBPA separately for every service element."""
    required = {
        "ES",
        "period_t",
        "period_t1",
        "NR_t",
        "NR_t1",
        "CRR_t1",
        "available_transition",
    }
    missing = required.difference(transition_long.columns)
    if missing:
        raise KeyError(
            f"transition_long is missing columns: {sorted(missing)}"
        )

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    es_names = sorted(
        transition_long["ES"].dropna().astype(str).unique(),
        key=es_sort_key,
    )

    for es in es_names:
        group = (
            transition_long.loc[
                transition_long["ES"] == es
            ]
            .copy()
            .reset_index(drop=True)
        )

        valid_mask = (
            group["available_transition"].astype(bool)
            & pd.to_numeric(group["NR_t"], errors="coerce").notna()
            & pd.to_numeric(group["CRR_t1"], errors="coerce").notna()
        )

        nr_t = pd.to_numeric(
            group.loc[valid_mask, "NR_t"],
            errors="raise",
        ).to_numpy(dtype=float)

        crr_t1 = pd.to_numeric(
            group.loc[valid_mask, "CRR_t1"],
            errors="raise",
        ).to_numpy(dtype=float)

        result = estimate_ibpa_for_element(
            es=es,
            nr_t=nr_t,
            crr_t1=crr_t1,
            tol=tol,
            max_iter=max_iter,
        )

        summary_rows.append(
            {
                "ES": es,
                "alpha_Beta": result.alpha,
                "beta_Beta": result.beta,
                "CM": result.cm,
                "Cost": result.cost,
                "CM_last_legacy": result.cm_last_legacy,
                "MSE_raw_vs_calibrated": result.mse,
                "N_available_transitions": int(valid_mask.sum()),
                "iterations": result.iterations,
                "converged": result.converged,
                "status": result.status,
            }
        )

        valid_group = group.loc[valid_mask].reset_index(drop=True)

        if len(result.cm_by_transition) != len(valid_group):
            raise RuntimeError(
                f"{es}: posterior length does not match valid transitions."
            )

        for i, row in valid_group.iterrows():
            detail_rows.append(
                {
                    "ES": es,
                    "period_t": row["period_t"],
                    "period_t1": row["period_t1"],
                    "NR_t": float(row["NR_t"]),
                    "NR_t1": float(row["NR_t1"]),
                    "CRR_t1": float(row["CRR_t1"]),
                    "raw_improvement_ratio": float(result.raw_ratio[i]),
                    "CM_t": float(result.cm_by_transition[i]),
                    "alpha_Beta": result.alpha,
                    "beta_Beta": result.beta,
                    "CM_overall": result.cm,
                    "Cost": result.cost,
                }
            )

    summary = pd.DataFrame(summary_rows)
    details = pd.DataFrame(detail_rows)

    if not summary.empty:
        summary["_sort"] = summary["ES"].map(es_sort_key)
        summary = (
            summary
            .sort_values("_sort")
            .drop(columns="_sort")
            .reset_index(drop=True)
        )

    return summary, details


# =============================================================================
# 7. SAVE OUTPUTS
# =============================================================================

def save_results(
    *,
    output_file: Path,
    sentiment_counts_long: pd.DataFrame,
    positive_count_matrix: pd.DataFrame,
    negative_count_matrix: pd.DataFrame,
    negative_rate_matrix: pd.DataFrame,
    crr_matrix: pd.DataFrame,
    transition_long: pd.DataFrame,
    ibpa_details: pd.DataFrame,
    ibpa_summary: pd.DataFrame,
    run_configuration: dict[str, Any],
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    config_df = pd.DataFrame(
        [
            {"parameter": key, "value": value}
            for key, value in run_configuration.items()
        ]
    )

    with pd.ExcelWriter(
        output_file,
        engine="openpyxl",
    ) as writer:
        sentiment_counts_long.to_excel(
            writer,
            sheet_name="sentiment_counts_long",
            index=False,
        )
        positive_count_matrix.to_excel(
            writer,
            sheet_name="positive_count",
            index=True,
        )
        negative_count_matrix.to_excel(
            writer,
            sheet_name="negative_count",
            index=True,
        )
        negative_rate_matrix.to_excel(
            writer,
            sheet_name="negative_rate_NR",
            index=True,
        )
        crr_matrix.to_excel(
            writer,
            sheet_name="corrective_rate_CRR",
            index=True,
        )
        transition_long.to_excel(
            writer,
            sheet_name="transition_input",
            index=False,
        )
        ibpa_details.to_excel(
            writer,
            sheet_name="IBPA_transition_details",
            index=False,
        )
        ibpa_summary.to_excel(
            writer,
            sheet_name="IBPA_summary",
            index=False,
        )
        config_df.to_excel(
            writer,
            sheet_name="Configuration",
            index=False,
        )

    summary_csv = output_file.with_suffix(".csv")
    ibpa_summary.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nSaved IBPA workbook to: {output_file}")
    print(f"Saved IBPA summary CSV to: {summary_csv}")


# =============================================================================
# 8. MAIN WORKFLOW
# =============================================================================

def run(
    *,
    input_file: Path,
    output_file: Path,
    sheet_name: str | int,
    date_column: str,
    cluster_column: str,
    sentiment_column: str,
    positive_label: str,
    negative_label: str,
    start_date: str,
    end_date: str,
    missing_rate_policy: str,
    tol: float,
    max_iter: int,
) -> None:
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {input_file}"
        )

    print("Loading finalized AOCS data...")
    df = pd.read_excel(
        input_file,
        sheet_name=sheet_name,
    )
    
    if df.empty:
        raise ValueError("Input worksheet is empty.")
    
    print(f"Loaded rows: {len(df):,}")
    
    
    # ------------------------------------------------------------------
    # Construct binary sentiment orientation from the service-level
    # sentiment score.
    #
    # sentiment >= 6  -> positive
    # sentiment <  6  -> negative
    # ------------------------------------------------------------------
    if "sentiment" not in df.columns:
        raise KeyError(
            "Input data must contain the 'sentiment' column."
        )
    
    df["sentiment"] = pd.to_numeric(
        df["sentiment"],
        errors="coerce",
    )
    
    invalid_sentiment = df["sentiment"].isna()
    
    if invalid_sentiment.any():
        raise ValueError(
            f"{int(invalid_sentiment.sum()):,} rows contain missing or "
            "non-numeric sentiment scores."
        )
    
    df[sentiment_column] = np.where(
        df["sentiment"] >= 6.0,
        positive_label,
        negative_label,
    )
    
    print(
        "Sentiment orientation constructed using the threshold: "
        "sentiment >= 6 -> positive; sentiment < 6 -> negative."
    )
    
    print(
        df[sentiment_column]
        .value_counts()
        .to_string()
    )
    
    print("\n1. Constructing quarterly negative review rates NR...")
    (
        sentiment_counts_long,
        negative_rate_matrix,
        positive_count_matrix,
        negative_count_matrix,
    ) = prepare_quarterly_negative_rates(
        df,
        date_column=date_column,
        cluster_column=cluster_column,
        sentiment_column=sentiment_column,
        positive_label=positive_label,
        negative_label=negative_label,
        start_date=start_date,
        end_date=end_date,
        missing_rate_policy=missing_rate_policy,
    )

    print(
        f"   Quarters: {len(negative_rate_matrix)} "
        f"({negative_rate_matrix.index[0]} to "
        f"{negative_rate_matrix.index[-1]})"
    )
    print(
        f"   Service elements: {len(negative_rate_matrix.columns)}"
    )

    print("\n2. Calculating corrective response rates CRR...")
    crr_matrix, transition_long = (
        calculate_corrective_response_rates(
            negative_rate_matrix
        )
    )
    print(f"   Adjacent transitions: {len(crr_matrix)}")

    print("\n3. Running IBPA for each service element...")
    ibpa_summary, ibpa_details = run_ibpa_for_all_elements(
        transition_long,
        tol=tol,
        max_iter=max_iter,
    )

    n_converged = int(ibpa_summary["converged"].sum())
    print(
        f"   Converged: {n_converged}/"
        f"{len(ibpa_summary)} service elements"
    )

    status_counts = (
        ibpa_summary["status"]
        .value_counts(dropna=False)
        .to_dict()
    )
    print(f"   Status counts: {status_counts}")

    run_configuration = {
        "input_file": str(input_file),
        "sheet_name": sheet_name,
        "date_column": date_column,
        "cluster_column": cluster_column,
        "sentiment_column": sentiment_column,
        "positive_label": positive_label,
        "negative_label": negative_label,
        "start_date": start_date,
        "end_date": end_date,
        "missing_rate_policy": missing_rate_policy,
        "tol": tol,
        "max_iter": max_iter,
        "n_input_rows": len(df),
        "n_quarters": len(negative_rate_matrix),
        "n_transitions": len(crr_matrix),
        "n_service_elements": len(negative_rate_matrix.columns),
        "formal_CM_definition": "mean of converged CM_t across available transitions",
        "formal_Cost_definition": "1 / CM",
    }

    print("\n4. Saving outputs...")
    save_results(
        output_file=output_file,
        sentiment_counts_long=sentiment_counts_long,
        positive_count_matrix=positive_count_matrix,
        negative_count_matrix=negative_count_matrix,
        negative_rate_matrix=negative_rate_matrix,
        crr_matrix=crr_matrix,
        transition_long=transition_long,
        ibpa_details=ibpa_details,
        ibpa_summary=ibpa_summary,
        run_configuration=run_configuration,
    )

    print("\nTop rows of the final implementation-difficulty table:")
    display_cols = [
        "ES",
        "alpha_Beta",
        "beta_Beta",
        "CM",
        "Cost",
        "CM_last_legacy",
        "N_available_transitions",
        "converged",
        "status",
    ]
    print(
        ibpa_summary[display_cols]
        .head(20)
        .to_string(index=False)
    )


# =============================================================================
# 9. COMMAND-LINE INTERFACE
# =============================================================================

def parse_sheet(value: str) -> str | int:
    """Allow --sheet 0 or --sheet Sheet1."""
    stripped = str(value).strip()
    if re.fullmatch(r"\d+", stripped):
        return int(stripped)
    return stripped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct quarterly negative review rates, calculate corrective "
            "response rates, and estimate service-element implementation "
            "difficulty using IBPA."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Finalized AOCS-level Excel file containing review dates (AOCS_finalized.xlsx), "
            "service-element labels, and positive/negative sentiment labels."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/08_improvement_difficulty/"
            "ibpa_improvement_difficulty.xlsx"
        ),
        help="Output Excel workbook.",
    )
    parser.add_argument(
        "--sheet",
        type=parse_sheet,
        default=0,
        help="Input worksheet name or zero-based worksheet index.",
    )

    parser.add_argument(
        "--date-column",
        default=DEFAULT_DATE_COLUMN,
        help=f"Review-date column. Default: {DEFAULT_DATE_COLUMN!r}.",
    )
    parser.add_argument(
        "--cluster-column",
        default=DEFAULT_CLUSTER_COLUMN,
        help=(
            "Final service-element identifier column. "
            f"Default: {DEFAULT_CLUSTER_COLUMN!r}."
        ),
    )
    parser.add_argument(
        "--sentiment-column",
        default=DEFAULT_SENTIMENT_COLUMN,
        help=(
            "Positive/negative sentiment-label column. "
            f"Default: {DEFAULT_SENTIMENT_COLUMN!r}."
        ),
    )
    parser.add_argument(
        "--positive-label",
        default=DEFAULT_POSITIVE_LABEL,
        help=f"Positive sentiment label. Default: {DEFAULT_POSITIVE_LABEL!r}.",
    )
    parser.add_argument(
        "--negative-label",
        default=DEFAULT_NEGATIVE_LABEL,
        help=f"Negative sentiment label. Default: {DEFAULT_NEGATIVE_LABEL!r}.",
    )

    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help=(
            "First included review date. "
            f"Default: {DEFAULT_START_DATE}."
        ),
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
        help=(
            "Last included review date. "
            f"Default: {DEFAULT_END_DATE}."
        ),
    )

    parser.add_argument(
        "--missing-rate-policy",
        choices=["zero", "exclude"],
        default="zero",
        help=(
            "'zero' reproduces the original preprocessing by assigning NR=0 "
            "when an ES has no positive/negative observations in a quarter; "
            "'exclude' removes such adjacent transitions from that ES's IBPA."
        ),
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=DEFAULT_TOL,
        help=f"IBPA convergence tolerance. Default: {DEFAULT_TOL}.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_MAX_ITER,
        help=f"Maximum IBPA iterations. Default: {DEFAULT_MAX_ITER}.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run(
        input_file=args.input,
        output_file=args.output,
        sheet_name=args.sheet,
        date_column=args.date_column,
        cluster_column=args.cluster_column,
        sentiment_column=args.sentiment_column,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
        start_date=args.start_date,
        end_date=args.end_date,
        missing_rate_policy=args.missing_rate_policy,
        tol=args.tol,
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    main()
