# -*- coding: utf-8 -*-
"""
M1-I managerial-scenario sensitivity analysis
==============================================

Purpose
-------
For M1-I_threshold_MO_SHADE only:

1. Extract the already selected robust representative from every
   (K, rho, seed) Pareto-front file.
2. Summarize robust-plan outcomes across seeds for every (K, rho) scenario.
3. Calculate service-element selection probabilities and improvement magnitudes.
4. Classify service elements as core, scenario-sensitive, stable-secondary,
   or peripheral.
5. Draw scenario-result heatmaps and service-element heatmaps.

Expected directory structure
----------------------------
ROOT_DIR/
    comparison_five_algorithms_formal_K10/
        run_metrics_recomputed.csv  # preferred, if present
        run_metrics.csv             # fallback
        fronts/
            K10_rho0p30_M1-I_threshold_MO_SHADE_seed_42.csv
            ...
    comparison_five_algorithms_formal_K15/
        ...
    comparison_five_algorithms_formal_K20/
        ...

The script uses the representative solution already stored by the comparison
runner. It first matches recommended_solution_id to solution_id. If that ID is
not available, it falls back to matching the three raw objective values.

Outputs
-------
OUTPUT_DIR/
    M1I_sensitivity_summary.xlsx
    m1i_robust_solutions_by_seed.csv
    m1i_scenario_summary_long.csv
    m1i_scenario_summary_wide.csv
    m1i_element_summary_by_scenario.csv
    m1i_element_classification.csv
    figures/*.png
    figures/*.pdf
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# 1. User configuration
# =============================================================================

# Change this path to the parent directory containing the K10/K15/K20 folders.
ROOT_DIR = Path(
    r"D:\AAApaper\online_review\CODE\five_algorithm_checkpoint_fix"
)

OUTPUT_DIR = ROOT_DIR / "M1I_sensitivity_analysis_V3"
FIGURE_DIR = OUTPUT_DIR / "figures"

ALGORITHM = "M1-I_threshold_MO_SHADE"

# Leave as None to use every scenario found in the files.
INCLUDE_K: tuple[int, ...] | None = (10, 15, 20)
INCLUDE_RHO: tuple[float, ...] | None = (0.30, 0.40, 0.50)

METRIC_FILE_PRIORITY = (
    "run_metrics.csv",
)

# The semi-continuous action-domain threshold used by the optimization model.
MIN_ACTION_MAGNITUDE = 0.5
NUMERICAL_EPS = 1e-10

# Classification rules. These thresholds should be reported in the paper.
# Core:
#   selected by at least CORE_MIN_PROBABILITY of seeds in EVERY scenario.
# Scenario-sensitive:
#   maximum-minus-minimum scenario probability >= SENSITIVE_PROBABILITY_RANGE,
#   and selected with probability >= SENSITIVE_MIN_MAX_PROBABILITY somewhere.
CORE_MIN_PROBABILITY = 0.80
SENSITIVE_PROBABILITY_RANGE = 0.40
SENSITIVE_MIN_MAX_PROBABILITY = 0.50

# Used only to describe whether K or rho is the stronger sensitivity driver.
DRIVER_DIFFERENCE_TOLERANCE = 0.10

# Element-figure display control.
#
# The full element statistics are always retained in CSV/XLSX outputs. Figures
# exclude elements that are never selected in any scenario. All core and
# scenario-sensitive elements are retained first; remaining positions are filled
# by selected elements with the highest overall selection probability.
MAX_HEATMAP_ELEMENTS = 40
ANNOTATE_ELEMENT_HEATMAP_UP_TO = 30
EXCLUDE_NEVER_SELECTED_ELEMENTS_FROM_FIGURES = True
SHOW_ZERO_CELLS_AS_BLANK = True

# None: show core/sensitive elements first and then the most frequently selected
# remaining elements up to MAX_HEATMAP_ELEMENTS.
# To display only the two focal classes in the paper, use:
FIGURE_ELEMENT_CLASSES = ("Core", "Stable-secondary", "Scenario-sensitive")
#FIGURE_ELEMENT_CLASSES: tuple[str, ...] | None = None

# A bubble matrix combines both element indicators:
# marker area = selection probability;
# marker colour = conditional mean improvement magnitude.
GENERATE_COMBINED_ELEMENT_BUBBLE_MATRIX = True
BUBBLE_MIN_SIZE = 18.0
BUBBLE_MAX_SIZE = 520.0

# Figure export.
FIGURE_DPI = 300
SAVE_PDF = False


# =============================================================================
# 2. General helpers
# =============================================================================

FRONT_REGEX = re.compile(
    r"K(?P<K>\d+)_rho(?P<rho>\d+p\d+)_"
    r"M1-I_threshold_MO_SHADE_seed_(?P<seed>\d+)\.csv$"
)


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV while tolerating UTF-8 BOM."""
    return pd.read_csv(path, encoding="utf-8-sig")


def natural_es_key(value: str) -> tuple[str, int, str]:
    """Sort ES_2 before ES_10."""
    text = str(value)
    match = re.search(r"(\d+)$", text)
    if match:
        return text[: match.start()], int(match.group(1)), text
    return text, math.inf, text


def discover_k_directories(root: Path) -> list[Path]:
    """Accept either a parent directory or one individual K directory."""
    root = root.resolve()

    if (root / "fronts").is_dir():
        return [root]

    directories = sorted(
        (
            path
            for path in root.glob("comparison_five_algorithms_formal_K*")
            if path.is_dir() and (path / "fronts").is_dir()
        ),
        key=lambda path: natural_es_key(path.name),
    )

    if not directories:
        raise FileNotFoundError(
            "No comparison_five_algorithms_formal_K*/fronts directories "
            f"were found under: {root}"
        )
    return directories


def find_metric_file(k_directory: Path) -> Path:
    """Prefer a metrics file that retains robust-solution identifiers."""
    existing: list[Path] = []

    for filename in METRIC_FILE_PRIORITY:
        candidate = k_directory / filename
        if not candidate.exists():
            continue

        existing.append(candidate)
        header = pd.read_csv(
            candidate,
            encoding="utf-8-sig",
            nrows=0,
        ).columns

        has_solution_id = "recommended_solution_id" in header
        has_objective_triplet = {
            "recommended_reputation",
            "recommended_choice_gain",
            "recommended_effective_cost",
        }.issubset(set(header))

        if has_solution_id or has_objective_triplet:
            return candidate

    if existing:
        warnings.warn(
            "A run-metrics file was found but it does not retain the robust "
            "solution ID or the complete recommended objective triplet. "
            f"The first available file will be tried: {existing[0]}"
        )
        return existing[0]

    raise FileNotFoundError(
        f"No run-metrics file was found in {k_directory}. Expected one of: "
        f"{METRIC_FILE_PRIORITY}"
    )


def parse_front_filename(path: Path) -> tuple[int, float, int]:
    match = FRONT_REGEX.search(path.name)
    if not match:
        raise ValueError(f"Unexpected M1-I front filename: {path.name}")
    k = int(match.group("K"))
    rho = float(match.group("rho").replace("p", "."))
    seed = int(match.group("seed"))
    return k, rho, seed


def first_scalar(
    frame: pd.DataFrame,
    column: str,
    fallback: float | int,
) -> float | int:
    if column not in frame.columns or frame.empty:
        return fallback
    values = frame[column].dropna()
    return fallback if values.empty else values.iloc[0]


def scenario_allowed(k: int, rho: float) -> bool:
    if INCLUDE_K is not None and int(k) not in {int(value) for value in INCLUDE_K}:
        return False
    if INCLUDE_RHO is not None and not any(
        np.isclose(float(rho), float(value)) for value in INCLUDE_RHO
    ):
        return False
    return True


def load_all_metrics(k_directories: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for directory in k_directories:
        metric_path = find_metric_file(directory)
        frame = read_csv(metric_path)
        frame["_metric_source_file"] = str(metric_path)
        frames.append(frame)

    metrics = pd.concat(frames, ignore_index=True, sort=False)

    if "algorithm" not in metrics.columns:
        raise KeyError("The run-metrics files do not contain an 'algorithm' column.")

    metrics = metrics.loc[metrics["algorithm"].astype(str) == ALGORITHM].copy()

    if "status" in metrics.columns:
        metrics = metrics.loc[
            metrics["status"].astype(str).str.lower() == "success"
        ].copy()

    required = {"max_active_actions", "coverage_threshold", "seed"}
    missing = required - set(metrics.columns)
    if missing:
        raise KeyError(f"Run metrics are missing columns: {sorted(missing)}")

    metrics["max_active_actions"] = pd.to_numeric(
        metrics["max_active_actions"], errors="raise"
    ).astype(int)
    metrics["coverage_threshold"] = pd.to_numeric(
        metrics["coverage_threshold"], errors="raise"
    ).astype(float)
    metrics["seed"] = pd.to_numeric(metrics["seed"], errors="raise").astype(int)

    metrics = metrics.loc[
        [
            scenario_allowed(int(k), float(rho))
            for k, rho in zip(
                metrics["max_active_actions"],
                metrics["coverage_threshold"],
            )
        ]
    ].copy()

    duplicate_keys = [
        "max_active_actions",
        "coverage_threshold",
        "algorithm",
        "seed",
    ]
    duplicate_mask = metrics.duplicated(duplicate_keys, keep=False)
    if duplicate_mask.any():
        warnings.warn(
            "Duplicate run-metrics rows were detected. The last row for each "
            "(K, rho, algorithm, seed) key will be retained."
        )
        metrics = metrics.drop_duplicates(duplicate_keys, keep="last")

    return metrics.reset_index(drop=True)


# =============================================================================
# 3. Robust representative extraction
# =============================================================================

def match_metric_row(
    metrics: pd.DataFrame,
    *,
    k: int,
    rho: float,
    seed: int,
) -> pd.Series:
    mask = (
        (metrics["max_active_actions"] == int(k))
        & np.isclose(metrics["coverage_threshold"], float(rho))
        & (metrics["seed"] == int(seed))
    )
    matched = metrics.loc[mask]

    if matched.empty:
        raise KeyError(
            f"No successful M1-I metric row for K={k}, rho={rho:.2f}, seed={seed}."
        )
    if len(matched) > 1:
        warnings.warn(
            f"Multiple M1-I metric rows for K={k}, rho={rho:.2f}, seed={seed}; "
            "the last row will be used."
        )
    return matched.iloc[-1]


def select_by_solution_id(
    front: pd.DataFrame,
    metric_row: pd.Series,
) -> pd.Series | None:
    if (
        "recommended_solution_id" not in metric_row.index
        or "solution_id" not in front.columns
        or pd.isna(metric_row["recommended_solution_id"])
    ):
        return None

    target_id = int(float(metric_row["recommended_solution_id"]))
    ids = pd.to_numeric(front["solution_id"], errors="coerce")
    matched = front.loc[ids == target_id]

    if matched.empty:
        return None
    if len(matched) > 1:
        warnings.warn(
            f"solution_id={target_id} is not unique in a Pareto front; "
            "the first matching row will be used."
        )
    return matched.iloc[0]


def select_by_objective_values(
    front: pd.DataFrame,
    metric_row: pd.Series,
) -> pd.Series | None:
    mapping = {
        "recommended_reputation": "reputation_improvement",
        "recommended_choice_gain": "choice_probability_gain",
        "recommended_effective_cost": "effective_cost",
    }

    available = [
        (metric_column, front_column)
        for metric_column, front_column in mapping.items()
        if (
            metric_column in metric_row.index
            and front_column in front.columns
            and pd.notna(metric_row[metric_column])
        )
    ]
    if not available:
        return None

    distance = np.zeros(len(front), dtype=float)
    for metric_column, front_column in available:
        values = pd.to_numeric(front[front_column], errors="coerce").to_numpy()
        target = float(metric_row[metric_column])
        finite_values = values[np.isfinite(values)]
        scale = max(
            abs(target),
            float(np.std(finite_values)) if finite_values.size else 0.0,
            1e-12,
        )
        distance += ((values - target) / scale) ** 2

    if not np.isfinite(distance).any():
        return None

    position = int(np.nanargmin(distance))
    selected = front.iloc[position]

    # Warn rather than silently accepting a materially different point.
    for metric_column, front_column in available:
        actual = float(selected[front_column])
        target = float(metric_row[metric_column])
        if not np.isclose(actual, target, rtol=1e-7, atol=1e-9):
            warnings.warn(
                "Representative matching used the nearest objective vector, "
                f"but {front_column} differs: selected={actual}, target={target}."
            )

    return selected


def extract_robust_solutions(
    k_directories: Iterable[Path],
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    selected_rows: list[pd.Series] = []
    seen_keys: set[tuple[int, float, int]] = set()

    for directory in k_directories:
        front_files = sorted(
            (directory / "fronts").glob(
                f"*_{ALGORITHM}_seed_*.csv"
            )
        )

        for path in front_files:
            filename_k, filename_rho, filename_seed = parse_front_filename(path)
            front = read_csv(path)

            k = int(
                first_scalar(
                    front,
                    "max_active_actions",
                    filename_k,
                )
            )
            rho = float(
                first_scalar(
                    front,
                    "coverage_threshold",
                    filename_rho,
                )
            )
            seed = int(first_scalar(front, "seed", filename_seed))

            if not scenario_allowed(k, rho):
                continue

            key = (k, round(rho, 10), seed)
            if key in seen_keys:
                warnings.warn(
                    f"Duplicate front detected for K={k}, rho={rho:.2f}, "
                    f"seed={seed}; file skipped: {path}"
                )
                continue
            seen_keys.add(key)

            if front.empty:
                raise RuntimeError(f"Empty Pareto front: {path}")

            metric_row = match_metric_row(
                metrics,
                k=k,
                rho=rho,
                seed=seed,
            )

            selected = select_by_solution_id(front, metric_row)
            selection_method = "recommended_solution_id"

            if selected is None:
                selected = select_by_objective_values(front, metric_row)
                selection_method = "nearest_raw_objective_vector"

            if selected is None:
                raise RuntimeError(
                    "Unable to identify the stored robust representative for "
                    f"K={k}, rho={rho:.2f}, seed={seed}. "
                    "The run-metrics row needs recommended_solution_id or the "
                    "three recommended raw objective columns."
                )

            selected = selected.copy()
            selected["max_active_actions"] = int(k)
            selected["coverage_threshold"] = float(rho)
            selected["seed"] = int(seed)
            selected["algorithm"] = ALGORITHM
            selected["_front_source_file"] = str(path)
            selected["_selection_method"] = selection_method
            selected_rows.append(selected)

    if not selected_rows:
        raise RuntimeError("No M1-I robust representative solutions were extracted.")

    result = pd.DataFrame(selected_rows)

    # Derive choice gain in percentage points when the front stores only probability.
    if (
        "choice_probability_gain_pp" not in result.columns
        and "choice_probability_gain" in result.columns
    ):
        result["choice_probability_gain_pp"] = (
            100.0
            * pd.to_numeric(
                result["choice_probability_gain"],
                errors="coerce",
            )
        )

    result = result.sort_values(
        ["max_active_actions", "coverage_threshold", "seed"]
    ).reset_index(drop=True)

    return result


# =============================================================================
# 4. Scenario-level robust-plan summary
# =============================================================================

ROBUST_OUTCOME_METRICS = {
    "reputation_improvement": "Reputation improvement",
    "choice_probability_gain_pp": "Choice-probability gain (pp)",
    "effective_cost": "Effective cost",
    "n_active_actions": "Number of active actions",
    "high_priority_coverage": "High-priority coverage",
    "priority_alignment": "Priority alignment",
}


def summarize_robust_outcomes(
    robust: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []

    for (k, rho), group in robust.groupby(
        ["max_active_actions", "coverage_threshold"],
        sort=True,
    ):
        for metric, label in ROBUST_OUTCOME_METRICS.items():
            if metric not in group.columns:
                continue

            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue

            rows.append(
                {
                    "max_active_actions": int(k),
                    "coverage_threshold": float(rho),
                    "metric": metric,
                    "metric_label": label,
                    "n_seeds": int(group["seed"].nunique()),
                    "mean": float(values.mean()),
                    "std": (
                        float(values.std(ddof=1))
                        if len(values) > 1
                        else 0.0
                    ),
                    "median": float(values.median()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            )

    long_summary = pd.DataFrame(rows)

    if long_summary.empty:
        return long_summary, pd.DataFrame()

    wide_parts: list[pd.DataFrame] = []
    for statistic in ("mean", "std", "median", "q25", "q75"):
        part = long_summary.pivot_table(
            index=["max_active_actions", "coverage_threshold", "n_seeds"],
            columns="metric",
            values=statistic,
            aggfunc="first",
        )
        part.columns = [f"{column}_{statistic}" for column in part.columns]
        wide_parts.append(part)

    wide_summary = pd.concat(wide_parts, axis=1).reset_index()
    wide_summary = wide_summary.sort_values(
        ["max_active_actions", "coverage_threshold"]
    ).reset_index(drop=True)

    return long_summary, wide_summary


# =============================================================================
# 5. Element-level probabilities and magnitudes
# =============================================================================

def identify_action_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in frame.columns
        if str(column).startswith("x_")
    ]
    if not columns:
        raise KeyError(
            "No service-element action columns with prefix 'x_' were found."
        )
    return sorted(columns, key=natural_es_key)


def summarize_elements_by_scenario(
    robust: pd.DataFrame,
    action_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for (k, rho), group in robust.groupby(
        ["max_active_actions", "coverage_threshold"],
        sort=True,
    ):
        n_seeds = int(group["seed"].nunique())

        for action_column in action_columns:
            values = pd.to_numeric(
                group[action_column],
                errors="coerce",
            ).fillna(0.0).to_numpy(dtype=float)

            selected = values >= (
                float(MIN_ACTION_MAGNITUDE) - NUMERICAL_EPS
            )
            active_values = values[selected]

            rows.append(
                {
                    "max_active_actions": int(k),
                    "coverage_threshold": float(rho),
                    "scenario": f"({int(k)},{float(rho):.1f})",
                    "ES": action_column.removeprefix("x_"),
                    "action_column": action_column,
                    "n_seeds": n_seeds,
                    "selection_count": int(selected.sum()),
                    "selection_probability": float(selected.mean()),
                    # Includes zeros and captures expected overall allocation.
                    "mean_improvement_all_seeds": float(values.mean()),
                    "std_improvement_all_seeds": (
                        float(np.std(values, ddof=1))
                        if len(values) > 1
                        else 0.0
                    ),
                    # Conditional magnitude answers: how much, once selected?
                    "mean_improvement_when_selected": (
                        float(active_values.mean())
                        if active_values.size
                        else np.nan
                    ),
                    "median_improvement_when_selected": (
                        float(np.median(active_values))
                        if active_values.size
                        else np.nan
                    ),
                    "q25_improvement_when_selected": (
                        float(np.quantile(active_values, 0.25))
                        if active_values.size
                        else np.nan
                    ),
                    "q75_improvement_when_selected": (
                        float(np.quantile(active_values, 0.75))
                        if active_values.size
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def endpoint_change(
    element_group: pd.DataFrame,
    *,
    axis: str,
) -> float:
    """Average high-end minus low-end selection probability."""
    grouped = (
        element_group.groupby(axis, sort=True)["selection_probability"]
        .mean()
        .sort_index()
    )
    if len(grouped) < 2:
        return np.nan
    return float(grouped.iloc[-1] - grouped.iloc[0])


def classify_elements(
    element_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for es, group in element_summary.groupby("ES", sort=False):
        probability = pd.to_numeric(
            group["selection_probability"],
            errors="coerce",
        ).dropna()

        mean_probability = float(probability.mean())
        minimum_probability = float(probability.min())
        maximum_probability = float(probability.max())
        probability_range = maximum_probability - minimum_probability

        delta_k = endpoint_change(
            group,
            axis="max_active_actions",
        )
        delta_rho = endpoint_change(
            group,
            axis="coverage_threshold",
        )

        if minimum_probability >= CORE_MIN_PROBABILITY:
            classification = "Core"
        elif (
            probability_range >= SENSITIVE_PROBABILITY_RANGE
            and maximum_probability >= SENSITIVE_MIN_MAX_PROBABILITY
        ):
            classification = "Scenario-sensitive"
        elif mean_probability >= 0.50:
            classification = "Stable-secondary"
        else:
            classification = "Peripheral"

        if not np.isfinite(delta_k) and not np.isfinite(delta_rho):
            dominant_driver = "Insufficient scenarios"
        elif not np.isfinite(delta_rho):
            dominant_driver = "K"
        elif not np.isfinite(delta_k):
            dominant_driver = "rho"
        elif abs(delta_k) > abs(delta_rho) + DRIVER_DIFFERENCE_TOLERANCE:
            dominant_driver = "K"
        elif abs(delta_rho) > abs(delta_k) + DRIVER_DIFFERENCE_TOLERANCE:
            dominant_driver = "rho"
        else:
            dominant_driver = "Mixed"

        rows.append(
            {
                "ES": es,
                "classification": classification,
                "mean_selection_probability": mean_probability,
                "minimum_selection_probability": minimum_probability,
                "maximum_selection_probability": maximum_probability,
                "selection_probability_range": probability_range,
                "selection_probability_std_across_scenarios": (
                    float(probability.std(ddof=1))
                    if len(probability) > 1
                    else 0.0
                ),
                "high_K_minus_low_K_probability": delta_k,
                "high_rho_minus_low_rho_probability": delta_rho,
                "dominant_sensitivity_driver": dominant_driver,
                "mean_improvement_all_scenarios": float(
                    pd.to_numeric(
                        group["mean_improvement_all_seeds"],
                        errors="coerce",
                    ).mean()
                ),
                "mean_conditional_improvement_all_scenarios": float(
                    pd.to_numeric(
                        group["mean_improvement_when_selected"],
                        errors="coerce",
                    ).mean()
                ),
            }
        )

    classification = pd.DataFrame(rows)

    class_order = {
        "Core": 0,
        "Stable-secondary": 1,
        "Scenario-sensitive": 2,
        "Peripheral": 3,
    }
    classification["_class_order"] = classification[
        "classification"
    ].map(class_order)

    classification = classification.sort_values(
        [
            "_class_order",
            "mean_selection_probability",
            "selection_probability_range",
            "ES",
        ],
        ascending=[True, False, False, True],
    ).drop(columns="_class_order").reset_index(drop=True)

    return classification


# =============================================================================
# 6. Plotting
# =============================================================================

def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / f"{stem}.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    if SAVE_PDF:
        fig.savefig(
            FIGURE_DIR / f"{stem}.pdf",
            bbox_inches="tight",
        )
    plt.close(fig)


def annotate_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    format_string: str,
) -> None:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return

    midpoint = (float(finite.min()) + float(finite.max())) / 2.0

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if not np.isfinite(value):
                continue
            # Let Matplotlib choose the text colour through its default cycle:
            # no explicit colour is set.
            ax.text(
                column,
                row,
                format(value, format_string),
                ha="center",
                va="center",
                fontsize=8,
                alpha=1.0 if value >= midpoint else 0.9,
            )


def plot_scenario_metric_interaction(
    scenario_summary_long: pd.DataFrame,
    *,
    metric: str,
    title: str,
    filename: str,
    y_label: str,
    uncertainty: str = "std",
) -> None:
    """Plot a two-factor interaction chart.

    The x-axis represents K and each line represents one rho level. Therefore,
    the chart displays the main effect of K, the vertical shift associated with
    rho, and any K-by-rho interaction. Non-parallel or crossing lines indicate
    that the effect of K depends on rho.

    Parameters
    ----------
    uncertainty:
        "std" draws mean +/- one standard deviation across seeds.
        "iqr" draws asymmetric q25/q75 error bars around the median.
    """
    subset = scenario_summary_long.loc[
        scenario_summary_long["metric"] == metric
    ].copy()

    if subset.empty:
        warnings.warn(
            f"Scenario metric not found and will not be plotted: {metric}"
        )
        return

    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    for rho, group in subset.groupby(
        "coverage_threshold",
        sort=True,
    ):
        ordered = group.sort_values("max_active_actions")
        x = ordered["max_active_actions"].to_numpy(dtype=float)

        if uncertainty == "iqr":
            center = ordered["median"].to_numpy(dtype=float)
            lower = center - ordered["q25"].to_numpy(dtype=float)
            upper = ordered["q75"].to_numpy(dtype=float) - center
            y_error = np.vstack(
                [
                    np.maximum(lower, 0.0),
                    np.maximum(upper, 0.0),
                ]
            )
        elif uncertainty == "std":
            center = ordered["mean"].to_numpy(dtype=float)
            y_error = ordered["std"].to_numpy(dtype=float)
        else:
            raise ValueError("uncertainty must be either 'std' or 'iqr'.")

        ax.errorbar(
            x,
            center,
            yerr=y_error,
            marker="o",
            linewidth=1.8,
            capsize=3.5,
            label=f"rho = {float(rho):.2f}",
        )

    k_values = sorted(
        pd.to_numeric(
            subset["max_active_actions"],
            errors="coerce",
        ).dropna().unique()
    )
    ax.set_xticks(k_values)
    ax.set_xlabel("Maximum active actions (K)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(title="Coverage threshold")
    save_figure(fig, filename)


def plot_scenario_metric_heatmap(
    scenario_summary_long: pd.DataFrame,
    *,
    metric: str,
    title: str,
    filename: str,
    number_format: str,
) -> None:
    subset = scenario_summary_long.loc[
        scenario_summary_long["metric"] == metric
    ]

    if subset.empty:
        warnings.warn(f"Scenario metric not found and will not be plotted: {metric}")
        return

    matrix = subset.pivot_table(
        index="max_active_actions",
        columns="coverage_threshold",
        values="mean",
        aggfunc="first",
    ).sort_index().sort_index(axis=1)

    values = matrix.to_numpy(dtype=float)

    fig, ax = plt.subplots(
        figsize=(
            max(6.5, 1.3 * len(matrix.columns) + 2.5),
            max(4.5, 0.9 * len(matrix.index) + 2.0),
        )
    )
    image = ax.imshow(values, aspect="auto")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels([f"{value:.2f}" for value in matrix.columns])
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels([str(int(value)) for value in matrix.index])
    ax.set_xlabel("Coverage threshold (rho)")
    ax.set_ylabel("Maximum active actions (K)")
    ax.set_title(title)

    annotate_heatmap(
        ax,
        values,
        format_string=number_format,
    )
    save_figure(fig, filename)


def selected_heatmap_elements(
    classification: pd.DataFrame,
) -> list[str]:
    """Choose a compact, substantively relevant element set for figures."""
    candidates = classification.copy()

    if EXCLUDE_NEVER_SELECTED_ELEMENTS_FROM_FIGURES:
        candidates = candidates.loc[
            pd.to_numeric(
                candidates["maximum_selection_probability"],
                errors="coerce",
            ).fillna(0.0) > NUMERICAL_EPS
        ].copy()

    if FIGURE_ELEMENT_CLASSES is not None:
        candidates = candidates.loc[
            candidates["classification"].isin(FIGURE_ELEMENT_CLASSES)
        ].copy()

    priority = candidates.loc[
        candidates["classification"].isin(
            ["Core", "Stable-secondary", "Scenario-sensitive"]
        ),
        "ES",
    ].tolist()

    remaining = candidates.loc[
        ~candidates["ES"].isin(priority)
    ].sort_values(
        [
            "mean_selection_probability",
            "maximum_selection_probability",
            "selection_probability_range",
        ],
        ascending=[False, False, False],
    )["ES"].tolist()

    # Never discard core or scenario-sensitive elements merely to obey a
    # display cap. The cap applies only to supplementary elements.
    if len(priority) >= MAX_HEATMAP_ELEMENTS:
        return priority

    return priority + remaining[: MAX_HEATMAP_ELEMENTS - len(priority)]


def scenario_column_order(
    element_summary: pd.DataFrame,
) -> list[str]:
    scenarios = (
        element_summary[
            ["max_active_actions", "coverage_threshold", "scenario"]
        ]
        .drop_duplicates()
        .sort_values(["max_active_actions", "coverage_threshold"])
    )
    return scenarios["scenario"].tolist()


def element_display_labels(
    elements: list[str],
    classification: pd.DataFrame,
) -> list[str]:
    """Append compact class tags to element labels."""
    class_lookup = classification.set_index("ES")["classification"].to_dict()
    tag_lookup = {
        "Core": "[C]",
        "Stable-secondary": "[SC]",
        "Scenario-sensitive": "[SS]",
        "Peripheral": "[P]",
    }
    return [
        f"{element} {tag_lookup.get(class_lookup.get(element, ''), '')}".strip()
        for element in elements
    ]


def plot_element_heatmap(
    element_summary: pd.DataFrame,
    classification: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    filename: str,
    number_format: str,
) -> None:
    elements = selected_heatmap_elements(classification)
    if not elements:
        warnings.warn("No selected elements are available for the heatmap.")
        return

    columns = scenario_column_order(element_summary)

    matrix = element_summary.pivot_table(
        index="ES",
        columns="scenario",
        values=value_column,
        aggfunc="first",
    )
    matrix = matrix.reindex(index=elements, columns=columns)

    values = matrix.to_numpy(dtype=float)

    if SHOW_ZERO_CELLS_AS_BLANK:
        values = values.copy()
        values[np.isclose(values, 0.0, atol=NUMERICAL_EPS)] = np.nan

    masked_values = np.ma.masked_invalid(values)

    fig, ax = plt.subplots(
        figsize=(
            max(10.0, 0.95 * len(columns) + 3.0),
            max(5.5, 0.36 * len(matrix.index) + 2.5),
        )
    )
    image = ax.imshow(masked_values, aspect="auto")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(
        matrix.columns,
        rotation=45,
        ha="right",
    )
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(
        element_display_labels(elements, classification)
    )
    #ax.set_xlabel("Managerial scenario")
    ax.set_ylabel("Service element")
    #ax.set_title(title + "\n(blank cells indicate zero selection/improvement)")
    ax.legend(
        title=title,
        loc="lower left",
        fontsize=12,
        bbox_to_anchor=(1.0, -0.101),
        frameon=False,
        borderaxespad=0
    )
    
    plt.show()


    if len(matrix.index) <= ANNOTATE_ELEMENT_HEATMAP_UP_TO:
        annotate_heatmap(
            ax,
            values,
            format_string=number_format,
        )

    save_figure(fig, filename)


def plot_element_bubble_matrix(
    element_summary: pd.DataFrame,
    classification: pd.DataFrame,
    *,
    filename: str = "element_probability_magnitude_bubble",
) -> None:
    """Combine selection probability and conditional magnitude in one matrix.

    Marker area represents the selection probability. Marker colour represents
    the mean improvement magnitude conditional on selection. Zero-probability
    cells are omitted, so sparse results do not occupy visual attention.
    """
    elements = selected_heatmap_elements(classification)
    if not elements:
        warnings.warn("No selected elements are available for the bubble matrix.")
        return

    columns = scenario_column_order(element_summary)
    subset = element_summary.loc[
        element_summary["ES"].isin(elements)
        & element_summary["scenario"].isin(columns)
    ].copy()

    element_position = {
        element: position
        for position, element in enumerate(elements)
    }
    scenario_position = {
        scenario: position
        for position, scenario in enumerate(columns)
    }

    subset["x_position"] = subset["scenario"].map(scenario_position)
    subset["y_position"] = subset["ES"].map(element_position)

    probability = pd.to_numeric(
        subset["selection_probability"],
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)

    magnitude = pd.to_numeric(
        subset["mean_improvement_when_selected"],
        errors="coerce",
    ).to_numpy(dtype=float)

    visible = (
        probability > NUMERICAL_EPS
    ) & np.isfinite(magnitude)

    if not visible.any():
        warnings.warn("No nonzero element selections exist for the bubble matrix.")
        return

    sizes = (
        BUBBLE_MIN_SIZE
        + (BUBBLE_MAX_SIZE - BUBBLE_MIN_SIZE)
        * probability[visible]
    )

    fig, ax = plt.subplots(
        figsize=(
            max(10.0, 0.95 * len(columns) + 3.0),
            max(5.5, 0.38 * len(elements) + 2.5),
        )
    )

    scatter = ax.scatter(
        subset.loc[visible, "x_position"].to_numpy(dtype=float),
        subset.loc[visible, "y_position"].to_numpy(dtype=float),
        s=sizes,
        c=magnitude[visible],
        alpha=0.82,
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Mean improvement when selected")

    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(
        columns,
        rotation=45,
        ha="right",
    )
    ax.set_yticks(np.arange(len(elements)))
    ax.set_yticklabels(
        element_display_labels(elements, classification)
    )
    ax.set_xlim(-0.6, len(columns) - 0.4)
    ax.set_ylim(len(elements) - 0.4, -0.6)
    ax.set_xlabel("Managerial scenario")
    ax.set_ylabel("Service element")
    ax.set_title(
        "M1-I element sensitivity: selection probability and improvement magnitude"
        "\n(marker area = selection probability; colour = conditional mean improvement)"
    )
    ax.grid(True, alpha=0.15)

    # Size legend based on representative probabilities.
    legend_probabilities = [0.25, 0.50, 0.75, 1.00]
    legend_handles = [
        ax.scatter(
            [],
            [],
            s=(
                BUBBLE_MIN_SIZE
                + (BUBBLE_MAX_SIZE - BUBBLE_MIN_SIZE) * probability_value
            ),
            alpha=0.82,
            label=f"{probability_value:.0%}",
        )
        for probability_value in legend_probabilities
    ]
    ax.legend(
        handles=legend_handles,
        title="Selection probability",
        loc="upper left",
        bbox_to_anchor=(1.13, 1.0),
    )

    save_figure(fig, filename)


def generate_all_figures(
    scenario_summary_long: pd.DataFrame,
    element_summary: pd.DataFrame,
    classification: pd.DataFrame,
) -> None:
    # ------------------------------------------------------------------
    # Primary objective results: two-factor interaction line charts.
    # x = K; one line per rho; error bars = mean +/- one seed-level std.
    # ------------------------------------------------------------------
    objective_line_plots = [
        (
            "reputation_improvement",
            "M1-I robust-plan reputation improvement across managerial scenarios",
            "objective_reputation_interaction",
            "Reputation improvement",
        ),
        (
            "choice_probability_gain_pp",
            "M1-I robust-plan choice-probability gain across managerial scenarios",
            "objective_choice_gain_interaction",
            "Choice-probability gain (percentage points)",
        ),
        (
            "effective_cost",
            "M1-I robust-plan effective cost across managerial scenarios",
            "objective_cost_interaction",
            "Effective cost",
        ),
    ]

    for metric, title, filename, y_label in objective_line_plots:
        plot_scenario_metric_interaction(
            scenario_summary_long,
            metric=metric,
            title=title,
            filename=filename,
            y_label=y_label,
            uncertainty="std",
        )

    # Structural/constraint indicators remain compact as 3-by-3 heatmaps.
    supplementary_scenario_heatmaps = [
        (
            "n_active_actions",
            "M1-I robust-plan number of active actions",
            "scenario_active_actions",
            ".1f",
        ),
        (
            "high_priority_coverage",
            "M1-I robust-plan high-priority coverage",
            "scenario_high_priority_coverage",
            ".2f",
        ),
        (
            "priority_alignment",
            "M1-I robust-plan priority alignment",
            "scenario_priority_alignment",
            ".2f",
        ),
    ]

    for metric, title, filename, number_format in supplementary_scenario_heatmaps:
        plot_scenario_metric_heatmap(
            scenario_summary_long,
            metric=metric,
            title=title,
            filename=filename,
            number_format=number_format,
        )

    # ------------------------------------------------------------------
    # Element-level sensitivity. Never-selected elements are excluded.
    # ------------------------------------------------------------------
    plot_element_heatmap(
        element_summary,
        classification,
        value_column="selection_probability",
        title="selection probability",
        filename="element_selection_probability_focused",
        number_format=".2f",
    )

    plot_element_heatmap(
        element_summary,
        classification,
        value_column="mean_improvement_when_selected",
        title="mean improvement magnitude",
        filename="element_mean_improvement_when_selected_focused",
        number_format=".2f",
    )

    if GENERATE_COMBINED_ELEMENT_BUBBLE_MATRIX:
        plot_element_bubble_matrix(
            element_summary,
            classification,
        )


# =============================================================================
# 7. Output workbook
# =============================================================================

def set_excel_layout(writer: pd.ExcelWriter) -> None:
    """Apply basic readable layout without changing the data."""
    for worksheet in writer.book.worksheets:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

        for column_cells in worksheet.columns:
            values = [
                "" if cell.value is None else str(cell.value)
                for cell in column_cells
            ]
            width = min(
                max(max((len(value) for value in values), default=0) + 2, 10),
                45,
            )
            worksheet.column_dimensions[
                column_cells[0].column_letter
            ].width = width


def save_outputs(
    robust: pd.DataFrame,
    scenario_long: pd.DataFrame,
    scenario_wide: pd.DataFrame,
    element_summary: pd.DataFrame,
    classification: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    robust.to_csv(
        OUTPUT_DIR / "m1i_robust_solutions_by_seed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    scenario_long.to_csv(
        OUTPUT_DIR / "m1i_scenario_summary_long.csv",
        index=False,
        encoding="utf-8-sig",
    )
    scenario_wide.to_csv(
        OUTPUT_DIR / "m1i_scenario_summary_wide.csv",
        index=False,
        encoding="utf-8-sig",
    )
    element_summary.to_csv(
        OUTPUT_DIR / "m1i_element_summary_by_scenario.csv",
        index=False,
        encoding="utf-8-sig",
    )
    classification.to_csv(
        OUTPUT_DIR / "m1i_element_classification.csv",
        index=False,
        encoding="utf-8-sig",
    )

    workbook_path = OUTPUT_DIR / "M1I_sensitivity_summary.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        robust.to_excel(
            writer,
            sheet_name="Robust_by_seed",
            index=False,
        )
        scenario_wide.to_excel(
            writer,
            sheet_name="Scenario_summary",
            index=False,
        )
        scenario_long.to_excel(
            writer,
            sheet_name="Scenario_long",
            index=False,
        )
        element_summary.to_excel(
            writer,
            sheet_name="Element_scenario",
            index=False,
        )
        classification.to_excel(
            writer,
            sheet_name="Element_class",
            index=False,
        )
        set_excel_layout(writer)


# =============================================================================
# 8. Validation and main
# =============================================================================

def validate_scenario_seed_counts(robust: pd.DataFrame) -> None:
    counts = (
        robust.groupby(
            ["max_active_actions", "coverage_threshold"]
        )["seed"]
        .nunique()
        .rename("n_seeds")
        .reset_index()
    )

    if counts["n_seeds"].nunique() > 1:
        warnings.warn(
            "The number of successful seeds differs across scenarios:\n"
            + counts.to_string(index=False)
        )

    duplicate_mask = robust.duplicated(
        ["max_active_actions", "coverage_threshold", "seed"],
        keep=False,
    )
    if duplicate_mask.any():
        duplicates = robust.loc[
            duplicate_mask,
            ["max_active_actions", "coverage_threshold", "seed"],
        ]
        raise RuntimeError(
            "Duplicate robust solutions exist for the same scenario and seed:\n"
            + duplicates.to_string(index=False)
        )


def print_compact_report(
    robust: pd.DataFrame,
    classification: pd.DataFrame,
) -> None:
    scenarios = robust[
        ["max_active_actions", "coverage_threshold"]
    ].drop_duplicates()

    print("\nM1-I sensitivity analysis completed.")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Robust solutions: {len(robust)}")
    print(
        "Seeds per scenario: "
        + str(
            robust.groupby(
                ["max_active_actions", "coverage_threshold"]
            )["seed"].nunique().to_dict()
        )
    )

    print("\nElement classification:")
    print(
        classification["classification"]
        .value_counts()
        .reindex(
            [
                "Core",
                "Stable-secondary",
                "Scenario-sensitive",
                "Peripheral",
            ],
            fill_value=0,
        )
        .to_string()
    )

    core = classification.loc[
        classification["classification"] == "Core",
        "ES",
    ].tolist()
    sensitive = classification.loc[
        classification["classification"] == "Scenario-sensitive",
        "ES",
    ].tolist()
    secondary = classification.loc[
        classification["classification"] == "Stable-secondary",
        "ES",
    ].tolist()

    print("\nCore elements:")
    print(" | ".join(core) if core else "(none under the configured threshold)")
    
    print("\nSecondary elements:")
    print(" | ".join(secondary) if core else "(none under the configured threshold)")

    print("\nScenario-sensitive elements:")
    print(
        " | ".join(sensitive)
        if sensitive
        else "(none under the configured threshold)"
    )

    print(f"\nOutput directory: {OUTPUT_DIR}")


def main() -> None:
    k_directories = discover_k_directories(ROOT_DIR)
    metrics = load_all_metrics(k_directories)

    robust = extract_robust_solutions(
        k_directories,
        metrics,
    )
    validate_scenario_seed_counts(robust)

    action_columns = identify_action_columns(robust)

    scenario_long, scenario_wide = summarize_robust_outcomes(robust)
    element_summary = summarize_elements_by_scenario(
        robust,
        action_columns,
    )
    classification = classify_elements(element_summary)

    save_outputs(
        robust,
        scenario_long,
        scenario_wide,
        element_summary,
        classification,
    )
    generate_all_figures(
        scenario_long,
        element_summary,
        classification,
    )
    print_compact_report(
        robust,
        classification,
    )


#if __name__ == "__main__":
#    main()


k_directories = discover_k_directories(ROOT_DIR)
metrics = load_all_metrics(k_directories)

robust = extract_robust_solutions(
    k_directories,
    metrics,
)
validate_scenario_seed_counts(robust)

action_columns = identify_action_columns(robust)

scenario_long, scenario_wide = summarize_robust_outcomes(robust)
element_summary = summarize_elements_by_scenario(
    robust,
    action_columns,
)
classification = classify_elements(element_summary)


save_outputs(
    robust,
    scenario_long,
    scenario_wide,
    element_summary,
    classification,
)
generate_all_figures(
    scenario_long,
    element_summary,
    classification,
)
print_compact_report(
    robust,
    classification,
)


def plot_element_heatmap(
    element_summary: pd.DataFrame,
    classification: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    filename: str,
    number_format: str,
) -> None:
    elements = selected_heatmap_elements(classification)
    if not elements:
        warnings.warn("No selected elements are available for the heatmap.")
        return

    columns = scenario_column_order(element_summary)

    matrix = element_summary.pivot_table(
        index="ES",
        columns="scenario",
        values=value_column,
        aggfunc="first",
    )
    matrix = matrix.reindex(
        index=elements,
        columns=columns,
    )

    values = matrix.to_numpy(dtype=float)

    if SHOW_ZERO_CELLS_AS_BLANK:
        values = values.copy()
        values[np.isclose(
            values,
            0.0,
            atol=NUMERICAL_EPS,
        )] = np.nan

    masked_values = np.ma.masked_invalid(values)

    fig, ax = plt.subplots(
        figsize=(
            max(10.0, 0.95 * len(columns) + 3.0),
            max(5.5, 0.36 * len(matrix.index) + 2.5),
        )
    )

    image = ax.imshow(
        masked_values,
        aspect="auto",
    )


    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(
        matrix.columns,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(
        element_display_labels(elements, classification)
    )

    ax.set_ylabel("Service element")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(
        title,
        fontsize=12,
        rotation=90,
        labelpad=12,
    )

    if len(matrix.index) <= ANNOTATE_ELEMENT_HEATMAP_UP_TO:
        annotate_heatmap(
            ax,
            values,
            format_string=number_format,
        )

    # 给旋转后的横轴标签和右下角名称留出空间
    fig.tight_layout()

    save_figure(fig, filename)

    # 如果 save_figure() 内部没有关闭图像，可以显式关闭
    plt.close(fig)
    

plot_element_heatmap(
    element_summary,
    classification,
    value_column="selection_probability",
    title="selection probability",
    filename="element_selection_probability_focused",
    number_format=".2f",
)

plot_element_heatmap(
    element_summary,
    classification,
    value_column="mean_improvement_when_selected",
    title="mean improvement magnitude",
    filename="element_mean_improvement_when_selected_focused",
    number_format=".2f",
)
