# -*- coding: utf-8 -*-
"""
M1-I: threshold-aware stratified IPEA-guided MO-SHADE
======================================================

This is a complete, standalone implementation of the M1-I algorithm for hotel
service-improvement planning. No runtime patch module or external optimizer
script is required.

Formal optimization model
-------------------------
For service element m with current performance Per_m and improvement x_m:

    DeltaS_m(x_m) = S_m(Per_m + x_m) - S_m(Per_m)

    F_rep(x) = sum_m DeltaS_m(x_m) / sum_m DeltaS_m(max_delta_m)

    DeltaV(x) = sum_m [beta_m_raw*x_m + zeta_R_raw*DeltaS_m(x_m)]

    F_choice(x) = logistic(logit(P0) + DeltaV(x)) - P0

    F_cost(x) = sum_m effective_cost_m*x_m

The optimizer internally minimizes:

    (-F_rep, -F_choice, F_cost/max_cost)

M1-I search prior
-----------------
The nonzero initial population is stratified into three independent layers:

    40% coverage-boundary layer:
        cost-weighted high-priority coverage lies in [rho, rho + delta];

    30% ordinary IPEA-priority layer:
        activation follows the original priority probabilities;

    30% uniform-exploration layer:
        all eligible service elements have equal activation probability.

IPEA therefore guides the initial search distribution but does not enter the
three objectives, SHADE crossover, environmental selection, or robust
representative-solution score. The minimum high-priority coverage is enforced
by feasibility-first constrained dominance, not by forcing every candidate
through a priority-aware repair.

Decision and managerial constraints
-----------------------------------
    x_m = 0 or x_m >= MIN_ACTION_MAGNITUDE
    0 <= x_m <= max_delta_m
    optional total budget cap
    optional maximum number of active actions
    high-priority resource coverage >= HIGH_PRIORITY_COVERAGE_MIN

Main outputs
------------
- moo_results_M1I/M1I_results.xlsx
- moo_results_M1I/Pareto_solutions.csv
- moo_results_M1I/M1I_results.pkl
- diagnostic and Pareto/convergence figures
"""

from __future__ import annotations

import math
import pickle
import random
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# 1. Configuration
# =============================================================================
BASE_DIR = Path(r"D:\AAApaper\online_review")

IPEA_SOURCE_FILE = BASE_DIR / "shap+TD_result.xlsx"
IPEA_SOURCE_SHEET = "IPEA-modified"

EFFECT_FILE = BASE_DIR / "IPEA_effect_results_pooled_MNL-2.xlsx"
EFFECT_SUMMARY_SHEET = "IPEA_summary"
EFFECT_DETAIL_SHEET = "IPEA_details"

OUTPUT_DIR = BASE_DIR / "moo_results_M1I"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_EXCEL = OUTPUT_DIR / "M1I_results.xlsx"
OUTPUT_CSV = OUTPUT_DIR / "Pareto_solutions.csv"
OUTPUT_PICKLE = OUTPUT_DIR / "M1I_results.pkl"

# Source columns
ES_COLUMN = "ES"
FEATURE_FALLBACK_COLUMN = "Feature"
DESCRIPTION_CANDIDATES = (
    "Sub_Issue",
    "sub_issue",
    "Description",
    "Feature_name",
)
IMPORTANCE_COLUMN = "Imp"
PERFORMANCE_COLUMN = "Per_focus"
MARKET_PERFORMANCE_COLUMN = "Per_market"
CATEGORY_COLUMN = "type"
COST_COLUMN = "维护性成本"
ELIGIBILITY_COLUMN = "eligible_for_moo"

# IPEA reference definitions
IMPORTANCE_REFERENCE_METHOD = "mean"   # "mean" or "median"
EFFECT_REFERENCE_METHOD = "median"     # "mean" or "median"

# MNL context
MOO_CONTEXT_PERIOD: int | None = None  # None selects the latest available
BASELINE_PROBABILITY_OVERRIDE: float | None = None

# Performance/action domain
MIN_PERFORMANCE = 1.0
MAX_PERFORMANCE = 10.0
ZERO_PERFORMANCE_AS_MISSING = True
ZERO_PERFORMANCE_ATOL = 1e-12
MIN_ACTION_MAGNITUDE = 0.5

# Optional operational constraints
# None means no hard budget cap; cost remains an optimization objective.
BUDGET_LIMIT: float | None = None
# Set an integer to impose a maximum number of simultaneously active actions.
MAX_ACTIVE_ACTIONS: int | None = 15

# IPEA priority guidance
HIGH_PRIORITY_LEVELS = (1, 2)
HIGH_PRIORITY_COVERAGE_MIN = 0.40
PRIORITY_GUIDANCE: dict[int, float] = {
    1: 1.00,
    2: 0.75,
    3: 0.50,
    4: 0.25,
}
PRIORITY_EXPLORATION_RATE = 0.20

# M1-I threshold-aware stratified initialization.
M1I_BOUNDARY_LAYER_SHARE = 0.40
M1I_PRIORITY_LAYER_SHARE = 0.30
M1I_UNIFORM_LAYER_SHARE = 0.30
M1I_BOUNDARY_BANDWIDTH = 0.05
M1I_BOUNDARY_MAX_ATTEMPTS = 100

# A zero reported cost creates free-action pathologies. A small effective
# floor is used for optimization and coverage, while raw cost is retained in
# outputs for transparency.
ZERO_COST_FLOOR_RATIO = 0.05

# MO-SHADE
SEED = 42
POPULATION_MULTIPLIER = 10
MIN_POPULATION_SIZE = 100
MAX_POPULATION_SIZE = 1200
FINAL_POPULATION_RATIO = 0.50
N_GENERATIONS = 500
SHADE_MEMORY_SIZE = 10
P_BEST_RATE = 0.20
ARCHIVE_RATE = 1.0
F_SCALE = 0.10
CR_STD = 0.10
INITIAL_ZERO_SOLUTION_RATE = 0.01
INITIAL_ACTIVE_FRACTION_RANGE = (0.10, 0.35)

# Convergence/archive
MIN_GENERATIONS_BEFORE_STOP = 80
CONVERGENCE_WINDOW = 30
FRONT_SHIFT_TOLERANCE = 1e-4
MAX_PARETO_ARCHIVE_SIZE = 1000
PRINT_EVERY = 10

# Pareto representative selection: no historical objective weights.
ROBUST_EPSILON = 0.05
ROBUST_SCORE_TIE_TOLERANCE = 0.01
RECOMMENDED_COST_TIER = "all"  # "low", "medium", "high", or "all"
REQUIRE_NONNEGATIVE_CHOICE_GAIN = True
CHOICE_GAIN_TOLERANCE = 1e-12
NONTRIVIAL_DELTA_TOLERANCE = 1e-8

# Plotting
SHOW_PLOTS = False
SAVE_PLOTS = True

# Numerics
EXPONENT_BOUND = 100.0
EPS = 1e-12
ROUND_DECIMALS_FOR_UNIQUENESS = 10


# =============================================================================
# 2. Data structures
# =============================================================================
@dataclass(frozen=True)
class ServiceOption:
    es: str
    description: str
    importance: float
    performance: float
    market_performance: float
    ipea_effect: float
    raw_cost: float
    effective_cost: float
    category: str
    params: tuple[float, ...]
    direct_effect_raw: float
    zone: int
    priority: int
    max_delta: float
    max_satisfaction_gain: float


@dataclass(frozen=True)
class ModelContext:
    period: int
    baseline_probability: float
    zeta_total_score_raw: float
    importance_reference: float
    effect_reference: float
    max_effective_cost: float
    total_max_satisfaction_gain: float


@dataclass
class Individual:
    x: np.ndarray
    objectives: tuple[float, float, float]
    violation: float
    components: dict[str, Any]
    origin: str = "population"
    F: float | None = None
    CR: float | None = None
    memory_index: int | None = None
    parent_index: int | None = None

    def copy(self) -> "Individual":
        return Individual(
            x=self.x.copy(),
            objectives=tuple(self.objectives),
            violation=float(self.violation),
            components=dict(self.components),
            origin=self.origin,
            F=self.F,
            CR=self.CR,
            memory_index=self.memory_index,
            parent_index=self.parent_index,
        )


# =============================================================================
# 3. General helpers
# =============================================================================
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_es(value: Any) -> str:
    if pd.isna(value):
        return ""

    if isinstance(value, (int, np.integer)):
        return f"ES_{int(value)}"

    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        if float(value).is_integer():
            return f"ES_{int(value)}"

    text = str(value).strip()
    match = re.fullmatch(r"(?i)ES[\s_-]?(\d+)", text)
    if match:
        return f"ES_{int(match.group(1))}"

    if re.fullmatch(r"\d+", text):
        return f"ES_{int(text)}"

    return text


def sort_es_key(es: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", str(es))
    if match:
        return int(match.group(1)), str(es)
    return 10**9, str(es)


def normalize_category(value: Any) -> str:
    text = str(value).strip().lower().replace("_", "-")
    mapping = {
        "excitement": "Attractive",
        "exciting": "Attractive",
        "attractive": "Attractive",
        "must-be": "Must-be",
        "must be": "Must-be",
        "mustbe": "Must-be",
        "linear": "One-dimensional",
        "one-dimensional": "One-dimensional",
        "one dimensional": "One-dimensional",
        "onedimensional": "One-dimensional",
        "indifferent": "Indifferent",
        "reverse": "Reverse",
        "questionable": "Questionable",
    }
    if text not in mapping:
        raise ValueError(f"Unsupported satisfaction category: {value!r}")
    return mapping[text]


def parse_boolean(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(float(value) != 0.0)
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0", ""}:
        return False
    raise ValueError(f"Cannot interpret Boolean value: {value!r}")


def require_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    name: str,
) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


def consistent_unique(
    series: pd.Series,
    name: str,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> float:
    values = (
        pd.to_numeric(series, errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
    )
    if values.size == 0:
        raise ValueError(f"No valid value found for {name}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite values found for {name}: {values.tolist()}")
    if not np.allclose(values, values[0], rtol=rtol, atol=atol):
        raise ValueError(
            f"Expected numerically consistent values for {name}, "
            f"found {values.tolist()}"
        )
    return float(values[0])


def calculate_partition_reference(series: pd.Series, method: str) -> float:
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty:
        raise ValueError("No finite values are available for a partition reference.")

    method_key = str(method).strip().lower()
    if method_key == "mean":
        return float(values.mean())
    if method_key == "median":
        return float(values.median())
    raise ValueError("Partition-reference method must be 'mean' or 'median'.")


def infer_es_series(df: pd.DataFrame) -> pd.Series:
    if ES_COLUMN in df.columns:
        return df[ES_COLUMN].map(normalize_es)
    if FEATURE_FALLBACK_COLUMN in df.columns:
        return df[FEATURE_FALLBACK_COLUMN].map(normalize_es)
    raise KeyError(
        f"{IPEA_SOURCE_SHEET} must contain {ES_COLUMN!r} or "
        f"{FEATURE_FALLBACK_COLUMN!r}."
    )


def infer_description(row: pd.Series, es: str) -> str:
    for column in DESCRIPTION_CANDIDATES:
        if column in row.index and pd.notna(row[column]):
            text = str(row[column]).strip()
            if text:
                return text
    return es


# =============================================================================
# 4. IPEA zones and priorities -- retained exactly as requested
# =============================================================================
def get_zone(
    importance: float,
    performance: float,
    effect: float,
    market_performance: float,
    importance_reference: float,
    effect_reference: float,
) -> int:
    values = {
        "importance": importance,
        "performance": performance,
        "effect": effect,
        "market_performance": market_performance,
        "importance_reference": importance_reference,
        "effect_reference": effect_reference,
    }

    invalid = {
        key: value
        for key, value in values.items()
        if not np.isfinite(value)
    }
    if invalid:
        raise ValueError(f"Invalid values passed to get_zone: {invalid}")

    low_importance = importance < importance_reference
    low_effect = effect < effect_reference
    below_market = performance < market_performance

    if not low_importance and not below_market:
        base_zone = 1
    elif low_importance and not below_market:
        base_zone = 2
    elif low_importance and below_market:
        base_zone = 3
    else:
        base_zone = 4

    return base_zone + (4 if low_effect else 0)


ZONE_PRIORITY_MAP: dict[int, int] = {
    4: 1,
    1: 2,
    3: 2,
    8: 2,
    2: 3,
    5: 3,
    7: 4,
    6: 4,
}


def get_priority(zone: int) -> int:
    zone_int = int(zone)
    if zone_int not in ZONE_PRIORITY_MAP:
        raise ValueError(f"Unsupported IPEA zone: {zone_int}")
    return int(ZONE_PRIORITY_MAP[zone_int])


# =============================================================================
# 5. Satisfaction and choice functions
# =============================================================================
def _safe_exp(exponent: float) -> float:
    return math.exp(float(np.clip(exponent, -EXPONENT_BOUND, EXPONENT_BOUND)))


def satisfaction_value(
    category: str,
    params: tuple[float, ...],
    performance: float,
) -> float:
    """Evaluate the fitted function directly on the original rating scale."""
    x = float(performance)
    if not MIN_PERFORMANCE <= x <= MAX_PERFORMANCE:
        raise ValueError(
            f"Performance must lie in [{MIN_PERFORMANCE}, {MAX_PERFORMANCE}], "
            f"received {x}."
        )

    if category == "Attractive":
        knot, a, b, c = params
        return float(a * _safe_exp(b * (x - knot)) + c)

    if category == "Must-be":
        knot, a, b, c = params
        return float(-a * _safe_exp(-b * (x - knot)) + c)

    if category in {"One-dimensional", "Reverse"}:
        a, c = params
        return float(a * x + c)

    if category == "Indifferent":
        (c,) = params
        return float(c)

    raise ValueError(f"Unsupported satisfaction category: {category!r}")


def satisfaction_gain(option: ServiceOption, delta: float) -> float:
    x = float(np.clip(delta, 0.0, option.max_delta))
    if 0.0 < x < MIN_ACTION_MAGNITUDE - EPS:
        raise ValueError(
            f"{option.es}: nonzero action {x} is below "
            f"MIN_ACTION_MAGNITUDE={MIN_ACTION_MAGNITUDE}."
        )
    p0 = option.performance
    p1 = p0 + x
    return float(
        satisfaction_value(option.category, option.params, p1)
        - satisfaction_value(option.category, option.params, p0)
    )


def stable_logistic(logit_value: float) -> float:
    z = float(logit_value)
    if z >= 0.0:
        return float(1.0 / (1.0 + math.exp(-min(z, 700.0))))
    exp_z = math.exp(max(z, -700.0))
    return float(exp_z / (1.0 + exp_z))


def probability_after_utility_change(
    baseline_probability: float,
    utility_change: float,
) -> float:
    p0 = float(baseline_probability)
    if not 0.0 < p0 < 1.0:
        raise ValueError(
            "baseline_probability must lie strictly in (0,1); "
            f"received {p0}."
        )
    dv = float(utility_change)
    if not np.isfinite(dv):
        raise ValueError(f"Non-finite utility change: {dv}")
    baseline_logit = math.log(p0) - math.log1p(-p0)
    return stable_logistic(baseline_logit + dv)


# =============================================================================
# 6. Input loading and option construction
# =============================================================================
def _read_float(
    row: pd.Series,
    column: str,
    *,
    default: float | None = None,
) -> float:
    if column not in row.index or pd.isna(row[column]):
        if default is None:
            raise KeyError(f"Missing required parameter {column!r}.")
        return float(default)
    value = float(row[column])
    if not np.isfinite(value):
        raise ValueError(f"Non-finite parameter {column!r}: {value}")
    return value


def build_satisfaction_params(
    row: pd.Series,
    category: str,
) -> tuple[float, ...]:
    if category in {"Attractive", "Must-be"}:
        knot = _read_float(row, "knot")
        a = _read_float(row, "a")
        b = _read_float(row, "b")
        c = _read_float(row, "c", default=0.0)
        if a <= 0.0 or b <= 0.0:
            raise ValueError(
                f"{row['ES_key']}: {category} requires a>0 and b>0; "
                f"received a={a}, b={b}."
            )
        return knot, a, b, c

    if category == "One-dimensional":
        a = _read_float(row, "a")
        c = _read_float(row, "c", default=0.0)
        if a <= 0.0:
            raise ValueError(
                f"{row['ES_key']}: One-dimensional requires a>0; a={a}."
            )
        return a, c

    if category == "Reverse":
        a = _read_float(row, "a")
        c = _read_float(row, "c", default=0.0)
        return a, c

    if category == "Indifferent":
        c = _read_float(row, "c", default=0.0)
        return (c,)

    raise ValueError(
        f"{row['ES_key']}: unsupported category for parameter construction: "
        f"{category!r}"
    )


def prepare_source_table(raw_source: pd.DataFrame) -> pd.DataFrame:
    source = raw_source.copy()
    source["ES_key"] = infer_es_series(source)

    require_columns(
        source,
        [
            "ES_key",
            IMPORTANCE_COLUMN,
            PERFORMANCE_COLUMN,
            MARKET_PERFORMANCE_COLUMN,
            CATEGORY_COLUMN,
            COST_COLUMN,
            "a",
        ],
        "IPEA source",
    )

    if source["ES_key"].eq("").any():
        raise ValueError("Some source rows have no valid ES identifier.")
    if source["ES_key"].duplicated().any():
        duplicates = source.loc[
            source["ES_key"].duplicated(keep=False), "ES_key"
        ].unique()
        raise ValueError(
            f"Duplicate ES identifiers in source: {duplicates.tolist()}"
        )

    numeric_columns = [
        IMPORTANCE_COLUMN,
        PERFORMANCE_COLUMN,
        MARKET_PERFORMANCE_COLUMN,
        COST_COLUMN,
    ]
    for column in numeric_columns:
        source[column] = pd.to_numeric(source[column], errors="coerce")

    invalid_numeric = source[numeric_columns].isna().any(axis=1)
    if invalid_numeric.any():
        bad = source.loc[invalid_numeric, ["ES_key", *numeric_columns]]
        raise ValueError(
            "Invalid source numeric values:\n"
            f"{bad.to_string(index=False)}"
        )

    if (source[COST_COLUMN] < 0.0).any():
        bad = source.loc[source[COST_COLUMN] < 0.0, ["ES_key", COST_COLUMN]]
        raise ValueError(
            "Negative maintenance cost:\n"
            f"{bad.to_string(index=False)}"
        )

    source["_category"] = source[CATEGORY_COLUMN].map(normalize_category)

    if ELIGIBILITY_COLUMN in source.columns:
        source["_eligible"] = source[ELIGIBILITY_COLUMN].map(parse_boolean)
    else:
        source["_eligible"] = source["_category"].isin(
            {"Attractive", "Must-be", "One-dimensional"}
        )

    invalid_eligible = source["_eligible"] & ~source["_category"].isin(
        {"Attractive", "Must-be", "One-dimensional"}
    )
    if invalid_eligible.any():
        bad = source.loc[invalid_eligible, ["ES_key", "_category"]]
        raise ValueError(
            "Only positive monotonic satisfaction categories may enter MOO:\n"
            f"{bad.to_string(index=False)}"
        )

    return source.reset_index(drop=True)


def load_inputs() -> tuple[
    list[ServiceOption],
    ModelContext,
    pd.DataFrame,
    pd.DataFrame,
]:
    raw_source = pd.read_excel(
        IPEA_SOURCE_FILE,
        sheet_name=IPEA_SOURCE_SHEET,
    )
    source_all = prepare_source_table(raw_source)

    summary = pd.read_excel(
        EFFECT_FILE,
        sheet_name=EFFECT_SUMMARY_SHEET,
    ).copy()
    details = pd.read_excel(
        EFFECT_FILE,
        sheet_name=EFFECT_DETAIL_SHEET,
    ).copy()

    require_columns(summary, ["ES", "AME_Choice_Probability"], "IPEA summary")
    require_columns(
        details,
        ["ES", "time_period", "P_focal", "direct_effect_raw"],
        "IPEA details",
    )

    summary["ES_key"] = summary["ES"].map(normalize_es)
    details["ES_key"] = details["ES"].map(normalize_es)

    if summary["ES_key"].duplicated().any():
        duplicates = summary.loc[
            summary["ES_key"].duplicated(keep=False), "ES_key"
        ].unique()
        raise ValueError(
            f"Duplicate ES identifiers in IPEA summary: {duplicates.tolist()}"
        )

    summary["AME_Choice_Probability"] = pd.to_numeric(
        summary["AME_Choice_Probability"], errors="coerce"
    )
    if summary["AME_Choice_Probability"].isna().any():
        bad = summary.loc[
            summary["AME_Choice_Probability"].isna(), "ES_key"
        ].tolist()
        raise ValueError(f"Invalid AME_Choice_Probability for: {bad}")

    source_all = source_all.merge(
        summary[["ES_key", "AME_Choice_Probability"]],
        on="ES_key",
        how="left",
        validate="one_to_one",
    )
    missing_effect = source_all.loc[
        source_all["AME_Choice_Probability"].isna(), "ES_key"
    ].tolist()
    if missing_effect:
        raise ValueError(f"Missing AME_Choice_Probability for: {missing_effect}")

    # Partition references use the complete valid IPEA table, before the MOO
    # eligibility filter, matching the diagnostic interpretation.
    importance_reference = calculate_partition_reference(
        source_all[IMPORTANCE_COLUMN],
        IMPORTANCE_REFERENCE_METHOD,
    )
    effect_reference = calculate_partition_reference(
        source_all["AME_Choice_Probability"],
        EFFECT_REFERENCE_METHOD,
    )

    details["time_period"] = pd.to_numeric(
        details["time_period"], errors="coerce"
    )
    details = details.loc[details["time_period"].notna()].copy()
    details["time_period"] = details["time_period"].astype(int)

    periods = sorted(details["time_period"].unique().tolist())
    if not periods:
        raise ValueError("No valid MNL context period exists in IPEA_details.")

    context_period = max(periods) if MOO_CONTEXT_PERIOD is None else int(
        MOO_CONTEXT_PERIOD
    )
    if context_period not in periods:
        raise ValueError(
            f"MOO_CONTEXT_PERIOD={context_period} is unavailable; "
            f"available periods={periods}."
        )

    current_details = details.loc[
        details["time_period"] == context_period
    ].copy()

    baseline_probability = consistent_unique(
        current_details["P_focal"],
        f"P_focal[t={context_period}]",
    )
    if BASELINE_PROBABILITY_OVERRIDE is not None:
        override = float(BASELINE_PROBABILITY_OVERRIDE)
        if not 0.0 < override < 1.0:
            raise ValueError("BASELINE_PROBABILITY_OVERRIDE must lie in (0,1).")
        baseline_probability = override

    if "zeta_total_score_raw" in current_details.columns:
        zeta_total_score_raw = consistent_unique(
            current_details["zeta_total_score_raw"],
            f"zeta_total_score_raw[t={context_period}]",
        )
    else:
        require_columns(
            current_details,
            ["beta_total_score_standardized", "sigma_total_score"],
            "IPEA details fallback",
        )
        beta_total_z = consistent_unique(
            current_details["beta_total_score_standardized"],
            f"beta_total_score_standardized[t={context_period}]",
        )
        sigma_total = consistent_unique(
            current_details["sigma_total_score"],
            f"sigma_total_score[t={context_period}]",
        )
        if sigma_total <= 0.0:
            raise ValueError("sigma_total_score must be positive.")
        zeta_total_score_raw = beta_total_z / sigma_total
        warnings.warn(
            "zeta_total_score_raw is absent; using beta_total_score_standardized "
            "/ sigma_total_score. The latest EWMA-adjusted MNL output is preferred.",
            RuntimeWarning,
        )

    direct_effect_map: dict[str, float] = {}
    for es, group in current_details.groupby("ES_key", sort=False):
        direct_effect_map[es] = consistent_unique(
            group["direct_effect_raw"],
            f"direct_effect_raw[ES={es},t={context_period}]",
        )

    source = source_all.loc[source_all["_eligible"]].copy().reset_index(drop=True)
    if source.empty:
        raise ValueError("No service elements are eligible for optimization.")

    positive_costs = source.loc[source[COST_COLUMN] > EPS, COST_COLUMN].to_numpy(
        dtype=float
    )
    if positive_costs.size:
        zero_cost_floor = max(
            float(np.median(positive_costs)) * ZERO_COST_FLOOR_RATIO,
            EPS,
        )
    else:
        zero_cost_floor = 1.0

    option_rows: list[dict[str, Any]] = []
    skipped_zero_performance: list[str] = []

    for _, row in source.iterrows():
        es = normalize_es(row["ES_key"])
        performance = float(row[PERFORMANCE_COLUMN])
        market_performance = float(row[MARKET_PERFORMANCE_COLUMN])

        if ZERO_PERFORMANCE_AS_MISSING and np.isclose(
            performance, 0.0, rtol=0.0, atol=ZERO_PERFORMANCE_ATOL
        ):
            skipped_zero_performance.append(es)
            continue

        if not MIN_PERFORMANCE <= performance <= MAX_PERFORMANCE:
            raise ValueError(
                f"{es}: performance={performance} is outside "
                f"[{MIN_PERFORMANCE},{MAX_PERFORMANCE}]."
            )
        if not np.isfinite(market_performance):
            raise ValueError(f"{es}: non-finite market performance.")

        category = str(row["_category"])
        params = build_satisfaction_params(row, category)
        importance = float(row[IMPORTANCE_COLUMN])
        ipea_effect = float(row["AME_Choice_Probability"])
        raw_cost = float(row[COST_COLUMN])
        effective_cost = raw_cost if raw_cost > EPS else zero_cost_floor

        zone = get_zone(
            importance=importance,
            performance=performance,
            effect=ipea_effect,
            market_performance=market_performance,
            importance_reference=importance_reference,
            effect_reference=effect_reference,
        )
        priority = get_priority(zone)
        max_delta = max(0.0, MAX_PERFORMANCE - performance)

        if max_delta < MIN_ACTION_MAGNITUDE - EPS:
            continue

        direct_effect_raw = float(direct_effect_map.get(es, 0.0))
        if not np.isfinite(direct_effect_raw):
            raise ValueError(f"{es}: non-finite direct_effect_raw.")

        current_s = satisfaction_value(category, params, performance)
        maximum_s = satisfaction_value(category, params, performance + max_delta)
        max_satisfaction_gain = float(maximum_s - current_s)
        if not np.isfinite(max_satisfaction_gain):
            raise ValueError(f"{es}: non-finite maximum satisfaction gain.")
        if max_satisfaction_gain <= EPS:
            raise ValueError(
                f"{es}: eligible element has non-positive maximum satisfaction "
                f"gain={max_satisfaction_gain}."
            )

        option_rows.append(
            {
                "es": es,
                "description": infer_description(row, es),
                "importance": importance,
                "performance": performance,
                "market_performance": market_performance,
                "ipea_effect": ipea_effect,
                "raw_cost": raw_cost,
                "effective_cost": effective_cost,
                "category": category,
                "params": params,
                "direct_effect_raw": direct_effect_raw,
                "zone": zone,
                "priority": priority,
                "max_delta": max_delta,
                "max_satisfaction_gain": max_satisfaction_gain,
            }
        )

    if not option_rows:
        raise ValueError("No valid service options remain after filtering.")

    options = [ServiceOption(**row) for row in option_rows]
    options.sort(key=lambda option: sort_es_key(option.es))

    if not any(option.priority in HIGH_PRIORITY_LEVELS for option in options):
        raise ValueError(
            "No eligible high-priority service element exists, so the high-"
            "priority coverage constraint cannot be defined."
        )

    max_effective_cost = float(
        sum(option.effective_cost * option.max_delta for option in options)
    )
    if max_effective_cost <= EPS:
        raise ValueError("Maximum effective implementation cost is zero.")

    total_max_satisfaction_gain = float(
        sum(option.max_satisfaction_gain for option in options)
    )
    if total_max_satisfaction_gain <= EPS:
        raise ValueError("Total maximum satisfaction gain is zero.")

    context = ModelContext(
        period=context_period,
        baseline_probability=baseline_probability,
        zeta_total_score_raw=zeta_total_score_raw,
        importance_reference=importance_reference,
        effect_reference=effect_reference,
        max_effective_cost=max_effective_cost,
        total_max_satisfaction_gain=total_max_satisfaction_gain,
    )

    diagnostics = pd.DataFrame(
        [
            {
                "ES": option.es,
                "Description": option.description,
                "Importance": option.importance,
                "Performance": option.performance,
                "Market_performance": option.market_performance,
                "AME_Choice_Probability": option.ipea_effect,
                "Direct_effect_raw": option.direct_effect_raw,
                "Zone": option.zone,
                "Priority": option.priority,
                "Raw_cost": option.raw_cost,
                "Effective_cost": option.effective_cost,
                "Max_delta": option.max_delta,
                "Max_satisfaction_gain": option.max_satisfaction_gain,
                "Category": option.category,
            }
            for option in options
        ]
    )

    if skipped_zero_performance:
        print(
            "Skipped zero-performance service elements as missing: "
            + ", ".join(skipped_zero_performance)
        )

    return options, context, diagnostics, current_details


# =============================================================================
# 7. Objective evaluation and constraints
# =============================================================================
def high_priority_resource_coverage(
    x: np.ndarray,
    options: Sequence[ServiceOption],
) -> float:
    costs = np.asarray([option.effective_cost for option in options], dtype=float)
    total = float(np.dot(costs, x))
    if total <= EPS:
        # Coverage is mathematically undefined for the zero-action plan.
        # By convention, return 1.0 so that the conditional coverage
        # requirement is treated as vacuously satisfied when no resource
        # is allocated. The zero-action plan is excluded separately from
        # nontrivial feasible solutions and final recommendations.
        return 1.0
    high_mask = np.asarray(
        [option.priority in HIGH_PRIORITY_LEVELS for option in options],
        dtype=bool,
    )
    high = float(np.dot(costs[high_mask], x[high_mask]))
    return float(high / total)


def evaluate_components(
    values: Sequence[float],
    options: Sequence[ServiceOption],
    context: ModelContext,
) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    if x.shape != (len(options),):
        raise ValueError(
            f"Decision vector shape {x.shape} differs from {(len(options),)}."
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("Decision vector contains non-finite values.")

    relative_satisfaction_gains: list[float] = []
    total_satisfaction_gain = 0.0
    total_utility_change = 0.0
    raw_cost = 0.0
    effective_cost = 0.0
    element_details: list[dict[str, Any]] = []

    for delta, option in zip(x, options):
        d = float(delta)
        gain = satisfaction_gain(option, d) if d > EPS else 0.0
        relative_gain = gain / option.max_satisfaction_gain
        relative_gain = float(np.clip(relative_gain, 0.0, 1.0))

        direct_change = option.direct_effect_raw * d
        mediated_change = context.zeta_total_score_raw * gain
        utility_change = direct_change + mediated_change

        relative_satisfaction_gains.append(relative_gain)
        total_satisfaction_gain += gain
        total_utility_change += utility_change
        raw_cost += option.raw_cost * d
        effective_cost += option.effective_cost * d

        element_details.append(
            {
                "ES": option.es,
                "Description": option.description,
                "Zone": option.zone,
                "Priority": option.priority,
                "Current_performance": option.performance,
                "Delta": d,
                "Target_performance": option.performance + d,
                "Satisfaction_gain": gain,
                "Relative_satisfaction_gain": relative_gain,
                "Direct_utility_change": direct_change,
                "Mediated_utility_change": mediated_change,
                "Total_utility_change": utility_change,
                "Raw_cost_contribution": option.raw_cost * d,
                "Effective_cost_contribution": option.effective_cost * d,
            }
        )

    reputation_improvement = float(
        total_satisfaction_gain / context.total_max_satisfaction_gain
    )
    reputation_improvement = float(np.clip(reputation_improvement, 0.0, 1.0))
    improved_probability = probability_after_utility_change(
        context.baseline_probability,
        total_utility_change,
    )
    probability_gain = float(
        improved_probability - context.baseline_probability
    )
    normalized_cost = float(effective_cost / context.max_effective_cost)
    coverage = high_priority_resource_coverage(x, options)
    n_active = int(np.count_nonzero(x >= MIN_ACTION_MAGNITUDE - EPS))

    # Priority alignment is a secondary, non-objective indicator.
    relative_array = np.asarray(relative_satisfaction_gains, dtype=float)
    guidance = np.asarray(
        [PRIORITY_GUIDANCE[option.priority] for option in options],
        dtype=float,
    )
    if relative_array.sum() > EPS:
        priority_alignment = float(
            np.dot(guidance, relative_array) / relative_array.sum()
        )
    else:
        priority_alignment = 0.0

    return {
        "x": x.copy(),
        "reputation_improvement": reputation_improvement,
        "total_satisfaction_gain": float(total_satisfaction_gain),
        "total_utility_change": float(total_utility_change),
        "baseline_probability": context.baseline_probability,
        "improved_probability": improved_probability,
        "probability_gain": probability_gain,
        "probability_gain_pp": 100.0 * probability_gain,
        "raw_cost": float(raw_cost),
        "effective_cost": float(effective_cost),
        "normalized_cost": normalized_cost,
        "high_priority_coverage": coverage,
        "priority_alignment": priority_alignment,
        "n_active_actions": n_active,
        "sum_delta": float(x.sum()),
        "element_details": element_details,
    }


def constraint_violation(
    values: Sequence[float],
    options: Sequence[ServiceOption],
    components: dict[str, Any] | None = None,
) -> float:
    x = np.asarray(values, dtype=float)
    max_delta = np.asarray([option.max_delta for option in options], dtype=float)

    lower_violation = float(np.maximum(-x, 0.0).sum())
    upper_violation = float(np.maximum(x - max_delta, 0.0).sum())

    small_mask = (x > EPS) & (x < MIN_ACTION_MAGNITUDE - EPS)
    minimum_action_violation = float(
        np.sum(MIN_ACTION_MAGNITUDE - x[small_mask])
    )

    if components is None:
        components = evaluate_components(x, options, CURRENT_CONTEXT)

    coverage_violation = max(
        0.0,
        HIGH_PRIORITY_COVERAGE_MIN
        - float(components["high_priority_coverage"]),
    )

    if BUDGET_LIMIT is None:
        budget_violation = 0.0
    else:
        budget_violation = max(
            0.0,
            float(components["effective_cost"]) - float(BUDGET_LIMIT),
        ) / max(float(BUDGET_LIMIT), EPS)

    if MAX_ACTIVE_ACTIONS is None:
        active_violation = 0.0
    else:
        active_violation = max(
            0.0,
            float(components["n_active_actions"] - MAX_ACTIVE_ACTIONS),
        ) / max(float(MAX_ACTIVE_ACTIONS), 1.0)

    return float(
        lower_violation
        + upper_violation
        + minimum_action_violation
        + coverage_violation
        + budget_violation
        + active_violation
    )


def make_individual(
    values: Sequence[float],
    options: Sequence[ServiceOption],
    context: ModelContext,
    *,
    origin: str,
    F: float | None = None,
    CR: float | None = None,
    memory_index: int | None = None,
    parent_index: int | None = None,
) -> Individual:
    x = np.asarray(values, dtype=float)
    components = evaluate_components(x, options, context)
    violation = constraint_violation(x, options, components)
    objectives = (
        -float(components["reputation_improvement"]),
        -float(components["probability_gain"]),
        float(components["normalized_cost"]),
    )
    return Individual(
        x=x.copy(),
        objectives=objectives,
        violation=violation,
        components=components,
        origin=origin,
        F=F,
        CR=CR,
        memory_index=memory_index,
        parent_index=parent_index,
    )


# Context used only by the fallback in constraint_violation; it is assigned
# in main before optimization starts.
CURRENT_CONTEXT: ModelContext


# =============================================================================
# 8. M1-I threshold-aware stratified initialization and neutral repair
# =============================================================================
@dataclass
class M1IInitializationDiagnostics:
    total_vectors: int = 0
    zero_vectors: int = 0
    boundary_vectors: int = 0
    priority_vectors: int = 0
    uniform_vectors: int = 0
    boundary_fallbacks: int = 0
    boundary_abs_error_sum: float = 0.0
    boundary_abs_error_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        count = max(self.boundary_abs_error_count, 1)
        return {
            "initial_total_vectors": self.total_vectors,
            "initial_zero_vectors": self.zero_vectors,
            "initial_boundary_vectors": self.boundary_vectors,
            "initial_priority_vectors": self.priority_vectors,
            "initial_uniform_vectors": self.uniform_vectors,
            "initial_boundary_fallbacks": self.boundary_fallbacks,
            "initial_boundary_fallback_rate": (
                self.boundary_fallbacks / max(self.boundary_vectors, 1)
            ),
            "initial_boundary_mean_abs_coverage_error": (
                self.boundary_abs_error_sum / count
                if self.boundary_abs_error_count > 0
                else np.nan
            ),
        }


M1I_INITIALIZATION_DIAGNOSTICS = M1IInitializationDiagnostics()


def reset_m1i_initialization_diagnostics() -> None:
    global M1I_INITIALIZATION_DIAGNOSTICS
    M1I_INITIALIZATION_DIAGNOSTICS = M1IInitializationDiagnostics()


def eligible_option_indices(
    options: Sequence[ServiceOption],
) -> np.ndarray:
    indices = np.flatnonzero(
        [
            option.max_delta >= MIN_ACTION_MAGNITUDE - EPS
            for option in options
        ]
    )
    if indices.size == 0:
        raise ValueError("No service element can satisfy the minimum action size.")
    return indices.astype(int)


def priority_activation_probabilities(
    options: Sequence[ServiceOption],
) -> np.ndarray:
    """Original M1 priority probabilities with uniform exploration."""
    guidance = np.asarray(
        [PRIORITY_GUIDANCE[option.priority] for option in options],
        dtype=float,
    )
    eligible = np.asarray(
        [option.max_delta >= MIN_ACTION_MAGNITUDE - EPS for option in options],
        dtype=bool,
    )
    guidance = np.where(eligible, guidance, 0.0)
    if guidance.sum() <= EPS:
        raise ValueError("No service element can satisfy the minimum action size.")

    priority_distribution = guidance / guidance.sum()
    uniform = eligible.astype(float) / eligible.sum()
    probabilities = (
        (1.0 - PRIORITY_EXPLORATION_RATE) * priority_distribution
        + PRIORITY_EXPLORATION_RATE * uniform
    )
    return probabilities / probabilities.sum()


def uniform_activation_probabilities(
    options: Sequence[ServiceOption],
) -> np.ndarray:
    eligible = eligible_option_indices(options)
    probabilities = np.zeros(len(options), dtype=float)
    probabilities[eligible] = 1.0 / eligible.size
    return probabilities


def sample_active_count(
    options: Sequence[ServiceOption],
    rng: np.random.Generator,
    *,
    minimum_required: int = 1,
) -> int:
    n = len(options)
    low_fraction, high_fraction = INITIAL_ACTIVE_FRACTION_RANGE
    min_active = max(minimum_required, int(math.ceil(low_fraction * n)))
    max_active = max(min_active, int(math.ceil(high_fraction * n)))
    max_active = min(max_active, n)

    if MAX_ACTIVE_ACTIONS is not None:
        if int(MAX_ACTIVE_ACTIONS) < minimum_required:
            raise ValueError(
                "MAX_ACTIVE_ACTIONS is too small for the requested initial layer: "
                f"K={MAX_ACTIVE_ACTIONS}, required={minimum_required}."
            )
        max_active = min(max_active, int(MAX_ACTIVE_ACTIONS), n)
        # Essential when K < ceil(low_fraction*n), e.g. n=119 and K=10.
        min_active = min(min_active, max_active)

    eligible_count = int(eligible_option_indices(options).size)
    max_active = min(max_active, eligible_count)
    min_active = min(min_active, max_active)
    if min_active < minimum_required or max_active < min_active:
        raise RuntimeError(
            f"Invalid initial active-count range: [{min_active},{max_active}]."
        )
    return int(rng.integers(min_active, max_active + 1))


def _snap_minimum_action(
    x: np.ndarray,
    options: Sequence[ServiceOption],
) -> np.ndarray:
    result = np.asarray(x, dtype=float).copy()
    max_delta = np.asarray([option.max_delta for option in options], dtype=float)
    result = np.clip(result, 0.0, max_delta)
    small = (result > EPS) & (result < MIN_ACTION_MAGNITUDE)
    result[small] = 0.0
    return result


def _enforce_max_active_actions(
    x: np.ndarray,
    options: Sequence[ServiceOption],
) -> np.ndarray:
    """Priority-neutral action-count repair."""
    result = np.asarray(x, dtype=float).copy()
    if MAX_ACTIVE_ACTIONS is None:
        return result
    if int(MAX_ACTIVE_ACTIONS) < 1:
        raise ValueError("MAX_ACTIVE_ACTIONS must be positive or None.")

    active = np.flatnonzero(result >= MIN_ACTION_MAGNITUDE - EPS)
    if active.size <= int(MAX_ACTIVE_ACTIONS):
        return result

    removal_order = sorted(
        active.tolist(),
        key=lambda idx: (
            result[idx],
            -options[idx].effective_cost,
            idx,
        ),
    )
    n_remove = active.size - int(MAX_ACTIVE_ACTIONS)
    for idx in removal_order[:n_remove]:
        result[idx] = 0.0
    return result


def _reduce_to_budget(
    x: np.ndarray,
    options: Sequence[ServiceOption],
) -> np.ndarray:
    """Priority-neutral budget repair."""
    result = np.asarray(x, dtype=float).copy()
    if BUDGET_LIMIT is None:
        return result

    costs = np.asarray([option.effective_cost for option in options], dtype=float)
    total = float(np.dot(costs, result))
    if total <= float(BUDGET_LIMIT) + EPS:
        return result

    active = np.flatnonzero(result >= MIN_ACTION_MAGNITUDE - EPS)
    order = sorted(
        active.tolist(),
        key=lambda idx: (
            result[idx],
            -options[idx].effective_cost,
            idx,
        ),
    )

    for idx in order:
        if total <= float(BUDGET_LIMIT) + EPS:
            break
        cost = float(costs[idx])
        current = float(result[idx])
        excess = total - float(BUDGET_LIMIT)
        reducible_above_min = max(
            0.0, current - MIN_ACTION_MAGNITUDE
        ) * cost

        if reducible_above_min >= excess - EPS:
            result[idx] -= excess / cost
            total = float(np.dot(costs, result))
            break

        if reducible_above_min > EPS:
            result[idx] = MIN_ACTION_MAGNITUDE
            total -= reducible_above_min

        if total > float(BUDGET_LIMIT) + EPS:
            total -= result[idx] * cost
            result[idx] = 0.0

    return result


def repair_vector(
    values: Sequence[float],
    options: Sequence[ServiceOption],
) -> np.ndarray:
    """Repair domain, minimum-action, K and budget constraints only.

    High-priority coverage is deliberately not forced by repair. It is handled
    by constraint violation and feasibility-first environmental selection.
    """
    x = np.asarray(values, dtype=float).copy()
    if x.shape != (len(options),):
        raise ValueError(
            f"Decision vector shape {x.shape} differs from {(len(options),)}."
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("Cannot repair non-finite decision values.")

    x = _snap_minimum_action(x, options)
    x = _enforce_max_active_actions(x, options)
    x = _reduce_to_budget(x, options)
    x = _snap_minimum_action(x, options)
    x = _enforce_max_active_actions(x, options)
    return _reduce_to_budget(x, options)


def create_vector_from_probabilities(
    options: Sequence[ServiceOption],
    rng: np.random.Generator,
    probabilities: np.ndarray,
) -> np.ndarray:
    n_active = sample_active_count(options, rng)
    selected = rng.choice(
        len(options),
        size=n_active,
        replace=False,
        p=np.asarray(probabilities, dtype=float),
    )
    x = np.zeros(len(options), dtype=float)
    for raw_idx in selected:
        idx = int(raw_idx)
        option = options[idx]
        x[idx] = float(
            rng.uniform(MIN_ACTION_MAGNITUDE, option.max_delta)
        )
    return repair_vector(x, options)


def _sample_group_indices(
    options: Sequence[ServiceOption],
    candidates: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if size < 1 or size > len(candidates):
        raise ValueError("Invalid boundary-layer group sample size.")
    guidance = np.asarray(
        [PRIORITY_GUIDANCE[options[int(i)].priority] for i in candidates],
        dtype=float,
    )
    probabilities = guidance / guidance.sum()
    return np.asarray(
        rng.choice(candidates, size=size, replace=False, p=probabilities),
        dtype=int,
    )


def _allocate_group_spend(
    lower: np.ndarray,
    upper: np.ndarray,
    target: float,
    rng: np.random.Generator,
) -> np.ndarray:
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    minimum = float(lower.sum())
    maximum = float(upper.sum())
    if target < minimum - EPS or target > maximum + EPS:
        raise ValueError("Target group spend is outside its feasible range.")

    spend = lower.copy()
    remaining = max(0.0, target - minimum)
    capacity = np.maximum(upper - spend, 0.0)
    for _ in range(max(20, 5 * len(spend))):
        active = np.flatnonzero(capacity > EPS)
        if remaining <= EPS or active.size == 0:
            break
        shares = rng.dirichlet(np.ones(active.size, dtype=float))
        actual = np.minimum(remaining * shares, capacity[active])
        used = float(actual.sum())
        spend[active] += actual
        capacity[active] -= actual
        remaining -= used
        if used <= EPS:
            break

    if remaining > EPS:
        for idx in np.argsort(-capacity):
            if remaining <= EPS:
                break
            addition = min(float(capacity[idx]), remaining)
            spend[idx] += addition
            remaining -= addition

    if remaining > 1e-9:
        raise RuntimeError(
            f"Boundary spend allocation left {remaining:.12g} unallocated."
        )
    return np.clip(spend, lower, upper)


def create_boundary_layer_vector(
    options: Sequence[ServiceOption],
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, bool]:
    """Construct a plan near the cost-weighted coverage boundary."""
    eligible = eligible_option_indices(options)
    high = np.asarray(
        [
            idx for idx in eligible
            if options[int(idx)].priority in HIGH_PRIORITY_LEVELS
        ],
        dtype=int,
    )
    low = np.asarray(
        [
            idx for idx in eligible
            if options[int(idx)].priority not in HIGH_PRIORITY_LEVELS
        ],
        dtype=int,
    )

    target_lower = float(HIGH_PRIORITY_COVERAGE_MIN)
    target_upper = min(
        target_lower + float(M1I_BOUNDARY_BANDWIDTH),
        0.98,
    )
    target_center = 0.5 * (target_lower + target_upper)

    if high.size == 0 or low.size == 0 or target_lower <= EPS:
        fallback = create_vector_from_probabilities(
            options, rng, priority_activation_probabilities(options)
        )
        error = abs(
            high_priority_resource_coverage(fallback, options) - target_center
        )
        return fallback, float(error), True

    best_vector: np.ndarray | None = None
    best_band_distance = float("inf")

    for _ in range(M1I_BOUNDARY_MAX_ATTEMPTS):
        n_active = sample_active_count(options, rng, minimum_required=2)
        target = float(rng.uniform(target_lower, target_upper))

        proposed_high = int(round(target * n_active)) + int(rng.integers(-1, 2))
        n_high = int(np.clip(proposed_high, 1, n_active - 1))
        n_high = min(n_high, int(high.size))
        n_low = n_active - n_high
        if n_low > low.size:
            n_low = int(low.size)
            n_high = n_active - n_low
        if n_high < 1 or n_low < 1:
            continue

        high_selected = _sample_group_indices(
            options, high, n_high, rng
        )
        low_selected = _sample_group_indices(
            options, low, n_low, rng
        )

        high_cost = np.asarray(
            [options[int(i)].effective_cost for i in high_selected],
            dtype=float,
        )
        low_cost = np.asarray(
            [options[int(i)].effective_cost for i in low_selected],
            dtype=float,
        )
        high_lower = high_cost * MIN_ACTION_MAGNITUDE
        low_lower = low_cost * MIN_ACTION_MAGNITUDE
        high_upper = high_cost * np.asarray(
            [options[int(i)].max_delta for i in high_selected], dtype=float
        )
        low_upper = low_cost * np.asarray(
            [options[int(i)].max_delta for i in low_selected], dtype=float
        )

        total_lower = max(
            float(high_lower.sum()) / max(target, EPS),
            float(low_lower.sum()) / max(1.0 - target, EPS),
        )
        total_upper = min(
            float(high_upper.sum()) / max(target, EPS),
            float(low_upper.sum()) / max(1.0 - target, EPS),
        )
        if total_lower > total_upper + EPS:
            continue

        total_spend = float(rng.uniform(total_lower, total_upper))
        try:
            high_spend = _allocate_group_spend(
                high_lower, high_upper, target * total_spend, rng
            )
            low_spend = _allocate_group_spend(
                low_lower, low_upper, (1.0 - target) * total_spend, rng
            )
        except (ValueError, RuntimeError):
            continue

        x = np.zeros(len(options), dtype=float)
        x[high_selected] = high_spend / high_cost
        x[low_selected] = low_spend / low_cost
        x = repair_vector(x, options)
        coverage = high_priority_resource_coverage(x, options)

        band_distance = max(
            target_lower - coverage,
            coverage - target_upper,
            0.0,
        )
        center_error = abs(coverage - target_center)
        if band_distance <= EPS:
            return x, float(center_error), False
        if band_distance < best_band_distance:
            best_band_distance = float(band_distance)
            best_vector = x.copy()

    if best_vector is not None:
        error = abs(
            high_priority_resource_coverage(best_vector, options) - target_center
        )
        return best_vector, float(error), True

    fallback = create_vector_from_probabilities(
        options, rng, priority_activation_probabilities(options)
    )
    error = abs(
        high_priority_resource_coverage(fallback, options) - target_center
    )
    return fallback, float(error), True


def create_m1i_vector(
    options: Sequence[ServiceOption],
    rng: np.random.Generator,
) -> np.ndarray:
    diagnostics = M1I_INITIALIZATION_DIAGNOSTICS
    diagnostics.total_vectors += 1

    if rng.random() < INITIAL_ZERO_SOLUTION_RATE:
        diagnostics.zero_vectors += 1
        return np.zeros(len(options), dtype=float)

    shares = np.asarray(
        [
            M1I_BOUNDARY_LAYER_SHARE,
            M1I_PRIORITY_LAYER_SHARE,
            M1I_UNIFORM_LAYER_SHARE,
        ],
        dtype=float,
    )
    layer = int(rng.choice(3, p=shares))

    if layer == 0:
        diagnostics.boundary_vectors += 1
        vector, error, fallback = create_boundary_layer_vector(options, rng)
        diagnostics.boundary_abs_error_sum += float(error)
        diagnostics.boundary_abs_error_count += 1
        diagnostics.boundary_fallbacks += int(fallback)
        return vector

    if layer == 1:
        diagnostics.priority_vectors += 1
        return create_vector_from_probabilities(
            options, rng, priority_activation_probabilities(options)
        )

    diagnostics.uniform_vectors += 1
    return create_vector_from_probabilities(
        options, rng, uniform_activation_probabilities(options)
    )

# =============================================================================
# 9. Constraint dominance, sorting and diversity
# =============================================================================
def pareto_dominates_objectives(
    a: Sequence[float],
    b: Sequence[float],
) -> bool:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    return bool(np.all(aa <= bb + EPS) and np.any(aa < bb - EPS))


def constrained_dominates(a: Individual, b: Individual) -> bool:
    a_feasible = a.violation <= EPS
    b_feasible = b.violation <= EPS
    if a_feasible and not b_feasible:
        return True
    if b_feasible and not a_feasible:
        return False
    if not a_feasible and not b_feasible:
        return a.violation < b.violation - EPS
    return pareto_dominates_objectives(a.objectives, b.objectives)


def nondominated_sort(population: Sequence[Individual]) -> list[list[int]]:
    """Vectorized feasibility-first nondominated sorting.

    The dominance matrix uses the same constrained-dominance rule as the
    scalar implementation but avoids the large nested-Python-loop constant.
    """
    n = len(population)
    if n == 0:
        return []

    objectives = np.asarray(
        [individual.objectives for individual in population], dtype=float
    )
    violations = np.asarray(
        [individual.violation for individual in population], dtype=float
    )
    if objectives.shape != (n, 3):
        raise ValueError("Population objective matrix has an invalid shape.")
    if not np.all(np.isfinite(objectives)) or not np.all(np.isfinite(violations)):
        raise ValueError("Population contains non-finite objectives/violations.")

    feasible = violations <= EPS
    dominates = np.zeros((n, n), dtype=bool)

    dominates |= feasible[:, None] & ~feasible[None, :]

    both_infeasible = ~feasible[:, None] & ~feasible[None, :]
    dominates |= both_infeasible & (
        violations[:, None] < violations[None, :] - EPS
    )

    both_feasible = feasible[:, None] & feasible[None, :]
    no_worse = np.all(
        objectives[:, None, :] <= objectives[None, :, :] + EPS,
        axis=2,
    )
    strictly_better = np.any(
        objectives[:, None, :] < objectives[None, :, :] - EPS,
        axis=2,
    )
    dominates |= both_feasible & no_worse & strictly_better
    np.fill_diagonal(dominates, False)

    dominated_counts = dominates.sum(axis=0).astype(int)
    remaining = np.ones(n, dtype=bool)
    fronts: list[list[int]] = []

    while np.any(remaining):
        front = np.flatnonzero(remaining & (dominated_counts == 0)).tolist()
        if not front:
            # Numerical safety fallback; constrained dominance should be acyclic.
            minimum = int(np.argmin(np.where(remaining, dominated_counts, np.inf)))
            front = [minimum]
        fronts.append(front)
        remaining[front] = False
        dominated_counts -= dominates[front].sum(axis=0).astype(int)
        dominated_counts[~remaining] = -1

    return fronts

def crowding_distances(
    population: Sequence[Individual],
    front: Sequence[int],
) -> dict[int, float]:
    if not front:
        return {}
    if len(front) <= 2:
        return {idx: float("inf") for idx in front}

    distances = {idx: 0.0 for idx in front}

    # For an all-infeasible front, violation is the primary diversity axis.
    if all(population[idx].violation > EPS for idx in front):
        values = np.asarray(
            [[population[idx].violation] for idx in front], dtype=float
        )
    else:
        values = np.asarray(
            [population[idx].objectives for idx in front], dtype=float
        )

    n_objectives = values.shape[1]
    for objective in range(n_objectives):
        order_local = np.argsort(values[:, objective])
        minimum = float(values[order_local[0], objective])
        maximum = float(values[order_local[-1], objective])
        distances[front[int(order_local[0])]] = float("inf")
        distances[front[int(order_local[-1])]] = float("inf")
        span = maximum - minimum
        if span <= EPS:
            continue
        for position in range(1, len(front) - 1):
            prev_value = float(values[order_local[position - 1], objective])
            next_value = float(values[order_local[position + 1], objective])
            idx = front[int(order_local[position])]
            if np.isfinite(distances[idx]):
                distances[idx] += (next_value - prev_value) / span

    return distances


def environmental_selection(
    candidates: Sequence[Individual],
    target_size: int,
) -> list[Individual]:
    if target_size < 1:
        raise ValueError("target_size must be positive.")
    fronts = nondominated_sort(candidates)
    selected: list[Individual] = []

    for front in fronts:
        remaining = target_size - len(selected)
        if remaining <= 0:
            break
        if len(front) <= remaining:
            selected.extend(candidates[idx].copy() for idx in front)
            continue

        distances = crowding_distances(candidates, front)
        ordered = sorted(
            front,
            key=lambda idx: (
                candidates[idx].violation,
                -distances[idx],
            ),
        )
        selected.extend(candidates[idx].copy() for idx in ordered[:remaining])
        break

    if len(selected) != target_size:
        raise RuntimeError(
            f"Environmental selection returned {len(selected)} rather than "
            f"{target_size} individuals."
        )
    return selected


def rank_and_crowding(
    population: Sequence[Individual],
) -> tuple[np.ndarray, np.ndarray]:
    ranks = np.full(len(population), fill_value=np.inf, dtype=float)
    crowding = np.zeros(len(population), dtype=float)
    fronts = nondominated_sort(population)
    for rank, front in enumerate(fronts):
        for idx in front:
            ranks[idx] = rank
        distances = crowding_distances(population, front)
        for idx, value in distances.items():
            crowding[idx] = value
    return ranks, crowding


# =============================================================================
# 10. SHADE mutation, crossover and adaptive memory
# =============================================================================
def sample_positive_cauchy(
    location: float,
    scale: float,
    rng: np.random.Generator,
) -> float:
    for _ in range(100):
        value = location + scale * math.tan(math.pi * (rng.random() - 0.5))
        if value > 0.0:
            return float(min(value, 1.0))
    return float(np.clip(location, 1e-6, 1.0))


def generate_trials(
    population: Sequence[Individual],
    mutation_archive: Sequence[np.ndarray],
    memory_f: np.ndarray,
    memory_cr: np.ndarray,
    options: Sequence[ServiceOption],
    context: ModelContext,
    rng: np.random.Generator,
) -> list[Individual]:
    n = len(population)
    if n < 4:
        raise ValueError("MO-SHADE requires at least four population members.")

    ranks, crowding = rank_and_crowding(population)
    ordered_indices = sorted(
        range(n),
        key=lambda idx: (
            ranks[idx],
            -crowding[idx],
            population[idx].violation,
        ),
    )
    top_count = max(2, int(math.ceil(P_BEST_RATE * n)))
    pbest_indices = ordered_indices[:top_count]

    union_vectors = [individual.x for individual in population] + [
        np.asarray(vector, dtype=float) for vector in mutation_archive
    ]

    trials: list[Individual] = []
    for target_index, target in enumerate(population):
        memory_index = int(rng.integers(0, len(memory_f)))
        F = sample_positive_cauchy(
            float(memory_f[memory_index]), F_SCALE, rng
        )
        CR = float(
            np.clip(
                rng.normal(float(memory_cr[memory_index]), CR_STD),
                0.0,
                1.0,
            )
        )

        pbest_index = int(rng.choice(pbest_indices))
        pbest = population[pbest_index].x

        r1_candidates = [idx for idx in range(n) if idx != target_index]
        r1_index = int(rng.choice(r1_candidates))
        r1 = population[r1_index].x

        excluded_vectors = {target_index, r1_index}
        union_indices = list(range(len(union_vectors)))
        valid_r2 = [
            idx
            for idx in union_indices
            if not (idx < n and idx in excluded_vectors)
        ]
        if not valid_r2:
            raise RuntimeError("No valid r2 vector exists for mutation.")
        r2 = union_vectors[int(rng.choice(valid_r2))]

        mutant = (
            target.x
            + F * (pbest - target.x)
            + F * (r1 - r2)
        )
        mutant = repair_vector(mutant, options)

        trial_values = target.x.copy()
        j_rand = int(rng.integers(0, len(options)))
        crossover_mask = rng.random(len(options)) < CR
        crossover_mask[j_rand] = True
        trial_values[crossover_mask] = mutant[crossover_mask]
        trial_values = repair_vector(trial_values, options)

        trials.append(
            make_individual(
                trial_values,
                options,
                context,
                origin="trial",
                F=F,
                CR=CR,
                memory_index=memory_index,
                parent_index=target_index,
            )
        )

    return trials


def normalized_improvement_gain(
    parent: Individual,
    trial: Individual,
) -> float:
    if parent.violation > EPS or trial.violation > EPS:
        return max(parent.violation - trial.violation, 0.0)

    parent_obj = np.asarray(parent.objectives, dtype=float)
    trial_obj = np.asarray(trial.objectives, dtype=float)
    scale = np.maximum(np.maximum(np.abs(parent_obj), np.abs(trial_obj)), 1e-8)
    gain = np.maximum((parent_obj - trial_obj) / scale, 0.0)
    return float(gain.sum())


def update_shade_memory(
    successful_trials: Sequence[tuple[Individual, float]],
    memory_f: np.ndarray,
    memory_cr: np.ndarray,
    memory_position: int,
) -> int:
    if not successful_trials:
        return memory_position

    F_values = np.asarray(
        [trial.F for trial, _ in successful_trials], dtype=float
    )
    CR_values = np.asarray(
        [trial.CR for trial, _ in successful_trials], dtype=float
    )
    gains = np.asarray([gain for _, gain in successful_trials], dtype=float)

    if gains.sum() <= EPS:
        gains = np.ones_like(gains)
    weights = gains / gains.sum()

    denominator = float(np.sum(weights * F_values))
    if denominator > EPS:
        memory_f[memory_position] = float(
            np.sum(weights * F_values**2) / denominator
        )
    memory_cr[memory_position] = float(np.sum(weights * CR_values))

    return int((memory_position + 1) % len(memory_f))


# =============================================================================
# 11. Pareto archive and convergence
# =============================================================================
def unique_individuals(
    individuals: Sequence[Individual],
) -> list[Individual]:
    seen: set[tuple[float, ...]] = set()
    unique: list[Individual] = []
    for individual in individuals:
        key = tuple(np.round(individual.x, ROUND_DECIMALS_FOR_UNIQUENESS))
        if key not in seen:
            seen.add(key)
            unique.append(individual.copy())
    return unique


def update_pareto_archive(
    archive: Sequence[Individual],
    candidates: Sequence[Individual],
) -> list[Individual]:
    feasible = [
        individual.copy()
        for individual in list(archive) + list(candidates)
        if individual.violation <= EPS
    ]
    feasible = unique_individuals(feasible)
    if not feasible:
        return []

    # With all candidates feasible, constrained sorting equals standard Pareto.
    fronts = nondominated_sort(feasible)
    first = [feasible[idx].copy() for idx in fronts[0]]

    if len(first) > MAX_PARETO_ARCHIVE_SIZE:
        distances = crowding_distances(first, list(range(len(first))))
        ordered = sorted(
            range(len(first)),
            key=lambda idx: -distances[idx],
        )
        first = [first[idx].copy() for idx in ordered[:MAX_PARETO_ARCHIVE_SIZE]]

    return first


def objective_front(archive: Sequence[Individual]) -> np.ndarray:
    if not archive:
        return np.empty((0, 3), dtype=float)
    return np.asarray([individual.objectives for individual in archive], dtype=float)


def symmetric_front_shift(
    previous: np.ndarray | None,
    current: np.ndarray,
) -> float:
    if previous is None or previous.size == 0 or current.size == 0:
        return float("inf")
    reference = np.vstack([previous, current])
    minimum = reference.min(axis=0)
    maximum = reference.max(axis=0)
    span = maximum - minimum
    span = np.where(span > EPS, span, 1.0)
    prev_norm = (previous - minimum) / span
    curr_norm = (current - minimum) / span

    distances = np.linalg.norm(
        prev_norm[:, None, :] - curr_norm[None, :, :], axis=2
    )
    forward = float(np.mean(np.min(distances, axis=1)))
    backward = float(np.mean(np.min(distances, axis=0)))
    return max(forward, backward)


def target_population_size(initial_size: int, generation: int) -> int:
    final_size = max(20, int(round(initial_size * FINAL_POPULATION_RATIO)))
    fraction = generation / max(N_GENERATIONS, 1)
    size = int(round(initial_size - fraction * (initial_size - final_size)))
    return max(4, size)


# =============================================================================
# 12. MO-SHADE solver
# =============================================================================
def run_mo_shade(
    options: Sequence[ServiceOption],
    context: ModelContext,
) -> tuple[list[Individual], pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    reset_m1i_initialization_diagnostics()

    initial_size = int(
        np.clip(
            POPULATION_MULTIPLIER * len(options),
            MIN_POPULATION_SIZE,
            MAX_POPULATION_SIZE,
        )
    )
    population = [
        make_individual(
            create_m1i_vector(options, rng),
            options,
            context,
            origin="initial_M1I",
        )
        for _ in range(initial_size)
    ]

    memory_f = np.full(SHADE_MEMORY_SIZE, 0.5, dtype=float)
    memory_cr = np.full(SHADE_MEMORY_SIZE, 0.5, dtype=float)
    memory_position = 0
    mutation_archive: list[np.ndarray] = []
    pareto_archive = update_pareto_archive([], population)

    history_rows: list[dict[str, Any]] = []
    initial_feasible = float(
        np.mean([individual.violation <= EPS for individual in population])
    )
    initial_nontrivial_feasible = float(
        np.mean(
            [
                individual.violation <= EPS
                and individual.components["sum_delta"] > NONTRIVIAL_DELTA_TOLERANCE
                for individual in population
            ]
        )
    )
    history_rows.append(
        {
            "generation": 0,
            "population_size": len(population),
            "pareto_archive_size": len(pareto_archive),
            "successful_trials": 0,
            "feasible_fraction": initial_feasible,
            "nontrivial_feasible_fraction": initial_nontrivial_feasible,
            "mean_high_priority_coverage": float(
                np.mean(
                    [
                        individual.components["high_priority_coverage"]
                        for individual in population
                    ]
                )
            ),
            "front_shift": np.nan,
            "best_reputation_improvement": (
                max(
                    individual.components["reputation_improvement"]
                    for individual in pareto_archive
                ) if pareto_archive else np.nan
            ),
            "best_choice_probability_gain": (
                max(
                    individual.components["probability_gain"]
                    for individual in pareto_archive
                ) if pareto_archive else np.nan
            ),
            "lowest_effective_cost": (
                min(
                    individual.components["effective_cost"]
                    for individual in pareto_archive
                ) if pareto_archive else np.nan
            ),
            "memory_f_mean": float(memory_f.mean()),
            "memory_cr_mean": float(memory_cr.mean()),
        }
    )

    previous_front = objective_front(pareto_archive)
    stable_counter = 0

    for generation in range(1, N_GENERATIONS + 1):
        trials = generate_trials(
            population,
            mutation_archive,
            memory_f,
            memory_cr,
            options,
            context,
            rng,
        )

        combined = [individual.copy() for individual in population] + [
            trial.copy() for trial in trials
        ]
        next_size = target_population_size(initial_size, generation)
        next_population = environmental_selection(combined, next_size)

        selected_keys = {
            tuple(np.round(individual.x, ROUND_DECIMALS_FOR_UNIQUENESS))
            for individual in next_population
        }
        successful_trials: list[tuple[Individual, float]] = []
        for trial in trials:
            key = tuple(np.round(trial.x, ROUND_DECIMALS_FOR_UNIQUENESS))
            if key not in selected_keys or trial.parent_index is None:
                continue
            parent = population[trial.parent_index]
            gain = normalized_improvement_gain(parent, trial)
            if gain > EPS or constrained_dominates(trial, parent):
                successful_trials.append((trial, max(gain, EPS)))
                mutation_archive.append(parent.x.copy())

        max_mutation_archive = max(1, int(round(ARCHIVE_RATE * next_size)))
        if len(mutation_archive) > max_mutation_archive:
            keep = rng.choice(
                len(mutation_archive),
                size=max_mutation_archive,
                replace=False,
            )
            mutation_archive = [mutation_archive[int(idx)] for idx in keep]

        memory_position = update_shade_memory(
            successful_trials, memory_f, memory_cr, memory_position
        )
        population = next_population
        pareto_archive = update_pareto_archive(pareto_archive, population)

        front = objective_front(pareto_archive)
        shift = symmetric_front_shift(previous_front, front)
        previous_front = front.copy()
        if generation >= MIN_GENERATIONS_BEFORE_STOP and shift <= FRONT_SHIFT_TOLERANCE:
            stable_counter += 1
        else:
            stable_counter = 0

        feasible_fraction = float(
            np.mean([individual.violation <= EPS for individual in population])
        )
        nontrivial_feasible_fraction = float(
            np.mean(
                [
                    individual.violation <= EPS
                    and individual.components["sum_delta"] > NONTRIVIAL_DELTA_TOLERANCE
                    for individual in population
                ]
            )
        )
        mean_coverage = float(
            np.mean(
                [
                    individual.components["high_priority_coverage"]
                    for individual in population
                ]
            )
        )

        if pareto_archive:
            best_rep = max(
                individual.components["reputation_improvement"]
                for individual in pareto_archive
            )
            best_choice = max(
                individual.components["probability_gain"]
                for individual in pareto_archive
            )
            lowest_cost = min(
                individual.components["effective_cost"]
                for individual in pareto_archive
            )
        else:
            best_rep = best_choice = lowest_cost = np.nan

        history_rows.append(
            {
                "generation": generation,
                "population_size": len(population),
                "pareto_archive_size": len(pareto_archive),
                "successful_trials": len(successful_trials),
                "feasible_fraction": feasible_fraction,
                "nontrivial_feasible_fraction": nontrivial_feasible_fraction,
                "mean_high_priority_coverage": mean_coverage,
                "front_shift": shift,
                "best_reputation_improvement": best_rep,
                "best_choice_probability_gain": best_choice,
                "lowest_effective_cost": lowest_cost,
                "memory_f_mean": float(memory_f.mean()),
                "memory_cr_mean": float(memory_cr.mean()),
            }
        )

        if generation == 1 or generation % PRINT_EVERY == 0:
            print(
                f"Generation {generation:4d}: pop={len(population):4d}, "
                f"Pareto={len(pareto_archive):4d}, feasible={feasible_fraction:.3f}, "
                f"nontrivial={nontrivial_feasible_fraction:.3f}, "
                f"shift={shift:.6g}, best_rep={best_rep:.4f}, "
                f"best_choice_pp={100.0*best_choice:.4f}"
            )

        if stable_counter >= CONVERGENCE_WINDOW:
            print(
                f"Converged after {generation} generations: front shift remained "
                f"below {FRONT_SHIFT_TOLERANCE} for {CONVERGENCE_WINDOW} generations."
            )
            break

    metadata = {
        "algorithm": "M1-I_threshold_aware_stratified_MO_SHADE",
        "initial_population_size": initial_size,
        "final_population_size": len(population),
        "generations_completed": len(history_rows) - 1,
        "pareto_archive_size": len(pareto_archive),
        "memory_f": memory_f.copy(),
        "memory_cr": memory_cr.copy(),
        **M1I_INITIALIZATION_DIAGNOSTICS.as_dict(),
    }
    return pareto_archive, pd.DataFrame(history_rows), metadata

# =============================================================================
# 13. Priority-neutral relative-robust representative selection
# =============================================================================
def benefit_minmax_normalize(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="raise").astype(float)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum <= EPS:
        return pd.Series(np.ones(len(values)), index=values.index, dtype=float)
    return (values - minimum) / (maximum - minimum)


def cost_minmax_normalize(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="raise").astype(float)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum <= EPS:
        return pd.Series(np.ones(len(values)), index=values.index, dtype=float)
    return (maximum - values) / (maximum - minimum)


def pareto_dataframe(
    archive: Sequence[Individual],
    options: Sequence[ServiceOption],
) -> tuple[pd.DataFrame, list[list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    details: list[list[dict[str, Any]]] = []
    for solution_id, individual in enumerate(archive, start=1):
        c = individual.components
        row: dict[str, Any] = {
            "solution_id": solution_id,
            "reputation_improvement": c["reputation_improvement"],
            "total_satisfaction_gain": c["total_satisfaction_gain"],
            "baseline_probability": c["baseline_probability"],
            "improved_probability": c["improved_probability"],
            "choice_probability_gain": c["probability_gain"],
            "choice_probability_gain_pp": c["probability_gain_pp"],
            "raw_cost": c["raw_cost"],
            "effective_cost": c["effective_cost"],
            "normalized_cost": c["normalized_cost"],
            "high_priority_coverage": c["high_priority_coverage"],
            "priority_alignment": c["priority_alignment"],
            "n_active_actions": c["n_active_actions"],
            "sum_delta": c["sum_delta"],
            "constraint_violation": individual.violation,
        }
        for option, value in zip(options, individual.x):
            row[f"x_{option.es}"] = float(value)
        rows.append(row)
        details.append(c["element_details"])
    return pd.DataFrame(rows), details


def assign_cost_tiers(frame: pd.DataFrame) -> pd.Series:
    if len(frame) < 3 or frame["effective_cost"].nunique() < 3:
        return pd.Series(["medium"] * len(frame), index=frame.index)
    q1 = float(frame["effective_cost"].quantile(1.0 / 3.0))
    q2 = float(frame["effective_cost"].quantile(2.0 / 3.0))
    return pd.Series(
        np.where(
            frame["effective_cost"] <= q1,
            "low",
            np.where(frame["effective_cost"] <= q2, "medium", "high"),
        ),
        index=frame.index,
    )


def calculate_relative_robust_scores(
    pareto: pd.DataFrame,
    epsilon: float = ROBUST_EPSILON,
) -> pd.DataFrame:
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("ROBUST_EPSILON must lie in [0,1].")

    candidates = pareto.loc[
        pareto["sum_delta"] > NONTRIVIAL_DELTA_TOLERANCE
    ].copy()
    if REQUIRE_NONNEGATIVE_CHOICE_GAIN:
        candidates = candidates.loc[
            candidates["choice_probability_gain"] >= -CHOICE_GAIN_TOLERANCE
        ].copy()
    if candidates.empty:
        raise ValueError(
            "No nontrivial Pareto solution satisfies the recommendation filter."
        )

    candidates["phi_reputation"] = benefit_minmax_normalize(
        candidates["reputation_improvement"]
    )
    candidates["phi_choice"] = benefit_minmax_normalize(
        candidates["choice_probability_gain"]
    )
    candidates["phi_cost"] = cost_minmax_normalize(
        candidates["effective_cost"]
    )
    phi = candidates[["phi_reputation", "phi_choice", "phi_cost"]]
    candidates["rho_minimum"] = phi.min(axis=1)
    candidates["phi_average"] = phi.mean(axis=1)
    candidates["robust_score"] = (
        (1.0 - epsilon) * candidates["rho_minimum"]
        + epsilon * candidates["phi_average"]
    )
    candidates["cost_tier"] = assign_cost_tiers(candidates)
    return candidates.sort_values(
        ["robust_score", "effective_cost", "n_active_actions", "solution_id"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)


def _choose_robust_candidate(candidates: pd.DataFrame) -> pd.Series:
    if candidates.empty:
        raise ValueError("Cannot select from an empty candidate table.")
    best_score = float(candidates["robust_score"].max())
    near_best = candidates.loc[
        candidates["robust_score"] >= best_score - ROBUST_SCORE_TIE_TOLERANCE
    ].copy()
    # Priority-neutral tie-breaking.
    near_best = near_best.sort_values(
        ["robust_score", "effective_cost", "n_active_actions", "solution_id"],
        ascending=[False, True, True, True],
    )
    return near_best.iloc[0].copy()


def select_robust_representatives(
    ranked: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.Series] = []
    for tier in ("low", "medium", "high"):
        subset = ranked.loc[ranked["cost_tier"] == tier]
        if subset.empty:
            continue
        chosen = _choose_robust_candidate(subset)
        chosen["selection_scope"] = tier
        rows.append(chosen)

    overall = _choose_robust_candidate(ranked)
    overall["selection_scope"] = "all"
    rows.append(overall)
    representatives = pd.DataFrame(rows).reset_index(drop=True)

    scope = RECOMMENDED_COST_TIER
    recommended = representatives.loc[
        representatives["selection_scope"] == scope
    ].copy()
    if recommended.empty:
        recommended = representatives.loc[
            representatives["selection_scope"] == "all"
        ].copy()
    return representatives, recommended.reset_index(drop=True)


def robust_epsilon_sensitivity(pareto: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for epsilon in (0.00, 0.05, 0.10, 0.20):
        ranked = calculate_relative_robust_scores(pareto, epsilon)
        chosen = _choose_robust_candidate(ranked)
        rows.append(
            {
                "epsilon": epsilon,
                "solution_id": int(chosen["solution_id"]),
                "robust_score": float(chosen["robust_score"]),
                "reputation_improvement": float(chosen["reputation_improvement"]),
                "choice_probability_gain": float(chosen["choice_probability_gain"]),
                "effective_cost": float(chosen["effective_cost"]),
                "high_priority_coverage": float(chosen["high_priority_coverage"]),
                "n_active_actions": int(chosen["n_active_actions"]),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# 14. Outputs and plots
# =============================================================================
def flatten_element_details(
    pareto: pd.DataFrame,
    details: Sequence[Sequence[dict[str, Any]]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (_, solution_row), solution_details in zip(pareto.iterrows(), details):
        solution_id = int(solution_row["solution_id"])
        for element in solution_details:
            rows.append({"solution_id": solution_id, **element})
    return pd.DataFrame(rows)


def build_initialization_audit(
    options: Sequence[ServiceOption],
    metadata: dict[str, Any],
) -> pd.DataFrame:
    rows = [
        {"metric": key, "value": value}
        for key, value in metadata.items()
        if key.startswith("initial_")
    ]
    probability_rows: list[dict[str, Any]] = []
    probabilities = priority_activation_probabilities(options)
    for priority in (1, 2, 3, 4):
        mask = np.asarray([option.priority == priority for option in options])
        if np.any(mask):
            probability_rows.append(
                {
                    "metric": f"M1_priority_probability_mean_P{priority}",
                    "value": float(probabilities[mask].mean()),
                }
            )
    return pd.DataFrame(rows + probability_rows)


def plot_pareto_front(pareto: pd.DataFrame, recommended: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        pareto["reputation_improvement"],
        pareto["choice_probability_gain_pp"],
        pareto["effective_cost"],
        s=24,
        alpha=0.7,
    )
    if not recommended.empty:
        ax.scatter(
            recommended["reputation_improvement"],
            recommended["choice_probability_gain_pp"],
            recommended["effective_cost"],
            s=100,
            marker="*",
        )
    ax.set_xlabel("Reputation improvement")
    ax.set_ylabel("Choice probability gain (percentage points)")
    ax.set_zlabel("Effective implementation cost")
    ax.set_title("M1-I three-objective Pareto front")
    fig.tight_layout()
    if SAVE_PLOTS:
        fig.savefig(FIGURE_DIR / "pareto_front_3d.png", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_convergence(history: pd.DataFrame) -> None:
    if history.empty:
        return
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    finite = history.loc[np.isfinite(history["front_shift"])].copy()
    if finite.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(finite["generation"], finite["front_shift"], linewidth=1.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Symmetric Pareto-front shift")
    ax.set_yscale("log")
    ax.set_title("M1-I MO-SHADE convergence")
    fig.tight_layout()
    if SAVE_PLOTS:
        fig.savefig(FIGURE_DIR / "convergence.png", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def save_outputs(
    archive: Sequence[Individual],
    options: Sequence[ServiceOption],
    diagnostics: pd.DataFrame,
    current_details: pd.DataFrame,
    history: pd.DataFrame,
    metadata: dict[str, Any],
    context: ModelContext,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    pareto, details = pareto_dataframe(archive, options)
    robust_ranked = calculate_relative_robust_scores(pareto)
    representatives, recommended = select_robust_representatives(robust_ranked)
    sensitivity = robust_epsilon_sensitivity(pareto)

    recommended_ids = set(recommended["solution_id"].astype(int))
    pareto["recommended"] = pareto["solution_id"].isin(recommended_ids)
    pareto.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    element_df = flatten_element_details(pareto, details)
    recommended_elements = element_df.loc[
        element_df["solution_id"].isin(recommended_ids)
        & (element_df["Delta"] > NONTRIVIAL_DELTA_TOLERANCE)
    ].copy()
    initialization_audit = build_initialization_audit(options, metadata)

    config_rows = {
        "algorithm": "M1-I_threshold_aware_stratified_MO_SHADE",
        "MOO_context_period": context.period,
        "baseline_probability": context.baseline_probability,
        "zeta_total_score_raw": context.zeta_total_score_raw,
        "importance_reference_method": IMPORTANCE_REFERENCE_METHOD,
        "importance_reference": context.importance_reference,
        "effect_reference_method": EFFECT_REFERENCE_METHOD,
        "effect_reference": context.effect_reference,
        "minimum_action_magnitude": MIN_ACTION_MAGNITUDE,
        "budget_limit": BUDGET_LIMIT,
        "max_active_actions": MAX_ACTIVE_ACTIONS,
        "high_priority_levels": str(HIGH_PRIORITY_LEVELS),
        "high_priority_coverage_min": HIGH_PRIORITY_COVERAGE_MIN,
        "priority_exploration_rate": PRIORITY_EXPLORATION_RATE,
        "M1I_boundary_layer_share": M1I_BOUNDARY_LAYER_SHARE,
        "M1I_priority_layer_share": M1I_PRIORITY_LAYER_SHARE,
        "M1I_uniform_layer_share": M1I_UNIFORM_LAYER_SHARE,
        "M1I_boundary_bandwidth": M1I_BOUNDARY_BANDWIDTH,
        "robust_epsilon": ROBUST_EPSILON,
        "recommended_cost_tier": RECOMMENDED_COST_TIER,
        **metadata,
    }
    config_df = pd.DataFrame(
        [{"parameter": key, "value": value} for key, value in config_rows.items()]
    )

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        pareto.to_excel(writer, sheet_name="Pareto_solutions", index=False)
        robust_ranked.to_excel(writer, sheet_name="Robust_ranking", index=False)
        representatives.to_excel(writer, sheet_name="Representatives", index=False)
        recommended.to_excel(writer, sheet_name="Recommended", index=False)
        sensitivity.to_excel(writer, sheet_name="Robust_sensitivity", index=False)
        recommended_elements.to_excel(
            writer, sheet_name="Recommended_elements", index=False
        )
        element_df.to_excel(writer, sheet_name="All_element_details", index=False)
        diagnostics.to_excel(writer, sheet_name="IPEA_diagnostics", index=False)
        initialization_audit.to_excel(
            writer, sheet_name="Initialization_audit", index=False
        )
        history.to_excel(writer, sheet_name="Convergence", index=False)
        config_df.to_excel(writer, sheet_name="Configuration", index=False)
        current_details.to_excel(writer, sheet_name="MNL_context", index=False)

    payload = {
        "pareto_archive": archive,
        "pareto_dataframe": pareto,
        "robust_ranking": robust_ranked,
        "representatives": representatives,
        "recommended": recommended,
        "diagnostics": diagnostics,
        "initialization_audit": initialization_audit,
        "history": history,
        "metadata": metadata,
        "context": context,
        "options": options,
    }
    with open(OUTPUT_PICKLE, "wb") as file:
        pickle.dump(payload, file)

    plot_pareto_front(pareto, recommended)
    plot_convergence(history)
    return pareto, recommended


# =============================================================================
# 15. Internal checks
# =============================================================================
def run_internal_checks() -> None:
    if set(ZONE_PRIORITY_MAP) != set(range(1, 9)):
        raise RuntimeError("ZONE_PRIORITY_MAP must cover zones 1-8 exactly.")
    for zone in range(1, 9):
        if get_priority(zone) not in {1, 2, 3, 4}:
            raise RuntimeError("Invalid priority mapping.")

    for p0 in (0.05, 0.25, 0.70):
        for dv in (-1.0, -0.1, 0.0, 0.3, 1.2):
            p1 = probability_after_utility_change(p0, dv)
            expected = p0 * math.exp(dv) / (1.0 - p0 + p0 * math.exp(dv))
            if not np.isclose(p1, expected, rtol=1e-12, atol=1e-14):
                raise RuntimeError("Probability-update identity failed.")

    shares = np.asarray(
        [
            M1I_BOUNDARY_LAYER_SHARE,
            M1I_PRIORITY_LAYER_SHARE,
            M1I_UNIFORM_LAYER_SHARE,
        ],
        dtype=float,
    )
    if np.any(shares < 0.0) or not np.isclose(shares.sum(), 1.0, atol=1e-12):
        raise ValueError("M1-I layer shares must be nonnegative and sum to one.")
    if not 0.0 < M1I_BOUNDARY_BANDWIDTH < 1.0:
        raise ValueError("M1I_BOUNDARY_BANDWIDTH must lie in (0,1).")
    if M1I_BOUNDARY_MAX_ATTEMPTS < 1:
        raise ValueError("M1I_BOUNDARY_MAX_ATTEMPTS must be positive.")
    if not 0.0 <= PRIORITY_EXPLORATION_RATE <= 1.0:
        raise ValueError("PRIORITY_EXPLORATION_RATE must lie in [0,1].")
    if not 0.0 <= HIGH_PRIORITY_COVERAGE_MIN <= 1.0:
        raise ValueError("HIGH_PRIORITY_COVERAGE_MIN must lie in [0,1].")
    if MIN_ACTION_MAGNITUDE <= 0.0:
        raise ValueError("MIN_ACTION_MAGNITUDE must be positive.")
    if MAX_ACTIVE_ACTIONS is not None and int(MAX_ACTIVE_ACTIONS) < 2:
        raise ValueError(
            "M1-I boundary layer requires MAX_ACTIVE_ACTIONS>=2 or None."
        )


# =============================================================================
# 16. Main
# =============================================================================
def main() -> None:
    global CURRENT_CONTEXT

    run_internal_checks()
    set_global_seed(SEED)

    print("Loading IPEA, satisfaction and MNL inputs...")
    options, context, diagnostics, current_details = load_inputs()
    CURRENT_CONTEXT = context

    print(
        f"Loaded {len(options)} eligible service elements; "
        f"MNL context period={context.period}; "
        f"baseline choice probability={context.baseline_probability:.6f}."
    )
    print(
        f"M1-I layers: boundary={M1I_BOUNDARY_LAYER_SHARE:.2f}, "
        f"priority={M1I_PRIORITY_LAYER_SHARE:.2f}, "
        f"uniform={M1I_UNIFORM_LAYER_SHARE:.2f}; "
        f"boundary band=[rho, rho+{M1I_BOUNDARY_BANDWIDTH:.2f}]."
    )
    print(
        f"Constraints: minimum action={MIN_ACTION_MAGNITUDE}, "
        f"coverage>={HIGH_PRIORITY_COVERAGE_MIN:.2f}, "
        f"budget={BUDGET_LIMIT}, max active={MAX_ACTIVE_ACTIONS}."
    )

    archive, history, metadata = run_mo_shade(options, context)
    if not archive:
        raise RuntimeError("M1-I MO-SHADE returned no feasible Pareto solution.")

    pareto, recommended = save_outputs(
        archive,
        options,
        diagnostics,
        current_details,
        history,
        metadata,
        context,
    )

    print("\nM1-I optimization completed.")
    print(f"Pareto solutions: {len(pareto)}")
    if not recommended.empty:
        row = recommended.iloc[0]
        print(
            "Recommended plan: "
            f"rep={row['reputation_improvement']:.6f}, "
            f"choice={row['choice_probability_gain_pp']:.4f} pp, "
            f"cost={row['effective_cost']:.6f}, "
            f"coverage={row['high_priority_coverage']:.4f}, "
            f"actions={int(row['n_active_actions'])}."
        )
    print(f"Workbook: {OUTPUT_EXCEL}")
    print(f"Pareto CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()



def _natural_es_key(column: str) -> tuple[str, int]:
    """Ensure x_ES_2 is ordered before x_ES_10."""
    match = re.search(r"(\d+)$", str(column))
    if match is None:
        return str(column), -1
    return str(column)[: match.start()], int(match.group(1))


def load_pooled_scenario_front(
    front_directory: str | Path,
    *,
    max_active_actions: int,
    coverage_threshold: float,
    algorithm: str = "M1-I_threshold_MO_SHADE",
    decision_decimals: int = 10,
) -> pd.DataFrame:
    """Load, validate, and decision-deduplicate all seed fronts."""

    front_directory = Path(front_directory)

    rho_tag = f"{float(coverage_threshold):.2f}".replace(".", "p")
    pattern = (
        f"K{int(max_active_actions)}_rho{rho_tag}_"
        f"{algorithm}_seed_*.csv"
    )
    files = sorted(front_directory.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No Pareto-front files match: {front_directory / pattern}"
        )

    frames: list[pd.DataFrame] = []

    for path in files:
        frame = pd.read_csv(path, encoding="utf-8-sig")

        if frame.empty:
            raise ValueError(f"Empty Pareto-front file: {path}")

        seed_match = re.search(r"_seed_(\d+)\.csv$", path.name)
        if seed_match is None:
            raise ValueError(
                f"Cannot extract the seed from filename: {path.name}"
            )
        seed = int(seed_match.group(1))

        # Validate scenario metadata when available.
        if "max_active_actions" in frame.columns:
            observed_k = pd.to_numeric(
                frame["max_active_actions"],
                errors="raise",
            )
            if not (observed_k == int(max_active_actions)).all():
                raise ValueError(
                    f"Inconsistent K in {path.name}."
                )

        if "coverage_threshold" in frame.columns:
            observed_rho = pd.to_numeric(
                frame["coverage_threshold"],
                errors="raise",
            )
            if not np.allclose(
                observed_rho,
                float(coverage_threshold),
                atol=1e-12,
            ):
                raise ValueError(
                    f"Inconsistent coverage threshold in {path.name}."
                )

        if "algorithm" in frame.columns:
            if not (
                frame["algorithm"].astype(str) == algorithm
            ).all():
                raise ValueError(
                    f"Inconsistent algorithm in {path.name}."
                )

        # solution_id is only unique within one file.
        if "solution_id" in frame.columns:
            frame = frame.rename(
                columns={"solution_id": "source_solution_id"}
            )

        frame["source_seed"] = seed
        frame["source_file"] = path.name
        frames.append(frame)

    pooled = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    required_columns = {
        "reputation_improvement",
        "choice_probability_gain",
        "effective_cost",
        "sum_delta",
        "constraint_violation",
    }
    missing = required_columns - set(pooled.columns)
    if missing:
        raise KeyError(
            f"The Pareto fronts are missing columns: {sorted(missing)}"
        )

    action_columns = sorted(
        [
            column
            for column in pooled.columns
            if str(column).startswith("x_")
        ],
        key=_natural_es_key,
    )
    if not action_columns:
        raise KeyError("No decision columns beginning with 'x_' were found.")

    # Canonicalize decision vectors only for duplicate identification.
    canonical_x = (
        pooled[action_columns]
        .apply(pd.to_numeric, errors="raise")
        .fillna(0.0)
    )
    canonical_x = canonical_x.mask(
        np.abs(canonical_x) <= EPS,
        0.0,
    )
    canonical_x = canonical_x.round(int(decision_decimals))

    pooled["_decision_key"] = [
        tuple(row)
        for row in canonical_x.to_numpy(dtype=float)
    ]

    # Record from which seeds each unique solution was recovered.
    provenance = (
        pooled.groupby("_decision_key", sort=False)
        .agg(
            source_seeds=(
                "source_seed",
                lambda values: " | ".join(
                    str(value)
                    for value in sorted(set(values))
                ),
            ),
            n_source_seeds=(
                "source_seed",
                "nunique",
            ),
            source_files=(
                "source_file",
                lambda values: " | ".join(
                    sorted(set(str(value) for value in values))
                ),
            ),
        )
        .reset_index()
    )

    # Identical decision vectors have deterministic objectives, so one row
    # can be retained while preserving the full provenance above.
    pooled_unique = (
        pooled.sort_values(
            [
                "constraint_violation",
                "effective_cost",
                "source_seed",
            ],
            ascending=[True, True, True],
        )
        .drop_duplicates(
            subset="_decision_key",
            keep="first",
        )
        .merge(
            provenance,
            on="_decision_key",
            how="left",
            validate="one_to_one",
        )
    )

    return pooled_unique.reset_index(drop=True)


# ===============================
# 重新筛选非支配解
# ================================

def refilter_pooled_nondominated_front(
    pooled: pd.DataFrame,
) -> pd.DataFrame:
    """Re-establish feasibility and nondominance after pooling seeds."""

    candidates = pooled.loc[
        (
            pd.to_numeric(
                pooled["sum_delta"],
                errors="coerce",
            )
            > NONTRIVIAL_DELTA_TOLERANCE
        )
        & (
            pd.to_numeric(
                pooled["constraint_violation"],
                errors="coerce",
            )
            <= EPS
        )
    ].copy()

    if REQUIRE_NONNEGATIVE_CHOICE_GAIN:
        candidates = candidates.loc[
            pd.to_numeric(
                candidates["choice_probability_gain"],
                errors="coerce",
            )
            >= -CHOICE_GAIN_TOLERANCE
        ].copy()

    if candidates.empty:
        raise ValueError(
            "No feasible nontrivial solution remains after pooling."
        )

    # Common minimization representation:
    # maximize reputation -> minimize -reputation
    # maximize choice     -> minimize -choice
    # minimize cost       -> minimize cost
    objective_values = np.column_stack(
        [
            -pd.to_numeric(
                candidates["reputation_improvement"],
                errors="raise",
            ).to_numpy(dtype=float),
            -pd.to_numeric(
                candidates["choice_probability_gain"],
                errors="raise",
            ).to_numpy(dtype=float),
            pd.to_numeric(
                candidates["effective_cost"],
                errors="raise",
            ).to_numpy(dtype=float),
        ]
    )

    nondominated = nondominated_mask_minimization(
        objective_values
    )

    pooled_front = candidates.loc[nondominated].copy()

    # solution_id must now be unique within the pooled Pareto set.
    pooled_front = pooled_front.sort_values(
        [
            "effective_cost",
            "n_active_actions",
            "reputation_improvement",
            "choice_probability_gain",
        ],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)

    pooled_front["solution_id"] = np.arange(
        1,
        len(pooled_front) + 1,
        dtype=int,
    )

    return pooled_front


# 统一鲁棒规则选择

def select_hitl_solution_from_scenario(
    front_directory: str | Path,
    *,
    max_active_actions: int,
    coverage_threshold: float,
    algorithm: str = "M1-I_threshold_MO_SHADE",
    epsilon: float = ROBUST_EPSILON,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
]:
    """Pool seed fronts and select one concrete HITL input solution."""

    pooled_unique = load_pooled_scenario_front(
        front_directory,
        max_active_actions=max_active_actions,
        coverage_threshold=coverage_threshold,
        algorithm=algorithm,
    )

    pooled_pareto = refilter_pooled_nondominated_front(
        pooled_unique
    )

    # Recalculate all relative attainments on the pooled Pareto set.
    ranked = calculate_relative_robust_scores(
        pooled_pareto,
        epsilon=float(epsilon),
    )

    representatives, recommended = (
        select_robust_representatives(ranked)
    )

    if recommended.empty:
        raise RuntimeError(
            "The pooled Pareto set did not produce a recommended solution."
        )

    selected = recommended.iloc[0].copy()

    action_columns = sorted(
        [
            column
            for column in pooled_pareto.columns
            if str(column).startswith("x_")
        ],
        key=_natural_es_key,
    )

    active_rows: list[dict[str, float | str]] = []
    for column in action_columns:
        magnitude = float(selected[column])
        if magnitude >= MIN_ACTION_MAGNITUDE - EPS:
            active_rows.append(
                {
                    "service_element": column.removeprefix("x_"),
                    "improvement_magnitude": magnitude,
                }
            )

    active_plan = pd.DataFrame(active_rows)

    # Final validation.
    if int(selected["n_active_actions"]) > int(max_active_actions):
        raise RuntimeError(
            "The selected HITL solution violates the action cap."
        )

    if float(selected["high_priority_coverage"]) < (
        float(coverage_threshold) - EPS
    ):
        raise RuntimeError(
            "The selected HITL solution violates the coverage threshold."
        )

    if float(selected["constraint_violation"]) > EPS:
        raise RuntimeError(
            "The selected HITL solution is infeasible."
        )

    return (
        pooled_unique,
        pooled_pareto,
        representatives,
        selected,
        active_plan,
    )


# =============================

front_dir = Path(
    r"D:\AAApaper\online_review\CODE"
    r"\comparison_five_algorithms_formal_K15"
    r"\fronts"
)

(
    pooled_unique,
    pooled_pareto,
    robust_representatives,
    hitl_solution,
    hitl_action_plan,
) = select_hitl_solution_from_scenario(
    front_dir,
    max_active_actions=15,
    coverage_threshold=0.50,
    algorithm="M1-I_threshold_MO_SHADE",
    epsilon=ROBUST_EPSILON,
)