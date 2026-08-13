# -*- coding: utf-8 -*-
"""
Unified five-algorithm benchmark for hotel-service MOO (optimized metrics)
======================================================

Algorithms
----------
M0_uniform_MO_SHADE
    Priority-neutral sparse initialization plus standard MO-SHADE.

M1_IPEA_MO_SHADE
    Original IPEA-priority initialization plus standard MO-SHADE.

M1-I_threshold_MO_SHADE
    Threshold-aware layered initialization (40% coverage-boundary, 30% IPEA
    priority and 30% uniform) plus otherwise unchanged MO-SHADE.

modified_MOEA_D
    Priority-neutral modified MOEA/D with simplex-lattice decomposition,
    range-corrected Tchebycheff scalarization, neighborhood DE/rand/1/bin,
    residual-violation penalty and an external feasible Pareto archive.

SHAMODE_2019
    Literature SHAMODE adapted to the same semi-continuous hotel decision
    representation, using priority-neutral initialization.

Common comparison rules
-----------------------
1. All algorithms call the same core objective evaluator, repair rules and
   constraint-violation function.
2. Every algorithm receives the same (K, rho) managerial scenario, paired seed
   and exact objective-function-evaluation budget.
3. Early stopping is disabled for the MO-SHADE variants.
4. Final Pareto fronts are nondominated-filtered and crowding-capped to the
   common initial MO-SHADE population size before HV, IGD+, epsilon+, spacing
   and coverage-indicator calculations. This removes archive-size advantage.
5. Objective normalization and the empirical reference front are constructed
   jointly within each fixed (K, rho) scenario across all algorithms and seeds.
6. HV-FE curves use the same FE fractions and never use a front produced after
   the requested checkpoint.
7. The same external relative-robust selector is applied only for descriptive
   comparison of representative plans; its internally normalized score is not
   compared across algorithms.

Required files in the same directory
------------------------------------
- MOO_MO_SHADE_IPEA_priority_initialization_fixed.py
- ipea_m1_extensions.py
- modified_moead_comparison_adapter.py
- shamode_comparison_adapter.py

Primary outputs
---------------
comparison_five_algorithms/
    Comprehensive_results.xlsx
    run_metrics.csv
    summary_by_scenario.csv
    pairwise_tests.csv
    friedman_tests.csv
    algorithm_ranks.csv
    pairwise_coverage.csv
    pairwise_coverage_summary.csv
    convergence_history.csv
    hv_fe_curve.csv
    hv_fe_summary.csv
    recommendation_stability.csv
    recommended_es_frequency.csv
    fronts/*.csv
    hv_fe_curves/*.png
    live_monitor/*.csv
    live_monitor/*.png
    logs/*.log
"""

from __future__ import annotations
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


import contextlib
from bisect import bisect_left
import gzip
import hashlib
import importlib.util
import json
import pickle
import itertools
import math
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from ipea_m1_extensions import (
    capture_core_patch_state,
    get_variant_diagnostics,
    install_variant,
    restore_core_patch_state,
    validate_extension_parameters,
)
import modified_moead_comparison_adapter as moead_adapter
import shamode_comparison_adapter as shamode_adapter
from modified_moead_comparison_adapter import (
    ModifiedMOEADConfig,
    add_mo_shade_evaluation_axis,
    planned_mo_shade_evaluations,
    run_internal_checks as run_moead_internal_checks,
    run_modified_moead,
)


# =============================================================================
# 1. Configuration
# =============================================================================
global METRIC_FRONT_CAP

SCRIPT_DIR = Path(__file__).resolve().parent
CORE_FILE = SCRIPT_DIR / "MOO_MO_SHADE_IPEA_priority_initialization_fixed.py"

RUN_MODE = "formal"  # "pilot" or "formal"
PILOT_SEEDS = (42, 43)
PILOT_GENERATIONS = 50

FORMAL_SEEDS = tuple(range(42, 62))
FORMAL_GENERATIONS = 300

# Recommended formal grid.  Edit here if a wider policy-sensitivity grid is
# required, but do not compare HV/IGD+ across different scenarios directly.
MAX_ACTIVE_ACTION_SCENARIOS = (10,)
COVERAGE_THRESHOLD_SCENARIOS = (0.30, 0.40, 0.50)

IPEA_EXPLORATION_RATE = 0.20
UNIFORM_EXPLORATION_RATE = 1.00

EXPERIMENT_POPULATION_MULTIPLIER = 4
EXPERIMENT_MIN_POPULATION_SIZE = 100
EXPERIMENT_MAX_POPULATION_SIZE = 500
FIXED_GENERATION_BUDGET = True
RANDOMIZE_EXECUTION_ORDER_WITHIN_SEED = True
SAVE_EACH_FRONT = True

MOEAD_CONFIG = ModifiedMOEADConfig(
    lattice_divisions=23,
    neighborhood_size=20,
    de_scale_factor=0.50,
    de_crossover_rate=0.50,
    constraint_penalty_factor=100.0,
    print_every_generations=10,
)

OUTPUT_DIR = SCRIPT_DIR / "comparison_five_algorithms_formal_K10_V4"
FRONT_DIR = OUTPUT_DIR / "fronts"
LOG_DIR = OUTPUT_DIR / "logs"
HV_FE_DIR = OUTPUT_DIR / "hv_fe_curves"
LIVE_MONITOR_DIR = OUTPUT_DIR / "live_monitor"
OUTPUT_EXCEL = OUTPUT_DIR / "Comprehensive_results.xlsx"

# Per-run checkpointing and restart behavior. A run is considered complete only
# after its completion marker has been atomically written.
ENABLE_RESUME = True
RETRY_FAILED_RUNS_ON_RESUME = True
STRICT_RESUME_SIGNATURE = True
RESUME_SCHEMA_VERSION = 1
RESUME_DIR = OUTPUT_DIR / "_resume"
RESUME_RUN_DIR = RESUME_DIR / "runs"
RESUME_MANIFEST = RESUME_DIR / "manifest.json"
RESUME_INDEX = RESUME_DIR / "run_metrics_partial.csv"

HV_REFERENCE_POINT = np.asarray([1.05, 1.05, 1.05], dtype=float)
MAX_REFERENCE_FRONT_SIZE = 5000
# Set in main() to the common initial MO-SHADE population size.
METRIC_FRONT_CAP: int | None = None
# Retained for backward-compatible configuration reporting. The current
# cumulative HV-FE implementation intentionally does not downsample checkpoint
# fronts because independent crowding truncation can alter exact HV and break
# monotonicity. The accelerated exact 3D sweep makes such truncation unnecessary.
MAX_HV_FE_FRONT_SIZE = 300
METRIC_EPS = 1e-12
# Block size for vectorized IGD+, epsilon+, and coverage calculations.
METRIC_BLOCK_SIZE = 256

# FE-aligned convergence monitoring. Each checkpoint uses the most recent
# archive available at or before the requested common FE share.
HV_FE_CHECKPOINT_FRACTIONS = (0.10, 0.25, 0.50, 0.75, 1.00)
SAVE_HV_FE_PLOTS = True
ENABLE_LIVE_MONITORING = True
# Recomputing all provisional Pareto metrics after every seed is quadratic in
# the number of completed seeds. Refresh every N paired seeds instead.
LIVE_MONITOR_EVERY_N_SEEDS = 5
LIVE_MONITOR_DECIMALS = 6


# =============================================================================
# 2. Algorithms
# =============================================================================
@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    family: str
    exploration_rate: float
    initialization: str
    optimizer: str


ALGORITHMS: tuple[AlgorithmSpec, ...] = (
    AlgorithmSpec(
        name="M0_uniform_MO_SHADE",
        family="MO_SHADE",
        exploration_rate=UNIFORM_EXPLORATION_RATE,
        initialization="uniform",
        optimizer="standard_MO_SHADE",
    ),
    AlgorithmSpec(
        name="M1_IPEA_MO_SHADE",
        family="MO_SHADE",
        exploration_rate=IPEA_EXPLORATION_RATE,
        initialization="ipea_priority",
        optimizer="standard_MO_SHADE",
    ),
    AlgorithmSpec(
        name="M1-I_threshold_MO_SHADE",
        family="MO_SHADE",
        exploration_rate=IPEA_EXPLORATION_RATE,
        initialization="threshold_stratified",
        optimizer="standard_MO_SHADE",
    ),
    AlgorithmSpec(
        name="modified_MOEA_D",
        family="MOEA_D",
        exploration_rate=UNIFORM_EXPLORATION_RATE,
        initialization="uniform",
        optimizer="modified_MOEA_D",
    ),
    AlgorithmSpec(
        name="SHAMODE_2019",
        family="SHAMODE",
        exploration_rate=UNIFORM_EXPLORATION_RATE,
        initialization="uniform",
        optimizer="literature_SHAMODE_2019",
    ),
)
ALGORITHM_NAMES = tuple(spec.name for spec in ALGORITHMS)
PAIRWISE_COMPARISONS = tuple(itertools.combinations(ALGORITHM_NAMES, 2))

# Direction: +1 means larger is better; -1 means smaller is better.
PRIMARY_TEST_METRICS: dict[str, int] = {
    "hypervolume": +1,
    "igd_plus": -1,
    "epsilon_plus": -1,
    "runtime_seconds": -1,
    "feasible_rate": +1,
    "nontrivial_feasible_rate": +1,
    "first_fe_nontrivial_feasible_95": -1,
    "hv_fe_auc": +1,
}


# =============================================================================
# 3. Core loading and state control
# =============================================================================
def load_core_module(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Core optimizer not found: {path}. Keep it beside this runner or "
            "edit CORE_FILE."
        )
    module_name = "hotel_moo_comparison_core"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import the core optimizer from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class OriginalCoreState:
    priority_exploration_rate: float
    max_active_actions: int | None
    coverage_threshold: float
    seed: int
    n_generations: int
    min_generations_before_stop: int
    convergence_window: int
    print_every: int
    population_multiplier: int
    min_population_size: int
    max_population_size: int


def capture_original_state(core: Any) -> OriginalCoreState:
    return OriginalCoreState(
        priority_exploration_rate=float(core.PRIORITY_EXPLORATION_RATE),
        max_active_actions=core.MAX_ACTIVE_ACTIONS,
        coverage_threshold=float(core.HIGH_PRIORITY_COVERAGE_MIN),
        seed=int(core.SEED),
        n_generations=int(core.N_GENERATIONS),
        min_generations_before_stop=int(core.MIN_GENERATIONS_BEFORE_STOP),
        convergence_window=int(core.CONVERGENCE_WINDOW),
        print_every=int(core.PRINT_EVERY),
        population_multiplier=int(core.POPULATION_MULTIPLIER),
        min_population_size=int(core.MIN_POPULATION_SIZE),
        max_population_size=int(core.MAX_POPULATION_SIZE),
    )


def restore_original_state(core: Any, state: OriginalCoreState) -> None:
    core.PRIORITY_EXPLORATION_RATE = state.priority_exploration_rate
    core.MAX_ACTIVE_ACTIONS = state.max_active_actions
    core.HIGH_PRIORITY_COVERAGE_MIN = state.coverage_threshold
    core.SEED = state.seed
    core.N_GENERATIONS = state.n_generations
    core.MIN_GENERATIONS_BEFORE_STOP = state.min_generations_before_stop
    core.CONVERGENCE_WINDOW = state.convergence_window
    core.PRINT_EVERY = state.print_every
    core.POPULATION_MULTIPLIER = state.population_multiplier
    core.MIN_POPULATION_SIZE = state.min_population_size
    core.MAX_POPULATION_SIZE = state.max_population_size


def apply_common_scenario(
    core: Any,
    state: OriginalCoreState,
    patch_state: Any,
    algorithm: AlgorithmSpec,
    *,
    seed: int,
    generations: int,
    max_active_actions: int,
    coverage_threshold: float,
) -> None:
    restore_original_state(core, state)
    restore_core_patch_state(core, patch_state)

    core.SEED = int(seed)
    core.N_GENERATIONS = int(generations)
    core.MAX_ACTIVE_ACTIONS = int(max_active_actions)
    core.HIGH_PRIORITY_COVERAGE_MIN = float(coverage_threshold)
    core.PRIORITY_EXPLORATION_RATE = float(algorithm.exploration_rate)
    core.POPULATION_MULTIPLIER = int(EXPERIMENT_POPULATION_MULTIPLIER)
    core.MIN_POPULATION_SIZE = int(EXPERIMENT_MIN_POPULATION_SIZE)
    core.MAX_POPULATION_SIZE = int(EXPERIMENT_MAX_POPULATION_SIZE)
    core.PRINT_EVERY = int(generations) + 1

    if algorithm.name == "M0_uniform_MO_SHADE":
        install_variant(core, patch_state, "M0_uniform_initialization")
    elif algorithm.name == "M1_IPEA_MO_SHADE":
        install_variant(core, patch_state, "M1_IPEA_priority_initialization")
    elif algorithm.name == "M1-I_threshold_MO_SHADE":
        install_variant(
            core,
            patch_state,
            "M1-I_threshold_stratified_initialization",
        )
    elif algorithm.name in {"modified_MOEA_D", "SHAMODE_2019"}:
        # External literature algorithms use their own priority-neutral
        # initialization and reproduction mechanisms.
        restore_core_patch_state(core, patch_state)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm.name}.")

    if FIXED_GENERATION_BUDGET:
        core.MIN_GENERATIONS_BEFORE_STOP = int(generations) + 1
        core.CONVERGENCE_WINDOW = int(generations) + 1


# =============================================================================
# 4. Pareto quality metrics
# =============================================================================
def nondominated_mask_minimization(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("points must be a non-empty two-dimensional array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("points contain non-finite values.")

    n, dimension = values.shape
    if dimension != 3:
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            dominated = np.any(
                np.all(values <= values[i] + METRIC_EPS, axis=1)
                & np.any(values < values[i] - METRIC_EPS, axis=1)
            )
            if dominated:
                keep[i] = False
        return keep

    order = np.lexsort((values[:, 2], values[:, 1], values[:, 0]))
    y_unique = np.unique(values[:, 1])
    y_rank = np.searchsorted(y_unique, values[:, 1], side="left") + 1
    tree = np.full(len(y_unique) + 1, np.inf, dtype=float)

    def query(rank: int) -> float:
        result = np.inf
        while rank > 0:
            result = min(result, float(tree[rank]))
            rank -= rank & -rank
        return result

    def update(rank: int, z_value: float) -> None:
        while rank < len(tree):
            if z_value < tree[rank]:
                tree[rank] = z_value
            rank += rank & -rank

    keep = np.zeros(n, dtype=bool)
    exact_seen: set[tuple[float, float, float]] = set()
    for index in order:
        point = tuple(float(value) for value in values[index])
        if point in exact_seen:
            continue
        exact_seen.add(point)
        if query(int(y_rank[index])) <= values[index, 2] + METRIC_EPS:
            continue
        keep[index] = True
        update(int(y_rank[index]), float(values[index, 2]))
    return keep


def crowding_downsample(points: np.ndarray, maximum_size: int) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if len(values) <= maximum_size:
        return values
    distances = np.zeros(len(values), dtype=float)
    for objective in range(values.shape[1]):
        order = np.argsort(values[:, objective], kind="mergesort")
        minimum = float(values[order[0], objective])
        maximum = float(values[order[-1], objective])
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        span = maximum - minimum
        if span <= METRIC_EPS:
            continue
        internal = order[1:-1]
        increments = (
            values[order[2:], objective] - values[order[:-2], objective]
        ) / span
        finite = np.isfinite(distances[internal])
        distances[internal[finite]] += increments[finite]
    selected = np.argsort(-distances, kind="mergesort")[:maximum_size]
    return values[selected]


def hypervolume_2d_minimization(
    points: np.ndarray,
    reference_y: float,
    reference_z: float,
) -> float:
    """Exact 2D minimization HV in O(n log n).

    The previous implementation repeatedly rescanned all points for every
    unique y coordinate, which was quadratic. Sorting once and using a
    cumulative minimum produces the same slice areas.
    """
    values = np.asarray(points, dtype=float)
    if values.size == 0:
        return 0.0
    values = values[
        (values[:, 0] < float(reference_y))
        & (values[:, 1] < float(reference_z))
    ]
    if values.size == 0:
        return 0.0

    order = np.argsort(values[:, 0], kind="mergesort")
    y = values[order, 0]
    z = values[order, 1]

    unique_y, starts = np.unique(y, return_index=True)
    minimum_z_at_y = np.minimum.reduceat(z, starts)
    best_z = np.minimum.accumulate(minimum_z_at_y)

    widths = np.diff(np.concatenate((unique_y, [float(reference_y)])))
    heights = np.maximum(float(reference_z) - best_z, 0.0)
    return float(np.dot(np.maximum(widths, 0.0), heights))


def hypervolume_3d_minimization(
    points: np.ndarray,
    reference: np.ndarray = HV_REFERENCE_POINT,
    *,
    assume_nondominated: bool = False,
) -> float:
    """Exact 3D minimization HV using an incremental 2D skyline sweep.

    After one nondominated filter, points are swept in ascending x. The active
    y-z projection is maintained as a two-dimensional skyline, avoiding the
    former nested repeated scans. The result is numerically equivalent to the
    original slicing definition but substantially faster for fronts containing
    hundreds of points.
    """
    values = np.asarray(points, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape (n, 3).")
    if reference.shape != (3,):
        raise ValueError("reference must have shape (3,).")

    values = values[np.all(values < reference, axis=1)]
    if values.size == 0:
        return 0.0
    if not assume_nondominated:
        values = values[nondominated_mask_minimization(values)]

    order = np.lexsort((values[:, 2], values[:, 1], values[:, 0]))
    values = values[order]
    x_values, starts = np.unique(values[:, 0], return_index=True)
    ends = np.concatenate((starts[1:], [len(values)]))

    skyline_y: list[float] = []
    skyline_z: list[float] = []
    volume = 0.0

    for position, (x_value, start_index, end_index) in enumerate(
        zip(x_values, starts, ends)
    ):
        for y_value, z_value in values[start_index:end_index, 1:]:
            y = float(y_value)
            z = float(z_value)
            insertion = bisect_left(skyline_y, y)

            # Equal-y points retain only the better z value.
            if insertion < len(skyline_y) and skyline_y[insertion] == y:
                if skyline_z[insertion] <= z + METRIC_EPS:
                    continue
                del skyline_y[insertion]
                del skyline_z[insertion]

            # A predecessor with no worse z dominates the new y-z point.
            if insertion > 0 and skyline_z[insertion - 1] <= z + METRIC_EPS:
                continue

            # Remove the contiguous successor range dominated by the new point.
            while (
                insertion < len(skyline_y)
                and skyline_z[insertion] >= z - METRIC_EPS
            ):
                del skyline_y[insertion]
                del skyline_z[insertion]

            skyline_y.insert(insertion, y)
            skyline_z.insert(insertion, z)

        next_x = (
            float(x_values[position + 1])
            if position + 1 < len(x_values)
            else float(reference[0])
        )
        width_x = max(next_x - float(x_value), 0.0)
        if width_x <= 0.0 or not skyline_y:
            continue

        y_array = np.asarray(skyline_y, dtype=float)
        z_array = np.asarray(skyline_z, dtype=float)
        widths_y = np.diff(
            np.concatenate((y_array, [float(reference[1])]))
        )
        heights_z = np.maximum(float(reference[2]) - z_array, 0.0)
        area_yz = float(np.dot(np.maximum(widths_y, 0.0), heights_z))
        volume += width_x * area_yz

    return float(volume)


def igd_plus(
    approximation: np.ndarray,
    reference_front: np.ndarray,
    *,
    block_size: int = METRIC_BLOCK_SIZE,
) -> float:
    """IGD+ using bounded-memory vectorized blocks."""
    approximation = np.asarray(approximation, dtype=float)
    reference_front = np.asarray(reference_front, dtype=float)
    if approximation.ndim != 2 or reference_front.ndim != 2:
        raise ValueError("approximation and reference_front must be 2D arrays.")
    if len(approximation) == 0 or len(reference_front) == 0:
        return np.nan

    total = 0.0
    count = 0
    block_size = max(int(block_size), 1)
    for start_index in range(0, len(reference_front), block_size):
        block = reference_front[start_index:start_index + block_size]
        differences = np.maximum(
            approximation[None, :, :] - block[:, None, :],
            0.0,
        )
        squared = np.einsum(
            "ijk,ijk->ij",
            differences,
            differences,
            optimize=True,
        )
        total += float(np.sqrt(squared.min(axis=1)).sum())
        count += int(len(block))
    return float(total / count)


def additive_epsilon_plus(
    approximation: np.ndarray,
    reference_front: np.ndarray,
    *,
    block_size: int = METRIC_BLOCK_SIZE,
) -> float:
    """Unary additive epsilon indicator; smaller is better."""
    approximation = np.asarray(approximation, dtype=float)
    reference_front = np.asarray(reference_front, dtype=float)
    if approximation.ndim != 2 or reference_front.ndim != 2:
        raise ValueError("approximation and reference_front must be 2D arrays.")
    if len(approximation) == 0 or len(reference_front) == 0:
        return np.nan

    required_maximum = -np.inf
    block_size = max(int(block_size), 1)
    for start_index in range(0, len(reference_front), block_size):
        block = reference_front[start_index:start_index + block_size]
        required = np.max(
            approximation[None, :, :] - block[:, None, :],
            axis=2,
        ).min(axis=1)
        required_maximum = max(required_maximum, float(required.max()))
    return float(required_maximum)


def spacing_metric(points: np.ndarray) -> float:
    values = np.asarray(points, dtype=float)
    if len(values) < 3:
        return np.nan
    distances = cdist(values, values, metric="euclidean")
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    return float(np.std(nearest, ddof=1))


def objective_extent(points: np.ndarray) -> float:
    values = np.asarray(points, dtype=float)
    if len(values) == 0:
        return np.nan
    return float(np.mean(values.max(axis=0) - values.min(axis=0)))


def semantic_front_to_minimization(front: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            -front["reputation_improvement"].to_numpy(dtype=float),
            -front["choice_probability_gain"].to_numpy(dtype=float),
            front["effective_cost"].to_numpy(dtype=float),
        ]
    )


def normalize_fronts_within_scenario(
    scenario_fronts: dict[tuple[str, int], pd.DataFrame],
) -> tuple[
    dict[tuple[str, int], np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    raw: dict[tuple[str, int], np.ndarray] = {}
    for key, frame in scenario_fronts.items():
        if frame.empty:
            continue
        values = semantic_front_to_minimization(frame)
        values = values[nondominated_mask_minimization(values)]
        if METRIC_FRONT_CAP is not None:
            values = crowding_downsample(values, int(METRIC_FRONT_CAP))
        raw[key] = values
    if not raw:
        raise ValueError("No successful Pareto fronts exist in the scenario.")
    pooled = np.vstack(list(raw.values()))
    ideal = pooled.min(axis=0)
    nadir = pooled.max(axis=0)
    span = np.where(nadir - ideal > METRIC_EPS, nadir - ideal, 1.0)
    normalized = {
        key: np.clip((values - ideal) / span, 0.0, 1.0)
        for key, values in raw.items()
    }
    pooled_normalized = np.vstack(list(normalized.values()))
    reference = pooled_normalized[
        nondominated_mask_minimization(pooled_normalized)
    ]
    reference = crowding_downsample(reference, MAX_REFERENCE_FRONT_SIZE)
    return normalized, reference, ideal, nadir


def coverage_indicator(
    a: np.ndarray,
    b: np.ndarray,
    *,
    block_size: int = METRIC_BLOCK_SIZE,
) -> float:
    """C(A,B), evaluated in bounded-memory vectorized blocks."""
    A = np.asarray(a, dtype=float)
    B = np.asarray(b, dtype=float)
    if len(B) == 0:
        return np.nan
    if len(A) == 0:
        return 0.0

    n_covered = 0
    block_size = max(int(block_size), 1)
    for start_index in range(0, len(B), block_size):
        block = B[start_index:start_index + block_size]
        dominated = np.any(
            np.all(
                A[None, :, :] <= block[:, None, :] + METRIC_EPS,
                axis=2,
            ),
            axis=1,
        )
        n_covered += int(dominated.sum())
    return float(n_covered / len(B))



# =============================================================================
# 4B. Pareto-archive snapshots for FE-aligned HV curves
# =============================================================================
def archive_to_semantic_minimization(
    archive: Sequence[Any],
    *,
    clean: bool = True,
) -> np.ndarray:
    """Convert a feasible archive to the three raw minimization objectives.

    The third component deliberately uses effective implementation cost rather
    than the optimizer's internally normalized cost.  Scenario-level joint
    normalization is applied later, exactly as for final-front metrics.
    """
    rows: list[list[float]] = []
    for individual in archive:
        if float(getattr(individual, "violation", np.inf)) > METRIC_EPS:
            continue
        components = getattr(individual, "components", {})
        try:
            rows.append(
                [
                    -float(components["reputation_improvement"]),
                    -float(components["probability_gain"]),
                    float(components["effective_cost"]),
                ]
            )
        except (KeyError, TypeError, ValueError):
            continue
    if not rows:
        return np.empty((0, 3), dtype=float)
    values = np.asarray(rows, dtype=float)
    if not clean:
        return values
    values = np.unique(np.round(values, 12), axis=0)
    return values[nondominated_mask_minimization(values)]


class ParetoSnapshotRecorder:
    """Capture Pareto archives at FE-aligned checkpoints.

    The two solvers update archives through different call paths:

    * MO-SHADE calls ``core.update_pareto_archive`` once after initialization
      and once after every generation.
    * modified MOEA/D calls the adapter-level ``update_external_archive`` once
      after initialization and once after every completed generation.

    Wrapping the adapter-level function for MOEA/D is essential.  Wrapping only
    ``core.update_pareto_archive`` is not reliable because the adapter may hold
    or route archive updates through its own module-level helper.
    """

    def __init__(
        self,
        core: Any,
        moead_module: Any,
        *,
        algorithm_family: str,
        evaluation_budget: int,
        initial_population_size: int,
    ) -> None:
        self.core = core
        self.moead_module = moead_module
        self.algorithm_family = str(algorithm_family)
        self.evaluation_budget = int(evaluation_budget)
        self.initial_population_size = int(initial_population_size)

        self.original_core_update = core.update_pareto_archive
        self.original_moead_update = moead_module.update_external_archive

        self.call_count = 0
        # Store every archive state in update order for both MO-SHADE and
        # modified MOEA/D.  FE values are attached later from the solver's
        # history table, which avoids estimating FE from callback counts.
        self.all_states: list[np.ndarray] = []

    def _capture_moead_state(
        self,
        archive: Sequence[Any],
    ) -> None:
        """Record every MOEA/D archive update in chronological order.

        The first state corresponds to the fully evaluated initial population;
        each subsequent state corresponds to one history row (one completed
        generation).  Checkpoint selection is performed only in ``finalize``.
        """
        self.call_count += 1
        points = archive_to_semantic_minimization(archive, clean=False)
        self.all_states.append(points)

    def __enter__(self) -> "ParetoSnapshotRecorder":
        if self.algorithm_family == "MO_SHADE":
            def wrapped_core_update(
                archive: Sequence[Any],
                candidates: Sequence[Any],
            ):
                result = self.original_core_update(archive, candidates)
                self.call_count += 1
                points = archive_to_semantic_minimization(result, clean=False)
                self.all_states.append(points)
                return result

            self.core.update_pareto_archive = wrapped_core_update

        elif self.algorithm_family == "MOEA_D":
            def wrapped_moead_update(
                base: Any,
                archive: Sequence[Any],
                candidates: Sequence[Any],
                archive_cap: int | None = None,
            ):
                result = self.original_moead_update(
                    base,
                    archive,
                    candidates,
                    archive_cap=archive_cap,
                )
                self._capture_moead_state(result)
                return result

            # run_modified_moead resolves this helper from its module globals,
            # so replacing it here intercepts every true MOEA/D archive update.
            self.moead_module.update_external_archive = wrapped_moead_update
        else:
            raise ValueError(
                f"Unsupported algorithm family for snapshot recording: "
                f"{self.algorithm_family!r}."
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.core.update_pareto_archive = self.original_core_update
        self.moead_module.update_external_archive = self.original_moead_update

    def finalize(
        self,
        history: pd.DataFrame,
        *,
        completed_evaluations: int,
        final_archive: Sequence[Any],
    ) -> list[dict[str, Any]]:
        final_points = archive_to_semantic_minimization(final_archive)
        timeline: list[tuple[int, np.ndarray]] = []

        if self.all_states:
            # State 0 is the archive after the complete initial population.
            timeline.append(
                (self.initial_population_size, self.all_states[0].copy())
            )

            # Every later state is paired with the exact FE stored by the
            # solver after that generation.  This creates a complete timeline,
            # so selecting the latest state at or before each target FE is valid.
            n_generation_states = min(
                len(history), max(len(self.all_states) - 1, 0)
            )
            for position in range(n_generation_states):
                fe = int(history.iloc[position]["function_evaluations"])
                timeline.append((fe, self.all_states[position + 1].copy()))

        if self.algorithm_family == "MOEA_D":
            expected_states = len(history) + 1
            if len(self.all_states) != expected_states:
                raise RuntimeError(
                    "Incomplete modified MOEA/D archive timeline: captured "
                    f"{len(self.all_states)} states, expected {expected_states} "
                    "(initial archive plus one state per history row); "
                    f"archive-update calls={self.call_count}."
                )

        if not timeline:
            raise RuntimeError(
                f"No archive states were captured for {self.algorithm_family}."
            )

        # Retain the exact final archive at the final FE.  For MOEA/D the 100%
        # checkpoint should already have been captured after the final batch,
        # but replacing it here guarantees identity with the returned archive.
        if timeline[-1][0] < int(completed_evaluations):
            timeline.append((int(completed_evaluations), final_points.copy()))
        elif timeline[-1][0] == int(completed_evaluations):
            timeline[-1] = (int(completed_evaluations), final_points.copy())

        timeline.sort(key=lambda item: item[0])
        records: list[dict[str, Any]] = []
        for fraction in HV_FE_CHECKPOINT_FRACTIONS:
            target_fe = max(
                self.initial_population_size,
                int(math.ceil(float(fraction) * self.evaluation_budget)),
            )
            candidates = [item for item in timeline if item[0] <= target_fe]
            if not candidates:
                # The initial complete population is the earliest valid state.
                chosen_fe, chosen_points = timeline[0]
            else:
                chosen_fe, chosen_points = candidates[-1]
            if chosen_points.size:
                chosen_points = np.unique(
                    np.round(np.asarray(chosen_points, dtype=float), 12),
                    axis=0,
                )
                chosen_points = chosen_points[
                    nondominated_mask_minimization(chosen_points)
                ]
            records.append(
                {
                    "checkpoint_fraction": float(fraction),
                    "target_function_evaluations": int(target_fe),
                    "function_evaluations": int(chosen_fe),
                    "points": chosen_points.copy(),
                }
            )



        return records


# =============================================================================
# 5. Run helpers
# =============================================================================
def scenario_tag(k: int, rho: float) -> str:
    return f"K{k}_rho{rho:.2f}".replace(".", "p")


def first_axis_at_or_above(
    history: pd.DataFrame,
    value_column: str,
    threshold: float,
    axis_column: str = "function_evaluations",
) -> float:
    if value_column not in history.columns or axis_column not in history.columns:
        return np.nan
    values = pd.to_numeric(history[value_column], errors="coerce")
    rows = history.loc[values >= threshold]
    return np.nan if rows.empty else float(rows.iloc[0][axis_column])


def active_es_from_recommendation(
    recommended: pd.Series,
    options: Sequence[Any],
    minimum_action: float,
) -> tuple[str, ...]:
    active: list[str] = []
    for option in options:
        column = f"x_{option.es}"
        if column in recommended.index and float(recommended[column]) >= (
            float(minimum_action) - METRIC_EPS
        ):
            active.append(str(option.es))
    return tuple(sorted(active))


def validate_front(
    pareto: pd.DataFrame,
    max_active_actions: int,
    coverage_threshold: float,
) -> None:
    if pareto.empty:
        raise RuntimeError("The feasible Pareto archive is empty.")
    if int(pareto["n_active_actions"].max()) > int(max_active_actions):
        raise RuntimeError("Pareto archive violates MAX_ACTIVE_ACTIONS.")
    if float(pareto["high_priority_coverage"].min()) < (
        float(coverage_threshold) - 1e-8
    ):
        raise RuntimeError("Pareto archive violates the coverage threshold.")


def select_common_representative(
    core: Any,
    pareto: pd.DataFrame,
) -> pd.Series:
    ranked = core.calculate_relative_robust_scores(
        pareto,
        epsilon=float(core.ROBUST_EPSILON),
    )
    _, representatives = core.select_robust_representatives(ranked)
    if representatives.empty:
        raise RuntimeError("No common robust representative was selected.")
    return representatives.iloc[0]


def convert_shamode_snapshots(
    snapshots: Sequence[Any],
    *,
    evaluation_budget: int,
    initial_population_size: int,
) -> list[dict[str, Any]]:
    """Convert SHAMODE trace objects to the common left-continuous FE records."""
    if not snapshots:
        return []
    ordered = sorted(snapshots, key=lambda item: int(item.actual_fe))
    records: list[dict[str, Any]] = []
    for fraction in HV_FE_CHECKPOINT_FRACTIONS:
        target_fe = max(
            int(initial_population_size),
            int(math.ceil(float(fraction) * int(evaluation_budget))),
        )
        candidates = [item for item in ordered if int(item.actual_fe) <= target_fe]
        chosen = candidates[-1] if candidates else ordered[0]
        # SHAMODE stores optimizer-internal objectives. Convert the normalized
        # cost component back to effective cost for the common metric scale.
        points = np.asarray(chosen.objectives, dtype=float)
        if points.size:
            points = points.copy()
            points[:, 2] *= float(_CURRENT_CONTEXT_FOR_SNAPSHOTS.max_effective_cost)
            points = points[nondominated_mask_minimization(points)]
        records.append(
            {
                "checkpoint_fraction": float(fraction),
                "target_function_evaluations": int(target_fe),
                "function_evaluations": int(chosen.actual_fe),
                "points": points,
            }
        )
    return records


_CURRENT_CONTEXT_FOR_SNAPSHOTS: Any = None

def run_one_experiment(
    core: Any,
    state: OriginalCoreState,
    patch_state: Any,
    algorithm: AlgorithmSpec,
    *,
    seed: int,
    generations: int,
    max_active_actions: int,
    coverage_threshold: float,
    options: Sequence[Any],
    context: Any,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, Any]],
]:
    apply_common_scenario(
        core,
        state,
        patch_state,
        algorithm,
        seed=seed,
        generations=generations,
        max_active_actions=max_active_actions,
        coverage_threshold=coverage_threshold,
    )
    if hasattr(core, "CURRENT_CONTEXT"):
        core.CURRENT_CONTEXT = context

    evaluation_budget = planned_mo_shade_evaluations(
        core,
        len(options),
        generations=generations,
    )
    tag = scenario_tag(max_active_actions, coverage_threshold)
    log_path = LOG_DIR / f"{tag}_{algorithm.name}_seed_{seed}.log"
    start = time.perf_counter()

    if algorithm.family in {"MO_SHADE", "SHAMODE"}:
        initial_population_size = int(
            np.clip(
                int(core.POPULATION_MULTIPLIER) * len(options),
                int(core.MIN_POPULATION_SIZE),
                int(core.MAX_POPULATION_SIZE),
            )
        )
    else:
        initial_population_size = int(
            math.comb(MOEAD_CONFIG.lattice_divisions + 2, 2)
        )

    try:
        if algorithm.family == "SHAMODE":
            global _CURRENT_CONTEXT_FOR_SNAPSHOTS
            _CURRENT_CONTEXT_FOR_SNAPSHOTS = context
            # Dense internal trace; converted below to the five common FE shares.
            shamode_adapter.TRACE_CHECKPOINTS = 101
            with log_path.open("w", encoding="utf-8") as log_file:
                with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(
                    log_file
                ):
                    archive, history, metadata, shamode_snapshots = (
                        shamode_adapter.run_shamode_2019(
                            core,
                            options,
                            context,
                            seed=int(seed),
                            evaluation_budget=int(evaluation_budget),
                        )
                    )
            completed_evaluations = int(metadata["function_evaluations_completed"])
            snapshot_records = convert_shamode_snapshots(
                shamode_snapshots,
                evaluation_budget=int(evaluation_budget),
                initial_population_size=int(initial_population_size),
            )
        else:
            recorder = ParetoSnapshotRecorder(
                core,
                moead_adapter,
                algorithm_family=algorithm.family,
                evaluation_budget=int(evaluation_budget),
                initial_population_size=initial_population_size,
            )
            with recorder:
                with log_path.open("w", encoding="utf-8") as log_file:
                    with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(
                        log_file
                    ):
                        if algorithm.family == "MO_SHADE":
                            archive, history, metadata = core.run_mo_shade(options, context)
                            history = add_mo_shade_evaluation_axis(
                                core,
                                history,
                                len(options),
                                generations=generations,
                            )
                            completed_evaluations = int(
                                history["function_evaluations"].iloc[-1]
                            )
                        elif algorithm.family == "MOEA_D":
                            archive, history, metadata = run_modified_moead(
                                core,
                                options,
                                context,
                                seed=int(seed),
                                evaluation_budget=int(evaluation_budget),
                                config=MOEAD_CONFIG,
                                archive_cap=int(core.MAX_PARETO_ARCHIVE_SIZE),
                            )
                            completed_evaluations = int(
                                metadata["function_evaluations_completed"]
                            )
                        else:
                            raise ValueError(
                                f"Unsupported algorithm family: {algorithm.family}."
                            )

            snapshot_records = recorder.finalize(
                history,
                completed_evaluations=int(completed_evaluations),
                final_archive=archive,
            )

        runtime = time.perf_counter() - start
        if not archive:
            raise RuntimeError("No feasible Pareto solution was returned.")

        pareto, _ = core.pareto_dataframe(archive, options)
        validate_front(pareto, max_active_actions, coverage_threshold)
        recommended = select_common_representative(core, pareto)
        active_es = active_es_from_recommendation(
            recommended,
            options,
            float(core.MIN_ACTION_MAGNITUDE),
        )

        first = history.iloc[0]
        last = history.iloc[-1]
        metrics: dict[str, Any] = {
            "status": "success",
            "error": "",
            "algorithm": algorithm.name,
            "algorithm_family": algorithm.family,
            "seed": int(seed),
            "max_active_actions": int(max_active_actions),
            "coverage_threshold": float(coverage_threshold),
            "scenario": tag,
            "generations_requested_for_mo_shade": int(generations),
            "generations_completed": int(metadata["generations_completed"]),
            "function_evaluation_budget": int(evaluation_budget),
            "function_evaluations_completed": int(completed_evaluations),
            "runtime_seconds": float(runtime),
            "pareto_size": int(len(pareto)),
            "first_recorded_feasible_fraction": float(first["feasible_fraction"]),
            "first_recorded_nontrivial_feasible_fraction": float(
                first.get("nontrivial_feasible_fraction", np.nan)
            ),
            "first_recorded_mean_priority_coverage": float(
                first.get("mean_high_priority_coverage", np.nan)
            ),
            # FR is the final proportion of the current population satisfying
            # all common constraints.  The nontrivial version additionally
            # excludes the zero-action baseline.
            "feasible_rate": float(last["feasible_fraction"]),
            "nontrivial_feasible_rate": float(
                last.get("nontrivial_feasible_fraction", np.nan)
            ),
            "mean_feasible_rate": float(
                pd.to_numeric(history["feasible_fraction"], errors="coerce").mean()
            ),
            "mean_nontrivial_feasible_rate": float(
                pd.to_numeric(
                    history.get(
                        "nontrivial_feasible_fraction",
                        pd.Series(np.nan, index=history.index),
                    ),
                    errors="coerce",
                ).mean()
            ),
            "final_feasible_fraction": float(last["feasible_fraction"]),
            "final_nontrivial_feasible_fraction": float(
                last.get("nontrivial_feasible_fraction", np.nan)
            ),
            "first_fe_feasible_95": first_axis_at_or_above(
                history, "feasible_fraction", 0.95
            ),
            "first_fe_nontrivial_feasible_95": first_axis_at_or_above(
                history, "nontrivial_feasible_fraction", 0.95
            ),
            "pareto_mean_high_priority_coverage": float(
                pareto["high_priority_coverage"].mean()
            ),
            "pareto_min_high_priority_coverage": float(
                pareto["high_priority_coverage"].min()
            ),
            "pareto_mean_priority_alignment": float(
                pareto["priority_alignment"].mean()
            ),
            "recommended_solution_id": int(recommended["solution_id"]),
            "recommended_reputation": float(
                recommended["reputation_improvement"]
            ),
            "recommended_choice_gain": float(
                recommended["choice_probability_gain"]
            ),
            "recommended_choice_gain_pp": float(
                recommended["choice_probability_gain_pp"]
            ),
            "recommended_effective_cost": float(
                recommended["effective_cost"]
            ),
            "recommended_high_priority_coverage": float(
                recommended["high_priority_coverage"]
            ),
            "recommended_priority_alignment": float(
                recommended["priority_alignment"]
            ),
            "recommended_n_active_actions": int(
                recommended["n_active_actions"]
            ),
            "recommended_active_es": " | ".join(active_es),
            "n_recommended_active_es": int(len(active_es)),
            **{f"algorithm_{key}": value for key, value in asdict(algorithm).items()},
        }
        if algorithm.family == "MO_SHADE":
            metrics.update(
                {
                    f"extension_{key}": value
                    for key, value in get_variant_diagnostics(core).items()
                    if key != "variant"
                }
            )
        elif algorithm.family == "MOEA_D":
            metrics.update(
                {
                    "moead_population_size": int(metadata["population_size"]),
                    "moead_lattice_divisions": int(
                        metadata["simplex_lattice_divisions"]
                    ),
                    "moead_neighborhood_size": int(metadata["neighborhood_size"]),
                }
            )
        elif algorithm.family == "SHAMODE":
            metrics.update(
                {
                    "shamode_memory_size": int(metadata["memory_size"]),
                    "shamode_mutation_archive_rate": float(
                        metadata["mutation_archive_rate"]
                    ),
                    "shamode_final_partial_batch": bool(
                        metadata["final_partial_batch"]
                    ),
                }
            )

        pareto = pareto.copy()
        pareto.insert(0, "coverage_threshold", float(coverage_threshold))
        pareto.insert(0, "max_active_actions", int(max_active_actions))
        pareto.insert(0, "seed", int(seed))
        pareto.insert(0, "algorithm", algorithm.name)

        history = history.copy()
        history.insert(0, "coverage_threshold", float(coverage_threshold))
        history.insert(0, "max_active_actions", int(max_active_actions))
        history.insert(0, "seed", int(seed))
        history.insert(0, "algorithm", algorithm.name)

        for record in snapshot_records:
            record.update(
                {
                    "algorithm": algorithm.name,
                    "seed": int(seed),
                    "max_active_actions": int(max_active_actions),
                    "coverage_threshold": float(coverage_threshold),
                    "scenario": tag,
                    "function_evaluation_budget": int(evaluation_budget),
                }
            )
        return metrics, pareto, history, snapshot_records

    except Exception as exc:
        runtime = time.perf_counter() - start
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write("\n\nEXCEPTION\n")
            log_file.write(traceback.format_exc())
        return (
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "algorithm": algorithm.name,
                "algorithm_family": algorithm.family,
                "seed": int(seed),
                "max_active_actions": int(max_active_actions),
                "coverage_threshold": float(coverage_threshold),
                "scenario": tag,
                "generations_requested_for_mo_shade": int(generations),
                "function_evaluation_budget": int(evaluation_budget),
                "runtime_seconds": float(runtime),
            },
            pd.DataFrame(),
            pd.DataFrame(),
            [],
        )


# =============================================================================
# 6. Scenario-level quality calculation
# =============================================================================
def attach_quality_metrics(
    metrics: pd.DataFrame,
    fronts: dict[tuple[int, float, str, int], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = metrics.copy()
    quality_columns = [
        "hypervolume",
        "igd_plus",
        "epsilon_plus",
        "spacing",
        "objective_extent",
        "metric_front_size",
        "reference_front_size",
    ]
    for column in quality_columns:
        result[column] = np.nan

    coverage_rows: list[dict[str, Any]] = []
    successful = result.loc[result["status"] == "success"]
    scenarios = successful[
        ["max_active_actions", "coverage_threshold"]
    ].drop_duplicates()

    for k, rho in scenarios.itertuples(index=False, name=None):
        scenario_fronts: dict[tuple[str, int], pd.DataFrame] = {}
        for (fk, frho, algorithm, seed), frame in fronts.items():
            if int(fk) == int(k) and np.isclose(float(frho), float(rho)):
                scenario_fronts[(algorithm, int(seed))] = frame
        normalized, reference, ideal, nadir = normalize_fronts_within_scenario(
            scenario_fronts
        )

        for (algorithm, seed), values in normalized.items():
            mask = (
                (result["status"] == "success")
                & (result["max_active_actions"] == int(k))
                & np.isclose(result["coverage_threshold"], float(rho))
                & (result["algorithm"] == algorithm)
                & (result["seed"] == int(seed))
            )
            result.loc[mask, "hypervolume"] = hypervolume_3d_minimization(
                values, assume_nondominated=True
            )
            result.loc[mask, "igd_plus"] = igd_plus(values, reference)
            result.loc[mask, "epsilon_plus"] = additive_epsilon_plus(
                values, reference
            )
            result.loc[mask, "spacing"] = spacing_metric(values)
            result.loc[mask, "objective_extent"] = objective_extent(values)
            result.loc[mask, "metric_front_size"] = int(len(values))
            result.loc[mask, "reference_front_size"] = int(len(reference))
            for objective, label in enumerate(("rep", "choice", "cost")):
                result.loc[mask, f"scenario_ideal_{label}"] = float(ideal[objective])
                result.loc[mask, f"scenario_nadir_{label}"] = float(nadir[objective])

        common_seeds = sorted(
            set.intersection(
                *[
                    {
                        seed
                        for algorithm, seed in normalized
                        if algorithm == algorithm_name
                    }
                    for algorithm_name in ALGORITHM_NAMES
                ]
            )
        )
        for seed in common_seeds:
            for algorithm_a, algorithm_b in PAIRWISE_COMPARISONS:
                a = normalized[(algorithm_a, seed)]
                b = normalized[(algorithm_b, seed)]
                coverage_rows.append(
                    {
                        "max_active_actions": int(k),
                        "coverage_threshold": float(rho),
                        "seed": int(seed),
                        "algorithm_A": algorithm_a,
                        "algorithm_B": algorithm_b,
                        "C_A_dominates_B": coverage_indicator(a, b),
                        "C_B_dominates_A": coverage_indicator(b, a),
                    }
                )

    return result, pd.DataFrame(coverage_rows)




def summarize_pairwise_coverage(coverage: pd.DataFrame) -> pd.DataFrame:
    """Summarize paired C(A,B) and C(B,A) values by managerial scenario."""
    if coverage.empty:
        return pd.DataFrame()
    frame = coverage.copy()
    frame["coverage_advantage_A_minus_B"] = (
        frame["C_A_dominates_B"] - frame["C_B_dominates_A"]
    )
    rows: list[dict[str, Any]] = []
    group_columns = [
        "max_active_actions",
        "coverage_threshold",
        "algorithm_A",
        "algorithm_B",
    ]
    for keys, group in frame.groupby(group_columns, sort=True):
        k, rho, algorithm_a, algorithm_b = keys
        row: dict[str, Any] = {
            "max_active_actions": int(k),
            "coverage_threshold": float(rho),
            "algorithm_A": algorithm_a,
            "algorithm_B": algorithm_b,
            "n_paired_runs": int(len(group)),
        }
        for column in (
            "C_A_dominates_B",
            "C_B_dominates_A",
            "coverage_advantage_A_minus_B",
        ):
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            row[f"{column}_median"] = float(values.median())
        rows.append(row)
    return pd.DataFrame(rows)


def build_hv_fe_curve(
    metrics: pd.DataFrame,
    fronts: dict[
        tuple[int, float, str, int],
        pd.DataFrame,
    ],
    snapshot_records: Sequence[dict[str, Any]],
) -> pd.DataFrame:
    """
    Calculate cumulative HV and IGD+ at common FE checkpoints.

    For each fixed scenario, algorithm, and seed, checkpoint fronts are
    accumulated chronologically. Historical nondominated solutions are
    therefore retained even when an optimizer's capacity-limited archive
    later removes them.

    Scenario-level ideal and nadir points are derived from final Pareto fronts
    and remain fixed across all checkpoints.
    """
    if not snapshot_records:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    successful = metrics.loc[
        metrics["status"] == "success"
    ].copy()

    scenarios = successful[
        [
            "max_active_actions",
            "coverage_threshold",
        ]
    ].drop_duplicates()

    for k, rho in scenarios.itertuples(
        index=False,
        name=None,
    ):
        # =============================================================
        # 1. Construct fixed scenario-level normalization
        # =============================================================
        scenario_fronts: dict[
            tuple[str, int],
            pd.DataFrame,
        ] = {}

        for (
            fk,
            frho,
            algorithm,
            seed,
        ), frame in fronts.items():
            if (
                int(fk) == int(k)
                and np.isclose(
                    float(frho),
                    float(rho),
                )
            ):
                scenario_fronts[
                    (
                        str(algorithm),
                        int(seed),
                    )
                ] = frame

        if not scenario_fronts:
            continue

        _, reference, ideal, nadir = (
            normalize_fronts_within_scenario(
                scenario_fronts
            )
        )

        span = np.where(
            nadir - ideal > METRIC_EPS,
            nadir - ideal,
            1.0,
        )

        # =============================================================
        # 2. Select records belonging to this scenario
        # =============================================================
        scenario_records = [
            record
            for record in snapshot_records
            if (
                int(
                    record["max_active_actions"]
                )
                == int(k)
                and np.isclose(
                    float(
                        record[
                            "coverage_threshold"
                        ]
                    ),
                    float(rho),
                )
            )
        ]

        # =============================================================
        # 3. Group records by algorithm and seed
        # =============================================================
        run_groups: dict[
            tuple[str, int],
            list[dict[str, Any]],
        ] = {}

        for record in scenario_records:
            run_key = (
                str(record["algorithm"]),
                int(record["seed"]),
            )
            run_groups.setdefault(
                run_key,
                [],
            ).append(record)

        # =============================================================
        # 4. Construct the cumulative attained front for every run
        # =============================================================
        for (
            algorithm,
            seed,
        ), run_records in run_groups.items():

            ordered_records = sorted(
                run_records,
                key=lambda item: (
                    int(
                        item[
                            "function_evaluations"
                        ]
                    ),
                    float(
                        item[
                            "checkpoint_fraction"
                        ]
                    ),
                ),
            )

            # Reset only once per algorithm × seed run.
            cumulative_points = np.empty(
                (0, 3),
                dtype=float,
            )

            previous_hv = -np.inf

            for record in ordered_records:
                current_points = np.asarray(
                    record["points"],
                    dtype=float,
                )

                # -----------------------------------------------------
                # Add the current checkpoint front to historical points
                # -----------------------------------------------------
                if current_points.size:
                    if (
                        current_points.ndim != 2
                        or current_points.shape[1] != 3
                    ):
                        raise ValueError(
                            "HV-FE snapshot points must "
                            "have shape (n, 3). "
                            f"Received "
                            f"{current_points.shape} for "
                            f"K={k}, rho={rho}, "
                            f"algorithm={algorithm}, "
                            f"seed={seed}."
                        )

                    current_points = current_points[
                        np.all(
                            np.isfinite(
                                current_points
                            ),
                            axis=1,
                        )
                    ]

                    if current_points.size:
                        if cumulative_points.size:
                            cumulative_points = (
                                np.vstack(
                                    [
                                        cumulative_points,
                                        current_points,
                                    ]
                                )
                            )
                        else:
                            cumulative_points = (
                                current_points.copy()
                            )

                        # Remove exact or numerically equivalent duplicates.
                        cumulative_points = np.unique(
                            np.round(
                                cumulative_points,
                                decimals=12,
                            ),
                            axis=0,
                        )

                        # Keep the cumulative nondominated attained front.
                        cumulative_points = (
                            cumulative_points[
                                nondominated_mask_minimization(
                                    cumulative_points
                                )
                            ]
                        )

                # -----------------------------------------------------
                # Calculate quality metrics from the cumulative front
                # -----------------------------------------------------
                if cumulative_points.size == 0:
                    hv = 0.0
                    igd = np.nan
                    n_points = 0

                else:
                    normalized = (
                        cumulative_points - ideal
                    ) / span

                    normalized = np.clip(
                        normalized,
                        0.0,
                        1.0,
                    )

                    normalized = normalized[
                        nondominated_mask_minimization(
                            normalized
                        )
                    ]

                    # Do not independently downsample each checkpoint.
                    hv = (
                        hypervolume_3d_minimization(
                            normalized,
                            assume_nondominated=True,
                        )
                    )

                    igd = igd_plus(
                        normalized,
                        reference,
                    )

                    n_points = int(
                        len(normalized)
                    )

                # -----------------------------------------------------
                # Cumulative HV must not decrease
                # -----------------------------------------------------
                if (
                    np.isfinite(previous_hv)
                    and hv < previous_hv - 1e-10
                ):
                    raise RuntimeError(
                        "Non-monotone cumulative HV "
                        "detected: "
                        f"K={k}, rho={rho}, "
                        f"algorithm={algorithm}, "
                        f"seed={seed}, "
                        f"FE="
                        f"{record['function_evaluations']}, "
                        f"previous_HV="
                        f"{previous_hv:.12f}, "
                        f"current_HV={hv:.12f}."
                    )

                previous_hv = max(
                    previous_hv,
                    float(hv),
                )

                rows.append(
                    {
                        "max_active_actions": int(k),
                        "coverage_threshold": float(
                            rho
                        ),
                        "scenario": scenario_tag(
                            int(k),
                            float(rho),
                        ),
                        "algorithm": str(
                            algorithm
                        ),
                        "seed": int(seed),
                        "checkpoint_fraction": float(
                            record[
                                "checkpoint_fraction"
                            ]
                        ),
                        "target_function_evaluations": int(
                            record[
                                "target_function_evaluations"
                            ]
                        ),
                        "function_evaluations": int(
                            record[
                                "function_evaluations"
                            ]
                        ),
                        "function_evaluation_budget": int(
                            record[
                                "function_evaluation_budget"
                            ]
                        ),
                        "actual_fe_fraction": float(
                            int(
                                record[
                                    "function_evaluations"
                                ]
                            )
                            / max(
                                int(
                                    record[
                                        "function_evaluation_budget"
                                    ]
                                ),
                                1,
                            )
                        ),
                        "hypervolume": float(hv),
                        "igd_plus": (
                            float(igd)
                            if np.isfinite(igd)
                            else np.nan
                        ),
                        "snapshot_front_size": int(
                            n_points
                        ),
                    }
                )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            [
                "max_active_actions",
                "coverage_threshold",
                "algorithm",
                "seed",
                "actual_fe_fraction",
                "checkpoint_fraction",
            ],
            kind="mergesort",
        ).reset_index(drop=True)

    return result


def attach_hv_fe_auc(
    metrics: pd.DataFrame,
    curve: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach the area under the HV-FE curve.

    A common artificial origin, (FE fraction, HV) = (0, 0), is added and
    integration is performed over the common interval [0, 1].
    """
    result = metrics.copy()
    result["hv_fe_auc"] = np.nan

    if curve.empty:
        return result

    keys = [
        "max_active_actions",
        "coverage_threshold",
        "algorithm",
        "seed",
    ]

    for group_key, group in curve.groupby(
        keys,
        sort=False,
    ):
        ordered = group.sort_values(
            [
                "actual_fe_fraction",
                "checkpoint_fraction",
            ],
            kind="mergesort",
        )

        x = pd.to_numeric(
            ordered["actual_fe_fraction"],
            errors="coerce",
        ).to_numpy(dtype=float)

        y = pd.to_numeric(
            ordered["hypervolume"],
            errors="coerce",
        ).to_numpy(dtype=float)

        valid = (
            np.isfinite(x)
            & np.isfinite(y)
        )

        x = x[valid]
        y = y[valid]

        if len(x) == 0:
            auc = np.nan

        else:
            x = np.clip(
                x,
                0.0,
                1.0,
            )

            order = np.argsort(
                x,
                kind="mergesort",
            )

            x = x[order]
            y = y[order]

            # If several nominal checkpoints refer to the same actual FE,
            # retain the maximum cumulative HV at that FE.
            unique_curve = (
                pd.DataFrame(
                    {
                        "actual_fe_fraction": x,
                        "hypervolume": y,
                    }
                )
                .groupby(
                    "actual_fe_fraction",
                    as_index=False,
                    sort=True,
                )["hypervolume"]
                .max()
            )

            x = unique_curve[
                "actual_fe_fraction"
            ].to_numpy(dtype=float)

            y = unique_curve[
                "hypervolume"
            ].to_numpy(dtype=float)

            # Add the common artificial origin.
            if x[0] > METRIC_EPS:
                x = np.concatenate(
                    (
                        [0.0],
                        x,
                    )
                )

                y = np.concatenate(
                    (
                        [0.0],
                        y,
                    )
                )

            else:
                x[0] = 0.0
                y[0] = 0.0

            # A successful complete run should reach 100% FE.
            if x[-1] < 1.0 - 1e-8:
                auc = np.nan

            else:
                x[-1] = 1.0

                trapezoid_function = getattr(
                    np,
                    "trapezoid",
                    np.trapz,
                )

                auc = float(
                    trapezoid_function(
                        y,
                        x,
                    )
                )

        (
            k,
            rho,
            algorithm,
            seed,
        ) = group_key

        mask = (
            (
                result[
                    "max_active_actions"
                ]
                == int(k)
            )
            & np.isclose(
                result[
                    "coverage_threshold"
                ],
                float(rho),
            )
            & (
                result["algorithm"]
                == algorithm
            )
            & (
                result["seed"]
                == int(seed)
            )
        )

        result.loc[
            mask,
            "hv_fe_auc",
        ] = auc

    return result



def summarize_hv_fe_curve(curve: pd.DataFrame) -> pd.DataFrame:
    if curve.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_columns = [
        "max_active_actions",
        "coverage_threshold",
        "algorithm",
        "checkpoint_fraction",
    ]
    for keys, group in curve.groupby(group_columns, sort=True):
        k, rho, algorithm, fraction = keys
        hv = pd.to_numeric(group["hypervolume"], errors="coerce").dropna()
        igd = pd.to_numeric(group["igd_plus"], errors="coerce").dropna()
        rows.append(
            {
                "max_active_actions": int(k),
                "coverage_threshold": float(rho),
                "algorithm": algorithm,
                "checkpoint_fraction": float(fraction),
                "n_runs": int(group["seed"].nunique()),
                "function_evaluations_median": float(
                    pd.to_numeric(
                        group["function_evaluations"], errors="coerce"
                    ).median()
                ),
                "actual_fe_fraction_median": float(
                    pd.to_numeric(
                        group["actual_fe_fraction"], errors="coerce"
                    ).median()
                ),
                "hypervolume_mean": float(hv.mean()) if len(hv) else np.nan,
                "hypervolume_std": (
                    float(hv.std(ddof=1)) if len(hv) > 1 else 0.0
                ),
                "hypervolume_q25": float(hv.quantile(0.25)) if len(hv) else np.nan,
                "hypervolume_median": float(hv.median()) if len(hv) else np.nan,
                "hypervolume_q75": float(hv.quantile(0.75)) if len(hv) else np.nan,
                "igd_plus_median": float(igd.median()) if len(igd) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_hv_fe_curves(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty or not SAVE_HV_FE_PLOTS:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for (k, rho), scenario in summary.groupby(
        ["max_active_actions", "coverage_threshold"], sort=True
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for algorithm, group in scenario.groupby("algorithm", sort=False):
            ordered = group.sort_values("actual_fe_fraction_median")
            x = 100.0 * ordered["actual_fe_fraction_median"].to_numpy(dtype=float)
            median = ordered["hypervolume_median"].to_numpy(dtype=float)
            q25 = ordered["hypervolume_q25"].to_numpy(dtype=float)
            q75 = ordered["hypervolume_q75"].to_numpy(dtype=float)
            line = ax.plot(x, median, marker="o", label=algorithm)[0]
            ax.fill_between(
                x,
                q25,
                q75,
                alpha=0.18,
                color=line.get_color(),
            )
        ax.set_xlabel("Actual function-evaluation budget used (%)")
        ax.set_ylabel("Hypervolume")
        ax.set_title(f"HV-FE convergence: K={int(k)}, rho={float(rho):.2f}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{scenario_tag(int(k), float(rho))}_HV_FE.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def print_run_monitor(metrics: dict[str, Any]) -> None:
    """Immediate per-run status line; HV is added after joint normalization."""
    if metrics.get("status") != "success":
        print(
            f"  -> FAILED after {metrics.get('runtime_seconds', np.nan):.2f}s: "
            f"{metrics.get('error', '')}",
            flush=True,
        )
        return
    print(
        "  -> completed | "
        f"FR={metrics.get('feasible_rate', np.nan):.{LIVE_MONITOR_DECIMALS}f} | "
        f"nontrivial_FR={metrics.get('nontrivial_feasible_rate', np.nan):.{LIVE_MONITOR_DECIMALS}f} | "
        f"Pareto={int(metrics.get('pareto_size', 0))} | "
        f"FE={int(metrics.get('function_evaluations_completed', 0))} | "
        f"time={metrics.get('runtime_seconds', np.nan):.2f}s",
        flush=True,
    )


def refresh_live_scenario_monitor(
    metric_rows: Sequence[dict[str, Any]],
    fronts: dict[tuple[int, float, str, int], pd.DataFrame],
    *,
    k: int,
    rho: float,
) -> None:
    """Recompute provisional joint metrics from all completed runs in a scenario."""
    if not ENABLE_LIVE_MONITORING:
        return
    metrics = pd.DataFrame(metric_rows)
    if metrics.empty:
        return
    subset = metrics.loc[
        (metrics["max_active_actions"] == int(k))
        & np.isclose(metrics["coverage_threshold"], float(rho))
    ].copy()
    if subset.empty or not (subset["status"] == "success").any():
        return
    scenario_fronts = {
        key: frame
        for key, frame in fronts.items()
        if int(key[0]) == int(k) and np.isclose(float(key[1]), float(rho))
    }
    provisional, coverage = attach_quality_metrics(subset, scenario_fronts)
    summary = summarize_by_scenario(provisional)
    coverage_summary = summarize_pairwise_coverage(coverage)

    LIVE_MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    tag = scenario_tag(int(k), float(rho))
    provisional.to_csv(
        LIVE_MONITOR_DIR / f"{tag}_live_run_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        LIVE_MONITOR_DIR / f"{tag}_live_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    coverage_summary.to_csv(
        LIVE_MONITOR_DIR / f"{tag}_live_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )

    display_columns = [
        "algorithm",
        "n_successful_runs",
        "hypervolume_median",
        "igd_plus_median",
        "feasible_rate_mean",
        "nontrivial_feasible_rate_mean",
        "pareto_size_mean",
        "runtime_seconds_mean",
    ]
    available = [column for column in display_columns if column in summary.columns]
    print(
        f"\n[PROVISIONAL MONITOR] K={int(k)}, rho={float(rho):.2f} "
        f"(joint normalization over completed runs)",
        flush=True,
    )
    if available:
        print(summary[available].to_string(index=False), flush=True)
    if not coverage_summary.empty:
        coverage_columns = [
            "algorithm_A",
            "algorithm_B",
            "C_A_dominates_B_mean",
            "C_B_dominates_A_mean",
            "coverage_advantage_A_minus_B_mean",
        ]
        print("Pairwise C(A,B):", flush=True)
        print(
            coverage_summary[coverage_columns].to_string(index=False),
            flush=True,
        )
    print("", flush=True)


# =============================================================================
# 7. Aggregation and inference
# =============================================================================
def summarize_by_scenario(metrics: pd.DataFrame) -> pd.DataFrame:
    successful = metrics.loc[metrics["status"] == "success"].copy()
    group_columns = ["max_active_actions", "coverage_threshold", "algorithm"]
    excluded = {
        "seed",
        "max_active_actions",
        "coverage_threshold",
        "recommended_solution_id",
    }
    numeric_columns = [
        column
        for column in successful.columns
        if pd.api.types.is_numeric_dtype(successful[column])
        and column not in excluded
        and not column.startswith("algorithm_")
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in successful.groupby(group_columns, sort=True):
        k, rho, algorithm = keys
        row: dict[str, Any] = {
            "max_active_actions": int(k),
            "coverage_threshold": float(rho),
            "algorithm": algorithm,
            "n_successful_runs": int(len(group)),
            "n_unique_seeds": int(group["seed"].nunique()),
        }
        for column in numeric_columns:
            values = (
                pd.to_numeric(group[column], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if values.empty:
                continue
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            row[f"{column}_median"] = float(values.median())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_rank_biserial(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[
        np.isfinite(differences) & (np.abs(differences) > METRIC_EPS)
    ]
    if differences.size == 0:
        return 0.0
    ranks = rankdata(np.abs(differences), method="average")
    positive = float(ranks[differences > 0].sum())
    negative = float(ranks[differences < 0].sum())
    denominator = positive + negative
    return 0.0 if denominator <= METRIC_EPS else (positive - negative) / denominator


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(values, np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return adjusted
    order = finite[np.argsort(values[finite])]
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def pairwise_tests(metrics: pd.DataFrame) -> pd.DataFrame:
    successful = metrics.loc[metrics["status"] == "success"].copy()
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[Any, Any]] = list(
        successful[["max_active_actions", "coverage_threshold"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    scopes.append(("pooled_exploratory", "pooled_exploratory"))

    for k, rho in scopes:
        if k == "pooled_exploratory":
            subset = successful
            keys = ["max_active_actions", "coverage_threshold", "seed"]
        else:
            subset = successful.loc[
                (successful["max_active_actions"] == int(k))
                & np.isclose(successful["coverage_threshold"], float(rho))
            ]
            keys = ["seed"]

        for algorithm_a, algorithm_b in PAIRWISE_COMPARISONS:
            a = subset.loc[subset["algorithm"] == algorithm_a].set_index(keys)
            b = subset.loc[subset["algorithm"] == algorithm_b].set_index(keys)
            common = a.index.intersection(b.index)

            for metric, direction in PRIMARY_TEST_METRICS.items():
                if metric not in a.columns or metric not in b.columns:
                    continue
                a_values = pd.to_numeric(a.loc[common, metric], errors="coerce")
                b_values = pd.to_numeric(b.loc[common, metric], errors="coerce")
                valid = a_values.notna() & b_values.notna()
                av = a_values.loc[valid].to_numpy(dtype=float)
                bv = b_values.loc[valid].to_numpy(dtype=float)
                differences = av - bv

                if len(differences) == 0:
                    statistic = np.nan
                    p_value = np.nan
                elif np.all(np.abs(differences) <= METRIC_EPS):
                    statistic = 0.0
                    p_value = 1.0
                else:
                    test = wilcoxon(
                        av,
                        bv,
                        zero_method="wilcox",
                        correction=False,
                        alternative="two-sided",
                        mode="auto",
                    )
                    statistic = float(test.statistic)
                    p_value = float(test.pvalue)

                signed = int(direction) * differences
                rows.append(
                    {
                        "max_active_actions": k,
                        "coverage_threshold": rho,
                        "algorithm_A": algorithm_a,
                        "algorithm_B": algorithm_b,
                        "metric": metric,
                        "larger_is_better": bool(direction > 0),
                        "n_pairs": int(len(differences)),
                        "A_mean": float(np.mean(av)) if len(av) else np.nan,
                        "B_mean": float(np.mean(bv)) if len(bv) else np.nan,
                        "mean_A_minus_B": (
                            float(np.mean(differences))
                            if len(differences)
                            else np.nan
                        ),
                        "A_win_rate": (
                            float(np.mean(signed > 0.0)) if len(signed) else np.nan
                        ),
                        "tie_rate": (
                            float(np.mean(np.abs(signed) <= METRIC_EPS))
                            if len(signed)
                            else np.nan
                        ),
                        "rank_biserial_expected_direction": paired_rank_biserial(
                            signed
                        ),
                        "wilcoxon_statistic": statistic,
                        "p_value": p_value,
                    }
                )

    output = pd.DataFrame(rows)
    if not output.empty:
        output["p_value_holm"] = np.nan
        correction_groups = [
            "max_active_actions",
            "coverage_threshold",
            "metric",
        ]
        for _, indices in output.groupby(correction_groups).groups.items():
            idx = list(indices)
            output.loc[idx, "p_value_holm"] = holm_adjust(
                output.loc[idx, "p_value"].to_numpy(dtype=float)
            )
    return output


def friedman_and_ranks(
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform scenario-level and pooled Friedman tests.

    Completely tied metrics are handled explicitly because SciPy's
    tie-correction denominator is zero when all algorithms have the same
    value in every block.
    """
    successful = metrics.loc[
        metrics["status"] == "success"
    ].copy()

    test_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []

    scopes: list[tuple[Any, Any]] = list(
        successful[
            [
                "max_active_actions",
                "coverage_threshold",
            ]
        ]
        .drop_duplicates()
        .itertuples(
            index=False,
            name=None,
        )
    )

    scopes.append(
        (
            "pooled_exploratory",
            "pooled_exploratory",
        )
    )

    for k, rho in scopes:
        if k == "pooled_exploratory":
            subset = successful

            block_columns = [
                "max_active_actions",
                "coverage_threshold",
                "seed",
            ]

        else:
            subset = successful.loc[
                (
                    successful[
                        "max_active_actions"
                    ]
                    == int(k)
                )
                & np.isclose(
                    successful[
                        "coverage_threshold"
                    ],
                    float(rho),
                )
            ]

            block_columns = ["seed"]

        for metric, direction in (
            PRIMARY_TEST_METRICS.items()
        ):
            if metric not in subset.columns:
                continue

            pivot = subset.pivot_table(
                index=block_columns,
                columns="algorithm",
                values=metric,
                aggfunc="first",
            )

            if not set(
                ALGORITHM_NAMES
            ).issubset(pivot.columns):
                continue

            pivot = pivot[
                list(ALGORITHM_NAMES)
            ].dropna()

            n_blocks = int(len(pivot))
            values_matrix = pivot.to_numpy(
                dtype=float
            )

            # ---------------------------------------------------------
            # Calculate block-wise algorithm ranks
            # ---------------------------------------------------------
            if n_blocks:
                rank_matrix = np.vstack(
                    [
                        rankdata(
                            -int(direction) * row,
                            method="average",
                        )
                        for row in values_matrix
                    ]
                )
            else:
                rank_matrix = np.empty(
                    (
                        0,
                        len(ALGORITHM_NAMES),
                    ),
                    dtype=float,
                )

            # ---------------------------------------------------------
            # Detect blocks carrying algorithm-comparison information
            # ---------------------------------------------------------
            if n_blocks:
                block_ranges = (
                    np.max(
                        values_matrix,
                        axis=1,
                    )
                    - np.min(
                        values_matrix,
                        axis=1,
                    )
                )

                informative_blocks = (
                    block_ranges > METRIC_EPS
                )

                n_informative_blocks = int(
                    informative_blocks.sum()
                )

                all_blocks_tied = bool(
                    n_informative_blocks == 0
                )
            else:
                n_informative_blocks = 0
                all_blocks_tied = False

            # ---------------------------------------------------------
            # Friedman test
            # ---------------------------------------------------------
            if n_blocks < 3:
                statistic = np.nan
                p_value = np.nan
                test_status = (
                    "insufficient_blocks"
                )

            elif all_blocks_tied:
                # All algorithms are identical in every block.
                # scipy.stats.friedmanchisquare would divide by a zero
                # tie-correction factor.
                statistic = 0.0
                p_value = 1.0
                test_status = (
                    "all_algorithms_tied"
                )

            else:
                test = friedmanchisquare(
                    *[
                        values_matrix[:, index]
                        for index in range(
                            len(ALGORITHM_NAMES)
                        )
                    ]
                )

                statistic = float(
                    test.statistic
                )

                p_value = float(
                    test.pvalue
                )

                if (
                    np.isfinite(statistic)
                    and np.isfinite(p_value)
                ):
                    test_status = "ok"
                else:
                    statistic = np.nan
                    p_value = np.nan
                    test_status = (
                        "nonfinite_test_result"
                    )

            test_rows.append(
                {
                    "max_active_actions": k,
                    "coverage_threshold": rho,
                    "metric": metric,
                    "n_blocks": int(n_blocks),
                    "n_informative_blocks": int(
                        n_informative_blocks
                    ),
                    "test_status": test_status,
                    "friedman_statistic": (
                        statistic
                    ),
                    "p_value": p_value,
                }
            )

            for index, algorithm in enumerate(
                ALGORITHM_NAMES
            ):
                rank_rows.append(
                    {
                        "max_active_actions": k,
                        "coverage_threshold": rho,
                        "metric": metric,
                        "algorithm": algorithm,
                        "n_blocks": int(
                            n_blocks
                        ),
                        "n_informative_blocks": int(
                            n_informative_blocks
                        ),
                        "mean_rank": (
                            float(
                                rank_matrix[
                                    :,
                                    index,
                                ].mean()
                            )
                            if n_blocks
                            else np.nan
                        ),
                    }
                )

    tests = pd.DataFrame(
        test_rows
    )

    if not tests.empty:
        tests["p_value_holm"] = (
            holm_adjust(
                tests[
                    "p_value"
                ].to_numpy(dtype=float)
            )
        )

    return (
        tests,
        pd.DataFrame(rank_rows),
    )


def recommendation_stability(metrics: pd.DataFrame) -> pd.DataFrame:
    successful = metrics.loc[metrics["status"] == "success"].copy()
    rows: list[dict[str, Any]] = []
    for keys, group in successful.groupby(
        ["max_active_actions", "coverage_threshold", "algorithm"],
        sort=True,
    ):
        sets = [
            set(str(value).split(" | ")) if str(value).strip() else set()
            for value in group["recommended_active_es"]
        ]
        similarities: list[float] = []
        for a, b in itertools.combinations(sets, 2):
            union = a | b
            similarities.append(1.0 if not union else len(a & b) / len(union))
        k, rho, algorithm = keys
        rows.append(
            {
                "max_active_actions": int(k),
                "coverage_threshold": float(rho),
                "algorithm": algorithm,
                "n_runs": int(len(group)),
                "mean_pairwise_jaccard": (
                    float(np.mean(similarities)) if similarities else np.nan
                ),
                "std_pairwise_jaccard": (
                    float(np.std(similarities, ddof=1))
                    if len(similarities) > 1
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def recommended_es_frequency(metrics: pd.DataFrame) -> pd.DataFrame:
    successful = metrics.loc[metrics["status"] == "success"].copy()
    rows: list[dict[str, Any]] = []
    for keys, group in successful.groupby(
        ["max_active_actions", "coverage_threshold", "algorithm"],
        sort=True,
    ):
        counts: dict[str, int] = {}
        for value in group["recommended_active_es"]:
            elements = [item for item in str(value).split(" | ") if item]
            for element in set(elements):
                counts[element] = counts.get(element, 0) + 1
        k, rho, algorithm = keys
        for element, count in sorted(counts.items()):
            rows.append(
                {
                    "max_active_actions": int(k),
                    "coverage_threshold": float(rho),
                    "algorithm": algorithm,
                    "ES": element,
                    "selection_count": int(count),
                    "selection_frequency": float(count / len(group)),
                }
            )
    return pd.DataFrame(rows)


# =============================================================================
# 8. Output
# =============================================================================
def configuration_table(
    core: Any,
    seeds: Sequence[int],
    generations: int,
    n_options: int,
) -> pd.DataFrame:
    rows = {
        "run_mode": RUN_MODE,
        "core_file": str(CORE_FILE),
        "algorithms": " | ".join(ALGORITHM_NAMES),
        "seeds": str(tuple(int(seed) for seed in seeds)),
        "mo_shade_generations": int(generations),
        "max_active_action_scenarios": str(MAX_ACTIVE_ACTION_SCENARIOS),
        "coverage_threshold_scenarios": str(COVERAGE_THRESHOLD_SCENARIOS),
        "population_multiplier": int(EXPERIMENT_POPULATION_MULTIPLIER),
        "minimum_population_size": int(EXPERIMENT_MIN_POPULATION_SIZE),
        "maximum_population_size": int(EXPERIMENT_MAX_POPULATION_SIZE),
        "n_service_elements": int(n_options),
        "planned_FE_per_run": int(
            planned_mo_shade_evaluations(core, n_options, generations)
        ),
        "fairness_budget": "equal objective-function evaluations",
        "common_objectives": (
            "maximize reputation; maximize choice probability; minimize cost"
        ),
        "common_constraints": (
            "minimum action; action cap K; high-priority coverage rho; "
            "optional common budget"
        ),
        "representative_plan_diagnostics": (
            "retained only as optional descriptive output; excluded from "
            "algorithm-performance tests"
        ),
        "moead_population": int(
            math.comb(MOEAD_CONFIG.lattice_divisions + 2, 2)
        ),
        "moead_neighborhood": int(MOEAD_CONFIG.neighborhood_size),
        "moead_F": float(MOEAD_CONFIG.de_scale_factor),
        "moead_CR": float(MOEAD_CONFIG.de_crossover_rate),
        "moead_penalty": float(MOEAD_CONFIG.constraint_penalty_factor),
        "shamode_memory_size": int(shamode_adapter.SHAMODE_MEMORY_SIZE),
        "shamode_mutation_archive_rate": float(shamode_adapter.SHAMODE_MUTATION_ARCHIVE_RATE),
        "metric_front_cap": METRIC_FRONT_CAP,
        "HV_reference_point": str(HV_REFERENCE_POINT.tolist()),
        "HV_FE_checkpoint_fractions": str(HV_FE_CHECKPOINT_FRACTIONS),
        "HV_FE_max_front_size": int(MAX_HV_FE_FRONT_SIZE),
        "live_monitor_every_n_seeds": int(LIVE_MONITOR_EVERY_N_SEEDS),
        "FR_definition": (
            "final proportion of the current population satisfying all common constraints"
        ),
        "nontrivial_FR_definition": (
            "FR additionally excluding the zero-action baseline"
        ),
        "C_A_B_definition": (
            "proportion of B's final Pareto points weakly dominated by at least one point in A"
        ),
        "pairwise_Holm_family": (
            "all algorithm pairs within the same scenario and performance metric"
        ),
    }
    return pd.DataFrame(
        [{"parameter": key, "value": value} for key, value in rows.items()]
    )


def save_outputs(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    friedman: pd.DataFrame,
    ranks: pd.DataFrame,
    coverage: pd.DataFrame,
    coverage_summary: pd.DataFrame,
    convergence: pd.DataFrame,
    hv_fe_curve: pd.DataFrame,
    hv_fe_summary: pd.DataFrame,
    stability: pd.DataFrame,
    frequency: pd.DataFrame,
    configuration: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "run_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(
        OUTPUT_DIR / "summary_by_scenario.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pairwise.to_csv(
        OUTPUT_DIR / "pairwise_tests.csv", index=False, encoding="utf-8-sig"
    )
    friedman.to_csv(
        OUTPUT_DIR / "friedman_tests.csv", index=False, encoding="utf-8-sig"
    )
    ranks.to_csv(
        OUTPUT_DIR / "algorithm_ranks.csv", index=False, encoding="utf-8-sig"
    )
    coverage.to_csv(
        OUTPUT_DIR / "pairwise_coverage.csv", index=False, encoding="utf-8-sig"
    )
    coverage_summary.to_csv(
        OUTPUT_DIR / "pairwise_coverage_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    convergence.to_csv(
        OUTPUT_DIR / "convergence_history.csv", index=False, encoding="utf-8-sig"
    )
    hv_fe_curve.to_csv(
        OUTPUT_DIR / "hv_fe_curve.csv", index=False, encoding="utf-8-sig"
    )
    hv_fe_summary.to_csv(
        OUTPUT_DIR / "hv_fe_summary.csv", index=False, encoding="utf-8-sig"
    )
    stability.to_csv(
        OUTPUT_DIR / "recommendation_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    frequency.to_csv(
        OUTPUT_DIR / "recommended_es_frequency.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="Run_metrics", index=False)
        summary.to_excel(writer, sheet_name="Scenario_summary", index=False)
        pairwise.to_excel(writer, sheet_name="Pairwise_tests", index=False)
        friedman.to_excel(writer, sheet_name="Friedman_tests", index=False)
        ranks.to_excel(writer, sheet_name="Algorithm_ranks", index=False)
        coverage.to_excel(writer, sheet_name="Coverage_indicator", index=False)
        coverage_summary.to_excel(
            writer, sheet_name="Coverage_summary", index=False
        )
        convergence.to_excel(writer, sheet_name="Convergence", index=False)
        hv_fe_curve.to_excel(writer, sheet_name="HV_FE_curve", index=False)
        hv_fe_summary.to_excel(writer, sheet_name="HV_FE_summary", index=False)
        stability.to_excel(writer, sheet_name="Rec_stability", index=False)
        frequency.to_excel(writer, sheet_name="ES_frequency", index=False)
        configuration.to_excel(writer, sheet_name="Configuration", index=False)



# =============================================================================
# 8B. Per-run checkpointing and resume support
# =============================================================================
RunKey = tuple[int, float, str, int]


def canonical_run_key(
    k: int,
    rho: float,
    algorithm: str,
    seed: int,
) -> RunKey:
    """Return a stable in-memory key for one algorithm run."""
    return (
        int(k),
        round(float(rho), 10),
        str(algorithm),
        int(seed),
    )


def run_checkpoint_stem(key: RunKey) -> str:
    k, rho, algorithm, seed = key
    return f"{scenario_tag(k, rho)}__{algorithm}__seed_{seed}"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable."
    )


def _temporary_path(path: Path) -> Path:
    return path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=_json_default,
                allow_nan=True,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_csv(
    frame: pd.DataFrame,
    path: Path,
    *,
    encoding: str = "utf-8",
    compression: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        frame.to_csv(
            temporary,
            index=False,
            encoding=encoding,
            compression=compression,
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_pickle_gzip(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with gzip.open(temporary, "wb", compresslevel=5) as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_resume_configuration(
    *,
    seeds: Sequence[int],
    generations: int,
    n_options: int,
) -> dict[str, Any]:
    """Build an experiment signature that prevents incompatible cache reuse."""
    runner_path = Path(__file__).resolve()
    adapter_path = Path(moead_adapter.__file__).resolve()
    shamode_path = Path(shamode_adapter.__file__).resolve()
    return {
        "schema_version": RESUME_SCHEMA_VERSION,
        "run_mode": RUN_MODE,
        "seeds": [int(seed) for seed in seeds],
        "generations": int(generations),
        "n_options": int(n_options),
        "max_active_action_scenarios": [
            int(value) for value in MAX_ACTIVE_ACTION_SCENARIOS
        ],
        "coverage_threshold_scenarios": [
            float(value) for value in COVERAGE_THRESHOLD_SCENARIOS
        ],
        "algorithms": [asdict(spec) for spec in ALGORITHMS],
        "population_multiplier": int(EXPERIMENT_POPULATION_MULTIPLIER),
        "minimum_population_size": int(EXPERIMENT_MIN_POPULATION_SIZE),
        "maximum_population_size": int(EXPERIMENT_MAX_POPULATION_SIZE),
        "fixed_generation_budget": bool(FIXED_GENERATION_BUDGET),
        "moead_config": asdict(MOEAD_CONFIG),
        "metric_block_size": int(METRIC_BLOCK_SIZE),
        "live_monitor_every_n_seeds": int(LIVE_MONITOR_EVERY_N_SEEDS),
        "hv_fe_checkpoint_fractions": [
            float(value) for value in HV_FE_CHECKPOINT_FRACTIONS
        ],
        "core_file": str(CORE_FILE.resolve()),
        "core_sha256": sha256_file(CORE_FILE),
        "runner_file": str(runner_path),
        "runner_sha256": sha256_file(runner_path),
        "moead_adapter_file": str(adapter_path),
        "moead_adapter_sha256": sha256_file(adapter_path),
        "shamode_adapter_file": str(shamode_path),
        "shamode_adapter_sha256": sha256_file(shamode_path),
    }


def configuration_signature(configuration: dict[str, Any]) -> str:
    encoded = json.dumps(
        configuration,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ensure_resume_manifest(configuration: dict[str, Any]) -> None:
    if not ENABLE_RESUME:
        return
    RESUME_DIR.mkdir(parents=True, exist_ok=True)
    RESUME_RUN_DIR.mkdir(parents=True, exist_ok=True)
    expected_signature = configuration_signature(configuration)
    expected = {
        "schema_version": RESUME_SCHEMA_VERSION,
        "signature": expected_signature,
        "configuration": configuration,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not RESUME_MANIFEST.exists():
        atomic_write_json(RESUME_MANIFEST, expected)
        return

    with RESUME_MANIFEST.open("r", encoding="utf-8") as handle:
        existing = json.load(handle)
    existing_signature = str(existing.get("signature", ""))
    if existing_signature == expected_signature:
        return

    message = (
        "Existing resume cache was created with a different experiment "
        f"configuration or code version: {RESUME_MANIFEST}. "
        "Delete the _resume directory or choose a new OUTPUT_DIR before "
        "starting a new formal experiment."
    )
    if STRICT_RESUME_SIGNATURE:
        raise RuntimeError(message)
    print(f"WARNING: {message}", flush=True)


def checkpoint_paths(key: RunKey) -> dict[str, Path]:
    directory = RESUME_RUN_DIR / run_checkpoint_stem(key)
    return {
        "directory": directory,
        "metrics": directory / "metrics.json",
        "pareto": directory / "pareto.csv.gz",
        "history": directory / "history.csv.gz",
        "snapshots": directory / "snapshots.pkl.gz",
        "complete": directory / "COMPLETE.json",
    }


def save_run_checkpoint(
    key: RunKey,
    metrics: dict[str, Any],
    pareto: pd.DataFrame,
    history: pd.DataFrame,
    snapshot_records: Sequence[dict[str, Any]],
) -> None:
    """Atomically commit one completed or failed run to the resume cache."""
    if not ENABLE_RESUME:
        return
    paths = checkpoint_paths(key)
    paths["directory"].mkdir(parents=True, exist_ok=True)

    # The marker is written last. A process termination before that point leaves
    # an intentionally incomplete run that will be recomputed on restart.
    if paths["complete"].exists():
        paths["complete"].unlink()

    atomic_write_json(paths["metrics"], metrics)
    status = str(metrics.get("status", "failed"))
    required_files = [paths["metrics"].name]

    if status == "success":
        if pareto.empty:
            raise RuntimeError(f"Cannot checkpoint successful run {key}: empty front.")
        atomic_write_csv(
            pareto,
            paths["pareto"],
            encoding="utf-8",
            compression="gzip",
        )
        atomic_write_csv(
            history,
            paths["history"],
            encoding="utf-8",
            compression="gzip",
        )
        atomic_write_pickle_gzip(paths["snapshots"], list(snapshot_records))
        required_files.extend(
            [
                paths["pareto"].name,
                paths["history"].name,
                paths["snapshots"].name,
            ]
        )

    marker = {
        "schema_version": RESUME_SCHEMA_VERSION,
        "run_key": {
            "max_active_actions": int(key[0]),
            "coverage_threshold": float(key[1]),
            "algorithm": str(key[2]),
            "seed": int(key[3]),
        },
        "status": status,
        "required_files": required_files,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(paths["complete"], marker)


def load_run_checkpoint(
    key: RunKey,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, Any]],
] | None:
    """Load one committed run, or return None when it must be recomputed."""
    if not ENABLE_RESUME:
        return None
    paths = checkpoint_paths(key)
    if not paths["complete"].exists():
        return None

    try:
        with paths["complete"].open("r", encoding="utf-8") as handle:
            marker = json.load(handle)
        with paths["metrics"].open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        cached_key = canonical_run_key(
            int(metrics["max_active_actions"]),
            float(metrics["coverage_threshold"]),
            str(metrics["algorithm"]),
            int(metrics["seed"]),
        )
        if cached_key != key:
            raise ValueError(
                f"Checkpoint key mismatch: expected {key}, found {cached_key}."
            )

        status = str(metrics.get("status", marker.get("status", "failed")))
        if status != "success":
            if RETRY_FAILED_RUNS_ON_RESUME:
                return None
            return metrics, pd.DataFrame(), pd.DataFrame(), []

        for required in ("pareto", "history", "snapshots"):
            if not paths[required].exists():
                raise FileNotFoundError(paths[required])

        pareto = pd.read_csv(paths["pareto"], compression="gzip")
        history = pd.read_csv(paths["history"], compression="gzip")
        with gzip.open(paths["snapshots"], "rb") as handle:
            snapshots = pickle.load(handle)
        if pareto.empty:
            raise ValueError("Cached successful Pareto front is empty.")
        if not isinstance(snapshots, list):
            raise TypeError("Cached snapshots must be a list.")
        return metrics, pareto, history, snapshots
    except Exception as exc:
        print(
            f"  -> ignoring invalid checkpoint {run_checkpoint_stem(key)}: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return None


def write_resume_index(
    rows_by_key: dict[RunKey, dict[str, Any]],
) -> None:
    if not ENABLE_RESUME:
        return
    ordered_rows = [
        rows_by_key[key]
        for key in sorted(
            rows_by_key,
            key=lambda item: (item[0], item[1], item[3], item[2]),
        )
    ]
    frame = pd.DataFrame(ordered_rows)
    atomic_write_csv(
        frame,
        RESUME_INDEX,
        encoding="utf-8-sig",
        compression=None,
    )


def snapshot_records_to_dataframe(
    snapshot_records: Sequence[dict[str, Any]],
) -> pd.DataFrame:
    """Flatten raw checkpoint point arrays for an auditable compressed export."""
    rows: list[dict[str, Any]] = []
    for record in snapshot_records:
        points = np.asarray(record.get("points", np.empty((0, 3))), dtype=float)
        if points.size == 0:
            continue
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"Snapshot points must have shape (n,3), received {points.shape}."
            )
        metadata = {
            key: value
            for key, value in record.items()
            if key != "points"
        }
        for point_index, point in enumerate(points):
            rows.append(
                {
                    **metadata,
                    "snapshot_point_index": int(point_index),
                    "min_objective_reputation": float(point[0]),
                    "min_objective_choice": float(point[1]),
                    "effective_cost": float(point[2]),
                }
            )
    return pd.DataFrame(rows)


# =============================================================================
# 9. Main
# =============================================================================
def main() -> None:
    global METRIC_FRONT_CAP
    validate_extension_parameters()
    run_moead_internal_checks(MOEAD_CONFIG)
    shamode_adapter.run_internal_checks()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FRONT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    HV_FE_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    if ENABLE_RESUME:
        RESUME_RUN_DIR.mkdir(parents=True, exist_ok=True)

    core = load_core_module(CORE_FILE)
    core.run_internal_checks()
    original_state = capture_original_state(core)
    patch_state = capture_core_patch_state(core)

    try:
        options, context, _, _ = core.load_inputs()
        if hasattr(core, "CURRENT_CONTEXT"):
            core.CURRENT_CONTEXT = context

        core.POPULATION_MULTIPLIER = int(EXPERIMENT_POPULATION_MULTIPLIER)
        core.MIN_POPULATION_SIZE = int(EXPERIMENT_MIN_POPULATION_SIZE)
        core.MAX_POPULATION_SIZE = int(EXPERIMENT_MAX_POPULATION_SIZE)
        METRIC_FRONT_CAP = int(
            np.clip(
                EXPERIMENT_POPULATION_MULTIPLIER * len(options),
                EXPERIMENT_MIN_POPULATION_SIZE,
                EXPERIMENT_MAX_POPULATION_SIZE,
            )
        )

        if RUN_MODE == "pilot":
            seeds = tuple(int(seed) for seed in PILOT_SEEDS)
            generations = int(PILOT_GENERATIONS)
        elif RUN_MODE == "formal":
            seeds = tuple(int(seed) for seed in FORMAL_SEEDS)
            generations = int(FORMAL_GENERATIONS)
        else:
            raise ValueError("RUN_MODE must be 'pilot' or 'formal'.")

        resume_configuration = build_resume_configuration(
            seeds=seeds,
            generations=generations,
            n_options=len(options),
        )
        ensure_resume_manifest(resume_configuration)

        expected_runs = (
            len(MAX_ACTIVE_ACTION_SCENARIOS)
            * len(COVERAGE_THRESHOLD_SCENARIOS)
            * len(seeds)
            * len(ALGORITHMS)
        )
        print(
            f"Starting comparison: algorithms={len(ALGORITHMS)}, "
            f"scenarios={len(MAX_ACTIVE_ACTION_SCENARIOS) * len(COVERAGE_THRESHOLD_SCENARIOS)}, "
            f"seeds={len(seeds)}, elements={len(options)}, "
            f"expected_runs={expected_runs}."
        )
        print(
            "Each algorithm receives FE budget = "
            f"{planned_mo_shade_evaluations(core, len(options), generations)}."
        )
        if ENABLE_RESUME:
            print(f"Resume cache: {RESUME_DIR}", flush=True)

        metric_rows_by_key: dict[RunKey, dict[str, Any]] = {}
        fronts: dict[tuple[int, float, str, int], pd.DataFrame] = {}
        histories_by_key: dict[RunKey, pd.DataFrame] = {}
        snapshots_by_key: dict[RunKey, list[dict[str, Any]]] = {}
        n_resumed = 0
        n_executed = 0

        for k in MAX_ACTIVE_ACTION_SCENARIOS:
            for rho in COVERAGE_THRESHOLD_SCENARIOS:
                for seed_index, seed in enumerate(seeds, start=1):
                    order = list(ALGORITHMS)
                    if RANDOMIZE_EXECUTION_ORDER_WITHIN_SEED:
                        order_rng = np.random.default_rng(
                            np.random.SeedSequence(
                                [
                                    int(seed),
                                    int(k),
                                    int(round(rho * 1000)),
                                ]
                            )
                        )
                        order = [
                            order[index]
                            for index in order_rng.permutation(len(order))
                        ]

                    for algorithm in order:
                        key = canonical_run_key(
                            int(k),
                            float(rho),
                            algorithm.name,
                            int(seed),
                        )
                        cached = load_run_checkpoint(key)
                        if cached is not None:
                            run_metrics, pareto, history, snapshot_records = cached
                            metric_rows_by_key[key] = run_metrics
                            n_resumed += 1
                            print(
                                f"Resumed {algorithm.name}: K={k}, "
                                f"rho={rho:.2f}, seed={seed}, "
                                f"status={run_metrics.get('status', 'unknown')}.",
                                flush=True,
                            )
                            if run_metrics.get("status") == "success":
                                front_key = (
                                    int(k),
                                    float(rho),
                                    algorithm.name,
                                    int(seed),
                                )
                                fronts[front_key] = pareto
                                histories_by_key[key] = history
                                snapshots_by_key[key] = list(snapshot_records)
                                if SAVE_EACH_FRONT:
                                    filename = (
                                        f"{scenario_tag(k, rho)}_{algorithm.name}_"
                                        f"seed_{seed}.csv"
                                    )
                                    front_path = FRONT_DIR / filename
                                    if not front_path.exists():
                                        atomic_write_csv(
                                            pareto,
                                            front_path,
                                            encoding="utf-8-sig",
                                            compression=None,
                                        )
                            continue

                        print(
                            f"Running {algorithm.name}: K={k}, rho={rho:.2f}, "
                            f"seed={seed}."
                        )
                        run_metrics, pareto, history, snapshot_records = (
                            run_one_experiment(
                                core,
                                original_state,
                                patch_state,
                                algorithm,
                                seed=int(seed),
                                generations=generations,
                                max_active_actions=int(k),
                                coverage_threshold=float(rho),
                                options=options,
                                context=context,
                            )
                        )
                        n_executed += 1

                        # Commit the run before any optional/reporting output.
                        # If the process stops later, this run will still resume.
                        save_run_checkpoint(
                            key,
                            run_metrics,
                            pareto,
                            history,
                            snapshot_records,
                        )
                        metric_rows_by_key[key] = run_metrics
                        write_resume_index(metric_rows_by_key)
                        print_run_monitor(run_metrics)

                        if run_metrics["status"] == "success":
                            front_key = (
                                int(k),
                                float(rho),
                                algorithm.name,
                                int(seed),
                            )
                            fronts[front_key] = pareto
                            histories_by_key[key] = history
                            snapshots_by_key[key] = list(snapshot_records)
                            if SAVE_EACH_FRONT:
                                filename = (
                                    f"{scenario_tag(k, rho)}_{algorithm.name}_"
                                    f"seed_{seed}.csv"
                                )
                                atomic_write_csv(
                                    pareto,
                                    FRONT_DIR / filename,
                                    encoding="utf-8-sig",
                                    compression=None,
                                )

                    # All algorithms for the paired seed are now either loaded
                    # or newly completed, so provisional metrics are coherent.
                    if (
                        ENABLE_LIVE_MONITORING
                        and (
                            seed_index % max(int(LIVE_MONITOR_EVERY_N_SEEDS), 1) == 0
                            or seed_index == len(seeds)
                        )
                    ):
                        refresh_live_scenario_monitor(
                            list(metric_rows_by_key.values()),
                            fronts,
                            k=int(k),
                            rho=float(rho),
                        )
                    write_resume_index(metric_rows_by_key)

        if len(metric_rows_by_key) != expected_runs:
            raise RuntimeError(
                f"Only {len(metric_rows_by_key)} of {expected_runs} expected "
                "run records are available after the experiment loop."
            )

        ordered_keys = sorted(
            metric_rows_by_key,
            key=lambda item: (item[0], item[1], item[3], item[2]),
        )
        metric_rows = [metric_rows_by_key[key] for key in ordered_keys]
        histories = [
            histories_by_key[key]
            for key in ordered_keys
            if key in histories_by_key
        ]
        all_snapshot_records = [
            record
            for key in ordered_keys
            for record in snapshots_by_key.get(key, [])
        ]

        metrics = pd.DataFrame(metric_rows)
        metrics, coverage = attach_quality_metrics(metrics, fronts)
        coverage_summary = summarize_pairwise_coverage(coverage)

        snapshot_points = snapshot_records_to_dataframe(all_snapshot_records)
        atomic_write_csv(
            snapshot_points,
            OUTPUT_DIR / "hv_fe_snapshot_points.csv.gz",
            encoding="utf-8",
            compression="gzip",
        )

        hv_fe_curve = build_hv_fe_curve(
            metrics,
            fronts,
            all_snapshot_records,
        )
        metrics = attach_hv_fe_auc(metrics, hv_fe_curve)
        hv_fe_summary = summarize_hv_fe_curve(hv_fe_curve)
        plot_hv_fe_curves(hv_fe_summary, HV_FE_DIR)
        summary = summarize_by_scenario(metrics)
        pairwise = pairwise_tests(metrics)
        friedman, ranks = friedman_and_ranks(metrics)
        convergence = (
            pd.concat(histories, ignore_index=True, sort=False)
            if histories
            else pd.DataFrame()
        )
        stability = recommendation_stability(metrics)
        frequency = recommended_es_frequency(metrics)
        configuration = configuration_table(
            core,
            seeds,
            generations,
            len(options),
        )

        save_outputs(
            metrics,
            summary,
            pairwise,
            friedman,
            ranks,
            coverage,
            coverage_summary,
            convergence,
            hv_fe_curve,
            hv_fe_summary,
            stability,
            frequency,
            configuration,
        )

        n_success = int((metrics["status"] == "success").sum())
        n_failed = int((metrics["status"] != "success").sum())
        print("\nComparison completed.")
        print(
            f"Successful runs: {n_success}; failed runs: {n_failed}; "
            f"resumed: {n_resumed}; newly executed: {n_executed}."
        )
        print(f"Workbook: {OUTPUT_EXCEL}")
        print(f"HV-FE curves: {HV_FE_DIR}")
        print(f"Live monitor files: {LIVE_MONITOR_DIR}")
        print(f"Resume cache retained at: {RESUME_DIR}")

    except KeyboardInterrupt:
        print(
            "\nExecution interrupted. Completed runs have been committed to "
            f"{RESUME_DIR}; rerun the same script to continue.",
            flush=True,
        )
        raise
    finally:
        restore_core_patch_state(core, patch_state)
        restore_original_state(core, original_state)


if __name__ == "__main__":
    main()

