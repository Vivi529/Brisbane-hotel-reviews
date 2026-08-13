# -*- coding: utf-8 -*-
"""
Comparable benchmark: proposed MO-SHADE versus literature SHAMODE
=================================================================

This runner adapts the SHAMODE algorithm of Panagant, Bureerat and Tai
(2019) to the same hotel-service three-objective MOO problem implemented by
``MOO_MO_SHADE_IPEA_priority_initialization_fixed(3).py``.

Algorithms
----------
M1_IPEA_MO_SHADE
    The proposed method in the supplied core file: IPEA-priority-guided sparse
    initialization, current-to-pbest/1, weighted SHADE memory, crowding-based
    environmental selection, linear population-size reduction, and a feasible
    external Pareto archive.

M0_uniform_MO_SHADE
    Mechanism-control ablation: the same MO-SHADE optimizer, but with a
    priority-neutral uniform sparse initializer.  It separates the effect of
    IPEA initialization from the effect of the MO-SHADE search mechanism.

SHAMODE_2019
    Literature-faithful SHAMODE mechanism adapted only where required by the
    hotel problem representation:
      * constant population size;
      * current-to-pbest/1 with pbest sampled uniformly from the current
        constrained Pareto archive;
      * external mutation archive size 1.4 * NP;
      * memory size H=5, initial MF=MCR=0.5;
      * ordinary (unweighted) Lehmer means for successful F and CR;
      * non-dominated-level selection with random truncation;
      * Pareto archive cap NP with random truncation;
      * priority-neutral sparse random initialization.

Fairness rules
--------------
1. All algorithms use exactly the same objective functions, action domain,
   repairs, constrained dominance, K limit, and high-priority coverage rule
   from the supplied core module.
2. M0 and SHAMODE use the same priority-neutral sparse initialization domain;
   SHAMODE is not given IPEA guidance.
3. Paired random seeds are used within every (K, rho) scenario.
4. The initial population size is identical across algorithms.
5. The primary budget is an equal number of objective-function evaluations
   (FE).  It is the FE count planned for the proposed MO-SHADE run.  SHAMODE
   uses complete generations plus, only when necessary, one final truncated
   offspring batch so that the FE budget is exact.
6. Pareto-quality metrics are computed after joint scenario-wise objective
   normalization.  Each front is capped to the common initial population size
   for metric calculation, preventing a larger output archive from receiving
   an artificial resolution advantage.
7. The common relative-robust selector is applied after optimization only; its
   score is not treated as an optimizer-performance metric.

Primary outputs
---------------
comparison_MO_SHADE_SHAMODE/
    Comprehensive_results.xlsx
    run_metrics.csv
    summary_by_scenario.csv
    pairwise_tests.csv
    friedman_tests.csv
    algorithm_ranks.csv
    pairwise_coverage.csv
    hv_fe_curve.csv
    convergence_history.csv
    fronts/*.csv
    figures/HV_FE_*.png
    logs/*.log

Dependencies
------------
numpy, pandas, scipy, matplotlib, openpyxl
"""

from __future__ import annotations

import contextlib
import importlib.util
import itertools
import math
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


# =============================================================================
# 1. Configuration
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
CORE_CANDIDATES = (
    SCRIPT_DIR / "MOO_MO_SHADE_IPEA_priority_initialization_fixed.py",
)

RUN_MODE = "pilot"  # "pilot" or "formal"

# Fast code/data check.
PILOT_SEEDS = (42, 43, 44)
PILOT_GENERATIONS = 50

# Formal paper experiment.  Thirty runs match the SHAMODE paper's replication
# count; the scenario grid tests the managerial constraints of this study.
FORMAL_SEEDS = tuple(range(42, 72))
FORMAL_GENERATIONS = 300
FORMAL_MAX_ACTIVE_ACTIONS = (10, 15, 20)
FORMAL_COVERAGE_THRESHOLDS = (0.30, 0.40, 0.50)

# Common population settings.  The initial NP is shared by all algorithms.
EXPERIMENT_POPULATION_MULTIPLIER = 4
EXPERIMENT_MIN_POPULATION_SIZE = 100
EXPERIMENT_MAX_POPULATION_SIZE = 500

# Current proposed initialization parameter from the supplied core.
IPEA_EXPLORATION_RATE = 0.20

# SHAMODE parameters reported in Panagant et al. (2019).
SHAMODE_MEMORY_SIZE = 5
SHAMODE_MUTATION_ARCHIVE_RATE = 1.40
SHAMODE_F_SCALE = 0.10
SHAMODE_CR_STD = 0.10

# Number of FE checkpoints used for HV-FE curves, including initial and final.
TRACE_CHECKPOINTS = 25

# Metric settings.
HV_REFERENCE_POINT = np.asarray([1.05, 1.05, 1.05], dtype=float)
MAX_REFERENCE_FRONT_SIZE = 5000
METRIC_EPS = 1e-12
SAVE_EACH_FRONT = True
SAVE_PLOTS = True

OUTPUT_DIR = SCRIPT_DIR / "comparison_MO_SHADE_SHAMODE_tau"
FRONT_DIR = OUTPUT_DIR / "fronts"
LOG_DIR = OUTPUT_DIR / "logs"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_EXCEL = OUTPUT_DIR / "Comprehensive_results.xlsx"


# =============================================================================
# 2. Algorithm definitions and state control
# =============================================================================
@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    family: str
    initialization: str
    description: str


ALGORITHMS: tuple[AlgorithmSpec, ...] = (
    AlgorithmSpec(
        name="M1_IPEA_MO_SHADE",
        family="MO_SHADE",
        initialization="ipea_priority",
        description="Proposed full method",
    ),
    AlgorithmSpec(
        name="M0_uniform_MO_SHADE",
        family="MO_SHADE",
        initialization="uniform_sparse",
        description="Priority-neutral MO-SHADE ablation",
    ),
    AlgorithmSpec(
        name="SHAMODE_2019",
        family="SHAMODE",
        initialization="uniform_sparse",
        description="Panagant et al. (2019) mechanism adapted to hotel MOO",
    ),
)
ALGORITHM_NAMES = tuple(spec.name for spec in ALGORITHMS)
PAIRWISE_COMPARISONS = tuple(itertools.combinations(ALGORITHM_NAMES, 2))

# +1: larger is better; -1: smaller is better.
PRIMARY_TEST_METRICS: dict[str, int] = {
    "hypervolume": +1,
    "igd_plus": -1,
    "epsilon_plus": -1,
    "runtime_seconds": -1,
    "final_feasible_fraction": +1,
    "first_fe_nontrivial_feasible_95": -1,
    "recommended_reputation": +1,
    "recommended_choice_gain": +1,
    "recommended_effective_cost": -1,
    "recommended_n_active_actions": -1,
}


@dataclass
class CoreState:
    seed: int
    n_generations: int
    max_active_actions: int | None
    coverage_threshold: float
    priority_exploration_rate: float
    population_multiplier: int
    min_population_size: int
    max_population_size: int
    print_every: int
    min_generations_before_stop: int
    convergence_window: int


def locate_core_file() -> Path:
    for path in CORE_CANDIDATES:
        if path.exists():
            return path
    candidates = "\n".join(f"  - {path}" for path in CORE_CANDIDATES)
    raise FileNotFoundError(
        "The MO-SHADE core file was not found. Expected one of:\n" + candidates
    )


def load_core_module(path: Path) -> Any:
    module_name = "hotel_moo_shamode_comparison_core"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import core optimizer from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def capture_core_state(core: Any) -> CoreState:
    return CoreState(
        seed=int(core.SEED),
        n_generations=int(core.N_GENERATIONS),
        max_active_actions=core.MAX_ACTIVE_ACTIONS,
        coverage_threshold=float(core.HIGH_PRIORITY_COVERAGE_MIN),
        priority_exploration_rate=float(core.PRIORITY_EXPLORATION_RATE),
        population_multiplier=int(core.POPULATION_MULTIPLIER),
        min_population_size=int(core.MIN_POPULATION_SIZE),
        max_population_size=int(core.MAX_POPULATION_SIZE),
        print_every=int(core.PRINT_EVERY),
        min_generations_before_stop=int(core.MIN_GENERATIONS_BEFORE_STOP),
        convergence_window=int(core.CONVERGENCE_WINDOW),
    )


def restore_core_state(core: Any, state: CoreState) -> None:
    core.SEED = state.seed
    core.N_GENERATIONS = state.n_generations
    core.MAX_ACTIVE_ACTIONS = state.max_active_actions
    core.HIGH_PRIORITY_COVERAGE_MIN = state.coverage_threshold
    core.PRIORITY_EXPLORATION_RATE = state.priority_exploration_rate
    core.POPULATION_MULTIPLIER = state.population_multiplier
    core.MIN_POPULATION_SIZE = state.min_population_size
    core.MAX_POPULATION_SIZE = state.max_population_size
    core.PRINT_EVERY = state.print_every
    core.MIN_GENERATIONS_BEFORE_STOP = state.min_generations_before_stop
    core.CONVERGENCE_WINDOW = state.convergence_window


def apply_scenario(
    core: Any,
    original: CoreState,
    *,
    seed: int,
    generations: int,
    max_active_actions: int,
    coverage_threshold: float,
) -> None:
    restore_core_state(core, original)
    core.SEED = int(seed)
    core.N_GENERATIONS = int(generations)
    core.MAX_ACTIVE_ACTIONS = int(max_active_actions)
    core.HIGH_PRIORITY_COVERAGE_MIN = float(coverage_threshold)
    core.PRIORITY_EXPLORATION_RATE = float(IPEA_EXPLORATION_RATE)
    core.POPULATION_MULTIPLIER = int(EXPERIMENT_POPULATION_MULTIPLIER)
    core.MIN_POPULATION_SIZE = int(EXPERIMENT_MIN_POPULATION_SIZE)
    core.MAX_POPULATION_SIZE = int(EXPERIMENT_MAX_POPULATION_SIZE)
    core.PRINT_EVERY = int(generations) + 1
    # Fixed-budget benchmarking: disable early stopping.
    core.MIN_GENERATIONS_BEFORE_STOP = int(generations) + 1
    core.CONVERGENCE_WINDOW = int(generations) + 1


# =============================================================================
# 3. Common initialization and evaluation-budget helpers
# =============================================================================
def initial_population_size(core: Any, n_options: int) -> int:
    return int(
        np.clip(
            int(core.POPULATION_MULTIPLIER) * int(n_options),
            int(core.MIN_POPULATION_SIZE),
            int(core.MAX_POPULATION_SIZE),
        )
    )


def mo_shade_population_schedule(
    core: Any,
    n_options: int,
    generations: int,
) -> list[int]:
    """Population size before reproduction at generations 1..G."""
    initial = initial_population_size(core, n_options)
    schedule = [initial]
    for generation in range(1, int(generations)):
        schedule.append(int(core.target_population_size(initial, generation)))
    return schedule


def planned_mo_shade_evaluations(
    core: Any,
    n_options: int,
    generations: int,
) -> int:
    initial = initial_population_size(core, n_options)
    schedule = mo_shade_population_schedule(core, n_options, generations)
    return int(initial + sum(schedule))


def create_sparse_initial_vector(
    core: Any,
    options: Sequence[Any],
    rng: np.random.Generator,
    mode: str,
) -> np.ndarray:
    """Shared semi-continuous sparse initializer.

    ``mode='ipea_priority'`` reproduces the supplied core's probability rule.
    ``mode='uniform_sparse'`` retains exactly the same sparsity and magnitude
    domain but samples eligible service elements uniformly.
    """
    n = len(options)
    if rng.random() < float(core.INITIAL_ZERO_SOLUTION_RATE):
        return np.zeros(n, dtype=float)

    low_fraction, high_fraction = core.INITIAL_ACTIVE_FRACTION_RANGE
    min_active = max(1, int(math.ceil(float(low_fraction) * n)))
    max_active = max(min_active, int(math.ceil(float(high_fraction) * n)))
    max_active = min(max_active, n)

    if core.MAX_ACTIVE_ACTIONS is not None:
        cap = int(core.MAX_ACTIVE_ACTIONS)
        if cap < 1:
            raise ValueError("MAX_ACTIVE_ACTIONS must be positive or None.")
        max_active = min(max_active, cap, n)
        min_active = min(min_active, max_active)

    eligible = np.asarray(
        [
            option.max_delta >= float(core.MIN_ACTION_MAGNITUDE) - float(core.EPS)
            for option in options
        ],
        dtype=bool,
    )
    if not eligible.any():
        raise ValueError("No service element can satisfy the minimum action size.")

    if mode == "ipea_priority":
        probabilities = np.asarray(
            core.priority_activation_probabilities(options), dtype=float
        )
    elif mode == "uniform_sparse":
        probabilities = eligible.astype(float)
        probabilities /= probabilities.sum()
    else:
        raise ValueError(f"Unsupported initialization mode: {mode!r}")

    n_active = int(rng.integers(min_active, max_active + 1))
    selected = rng.choice(n, size=n_active, replace=False, p=probabilities)
    x = np.zeros(n, dtype=float)
    for raw_index in selected:
        index = int(raw_index)
        option = options[index]
        x[index] = float(
            rng.uniform(float(core.MIN_ACTION_MAGNITUDE), option.max_delta)
        )
    return core.repair_vector(x, options)


# =============================================================================
# 4. FE trace collection
# =============================================================================
@dataclass
class FrontSnapshot:
    target_fe: int
    actual_fe: int
    generation: int
    objectives: np.ndarray


class TraceCollector:
    """Store the most recent archive not exceeding each FE checkpoint."""

    def __init__(self, initial_fe: int, final_fe: int, n_points: int) -> None:
        if final_fe < initial_fe:
            raise ValueError("final_fe must be at least initial_fe.")
        raw = np.linspace(initial_fe, final_fe, max(int(n_points), 2))
        self.thresholds = sorted(set(int(round(value)) for value in raw))
        self.position = 0
        self.previous_fe: int | None = None
        self.previous_generation: int | None = None
        self.previous_objectives: np.ndarray | None = None
        self.snapshots: list[FrontSnapshot] = []

    @staticmethod
    def _archive_objectives(archive: Sequence[Any]) -> np.ndarray:
        if not archive:
            return np.empty((0, 3), dtype=float)
        return np.asarray(
            [individual.objectives for individual in archive], dtype=float
        )

    def update(
        self,
        function_evaluations: int,
        generation: int,
        archive: Sequence[Any],
    ) -> None:
        current_fe = int(function_evaluations)
        current_objectives = self._archive_objectives(archive)

        if self.previous_fe is None:
            self.previous_fe = current_fe
            self.previous_generation = int(generation)
            self.previous_objectives = current_objectives.copy()

        while (
            self.position < len(self.thresholds)
            and self.thresholds[self.position] <= current_fe
        ):
            threshold = self.thresholds[self.position]
            if threshold == current_fe:
                selected_fe = current_fe
                selected_generation = int(generation)
                selected_objectives = current_objectives
            else:
                # Conservative left-continuous trace: never use a front produced
                # after the requested FE checkpoint.
                selected_fe = int(self.previous_fe)
                selected_generation = int(self.previous_generation)
                selected_objectives = np.asarray(
                    self.previous_objectives, dtype=float
                )
            self.snapshots.append(
                FrontSnapshot(
                    target_fe=int(threshold),
                    actual_fe=selected_fe,
                    generation=selected_generation,
                    objectives=selected_objectives.copy(),
                )
            )
            self.position += 1

        self.previous_fe = current_fe
        self.previous_generation = int(generation)
        self.previous_objectives = current_objectives.copy()

    def finalize(
        self,
        function_evaluations: int,
        generation: int,
        archive: Sequence[Any],
    ) -> list[FrontSnapshot]:
        self.update(function_evaluations, generation, archive)
        current_objectives = self._archive_objectives(archive)
        while self.position < len(self.thresholds):
            self.snapshots.append(
                FrontSnapshot(
                    target_fe=int(self.thresholds[self.position]),
                    actual_fe=int(function_evaluations),
                    generation=int(generation),
                    objectives=current_objectives.copy(),
                )
            )
            self.position += 1
        return list(self.snapshots)


# =============================================================================
# 5. Proposed MO-SHADE runner with explicit FE trace
# =============================================================================
def population_diagnostics(core: Any, population: Sequence[Any]) -> dict[str, float]:
    if not population:
        return {
            "feasible_fraction": 0.0,
            "nontrivial_feasible_fraction": 0.0,
            "mean_high_priority_coverage": np.nan,
        }
    feasible = np.asarray(
        [individual.violation <= float(core.EPS) for individual in population],
        dtype=bool,
    )
    nontrivial = np.asarray(
        [
            individual.components["sum_delta"]
            > float(core.NONTRIVIAL_DELTA_TOLERANCE)
            for individual in population
        ],
        dtype=bool,
    )
    coverage = np.asarray(
        [
            individual.components["high_priority_coverage"]
            for individual in population
        ],
        dtype=float,
    )
    return {
        "feasible_fraction": float(feasible.mean()),
        "nontrivial_feasible_fraction": float((feasible & nontrivial).mean()),
        "mean_high_priority_coverage": float(coverage.mean()),
    }


def archive_diagnostics(archive: Sequence[Any]) -> dict[str, float]:
    if not archive:
        return {
            "pareto_archive_size": 0,
            "best_reputation_improvement": np.nan,
            "best_choice_probability_gain": np.nan,
            "lowest_effective_cost": np.nan,
        }
    return {
        "pareto_archive_size": int(len(archive)),
        "best_reputation_improvement": float(
            max(ind.components["reputation_improvement"] for ind in archive)
        ),
        "best_choice_probability_gain": float(
            max(ind.components["probability_gain"] for ind in archive)
        ),
        "lowest_effective_cost": float(
            min(ind.components["effective_cost"] for ind in archive)
        ),
    }


def run_comparable_mo_shade(
    core: Any,
    options: Sequence[Any],
    context: Any,
    *,
    seed: int,
    generations: int,
    initialization: str,
    evaluation_budget: int,
) -> tuple[list[Any], pd.DataFrame, dict[str, Any], list[FrontSnapshot]]:
    rng = np.random.default_rng(int(seed))
    initial_size = initial_population_size(core, len(options))

    population = [
        core.make_individual(
            create_sparse_initial_vector(core, options, rng, initialization),
            options,
            context,
            origin="initial",
        )
        for _ in range(initial_size)
    ]
    function_evaluations = initial_size

    memory_f = np.full(int(core.SHADE_MEMORY_SIZE), 0.5, dtype=float)
    memory_cr = np.full(int(core.SHADE_MEMORY_SIZE), 0.5, dtype=float)
    memory_position = 0
    mutation_archive: list[np.ndarray] = []
    pareto_archive = core.update_pareto_archive([], population)

    trace = TraceCollector(
        initial_fe=initial_size,
        final_fe=int(evaluation_budget),
        n_points=TRACE_CHECKPOINTS,
    )
    trace.update(function_evaluations, 0, pareto_archive)

    history_rows: list[dict[str, Any]] = []
    initial_row = {
        "generation": 0,
        "function_evaluations": int(function_evaluations),
        "population_size": int(len(population)),
        "successful_trials": 0,
        "memory_f_mean": float(memory_f.mean()),
        "memory_cr_mean": float(memory_cr.mean()),
        **population_diagnostics(core, population),
        **archive_diagnostics(pareto_archive),
    }
    history_rows.append(initial_row)

    for generation in range(1, int(generations) + 1):
        trials = core.generate_trials(
            population,
            mutation_archive,
            memory_f,
            memory_cr,
            options,
            context,
            rng,
        )
        function_evaluations += len(trials)

        combined = [individual.copy() for individual in population] + [
            trial.copy() for trial in trials
        ]
        next_size = int(core.target_population_size(initial_size, generation))
        next_population = core.environmental_selection(combined, next_size)

        selected_keys = {
            tuple(np.round(individual.x, core.ROUND_DECIMALS_FOR_UNIQUENESS))
            for individual in next_population
        }
        successful_trials: list[tuple[Any, float]] = []
        for trial in trials:
            trial_key = tuple(
                np.round(trial.x, core.ROUND_DECIMALS_FOR_UNIQUENESS)
            )
            if trial_key not in selected_keys or trial.parent_index is None:
                continue
            parent = population[int(trial.parent_index)]
            parent_key = tuple(
                np.round(parent.x, core.ROUND_DECIMALS_FOR_UNIQUENESS)
            )
            parent_survives = parent_key in selected_keys
            if (not parent_survives) or core.constrained_dominates(trial, parent):
                gain = core.normalized_improvement_gain(parent, trial)
                successful_trials.append((trial, max(float(gain), float(core.EPS))))
                mutation_archive.append(parent.x.copy())

        max_mutation_archive = max(
            1, int(round(float(core.ARCHIVE_RATE) * next_size))
        )
        if len(mutation_archive) > max_mutation_archive:
            keep = rng.choice(
                len(mutation_archive),
                size=max_mutation_archive,
                replace=False,
            )
            mutation_archive = [mutation_archive[int(index)] for index in keep]

        memory_position = core.update_shade_memory(
            successful_trials,
            memory_f,
            memory_cr,
            memory_position,
        )
        population = next_population
        pareto_archive = core.update_pareto_archive(pareto_archive, population)
        trace.update(function_evaluations, generation, pareto_archive)

        history_rows.append(
            {
                "generation": int(generation),
                "function_evaluations": int(function_evaluations),
                "population_size": int(len(population)),
                "successful_trials": int(len(successful_trials)),
                "memory_f_mean": float(memory_f.mean()),
                "memory_cr_mean": float(memory_cr.mean()),
                **population_diagnostics(core, population),
                **archive_diagnostics(pareto_archive),
            }
        )

    if function_evaluations != int(evaluation_budget):
        raise RuntimeError(
            "MO-SHADE FE accounting mismatch: "
            f"completed={function_evaluations}, planned={evaluation_budget}."
        )

    snapshots = trace.finalize(
        function_evaluations, int(generations), pareto_archive
    )
    metadata = {
        "initial_population_size": int(initial_size),
        "final_population_size": int(len(population)),
        "generations_completed": int(generations),
        "function_evaluations_completed": int(function_evaluations),
        "pareto_archive_size": int(len(pareto_archive)),
        "memory_f": memory_f.copy(),
        "memory_cr": memory_cr.copy(),
        "initialization": initialization,
    }
    return pareto_archive, pd.DataFrame(history_rows), metadata, snapshots


# =============================================================================
# 6. Literature SHAMODE implementation
# =============================================================================
def ordinary_lehmer_mean(values: Iterable[float], eps: float) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("Lehmer mean requires at least one value.")
    denominator = float(array.sum())
    if denominator <= eps:
        return 0.0
    return float(np.sum(array**2) / denominator)


def random_cap(
    values: Sequence[Any],
    cap: int,
    rng: np.random.Generator,
) -> list[Any]:
    if len(values) <= int(cap):
        return [value.copy() if hasattr(value, "copy") else value for value in values]
    keep = rng.choice(len(values), size=int(cap), replace=False)
    return [
        values[int(index)].copy()
        if hasattr(values[int(index)], "copy")
        else values[int(index)]
        for index in keep
    ]


def constrained_pareto_archive(
    core: Any,
    candidates: Sequence[Any],
    cap: int,
    rng: np.random.Generator,
) -> list[Any]:
    unique = core.unique_individuals(candidates)
    if not unique:
        return []
    first_front = core.nondominated_sort(unique)[0]
    archive = [unique[int(index)].copy() for index in first_front]
    return random_cap(archive, cap, rng)


def feasible_pareto_archive_random_cap(
    core: Any,
    archive: Sequence[Any],
    candidates: Sequence[Any],
    cap: int,
    rng: np.random.Generator,
) -> list[Any]:
    feasible = [
        individual.copy()
        for individual in list(archive) + list(candidates)
        if individual.violation <= float(core.EPS)
    ]
    feasible = core.unique_individuals(feasible)
    if not feasible:
        return []
    first_front = core.nondominated_sort(feasible)[0]
    front = [feasible[int(index)].copy() for index in first_front]
    return random_cap(front, cap, rng)


def shamode_environmental_selection(
    core: Any,
    candidates: Sequence[Any],
    population_size: int,
    rng: np.random.Generator,
) -> tuple[list[Any], list[int]]:
    """Highest non-dominated levels; random truncation of the boundary front."""
    selected_indices: list[int] = []
    for front in core.nondominated_sort(candidates):
        remaining = int(population_size) - len(selected_indices)
        if remaining <= 0:
            break
        if len(front) <= remaining:
            selected_indices.extend(int(index) for index in front)
        else:
            chosen = rng.choice(front, size=remaining, replace=False)
            selected_indices.extend(int(index) for index in chosen)
            break
    if len(selected_indices) != int(population_size):
        raise RuntimeError("SHAMODE environmental selection size mismatch.")
    return [candidates[index].copy() for index in selected_indices], selected_indices


def generate_shamode_trials(
    core: Any,
    population: Sequence[Any],
    pareto_guidance_archive: Sequence[Any],
    mutation_archive: Sequence[np.ndarray],
    memory_f: np.ndarray,
    memory_cr: np.ndarray,
    options: Sequence[Any],
    context: Any,
    rng: np.random.Generator,
    target_indices: Sequence[int],
) -> list[Any]:
    population_size = len(population)
    if population_size < 4:
        raise ValueError("SHAMODE requires at least four population members.")
    guidance = list(pareto_guidance_archive)
    if not guidance:
        guidance = constrained_pareto_archive(
            core, population, population_size, rng
        )
    if not guidance:
        raise RuntimeError("SHAMODE has no pbest guidance solution.")

    union_vectors = [individual.x for individual in population] + [
        np.asarray(vector, dtype=float) for vector in mutation_archive
    ]
    trials: list[Any] = []

    for raw_target_index in target_indices:
        target_index = int(raw_target_index)
        target = population[target_index]
        memory_index = int(rng.integers(0, len(memory_f)))
        F = core.sample_positive_cauchy(
            float(memory_f[memory_index]), SHAMODE_F_SCALE, rng
        )
        CR = float(
            np.clip(
                rng.normal(float(memory_cr[memory_index]), SHAMODE_CR_STD),
                0.0,
                1.0,
            )
        )

        pbest = guidance[int(rng.integers(0, len(guidance)))].x
        r1_candidates = [
            index for index in range(population_size) if index != target_index
        ]
        r1_index = int(rng.choice(r1_candidates))
        r1 = population[r1_index].x

        valid_r2 = [
            index
            for index in range(len(union_vectors))
            if not (
                index < population_size
                and index in {target_index, r1_index}
            )
        ]
        if not valid_r2:
            raise RuntimeError("No valid SHAMODE r2 vector exists.")
        r2 = union_vectors[int(rng.choice(valid_r2))]

        mutant = target.x + F * (pbest - target.x) + F * (r1 - r2)
        mutant = core.repair_vector(mutant, options)

        trial_values = target.x.copy()
        j_rand = int(rng.integers(0, len(options)))
        mask = rng.random(len(options)) < CR
        mask[j_rand] = True
        trial_values[mask] = mutant[mask]
        trial_values = core.repair_vector(trial_values, options)

        trials.append(
            core.make_individual(
                trial_values,
                options,
                context,
                origin="shamode_trial",
                F=F,
                CR=CR,
                memory_index=memory_index,
                parent_index=target_index,
            )
        )
    return trials


def run_shamode_2019(
    core: Any,
    options: Sequence[Any],
    context: Any,
    *,
    seed: int,
    evaluation_budget: int,
) -> tuple[list[Any], pd.DataFrame, dict[str, Any], list[FrontSnapshot]]:
    rng = np.random.default_rng(int(seed))
    population_size = initial_population_size(core, len(options))
    if int(evaluation_budget) < population_size:
        raise ValueError("SHAMODE evaluation budget is below its initial NP.")

    population = [
        core.make_individual(
            create_sparse_initial_vector(
                core, options, rng, mode="uniform_sparse"
            ),
            options,
            context,
            origin="shamode_initial",
        )
        for _ in range(population_size)
    ]
    function_evaluations = population_size

    memory_f = np.full(SHAMODE_MEMORY_SIZE, 0.5, dtype=float)
    memory_cr = np.full(SHAMODE_MEMORY_SIZE, 0.5, dtype=float)
    memory_position = 0
    mutation_archive: list[np.ndarray] = []

    # Internal pbest archive may temporarily contain infeasible solutions when
    # no feasible solution exists; constrained dominance immediately gives
    # feasible solutions precedence once found.
    guidance_archive = constrained_pareto_archive(
        core, population, population_size, rng
    )
    feasible_archive = feasible_pareto_archive_random_cap(
        core, [], population, population_size, rng
    )

    trace = TraceCollector(
        initial_fe=population_size,
        final_fe=int(evaluation_budget),
        n_points=TRACE_CHECKPOINTS,
    )
    trace.update(function_evaluations, 0, feasible_archive)

    history_rows: list[dict[str, Any]] = [
        {
            "generation": 0,
            "function_evaluations": int(function_evaluations),
            "population_size": int(population_size),
            "offspring_evaluated": 0,
            "successful_trials": 0,
            "guidance_archive_size": int(len(guidance_archive)),
            "mutation_archive_size": 0,
            "memory_f_mean": float(memory_f.mean()),
            "memory_cr_mean": float(memory_cr.mean()),
            **population_diagnostics(core, population),
            **archive_diagnostics(feasible_archive),
        }
    ]

    generation = 0
    while function_evaluations < int(evaluation_budget):
        generation += 1
        remaining = int(evaluation_budget) - function_evaluations
        offspring_count = min(population_size, remaining)

        if offspring_count == population_size:
            target_indices = np.arange(population_size, dtype=int)
        else:
            # Exact FE accounting only for the final incomplete batch.
            target_indices = rng.choice(
                population_size, size=offspring_count, replace=False
            )

        trials = generate_shamode_trials(
            core,
            population,
            guidance_archive,
            mutation_archive,
            memory_f,
            memory_cr,
            options,
            context,
            rng,
            target_indices,
        )
        function_evaluations += len(trials)

        combined = [individual.copy() for individual in population] + [
            trial.copy() for trial in trials
        ]
        next_population, selected_indices = shamode_environmental_selection(
            core, combined, population_size, rng
        )

        selected_trial_positions = [
            index - population_size
            for index in selected_indices
            if index >= population_size
        ]
        successful_trials = [trials[position] for position in selected_trial_positions]

        # Paper rule: parents of offspring that survive environmental selection
        # enter the external reproduction archive.
        for trial in successful_trials:
            if trial.parent_index is None:
                raise RuntimeError("A SHAMODE successful trial has no parent index.")
            mutation_archive.append(
                population[int(trial.parent_index)].x.copy()
            )

        mutation_cap = max(
            1, int(round(SHAMODE_MUTATION_ARCHIVE_RATE * population_size))
        )
        if len(mutation_archive) > mutation_cap:
            keep = rng.choice(
                len(mutation_archive), size=mutation_cap, replace=False
            )
            mutation_archive = [mutation_archive[int(index)] for index in keep]

        if successful_trials:
            memory_f[memory_position] = ordinary_lehmer_mean(
                [float(trial.F) for trial in successful_trials],
                float(core.EPS),
            )
            memory_cr[memory_position] = ordinary_lehmer_mean(
                [float(trial.CR) for trial in successful_trials],
                float(core.EPS),
            )
            memory_position = (memory_position + 1) % SHAMODE_MEMORY_SIZE

        population = next_population
        guidance_archive = constrained_pareto_archive(
            core,
            list(guidance_archive) + list(trials),
            population_size,
            rng,
        )
        feasible_archive = feasible_pareto_archive_random_cap(
            core,
            feasible_archive,
            trials,
            population_size,
            rng,
        )
        trace.update(function_evaluations, generation, feasible_archive)

        history_rows.append(
            {
                "generation": int(generation),
                "function_evaluations": int(function_evaluations),
                "population_size": int(population_size),
                "offspring_evaluated": int(len(trials)),
                "successful_trials": int(len(successful_trials)),
                "guidance_archive_size": int(len(guidance_archive)),
                "mutation_archive_size": int(len(mutation_archive)),
                "memory_f_mean": float(memory_f.mean()),
                "memory_cr_mean": float(memory_cr.mean()),
                **population_diagnostics(core, population),
                **archive_diagnostics(feasible_archive),
            }
        )

    if function_evaluations != int(evaluation_budget):
        raise RuntimeError("SHAMODE did not consume the exact FE budget.")

    snapshots = trace.finalize(
        function_evaluations, generation, feasible_archive
    )
    metadata = {
        "initial_population_size": int(population_size),
        "final_population_size": int(population_size),
        "generations_completed": int(generation),
        "function_evaluations_completed": int(function_evaluations),
        "pareto_archive_size": int(len(feasible_archive)),
        "guidance_archive_size": int(len(guidance_archive)),
        "mutation_archive_size": int(len(mutation_archive)),
        "memory_f": memory_f.copy(),
        "memory_cr": memory_cr.copy(),
        "memory_size": int(SHAMODE_MEMORY_SIZE),
        "mutation_archive_rate": float(SHAMODE_MUTATION_ARCHIVE_RATE),
        "final_partial_batch": bool(
            history_rows[-1].get("offspring_evaluated", population_size)
            < population_size
        ),
        "initialization": "uniform_sparse",
    }
    return feasible_archive, pd.DataFrame(history_rows), metadata, snapshots


# =============================================================================
# 7. Pareto quality metrics
# =============================================================================
def nondominated_mask_minimization(points: np.ndarray) -> np.ndarray:
    """Return a non-dominated mask for minimization.

    The three-objective path uses a Fenwick-tree sweep and is O(n log n),
    which is important when the pooled scenario reference front contains many
    runs.  A generic O(n^2) fallback is retained for other dimensions.
    """
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("points must be a non-empty 2D array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("points contain non-finite values.")

    n, dimension = values.shape
    if dimension != 3:
        keep = np.ones(n, dtype=bool)
        for index in range(n):
            dominated = np.any(
                np.all(values <= values[index] + METRIC_EPS, axis=1)
                & np.any(values < values[index] - METRIC_EPS, axis=1)
            )
            if dominated:
                keep[index] = False
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
    if len(values) <= int(maximum_size):
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
    selected = np.argsort(-distances, kind="mergesort")[: int(maximum_size)]
    return values[selected]


def hypervolume_2d_minimization(
    points: np.ndarray,
    reference_y: float,
    reference_z: float,
) -> float:
    values = np.asarray(points, dtype=float)
    if values.size == 0:
        return 0.0
    values = values[
        (values[:, 0] < reference_y) & (values[:, 1] < reference_z)
    ]
    if values.size == 0:
        return 0.0
    y_values = np.unique(values[:, 0])
    area = 0.0
    best_z = reference_z
    for position, y_value in enumerate(y_values):
        best_z = min(best_z, float(values[values[:, 0] <= y_value, 1].min()))
        next_y = (
            float(y_values[position + 1])
            if position + 1 < len(y_values)
            else reference_y
        )
        area += max(next_y - float(y_value), 0.0) * max(
            reference_z - best_z, 0.0
        )
    return float(area)


def hypervolume_3d_minimization(
    points: np.ndarray,
    reference: np.ndarray = HV_REFERENCE_POINT,
) -> float:
    values = np.asarray(points, dtype=float)
    if values.size == 0:
        return 0.0
    reference = np.asarray(reference, dtype=float)
    values = values[np.all(values < reference, axis=1)]
    if values.size == 0:
        return 0.0
    values = values[nondominated_mask_minimization(values)]
    x_values = np.unique(values[:, 0])
    volume = 0.0
    for position, x_value in enumerate(x_values):
        active = values[values[:, 0] <= x_value, 1:]
        next_x = (
            float(x_values[position + 1])
            if position + 1 < len(x_values)
            else float(reference[0])
        )
        volume += max(next_x - float(x_value), 0.0) * hypervolume_2d_minimization(
            active, float(reference[1]), float(reference[2])
        )
    return float(volume)


def igd_plus(approximation: np.ndarray, reference_front: np.ndarray) -> float:
    approximation = np.asarray(approximation, dtype=float)
    reference_front = np.asarray(reference_front, dtype=float)
    if approximation.size == 0 or reference_front.size == 0:
        return np.inf
    distances = []
    for reference_point in reference_front:
        differences = np.maximum(approximation - reference_point, 0.0)
        distances.append(float(np.min(np.linalg.norm(differences, axis=1))))
    return float(np.mean(distances))


def additive_epsilon_plus(
    approximation: np.ndarray,
    reference_front: np.ndarray,
) -> float:
    approximation = np.asarray(approximation, dtype=float)
    reference_front = np.asarray(reference_front, dtype=float)
    if approximation.size == 0 or reference_front.size == 0:
        return np.inf
    required = []
    for reference_point in reference_front:
        per_approximation = np.max(approximation - reference_point, axis=1)
        required.append(float(np.min(per_approximation)))
    return float(np.max(required))


def spacing_metric(points: np.ndarray) -> float:
    values = np.asarray(points, dtype=float)
    if len(values) < 3:
        return np.nan
    distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    return float(np.std(distances.min(axis=1), ddof=1))


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
            front["normalized_cost"].to_numpy(dtype=float),
        ]
    )


def coverage_indicator(a: np.ndarray, b: np.ndarray) -> float:
    """C(A,B): fraction of B weakly dominated by at least one point in A."""
    A = np.asarray(a, dtype=float)
    B = np.asarray(b, dtype=float)
    if len(B) == 0:
        return np.nan
    if len(A) == 0:
        return 0.0
    covered = [
        bool(np.any(np.all(A <= point + METRIC_EPS, axis=1))) for point in B
    ]
    return float(np.mean(covered))


def normalize_scenario_fronts(
    scenario_fronts: dict[tuple[str, int], pd.DataFrame],
    metric_front_cap: int,
) -> tuple[
    dict[tuple[str, int], np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    raw = {
        key: semantic_front_to_minimization(frame)
        for key, frame in scenario_fronts.items()
        if not frame.empty
    }
    if not raw:
        raise ValueError("No successful Pareto fronts exist in the scenario.")
    pooled = np.vstack(list(raw.values()))
    ideal = pooled.min(axis=0)
    nadir = pooled.max(axis=0)
    span = np.where(nadir - ideal > METRIC_EPS, nadir - ideal, 1.0)

    normalized: dict[tuple[str, int], np.ndarray] = {}
    for key, values in raw.items():
        current = np.clip((values - ideal) / span, 0.0, 1.0)
        current = current[nondominated_mask_minimization(current)]
        current = crowding_downsample(current, metric_front_cap)
        normalized[key] = current

    pooled_normalized = np.vstack(list(normalized.values()))
    reference = pooled_normalized[
        nondominated_mask_minimization(pooled_normalized)
    ]
    reference = crowding_downsample(reference, MAX_REFERENCE_FRONT_SIZE)
    return normalized, reference, ideal, nadir


# =============================================================================
# 8. Run helpers
# =============================================================================
def scenario_tag(k: int, rho: float) -> str:
    return f"K{k}_rho{rho:.2f}".replace(".", "p")


def first_fe_at_or_above(
    history: pd.DataFrame,
    column: str,
    threshold: float,
) -> float:
    if column not in history.columns:
        return np.nan
    values = pd.to_numeric(history[column], errors="coerce")
    rows = history.loc[values >= float(threshold)]
    if rows.empty:
        return np.nan
    return float(rows.iloc[0]["function_evaluations"])


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


def active_es_from_recommendation(
    recommended: pd.Series,
    options: Sequence[Any],
    minimum_action: float,
) -> tuple[str, ...]:
    active = []
    for option in options:
        column = f"x_{option.es}"
        if column in recommended.index and float(recommended[column]) >= (
            minimum_action - METRIC_EPS
        ):
            active.append(str(option.es))
    return tuple(sorted(active))


def select_common_representative(core: Any, pareto: pd.DataFrame) -> pd.Series:
    ranked = core.calculate_relative_robust_scores(
        pareto, epsilon=float(core.ROBUST_EPSILON)
    )
    _, recommended = core.select_robust_representatives(ranked)
    if recommended.empty:
        raise RuntimeError("No common robust representative was selected.")
    return recommended.iloc[0]


def run_one_experiment(
    core: Any,
    original_state: CoreState,
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
    list[FrontSnapshot],
]:
    apply_scenario(
        core,
        original_state,
        seed=seed,
        generations=generations,
        max_active_actions=max_active_actions,
        coverage_threshold=coverage_threshold,
    )
    evaluation_budget = planned_mo_shade_evaluations(
        core, len(options), generations
    )
    tag = scenario_tag(max_active_actions, coverage_threshold)
    log_path = LOG_DIR / f"{tag}_{algorithm.name}_seed_{seed}.log"
    start = time.perf_counter()

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(
                log_file
            ):
                if algorithm.family == "MO_SHADE":
                    archive, history, metadata, snapshots = run_comparable_mo_shade(
                        core,
                        options,
                        context,
                        seed=seed,
                        generations=generations,
                        initialization=algorithm.initialization,
                        evaluation_budget=evaluation_budget,
                    )
                elif algorithm.family == "SHAMODE":
                    archive, history, metadata, snapshots = run_shamode_2019(
                        core,
                        options,
                        context,
                        seed=seed,
                        evaluation_budget=evaluation_budget,
                    )
                else:
                    raise ValueError(f"Unsupported algorithm family: {algorithm.family}")

        runtime = time.perf_counter() - start
        if not archive:
            raise RuntimeError("No feasible Pareto solution was returned.")

        pareto, _ = core.pareto_dataframe(archive, options)
        validate_front(pareto, max_active_actions, coverage_threshold)
        recommended = select_common_representative(core, pareto)
        active_es = active_es_from_recommendation(
            recommended, options, float(core.MIN_ACTION_MAGNITUDE)
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
            "function_evaluations_completed": int(
                metadata["function_evaluations_completed"]
            ),
            "initial_population_size": int(metadata["initial_population_size"]),
            "final_population_size": int(metadata["final_population_size"]),
            "runtime_seconds": float(runtime),
            "pareto_size": int(len(pareto)),
            "initial_feasible_fraction": float(first["feasible_fraction"]),
            "initial_nontrivial_feasible_fraction": float(
                first["nontrivial_feasible_fraction"]
            ),
            "initial_mean_high_priority_coverage": float(
                first["mean_high_priority_coverage"]
            ),
            "final_feasible_fraction": float(last["feasible_fraction"]),
            "final_nontrivial_feasible_fraction": float(
                last["nontrivial_feasible_fraction"]
            ),
            "first_fe_feasible_95": first_fe_at_or_above(
                history, "feasible_fraction", 0.95
            ),
            "first_fe_nontrivial_feasible_95": first_fe_at_or_above(
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
            **{
                f"algorithm_{key}": value
                for key, value in asdict(algorithm).items()
            },
        }
        if algorithm.family == "SHAMODE":
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
        return metrics, pareto, history, snapshots

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
# 9. Scenario-level metrics and HV-FE curves
# =============================================================================
def attach_quality_metrics(
    metrics: pd.DataFrame,
    fronts: dict[tuple[int, float, str, int], pd.DataFrame],
    traces: dict[tuple[int, float, str, int], list[FrontSnapshot]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result = metrics.copy()
    for column in (
        "hypervolume",
        "igd_plus",
        "epsilon_plus",
        "spacing",
        "objective_extent",
        "reference_front_size",
    ):
        result[column] = np.nan

    coverage_rows: list[dict[str, Any]] = []
    hv_fe_rows: list[dict[str, Any]] = []
    successful = result.loc[result["status"] == "success"]
    scenarios = successful[
        ["max_active_actions", "coverage_threshold"]
    ].drop_duplicates()

    for k, rho in scenarios.itertuples(index=False, name=None):
        scenario_fronts: dict[tuple[str, int], pd.DataFrame] = {}
        for (fk, frho, algorithm, seed), frame in fronts.items():
            if int(fk) == int(k) and np.isclose(float(frho), float(rho)):
                scenario_fronts[(algorithm, int(seed))] = frame
        if not scenario_fronts:
            continue

        scenario_rows = successful.loc[
            (successful["max_active_actions"] == int(k))
            & np.isclose(successful["coverage_threshold"], float(rho))
        ]
        metric_front_cap = int(scenario_rows["initial_population_size"].min())
        normalized, reference, ideal, nadir = normalize_scenario_fronts(
            scenario_fronts, metric_front_cap
        )
        span = np.where(nadir - ideal > METRIC_EPS, nadir - ideal, 1.0)

        for (algorithm, seed), values in normalized.items():
            mask = (
                (result["status"] == "success")
                & (result["max_active_actions"] == int(k))
                & np.isclose(result["coverage_threshold"], float(rho))
                & (result["algorithm"] == algorithm)
                & (result["seed"] == int(seed))
            )
            result.loc[mask, "hypervolume"] = hypervolume_3d_minimization(values)
            result.loc[mask, "igd_plus"] = igd_plus(values, reference)
            result.loc[mask, "epsilon_plus"] = additive_epsilon_plus(
                values, reference
            )
            result.loc[mask, "spacing"] = spacing_metric(values)
            result.loc[mask, "objective_extent"] = objective_extent(values)
            result.loc[mask, "reference_front_size"] = int(len(reference))
            for objective, label in enumerate(("rep", "choice", "cost")):
                result.loc[mask, f"scenario_ideal_{label}"] = float(
                    ideal[objective]
                )
                result.loc[mask, f"scenario_nadir_{label}"] = float(
                    nadir[objective]
                )

            key = (int(k), float(rho), algorithm, int(seed))
            for snapshot in traces.get(key, []):
                raw = np.asarray(snapshot.objectives, dtype=float)
                if raw.size == 0:
                    hv = 0.0
                    front_size = 0
                else:
                    current = np.clip((raw - ideal) / span, 0.0, 1.0)
                    current = current[nondominated_mask_minimization(current)]
                    current = crowding_downsample(current, metric_front_cap)
                    hv = hypervolume_3d_minimization(current)
                    front_size = int(len(current))
                budget = int(
                    scenario_rows.loc[
                        (scenario_rows["algorithm"] == algorithm)
                        & (scenario_rows["seed"] == int(seed)),
                        "function_evaluation_budget",
                    ].iloc[0]
                )
                hv_fe_rows.append(
                    {
                        "max_active_actions": int(k),
                        "coverage_threshold": float(rho),
                        "scenario": scenario_tag(int(k), float(rho)),
                        "algorithm": algorithm,
                        "seed": int(seed),
                        "target_fe": int(snapshot.target_fe),
                        "actual_fe": int(snapshot.actual_fe),
                        "fe_ratio": float(snapshot.target_fe / budget),
                        "generation": int(snapshot.generation),
                        "hypervolume": float(hv),
                        "front_size_for_metric": front_size,
                    }
                )

        common_seeds_by_algorithm = [
            {
                seed
                for algorithm, seed in normalized
                if algorithm == algorithm_name
            }
            for algorithm_name in ALGORITHM_NAMES
        ]
        common_seeds = (
            sorted(set.intersection(*common_seeds_by_algorithm))
            if common_seeds_by_algorithm
            else []
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

    hv_fe = pd.DataFrame(hv_fe_rows)
    if not hv_fe.empty:
        aggregate_keys = [
            "max_active_actions",
            "coverage_threshold",
            "scenario",
            "algorithm",
            "target_fe",
            "fe_ratio",
        ]
        aggregate = hv_fe.groupby(aggregate_keys, as_index=False).agg(
            count=("hypervolume", "count"),
            mean=("hypervolume", "mean"),
            std=("hypervolume", "std"),
        )
        aggregate["std"] = aggregate["std"].fillna(0.0)
        aggregate["ci95"] = 1.96 * aggregate["std"] / np.sqrt(
            aggregate["count"].clip(lower=1)
        )
        hv_fe = hv_fe.merge(
            aggregate,
            on=[
                "max_active_actions",
                "coverage_threshold",
                "scenario",
                "algorithm",
                "target_fe",
                "fe_ratio",
            ],
            how="left",
            suffixes=("", "_aggregate"),
        )

    return result, pd.DataFrame(coverage_rows), hv_fe


# =============================================================================
# 10. Aggregation and statistical inference
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
    rows = []
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
    scopes = list(
        successful[["max_active_actions", "coverage_threshold"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    for k, rho in scopes:
        subset = successful.loc[
            (successful["max_active_actions"] == int(k))
            & np.isclose(successful["coverage_threshold"], float(rho))
        ]
        for algorithm_a, algorithm_b in PAIRWISE_COMPARISONS:
            a = subset.loc[subset["algorithm"] == algorithm_a].set_index("seed")
            b = subset.loc[subset["algorithm"] == algorithm_b].set_index("seed")
            common = a.index.intersection(b.index)

            for metric, direction in PRIMARY_TEST_METRICS.items():
                if metric not in a.columns or metric not in b.columns:
                    continue
                av = pd.to_numeric(a.loc[common, metric], errors="coerce")
                bv = pd.to_numeric(b.loc[common, metric], errors="coerce")
                valid = av.notna() & bv.notna()
                a_values = av.loc[valid].to_numpy(dtype=float)
                b_values = bv.loc[valid].to_numpy(dtype=float)
                differences = a_values - b_values

                if len(differences) == 0:
                    statistic = np.nan
                    p_value = np.nan
                elif np.all(np.abs(differences) <= METRIC_EPS):
                    statistic = 0.0
                    p_value = 1.0
                else:
                    test = wilcoxon(
                        a_values,
                        b_values,
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
                        "max_active_actions": int(k),
                        "coverage_threshold": float(rho),
                        "metric": metric,
                        "direction": "larger_better"
                        if direction > 0
                        else "smaller_better",
                        "algorithm_A": algorithm_a,
                        "algorithm_B": algorithm_b,
                        "n_pairs": int(len(differences)),
                        "A_mean": float(np.mean(a_values))
                        if len(a_values)
                        else np.nan,
                        "B_mean": float(np.mean(b_values))
                        if len(b_values)
                        else np.nan,
                        "A_minus_B_mean": float(np.mean(differences))
                        if len(differences)
                        else np.nan,
                        "win_rate_A": float(np.mean(signed > METRIC_EPS))
                        if len(signed)
                        else np.nan,
                        "tie_rate": float(np.mean(np.abs(signed) <= METRIC_EPS))
                        if len(signed)
                        else np.nan,
                        "wilcoxon_statistic": statistic,
                        "p_value": p_value,
                        "rank_biserial_A_better": paired_rank_biserial(signed),
                    }
                )

    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_holm"] = np.nan
        group_columns = [
            "max_active_actions",
            "coverage_threshold",
            "metric",
        ]
        for _, indices in result.groupby(group_columns).groups.items():
            positions = list(indices)
            result.loc[positions, "p_holm"] = holm_adjust(
                result.loc[positions, "p_value"].to_numpy(dtype=float)
            )
        result["significant_0p05"] = result["p_holm"] < 0.05
    return result


def friedman_tests(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    successful = metrics.loc[metrics["status"] == "success"].copy()
    test_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []

    scopes = list(
        successful[["max_active_actions", "coverage_threshold"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    for k, rho in scopes:
        subset = successful.loc[
            (successful["max_active_actions"] == int(k))
            & np.isclose(successful["coverage_threshold"], float(rho))
        ]
        for metric, direction in PRIMARY_TEST_METRICS.items():
            pivot = subset.pivot_table(
                index="seed", columns="algorithm", values=metric, aggfunc="first"
            )
            if not set(ALGORITHM_NAMES).issubset(pivot.columns):
                continue
            pivot = pivot.loc[:, list(ALGORITHM_NAMES)].dropna()
            if len(pivot) < 2:
                continue
            arrays = [pivot[name].to_numpy(dtype=float) for name in ALGORITHM_NAMES]
            if all(np.allclose(arrays[0], array) for array in arrays[1:]):
                statistic, p_value = 0.0, 1.0
            else:
                test = friedmanchisquare(*arrays)
                statistic, p_value = float(test.statistic), float(test.pvalue)
            test_rows.append(
                {
                    "max_active_actions": int(k),
                    "coverage_threshold": float(rho),
                    "metric": metric,
                    "n_complete_seeds": int(len(pivot)),
                    "friedman_chi_square": statistic,
                    "p_value": p_value,
                }
            )

            rank_matrix = []
            for _, row in pivot.iterrows():
                values = row.to_numpy(dtype=float)
                ranking_values = -values if direction > 0 else values
                rank_matrix.append(rankdata(ranking_values, method="average"))
            mean_ranks = np.mean(np.asarray(rank_matrix), axis=0)
            for algorithm, mean_rank in zip(ALGORITHM_NAMES, mean_ranks):
                rank_rows.append(
                    {
                        "max_active_actions": int(k),
                        "coverage_threshold": float(rho),
                        "metric": metric,
                        "algorithm": algorithm,
                        "mean_rank": float(mean_rank),
                        "n_complete_seeds": int(len(pivot)),
                    }
                )

    return pd.DataFrame(test_rows), pd.DataFrame(rank_rows)


# =============================================================================
# 11. Plots and outputs
# =============================================================================
def plot_hv_fe_curves(hv_fe: pd.DataFrame) -> None:
    if hv_fe.empty or not SAVE_PLOTS:
        return
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    aggregate = hv_fe[
        [
            "max_active_actions",
            "coverage_threshold",
            "scenario",
            "algorithm",
            "target_fe",
            "fe_ratio",
            "mean",
            "ci95",
        ]
    ].drop_duplicates()

    for scenario, group in aggregate.groupby("scenario", sort=True):
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for algorithm, algorithm_group in group.groupby("algorithm", sort=False):
            algorithm_group = algorithm_group.sort_values("fe_ratio")
            x = algorithm_group["fe_ratio"].to_numpy(dtype=float)
            y = algorithm_group["mean"].to_numpy(dtype=float)
            ci = algorithm_group["ci95"].to_numpy(dtype=float)
            ax.plot(x, y, linewidth=1.8, label=algorithm)
            ax.fill_between(x, y - ci, y + ci, alpha=0.15)
        ax.set_xlabel("Fraction of equal FE budget")
        ax.set_ylabel("Scenario-normalized hypervolume")
        ax.set_title(f"HV-FE convergence: {scenario}")
        ax.set_xlim(0.0, 1.0)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"HV_FE_{scenario}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def configuration_frame(
    core_file: Path,
    seeds: Sequence[int],
    generations: int,
    scenarios: Sequence[tuple[int, float]],
) -> pd.DataFrame:
    rows: dict[str, Any] = {
        "core_file": str(core_file),
        "run_mode": RUN_MODE,
        "seeds": str(tuple(int(seed) for seed in seeds)),
        "n_seeds": int(len(seeds)),
        "mo_shade_generations": int(generations),
        "scenarios": str(tuple(scenarios)),
        "population_multiplier": EXPERIMENT_POPULATION_MULTIPLIER,
        "min_population_size": EXPERIMENT_MIN_POPULATION_SIZE,
        "max_population_size": EXPERIMENT_MAX_POPULATION_SIZE,
        "ipea_exploration_rate": IPEA_EXPLORATION_RATE,
        "shamode_memory_size": SHAMODE_MEMORY_SIZE,
        "shamode_mutation_archive_rate": SHAMODE_MUTATION_ARCHIVE_RATE,
        "shamode_f_scale": SHAMODE_F_SCALE,
        "shamode_cr_std": SHAMODE_CR_STD,
        "trace_checkpoints": TRACE_CHECKPOINTS,
        "hv_reference_point": str(HV_REFERENCE_POINT.tolist()),
        "metric_front_cap": "common initial population size within scenario",
        "primary_comparison": "M1_IPEA_MO_SHADE versus SHAMODE_2019",
        "mechanism_control": "M0_uniform_MO_SHADE versus SHAMODE_2019",
        "ipea_ablation": "M1_IPEA_MO_SHADE versus M0_uniform_MO_SHADE",
    }
    for index, algorithm in enumerate(ALGORITHMS, start=1):
        rows[f"algorithm_{index}"] = str(asdict(algorithm))
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
    hv_fe: pd.DataFrame,
    convergence: pd.DataFrame,
    configuration: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "run_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(
        OUTPUT_DIR / "summary_by_scenario.csv", index=False, encoding="utf-8-sig"
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
    hv_fe.to_csv(
        OUTPUT_DIR / "hv_fe_curve.csv", index=False, encoding="utf-8-sig"
    )
    convergence.to_csv(
        OUTPUT_DIR / "convergence_history.csv", index=False, encoding="utf-8-sig"
    )

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="Run_metrics", index=False)
        summary.to_excel(writer, sheet_name="Scenario_summary", index=False)
        pairwise.to_excel(writer, sheet_name="Pairwise_tests", index=False)
        friedman.to_excel(writer, sheet_name="Friedman_tests", index=False)
        ranks.to_excel(writer, sheet_name="Algorithm_ranks", index=False)
        coverage.to_excel(writer, sheet_name="Coverage_indicator", index=False)
        hv_fe.to_excel(writer, sheet_name="HV_FE_curve", index=False)
        convergence.to_excel(writer, sheet_name="Convergence", index=False)
        configuration.to_excel(writer, sheet_name="Configuration", index=False)

    plot_hv_fe_curves(hv_fe)


# =============================================================================
# 12. Internal checks
# =============================================================================
def run_internal_checks() -> None:
    if SHAMODE_MEMORY_SIZE != 5:
        raise ValueError("The paper-faithful SHAMODE memory size must be H=5.")
    if not np.isclose(SHAMODE_MUTATION_ARCHIVE_RATE, 1.4):
        raise ValueError("The paper-faithful SHAMODE archive rate must be 1.4.")
    if TRACE_CHECKPOINTS < 2:
        raise ValueError("TRACE_CHECKPOINTS must be at least 2.")
    if len(ALGORITHMS) != 3 or len(set(ALGORITHM_NAMES)) != 3:
        raise ValueError("Exactly three unique algorithms are required.")

    # Basic metric identities.
    simple = np.asarray([[0.2, 0.2, 0.2]], dtype=float)
    hv = hypervolume_3d_minimization(simple, np.ones(3))
    expected = 0.8**3
    if not np.isclose(hv, expected, rtol=1e-12, atol=1e-12):
        raise RuntimeError("3D hypervolume internal check failed.")
    if not np.isclose(coverage_indicator(simple, simple), 1.0):
        raise RuntimeError("Coverage-indicator internal check failed.")


# =============================================================================
# 13. Main
# =============================================================================
def resolve_experiment() -> tuple[tuple[int, ...], int, tuple[tuple[int, float], ...]]:
    mode = str(RUN_MODE).strip().lower()
    if mode == "pilot":
        scenarios = tuple(
            (int(k), float(rho))
            for k in FORMAL_MAX_ACTIVE_ACTIONS
            for rho in FORMAL_COVERAGE_THRESHOLDS
        )
        return (
            tuple(int(seed) for seed in PILOT_SEEDS),
            int(PILOT_GENERATIONS),
            scenarios,
        )
    if mode == "formal":
        scenarios = tuple(
            (int(k), float(rho))
            for k in FORMAL_MAX_ACTIVE_ACTIONS
            for rho in FORMAL_COVERAGE_THRESHOLDS
        )
        return (
            tuple(int(seed) for seed in FORMAL_SEEDS),
            int(FORMAL_GENERATIONS),
            scenarios,
        )
    raise ValueError("RUN_MODE must be 'pilot' or 'formal'.")


def main() -> None:
    run_internal_checks()
    core_file = locate_core_file()
    core = load_core_module(core_file)
    core.run_internal_checks()
    original_state = capture_core_state(core)

    seeds, generations, scenarios = resolve_experiment()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FRONT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Input data are loaded once.  Scenario globals affect only optimization,
    # not the definitions of service options or the MNL context.
    options, context, _, _ = core.load_inputs()

    metric_rows: list[dict[str, Any]] = []
    history_frames: list[pd.DataFrame] = []
    fronts: dict[tuple[int, float, str, int], pd.DataFrame] = {}
    traces: dict[tuple[int, float, str, int], list[FrontSnapshot]] = {}

    total_runs = len(seeds) * len(scenarios) * len(ALGORITHMS)
    completed = 0
    print(
        f"Starting {RUN_MODE} comparison: {total_runs} runs, "
        f"{len(options)} service elements."
    )

    try:
        for max_active_actions, coverage_threshold in scenarios:
            for seed in seeds:
                # Paired seed; alternate execution order to reduce systematic
                # thermal/cache ordering effects.
                order = list(ALGORITHMS)
                if seed % 2:
                    order.reverse()
                for algorithm in order:
                    completed += 1
                    print(
                        f"[{completed}/{total_runs}] {scenario_tag(max_active_actions, coverage_threshold)} "
                        f"seed={seed} algorithm={algorithm.name}"
                    )
                    metrics, pareto, history, snapshots = run_one_experiment(
                        core,
                        original_state,
                        algorithm,
                        seed=seed,
                        generations=generations,
                        max_active_actions=max_active_actions,
                        coverage_threshold=coverage_threshold,
                        options=options,
                        context=context,
                    )
                    metric_rows.append(metrics)
                    if metrics.get("status") == "success":
                        key = (
                            int(max_active_actions),
                            float(coverage_threshold),
                            algorithm.name,
                            int(seed),
                        )
                        fronts[key] = pareto
                        traces[key] = snapshots
                        history_frames.append(history)
                        if SAVE_EACH_FRONT:
                            pareto.to_csv(
                                FRONT_DIR
                                / (
                                    f"{scenario_tag(max_active_actions, coverage_threshold)}_"
                                    f"{algorithm.name}_seed_{seed}.csv"
                                ),
                                index=False,
                                encoding="utf-8-sig",
                            )
                    else:
                        print(f"  FAILED: {metrics.get('error')}")
    finally:
        restore_core_state(core, original_state)

    metrics = pd.DataFrame(metric_rows)
    metrics, coverage, hv_fe = attach_quality_metrics(metrics, fronts, traces)
    summary = summarize_by_scenario(metrics)
    pairwise = pairwise_tests(metrics)
    friedman, ranks = friedman_tests(metrics)
    convergence = (
        pd.concat(history_frames, ignore_index=True)
        if history_frames
        else pd.DataFrame()
    )
    configuration = configuration_frame(
        core_file, seeds, generations, scenarios
    )
    save_outputs(
        metrics,
        summary,
        pairwise,
        friedman,
        ranks,
        coverage,
        hv_fe,
        convergence,
        configuration,
    )

    failures = metrics.loc[metrics["status"] != "success"]
    print("\nComparison completed.")
    print(f"Successful runs: {int((metrics['status'] == 'success').sum())}")
    print(f"Failed runs: {len(failures)}")
    print(f"Workbook: {OUTPUT_EXCEL}")
    if not failures.empty:
        print("Inspect logs for failed runs:")
        print(failures[["scenario", "algorithm", "seed", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
