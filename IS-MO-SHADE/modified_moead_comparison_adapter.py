# -*- coding: utf-8 -*-
"""
Reusable modified MOEA/D adapter for fair algorithm comparison
===============================================================

This module contains only the optimization mechanism.  It does not load a
problem file and does not write standalone outputs.  A comparison runner passes
in the already-loaded hotel MOO core module, ensuring that M0, M1-I and
modified MOEA/D use exactly the same:

- input data and ServiceOption objects;
- objective evaluator;
- repair function and decision domain;
- constraint-violation function;
- feasible Pareto archive definition;
- scenario values for MAX_ACTIVE_ACTIONS and coverage threshold.

The modified MOEA/D mechanism retains:

- H=23 simplex-lattice weights (300 subproblems for three objectives);
- Euclidean weight-vector neighborhoods;
- range-corrected modified Tchebycheff decomposition;
- neighborhood DE/rand/1/bin with F=0.5 and CR=0.5;
- normalized constraint penalty for residual violations;
- external feasible nondominated archive.

The stopping criterion is an externally supplied objective-evaluation budget.
This is essential for fair comparison with MO-SHADE's shrinking population.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


# =============================================================================
# 1. Algorithm configuration
# =============================================================================
LATTICE_DIVISIONS = 23
NEIGHBORHOOD_SIZE = 20
DE_SCALE_FACTOR = 0.50
DE_CROSSOVER_RATE = 0.50
CONSTRAINT_PENALTY_FACTOR = 100.0
MIN_OBJECTIVE_RANGE = 1e-10
WEIGHT_ZERO_TOLERANCE = 0.0
PRINT_EVERY_GENERATIONS = 10


@dataclass(frozen=True)
class ModifiedMOEADConfig:
    lattice_divisions: int = LATTICE_DIVISIONS
    neighborhood_size: int = NEIGHBORHOOD_SIZE
    de_scale_factor: float = DE_SCALE_FACTOR
    de_crossover_rate: float = DE_CROSSOVER_RATE
    constraint_penalty_factor: float = CONSTRAINT_PENALTY_FACTOR
    minimum_objective_range: float = MIN_OBJECTIVE_RANGE
    weight_zero_tolerance: float = WEIGHT_ZERO_TOLERANCE
    print_every_generations: int = PRINT_EVERY_GENERATIONS


# =============================================================================
# 2. Fair objective-evaluation budget
# =============================================================================
def initial_mo_shade_population_size(base: Any, n_options: int) -> int:
    return int(
        np.clip(
            int(base.POPULATION_MULTIPLIER) * int(n_options),
            int(base.MIN_POPULATION_SIZE),
            int(base.MAX_POPULATION_SIZE),
        )
    )


def planned_mo_shade_evaluations(
    base: Any,
    n_options: int,
    generations: int | None = None,
) -> int:
    """Return the exact planned FE count of the configured MO-SHADE run."""
    n_generations = int(base.N_GENERATIONS if generations is None else generations)
    if n_generations < 1:
        raise ValueError("generations must be positive.")

    initial_size = initial_mo_shade_population_size(base, n_options)
    evaluations = initial_size
    current_size = initial_size

    # target_population_size uses base.N_GENERATIONS in the current core.
    original_generations = int(base.N_GENERATIONS)
    try:
        base.N_GENERATIONS = n_generations
        for generation in range(1, n_generations + 1):
            evaluations += current_size
            current_size = int(base.target_population_size(initial_size, generation))
    finally:
        base.N_GENERATIONS = original_generations

    return int(evaluations)


def add_mo_shade_evaluation_axis(
    base: Any,
    history: pd.DataFrame,
    n_options: int,
    generations: int | None = None,
) -> pd.DataFrame:
    """Add estimated cumulative FE to MO-SHADE history rows.

    The current core records one history row after every generation but does
    not store objective-evaluation counts.  Under fixed generations, each
    generation evaluates one trial per member of the population entering that
    generation, so the cumulative count is deterministic.
    """
    result = history.copy()
    if result.empty:
        result["function_evaluations"] = pd.Series(dtype=int)
        return result

    n_generations = int(base.N_GENERATIONS if generations is None else generations)
    initial_size = initial_mo_shade_population_size(base, n_options)
    cumulative = initial_size
    current_size = initial_size
    fe_by_generation: dict[int, int] = {}

    original_generations = int(base.N_GENERATIONS)
    try:
        base.N_GENERATIONS = n_generations
        for generation in range(1, n_generations + 1):
            cumulative += current_size
            fe_by_generation[generation] = int(cumulative)
            current_size = int(base.target_population_size(initial_size, generation))
    finally:
        base.N_GENERATIONS = original_generations

    generation_values = pd.to_numeric(result["generation"], errors="coerce")
    result["function_evaluations"] = generation_values.map(fe_by_generation)
    return result


# =============================================================================
# 3. Decomposition and neighborhoods
# =============================================================================
def simplex_lattice_weights(n_objectives: int, divisions: int) -> np.ndarray:
    if n_objectives != 3:
        raise ValueError(
            "The hotel model uses exactly three objectives; received "
            f"n_objectives={n_objectives}."
        )
    if divisions < 1:
        raise ValueError("divisions must be positive.")

    vectors: list[tuple[float, float, float]] = []
    for i in range(divisions + 1):
        for j in range(divisions - i + 1):
            k = divisions - i - j
            vectors.append((i / divisions, j / divisions, k / divisions))

    weights = np.asarray(vectors, dtype=float)
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-12):
        raise RuntimeError("Simplex-lattice vectors do not sum to one.")
    return weights


def build_weight_neighborhoods(
    weights: np.ndarray,
    neighborhood_size: int,
) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    n = len(values)
    if not 3 <= int(neighborhood_size) <= n:
        raise ValueError(
            f"neighborhood_size must lie in [3,{n}], received "
            f"{neighborhood_size}."
        )
    distances = np.linalg.norm(
        values[:, None, :] - values[None, :, :],
        axis=2,
    )
    return np.argsort(distances, axis=1)[:, : int(neighborhood_size)]


# =============================================================================
# 4. Priority-neutral initialization
# =============================================================================
def neutral_random_vector(
    base: Any,
    options: Sequence[Any],
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(options)
    if n == 0:
        raise ValueError("No service options are available.")

    if rng.random() < float(base.INITIAL_ZERO_SOLUTION_RATE):
        return np.zeros(n, dtype=float)

    low_fraction, high_fraction = base.INITIAL_ACTIVE_FRACTION_RANGE
    minimum = max(1, int(math.ceil(float(low_fraction) * n)))
    maximum = max(minimum, int(math.ceil(float(high_fraction) * n)))
    maximum = min(maximum, n)

    if base.MAX_ACTIVE_ACTIONS is not None:
        cap = int(base.MAX_ACTIVE_ACTIONS)
        if cap < 1:
            raise ValueError("MAX_ACTIVE_ACTIONS must be positive or None.")
        maximum = min(maximum, cap)
        minimum = min(minimum, maximum)

    if maximum < 1 or minimum > maximum:
        raise RuntimeError(
            f"Invalid active-count interval [{minimum},{maximum}]."
        )

    n_active = int(rng.integers(minimum, maximum + 1))
    selected = rng.choice(n, size=n_active, replace=False)

    vector = np.zeros(n, dtype=float)
    for raw_index in selected:
        index = int(raw_index)
        option = options[index]
        vector[index] = float(
            rng.uniform(float(base.MIN_ACTION_MAGNITUDE), float(option.max_delta))
        )
    return base.repair_vector(vector, options)


def create_initial_population(
    base: Any,
    population_size: int,
    options: Sequence[Any],
    context: Any,
    rng: np.random.Generator,
) -> list[Any]:
    return [
        base.make_individual(
            neutral_random_vector(base, options, rng),
            options,
            context,
            origin="modified_moead_initial_random",
        )
        for _ in range(int(population_size))
    ]


# =============================================================================
# 5. Scalarization and constraint penalty
# =============================================================================
def objective_matrix(individuals: Sequence[Any]) -> np.ndarray:
    if not individuals:
        return np.empty((0, 3), dtype=float)
    matrix = np.asarray(
        [individual.objectives for individual in individuals],
        dtype=float,
    )
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError(f"Invalid objective matrix shape: {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Objective matrix contains non-finite values.")
    return matrix


def modified_tchebycheff(
    individual: Any,
    weight: np.ndarray,
    ideal_point: np.ndarray,
    nadir_point: np.ndarray,
    config: ModifiedMOEADConfig,
) -> float:
    objective = np.asarray(individual.objectives, dtype=float)
    lam = np.asarray(weight, dtype=float)
    if config.weight_zero_tolerance > 0.0:
        lam = np.maximum(lam, float(config.weight_zero_tolerance))

    spans = np.maximum(
        np.asarray(nadir_point) - np.asarray(ideal_point),
        float(config.minimum_objective_range),
    )
    normalized_deviation = np.abs(objective - ideal_point) / spans
    scalar_value = float(np.max(lam * normalized_deviation))
    penalty = float(config.constraint_penalty_factor) * max(
        float(individual.violation), 0.0
    )
    return scalar_value + penalty


# =============================================================================
# 6. Reproduction and neighborhood replacement
# =============================================================================
def generate_offspring(
    base: Any,
    subproblem_index: int,
    population: Sequence[Any],
    neighborhoods: np.ndarray,
    options: Sequence[Any],
    context: Any,
    rng: np.random.Generator,
    config: ModifiedMOEADConfig,
) -> Any:
    neighborhood = np.asarray(neighborhoods[subproblem_index], dtype=int)
    if neighborhood.size < 3:
        raise RuntimeError("At least three neighboring subproblems are required.")

    r1, r2, r3 = rng.choice(neighborhood, size=3, replace=False)
    x1 = population[int(r1)].x
    x2 = population[int(r2)].x
    x3 = population[int(r3)].x

    mutant = x1 + float(config.de_scale_factor) * (x2 - x3)
    mutant = base.repair_vector(mutant, options)

    target = population[int(subproblem_index)].x
    trial = target.copy()
    forced_dimension = int(rng.integers(0, len(options)))
    mask = rng.random(len(options)) <= float(config.de_crossover_rate)
    mask[forced_dimension] = True
    trial[mask] = mutant[mask]
    trial = base.repair_vector(trial, options)

    return base.make_individual(
        trial,
        options,
        context,
        origin="modified_moead_trial",
        F=float(config.de_scale_factor),
        CR=float(config.de_crossover_rate),
        parent_index=int(subproblem_index),
    )


def replace_neighborhood(
    offspring: Any,
    population: list[Any],
    neighborhood: np.ndarray,
    weights: np.ndarray,
    ideal_point: np.ndarray,
    nadir_point: np.ndarray,
    rng: np.random.Generator,
    config: ModifiedMOEADConfig,
) -> int:
    """
    Vectorized neighborhood replacement for modified MOEA/D.

    The offspring is compared with every neighboring subproblem using the
    same modified Tchebycheff scalarization and constraint penalty as the
    original sequential implementation.
    """
    neighborhood_array = np.asarray(neighborhood, dtype=int)

    if neighborhood_array.ndim != 1 or neighborhood_array.size == 0:
        raise ValueError("neighborhood must be a non-empty one-dimensional array.")

    # Preserve randomized neighborhood-processing semantics.
    order = rng.permutation(neighborhood_array)

    neighbor_weights = np.asarray(weights[order], dtype=float)

    span = np.maximum(
        np.asarray(nadir_point, dtype=float)
        - np.asarray(ideal_point, dtype=float),
        float(config.minimum_objective_range),
    )

    # ------------------------------------------------------------------
    # 1. Offspring scalar values for all neighboring subproblems
    # ------------------------------------------------------------------
    child_objectives = np.asarray(
        offspring.objectives,
        dtype=float,
    )

    child_deviation = (
        np.abs(child_objectives - ideal_point)
        / span
    )

    child_values = np.max(
        neighbor_weights * child_deviation[None, :],
        axis=1,
    )

    child_penalty = (
        float(config.constraint_penalty_factor)
        * max(float(offspring.violation), 0.0)
    )
    child_values = child_values + child_penalty

    # ------------------------------------------------------------------
    # 2. Incumbent scalar values for the corresponding subproblems
    # ------------------------------------------------------------------
    incumbent_objectives = np.asarray(
        [
            population[int(index)].objectives
            for index in order
        ],
        dtype=float,
    )

    incumbent_deviation = (
        np.abs(
            incumbent_objectives
            - ideal_point[None, :]
        )
        / span[None, :]
    )

    incumbent_values = np.max(
        neighbor_weights * incumbent_deviation,
        axis=1,
    )

    incumbent_penalties = np.asarray(
        [
            float(config.constraint_penalty_factor)
            * max(
                float(population[int(index)].violation),
                0.0,
            )
            for index in order
        ],
        dtype=float,
    )

    incumbent_values = (
        incumbent_values
        + incumbent_penalties
    )

    # ------------------------------------------------------------------
    # 3. Replace every neighboring subproblem improved by the offspring
    # ------------------------------------------------------------------
    replace_mask = (
        child_values
        < incumbent_values - 1e-12
    )

    replacement_indices = order[replace_mask]

    for raw_index in replacement_indices:
        population[int(raw_index)] = offspring.copy()

    return int(replacement_indices.size)


def update_external_archive(
    base: Any,
    archive: Sequence[Any],
    candidates: Sequence[Any],
    archive_cap: int | None = None,
) -> list[Any]:
    original_cap = int(base.MAX_PARETO_ARCHIVE_SIZE)
    try:
        if archive_cap is not None:
            base.MAX_PARETO_ARCHIVE_SIZE = int(archive_cap)
        return base.update_pareto_archive(archive, candidates)
    finally:
        base.MAX_PARETO_ARCHIVE_SIZE = original_cap


# =============================================================================
# 7. History helpers
# =============================================================================
def _population_diagnostics(base: Any, population: Sequence[Any]) -> dict[str, float]:
    feasible = np.asarray(
        [individual.violation <= float(base.EPS) for individual in population],
        dtype=bool,
    )
    nontrivial = np.asarray(
        [
            individual.violation <= float(base.EPS)
            and individual.components["sum_delta"]
            > float(base.NONTRIVIAL_DELTA_TOLERANCE)
            for individual in population
        ],
        dtype=bool,
    )
    coverages = np.asarray(
        [
            individual.components["high_priority_coverage"]
            for individual in population
        ],
        dtype=float,
    )
    violations = np.asarray(
        [individual.violation for individual in population],
        dtype=float,
    )
    return {
        "feasible_fraction": float(feasible.mean()),
        "nontrivial_feasible_fraction": float(nontrivial.mean()),
        "mean_high_priority_coverage": float(coverages.mean()),
        "mean_constraint_violation": float(violations.mean()),
    }


def _archive_diagnostics(archive: Sequence[Any]) -> dict[str, float]:
    if not archive:
        return {
            "best_reputation_improvement": np.nan,
            "best_choice_probability_gain": np.nan,
            "lowest_effective_cost": np.nan,
        }
    return {
        "best_reputation_improvement": float(
            max(
                individual.components["reputation_improvement"]
                for individual in archive
            )
        ),
        "best_choice_probability_gain": float(
            max(individual.components["probability_gain"] for individual in archive)
        ),
        "lowest_effective_cost": float(
            min(individual.components["effective_cost"] for individual in archive)
        ),
    }


# =============================================================================
# 8. Solver
# =============================================================================
def run_modified_moead(
    base: Any,
    options: Sequence[Any],
    context: Any,
    *,
    seed: int,
    evaluation_budget: int,
    config: ModifiedMOEADConfig | None = None,
    archive_cap: int | None = None,
) -> tuple[list[Any], pd.DataFrame, dict[str, Any]]:
    """Run modified MOEA/D under an exact objective-evaluation budget."""
    settings = config or ModifiedMOEADConfig()

    evaluation_budget = int(evaluation_budget)
    if evaluation_budget < 1:
        raise ValueError("evaluation_budget must be positive.")

    rng = np.random.default_rng(int(seed))

    weights = simplex_lattice_weights(
        3,
        int(settings.lattice_divisions),
    )
    population_size = len(weights)

    if evaluation_budget < population_size:
        raise ValueError(
            f"Evaluation budget {evaluation_budget} is smaller than modified "
            f"MOEA/D population size {population_size}."
        )

    neighborhoods = build_weight_neighborhoods(
        weights,
        int(settings.neighborhood_size),
    )

    population = create_initial_population(
        base,
        population_size,
        options,
        context,
        rng,
    )
    evaluations = population_size

    initial_objectives = objective_matrix(population)
    ideal_point = initial_objectives.min(axis=0)
    nadir_point = initial_objectives.max(axis=0)

    external_archive = update_external_archive(
        base,
        [],
        population,
        archive_cap=archive_cap,
    )

    history_rows: list[dict[str, Any]] = []
    generation = 0

    while evaluations < evaluation_budget:
        generation += 1
        replacements = 0
        offspring_count = 0

        # 必须每代重新初始化，不能跨代累积。
        generation_offspring: list[Any] = []

        # 保留MOEA/D的序贯更新机制：
        # 每生成一个子代，立即更新理想点并替换邻域。
        for raw_subproblem in rng.permutation(population_size):
            if evaluations >= evaluation_budget:
                break

            subproblem = int(raw_subproblem)

            offspring = generate_offspring(
                base,
                subproblem,
                population,
                neighborhoods,
                options,
                context,
                rng,
                settings,
            )

            generation_offspring.append(offspring)
            evaluations += 1
            offspring_count += 1

            objective = np.asarray(
                offspring.objectives,
                dtype=float,
            )

            ideal_point = np.minimum(
                ideal_point,
                objective,
            )
            nadir_point = np.maximum(
                nadir_point,
                objective,
            )

            replacements += replace_neighborhood(
                offspring,
                population,
                neighborhoods[subproblem],
                weights,
                ideal_point,
                nadir_point,
                rng,
                settings,
            )

        # 一代中的全部子代生成完成后，只更新一次外部档案。
        if generation_offspring:
            external_archive = update_external_archive(
                base,
                external_archive,
                generation_offspring,
                archive_cap=archive_cap,
            )

        row = {
            "generation": int(generation),
            "function_evaluations": int(evaluations),
            "population_size": int(population_size),
            "pareto_archive_size": int(len(external_archive)),
            "offspring_evaluated": int(offspring_count),
            "neighborhood_replacements": int(replacements),
            "front_shift": np.nan,
            **_population_diagnostics(
                base,
                population,
            ),
            **_archive_diagnostics(
                external_archive,
            ),
        }
        history_rows.append(row)

        print_interval = int(settings.print_every_generations)
        if (
            generation == 1
            or (
                print_interval > 0
                and generation % print_interval == 0
            )
        ):
            best_choice = row["best_choice_probability_gain"]

            print(
                f"modified MOEA/D generation {generation:4d}: "
                f"FE={evaluations:7d}/{evaluation_budget}, "
                f"EP={len(external_archive):4d}, "
                f"feasible={row['feasible_fraction']:.3f}, "
                f"best_choice_pp={100.0 * best_choice:.4f}"
            )

    metadata = {
        "algorithm": "modified_MOEA_D",
        "initial_population_size": int(population_size),
        "final_population_size": int(population_size),
        "population_size": int(population_size),
        "generations_completed": int(generation),
        "pareto_archive_size": int(len(external_archive)),
        "function_evaluations_completed": int(evaluations),
        "maximum_function_evaluations": int(evaluation_budget),
        "simplex_lattice_divisions": int(
            settings.lattice_divisions
        ),
        "neighborhood_size": int(
            settings.neighborhood_size
        ),
        "DE_scale_factor": float(
            settings.de_scale_factor
        ),
        "DE_crossover_rate": float(
            settings.de_crossover_rate
        ),
        "constraint_penalty_factor": float(
            settings.constraint_penalty_factor
        ),
        "archive_update_frequency": "once_per_generation",
        "archive_cap": archive_cap,
        "IPEA_guided_initialization": False,
        "objective_vector": (
            "(-reputation_improvement, "
            "-choice_probability_gain, "
            "normalized_cost)"
        ),
        "ideal_point": ideal_point.copy(),
        "nadir_point": nadir_point.copy(),
    }

    return (
        external_archive,
        pd.DataFrame(history_rows),
        metadata,
    )

# =============================================================================
# 9. Internal validation
# =============================================================================
def run_internal_checks(config: ModifiedMOEADConfig | None = None) -> None:
    settings = config or ModifiedMOEADConfig()
    weights = simplex_lattice_weights(3, int(settings.lattice_divisions))
    expected = math.comb(int(settings.lattice_divisions) + 2, 2)
    if len(weights) != expected:
        raise RuntimeError(
            f"Expected {expected} weight vectors, obtained {len(weights)}."
        )
    neighborhoods = build_weight_neighborhoods(
        weights,
        int(settings.neighborhood_size),
    )
    if neighborhoods.shape != (len(weights), int(settings.neighborhood_size)):
        raise RuntimeError("Neighborhood matrix has an invalid shape.")
    if not 0.0 < float(settings.de_scale_factor) <= 2.0:
        raise ValueError("DE scale factor must lie in (0,2].")
    if not 0.0 <= float(settings.de_crossover_rate) <= 1.0:
        raise ValueError("DE crossover rate must lie in [0,1].")
    if float(settings.constraint_penalty_factor) <= 0.0:
        raise ValueError("Constraint penalty factor must be positive.")
