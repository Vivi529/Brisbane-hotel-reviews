# -*- coding: utf-8 -*-
"""Independent M1-I and M1-C extensions for IPEA-guided MO-SHADE.

This module patches an already loaded
``MOO_MO_SHADE_IPEA_priority_initialization_fixed.py`` module in memory.
It deliberately keeps the two mechanisms independent:

M1-I
    Threshold-aware stratified initialization only. The initial population is
    sampled from three layers: coverage-boundary, ordinary IPEA-priority, and
    uniform exploration. The crossover operator remains standard MO-SHADE.

M1-C
    The original M1 priority initialization is retained. Only the binomial
    crossover probabilities become priority-dependent and decay to the
    ordinary SHADE crossover probability over generations.

Neither extension changes the three objectives, constraint violation,
constraint dominance, repair, environmental selection, SHADE memory update,
or representative-solution selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np


# =============================================================================
# Extension parameters
# =============================================================================
# M1-I population mixture.
M1I_BOUNDARY_LAYER_SHARE = 0.40
M1I_PRIORITY_LAYER_SHARE = 0.30
M1I_UNIFORM_LAYER_SHARE = 0.30
M1I_BOUNDARY_BANDWIDTH = 0.05
M1I_BOUNDARY_MAX_ATTEMPTS = 100

# M1-C decayed priority crossover.
M1C_LAMBDA_0 = 0.40
M1C_DECAY_KAPPA = 2.0
M1C_CR_MIN = 0.05
M1C_CR_MAX = 0.95

SUPPORTED_VARIANTS = {
    "M0_uniform_initialization",
    "M1_IPEA_priority_initialization",
    "M1-I_threshold_stratified_initialization",
    "M1-C_decayed_priority_crossover",
}


@dataclass(frozen=True)
class CorePatchState:
    create_priority_guided_vector: Callable[..., np.ndarray]
    generate_trials: Callable[..., list[Any]]


def capture_core_patch_state(core: Any) -> CorePatchState:
    """Capture the two functions that the extensions are allowed to replace."""
    return CorePatchState(
        create_priority_guided_vector=core.create_priority_guided_vector,
        generate_trials=core.generate_trials,
    )


def restore_core_patch_state(core: Any, state: CorePatchState) -> None:
    core.create_priority_guided_vector = state.create_priority_guided_vector
    core.generate_trials = state.generate_trials


def validate_extension_parameters() -> None:
    shares = np.asarray(
        [
            M1I_BOUNDARY_LAYER_SHARE,
            M1I_PRIORITY_LAYER_SHARE,
            M1I_UNIFORM_LAYER_SHARE,
        ],
        dtype=float,
    )
    if np.any(shares < 0.0) or not np.isclose(shares.sum(), 1.0, atol=1e-12):
        raise ValueError("M1-I layer shares must be nonnegative and sum to 1.")
    if not 0.0 < M1I_BOUNDARY_BANDWIDTH < 1.0:
        raise ValueError("M1I_BOUNDARY_BANDWIDTH must lie in (0,1).")
    if M1I_BOUNDARY_MAX_ATTEMPTS < 1:
        raise ValueError("M1I_BOUNDARY_MAX_ATTEMPTS must be positive.")
    if not 0.0 <= M1C_LAMBDA_0 <= 1.0:
        raise ValueError("M1C_LAMBDA_0 must lie in [0,1].")
    if M1C_DECAY_KAPPA <= 0.0:
        raise ValueError("M1C_DECAY_KAPPA must be positive.")
    if not 0.0 <= M1C_CR_MIN < M1C_CR_MAX <= 1.0:
        raise ValueError("M1-C crossover bounds are invalid.")


def _new_diagnostics(variant_name: str) -> dict[str, Any]:
    return {
        "variant": variant_name,
        "initial_vectors": 0,
        "initial_zero_vectors": 0,
        "initial_boundary_vectors": 0,
        "initial_priority_vectors": 0,
        "initial_uniform_vectors": 0,
        "boundary_fallbacks": 0,
        "boundary_abs_error_sum": 0.0,
        "boundary_abs_error_count": 0,
        "crossover_calls": 0,
        "lambda_sum": 0.0,
        "lambda_first": np.nan,
        "lambda_last": np.nan,
        "priority_cr_probability_sum": {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
        "priority_cr_probability_count": {1: 0, 2: 0, 3: 0, 4: 0},
    }


def reset_variant_diagnostics(core: Any, variant_name: str) -> None:
    core._ipea_extension_diagnostics = _new_diagnostics(variant_name)


def get_variant_diagnostics(core: Any) -> dict[str, Any]:
    diagnostics = getattr(core, "_ipea_extension_diagnostics", None)
    if diagnostics is None:
        return {}

    output = {
        key: value
        for key, value in diagnostics.items()
        if key not in {
            "priority_cr_probability_sum",
            "priority_cr_probability_count",
        }
    }
    count = int(diagnostics.get("boundary_abs_error_count", 0))
    output["boundary_mean_abs_coverage_error"] = (
        float(diagnostics["boundary_abs_error_sum"] / count)
        if count > 0
        else np.nan
    )
    calls = int(diagnostics.get("crossover_calls", 0))
    output["mean_crossover_lambda"] = (
        float(diagnostics["lambda_sum"] / calls) if calls > 0 else np.nan
    )
    for priority in (1, 2, 3, 4):
        p_count = int(diagnostics["priority_cr_probability_count"][priority])
        output[f"mean_crossover_probability_priority_{priority}"] = (
            float(diagnostics["priority_cr_probability_sum"][priority] / p_count)
            if p_count > 0
            else np.nan
        )
    return output


# =============================================================================
# Shared initialization helpers
# =============================================================================
def _eligible_indices(core: Any, options: Sequence[Any]) -> np.ndarray:
    eligible = np.asarray(
        [
            option.max_delta >= core.MIN_ACTION_MAGNITUDE - core.EPS
            for option in options
        ],
        dtype=bool,
    )
    indices = np.flatnonzero(eligible)
    if indices.size == 0:
        raise ValueError("No service element can satisfy the minimum action size.")
    return indices


def _sample_active_count(
    core: Any,
    options: Sequence[Any],
    rng: np.random.Generator,
) -> int:
    n = len(options)
    low_fraction, high_fraction = core.INITIAL_ACTIVE_FRACTION_RANGE
    minimum = max(1, int(math.ceil(float(low_fraction) * n)))
    maximum = max(minimum, int(math.ceil(float(high_fraction) * n)))
    maximum = min(maximum, n)

    if core.MAX_ACTIVE_ACTIONS is not None:
        cap = int(core.MAX_ACTIVE_ACTIONS)
        if cap < 1:
            raise ValueError("MAX_ACTIVE_ACTIONS must be positive or None.")
        maximum = min(maximum, cap)
        minimum = min(minimum, maximum)

    if minimum < 1 or maximum < minimum:
        raise RuntimeError(
            f"Invalid active-count interval: minimum={minimum}, maximum={maximum}."
        )
    return int(rng.integers(minimum, maximum + 1))


def _probabilities_for_mode(
    core: Any,
    options: Sequence[Any],
    mode: str,
) -> np.ndarray:
    eligible = _eligible_indices(core, options)
    probabilities = np.zeros(len(options), dtype=float)

    if mode == "uniform":
        probabilities[eligible] = 1.0 / eligible.size
        return probabilities

    if mode != "priority":
        raise ValueError(f"Unsupported initialization probability mode: {mode!r}")

    # Reproduce M1 exactly without permanently changing the global rate.
    probabilities = np.asarray(
        core.priority_activation_probabilities(options), dtype=float
    )
    if probabilities.shape != (len(options),):
        raise RuntimeError("priority_activation_probabilities returned wrong shape.")
    return probabilities / probabilities.sum()


def _create_probability_vector(
    core: Any,
    options: Sequence[Any],
    rng: np.random.Generator,
    probabilities: np.ndarray,
) -> np.ndarray:
    n_active = _sample_active_count(core, options, rng)
    selected = rng.choice(
        len(options),
        size=n_active,
        replace=False,
        p=np.asarray(probabilities, dtype=float),
    )
    x = np.zeros(len(options), dtype=float)
    for raw_index in selected:
        index = int(raw_index)
        option = options[index]
        x[index] = float(
            rng.uniform(core.MIN_ACTION_MAGNITUDE, option.max_delta)
        )
    return core.repair_vector(x, options)


def _allocate_group_spend(
    lower: np.ndarray,
    upper: np.ndarray,
    target: float,
    rng: np.random.Generator,
    eps: float,
) -> np.ndarray:
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != upper.shape or lower.ndim != 1:
        raise ValueError("Spend bounds must be one-dimensional and aligned.")
    minimum = float(lower.sum())
    maximum = float(upper.sum())
    if target < minimum - eps or target > maximum + eps:
        raise ValueError(
            f"Target spend {target} is outside [{minimum}, {maximum}]."
        )

    spend = lower.copy()
    remaining = max(0.0, float(target - minimum))
    capacity = np.maximum(upper - spend, 0.0)

    for _ in range(max(20, 5 * len(spend))):
        active = np.flatnonzero(capacity > eps)
        if remaining <= eps or active.size == 0:
            break
        shares = rng.dirichlet(np.ones(active.size, dtype=float))
        proposed = remaining * shares
        actual = np.minimum(proposed, capacity[active])
        used = float(actual.sum())
        spend[active] += actual
        capacity[active] -= actual
        remaining -= used
        if used <= eps:
            break

    # Deterministic numerical completion.
    if remaining > eps:
        for index in np.argsort(-capacity):
            if remaining <= eps:
                break
            addition = min(float(capacity[index]), remaining)
            spend[index] += addition
            remaining -= addition

    if remaining > max(1e-9, 100.0 * eps):
        raise RuntimeError(
            f"Failed to allocate group spend; unallocated={remaining}."
        )
    return np.clip(spend, lower, upper)


def _sample_group_indices(
    core: Any,
    options: Sequence[Any],
    candidates: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if size < 1 or size > len(candidates):
        raise ValueError("Invalid group sample size.")
    guidance = np.asarray(
        [core.PRIORITY_GUIDANCE[options[int(i)].priority] for i in candidates],
        dtype=float,
    )
    guidance = np.maximum(guidance, 0.0)
    probabilities = (
        guidance / guidance.sum()
        if guidance.sum() > core.EPS
        else np.full(len(candidates), 1.0 / len(candidates))
    )
    return np.asarray(
        rng.choice(candidates, size=size, replace=False, p=probabilities),
        dtype=int,
    )


def _create_boundary_vector(
    core: Any,
    options: Sequence[Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, bool]:
    """Create a nonzero plan with cost-weighted coverage near rho.

    Returns
    -------
    vector, absolute_error, used_fallback
    """
    eligible = _eligible_indices(core, options)
    high = np.asarray(
        [
            index
            for index in eligible
            if options[int(index)].priority in core.HIGH_PRIORITY_LEVELS
        ],
        dtype=int,
    )
    low = np.asarray(
        [
            index
            for index in eligible
            if options[int(index)].priority not in core.HIGH_PRIORITY_LEVELS
        ],
        dtype=int,
    )
    if high.size == 0 or low.size == 0:
        probabilities = _probabilities_for_mode(core, options, "priority")
        fallback = _create_probability_vector(core, options, rng, probabilities)
        error = abs(
            core.high_priority_resource_coverage(fallback, options)
            - float(core.HIGH_PRIORITY_COVERAGE_MIN)
        )
        return fallback, float(error), True

    rho = float(core.HIGH_PRIORITY_COVERAGE_MIN)
    upper_target = min(rho + M1I_BOUNDARY_BANDWIDTH, 0.98)
    if upper_target < rho:
        upper_target = rho

    best_vector: np.ndarray | None = None
    best_error = float("inf")

    for _ in range(M1I_BOUNDARY_MAX_ATTEMPTS):
        n_active = max(2, _sample_active_count(core, options, rng))
        n_active = min(n_active, int(high.size + low.size))

        target = float(rng.uniform(rho, upper_target))
        expected_high = int(round(target * n_active))
        jitter = int(rng.integers(-1, 2))
        n_high = int(np.clip(expected_high + jitter, 1, n_active - 1))
        n_high = min(n_high, int(high.size))
        n_low = n_active - n_high
        if n_low > low.size:
            n_low = int(low.size)
            n_high = n_active - n_low
        if n_high < 1 or n_low < 1 or n_high > high.size or n_low > low.size:
            continue

        high_selected = _sample_group_indices(
            core, options, high, n_high, rng
        )
        low_selected = _sample_group_indices(core, options, low, n_low, rng)

        high_cost = np.asarray(
            [options[int(i)].effective_cost for i in high_selected], dtype=float
        )
        low_cost = np.asarray(
            [options[int(i)].effective_cost for i in low_selected], dtype=float
        )
        high_lower = high_cost * float(core.MIN_ACTION_MAGNITUDE)
        low_lower = low_cost * float(core.MIN_ACTION_MAGNITUDE)
        high_upper = high_cost * np.asarray(
            [options[int(i)].max_delta for i in high_selected], dtype=float
        )
        low_upper = low_cost * np.asarray(
            [options[int(i)].max_delta for i in low_selected], dtype=float
        )

        total_lower = max(
            float(high_lower.sum()) / max(target, core.EPS),
            float(low_lower.sum()) / max(1.0 - target, core.EPS),
        )
        total_upper = min(
            float(high_upper.sum()) / max(target, core.EPS),
            float(low_upper.sum()) / max(1.0 - target, core.EPS),
        )
        if total_lower > total_upper + core.EPS:
            continue

        total_spend = float(rng.uniform(total_lower, total_upper))
        high_target_spend = target * total_spend
        low_target_spend = (1.0 - target) * total_spend

        try:
            high_spend = _allocate_group_spend(
                high_lower, high_upper, high_target_spend, rng, core.EPS
            )
            low_spend = _allocate_group_spend(
                low_lower, low_upper, low_target_spend, rng, core.EPS
            )
        except (ValueError, RuntimeError):
            continue

        vector = np.zeros(len(options), dtype=float)
        vector[high_selected] = high_spend / high_cost
        vector[low_selected] = low_spend / low_cost
        vector = core.repair_vector(vector, options)

        coverage = float(core.high_priority_resource_coverage(vector, options))
        distance_to_band = (
            rho - coverage
            if coverage < rho
            else coverage - upper_target
            if coverage > upper_target
            else 0.0
        )
        center = 0.5 * (rho + upper_target)
        error = abs(coverage - center)
        if distance_to_band <= core.EPS:
            return vector, float(error), False
        if distance_to_band < best_error:
            best_error = float(distance_to_band)
            best_vector = vector.copy()

    if best_vector is not None:
        return best_vector, float(best_error), True

    probabilities = _probabilities_for_mode(core, options, "priority")
    fallback = _create_probability_vector(core, options, rng, probabilities)
    error = abs(
        core.high_priority_resource_coverage(fallback, options)
        - float(core.HIGH_PRIORITY_COVERAGE_MIN)
    )
    return fallback, float(error), True


def make_m1i_initializer(core: Any) -> Callable[..., np.ndarray]:
    shares = np.asarray(
        [
            M1I_BOUNDARY_LAYER_SHARE,
            M1I_PRIORITY_LAYER_SHARE,
            M1I_UNIFORM_LAYER_SHARE,
        ],
        dtype=float,
    )

    def create_threshold_stratified_vector(
        options: Sequence[Any],
        rng: np.random.Generator,
    ) -> np.ndarray:
        diagnostics = core._ipea_extension_diagnostics
        diagnostics["initial_vectors"] += 1

        if rng.random() < float(core.INITIAL_ZERO_SOLUTION_RATE):
            diagnostics["initial_zero_vectors"] += 1
            return np.zeros(len(options), dtype=float)

        layer = int(rng.choice(3, p=shares))
        if layer == 0:
            diagnostics["initial_boundary_vectors"] += 1
            vector, error, fallback = _create_boundary_vector(core, options, rng)
            diagnostics["boundary_abs_error_sum"] += float(error)
            diagnostics["boundary_abs_error_count"] += 1
            diagnostics["boundary_fallbacks"] += int(fallback)
            return vector

        if layer == 1:
            diagnostics["initial_priority_vectors"] += 1
            probabilities = _probabilities_for_mode(core, options, "priority")
            return _create_probability_vector(core, options, rng, probabilities)

        diagnostics["initial_uniform_vectors"] += 1
        probabilities = _probabilities_for_mode(core, options, "uniform")
        return _create_probability_vector(core, options, rng, probabilities)

    return create_threshold_stratified_vector


# =============================================================================
# M1-C decayed, priority-dependent crossover
# =============================================================================
def _decay_strength(generation: int, total_generations: int) -> float:
    if total_generations <= 1:
        fraction = 1.0
    else:
        fraction = (generation - 1) / (total_generations - 1)
    remaining = max(0.0, 1.0 - fraction)
    return float(M1C_LAMBDA_0 * remaining**M1C_DECAY_KAPPA)


def _priority_crossover_probabilities(
    core: Any,
    options: Sequence[Any],
    base_cr: float,
    strength: float,
) -> np.ndarray:
    guidance = np.asarray(
        [core.PRIORITY_GUIDANCE[option.priority] for option in options],
        dtype=float,
    )
    minimum = float(guidance.min())
    maximum = float(guidance.max())
    if maximum - minimum <= core.EPS:
        normalized = np.full(len(options), 0.5, dtype=float)
    else:
        normalized = (guidance - minimum) / (maximum - minimum)

    # Centering preserves the average crossover pressure approximately; only
    # its allocation across service-element dimensions changes.
    centered = normalized - float(normalized.mean())
    probabilities = float(base_cr) + float(strength) * centered
    return np.clip(probabilities, M1C_CR_MIN, M1C_CR_MAX)


def make_m1c_generate_trials(core: Any) -> Callable[..., list[Any]]:
    call_state = {"generation": 0}

    def generate_trials(
        population: Sequence[Any],
        mutation_archive: Sequence[np.ndarray],
        memory_f: np.ndarray,
        memory_cr: np.ndarray,
        options: Sequence[Any],
        context: Any,
        rng: np.random.Generator,
    ) -> list[Any]:
        call_state["generation"] += 1
        generation = int(call_state["generation"])
        strength = _decay_strength(generation, int(core.N_GENERATIONS))

        diagnostics = core._ipea_extension_diagnostics
        diagnostics["crossover_calls"] += 1
        diagnostics["lambda_sum"] += float(strength)
        if diagnostics["crossover_calls"] == 1:
            diagnostics["lambda_first"] = float(strength)
        diagnostics["lambda_last"] = float(strength)

        n = len(population)
        if n < 4:
            raise ValueError("MO-SHADE requires at least four population members.")

        ranks, crowding = core.rank_and_crowding(population)
        ordered_indices = sorted(
            range(n),
            key=lambda idx: (
                ranks[idx],
                -crowding[idx],
                population[idx].violation,
            ),
        )
        top_count = max(2, int(math.ceil(core.P_BEST_RATE * n)))
        pbest_indices = ordered_indices[:top_count]

        union_vectors = [individual.x for individual in population] + [
            np.asarray(vector, dtype=float) for vector in mutation_archive
        ]

        trials: list[Any] = []
        for target_index, target in enumerate(population):
            memory_index = int(rng.integers(0, len(memory_f)))
            F = core.sample_positive_cauchy(
                float(memory_f[memory_index]), core.F_SCALE, rng
            )
            CR = float(
                np.clip(
                    rng.normal(float(memory_cr[memory_index]), core.CR_STD),
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
            valid_r2 = [
                idx
                for idx in range(len(union_vectors))
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
            mutant = core.repair_vector(mutant, options)

            crossover_probabilities = _priority_crossover_probabilities(
                core, options, CR, strength
            )
            for priority in (1, 2, 3, 4):
                mask = np.asarray(
                    [option.priority == priority for option in options],
                    dtype=bool,
                )
                if np.any(mask):
                    diagnostics["priority_cr_probability_sum"][priority] += float(
                        crossover_probabilities[mask].mean()
                    )
                    diagnostics["priority_cr_probability_count"][priority] += 1

            trial_values = target.x.copy()
            j_rand = int(rng.integers(0, len(options)))
            crossover_mask = rng.random(len(options)) < crossover_probabilities
            # Keep the standard DE guarantee of at least one mutant dimension.
            crossover_mask[j_rand] = True
            trial_values[crossover_mask] = mutant[crossover_mask]
            trial_values = core.repair_vector(trial_values, options)

            trials.append(
                core.make_individual(
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

    return generate_trials


# =============================================================================
# Public installation API
# =============================================================================
def install_variant(
    core: Any,
    state: CorePatchState,
    variant_name: str,
) -> None:
    """Install exactly one search variant into the loaded core module."""
    validate_extension_parameters()
    if variant_name not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"Unsupported variant {variant_name!r}; "
            f"supported={sorted(SUPPORTED_VARIANTS)}."
        )

    restore_core_patch_state(core, state)
    reset_variant_diagnostics(core, variant_name)

    if variant_name == "M1-I_threshold_stratified_initialization":
        core.create_priority_guided_vector = make_m1i_initializer(core)
    elif variant_name == "M1-C_decayed_priority_crossover":
        core.generate_trials = make_m1c_generate_trials(core)
