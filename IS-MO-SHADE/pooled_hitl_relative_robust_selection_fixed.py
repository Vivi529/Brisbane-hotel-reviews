# -*- coding: utf-8 -*-
"""
Pool Pareto fronts from repeated M1-I runs and select one relative-robust
solution for HITL strategy generation.

This module is standalone with respect to selection. It expects the saved
Pareto-front CSV files produced by the comparison runner.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Selection configuration
# ---------------------------------------------------------------------------
MIN_ACTION_MAGNITUDE = 0.5
METRIC_EPS = 1e-12
EPS = 1e-12
ROUND_DECIMALS_FOR_UNIQUENESS = 10

ROBUST_EPSILON = 0.05
ROBUST_SCORE_TIE_TOLERANCE = 0.00
RECOMMENDED_COST_TIER = "all"  # "low", "medium", "high", or "all"
REQUIRE_NONNEGATIVE_CHOICE_GAIN = True
CHOICE_GAIN_TOLERANCE = 1e-12
NONTRIVIAL_DELTA_TOLERANCE = 1e-8

OBJECTIVE_CONSISTENCY_RTOL = 1e-9
OBJECTIVE_CONSISTENCY_ATOL = 1e-10


def _natural_es_key(column: str) -> tuple[str, int]:
    """Ensure x_ES_2 is ordered before x_ES_10."""
    match = re.search(r"(\d+)$", str(column))
    if match is None:
        return str(column), -1
    return str(column)[: match.start()], int(match.group(1))


def _required_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="raise")
    values = result[columns].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite values found in columns: {columns}")
    return result


def _validate_duplicate_decision_objectives(
    pooled: pd.DataFrame,
    objective_columns: list[str],
) -> None:
    """Identical decisions should have deterministic objective values."""
    for decision_key, group in pooled.groupby("_decision_key", sort=False):
        if len(group) < 2:
            continue
        values = group[objective_columns].to_numpy(dtype=float)
        reference = values[0]
        if not np.allclose(
            values,
            reference[None, :],
            rtol=OBJECTIVE_CONSISTENCY_RTOL,
            atol=OBJECTIVE_CONSISTENCY_ATOL,
        ):
            raise ValueError(
                "The same rounded decision vector has inconsistent objective "
                f"values across saved fronts. Decision key starts with "
                f"{decision_key[:3]!r}; sources={group['source_file'].tolist()}. "
                "Re-evaluate the solutions with one common evaluator before pooling."
            )


def load_pooled_scenario_front(
    front_directory: str | Path,
    *,
    max_active_actions: int,
    coverage_threshold: float,
    algorithm: str = "M1-I_threshold_MO_SHADE",
    decision_decimals: int = ROUND_DECIMALS_FOR_UNIQUENESS,
) -> pd.DataFrame:
    """Load, validate, and decision-deduplicate all seed fronts."""
    front_directory = Path(front_directory)

    if not front_directory.is_dir():
        raise NotADirectoryError(f"Front directory does not exist: {front_directory}")

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
            raise ValueError(f"Cannot extract the seed from filename: {path.name}")
        source_seed = int(seed_match.group(1))

        if "seed" in frame.columns:
            observed_seed = pd.to_numeric(frame["seed"], errors="raise")
            if not (observed_seed == source_seed).all():
                raise ValueError(f"Inconsistent seed metadata in {path.name}.")

        if "max_active_actions" in frame.columns:
            observed_k = pd.to_numeric(
                frame["max_active_actions"], errors="raise"
            )
            if not (observed_k == int(max_active_actions)).all():
                raise ValueError(f"Inconsistent K in {path.name}.")

        if "coverage_threshold" in frame.columns:
            observed_rho = pd.to_numeric(
                frame["coverage_threshold"], errors="raise"
            )
            if not np.allclose(
                observed_rho,
                float(coverage_threshold),
                atol=EPS,
                rtol=0.0,
            ):
                raise ValueError(
                    f"Inconsistent coverage threshold in {path.name}."
                )

        if "algorithm" in frame.columns:
            if not (frame["algorithm"].astype(str) == algorithm).all():
                raise ValueError(f"Inconsistent algorithm in {path.name}.")

        if "source_solution_id" in frame.columns:
            raise ValueError(
                f"{path.name} already contains source_solution_id; "
                "the file appears to have been pooled previously."
            )
        if "solution_id" in frame.columns:
            frame = frame.rename(
                columns={"solution_id": "source_solution_id"}
            )
        else:
            # Preserve deterministic provenance even if the original export
            # omitted solution_id.
            frame["source_solution_id"] = np.arange(
                1,
                len(frame) + 1,
                dtype=int,
            )

        frame["source_seed"] = source_seed
        frame["source_file"] = path.name
        frames.append(frame)

    pooled = pd.concat(frames, ignore_index=True, sort=False)

    required_columns = {
        "reputation_improvement",
        "choice_probability_gain",
        "effective_cost",
        "sum_delta",
        "constraint_violation",
        "high_priority_coverage",
        "n_active_actions",
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

    numeric_columns = [
        *action_columns,
        "reputation_improvement",
        "choice_probability_gain",
        "effective_cost",
        "sum_delta",
        "constraint_violation",
        "high_priority_coverage",
        "n_active_actions",
    ]
    pooled = _required_numeric(pooled, numeric_columns)

    canonical_x = pooled[action_columns].copy()
    canonical_x = canonical_x.mask(
        np.abs(canonical_x) <= EPS,
        0.0,
    ).round(int(decision_decimals))

    pooled["_decision_key"] = [
        tuple(row)
        for row in canonical_x.to_numpy(dtype=float)
    ]

    _validate_duplicate_decision_objectives(
        pooled,
        [
            "reputation_improvement",
            "choice_probability_gain",
            "effective_cost",
        ],
    )

    provenance = (
        pooled.groupby("_decision_key", sort=False)
        .agg(
            source_seeds=(
                "source_seed",
                lambda values: " | ".join(
                    str(value) for value in sorted(set(values))
                ),
            ),
            n_source_seeds=("source_seed", "nunique"),
            source_files=(
                "source_file",
                lambda values: " | ".join(
                    sorted(set(str(value) for value in values))
                ),
            ),
        )
        .reset_index()
    )

    pooled_unique = (
        pooled.sort_values(
            [
                "constraint_violation",
                "effective_cost",
                "n_active_actions",
                "source_seed",
                "source_solution_id",
            ],
            ascending=[True, True, True, True, True],
            na_position="last",
        )
        .drop_duplicates(subset=["_decision_key"], keep="first")
        .merge(
            provenance,
            on="_decision_key",
            how="left",
            validate="one_to_one",
        )
        .reset_index(drop=True)
    )

    # Recompute the number of active actions from the actual decision vector.
    recomputed_active = (
        pooled_unique[action_columns].to_numpy(dtype=float)
        >= (MIN_ACTION_MAGNITUDE - EPS)
    ).sum(axis=1)
    reported_active = pooled_unique["n_active_actions"].to_numpy(dtype=float)
    if not np.allclose(recomputed_active, reported_active, atol=0.0, rtol=0.0):
        raise ValueError(
            "At least one saved n_active_actions value is inconsistent with "
            f"MIN_ACTION_MAGNITUDE={MIN_ACTION_MAGNITUDE}."
        )

    return pooled_unique


def nondominated_mask_minimization(points: np.ndarray) -> np.ndarray:
    """Return one representative for each nondominated objective vector."""
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


def nondominated_mask_keep_objective_duplicates(
    points: np.ndarray,
) -> np.ndarray:
    """Keep every decision whose objective vector is nondominated.

    Distinct action plans with identical objective values remain available for
    the later parsimony tie-break.
    """
    values = np.asarray(points, dtype=float)
    unique_values, inverse = np.unique(values, axis=0, return_inverse=True)
    unique_keep = nondominated_mask_minimization(unique_values)
    return unique_keep[inverse]


def refilter_pooled_nondominated_front(
    pooled: pd.DataFrame,
) -> pd.DataFrame:
    """Re-establish admissibility and nondominance after pooling seeds."""
    required = {
        "reputation_improvement",
        "choice_probability_gain",
        "effective_cost",
        "sum_delta",
        "constraint_violation",
        "n_active_actions",
        "high_priority_coverage",
    }
    missing = required - set(pooled.columns)
    if missing:
        raise KeyError(f"Pooled data are missing columns: {sorted(missing)}")

    candidates = pooled.loc[
        (pooled["sum_delta"] > NONTRIVIAL_DELTA_TOLERANCE)
        & (pooled["constraint_violation"] <= EPS)
    ].copy()

    if REQUIRE_NONNEGATIVE_CHOICE_GAIN:
        candidates = candidates.loc[
            candidates["choice_probability_gain"]
            >= -CHOICE_GAIN_TOLERANCE
        ].copy()

    if candidates.empty:
        raise ValueError(
            "No feasible nontrivial solution remains after pooling."
        )

    objective_values = np.column_stack(
        [
            -candidates["reputation_improvement"].to_numpy(dtype=float),
            -candidates["choice_probability_gain"].to_numpy(dtype=float),
            candidates["effective_cost"].to_numpy(dtype=float),
        ]
    )

    nondominated = nondominated_mask_keep_objective_duplicates(
        objective_values
    )
    pooled_front = candidates.loc[nondominated].copy()

    pooled_front = pooled_front.sort_values(
        [
            "effective_cost",
            "n_active_actions",
            "reputation_improvement",
            "choice_probability_gain",
            "source_seed",
            "source_solution_id",
        ],
        ascending=[True, True, False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)

    pooled_front["solution_id"] = np.arange(
        1,
        len(pooled_front) + 1,
        dtype=int,
    )
    return pooled_front


def _benefit_normalize(values: pd.Series) -> pd.Series:
    minimum = float(values.min())
    maximum = float(values.max())
    span = maximum - minimum
    if span <= EPS:
        return pd.Series(np.ones(len(values)), index=values.index, dtype=float)
    return (values - minimum) / span


def _cost_satisfaction_normalize(values: pd.Series) -> pd.Series:
    minimum = float(values.min())
    maximum = float(values.max())
    span = maximum - minimum
    if span <= EPS:
        return pd.Series(np.ones(len(values)), index=values.index, dtype=float)
    return (maximum - values) / span


def assign_cost_tiers(frame: pd.DataFrame) -> pd.Series:
    if len(frame) < 3 or frame["effective_cost"].nunique() < 3:
        return pd.Series(["medium"] * len(frame), index=frame.index)
    q1 = float(frame["effective_cost"].quantile(1.0 / 3.0))
    q2 = float(frame["effective_cost"].quantile(2.0 / 3.0))
    tiers = np.where(
        frame["effective_cost"] <= q1,
        "low",
        np.where(frame["effective_cost"] <= q2, "medium", "high"),
    )
    return pd.Series(tiers, index=frame.index)


def calculate_relative_robust_scores(
    pareto: pd.DataFrame,
    epsilon: float = ROBUST_EPSILON,
) -> pd.DataFrame:
    """Calculate relative max-min scores on one common Pareto set."""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("Robust epsilon must lie in [0,1].")

    candidates = pareto.loc[
        (pareto["sum_delta"] > NONTRIVIAL_DELTA_TOLERANCE)
        & (pareto["constraint_violation"] <= EPS)
    ].copy()

    if REQUIRE_NONNEGATIVE_CHOICE_GAIN:
        candidates = candidates.loc[
            candidates["choice_probability_gain"]
            >= -CHOICE_GAIN_TOLERANCE
        ].copy()

    if candidates.empty:
        raise ValueError(
            "No nontrivial feasible Pareto solution satisfies the robust-"
            "selection admissibility rules."
        )

    candidates["phi_reputation"] = _benefit_normalize(
        candidates["reputation_improvement"]
    )
    candidates["phi_choice"] = _benefit_normalize(
        candidates["choice_probability_gain"]
    )
    candidates["phi_cost"] = _cost_satisfaction_normalize(
        candidates["effective_cost"]
    )

    phi_columns = ["phi_reputation", "phi_choice", "phi_cost"]
    candidates["minimum_attainment"] = candidates[phi_columns].min(axis=1)
    candidates["phi_average"] = candidates[phi_columns].mean(axis=1)
    candidates["robust_epsilon"] = float(epsilon)
    candidates["robust_score"] = (
        (1.0 - epsilon) * candidates["minimum_attainment"]
        + epsilon * candidates["phi_average"]
    )
    candidates["cost_tier"] = assign_cost_tiers(candidates)

    return candidates.sort_values(
        [
            "robust_score",
            "effective_cost",
            "n_active_actions",
            "solution_id",
        ],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)


def _choose_near_best_robust(candidates: pd.DataFrame) -> pd.Series:
    """Choose a parsimonious solution within the near-best score band."""
    if candidates.empty:
        raise ValueError("Cannot choose from an empty robust candidate set.")

    best_score = float(candidates["robust_score"].max())
    near_best = candidates.loc[
        candidates["robust_score"]
        >= best_score - ROBUST_SCORE_TIE_TOLERANCE
    ].copy()

    # The tolerance is meaningful only if cost/parsimony precede the tiny score
    # difference inside the admitted near-best band.
    near_best = near_best.sort_values(
        [
            "effective_cost",
            "n_active_actions",
            "robust_score",
            "solution_id",
        ],
        ascending=[True, True, False, True],
    )
    return near_best.iloc[0].copy()


def select_robust_representatives(
    ranked: pd.DataFrame,
    *,
    recommended_cost_tier: str = RECOMMENDED_COST_TIER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    allowed_tiers = {"low", "medium", "high", "all"}
    if recommended_cost_tier not in allowed_tiers:
        raise ValueError(
            f"recommended_cost_tier must be one of {sorted(allowed_tiers)}."
        )

    selected_rows: list[pd.Series] = []
    for tier in ("low", "medium", "high"):
        tier_candidates = ranked.loc[ranked["cost_tier"] == tier].copy()
        if tier_candidates.empty:
            continue
        chosen = _choose_near_best_robust(tier_candidates)
        chosen["selection_scope"] = tier
        selected_rows.append(chosen)

    overall = _choose_near_best_robust(ranked)
    overall["selection_scope"] = "all"
    selected_rows.append(overall)

    representatives = (
        pd.DataFrame(selected_rows)
        .drop_duplicates(subset=["solution_id", "selection_scope"])
        .reset_index(drop=True)
    )

    if recommended_cost_tier == "all":
        recommended = representatives.loc[
            representatives["selection_scope"] == "all"
        ].copy()
    else:
        recommended = representatives.loc[
            representatives["selection_scope"] == recommended_cost_tier
        ].copy()
        if recommended.empty:
            recommended = representatives.loc[
                representatives["selection_scope"] == "all"
            ].copy()

    return representatives, recommended.reset_index(drop=True)


def select_hitl_solution_from_scenario(
    front_directory: str | Path,
    *,
    max_active_actions: int,
    coverage_threshold: float,
    algorithm: str = "M1-I_threshold_MO_SHADE",
    epsilon: float = ROBUST_EPSILON,
    recommended_cost_tier: str = RECOMMENDED_COST_TIER,
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

    pooled_pareto = refilter_pooled_nondominated_front(pooled_unique)

    ranked = calculate_relative_robust_scores(
        pooled_pareto,
        epsilon=float(epsilon),
    )

    representatives, recommended = select_robust_representatives(
        ranked,
        recommended_cost_tier=recommended_cost_tier,
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

    active_mask = np.asarray(
        [
            float(selected[column]) >= MIN_ACTION_MAGNITUDE - EPS
            for column in action_columns
        ],
        dtype=bool,
    )
    recomputed_n_active = int(active_mask.sum())

    active_plan = pd.DataFrame(
        [
            {
                "service_element": column.removeprefix("x_"),
                "improvement_magnitude": float(selected[column]),
            }
            for column, active in zip(action_columns, active_mask)
            if active
        ]
    )

    if recomputed_n_active > int(max_active_actions):
        raise RuntimeError(
            "The selected HITL solution violates the action cap."
        )
    if recomputed_n_active != int(selected["n_active_actions"]):
        raise RuntimeError(
            "The selected solution's n_active_actions is inconsistent with "
            "its decision vector."
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
    if REQUIRE_NONNEGATIVE_CHOICE_GAIN and float(
        selected["choice_probability_gain"]
    ) < -CHOICE_GAIN_TOLERANCE:
        raise RuntimeError(
            "The selected HITL solution violates the nonnegative-choice rule."
        )

    return (
        pooled_unique,
        pooled_pareto,
        representatives,
        selected,
        active_plan,
    )


def main() -> None:
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
        recommended_cost_tier="all",
    )

    output_dir = front_dir.parent / "pooled_HITL_selection"
    output_dir.mkdir(parents=True, exist_ok=True)

    pooled_unique.to_csv(
        output_dir / "K15_rho0p50_pooled_unique.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pooled_pareto.to_csv(
        output_dir / "K15_rho0p50_pooled_nondominated.csv",
        index=False,
        encoding="utf-8-sig",
    )
    robust_representatives.to_csv(
        output_dir / "K15_rho0p50_robust_representatives.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([hitl_solution]).to_csv(
        output_dir / "K15_rho0p50_HITL_solution.csv",
        index=False,
        encoding="utf-8-sig",
    )
    hitl_action_plan.to_csv(
        output_dir / "K15_rho0p50_HITL_action_plan.csv",
        index=False,
        encoding="utf-8-sig",
    )

    display_columns = [
        "solution_id",
        "reputation_improvement",
        "choice_probability_gain_pp",
        "effective_cost",
        "high_priority_coverage",
        "n_active_actions",
        "minimum_attainment",
        "phi_average",
        "robust_score",
        "source_seed",
        "source_solution_id",
    ]
    available = [
        column for column in display_columns
        if column in hitl_solution.index
    ]
    print(hitl_solution[available].to_string())
    print("\nActive HITL plan:")
    print(hitl_action_plan.to_string(index=False))
    print(f"\nOutputs written to: {output_dir}")


#if __name__ == "__main__":
#    main()

front_dir = Path(
    r"D:\AAApaper\online_review\复现\five_algorithm_checkpoint\comparison_five_algorithms_formal_K15_V4"
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
    recommended_cost_tier="all",
)

output_dir = front_dir.parent / "pooled_HITL_selection"
output_dir.mkdir(parents=True, exist_ok=True)


pd.DataFrame([hitl_solution]).to_csv(
    output_dir / "K15_rho0p50_HITL_solution.csv",
    index=False,
    encoding="utf-8-sig",
)
hitl_action_plan.to_csv(
    output_dir / "K15_rho0p50_HITL_action_plan.csv",
    index=False,
    encoding="utf-8-sig",
)

display_columns = [
    "solution_id",
    "reputation_improvement",
    "choice_probability_gain_pp",
    "effective_cost",
    "high_priority_coverage",
    "n_active_actions",
    "minimum_attainment",
    "phi_average",
    "robust_score",
    "source_seed",
    "source_solution_id",
]
available = [
    column for column in display_columns
    if column in hitl_solution.index
]
print(hitl_solution[available].to_string())
print("\nActive HITL plan:")
print(hitl_action_plan.to_string(index=False))
print(f"\nOutputs written to: {output_dir}")