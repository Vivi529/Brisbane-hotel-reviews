# -*- coding: utf-8 -*-
"""
Complete LightGBM + Optuna + out-of-fold SHAP severity workflow
===============================================================

Purpose
-------
Estimate the RPN severity of each service element from LightGBM SHAP values.
The formal severity measure is the repeated-cross-fitted mean absolute SHAP
value among reviews in which the corresponding service element is observed:

    s_m = E(|phi_im^OOF| | ES_m is observed)

The final severity score is min-max normalized to [1, 10]. Occurrence is not
used in this script and should be calculated separately at the review level.

Main safeguards
---------------
1. Hyperparameters are tuned only on the training portion of a fixed holdout
   split; the holdout portion is used for unbiased model evaluation.
2. Severity is computed from out-of-fold SHAP values, not in-sample SHAP.
3. SHAP values are averaged only over observations where the service element
   is non-missing, reducing overlap between Severity and Occurrence.
4. Missing-indicator SHAP analysis is diagnostic only and is never merged into
   the formal severity score.
5. Stability is assessed by repeatedly refitting all cross-validation models
   under different fold seeds.
6. Optuna-level and LightGBM-level parallelism are not allowed to oversubscribe
   CPU threads.

Expected input
--------------
An Excel sheet containing:
    ES_1, ES_2, ..., ES_122
    total score

Outputs
-------
- SHAP_severity_results.xlsx
- SHAP_severity.csv
- best_params.json
- optuna_trials.csv
- repeated_severity_vectors.csv
- oof_shap_seed_<seed>.npz (optional)
- figures/*.png

Run
---
Edit the CONFIGURATION section or use command-line arguments, for example:

python LightGBM_SHAP_Severity_complete_fixed.py \
    --input "D:/AAApaper/online_review/S3-AOP_sent.xlsx" \
    --sheet Sheet1 \
    --output-dir "D:/AAApaper/online_review/shap_severity_results"

For a short code/data check:

python LightGBM_SHAP_Severity_complete_fixed.py --quick-test
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import random
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:  # compatibility with older scikit-learn
    root_mean_squared_error = None



# =============================================================================
# 1. CONFIGURATION
# =============================================================================
INPUT_FILE = Path("data/processed/review_service_matrix.xlsx")
INPUT_SHEET = "Sheet1"
OUTPUT_DIR = Path("outputs/shap_severity")

FEATURE_PREFIX = "ES_"
N_SERVICE_ELEMENTS = 122
TARGET_COLUMN = "total score"

# Data validation
MIN_NONMISSING_PER_FEATURE = 5
DROP_ROWS_WITH_MISSING_TARGET = True
TARGET_MIN: float | None = 1.0
TARGET_MAX: float | None = 10.0

# Fixed holdout evaluation
HOLDOUT_SIZE = 0.20
HOLDOUT_RANDOM_STATE = 42

# Optuna tuning
N_TRIALS = 200
OPTUNA_CV_SPLITS = 5
OPTUNA_RANDOM_STATE = 42
OPTUNA_N_JOBS = 1
OPTUNA_STARTUP_TRIALS = 20
OPTUNA_WARMUP_STEPS = 1
OPTUNA_TIMEOUT_SECONDS: int | None = None

# LightGBM
NUM_BOOST_ROUND = 5000
EARLY_STOPPING_ROUNDS = 50
# When OPTUNA_N_JOBS > 1, each model is forced to one thread.
LGB_NUM_THREADS = min(8, max(1, (os.cpu_count() or 1) - 1))

# Repeated OOF SHAP stability
OOF_CV_SPLITS = 5
STABILITY_SEEDS = (42, 43, 44, 45, 46)
SAVE_OOF_SHAP = True
CHECK_SHAP_ADDITIVITY = False

# Severity definition and stability summaries
SEVERITY_SCALE_MIN = 1.0
SEVERITY_SCALE_MAX = 10.0
TOP_K = 20
SEVERITY_WINSORIZE = False
SEVERITY_WINSOR_LOWER = 0.05
SEVERITY_WINSOR_UPPER = 0.95

# Optional diagnostic: explicit missing indicators
RUN_MISSINGNESS_DIAGNOSTIC = True
MISSINGNESS_DIAGNOSTIC_MAX_ROWS = 2000

# Plots
SAVE_PLOTS = True
SHOW_PLOTS = False
PLOT_TOP_N = 20

# Reproducibility and numerics
GLOBAL_SEED = 42
EPS = 1e-12


# =============================================================================
# 2. DATA STRUCTURES
# =============================================================================
@dataclass(frozen=True)
class RunConfig:
    input_file: str
    input_sheet: str
    output_dir: str
    n_features: int
    target_column: str
    holdout_size: float
    n_trials: int
    optuna_cv_splits: int
    optuna_n_jobs: int
    num_boost_round: int
    early_stopping_rounds: int
    oof_cv_splits: int
    stability_seeds: tuple[int, ...]
    top_k: int
    run_missingness_diagnostic: bool
    severity_winsorize: bool


@dataclass
class OOFRunResult:
    seed: int
    severity_raw: pd.Series
    severity_negative_raw: pd.Series
    n_observed: pd.Series
    fold_metrics: pd.DataFrame
    oof_predictions: np.ndarray
    oof_shap: np.ndarray | None


# =============================================================================
# 3. GENERAL HELPERS
# =============================================================================
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def rmse_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if root_mean_squared_error is not None:
        return float(root_mean_squared_error(y_true, y_pred))
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def safe_json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def normalize_to_scale(
    values: pd.Series,
    lower: float = 1.0,
    upper: float = 10.0,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    if numeric.isna().any() or not np.all(np.isfinite(numeric.to_numpy())):
        raise ValueError("Cannot normalize non-finite severity values.")

    minimum = float(numeric.min())
    maximum = float(numeric.max())
    span = maximum - minimum

    if span <= EPS:
        # No relative distinction is available. Assign the minimum risk level.
        return pd.Series(lower, index=numeric.index, dtype=float)

    return lower + (upper - lower) * (numeric - minimum) / span


def winsorize_series(values: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
    if not 0.0 <= lower_q < upper_q <= 1.0:
        raise ValueError("Winsor quantiles must satisfy 0 <= lower < upper <= 1.")
    lower = float(values.quantile(lower_q))
    upper = float(values.quantile(upper_q))
    return values.clip(lower=lower, upper=upper)


def topk_jaccard(a: pd.Series, b: pd.Series, k: int) -> float:
    k_eff = min(k, len(a), len(b))
    top_a = set(a.nlargest(k_eff).index)
    top_b = set(b.nlargest(k_eff).index)
    union = top_a | top_b
    return float(len(top_a & top_b) / len(union)) if union else 1.0


def ensure_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, figure_dir


# =============================================================================
# 4. INPUT LOADING AND VALIDATION
# =============================================================================
def expected_feature_names(n_features: int) -> list[str]:
    return [f"{FEATURE_PREFIX}{i}" for i in range(1, n_features + 1)]


def load_and_validate_data(
    input_file: Path,
    sheet_name: str,
    feature_names: Sequence[str],
    target_column: str,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    if not input_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    raw = pd.read_excel(input_file, sheet_name=sheet_name)
    if raw.empty:
        raise ValueError("The input worksheet is empty.")

    required = [*feature_names, target_column]
    missing_columns = [column for column in required if column not in raw.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    data = raw[required].copy()
    for feature in feature_names:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
    data[target_column] = pd.to_numeric(data[target_column], errors="coerce")

    missing_target = data[target_column].isna()
    if missing_target.any():
        if DROP_ROWS_WITH_MISSING_TARGET:
            data = data.loc[~missing_target].copy()
        else:
            raise ValueError(
                f"Target contains {int(missing_target.sum())} missing/non-numeric rows."
            )

    if data.empty:
        raise ValueError("No rows remain after target validation.")

    y = data[target_column].to_numpy(dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("Target contains NaN or infinite values.")

    if TARGET_MIN is not None and np.any(y < TARGET_MIN - EPS):
        warnings.warn(
            f"Some target values are below TARGET_MIN={TARGET_MIN}.",
            RuntimeWarning,
        )
    if TARGET_MAX is not None and np.any(y > TARGET_MAX + EPS):
        warnings.warn(
            f"Some target values are above TARGET_MAX={TARGET_MAX}.",
            RuntimeWarning,
        )

    X = data[list(feature_names)].astype(np.float32)
    nonmissing_counts = X.notna().sum(axis=0)
    all_missing = nonmissing_counts[nonmissing_counts == 0]
    if not all_missing.empty:
        raise ValueError(
            "The following service-element columns are completely missing: "
            f"{all_missing.index.tolist()}"
        )

    sparse_features = nonmissing_counts[
        nonmissing_counts < MIN_NONMISSING_PER_FEATURE
    ]
    if not sparse_features.empty:
        warnings.warn(
            "Some features have fewer than "
            f"{MIN_NONMISSING_PER_FEATURE} observed values: "
            f"{sparse_features.to_dict()}",
            RuntimeWarning,
        )

    diagnostics = pd.DataFrame(
        {
            "ES": feature_names,
            "n_rows": len(X),
            "n_observed": [int(nonmissing_counts[name]) for name in feature_names],
            "observed_fraction": [
                float(nonmissing_counts[name] / len(X)) for name in feature_names
            ],
            "mean_when_observed": [float(X[name].mean(skipna=True)) for name in feature_names],
            "std_when_observed": [float(X[name].std(skipna=True)) for name in feature_names],
            "min_when_observed": [float(X[name].min(skipna=True)) for name in feature_names],
            "max_when_observed": [float(X[name].max(skipna=True)) for name in feature_names],
        }
    )

    return X.reset_index(drop=True), y, diagnostics


# =============================================================================
# 5. LIGHTGBM PARAMETER HANDLING
# =============================================================================
def base_lgb_params(seed: int, num_threads: int) -> dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "num_threads": int(num_threads),
        "use_missing": True,
        "zero_as_missing": False,
        "deterministic": True,
        "force_col_wise": True,
        "seed": int(seed),
        "bagging_seed": int(seed),
        "feature_fraction_seed": int(seed),
        "extra_seed": int(seed),
        "data_random_seed": int(seed),
    }


def complete_lgb_params(
    tuned_params: dict[str, Any],
    *,
    seed: int,
    num_threads: int,
) -> dict[str, Any]:
    params = {**base_lgb_params(seed, num_threads), **tuned_params}
    if float(params.get("bagging_fraction", 1.0)) < 1.0:
        params["bagging_freq"] = max(1, int(params.get("bagging_freq", 1)))
    else:
        params["bagging_freq"] = 0
    return params


# =============================================================================
# 6. OPTUNA HYPERPARAMETER TUNING
# =============================================================================
def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    n_trials: int,
    n_splits: int,
    random_state: int,
    n_jobs: int,
    timeout_seconds: int | None,
    log_file: Path,
) -> tuple[dict[str, Any], optuna.Study, pd.DataFrame]:
    if n_splits < 2:
        raise ValueError("Optuna CV requires at least two folds.")

    # Avoid nested thread oversubscription.
    model_threads = 1 if n_jobs > 1 else LGB_NUM_THREADS
    logger = logging.getLogger("optuna_objective")
    logger.setLevel(logging.ERROR)
    logger.handlers.clear()
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.ERROR)
    logger.addHandler(file_handler)

    def objective(trial: optuna.Trial) -> float:
        max_depth = trial.suggest_int("max_depth", 4, 8)
        suggested = {
            "max_depth": max_depth,
            "num_leaves": trial.suggest_int(
                "num_leaves",
                2 ** (max_depth - 1),
                2**max_depth,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-3, 5e-2, log=True
            ),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 3.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 3.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 30, 120),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
            "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
        }

        fold_scores: list[float] = []
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        try:
            for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train), start=1):
                X_tr = X_train.iloc[tr_idx].astype(np.float32)
                X_va = X_train.iloc[va_idx].astype(np.float32)
                y_tr = y_train[tr_idx]
                y_va = y_train[va_idx]

                params = complete_lgb_params(
                    suggested,
                    seed=random_state + fold,
                    num_threads=model_threads,
                )

                train_set = lgb.Dataset(
                    X_tr,
                    label=y_tr,
                    feature_name=X_train.columns.tolist(),
                    free_raw_data=False,
                )
                valid_set = lgb.Dataset(
                    X_va,
                    label=y_va,
                    reference=train_set,
                    feature_name=X_train.columns.tolist(),
                    free_raw_data=False,
                )

                model = lgb.train(
                    params,
                    train_set,
                    num_boost_round=NUM_BOOST_ROUND,
                    valid_sets=[valid_set],
                    valid_names=["validation"],
                    callbacks=[
                        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)
                    ],
                )

                predictions = model.predict(
                    X_va,
                    num_iteration=model.best_iteration,
                )
                fold_scores.append(rmse_score(y_va, predictions))

                # Optuna step is zero-indexed and unique within each trial.
                trial.report(float(np.mean(fold_scores)), step=fold - 1)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.mean(fold_scores))

        except optuna.TrialPruned:
            raise
        except Exception as exc:
            logger.exception("Trial %s failed: %s", trial.number, exc)
            raise

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            multivariate=False,
            seed=random_state,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=OPTUNA_STARTUP_TRIALS,
            n_warmup_steps=OPTUNA_WARMUP_STEPS,
            n_min_trials=5,
        ),
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        timeout=timeout_seconds,
        gc_after_trial=True,
        catch=(lgb.basic.LightGBMError, ValueError, FloatingPointError),
        show_progress_bar=False,
    )

    complete_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and np.isfinite(trial.value)
    ]
    if not complete_trials:
        raise RuntimeError(
            "Optuna produced no valid completed trial. Check the input data and log file."
        )

    best_trial = min(complete_trials, key=lambda trial: float(trial.value))
    best_params = dict(best_trial.params)

    trials_df = study.trials_dataframe(
        attrs=("number", "value", "datetime_start", "datetime_complete", "duration", "params", "state")
    )
    return best_params, study, trials_df


# =============================================================================
# 7. FIXED HOLDOUT MODEL EVALUATION
# =============================================================================
def train_and_evaluate_holdout(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    best_params: dict[str, Any],
    *,
    seed: int,
) -> tuple[lgb.Booster, pd.DataFrame, pd.DataFrame]:
    params = complete_lgb_params(
        best_params,
        seed=seed,
        num_threads=LGB_NUM_THREADS,
    )

    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=X_train.columns.tolist(),
        free_raw_data=False,
    )
    test_set = lgb.Dataset(
        X_test,
        label=y_test,
        reference=train_set,
        feature_name=X_train.columns.tolist(),
        free_raw_data=False,
    )

    model = lgb.train(
        params,
        train_set,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[test_set],
        valid_names=["holdout"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    predictions = model.predict(X_test, num_iteration=model.best_iteration)
    metrics = pd.DataFrame(
        [
            {
                "sample": "holdout",
                "n_train": len(X_train),
                "n_test": len(X_test),
                "best_iteration": int(model.best_iteration),
                "rmse": rmse_score(y_test, predictions),
                "mae": float(mean_absolute_error(y_test, predictions)),
                "r2": float(r2_score(y_test, predictions)),
                "target_mean": float(np.mean(y_test)),
                "prediction_mean": float(np.mean(predictions)),
            }
        ]
    )
    prediction_df = pd.DataFrame(
        {
            "actual": y_test,
            "predicted": predictions,
            "residual": y_test - predictions,
        }
    )
    return model, metrics, prediction_df


# =============================================================================
# 8. SHAP EXTRACTION
# =============================================================================
def extract_tree_shap_values(
    model: lgb.Booster,
    X_explain: pd.DataFrame,
) -> np.ndarray:
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )

    # The Explanation API is preferred in recent SHAP versions.
    explanation = explainer(
        X_explain,
        check_additivity=CHECK_SHAP_ADDITIVITY,
    )
    values = np.asarray(explanation.values, dtype=float)

    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.shape != X_explain.shape:
        raise ValueError(
            f"Unexpected SHAP shape {values.shape}; expected {X_explain.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("SHAP output contains NaN or infinite values.")
    return values


# =============================================================================
# 9. REPEATED OOF SHAP SEVERITY
# =============================================================================
def calculate_oof_shap_severity(
    X: pd.DataFrame,
    y: np.ndarray,
    best_params: dict[str, Any],
    *,
    n_splits: int,
    seed: int,
    retain_oof_shap: bool,
) -> OOFRunResult:
    if n_splits < 2:
        raise ValueError("OOF CV requires at least two folds.")

    n_samples, n_features = X.shape
    feature_names = X.columns.tolist()
    oof_predictions = np.full(n_samples, np.nan, dtype=float)
    oof_shap = np.full((n_samples, n_features), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X), start=1):
        X_train_fold = X.iloc[train_idx].astype(np.float32)
        X_valid_fold = X.iloc[valid_idx].astype(np.float32)
        y_train_fold = y[train_idx]
        y_valid_fold = y[valid_idx]

        fold_seed = seed * 100 + fold
        params = complete_lgb_params(
            best_params,
            seed=fold_seed,
            num_threads=LGB_NUM_THREADS,
        )

        train_set = lgb.Dataset(
            X_train_fold,
            label=y_train_fold,
            feature_name=feature_names,
            free_raw_data=False,
        )
        valid_set = lgb.Dataset(
            X_valid_fold,
            label=y_valid_fold,
            reference=train_set,
            feature_name=feature_names,
            free_raw_data=False,
        )

        model = lgb.train(
            params,
            train_set,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[valid_set],
            valid_names=["validation"],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )

        predictions = model.predict(
            X_valid_fold,
            num_iteration=model.best_iteration,
        )
        shap_values = extract_tree_shap_values(model, X_valid_fold)

        oof_predictions[valid_idx] = predictions
        oof_shap[valid_idx, :] = shap_values

        fold_rows.append(
            {
                "seed": seed,
                "fold": fold,
                "n_train": len(train_idx),
                "n_valid": len(valid_idx),
                "best_iteration": int(model.best_iteration),
                "rmse": rmse_score(y_valid_fold, predictions),
                "mae": float(mean_absolute_error(y_valid_fold, predictions)),
                "r2": float(r2_score(y_valid_fold, predictions)),
            }
        )

    if np.isnan(oof_predictions).any() or np.isnan(oof_shap).any():
        raise RuntimeError("OOF prediction/SHAP arrays are incomplete.")

    severity_abs: dict[str, float] = {}
    severity_negative: dict[str, float] = {}
    n_observed: dict[str, int] = {}

    for j, feature in enumerate(feature_names):
        observed_mask = X[feature].notna().to_numpy()
        count = int(observed_mask.sum())
        n_observed[feature] = count

        if count == 0:
            severity_abs[feature] = 0.0
            severity_negative[feature] = 0.0
            continue

        observed_values = oof_shap[observed_mask, j]
        severity_abs[feature] = float(np.mean(np.abs(observed_values)))
        severity_negative[feature] = float(
            np.mean(np.maximum(-observed_values, 0.0))
        )

    return OOFRunResult(
        seed=seed,
        severity_raw=pd.Series(severity_abs, dtype=float),
        severity_negative_raw=pd.Series(severity_negative, dtype=float),
        n_observed=pd.Series(n_observed, dtype=int),
        fold_metrics=pd.DataFrame(fold_rows),
        oof_predictions=oof_predictions,
        oof_shap=oof_shap if retain_oof_shap else None,
    )


def aggregate_repeated_severity(
    results: Sequence[OOFRunResult],
    *,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not results:
        raise ValueError("No repeated OOF result was supplied.")

    feature_order = results[0].severity_raw.index.tolist()
    for result in results[1:]:
        if result.severity_raw.index.tolist() != feature_order:
            raise ValueError("Feature order differs across repeated OOF runs.")

    severity_vectors = pd.DataFrame(
        [result.severity_raw.loc[feature_order] for result in results],
        index=[f"seed_{result.seed}" for result in results],
    )
    negative_vectors = pd.DataFrame(
        [result.severity_negative_raw.loc[feature_order] for result in results],
        index=[f"seed_{result.seed}" for result in results],
    )

    raw_mean = severity_vectors.mean(axis=0)
    raw_for_scale = raw_mean.copy()
    if SEVERITY_WINSORIZE:
        raw_for_scale = winsorize_series(
            raw_for_scale,
            SEVERITY_WINSOR_LOWER,
            SEVERITY_WINSOR_UPPER,
        )

    severity_score = normalize_to_scale(
        raw_for_scale,
        lower=SEVERITY_SCALE_MIN,
        upper=SEVERITY_SCALE_MAX,
    )

    std = severity_vectors.std(axis=0, ddof=1) if len(results) > 1 else pd.Series(0.0, index=feature_order)
    cv = std / (raw_mean + EPS)
    k_eff = min(top_k, len(feature_order))
    topk_mask = pd.DataFrame(
        False,
        index=severity_vectors.index,
        columns=severity_vectors.columns,
    )
    for run_name, row in severity_vectors.iterrows():
        # Stable sorting ensures exactly k features even when SHAP values tie.
        selected = row.sort_values(ascending=False, kind="mergesort").head(k_eff).index
        topk_mask.loc[run_name, selected] = True
    topk_frequency = topk_mask.mean(axis=0)

    n_observed = results[0].n_observed.loc[feature_order]
    severity_df = pd.DataFrame(
        {
            "ES": feature_order,
            "n_observed_reviews": n_observed.to_numpy(dtype=int),
            "shap_severity_raw_mean": raw_mean.to_numpy(dtype=float),
            "shap_severity_raw_std": std.to_numpy(dtype=float),
            "shap_severity_cv": cv.to_numpy(dtype=float),
            f"top{top_k}_frequency": topk_frequency.to_numpy(dtype=float),
            "negative_shap_raw_mean": negative_vectors.mean(axis=0).to_numpy(dtype=float),
            "Severity": severity_score.to_numpy(dtype=float),
        }
    ).sort_values("shap_severity_raw_mean", ascending=False).reset_index(drop=True)
    severity_df.insert(0, "severity_rank", np.arange(1, len(severity_df) + 1))

    pairwise_rows: list[dict[str, Any]] = []
    for (name_i, row_i), (name_j, row_j) in itertools.combinations(
        severity_vectors.iterrows(), 2
    ):
        rho, p_value = spearmanr(row_i, row_j)
        pairwise_rows.append(
            {
                "run_i": name_i,
                "run_j": name_j,
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "spearman_p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                f"top{top_k}_jaccard": topk_jaccard(row_i, row_j, top_k),
            }
        )
    pairwise_df = pd.DataFrame(pairwise_rows)

    stability_summary = pd.DataFrame(
        [
            {
                "n_repeated_oof_runs": len(results),
                "mean_pairwise_spearman": (
                    float(pairwise_df["spearman_rho"].mean())
                    if not pairwise_df.empty
                    else 1.0
                ),
                "std_pairwise_spearman": (
                    float(pairwise_df["spearman_rho"].std(ddof=1))
                    if len(pairwise_df) > 1
                    else 0.0
                ),
                f"mean_top{top_k}_jaccard": (
                    float(pairwise_df[f"top{top_k}_jaccard"].mean())
                    if not pairwise_df.empty
                    else 1.0
                ),
                f"minimum_top{top_k}_jaccard": (
                    float(pairwise_df[f"top{top_k}_jaccard"].min())
                    if not pairwise_df.empty
                    else 1.0
                ),
                "median_feature_cv": float(severity_df["shap_severity_cv"].median()),
                "mean_feature_cv": float(severity_df["shap_severity_cv"].mean()),
            }
        ]
    )

    return severity_df, severity_vectors, pairwise_df, stability_summary


# =============================================================================
# 10. OPTIONAL MISSINGNESS DIAGNOSTIC
# =============================================================================
def run_missingness_diagnostic(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    best_params: dict[str, Any],
    *,
    seed: int,
) -> pd.DataFrame:
    missing_train = X_train.isna().astype(np.float32)
    missing_test = X_test.isna().astype(np.float32)
    missing_train.columns = [f"{name}_missing" for name in X_train.columns]
    missing_test.columns = [f"{name}_missing" for name in X_test.columns]

    X_train_augmented = pd.concat(
        [X_train.reset_index(drop=True), missing_train.reset_index(drop=True)],
        axis=1,
    )
    X_test_augmented = pd.concat(
        [X_test.reset_index(drop=True), missing_test.reset_index(drop=True)],
        axis=1,
    )

    params = complete_lgb_params(
        best_params,
        seed=seed,
        num_threads=LGB_NUM_THREADS,
    )

    train_set = lgb.Dataset(
        X_train_augmented,
        label=y_train,
        feature_name=X_train_augmented.columns.tolist(),
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        X_test_augmented,
        label=y_test,
        reference=train_set,
        feature_name=X_train_augmented.columns.tolist(),
        free_raw_data=False,
    )

    model = lgb.train(
        params,
        train_set,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[valid_set],
        valid_names=["holdout"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    if len(X_test_augmented) > MISSINGNESS_DIAGNOSTIC_MAX_ROWS:
        sample = X_test_augmented.sample(
            n=MISSINGNESS_DIAGNOSTIC_MAX_ROWS,
            random_state=seed,
        )
    else:
        sample = X_test_augmented

    shap_values = extract_tree_shap_values(model, sample)
    mean_abs = np.mean(np.abs(shap_values), axis=0)

    result = pd.DataFrame(
        {
            "feature": sample.columns,
            "mean_abs_shap": mean_abs,
            "feature_type": [
                "missing_indicator" if name.endswith("_missing") else "service_element"
                for name in sample.columns
            ],
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result


# =============================================================================
# 11. PLOTS
# =============================================================================
def plot_top_severity(severity_df: pd.DataFrame, figure_dir: Path) -> None:
    top = severity_df.head(PLOT_TOP_N).sort_values("Severity", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["ES"], top["Severity"])
    ax.set_xlabel("SHAP-based severity (1-10)")
    ax.set_ylabel("Service element")
    ax.set_title(f"Top {min(PLOT_TOP_N, len(severity_df))} service elements by severity")
    fig.tight_layout()
    if SAVE_PLOTS:
        fig.savefig(figure_dir / "severity_top_elements.png", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_holdout_predictions(prediction_df: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(prediction_df["actual"], prediction_df["predicted"], alpha=0.5, s=18)
    minimum = float(min(prediction_df["actual"].min(), prediction_df["predicted"].min()))
    maximum = float(max(prediction_df["actual"].max(), prediction_df["predicted"].max()))
    ax.plot([minimum, maximum], [minimum, maximum], linestyle="--", linewidth=1.0)
    ax.set_xlabel("Actual overall rating")
    ax.set_ylabel("Predicted overall rating")
    ax.set_title("LightGBM holdout predictions")
    fig.tight_layout()
    if SAVE_PLOTS:
        fig.savefig(figure_dir / "holdout_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_missingness_diagnostic(diagnostic: pd.DataFrame, figure_dir: Path) -> None:
    indicators = diagnostic.loc[
        diagnostic["feature_type"] == "missing_indicator"
    ].head(PLOT_TOP_N).sort_values("mean_abs_shap", ascending=True)
    if indicators.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(indicators["feature"], indicators["mean_abs_shap"])
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("Missing indicator")
    ax.set_title("Missingness diagnostic (not used in Severity)")
    fig.tight_layout()
    if SAVE_PLOTS:
        fig.savefig(figure_dir / "missing_indicator_shap.png", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# =============================================================================
# 12. OUTPUTS
# =============================================================================
def save_results(
    *,
    output_dir: Path,
    severity_df: pd.DataFrame,
    repeated_vectors: pd.DataFrame,
    pairwise_stability: pd.DataFrame,
    stability_summary: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    holdout_metrics: pd.DataFrame,
    holdout_predictions: pd.DataFrame,
    data_diagnostics: pd.DataFrame,
    optuna_trials: pd.DataFrame,
    best_params: dict[str, Any],
    config: RunConfig,
    missingness_diagnostic: pd.DataFrame,
) -> None:
    output_dir, figure_dir = ensure_output_dirs(output_dir)

    severity_df.to_csv(output_dir / "SHAP_severity.csv", index=False, encoding="utf-8-sig")
    repeated_vectors.to_csv(
        output_dir / "repeated_severity_vectors.csv",
        index=True,
        encoding="utf-8-sig",
    )
    optuna_trials.to_csv(output_dir / "optuna_trials.csv", index=False, encoding="utf-8-sig")
    holdout_predictions.to_csv(
        output_dir / "holdout_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(output_dir / "best_params.json", "w", encoding="utf-8") as file:
        json.dump(
            {key: safe_json_value(value) for key, value in best_params.items()},
            file,
            ensure_ascii=False,
            indent=2,
        )

    config_df = pd.DataFrame(
        [
            {"parameter": key, "value": safe_json_value(value)}
            for key, value in asdict(config).items()
        ]
    )

    workbook = output_dir / "SHAP_severity_results.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        severity_df.to_excel(writer, sheet_name="Severity", index=False)
        holdout_metrics.to_excel(writer, sheet_name="Holdout_performance", index=False)
        fold_metrics.to_excel(writer, sheet_name="OOF_fold_metrics", index=False)
        stability_summary.to_excel(writer, sheet_name="Stability_summary", index=False)
        pairwise_stability.to_excel(writer, sheet_name="Pairwise_stability", index=False)
        repeated_vectors.T.to_excel(writer, sheet_name="Repeated_severity", index=True)
        data_diagnostics.to_excel(writer, sheet_name="Data_diagnostics", index=False)
        optuna_trials.to_excel(writer, sheet_name="Optuna_trials", index=False)
        holdout_predictions.to_excel(writer, sheet_name="Holdout_predictions", index=False)
        missingness_diagnostic.to_excel(
            writer,
            sheet_name="Missingness_diagnostic",
            index=False,
        )
        config_df.to_excel(writer, sheet_name="Configuration", index=False)

    plot_top_severity(severity_df, figure_dir)
    plot_holdout_predictions(holdout_predictions, figure_dir)
    if not missingness_diagnostic.empty:
        plot_missingness_diagnostic(missingness_diagnostic, figure_dir)


# =============================================================================
# 13. MAIN WORKFLOW
# =============================================================================
def run_workflow(
    *,
    input_file: Path,
    input_sheet: str,
    output_dir: Path,
    n_features: int,
    target_column: str,
    n_trials: int,
    optuna_cv_splits: int,
    oof_cv_splits: int,
    stability_seeds: Sequence[int],
    quick_test: bool,
) -> None:
    set_global_seed(GLOBAL_SEED)
    output_dir, _ = ensure_output_dirs(output_dir)

    if quick_test:
        n_trials = min(n_trials, 2)
        optuna_cv_splits = min(optuna_cv_splits, 2)
        oof_cv_splits = min(oof_cv_splits, 2)
        stability_seeds = tuple(stability_seeds[:1]) or (42,)
        global NUM_BOOST_ROUND, EARLY_STOPPING_ROUNDS
        NUM_BOOST_ROUND = min(NUM_BOOST_ROUND, 80)
        EARLY_STOPPING_ROUNDS = min(EARLY_STOPPING_ROUNDS, 10)

    feature_names = expected_feature_names(n_features)
    run_missingness_diagnostic_flag = RUN_MISSINGNESS_DIAGNOSTIC and not quick_test
    print("Loading and validating data...")
    X, y, data_diagnostics = load_and_validate_data(
        input_file,
        input_sheet,
        feature_names,
        target_column,
    )
    print(
        f"Loaded {len(X)} reviews, {len(feature_names)} service elements; "
        f"target mean={float(np.mean(y)):.4f}."
    )

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=HOLDOUT_SIZE,
        random_state=HOLDOUT_RANDOM_STATE,
        shuffle=True,
    )
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]

    print(
        f"Tuning LightGBM with {n_trials} Optuna trials and "
        f"{optuna_cv_splits}-fold CV..."
    )
    best_params, study, optuna_trials = tune_hyperparameters(
        X_train,
        y_train,
        n_trials=n_trials,
        n_splits=optuna_cv_splits,
        random_state=OPTUNA_RANDOM_STATE,
        n_jobs=OPTUNA_N_JOBS,
        timeout_seconds=OPTUNA_TIMEOUT_SECONDS,
        log_file=output_dir / "optuna_errors.log",
    )
    print(f"Best tuning RMSE: {float(study.best_value):.6f}")
    print("Best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    print("Training and evaluating the fixed holdout model...")
    holdout_model, holdout_metrics, holdout_predictions = train_and_evaluate_holdout(
        X_train,
        X_test,
        y_train,
        y_test,
        best_params,
        seed=HOLDOUT_RANDOM_STATE,
    )
    print(holdout_metrics.to_string(index=False))

    repeated_results: list[OOFRunResult] = []
    all_fold_metrics: list[pd.DataFrame] = []

    print(
        f"Calculating repeated {oof_cv_splits}-fold OOF SHAP severity for "
        f"seeds={tuple(stability_seeds)}..."
    )
    for seed in stability_seeds:
        result = calculate_oof_shap_severity(
            X,
            y,
            best_params,
            n_splits=oof_cv_splits,
            seed=int(seed),
            retain_oof_shap=SAVE_OOF_SHAP,
        )
        repeated_results.append(result)
        all_fold_metrics.append(result.fold_metrics)

        if SAVE_OOF_SHAP and result.oof_shap is not None:
            np.savez_compressed(
                output_dir / f"oof_shap_seed_{seed}.npz",
                shap_values=result.oof_shap,
                predictions=result.oof_predictions,
                feature_names=np.asarray(feature_names, dtype=object),
                seed=np.asarray([seed], dtype=int),
            )

        mean_rmse = float(result.fold_metrics["rmse"].mean())
        print(f"  seed={seed}: mean OOF RMSE={mean_rmse:.6f}")

    severity_df, repeated_vectors, pairwise_stability, stability_summary = (
        aggregate_repeated_severity(repeated_results, top_k=TOP_K)
    )
    fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)

    if run_missingness_diagnostic_flag:
        print("Running missing-indicator diagnostic (not used in Severity)...")
        missingness_diagnostic = run_missingness_diagnostic(
            X_train,
            X_test,
            y_train,
            y_test,
            best_params,
            seed=HOLDOUT_RANDOM_STATE,
        )
    else:
        missingness_diagnostic = pd.DataFrame(
            columns=["rank", "feature", "mean_abs_shap", "feature_type"]
        )

    config = RunConfig(
        input_file=str(input_file),
        input_sheet=input_sheet,
        output_dir=str(output_dir),
        n_features=n_features,
        target_column=target_column,
        holdout_size=HOLDOUT_SIZE,
        n_trials=n_trials,
        optuna_cv_splits=optuna_cv_splits,
        optuna_n_jobs=OPTUNA_N_JOBS,
        num_boost_round=NUM_BOOST_ROUND,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        oof_cv_splits=oof_cv_splits,
        stability_seeds=tuple(int(seed) for seed in stability_seeds),
        top_k=TOP_K,
        run_missingness_diagnostic=run_missingness_diagnostic_flag,
        severity_winsorize=SEVERITY_WINSORIZE,
    )

    save_results(
        output_dir=output_dir,
        severity_df=severity_df,
        repeated_vectors=repeated_vectors,
        pairwise_stability=pairwise_stability,
        stability_summary=stability_summary,
        fold_metrics=fold_metrics,
        holdout_metrics=holdout_metrics,
        holdout_predictions=holdout_predictions,
        data_diagnostics=data_diagnostics,
        optuna_trials=optuna_trials,
        best_params=best_params,
        config=config,
        missingness_diagnostic=missingness_diagnostic,
    )

    print("\nCompleted.")
    print(f"Severity workbook: {output_dir / 'SHAP_severity_results.xlsx'}")
    print(f"Severity CSV:      {output_dir / 'SHAP_severity.csv'}")
    print("Top severity results:")
    print(
        severity_df[
            [
                "severity_rank",
                "ES",
                "shap_severity_raw_mean",
                "shap_severity_cv",
                f"top{TOP_K}_frequency",
                "Severity",
            ]
        ].head(20).to_string(index=False)
    )


# =============================================================================
# 14. COMMAND-LINE INTERFACE
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate service-element RPN Severity using repeated OOF LightGBM SHAP."
    )
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--sheet", type=str, default=INPUT_SHEET)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-features", type=int, default=N_SERVICE_ELEMENTS)
    parser.add_argument("--target", type=str, default=TARGET_COLUMN)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--optuna-folds", type=int, default=OPTUNA_CV_SPLITS)
    parser.add_argument("--oof-folds", type=int, default=OOF_CV_SPLITS)
    parser.add_argument(
        "--stability-seeds",
        type=int,
        nargs="+",
        default=list(STABILITY_SEEDS),
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a short pipeline test with fewer trials/folds/boosting rounds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_workflow(
        input_file=args.input,
        input_sheet=args.sheet,
        output_dir=args.output_dir,
        n_features=args.n_features,
        target_column=args.target,
        n_trials=args.n_trials,
        optuna_cv_splits=args.optuna_folds,
        oof_cv_splits=args.oof_folds,
        stability_seeds=tuple(args.stability_seeds),
        quick_test=bool(args.quick_test),
    )


if __name__ == "__main__":
    main()


