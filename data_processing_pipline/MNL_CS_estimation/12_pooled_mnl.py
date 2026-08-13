# RELEASE NOTE
# ------------
# This file preserves the complete pooled-MNL implementation used in the study,
# including Lasso stability screening, dynamic period-specific choice sets,
# 3-period EWMA covariates, pooled count-weighted estimation, parametric
# bootstrap inference, t18 validation, and IPEA average marginal effects.
#
# The original local D:/ paths are retained in the configuration block and
# should be replaced with the corresponding paths listed in README.md before
# public release.
#
# -*- coding: utf-8 -*-
"""
Pooled MNL estimation and IPEA average marginal effects (AME)
================================================================

Choice-set rule
---------------
For each period t, only hotels with eligible observations in period t enter that period's choice set. The pooled MNL uses one common parameter vector across training periods t=1,...,17, while each period retains its own dynamic softmax denominator and period-specific covariates/counts.

Estimation
----------
REG_LAMBDA is fixed at 0.0. The common MNL parameters are estimated by
maximizing the count-weighted pooled log-likelihood across all training
periods. Multiple starting values and a pooled within-market rank diagnostic
are used because no ridge penalty is available to regularize unidentified
or weakly identified directions.

Scale conventions
-----------------
1. MNL covariates (including total_score and selected ES variables) are
   standardized using training periods t=1,...,17 only and then represented by
   the recursive rolling-window EWMA used in each period-specific market.
2. GRF Tau_Hat_Mean is estimated on the original 1--10 scale.
3. beta_z[k] / sigma[k] is the utility effect of a one-unit increase in the
   corresponding EWMA state. To interpret an intervention as a one-unit change
   in the latest raw observation, this state effect is multiplied by the
   latest-observation EWMA response weight lambda[a,t]:

       lambda[a,t] = 1                         if the window has one observation,
                     1 - WINDOW_GAMMA          otherwise.

4. The latest-raw-observation utility effect of service element m is

       Eff_raw[m,t]
       = lambda[a,t] * beta_z[m] / sigma_ES[m]
         + lambda[a,t] * beta_z[total_score] / sigma_total_score * tau[m].

   The output retains both the unadjusted EWMA-state effects and the adjusted
   latest-raw-observation effects. Downstream weight learning and optimization
   should use direct_effect_raw and zeta_total_score_raw and must not multiply
   the EWMA response weight again.
5. The focal hotel's period-specific probability marginal effect is

       ME[m,t] = P[a,t] * (1-P[a,t]) * Eff_raw[m,t].

6. The IPEA effect is

       AME[m] = sum_t w[t] * ME[m,t],

   using only periods in which the focal hotel belongs to the current dynamic
   choice set. By default, valid periods are equally weighted.
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import GroupKFold

from sklearn.utils import resample
from sklearn.linear_model import LinearRegression

from matplotlib.ticker import MaxNLocator

# ==========================================
#  变量筛选 
#===========================================

df_logit = pd.read_excel(r"D:\AAApaper\online_review\AOP-竞争酒店-new.xlsx", sheet_name='季度划分')
df_pic = pd.read_csv(r"D:\AAApaper\online_review\pics\competitive.csv")
df_pic["id"] = df_pic["id"].astype(str)

# ---- 为每个酒店构建聚合向量 ------
agg_vecs = []
agg_ids = []

for idx, row in df_pic.iterrows():
    hid = row["id"]
    emb_path = f"D:/AAApaper/online_review/pics/hotel_embeddings/{hid}.npy"
    if os.path.exists(emb_path):
        arr = np.load(emb_path)
        if arr.ndim == 1:
            arr = arr.reshape(1,-1)
        agg = arr.mean(axis=0)   # mean pooling
    else:
        agg = np.zeros(512)
    agg_vecs.append(agg)
    agg_ids.append(hid)

agg_matrix = np.vstack(agg_vecs)  # (N_hotels,512)


N_hotels = agg_matrix.shape[0]
max_pca_dim = min(N_hotels, 512)

print("允许的最大 PCA 维度 =", max_pca_dim)

pca = PCA(n_components=max_pca_dim)
pca.fit(agg_matrix)

plt.figure(figsize=(7,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("PCA components")
plt.ylabel("Cumulative explained variance")
plt.title("Explained variance curve")
plt.grid()
plt.show()

# ---- PCA ----
N_PCA = 13 
pca = PCA(n_components=N_PCA)
reduced = pca.fit_transform(agg_matrix)

# ---- 标准化（可选） ----
#scaler = StandardScaler()
#reduced_scaled = scaler.fit_transform(reduced)
# ---- 合并回 df ----
col_names = [f"img_pca_{i}" for i in range(N_PCA)]
df_img = pd.DataFrame(reduced, columns=col_names)
df_img["id"] = agg_ids


# 合并（按 id）
df_merged = df_pic.merge(df_img, on=["id"], how="left")

# 缺失填0（若没有embedding）
for c in col_names:
    df_merged[c] = df_merged[c].fillna(0.0)
    

df_logit['Name'] = df_logit['Name'].str.strip().str.lower()
df_merged['Name'] = df_merged['Name'].str.strip().str.lower()

df = pd.merge(
    df_logit,
    df_merged,
    on=['Name'],
    how='left'
)

feature_columns1 = [col for col in df.columns if col.startswith('ES_')]

feature_columns2 = ['Num', 'Star', 'Rating', 'pagerank_score', 'distance from centre (km)', 'price', 'total_score']


feature_columns =  feature_columns2 + feature_columns1


image_cols = [f"img_pca_{i}" for i in range(N_PCA)]
feature_columns = feature_columns + image_cols

X = df[feature_columns]  # 特征变量
y = df['Y'] # 目标变量


# 标准化特征
scaler = StandardScaler()
numeric_cols = X.columns[X.dtypes != 'object']
X_numeric = X[numeric_cols]
X_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=numeric_cols)


#必须保留的特征
must_keep =  ['price']
#其他特征
optional_features = [c for c in X_scaled.columns if c not in must_keep]

#拆分
X_fixed = X_scaled[must_keep]
X_optional = X_scaled[optional_features]

#=== 3. 控制固定变量，回归残差 ===
ols = LinearRegression().fit(X_fixed, y)
y_resid = y - ols.predict(X_fixed)


#=== 4. 用 LassoCV 找最优 alpha ===
groups = df.loc[X_optional.index, "Name"]

group_cv = GroupKFold(n_splits=5)

cv_splits = list(
    group_cv.split(
        X_optional,
        y_resid,
        groups=groups
    )
)

lasso_cv = LassoCV(
    cv=cv_splits,
    selection="cyclic",
    max_iter=10000
).fit(
    X_optional,
    np.asarray(y_resid)
)  
      
best_alpha = lasso_cv.alpha_
print("Best alpha:", best_alpha)
     


# ==========================================
# 2. Bootstrap稳定性选择
# ==========================================

seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
n_bootstrap_per_seed = 1000
n_features = X_optional.shape[1]

selection_counts = np.zeros(
    n_features,
    dtype=np.uint64
)

frequency_by_seed = []

for seed in seeds:
    rng = np.random.default_rng(seed)

    coef_matrix_seed = np.zeros(
        (n_bootstrap_per_seed, n_features),
        dtype=np.uint8
    )

    for b in range(n_bootstrap_per_seed):
        sample_idx = rng.choice(
            len(X_optional),
            size=len(X_optional),
            replace=True
        )

        Xb = X_optional.iloc[sample_idx]
        yb = np.asarray(y_resid)[sample_idx]

        lasso = Lasso(
            alpha=best_alpha,
            selection="cyclic",
            max_iter=10000
        ).fit(Xb, yb)

        coef_matrix_seed[b, :] = (
            np.abs(lasso.coef_) > 1e-8
        ).astype(np.uint8)

    seed_frequency = coef_matrix_seed.mean(axis=0)
    frequency_by_seed.append(seed_frequency)

    selection_counts += coef_matrix_seed.sum(
        axis=0,
        dtype=np.uint64
    )


# ==========================================
# 3. 汇总不同种子的结果
# ==========================================

frequency_by_seed = np.vstack(frequency_by_seed)

total_bootstrap = len(seeds) * n_bootstrap_per_seed

overall_frequency = (
    selection_counts.astype(np.float64)
    / total_bootstrap
)

stability_results = pd.DataFrame({
    "feature": X_optional.columns,
    "overall_frequency": overall_frequency,
    "mean_across_seeds": frequency_by_seed.mean(axis=0),
    "sd_across_seeds": frequency_by_seed.std(axis=0, ddof=1),
    "min_across_seeds": frequency_by_seed.min(axis=0),
    "max_across_seeds": frequency_by_seed.max(axis=0)
}).sort_values(
    "overall_frequency",
    ascending=False
).reset_index(drop=True)

print(stability_results)

# ==========================================
# 4. 按预设阈值确定最终变量
# ==========================================
selection_threshold = 0.50

selected_other = stability_results.loc[
    stability_results["overall_frequency"] >= selection_threshold,
    "feature"
].tolist()

selected_features_combined = must_keep + selected_other

print("Bootstrap总次数:", total_bootstrap)
print("入选阈值:", selection_threshold)
print("最终变量数量:", len(selected_features_combined))
print("最终进入MNL的变量:", selected_features_combined)

selected_features=pd.DataFrame(selected_features_combined)


# =============================================================================
# MNL estimation
# =============================================================================
# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = Path(r"D:\AAApaper\online_review")
LOGIT_FILE = BASE_DIR / "AOP-竞争酒店-new.xlsx"
LOGIT_SHEET = "季度划分"
PICTURE_META_FILE = BASE_DIR / "pics" / "competitive.csv"
EMBEDDING_DIR = BASE_DIR / "pics" / "hotel_embeddings"
TAU_FILE = BASE_DIR / "shap+TD_result.xlsx"
TAU_SHEET = "IPEA-modified"
OUTPUT_FILE = BASE_DIR / "IPEA_effect_results_pooled_MNL-stab2.xlsx"

FOCAL_HOTEL = "meriton suites herschel street brisbane"
MEDIATOR_NAME = "total_score"

TRAIN_PERIOD_START = 1
TRAIN_PERIOD_END = 17
TEST_PERIOD = 18
# "contemporaneous": use t18 covariates and t18 observed shares to assess
# transfer of the pooled t1--t17 parameters without refitting. This is
# held-out-period validation, not a strict one-step-ahead forecast.
# "lagged": use covariates available through t17 to predict t18 shares.
VALIDATION_COVARIATE_MODE = "contemporaneous"
MIN_RECORDS = 20

N_PCA = 13
WINDOW_SIZE = 3
WINDOW_GAMMA = 0.6

# Equal weighting across valid focal-hotel periods.
AME_TIME_GAMMA = 1.0

REG_LAMBDA = 0.0
POOLED_N_STARTS = 10
BOOTSTRAP_N_STARTS = 1
N_BOOTSTRAP = 100
SEED = 42
PLOT_PERIOD_FIT = False
# Disable the three separate t18 figures; use one combined comparison figure.
PLOT_TEST_FIT = False
PLOT_T18_MODEL_COMPARISON = True


# Basic-characteristics model requested by the user. The price coefficient is estimated separately as alpha and is therefore not included in this list.
FEATURE_COLUMNS_BASIC = ["Num", "Star", "distance from centre (km)", "total_score", "Rating", 'Cleanliness', "img_pca_3", "img_pca_5"]

FEATURE_COLUMNS_Att = ["Num", "Star", "distance from centre (km)", "total_score", "Rating", 'Cleanliness', "img_pca_3", "img_pca_5", 'Comfort and Security', 'Food and Drink', 'Location', 'Parking', 'Public area',	'Reception', 'Room and Facility', 'Staff and Professionalism', 'Value', 'Others'] 

# Fine-grained service-element model used for IPEA.
FEATURE_COLUMNS_ES = ["Num", "Star", "distance from centre (km)", "total_score", "Rating", 'ES_2', 'ES_3', 'ES_6', 'ES_7', 'ES_13', 'ES_14', 'ES_23', 'ES_24', 'ES_29', 'ES_33', 'ES_40', 'ES_44', 'ES_52', 'ES_56', 'ES_60', 'ES_62', 'ES_64', 'ES_65', 'ES_71', 'ES_72', 'ES_77', 'ES_80', 'ES_82', 'ES_91', 'ES_93', 'ES_99', 'ES_105', 'ES_109', 'ES_118', 'ES_119', 'ES_120', 'ES_122', "img_pca_3", "img_pca_5"]



MODEL_SPECS: dict[str, list[str]] = {
    "Basic": FEATURE_COLUMNS_BASIC,
    "ES": FEATURE_COLUMNS_ES,
    'Att':FEATURE_COLUMNS_Att
}

ALL_REQUIRED_FEATURES = list(
    dict.fromkeys(FEATURE_COLUMNS_BASIC + FEATURE_COLUMNS_ES + FEATURE_COLUMNS_Att)
)


# The price coefficient is handled separately and constrained non-positive.
# Feature-specific coefficient bounds are defined by name to avoid positional
# mistakes. All omitted features are unconstrained.
FEATURE_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    
}
PRICE_BOUND: tuple[float | None, float | None] = (None, 0.0)



# =============================================================================
# General helpers
# =============================================================================
def normalize_name(value: Any) -> str:
    return str(value).strip().lower()


def normalize_es_name(value: Any) -> str:
    """Normalize ES2, ES-2, ES_2, and ES 2 to ES_2."""
    text = str(value).strip()
    match = re.fullmatch(r"(?i)ES[\s_-]?(\d+)", text)
    if match:
        return f"ES_{int(match.group(1))}"
    return text


def validate_required_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    data_name: str,
) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise KeyError(f"{data_name} is missing required columns: {missing}")


def softmax(utilities: np.ndarray) -> np.ndarray:
    u = np.asarray(utilities, dtype=float)
    if u.ndim != 1 or u.size == 0:
        raise ValueError("softmax expects a non-empty one-dimensional array.")
    if not np.all(np.isfinite(u)):
        raise ValueError("Utilities contain NaN or infinite values.")
    shifted = u - np.max(u)
    exp_u = np.exp(shifted)
    denominator = exp_u.sum()
    if not np.isfinite(denominator) or denominator <= 0:
        raise FloatingPointError("Invalid softmax denominator.")
    return exp_u / denominator


def exponential_weighted_average(
    values: Iterable[np.ndarray | float],
    gamma: float = 0.9,
) -> np.ndarray | float:
    """
    Sequential EWMA matching the original implementation.

    Larger gamma retains more of the earlier EWMA; smaller gamma gives the
    newest observation more weight.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1].")

    values_list = list(values)
    if not values_list:
        raise ValueError("values cannot be empty.")

    ewma = np.asarray(values_list[0], dtype=float)
    for value in values_list[1:]:
        ewma = gamma * ewma + (1.0 - gamma) * np.asarray(value, dtype=float)

    if ewma.ndim == 0:
        return float(ewma)
    return ewma


def latest_ewma_observation_weight(
    n_observations: int,
    gamma: float,
) -> float:
    """
    Return the response of the recursive EWMA state to a one-unit change in
    its latest raw observation.

    The EWMA implementation initializes the state with the first available
    observation and applies ``ewma = gamma * ewma + (1-gamma) * value`` for
    every subsequent observation. Consequently, the latest observation has
    weight 1 when it is the only observation in the window and weight
    ``1-gamma`` otherwise.
    """
    if n_observations < 1:
        raise ValueError("n_observations must be at least 1.")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1].")
    if n_observations == 1:
        return 1.0
    return float(1.0 - gamma)


def _consistent_scalar(
    series: pd.Series,
    variable_name: str,
    hotel_name: str,
    time_period: int,
) -> float:
    """
    Return a scalar when duplicate rows contain the same value.

    This prevents accidental over-counting after a merge. If genuinely
    different values occur for one hotel-period, the data must be aggregated
    explicitly rather than silently taking the first row.
    """
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError(
            f"No valid {variable_name} for hotel={hotel_name!r}, period={time_period}."
        )
    if not np.allclose(values, values[0], rtol=1e-10, atol=1e-12):
        raise ValueError(
            f"Inconsistent duplicate {variable_name} values for "
            f"hotel={hotel_name!r}, period={time_period}: {values.tolist()}"
        )
    return float(values[0])


# =============================================================================
# Image/PCA preparation and source-data merge
# =============================================================================
def load_and_merge_data() -> pd.DataFrame:
    df_logit = pd.read_excel(LOGIT_FILE, sheet_name=LOGIT_SHEET)
    df_pic = pd.read_csv(PICTURE_META_FILE)

    validate_required_columns(df_logit, ["Name"], "logit data")
    validate_required_columns(df_pic, ["id", "Name"], "picture metadata")

    df_logit = df_logit.copy()
    df_pic = df_pic.copy()
    df_logit["Name"] = df_logit["Name"].map(normalize_name)
    df_pic["Name"] = df_pic["Name"].map(normalize_name)
    df_pic["id"] = df_pic["id"].astype(str)

    agg_vecs: list[np.ndarray] = []
    agg_ids: list[str] = []

    for _, row in df_pic.iterrows():
        hotel_id = row["id"]
        embedding_path = EMBEDDING_DIR / f"{hotel_id}.npy"

        if embedding_path.exists():
            array = np.load(embedding_path)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            if array.ndim != 2:
                raise ValueError(
                    f"Embedding file {embedding_path} must be 1D or 2D; "
                    f"received shape {array.shape}."
                )
            aggregate_vector = np.asarray(array, dtype=float).mean(axis=0)
        else:
            aggregate_vector = np.zeros(512, dtype=float)

        if aggregate_vector.shape[0] != 512:
            raise ValueError(
                f"Embedding file for id={hotel_id!r} has dimension "
                f"{aggregate_vector.shape[0]}, expected 512."
            )

        agg_vecs.append(aggregate_vector)
        agg_ids.append(hotel_id)

    if not agg_vecs:
        raise ValueError("No picture metadata rows were loaded.")

    agg_matrix = np.vstack(agg_vecs)
    max_pca_dim = min(agg_matrix.shape[0], agg_matrix.shape[1])
    if N_PCA > max_pca_dim:
        raise ValueError(
            f"N_PCA={N_PCA} exceeds the maximum available dimension "
            f"{max_pca_dim}."
        )

    print("允许的最大 PCA 维度 =", max_pca_dim)
    pca = PCA(n_components=N_PCA, random_state=SEED)
    reduced = pca.fit_transform(agg_matrix)

    image_columns = [f"img_pca_{i}" for i in range(N_PCA)]
    df_img = pd.DataFrame(reduced, columns=image_columns)
    df_img["id"] = agg_ids

    image_meta = df_pic[["id", "Name"]].merge(df_img, on="id", how="left")
    image_meta[image_columns] = image_meta[image_columns].fillna(0.0)

    # Guarantee one image representation per hotel name. This avoids duplicate
    # hotel-period rows after the merge when the metadata contains multiple IDs.
    image_by_hotel = (
        image_meta.groupby("Name", as_index=False, sort=True)[image_columns]
        .mean()
    )

    merged = df_logit.merge(image_by_hotel, on="Name", how="left", validate="many_to_one")
    merged[image_columns] = merged[image_columns].fillna(0.0)

    validate_required_columns(
        merged,
        [
            "Name",
            "time_group_id",
            "record_num",
            "price",
            *ALL_REQUIRED_FEATURES,
        ],
        "merged MNL data",
    )
    return merged


# =============================================================================
# Training-period standardization
# =============================================================================
def prepare_standardized_model_data(
    df: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Filter eligible hotel-period observations and standardize MNL covariates.

    Only t=1,...,17 observations are used to estimate means/standard deviations.
    The original and standardized frames are both returned.
    """
    model_raw = df.loc[pd.to_numeric(df["record_num"], errors="coerce") > MIN_RECORDS].copy()
    model_raw["Name"] = model_raw["Name"].map(normalize_name)

    model_raw["time_group_id"] = pd.to_numeric(
        model_raw["time_group_id"], errors="raise"
    ).astype(int)

    for column in ["record_num", "price", *feature_names]:
        model_raw[column] = pd.to_numeric(model_raw[column], errors="coerce")

    train_mask = model_raw["time_group_id"].between(
        TRAIN_PERIOD_START, TRAIN_PERIOD_END
    )
    if not train_mask.any():
        raise ValueError("No eligible training observations were found.")

    train_features = model_raw.loc[train_mask, feature_names]
    missing_by_column = train_features.isna().sum()
    missing_columns = missing_by_column[missing_by_column > 0].to_dict()
    if missing_columns:
        raise ValueError(
            "Training MNL variables contain missing values. Resolve them before "
            f"estimation: {missing_columns}"
        )

    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0, ddof=0)

    invalid_std = feature_std[(~np.isfinite(feature_std)) | (feature_std <= 0)]
    if not invalid_std.empty:
        raise ValueError(
            "Features with zero/invalid training standard deviation: "
            f"{invalid_std.to_dict()}"
        )

    all_feature_missing = model_raw[feature_names].isna().sum()
    all_missing_columns = all_feature_missing[all_feature_missing > 0].to_dict()
    if all_missing_columns:
        raise ValueError(
            "MNL variables contain missing values outside the training subset: "
            f"{all_missing_columns}"
        )

    model_z = model_raw.copy()
    model_z[feature_names] = (
        model_raw[feature_names] - feature_mean
    ) / feature_std

    return model_raw, model_z, feature_mean, feature_std


def calculate_vif(model_z: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    train_mask = model_z["time_group_id"].between(
        TRAIN_PERIOD_START, TRAIN_PERIOD_END
    )
    x_vif = model_z.loc[train_mask, feature_names].astype(float)

    vif_values: list[float] = []
    for index in range(x_vif.shape[1]):
        try:
            vif_values.append(float(variance_inflation_factor(x_vif.values, index)))
        except Exception:
            vif_values.append(np.inf)

    return pd.DataFrame({"feature": x_vif.columns, "VIF": vif_values})


# =============================================================================
# Pooled MNL estimation helpers
# =============================================================================
def make_parameter_bounds(
    feature_names: list[str],
) -> list[tuple[float | None, float | None]]:
    beta_bounds = [FEATURE_BOUNDS.get(name, (None, None)) for name in feature_names]
    return [PRICE_BOUND, *beta_bounds]


def initialize_parameters(
    feature_names: list[str],
    rng: np.random.Generator,
) -> np.ndarray:
    alpha = np.array([rng.uniform(-0.1, -0.01)], dtype=float)
    beta = rng.normal(0.0, 0.05, size=len(feature_names))

    for index, name in enumerate(feature_names):
        lower, upper = FEATURE_BOUNDS.get(name, (None, None))
        if lower is not None:
            beta[index] = max(beta[index], lower + 1e-6)
        if upper is not None:
            beta[index] = min(beta[index], upper - 1e-6)

    return np.concatenate([alpha, beta])


def pooled_negative_log_likelihood(
    params: np.ndarray,
    markets: list[dict[str, Any]],
    reg_lambda: float = REG_LAMBDA,
) -> float:
    """Count-weighted pooled multinomial negative log-likelihood."""
    if reg_lambda != 0.0:
        raise ValueError(
            "This pooled-MNL implementation requires REG_LAMBDA=0.0."
        )

    alpha = float(params[0])
    beta = np.asarray(params[1:], dtype=float)
    total_nll = 0.0

    for market in markets:
        x = np.asarray(market["x"], dtype=float)
        price = np.asarray(market["price"], dtype=float)
        counts = np.asarray(market["counts"], dtype=float)

        utilities = alpha * price + x @ beta
        probabilities = np.clip(softmax(utilities), 1e-12, 1.0)
        total_nll -= float(np.sum(counts * np.log(probabilities)))

    return total_nll


def predict_probabilities(
    x: np.ndarray,
    price: np.ndarray,
    alpha: float,
    beta: np.ndarray,
) -> np.ndarray:
    utilities = (
        float(alpha) * np.asarray(price, dtype=float)
        + np.asarray(x, dtype=float) @ np.asarray(beta, dtype=float)
    )
    return softmax(utilities)


def evaluate_fit(
    x: np.ndarray,
    price: np.ndarray,
    alpha: float,
    beta: np.ndarray,
    y_actual: np.ndarray,
) -> dict[str, Any]:
    p_pred = predict_probabilities(x, price, alpha, beta)
    y = np.asarray(y_actual, dtype=float)
    if y.ndim != 1 or y.shape != p_pred.shape:
        raise ValueError("y_actual has an invalid shape.")
    if y.sum() <= 0:
        raise ValueError("y_actual must sum to a positive value.")
    y = y / y.sum()

    mse = float(np.mean((p_pred - y) ** 2))
    mae = float(np.mean(np.abs(p_pred - y)))
    kl = float(np.sum(y * np.log((y + 1e-10) / (p_pred + 1e-10))))

    norm_product = float(np.linalg.norm(p_pred) * np.linalg.norm(y))
    cosine = float(np.dot(p_pred, y) / norm_product) if norm_product > 0 else np.nan
    relative_error = float(np.mean(np.abs(p_pred - y) / (y + 1e-8)))

    spearman = float(spearmanr(p_pred, y).statistic) if len(y) > 1 else np.nan
    pearson = float(pearsonr(p_pred, y).statistic) if len(y) > 1 else np.nan
    kendall = float(kendalltau(p_pred, y).statistic) if len(y) > 1 else np.nan

    tss = float(np.sum((y - y.mean()) ** 2))
    rss = float(np.sum((y - p_pred) ** 2))
    r2 = float(1.0 - rss / tss) if tss > 0 else np.nan

    k = min(5, len(y))
    top_pred = set(p_pred.argsort()[::-1][:k])
    top_actual = set(y.argsort()[::-1][:k])
    top_k_overlap = len(top_pred & top_actual) / k

    return {
        "MSE": mse,
        "MAE": mae,
        "KL": kl,
        "Cosine Similarity": cosine,
        "Mean Relative Error": relative_error,
        "Spearman Correlation": spearman,
        "Pearson Correlation": pearson,
        "Kendall Tau": kendall,
        "R2": r2,
        "Top-K Consistency": float(top_k_overlap),
        "Predicted Probabilities": p_pred,
    }


def pooled_design_diagnostics(
    markets: list[dict[str, Any]],
    feature_names: list[str],
) -> dict[str, float | int]:
    """
    Check the rank of within-market differences, which identify MNL slopes.

    For each market, the last alternative is used as the reference. The first
    column is the raw-price difference and the remaining columns are the
    standardized feature differences.
    """
    differenced_rows: list[np.ndarray] = []

    for market in markets:
        x = np.asarray(market["x"], dtype=float)
        price = np.asarray(market["price"], dtype=float)
        if x.shape[0] < 2:
            continue
        reference = np.concatenate([[price[-1]], x[-1]])
        design = np.column_stack([price, x])
        differenced_rows.extend(design[:-1] - reference)

    if not differenced_rows:
        raise ValueError("No within-market contrasts are available.")

    matrix = np.vstack(differenced_rows)
    n_parameters = 1 + len(feature_names)
    rank = int(np.linalg.matrix_rank(matrix))
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    positive = singular_values[singular_values > np.finfo(float).eps]
    condition_number = (
        float(positive.max() / positive.min())
        if positive.size > 0
        else np.inf
    )

    diagnostics = {
        "n_markets": len(markets),
        "n_within_market_contrasts": int(matrix.shape[0]),
        "n_parameters": n_parameters,
        "design_rank": rank,
        "condition_number": condition_number,
    }

    if rank < n_parameters:
        raise RuntimeError(
            "The pooled MNL is not identified without regularization: "
            f"within-market design rank={rank}, parameters={n_parameters}. "
            "Reduce/re-specify predictors before using REG_LAMBDA=0."
        )

    return diagnostics


def estimate_pooled_mnl(
    markets: list[dict[str, Any]],
    feature_names: list[str],
    bounds: list[tuple[float | None, float | None]],
    rng: np.random.Generator,
    n_starts: int = POOLED_N_STARTS,
    initial_params: np.ndarray | None = None,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Estimate one common MNL parameter vector across all training markets."""
    if REG_LAMBDA != 0.0:
        raise ValueError("REG_LAMBDA must equal 0.0 for pooled MNL estimation.")
    if not markets:
        raise ValueError("markets cannot be empty.")
    if n_starts < 1:
        raise ValueError("n_starts must be at least 1.")

    starts: list[np.ndarray] = []
    if initial_params is not None:
        initial = np.asarray(initial_params, dtype=float)
        if initial.shape != (1 + len(feature_names),):
            raise ValueError("initial_params has an invalid shape.")
        starts.append(initial.copy())

    while len(starts) < n_starts:
        starts.append(initialize_parameters(feature_names, rng))

    best_result = None
    successful_objectives: list[float] = []

    for start_index, start in enumerate(starts, start=1):
        result = minimize(
            fun=pooled_negative_log_likelihood,
            x0=start,
            args=(markets, REG_LAMBDA),
            bounds=bounds,
            method="L-BFGS-B",
            options={
                "disp": False,
                "maxiter": 20000,
                "maxfun": 200000,
                "gtol": 1e-7,
                "ftol": 1e-11,
                "maxls": 80,
            },
        )

        if result.success and np.all(np.isfinite(result.x)) and np.isfinite(result.fun):
            successful_objectives.append(float(result.fun))
            if best_result is None or result.fun < best_result.fun:
                best_result = result

    if best_result is None:
        raise RuntimeError(
            f"Pooled MNL failed for all {len(starts)} starting values."
        )

    diagnostics = {
        "n_starts": len(starts),
        "n_successful_starts": len(successful_objectives),
        "best_negative_log_likelihood": float(best_result.fun),
        "objective_range_across_successful_starts": (
            float(max(successful_objectives) - min(successful_objectives))
            if successful_objectives
            else np.nan
        ),
        "optimizer_message": str(best_result.message),
        "optimizer_iterations": int(getattr(best_result, "nit", -1)),
    }

    return (
        float(best_result.x[0]),
        np.asarray(best_result.x[1:], dtype=float),
        diagnostics,
    )

# =============================================================================
# Current-period choice set and window-smoothed variables
# =============================================================================
def build_current_choice_set(
    model_z: pd.DataFrame,
    time_period: int,
    feature_names: list[str],
    window_size: int,
    gamma: float,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the period-t MNL data under Choice-set Rule B.

    Only hotels with an eligible row in ``time_period`` enter the period-t
    choice set. For each such hotel, the EWMA is calculated from available
    observations in the actual calendar window [t-window_size+1, t]. A hotel
    absent in t is never carried forward into the choice set.

    The final returned vector contains, in the same hotel order, the response
    of each hotel's EWMA state to a one-unit change in its latest raw
    observation. This response is used only to translate estimated EWMA-state
    coefficients into latest-period intervention effects; it is not an
    additional MNL regressor.
    """
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")

    current_period = model_z.loc[model_z["time_group_id"] == time_period].copy()
    current_hotels = sorted(
        current_period["Name"].dropna().map(normalize_name).unique()
    )

    if len(current_hotels) < 2:
        raise ValueError(
            f"Period {time_period} has fewer than two eligible current hotels."
        )

    window_start = max(TRAIN_PERIOD_START, time_period - window_size + 1)

    x_rows: list[np.ndarray] = []
    price_rows: list[float] = []
    count_rows: list[float] = []
    latest_weight_rows: list[float] = []

    for hotel_name in current_hotels:
        hotel_window = model_z.loc[
            (model_z["Name"] == hotel_name)
            & model_z["time_group_id"].between(window_start, time_period)
        ].copy()

        if hotel_window.empty:
            raise RuntimeError(
                f"Internal error: no window data for current hotel {hotel_name!r}."
            )

        period_x: list[np.ndarray] = []
        period_price: list[float] = []
        period_counts: list[float] = []

        for hist_t, hist_group in hotel_window.groupby("time_group_id", sort=True):
            period_x.append(
                hist_group[feature_names]
                .astype(float)
                .mean(axis=0)
                .to_numpy(dtype=float)
            )
            period_price.append(
                _consistent_scalar(
                    hist_group["price"], "price", hotel_name, int(hist_t)
                )
            )
            period_counts.append(
                _consistent_scalar(
                    hist_group["record_num"],
                    "record_num",
                    hotel_name,
                    int(hist_t),
                )
            )

        x_rows.append(
            np.asarray(exponential_weighted_average(period_x, gamma), dtype=float)
        )
        price_rows.append(
            float(exponential_weighted_average(period_price, gamma))
        )
        count_rows.append(
            float(exponential_weighted_average(period_counts, gamma))
        )
        latest_weight_rows.append(
            latest_ewma_observation_weight(
                n_observations=len(period_x),
                gamma=gamma,
            )
        )

    x_current = np.vstack(x_rows)
    price_current = np.asarray(price_rows, dtype=float)
    count_current = np.asarray(count_rows, dtype=float)
    latest_state_weights = np.asarray(latest_weight_rows, dtype=float)

    if np.any(count_current < 0) or count_current.sum() <= 0:
        raise ValueError(f"Invalid period-{time_period} smoothed counts.")

    expected_shape = (len(current_hotels),)
    if latest_state_weights.shape != expected_shape:
        raise RuntimeError(
            "The number of latest-observation EWMA response weights differs "
            "from the current choice-set size."
        )
    if not np.all(np.isfinite(latest_state_weights)):
        raise ValueError("EWMA response weights contain NaN or infinite values.")
    if np.any((latest_state_weights < 0.0) | (latest_state_weights > 1.0)):
        raise ValueError("EWMA response weights must lie in [0, 1].")

    return (
        current_hotels,
        x_current,
        price_current,
        count_current,
        latest_state_weights,
    )


def build_training_markets(
    model_z: pd.DataFrame,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    """Build all t1--t17 markets while preserving each dynamic choice set."""
    markets: list[dict[str, Any]] = []

    for time_period in range(TRAIN_PERIOD_START, TRAIN_PERIOD_END + 1):
        try:
            (
                hotel_order,
                x_current,
                price_current,
                y_counts,
                latest_state_weights,
            ) = build_current_choice_set(
                model_z=model_z,
                time_period=time_period,
                feature_names=feature_names,
                window_size=WINDOW_SIZE,
                gamma=WINDOW_GAMMA,
            )
        except ValueError as exc:
            print(f"跳过时间段 {time_period}: {exc}")
            continue

        markets.append(
            {
                "time_period": int(time_period),
                "hotel_order": list(hotel_order),
                "x": np.asarray(x_current, dtype=float),
                "price": np.asarray(price_current, dtype=float),
                "counts": np.asarray(y_counts, dtype=float),
                "latest_state_weights": np.asarray(
                    latest_state_weights,
                    dtype=float,
                ),
            }
        )

    if not markets:
        raise RuntimeError("No valid training markets were constructed.")
    return markets


def pooled_parametric_bootstrap(
    markets: list[dict[str, Any]],
    alpha_hat: float,
    beta_hat: np.ndarray,
    feature_names: list[str],
    bounds: list[tuple[float | None, float | None]],
    rng: np.random.Generator,
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple[np.ndarray, np.ndarray]:
    """Parametric bootstrap preserving every period-specific choice set."""
    if n_bootstrap <= 0:
        return np.empty(0), np.empty((0, len(feature_names)))

    fitted_probabilities = [
        predict_probabilities(
            market["x"], market["price"], alpha_hat, beta_hat
        )
        for market in markets
    ]
    initial = np.concatenate([[alpha_hat], np.asarray(beta_hat, dtype=float)])

    alpha_samples: list[float] = []
    beta_samples: list[np.ndarray] = []
    attempts = 0
    max_attempts = max(3 * n_bootstrap, n_bootstrap)

    while len(alpha_samples) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        bootstrap_markets: list[dict[str, Any]] = []

        for market, probabilities in zip(markets, fitted_probabilities):
            n_draws = max(
                int(round(float(np.sum(market["counts"])))),
                len(probabilities),
            )
            simulated_counts = rng.multinomial(n_draws, probabilities).astype(float)
            bootstrap_market = dict(market)
            bootstrap_market["counts"] = simulated_counts
            bootstrap_markets.append(bootstrap_market)

        try:
            alpha_b, beta_b, _ = estimate_pooled_mnl(
                markets=bootstrap_markets,
                feature_names=feature_names,
                bounds=bounds,
                rng=rng,
                n_starts=BOOTSTRAP_N_STARTS,
                initial_params=initial,
            )
        except RuntimeError:
            continue

        alpha_samples.append(alpha_b)
        beta_samples.append(beta_b)

    if len(alpha_samples) < max(10, n_bootstrap // 2):
        raise RuntimeError(
            f"Only {len(alpha_samples)} of {n_bootstrap} pooled bootstrap "
            "replications converged."
        )

    return np.asarray(alpha_samples), np.vstack(beta_samples)


def fit_pooled_model(
    model_z: pd.DataFrame,
    feature_names: list[str],
    model_name: str = "MNL",
) -> dict[str, Any]:
    """Fit one common MNL parameter vector over all training periods."""
    rng = np.random.default_rng(SEED)
    bounds = make_parameter_bounds(feature_names)
    markets = build_training_markets(model_z, feature_names)
    rank_diagnostics = pooled_design_diagnostics(markets, feature_names)

    print(f"\n=== {model_name}: pooled MNL over t1--t17 ===")
    print("Pooled design diagnostics:", rank_diagnostics)

    alpha_hat, beta_hat, optimizer_diagnostics = estimate_pooled_mnl(
        markets=markets,
        feature_names=feature_names,
        bounds=bounds,
        rng=rng,
        n_starts=POOLED_N_STARTS,
    )

    alpha_boot, beta_boot = pooled_parametric_bootstrap(
        markets=markets,
        alpha_hat=alpha_hat,
        beta_hat=beta_hat,
        feature_names=feature_names,
        bounds=bounds,
        rng=rng,
        n_bootstrap=N_BOOTSTRAP,
    )

    parameter_names = ["alpha", *feature_names]
    variable_names = ["price", *feature_names]
    estimates = np.concatenate([[alpha_hat], beta_hat])

    if alpha_boot.size > 0:
        bootstrap_matrix = np.column_stack([alpha_boot, beta_boot])
        lower95 = np.percentile(bootstrap_matrix, 2.5, axis=0)
        upper95 = np.percentile(bootstrap_matrix, 97.5, axis=0)
    else:
        lower95 = np.full_like(estimates, np.nan)
        upper95 = np.full_like(estimates, np.nan)

    parameter_summary = pd.DataFrame(
        {
            "Parameter": parameter_names,
            "Variable": variable_names,
            "Estimate": estimates,
            "Lower95": lower95,
            "Upper95": upper95,
        }
    )

    period_results: list[dict[str, Any]] = []
    total_counts = 0.0
    total_alternatives = 0

    for market in markets:
        y_counts = np.asarray(market["counts"], dtype=float)
        y_probability = y_counts / y_counts.sum()
        metrics = evaluate_fit(
            x=market["x"],
            price=market["price"],
            alpha=alpha_hat,
            beta=beta_hat,
            y_actual=y_probability,
        )
        predicted = np.asarray(metrics["Predicted Probabilities"], dtype=float)

        period_results.append(
            {
                "time_period": int(market["time_period"]),
                "alpha": float(alpha_hat),
                "beta": np.asarray(beta_hat, dtype=float).copy(),
                "feature_names": list(feature_names),
                "hotel_order": list(market["hotel_order"]),
                "predicted_probabilities": predicted.copy(),
                "observed_probabilities": y_probability.copy(),
                "smoothed_counts": y_counts.copy(),
                "latest_state_weights": np.asarray(
                    market["latest_state_weights"],
                    dtype=float,
                ).copy(),
                "metrics": metrics,
            }
        )

        total_counts += float(y_counts.sum())
        total_alternatives += len(market["hotel_order"])

        print(
            f"t={market['time_period']}: choice set={len(market['hotel_order'])}, "
            f"KL={metrics['KL']:.4f}, R2={metrics['R2']:.4f}, "
            f"Top-K={metrics['Top-K Consistency']:.4f}"
        )

        if PLOT_PERIOD_FIT:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(y_probability, label="Observed share", marker="o")
            ax.plot(predicted, label="Pooled-MNL probability", marker="x")
            ax.set_title(
                f"{model_name}: pooled-MNL fit in period {market['time_period']}"
            )
            ax.set_xlabel("Hotel index in current-period choice set")
            ax.set_ylabel("Probability")
            ax.legend()
            ax.grid(True)
            fig.tight_layout()
            plt.show()
            plt.close(fig)

    log_likelihood = -pooled_negative_log_likelihood(
        np.concatenate([[alpha_hat], beta_hat]), markets, REG_LAMBDA
    )
    n_parameters = 1 + len(feature_names)
    aic = 2 * n_parameters - 2 * log_likelihood
    bic = np.log(max(total_counts, 1.0)) * n_parameters - 2 * log_likelihood

    pooled_fit_summary = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Training_Start": TRAIN_PERIOD_START,
                "Training_End": TRAIN_PERIOD_END,
                "N_Markets": len(markets),
                "Total_Alternatives_Across_Markets": total_alternatives,
                "Total_Smoothed_Counts": total_counts,
                "N_Parameters": n_parameters,
                "REG_LAMBDA": REG_LAMBDA,
                "Log_Likelihood": log_likelihood,
                "AIC": aic,
                "BIC": bic,
                **rank_diagnostics,
                **optimizer_diagnostics,
            }
        ]
    )

    return {
        "model_name": model_name,
        "alpha": float(alpha_hat),
        "beta": np.asarray(beta_hat, dtype=float).copy(),
        "feature_names": list(feature_names),
        "parameter_summary": parameter_summary,
        "pooled_fit_summary": pooled_fit_summary,
        "period_results": period_results,
        "markets": markets,
    }


# =============================================================================
# Out-of-sample validation on t=18
# =============================================================================
def build_test_period_data(
    model_z: pd.DataFrame,
    time_period: int,
    feature_names: list[str],
    window_size: int,
    gamma: float,
    covariate_mode: str = VALIDATION_COVARIATE_MODE,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Build the t-period validation choice set under Choice-set Rule B.

    Only hotels with an eligible observation in ``time_period`` enter the test
    choice set. With ``covariate_mode="contemporaneous"``, covariates and price
    are constructed from the rolling window ending in t, while t-period counts
    are used only as validation outcomes. This tests whether parameters learned
    through t-1 transfer to the held-out t-period cross-section without
    refitting. With ``covariate_mode="lagged"``, the predictor window ends in
    t-1, which is the stricter one-step-ahead forecasting design.
    """
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")
    if covariate_mode not in {"contemporaneous", "lagged"}:
        raise ValueError(
            "covariate_mode must be 'contemporaneous' or 'lagged'."
        )

    current_period = model_z.loc[
        model_z["time_group_id"] == time_period
    ].copy()
    current_hotels = sorted(
        current_period["Name"].dropna().map(normalize_name).unique()
    )

    if len(current_hotels) < 2:
        raise ValueError(
            f"Test period {time_period} has fewer than two eligible current hotels."
        )

    predictor_end_period = (
        time_period if covariate_mode == "contemporaneous" else time_period - 1
    )
    window_start = max(
        TRAIN_PERIOD_START, predictor_end_period - window_size + 1
    )

    x_rows: list[np.ndarray] = []
    price_rows: list[float] = []
    actual_count_rows: list[float] = []

    for hotel_name in current_hotels:
        hotel_window = model_z.loc[
            (model_z["Name"] == hotel_name)
            & model_z["time_group_id"].between(
                window_start, predictor_end_period
            )
        ].copy()

        if hotel_window.empty:
            raise RuntimeError(
                f"No predictor-window data for current hotel {hotel_name!r} "
                f"under covariate_mode={covariate_mode!r}."
            )

        period_x: list[np.ndarray] = []
        period_price: list[float] = []

        for hist_t, hist_group in hotel_window.groupby("time_group_id", sort=True):
            period_x.append(
                hist_group[feature_names]
                .astype(float)
                .mean(axis=0)
                .to_numpy(dtype=float)
            )
            period_price.append(
                _consistent_scalar(
                    hist_group["price"], "price", hotel_name, int(hist_t)
                )
            )

        # Current-period observed count is the validation target.
        current_hotel_rows = current_period.loc[
            current_period["Name"] == hotel_name
        ]
        actual_count = _consistent_scalar(
            current_hotel_rows["record_num"],
            "record_num",
            hotel_name,
            int(time_period),
        )

        x_rows.append(
            np.asarray(
                exponential_weighted_average(period_x, gamma),
                dtype=float,
            )
        )
        price_rows.append(
            float(exponential_weighted_average(period_price, gamma))
        )
        actual_count_rows.append(actual_count)

    x_test = np.vstack(x_rows)
    price_test = np.asarray(price_rows, dtype=float)
    actual_counts = np.asarray(actual_count_rows, dtype=float)

    if np.any(~np.isfinite(x_test)) or np.any(~np.isfinite(price_test)):
        raise ValueError(f"Invalid covariates or prices in test period {time_period}.")
    if np.any(actual_counts < 0) or actual_counts.sum() <= 0:
        raise ValueError(f"Invalid observed counts in test period {time_period}.")

    return (
        current_hotels,
        x_test,
        price_test,
        actual_counts,
        int(window_start),
        int(predictor_end_period),
    )


def validate_on_test_period(
    model_z: pd.DataFrame,
    pooled_model: dict[str, Any],
    feature_names: list[str],
    test_period: int = TEST_PERIOD,
    covariate_mode: str = VALIDATION_COVARIATE_MODE,
    plot_file: Path | None = None,
    model_name: str = "MNL",
) -> dict[str, Any]:
    """Apply pooled t1--t17 parameters to the dynamic t18 choice set."""
    source_features = list(pooled_model["feature_names"])
    if source_features != list(feature_names):
        raise ValueError(
            "The validation feature order differs from the pooled model order."
        )

    (
        hotel_order,
        x_test,
        price_test,
        actual_counts,
        predictor_window_start,
        predictor_window_end,
    ) = build_test_period_data(
        model_z=model_z,
        time_period=test_period,
        feature_names=feature_names,
        window_size=WINDOW_SIZE,
        gamma=WINDOW_GAMMA,
        covariate_mode=covariate_mode,
    )

    alpha = float(pooled_model["alpha"])
    beta = np.asarray(pooled_model["beta"], dtype=float)
    observed_probabilities = actual_counts / actual_counts.sum()

    metrics = evaluate_fit(
        x=x_test,
        price=price_test,
        alpha=alpha,
        beta=beta,
        y_actual=observed_probabilities,
    )
    predicted_probabilities = np.asarray(
        metrics["Predicted Probabilities"], dtype=float
    )

    hotel_predictions = pd.DataFrame(
        {
            "hotel_index": np.arange(len(hotel_order), dtype=int),
            "Name": hotel_order,
            "Observed_Count": actual_counts,
            "Observed_Probability": observed_probabilities,
            "Predicted_Probability": predicted_probabilities,
            "Absolute_Error": np.abs(predicted_probabilities - observed_probabilities),
            "Relative_Error": np.abs(predicted_probabilities - observed_probabilities)
            / (observed_probabilities + 1e-8),
        }
    )
    hotel_predictions["Observed_Rank"] = (
        hotel_predictions["Observed_Probability"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    hotel_predictions["Predicted_Rank"] = (
        hotel_predictions["Predicted_Probability"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    print(
        f"\n=== {model_name}: t={test_period} validation using pooled "
        f"t{TRAIN_PERIOD_START}--t{TRAIN_PERIOD_END} parameters ==="
    )
    print(f"Validation choice-set size: {len(hotel_order)}")
    print(
        f"Validation covariate mode: {covariate_mode}; predictor window: "
        f"t{predictor_window_start}-t{predictor_window_end}"
    )
    for key, value in metrics.items():
        if key != "Predicted Probabilities":
            print(f"  {key}: {value:.4f}" if np.isfinite(value) else f"  {key}: nan")

    if plot_file is None:
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", model_name).strip("_")
        plot_file = BASE_DIR / "figures" / f"mnl_{safe_name}_t18_validation.png"

    if PLOT_TEST_FIT:
        plot_file.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(
            observed_probabilities,
            label="Observed t18 share",
            marker="o",
            linewidth=2,
        )
        ax.plot(
            predicted_probabilities,
            label="Pooled-MNL t18 probability",
            marker="x",
            linewidth=2,
        )
        ax.set_title(
            f"{model_name}: t={test_period} validation "
            f"(pooled t{TRAIN_PERIOD_START}--t{TRAIN_PERIOD_END})"
        )
        ax.set_xlabel("Hotel index in the t18 choice set")
        ax.set_ylabel("Choice probability")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(plot_file, dpi=400, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    return {
        "test_period": int(test_period),
        "parameter_source": f"pooled_t{TRAIN_PERIOD_START}_t{TRAIN_PERIOD_END}",
        "training_period_start": TRAIN_PERIOD_START,
        "training_period_end": TRAIN_PERIOD_END,
        "choice_set_size": len(hotel_order),
        "covariate_mode": covariate_mode,
        "predictor_window_start": predictor_window_start,
        "predictor_window_end": predictor_window_end,
        "hotel_order": list(hotel_order),
        "observed_counts": actual_counts.copy(),
        "observed_probabilities": observed_probabilities.copy(),
        "predicted_probabilities": predicted_probabilities.copy(),
        "metrics": metrics,
        "hotel_predictions": hotel_predictions,
        "plot_file": str(plot_file),
        "model_name": model_name,
    }


# =============================================================================
# Direct probability AMEs for all predictors in an MNL specification
# =============================================================================
def calculate_predictor_ame(
    period_results: list[dict[str, Any]],
    feature_std: pd.Series,
    focal_hotel: str,
    time_gamma: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate focal-hotel probability AMEs for the latest raw observations.

    Estimated MNL coefficients apply to the rolling EWMA states. For a
    standardized predictor k, the effect of a one-unit increase in its EWMA
    state is ``beta_z[k] / sigma[k]``. The effect of increasing the latest raw
    observation by one unit is therefore

        latest_response_weight[a,t] * beta_z[k] / sigma[k].

    Price is not standardized, but it is EWMA-smoothed in the same way, so its
    latest-raw-observation effect is ``latest_response_weight[a,t] * alpha``.

    The period-specific own-probability marginal effect is

        ME[k,t] = P[a,t] * (1-P[a,t]) * current_raw_utility_effect[k,t].
    """
    if not 0.0 < time_gamma <= 1.0:
        raise ValueError("time_gamma must be in (0, 1].")
    if not period_results:
        raise ValueError("period_results cannot be empty.")

    focal_hotel = normalize_name(focal_hotel)
    period_records: list[dict[str, Any]] = []

    for result in period_results:
        time_period = int(result["time_period"])
        hotel_order = [normalize_name(name) for name in result["hotel_order"]]
        if focal_hotel not in hotel_order:
            continue

        probabilities = np.asarray(result["predicted_probabilities"], dtype=float)
        if probabilities.shape != (len(hotel_order),):
            raise ValueError(
                f"Period {time_period}: probability length differs from the "
                "dynamic choice-set size."
            )

        latest_state_weights = np.asarray(
            result.get("latest_state_weights"),
            dtype=float,
        )
        if latest_state_weights.shape != (len(hotel_order),):
            raise ValueError(
                f"Period {time_period}: latest-state-weight length differs "
                "from the dynamic choice-set size."
            )
        if not np.all(np.isfinite(latest_state_weights)):
            raise ValueError(
                f"Period {time_period}: latest-state weights contain invalid values."
            )

        focal_index = hotel_order.index(focal_hotel)
        p_focal = float(probabilities[focal_index])
        ewma_response_weight = float(latest_state_weights[focal_index])

        if not 0.0 < p_focal < 1.0:
            raise ValueError(
                f"Period {time_period}: invalid focal probability {p_focal}."
            )
        if not 0.0 <= ewma_response_weight <= 1.0:
            raise ValueError(
                f"Period {time_period}: invalid EWMA response weight "
                f"{ewma_response_weight}."
            )

        feature_names = list(result["feature_names"])
        beta = np.asarray(result["beta"], dtype=float)
        if beta.shape != (len(feature_names),):
            raise ValueError(
                f"Period {time_period}: beta length differs from feature-name length."
            )

        period_records.append(
            {
                "time_period": time_period,
                "p_focal": p_focal,
                "ewma_response_weight": ewma_response_weight,
                "alpha": float(result["alpha"]),
                "beta_map": dict(zip(feature_names, beta)),
                "feature_names": feature_names,
            }
        )

    if not period_records:
        raise ValueError(
            f"The focal hotel {focal_hotel!r} is absent from all fitted "
            "dynamic choice sets."
        )

    period_records.sort(key=lambda row: row["time_period"])
    latest_period = max(row["time_period"] for row in period_records)
    raw_weights = np.asarray(
        [time_gamma ** (latest_period - row["time_period"]) for row in period_records],
        dtype=float,
    )
    weights = raw_weights / raw_weights.sum()
    for row, weight in zip(period_records, weights):
        row["weight"] = float(weight)

    reference_features = period_records[0]["feature_names"]
    for row in period_records[1:]:
        if row["feature_names"] != reference_features:
            raise ValueError("The predictor order changes across fitted periods.")

    predictor_specs = [("alpha", "price", False)] + [
        (name, name, True) for name in reference_features
    ]

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for parameter_name, variable_name, is_standardized in predictor_specs:
        variable_period_rows: list[dict[str, Any]] = []

        if is_standardized:
            if variable_name not in feature_std.index:
                raise KeyError(
                    f"Predictor {variable_name!r} is absent from feature_std."
                )
            sigma = float(feature_std[variable_name])
            if not np.isfinite(sigma) or sigma <= 0.0:
                raise ValueError(
                    f"Invalid training standard deviation for "
                    f"{variable_name!r}: {sigma}."
                )
        else:
            sigma = 1.0

        for row in period_records:
            coefficient = (
                row["alpha"]
                if parameter_name == "alpha"
                else float(row["beta_map"][variable_name])
            )
            state_raw_utility_effect = coefficient / sigma
            current_raw_utility_effect = (
                row["ewma_response_weight"] * state_raw_utility_effect
            )
            p_factor = row["p_focal"] * (1.0 - row["p_focal"])
            me_probability = p_factor * current_raw_utility_effect
            weighted_me = row["weight"] * me_probability

            if not np.isclose(
                current_raw_utility_effect,
                row["ewma_response_weight"] * state_raw_utility_effect,
                rtol=1e-12,
                atol=1e-14,
            ):
                raise RuntimeError(
                    f"Period {row['time_period']}, variable {variable_name}: "
                    "EWMA effect conversion identity failed."
                )

            detail = {
                "Parameter": parameter_name,
                "Variable": variable_name,
                "time_period": row["time_period"],
                "weight": row["weight"],
                "P_focal": row["p_focal"],
                "P_factor": p_factor,
                "EWMA_Response_Weight": row["ewma_response_weight"],
                "Coefficient_Estimate": coefficient,
                "Training_Std": np.nan if not is_standardized else sigma,
                "State_Raw_Utility_Effect": state_raw_utility_effect,
                "Raw_Utility_Effect": current_raw_utility_effect,
                "ME_Choice_Probability": me_probability,
                "Weighted_ME_Choice_Probability": weighted_me,
            }
            detail_rows.append(detail)
            variable_period_rows.append(detail)

        ame = float(
            sum(
                r["Weighted_ME_Choice_Probability"]
                for r in variable_period_rows
            )
        )
        summary_rows.append(
            {
                "Parameter": parameter_name,
                "Variable": variable_name,
                "Scale_Interpretation": (
                    "per one-unit increase in the latest raw price observation"
                    if parameter_name == "alpha"
                    else (
                        "per one-unit increase in the latest raw observation "
                        f"of {variable_name}"
                    )
                ),
                "Average_Coefficient_Estimate": float(
                    sum(
                        r["weight"] * r["Coefficient_Estimate"]
                        for r in variable_period_rows
                    )
                ),
                "Average_EWMA_Response_Weight": float(
                    sum(
                        r["weight"] * r["EWMA_Response_Weight"]
                        for r in variable_period_rows
                    )
                ),
                "Average_State_Raw_Utility_Effect": float(
                    sum(
                        r["weight"] * r["State_Raw_Utility_Effect"]
                        for r in variable_period_rows
                    )
                ),
                "Average_Raw_Utility_Effect": float(
                    sum(
                        r["weight"] * r["Raw_Utility_Effect"]
                        for r in variable_period_rows
                    )
                ),
                "Average_Focal_Probability": float(
                    sum(
                        r["weight"] * r["P_focal"]
                        for r in variable_period_rows
                    )
                ),
                "AME_Choice_Probability": ame,
                "AME_Choice_Percentage_Point": 100.0 * ame,
                "N_Periods": len(variable_period_rows),
                "First_Period": min(
                    r["time_period"] for r in variable_period_rows
                ),
                "Last_Period": max(
                    r["time_period"] for r in variable_period_rows
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)

    identity = (
        detail_df.groupby(["Parameter", "Variable"], as_index=False)[
            "Weighted_ME_Choice_Probability"
        ]
        .sum()
        .rename(columns={"Weighted_ME_Choice_Probability": "AME_check"})
    )
    check = summary_df.merge(
        identity,
        on=["Parameter", "Variable"],
        how="left",
        validate="one_to_one",
    )
    max_difference = float(
        np.max(np.abs(check["AME_Choice_Probability"] - check["AME_check"]))
    )
    if max_difference > 1e-12:
        raise RuntimeError(
            f"Predictor AME identity check failed; max difference={max_difference}."
        )

    return summary_df, detail_df


# =============================================================================
# IPEA average marginal effects
# =============================================================================
def calculate_ipea_ame(
    tau_df: pd.DataFrame,
    period_results: list[dict[str, Any]],
    feature_std: pd.Series,
    focal_hotel: str,
    mediator_name: str = MEDIATOR_NAME,
    time_gamma: float = AME_TIME_GAMMA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate latest-raw-observation service-element effects and focal AMEs.

    ``beta_z / sigma`` describes a one-unit change in an EWMA state. The
    corresponding effect of a one-unit change in the latest raw observation is
    obtained by multiplying by the focal hotel's period-specific EWMA response
    weight. Both state-scale and latest-raw-observation effects are exported.
    """
    if not 0.0 < time_gamma <= 1.0:
        raise ValueError("time_gamma must be in (0, 1].")

    validate_required_columns(tau_df, ["ES", "Tau_Hat_Mean"], "tau data")
    tau_work = tau_df.copy()
    tau_work["ES"] = tau_work["ES"].map(normalize_es_name)

    if tau_work["ES"].duplicated().any():
        duplicates = sorted(
            tau_work.loc[tau_work["ES"].duplicated(keep=False), "ES"].unique()
        )
        raise ValueError(f"Duplicate ES identifiers in tau table: {duplicates}")

    tau_work["Tau_Hat_Mean"] = pd.to_numeric(
        tau_work["Tau_Hat_Mean"], errors="coerce"
    )
    invalid_tau = tau_work.loc[
        ~np.isfinite(tau_work["Tau_Hat_Mean"]), "ES"
    ].tolist()
    if invalid_tau:
        raise ValueError(f"Invalid Tau_Hat_Mean for ES variables: {invalid_tau}")

    if mediator_name not in feature_std.index:
        raise KeyError(f"{mediator_name!r} is absent from feature_std.")
    sigma_total_score = float(feature_std[mediator_name])
    if not np.isfinite(sigma_total_score) or sigma_total_score <= 0.0:
        raise ValueError(
            f"Invalid standard deviation for {mediator_name}: {sigma_total_score}"
        )

    focal_name = normalize_name(focal_hotel)
    period_records: list[dict[str, Any]] = []

    for result in period_results:
        time_period = int(result["time_period"])
        feature_names = [
            normalize_es_name(name) for name in result["feature_names"]
        ]
        beta = np.asarray(result["beta"], dtype=float)

        if beta.shape != (len(feature_names),):
            raise ValueError(
                f"Period {time_period}: beta length differs from feature_names length."
            )

        beta_map = dict(zip(feature_names, beta))
        if mediator_name not in beta_map:
            raise KeyError(
                f"Period {time_period}: mediator {mediator_name!r} is absent from MNL."
            )

        hotel_order = [normalize_name(name) for name in result["hotel_order"]]

        # Under Choice-set Rule B, the focal hotel contributes only in periods
        # in which it is actually in the current-period choice set.
        if focal_name not in hotel_order:
            continue

        probabilities = np.asarray(result["predicted_probabilities"], dtype=float)
        if probabilities.shape != (len(hotel_order),):
            raise ValueError(
                f"Period {time_period}: probability vector length is inconsistent."
            )
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-10):
            raise ValueError(
                f"Period {time_period}: probabilities sum to {probabilities.sum()}."
            )

        latest_state_weights = np.asarray(
            result.get("latest_state_weights"),
            dtype=float,
        )
        if latest_state_weights.shape != (len(hotel_order),):
            raise ValueError(
                f"Period {time_period}: latest-state-weight length is inconsistent."
            )
        if not np.all(np.isfinite(latest_state_weights)):
            raise ValueError(
                f"Period {time_period}: latest-state weights contain invalid values."
            )

        focal_index = hotel_order.index(focal_name)
        p_focal = float(probabilities[focal_index])
        ewma_response_weight = float(latest_state_weights[focal_index])

        if not 0.0 < p_focal < 1.0:
            raise ValueError(
                f"Period {time_period}: invalid focal probability {p_focal}."
            )
        if not 0.0 <= ewma_response_weight <= 1.0:
            raise ValueError(
                f"Period {time_period}: invalid EWMA response weight "
                f"{ewma_response_weight}."
            )

        period_records.append(
            {
                "time_period": time_period,
                "p_focal": p_focal,
                "beta_map": beta_map,
                "choice_set_size": len(hotel_order),
                "ewma_response_weight": ewma_response_weight,
            }
        )

    if not period_records:
        raise ValueError(
            f"The focal hotel {focal_name!r} is absent from all valid "
            "current-period choice sets."
        )

    period_records.sort(key=lambda row: row["time_period"])
    latest_period = max(row["time_period"] for row in period_records)
    raw_weights = np.asarray(
        [time_gamma ** (latest_period - row["time_period"]) for row in period_records],
        dtype=float,
    )
    normalized_weights = raw_weights / raw_weights.sum()
    for row, weight in zip(period_records, normalized_weights):
        row["weight"] = float(weight)

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for _, tau_row in tau_work.iterrows():
        es_name = normalize_es_name(tau_row["ES"])
        tau_mean = float(tau_row["Tau_Hat_Mean"])
        es_period_rows: list[dict[str, Any]] = []

        for record in period_records:
            beta_map = record["beta_map"]
            beta_total_z = float(beta_map[mediator_name])
            beta_es_z = float(beta_map.get(es_name, 0.0))
            ewma_response_weight = float(record["ewma_response_weight"])

            if es_name in beta_map:
                if es_name not in feature_std.index:
                    raise KeyError(
                        f"{es_name!r} enters MNL but is absent from feature_std."
                    )
                sigma_es = float(feature_std[es_name])
                if not np.isfinite(sigma_es) or sigma_es <= 0.0:
                    raise ValueError(
                        f"Invalid original-scale standard deviation for "
                        f"{es_name}: {sigma_es}"
                    )
                direct_effect_state_raw = beta_es_z / sigma_es
            else:
                sigma_es = np.nan
                direct_effect_state_raw = 0.0

            zeta_total_score_state_raw = beta_total_z / sigma_total_score

            # Convert EWMA-state effects into effects of a one-unit increase in
            # the latest raw period observation. Downstream code must use these
            # adjusted fields directly and must not apply the EWMA weight again.
            direct_effect_raw = (
                ewma_response_weight * direct_effect_state_raw
            )
            zeta_total_score_raw = (
                ewma_response_weight * zeta_total_score_state_raw
            )
            mediated_effect_raw = zeta_total_score_raw * tau_mean
            eff_raw = direct_effect_raw + mediated_effect_raw

            if not np.isclose(
                direct_effect_raw,
                ewma_response_weight * direct_effect_state_raw,
                rtol=1e-12,
                atol=1e-14,
            ):
                raise RuntimeError(
                    f"{es_name}, period {record['time_period']}: direct-effect "
                    "EWMA conversion identity failed."
                )
            if not np.isclose(
                zeta_total_score_raw,
                ewma_response_weight * zeta_total_score_state_raw,
                rtol=1e-12,
                atol=1e-14,
            ):
                raise RuntimeError(
                    f"{es_name}, period {record['time_period']}: mediator-effect "
                    "EWMA conversion identity failed."
                )
            if not np.isclose(
                mediated_effect_raw,
                zeta_total_score_raw * tau_mean,
                rtol=1e-12,
                atol=1e-14,
            ):
                raise RuntimeError(
                    f"{es_name}, period {record['time_period']}: mediated-effect "
                    "identity failed."
                )

            p_focal = float(record["p_focal"])
            p_factor = p_focal * (1.0 - p_focal)
            me_probability = p_factor * eff_raw
            weighted_me = float(record["weight"]) * me_probability

            period_row = {
                "ES": es_name,
                "Tau_Hat_Mean": tau_mean,
                "time_period": int(record["time_period"]),
                "choice_set_size": int(record["choice_set_size"]),
                "weight": float(record["weight"]),
                "P_focal": p_focal,
                "P_factor": p_factor,
                "ewma_response_weight": ewma_response_weight,
                "beta_ES_standardized": beta_es_z,
                "sigma_ES": sigma_es,
                "direct_effect_state_raw": direct_effect_state_raw,
                "direct_effect_raw": direct_effect_raw,
                "beta_total_score_standardized": beta_total_z,
                "sigma_total_score": sigma_total_score,
                "zeta_total_score_state_raw": zeta_total_score_state_raw,
                "zeta_total_score_raw": zeta_total_score_raw,
                "mediated_effect_raw": mediated_effect_raw,
                "Eff_raw": eff_raw,
                "ME_probability": me_probability,
                "weighted_ME_probability": weighted_me,
            }
            detail_rows.append(period_row)
            es_period_rows.append(period_row)

        ame = float(
            sum(row["weighted_ME_probability"] for row in es_period_rows)
        )
        avg_ewma_response = float(
            sum(
                row["weight"] * row["ewma_response_weight"]
                for row in es_period_rows
            )
        )
        avg_direct_state = float(
            sum(
                row["weight"] * row["direct_effect_state_raw"]
                for row in es_period_rows
            )
        )
        avg_direct = float(
            sum(
                row["weight"] * row["direct_effect_raw"]
                for row in es_period_rows
            )
        )
        avg_zeta_state = float(
            sum(
                row["weight"] * row["zeta_total_score_state_raw"]
                for row in es_period_rows
            )
        )
        avg_zeta = float(
            sum(
                row["weight"] * row["zeta_total_score_raw"]
                for row in es_period_rows
            )
        )
        avg_mediated = float(
            sum(
                row["weight"] * row["mediated_effect_raw"]
                for row in es_period_rows
            )
        )
        avg_eff = float(
            sum(row["weight"] * row["Eff_raw"] for row in es_period_rows)
        )
        avg_probability = float(
            sum(row["weight"] * row["P_focal"] for row in es_period_rows)
        )
        avg_p_factor = float(
            sum(row["weight"] * row["P_factor"] for row in es_period_rows)
        )

        summary_rows.append(
            {
                "ES": es_name,
                "Average_EWMA_Response_Weight": avg_ewma_response,
                "Average_Direct_Effect_State_Raw": avg_direct_state,
                "Average_Direct_Effect_Raw": avg_direct,
                "Average_Zeta_Total_Score_State_Raw": avg_zeta_state,
                "Average_Zeta_Total_Score_Raw": avg_zeta,
                "Average_Mediated_Effect_Raw": avg_mediated,
                "Average_Eff_Raw": avg_eff,
                "Average_Focal_Probability": avg_probability,
                "Average_P_Factor": avg_p_factor,
                "AME_Choice_Probability": ame,
                "AME_Choice_Percentage_Point": 100.0 * ame,
                "N_Periods": len(es_period_rows),
                "First_Period": min(
                    row["time_period"] for row in es_period_rows
                ),
                "Last_Period": max(
                    row["time_period"] for row in es_period_rows
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)

    result_df = tau_work.merge(
        summary_df,
        on="ES",
        how="left",
        validate="one_to_one",
    )

    if result_df["AME_Choice_Probability"].isna().any():
        invalid = result_df.loc[
            result_df["AME_Choice_Probability"].isna(), "ES"
        ].tolist()
        raise RuntimeError(f"AME is missing for ES variables: {invalid}")

    # Independent numerical identity checks.
    ame_check = (
        detail_df.groupby("ES", as_index=False)["weighted_ME_probability"]
        .sum()
        .rename(columns={"weighted_ME_probability": "AME_check"})
    )
    check = result_df[["ES", "AME_Choice_Probability"]].merge(
        ame_check,
        on="ES",
        how="left",
        validate="one_to_one",
    )
    max_difference = float(
        np.max(np.abs(check["AME_Choice_Probability"] - check["AME_check"]))
    )
    if max_difference > 1e-12:
        raise RuntimeError(
            f"AME identity check failed; max difference={max_difference}"
        )

    direct_identity_error = float(
        np.max(
            np.abs(
                detail_df["direct_effect_raw"]
                - detail_df["ewma_response_weight"]
                * detail_df["direct_effect_state_raw"]
            )
        )
    )
    zeta_identity_error = float(
        np.max(
            np.abs(
                detail_df["zeta_total_score_raw"]
                - detail_df["ewma_response_weight"]
                * detail_df["zeta_total_score_state_raw"]
            )
        )
    )
    mediated_identity_error = float(
        np.max(
            np.abs(
                detail_df["mediated_effect_raw"]
                - detail_df["zeta_total_score_raw"]
                * detail_df["Tau_Hat_Mean"]
            )
        )
    )
    if max(
        direct_identity_error,
        zeta_identity_error,
        mediated_identity_error,
    ) > 1e-12:
        raise RuntimeError(
            "IPEA effect conversion identity check failed: "
            f"direct={direct_identity_error}, zeta={zeta_identity_error}, "
            f"mediated={mediated_identity_error}."
        )

    return result_df, detail_df


# =============================================================================
# Output
# =============================================================================
def _period_metrics_frame(
    period_results: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in period_results:
        row = {
            "time_period": result["time_period"],
            "choice_set_size": len(result["hotel_order"]),
            "hotel_order": " | ".join(result["hotel_order"]),
        }
        row.update(
            {
                key: value
                for key, value in result["metrics"].items()
                if key != "Predicted Probabilities"
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _test_metric_row(
    model_name: str,
    test_validation: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "Model": model_name,
        "test_period": test_validation["test_period"],
        "parameter_source": test_validation["parameter_source"],
        "training_period_start": test_validation["training_period_start"],
        "training_period_end": test_validation["training_period_end"],
        "choice_set_size": test_validation["choice_set_size"],
        "covariate_mode": test_validation["covariate_mode"],
        "predictor_window_start": test_validation["predictor_window_start"],
        "predictor_window_end": test_validation["predictor_window_end"],
    }
    row.update(
        {
            key: value
            for key, value in test_validation["metrics"].items()
            if key != "Predicted Probabilities"
        }
    )
    return row



def build_t18_model_comparison(
    model_outputs: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Align the t18 hotel-level predictions from all model specifications.

    The first model in ``model_outputs`` defines the plotting order. Every
    model must contain exactly the same t18 hotel choice set, and the observed
    probabilities must agree after alignment by hotel name.
    """
    if not model_outputs:
        raise ValueError("model_outputs cannot be empty.")

    model_names = list(model_outputs.keys())
    reference_model = model_names[0]
    reference_predictions = model_outputs[reference_model]["test_validation"][
        "hotel_predictions"
    ].copy()

    required_columns = {
        "hotel_index",
        "Name",
        "Observed_Count",
        "Observed_Probability",
        "Predicted_Probability",
        "Absolute_Error",
    }
    validate_required_columns(
        reference_predictions,
        required_columns,
        f"{reference_model} t18 predictions",
    )

    if reference_predictions["Name"].duplicated().any():
        duplicates = reference_predictions.loc[
            reference_predictions["Name"].duplicated(keep=False), "Name"
        ].tolist()
        raise ValueError(
            f"Duplicate hotel names in {reference_model} t18 predictions: {duplicates}"
        )

    reference_predictions = reference_predictions.sort_values(
        "hotel_index"
    ).reset_index(drop=True)
    hotel_order = reference_predictions["Name"].tolist()
    reference_names = set(hotel_order)

    fit_comparison = reference_predictions[
        ["hotel_index", "Name", "Observed_Count", "Observed_Probability"]
    ].copy()
    absolute_error_rows: list[pd.DataFrame] = []

    for model_name in model_names:
        predictions = model_outputs[model_name]["test_validation"][
            "hotel_predictions"
        ].copy()
        validate_required_columns(
            predictions,
            required_columns,
            f"{model_name} t18 predictions",
        )

        if predictions["Name"].duplicated().any():
            duplicates = predictions.loc[
                predictions["Name"].duplicated(keep=False), "Name"
            ].tolist()
            raise ValueError(
                f"Duplicate hotel names in {model_name} t18 predictions: {duplicates}"
            )

        model_names_set = set(predictions["Name"])
        if model_names_set != reference_names:
            missing = sorted(reference_names - model_names_set)
            extra = sorted(model_names_set - reference_names)
            raise ValueError(
                f"The t18 choice set for {model_name} differs from "
                f"{reference_model}. Missing={missing}; extra={extra}."
            )

        aligned = predictions.set_index("Name").reindex(hotel_order).reset_index()
        if aligned["Predicted_Probability"].isna().any():
            raise RuntimeError(
                f"Failed to align t18 predictions for model {model_name}."
            )

        observed_aligned = aligned["Observed_Probability"].to_numpy(dtype=float)
        observed_reference = fit_comparison["Observed_Probability"].to_numpy(
            dtype=float
        )
        if not np.allclose(
            observed_aligned,
            observed_reference,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                f"Observed t18 probabilities differ across model data for {model_name}."
            )

        fit_comparison[f"{model_name}_Predicted_Probability"] = aligned[
            "Predicted_Probability"
        ].to_numpy(dtype=float)

        error_frame = pd.DataFrame(
            {
                "hotel_index": fit_comparison["hotel_index"].to_numpy(dtype=int),
                "Name": hotel_order,
                "Model": model_name,
                "Observed_Probability": observed_reference,
                "Predicted_Probability": aligned[
                    "Predicted_Probability"
                ].to_numpy(dtype=float),
                "Absolute_Prediction_Error": aligned["Absolute_Error"].to_numpy(
                    dtype=float
                ),
            }
        )
        absolute_error_rows.append(error_frame)

    absolute_errors = pd.concat(absolute_error_rows, ignore_index=True)
    error_summary = (
        absolute_errors.groupby("Model", sort=False)["Absolute_Prediction_Error"]
        .agg(
            N="count",
            Mean="mean",
            Median="median",
            Std="std",
            Min="min",
            Q1=lambda values: values.quantile(0.25),
            Q3=lambda values: values.quantile(0.75),
            Max="max",
        )
        .reset_index()
    )
    error_summary["MAE"] = error_summary["Mean"]

    return fit_comparison, absolute_errors, error_summary



# Unified dimensions and typography for figures that will be
# vertically combined in LaTeX.
T18_FIGSIZE = (13, 5.5)

T18_AXIS_LABEL_FONTSIZE = 14
T18_TICK_FONTSIZE = 12
T18_LEGEND_FONTSIZE = 12

# Use identical margins so that the actual plotting areas are comparable.
T18_SUBPLOT_ADJUST = {
    "left": 0.10,
    "right": 0.98,
    "bottom": 0.10,
    "top": 0.96,
}


def plot_t18_model_comparison(
    model_outputs: dict[str, dict[str, Any]],
    output_dir: Path,
    test_period: int = TEST_PERIOD,
) -> dict[str, Any]:
    """Create one fit-comparison plot and one absolute-error boxplot."""
    fit_comparison, absolute_errors, error_summary = (
        build_t18_model_comparison(model_outputs)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    fit_plot_file = output_dir / f"mnl_models_t{test_period}_fit_comparison.png"
    error_plot_file = (
        output_dir / f"mnl_models_t{test_period}_absolute_error_boxplot.png"
    )

    x = fit_comparison["hotel_index"].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        x,
        fit_comparison["Observed_Probability"],
        label=T18_PLOT_LABELS.get(
            "Observed", f"Observed t{test_period} share"
        ),
        marker="o",
        linestyle="-",
        linewidth=2.5,
        markersize=4,
    )

    markers = ["x", "s", "^", "D", "v", "P", "*"]
    for model_index, model_name in enumerate(model_outputs.keys()):
        ax.plot(
            x,
            fit_comparison[f"{model_name}_Predicted_Probability"],
            label=T18_PLOT_LABELS.get(
                model_name, f"{model_name} prediction"
            ),
            marker=markers[model_index % len(markers)],
            linestyle=T18_MODEL_LINESTYLES.get(model_name, "-"),
            linewidth=1.8,
            markersize=4,
        )

    ax.set_xlabel(
        "Hotel index",
        fontsize=T18_AXIS_LABEL_FONTSIZE,
    )
    ax.set_ylabel(
        "Choice probability",
        fontsize=T18_AXIS_LABEL_FONTSIZE,
    )
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax.tick_params(
        axis="both",
        labelsize=T18_TICK_FONTSIZE,
    )
    
    ax.legend(
        ncol=2,
        fontsize=T18_LEGEND_FONTSIZE,
    )
    
    ax.grid(True, alpha=0.3)
    
    fig.subplots_adjust(**T18_SUBPLOT_ADJUST)
    fig.savefig(
        fit_plot_file,
        dpi=400,
    )
    plt.show()
    plt.close(fig)

    model_names = list(model_outputs.keys())
    box_tick_labels = [
        T18_PLOT_LABELS.get(model_name, model_name)
        for model_name in model_names
    ]
    box_data = [
        absolute_errors.loc[
            absolute_errors["Model"] == model_name,
            "Absolute_Prediction_Error",
        ].to_numpy(dtype=float)
        for model_name in model_names
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    try:
        ax.boxplot(
            box_data,
            tick_labels=box_tick_labels,
            showmeans=True,
            meanline=False,
        )
    except TypeError:
        # Compatibility with Matplotlib versions earlier than 3.9.
        ax.boxplot(
            box_data,
            labels=box_tick_labels,
            showmeans=True,
            meanline=False,
        )

    
    ax.set_ylabel(
        "Absolute prediction error",
        fontsize=T18_AXIS_LABEL_FONTSIZE,
    )
    
    ax.tick_params(
        axis="both",
        labelsize=T18_TICK_FONTSIZE,
    )
    
    ax.grid(True, axis="y", alpha=0.3)
    
    fig.subplots_adjust(**T18_SUBPLOT_ADJUST)
    fig.savefig(
        error_plot_file,
        dpi=400,
    )
    plt.show()
    plt.close(fig)

    print(f"Combined t{test_period} fit plot saved to: {fit_plot_file}")
    print(f"Absolute-error boxplot saved to: {error_plot_file}")
    print("\nAbsolute prediction error summary:")
    print(error_summary.to_string(index=False))

    return {
        "fit_comparison": fit_comparison,
        "absolute_errors": absolute_errors,
        "error_summary": error_summary,
        "fit_plot_file": str(fit_plot_file),
        "error_plot_file": str(error_plot_file),
    }




def save_all_results(
    output_file: Path,
    model_outputs: dict[str, dict[str, Any]],
    ipea_summary: pd.DataFrame,
    ipea_details: pd.DataFrame,
    t18_comparison: dict[str, Any],
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    test_comparison = pd.DataFrame(
        [
            _test_metric_row(model_name, bundle["test_validation"])
            for model_name, bundle in model_outputs.items()
        ]
    )

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        test_comparison.to_excel(
            writer, sheet_name="Model_comparison_t18", index=False
        )
        t18_comparison["fit_comparison"].to_excel(
            writer, sheet_name="t18_fit_comparison", index=False
        )
        t18_comparison["absolute_errors"].to_excel(
            writer, sheet_name="t18_absolute_errors", index=False
        )
        t18_comparison["error_summary"].to_excel(
            writer, sheet_name="t18_error_summary", index=False
        )
        ipea_summary.to_excel(writer, sheet_name="IPEA_summary", index=False)
        ipea_details.to_excel(writer, sheet_name="IPEA_details", index=False)

        for model_name, bundle in model_outputs.items():
            prefix = model_name[:10]
            period_metrics = _period_metrics_frame(bundle["period_results"])
            scaler = pd.DataFrame(
                {
                    "Feature": bundle["feature_mean"].index,
                    "Training_Mean": bundle["feature_mean"].values,
                    "Training_Std": bundle["feature_std"]
                    .reindex(bundle["feature_mean"].index)
                    .values,
                }
            )

            bundle["predictor_ame"].to_excel(
                writer, sheet_name=f"{prefix}_predictor_AME", index=False
            )
            bundle["predictor_ame_details"].to_excel(
                writer, sheet_name=f"{prefix}_AME_details", index=False
            )
            period_metrics.to_excel(
                writer, sheet_name=f"{prefix}_period_metrics", index=False
            )
            bundle["pooled_model"]["parameter_summary"].to_excel(
                writer, sheet_name=f"{prefix}_pooled_params", index=False
            )
            bundle["pooled_model"]["pooled_fit_summary"].to_excel(
                writer, sheet_name=f"{prefix}_pooled_fit", index=False
            )
            bundle["test_validation"]["hotel_predictions"].to_excel(
                writer, sheet_name=f"{prefix}_t18_predictions", index=False
            )
            scaler.to_excel(
                writer, sheet_name=f"{prefix}_standardization", index=False
            )
            bundle["vif"].to_excel(
                writer, sheet_name=f"{prefix}_VIF", index=False
            )

    print(f"Results saved to: {output_file}")


# =============================================================================
# Main
# =============================================================================
#def main() -> None:
if REG_LAMBDA != 0.0:
    raise ValueError("This script requires REG_LAMBDA=0.0.")

np.random.seed(SEED)
random.seed(SEED)

merged_data = load_and_merge_data()
model_outputs: dict[str, dict[str, Any]] = {}


T18_PLOT_LABELS: dict[str, str] = {
    "Observed": "Observed",
    "Basic": "Basic characteristics",
    "ES": "With service elements",
    "Att": "With attributes",
}

# Per-model line styles in the combined t18 fit figure.
# Basic and Att are displayed as dashed lines, as requested.
T18_MODEL_LINESTYLES: dict[str, str] = {
    "Basic": "--",
    "ES": "-",
    "Att": "--",
}


for model_name, feature_names in MODEL_SPECS.items():
    print("\n" + "=" * 80)
    print(f"Estimating pooled model: {model_name}")
    print("Features:", feature_names)

    model_raw, model_z, feature_mean, feature_std = (
        prepare_standardized_model_data(
            merged_data,
            feature_names=list(feature_names),
        )
    )

    print(f"Eligible observations ({model_name}): {len(model_z)}")
    vif = calculate_vif(model_z, list(feature_names))
    print(vif.sort_values("VIF", ascending=False).head(20))

    pooled_model = fit_pooled_model(
        model_z=model_z,
        feature_names=list(feature_names),
        model_name=model_name,
    )
    period_results = pooled_model["period_results"]

    plot_file = (
        BASE_DIR
        / "figures"
        / f"mnl_{model_name.lower()}_pooled_t18_validation.png"
    )
    test_validation = validate_on_test_period(
        model_z=model_z,
        pooled_model=pooled_model,
        feature_names=list(feature_names),
        test_period=TEST_PERIOD,
        covariate_mode=VALIDATION_COVARIATE_MODE,
        plot_file=plot_file,
        model_name=model_name,
    )

    predictor_ame, predictor_ame_details = calculate_predictor_ame(
        period_results=period_results,
        feature_std=feature_std,
        focal_hotel=FOCAL_HOTEL,
        time_gamma=AME_TIME_GAMMA,
    )

    model_outputs[model_name] = {
        "model_raw": model_raw,
        "model_z": model_z,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "vif": vif,
        "pooled_model": pooled_model,
        "period_results": period_results,
        "test_validation": test_validation,
        "predictor_ame": predictor_ame,
        "predictor_ame_details": predictor_ame_details,
    }

if PLOT_T18_MODEL_COMPARISON:
    t18_comparison = plot_t18_model_comparison(
        model_outputs=model_outputs,
        output_dir=BASE_DIR / "figures",
        test_period=TEST_PERIOD,
    )
else:
    fit_comparison, absolute_errors, error_summary = (
        build_t18_model_comparison(model_outputs)
    )
    t18_comparison = {
        "fit_comparison": fit_comparison,
        "absolute_errors": absolute_errors,
        "error_summary": error_summary,
        "fit_plot_file": None,
        "error_plot_file": None,
    }

es_output = model_outputs["ES"]
tau_df = pd.read_excel(TAU_FILE, sheet_name=TAU_SHEET)
ipea_summary, ipea_details = calculate_ipea_ame(
    tau_df=tau_df,
    period_results=es_output["period_results"],
    feature_std=es_output["feature_std"],
    focal_hotel=FOCAL_HOTEL,
    mediator_name=MEDIATOR_NAME,
    time_gamma=AME_TIME_GAMMA,
)

save_all_results(
    output_file=OUTPUT_FILE,
    model_outputs=model_outputs,
    ipea_summary=ipea_summary,
    ipea_details=ipea_details,
    t18_comparison=t18_comparison,
)

print("\nBasic-model direct probability AMEs (pooled MNL):")
print(model_outputs["Basic"]["predictor_ame"].to_string(index=False))

display_columns = [
    column
    for column in [
        "ES",
        "Tau_Hat_Mean",
        "Average_EWMA_Response_Weight",
        "Average_Direct_Effect_State_Raw",
        "Average_Direct_Effect_Raw",
        "Average_Zeta_Total_Score_Raw",
        "Average_Mediated_Effect_Raw",
        "Average_Eff_Raw",
        "AME_Choice_Probability",
        "AME_Choice_Percentage_Point",
        "N_Periods",
    ]
    if column in ipea_summary.columns
]
print("\nIPEA service-element AMEs (pooled MNL):")
print(ipea_summary[display_columns].head(20).to_string(index=False))


