# Effect Estimation and Improvement-Difficulty Modules

This folder contains the three analytical modules that follow the review-processing and service-element construction stages.

## 1. Position in the full workflow

```text
review_service_matrix.xlsx
        |
        +-------------------------------+
        |                               |
        v                               v
GRF / Kano                         Pooled MNL
11_grf_kano_estimation.R           12_pooled_mnl.py
        |                               |
        |                               |
        +-------------+-----------------+
                      |
                      v
             service-element effects
                      |
                      v
          IPEA / multi-objective model

negative-rate transition table
        |
        v
13_ibpa_improvement_difficulty.py
        |
        v
CM_m = final posterior h
Cost_m = 1 / CM_m
        |
        v
multi-objective implementation-difficulty objective
```

## 2. Recommended data folders

```text
data/
├── 06_choice_satisfaction/
│   ├── competitive_hotel_reviews.xlsx
│   └── grf_input.xlsx
│
├── 07_effect_estimation/
│   ├── grf_kano/
│   │   ├── grf_input_diagnostics.csv
│   │   ├── tau_estimates_raw.rds
│   │   ├── satisfaction_shape_summary.csv
│   │   ├── shape_classification_diagnostics.csv
│   │   ├── satisfaction_function_parameters.csv
│   │   ├── satisfaction_shape_and_parameters.csv
│   │   └── moo_satisfaction_parameters.csv
│   │
│   └── mnl/
│       └── pooled_mnl_results.xlsx
│
└── 08_improvement_difficulty/
    ├── ibpa_input.xlsx
    ├── ibpa_improvement_difficulty.xlsx
    └── ibpa_improvement_difficulty.csv
```

## 3. GRF / Kano module

### Script

`11_grf_kano_estimation.R`

### Input

`data/06_choice_satisfaction/grf_input.xlsx`

This corresponds to the working research file:

```text
S4-grf.xlsx
```

### Main calculations

For each service element `ES_m`, the script:

1. estimates the conditional marginal rating effect with a generalized random forest;
2. obtains the global average marginal effect `Tau_Hat_Mean`;
3. smooths the estimated marginal-effect profile over observed service performance;
4. assesses turning-point stability;
5. compares broad lower- and upper-performance marginal effects;
6. classifies the dominant Kano-type satisfaction shape;
7. recovers parsimonious satisfaction-function parameters for downstream MOO.

### Important outputs

| File | Meaning |
|---|---|
| `tau_estimates_raw.rds` | Observation-level GRF estimates for all service elements. |
| `satisfaction_shape_summary.csv` | Average effect, lower/upper effects, bootstrap stability, and Kano-type classification. |
| `satisfaction_function_parameters.csv` | Parameters of the fitted Attractive, Must-be, or One-dimensional satisfaction functions. |
| `satisfaction_shape_and_parameters.csv` | Combined diagnostic and parameter table. |
| `moo_satisfaction_parameters.csv` | Compact model-ready satisfaction-function table used by the optimization module. |

### Reproducibility note

The original notebook selected several control blocks by column position after dummy expansion. The release script currently preserves those positional selectors exactly because they reproduce the research run. Before final public release, it is preferable to replace them with explicit column names once the schema of the released `grf_input.xlsx` is fixed.

## 4. Pooled MNL module

### Script

`12_pooled_mnl.py`

The uploaded MNL implementation is already relatively complete and is therefore preserved as the main estimation script.

Its workflow includes:

1. visual-feature aggregation and PCA;
2. Lasso-based variable screening;
3. stability selection using 10 random seeds and 1,000 bootstrap samples per seed;
4. construction of dynamic period-specific choice sets;
5. three-period rolling EWMA covariates;
6. pooled count-weighted MNL estimation over training periods;
7. parametric bootstrap inference;
8. held-out-period validation;
9. direct and rating-mediated service-element effects;
10. focal-hotel average marginal effects used by IPEA.

The MNL module consumes the GRF average effect `Tau_Hat_Mean` when constructing the mediated effect of a service element through overall rating.

### Recommended output

```text
data/07_effect_estimation/mnl/pooled_mnl_results.xlsx
```

The original working output was:

```text
IPEA_effect_results_pooled_MNL-stab2.xlsx
```

## 5. IBPA implementation-difficulty module

### Script

`13_ibpa_improvement_difficulty.py`

### Input

The original notebook reads two matrices from the `neg_ratio` worksheet:

```text
x: B21:DS37
n: B2:DS18
```

Both matrices have periods in rows and service elements in columns.

For public release, the same data should be placed in:

```text
data/08_improvement_difficulty/ibpa_input.xlsx
```

### Iterative update

The initial value is

```text
h_t = x_t / n_t
```

for nonzero `n_t`. At each iteration, the mean and variance of the current `h` values are used to update the Beta prior parameters `alpha` and `beta`, followed by

```text
h_t = (x_t + alpha) / (n_t + alpha + beta).
```

### Final implementation-difficulty measure

The original notebook reports the last element of the converged posterior vector:

```text
CM_m = h_final = h[-1].
```

The implementation difficulty used downstream is then

```text
Cost_m = 1 / CM_m.
```

The cleaned script writes both quantities explicitly:

| Column | Meaning |
|---|---|
| `CM` | Final-period posterior `h` from IBPA; larger values indicate greater improvability. |
| `Cost` | `1 / CM`; larger values indicate greater implementation difficulty. |
| `alpha`, `beta` | Final empirical-Bayes prior parameters. |
| `MSE` | Difference between raw `x/n` values and posterior `h` values. |
| `converged` | Whether the iterative parameter updates met the tolerance criterion. |
| `status` | Diagnostic termination status. |

The workbook also contains an `h_by_period` sheet so that readers can inspect the full posterior sequence rather than only the final value.

## 6. Legacy-to-release filename correspondence

| Release filename | Original working filename |
|---|---|
| `06_choice_satisfaction/grf_input.xlsx` | `S4-grf.xlsx` |
| `07_effect_estimation/grf_kano/*` | `grf_shape_outputs/*` |
| `07_effect_estimation/mnl/pooled_mnl_results.xlsx` | `IPEA_effect_results_pooled_MNL-stab2.xlsx` |
| `08_improvement_difficulty/ibpa_input.xlsx` | `shap+TD_result.xlsx` (`neg_ratio` sheet) |
| `08_improvement_difficulty/ibpa_improvement_difficulty.xlsx` | Previously produced interactively in `IBPA.ipynb` |

## 7. Recommended execution order

GRF/Kano:

```bash
Rscript 11_grf_kano_estimation.R \
  --input data/06_choice_satisfaction/grf_input.xlsx \
  --output-dir data/07_effect_estimation/grf_kano
```

IBPA:

```bash
python 13_ibpa_improvement_difficulty.py \
  --input data/08_improvement_difficulty/ibpa_input.xlsx \
  --output data/08_improvement_difficulty/ibpa_improvement_difficulty.xlsx
```

The MNL script should be run after updating its configuration block to point to the corresponding released data, image metadata, hotel-image embeddings, and GRF effect table.

## 8. Remaining release check

Before publishing the GRF script, replace the remaining positional covariate selections with explicit column names if possible. This is the only major schema-related dependency that still relies on the exact column order of the original working workbook.
