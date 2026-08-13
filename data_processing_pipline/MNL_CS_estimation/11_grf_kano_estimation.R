# GRF marginal-effect, Kano-shape, and satisfaction-function estimation
# ---------------------------------------------------------------------
# Source: cleaned from the original grf_estimate.ipynb research notebook.
#
# Main input:
#   data/06_choice_satisfaction/grf_input.xlsx
#
# Main outputs:
#   grf_input_diagnostics.csv
#   tau_estimates_raw.rds
#   satisfaction_shape_summary.csv
#   shape_classification_diagnostics.csv
#   satisfaction_function_parameters.csv
#   satisfaction_shape_and_parameters.csv
#   moo_satisfaction_parameters.csv
#
# Example:
#   Rscript 11_grf_kano_estimation.R \
#     --input data/06_choice_satisfaction/grf_input.xlsx \
#     --output-dir data/07_effect_estimation/grf_kano
#
# IMPORTANT:
# The original analysis selected several control blocks by their positions
# after dummy expansion (columns 8:11, 14:135, and 136:180). This cleaned
# release preserves that exact research specification for reproducibility.
# Once the released grf_input.xlsx schema is frozen, replacing these positional
# selectors with explicit column names is recommended.

# ============================================================
# 0. Packages and configuration
# ============================================================
required_packages <- c(
  "readxl", "grf", "dplyr", "ggplot2", "tibble", "purrr"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    "Please install the following R packages first: ",
    paste(missing_packages, collapse = ", ")
  )
}

get_arg <- function(flag, default = NULL) {
  args <- commandArgs(trailingOnly = TRUE)
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) {
    return(default)
  }
  args[idx + 1L]
}

DATA_PATH <- get_arg(
  "--input",
  "data/06_choice_satisfaction/grf_input.xlsx"
)
OUTPUT_DIR <- get_arg(
  "--output-dir",
  "data/07_effect_estimation/grf_kano"
)
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

SEED <- 42
NUM_TREES <- 2000
MIN_GRF_OBS <- 30
MIN_GRF_UNIQUE_W <- 10

LOESS_SPAN <- 0.80
GRID_N <- 200
MIN_UNIQUE_WP <- 20
MIN_SMOOTH_OBS <- 50
MIN_WP_RANGE <- 1.50

KNOT_BOOT_B <- 200
MIN_KNOT_DETECTION_RATE <- 0.60
MAX_RELATIVE_KNOT_SD <- 0.25

# Shape classification uses broad lower/upper performance regions,
# not observations immediately adjacent to a turning point.
SLOPE_BOOT_B <- 500
TAIL_FRACTION <- 0.20
MIN_SIDE_N <- 20
MAX_SIDE_N <- 100
MIN_SLOPE_VALID_RATE <- 0.60


set.seed(SEED)

RELATIVE_SHAPE_THRESHOLD <- 0.03
MIN_DIRECTION_PROBABILITY <- 0.70
PRACTICAL_EFFECT_TOL <- 0.005

# ============================================================
# 1. Read data and construct matrices
# ============================================================
if (!file.exists(DATA_PATH)) {
  stop("Input file does not exist: ", DATA_PATH)
}

data <- read_excel(DATA_PATH)

required_columns <- c(
  "total_score", "location", "time_group_id", "Traveler_type"
)
missing_columns <- setdiff(required_columns, names(data))
if (length(missing_columns) > 0) {
  stop("Missing required columns: ", paste(missing_columns, collapse = ", "))
}


data$time_group_id <- as.factor(data$time_group_id)

location_dummies <- model.matrix(~ location - 1, data = data)
time_dummies <- model.matrix(~ time_group_id - 1, data = data)
traveler_dummies <- model.matrix(~ Traveler_type - 1, data = data)

data_with_dummies <- cbind(
  data,
  location_dummies,
  time_dummies,
  traveler_dummies
)

data_with_dummies$location <- NULL
data_with_dummies$time_group_id <- NULL
data_with_dummies$Traveler_type <- NULL

if (ncol(data_with_dummies) < 180) {
  stop(
    "data_with_dummies requires at least 180 columns, but contains only ",
    ncol(data_with_dummies),
    "."
  )
}

# Preserve the user's original positional definitions.
X_others1 <- data.matrix(data_with_dummies[, 8:11, drop = FALSE])
X_dummies <- data.matrix(data_with_dummies[, 136:180, drop = FALSE])
features <- data.matrix(data_with_dummies[, 14:135, drop = FALSE])
score <- suppressWarnings(
  as.numeric(trimws(as.character(data_with_dummies$total_score)))
)

storage.mode(X_others1) <- "double"
storage.mode(X_dummies) <- "double"
storage.mode(features) <- "double"

if (is.null(colnames(features))) {
  colnames(features) <- paste0("ES_", seq_len(ncol(features)))
}

message("Observations: ", nrow(features))
message("Service elements: ", ncol(features))
message("Covariates excluding other ESs: ", ncol(X_others1) + ncol(X_dummies))

cat("\n========== Basic diagnostics ==========\n")

cat("Rows:", nrow(features), "\n")
cat("ES columns:", ncol(features), "\n")
cat("Score NA/non-finite:", sum(!is.finite(score)), "\n")

cat(
  "X_others1 NA/non-finite:",
  sum(!is.finite(X_others1)),
  "\n"
)

cat(
  "X_dummies NA/non-finite:",
  sum(!is.finite(X_dummies)),
  "\n"
)

cat(
  "Features NA/non-finite:",
  sum(!is.finite(features)),
  "\n"
)

cat(
  "Rows complete in all ES columns:",
  sum(complete.cases(features)),
  "\n"
)

cat(
  "Rows complete in all model variables:",
  sum(
    complete.cases(
      cbind(
        score,
        X_others1,
        X_dummies,
        features
      )
    )
  ),
  "\n"
)

cat("\nFeature names:\n")
print(colnames(features))

cat("\nFirst ES diagnostics:\n")
for (j in seq_len(min(10, ncol(features)))) {

  Wp <- features[, j]

  X_test <- cbind(
    X_others1,
    X_dummies,
    features[, -j, drop = FALSE]
  )

  complete_n <- sum(
    complete.cases(
      cbind(
        score,
        Wp,
        X_test
      )
    )
  )

  cat(
    colnames(features)[j],
    "| valid W =", sum(is.finite(Wp)),
    "| unique W =", length(unique(Wp[is.finite(Wp)])),
    "| complete model rows =", complete_n,
    "\n"
  )
}

# ============================================================
# 2. GRF estimator for one service element
# ============================================================
estimate_tau_only <- function(
    Wp,
    Y,
    X,
    num_trees = NUM_TREES,
    seed = SEED,
    min_obs = MIN_GRF_OBS,
    min_unique_w = MIN_GRF_UNIQUE_W
) {

  Wp <- as.numeric(Wp)
  Y <- as.numeric(Y)
  X <- as.matrix(X)

  storage.mode(X) <- "double"

  keep <- (
    is.finite(Wp) &
    is.finite(Y) &
    complete.cases(X)
  )

  Wp <- Wp[keep]
  Y <- Y[keep]
  X <- X[
    keep,
    ,
    drop = FALSE
  ]

  if (length(Wp) < min_obs) {
    return(NULL)
  }

  if (
    length(unique(Wp)) < min_unique_w ||
    !is.finite(stats::sd(Wp)) ||
    stats::sd(Wp) <= 0 ||
    !is.finite(stats::sd(Y)) ||
    stats::sd(Y) <= 0
  ) {
    return(NULL)
  }

  # 后面的 regression_forest 和 causal_forest 保持不变
  forest_y <- regression_forest(
    X,
    Y,
    num.trees = num_trees,
    seed = seed
  )

  forest_w <- regression_forest(
    X,
    Wp,
    num.trees = num_trees,
    seed = seed + 1L
  )

  Y_hat <- predict(forest_y)$predictions
  W_hat <- predict(forest_w)$predictions

  forest_tau <- causal_forest(
    X,
    Y,
    Wp,
    Y.hat = Y_hat,
    W.hat = W_hat,
    num.trees = num_trees,
    seed = seed + 2L
  )

  pred <- predict(
    forest_tau,
    estimate.variance = TRUE
  )

  ate <- tryCatch(
    average_treatment_effect(
      forest_tau,
      target.sample = "all"
    ),
    error = function(e) NULL
  )

  if (!is.null(ate) && length(ate) >= 2 && all(is.finite(ate[1:2]))) {
    tau_mean <- unname(ate[1])
    tau_se <- unname(ate[2])
    tau_mean_method <- "grf_average_treatment_effect"
  } else {
    tau_mean <- mean(pred$predictions, na.rm = TRUE)
    tau_se <- stats::sd(pred$predictions, na.rm = TRUE) /
      sqrt(sum(is.finite(pred$predictions)))
    tau_mean_method <- "mean_predicted_tau_fallback"
  }

  result <- data.frame(
    Wp = Wp,
    tau = as.numeric(pred$predictions),
    tau_var = as.numeric(pred$variance.estimates),
    tau_mean_global = tau_mean,
    tau_se_global = tau_se,
    tau_mean_method = tau_mean_method,
    stringsAsFactors = FALSE
  )

  result <- result |>
    dplyr::filter(is.finite(Wp), is.finite(tau)) |>
    dplyr::arrange(Wp)

  if (nrow(result) < min_obs) {
    return(NULL)
  }

  result
}

# ============================================================
# Estimate all service elements.
# ============================================================
X_base <- cbind(
  X_others1,
  X_dummies
)

X_base <- as.matrix(X_base)
storage.mode(X_base) <- "double"

tau_results <- vector(
  "list",
  ncol(features)
)

names(tau_results) <- colnames(features)

grf_diagnostics <- vector(
  "list",
  ncol(features)
)

for (j in seq_len(ncol(features))) {

  fname <- colnames(features)[j]

  message(
    "Estimating tau for: ",
    fname,
    " (",
    j,
    "/",
    ncol(features),
    ")"
  )

  Wp_all <- as.numeric(
    features[, j]
  )

  # Only require:
  # 1. current ES performance is observed;
  # 2. total score is observed;
  # 3. common control variables are observed.
  keep <- (
    is.finite(Wp_all) &
    is.finite(score) &
    complete.cases(X_base)
  )

  n_valid <- sum(keep)

  if (n_valid < MIN_GRF_OBS) {

    grf_diagnostics[[j]] <- data.frame(
      Feature = fname,
      n_total = length(Wp_all),
      n_valid_ES = sum(is.finite(Wp_all)),
      n_model = n_valid,
      unique_W = length(
        unique(Wp_all[keep])
      ),
      status = "insufficient_observations",
      stringsAsFactors = FALSE
    )

    tau_results[[fname]] <- NULL
    next
  }

  Wp_j <- Wp_all[keep]
  Y_j <- score[keep]

  X_j <- X_base[
    keep,
    ,
    drop = FALSE
  ]

  # Remove covariates that are constant within the current ES subset.
  variable_column <- apply(
    X_j,
    2,
    function(z) {
      z <- z[is.finite(z)]

      length(z) >= 2 &&
        is.finite(stats::sd(z)) &&
        stats::sd(z) > 0
    }
  )

  X_j <- X_j[
    ,
    variable_column,
    drop = FALSE
  ]

  if (ncol(X_j) == 0) {

    grf_diagnostics[[j]] <- data.frame(
      Feature = fname,
      n_total = length(Wp_all),
      n_valid_ES = sum(is.finite(Wp_all)),
      n_model = n_valid,
      unique_W = length(unique(Wp_j)),
      status = "no_variable_covariates",
      stringsAsFactors = FALSE
    )

    tau_results[[fname]] <- NULL
    next
  }

  tau_results[[fname]] <- tryCatch(
    estimate_tau_only(
      Wp = Wp_j,
      Y = Y_j,
      X = X_j,
      num_trees = NUM_TREES,
      seed = SEED + 10L * j
    ),
    error = function(e) {

      warning(
        "GRF failed for ",
        fname,
        ": ",
        conditionMessage(e)
      )

      NULL
    }
  )

  grf_diagnostics[[j]] <- data.frame(
    Feature = fname,
    n_total = length(Wp_all),
    n_valid_ES = sum(is.finite(Wp_all)),
    n_model = n_valid,
    unique_W = length(unique(Wp_j)),
    n_covariates = ncol(X_j),
    status = if (
      is.null(tau_results[[fname]])
    ) {
      "GRF_failed_or_invalid"
    } else {
      "success"
    },
    stringsAsFactors = FALSE
  )
}

grf_diagnostic_df <- dplyr::bind_rows(
  grf_diagnostics
)

write.csv(
  grf_diagnostic_df,
  file.path(
    OUTPUT_DIR,
    "grf_input_diagnostics.csv"
  ),
  row.names = FALSE
)

message(
  "Valid GRF results: ",
  sum(
    !vapply(
      tau_results,
      is.null,
      logical(1)
    )
  )
)

print(
  sort(
    table(
      grf_diagnostic_df$status
    ),
    decreasing = TRUE
  )
)

# ============================================================
# 3. Check and save GRF results
# ============================================================
valid_grf_n <- sum(
  !vapply(
    tau_results,
    is.null,
    logical(1)
  )
)

message(
  "Valid GRF results: ",
  valid_grf_n
)

if (valid_grf_n == 0) {
  stop(
    "No valid GRF results were obtained. ",
    "Check grf_input_diagnostics.csv."
  )
}

saveRDS(
  tau_results,
  file.path(
    OUTPUT_DIR,
    "tau_estimates_raw.rds"
  )
)

# ============================================================
# 3. Smooth tau(W) and calculate its numerical derivative
# ============================================================
compute_tau_derivative <- function(
    df,
    span = LOESS_SPAN,
    grid_n = GRID_N,
    min_unique_wp = MIN_UNIQUE_WP,
    min_obs = MIN_SMOOTH_OBS,
    min_wp_range = MIN_WP_RANGE
) {
  if (is.null(df)) {
    return(NULL)
  }

  df <- df[
    is.finite(df$tau) & is.finite(df$Wp),
    ,
    drop = FALSE
  ]

  if (nrow(df) < min_obs) {
    return(NULL)
  }

  support_quantiles <- stats::quantile(
    df$Wp,
    probs = c(0.10, 0.90),
    na.rm = TRUE,
    names = FALSE
  )

  wp_lo <- support_quantiles[1]
  wp_hi <- support_quantiles[2]

  if (
    !all(is.finite(support_quantiles)) ||
    (wp_hi - wp_lo) < min_wp_range
  ) {
    return(NULL)
  }

  df_support <- df[
    df$Wp >= wp_lo & df$Wp <= wp_hi,
    ,
    drop = FALSE
  ]

  if (nrow(df_support) < min_obs) {
    return(NULL)
  }

  # Aggregate duplicate performance values while preserving their frequency as LOESS weights.
  df_agg <- df_support |>
    dplyr::group_by(Wp) |>
    dplyr::summarise(
      tau = mean(tau, na.rm = TRUE),
      n_at_wp = dplyr::n(),
      .groups = "drop"
    ) |>
    dplyr::arrange(Wp)

  if (nrow(df_agg) < min_unique_wp) {
    return(NULL)
  }

  lo <- tryCatch(
    loess(
      tau ~ Wp,
      data = df_agg,
      weights = n_at_wp,
      span = span,
      degree = 1,
      control = loess.control(
        surface = "direct",
        trace.hat = "approximate"
      )
    ),
    error = function(e) NULL
  )

  if (is.null(lo)) {
    return(NULL)
  }

  grid_wp <- seq(wp_lo, wp_hi, length.out = grid_n)
  tau_grid <- as.numeric(
    predict(lo, newdata = data.frame(Wp = grid_wp))
  )

  if (length(tau_grid) != grid_n || any(!is.finite(tau_grid))) {
    return(NULL)
  }

  d_tau <- diff(tau_grid) / diff(grid_wp)
  grid_mid <- (grid_wp[-1] + grid_wp[-length(grid_wp)]) / 2
  tau_mid <- (tau_grid[-1] + tau_grid[-length(tau_grid)]) / 2

  data.frame(
    Wp_mid = grid_mid,
    tau_smooth = tau_mid,
    d_tau = d_tau,
    support_low = wp_lo,
    support_high = wp_hi,
    support_range = wp_hi - wp_lo,
    n_obs = nrow(df),
    n_support_obs = nrow(df_support),
    n_unique_wp = nrow(df_agg)
  )
}

derivative_results <- lapply(
  tau_results,
  function(df) {

    if (is.null(df)) {
      return(NULL)
    }

    compute_tau_derivative(
      df = df,
      span = 0.8,
      grid_n = 200,
      min_unique_wp = 20,
      min_obs = MIN_GRF_OBS,
      min_wp_range = 1.5
    )
  }
)


valid_derivative_n <- sum(
  !vapply(
    derivative_results,
    is.null,
    logical(1)
  )
)

message(
  "Valid derivative results: ",
  valid_derivative_n
)


saveRDS(
  derivative_results,
  file.path(
    OUTPUT_DIR,
    "tau_derivative_results.rds"
  )
)

message(
  "Elements without a reliable smoothed derivative: ",
  sum(vapply(derivative_results, is.null, logical(1)))
)

# ============================================================
# 4. Detect the strongest turning-point candidate in tau(W)
# ============================================================
detect_turning_point <- function(
    deriv_df,
    min_grid_points = 20,
    zero_abs_tol = 1e-10,
    zero_rel_tol = 0.01,
    min_crossing_strength_ratio = 0.05
) {
  if (is.null(deriv_df) || nrow(deriv_df) < min_grid_points) {
    return(NULL)
  }

  deriv_df <- deriv_df[
    is.finite(deriv_df$Wp_mid) & is.finite(deriv_df$d_tau),
    ,
    drop = FALSE
  ]

  if (nrow(deriv_df) < min_grid_points) {
    return(NULL)
  }

  x <- deriv_df$Wp_mid
  y <- deriv_df$d_tau
  y_scale <- max(abs(y), na.rm = TRUE)

  if (!is.finite(y_scale) || y_scale <= zero_abs_tol) {
    return(NULL)
  }

  zero_tol <- max(zero_abs_tol, zero_rel_tol * y_scale)
  sign_class <- ifelse(y > zero_tol, 1L, ifelse(y < -zero_tol, -1L, 0L))
  nonzero_idx <- which(sign_class != 0L)

  if (length(nonzero_idx) < 2) {
    return(NULL)
  }

  left_idx <- nonzero_idx[-length(nonzero_idx)]
  right_idx <- nonzero_idx[-1]
  opposite <- sign_class[left_idx] != sign_class[right_idx]

  left_idx <- left_idx[opposite]
  right_idx <- right_idx[opposite]

  if (length(left_idx) == 0) {
    return(NULL)
  }

  crossing_strength <- abs(y[right_idx] - y[left_idx])
  minimum_strength <- min_crossing_strength_ratio * y_scale
  keep <- crossing_strength >= minimum_strength

  left_idx <- left_idx[keep]
  right_idx <- right_idx[keep]
  crossing_strength <- crossing_strength[keep]

  if (length(left_idx) == 0) {
    return(NULL)
  }

  knot_values <- vapply(
    seq_along(left_idx),
    function(j) {
      i1 <- left_idx[j]
      i2 <- right_idx[j]
      x1 <- x[i1]
      x2 <- x[i2]
      y1 <- y[i1]
      y2 <- y[i2]
      denominator <- y2 - y1

      if (!is.finite(denominator) || abs(denominator) <= zero_abs_tol) {
        return((x1 + x2) / 2)
      }

      x1 - y1 * (x2 - x1) / denominator
    },
    numeric(1)
  )

  selected <- which.max(crossing_strength)
  i1 <- left_idx[selected]
  i2 <- right_idx[selected]

  data.frame(
    knot = knot_values[selected],
    d_tau_left = y[i1],
    d_tau_right = y[i2],
    crossing_strength = crossing_strength[selected],
    n_crossings = length(knot_values),
    stringsAsFactors = FALSE
  )
}

# Compatibility alias for earlier code.
detect_inflection_point <- detect_turning_point

turning_point_candidates <- lapply(
  derivative_results,
  detect_turning_point
)

saveRDS(
  turning_point_candidates,
  file.path(OUTPUT_DIR, "turning_point_candidates.rds")
)

# ============================================================
# 5. Bootstrap stability of the turning-point reference
# ============================================================
bootstrap_turning_point_once <- function(
    df,
    span = LOESS_SPAN,
    grid_n = GRID_N,
    min_unique_wp = MIN_UNIQUE_WP,
    min_obs = MIN_SMOOTH_OBS,
    min_wp_range = MIN_WP_RANGE
) {
  if (is.null(df)) {
    return(NA_real_)
  }

  df <- df[
    is.finite(df$Wp) & is.finite(df$tau),
    ,
    drop = FALSE
  ]

  if (nrow(df) < min_obs) {
    return(NA_real_)
  }

  idx <- sample.int(nrow(df), nrow(df), replace = TRUE)
  d <- df[idx, , drop = FALSE]

  deriv <- compute_tau_derivative(
    d,
    span = span,
    grid_n = grid_n,
    min_unique_wp = min_unique_wp,
    min_obs = min_obs,
    min_wp_range = min_wp_range
  )

  candidate <- detect_turning_point(deriv)
  if (is.null(candidate)) {
    return(NA_real_)
  }

  as.numeric(candidate$knot[1])
}

bootstrap_turning_point <- function(
    df,
    B = KNOT_BOOT_B,
    span = LOESS_SPAN,
    grid_n = GRID_N,
    min_unique_wp = MIN_UNIQUE_WP,
    min_obs = MIN_SMOOTH_OBS,
    min_wp_range = MIN_WP_RANGE,
    min_detection_rate = MIN_KNOT_DETECTION_RATE,
    max_relative_sd = MAX_RELATIVE_KNOT_SD,
    seed = SEED
) {
  empty_result <- function(reason, detection_rate = NA_real_) {
    list(
      stable = FALSE,
      reason = reason,
      knot = NA_real_,
      knot_sd = NA_real_,
      relative_knot_sd = NA_real_,
      knot_detection_rate = detection_rate,
      knot_ci_low = NA_real_,
      knot_ci_high = NA_real_,
      n_valid_knots = 0L,
      B = B,
      support_low = NA_real_,
      support_high = NA_real_,
      support_range = NA_real_,
      knot_samples = numeric(0)
    )
  }

  if (is.null(df)) {
    return(empty_result("missing_tau_result"))
  }

  df <- df[
    is.finite(df$Wp) & is.finite(df$tau),
    ,
    drop = FALSE
  ]

  if (nrow(df) < min_obs) {
    return(empty_result("insufficient_observations"))
  }

  support_quantiles <- stats::quantile(
    df$Wp,
    probs = c(0.10, 0.90),
    na.rm = TRUE,
    names = FALSE
  )
  support_range <- support_quantiles[2] - support_quantiles[1]

  if (!all(is.finite(support_quantiles)) || support_range < min_wp_range) {
    result <- empty_result("insufficient_performance_range")
    result$support_low <- support_quantiles[1]
    result$support_high <- support_quantiles[2]
    result$support_range <- support_range
    return(result)
  }

  set.seed(seed)
  knots <- replicate(
    B,
    bootstrap_turning_point_once(
      df = df,
      span = span,
      grid_n = grid_n,
      min_unique_wp = min_unique_wp,
      min_obs = min_obs,
      min_wp_range = min_wp_range
    )
  )

  valid_knots <- as.numeric(knots[is.finite(knots)])
  detection_rate <- length(valid_knots) / B

  result <- empty_result("unstable_turning_point", detection_rate)
  result$support_low <- support_quantiles[1]
  result$support_high <- support_quantiles[2]
  result$support_range <- support_range
  result$n_valid_knots <- length(valid_knots)
  result$knot_samples <- valid_knots

  if (detection_rate < min_detection_rate) {
    result$reason <- "low_detection_rate"
    return(result)
  }

  if (length(valid_knots) < 2) {
    result$reason <- "too_few_valid_knots"
    return(result)
  }

  knot_median <- stats::median(valid_knots)
  knot_sd <- stats::sd(valid_knots)
  relative_knot_sd <- knot_sd / support_range
  knot_ci <- stats::quantile(
    valid_knots,
    probs = c(0.025, 0.975),
    na.rm = TRUE,
    names = FALSE
  )

  result$knot <- knot_median
  result$knot_sd <- knot_sd
  result$relative_knot_sd <- relative_knot_sd
  result$knot_ci_low <- knot_ci[1]
  result$knot_ci_high <- knot_ci[2]

  if (!is.finite(relative_knot_sd) || relative_knot_sd > max_relative_sd) {
    result$reason <- "diffuse_knot_distribution"
    return(result)
  }

  result$stable <- TRUE
  result$reason <- "bootstrap_stable_turning_point"
  result
}

# Compatibility alias.
bootstrap_inflection_point <- bootstrap_turning_point

knot_bootstrap_results <- Map(
  function(df, j) {
    bootstrap_turning_point(
      df = df,
      seed = SEED + 1000L + j
    )
  },
  tau_results,
  seq_along(tau_results)
)
names(knot_bootstrap_results) <- names(tau_results)

saveRDS(
  knot_bootstrap_results,
  file.path(OUTPUT_DIR, "turning_point_bootstrap_results.rds")
)

# ============================================================
# 6. Summarize turning-point bootstrap results
# ============================================================
knot_summary <- dplyr::bind_rows(
  lapply(names(knot_bootstrap_results), function(fname) {
    info <- knot_bootstrap_results[[fname]]

    data.frame(
      Feature = fname,
      stable_knot = isTRUE(info$stable),
      reason = info$reason,
      knot = info$knot,
      knot_sd = info$knot_sd,
      relative_knot_sd = info$relative_knot_sd,
      knot_detection_rate = info$knot_detection_rate,
      knot_ci_low = info$knot_ci_low,
      knot_ci_high = info$knot_ci_high,
      n_valid_knots = info$n_valid_knots,
      support_low = info$support_low,
      support_high = info$support_high,
      support_range = info$support_range,
      stringsAsFactors = FALSE
    )
  })
)

write.csv(
  knot_summary,
  file.path(OUTPUT_DIR, "turning_point_bootstrap_summary.csv"),
  row.names = FALSE
)

print(table(knot_summary$stable_knot, useNA = "ifany"))
print(sort(table(knot_summary$reason), decreasing = TRUE))

# ============================================================
# 7. Choose an auxiliary reference point and estimate broad
#    lower-/upper-performance marginal effects
# ============================================================
choose_reference_point <- function(df, knot_info) {
  df <- df[is.finite(df$Wp), , drop = FALSE]
  if (nrow(df) == 0) {
    return(NULL)
  }

  support <- stats::quantile(
    df$Wp,
    probs = c(0.10, 0.90),
    na.rm = TRUE,
    names = FALSE
  )

  support_df <- df[
    df$Wp >= support[1] & df$Wp <= support[2],
    ,
    drop = FALSE
  ]

  if (nrow(support_df) == 0) {
    support_df <- df
  }

  # The stable turning point is retained only as a numerical
  # reference/centering point. Otherwise use the support median.
  if (
    !is.null(knot_info) &&
    isTRUE(knot_info$stable) &&
    is.finite(knot_info$knot)
  ) {
    reference_point <- min(
      max(knot_info$knot, support[1]),
      support[2]
    )
    source <- "bootstrap_stable_turning_point_reference"
  } else {
    reference_point <- stats::median(
      support_df$Wp,
      na.rm = TRUE
    )
    source <- "median_support_reference"
  }

  list(
    reference_point = as.numeric(reference_point),
    reference_source = source,
    support_low = support[1],
    support_high = support[2]
  )
}


estimate_tau_region_slopes <- function(
    df,
    tail_fraction = TAIL_FRACTION,
    min_side_n = MIN_SIDE_N,
    max_side_n = MAX_SIDE_N,
    support_probs = c(0.10, 0.90)
) {
  if (is.null(df)) {
    return(NULL)
  }

  df <- df[
    is.finite(df$Wp) & is.finite(df$tau),
    ,
    drop = FALSE
  ]

  if (nrow(df) < 2 * min_side_n) {
    return(NULL)
  }

  support <- stats::quantile(
    df$Wp,
    probs = support_probs,
    na.rm = TRUE,
    names = FALSE
  )

  if (
    length(support) != 2 ||
    !all(is.finite(support)) ||
    support[2] <= support[1]
  ) {
    return(NULL)
  }

  d <- df[
    df$Wp >= support[1] &
    df$Wp <= support[2],
    ,
    drop = FALSE
  ]

  if (nrow(d) < 2 * min_side_n) {
    return(NULL)
  }

  d <- d[
    order(d$Wp),
    ,
    drop = FALSE
  ]

  side_n <- max(
    min_side_n,
    ceiling(tail_fraction * nrow(d))
  )

  side_n <- min(
    side_n,
    max_side_n,
    floor(nrow(d) / 2)
  )

  if (side_n < min_side_n) {
    return(NULL)
  }

  # Lowest and highest balanced portions of the reliable support.
  left <- d[
    seq_len(side_n),
    ,
    drop = FALSE
  ]

  right <- d[
    (nrow(d) - side_n + 1):nrow(d),
    ,
    drop = FALSE
  ]

  data.frame(
    Per_left = mean(left$Wp, na.rm = TRUE),
    Per_right = mean(right$Wp, na.rm = TRUE),
    tau_left = mean(left$tau, na.rm = TRUE),
    tau_right = mean(right$tau, na.rm = TRUE),
    tau_diff = mean(right$tau, na.rm = TRUE) -
      mean(left$tau, na.rm = TRUE),
    n_left = nrow(left),
    n_right = nrow(right),
    support_low = support[1],
    support_high = support[2],
    stringsAsFactors = FALSE
  )
}

# ============================================================
# 8. Bootstrap the broad lower-/upper-performance contrast
# ============================================================
bootstrap_tau_region_slopes <- function(
    df,
    B = SLOPE_BOOT_B,
    tail_fraction = TAIL_FRACTION,
    min_side_n = MIN_SIDE_N,
    max_side_n = MAX_SIDE_N,
    min_valid_rate = MIN_SLOPE_VALID_RATE,
    seed = SEED
) {
  if (is.null(df)) {
    return(NULL)
  }

  df <- df[
    is.finite(df$Wp) & is.finite(df$tau),
    ,
    drop = FALSE
  ]

  if (nrow(df) < 2 * min_side_n) {
    return(NULL)
  }

  set.seed(seed)

  res <- replicate(B, {
    idx <- sample.int(
      nrow(df),
      nrow(df),
      replace = TRUE
    )

    d <- df[
      idx,
      ,
      drop = FALSE
    ]

    slope <- estimate_tau_region_slopes(
      df = d,
      tail_fraction = tail_fraction,
      min_side_n = min_side_n,
      max_side_n = max_side_n
    )

    if (is.null(slope)) {
      return(c(
        Per_left = NA_real_,
        Per_right = NA_real_,
        tau_left = NA_real_,
        tau_right = NA_real_,
        tau_diff = NA_real_
      ))
    }

    c(
      Per_left = slope$Per_left,
      Per_right = slope$Per_right,
      tau_left = slope$tau_left,
      tau_right = slope$tau_right,
      tau_diff = slope$tau_diff
    )
  })

  if (is.null(dim(res))) {
    res <- matrix(res, ncol = 1)
  }

  valid <- apply(
    res,
    2,
    function(x) all(is.finite(x))
  )

  res <- res[
    ,
    valid,
    drop = FALSE
  ]

  valid_rate <- ncol(res) / B

  if (
    ncol(res) < 2 ||
    valid_rate < min_valid_rate
  ) {
    return(NULL)
  }

  get_ci <- function(name) {
    stats::quantile(
      res[name, ],
      probs = c(0.025, 0.975),
      na.rm = TRUE,
      names = FALSE
    )
  }

  left_ci <- get_ci("tau_left")
  right_ci <- get_ci("tau_right")
  diff_ci <- get_ci("tau_diff")

  data.frame(
    Per_left = mean(
      res["Per_left", ],
      na.rm = TRUE
    ),
    Per_right = mean(
      res["Per_right", ],
      na.rm = TRUE
    ),
    tau_left_mean = mean(
      res["tau_left", ],
      na.rm = TRUE
    ),
    tau_left_ci_low = left_ci[1],
    tau_left_ci_high = left_ci[2],
    tau_right_mean = mean(
      res["tau_right", ],
      na.rm = TRUE
    ),
    tau_right_ci_low = right_ci[1],
    tau_right_ci_high = right_ci[2],
    tau_diff_mean = mean(
      res["tau_diff", ],
      na.rm = TRUE
    ),
    tau_diff_sd = stats::sd(
      res["tau_diff", ],
      na.rm = TRUE
    ),
    tau_diff_ci_low = diff_ci[1],
    tau_diff_ci_high = diff_ci[2],
    diff_positive_probability = mean(
      res["tau_diff", ] > 0,
      na.rm = TRUE
    ),
    diff_negative_probability = mean(
      res["tau_diff", ] < 0,
      na.rm = TRUE
    ),
    slope_boot_valid_rate = valid_rate,
    n_boot = ncol(res),
    stringsAsFactors = FALSE
  )
}

# ============================================================
# 9. Classify the dominant local satisfaction shape
# ============================================================
classify_satisfaction_shape <- function(
    tau_mean,
    slope_info,
    relative_threshold = RELATIVE_SHAPE_THRESHOLD,
    min_direction_probability = MIN_DIRECTION_PROBABILITY,
    effect_tol = PRACTICAL_EFFECT_TOL,
    eps = 1e-8
) {
  if (!is.finite(tau_mean)) {
    return("Questionable")
  }

  if (is.null(slope_info)) {
    if (abs(tau_mean) <= effect_tol) {
      return("Indifferent")
    }
    if (tau_mean < -effect_tol) {
      return("Reverse")
    }
    return("One-dimensional")
  }

  tau_left <- as.numeric(
    slope_info$tau_left_mean[1]
  )
  tau_right <- as.numeric(
    slope_info$tau_right_mean[1]
  )
  diff_mean <- as.numeric(
    slope_info$tau_diff_mean[1]
  )
  p_positive <- as.numeric(
    slope_info$diff_positive_probability[1]
  )
  p_negative <- as.numeric(
    slope_info$diff_negative_probability[1]
  )

  if (
    !all(is.finite(c(
      tau_left,
      tau_right,
      diff_mean,
      p_positive,
      p_negative
    )))
  ) {
    return("Questionable")
  }

  if (
    abs(tau_left) <= effect_tol &&
    abs(tau_right) <= effect_tol
  ) {
    return("Indifferent")
  }

  if (
    tau_left < -effect_tol &&
    tau_right < -effect_tol
  ) {
    return("Reverse")
  }

  # Mixed signs or a non-positive local marginal effect cannot be
  # represented by the positive monotonic Kano functions.
  if (
    tau_left <= effect_tol ||
    tau_right <= effect_tol
  ) {
    return("Questionable")
  }

  relative_change <- diff_mean /
    (abs(tau_mean) + eps)

  if (
    relative_change >= relative_threshold &&
    p_positive >= min_direction_probability
  ) {
    return("Attractive")
  }

  if (
    relative_change <= -relative_threshold &&
    p_negative >= min_direction_probability
  ) {
    return("Must-be")
  }

  # One-dimensional is reserved for practically small slope changes.
  if (abs(relative_change) < relative_threshold) {
    return("One-dimensional")
  }

  # A sizeable but directionally unstable contrast is not evidence
  # of linearity; retain it as Questionable.
  "Questionable"
}


explain_shape_classification <- function(
    tau_mean,
    slope_info,
    shape_type,
    relative_threshold = RELATIVE_SHAPE_THRESHOLD,
    min_direction_probability = MIN_DIRECTION_PROBABILITY,
    eps = 1e-8
) {
  if (is.null(slope_info)) {
    return("No reliable lower/upper contrast; conservative fallback")
  }

  relative_change <- as.numeric(
    slope_info$tau_diff_mean[1]
  ) / (abs(tau_mean) + eps)

  p_positive <- as.numeric(
    slope_info$diff_positive_probability[1]
  )
  p_negative <- as.numeric(
    slope_info$diff_negative_probability[1]
  )

  if (shape_type == "Attractive") {
    return(
      paste0(
        "Positive lower-to-upper slope change; relative change=",
        round(relative_change, 4),
        "; P(diff>0)=",
        round(p_positive, 3)
      )
    )
  }

  if (shape_type == "Must-be") {
    return(
      paste0(
        "Negative lower-to-upper slope change; relative change=",
        round(relative_change, 4),
        "; P(diff<0)=",
        round(p_negative, 3)
      )
    )
  }

  if (shape_type == "One-dimensional") {
    return(
      paste0(
        "Practically small lower-to-upper slope change; |relative change|<",
        relative_threshold
      )
    )
  }

  if (shape_type == "Questionable") {
    return(
      paste0(
        "Slope contrast is sizeable but directionally unstable; max direction probability=",
        round(max(p_positive, p_negative), 3),
        " < ",
        min_direction_probability
      )
    )
  }

  shape_type
}

# ============================================================
# 10. Build the final shape summary
# ============================================================
shape_results <- lapply(
  seq_along(tau_results),
  function(j) {
    fname <- names(tau_results)[j]
    df <- tau_results[[fname]]

    if (
      is.null(df) ||
      nrow(df) == 0
    ) {
      return(data.frame(
        Feature = fname,
        Tau_Hat_Mean = NA_real_,
        AME_SE = NA_real_,
        reference_point = NA_real_,
        knot = NA_real_,
        reference_source = "unavailable",
        stable_turning_point = FALSE,
        shape_type = "Questionable",
        raw_shape_candidate = "Unavailable",
        classification_reason = "GRF result unavailable",
        shape_basis = "GRF result unavailable",
        stringsAsFactors = FALSE
      ))
    }

    tau_mean <- unique(
      df$tau_mean_global[
        is.finite(df$tau_mean_global)
      ]
    )[1]

    tau_se <- unique(
      df$tau_se_global[
        is.finite(df$tau_se_global)
      ]
    )[1]

    if (!is.finite(tau_mean)) {
      tau_mean <- mean(
        df$tau,
        na.rm = TRUE
      )
    }

    if (!is.finite(tau_se)) {
      tau_se <- stats::sd(
        df$tau,
        na.rm = TRUE
      ) / sqrt(nrow(df))
    }

    knot_info <- knot_bootstrap_results[[fname]]
    ref_info <- choose_reference_point(
      df,
      knot_info
    )

    if (is.null(ref_info)) {
      return(data.frame(
        Feature = fname,
        Tau_Hat_Mean = tau_mean,
        AME_SE = tau_se,
        reference_point = NA_real_,
        knot = NA_real_,
        reference_source = "unavailable",
        stable_turning_point = FALSE,
        shape_type = classify_satisfaction_shape(
          tau_mean,
          NULL
        ),
        raw_shape_candidate = "Unavailable",
        classification_reason = "No usable performance support",
        shape_basis = "No usable performance support",
        stringsAsFactors = FALSE
      ))
    }

    # Important: classification compares broad lower and upper
    # performance regions. The turning point is not used to select
    # immediately adjacent observations.
    slope_info <- bootstrap_tau_region_slopes(
      df = df,
      seed = SEED + 5000L + j
    )

    shape_type <- classify_satisfaction_shape(
      tau_mean = tau_mean,
      slope_info = slope_info
    )

    base <- data.frame(
      Feature = fname,
      Tau_Hat_Mean = tau_mean,
      AME_SE = tau_se,
      reference_point = ref_info$reference_point,
      knot = ref_info$reference_point,
      reference_source = ref_info$reference_source,
      stable_turning_point = isTRUE(knot_info$stable),
      knot_detection_rate = knot_info$knot_detection_rate,
      knot_sd = knot_info$knot_sd,
      relative_knot_sd = knot_info$relative_knot_sd,
      knot_ci_low = knot_info$knot_ci_low,
      knot_ci_high = knot_info$knot_ci_high,
      support_low = ref_info$support_low,
      support_high = ref_info$support_high,
      support_range = ref_info$support_high -
        ref_info$support_low,
      n_obs = nrow(df),
      shape_type = shape_type,
      type = shape_type,
      stringsAsFactors = FALSE
    )

    if (is.null(slope_info)) {
      base$Per_left <- NA_real_
      base$Per_right <- NA_real_
      base$tau_left_mean <- NA_real_
      base$tau_left_ci_low <- NA_real_
      base$tau_left_ci_high <- NA_real_
      base$tau_right_mean <- NA_real_
      base$tau_right_ci_low <- NA_real_
      base$tau_right_ci_high <- NA_real_
      base$tau_diff_mean <- NA_real_
      base$tau_diff_sd <- NA_real_
      base$tau_diff_ci_low <- NA_real_
      base$tau_diff_ci_high <- NA_real_
      base$diff_positive_probability <- NA_real_
      base$diff_negative_probability <- NA_real_
      base$relative_tau_change <- NA_real_
      base$slope_boot_valid_rate <- NA_real_
      base$raw_shape_candidate <- "Unavailable"
      base$classification_reason <-
        "No reliable lower/upper contrast; conservative fallback"
      base$shape_basis <- paste0(
        ref_info$reference_source,
        "; insufficient broad-region support"
      )
      return(base)
    }

    relative_change <- slope_info$tau_diff_mean /
      (abs(tau_mean) + 1e-8)

    raw_candidate <- if (
      relative_change >= RELATIVE_SHAPE_THRESHOLD
    ) {
      "Attractive_candidate"
    } else if (
      relative_change <= -RELATIVE_SHAPE_THRESHOLD
    ) {
      "Must-be_candidate"
    } else {
      "One-dimensional_candidate"
    }

    base$Per_left <- slope_info$Per_left
    base$Per_right <- slope_info$Per_right
    base$tau_left_mean <- slope_info$tau_left_mean
    base$tau_left_ci_low <- slope_info$tau_left_ci_low
    base$tau_left_ci_high <- slope_info$tau_left_ci_high
    base$tau_right_mean <- slope_info$tau_right_mean
    base$tau_right_ci_low <- slope_info$tau_right_ci_low
    base$tau_right_ci_high <- slope_info$tau_right_ci_high
    base$tau_diff_mean <- slope_info$tau_diff_mean
    base$tau_diff_sd <- slope_info$tau_diff_sd
    base$tau_diff_ci_low <- slope_info$tau_diff_ci_low
    base$tau_diff_ci_high <- slope_info$tau_diff_ci_high
    base$diff_positive_probability <-
      slope_info$diff_positive_probability
    base$diff_negative_probability <-
      slope_info$diff_negative_probability
    base$relative_tau_change <- relative_change
    base$slope_boot_valid_rate <-
      slope_info$slope_boot_valid_rate
    base$raw_shape_candidate <- raw_candidate
    base$classification_reason <-
      explain_shape_classification(
        tau_mean = tau_mean,
        slope_info = slope_info,
        shape_type = shape_type
      )
    base$shape_basis <- paste0(
      "balanced lower/upper performance regions; ",
      ref_info$reference_source,
      " retained only for function centering"
    )

    base
  }
)

shape_summary <- dplyr::bind_rows(
  shape_results
)

# Compatibility columns used by the earlier Excel/Python workflow.
shape_summary[["Per-"]] <- shape_summary$Per_left
shape_summary[["Per+"]] <- shape_summary$Per_right

write.csv(
  shape_summary,
  file.path(
    OUTPUT_DIR,
    "satisfaction_shape_summary.csv"
  ),
  row.names = FALSE
)

saveRDS(
  shape_summary,
  file.path(
    OUTPUT_DIR,
    "satisfaction_shape_summary.rds"
  )
)

message("\nFinal classifications:")
print(
  sort(
    table(shape_summary$shape_type),
    decreasing = TRUE
  )
)

message("\nMagnitude-only candidates:")
print(
  sort(
    table(shape_summary$raw_shape_candidate),
    decreasing = TRUE
  )
)

message("\nReference sources:")
print(
  sort(
    table(shape_summary$reference_source),
    decreasing = TRUE
  )
)

diagnostic_columns <- c(
  "Feature",
  "Tau_Hat_Mean",
  "Per_left",
  "Per_right",
  "tau_left_mean",
  "tau_right_mean",
  "tau_diff_mean",
  "relative_tau_change",
  "diff_positive_probability",
  "diff_negative_probability",
  "raw_shape_candidate",
  "shape_type",
  "classification_reason"
)

write.csv(
  shape_summary[
    ,
    intersect(
      diagnostic_columns,
      names(shape_summary)
    ),
    drop = FALSE
  ],
  file.path(
    OUTPUT_DIR,
    "shape_classification_diagnostics.csv"
  ),
  row.names = FALSE
)

# ============================================================
# 11. Fit parsimonious satisfaction-function parameters
# ============================================================
attractive_value <- function(x, knot, a, b, c = 0) {
  a * exp(b * (x - knot)) + c
}

attractive_derivative <- function(x, knot, a, b) {
  a * b * exp(b * (x - knot))
}

must_be_value <- function(x, knot, a, b, c = 0) {
  -a * exp(-b * (x - knot)) + c
}

must_be_derivative <- function(x, knot, a, b) {
  a * b * exp(-b * (x - knot))
}

linear_value <- function(x, a, c = 0) {
  a * x + c
}

fit_satisfaction_parameters <- function(
    row,
    df,
    eps = 1e-10,
    max_log_slope_change = 10
) {
  type <- as.character(row$shape_type)
  tau_mean <- as.numeric(row$Tau_Hat_Mean)

  result <- data.frame(
    Feature = as.character(row$Feature),
    shape_type = type,
    function_form = NA_character_,
    knot = as.numeric(row$reference_point),
    a = NA_real_,
    b = NA_real_,
    c = 0,
    fitted_tau_left = NA_real_,
    fitted_tau_right = NA_real_,
    fitted_average_tau = NA_real_,
    global_tau_relative_error = NA_real_,
    eligible_for_moo = FALSE,
    fit_status = "not_fitted",
    stringsAsFactors = FALSE
  )

  if (!is.finite(tau_mean)) {
    result$fit_status <- "invalid_global_tau"
    return(result)
  }

  if (type == "One-dimensional") {
    result$function_form <- "linear"
    result$a <- tau_mean
    result$b <- 0
    result$fitted_average_tau <- tau_mean
    result$global_tau_relative_error <- 0
    result$eligible_for_moo <- tau_mean > 0
    result$fit_status <- "local_linear_approximation"
    return(result)
  }

  if (type == "Indifferent") {
    result$function_form <- "constant"
    result$a <- 0
    result$b <- 0
    result$fitted_average_tau <- 0
    result$global_tau_relative_error <- abs(tau_mean)
    result$eligible_for_moo <- FALSE
    result$fit_status <- "practically_zero_effect"
    return(result)
  }

  if (type == "Reverse") {
    result$function_form <- "linear_reverse_diagnostic"
    result$a <- tau_mean
    result$b <- 0
    result$fitted_average_tau <- tau_mean
    result$global_tau_relative_error <- 0
    result$eligible_for_moo <- FALSE
    result$fit_status <- "reverse_excluded_from_positive_improvement"
    return(result)
  }

  if (type == "Questionable") {
    result$fit_status <- "questionable_shape_excluded"
    return(result)
  }

  x_left <- as.numeric(row$Per_left)
  x_right <- as.numeric(row$Per_right)
  tau_left <- as.numeric(row$tau_left_mean)
  tau_right <- as.numeric(row$tau_right_mean)
  knot <- as.numeric(row$reference_point)

  needed <- c(x_left, x_right, tau_left, tau_right, knot)
  if (!all(is.finite(needed)) || x_right <= x_left || tau_left <= 0 || tau_right <= 0) {
    result$function_form <- "linear_fallback"
    result$a <- tau_mean
    result$b <- 0
    result$fitted_average_tau <- tau_mean
    result$global_tau_relative_error <- 0
    result$eligible_for_moo <- tau_mean > 0
    result$fit_status <- "invalid_exponential_anchors_linear_fallback"
    return(result)
  }

  dx <- x_right - x_left

  if (type == "Attractive") {
    b <- log(tau_right / tau_left) / dx
    exponent_sign <- 1
    form <- "attractive_exponential"
  } else if (type == "Must-be") {
    b <- -log(tau_right / tau_left) / dx
    exponent_sign <- -1
    form <- "must_be_exponential"
  } else {
    result$fit_status <- "unsupported_shape"
    return(result)
  }

  support_range <- as.numeric(row$support_range)
  if (
    !is.finite(b) || b <= eps ||
    (is.finite(support_range) && b * support_range > max_log_slope_change)
  ) {
    result$function_form <- "linear_fallback"
    result$a <- tau_mean
    result$b <- 0
    result$fitted_average_tau <- tau_mean
    result$global_tau_relative_error <- 0
    result$eligible_for_moo <- tau_mean > 0
    result$fit_status <- "unstable_exponential_linear_fallback"
    return(result)
  }

  if (type == "Attractive") {
    a <- tau_left / (b * exp(b * (x_left - knot)))
    fitted_left <- attractive_derivative(x_left, knot, a, b)
    fitted_right <- attractive_derivative(x_right, knot, a, b)
  } else {
    a <- tau_left / (b * exp(-b * (x_left - knot)))
    fitted_left <- must_be_derivative(x_left, knot, a, b)
    fitted_right <- must_be_derivative(x_right, knot, a, b)
  }

  if (!is.finite(a) || a <= 0) {
    result$function_form <- "linear_fallback"
    result$a <- tau_mean
    result$b <- 0
    result$fitted_average_tau <- tau_mean
    result$global_tau_relative_error <- 0
    result$eligible_for_moo <- tau_mean > 0
    result$fit_status <- "invalid_exponential_scale_linear_fallback"
    return(result)
  }

  observed_w <- df$Wp[is.finite(df$Wp)]
  if (type == "Attractive") {
    fitted_tau <- attractive_derivative(observed_w, knot, a, b)
  } else {
    fitted_tau <- must_be_derivative(observed_w, knot, a, b)
  }

  fitted_average_tau <- mean(fitted_tau, na.rm = TRUE)
  relative_error <- abs(fitted_average_tau - tau_mean) / (abs(tau_mean) + eps)

  result$function_form <- form
  result$a <- a
  result$b <- b
  result$c <- 0
  result$fitted_tau_left <- fitted_left
  result$fitted_tau_right <- fitted_right
  result$fitted_average_tau <- fitted_average_tau
  result$global_tau_relative_error <- relative_error
  result$eligible_for_moo <- TRUE
  result$fit_status <- "success"
  result
}

parameter_results <- dplyr::bind_rows(
  lapply(seq_len(nrow(shape_summary)), function(i) {
    fname <- as.character(shape_summary$Feature[i])
    fit_satisfaction_parameters(
      row = shape_summary[i, , drop = FALSE],
      df = tau_results[[fname]]
    )
  })
)

final_results <- shape_summary |>
  dplyr::left_join(
    parameter_results,
    by = c("Feature", "shape_type"),
    suffix = c("", "_parameter")
  )

write.csv(
  parameter_results,
  file.path(OUTPUT_DIR, "satisfaction_function_parameters.csv"),
  row.names = FALSE
)

write.csv(
  final_results,
  file.path(OUTPUT_DIR, "satisfaction_shape_and_parameters.csv"),
  row.names = FALSE
)

saveRDS(
  final_results,
  file.path(OUTPUT_DIR, "satisfaction_shape_and_parameters.rds")
)

print(sort(table(parameter_results$fit_status), decreasing = TRUE))

# ============================================================
# 12. MOO helper functions and diagnostics
# ============================================================
satisfaction_value <- function(x, row) {
  form <- as.character(row$function_form)
  a <- as.numeric(row$a)
  b <- as.numeric(row$b)
  c <- as.numeric(row$c)
  knot <- as.numeric(row$knot)

  if (form == "attractive_exponential") {
    return(attractive_value(x, knot, a, b, c))
  }
  if (form == "must_be_exponential") {
    return(must_be_value(x, knot, a, b, c))
  }
  if (form %in% c("linear", "linear_fallback", "linear_reverse_diagnostic")) {
    return(linear_value(x, a, c))
  }
  if (form == "constant") {
    return(rep(c, length(x)))
  }

  stop("Unsupported or unavailable function form: ", form)
}

satisfaction_change <- function(current_performance, improvement, row) {
  target_performance <- pmin(
    10,
    pmax(1, current_performance + improvement)
  )

  satisfaction_value(target_performance, row) -
    satisfaction_value(current_performance, row)
}

# Classification count plot.
type_plot <- ggplot(shape_summary, aes(x = shape_type)) +
  geom_bar() +
  coord_flip() +
  labs(
    x = NULL,
    y = "Number of service elements",
    title = "Dominant local satisfaction-shape classifications"
  ) +
  theme_minimal(base_size = 12)

ggsave(
  filename = file.path(OUTPUT_DIR, "shape_type_counts.png"),
  plot = type_plot,
  width = 8,
  height = 5,
  dpi = 300
)

# Export a compact model-ready table for MOO.
moo_parameters <- final_results |>
  dplyr::select(
    Feature,
    Tau_Hat_Mean,
    AME_SE,
    shape_type,
    function_form,
    knot = reference_point,
    a,
    b,
    c,
    support_low,
    support_high,
    eligible_for_moo,
    fit_status,
    global_tau_relative_error
  )

write.csv(
  moo_parameters,
  file.path(OUTPUT_DIR, "moo_satisfaction_parameters.csv"),
  row.names = FALSE
)

message("All outputs were written to: ", OUTPUT_DIR)