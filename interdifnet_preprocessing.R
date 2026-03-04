# =============================================================================
# InterDIFNet Pre-Processing
# =============================================================================
# Generates TLP feature matrix for InterDIFNet neural network.
# Supports ONLY 3-group and 10-group configurations.
# =============================================================================

difnet_generate_features <- function(
    item_responses,
    group_assignments,
    num_groups = NULL,
    seed = 123,
    output_path = "interdifnet_features.csv",
    write_output = TRUE,
    verbose = TRUE
) {

  # ---------------------------------------------------------------------------
  # Dependency Checks
  # ---------------------------------------------------------------------------

  if (!requireNamespace("VEMIRT", quietly = TRUE)) {
    stop("Package 'VEMIRT' is required but not installed.")
  }

  if (!requireNamespace("stringr", quietly = TRUE)) {
    stop("Package 'stringr' is required but not installed.")
  }

  # ---------------------------------------------------------------------------
  # Input Validation
  # ---------------------------------------------------------------------------

  if (!is.data.frame(item_responses) && !is.matrix(item_responses)) {
    stop("item_responses must be a data frame or matrix.")
  }

  item_responses <- as.data.frame(item_responses)

  if (anyNA(item_responses)) {
    stop("Item responses contain missing values.")
  }

  if (!all(sapply(item_responses, is.numeric))) {
    stop("Item responses must be numeric.")
  }

  if (!all(unlist(item_responses) %in% c(0,1))) {
    stop("Item responses must be binary (0/1).")
  }

  if (!is.numeric(group_assignments) || anyNA(group_assignments)) {
    stop("group_assignments must be numeric without NAs.")
  }

  if (nrow(item_responses) != length(group_assignments)) {
    stop("Mismatch between number of examinees and group vector length.")
  }

  N <- nrow(item_responses)
  J <- ncol(item_responses)

  if (is.null(num_groups)) {
    num_groups <- max(group_assignments)
  }

  S <- as.integer(num_groups)

  if (!(S %in% c(3, 10))) {
    stop("DIFNet currently supports only 3-group or 10-group models.")
  }

  if (!identical(sort(unique(group_assignments)), 1:S)) {
    stop("Groups must be consecutive integers 1:S.")
  }

  group_sizes <- as.numeric(table(group_assignments))

  if (any(group_sizes < 30)) {
    warning("Some groups have fewer than 30 examinees. Estimates may be unstable.")
  }

  item_var <- apply(item_responses, 2, var)

  if (any(item_var == 0)) {
    stop("Some items have zero variance (all 0 or all 1). Please remove degenerate items.")
  }

  if (verbose) {
    message("Data: N=", N, ", J=", J, ", Groups=", S)
  }

  # ---------------------------------------------------------------------------
  # Estimation
  # ---------------------------------------------------------------------------

  set.seed(seed)

  if (verbose) message("Estimating TLP parameters...")

  TLP_model <- tryCatch({

    VEMIRT::D2PL_pair_em(
      data = item_responses,
      group = as.integer(group_assignments),
      Lambda0 = seq(0.1, 1.5, by = 0.1),
      Tau = c(Inf, seq(0.05, 0.5, by = 0.05)),
      verbose = FALSE
    )

  }, error = function(e) {
    stop("TLP estimation failed: ", conditionMessage(e))
  })

  bic_vals <- sapply(TLP_model$all, `[[`, "BIC")
  best_model <- TLP_model$all[[which.min(bic_vals)]]

  TLP_a <- best_model$a
  TLP_b <- best_model$b

  # ---------------------------------------------------------------------------
  # Pair Matrix
  # ---------------------------------------------------------------------------

  pair_matrix <- t(combn(1:S, 2))
  num_pairs <- nrow(pair_matrix)

  # ---------------------------------------------------------------------------
  # Pairwise Differences
  # ---------------------------------------------------------------------------

  TLP_d_a <- sapply(1:num_pairs, function(i) {
    g1 <- pair_matrix[i,1]
    g2 <- pair_matrix[i,2]
    TLP_a[g1, ] - TLP_a[g2, ]
  })

  TLP_d_b <- sapply(1:num_pairs, function(i) {
    g1 <- pair_matrix[i,1]
    g2 <- pair_matrix[i,2]
    TLP_b[g1, ] - TLP_b[g2, ]
  })

  # ---------------------------------------------------------------------------
  # Reshape
  # ---------------------------------------------------------------------------

  TLP_a <- t(TLP_a)
  TLP_b <- t(TLP_b)

  colnames(TLP_a) <- paste0("TLP_a_Group", 1:S)
  colnames(TLP_b) <- paste0("TLP_b_Group", 1:S)

  colnames(TLP_d_a) <- paste0(
    "TLP_d.a_Group",
    pair_matrix[,1],
    "Group",
    pair_matrix[,2]
  )

  colnames(TLP_d_b) <- paste0(
    "TLP_d.b_Group",
    pair_matrix[,1],
    "Group",
    pair_matrix[,2]
  )

  features <- cbind(TLP_a, TLP_b, TLP_d_a, TLP_d_b)
  features <- as.data.frame(features)

  if (verbose) {
    message("Feature matrix generated: ",
            nrow(features), " items x ",
            ncol(features), " features")
  }

  # ---------------------------------------------------------------------------
  # Write Output (Optional)
  # ---------------------------------------------------------------------------

  if (write_output) {

    if (file.exists(output_path)) {
      warning("Output file already exists and will be overwritten: ", output_path)
    }

    write.csv(features, output_path, row.names = FALSE)

    if (verbose) {
      message("Features written to: ", normalizePath(output_path))
    }
  }

  # ---------------------------------------------------------------------------
  # Return
  # ---------------------------------------------------------------------------

  return(list(
    features = features,
    metadata = list(
      num_items = J,
      num_examinees = N,
      num_groups = S,
      num_pairs = num_pairs,
      group_sizes = group_sizes,
      seed = seed,
      output_path = if (write_output) output_path else NULL
    )
  ))
}