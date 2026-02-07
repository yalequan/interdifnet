#' Generate Training Data for Ten-Group DIF Detection
#'
#' This script generates simulated training datasets for detecting differential
#' item functioning (DIF) in item response theory (IRT) models with ten groups.
#' The data generation follows a 2-parameter logistic (2PL) IRT model with
#' varying DIF patterns across groups.
#'
#' @section Data Generation Parameters:
#' - Number of groups (S): 10
#' - Number of items (J): 10
#' - Sample size (N): Randomly selected between 1000 and 5000 (in 500 increments)
#' - Number of DIF items (m): Randomly selected from {2, 3, 4}
#' - DIF types: Uniform (b), non-uniform (a), or both (ab)
#'
#' @section Group Configuration:
#' - Group size ratios: {0.10, 0.10, 0.15, 0.15, 0.05, 0.05, 0.15, 0.15, 0.05, 0.05}
#' - Ability distribution: Normal with group-specific means and unit variance
#' - Group means: Reference groups (1-2) = 0, Groups 3-6 = 1, Groups 7-10 = -1
#'
#' @section DIF Implementation:
#' - Reference groups: Groups 1 and 2 (no DIF)
#' - DIF magnitude for discrimination (a): Drawn from [-1, 1] by 0.25 (excluding 0)
#' - DIF magnitude for difficulty (b): Drawn from [-1.5, 1.5] by 0.25 (excluding 0)
#'
#' @section Output:
#' Each replication saves an RData file containing:
#' - y: N x J matrix of binary item responses
#' - a: S x J matrix of discrimination parameters
#' - b: S x J matrix of difficulty parameters
#' - d_a, d_b: S x S x J arrays of pairwise parameter differences
#' - Labels_a, Labels_b, Labels_DIF: Binary DIF indicators for pairwise comparisons
#' - temp_d_a, temp_d_b: Continuous parameter differences
#' - selected_items: Indices of items with DIF
#' - g: Group membership vector
#'
#' @author Yale Quan
#' @date 2024-07-01
#'
#' @examples
#' # This script is designed to be run as a standalone file
#' # It will generate 800 training datasets in the current directory
#'
#' @note
#' - Total replications: 500
#' - Random seed: Set based on group count and replication number for reproducibility
#' - File naming: "Ten_Group_Training_Data_Replication_[r].RData"

# Disable scientific notation for cleaner output
options(scipen = 9999)

# Clear workspace
rm(list = ls())

# Setup Parameters ----

S <- 10  # Number of groups
J <- 10  # Number of items

# Group size proportions (must sum to 1.0)
ratio <- c(0.1, 0.1, 0.15, 0.15, 0.05, 0.05, 0.15, 0.15, 0.05, 0.05)

# Standard deviations for ability distribution (one per group)
sigma <- c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

# Grid for discrimination parameter DIF (excluding 0)
grid.a <- seq(-1, 1, by = 0.25)
grid.a <- grid.a[grid.a != 0]

# Grid for difficulty parameter DIF (excluding 0)
grid.b <- seq(-1.5, 1.5, by = 0.25)
grid.b <- grid.b[grid.b != 0]

# Mean ability parameters for each group
dif.mu <- c(0, 0, 1, 1, 1, 1, -1, -1, -1, -1)

# Main Data Generation Loop ----
# Generate 500 independent training datasets with varying configurations

for (r in 1:500) {
  # Set random seed for reproducibility
  set.seed(202407)
  set.seed(sample.int(2147483647, 10000)[S * r])
  
  # Randomly determine sample size (1000, 1500, 2000, ..., 5000)
  N <- sample(c(seq(from = 1, to = 5, by = 0.5)), 1) * 1000
  
  # Randomly select number of DIF items (2, 3, or 4)
  m <- sample(c(2, 3, 4), 1)
  
  # Assign examinees to groups based on specified ratios
  g <- do.call(c, lapply(1:S, function(s) rep(s, N * ratio[s])))
  
  # Randomly select which items will exhibit DIF
  selected_items <- sample(1:J, m, replace = FALSE)
  
  # Sample DIF magnitudes for discrimination parameters
  # Reference groups (1 and 2) have no DIF
  dif.a <- sample(grid.a, size = S, replace = TRUE)
  dif.a[c(1, 2)] <- 0
  
  # Sample DIF magnitudes for difficulty parameters
  # Reference groups (1 and 2) have no DIF
  dif.b <- sample(grid.b, size = S, replace = TRUE)
  dif.b[c(1, 2)] <- 0
  
  # Generate Item Parameters ----
  
  # Randomly select type of DIF: "b" (uniform), "a" (non-uniform), or "ab" (both)
  dif_types <- c("b", "ab", "a")
  rep_dif <- sample(dif_types, 1)
  
  # Generate baseline discrimination parameters (between 1.5 and 2.5)
  a0 <- runif(J, 1.5, 2.5)
  
  # Generate baseline difficulty parameters (standard normal)
  b0 <- rnorm(J)
  
  # Replicate baseline parameters for all groups
  a <- t(replicate(S, a0))
  b <- t(replicate(S, b0))
  
  # Apply DIF shifts to selected items based on DIF type
  if (rep_dif == "ab") {
    # Both discrimination and difficulty DIF
    a[, selected_items] <- a[, selected_items] + dif.a
    b[, selected_items] <- b[, selected_items] + dif.b
  } else if (rep_dif == "b") {
    # Uniform DIF (difficulty only)
    b[, selected_items] <- b[, selected_items] + dif.b
  } else if (rep_dif == "a") {
    # Non-uniform DIF (discrimination only)
    a[, selected_items] <- a[, selected_items] + dif.a
  }
  
  # Compute Pairwise Parameter Differences ----
  
  # Initialize 3D arrays to store pairwise differences
  # d_a[k, n, j] = discrimination difference between groups k and n for item j
  # d_b[k, n, j] = difficulty difference between groups k and n for item j
  d_a <- array(0, c(S, S, J))
  d_b <- array(0, c(S, S, J))
  
  # Compute all pairwise differences
  for (k in 1:(S - 1)) {
    for (n in (k + 1):S) {
      d_a[k, n, ] <- a[k, ] - a[n, ]
      d_b[k, n, ] <- b[k, ] - b[n, ]
    }
  }
  
  # Create Training Labels ----
  
  # Compute pairwise differences for discrimination parameters
  # Each column represents one group comparison
  temp_d_a <- combn(1:nrow(a), 2, function(idx) {
    diff <- a[idx[1], ] - a[idx[2], ]
    return(diff)
  })
  colnames(temp_d_a) <- apply(
    combn(1:ncol(a), 2), 2,
    function(idx) paste0("d.a_", "Group", idx[1], "Group", idx[2])
  )
  
  # Compute pairwise differences for difficulty parameters
  temp_d_b <- combn(1:nrow(a), 2, function(idx) {
    diff <- b[idx[1], ] - b[idx[2], ]
    return(diff)
  })
  colnames(temp_d_b) <- apply(
    combn(1:ncol(b), 2), 2,
    function(idx) paste0("d.b_", "Group", idx[1], "Group", idx[2])
  )
  
  # Create binary labels for discrimination DIF (1 = DIF present, 0 = no DIF)
  Labels_a <- temp_d_a
  Labels_a[Labels_a != 0] <- 1
  colnames(Labels_a) <- apply(
    combn(1:ncol(a), 2), 2,
    function(idx) paste0("DIF_a_", "Group", idx[1], "Group", idx[2])
  )
  
  # Create binary labels for difficulty DIF
  Labels_b <- temp_d_b
  Labels_b[Labels_b != 0] <- 1
  colnames(Labels_b) <- apply(
    combn(1:ncol(b), 2), 2,
    function(idx) paste0("DIF_b_", "Group", idx[1], "Group", idx[2])
  )
  
  # Create overall DIF labels (DIF in either a or b or both)
  Labels_DIF <- pmax(Labels_a, Labels_b)
  colnames(Labels_DIF) <- apply(
    combn(1:ncol(b), 2), 2,
    function(idx) paste0("DIF_", "Group", idx[1], "Group", idx[2])
  )
  
  # Generate Observed Item Responses ----
  
  # Simulate responses using 2PL IRT model:
  # P(Y_ij = 1 | theta, a, b) = logit^(-1)(a_j * theta_i - b_j)
  # where theta_i ~ N(mu_s, sigma_s) for examinee i in group s
  y <- t(sapply(1:N, function(n) {
    s <- g[n]  # Group membership for examinee n
    theta <- rnorm(1, dif.mu[s], sigma[s])  # Sample ability from group distribution
    rbinom(J, 1, plogis(a[s, ] * theta - b[s, ]))  # Generate binary responses
  }))
  
  # Save Workspace ----
  
  # Save all objects to RData file for later use in training
  save.image(file = paste0("Ten_Group_Training_Data_Replication_", r, ".RData"))
}

# End of script
