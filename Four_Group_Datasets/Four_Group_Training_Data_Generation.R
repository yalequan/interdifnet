################################################################################
# Four Group DIF Training Data Generation
#
# This script generates simulated training data for four-group Differential Item 
# Functioning (DIF) detection. It creates multiple replications of IRT 2PL model 
# data with varying sample sizes, numbers of DIF items, and types of DIF 
# (non-uniform, uniform, or both).
#
# Simulation Parameters:
#   - Groups (S): 4 comparison groups
#   - Items (J): 10 items per assessment
#   - Sample Size (N): Randomly selected from 250-2000 (weighted toward N < 1000)
#   - DIF Items (m): Randomly 2, 3, or 4 items with DIF
#   - DIF Types: Non-uniform (a-parameter), uniform (b-parameter), or both
#   - Replications: 1000 datasets
#
# DIF Parameters:
#   - Non-uniform DIF (a-parameter): Grid from -1 to 1 (excluding 0)
#   - Uniform DIF (b-parameter): Grid from -1.5 to 1.5 (excluding 0)
#   - Reference groups: Groups 1 and 2 (no DIF)
#   - Focal groups: Groups 3 and 4 (DIF possible)
#
# Output:
#   - Four_Group_Training_Data_Replication_*.RData files containing:
#     * Item parameters (a, b) for all groups
#     * Response data (y) from 2PL IRT model
#     * DIF labels for all pairwise group comparisons (Labels_a, Labels_b)
#     * Group assignments (g), sample sizes (N), and DIF item indices
#
# Created on: December 2024
# @author: Yale Quan
################################################################################

options(scipen = 9999)

rm(list = ls())

# Setup Parameters ----

S <- 4  # Number of groups
J <- 10 # Number of items
dif.mu <- c(0, 0, 1, -1)  # Mean ability for each group
sigma <- c(1, 1, 1, 1)    # SD of ability for each group

# DIF parameter grids
grid.a <- seq(-1, 1, by = 0.25)
grid.a <- grid.a[grid.a != 0]    # Non-uniform DIF range (exclude 0)
grid.b <- seq(-1.5, 1.5, by = 0.25)
grid.b <- grid.b[grid.b != 0]    # Uniform DIF range (exclude 0)

# Group size proportions
ratio <- c(.3, .3, .2, .2)

# Sample size distribution (weighted toward smaller samples)
n_vals <- 250:2000
valid_n <- n_vals[n_vals %% 10 == 0]  # Ensure integer group sizes
weights <- ifelse(valid_n < 1000, 3, 1)  # 3x more likely if N < 1000
weights <- weights / sum(weights)

# Generate Training Data Replications ----

for (r in 1:1000) {
  set.seed(202407)
  set.seed(sample.int(2147483647, 10000)[S * r])
  
  # Randomly sample training data parameters
  N <- sample(valid_n, size = 1, prob = weights)  # Total sample size
  m <- sample(c(2, 3, 4), 1)                      # Number of DIF items
  
  # Assign participants to groups based on specified ratios
  g <- do.call(c, lapply(1:S, function(s) rep(s, N * ratio[s])))
  
  # Randomly select which items will have DIF
  selected_items <- sample(1:J, m, replace = FALSE)
  
  # Generate DIF parameter shifts for focal groups
  dif.a <- sample(grid.a, size = S, replace = TRUE)
  dif.a[c(1, 2)] <- 0  # Groups 1 and 2 are reference groups (no DIF)
  dif.b <- sample(grid.b, size = S, replace = TRUE)
  dif.b[c(1, 2)] <- 0  # Groups 1 and 2 are reference groups (no DIF)
  
  # Randomly select DIF type: non-uniform (a), uniform (b), or both (ab)
  dif_types <- c("b", "ab", "a")
  rep_dif <- sample(dif_types, 1)
  
  # Generate baseline item parameters (common across groups)
  a0 <- runif(J, 1.5, 2.5)  # Discrimination parameters
  b0 <- rnorm(J)            # Difficulty parameters
  a <- t(replicate(S, a0))
  b <- t(replicate(S, b0))
  
  # Apply DIF shifts to selected items
  if (rep_dif == "ab") {
    # Both non-uniform and uniform DIF
    a[, selected_items] <- a[, selected_items] + dif.a
    b[, selected_items] <- b[, selected_items] + dif.b
  } else if (rep_dif == "b") {
    # Uniform DIF only
    b[, selected_items] <- b[, selected_items] + dif.b
  } else if (rep_dif == "a") {
    # Non-uniform DIF only
    a[, selected_items] <- a[, selected_items] + dif.a
  }
  
  # Calculate pairwise DIF differences between all groups
  d_a <- array(0, c(S, S, J))
  d_b <- array(0, c(S, S, J))
  for (k in 1:(S - 1)) {
    for (n in (k + 1):S) {
      d_a[k, n, ] <- a[k, ] - a[n, ]
      d_b[k, n, ] <- b[k, ] - b[n, ]
    }
  }
  
  # Create continuous DIF difference matrices for all pairwise comparisons
  temp_d_a <- combn(1:nrow(a), 2, function(idx) {
    diff <- a[idx[1], ] - a[idx[2], ]
    return(diff)
  })
  colnames(temp_d_a) <- apply(combn(1:S, 2), 2,
                              function(idx) paste0("d.a_", "Group", idx[1],
                                                   "Group", idx[2]))
  
  temp_d_b <- combn(1:nrow(a), 2, function(idx) {
    diff <- b[idx[1], ] - b[idx[2], ]
    return(diff)
  })
  colnames(temp_d_b) <- apply(combn(1:S, 2), 2,
                              function(idx) paste0("d.b_", "Group",
                                                   idx[1], "Group", idx[2]))
  
  # Create binary DIF labels (0 = no DIF, 1 = DIF present)
  # Non-uniform DIF labels
  Labels_a <- temp_d_a
  Labels_a[Labels_a != 0] <- 1
  colnames(Labels_a) <- apply(combn(1:S, 2), 2, 
                              function(idx) paste0("DIF_a_", "Group", 
                                                   idx[1], "Group", idx[2]))
  
  # Uniform DIF labels
  Labels_b <- temp_d_b
  Labels_b[Labels_b != 0] <- 1
  colnames(Labels_b) <- apply(combn(1:S, 2), 2, 
                              function(idx) paste0("DIF_b_", "Group", 
                                                   idx[1], "Group", idx[2]))
  
  # Overall DIF labels (DIF present if either type exists)
  Labels_DIF <- pmax(Labels_a, Labels_b)
  colnames(Labels_DIF) <- apply(combn(1:S, 2), 2, 
                                function(idx) paste0("DIF_", "Group", 
                                                     idx[1], "Group", idx[2]))
  
  # Generate observed item responses using 2PL IRT model
  # P(Y = 1 | theta) = logit^-1(a * theta - b)
  y <- t(sapply(1:N, function(n) {
    s <- g[n]
    theta <- rnorm(1, dif.mu[s], sigma[s])
    rbinom(J, 1, plogis(a[s, ] * theta - b[s, ]))
  }))
  
  # Save all workspace variables for this replication
  save.image(file = paste0("Ten_Group_Training_Data_Replication_", r, ".RData"))
}

