#' Three-Group Training Data Generation
#'
#' @description
#' This script generates simulated training datasets for the three-group scenario
#' of the InterDIFNet study. It creates 100 replications with randomized
#' conditions (sample size, DIF percentage, DIF type) to train deep learning
#' models for DIF detection. The three-group design provides a simpler validation
#' scenario compared to the main 10-group study.
#'
#' @section Purpose:
#' Generates training data for the simplified three-group scenario. While the
#' main study uses 10 groups with 45 pairwise comparisons, this three-group
#' scenario with only 3 pairwise comparisons serves multiple purposes:
#' - Rapid prototyping and initial model development
#' - Method validation in a simpler setting
#' - Clearer interpretation of group-specific DIF patterns
#' - Computational efficiency for hyperparameter tuning
#' - Verification that methods generalize across different numbers of groups
#'
#' @section Training Data Design:
#' Creates 100 training replications (replications 501-600) with randomized:
#' - Sample Size: N ~ 250-2000 (increments of 5), weighted toward N < 1000
#' - DIF Items: m ~ Uniform{2, 3, 4} items with DIF (20%, 30%, 40%)
#' - DIF Type: Uniform{a-DIF, b-DIF, both} for discrimination/difficulty shifts
#' - DIF Magnitude: Random draws from predefined grids
#' - Groups: 3 groups with 60%/20%/20% ratio (majority in reference)
#' - Pairwise Comparisons: 3 (Group1vs2, Group1vs3, Group2vs3)
#'
#' @section DIF Implementation:
#' Three types of DIF are randomly applied:
#' 1. **a-DIF (Discrimination)**: Non-uniform DIF via slopes
#'    - Grid: seq(-1, 1, by=0.25), excluding 0
#'    - Group 1 (reference): shift = 0
#'    - Groups 2-3: random draws from grid
#' 2. **b-DIF (Difficulty)**: Uniform DIF via intercepts
#'    - Grid: seq(-1.5, 1.5, by=0.25), excluding 0
#'    - Group 1 (reference): shift = 0
#'    - Groups 2-3: random draws from grid
#' 3. **ab-DIF (Both)**: Combined discrimination and difficulty shifts
#'
#' @section Population Parameters:
#' - Groups: S = 3 (reference, positive, negative)
#' - Items: J = 10
#' - Group Means: μ = (0, +1, -1)
#' - Group SDs: σ = (1, 1, 1)
#' - Base Discrimination: a₀ ~ Uniform(1.5, 2.5)
#' - Base Difficulty: b₀ ~ N(0, 1)
#' - Group Ratio: 60% : 20% : 20%
#'
#' @section DIF Parameter Grids:
#' - Discrimination Shifts (a): {-1, -0.75, -0.5, -0.25, +0.25, +0.5, +0.75, +1}
#' - Difficulty Shifts (b): {±0.25, ±0.5, ±0.75, ±1, ±1.25, ±1.5}
#' - Reference Group: Always 0 shift (baseline for comparisons)
#' - Focal Groups: Random selection from grids
#'
#' @section Sample Size Distribution:
#' Training data uses weighted sampling for realistic variation:
#' - Range: N ∈ {250, 255, 260, ..., 1995, 2000}
#' - Constraint: N divisible by 5 (ensures integer group sizes)
#' - Weighting: P(N < 1000) = 3 × P(N ≥ 1000)
#' - Rationale: Smaller samples more common in practice
#'
#' @section Output Variables:
#' Each RData file contains:
#' - `y`: N × J response matrix (binary item responses)
#' - `g`: N-vector of group membership (1, 2, or 3)
#' - `a`: S × J matrix of discrimination parameters
#' - `b`: S × J matrix of difficulty parameters
#' - `d_a`: S × S × J array of pairwise discrimination differences
#' - `d_b`: S × S × J array of pairwise difficulty differences
#' - `Labels_a`: Binary labels for a-DIF across 3 pairwise comparisons
#' - `Labels_b`: Binary labels for b-DIF across 3 pairwise comparisons
#' - `Labels_DIF`: Binary labels for any DIF across 3 pairwise comparisons
#' - `N`, `m`, `selected_items`, `rep_dif`: Simulation metadata
#'
#' @section Workflow:
#' 1. Set fixed population parameters (S, J, μ, σ, grids, ratios)
#' 2. Define sample size distribution with weighting
#' 3. For each replication (r = 501 to 600):
#'    a. Set random seed for reproducibility
#'    b. Sample random N, m, and DIF type
#'    c. Assign group membership based on ratios
#'    d. Select items to have DIF
#'    e. Generate base item parameters (a₀, b₀)
#'    f. Apply DIF shifts to selected items
#'    g. Calculate pairwise parameter differences
#'    h. Create binary DIF labels for all comparisons
#'    i. Generate theta values by group
#'    j. Generate binary responses using 2PL model
#'    k. Save complete workspace to RData file
#'
#' @section Label Generation:
#' Binary labels indicate presence of DIF for each pairwise comparison:
#' - `Labels_a`: 1 if discrimination differs between groups, 0 otherwise
#' - `Labels_b`: 1 if difficulty differs between groups, 0 otherwise
#' - `Labels_DIF`: 1 if either a or b differs (combined DIF indicator)
#' - Dimensionality: 3 comparisons × 10 items = 30 binary labels per type
#'
#' @section Computational Details:
#' - Replications: 100 training files (501-600)
#' - Seed Strategy: Deterministic based on replication number
#' - Memory: Full workspace saved per replication
#' - Output Format: RData files for direct loading in R
#' - Three-group advantage: ~15x faster than 10-group scenario
#'
#' @author Yale Quan
#' @date February 2026
#'
#' @examples
#' # This script is designed to be run as a standalone batch process:
#' # source("Three_Group_Training_Data_Generation.R")
#' 
#' # Output file naming convention:
#' # "Three_Group_Training_Data_Replication_{r}.RData"
#' # Example: "Three_Group_Training_Data_Replication_550.RData"
#' 
#' # To load a specific replication:
#' # load("Three_Group_Training_Data_Replication_550.RData")

# Setup ----
options(scipen = 9999)
rm(list = ls())

# Population Parameters ----
S <- 3  # Number of groups (reference, positive, negative)
J <- 10 # Number of items
dif.mu <- c(0, 1, -1) # Group-specific ability means
sigma <- c(1, 1, 1)   # Group-specific ability SDs
grid.a <- seq(-1, 1, by = 0.25) # Discrimination shift grid
grid.a <- grid.a[grid.a != 0]   # Exclude zero shift
grid.b <- seq(-1.5, 1.5, by = 0.25) # Difficulty shift grid
grid.b <- grid.b[grid.b != 0]       # Exclude zero shift
ratio <- c(.6, .2, .2) # Group distribution: 60%/20%/20%

# Sample Size Distribution ----
# Weighted toward smaller samples (N < 1000 more likely)
n_vals <- 250:2000
valid_n <- n_vals[n_vals %% 5 == 0] # Ensure integer group sizes with ratio
weights <- ifelse(valid_n < 1000, 3, 1)  # 3x probability for N < 1000
weights <- weights / sum(weights)

# Main Loop: Generate Training Data ----
for (r in 501:600) { # 100 training replications
  
  # Random Seed for Reproducibility ----
  set.seed(202407)
  set.seed(sample.int(2147483647, 10000)[S * r])
  
  # Random Sample Size ----
  N <- sample(valid_n, size = 1, prob = weights)
  
  # Random Number of DIF Items ----
  m <- sample(c(2, 3, 4), 1) # 20%, 30%, or 40% DIF
  
  # Group Membership Assignment ----
  g <- do.call(c, lapply(1:S, function(s) rep(s, N * ratio[s])))
  
  # Select DIF Items ----
  selected_items <- sample(1:J, m, replace = FALSE)
  
  # Generate DIF Shifts ----
  dif.a <- sample(grid.a, size = S, replace = TRUE)
  dif.a[1] <- 0 # Reference group has no shift
  dif.b <- sample(grid.b, size = S, replace = TRUE)
  dif.b[1] <- 0 # Reference group has no shift
  
  # Select DIF Type ----
  dif_types <- c("b", "ab", "a")  # b-DIF, both, or a-DIF
  rep_dif <- sample(dif_types, 1)
  
  # Generate Base Item Parameters ----
  a0 <- runif(J, 1.5, 2.5) # Base discrimination
  b0 <- rnorm(J)           # Base difficulty
  a <- t(replicate(S, a0)) # Replicate across groups
  b <- t(replicate(S, b0))
  
  # Apply DIF Shifts to Selected Items ----
  if (rep_dif == "ab") {
    a[, selected_items] <- a[, selected_items] + dif.a
    b[, selected_items] <- b[, selected_items] + dif.b
  } else if (rep_dif == "b") {
    b[, selected_items] <- b[, selected_items] + dif.b
  } else if (rep_dif == "a") {
    a[, selected_items] <- a[, selected_items] + dif.a
  }
  
  # Calculate Pairwise Parameter Differences ----
  d_a <- array(0, c(S, S, J))
  d_b <- array(0, c(S, S, J))
  for (k in 1:(S - 1)) {
    for (n in (k + 1):S) {
      d_a[k, n, ] <- a[k, ] - a[n, ]
      d_b[k, n, ] <- b[k, ] - b[n, ]
    }
  }
  
  # Create Pairwise Difference Matrices ----
  temp_d_a <- combn(1:nrow(a), 2, function(idx) {
    diff <- a[idx[1], ] - a[idx[2], ]
    return(diff)
  })
  colnames(temp_d_a) <- apply(combn(1:ncol(temp_d_a), 2), 2,
                               function(idx) paste0("d.a_", "Group", idx[1],
                                                    "Group", idx[2]))
  
  temp_d_b <- combn(1:nrow(a), 2, function(idx) {
    diff <- b[idx[1], ] - b[idx[2], ]
    return(diff)
  })
  colnames(temp_d_b) <- apply(combn(1:ncol(temp_d_b), 2), 2,
                               function(idx) paste0("d.b_", "Group",
                                                    idx[1], "Group", idx[2]))
  
  # Create Binary Labels for a-DIF ----
  Labels_a <- temp_d_a
  Labels_a[Labels_a != 0] <- 1
  colnames(Labels_a) <- apply(combn(1:ncol(Labels_a), 2), 2, 
                               function(idx) paste0("DIF_a_", "Group", 
                                                    idx[1], "Group", idx[2]))
  
  # Create Binary Labels for b-DIF ----
  Labels_b <- temp_d_b
  Labels_b[Labels_b != 0] <- 1
  colnames(Labels_b) <- apply(combn(1:ncol(Labels_b), 2), 2, 
                               function(idx) paste0("DIF_b_", "Group", 
                                                    idx[1], "Group", idx[2]))
  
  # Create Combined DIF Labels ----
  Labels_DIF <- pmax(Labels_a, Labels_b)
  colnames(Labels_DIF) <- apply(combn(1:ncol(Labels_DIF), 2), 2, 
                                 function(idx) paste0("DIF_", "Group", 
                                                      idx[1], "Group", idx[2]))
  
  # Generate Latent Abilities and Responses ----
  y <- t(sapply(1:N, function(n) {
    s <- g[n]
    theta <- rnorm(1, dif.mu[s], sigma[s])
    rbinom(J, 1, plogis(a[s, ] * theta - b[s, ]))
  }))
  
  # Save Workspace ----
  save.image(file = paste0("Three_Group_Training_Data_Replication_", r, ".RData"))
}
