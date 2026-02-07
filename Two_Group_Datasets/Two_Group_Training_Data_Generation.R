#' Two-Group Training Data Generation
#'
#' @description
#' This script generates simulated training datasets for the two-group scenario
#' of the InterDIFNet study. It creates IRT response data under various
#' conditions of differential item functioning (DIF) for training deep learning
#' models in the simplest classical two-group setting.
#'
#' @section Purpose:
#' Generates training data for the two-group scenario (classical baseline
#' condition). While the main study uses 10 groups with 45 pairwise comparisons,
#' and validation uses 3 groups with 3 pairwise comparisons, this two-group
#' scenario with only 1 pairwise comparison provides:
#' - Training data for baseline two-group DIF detection
#' - Simplest possible multi-group scenario (S=2)
#' - Classical reference vs focal group comparison
#' - Benchmark for comparing with complex multi-group scenarios
#' - Traditional DIF setting used in most literature
#'
#' @section Simulation Parameters:
#' - **Number of Replications**: 600 training datasets
#' - **Groups (S)**: 2 (reference and focal)
#' - **Items (J)**: 10
#' - **Group Distribution**: 60% reference, 40% focal
#' - **Sample Sizes (N)**: 100-1000 (multiples of 5)
#'   * Higher probability for N < 750 (3x weight)
#'   * Ensures integer group sizes with ratio 60%/40%
#' - **DIF Items (m)**: Randomly 2, 3, or 4 items per dataset
#' - **DIF Types**: "a" (discrimination), "b" (difficulty), "ab" (both)
#'
#' @section IRT Model:
#' Uses the 2-Parameter Logistic (2PL) model:
#' \deqn{P(Y_{ij} = 1 | \theta_i, a_j, b_j) = \frac{1}{1 + \exp(-a_j(\theta_i - b_j))}}
#' where:
#' - \eqn{\theta_i}: Latent ability for person i ~ N(μ_s, σ_s)
#' - \eqn{a_j}: Discrimination parameter ~ Uniform(1.5, 2.5)
#' - \eqn{b_j}: Difficulty parameter ~ N(0, 1)
#' - Group means: μ = (0, 1) for reference and focal groups
#' - Group standard deviations: σ = (1, 1)
#'
#' @section DIF Implementation:
#' DIF is introduced through parameter shifts:
#' - **Discrimination shift grid**: seq(-1, 1, by=0.25), excluding 0
#' - **Difficulty shift grid**: seq(-1.5, 1.5, by=0.25), excluding 0
#' - **Reference group**: Always has zero shift (baseline)
#' - **Focal group**: Random shift from grid
#' - **DIF pattern**: Randomly selected as:
#'   * "b": Difficulty DIF only (uniform DIF)
#'   * "a": Discrimination DIF only (non-uniform DIF)
#'   * "ab": Both types (compound DIF)
#'
#' @section Output Structure:
#' Each replication saves an RData file containing:
#' - **y**: N × J response matrix (0/1)
#' - **g**: Group membership vector (1=reference, 2=focal)
#' - **a**: S × J discrimination parameters (true values)
#' - **b**: S × J difficulty parameters (true values)
#' - **d_a**: S × S × J discrimination differences array
#' - **d_b**: S × S × J difficulty differences array
#' - **temp_d_a**: Pairwise discrimination differences (1 comparison)
#' - **temp_d_b**: Pairwise difficulty differences (1 comparison)
#' - **Labels_a**: Binary DIF labels for discrimination
#' - **Labels_b**: Binary DIF labels for difficulty
#' - **Labels_DIF**: Combined binary DIF labels (1=DIF, 0=no DIF)
#' - **Simulation metadata**: S, J, N, m, ratio, dif.mu, sigma, etc.
#'
#' @section Two-Group Design:
#' - **Classical Setting**: Traditional reference vs focal comparison
#' - **Single Comparison**: Only 1 pairwise comparison possible
#' - **Simplest Scenario**: Baseline for multi-group complexity
#' - **Standard Distribution**: 60%/40% split common in DIF literature
#' - **Use Case**: Training baseline models for comparison with 3-group and 10-group
#'
#' @section Workflow:
#' 1. Set seed for reproducibility (based on replication number)
#' 2. Randomly sample N from weighted distribution
#' 3. Randomly select m DIF items (2, 3, or 4)
#' 4. Randomly select DIF type ("a", "b", or "ab")
#' 5. Generate baseline item parameters (a0, b0)
#' 6. Apply DIF shifts to selected items
#' 7. Calculate all pairwise parameter differences (1 pair)
#' 8. Create binary DIF labels for classification
#' 9. Generate ability parameters by group
#' 10. Generate observed binary responses
#' 11. Save complete workspace to RData file
#'
#' @section Random Sampling Strategy:
#' - **Sample size**: Weighted toward smaller N (N < 750 gets 3x probability)
#' - **DIF items**: Uniform over {2, 3, 4} representing 20%, 30%, 40% DIF
#' - **DIF type**: Uniform over {"a", "b", "ab"}
#' - **Parameter shifts**: Uniform from predefined grids
#' - **Abilities**: Normal with group-specific means and unit variance
#' - **Seeding**: Deterministic based on replication number for reproducibility
#'
#' @author Yale Quan
#' @date February 2026
#'
#' @references
#' Lord, F. M. (1980). Applications of item response theory to practical
#' testing problems. Lawrence Erlbaum Associates.
#'
#' Thissen, D., Steinberg, L., & Wainer, H. (1993). Detection of differential
#' item functioning using the parameters of item response models. In
#' P. W. Holland & H. Wainer (Eds.), Differential item functioning (pp. 67-113).
#' Lawrence Erlbaum Associates.
#'
#' @note
#' - This script generates TWO-GROUP TRAINING data (simplest scenario)
#' - For main analysis with 10 groups, see Ten_Group_Training_Data_Generation.R
#' - For validation with 3 groups, see Three_Group_Training_Data_Generation.R
#' - Output files used by Two_Group_Training_Data_Parameters.R for estimation
#' - Generates 600 replications for robust training
#' - Each dataset has random N, m, and DIF configuration
#' - Two groups create simplest scenario with only 1 pairwise comparison
#' - Classical two-group setting for baseline model training
#'
#' @examples
#' # This script is designed to be run as a standalone batch process:
#' # source("Two_Group_Training_Data_Generation.R")
#' 
#' # Output file naming convention:
#' # "Two_Group_Training_Data_Replication_{r}.RData"
#' # Example: "Two_Group_Training_Data_Replication_150.RData"
#' 
#' # To load a specific replication:
#' # load("Two_Group_Training_Data_Replication_1.RData")
#' # dim(y)  # Check response matrix dimensions
#' # table(g)  # Check group sizes
#' # sum(Labels_DIF)  # Count DIF items

# Setup ----
options(scipen = 9999)
rm(list = ls())

S <- 2  # Number of groups (rows)
J <- 10 # Number of items (columns)
dif.mu <- c(0, 1) # Group means (reference, focal)
sigma <- c(1, 1) # Group standard deviations
grid.a <- seq(-1, 1, by = 0.25) # Discrimination shift grid
grid.a <- grid.a[grid.a != 0] # Exclude zero (no DIF)
grid.b <- seq(-1.5, 1.5, by = 0.25) # Difficulty shift grid
grid.b <- grid.b[grid.b != 0] # Exclude zero (no DIF)
ratio <- c(.6, .4) # Group size ratio (60% reference, 40% focal)

# Sample Size Distribution ----
# Higher probability for N < 750
n_vals <- 100:1000
valid_n <- n_vals[n_vals %% 5 == 0] # Ensure integer group sizes with ratio
weights <- ifelse(valid_n < 750, 3, 1)  # 3x more likely if N < 750
weights <- weights / sum(weights)

# Generate Training Data ----
for (r in 1:600) { # Number of datasets to generate
  set.seed(202407)
  set.seed(sample.int(2147483647, 10000)[S * r])
  
  # Random sample size for training data ----
  N <- sample(valid_n, size = 1, prob = weights)
  
  # Random number of DIF items ----
  m <- sample(c(2, 3, 4), 1) # 20%, 30%, or 40% DIF
  
  # Group sizes ----
  g <- do.call(c, lapply(1:S, function(s) rep(s, N * ratio[s])))
  
  # Randomly select items to have DIF ----
  selected_items <- sample(1:J, m, replace = FALSE)
  
  # Randomly draw DIF shifts ----
  dif.a <- sample(grid.a, size = S, replace = TRUE)
  dif.a[1] <- 0 # Reference group has no shift
  dif.b <- sample(grid.b, size = S, replace = TRUE)
  dif.b[1] <- 0 # Reference group has no shift
  
  # Randomly select DIF type ----
  dif_types <- c("b", "ab", "a") # Difficulty, both, discrimination
  rep_dif <- sample(dif_types, 1)
  
  # Generate baseline item parameters ----
  a0 <- runif(J, 1.5, 2.5) # Discrimination
  b0 <- rnorm(J) # Difficulty
  a <- t(replicate(S, a0))
  b <- t(replicate(S, b0))
  
  # Apply DIF shifts to selected items ----
  if (rep_dif == "ab") {
    a[, selected_items] <- a[, selected_items] + dif.a
    b[, selected_items] <- b[, selected_items] + dif.b
  } else if (rep_dif == "b") {
    b[, selected_items] <- b[, selected_items] + dif.b
  } else if (rep_dif == "a") {
    a[, selected_items] <- a[, selected_items] + dif.a
  }
  
  # Calculate pairwise parameter differences ----
  d_a <- array(0, c(S, S, J))
  d_b <- array(0, c(S, S, J))
  for (k in 1:(S - 1)) {
    for (n in (k + 1):S) {
      d_a[k, n, ] <- a[k, ] - a[n, ]
      d_b[k, n, ] <- b[k, ] - b[n, ]
    }
  }
  
  # Create pairwise difference matrices ----
  temp_d_a <- combn(1:nrow(a), 2, function(idx) {
    diff <- a[idx[1], ] - a[idx[2], ]
    return(diff)
  })
  colnames(temp_d_a) <- "d.a"
  
  temp_d_b <- combn(1:nrow(a), 2, function(idx) {
    diff <- b[idx[1], ] - b[idx[2], ]
    return(diff)
  })
  colnames(temp_d_b) <- "d.b"
  
  # Create binary DIF labels ----
  Labels_a <- temp_d_a
  Labels_a[Labels_a != 0] <- 1
  colnames(Labels_a) <- "DIF_a"
  
  Labels_b <- temp_d_b
  Labels_b[Labels_b != 0] <- 1
  colnames(Labels_b) <- "DIF_b"
  
  Labels_DIF <- pmax(Labels_a, Labels_b)
  colnames(Labels_DIF) <- "DIF"
  
  # Generate observed responses ----
  y <- t(sapply(1:N, function(n) {
    s <- g[n]
    theta <- rnorm(1, dif.mu[s], sigma[s])
    rbinom(J, 1, plogis(a[s, ] * theta - b[s, ]))
  }))
  
  # Save workspace ----
  save.image(file = paste0("Two_Group_Training_Data_Replication_", r, ".RData"))
}
