#' Two-Group Testing Data Generation
#'
#' @description
#' This script generates simulated testing datasets for the two-group scenario
#' of the InterDIFNet study. It creates systematic replications with fixed
#' conditions (sample size, DIF percentage) to evaluate deep learning models
#' for DIF detection. The two-group design represents the simplest and most
#' traditional setting for DIF analysis.
#'
#' @section Testing Data Design:
#' Creates 1,500 testing replications with systematic conditions:
#' - Sample Sizes: N = 125, 250, 500 (3 levels)
#' - DIF Percentages: 20% (2/10 items), 40% (4/10 items) (2 levels)
#' - Replications: 250 per condition
#' - Groups: 2 groups with 60%/40% ratio (reference/focal)
#' - Pairwise Comparisons: 1 (reference vs focal)
#' - Total Files: 3 N × 2 DIF% × 250 reps = 1,500 testing files
#'
#' @section Two-Group Structure:
#' - Group 1 (Reference): 60% of sample, μ=0, no DIF shifts (baseline)
#' - Group 2 (Focal): 40% of sample, μ=+1, positive DIF shifts
#' - Simplest Structure: Only one comparison (traditional DIF setting)
#' - Classical Context: Most DIF methods designed for this case
#' - Easier to Interpret: Clear reference vs focal comparison
#' 
#' @section DIF Implementation:
#' Two types of DIF are randomly applied (simplified from 3-group/10-group):
#' 1. **b-DIF (Difficulty only)**: Uniform DIF via intercept shifts
#'    - Reference group: shift = 0
#'    - Focal group: shift = +1.5
#' 2. **ab-DIF (Both)**: Combined discrimination and difficulty shifts
#'    - Reference group: shifts = (0, 0)
#'    - Focal group: shifts = (+1, +1.5) for (a, b)
#'
#' @section Population Parameters:
#' - Groups: S = 2 (reference, focal)
#' - Items: J = 10
#' - Group Means: μ = (0, +1)
#' - Group SDs: σ = (1, 1)
#' - Base Discrimination: a₀ ~ Uniform(1.5, 2.5)
#' - Base Difficulty: b₀ ~ N(0, 1)
#' - Group Ratio: 60% : 40% (reference : focal)
#' - DIF Shifts: a = (0, +1), b = (0, +1.5)
#'
#' @section Testing Conditions:
#' Fixed experimental design for systematic evaluation:
#' - **Sample Sizes**: 125, 250, 500
#'   - Smaller than 10-group (1000-4000) and 3-group (250-1000)
#'   - Typical for traditional two-group DIF studies
#' - **DIF Percentages**: 20% (2 items), 40% (4 items)
#'   - Same percentages as other scenarios
#'   - 2 or 4 items out of 10 with DIF
#' - **DIF Types**: b-DIF or ab-DIF (randomly selected)
#'   - Simpler than training data (which includes a-DIF)
#'
#' @section Output Variables:
#' Each RData file contains:
#' - `y`: N × J response matrix (binary item responses)
#' - `g`: N-vector of group membership (1 or 2)
#' - `a`: S × J matrix of discrimination parameters
#' - `b`: S × J matrix of difficulty parameters
#' - `d_a`: S × S × J array of pairwise discrimination differences
#' - `d_b`: S × S × J array of pairwise difficulty differences
#' - `Labels_a`: Binary labels for a-DIF (single comparison)
#' - `Labels_b`: Binary labels for b-DIF (single comparison)
#' - `Labels_DIF`: Binary labels for any DIF (single comparison)
#' - `N`, `m`, `selected_items`, `rep_dif`: Simulation metadata
#'
#' @section Workflow:
#' 1. Set fixed population parameters (S, J, μ, σ, DIF shifts, ratios)
#' 2. Triple nested loop: N sizes × DIF percentages × Replications
#'    a. Set random seed for reproducibility
#'    b. Assign group membership based on 60%/40% ratio
#'    c. Select m items to have DIF (fixed per condition)
#'    d. Randomly select DIF type (b-DIF or ab-DIF)
#'    e. Generate base item parameters (a₀, b₀)
#'    f. Apply DIF shifts to selected items
#'    g. Calculate pairwise parameter differences (single comparison)
#'    h. Create binary DIF labels
#'    i. Generate theta values by group
#'    j. Generate binary responses using 2PL model
#'    k. Save complete workspace to RData file
#'
#' @section Label Generation:
#' Binary labels indicate presence of DIF for the single pairwise comparison:
#' - `Labels_a`: 1 if discrimination differs between groups, 0 otherwise
#' - `Labels_b`: 1 if difficulty differs between groups, 0 otherwise
#' - `Labels_DIF`: 1 if either a or b differs (combined DIF indicator)
#' - Dimensionality: 1 comparison × 10 items = 10 binary labels per type
#' - Simplest label structure across all scenarios
#'
#' @author Yale Quan
#' @date February 2026
#'
#' @examples
#' # This script is designed to be run as a standalone batch process:
#' # source("Two_Group_Testing_Data_Generation.R")
#' 
#' # Output file naming convention:
#' # "Two_Group_Testing_Data_{N}_{DIF_perc}_Replication_{r}.RData"
#' # Example: "Two_Group_Testing_Data_250_20_Replication_100.RData"
#' 
#' # To load a specific replication:
#' # load("Two_Group_Testing_Data_250_20_Replication_100.RData")

# Setup ----
options(scipen = 9999)
rm(list = ls())

# Population Parameters ----
S <- 2  # Number of groups (reference, focal)
J <- 10 # Number of items
dif.mu <- c(0, 1)    # Group-specific ability means
sigma <- c(1, 1)     # Group-specific ability SDs
dif.a <- c(0, 1)     # Discrimination shifts (reference, focal)
dif.b <- c(0, 1.5)   # Difficulty shifts (reference, focal)
ratio <- c(.6, .4)   # Group distribution: 60%/40%

# Main Loop: Generate Testing Data ----
for (N in c(125, 250, 500)) { # Sample sizes (smaller than 3/10-group)
  for (m in c(2, 4)) { # Number of DIF items (20%, 40%)
    for (r in 1:500) { # 500 replications per condition
      
      # Random Seed for Reproducibility ----
      set.seed(202407)
      set.seed(sample.int(2147483647, 10000)[S * r])
      
      # Group Membership Assignment ----
      g <- do.call(c, lapply(1:S, function(s) rep(s, N * ratio[s])))
      
      # Select DIF Items ----
      selected_items <- sample(1:J, m, replace = FALSE)
      
      # Select DIF Type ----
      dif_types <- c("b", "ab")  # b-DIF or both (no pure a-DIF)
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
      
      # Create Pairwise Difference Vectors ----
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
      
      # Create Binary Labels for a-DIF ----
      Labels_a <- temp_d_a
      Labels_a[Labels_a != 0] <- 1
      colnames(Labels_a) <- "DIF_a"
      
      # Create Binary Labels for b-DIF ----
      Labels_b <- temp_d_b
      Labels_b[Labels_b != 0] <- 1
      colnames(Labels_b) <- "DIF_b"
      
      # Create Combined DIF Labels ----
      Labels_DIF <- pmax(Labels_a, Labels_b)
      colnames(Labels_DIF) <- "DIF"
      
      # Generate Latent Abilities and Responses ----
      y <- t(sapply(1:N, function(n) {
        s <- g[n]
        theta <- rnorm(1, dif.mu[s], sigma[s])
        rbinom(J, 1, plogis(a[s, ] * theta - b[s, ]))
      }))
      
      # Save Workspace ----
      save.image(file = paste0("Two_Group_Testing_Data_", N, "_", m * 10, 
                              "_Replication_", r, ".RData"))
    }
  }
}

