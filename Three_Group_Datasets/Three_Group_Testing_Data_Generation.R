#' Generate Simulated Testing Data with Differential Item Functioning
#'
#' This script generates simulated item response data for testing the InterDIFNet
#' model with a three-group scenario. Each dataset contains binary responses from
#' 3 groups to 10 items under a 2-Parameter Logistic (2PL) IRT model with varying
#' levels of differential item functioning (DIF).
#'
#' @section Overview:
#' The script performs the following operations:
#' 1. Sets up population parameters for 3 distinct groups
#' 2. Randomly selects items to exhibit DIF
#' 3. Generates item parameters with DIF shifts
#' 4. Simulates ability parameters (theta) for each respondent
#' 5. Generates binary response data using the 2PL model
#' 6. Creates binary labels indicating DIF presence between group pairs
#' 7. Saves complete workspace for downstream analysis
#'
#' @section IRT Model Specification:
#' - **Model**: 2-Parameter Logistic (2PL)
#' - **Response Probability**: P(Y = 1 | theta) = plogis(a * theta - b)
#' - **Parameters**:
#'   - a: Discrimination parameter (1.5 to 2.5, uniform distribution)
#'   - b: Difficulty parameter (standard normal distribution)
#'   - theta: Ability parameter (varies by group)
#'
#' @section DIF Implementation:
#' Two types of DIF are randomly applied:
#' - **Uniform DIF ("b")**: Only difficulty parameters differ across groups
#' - **Non-uniform DIF ("ab")**: Both discrimination and difficulty differ
#'
#' DIF magnitude patterns across 3 groups:
#' - Group 1: No DIF (reference group, μ = 0)
#' - Group 2: Positive DIF (a: +1, b: +1.5, μ = 1)
#' - Group 3: Negative DIF (a: -1, b: -1.5, μ = -1)
#'
#' @section Group Structure:
#' - **Number of groups**: 3
#' - **Group size ratios**: Unequal (60%, 20%, 20%)
#' - **Ability distributions**: Groups differ in mean ability (μ: 0, 1, -1)
#' - **Variance**: 1 for all groups
#'
#' @section Experimental Conditions:
#' The script generates data across multiple conditions:
#' - **Sample sizes**: 250, 500, 1000 respondents per dataset
#' - **DIF percentages**: 20%, 40% (2 or 4 items with DIF out of 10)
#' - **Replications**: 125 independent datasets per condition
#' - **Total datasets**: 3 sample sizes × 2 DIF% × 125 replications = 750 files
#'
#' @section Random Seed Strategy:
#' - Base seed: 202407
#' - Replication-specific seeds derived from base seed
#' - Ensures reproducibility while maintaining independence across replications
#'
#' @section Output Files:
#' - Format: "Three_Group_Testing_Data_[N]_[perc]_Replication_[r].RData"
#' - Contains:
#'   - y: Binary response matrix (N × J)
#'   - g: Group membership vector (length N)
#'   - a, b: Item parameters by group (S × J matrices)
#'   - theta: Ability parameters (length N)
#'   - d_a, d_b: Pairwise parameter differences (S × S × J arrays)
#'   - temp_d_a, temp_d_b: Parameter difference matrices for all pairs
#'   - Labels_a, Labels_b, Labels_DIF: Binary DIF indicators
#'   - selected_items: Items with DIF
#'   - rep_dif: DIF type ("b" or "ab")
#'
#' @section Usage Notes:
#' - This is a testing data generation script (NOT training data)
#' - Generated data is used to evaluate model performance under varied conditions
#' - Sample sizes vary to assess model scalability
#' - DIF percentages vary to test sensitivity and specificity
#'
#' @author Yale Quan
#' @date 2024-07-01
#'
#' @note
#' - Processing time: ~10-20 seconds per condition
#' - Memory usage: Moderate (stores complete workspace per file)
#' - Total storage: ~1-2 GB for all 750 files
#' - DIF items are randomly selected for each replication
#'
#' @examples
#' # This script is designed to be run as a standalone file
#' # It will generate all testing conditions automatically
#' # source("Three_Group_Testing_Data_Generation.R")

# Setup and Configuration ----

# Clear workspace and set options
options(scipen = 9999)  # Disable scientific notation
rm(list = ls())

# Population Parameters ----

S <- 3   # Number of groups
J <- 10  # Number of items

# Group-specific ability distribution means (reference, high, low)
dif.mu <- c(0, 1, -1)

# Group-specific ability distribution standard deviations
sigma <- c(1, 1, 1)

# DIF magnitude parameters by group
dif.a <- c(0, 1, -1)    # Discrimination shifts
dif.b <- c(0, 1.5, -1.5)  # Difficulty shifts

# Group size ratios (proportions summing to 1)
ratio <- c(.6, .2, .2)  # Majority in reference group

# Main Data Generation Loop ----
# Iterate through all experimental conditions

for (N in c(250, 500, 1000)) {  # Sample sizes
  for (m in c(2, 4)) {  # Number of DIF items (20%, 40%)
    for (r in 1:125) {  # Replications per condition
      
      # Set Random Seed for Reproducibility ----
      set.seed(202407)
      set.seed(sample.int(2147483647, 10000)[S * r])
      
      # Generate Group Membership Vector ----
      # Assign respondents to groups based on specified ratios
      g <- do.call(c, lapply(1:S, function(s) rep(s, N * ratio[s])))
      
      # Select DIF Items ----
      # Randomly select m items to exhibit DIF
      selected_items <- sample(1:J, m, replace = FALSE)
      
      # Generate Base Item Parameters ----
      # Randomly select DIF type: uniform ("b") or non-uniform ("ab")
      dif_types <- c("b", "ab")
      rep_dif <- sample(dif_types, 1)
      
      # Generate baseline parameters (same for all groups initially)
      a0 <- runif(J, 1.5, 2.5)  # Discrimination: uniform(1.5, 2.5)
      b0 <- rnorm(J)  # Difficulty: standard normal
      
      # Replicate baseline parameters across all groups
      a <- t(replicate(S, a0))
      b <- t(replicate(S, b0))
      
      # Apply DIF Shifts ----
      # Add group-specific shifts to selected items based on DIF type
      if (rep_dif == "ab") {
        # Non-uniform DIF: shift both discrimination and difficulty
        a[, selected_items] <- a[, selected_items] + dif.a
        b[, selected_items] <- b[, selected_items] + dif.b
      } else if (rep_dif == "b") {
        # Uniform DIF: shift only difficulty parameters
        b[, selected_items] <- b[, selected_items] + dif.b
      }
      
      # Compute Pairwise Parameter Differences ----
      # Store differences between all group pairs for each item
      d_a <- array(0, c(S, S, J))
      d_b <- array(0, c(S, S, J))
      for (k in 1:(S - 1)) {
        for (n in (k + 1):S) {
          d_a[k, n, ] <- a[k, ] - a[n, ]
          d_b[k, n, ] <- b[k, ] - b[n, ]
        }
      }
      
      # Create Parameter Difference Matrices ----
      # Compute all pairwise group differences for downstream analysis
      
      # Discrimination parameter differences
      temp_d_a <- combn(1:nrow(a), 2, function(idx) {
        diff <- a[idx[1], ] - a[idx[2], ]
        return(diff)
      })
      colnames(temp_d_a) <- apply(
        combn(1:ncol(temp_d_a), 2), 2,
        function(idx) paste0("d.a_", "Group", idx[1], "Group", idx[2])
      )
      
      # Difficulty parameter differences
      temp_d_b <- combn(1:nrow(a), 2, function(idx) {
        diff <- b[idx[1], ] - b[idx[2], ]
        return(diff)
      })
      colnames(temp_d_b) <- apply(
        combn(1:ncol(temp_d_b), 2), 2,
        function(idx) paste0("d.b_", "Group", idx[1], "Group", idx[2])
      )
      
      # Create Binary DIF Labels ----
      # Convert continuous differences to binary indicators (0 = no DIF, 1 = DIF)
      
      # Label discrimination DIF
      Labels_a <- temp_d_a
      Labels_a[Labels_a != 0] <- 1
      colnames(Labels_a) <- apply(
        combn(1:ncol(Labels_a), 2), 2,
        function(idx) paste0("DIF_a_", "Group", idx[1], "Group", idx[2])
      )
      
      # Label difficulty DIF
      Labels_b <- temp_d_b
      Labels_b[Labels_b != 0] <- 1
      colnames(Labels_b) <- apply(
        combn(1:ncol(Labels_b), 2), 2,
        function(idx) paste0("DIF_b_", "Group", idx[1], "Group", idx[2])
      )
      
      # Combined DIF label (DIF in either discrimination or difficulty)
      Labels_DIF <- pmax(Labels_a, Labels_b)
      colnames(Labels_DIF) <- apply(
        combn(1:ncol(Labels_DIF), 2), 2,
        function(idx) paste0("DIF_", "Group", idx[1], "Group", idx[2])
      )
      
      # Generate Ability Parameters (Theta) ----
      # Sample from group-specific normal distributions
      theta <- sapply(1:N, function(n) {
        s <- g[n]
        rnorm(1, dif.mu[s], sigma[s])
      })
      
      # Generate Binary Response Data ----
      # Apply 2PL model: P(Y = 1) = plogis(a * theta - b)
      y <- t(sapply(1:N, function(n) {
        s <- g[n]
        rbinom(J, 1, plogis(a[s, ] * theta[n] - b[s, ]))
      }))
      
      # Save Complete Workspace ----
      # Save all variables for downstream parameter estimation and DIF testing
      save.image(
        file = paste0(
          "Three_Group_Testing_Data_", N, "_", m * 10,
          "_Replication_", r, ".RData"
        )
      )
    }
  }
}

# End of script

