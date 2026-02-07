#' Estimate IRT Parameters and DIF Statistics for Training Data
#'
#' This script processes simulated training data to estimate item response theory
#' (IRT) parameters and compute various differential item functioning (DIF)
#' detection statistics. These estimates serve as input features for the
#' InterDIFNet deep learning model.
#'
#' @section Overview:
#' The script performs the following operations for each training dataset:
#' 1. Loads simulated response data from RData files
#' 2. Uses VEMIRT package to utilize the truncated lasso penalty (TLP) for parameter estimation
#' 3. Computes pairwise DIF statistics across all group combinations
#' 4. Saves estimated parameters and test statistics for model training
#'
#' @section IRT Parameter Estimation:
#' - Uses VEMIRT package to utilize the truncated lasso penalty (TLP) for parameter estimation
#' - Model: 2-Parameter Logistic (2PL)
#' - Tuning: BIC-based selection of penalty parameters (Lambda, Tau)
#' - Output: Discrimination (a) and difficulty (b) parameters per group
#'
#' @section DIF Detection Methods:
#' The following pairwise DIF tests are computed:
#' - **Mantel-Haenszel (MH)**: Non-parametric test for uniform DIF
#' - **Logistic Regression (LR)**: Tests for uniform, non-uniform, and combined DIF
#' - **SIB Test**: Standardized item bias for uniform DIF
#' - **Crossing-SIB (CSIB)**: Tests for non-uniform DIF
#' - **Standardized D-statistic**: Effect size measure for DIF magnitude
#'
#' @section Computational Details:
#' - Parallel processing: Uses 3 cores via mirtCluster()
#' - Convergence tolerance: 1e-2 for computational efficiency
#' - Error handling: Skips failed replications and continues processing
#' - Purification: Applied with fallback when convergence issues arise
#'
#' @section Input Files:
#' - Format: "Ten_Group_Training_Data_Replication_[r].RData"
#' - Contains: Response matrix (y), group vector (g), true parameters (a, b)
#'
#' @section Output Files:
#' - Format: "Estimated_Training_data_Replication[r].RData"
#' - Contains: All estimated parameters, test statistics, and original data
#' - Variables:
#'   - Results.VEMIRT: Parameter estimates and pairwise differences
#'   - Results.MH: Mantel-Haenszel statistics
#'   - Results.LR: Logistic regression statistics
#'   - Results.SIB, Results.CSIB: SIB test statistics
#'   - Results.D: Standardized D-statistics
#'
#' @section Processing Range:
#' - Current: Replications 350-400 (modify loop range as needed)
#' - Total available: 800 training datasets
#' - Skip logic: Automatically skips already-processed files
#'
#' @author Yale Quan
#' @date 2024-07-01
#'
#' @note
#' - Processing time: Approximately 2-5 minutes per replication
#' - Memory usage: Moderate (dependent on sample size)
#' - Dependencies: tidyverse, VEMIRT, mirt, difR, DFIT, dplyr
#'
#' @examples
#' # This script is designed to be run as a standalone file
#' # Modify the loop range (350:400) to process different replications
#' # source("Ten_Group_Training_Data_Parameters.R")

# Setup and Configuration ----

# Clear workspace and set options
remove(list = ls())
options(scipen = 9999)  # Disable scientific notation

# Load required packages
library(tidyverse)  # Data manipulation and visualization
library(VEMIRT)     # Penalized IRT parameter estimation
library(mirt)       # Multidimensional IRT modeling
library(difR)       # DIF detection methods
library(DFIT)       # Differential functioning of items and tests
library(dplyr)      # Data manipulation

# Configure parallel processing for mirt functions
mirtCluster(3)

# Data Processing Loop ----

S <- 10  # Number of groups
J <- 10  # Number of items

for (r in 350:400) {
  # Check if file has already been estimated
  filename <- paste0("Estimated_Training_data_Replication", r, ".RData")
  
  if (file.exists(filename)) {
    cat("\n")
    message("Replication ", r, " already exists, skipping to next")
    next
  }
  
  message("Processing Replication: ", r)
  
  # Load simulated training data
  load(file = paste0("Ten_Group_Training_Data_Replication_", r, ".RData"))
  
  # Setup Pairwise Group Comparisons ----
  
  # Generate all possible pairwise group combinations
  temp_g <- 1:S
  possible_pairs <- choose(S, 2)  # Number of pairwise combinations
  pair_matrix <- matrix(data = NA, ncol = 2, nrow = possible_pairs)
  item_num <- seq(1:ncol(y))
  group_size <- as.numeric(table(g))
  
  # Populate pair matrix with all group combinations
  counter <- 1
  for (i in 1:(S - 1)) {
    for (j in (i + 1):S) {
      pair_matrix[counter, ] <- c(temp_g[i], temp_g[j])
      counter <- counter + 1
    }
  }
  
  # Prepare data for analysis
  J <- ncol(y)  # Number of items
  y <- as.data.frame(y)
  N <- nrow(y)  # Sample size
  
  # Error Handling Setup ----
  skip_to_next <- FALSE
  tryCatch({
    # VEMIRT Parameter Estimation ----
    
    cat("\nEstimating IRT Parameters with VEMIRT (2PL Model)")
    
    # Prepare data for VEMIRT
    VEMIRT_df <- list()
    VEMIRT_df[[1]] <- as.data.frame(y)
    VEMIRT_df[[2]] <- g
    
    # Fit 2PL model with multiple penalty parameters
    # Lambda0: Controls discrimination parameter regularization
    # Tau: Controls difficulty parameter regularization
    VEMIRT.m1 <- D2PL_pair_em(
      data = VEMIRT_df[[1]],
      group = VEMIRT_df[[2]],
      Lambda0 = seq(0.1, 1.5, by = 0.1),
      Tau = c(Inf, seq(0.05, 0.5, by = 0.05)),
      verbose = FALSE
    )
    
    # Select best model based on BIC
    bic <- sapply(VEMIRT.m1$all, `[[`, "BIC")
    temp <- VEMIRT.m1$all[[which.min(bic)]]
    VEMIRT_a <- temp$a
    VEMIRT_b <- temp$b
    
    # Compute pairwise differences in discrimination parameters
    VEMIRT_d.a <- combn(1:nrow(VEMIRT_a), 2, function(idx) {
      diff <- VEMIRT_a[idx[1], ] - VEMIRT_a[idx[2], ]
      return(diff)
    })
    colnames(VEMIRT_d.a) <- apply(
      combn(1:ncol(VEMIRT_a), 2), 2,
      function(idx) paste0("d.a_", "Group", idx[1], "Group", idx[2])
    )
    
    # Compute pairwise differences in difficulty parameters
    VEMIRT_d.b <- combn(1:nrow(VEMIRT_b), 2, function(idx) {
      diff <- VEMIRT_b[idx[1], ] - VEMIRT_b[idx[2], ]
      return(diff)
    })
    colnames(VEMIRT_d.b) <- apply(
      combn(1:ncol(VEMIRT_b), 2), 2,
      function(idx) paste0("d.b_", "Group", idx[1], "Group", idx[2])
    )
    
    # Reshape parameter estimates for storage
    VEMIRT_a <- t(VEMIRT_a)
    VEMIRT_b <- t(VEMIRT_b)
    colnames(VEMIRT_a) <- paste0("VEMIRT_a_Group", 1:S)
    colnames(VEMIRT_b) <- paste0("VEMIRT_b_Group", 1:S)
    
    # Combine all VEMIRT results
    Results.VEMIRT <- cbind(VEMIRT_a, VEMIRT_b, VEMIRT_d.a, VEMIRT_d.b)
  },
  error = function(e) {
    cat("\n")
    message("Error: ", e)
    skip_to_next <<- TRUE
  })
  
  if (skip_to_next) {
    next
  }
    
  # Mantel-Haenszel Test ----
  
  cat("\nMantel-Haenszel Pairwise Test")
  MH_Results.L <- tibble()
  
  for (k in 1:possible_pairs) {
    # Extract pairwise data
    pairwise_df <- as.data.frame(cbind(g, y)) %>%
      dplyr::filter(g %in% pair_matrix[k, ])
    pairwise_items <- dplyr::select(pairwise_df, -g)
    pairwise_groups <- as.character(pairwise_df$g)
    
    # Fit Mantel-Haenszel test with purification
    fitMH <- tryCatch({
      difMH(
        pairwise_items,
        group = pairwise_groups,
        focal.name = unique(pairwise_groups)[1],
        purify = TRUE,
        p.adjust.method = "BH"
      )
    }, error = function(e) {
      difMH(
        pairwise_items,
        group = pairwise_groups,
        focal.name = unique(pairwise_groups)[1],
        purify = FALSE,
        p.adjust.method = "BH"
      )
    })
    
    # Store results
    temp <- paste0("MH_Comparison_", pair_matrix[k, 1], "v", pair_matrix[k, 2])
    MH_stat <- fitMH$MH
    MH_df <- 1
    MH_Results.L <- rbind(
      MH_Results.L,
      cbind(item_num, MH_stat, MH_df, temp)
    )
  }
  
  # Reshape to wide format
  Results.MH <- MH_Results.L %>%
    pivot_wider(names_from = temp, values_from = MH_stat) %>%
    select(-item_num)
    
  # Logistic Regression Test ----
  
  cat("\nLogistic Regression Pairwise Test")
  LR_Results_Full.L <- tibble()
  
  for (k in 1:possible_pairs) {
    # Extract pairwise data
    pairwise_df <- cbind(g, y) %>%
      dplyr::filter(g %in% pair_matrix[k, ])
    pairwise_items <- dplyr::select(pairwise_df, -g)
    pairwise_groups <- as.character(pairwise_df$g)
    
    # Test for uniform DIF
    fitLR_uni <- tryCatch({
      difLogistic(
        pairwise_items,
        group = pairwise_groups,
        focal.name = unique(pairwise_groups)[1],
        type = "udif",
        purify = FALSE,
        p.adjust.method = "BH"
      )
    }, error = function(e) {
      difLogistic(
        pairwise_items,
        group = pairwise_groups,
        focal.name = unique(pairwise_groups)[1],
        type = "udif",
        purify = TRUE,
        p.adjust.method = "BH"
      )
    })
    
    # Test for non-uniform DIF
    fitLR_nuni <- tryCatch({
      difLogistic(
        pairwise_items,
        group = pairwise_groups,
        focal.name = unique(pairwise_groups)[1],
        type = "nudif",
        purify = FALSE,
        p.adjust.method = "BH"
      )
    }, error = function(e) {
      difLogistic(
        pairwise_items,
        group = pairwise_groups,
        focal.name = unique(pairwise_groups)[1],
        type = "nudif",
        purify = TRUE,
        p.adjust.method = "BH"
      )
    })
    
    # Test for both types of DIF
    fitLR_both <- tryCatch({
      difLogistic(
        pairwise_items,
        group = pairwise_groups,
        focal.name = unique(pairwise_groups)[1],
        type = "both",
        purify = FALSE,
        p.adjust.method = "BH"
      )
    }, error = function(e) {
      difLogistic(
        pairwise_items,
        group = pairwise_groups,
        focal.name = unique(pairwise_groups)[1],
        type = "both",
        purify = TRUE,
        p.adjust.method = "BH"
      )
    })
    
    # Store uniform DIF results
    temp <- paste0("LR_unifdif_Comparison_", pair_matrix[k, 1], "v",
                   pair_matrix[k, 2])
    LR_stat <- fitLR_uni$Logistik
    LR_df <- 1
    LR_Results_uniform.L <- cbind(item_num, LR_stat, LR_df, temp)
    
    # Store non-uniform DIF results
    temp <- paste0("LR_nonunifdif_Comparison_", pair_matrix[k, 1], "v",
                   pair_matrix[k, 2])
    LR_stat <- fitLR_nuni$Logistik
    LR_df <- 1
    LR_Results_nonuniform.L <- cbind(item_num, LR_stat, LR_df, temp)
    
    # Store both types DIF results
    temp <- paste0("LR_both_dif_Comparison_", pair_matrix[k, 1], "v",
                   pair_matrix[k, 2])
    LR_stat <- fitLR_both$Logistik
    LR_df <- 2
    LR_Results_both.L <- cbind(item_num, LR_stat, LR_df, temp)
    
    # Combine all results
    LR_Results_Full.L <- rbind(
      LR_Results_Full.L,
      LR_Results_uniform.L,
      LR_Results_nonuniform.L,
      LR_Results_both.L
    )
  }
  
  # Reshape to wide format
  Results.LR <- LR_Results_Full.L %>%
    pivot_wider(names_from = temp, values_from = c(LR_stat, LR_df)) %>%
    select(-item_num)
    
  # SIB Test (Standardized Item Bias) ----
  
  cat("\nSIB Test (Uniform DIF)")
  SIB_Results.L <- tibble()
  
  for (k in 1:possible_pairs) {
    # Extract pairwise data
    pairwise_df <- cbind(g, y) %>%
      dplyr::filter(g %in% pair_matrix[k, ])
    pairwise_items <- dplyr::select(pairwise_df, -g)
    pairwise_groups <- as.character(pairwise_df$g)
    
    # Fit SIB test for uniform DIF
    fitSIB <- tryCatch({
      difSIBTEST(
        pairwise_items,
        group = pairwise_groups,
        purify = TRUE,
        focal.name = unique(pairwise_groups)[1],
        type = "udif"
      )
    }, error = function(e) {
      difSIBTEST(
        pairwise_items,
        group = pairwise_groups,
        purify = FALSE,
        focal.name = unique(pairwise_groups)[1],
        type = "udif"
      )
    })
    
    # Store results
    temp <- paste0("SIB_Comparison_", pair_matrix[k, 1], "v", pair_matrix[k, 2])
    SIB_stat <- fitSIB$Beta
    SIB_DF <- fitSIB$y
    SIB_Results.L <- rbind(
      SIB_Results.L,
      cbind(item_num, SIB_stat, SIB_DF, temp)
    )
  }
  
  # Reshape to wide format
  Results.SIB <- SIB_Results.L %>%
    pivot_wider(names_from = temp, values_from = SIB_stat) %>%
    select(-item_num)
    
  # Crossing-SIB Test (Non-uniform DIF) ----
  
  cat("\nCSIB Test (Non-uniform DIF)")
  CSIB_Results.L <- tibble()
  
  for (k in 1:possible_pairs) {
    # Extract pairwise data
    pairwise_df <- cbind(g, y) %>%
      dplyr::filter(g %in% pair_matrix[k, ])
    pairwise_items <- dplyr::select(pairwise_df, -g)
    pairwise_groups <- as.character(pairwise_df$g)
    
    # Fit Crossing-SIB test for non-uniform DIF
    fitCSIB <- tryCatch({
      difSIBTEST(
        pairwise_items,
        group = pairwise_groups,
        purify = TRUE,
        focal.name = unique(pairwise_groups)[1],
        type = "nudif"
      )
    }, error = function(e) {
      difSIBTEST(
        pairwise_items,
        group = pairwise_groups,
        purify = FALSE,
        focal.name = unique(pairwise_groups)[1],
        type = "nudif"
      )
    })
    
    # Store results
    temp <- paste0("CSIB_Comparison_", pair_matrix[k, 1], "v", pair_matrix[k, 2])
    CSIB_stat <- fitCSIB$Beta
    CSIB_DF <- fitCSIB$y
    CSIB_Results.L <- rbind(
      CSIB_Results.L,
      cbind(item_num, CSIB_stat, CSIB_DF, temp)
    )
  }
  
  # Reshape to wide format
  Results.CSIB <- CSIB_Results.L %>%
    pivot_wider(names_from = temp, values_from = c(CSIB_stat, CSIB_DF)) %>%
    select(-item_num)
    
  # Standardized D-Statistic ----
  
  cat("\nStandardized D-Statistic Test")
  D_stat.L <- tibble()
  
  for (k in 1:possible_pairs) {
    # Extract pairwise data
    pairwise_df <- as.data.frame(cbind(g, y)) %>%
      dplyr::filter(g %in% pair_matrix[k, ])
    pairwise_items <- dplyr::select(pairwise_df, -g)
    pairwise_groups <- as.character(pairwise_df$g)
    
    # Compute standardized D-statistic (effect size)
    fitD <- tryCatch({
      difR::difStd(
        Data = pairwise_items,
        group = pairwise_groups,
        purify = TRUE,
        focal.name = unique(pairwise_groups)[1]
      )
    }, error = function(e) {
      difR::difStd(
        Data = pairwise_items,
        group = pairwise_groups,
        purify = FALSE,
        focal.name = unique(pairwise_groups)[1]
      )
    })
    
    # Store results
    temp <- paste0("D_Stat_Comparison_",
                   pair_matrix[k, 1], "v", pair_matrix[k, 2])
    D_stat <- fitD$PDIF
    D_stat.L <- rbind(
      D_stat.L,
      cbind(item_num, D_stat, temp)
    )
  }
  
  # Reshape to wide format
  Results.D <- D_stat.L %>%
    pivot_wider(names_from = temp, values_from = D_stat) %>%
    select(-item_num)
  
  # Save Workspace ----
  
  # Save all results to RData file for model training
  save.image(file = filename)
  
  cat("\n")
}

# End of script


