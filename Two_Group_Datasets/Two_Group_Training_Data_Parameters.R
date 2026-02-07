#' Two-Group Training Data Parameter Estimation
#'
#' @description
#' This script estimates IRT parameters and DIF statistics for training datasets
#' in the two-group scenario of the InterDIFNet study. It processes simulated
#' training data with varying sample sizes and DIF configurations across
#' replications, generating features for training deep learning-based DIF
#' detection models in the simplest classical setting.
#'
#' @section Purpose:
#' Estimates parameters and DIF statistics for training data in the two-group
#' scenario.
#'
#' @section Parameter Estimation Method:
#' Uses VEMIRT package to utilize the truncated lasso penalty (TLP) for parameter estimation
#' - Penalty Parameters: Lambda (L1) and Tau (L2/Ridge)
#' - Lambda Grid: seq(0.1, 1.5, by=0.1) for sparsity control
#' - Tau Grid: c(Inf, seq(0.05, 0.5, by=0.05)) for shrinkage
#' - Output: Group-specific item parameters (a, b) with DIF detection
#' - Verbose: TRUE for monitoring convergence
#'
#' @section DIF Tests Implemented:
#' Five complementary DIF detection methods for 1 pairwise comparison:
#' 1. **Mantel-Haenszel (MH)**: Chi-square test for uniform DIF
#'    - Purification: Adaptive with fallback (purify=TRUE, then FALSE)
#'    - Adjustment: Benjamini-Hochberg FDR correction
#' 2. **Logistic Regression (LR)**: Three types tested
#'    - Uniform DIF: Group effect only (df=1)
#'    - Non-uniform DIF: Group × Total interaction (df=1)
#'    - Both: Combined test (df=2)
#' 3. **SIB (Simultaneous Item Bias)**: Uniform DIF test
#'    - Type: udif (uniform differential item functioning)
#' 4. **CSIB (Crossing SIB)**: Non-uniform DIF test
#'    - Type: nudif (non-uniform differential item functioning)
#' 5. **Standardized D-statistic**: Effect size measure
#'    - Scale: PDIF (probability difference metric)
#'
#' @section Output Features:
#' For each training file, the script generates:
#' - TLP Parameters: a and b estimates for 2 groups (4 features per item)
#' - Pairwise Differences: d.a and d.b for 1 comparison (2 features per item)
#' - MH Statistics: 1 pairwise comparison (1 feature per item)
#' - LR Statistics: 3 tests (uniform/non-uniform/both) (3 features per item)
#' - SIB Statistics: 1 pairwise comparison (1 feature per item)
#' - CSIB Statistics: 1 pairwise comparison (1 feature per item)
#' - D-statistics: 1 pairwise comparison (1 feature per item)
#'
#' @section Workflow:
#' 1. Loop through replications (default: 101-200, configurable)
#' 2. Check if file already estimated (skip if exists)
#' 3. Load simulated training data from RData file
#' 4. Generate pairwise group combination (single pair: 1 vs 2)
#' 5. Estimate TLP parameters with BIC selection
#' 6. Calculate pairwise parameter differences
#' 7. Run five DIF tests on the single pairwise comparison
#' 8. Combine all results into comprehensive feature set
#' 9. Save estimated parameters to RData file
#' 10. Error handling with skip-to-next on failures
#'
#' @author Yale Quan
#' @date February 2026
#'
#' @examples
#' # This script is designed to be run as a standalone batch process:
#' # source("Two_Group_Training_Data_Parameters.R")
#' 
#' # Output file naming convention:
#' # "Estimated_Training_data_Replication{r}.RData"
#' # Example: "Estimated_Training_data_Replication150.RData"
#' 
#' # Input file requirements:
#' # Must have corresponding files from Two_Group_Training_Data_Generation.R:
#' # "Two_Group_Training_Data_Replication_{r}.RData"

# Setup ----
remove(list = ls())
options(scipen = 9999)
library(tidyverse)
library(VEMIRT)
library(mirt)
library(difR)
library(DFIT)
library(dplyr)

mirtCluster(3) # Parallel processing for mirt

# Main Loop: Process All Training Replications ----
S <- 2  # Number of groups (reference, focal)
J <- 10 # Number of items

for (r in 101:200) { # Replication range (configurable)
  # Check if file has already been estimated ----
  filename <- paste0("Estimated_Training_data_Replication", r, ".RData")
  if (file.exists(filename)) {
    cat("\n")
    message("Replication ", r, " already exists, Skipping to next")
    next
  }
  message("Replication: ", r)
  
  # Load training data ----
  load(file = paste0("Two_Group_Training_Data_Replication_", r, ".RData"))
  
  # Generate pairwise group combination ----
  temp_g <- 1:S
  possible_pairs <- choose(S, 2) # 1 pairwise comparison
  pair_matrix <- matrix(data = NA, ncol = 2, nrow = possible_pairs)
  item_num <- seq(1:ncol(y))
  group_size <- as.numeric(table(g))
  
  # Populate pair_matrix (only one pair: 1 vs 2)
  counter <- 1
  for (i in 1:(S - 1)) {
    for (j in (i + 1):S) {
      pair_matrix[counter, ] <- c(temp_g[i], temp_g[j])
      counter <- counter + 1
    }
  }
  
  J <- ncol(y) # Number of items
  y <- as.data.frame(y)
  N <- nrow(y)
  
  skip_to_next <- FALSE
  tryCatch({
    # VEMIRT Parameter Estimation ----
    cat("\nVEMIRT Parameter Estimation")
    cat("\n")
    VEMIRT_df <- list()
    VEMIRT_df[[1]] <- as.data.frame(y)
    VEMIRT_df[[2]] <- g
    
    # Estimate 2PL parameters with penalized likelihood
    VEMIRT.m1 <- D2PL_pair_em(
      data = VEMIRT_df[[1]],
      group = VEMIRT_df[[2]],
      Lambda0 = seq(0.1, 1.5, by = 0.1),  # L1 penalty grid
      Tau = c(Inf, seq(0.05, 0.5, by = 0.05)), # L2 penalty grid
      verbose = TRUE
    )
    # Select best model based on BIC
    bic <- sapply(VEMIRT.m1$all, `[[`, 'BIC')
    temp <- VEMIRT.m1$all[[which.min(bic)]]
    VEMIRT_a <- temp$a
    VEMIRT_b <- temp$b
    
    # Pairwise Differences in discrimination (a) estimates ----
    VEMIRT_d.a <- combn(1:nrow(VEMIRT_a), 2, function(idx) {
      diff <- VEMIRT_a[idx[1], ] - VEMIRT_a[idx[2], ]
      return(diff)
    })
    colnames(VEMIRT_d.a) <- apply(combn(1:ncol(VEMIRT_d.a), 2), 2, 
                                  function(idx) paste0("d.a_", "Group", 
                                                       idx[1], "Group", idx[2]))
    
    # Pairwise Differences in difficulty (b) estimates ----
    VEMIRT_d.b <- combn(1:nrow(VEMIRT_b), 2, function(idx) {
      diff <- VEMIRT_b[idx[1], ] - VEMIRT_b[idx[2], ]
      return(diff)
    })
    colnames(VEMIRT_d.b) <- apply(combn(1:ncol(VEMIRT_d.b), 2), 2, 
                                  function(idx) paste0("d.b_", "Group", 
                                                       idx[1], "Group", idx[2]))
    
    # Reshape parameter estimates for storage ----
    VEMIRT_a <- t(VEMIRT_a)
    VEMIRT_b <- t(VEMIRT_b)
    colnames(VEMIRT_a) <- paste0("VEMIRT_a_Group", paste0(1:S))
    colnames(VEMIRT_b) <- paste0("VEMIRT_b_Group", paste0(1:S))
    
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
  
  # Mantel-Haenszel Pairwise Test ----
  cat("\nMantel-Haenszel pairwise test")
  MH_Results.L <- tibble()
  for (k in 1:possible_pairs) {
    pairwise_df <- as.data.frame(cbind(g, y)) %>%
      dplyr::filter(g %in% pair_matrix[k, ])
    pairwise_items <- dplyr::select(pairwise_df, -g)
    pairwise_groups <- as.character(pairwise_df$g)
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
    temp <- paste0("MH_Comparison_", pair_matrix[k, 1], "v", pair_matrix[k, 2])
    MH_stat <- fitMH$MH
    MH_df <- 1
    MH_Results.L <- rbind(
      MH_Results.L,
      cbind(item_num, MH_stat, MH_df, temp)
    )
  }
  Results.MH <- MH_Results.L %>%
    pivot_wider(names_from = temp, values_from = MH_stat) %>%
    select(-item_num)
  
  # Logistic Regression ----
  cat("\nLogistic Regression")
  LR_Results_Full.L <- tibble()
  for (k in 1:possible_pairs) {
    pairwise_df <- cbind(g, y) %>%
      dplyr::filter(g %in% pair_matrix[k, ])
    pairwise_items <- dplyr::select(pairwise_df, -g)
    pairwise_groups <- as.character(pairwise_df$g)
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
    temp <- paste0("LR_unifdif_Comparison_", pair_matrix[k, 1], "v",
                   pair_matrix[k, 2])
    LR_stat <- fitLR_uni$Logistik
    LR_df <- 1
    LR_Results_uniform.L <- cbind(item_num, LR_stat, LR_df, temp)
    
    temp <- paste0("LR_nonunifdif_Comparison_", pair_matrix[k, 1], "v",
                   pair_matrix[k, 2])
    LR_stat <- fitLR_nuni$Logistik
    LR_df <- 1
    LR_Results_nonuniform.L <- cbind(item_num, LR_stat, LR_df, temp)
    
    temp <- paste0("LR_both_dif_Comparison_", pair_matrix[k, 1], "v",
                   pair_matrix[k, 2])
    LR_stat <- fitLR_both$Logistik
    LR_df <- 2
    LR_Results_both.L <- cbind(item_num, LR_stat, LR_df, temp)
    
    LR_Results_Full.L <- rbind(
      LR_Results_Full.L,
      LR_Results_uniform.L,
      LR_Results_nonuniform.L,
      LR_Results_both.L
    )
  }
  Results.LR <- LR_Results_Full.L %>%
    pivot_wider(names_from = temp, values_from = c(LR_stat, LR_df)) %>%
    select(-item_num)
  
  # SIB Test ----
  cat("\nSIB Test")
  SIB_Results.L <- tibble()
  for (k in 1:possible_pairs) {
    pairwise_df <- cbind(g, y) %>%
      dplyr::filter(g %in% pair_matrix[k, ])
    pairwise_items <- dplyr::select(pairwise_df, -g)
    pairwise_groups <- as.character(pairwise_df$g)
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
    temp <- paste0("SIB_Comparison_", pair_matrix[k, 1], "v", pair_matrix[k, 2])
    SIB_stat <- fitSIB$Beta
    SIB_DF <- fitSIB$y
    SIB_Results.L <- rbind(
      SIB_Results.L,
      cbind(item_num, SIB_stat, SIB_DF, temp)
    )
  }
  Results.SIB <- SIB_Results.L %>%
    pivot_wider(names_from = temp, values_from = SIB_stat) %>%
    select(-item_num)
  
  # Crossing SIB Test ----
  cat("\nCSIB Test")
  CSIB_Results.L <- tibble()
  for (k in 1:possible_pairs) {
    pairwise_df <- cbind(g, y) %>%
      dplyr::filter(g %in% pair_matrix[k, ])
    pairwise_items <- dplyr::select(pairwise_df, -g)
    pairwise_groups <- as.character(pairwise_df$g)
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
    temp <- paste0("CSIB_Comparison_", pair_matrix[k, 1], "v", pair_matrix[k, 2])
    CSIB_stat <- fitCSIB$Beta
    CSIB_DF <- fitCSIB$y
    CSIB_Results.L <- rbind(
      CSIB_Results.L,
      cbind(item_num, CSIB_stat, CSIB_DF, temp)
    )
  }
  Results.CSIB <- CSIB_Results.L %>%
    pivot_wider(names_from = temp, values_from = c(CSIB_stat, CSIB_DF)) %>%
    select(-item_num)
  
  # Standardized D-stat ----
  cat("\nStandardized D-stat")
  D_stat.L <- tibble()
  for (k in 1:possible_pairs) {
    pairwise_df <- as.data.frame(cbind(g, y)) %>%
      dplyr::filter(g %in% pair_matrix[k, ])
    pairwise_items <- dplyr::select(pairwise_df, -g)
    pairwise_groups <- as.character(pairwise_df$g)
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
    temp <- paste0("D_Stat_Comparison_",
                   pair_matrix[k, 1], "v", pair_matrix[k, 2])
    D_stat <- fitD$PDIF
    D_stat.L <- rbind(
      D_stat.L,
      cbind(item_num, D_stat, temp)
    )
  }
  
  Results.D <- D_stat.L %>%
    pivot_wider(names_from = temp, values_from = D_stat) %>%
    select(-item_num)
  
  # Save Results ----
  save.image(file = filename)
  
  cat("\n")
}


