#' Three-Group Training Data Parameter Estimation
#'
#' @description
#' This script estimates IRT parameters and DIF statistics for training datasets
#' in the three-group scenario of the InterDIFNet study. It processes 100
#' simulated training replications with randomized conditions (sample size, DIF
#' percentage, DIF type) to generate features for training deep learning models
#' for DIF detection.
#'
#' @section Data Processing:
#' The script processes 100 training files (replications 1-500, subset exists)
#' with randomized experimental conditions:
#' - Sample Sizes: Weighted random (250-2000, favoring N < 1000)
#' - DIF Items: Random selection of 2, 3, or 4 items (20%, 30%, 40%)
#' - DIF Type: Random selection of a-DIF, b-DIF, or both
#' - Groups: 3 groups with 60%/20%/20% distribution
#' - Pairwise Comparisons: 3 (Group1vs2, Group1vs3, Group2vs3)
#' - Total Training Files: 100 replications (vs 51 in 10-group scenario)
#'
#' @section Parameter Estimation Method:
#' Uses VEMIRT to perform the Truncated Lasso Penalty (TLP) estimation of 2PL parameters:
#' - Penalty Parameters: Lambda (L1) and Tau (L2/Ridge)
#' - Lambda Grid: seq(0.1, 1.5, by=0.1) for sparsity control
#' - Tau Grid: c(Inf, seq(0.05, 0.5, by=0.05)) for shrinkage
#' - Selection: BIC-based optimal penalty selection
#' - Output: Group-specific item parameters (a, b) with DIF detection
#' - Verbose: TRUE for monitoring convergence (training phase)
#'
#' @section DIF Tests Implemented:
#' Five complementary DIF detection methods for 3 pairwise comparisons:
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
#' - VEMIRT Parameters: a and b estimates for 3 groups (6 features per item)
#' - Pairwise Differences: d.a and d.b for 3 comparisons (6 features per item)
#' - MH Statistics: 3 pairwise comparisons (3 features per item)
#' - LR Statistics: 9 tests (3 types × 3 comparisons) (9 features per item)
#' - SIB Statistics: 3 pairwise comparisons (3 features per item)
#' - CSIB Statistics: 3 pairwise comparisons (3 features per item)
#' - D-statistics: 3 pairwise comparisons (3 features per item)
#' Total: 36 features per item for 10 items = 360-dimensional feature vector
#'
#' @section Workflow:
#' 1. Loop through replications (r = 1 to 500, processing existing files)
#' 2. Check if file already estimated (skip if exists)
#' 3. Load simulated training data from RData file
#' 4. Generate all pairwise group combinations (3 pairs)
#' 5. Estimate VEMIRT parameters with BIC selection
#' 6. Calculate pairwise parameter differences
#' 7. Run five DIF tests on each pairwise comparison
#' 8. Combine all results into comprehensive feature set
#' 9. Save estimated parameters to RData file
#' 10. Error handling with skip-to-next on failures
#'
#' @author Yale Quan
#' @date Updated - February 2026
#'
#' @examples
#' # This script is designed to be run as a standalone batch process:
#' # source("Three_Group_Training_Data_Parameters.R")
#' 
#' # Output file naming convention:
#' # "Estimated_Training_data_Replication{r}.RData"
#' # Example: "Estimated_Training_data_Replication50.RData"
#' 
#' # Input file requirements:
#' # Must have corresponding files from Three_Group_Training_Data_Generation.R:
#' # "Three_Group_Training_Data_Replication_{r}.RData"

# Setup ----
remove(list = ls())
options(scipen = 9999)
library(tidyverse)
library(VEMIRT)
library(mirt)
library(difR)
library(DFIT)
library(dplyr)

# Main Loop: Process All Training Replications ----
S <- 3  # Number of groups (reference, positive, negative)
J <- 10 # Number of items

for (r in 1:500) {
  
  # Check if file has already been estimated ----
  filename <- paste0("Estimated_Training_data_Replication", r, ".RData")
  if (file.exists(filename)) {
    cat("\n")
    message("Replication ", r, " already exists, Skipping to next")
    next
  }
  
  message("3-Group Replication: ", r)
  load(file = paste0("Three_Group_Training_Data_Replication_", r, ".RData"))
  
  # Generate all pairwise group combinations ----
  temp_g <- 1:S
  possible_pairs <- choose(S, 2) # 3 pairwise comparisons
  pair_matrix <- matrix(data = NA, ncol = 2, nrow = possible_pairs)
  item_num <- seq(1:ncol(y))
  group_size <- as.numeric(table(g))
  
  # Populate pair_matrix with all combinations
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
    cat("VEMIRT Parameter Estimation")
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
      verbose = TRUE # Verbose output for training phase
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
    pairwise_df <- as.data.frame(cbind(g, y)) |>
      dplyr::filter(g %in% pair_matrix[k, ])
    pairwise_items <- dplyr::select(pairwise_df, -g)
    pairwise_groups <- as.character(pairwise_df$g)
    
    # Mantel-Haenszel test with purification
    fitMH <- tryCatch({
      difMH(pairwise_items, 
            group = pairwise_groups, 
            focal.name = unique(pairwise_groups)[1],
            purify = TRUE,
            p.adjust.method = "BH")
    }, error = function(e) {
      difMH(pairwise_items, 
            group = pairwise_groups, 
            focal.name = unique(pairwise_groups)[1],
            purify = FALSE,
            p.adjust.method = "BH")
    })
    
    temp <- paste0("MH_Comparison_", pair_matrix[k, 1], "v", pair_matrix[k, 2])
    MH_stat <- fitMH$MH
    MH_df <- 1
    MH_Results.L <- rbind(MH_Results.L,
                          cbind(item_num, MH_stat, MH_df, temp))
  }
  Results.MH <- MH_Results.L |>
    pivot_wider(names_from = temp, values_from = MH_stat) |>
    select(-item_num)
  
  # Logistic Regression ----
  cat("\nLogistic Regression")
  LR_Results_Full.L <- tibble()
  for (k in 1:possible_pairs) {
    pairwise_df <- cbind(g, y) |>
      dplyr::filter(g %in% pair_matrix[k, ])
    pairwise_items <- dplyr::select(pairwise_df, -g)
    pairwise_groups <- as.character(pairwise_df$g)
    
    # Uniform DIF test
    fitLR_uni <- tryCatch({
      difLogistic(pairwise_items, 
                  group = pairwise_groups, 
                  focal.name = unique(pairwise_groups)[1], 
                  type = "udif", 
                  purify = FALSE,
                  p.adjust.method = "BH")
    }, error = function(e) {
      difLogistic(pairwise_items, 
                  group = pairwise_groups, 
                  focal.name = unique(pairwise_groups)[1], 
                  type = "udif", 
                  purify = TRUE,
                  p.adjust.method = "BH")
    })
    
    # Non-uniform DIF test
    fitLR_nuni <- tryCatch({
      difLogistic(pairwise_items, 
                  group = pairwise_groups, 
                  focal.name = unique(pairwise_groups)[1], 
                  type = "nudif", 
                  purify = FALSE,
                  p.adjust.method = "BH")
    }, error = function(e) {
      difLogistic(pairwise_items, 
                  group = pairwise_groups, 
                  focal.name = unique(pairwise_groups)[1], 
                  type = "nudif", 
                  purify = TRUE,
                  p.adjust.method = "BH")
    })
    
    # Both uniform and non-uniform DIF test
    fitLR_both <- tryCatch({
      difLogistic(pairwise_items, 
                  group = pairwise_groups, 
                  focal.name = unique(pairwise_groups)[1], 
                  type = "both", 
                  purify = FALSE,
                  p.adjust.method = "BH")
    }, error = function(e) {
      difLogistic(pairwise_items, 
                  group = pairwise_groups, 
                  focal.name = unique(pairwise_groups)[1], 
                  type = "both", 
                  purify = TRUE,
                  p.adjust.method = "BH")
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
    
    # Store both DIF results
    temp <- paste0("LR_both_dif_Comparison_", pair_matrix[k, 1], "v", 
                  pair_matrix[k, 2])
    LR_stat <- fitLR_both$Logistik
    LR_df <- 2
    LR_Results_both.L <- cbind(item_num, LR_stat, LR_df, temp)
    
    LR_Results_Full.L <- rbind(LR_Results_Full.L, 
                               LR_Results_uniform.L,
                               LR_Results_nonuniform.L,
                               LR_Results_both.L)
  }
  Results.LR <- LR_Results_Full.L |>
    pivot_wider(names_from = temp, values_from = c(LR_stat, LR_df)) |>
    select(-item_num)
  
  # SIB Test (Simultaneous Item Bias for Uniform DIF) ----
  cat("\nSIB Test")
  SIB_Results.L <- tibble()
  for (k in 1:possible_pairs) {
    pairwise_df <- cbind(g, y) |>
      dplyr::filter(g %in% pair_matrix[k, ])
    pairwise_items <- dplyr::select(pairwise_df, -g)
    pairwise_groups <- as.character(pairwise_df$g)
    
    fitSIB <- tryCatch({
      difSIBTEST(pairwise_items, 
                 group = pairwise_groups, 
                 purify = TRUE,
                 focal.name = unique(pairwise_groups)[1], 
                 type = "udif")
    }, error = function(e) {
      difSIBTEST(pairwise_items, 
                 group = pairwise_groups, 
                 purify = FALSE,
                 focal.name = unique(pairwise_groups)[1], 
                 type = "udif")
    })
    
    temp <- paste0("SIB_Comparison_", pair_matrix[k, 1], "v", pair_matrix[k, 2])
    SIB_stat <- fitSIB$Beta
    SIB_DF <- fitSIB$y
    SIB_Results.L <- rbind(SIB_Results.L,
                           cbind(item_num, SIB_stat, SIB_DF, temp))
  }
  Results.SIB <- SIB_Results.L |>
    pivot_wider(names_from = temp, values_from = SIB_stat) |>
    select(-item_num)
  
  # CSIB Test (Crossing SIB for Non-Uniform DIF) ----
  cat("\nCSIB Test")
  CSIB_Results.L <- tibble()
  for (k in 1:possible_pairs) {
    pairwise_df <- cbind(g, y) |>
      dplyr::filter(g %in% pair_matrix[k, ])
    pairwise_items <- dplyr::select(pairwise_df, -g)
    pairwise_groups <- as.character(pairwise_df$g)
    
    fitCSIB <- tryCatch({
      difSIBTEST(pairwise_items, 
                 group = pairwise_groups, 
                 purify = TRUE,
                 focal.name = unique(pairwise_groups)[1], 
                 type = "nudif")
    }, error = function(e) {
      difSIBTEST(pairwise_items, 
                 group = pairwise_groups, 
                 purify = FALSE,
                 focal.name = unique(pairwise_groups)[1], 
                 type = "nudif")
    })
    
    temp <- paste0("CSIB_Comparison_", pair_matrix[k, 1], "v", pair_matrix[k, 2])
    CSIB_stat <- fitCSIB$Beta
    CSIB_DF <- fitCSIB$y
    CSIB_Results.L <- rbind(CSIB_Results.L,
                            cbind(item_num, CSIB_stat, CSIB_DF, temp))
  }
  Results.CSIB <- CSIB_Results.L |>
    pivot_wider(names_from = temp, values_from = c(CSIB_stat, CSIB_DF)) |>
    select(-item_num)
  
  # Standardized D-statistic ----
  cat("\nStandardized D-stat")
  D_stat.L <- tibble()
  for (k in 1:possible_pairs) {
    pairwise_df <- as.data.frame(cbind(g, y)) |>
      dplyr::filter(g %in% pair_matrix[k, ])
    pairwise_items <- dplyr::select(pairwise_df, -g)
    pairwise_groups <- as.character(pairwise_df$g)
    
    fitD <- tryCatch({
      difR::difStd(Data = pairwise_items, 
                   group = pairwise_groups, 
                   purify = TRUE,
                   focal.name = unique(pairwise_groups)[1])
    }, error = function(e) {
      difR::difStd(Data = pairwise_items, 
                   group = pairwise_groups, 
                   purify = FALSE,
                   focal.name = unique(pairwise_groups)[1])
    })
    
    temp <- paste0("D_Stat_Comparison_", 
                  pair_matrix[k, 1], "v", pair_matrix[k, 2])
    D_stat <- fitD$PDIF
    D_stat.L <- rbind(D_stat.L,
                     cbind(item_num, D_stat, temp))
  }
  
  Results.D <- D_stat.L |>
    pivot_wider(names_from = temp, values_from = D_stat) |>
    select(-item_num)
  
  # Save workspace ----
  save.image(file = filename)
  
  cat("\n")
}


