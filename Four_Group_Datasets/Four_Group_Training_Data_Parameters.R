# ==============================================================================
# Four-Group Training Data Parameter Estimation
# ==============================================================================
#
# Author: Yale Quan
# Date: February 7, 2026
#
# Description:
#   This script estimates item parameters for four-group DIF training data using
#   multiple DIF detection methods. It processes 100 replications of simulated
#   data and applies various pairwise comparison techniques.
#
# Input:
#   - Four_Group_Training_Data_Replication_[r].RData (r = 1 to 100)
#     Contains: y (item responses), g (group membership)
#
# Output:
#   - Estimated_Four_Group_Training_data_Replication[r].RData
#     Contains parameter estimates from all DIF detection methods
#
# Methods:
#   - TLP (Truncated Lasso Penalty): Regularized IRT estimation via VEMIRT package
#   - MH (Mantel-Haenszel): Chi-square test for uniform DIF
#   - LR (Logistic Regression): Tests for uniform, non-uniform, and both DIF types
#   - SIB (Simultaneous Item Bias): Standardized uniform DIF statistic
#   - CSIB (Crossing SIB): Non-uniform DIF detection via item crossing
#   - D-statistic: Standardized mean difference in probabilities
#
# ==============================================================================

remove(list = ls())
options(scipen = 9999)
library(tidyverse)
library(VEMIRT)
library(mirt)
library(difR)
library(DFIT)
library(dplyr)

# Process 100 Replications ----
for (r in 1:100) {
  # Check if file has already been estimated
  filename <- paste0("Estimated_Four_Group_Training_data_Replication", r, ".RData")
  if (file.exists(filename)) {
    cat("\n")
    message("Replication ", r, " already exists, Skipping to next")
    next
  }
  message("Four-Group Replication: ", r)
  load(file = paste0("Four_Group_Training_Data_Replication_", r, ".RData"))
  
  # Setup data dimensions and pairwise comparisons
  S <- max(g)  # Number of groups
  J <- ncol(y)  # Number of items
  N <- nrow(y)  # Number of examinees
  
  # Generate all pairwise group combinations
  temp_g <- 1:S
  possible_pairs <- choose(S, 2)  # C(4,2) = 6 pairwise comparisons
  pair_matrix <- matrix(data = NA, ncol = 2, nrow = possible_pairs)
  item_num <- seq(1:ncol(y))
  group_size <- as.numeric(table(g))
  
  # Build pair matrix: (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
  counter <- 1
  for (i in 1:(S - 1)) {
    for (j in (i + 1):S) {
      pair_matrix[counter, ] <- c(temp_g[i], temp_g[j])
      counter <- counter + 1
    }
  }
  
  y <- as.data.frame(y)
  skip_to_next <- FALSE
  tryCatch({
    # TLP (Truncated Lasso Penalty) Parameter Estimation via VEMIRT ----
    cat("TLP Parameter Estimation")
    cat("\n")
    VEMIRT_df <- list()
    VEMIRT_df[[1]] <- as.data.frame(y)
    VEMIRT_df[[2]] <- g
    
    # Estimate 2PL IRT parameters with regularization
    VEMIRT.m1 <- D2PL_pair_em(
      data = VEMIRT_df[[1]],
      group = VEMIRT_df[[2]],
      Lambda0 = seq(0.1, 1.5, by = 0.1),  # Regularization grid
      Tau = c(Inf, seq(0.05, 0.5, by = 0.05)),  # Threshold grid
      verbose = T
    )
    
    # Select best model by BIC
    bic <- sapply(VEMIRT.m1$all, `[[`, 'BIC')
    temp <- VEMIRT.m1$all[[which.min(bic)]]
    VEMIRT_a <- temp$a  # Discrimination parameters
    VEMIRT_b <- temp$b  # Difficulty parametersb  # Difficulty parameters
    
    # Compute pairwise differences in discrimination (a) parameters
    VEMIRT_d.a <- combn(1:nrow(VEMIRT_a), 2, function(idx) {
      diff <- VEMIRT_a[idx[1], ] - VEMIRT_a[idx[2], ]
      return(diff)
    })
    colnames(VEMIRT_d.a) <- apply(
      combn(1:ncol(VEMIRT_d.a), 2), 2,
      function(idx) paste0("d.a_", "Group", idx[1], "Group", idx[2])
    )
    
    # Compute pairwise differences in difficulty (b) parameters
    VEMIRT_d.b <- combn(1:nrow(VEMIRT_b), 2, function(idx) {
      diff <- VEMIRT_b[idx[1], ] - VEMIRT_b[idx[2], ]
      return(diff)
    })
    colnames(VEMIRT_d.b) <- apply(
      combn(1:ncol(VEMIRT_d.b), 2), 2,
      function(idx) paste0("d.b_", "Group", idx[1], "Group", idx[2])
    )
    
    # Reshape parameter estimates for storage (items x groups)
    VEMIRT_a <- t(VEMIRT_a)
    VEMIRT_b <- t(VEMIRT_b)
    colnames(VEMIRT_a) <- paste0("VEMIRT_a_Group", 1:S)
    colnames(VEMIRT_b) <- paste0("VEMIRT_b_Group", 1:S)
    
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
  
  # Mantel-Haenszel Test (Uniform DIF) ----
  cat("\nMantel-Haenszel pairwise test")
  MH_Results.L <- tibble()
  for (k in 1:possible_pairs) {
    # Extract pairwise comparison data
    pairwise_df <- as.data.frame(cbind(g, y)) %>%
      dplyr::filter(g %in% pair_matrix[k, ])
    pairwise_items <- dplyr::select(pairwise_df, -g)
    paiwise_groups <- as.character(pairwise_df$g)
    
    # Apply Mantel-Haenszel test with purification fallback
    fitMH <- tryCatch({
      difMH(
        pairwise_items,
        group = paiwise_groups,
        focal.name = unique(paiwise_groups)[1],
        purify = T,
        p.adjust.method = "BH"
      )
    }, error = function(e) {
      difMH(
        pairwise_items,
        group = paiwise_groups,
        focal.name = unique(paiwise_groups)[1],
        purify = F,
        p.adjust.method = "BH"
      )
    })
    
    temp <- paste0("MH_Comparison_", pair_matrix[k, 1], "v", pair_matrix[k, 2])
    MH_stat <- fitMH$MH
    MH_df <- 1
    MH_Results.L <- rbind(MH_Results.L, cbind(item_num, MH_stat, MH_df, temp))
  }
  
  Results.MH <- MH_Results.L %>%
    pivot_wider(names_from = temp, values_from = MH_stat) %>%
    select(-item_num)
  
  # Logistic Regression (Uniform, Non-uniform, Both DIF) ----
  cat("\nLogistic Regression")
  LR_Results_Full.L <- tibble()
  for (k in 1:possible_pairs) {
    # Extract pairwise comparison data
    pairwise_df <- cbind(g, y) %>%
      dplyr::filter(g %in% pair_matrix[k, ])
    pairwise_items <- dplyr::select(pairwise_df, -g)
    paiwise_groups <- as.character(pairwise_df$g)
    
    # Test for uniform DIF (difficulty differences only)
    fitLR_uni <- tryCatch({
      difLogistic(
        pairwise_items,
        group = paiwise_groups,
        focal.name = unique(paiwise_groups)[1],
        type = "udif",
        purify = F,
        p.adjust.method = "BH"
      )
    }, error = function(e) {
      difLogistic(
        pairwise_items,
        group = paiwise_groups,
        focal.name = unique(paiwise_groups)[1],
        type = "udif",
        purify = T,
        p.adjust.method = "BH"
      )
    })
    
    # Test for non-uniform DIF (discrimination differences)
    fitLR_nuni <- tryCatch({
      difLogistic(
        pairwise_items,
        group = paiwise_groups,
        focal.name = unique(paiwise_groups)[1],
        type = "nudif",
        purify = F,
        p.adjust.method = "BH"
      )
    }, error = function(e) {
      difLogistic(
        pairwise_items,
        group = paiwise_groups,
        focal.name = unique(paiwise_groups)[1],
        type = "nudif",
        purify = T,
        p.adjust.method = "BH"
      )
    })
    
    # Test for both uniform and non-uniform DIF
    fitLR_both <- tryCatch({
      difLogistic(
        pairwise_items,
        group = paiwise_groups,
        focal.name = unique(paiwise_groups)[1],
        type = "both",
        purify = F,
        p.adjust.method = "BH"
      )
    }, error = function(e) {
      difLogistic(
        pairwise_items,
        group = paiwise_groups,
        focal.name = unique(paiwise_groups)[1],
        type = "both",
        purify = T,
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
    
    # Store both DIF results
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
  
  # SIB Test (Simultaneous Item Bias - Uniform DIF) ----
  cat("\nSIB Test")
  SIB_Results.L <- tibble()
  for (k in 1:possible_pairs) {
    # Extract pairwise comparison data
    pairwise_df <- cbind(g, y) %>%
      dplyr::filter(g %in% pair_matrix[k, ])
    pairwise_items <- dplyr::select(pairwise_df, -g)
    paiwise_groups <- as.character(pairwise_df$g)
    
    # Apply SIB test for uniform DIF with purification fallback
    fitSIB <- tryCatch({
      difSIBTEST(
        pairwise_items,
        group = paiwise_groups,
        purify = T,
        focal.name = unique(paiwise_groups)[1],
        type = "udif"
      )
    }, error = function(e) {
      difSIBTEST(
        pairwise_items,
        group = paiwise_groups,
        purify = F,
        focal.name = unique(paiwise_groups)[1],
        type = "udif"
      )
    })
    
    temp <- paste0("SIB_Comparison_", pair_matrix[k, 1], "v", pair_matrix[k, 2])
    SIB_stat <- fitSIB$Beta
    SIB_DF <- fitSIB$y
    SIB_Results.L <- rbind(SIB_Results.L, cbind(item_num, SIB_stat, SIB_DF, temp))
  }
  
  Results.SIB <- SIB_Results.L %>%
    pivot_wider(names_from = temp, values_from = SIB_stat) %>%
    select(-item_num)
  
  # CSIB Test (Crossing SIB - Non-uniform DIF) ----
  cat("\nCSIB Test")
  CSIB_Results.L <- tibble()
  for (k in 1:possible_pairs) {
    # Extract pairwise comparison data
    pairwise_df <- cbind(g, y) %>%
      dplyr::filter(g %in% pair_matrix[k, ])
    pairwise_items <- dplyr::select(pairwise_df, -g)
    paiwise_groups <- as.character(pairwise_df$g)
    
    # Apply CSIB test for non-uniform DIF with purification fallback
    fitCSIB <- tryCatch({
      difSIBTEST(
        pairwise_items,
        group = paiwise_groups,
        purify = T,
        focal.name = unique(paiwise_groups)[1],
        type = "nudif"
      )
    }, error = function(e) {
      difSIBTEST(
        pairwise_items,
        group = paiwise_groups,
        purify = F,
        focal.name = unique(paiwise_groups)[1],
        type = "nudif"
      )
    })
    
    temp <- paste0("CSIB_Comparison_", pair_matrix[k, 1], "v", pair_matrix[k, 2])
    CSIB_stat <- fitCSIB$Beta
    CSIB_DF <- fitCSIB$y
    CSIB_Results.L <- rbind(CSIB_Results.L, cbind(item_num, CSIB_stat, CSIB_DF, temp))
  }
  
  Results.CSIB <- CSIB_Results.L %>%
    pivot_wider(names_from = temp, values_from = c(CSIB_stat, CSIB_DF)) %>%
    select(-item_num)
  
  # Standardized D-statistic (Effect Size) ----
  cat("\nStandardized D-stat")
  D_stat.L <- tibble()
  for (k in 1:possible_pairs) {
    # Extract pairwise comparison data
    pairwise_df <- as.data.frame(cbind(g, y)) %>%
      dplyr::filter(g %in% pair_matrix[k, ])
    pairwise_items <- dplyr::select(pairwise_df, -g)
    paiwise_groups <- as.character(pairwise_df$g)
    
    # Compute standardized mean difference with purification fallback
    fitD <- tryCatch({
      difR::difStd(
        Data = pairwise_items,
        group = paiwise_groups,
        purify = T,
        focal.name = unique(paiwise_groups)[1]
      )
    }, error = function(e) {
      difR::difStd(
        Data = pairwise_items,
        group = paiwise_groups,
        purify = F,
        focal.name = unique(paiwise_groups)[1]
      )
    })
    
    temp <- paste0("D_Stat_Comparison_", pair_matrix[k, 1], "v", pair_matrix[k, 2])
    D_stat <- fitD$PDIF
    D_stat.L <- rbind(D_stat.L, cbind(item_num, D_stat, temp))
  }
  
  Results.D <- D_stat.L %>%
    pivot_wider(names_from = temp, values_from = D_stat) %>%
    select(-item_num)
  
  # Save Results ----
  save.image(file = filename)
  cat("\n")
}


