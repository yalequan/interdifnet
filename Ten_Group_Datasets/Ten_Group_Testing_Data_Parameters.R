#' Estimate IRT Parameters and DIF Statistics for Testing Data
#'
#' This script processes simulated testing data to estimate item response theory
#' (IRT) parameters and compute various differential item functioning (DIF)
#' detection statistics across multiple experimental conditions. These estimates
#' are used to evaluate the InterDIFNet model performance.
#'
#' @section Overview:
#' The script performs the following operations for each testing condition:
#' 1. Loads simulated response data from RData files
#' 2. Estimates 2PL IRT parameters using VEMIRT (penalized estimation)
#' 3. Computes pairwise DIF statistics across all group combinations
#' 4. Saves estimated parameters and test statistics for model evaluation
#'
#' @section IRT Parameter Estimation:
#' - Method: VEMIRT (Variational EM for IRT with regularization)
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
#' @section Experimental Conditions:
#' The script processes data across multiple conditions:
#' - **Sample sizes**: 1000, 2000, 4000
#' - **DIF percentages**: 20%, 40% (2 or 4 items with DIF out of 10)
#' - **Replications**: 101-125 (25 replications per condition)
#' - **Groups**: 10 comparison groups
#'
#' @section Computational Details:
#' - Convergence tolerance: Default VEMIRT settings
#' - Error handling: Skips failed replications and continues processing
#' - Purification: Applied with fallback when convergence issues arise
#' - Verbose output: Enabled for VEMIRT to monitor progress
#'
#' @section Input Files:
#' - Format: "Ten_Group_Testing_Data_[N]_[perc]_Replication_[r].RData"
#' - Contains: Response matrix (y), group vector (g), true parameters (a, b)
#' - N: Sample size (1000, 2000, or 4000)
#' - perc: DIF percentage (20 or 40)
#' - r: Replication number (101-125)
#'
#' @section Output Files:
#' - Format: "Estimated_Ten_Group_Testing_data_[N]_[perc]_Replication[r].RData"
#' - Contains: All estimated parameters, test statistics, and original data
#' - Variables:
#'   - Results.VEMIRT: Parameter estimates and pairwise differences
#'   - Results.MH: Mantel-Haenszel statistics
#'   - Results.LR: Logistic regression statistics
#'   - Results.SIB, Results.CSIB: SIB test statistics
#'   - Results.D: Standardized D-statistics
#'
#' @author Yale Quan
#' @date 2024-07-01
#' 
#' #' @note
#' - Processing time: Approximately 2-5 minutes per replication
#' - Memory usage: Moderate (dependent on sample size)
#' - Dependencies: tidyverse, VEMIRT, mirt, difR, DFIT, dplyr
#'
#' @examples
#' # This script is designed to be run as a standalone file
#' # It will process all conditions automatically
#' # source("Ten_Group_Testing_Data_Parameters.R")

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

# Experimental Conditions ----

S <- 10  # Number of groups
J <- 10  # Number of items
j_dif <- c(2, 4)  # Number of items with DIF
DIF_perc <- j_dif * 10  # DIF percentages (20%, 40%)
N_sizes <- c(1000, 2000, 4000)  # Sample sizes to test

# Main Processing Loop ----
# Iterate through all experimental conditions

for (N in N_sizes) {
  for (perc in DIF_perc) {
    for (r in 101:125) {
      # Error Handling Setup ----
      skip_to_next <- FALSE
      
      tryCatch({
        cat("\nEstimating: Groups:", S, "| N:", N, "| DIF%:", perc,
            "| Replication:", r)
        
        # Check if file has already been estimated
        filename <- paste0("Estimated_Ten_Group_Testing_data_", N, "_", perc,
                          "_Replication", r, ".RData")
        
        if (file.exists(filename)) {
          cat("\n")
          message("File already exists, skipping to next")
          next
        }
        
        # Load simulated testing data
        load(paste0("Ten_Group_Testing_Data_", N, "_", perc,
                   "_Replication_", r, ".RData"))
        cat("\n")
        
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
        
                # Prepare data for analysis
        J <- ncol(y)  # Number of items
        y <- as.data.frame(y)
        
        # VEMIRT Parameter Estimation ----
        
        cat("\nEstimating IRT Parameters with VEMIRT (2PL Model)")
        cat("\n")
        
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
          verbose = TRUE
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
        
        
        # # LRT DIF Test pairwise ----
        # cat("\nAnchor Item Selection")
        # # Select anchor items (Fit fully constrained model, assumes no DIF,
        # # an no impact)
        # md.cons0 = multipleGroup(y, 1, 
        #                          group = as.character(g),
        #                          SE=TRUE,
        #                          TOL = 1e-2,
        #                          verbose = F,
        #                          invariance=c('free_means', 'free_var',
        #                                       colnames(y)))
        # d=DIF(md.cons0, which.par = c('a1', 'd'), p.adjust = 'BH',scheme = 'drop',
        #       verbose = F, TOL = 1e-2, pairwise = T)
        # d$ratio1=d$X2/d$df
        # 
        # # n_anchor is the number of anchors you want to use
        # n_anchor=1
        # anchors_number=as.numeric(order(d$ratio1)[1:n_anchor])
        # anchors=colnames(y)[anchors_number]
        # items2test = colnames(y)[-anchors_number]
        # 
        # # Fit model with anchors
        # mirt.2 = multipleGroup(data = y, model = 1, 
        #                        SE = T,
        #                        TOL = 1e-2,
        #                        verbose = F,
        #                        group = as.character(g), 
        #                        invariance = c(anchors, "free_means"))
        # 
        # cat("\nLRT Pairwise Nonuniform Test")
        # DIF_LRT.L = tibble()
        # 
        # 
        # DIF_LRT = DIF(mirt.2, which.par = c("a1", "d"), Wald = F, scheme = "add",
        #               verbose = F, TOL = 1e-2, items2test = items2test,
        #               pairwise = T)
        # 
        # # Clean the output
        # DIF_LRT = as.data.frame(DIF_LRT)
        # DIF_LRT$item <- as.numeric(gsub("V", "", DIF_LRT$item))
        # DIF_LRT$groups <- gsub(",", "v", DIF_LRT$groups)
        # temp_LRT_Results = cbind(DIF_LRT$item,
        #                          DIF_LRT$X2,
        #                          DIF_LRT$df,
        #                          DIF_LRT$groups)
        # DIF_LRT.L = as.data.frame(temp_LRT_Results)
        # colnames(DIF_LRT.L) = c("item_num", "X2", "df", "groups")
        # DIF_LRT.L$groups = paste0("nunifDIF_LRT_",DIF_LRT.L$groups)
        # 
        # 
        # DIF_LRT.L$item_num = as.numeric(DIF_LRT.L$item_num)
        # DIF_LRT.L$X2 = as.numeric(DIF_LRT.L$X2)
        # DIF_LRT.L$df = as.numeric(DIF_LRT.L$df)
        # 
        # Results.LRT_nonuniform = DIF_LRT.L %>%
        #   pivot_wider(id_cols = item_num,
        #               names_from = groups, 
        #               values_from = c(X2, df)) 
        # # Handle anchor items after pivot
        # temp_row = Results.LRT_nonuniform[1,]
        # temp_row[] = NA
        # temp_row$item_num = anchors_number
        # 
        # Results.LRT_nonuniform = rbind(Results.LRT_nonuniform,temp_row)
        # Results.LRT_nonuniform <- Results.LRT_nonuniform[order(Results.LRT_nonuniform$item_num), ]
        # 
        # 
        # cat("\nLRT Pairwise Uniform Test")
        # DIF_LRT.L = tibble()
        # DIF_LRT = DIF(mirt.2, which.par = c("d"), Wald = F, scheme = "add",
        #               verbose = F, TOL = 1e-2, items2test = items2test,
        #               pairwise = T)
        # 
        # # Clean the output
        # DIF_LRT = as.data.frame(DIF_LRT)
        # DIF_LRT$item <- as.numeric(gsub("V", "", DIF_LRT$item))
        # DIF_LRT$groups <- gsub(",", "v", DIF_LRT$groups)
        # temp_LRT_Results = cbind(DIF_LRT$item,
        #                          DIF_LRT$X2,
        #                          DIF_LRT$df,
        #                          DIF_LRT$groups)
        # DIF_LRT.L = as.data.frame(temp_LRT_Results)
        # colnames(DIF_LRT.L) = c("item_num", "X2", "df", "groups")
        # DIF_LRT.L$groups = paste0("unifDIF_LRT_",DIF_LRT.L$groups)
        # 
        # 
        # DIF_LRT.L$item_num = as.numeric(DIF_LRT.L$item_num)
        # DIF_LRT.L$X2 = as.numeric(DIF_LRT.L$X2)
        # DIF_LRT.L$df = as.numeric(DIF_LRT.L$df)
        # 
        # Results.LRT_uniform = DIF_LRT.L %>%
        #   pivot_wider(id_cols = item_num,
        #               names_from = groups, 
        #               values_from = c(X2, df)) 
        # # Handle anchor items after pivot
        # temp_row = Results.LRT_uniform[1,]
        # temp_row[] = NA
        # temp_row$item_num = anchors_number
        # 
        # Results.LRT_uniform = rbind(Results.LRT_uniform,temp_row)
        # Results.LRT_uniform <- Results.LRT_uniform[order(Results.LRT_uniform$item_num), ]
        # 
        # # Wald DIF Test pairwise ----
        # cat("\nWald Pairwise Nonuniform Test")
        # DIF_Wald.L = tibble()
        # DIF_Wald = DIF(mirt.2, which.par = c("a1", "d"), Wald = T, scheme = "add",
        #                verbose = F, TOL = 1e-2, items2test = items2test,
        #                pairwise = T)
        # 
        # # Clean the output
        # DIF_Wald = as.data.frame(DIF_Wald)
        # DIF_Wald$item <- as.numeric(gsub("V", "", DIF_Wald$item))
        # DIF_Wald$groups <- gsub(",", "v", DIF_Wald$groups)
        # DIF_Wald.L = as.data.frame(DIF_Wald)
        # colnames(DIF_Wald.L) = c("item_num", "groups", "W", "df", "p")
        # DIF_Wald.L$groups = paste0("nonunifDIF_Wald_",DIF_Wald.L$groups)
        # 
        # 
        # DIF_Wald.L$item_num = as.numeric(DIF_Wald.L$item_num)
        # DIF_Wald.L$W = as.numeric(DIF_Wald.L$W)
        # DIF_Wald.L$df = as.numeric(DIF_Wald.L$df)
        # 
        # Results.Wald_nonuniform = DIF_Wald.L %>%
        #   pivot_wider(id_cols = item_num,
        #               names_from = groups, 
        #               values_from = c(W, df)) 
        # # Handle anchor items after pivot
        # temp_row = Results.Wald_nonuniform[1,]
        # temp_row[] = NA
        # temp_row$item_num = anchors_number
        # 
        # Results.Wald_nonuniform = rbind(Results.Wald_nonuniform,temp_row)
        # Results.Wald_nonuniform <- Results.Wald_nonuniform[order(Results.Wald_nonuniform$item_num), ]
        # 
        # 
        # cat("\nWald Pairwise Uniform Test")
        # DIF_Wald.L = tibble()
        # DIF_Wald = DIF(mirt.2, which.par = c("d"), Wald = T, scheme = "add",
        #                verbose = F, TOL = 1e-2, items2test = items2test,
        #                pairwise = T)
        # 
        # # Clean the output
        # DIF_Wald = as.data.frame(DIF_Wald)
        # DIF_Wald$item <- as.numeric(gsub("V", "", DIF_Wald$item))
        # DIF_Wald$groups <- gsub(",", "v", DIF_Wald$groups)
        # DIF_Wald.L = as.data.frame(DIF_Wald)
        # colnames(DIF_Wald.L) = c("item_num", "groups", "W", "df", "p")
        # DIF_Wald.L$groups = paste0("unifDIF_Wald_",DIF_Wald.L$groups)
        # 
        # 
        # DIF_Wald.L$item_num = as.numeric(DIF_Wald.L$item_num)
        # DIF_Wald.L$W = as.numeric(DIF_Wald.L$W)
        # DIF_Wald.L$df = as.numeric(DIF_Wald.L$df)
        # 
        # Results.Wald_uniform = DIF_Wald.L %>%
        #   pivot_wider(id_cols = item_num,
        #               names_from = groups, 
        #               values_from = c(W, df)) 
        # # Handle anchor items after pivot
        # temp_row = Results.Wald_uniform[1,]
        # temp_row[] = NA
        # temp_row$item_num = anchors_number
        # 
        # Results.Wald_uniform = rbind(Results.Wald_uniform,temp_row)
        # Results.Wald_uniform <- Results.Wald_uniform[order(Results.Wald_uniform$item_num), ]
        
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
        
        # Save all results to RData file for model evaluation
        save.image(file = filename)
      },
      error = function(e) {
        cat("\n")
        message("Error: ", e)
        skip_to_next <<- TRUE
      })
      
      if (skip_to_next) {
        next
      }
      cat("\n")
    }
  }
}

# End of script


