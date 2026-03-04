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
S <- 5
J <- 10 # Number of items

for (r in 1:500) {

  # Check if file has already been estimated ----
  filename <- paste0("Estimated_Training_data_Replication", r, ".RData")
  if (file.exists(filename)) {
    cat("\n")
    message("Replication ", r, " already exists, Skipping to next")
    next
  }

  message(S,"-Group Replication: ", r)
  load(file = paste0(S, "_Group_Training_Data_Replication_", r, ".RData"))

  # Generate all pairwise group combinations ----
  temp_g <- 1:S
  pair_matrix <- t(combn(unique(g), 2))
  possible_pairs <- nrow(pair_matrix)
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

    group_pairs <- combn(1:S, 2)
    
    # Discrimination differences
    VEMIRT_d.a <- combn(1:S, 2, function(idx) {
      VEMIRT_a[idx[1], ] - VEMIRT_a[idx[2], ]
    })
    
    colnames(VEMIRT_d.a) <- apply(group_pairs, 2,
                                  function(idx) paste0("VEMIRT_d.a_Group", idx[1], "_Group", idx[2])
    )
    
    # Difficulty differences
    VEMIRT_d.b <- combn(1:S, 2, function(idx) {
      VEMIRT_b[idx[1], ] - VEMIRT_b[idx[2], ]
    })
    
    colnames(VEMIRT_d.b) <- apply(group_pairs, 2,
                                  function(idx) paste0("VEMIRT_d.b_Group", idx[1], "_Group", idx[2])
    )
    # Reshape parameter estimates for storage ----
    VEMIRT_a <- t(VEMIRT_a)
    VEMIRT_b <- t(VEMIRT_b)
    group_labels <- sort(unique(g))
    
    colnames(VEMIRT_a) <- paste0("VEMIRT_a_Group", group_labels)
    colnames(VEMIRT_b) <- paste0("VEMIRT_b_Group", group_labels)

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
  for (k in seq_len(nrow(pair_matrix))) {
    pairwise_df <- as.data.frame(cbind(g, y)) %>%
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
  Results.MH <- MH_Results.L %>%
    pivot_wider(names_from = temp, values_from = MH_stat) %>%
    select(-item_num)

  # Logistic Regression ----
  cat("\nLogistic Regression")
  LR_Results_Full.L <- tibble()
  for (k in seq_len(nrow(pair_matrix))) {
    pairwise_df <- cbind(g, y) %>%
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
  Results.LR <- LR_Results_Full.L %>%
    pivot_wider(names_from = temp, values_from = c(LR_stat, LR_df)) %>%
    select(-item_num)

  # SIB Test (Simultaneous Item Bias for Uniform DIF) ----
  cat("\nSIB Test")
  SIB_Results.L <- tibble()
  for (k in seq_len(nrow(pair_matrix))) {
    pairwise_df <- cbind(g, y) %>%
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
  Results.SIB <- SIB_Results.L %>%
    pivot_wider(names_from = temp, values_from = SIB_stat) %>%
    select(-item_num)

  # CSIB Test (Crossing SIB for Non-Uniform DIF) ----
  cat("\nCSIB Test")
  CSIB_Results.L <- tibble()
  for (k in seq_len(nrow(pair_matrix))) {
    pairwise_df <- cbind(g, y) %>%
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
  Results.CSIB <- CSIB_Results.L %>%
    pivot_wider(names_from = temp, values_from = c(CSIB_stat, CSIB_DF)) %>%
    select(-item_num)

  # Standardized D-statistic ----
  cat("\nStandardized D-stat")
  D_stat.L <- tibble()
  for (k in seq_len(nrow(pair_matrix))) {
    pairwise_df <- as.data.frame(cbind(g, y)) %>%
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

  Results.D <- D_stat.L %>%
    pivot_wider(names_from = temp, values_from = D_stat) %>%
    select(-item_num)

  # Save workspace ----
  save.image(file = filename)

  cat("\n")
}


