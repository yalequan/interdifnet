remove(list = ls())
options(scipen = 9999)
library(tidyverse)
library(VEMIRT)
library(mirt)
library(difR)
library(DFIT)
library(dplyr)

# Load the data ----
S <- 3  # Number of groups
J <- 10 # Number of items
j_dif <- c(2,4)
DIF_perc <- j_dif * 10
N_sizes = c(250, 500, 1000) # 250, 500, 1000
for(N in N_sizes){
  for(perc in DIF_perc){
    for (r in 1:100) {
      skip_to_next <- FALSE
      tryCatch({
        cat("\nEstimating: Groups = ", S, "N =", N, ", DIF Percentage =", perc, ", Replication:", r)
        # Check if file has already been estimated
        filename = paste0('Estimated_Three_Group_Testing_data_',N,'_', perc, '_Replication',r,'.RData')
        if (file.exists(filename)) {
          cat("\n")
          message("File already exists, Skipping to next")
          next
        }
        
        load(paste0("Three_Group_Testing_Data_", N, "_", perc, "_Replication_", r, ".RData"))
        cat("\n")
        
        # get all pairwise combinations
        temp_g = 1:S
        possible_pairs = choose(S, 2)
        pair_matrix = matrix(data = NA, ncol = 2, nrow = possible_pairs)
        item_num = seq(1:ncol(y))
        group_size = as.numeric(table(g))
        # Initialize a counter for the matrix row
        counter = 1
        for (i in 1:(S - 1)) {
          for (j in (i + 1):S) {
            pair_matrix[counter, ] = c(temp_g[i], temp_g[j])
            counter = counter + 1
          }
        }
        
        J = ncol(y) # Number of items
        
        y = as.data.frame(y)
        
        # TLP Parameter Estimation ----
        cat("\nTLP Estimation")
        cat("\n") 
        VEMIRT_df = list()
        VEMIRT_df[[1]] = as.data.frame(y)
        VEMIRT_df[[2]] =  g
        VEMIRT.m1 = D2PL_pair_em(data = VEMIRT_df[[1]],
                                 group = VEMIRT_df[[2]],
                                 Lambda0 = seq(0.1, 1.5, by = 0.1), 
                                 Tau = c(Inf, seq(0.05, 0.5, by = 0.05)),
                                 verbose = F)
        bic <- sapply(VEMIRT.m1$all, `[[`, 'BIC')
        temp = VEMIRT.m1$all[[which.min(bic)]]
        VEMIRT_a = temp$a
        VEMIRT_b = temp$b
        
        # Paiwise Differences in a estimates
        VEMIRT_d.a <- combn(1:nrow(VEMIRT_a), 2, function(idx) {
          diff <- VEMIRT_a[idx[1], ] - VEMIRT_a[idx[2], ]
          return(diff)
        })
        colnames(VEMIRT_d.a) = apply(combn(1:ncol(VEMIRT_d.a), 2), 2, 
                                     function(idx) paste0("d.a_", "Group", 
                                                          idx[1], "Group", idx[2]))
        
        # Paiwise Differences in b estimates
        VEMIRT_d.b <- combn(1:nrow(VEMIRT_b), 2, function(idx) {
          # idx contains the indices of the pair
          diff <- VEMIRT_b[idx[1], ] - VEMIRT_b[idx[2], ]
          return(diff)
        })
        colnames(VEMIRT_d.b) <- apply(combn(1:ncol(VEMIRT_d.b), 2), 2, 
                                      function(idx) paste0("d.b_", "Group", 
                                                           idx[1], "Group", idx[2]))
        
        # Reshape parameter estimates for storage
        VEMIRT_a <- t(VEMIRT_a)
        VEMIRT_b <- t(VEMIRT_b)
        colnames(VEMIRT_a) <- paste0("VEMIRT_a_Group", paste0(1:S))
        colnames(VEMIRT_b) <- paste0("VEMIRT_b_Group", paste0(1:S))
        
        Results.VEMIRT = cbind(VEMIRT_a, VEMIRT_b, VEMIRT_d.a, VEMIRT_d.b)
        
        # Mantel-Haenszel pairwise test ----
        cat("\nMantel-Haenszel pairwise test")
        MH_Results.L = tibble()
        for (k in 1:possible_pairs){
          pairwise_df = as.data.frame(cbind(g, y)) %>%
            dplyr::filter(g %in% pair_matrix[k, ])
          pairwise_items = dplyr::select(pairwise_df, -g)
          paiwise_groups = as.character(pairwise_df$g)
          fitMH <- tryCatch({
            difMH(pairwise_items, 
                  group = paiwise_groups, 
                  focal.name = unique(paiwise_groups)[1],
                  purify = T,
                  p.adjust.method = "BH")
          }, error = function(e){
            difMH(pairwise_items, 
                  group = paiwise_groups, 
                  focal.name = unique(paiwise_groups)[1],
                  purify = F,
                  p.adjust.method = "BH")
          })
          temp = paste0("MH_Comparison_", pair_matrix[k, 1], "v", pair_matrix[k, 2])
          MH_stat = fitMH$MH
          MH_df = 1
          MH_Results.L = rbind(MH_Results.L,
                               cbind(item_num, MH_stat, MH_df, temp))
        }
        Results.MH = MH_Results.L %>%
          pivot_wider(names_from = temp, values_from = MH_stat) %>%
          select(-item_num)
        
        # Logistic Regression ----
        cat("\nLogistic Regression")
        LR_Results_Full.L = tibble()
        for (k in 1:possible_pairs){
          pairwise_df = cbind(g, y) %>%
            dplyr::filter(g %in% pair_matrix[k, ])
          pairwise_items = dplyr::select(pairwise_df, -g)
          paiwise_groups = as.character(pairwise_df$g)
          fitLR_uni <- tryCatch({
            difLogistic(pairwise_items, 
                        group = paiwise_groups, 
                        focal.name = unique(paiwise_groups)[1], 
                        type = "udif", 
                        purify = F,
                        p.adjust.method = "BH")
          }, error = function(e){
            difLogistic(pairwise_items, 
                        group = paiwise_groups, 
                        focal.name = unique(paiwise_groups)[1], 
                        type = "udif", 
                        purify = T,
                        p.adjust.method = "BH")
          })
          fitLR_nuni <- tryCatch({
            difLogistic(pairwise_items, 
                        group = paiwise_groups, 
                        focal.name = unique(paiwise_groups)[1], 
                        type = "nudif", 
                        purify = F,
                        p.adjust.method = "BH")
          }, error = function(e){
            difLogistic(pairwise_items, 
                        group = paiwise_groups, 
                        focal.name = unique(paiwise_groups)[1], 
                        type = "nudif", 
                        purify = T,
                        p.adjust.method = "BH")
          })
          fitLR_both <- tryCatch({
            difLogistic(pairwise_items, 
                        group = paiwise_groups, 
                        focal.name = unique(paiwise_groups)[1], 
                        type = "both", 
                        purify = F,
                        p.adjust.method = "BH")
          }, error = function(e){
            difLogistic(pairwise_items, 
                        group = paiwise_groups, 
                        focal.name = unique(paiwise_groups)[1], 
                        type = "both", 
                        purify = T,
                        p.adjust.method = "BH")
          })
          temp = paste0("LR_unifdif_Comparison_", pair_matrix[k, 1], "v", 
                        pair_matrix[k, 2])
          LR_stat = fitLR_uni$Logistik
          LR_df = 1
          LR_Results_uniform.L = cbind(item_num, LR_stat, LR_df, temp)
          
          temp = paste0("LR_nonunifdif_Comparison_", pair_matrix[k, 1], "v",
                        pair_matrix[k, 2])
          LR_stat = fitLR_nuni$Logistik
          LR_df = 1
          LR_Results_nonuniform.L = cbind(item_num, LR_stat, LR_df, temp)
          
          temp = paste0("LR_both_dif_Comparison_", pair_matrix[k, 1], "v", 
                        pair_matrix[k, 2])
          LR_stat = fitLR_both$Logistik
          LR_df = 2
          LR_Results_both.L = cbind(item_num, LR_stat, LR_df, temp)
          
          LR_Results_Full.L = rbind(LR_Results_Full.L, 
                                    LR_Results_uniform.L,
                                    LR_Results_nonuniform.L,
                                    LR_Results_both.L)
        }
        Results.LR = LR_Results_Full.L %>%
          pivot_wider(names_from = temp, values_from = c(LR_stat, LR_df)) %>%
          select(-item_num)
        
        # SIB Test ----
        cat("\nSIB Test")
        SIB_Results.L = tibble()
        for (k in 1:possible_pairs){
          pairwise_df = cbind(g, y) %>%
            dplyr::filter(g %in% pair_matrix[k, ])
          pairwise_items = dplyr::select(pairwise_df, -g)
          paiwise_groups = as.character(pairwise_df$g)
          fitSIB <- tryCatch({
            difSIBTEST(pairwise_items, 
                       group = paiwise_groups, 
                       purify = T,
                       focal.name = unique(paiwise_groups)[1], 
                       type = "udif")
          }, error = function(e){
            difSIBTEST(pairwise_items, 
                       group = paiwise_groups, 
                       purify = F,
                       focal.name = unique(paiwise_groups)[1], 
                       type = "udif")
          })
          temp = paste0("SIB_Comparison_", pair_matrix[k, 1], "v", pair_matrix[k, 2])
          SIB_stat = fitSIB$Beta
          SIB_DF = fitSIB$y
          SIB_Results.L = rbind(SIB_Results.L,
                                cbind(item_num, SIB_stat, SIB_DF, temp))
        }
        Results.SIB = SIB_Results.L %>%
          pivot_wider(names_from = temp, values_from = SIB_stat) %>%
          select(-item_num)
        
        # Crossing SIB Test ----
        cat("\nCSIB Test")
        CSIB_Results.L = tibble()
        for (k in 1:possible_pairs){
          pairwise_df = cbind(g, y) %>%
            dplyr::filter(g %in% pair_matrix[k, ])
          pairwise_items = dplyr::select(pairwise_df, -g)
          paiwise_groups = as.character(pairwise_df$g)
          fitCSIB <- tryCatch({
            difSIBTEST(pairwise_items, 
                       group = paiwise_groups, 
                       purify = T,
                       focal.name = unique(paiwise_groups)[1], 
                       type = "nudif")
          }, error = function(e){
            difSIBTEST(pairwise_items, 
                       group = paiwise_groups, 
                       purify = F,
                       focal.name = unique(paiwise_groups)[1], 
                       type = "nudif")
          })
          temp = paste0("CSIB_Comparison_", pair_matrix[k, 1], "v", pair_matrix[k, 2])
          CSIB_stat = fitCSIB$Beta
          CSIB_DF = fitCSIB$y
          CSIB_Results.L = rbind(CSIB_Results.L,
                                 cbind(item_num, CSIB_stat, CSIB_DF, temp))
        }
        Results.CSIB = CSIB_Results.L %>%
          pivot_wider(names_from = temp, values_from = c(CSIB_stat, CSIB_DF)) %>%
          select(-item_num)
        
        # Standardized D-stat ----
        cat("\nStandardized D-stat")
        D_stat.L = tibble()
        for (k in 1:possible_pairs){
          pairwise_df = as.data.frame(cbind(g, y)) %>%
            dplyr::filter(g %in% pair_matrix[k, ])
          pairwise_items = dplyr::select(pairwise_df, -g)
          paiwise_groups = as.character(pairwise_df$g)
          fitD <- tryCatch({
            difR::difStd(Data = pairwise_items, 
                         group = paiwise_groups, 
                         purify = T,
                         focal.name = unique(paiwise_groups)[1])
          }, error = function(e){
            difR::difStd(Data = pairwise_items, 
                         group = paiwise_groups, 
                         purify = F,
                         focal.name = unique(paiwise_groups)[1])
          })
          temp = paste0("D_Stat_Comparison_", 
                        pair_matrix[k, 1], "v", pair_matrix[k, 2])
          D_stat = fitD$PDIF
          D_stat.L = rbind(D_stat.L,
                           cbind(item_num, D_stat, temp))
        }
        
        Results.D = D_stat.L %>%
          pivot_wider(names_from = temp, values_from = D_stat) %>%
          select(-item_num)
        
        # Save   
        save.image(file = filename)
      },
      
      error = function(e){
        cat("\n")
        message("Error: ", e)
        skip_to_next <<- TRUE})
      if(skip_to_next) {
        next }
      cat("\n")
    }
  }
}


