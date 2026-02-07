################################################################################
# Four Group DIF Data Files for Python
#
# This script converts R training data files (.RData) into CSV files for use
# with the InterDIFNet Python package. It processes multiple DIF detection test
# results and creates standardized CSV outputs for training neural networks.
#
# Input Files:
#   - Estimated_Four_Group_Training_Data_Replication*.RData (500 replications)
#   - Estimated_Empirical_Testing_Parameters.Rdata
#
# Output Files:
#   - Four_Group_Training_data_ALL_Replication*.csv (all DIF test features)
#   - Four_Group_Training_data_TLP_Replication*.csv (TLP features only)
#   - Four_Group_Training_labels_Replication*.csv (DIF labels)
#   - Epirical_Testing_data_ALL.csv (empirical test data, all features)
#   - Epirical_Testing_data_TLP.csv (empirical test data, TLP only)
#
# DIF Tests Included:
#   - CSIB: Crossing-SIBTEST
#   - D: Standardized D statistic
#   - LR: Logistic Regression (uniform, non-uniform, both)
#   - MH: Mantel-Haenszel
#   - SIB: SIBTEST
#   - TLP: Truncated Lasso Penalty (via VEMIRT package)
#
# Created on: December 2024
# @author: Yale Quan
################################################################################

remove(list = ls())

library(tidyverse)
library(dplyr)

# Training data and Labels
training_file_counter <- 1 
for (rep in 1:500) {
  # Check if training data file exists for this replication
  file_path <- paste0('Estimated_Four_Group_Training_Data_Replication', rep, '.RData')
  if (!file.exists(file_path)) {
    next
  }
  load(file_path)
  
  # Convert VEMIRT results to data frame
  Results.VEMIRT <- Results.VEMIRT %>% as.data.frame()
  
  # Standardize column names across all DIF test results
  # CSIB: Crossing-SIBTEST
  names(Results.CSIB) <- names(Results.CSIB) %>% 
    gsub(pattern = "CSIB_Comparison", replacement = "CSIB", x = .)
  
  # D-statistic: Standardized difference measure
  names(Results.D) <- names(Results.D) %>% 
    gsub(pattern = "D_Stat_Comparison.", replacement = "Standardized-D_", x = .)
  
  # SIB: SIBTEST
  names(Results.SIB) <- names(Results.SIB) %>% 
    gsub(pattern = "SIB_Comparison", replacement = "SIB", x = .)
  
  # MH: Mantel-Haenszel
  names(Results.MH) <- names(Results.MH) %>% 
    gsub(pattern = "MH_Comparison", replacement = "Mantel-Hanzel", x = .) %>% 
    gsub(pattern = "MH", replacement = "Mantel-Hanzel", x = .)
  
  # TLP: Truncated Lasso Penalty (VEMIRT package output)
  names(Results.VEMIRT) <- names(Results.VEMIRT) %>% 
    gsub(pattern = "VEMIRT_a", replacement = "TLP_a", x = .) %>% 
    gsub(pattern = "VEMIRT_b", replacement = "TLP_b", x = .) %>% 
    gsub(pattern = "d.a", replacement = "TLP_d.a", x = .) %>% 
    gsub(pattern = "d.b", replacement = "TLP_d.b", x = .)
  
  # LR: Logistic Regression (uniform, non-uniform, and both DIF)
  names(Results.LR) <- names(Results.LR) %>% 
    gsub(pattern = "LR_stat_LR_unifdif", replacement = "LR-Unif", x = .) %>% 
    gsub(pattern = "LR_stat_LR_nonunifdif", replacement = "LR-Nunif", x = .) %>% 
    gsub(pattern = "LR_stat_LR_both_dif", replacement = "LR-DIF", x = .) %>% 
    gsub(pattern = "LR_df_LR_unifdif", replacement = "LR-Unif_df", x = .) %>% 
    gsub(pattern = "LR_df_LR_nonunifdif", replacement = "LR-Nunif_df", x = .) %>% 
    gsub(pattern = "LR_df_LR_both_dif", replacement = "LR-DIF_df", x = .)
  
  # Combine all DIF test features
  out.ALL.features <- cbind(
    Results.CSIB,
    Results.D,
    Results.LR,
    Results.MH,
    Results.SIB,
    Results.VEMIRT
  )
  
  # Preserve column names and reshape to matrix format
  feature_names <- colnames(out.ALL.features)
  J <- nrow(out.ALL.features)
  out.ALL.features <- out.ALL.features %>% 
    as.matrix() %>% 
    as.numeric() %>% 
    matrix(nrow = J)
  colnames(out.ALL.features) <- feature_names
  out.ALL.features <- out.ALL.features %>% as.data.frame()
  
  # Extract TLP-only features
  out.TLP.features <- Results.VEMIRT
  
  # Combine DIF labels (non-uniform and uniform)
  out.labels <- cbind(Labels_a, Labels_b) %>% as.data.frame()
  
  # Write output CSV files
  write_csv(out.ALL.features, paste0("Four_Group_Training_data_ALL_Replication", training_file_counter, ".csv"))
  write_csv(out.TLP.features, paste0("Four_Group_Training_data_TLP_Replication", training_file_counter, ".csv"))
  write_csv(out.labels, paste0("Four_Group_Training_labels_Replication", training_file_counter, ".csv"))

  training_file_counter <- training_file_counter + 1
}

# Clear environment
remove(list = ls())

# Process empirical testing data
file_path <- "Estimated_Empirical_Testing_Parameters.Rdata"
if (!file.exists(file_path)) {
  stop("Empirical testing data file not found")
}
load(file_path)

# Convert VEMIRT results to data frame
Results.VEMIRT <- Results.VEMIRT %>% as.data.frame()

# Standardize column names across all DIF test results
# CSIB: Crossing-SIBTEST
names(Results.CSIB) <- names(Results.CSIB) %>% 
  gsub(pattern = "CSIB_Comparison", replacement = "CSIB", x = .)

# D-statistic: Standardized difference measure
names(Results.D) <- names(Results.D) %>% 
  gsub(pattern = "D_Stat_Comparison.", replacement = "Standardized-D_", x = .)

# SIB: SIBTEST
names(Results.SIB) <- names(Results.SIB) %>% 
  gsub(pattern = "SIB_Comparison", replacement = "SIB", x = .)

# MH: Mantel-Haenszel
names(Results.MH) <- names(Results.MH) %>% 
  gsub(pattern = "MH_Comparison", replacement = "Mantel-Hanzel", x = .) %>% 
  gsub(pattern = "MH", replacement = "Mantel-Hanzel", x = .)

# TLP: Truncated Lasso Penalty (VEMIRT package output)
names(Results.VEMIRT) <- names(Results.VEMIRT) %>% 
  gsub(pattern = "VEMIRT_a", replacement = "TLP_a", x = .) %>% 
  gsub(pattern = "VEMIRT_b", replacement = "TLP_b", x = .) %>% 
  gsub(pattern = "d.a", replacement = "TLP_d.a", x = .) %>% 
  gsub(pattern = "d.b", replacement = "TLP_d.b", x = .)

# LR: Logistic Regression (uniform, non-uniform, and both DIF)
names(Results.LR) <- names(Results.LR) %>% 
  gsub(pattern = "LR_stat_LR_unifdif", replacement = "LR-Unif", x = .) %>% 
  gsub(pattern = "LR_stat_LR_nonunifdif", replacement = "LR-Nunif", x = .) %>% 
  gsub(pattern = "LR_stat_LR_both_dif", replacement = "LR-DIF", x = .) %>% 
  gsub(pattern = "LR_df_LR_unifdif", replacement = "LR-Unif_df", x = .) %>% 
  gsub(pattern = "LR_df_LR_nonunifdif", replacement = "LR-Nunif_df", x = .) %>% 
  gsub(pattern = "LR_df_LR_both_dif", replacement = "LR-DIF_df", x = .)

# Combine all DIF test features
out.ALL.features <- cbind(
  Results.CSIB,
  Results.D,
  Results.LR,
  Results.MH,
  Results.SIB,
  Results.VEMIRT
)

# Preserve column names and reshape to matrix format
feature_names <- colnames(out.ALL.features)
J <- nrow(out.ALL.features)
out.ALL.features <- out.ALL.features %>% 
  as.matrix() %>% 
  as.numeric() %>% 
  matrix(nrow = J)
colnames(out.ALL.features) <- feature_names
out.ALL.features <- out.ALL.features %>% as.data.frame()

# Extract TLP-only features
out.TLP.features <- Results.VEMIRT

# Write output CSV files for empirical testing data
write_csv(out.ALL.features, "Epirical_Testing_data_ALL.csv")
write_csv(out.TLP.features, "Epirical_Testing_data_TLP.csv")





