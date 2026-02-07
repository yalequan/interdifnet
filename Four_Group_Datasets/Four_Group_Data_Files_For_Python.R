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
  # Check if training data exists
  file_path <- paste0('Estimated_Four_Group_Training_Data_Replication',rep,'.RData')
  if (!file.exists(file_path)) {
    next
  }
  load(file_path)
  
  # Fix naming
  Results.VEMIRT = as.data.frame(Results.VEMIRT)
  
  names(Results.CSIB) <- gsub(x = names(Results.CSIB), 
                           pattern = "CSIB_Comparison", 
                           replacement = "CSIB") 
  names(Results.D) <- gsub(x = names(Results.D), 
                           pattern = "D_Stat_Comparison.", 
                           replacement = "Standardized-D_") 
  names(Results.SIB) <- gsub(x = names(Results.SIB), 
                              pattern = "SIB_Comparison", 
                              replacement = "SIB") 
  names(Results.MH) <- gsub(x = names(Results.MH), 
                             pattern = "MH_Comparison", 
                             replacement = "Mantel-Hanzel") 
  names(Results.MH) <- gsub(x = names(Results.MH), 
                            pattern = "MH", 
                            replacement = "Mantel-Hanzel") 
  names(Results.VEMIRT) <- gsub(x = names(Results.VEMIRT), 
                            pattern = "VEMIRT_a", 
                            replacement = "TLP_a") 
  names(Results.VEMIRT) <- gsub(x = names(Results.VEMIRT), 
                            pattern = "VEMIRT_b", 
                            replacement = "TLP_b") 
  names(Results.VEMIRT) <- gsub(x = names(Results.VEMIRT), 
                                pattern = "d.a", 
                                replacement = "TLP_d.a") 
  names(Results.VEMIRT) <- gsub(x = names(Results.VEMIRT), 
                                pattern = "d.b", 
                                replacement = "TLP_d.b") 
  names(Results.LR) <- gsub(x = names(Results.LR), 
                                pattern = "LR_stat_LR_unifdif", 
                                replacement = "LR-Unif") 
  names(Results.LR) <- gsub(x = names(Results.LR), 
                                pattern = "LR_stat_LR_nonunifdif", 
                                replacement = "LR-Nunif") 
  names(Results.LR) <- gsub(x = names(Results.LR), 
                            pattern = "LR_stat_LR_both_dif", 
                            replacement = "LR-DIF") 
  names(Results.LR) <- gsub(x = names(Results.LR), 
                            pattern = "LR_df_LR_unifdif", 
                            replacement = "LR-Unif_df") 
  names(Results.LR) <- gsub(x = names(Results.LR), 
                            pattern = "LR_df_LR_nonunifdif", 
                            replacement = "LR-Nunif_df") 
  names(Results.LR) <- gsub(x = names(Results.LR), 
                            pattern = "LR_df_LR_both_dif", 
                            replacement = "LR-DIF_df") 
  
  out.ALL.features = cbind(Results.CSIB,
                       Results.D,
                       Results.LR,
                       #Results.LRT_nonuniform,
                       #Results.LRT_uniform,
                       Results.MH,
                       Results.SIB,
                       #Results.Wald_nonuniform,
                       #Results.Wald_uniform,
                       Results.VEMIRT)
  
  feature_names = colnames(out.ALL.features)
  J = nrow(out.ALL.features)
  out.ALL.features = as.matrix(out.ALL.features)
  out.ALL.features = as.numeric(out.ALL.features)
  out.ALL.features = matrix(out.ALL.features, nrow = J)
  colnames(out.ALL.features) = feature_names
  out.ALL.features = data.frame(out.ALL.features)
  
  out.TLP.features = Results.VEMIRT
  out.labels = as.data.frame(cbind(Labels_a,
                     Labels_b))
  
  write_csv(out.ALL.features, paste0("Four_Group_Training_data_ALL_Replication",training_file_counter,".csv"))
  write_csv(out.TLP.features, paste0("Four_Group_Training_data_TLP_Replication",training_file_counter,".csv"))
  write_csv(out.labels, paste0("Four_Group_Training_labels_Replication",training_file_counter,".csv"))

  training_file_counter <- training_file_counter + 1
}

remove(list = ls())

file_path <- "Estimated_Empirical_Testing_Parameters.Rdata"
if (!file.exists(file_path)) {
  next
}
load(file_path)
# Fix naming
Results.VEMIRT = as.data.frame(Results.VEMIRT)

names(Results.CSIB) <- gsub(x = names(Results.CSIB), 
                            pattern = "CSIB_Comparison", 
                            replacement = "CSIB") 
names(Results.D) <- gsub(x = names(Results.D), 
                         pattern = "D_Stat_Comparison.", 
                         replacement = "Standardized-D_") 
names(Results.SIB) <- gsub(x = names(Results.SIB), 
                           pattern = "SIB_Comparison", 
                           replacement = "SIB") 
names(Results.MH) <- gsub(x = names(Results.MH), 
                          pattern = "MH_Comparison", 
                          replacement = "Mantel-Hanzel") 
names(Results.MH) <- gsub(x = names(Results.MH), 
                          pattern = "MH", 
                          replacement = "Mantel-Hanzel") 
names(Results.VEMIRT) <- gsub(x = names(Results.VEMIRT), 
                              pattern = "VEMIRT_a", 
                              replacement = "TLP_a") 
names(Results.VEMIRT) <- gsub(x = names(Results.VEMIRT), 
                              pattern = "VEMIRT_b", 
                              replacement = "TLP_b") 
names(Results.VEMIRT) <- gsub(x = names(Results.VEMIRT), 
                              pattern = "d.a", 
                              replacement = "TLP_d.a") 
names(Results.VEMIRT) <- gsub(x = names(Results.VEMIRT), 
                              pattern = "d.b", 
                              replacement = "TLP_d.b") 
names(Results.LR) <- gsub(x = names(Results.LR), 
                          pattern = "LR_stat_LR_unifdif", 
                          replacement = "LR-Unif") 
names(Results.LR) <- gsub(x = names(Results.LR), 
                          pattern = "LR_stat_LR_nonunifdif", 
                          replacement = "LR-Nunif") 
names(Results.LR) <- gsub(x = names(Results.LR), 
                          pattern = "LR_stat_LR_both_dif", 
                          replacement = "LR-DIF") 
names(Results.LR) <- gsub(x = names(Results.LR), 
                          pattern = "LR_df_LR_unifdif", 
                          replacement = "LR-Unif_df") 
names(Results.LR) <- gsub(x = names(Results.LR), 
                          pattern = "LR_df_LR_nonunifdif", 
                          replacement = "LR-Nunif_df") 
names(Results.LR) <- gsub(x = names(Results.LR), 
                          pattern = "LR_df_LR_both_dif", 
                          replacement = "LR-DIF_df") 

# Fix column names
out.ALL.features = cbind(Results.CSIB,
                         Results.D,
                         Results.LR,
                         # Results.LRT_nonuniform,
                         # Results.LRT_uniform,
                         Results.MH,
                         Results.SIB,
                         # Results.Wald_nonuniform,
                         # Results.Wald_uniform,
                         Results.VEMIRT)

feature_names = colnames(out.ALL.features)
J = nrow(out.ALL.features)
out.ALL.features = as.matrix(out.ALL.features)
out.ALL.features = as.numeric(out.ALL.features)
out.ALL.features = matrix(out.ALL.features, nrow = J)
colnames(out.ALL.features) = feature_names
out.ALL.features = data.frame(out.ALL.features)

out.TLP.features = Results.VEMIRT

write_csv(out.ALL.features, paste0("Epirical_Testing_data_ALL.csv"))
write_csv(out.TLP.features, paste0("Epirical_Testing_data_TLP.csv"))





