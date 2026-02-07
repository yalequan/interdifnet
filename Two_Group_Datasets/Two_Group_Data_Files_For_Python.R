#' Two-Group Data Files Export for Python/Deep Learning
#'
#' @description
#' This script converts estimated R data files to CSV format for use with
#' Python-based deep learning frameworks (PyTorch, TensorFlow). It processes
#' both training and testing data from the two-group scenario of the InterDIFNet
#' study, creating three types of CSV files per replication: ALL features,
#' TLP features only, and binary DIF labels.
#'
#' @section Data Processing Pipeline:
#' 1. **Training Data**: Processes replications 1-500 (subset exists)
#'    - Randomized conditions: N, m (DIF items), DIF type
#'    - Used for training deep learning models
#'    - Variable experimental conditions across replications
#' 
#' 2. **Testing Data**: Processes all N × DIF% × replication combinations
#'    - Sample Sizes: N = 125, 250, 500
#'    - DIF Percentages: 20% (2/10 items), 40% (4/10 items)
#'    - Replications: Up to 500 per condition
#'    - Fixed conditions for systematic evaluation
#'
#' @section Feature Name Standardization:
#' Original VEMIRT output is renamed to "TLP" (Truncated Lasso Penalty) for
#' consistency across all scenario files:
#' - `VEMIRT_a_*` → `TLP_a_*` (discrimination parameters)
#' - `VEMIRT_b_*` → `TLP_b_*` (difficulty parameters)
#' - `d.a_*` → `TLP_d.a_*` (discrimination differences)
#' - `d.b_*` → `TLP_d.b_*` (difficulty differences)
#' 
#' Other test statistics are also standardized:
#' - `MH_Comparison_*` → `Mantel-Hanzel_*`
#' - `D_Stat_Comparison_*` → `Standardized-D_*`
#' - `LR_stat_LR_unifdif_*` → `LR-Unif_*`
#' - `LR_stat_LR_nonunifdif_*` → `LR-Nunif_*`
#' - `LR_stat_LR_both_dif_*` → `LR-DIF_*`
#' - `CSIB_Comparison_*` → `CSIB_*`
#' - `SIB_Comparison_*` → `SIB_*`
#'
#' @section Output Files:
#' For each replication, three CSV files are created:
#' 
#' 1. **ALL Features** (`*_ALL_*.csv`):
#'    - Complete feature set from all DIF tests
#'    - Includes: CSIB, D-stat, LR (3 types), MH, SIB, VEMIRT/TLP
#'    - Dimensions: 10 items × ~14 features = ~140-dimensional vectors
#'    - One pairwise comparison (Group1 vs Group2)
#'    - For comprehensive model training
#' 
#' 2. **TLP Features** (`*_TLP_*.csv`):
#'    - Only VEMIRT/TLP parameters and differences
#'    - Includes: TLP_a, TLP_b, TLP_d.a, TLP_d.b for 2 groups
#'    - Dimensions: 10 items × 6 features = 60-dimensional vectors
#'    - For isolated evaluation of penalized IRT estimation
#' 
#' 3. **Labels** (`*_labels_*.csv`):
#'    - Binary ground truth DIF indicators
#'    - `Labels_a`: a-DIF (discrimination) presence
#'    - `Labels_b`: b-DIF (difficulty) presence
#'    - Dimensions: 10 items × 2 label types
#'    - For supervised learning and performance evaluation
#'
#' @section File Naming Conventions:
#' Training files:
#' - `Two_Group_Training_data_ALL_Replication{counter}.csv`
#' - `Two_Group_Training_data_TLP_Replication{counter}.csv`
#' - `Two_Group_Training_labels_Replication{counter}.csv`
#' 
#' Testing files:
#' - `Two_Group_Testing_data_ALL_{N}_{perc}_Replication{counter}.csv`
#' - `Two_Group_Testing_data_TLP_{N}_{perc}_Replication{counter}.csv`
#' - `Two_Group_Testing_data_labels_{N}_{perc}_Replication{counter}.csv`
#' 
#' Counter increments only for existing files (handles missing replications).
#'
#' @section Workflow:
#' **Training Data Processing**:
#' 1. Loop through replications 1-500
#' 2. Check if estimated RData file exists (skip if not)
#' 3. Load estimated parameters and DIF statistics
#' 4. Apply name standardization transformations
#' 5. Combine features into ALL and TLP datasets
#' 6. Reshape to ensure correct dimensions (10 items)
#' 7. Extract binary labels
#' 8. Write three CSV files
#' 9. Increment counter
#' 
#' **Testing Data Processing**:
#' 1. Triple nested loop: N sizes × DIF percentages × replications
#' 2. Same processing steps as training data
#' 3. Separate file naming by condition (N, perc)
#' 4. Independent counter per condition combination
#'
#' @section Data Structure:
#' Each CSV file has 10 rows (items) and varying columns:
#' - ALL features: ~14 features per item (one comparison)
#' - TLP features: 6 features per item (2 groups × 2 params + 1 diff × 2 params)
#' - Labels: 2 binary indicators per item (a-DIF, b-DIF)
#' 
#' All values are numeric, ready for direct loading in Python.
#'
#' @section Python Integration:
#' Output CSV files are designed for seamless Python integration:
#' ```python
#' import pandas as pd
#' import numpy as np
#' 
#' # Load features and labels
#' X_all = pd.read_csv("Two_Group_Training_data_ALL_Replication1.csv")
#' X_tlp = pd.read_csv("Two_Group_Training_data_TLP_Replication1.csv")
#' y = pd.read_csv("Two_Group_Training_labels_Replication1.csv")
#' 
#' # Convert to numpy arrays for PyTorch/TensorFlow
#' X_all_np = X_all.values
#' X_tlp_np = X_tlp.values
#' y_np = y.values
#' ```
#'
#' @author Yale Quan
#' @date February 2026
#'
#' @examples
#' # This script is designed to be run as a standalone batch process:
#' # source("Two_Group_Data_Files_For_Python.R")
#' 
#' # Output files for replication 1, N=250, 20% DIF:
#' # "Two_Group_Testing_data_ALL_250_20_Replication1.csv"
#' # "Two_Group_Testing_data_TLP_250_20_Replication1.csv"
#' # "Two_Group_Testing_data_labels_250_20_Replication1.csv"

# Setup ----
remove(list = ls())
library(tidyverse)
library(dplyr)

# Process Training Data ----
training_file_counter <- 1 
for (rep in 1:500) {
  
  # Check if training data exists ----
  file_path <- paste0('Estimated_Training_data_Replication', rep, '.RData')
  if (!file.exists(file_path)) {
    next
  }
  load(file_path)
  
  # Standardize Feature Names ----
  Results.VEMIRT <- as.data.frame(Results.VEMIRT)
  
  # CSIB naming
  names(Results.CSIB) <- gsub(x = names(Results.CSIB), 
                              pattern = "CSIB_Comparison", 
                              replacement = "CSIB")
  
  # D-statistic naming
  names(Results.D) <- gsub(x = names(Results.D), 
                           pattern = "D_Stat_Comparison.", 
                           replacement = "Standardized-D_")
  
  # SIB naming
  names(Results.SIB) <- gsub(x = names(Results.SIB), 
                             pattern = "SIB_Comparison", 
                             replacement = "SIB")
  
  # Mantel-Haenszel naming
  names(Results.MH) <- gsub(x = names(Results.MH), 
                            pattern = "MH_Comparison", 
                            replacement = "Mantel-Hanzel")
  names(Results.MH) <- gsub(x = names(Results.MH), 
                            pattern = "MH", 
                            replacement = "Mantel-Hanzel")
  
  # VEMIRT → TLP naming (discrimination parameters)
  names(Results.VEMIRT) <- gsub(x = names(Results.VEMIRT), 
                                pattern = "VEMIRT_a", 
                                replacement = "TLP_a")
  
  # VEMIRT → TLP naming (difficulty parameters)
  names(Results.VEMIRT) <- gsub(x = names(Results.VEMIRT), 
                                pattern = "VEMIRT_b", 
                                replacement = "TLP_b")
  
  # Parameter differences naming
  names(Results.VEMIRT) <- gsub(x = names(Results.VEMIRT), 
                                pattern = "d.a", 
                                replacement = "TLP_d.a")
  names(Results.VEMIRT) <- gsub(x = names(Results.VEMIRT), 
                                pattern = "d.b", 
                                replacement = "TLP_d.b")
  
  # Logistic Regression statistics naming
  names(Results.LR) <- gsub(x = names(Results.LR), 
                            pattern = "LR_stat_LR_unifdif", 
                            replacement = "LR-Unif")
  names(Results.LR) <- gsub(x = names(Results.LR), 
                            pattern = "LR_stat_LR_nonunifdif", 
                            replacement = "LR-Nunif")
  names(Results.LR) <- gsub(x = names(Results.LR), 
                            pattern = "LR_stat_LR_both_dif", 
                            replacement = "LR-DIF")
  
  # Logistic Regression degrees of freedom naming
  names(Results.LR) <- gsub(x = names(Results.LR), 
                            pattern = "LR_df_LR_unifdif", 
                            replacement = "LR-Unif_df")
  names(Results.LR) <- gsub(x = names(Results.LR), 
                            pattern = "LR_df_LR_nonunifdif", 
                            replacement = "LR-Nunif_df")
  names(Results.LR) <- gsub(x = names(Results.LR), 
                            pattern = "LR_df_LR_both_dif", 
                            replacement = "LR-DIF_df")
  
  # Combine All Features ----
  out.ALL.features <- cbind(Results.CSIB,
                            Results.D,
                            Results.LR,
                            Results.MH,
                            Results.SIB,
                            Results.VEMIRT)
  
  # Reshape to ensure correct dimensions ----
  feature_names <- colnames(out.ALL.features)
  out.ALL.features <- as.matrix(out.ALL.features)
  out.ALL.features <- as.numeric(out.ALL.features)
  out.ALL.features <- matrix(out.ALL.features, nrow = 10)
  out.ALL.features <- data.frame(out.ALL.features)
  colnames(out.ALL.features) <- feature_names
  
  # Extract TLP features and labels ----
  out.TLP.features <- Results.VEMIRT
  out.labels <- as.data.frame(cbind(Labels_a, Labels_b))
  
  # Write CSV files ----
  write_csv(out.ALL.features, 
            paste0("Two_Group_Training_data_ALL_Replication", training_file_counter, ".csv"))
  write_csv(out.TLP.features, 
            paste0("Two_Group_Training_data_TLP_Replication", training_file_counter, ".csv"))
  write_csv(out.labels, 
            paste0("Two_Group_Training_labels_Replication", training_file_counter, ".csv"))
  
  training_file_counter <- training_file_counter + 1
}

# Clear workspace before testing data ----
remove(list = ls())

# Process Testing Data ----
for (N in c(125, 250, 500)) {
  for (p in c(20, 40)) {
    testing_file_counter <- 1
    
    for (r in 1:500) {
      
      # Check if testing data exists ----
      file_path <- paste0('Estimated_Two_Group_Testing_data_', N, '_', p, 
                         '_Replication', r, '.RData')
      if (!file.exists(file_path)) {
        next
      }
      load(file_path)
      
      # Standardize Feature Names ----
      Results.VEMIRT <- as.data.frame(Results.VEMIRT)
      
      # CSIB naming
      names(Results.CSIB) <- gsub(x = names(Results.CSIB), 
                                  pattern = "CSIB_Comparison", 
                                  replacement = "CSIB")
      
      # D-statistic naming
      names(Results.D) <- gsub(x = names(Results.D), 
                               pattern = "D_Stat_Comparison.", 
                               replacement = "Standardized-D_")
      
      # SIB naming
      names(Results.SIB) <- gsub(x = names(Results.SIB), 
                                 pattern = "SIB_Comparison", 
                                 replacement = "SIB")
      
      # Mantel-Haenszel naming
      names(Results.MH) <- gsub(x = names(Results.MH), 
                                pattern = "MH_Comparison", 
                                replacement = "Mantel-Hanzel")
      names(Results.MH) <- gsub(x = names(Results.MH), 
                                pattern = "MH", 
                                replacement = "Mantel-Hanzel")
      
      # VEMIRT → TLP naming (discrimination parameters)
      names(Results.VEMIRT) <- gsub(x = names(Results.VEMIRT), 
                                    pattern = "VEMIRT_a", 
                                    replacement = "TLP_a")
      
      # VEMIRT → TLP naming (difficulty parameters)
      names(Results.VEMIRT) <- gsub(x = names(Results.VEMIRT), 
                                    pattern = "VEMIRT_b", 
                                    replacement = "TLP_b")
      
      # Parameter differences naming
      names(Results.VEMIRT) <- gsub(x = names(Results.VEMIRT), 
                                    pattern = "d.a", 
                                    replacement = "TLP_d.a")
      names(Results.VEMIRT) <- gsub(x = names(Results.VEMIRT), 
                                    pattern = "d.b", 
                                    replacement = "TLP_d.b")
      
      # Logistic Regression statistics naming
      names(Results.LR) <- gsub(x = names(Results.LR), 
                                pattern = "LR_stat_LR_unifdif", 
                                replacement = "LR-Unif")
      names(Results.LR) <- gsub(x = names(Results.LR), 
                                pattern = "LR_stat_LR_nonunifdif", 
                                replacement = "LR-Nunif")
      names(Results.LR) <- gsub(x = names(Results.LR), 
                                pattern = "LR_stat_LR_both_dif", 
                                replacement = "LR-DIF")
      
      # Logistic Regression degrees of freedom naming
      names(Results.LR) <- gsub(x = names(Results.LR), 
                                pattern = "LR_df_LR_unifdif", 
                                replacement = "LR-Unif_df")
      names(Results.LR) <- gsub(x = names(Results.LR), 
                                pattern = "LR_df_LR_nonunifdif", 
                                replacement = "LR-Nunif_df")
      names(Results.LR) <- gsub(x = names(Results.LR), 
                                pattern = "LR_df_LR_both_dif", 
                                replacement = "LR-DIF_df")
      
      # Combine All Features ----
      out.ALL.features <- cbind(Results.CSIB,
                                Results.D,
                                Results.LR,
                                Results.MH,
                                Results.SIB,
                                Results.VEMIRT)
      
      # Reshape to ensure correct dimensions ----
      feature_names <- colnames(out.ALL.features)
      out.ALL.features <- as.matrix(out.ALL.features)
      out.ALL.features <- as.numeric(out.ALL.features)
      out.ALL.features <- matrix(out.ALL.features, nrow = 10)
      out.ALL.features <- data.frame(out.ALL.features)
      colnames(out.ALL.features) <- feature_names
      
      # Extract TLP features and labels ----
      out.TLP.features <- Results.VEMIRT
      out.labels <- as.data.frame(cbind(Labels_a, Labels_b))
      
      # Write CSV files ----
      write_csv(out.ALL.features, 
                paste0("Two_Group_Testing_data_ALL_", N, "_", p, "_Replication", 
                      testing_file_counter, ".csv"))
      write_csv(out.TLP.features, 
                paste0("Two_Group_Testing_data_TLP_", N, "_", p, "_Replication", 
                      testing_file_counter, ".csv"))
      write_csv(out.labels, 
                paste0("Two_Group_Testing_data_labels_", N, "_", p, "_Replication", 
                      testing_file_counter, ".csv"))
      
      testing_file_counter <- testing_file_counter + 1
    }
  }
}


