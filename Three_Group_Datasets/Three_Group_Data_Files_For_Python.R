#' Export Training and Testing Data to CSV Files for Python Analysis
#'
#' This script processes estimated IRT parameters and DIF test statistics from
#' RData files and exports them as CSV files suitable for deep learning analysis
#' in Python. It handles both training data (for model development) and testing
#' data (for model evaluation under varied conditions) for the three-group scenario.
#'
#' @section Overview:
#' The script performs the following operations:
#' 1. Loads estimated parameters and DIF statistics from RData files
#' 2. Standardizes column names across all DIF test methods
#' 3. Combines results into comprehensive feature matrices
#' 4. Exports three CSV files per replication:
#'    - ALL features: Complete set of IRT parameters and DIF statistics
#'    - TLP features: VEMIRT (penalized IRT) parameters only
#'    - Labels: Binary DIF indicators for training/validation
#'
#' @section Three-Group Context:
#' This script processes data from simulations with 3 comparison groups:
#' - Simpler scenario than 10-group case
#' - 3 groups = 3 pairwise comparisons (fewer than 45 in 10-group case)
#' - Same 10 items per dataset
#' - Useful for model validation and comparison studies
#'
#' @section Data Structure:
#' Each CSV file contains:
#' - **Rows**: 10 items
#' - **Columns**: Features from all DIF detection methods
#' - **Format**: Numeric values suitable for neural network input
#'
#' @section Feature Sets:
#' - **CSIB**: Crossing-SIB test statistics for non-uniform DIF
#' - **D-statistic**: Standardized effect size measures
#' - **LR**: Logistic regression (uniform, non-uniform, both types)
#' - **MH**: Mantel-Haenszel statistics
#' - **SIB**: Standardized item bias for uniform DIF
#' - **VEMIRT/TLP**: Penalized IRT parameter estimates and differences
#'
#' @section Name Standardization:
#' Column names are standardized for consistency:
#' - VEMIRT → TLP (Truncated Lasso Penalized IRT)
#' - MH_Comparison → Mantel-Hanzel
#' - LR_stat → LR-Unif/LR-Nunif/LR-DIF
#' - CSIB_Comparison → CSIB
#' - SIB_Comparison → SIB
#' - D_Stat_Comparison → Standardized-D
#'
#' @section Training Data Export:
#' - Source: Estimated_Three_Group_Training_data_Replication[1-500].RData
#' - Output files per replication:
#'   1. Three_Group_Training_data_ALL_Replication[n].csv
#'   2. Three_Group_Training_data_TLP_Replication[n].csv
#'   3. Three_Group_Training_labels_Replication[n].csv
#' - Purpose: Train InterDIFNet deep learning model
#'
#' @section Testing Data Export:
#' - Source: Estimated_Three_Group_Testing_data_[N]_[perc]_Replication[1-500].RData
#' - Conditions:
#'   - Sample sizes (N): 250, 500, 1000
#'   - DIF percentages (perc): 20%, 40%
#' - Output files per replication:
#'   1. Three_Group_Testing_data_ALL_[N]_[perc]_Replication[n].csv
#'   2. Three_Group_Testing_data_TLP_[N]_[perc]_Replication[n].csv
#'   3. Three_Group_Testing_data_labels_[N]_[perc]_Replication[n].csv
#' - Purpose: Evaluate model performance under varied conditions
#'
#' @section File Handling:
#' - Skips non-existent files (allows partial processing)
#' - Sequential counter for output file numbering
#' - Workspace cleared between training and testing sections
#'
#' @section Data Transformation:
#' 1. Convert results to data frame format
#' 2. Apply name standardization via gsub()
#' 3. Combine all features using cbind()
#' 4. Convert to matrix, flatten to numeric vector, reshape
#' 5. Restore column names and convert back to data frame
#' 6. Export using write_csv()
#'
#' @author Yale Quan
#' @date 2024-07-01
#'
#' @note
#' - Processing time: ~1-3 minutes for all files (faster than 10-group)
#' - Output directory: Same as input RData files
#' - Missing replications are skipped automatically
#'
#' @examples
#' # This script is designed to be run as a standalone file
#' # source("Three_Group_Data_Files_For_Python.R")

# Setup and Configuration ----

# Clear workspace
remove(list = ls())

# Load required packages
library(tidyverse)  # Data manipulation and CSV export
library(dplyr)      # Data frame operations

# Training Data Export ----
# Process and export training data for InterDIFNet model development

training_file_counter <- 1
for (rep in 1:500) {
  # Check if Training Data Exists ----
  file_path <- paste0("Estimated_Three_Group_Training_data_Replication", rep, ".RData")
  if (!file.exists(file_path)) {
    next
  }
  load(file_path)
  
  # Standardize Column Names ----
  # Convert VEMIRT results to data frame format
  Results.VEMIRT <- as.data.frame(Results.VEMIRT)
  
  # CSIB: Crossing-SIB test for non-uniform DIF
  names(Results.CSIB) <- gsub(
    x = names(Results.CSIB),
    pattern = "CSIB_Comparison",
    replacement = "CSIB"
  )
  
  # D-statistic: Standardized effect size
  names(Results.D) <- gsub(
    x = names(Results.D),
    pattern = "D_Stat_Comparison.",
    replacement = "Standardized-D_"
  )
  
  # SIB: Standardized item bias for uniform DIF
  names(Results.SIB) <- gsub(
    x = names(Results.SIB),
    pattern = "SIB_Comparison",
    replacement = "SIB"
  )
  
  # Mantel-Haenszel: Non-parametric DIF test
  names(Results.MH) <- gsub(
    x = names(Results.MH),
    pattern = "MH_Comparison",
    replacement = "Mantel-Hanzel"
  )
  names(Results.MH) <- gsub(
    x = names(Results.MH),
    pattern = "MH",
    replacement = "Mantel-Hanzel"
  )
  
  # VEMIRT/TLP: Penalized IRT parameter estimates
  names(Results.VEMIRT) <- gsub(
    x = names(Results.VEMIRT),
    pattern = "VEMIRT_a",
    replacement = "TLP_a"
  )
  names(Results.VEMIRT) <- gsub(
    x = names(Results.VEMIRT),
    pattern = "VEMIRT_b",
    replacement = "TLP_b"
  )
  names(Results.VEMIRT) <- gsub(
    x = names(Results.VEMIRT),
    pattern = "d.a",
    replacement = "TLP_d.a"
  )
  names(Results.VEMIRT) <- gsub(
    x = names(Results.VEMIRT),
    pattern = "d.b",
    replacement = "TLP_d.b"
  )
  
  # Logistic Regression: Uniform, non-uniform, and combined DIF tests
  names(Results.LR) <- gsub(
    x = names(Results.LR),
    pattern = "LR_stat_LR_unifdif",
    replacement = "LR-Unif"
  )
  names(Results.LR) <- gsub(
    x = names(Results.LR),
    pattern = "LR_stat_LR_nonunifdif",
    replacement = "LR-Nunif"
  )
  names(Results.LR) <- gsub(
    x = names(Results.LR),
    pattern = "LR_stat_LR_both_dif",
    replacement = "LR-DIF"
  )
  names(Results.LR) <- gsub(
    x = names(Results.LR),
    pattern = "LR_df_LR_unifdif",
    replacement = "LR-Unif_df"
  )
  names(Results.LR) <- gsub(
    x = names(Results.LR),
    pattern = "LR_df_LR_nonunifdif",
    replacement = "LR-Nunif_df"
  )
  names(Results.LR) <- gsub(
    x = names(Results.LR),
    pattern = "LR_df_LR_both_dif",
    replacement = "LR-DIF_df"
  )
  
  # Combine All Features ----
  # Create comprehensive feature matrix with all DIF test results
  out.ALL.features <- cbind(
    Results.CSIB,
    Results.D,
    Results.LR,
    Results.MH,
    Results.SIB,
    Results.VEMIRT
  )
  
  # Transform to Proper Format ----
  # Ensure data is in correct shape (10 items × features)
  feature_names <- colnames(out.ALL.features)
  out.ALL.features <- as.matrix(out.ALL.features)
  out.ALL.features <- as.numeric(out.ALL.features)
  out.ALL.features <- matrix(out.ALL.features, nrow = 10)
  out.ALL.features <- data.frame(out.ALL.features)
  colnames(out.ALL.features) <- feature_names
  
  # Extract TLP-only features and labels
  out.TLP.features <- Results.VEMIRT
  out.labels <- as.data.frame(cbind(Labels_a, Labels_b))
  
  # Export to CSV Files ----
  write_csv(
    out.ALL.features,
    paste0("Three_Group_Training_data_ALL_Replication", training_file_counter, ".csv")
  )
  write_csv(
    out.TLP.features,
    paste0("Three_Group_Training_data_TLP_Replication", training_file_counter, ".csv")
  )
  write_csv(
    out.labels,
    paste0("Three_Group_Training_labels_Replication", training_file_counter, ".csv")
  )
  
  training_file_counter <- training_file_counter + 1
}

# Clear Workspace Before Testing Data ----
remove(list = ls())

# Testing Data Export ----
# Process and export testing data for model evaluation under varied conditions

for (N in c(250, 500, 1000)) {  # Sample sizes (smaller than 10-group case)
  for (p in c(20, 40)) {  # DIF percentages
    testing_file_counter <- 1
    for (r in 1:500) {  # Replications
      # Check if Testing Data Exists ----
      file_path <- paste0(
        "Estimated_Three_Group_Testing_data_", N, "_", p, "_Replication", r, ".RData"
      )
      if (!file.exists(file_path)) {
        next
      }
      load(file_path)
      
      # Standardize Column Names ----
      # Convert VEMIRT results to data frame format
      Results.VEMIRT <- as.data.frame(Results.VEMIRT)
      
      # CSIB: Crossing-SIB test for non-uniform DIF
      names(Results.CSIB) <- gsub(
        x = names(Results.CSIB),
        pattern = "CSIB_Comparison",
        replacement = "CSIB"
      )
      
      # D-statistic: Standardized effect size
      names(Results.D) <- gsub(
        x = names(Results.D),
        pattern = "D_Stat_Comparison.",
        replacement = "Standardized-D_"
      )
      
      # SIB: Standardized item bias for uniform DIF
      names(Results.SIB) <- gsub(
        x = names(Results.SIB),
        pattern = "SIB_Comparison",
        replacement = "SIB"
      )
      
      # Mantel-Haenszel: Non-parametric DIF test
      names(Results.MH) <- gsub(
        x = names(Results.MH),
        pattern = "MH_Comparison",
        replacement = "Mantel-Hanzel"
      )
      names(Results.MH) <- gsub(
        x = names(Results.MH),
        pattern = "MH",
        replacement = "Mantel-Hanzel"
      )
      
      # VEMIRT/TLP: Penalized IRT parameter estimates
      names(Results.VEMIRT) <- gsub(
        x = names(Results.VEMIRT),
        pattern = "VEMIRT_a",
        replacement = "TLP_a"
      )
      names(Results.VEMIRT) <- gsub(
        x = names(Results.VEMIRT),
        pattern = "VEMIRT_b",
        replacement = "TLP_b"
      )
      names(Results.VEMIRT) <- gsub(
        x = names(Results.VEMIRT),
        pattern = "d.a",
        replacement = "TLP_d.a"
      )
      names(Results.VEMIRT) <- gsub(
        x = names(Results.VEMIRT),
        pattern = "d.b",
        replacement = "TLP_d.b"
      )
      
      # Logistic Regression: Uniform, non-uniform, and combined DIF tests
      names(Results.LR) <- gsub(
        x = names(Results.LR),
        pattern = "LR_stat_LR_unifdif",
        replacement = "LR-Unif"
      )
      names(Results.LR) <- gsub(
        x = names(Results.LR),
        pattern = "LR_stat_LR_nonunifdif",
        replacement = "LR-Nunif"
      )
      names(Results.LR) <- gsub(
        x = names(Results.LR),
        pattern = "LR_stat_LR_both_dif",
        replacement = "LR-DIF"
      )
      names(Results.LR) <- gsub(
        x = names(Results.LR),
        pattern = "LR_df_LR_unifdif",
        replacement = "LR-Unif_df"
      )
      names(Results.LR) <- gsub(
        x = names(Results.LR),
        pattern = "LR_df_LR_nonunifdif",
        replacement = "LR-Nunif_df"
      )
      names(Results.LR) <- gsub(
        x = names(Results.LR),
        pattern = "LR_df_LR_both_dif",
        replacement = "LR-DIF_df"
      )
      
      # Combine All Features ----
      # Create comprehensive feature matrix with all DIF test results
      out.ALL.features <- cbind(
        Results.CSIB,
        Results.D,
        Results.LR,
        Results.MH,
        Results.SIB,
        Results.VEMIRT
      )
      
      # Transform to Proper Format ----
      # Ensure data is in correct shape (10 items × features)
      feature_names <- colnames(out.ALL.features)
      out.ALL.features <- as.matrix(out.ALL.features)
      out.ALL.features <- as.numeric(out.ALL.features)
      out.ALL.features <- matrix(out.ALL.features, nrow = 10)
      out.ALL.features <- data.frame(out.ALL.features)
      colnames(out.ALL.features) <- feature_names
      
      # Extract TLP-only features and labels
      out.TLP.features <- Results.VEMIRT
      out.labels <- as.data.frame(cbind(Labels_a, Labels_b))
      
      # Export to CSV Files ----
      write_csv(
        out.ALL.features,
        paste0(
          "Three_Group_Testing_data_ALL_", N, "_", p,
          "_Replication", testing_file_counter, ".csv"
        )
      )
      write_csv(
        out.TLP.features,
        paste0(
          "Three_Group_Testing_data_TLP_", N, "_", p,
          "_Replication", testing_file_counter, ".csv"
        )
      )
      write_csv(
        out.labels,
        paste0(
          "Three_Group_Testing_data_labels_", N, "_", p,
          "_Replication", testing_file_counter, ".csv"
        )
      )
      
      testing_file_counter <- testing_file_counter + 1
    }
  }
}

# End of script


