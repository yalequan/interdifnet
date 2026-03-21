# InterDIFNet

A multi-label neural network approach for detecting small sample intersectional differential item functioning (DIF). This is the code repository needed to reproduce the results from Quan & Wang (2026) cited below.

## Citation

If you use InterDIFNet in your research, please cite:

```bibtex
@article{interdifnet,
  title= {Using multi-label classification neural networks to detect intersectional DIF},
  author= {Yale Quan and Chun Wang},
  journal={British Journal of Mathematical and Statistical Psychology},
  year= {In Press}
}
```
and 
```bibtex
@misc{https://doi.org/10.5281/zenodo.18626820,
  doi = {10.5281/ZENODO.18626820},
  url = {https://zenodo.org/doi/10.5281/zenodo.18626820},
  author = {Yale Quan,  },
  title = {yalequan/interdifnet: InterDIFNet Initial Release V1.0.0},
  publisher = {Zenodo},
  year = {2026},
  copyright = {GNU Affero General Public License v3.0 only}
}
```

## Overview

InterDIFNet addresses the challenge of detecting intersectional differential item functioning in small sample settings using deep learning. Traditional DIF detection methods often struggle with limited data and multiple intersecting demographic factors. This repository provides a neural network-based solution designed to handle these complexities.

## Repository Structure

```
InterDIFNet/
├── Four_Group_Datasets                           # Folder for storing training and testing data
├───── Four_Group_Data_Files_For_Python.R         # R script for generating CSV files for python
├───── Four_Group_Training_Data_Generation.R      # R script to generate training data
├───── Four_Group_Training_Data_Parameters.R      # R script to estmate the training data parameters
├── Thre_Group_Datasets                           # Folder for storing training and testing data
├───── Three_Group_Data_Files_For_Python.R        # R script for generating CSV files for python
├───── Three_Group_Training_Data_Generation.R     # R script to generate training data
├───── Three_Group_Training_Data_Parameters.R     # R script to estmate the training data parameters
├───── Three_Group_Testing_Data_Generation.R      # R script to generate testing data
├───── Three_Group_Testing_Data_Parameters.R      # R script to estmate the testing data parameters
├── Ten_Group_Datasets                            # Folder for storing training and testing data
├───── Ten_Group_Data_Files_For_Python.R          # R script for generating CSV files for python
├───── Ten_Group_Training_Data_Generation.R       # R script to generate training data
├───── Ten_Group_Training_Data_Parameters.R       # R script to estmate the training data parameters
├───── Ten_Group_Testing_Data_Generation.R        # R script to generate testing data
├───── Ten_Group_Testing_Data_Parameters.R        # R script to estmate the testing data parameters
├── Generic_training_code.R                       # R script to generate training features
├── interdifnet_preprocessing.R                   # R script to generate InterDIFNet data from user provided data
├── InterDIFNet Package Dependencies.md           # Markdown file explaining how package dependencies are handled
├── InterDIFNet.py                                # Python code for main InterDIFNet functions
├── InterDIFNet_Function_Calls.py                 # Python code with InterDIFNet function calls
└── README.md                                     # This file
```

## Requirements

See the InterDIFNet Package Dependencies.md file for detailed explanations

## Required Packages

The following packages are automatically managed:

| Package | PyPI Name | Import Name |
|---------|-----------|-------------|
| NumPy | `numpy` | `numpy` |
| Pandas | `pandas` | `pandas` |
| TensorFlow | `tensorflow` | `tensorflow` |
| Scikit-learn | `scikit-learn` | `sklearn` |
| Matplotlib | `matplotlib` | `matplotlib` |
| Seaborn | `seaborn` | `seaborn` |
| Scikit-multilearn | `scikit-multilearn` | `skmultilearn` |
| NetworkX | `networkx` | `networkx` |

## Simulation Study Workflow

**Note** When you download the InterDIFNet folder please respect the folder hierarchy. This current version is sensitive to folder naming.

1. **Download InterDIFNet**: Download the InterDIFNet folder to your computer. 
2. **Generate Training Data**: Create a large synthetic dataset with known DIF patterns.
  - Within the folder corresponding to the group setup you want to simulate.
    - Use the `_Training_Data_Parameters.R` script to estimate the features used to train the network.
    - Use the training data generation `_Training_Data_Generation.R` script within the training data folders.
3. **Generate Testing Data**: Create test datasets to evaluate model performance
  - Within the dataset folder
    - Use the `_Training_Data_Parameters.R` script to estimate the features used to train the network.
    - Use the `_Testing_Data_Generation.R` and `Testing_Data_Parameters.R` to generate and estimate the testing data features.
4. **Run Simulation**: Train the neural network and evaluate its ability to detect intersectional DIF
  - Within `InterDIFNet_Function_Calls.py` find the `train_InterDIFNet()` function that corresponds to the group size you generated data for
  - Then run the `Simulation_Study()` function
5. **Expected Output**: DIF Detection Results and Type 1 Error and Power from the simulation study

## Using Your Own Observed Data

In addition to simulation studies, users may preprocess real observed item response data for use with the InterDIFNet neural model.

### Feature preprocessing is handled by:

Feature preprocessing is handled by:  `interdifnet_preprocessing.R`

**Example Function Call**
```r
# Load feature generator
source("interdifnet_preprocessing.R")

# Load user datasets
item_matrix <- read.csv("responses.csv") # Item response matrix
group_data  <- read.csv("groups.csv") # Column denoting group membership
group_vector <- as.numeric(group_data[[1]])

# Generate TLP-based InterDIFNet features
result <- interdifnet_preprocessing(
  item_responses = item_matrix,
  group_assignments = group_vector,
  num_groups = 3,
  output_path = "interdifnet_features.csv",
  seed = 123
)

# Inspect feature matrix
head(result$features)
```

### Required Input Files

Two CSV files are required:

1. Binary Item Response File (responses.csv)
* Rows = examinees
* Columns = items


**Example:**

| Item1 | Item2 | Item3 |
|-------|-------|-------|
| 1     | 0     | 1     |
| 0     | 1     | 1     |
| 1     | 1     | 0     |

2. Group Assignment File (groups.csv)

Single column
* Integers denoting group membership
* Must match number of rows in response file

**Example (3-group model):** 

| group |
|-------|
| 1     |
| 2     |
| 1     |

Please check the models folder to determine the number of groups supported.

## DIF Detection

After generating the feature file, users can apply a trained InterDIFNet model using the DIF_Detection function inside `InterDIFNet.py`

**Example Usage**
```py
from InterDIFNet import DIF_Detection

DIF_Detection(
    data_filename="interdifnet_features.csv",
    model_name="InterDIFNet_Three_Group_model",
    verbose=True,
    save_results=True,
    output_filename="interdifnet_results.csv"
)
```

If only one trained model exists in `./models/`, `model_name` may be omitted.

**Full Workflow Summary (Observed Data)**

1) Prepare binary item response matrix
2) Prepare group assignment vector
3) Run `interdifnet_preprocessing()` in R
4) Run `DIF_Detection()` in Python
5) Interpret DIF classification results

## Method Details

InterDIFNet employs a multi-label neural network architecture to detectcintersectional DIF between many small groups simultaneously.

The approach is specifically designed for small sample scenarios where traditional methods may lack statistical power.

## License

This project is licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later).
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

## Contact

For questions or issues, please:
- Open an issue on GitHub
