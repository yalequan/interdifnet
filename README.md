# InterDIFNet

A multi-label neural network approach for detecting small sample intersectional differential item functioning (DIF).

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

## Usage

### 1. Generate Training Data

The training data generation scripts with each folder create training data with specific group numbers

### 2. Generate Testing Data

The testing data generation scripts with each folder create training data with specific group numbers

### 3. Run Simulation Study

Within the InterDIFNet_Function_Calls.py are the functions needed to run the simulation study

**Expected outputs:**
- Model performance metrics (Power and Type 1 Error)
- DIF detection results
- Visualization plots
- Summary statistics

## Workflow

1. **Generate Training Data**: Create a large synthetic dataset with known DIF patterns
2. **Generate Testing Data**: Create test datasets to evaluate model performance
3. **Run Simulation**: Train the neural network and evaluate its ability to detect intersectional DIF

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
