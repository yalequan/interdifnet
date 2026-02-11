#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 11:19:17 2025

@author: yalequan
"""

from InterDIFNet import Simulation_Study, train_InterDIFNet, DIF_Detection

# Train and Save Model for Ten Group Scenario
training_results = train_InterDIFNet(
    groups="Ten",
    feature_selection="TLP",
    merged=True,
    save_model=True,
    model_name= "InterDIFNet_Ten_Group",
    model_dir="./models/"
)

# Load the saved model and test
training_results, testing_results = Simulation_Study(
    groups="Ten",
    feature_selection="TLP",
    sizes=[1000, 2000, 4000],
    merged=True,
    load_existing=True,
    model_name= "InterDIFNet_Ten_Group",
    save_results=True, 
    model_dir="./models/"
    )

print("Ten Group Results")
print(testing_results)

# Train and Save Model for Three Group Scenario
training_results = train_InterDIFNet(
    groups="Three",
    feature_selection="TLP",
    merged=True,
    save_model=True,
    model_name= "InterDIFNet_Three_Group",
    model_dir="./models/"
)

# Load the saved model and test
training_results, testing_results = Simulation_Study(
    groups="Three",
    sizes=[250, 500, 1000],
    feature_selection="TLP",
    merged=True,
    load_existing=True,
    model_name= "InterDIFNet_Three_Group",
    save_results=True, 
    model_dir="./models/"
    )

print("Three Group Results")
print(testing_results)


# Empirical Example

for r in range(1, 100):
    print(f"\nEstimating Empirical Dataset Replication {r}")
    trained_empirical_network = train_InterDIFNet(groups = "Four",
                                                  feature_selection = "TLP",
                                                  merged = True,
                                                  save_model= True,
                                                  model_name= f"InterDIFNet_Empirical_Replication{r}",
                                                  model_dir= "./Empirical_models/")

    empirical_results = DIF_Detection(data_filename="Empirical_Testing_data.csv",
                                      model_name=f"InterDIFNet_Empirical_Replication{r}",
                                      save_results=True,
                                      output_filename=f"Empirical_Replication{r}",
                                      model_dir="./Empirical_models/")


