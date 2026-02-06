#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIF Model Utilities
Functions for training, evaluating, and applying DIF detection models.

Created on Mon Dec 23 18:29:29 2024
@author: yalequan
"""

import numpy as np
import pandas as pd
import glob
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from skmultilearn.model_selection import iterative_train_test_split
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.metrics import AUC
from sklearn.metrics import roc_curve
from sklearn.utils import shuffle
from itertools import combinations
import seaborn as sns
import tensorflow.keras.backend as K
import gc
import tensorflow as tf
import re
import networkx as nx

#%% Loss Function

def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = K.pow((1 - p_t), gamma)
        return -K.mean(alpha_factor * modulating_factor * K.log(p_t))
    return loss


#%% Model Creation
def build_separate_models(groups, n_inputs, n_outputs_per_set, loss):
    """
    Build completely separate models for each label set.
    
    Parameters:
    -----------
    groups : str
        String indicating number of groups (e.g., "Ten", "Three", "Two")
    n_inputs : int
        Number of input features
    n_outputs_per_set : int
        Number of outputs per model
    loss : str
        Loss function ("focal_loss", "binary_crossentropy")
    
    Returns:
    --------
    tuple
        (model_set1, model_set2) - Two compiled Keras models
    """
    if loss == "focal_loss":
        def create_single_model(name_suffix=""):
            model = Sequential(name=f"DIF_model_{name_suffix}")
            model.add(Input(shape=(n_inputs,)))
            
            model.add(Dense(
                512, 
                activation='relu',
                kernel_initializer='he_uniform',
                kernel_regularizer=l2(1e-4)
            ))
            model.add(BatchNormalization())
            model.add(Dropout(0.4))
            
            model.add(Dense(
                256,
                activation='relu',
                kernel_initializer='he_uniform',
                kernel_regularizer=l2(1e-4)
            ))
            model.add(BatchNormalization())
            model.add(Dropout(0.4))
            
            model.add(Dense(
                128,
                activation='relu',
                kernel_initializer='he_uniform',
                kernel_regularizer=l2(1e-4)
            ))
            model.add(BatchNormalization())
            model.add(Dropout(0.4))
            
            model.add(Dense(
                n_outputs_per_set, 
                activation='sigmoid'
            ))
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss=focal_loss(gamma=2., alpha=0.25),               
                metrics=['accuracy', AUC(name='auc')]
            )
            return model
        
        # Build separate models for each DIF set
        model_set1 = create_single_model("set1")
        model_set2 = create_single_model("set2")
        return model_set1, model_set2
    
    elif loss == "binary_crossentropy":
        def create_single_model(name_suffix=""):
            model = Sequential(name=f"DIF_model_{name_suffix}")
            model.add(Input(shape=(n_inputs,)))
            
            model.add(Dense(
                512, 
                activation='relu',
                kernel_initializer='he_uniform',
                kernel_regularizer=l2(1e-4)
            ))
            model.add(BatchNormalization())
            model.add(Dropout(0.4))
            
            model.add(Dense(
                256,
                activation='relu',
                kernel_initializer='he_uniform',
                kernel_regularizer=l2(1e-4)
            ))
            model.add(BatchNormalization())
            model.add(Dropout(0.4))
            
            model.add(Dense(
                128,
                activation='relu',
                kernel_initializer='he_uniform',
                kernel_regularizer=l2(1e-4)
            ))
            model.add(BatchNormalization())
            model.add(Dropout(0.4))
            
            model.add(Dense(
                n_outputs_per_set, 
                activation='sigmoid'
            ))
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', AUC(name='auc')]
            )
            return model
        
        # Build separate models for each DIF set
        model_set1 = create_single_model("set1")
        model_set2 = create_single_model("set2")
        return model_set1, model_set2

def build_merged_model(groups, n_inputs, n_outputs_per_set, loss):
    """
    Build a single model with two heads for both label sets.
    
    Parameters:
    -----------
    groups : str
        String indicating number of groups (e.g., "Ten", "Three", "Two")
    n_inputs : int
        Number of input features
    n_outputs_per_set : int
        Number of outputs per model head
    loss : str
        Loss function ("focal_loss", "binary_crossentropy")
    
    Returns:
    --------
    Model
        Single compiled Keras model with two outputs
    """
    
    # Input layer
    inputs = Input(shape=(n_inputs,), name='input')
    
    # Shared feature extraction layers
    x = Dense(
        512, 
        activation='relu',
        kernel_initializer='he_uniform',
        kernel_regularizer=l2(1e-4),
        name='shared_dense_1'
    )(inputs)
    x = BatchNormalization(name='shared_bn_1')(x)
    x = Dropout(0.4, name='shared_dropout_1')(x)
    
    x = Dense(
        256,
        activation='relu',
        kernel_initializer='he_uniform',
        kernel_regularizer=l2(1e-4),
        name='shared_dense_2'
    )(x)
    x = BatchNormalization(name='shared_bn_2')(x)
    x = Dropout(0.4, name='shared_dropout_2')(x)
    
    x = Dense(
        128,
        activation='relu',
        kernel_initializer='he_uniform',
        kernel_regularizer=l2(1e-4),
        name='shared_dense_3'
    )(x)
    shared_features = BatchNormalization(name='shared_bn_3')(x)
    shared_features = Dropout(0.4, name='shared_dropout_3')(shared_features)
    
    # Head 1 for set 1
    output_a = Dense(
        n_outputs_per_set, 
        activation='sigmoid',
        name='output_set1'
    )(shared_features)
    
    # Head 2 for set 2  
    output_b = Dense(
        n_outputs_per_set, 
        activation='sigmoid',
        name='output_set2'
    )(shared_features)
    
    # Create the model
    model = Model(
        inputs=inputs, 
        outputs=[output_a, output_b], 
        name=f"DIF_merged_model_{groups}"
    )
    
    # Compile based on loss function
    if loss == "focal_loss":
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'output_set1': focal_loss(gamma=2., alpha=0.25),
                'output_set2': focal_loss(gamma=2., alpha=0.25)
            },
            metrics={
                'output_set1': ['accuracy', AUC(name='auc_set1')],
                'output_set2': ['accuracy', AUC(name='auc_set2')]
            }
        )
    elif loss == "binary_crossentropy":
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'output_set1': 'binary_crossentropy',
                'output_set2': 'binary_crossentropy'
            },
            metrics={
                'output_set1': ['accuracy', AUC(name='auc_set1')],
                'output_set2': ['accuracy', AUC(name='auc_set2')]
            }
        )
    
    return model

#%% Data Pre-Processing
def clean_data(data, labels):
    """
    Clean data by removing NaN and infinite values.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    labels : pd.DataFrame
        Input labels
    
    Returns:
    --------
    tuple
        (cleaned_data, cleaned_labels)
    """
    # Drop anchor items (NaN values)
    nan_indices = data[data.isna().any(axis=1)].index
    data = data.drop(index=nan_indices)
    labels = labels.drop(index=nan_indices)

    # Fix +inf and -inf values
    data = data.apply(lambda col: col.apply(lambda x: 1e10 if x == np.inf else (-1e10 if x == -np.inf else x)))
    
    return data, labels

def prepare_label_sets(training_labels, merged=False):
    """
    Helper function to split labels into DIF_a and DIF_b sets.
    
    Parameters:
    -----------
    training_labels : pd.DataFrame
        DataFrame containing all labels
    merged : bool, optional
        If True, returns labels in dictionary format for merged model
    
    Returns:
    --------
    tuple or dict
        If merged=False: (y_set1, y_set2, set1_cols, set2_cols)
        If merged=True: Returns same tuple plus a dictionary format
    """
    all_columns = training_labels.columns.tolist()
    
    # Split by DIF_a and DIF_b patterns
    set1_cols = [col for col in all_columns if 'DIF_a' in col]
    set2_cols = [col for col in all_columns if 'DIF_b' in col]
    
    y_set1 = training_labels[set1_cols].values
    y_set2 = training_labels[set2_cols].values
    
    if merged:
        y_dict = {
            'output_set1': y_set1,
            'output_set2': y_set2
        }
        return y_set1, y_set2, set1_cols, set2_cols, y_dict
    else:
        return y_set1, y_set2, set1_cols, set2_cols


def load_training_data(groups, training_features=None, replications=500, 
                       data_pattern="Training_data_ALL_Replication*.csv", 
                       labels_pattern="Training_labels_Replication*.csv",
                       merged=False):
    """
    Load and preprocess training data and labels.
    
    Parameters:
    -----------
    groups : str
        String indicating number of groups (e.g., "Ten", "Three", "Two")
    training_features : str
        String or index indicating which features to load. 
        default to all features
    replications : int
        Number of replications to load (default: 250)
    data_pattern : str
        Glob pattern for training data files
    labels_pattern : str
        Glob pattern for training labels files
    merged : bool
        If True, creates a dict for y to be used in the merged models
    
    Returns:
    --------
    tuple
        (training_data, training_labels, feature_names, DIF_tests)
    """
    
    # Load training data
    training_csv_files = sorted(glob.glob(f"{groups}_Group_{data_pattern}"), 
                               key=lambda x: int(x.split('Replication')[1].split('.')[0]))
    training_csv_files = training_csv_files[:replications]
    training_data = pd.concat((pd.read_csv(file) for file in training_csv_files), ignore_index=True)

    # Load training labels
    training_labels_csv_files = sorted(glob.glob(f"{groups}_Group_{labels_pattern}"), 
                                     key=lambda x: int(x.split('Replication')[1].split('.')[0]))
    training_labels_csv_files = training_labels_csv_files[:replications]
    training_labels = pd.concat((pd.read_csv(file) for file in training_labels_csv_files), ignore_index=True)

    # Clean data
    training_data, training_labels = clean_data(training_data, training_labels)
    
    # Subset training data if needed
    if isinstance(training_features, str):
        training_data = training_data.filter(like=training_features, axis=1)
       
    elif training_features is None:
        training_data = training_data.copy()
    
    else:
        training_data = training_data[training_features]
    
    # Get DIF Tests in the training data
    DIF_tests = [p for p in training_data.columns.str.extract(r'^([^_]+)_')[0].unique() if pd.notna(p)]
    feature_names = training_data.columns
    
    temp_cols = [col for col in training_labels.columns if 'DIF_b' in col]
    
    if len(temp_cols) > 1:
        # For multiple group comparisons (e.g., Ten groups = 45 pairwise comparisons)
        group_ids = set()
        for label in temp_cols:
            pair = label.split('DIF_b_')[1]  # e.g., 'Group1Group2'
            # Extract individual group numbers by splitting on 'Group' and filtering out empty strings
            for g in pair.split('Group'):
                if g:
                    group_ids.add(g)
        num_groups = len(group_ids)
    else:
        # For simple two-group comparison
        num_groups = 2
    
    print(f"Number of groups detected in training dataset: {num_groups}")
    
    return training_data, training_labels, feature_names, DIF_tests

def evaluate_split_ratios(
    training_data,
    training_labels,
    groups,
    loss,
    split_ratios=[0.5, 0.4, 0.3, 0.2, 0.1],
    split_labels=["50/50", "60/40", "70/30", "80/20", "90/10"],
    n_repeats=10,
    verbose=False,
    plotting=True,
    merged=False
):
    """
    Evaluate macro-F1 performance across different train/validation splits using repeated stratified sampling.

    Parameters
    ----------
    training_data : pd.DataFrame
        Feature matrix.
    training_labels : pd.DataFrame
        Multi-label binary target matrix.
    groups : str
        String indicating number of groups (e.g., "Ten", "Three", "Two")
    split_ratios : list of float, optional
        List of validation set proportions to test. Default is [0.5, 0.4, 0.3, 0.2, 0.1].
    split_labels : list of str, optional
        Human-readable labels for each split ratio.
    n_repeats : int, optional
        Number of random splits per ratio to average over. Default is 10.
    verbose : bool, optional
        Whether to print progress information.
    plotting : bool, optional
        Whether to print plots
    merged : bool, optional
        If True, uses merged model architecture. If False, uses separate models.

    Returns
    -------
    perf_df : pd.DataFrame
        Summary dataframe with mean and std macro-F1 scores across splits.
    best_split_avg : float
        Value for splitting data
    """
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=25,
        restore_best_weights=True,
        verbose=0,
        start_from_epoch=20,
        mode='min'
    )

    performance_summary = []

    X_np = training_data.to_numpy()
    Y_np = training_labels.to_numpy()
    
    for test_size, label in zip(split_ratios, split_labels):
        if verbose:
            print(f"\n=== Evaluating {label} Split ===")
        
        f1_scores_a = []
        f1_scores_b = []
    
        for i in range(n_repeats):
            if verbose:
                print(f"  Repetition {i+1}/{n_repeats}")
                
            X_np, Y_np = shuffle(X_np, Y_np, random_state=i)

            X_train, y_train_combined, X_val, y_val_combined = iterative_train_test_split(X_np, Y_np, test_size=test_size)
    
            y_train_df = pd.DataFrame(y_train_combined, columns=training_labels.columns)
            y_val_df = pd.DataFrame(y_val_combined, columns=training_labels.columns)
    
            # Prepare label sets based on merged flag
            if merged:
                y_train_set1, y_train_set2, set1_cols, set2_cols, y_train_dict = prepare_label_sets(y_train_df, merged=True)
                y_val_DIF_a, y_val_DIF_b, _, _, y_val_dict = prepare_label_sets(y_val_df, merged=True)
            else:
                y_train_set1, y_train_set2, set1_cols, set2_cols = prepare_label_sets(y_train_df, merged=False)
                y_val_DIF_a, y_val_DIF_b, _, _ = prepare_label_sets(y_val_df, merged=False)
    
            scaler = StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_val_scaled = scaler.transform(X_val)
    
            if merged:
                # Use merged model
                model = build_merged_model(groups=groups, 
                                         n_inputs=X_train.shape[1], 
                                         n_outputs_per_set=y_val_DIF_a.shape[1],
                                         loss=loss)
                
                # Train merged model
                model.fit(X_train_scaled, y_train_dict,
                         validation_data=(X_val_scaled, y_val_dict),
                         epochs=100, batch_size=32,
                         callbacks=[early_stopping], verbose=0)
                
                # Get predictions from merged model
                predictions = model.predict(X_val_scaled, verbose=0)
                val_pred_a = predictions[0]  # output_set1
                val_pred_b = predictions[1]  # output_set2
                
            else:
                # Use separate models
                model_dif_a, model_dif_b = build_separate_models(groups=groups, 
                                                               n_inputs=X_train.shape[1], 
                                                               n_outputs_per_set=y_val_DIF_a.shape[1],
                                                               loss=loss)
        
                model_dif_a.fit(X_train_scaled, y_train_set1,
                               validation_data=(X_val_scaled, y_val_DIF_a),
                               epochs=100, batch_size=32,
                               callbacks=[early_stopping], verbose=0)
        
                model_dif_b.fit(X_train_scaled, y_train_set2,
                               validation_data=(X_val_scaled, y_val_DIF_b),
                               epochs=100, batch_size=32,
                               callbacks=[early_stopping], verbose=0)
        
                val_pred_a = model_dif_a.predict(X_val_scaled, verbose=0)
                val_pred_b = model_dif_b.predict(X_val_scaled, verbose=0)
    
            # Calculate F1 scores (same for both merged and separate models)
            _, macro_f1_a, _, _ = macro_f1_vs_threshold(y_val_DIF_a, val_pred_a, f"{label} — rep {i+1} DIF_a")
            _, macro_f1_b, _, _ = macro_f1_vs_threshold(y_val_DIF_b, val_pred_b, f"{label} — rep {i+1} DIF_b")
    
            f1_scores_a.append(macro_f1_a)
            f1_scores_b.append(macro_f1_b)
    
        performance_summary.append({
            "Split": label,
            "Ratio": test_size,
            "F1_DIF_a_Mean": np.mean(f1_scores_a),
            "F1_DIF_a_Std": np.std(f1_scores_a),
            "F1_DIF_b_Mean": np.mean(f1_scores_b),
            "F1_DIF_b_Std": np.std(f1_scores_b)
        })

    
    # Convert to DataFrame
    perf_df = pd.DataFrame(performance_summary)
    perf_df["F1_Avg"] = (perf_df["F1_DIF_a_Mean"] + perf_df["F1_DIF_b_Mean"]) / 2
    best_split_avg = perf_df.loc[perf_df["F1_Avg"].idxmax(), "Ratio"]
    
    if plotting:
        # Plot with error bars
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(perf_df["Split"], perf_df["F1_DIF_a_Mean"], yerr=perf_df["F1_DIF_a_Std"],
                    label="DIF_a", fmt='-o', capsize=5)
        ax.errorbar(perf_df["Split"], perf_df["F1_DIF_b_Mean"], yerr=perf_df["F1_DIF_b_Std"],
                    label="DIF_b", fmt='-o', capsize=5)
        
        model_type = "Merged" if merged else "Separate"
        ax.set_title(f"Macro-F1 vs Train/Validation Split ({model_type} Models)", fontsize=14, fontweight='bold')
        ax.set_ylabel("Macro-F1")
        ax.set_xlabel("Train/Validation Split")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()

    return perf_df, best_split_avg

#%% InterDIFNet Training

def prepare_train_val_split(training_data, training_labels, test_size=0.2, random_state=12345, merged=False):
    """
    Prepare training and validation splits with proper scaling.
    
    Parameters:
    -----------
    training_data : pd.DataFrame
        Training data
    training_labels : pd.DataFrame  
        Training labels
    test_size : float
        Proportion of data for validation (default: 0.2)
    random_state : int
        Random state for reproducibility
    merged : bool
        If True, returns labels in dictionary format for merged model
    
    Returns:
    --------
    tuple
        For merged=False: (X_train_scaled, X_val_scaled, y_train_set1, y_train_set2, 
                          y_val_set1, y_val_set2, scaler, set1_cols, set2_cols)
        For merged=True: Also includes y_train_dict and y_val_dict
    """
    # Shuffle training data
    training_data, training_labels = shuffle(training_data, training_labels, random_state=random_state)

    # Split the data
    X_train, y_train_combined, X_val, y_val_combined = iterative_train_test_split(
        training_data.to_numpy(), training_labels.to_numpy(), test_size=test_size)

    # Convert back to DataFrame for label splitting
    y_train_df = pd.DataFrame(y_train_combined, columns=training_labels.columns)
    y_val_df = pd.DataFrame(y_val_combined, columns=training_labels.columns)
        
    # Split into separate label sets
    if merged:
        y_train_set1, y_train_set2, set1_cols, set2_cols, y_train_dict = prepare_label_sets(y_train_df, merged=True)
        y_val_set1, y_val_set2, _, _, y_val_dict = prepare_label_sets(y_val_df, merged=True)
    else:
        y_train_set1, y_train_set2, set1_cols, set2_cols = prepare_label_sets(y_train_df, merged=False)
        y_val_set1, y_val_set2, _, _ = prepare_label_sets(y_val_df, merged=False)

    # Scaling
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    if merged:
        return (X_train_scaled, X_val_scaled, y_train_set1, y_train_set2, 
                y_val_set1, y_val_set2, scaler, set1_cols, set2_cols,
                y_train_dict, y_val_dict)
    else:
        return (X_train_scaled, X_val_scaled, y_train_set1, y_train_set2, 
                y_val_set1, y_val_set2, scaler, set1_cols, set2_cols)


def train_models(X_train_scaled, X_val_scaled, y_train_set1, y_train_set2, 
                y_val_set1, y_val_set2, groups, epochs, batch_size, loss, 
                merged=False, y_train_dict=None, y_val_dict=None):
    """
    Train either separate DIF models or a single merged model.
    
    Parameters:
    -----------
    X_train_scaled : np.ndarray
        Scaled training features
    X_val_scaled : np.ndarray
        Scaled validation features  
    y_train_set1, y_train_set2 : np.ndarray
        Training labels for each set
    y_val_set1, y_val_set2 : np.ndarray
        Validation labels for each set
    groups : str
        String indicating number of groups (e.g., "Ten", "Three", "Two")
    epochs : int
        Number of training epochs (default: 200)
    batch_size : int
        Batch size for training (default: 32)
    loss : str
        Loss function to use
    merged : bool
        If True, trains a single merged model. If False, trains separate models.
    y_train_dict, y_val_dict : dict, optional
        Dictionary format labels for merged model (required if merged=True)
    
    Returns:
    --------
    tuple
        For merged=False: (model_dif_a, model_dif_b, history_a, history_b)
        For merged=True: (merged_model, None, history, None)
    """
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=25,
        restore_best_weights=True,
        verbose=1,
        start_from_epoch=20,
        mode='min'
    )

    n_inputs = X_train_scaled.shape[1]
    n_outputs_per_set = y_val_set1.shape[1]
    
    if merged:
        if y_train_dict is None or y_val_dict is None:
            raise ValueError("y_train_dict and y_val_dict are required when merged=True")
            
        # Build and train merged model
        print("Training Merged DIF model")
        model = build_merged_model(groups=groups,
                                 n_inputs=n_inputs,
                                 n_outputs_per_set=n_outputs_per_set,
                                 loss=loss)
        #print(type(y_train_dict['output_set1']), y_train_dict['output_set1'].shape)

        history = model.fit(
            X_train_scaled, 
            y_train_dict,
            validation_data=(X_val_scaled, y_val_dict),
            epochs=epochs, 
            batch_size=batch_size,
            callbacks=early_stopping, 
            verbose=0
        )
        
        return model, None, history, None
        
    else:
        # Build separate models
        print("Training Seperate DIF model")
        model_dif_a, model_dif_b = build_separate_models(groups=groups,
                                                         n_inputs=n_inputs,
                                                         n_outputs_per_set=n_outputs_per_set,
                                                         loss=loss)

        # Train DIF_a model
        print("Training Non-Uniform DIF model")
        history_a = model_dif_a.fit(
            X_train_scaled, 
            y_train_set1,
            validation_data=(X_val_scaled, y_val_set1),
            epochs=epochs, 
            batch_size=batch_size,
            callbacks=early_stopping, 
            verbose=0
        )

        # Train DIF_b model
        print("Training Uniform DIF model")
        history_b = model_dif_b.fit(
            X_train_scaled,
            y_train_set2,
            validation_data=(X_val_scaled, y_val_set2),
            epochs=epochs, 
            batch_size=batch_size,
            callbacks=early_stopping, 
            verbose=0
        )
        
        return model_dif_a, model_dif_b, history_a, history_b


def youden_vs_threshold(y_true_set, y_pred_set, set_name, verbose=False):
    """
    Find optimal threshold using Youden's J statistic.
    
    Parameters:
    -----------
    y_true_set : np.ndarray
        True labels
    y_pred_set : np.ndarray
        Predicted probabilities
    set_name : str
        Name for plotting
    verbose : bool
        Controls printing
    
    Returns:
    --------
    tuple
        (opt_threshold, opt_tpr, opt_fpr)
    """
    fpr, tpr, thresholds = roc_curve(y_true_set.ravel(), y_pred_set.ravel())
    youden = tpr - fpr

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, youden, color='royalblue', lw=2)
    ax.set_title(f'Youden Index vs Threshold — {set_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Youden Index (TPR - FPR)', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Highlight the optimal threshold
    opt_idx = np.argmax(youden)
    opt_threshold = thresholds[opt_idx]
    opt_tpr = tpr[opt_idx]
    opt_fpr = fpr[opt_idx]
    
    ax.axvline(opt_threshold, color='red', linestyle='--', label=f'Opt. threshold = {opt_threshold:.3f}')
    ax.legend()
    plt.tight_layout()
    plt.show()

    if verbose:
        print(f"{set_name} optimal threshold: {opt_threshold:.3f}")
        print(f"Validation {set_name} TPR at optimal threshold: {opt_tpr:.3f}")
        print(f"Validation {set_name} FPR at optimal threshold: {opt_fpr:.3f}")

    return opt_threshold, opt_tpr, opt_fpr


def macro_f1_vs_threshold(y_true_set, y_pred_set, set_name, 
                          threshold_range=None, verbose=False,
                          plotting=False):
    """
    Find optimal threshold using macro-F1 score for multilabel classification.
    
    Parameters:
    -----------
    y_true_set : np.ndarray
        Ground truth labels (n_samples, n_labels)
    y_pred_set : np.ndarray
        Predicted probabilities (n_samples, n_labels)
    set_name : str
        Name for plotting and output
    threshold_range : np.ndarray, optional
        Array of thresholds to test (default: 100 values from 0.01 to 0.99)
    verbose : bool
        Controls printing
    plotting : bool
        Whether to show plots
    
    Returns:
    --------
    tuple
        (opt_threshold, opt_macro_f1, opt_macro_precision, opt_macro_recall)
    """
    if threshold_range is None:
        threshold_range = np.linspace(0.01, 1, 100)
    
    macro_f1_scores = []
    macro_precision_scores = []
    macro_recall_scores = []
    
    n_labels = y_true_set.shape[1]
    
    for threshold in threshold_range:
        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred_set >= threshold).astype(int)
        
        # Calculate F1 for each label individually
        label_f1_scores = []
        label_precision_scores = []
        label_recall_scores = []
        
        for label_idx in range(n_labels):
            y_true_label = y_true_set[:, label_idx]
            y_pred_label = y_pred_binary[:, label_idx]
            
            # Calculate TP, FP, FN for this label
            tp = np.sum((y_true_label == 1) & (y_pred_label == 1))
            fp = np.sum((y_true_label == 0) & (y_pred_label == 1))
            fn = np.sum((y_true_label == 1) & (y_pred_label == 0))
            
            # Calculate precision, recall, F1 for this label
            if tp + fp == 0:
                precision = 0.0
            else:
                precision = tp / (tp + fp)
            
            if tp + fn == 0:
                recall = 0.0
            else:
                recall = tp / (tp + fn)
            
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
            
            label_f1_scores.append(f1)
            label_precision_scores.append(precision)
            label_recall_scores.append(recall)
        
        # Calculate macro averages
        macro_f1 = np.mean(label_f1_scores)
        macro_precision = np.mean(label_precision_scores)
        macro_recall = np.mean(label_recall_scores)
        
        macro_f1_scores.append(macro_f1)
        macro_precision_scores.append(macro_precision)
        macro_recall_scores.append(macro_recall)
    
    # Find optimal threshold
    opt_idx = np.argmax(macro_f1_scores)
    opt_threshold = threshold_range[opt_idx]
    opt_macro_f1 = macro_f1_scores[opt_idx]
    opt_macro_precision = macro_precision_scores[opt_idx]
    opt_macro_recall = macro_recall_scores[opt_idx]
    
    if plotting:
        # Plot results
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(threshold_range, macro_f1_scores, color='royalblue', lw=2, label='Macro-F1')
        ax.plot(threshold_range, macro_precision_scores, color='green', lw=1.5, alpha=0.7, label='Macro-Precision')
        ax.plot(threshold_range, macro_recall_scores, color='orange', lw=1.5, alpha=0.7, label='Macro-Recall')
        
        ax.set_title(f'Macro-F1 vs Threshold — {set_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Highlight the optimal threshold
        ax.axvline(opt_threshold, color='red', linestyle='--', 
                   label=f'Opt. threshold = {opt_threshold:.3f}\nMacro-F1 = {opt_macro_f1:.3f}')
        ax.legend()
        plt.tight_layout()
        plt.show()
    
    if verbose:
        print(f"{set_name} optimal threshold: {opt_threshold:.3f}")
        print(f"Validation {set_name} Macro-F1 at optimal threshold: {opt_macro_f1:.3f}")
        print(f"Validation {set_name} Macro-Precision at optimal threshold: {opt_macro_precision:.3f}")
        print(f"Validation {set_name} Macro-Recall at optimal threshold: {opt_macro_recall:.3f}")
        
    return opt_threshold, opt_macro_f1, opt_macro_precision, opt_macro_recall

def evaluate_model_f1_score(model, X_val, y_val, scenario_name, rep, label):
    """
    Evaluate macro F1 score and avoid storing predictions.
    """
    preds = model.predict(X_val, verbose=0)
    _, macro_f1, _, _ = macro_f1_vs_threshold(
        y_val, preds, f"{scenario_name}_rep{rep}_{label}", verbose=False,
        plotting=False
    )
    del preds
    return macro_f1


def plot_training_history(history_a, history_b=None, merged=False):
    """
    Plot training and validation loss for models.
    
    Parameters:
    -----------
    history_a : keras.callbacks.History
        Training history for model A (or merged model if merged=True)
    history_b : keras.callbacks.History, optional
        Training history for model B (ignored if merged=True)
    merged : bool
        If True, plots merged model history. If False, plots separate model histories.
    """
    if merged:
        # Plot merged model history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Combined loss
        ax1.plot(history_a.history['loss'], label='Training', color='blue', linewidth=2)
        ax1.plot(history_a.history['val_loss'], label='Validation', color='green', linewidth=2)
        ax1.set_title('Merged DIF Model - Total Loss', fontsize=18, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Individual output losses (if available)
        if 'output_set1_loss' in history_a.history:
            ax2.plot(history_a.history['output_set1_loss'], label='DIF_a Loss', color='red', linewidth=2)
            ax2.plot(history_a.history['output_set2_loss'], label='DIF_b Loss', color='orange', linewidth=2)
            ax2.plot(history_a.history['val_output_set1_loss'], label='Val DIF_a Loss', color='red', linestyle='--', linewidth=2)
            ax2.plot(history_a.history['val_output_set2_loss'], label='Val DIF_b Loss', color='orange', linestyle='--', linewidth=2)
            ax2.set_title('Merged DIF Model - Individual Output Losses', fontsize=18, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.axis('off')  # Hide second subplot if individual losses not available
        
    else:
        # Plot separate model histories
        all_losses = (history_a.history['loss'] +
                      history_b.history['loss'] +
                      history_a.history['val_loss'] +
                      history_b.history['val_loss'])

        y_min = min(all_losses) - 0.10
        y_max = max(all_losses) + 0.10

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # DIF_a Loss
        ax1.plot(history_a.history['loss'], label='Training', color='blue', linewidth=2)
        ax1.plot(history_a.history['val_loss'], label='Validation', color='green', linewidth=2)
        ax1.set_title('Non-Uniform DIF BCE Loss Comparison', fontsize=18, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Binary Crossentropy Loss')
        ax1.set_ylim(y_min, y_max)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # DIF_b Loss
        ax2.plot(history_b.history['loss'], label='Training', color='red', linewidth=2)
        ax2.plot(history_b.history['val_loss'], label='Validation', color='orange', linewidth=2)
        ax2.set_title('Uniform DIF BCE Loss Comparison', fontsize=18, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Binary Crossentropy Loss')
        ax2.set_ylim(y_min, y_max)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    

def find_optimal_thresholds(model_dif_a, model_dif_b, X_val_scaled,
                            y_val_set1, y_val_set2, merged=False):
    """
    Find optimal thresholds for models.
    
    Parameters:
    -----------
    model_dif_a : keras.Model
        First model (or merged model if merged=True)
    model_dif_b : keras.Model, optional
        Second model (ignored if merged=True)
    X_val_scaled : np.ndarray
        Validation features
    y_val_set1, y_val_set2 : np.ndarray
        Validation labels
    merged : bool
        If True, treats model_dif_a as merged model
        
    Returns:
    --------
    tuple
        (opt_thr_a, opt_thr_b)
    """
    if merged:
        # Get predictions from merged model
        predictions = model_dif_a.predict(X_val_scaled, verbose=0)
        val_pred_a = predictions[0]  # output_set1
        val_pred_b = predictions[1]  # output_set2
    else:
        # Get predictions from separate models
        val_pred_a = model_dif_a.predict(X_val_scaled, verbose=0)
        val_pred_b = model_dif_b.predict(X_val_scaled, verbose=0)
    
    # Using macro-F1 method
    opt_thr_a, macro_f1_a, _, _ = macro_f1_vs_threshold(
        y_val_set1, val_pred_a, set_name="Non-Uniform DIF", plotting=True
    )
    opt_thr_b, macro_f1_b, _, _ = macro_f1_vs_threshold(
        y_val_set2, val_pred_b, set_name="Uniform DIF", plotting=True
    )
    
    print(f"\nNon-Uniform DIF Threshold: {opt_thr_a:.3f}")
    print(f"\nUniform DIF Threshold: {opt_thr_b:.3f}")

    return opt_thr_a, opt_thr_b


def complete_training_pipeline(groups, loss, replications=500, epochs=200, batch_size=32, 
                              data_pattern="Training_data_ALL_Replication*.csv", 
                              labels_pattern="Training_labels_Replication*.csv",
                              training_features=None,
                              plot_results=True, threshold_method='macro_f1',
                              val_split=False, merged=False):
    """
    Complete training pipeline from data loading to model training and threshold optimization.
    
    Parameters:
    -----------
    groups : str
        String indicating number of groups (e.g., "Ten", "Three", "Two")
    loss : str
        Loss function to use
    replications : int
        Number of replications to load (default: 500)
    epochs : int
        Number of training epochs (default: 200)
    batch_size : int
        Batch size for training (default: 32)
    data_pattern : str
        Glob pattern for training data files
    labels_pattern : str
        Glob pattern for training labels files
    training_features : str
        Set to "TLP" to only use TLP input features
    plot_results : bool
        Whether to plot training history and threshold curves (default: True)
    threshold_method : str
        Method for threshold optimization: 'macro_f1' or 'youden' (default: 'macro_f1')
    val_split : bool
        Whether to test different validation split ratios (default: False)
    merged : bool
        If True, uses merged model architecture. If False, uses separate models.
    
    Returns:
    --------
    dict
        Dictionary containing all trained components
    """
    print("\nLoading training data")
    training_data, training_labels, feature_names, DIF_tests = load_training_data(
        groups,
        training_features=training_features,
        replications=replications, 
        data_pattern=data_pattern, 
        labels_pattern=labels_pattern
    )
    
    if val_split:
        print("Testing Training/Validation Split Ratios")
        _, val_ratio = evaluate_split_ratios(
            training_data,
            training_labels,
            groups,
            loss,
            split_ratios=[0.5, 0.4, 0.3, 0.2, 0.1],
            split_labels=["50/50", "60/40", "70/30", "80/20", "90/10"],
            n_repeats=10,
            verbose=False,
            plotting=True,
            merged=merged
        )
    else: 
        val_ratio = 0.20
    
    print("Performing Stratified Random Iterative Train/Validation split")
    split_results = prepare_train_val_split(
        training_data, training_labels, test_size=val_ratio, merged=merged
    )
    
    if merged:
        (X_train_scaled, X_val_scaled, y_train_set1, y_train_set2, 
         y_val_set1, y_val_set2, scaler, set1_cols, set2_cols,
         y_train_dict, y_val_dict) = split_results
    else:
        (X_train_scaled, X_val_scaled, y_train_set1, y_train_set2, 
         y_val_set1, y_val_set2, scaler, set1_cols, set2_cols) = split_results
        y_train_dict = y_val_dict = None
    
    #print("Training models")
    model_dif_a, model_dif_b, history_a, history_b = train_models(
        X_train_scaled, X_val_scaled, y_train_set1, y_train_set2, 
        y_val_set1, y_val_set2, groups, epochs, batch_size, loss,
        merged=merged, y_train_dict=y_train_dict, y_val_dict=y_val_dict
    )
    
    if plot_results:
        print("Plotting training history")
        plot_training_history(history_a, history_b, merged=merged)
    
    print("Calculating optimal thresholds")
    
    if merged:
        # Get predictions from merged model
        predictions = model_dif_a.predict(X_val_scaled, verbose=0)
        val_pred_a = predictions[0]  # output_set1
        val_pred_b = predictions[1]  # output_set2
    else:
        # Get predictions from separate models
        val_pred_a = model_dif_a.predict(X_val_scaled, verbose=0)
        val_pred_b = model_dif_b.predict(X_val_scaled, verbose=0)
    
    if threshold_method == 'youden':
        opt_thr_a, tpr_a, fpr_a = youden_vs_threshold(y_val_set1, val_pred_a, set_name="DIF_a")
        opt_thr_b, tpr_b, fpr_b = youden_vs_threshold(y_val_set2, val_pred_b, set_name="DIF_b")
    elif threshold_method == 'macro_f1':
        opt_thr_a, macro_f1_a, macro_prec_a, macro_rec_a = macro_f1_vs_threshold(
            y_val_set1, val_pred_a, set_name="Non-Uniform DIF", plotting=plot_results
        )
        opt_thr_b, macro_f1_b, macro_prec_b, macro_rec_b = macro_f1_vs_threshold(
            y_val_set2, val_pred_b, set_name="Uniform DIF", plotting=plot_results
        )
    else:
        raise ValueError("threshold_method must be 'youden' or 'macro_f1'")
    
    model_type = "Merged" if merged else "Separate"
    print(f"Training complete ({model_type} model)")
    print(f"DIF_a optimal threshold: {opt_thr_a:.3f}")
    print(f"DIF_b optimal threshold: {opt_thr_b:.3f}")
    
    if merged:
        return {
            'merged_model': model_dif_a,
            'scaler': scaler,
            'feature_names': feature_names,
            'set1_cols': set1_cols,
            'set2_cols': set2_cols,
            'opt_thr_a': opt_thr_a,
            'opt_thr_b': opt_thr_b,
            'history_a': history_a,
            'history_b': history_b,
            'val_ratio': val_ratio,
            'loss': loss,
            'merged': merged
        }
    else:
        return {
            'model_dif_a': model_dif_a,
            'model_dif_b': model_dif_b,
            'scaler': scaler,
            'feature_names': feature_names,
            'set1_cols': set1_cols,
            'set2_cols': set2_cols,
            'opt_thr_a': opt_thr_a,
            'opt_thr_b': opt_thr_b,
            'history_a': history_a,
            'history_b': history_b,
            'val_ratio': val_ratio,
            'loss': loss,
            'merged': merged
        }

#%% DIF Ablation Functions
def find_columns_by_prefix(df, prefix):
    """
    Find all columns that start with a specific prefix.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to search
    prefix : str or list
        Prefix(es) to search for (e.g., 'CSIB', 'SIB')

    Returns:
    --------
    list
        List of column names that match the prefix
    """
    if isinstance(prefix, str):
        prefixes = [prefix]
    else:
        prefixes = prefix.copy()
    return [col for col in df.columns if any(col.startswith(p) for p in prefixes)]


def generate_all_combinations(prefixes):
    """
    Generate all possible combinations of prefixes to permute.

    Parameters:
    -----------
    prefixes : list
        List of prefix strings

    Returns:
    --------
    list
        List of tuples representing all possible combinations
    """
    all_combinations = []
    for r in range(1, len(prefixes) + 1):
        for combo in combinations(prefixes, r):
            all_combinations.append(combo)
    return all_combinations


def create_scenario_name(permuted_prefixes):
    """
    Create a readable name for each ablation scenario.

    Parameters:
    -----------
    permuted_prefixes : tuple
        Tuple of prefixes that are permuted in this scenario

    Returns:
    --------
    str
        String name for the scenario
    """
    if len(permuted_prefixes) == 0:
        return "baseline"
    elif len(permuted_prefixes) == 1:
        return f"{permuted_prefixes[0]}"
    else:
        return "+".join(sorted(permuted_prefixes))


def build_evaluation_models(n_inputs, n_outputs_per_set, groups, loss, merged=False):
    """
    Build lightweight models for ablation evaluation.
    
    Parameters:
    -----------
    n_inputs : int
        Number of input features
    n_outputs_per_set : int
        Number of outputs per model
    groups : str
        String indicating number of groups (e.g., "Ten", "Three", "Two")
    loss : str
        Loss function to use
    merged : bool
        If True, builds merged model. If False, builds separate models.
        
    Returns:
    --------
    Model or tuple
        For merged=True: Single merged model
        For merged=False: (model_set1, model_set2)
    """
    if merged:
        return build_merged_model(groups=groups,
                                n_inputs=n_inputs, 
                                n_outputs_per_set=n_outputs_per_set,
                                loss=loss)
    else:
        return build_separate_models(groups=groups,
                                   n_inputs=n_inputs, 
                                   n_outputs_per_set=n_outputs_per_set,
                                   loss=loss)


def get_early_stopping_callback():
    """
    Get early stopping callback for ablation training.
    
    Returns:
    --------
    EarlyStopping
        Configured early stopping callback
    """
    return EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=25,
        restore_best_weights=True,
        verbose=0,
        start_from_epoch=20,
        mode='min'
    )


def plot_top_ablation_results(results_df, top_k=10, 
                              score_col="Mean_F1_Overall",
                              reference="baseline"):
    """
    Plot top-k scenarios with largest F1 performance drop from baseline.

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from run_ablation_study()
    top_k : int
        Number of top scenarios to plot
    score_col : str
        Which score to rank by ("Mean_F1_Overall", "F1_DIF_a_Mean" or "F1_DIF_b_Mean")
    reference : str
        Which DIF Test serves as reference group
    """
    # Identify reference score
    if reference == "baseline":
        baseline_score = results_df.loc[0, score_col]
    elif isinstance(reference, int):
        baseline_score = results_df.loc[reference, score_col]
    elif isinstance(reference, str):
        ref_row = results_df[results_df["Scenario"] == reference]
        if ref_row.empty:
            raise ValueError(f"Reference scenario '{reference}' not found.")
        baseline_score = ref_row.iloc[0][score_col]
    else:
        raise ValueError("Invalid reference. Use 'baseline', a row index, or a scenario name.")
   
    # Compute drop
    results_df = results_df.copy()
    results_df["Drop_From_Reference"] = baseline_score - results_df[score_col]
   
    # Exclude reference row for ranking
    results_no_ref = results_df[results_df[score_col] != baseline_score].copy()
   
    # Top-k
    top_df = results_no_ref.sort_values("Drop_From_Reference", ascending=False).head(top_k)
   
    # Plot - Fixed to resolve seaborn deprecation warning
    plt.figure(figsize=(10, 6))
    sns.barplot(
        y="Permuted_Prefixes", 
        x="Drop_From_Reference", 
        hue="Permuted_Prefixes",
        data=top_df,
        palette="rocket",
        legend=False
    )
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel(f"{score_col} Drop from Reference")
    plt.ylabel("Permuted Prefix Group(s)")
    plt.title(f"Top {top_k} Ablations by Drop from Reference ({score_col})")
    plt.tight_layout()
    plt.show()


def print_best_scenario(results_df, merged=False):
    """
    Print the best performing scenario (highest average F1 across DIF_a and DIF_b).
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from ablation study
    merged : bool
        Whether results are from merged model
    """
    
    best_row = results_df.loc[results_df["Mean_F1_Overall"].idxmax()]
    
    model_type = "Merged" if merged else "Separate"
    print(f"Best Scenario (Highest Mean F1) - {model_type} Model:")
    print(f"  Scenario: {best_row['Scenario']}")
    print(f"  Permuted Prefixes: {best_row['Permuted_Prefixes']}")
    print(f"  F1 DIF_a Mean: {best_row['F1_DIF_a_Mean']:.4f}")
    print(f"  F1 DIF_b Mean: {best_row['F1_DIF_b_Mean']:.4f}")
    print(f"  Average F1: {best_row['Mean_F1_Overall']:.4f}")


def evaluate_merged_model_f1_score(model, X_val, y_val_a, y_val_b, scenario_name, rep):
    """
    Evaluate macro F1 score for merged model and avoid storing predictions.
    
    Parameters:
    -----------
    model : keras.Model
        Merged model with two outputs
    X_val : np.ndarray
        Validation features
    y_val_a, y_val_b : np.ndarray
        Validation labels for DIF_a and DIF_b
    scenario_name : str
        Name of current scenario
    rep : int
        Repetition number
        
    Returns:
    --------
    tuple
        (macro_f1_a, macro_f1_b)
    """
    predictions = model.predict(X_val, verbose=0)
    pred_a = predictions[0]  # output_set1
    pred_b = predictions[1]  # output_set2
    
    _, macro_f1_a, _, _ = macro_f1_vs_threshold(
        y_val_a, pred_a, f"{scenario_name}_rep{rep}_DIF_a", verbose=False, plotting=False
    )
    _, macro_f1_b, _, _ = macro_f1_vs_threshold(
        y_val_b, pred_b, f"{scenario_name}_rep{rep}_DIF_b", verbose=False, plotting=False
    )
    
    del predictions, pred_a, pred_b
    return macro_f1_a, macro_f1_b


def run_ablation_study(training_data, training_labels, DIF_tests, groups,
                       set1_cols, set2_cols, n_outputs_per_set, loss,
                       val_size=0.2,
                       scenario_prefix_combinations=None,
                       n_repeats=5,
                       random_state=12345,
                       reference="baseline",
                       skip_all_permuted=True,
                       verbose=False,
                       merged=False):
    """
    Run ablation study by permuting feature groups defined by prefixes.
    
    Parameters:
    -----------
    training_data : pd.DataFrame
        Original training features (before permutation)
    training_labels : pd.DataFrame
        Label DataFrame for both DIF_a and DIF_b
    DIF_tests : list
        List of all available feature group prefixes
    groups : str
        String indicating number of groups (e.g., "Ten", "Three", "Two")
    set1_cols, set2_cols : list
        Names of DIF_a and DIF_b columns
    n_outputs_per_set : int
        Number of outputs per model
    loss : str
        Loss function to use
    val_size : float
        Validation size for both models
    scenario_prefix_combinations : list of tuples, optional
        List of prefix combinations to permute (if None, generate all)
    n_repeats : int
        Number of repetitions for each scenario to average performance
    random_state : int
        Seed for reproducibility
    reference : str
        Scenario used for reference group. Default is baseline, no permutation
    skip_all_permuted : bool
        If True, skip the scenario where all feature groups are permuted
    verbose : bool
        Whether to print detailed progress
    merged : bool
        If True, uses merged model architecture. If False, uses separate models.
        
    Returns:
    --------
    tuple
        (results_df, selected_features, optimal_training_data) - Results and optimal feature sets
    """
    np.random.seed(random_state)
    model_type = "Merged" if merged else "Separate"
    print(f"Running Ablation Feature Selection Algorithm ({model_type} Model)")
    
    # Generate all prefix permutations if not provided
    if scenario_prefix_combinations is None:
        scenario_prefix_combinations = generate_all_combinations(DIF_tests)
        
    if skip_all_permuted:
        all_prefixes_tuple = tuple(DIF_tests)
        scenario_prefix_combinations = [combo for combo in scenario_prefix_combinations 
                                      if combo != all_prefixes_tuple]

    # Add baseline (no permutation) as first scenario
    scenario_prefix_combinations = [()] + list(scenario_prefix_combinations)

    # Container for performance metrics
    scenario_results = []

    # Loop through each permutation scenario
    for scenario_idx, permuted_prefixes in enumerate(scenario_prefix_combinations):
        scenario_name = create_scenario_name(permuted_prefixes)
        print(f"[{scenario_idx+1}/{len(scenario_prefix_combinations)}] Running scenario: {scenario_name}")
        
        f1_a_scores = []
        f1_b_scores = []

        for repeat in range(n_repeats):
            if verbose:
                print(f"  Repetition {repeat+1}/{n_repeats}")

            # Create copy of training data
            data_permuted = training_data.copy()

            # Permute columns for selected prefixes only (if any)
            if len(permuted_prefixes) > 0:
                for prefix in permuted_prefixes:
                    prefix_cols = find_columns_by_prefix(data_permuted, prefix)
                    for col in prefix_cols:
                        data_permuted[col] = np.random.permutation(data_permuted[col].values)

            # Convert to numpy for compatibility
            X_np = data_permuted.to_numpy()
            Y_np = training_labels.to_numpy()

            if merged:
                # Single split for merged model
                X_train, y_train, X_val, y_val = iterative_train_test_split(
                    X_np, Y_np, test_size=val_size
                )
                
                # Process labels for merged model
                y_train_df = pd.DataFrame(y_train, columns=training_labels.columns)
                y_val_df = pd.DataFrame(y_val, columns=training_labels.columns)
                
                y_train_DIF_a, y_train_DIF_b, _, _, y_train_dict = prepare_label_sets(y_train_df, merged=True)
                y_val_DIF_a, y_val_DIF_b, _, _, y_val_dict = prepare_label_sets(y_val_df, merged=True)
                
                # Standard scaling
                scaler = StandardScaler().fit(X_train)
                X_train_scaled = scaler.transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Build and train merged model
                model = build_evaluation_models(
                    n_inputs=X_train.shape[1], 
                    n_outputs_per_set=n_outputs_per_set,
                    groups=groups,
                    loss=loss,
                    merged=True
                )
                
                # Early stopping
                early_stopping = get_early_stopping_callback()
                
                # Train merged model
                model.fit(
                    X_train_scaled, y_train_dict,
                    validation_data=(X_val_scaled, y_val_dict),
                    epochs=200, batch_size=32,
                    callbacks=[early_stopping], verbose=0
                )
                
                # Evaluate merged model
                macro_f1_a, macro_f1_b = evaluate_merged_model_f1_score(
                    model, X_val_scaled, y_val_DIF_a, y_val_DIF_b, scenario_name, repeat+1
                )
                
                # Cleanup
                del model
                
            else:
                # Separate splits for separate models (original approach)
                X_train_a, y_train_a, X_val_a, y_val_a = iterative_train_test_split(
                    X_np, Y_np, test_size=val_size
                )
                X_train_b, y_train_b, X_val_b, y_val_b = iterative_train_test_split(
                    X_np, Y_np, test_size=val_size
                )

                # Process labels
                y_train_df_a = pd.DataFrame(y_train_a, columns=training_labels.columns)
                y_val_df_a = pd.DataFrame(y_val_a, columns=training_labels.columns)
                y_train_df_b = pd.DataFrame(y_train_b, columns=training_labels.columns)
                y_val_df_b = pd.DataFrame(y_val_b, columns=training_labels.columns)

                y_train_DIF_a = y_train_df_a[set1_cols].values
                y_val_DIF_a = y_val_df_a[set1_cols].values
                y_train_DIF_b = y_train_df_b[set2_cols].values
                y_val_DIF_b = y_val_df_b[set2_cols].values

                # Standard scaling
                scaler_a = StandardScaler().fit(X_train_a)
                X_train_a_scaled = scaler_a.transform(X_train_a)
                X_val_a_scaled = scaler_a.transform(X_val_a)

                scaler_b = StandardScaler().fit(X_train_b)
                X_train_b_scaled = scaler_b.transform(X_train_b)
                X_val_b_scaled = scaler_b.transform(X_val_b)

                # Build separate models
                model_dif_a, model_dif_b = build_evaluation_models(
                    n_inputs=X_train_a.shape[1], 
                    n_outputs_per_set=n_outputs_per_set,
                    groups=groups,
                    loss=loss,
                    merged=False
                )
                
                # Early stopping
                early_stopping = get_early_stopping_callback()

                # Train model for DIF_a
                model_dif_a.fit(
                    X_train_a_scaled, y_train_DIF_a,
                    validation_data=(X_val_a_scaled, y_val_DIF_a),
                    epochs=200, batch_size=32,
                    callbacks=[early_stopping], verbose=0
                )

                # Train model for DIF_b
                model_dif_b.fit(
                    X_train_b_scaled, y_train_DIF_b,
                    validation_data=(X_val_b_scaled, y_val_DIF_b),
                    epochs=200, batch_size=32,
                    callbacks=[early_stopping], verbose=0
                )
                
                # Predict and evaluate only summary scores
                macro_f1_a = evaluate_model_f1_score(model_dif_a, X_val_a_scaled, y_val_DIF_a, scenario_name, repeat+1, "DIF_a")
                macro_f1_b = evaluate_model_f1_score(model_dif_b, X_val_b_scaled, y_val_DIF_b, scenario_name, repeat+1, "DIF_b")
                
                # Cleanup memory
                del model_dif_a, model_dif_b
            
            # Store summary scores
            f1_a_scores.append(macro_f1_a)
            f1_b_scores.append(macro_f1_b)
            
            # Cleanup
            tf.keras.backend.clear_session()
            gc.collect()

        # Save summary for scenario
        scenario_results.append({
            "Scenario": scenario_name,
            "Permuted_Prefixes": "+".join(permuted_prefixes) if permuted_prefixes else "None",
            "F1_DIF_a_Mean": np.mean(f1_a_scores),
            "F1_DIF_a_Std": np.std(f1_a_scores),
            "F1_DIF_b_Mean": np.mean(f1_b_scores),
            "F1_DIF_b_Std": np.std(f1_b_scores),
            "F1_DIF_a_Scores": f1_a_scores,
            "F1_DIF_b_Scores": f1_b_scores
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(scenario_results)
    
    # Get best overall
    results_df = results_df.copy()
    results_df["Mean_F1_Overall"] = results_df[["F1_DIF_a_Mean", "F1_DIF_b_Mean"]].mean(axis=1)
    best_row = results_df.loc[results_df["Mean_F1_Overall"].idxmax()]
    best_prefixes = best_row["Permuted_Prefixes"]
    if best_prefixes != "None":
        DIF_prefixes = best_prefixes.split("+")
        selected_features = [col for col in training_data.columns 
                           if any(col.startswith(prefix) for prefix in DIF_prefixes)]
        optimal_training_data = training_data[selected_features]
    else:
        selected_features = training_data.columns
        optimal_training_data = training_data.copy()
        
    # Plot results
    plot_top_ablation_results(results_df, top_k=10,
                              score_col="Mean_F1_Overall",
                              reference=reference)
    
    # Print the results
    print_best_scenario(results_df, merged=merged)
    
    return results_df, selected_features, optimal_training_data

#%% DIF Detection on Testing Data
def evaluate_models_on_test_sets(groups, model_dif_a, model_dif_b, scaler, 
                                set1_cols, set2_cols, opt_thr_a, opt_thr_b, 
                                sizes, selected_features = None, percentages=[20, 40], 
                                replications=range(1, 101), verbose = False,
                                save_results = False, merged=False):
    """
    Evaluate trained models on test sets with different configurations.

    Parameters:
    -----------
    groups : str
        String indicating number of groups (e.g., "Ten", "Three", "Two")
    model_dif_a, model_dif_b : keras.Model
        Trained models or merged model if merged=True
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler from training
    selected_features : list
        Names of features to use. Default is to use all of them
    set1_cols, set2_cols : list
        Column names for each label set
    opt_thr_a, opt_thr_b : float
        Optimal thresholds for each model
    sizes : list
        Sample sizes to evaluate 
    percentages : list
        DIF percentages to evaluate (default: [20, 40])
    replications : range
        Replication numbers to process (default: range(1, 101))
    verbose : bool
        Controls printing
    merged : bool
        Whether model_dif_a is a merged model

    Returns:
    --------
    pd.DataFrame
        Summary results with TPR/FPR for each configuration
    """
    DIF_summary_results = []

    # Get number of groups
    if len(set1_cols) > 1:
        group_ids = set()
        for label in set1_cols:
            pair = label.split('DIF_a_')[1]
            for g in pair.split('Group'):
                if g:
                    group_ids.add(g)
        num_groups = len(group_ids)
    else:
        num_groups = 2

    print(f"Number Of Groups In The Data: {num_groups}")

    for n in sizes:
        for p in percentages:
            group_results = {group: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for group in ['DIF_a', 'DIF_b']}

            print(f"Processing - N: {n}, DIF Percent: {p}%")

            for r in replications:
                filename_testing = f"{groups}_Group_Testing_data_ALL_{n}_{p}_Replication{r}.csv"
                filename_testing_labels = f"{groups}_Group_Testing_data_labels_{n}_{p}_Replication{r}.csv"

                if not os.path.exists(filename_testing) or not os.path.exists(filename_testing_labels):
                    if verbose:
                        print(f"Skipping Replication {r}: Missing file(s)")
                    continue

                testing_data = pd.read_csv(filename_testing)
                testing_labels = pd.read_csv(filename_testing_labels)

                if selected_features is not None:
                    testing_data = testing_data[selected_features]
                    testing_data, testing_labels = clean_data(testing_data, testing_labels)
                    X_test_scaled = pd.DataFrame(scaler.transform(testing_data.to_numpy()))
                    X_test_scaled.columns = selected_features

                else:
                    testing_data, testing_labels = clean_data(testing_data, testing_labels)
                    X_test_scaled = pd.DataFrame(scaler.transform(testing_data.to_numpy()))
                    X_test_scaled.columns = testing_data.columns
                    
                # Get predictions
                if merged:
                    predictions = model_dif_a.predict(X_test_scaled, verbose=0)
                    temp_results_dif_a = predictions[0]
                    temp_results_dif_b = predictions[1]
                else:
                    temp_results_dif_a = model_dif_a.predict(X_test_scaled, verbose=0)
                    temp_results_dif_b = model_dif_b.predict(X_test_scaled, verbose=0)

                J = X_test_scaled.shape[0]
                L_a = len(set1_cols)
                L_b = len(set2_cols)

                temp_results_dif_a = np.array(temp_results_dif_a).reshape(J, L_a)
                temp_results_dif_b = np.array(temp_results_dif_b).reshape(J, L_b)

                temp_results_dif_a = pd.DataFrame(temp_results_dif_a, columns=set1_cols)
                temp_results_dif_b = pd.DataFrame(temp_results_dif_b, columns=set2_cols)

                binary_preds_a = (temp_results_dif_a > opt_thr_a).astype(int)
                binary_preds_b = (temp_results_dif_b > opt_thr_b).astype(int)

                predicted_DIF_labels = pd.concat([binary_preds_a, binary_preds_b], axis=1)
                predicted_DIF_labels = predicted_DIF_labels[testing_labels.columns]
                predicted_DIF_labels.columns = testing_labels.columns

                if save_results:
                    output_filename = f"Classification_Results_{groups}_{n}_{p}_Replication{r}.csv"
                    predicted_DIF_labels.to_csv(output_filename, index=False)

                group_pairs = list(combinations(range(1, num_groups + 1), 2))

                for group in ['DIF_a', 'DIF_b']:
                    if num_groups > 2:
                        pred_cols = [f'{group}_Group{pair[0]}Group{pair[1]}' for pair in group_pairs]
                        true_cols = [f'{group}_Group{pair[0]}Group{pair[1]}' for pair in group_pairs]
                    else:
                        pred_cols = group
                        true_cols = group

                    pred_flattened = predicted_DIF_labels[pred_cols].values.flatten()
                    true_flattened = testing_labels[true_cols].values.flatten()

                    TP = ((pred_flattened == 1) & (true_flattened == 1)).sum()
                    FP = ((pred_flattened == 1) & (true_flattened == 0)).sum()
                    TN = ((pred_flattened == 0) & (true_flattened == 0)).sum()
                    FN = ((pred_flattened == 0) & (true_flattened == 1)).sum()

                    group_results[group]['TP'] += TP
                    group_results[group]['FP'] += FP
                    group_results[group]['TN'] += TN
                    group_results[group]['FN'] += FN

            for group, metrics in group_results.items():
                DIF_summary_results.append({
                    'N': n,
                    'Perc': p,
                    'Group': group,
                    'TP': metrics['TP'],
                    'FP': metrics['FP'],
                    'TN': metrics['TN'],
                    'FN': metrics['FN']
                })

    DIF_summary_results = pd.DataFrame(DIF_summary_results)
    DIF_summary_results['TPR'] = DIF_summary_results['TP'] / (DIF_summary_results['TP'] + DIF_summary_results['FN'])
    DIF_summary_results['FPR'] = DIF_summary_results['FP'] / (DIF_summary_results['FP'] + DIF_summary_results['TN'])

    return DIF_summary_results


#%% Clustering DIF Item Functions
def load_dif_data(groups, n, perc, r):

    dif_data = {'DIF_a': pd.DataFrame(), 'DIF_b': pd.DataFrame()}
    
    filepath = f"Classification_Results_{groups}_{n}_{perc}_Replication{r}.csv"
    
    df = pd.read_csv(filepath)
    
    # Check which DIF type this file contains
    dif_a_cols = [col for col in df.columns if col.startswith('DIF_a_')]
    dif_b_cols = [col for col in df.columns if col.startswith('DIF_b_')]
    
    if dif_a_cols:
        #print(f"  Found {len(dif_a_cols)} DIF_a columns")
        if dif_data['DIF_a'].empty:
            dif_data['DIF_a'] = df.copy()
        else:
            # Merge with existing DIF_a data
            dif_data['DIF_a'] = pd.concat([dif_data['DIF_a'], df], ignore_index=True)
    
    if dif_b_cols:
        #print(f"  Found {len(dif_b_cols)} DIF_b columns")
        if dif_data['DIF_b'].empty:
            dif_data['DIF_b'] = df.copy()
        else:
            # Merge with existing DIF_b data
            dif_data['DIF_b'] = pd.concat([dif_data['DIF_b'], df], ignore_index=True)
    
    if not dif_a_cols and not dif_b_cols:
        raise KeyError(f"Error: No DIF_a or DIF_b columns found in {filepath}")
    
    return dif_data

def extract_groups_from_columns(dif_data, dif_type):
    """
    Extract unique group names from column headers for specified DIF type
    """
    column_names = dif_data.columns
    groups = set()
    pattern = f'{dif_type}_Group(\\d+)Group(\\d+)'
    
    for col in column_names:
        matches = re.findall(pattern, col)
        for match in matches:
            groups.add(f'Group{match[0]}')
            groups.add(f'Group{match[1]}')
    
    return sorted(list(groups), key=lambda x: int(x.replace('Group', '')))

def create_dif_matrix_per_item(dif_data, groups, item_index, dif_type='DIF_a'):
    """
    Create DIF matrix from pairwise comparison data for a specific item
    
    Parameters:
        df: DataFrame containing DIF data
        groups: List of group names
        item_index: Row index for the specific item
        dif_type: 'DIF_a' or 'DIF_b'
    
    Returns:
        pd.DataFrame: DIF matrix for the specified item
    """
    
    # Check for missing data in the specific row
    item_row = dif_data.iloc[item_index]
    dif_cols = [col for col in dif_data.columns if col.startswith(f'{dif_type}_')]
    
    if item_row[dif_cols].isna().any() or item_row[dif_cols].map(np.isinf).any():
        print(f"Warning: Missing or infinite data found in item {item_index}")
    
    dif_matrix = pd.DataFrame(0.0, index=groups, columns=groups)
    
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            group1, group2 = groups[i], groups[j]
            
            # Check both possible column name orders
            col_name1 = f'{dif_type}_{group1}{group2}'
            col_name2 = f'{dif_type}_{group2}{group1}'
            
            if col_name1 in dif_data.columns:
                dif_value = item_row[col_name1]
            elif col_name2 in dif_data.columns:
                dif_value = item_row[col_name2]
            else:
                dif_value = np.nan
                
            dif_matrix.loc[group1, group2] = dif_value
            dif_matrix.loc[group2, group1] = dif_value
            
    return dif_matrix

def Floyd_Warshall_Closure(dif_matrix, threshold, verbose=False):
    """
    Enforce transitive closure on DIF matrix using Floyd-Warshall-like algorithm.
    If there's a path of similar groups (DIF < threshold), all pairs in that path 
    should be considered similar.
    
    Parameters:
        dif_matrix: pandas DataFrame with DIF values
        threshold: DIF threshold for similarity
        verbose: Whether to print detailed information about changes
    
    Returns:
        pandas DataFrame: Modified DIF matrix with transitive closure enforced
        dict: Information about changes made
    """
    # Work with a copy to avoid modifying original
    modified_matrix = dif_matrix.copy()
    groups = list(dif_matrix.index)
    n_groups = len(groups)
    
    changes_made = []
    
    if verbose:
        print(f"Enforcing transitive closure with threshold {threshold}")
        print(f"Initial similarity pairs (DIF < {threshold}):")
        initial_pairs = []
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                if modified_matrix.iloc[i, j] < threshold:
                    initial_pairs.append((groups[i], groups[j], modified_matrix.iloc[i, j]))
        for g1, g2, dif_val in initial_pairs:
            print(f"  {g1} - {g2}: {dif_val:.4f}")
    
    # Use Floyd-Warshall-like algorithm for transitive closure
    # For each potential intermediate node k
    for k in range(n_groups):
        group_k = groups[k]
        
        # For each pair of nodes i, j
        for i in range(n_groups):
            for j in range(n_groups):
                if i == j or i == k or j == k:
                    continue
                
                group_i, group_j = groups[i], groups[j]
                
                # Get current DIF values
                dif_ik = modified_matrix.iloc[i, k]  # i to k
                dif_kj = modified_matrix.iloc[k, j]  # k to j
                dif_ij = modified_matrix.iloc[i, j]  # i to j (current)
                
                # Check if we have a path i~k~j where both edges are similar
                if (not np.isnan(dif_ik) and not np.isnan(dif_kj) and 
                    dif_ik < threshold and dif_kj < threshold):
                    
                    # Calculate new DIF value for i~j through path i~k~j
                    # Use maximum of the path (weakest link determines strength)
                    new_dif_ij = max(dif_ik, dif_kj)
                    
                    # If current i~j relationship is weaker (higher DIF) than the path,
                    # or if i~j was not similar before, update it
                    if (np.isnan(dif_ij) or dif_ij >= threshold or new_dif_ij < dif_ij):
                        
                        # Only make changes if it improves the relationship
                        if np.isnan(dif_ij) or new_dif_ij < dif_ij:
                            # Update both symmetric positions
                            modified_matrix.iloc[i, j] = new_dif_ij
                            modified_matrix.iloc[j, i] = new_dif_ij
                            
                            changes_made.append({
                                'type': 'transitive_closure',
                                'path': (group_i, group_k, group_j),
                                'original_dif_ij': dif_ij,
                                'new_dif_ij': new_dif_ij,
                                'dif_ik': dif_ik,
                                'dif_kj': dif_kj,
                                'reason': f"{group_i}~{group_k} ({dif_ik:.4f}) and {group_k}~{group_j} ({dif_kj:.4f}) → {group_i}~{group_j} ({new_dif_ij:.4f})"
                            })
                            
                            if verbose:
                                original_str = f"{dif_ij:.4f}" if not np.isnan(dif_ij) else "None"
                                print(f"  Path {group_i}→{group_k}→{group_j}: {group_i}~{group_j} updated from {original_str} to {new_dif_ij:.4f}")
    
    # Summary information
    closure_info = {
        'total_changes': len(changes_made),
        'changes_detail': changes_made,
        'final_similarity_pairs': []
    }
    
    # Collect final similarity pairs
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            if modified_matrix.iloc[i, j] < threshold:
                closure_info['final_similarity_pairs'].append(
                    (groups[i], groups[j], modified_matrix.iloc[i, j])
                )
    
    if verbose:
        print("\nTransitive closure completed")
        print(f"Total changes made: {closure_info['total_changes']}")
        print(f"Final similarity pairs (DIF < {threshold}):")
        for g1, g2, dif_val in closure_info['final_similarity_pairs']:
            print(f"  {g1} - {g2}: {dif_val:.4f}")
    
    return modified_matrix, closure_info

def connected_components_clustering(dif_matrix, groups, dif_threshold=0.01, 
                                    dif_type='DIF_a', verbose_closure=False):
    """
    Find connected components where groups have DIF below threshold
    Groups in same component don't display DIF with each other
    Now with transitive closure enforcement
    """
    # Apply transitive closure first
    modified_matrix, closure_info = Floyd_Warshall_Closure(
        dif_matrix, dif_threshold, verbose=verbose_closure
    )
    
    # Create graph using the transitively closed matrix
    G = nx.Graph()
    G.add_nodes_from(groups)
    
    # Add edges for group pairs with DIF below threshold
    edges_added = 0
    edge_details = []
    
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            dif_value = modified_matrix.iloc[i, j]
            if not np.isnan(dif_value) and dif_value < dif_threshold:
                G.add_edge(groups[i], groups[j], weight=1 - dif_value, dif=dif_value)
                edges_added += 1
                edge_details.append((groups[i], groups[j], dif_value))
    
    # Find connected components
    connected_components = list(nx.connected_components(G))
    
    # Create cluster assignments
    cluster_dict = {}
    for cluster_id, component in enumerate(connected_components, 1):
        for group in component:
            cluster_dict[group] = cluster_id
    
    # Handle isolated nodes (groups that don't cluster with any other)
    isolated_count = 0
    for group in groups:
        if group not in cluster_dict:
            cluster_dict[group] = len(connected_components) + 1 + isolated_count
            isolated_count += 1
    
    return cluster_dict, connected_components, G, edge_details, closure_info

def visualize_dif_matrix_per_item(dif_matrix, dif_type, item_index):
    """Visualize DIF matrix as heatmap for a specific item"""
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(dif_matrix, dtype=bool))
        
    sns.heatmap(dif_matrix, mask=mask, annot=True, cmap=("PiYG"), 
               fmt='.4f', square=True, linewidths=0.5, 
               cbar_kws={'label': 'Pairwise DIF Probability'},
               vmin=0, vmax=1, center=0.50)
    
    if dif_type == 'DIF_a':
        plt.title(f'DIF on a - Item {item_index} Pairwise Probability Matrix')
    elif dif_type == 'DIF_b':
        plt.title(f'DIF on b - Item {item_index} Pairwise Probability Matrix')
    plt.tight_layout()
    plt.show()


def visualize_connected_components_per_item(G, cluster_dict, groups, dif_threshold, 
                                          dif_type='DIF_a', item_index=0):
    """Visualize the graph with clusters colored for a specific item"""
    plt.figure(figsize=(12, 8))
    
    # Create layout
    try:
        pos = nx.spring_layout(G, seed=42, k=3, iterations=100)
    except:
        pos = nx.circular_layout(G)
    
    # Get unique clusters and assign colors
    unique_clusters = sorted(set(cluster_dict.values()))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
    
    # Draw nodes colored by cluster
    for cluster_id in unique_clusters:
        cluster_nodes = [g for g in groups if cluster_dict[g] == cluster_id]
        nx.draw_networkx_nodes(G, pos, nodelist=cluster_nodes, 
                             node_color=[colors[cluster_id-1]], 
                             node_size=3000, alpha=0.8,
                             edgecolors='black', linewidths=1)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=1, edge_color='black', width=2)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Uncomment to Add edge labels with DIF values
    # edge_labels = {}
    # for u, v, d in G.edges(data=True):
    #     if 'dif' in d:
    #         edge_labels[(u, v)] = f"{d['dif']:.3f}"
    
    #nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
    if dif_type == "DIF_b":
        plt.title(f'Item {item_index + 1} - Connected Components\nSimilar b Parameters (Threshold < {dif_threshold})', 
                  fontsize=14, fontweight='bold')
    elif dif_type == "DIF_a":
        plt.title(f'Item {item_index + 1} - Connected Components\nSimilar a Parameters (Threshold < {dif_threshold})', 
                  fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def analyze_item_threshold(dif_matrix, groups, threshold, dif_type='DIF_a', item_index=0, verbose_closure=False):
    """Analyze a single threshold value for a specific item and return results"""
    cluster_dict, components, G, edges, closure_info = connected_components_clustering(
        dif_matrix, groups, threshold, dif_type, verbose_closure)
    
    # Create results DataFrame
    cluster_df = pd.DataFrame({
        'Group': groups, 
        'Cluster': [cluster_dict[g] for g in groups]
    }).sort_values(by='Cluster')
    
    # Calculate statistics
    num_clusters = len(set(cluster_dict.values()))
    num_edges = G.number_of_edges()
    cluster_sizes = cluster_df['Cluster'].value_counts().sort_index()
    multi_group_clusters = cluster_sizes[cluster_sizes > 1]
    
    # Debugging Code
    '''
    Prints the DIF matrix before and after transitive closure. Lists the
    changes and the number of chnages made.
    '''
    # print(f"Before transitive closure: {dif_matrix.values}")
    # modified_matrix, closure_info = Floyd_Warshall_Closure(dif_matrix, threshold, verbose=True)
    # print(f"After transitive closure: {modified_matrix.values}")
    # print(f"Changes made: {closure_info['total_changes']}")
    
    return {
        'item_index': item_index,
        'threshold': threshold,
        'cluster_df': cluster_df,
        'components': components,
        'graph': G,
        'edges': edges,
        'closure_info': closure_info,
        'num_clusters': num_clusters,
        'num_edges': num_edges,
        'cluster_sizes': cluster_sizes,
        'multi_group_clusters': multi_group_clusters
    }

def find_recommended_threshold_per_item(dif_matrix, groups, dif_type='DIF_a', 
                                      item_index=0, test_thresholds=None, verbose_closure=False):
    """
    Find the recommended threshold for a specific item
    """
    if test_thresholds is None:
        # Default threshold range
        test_thresholds = [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    
    all_results = {}
    for threshold in test_thresholds:
        result = analyze_item_threshold(dif_matrix, groups, threshold, dif_type, item_index, verbose_closure)
        all_results[threshold] = result
    
    # Create summary comparison
    summary_df = pd.DataFrame([
        {
            'Threshold': threshold,
            'Num_Clusters': result['num_clusters'],
            'Num_Edges': result['num_edges'],
            'Multi_Group_Clusters': len(result['multi_group_clusters']),
            'Largest_Cluster_Size': result['cluster_sizes'].max() if len(result['cluster_sizes']) > 0 else 0
        }
        for threshold, result in all_results.items()
    ])
    
    # Find recommended threshold
    meaningful_thresholds = summary_df[summary_df['Multi_Group_Clusters'] > 0]
    
    recommended_threshold = None
    if len(meaningful_thresholds) > 0:
        # Choose the smallest threshold that creates multi-group clusters
        recommended_threshold = meaningful_thresholds.iloc[0]['Threshold']
    else:
        # If no threshold creates clusters, use a middle value for visualization
        recommended_threshold = test_thresholds[len(test_thresholds)//2]
    
    return summary_df, recommended_threshold, all_results


def display_item_results(dif_matrix, groups, threshold, dif_type='DIF_a', item_index=0, verbose_closure=False):
    """Display detailed results for a specific item and threshold"""
    cluster_dict, components, G, edges, closure_info = connected_components_clustering(
        dif_matrix, groups, threshold, dif_type, verbose_closure)
    
    # Create results DataFrame
    cluster_df = pd.DataFrame({
        'Group': groups, 
        'Cluster': [cluster_dict[g] for g in groups]
    }).sort_values(by='Cluster')
    
    # Calculate statistics
    num_clusters = len(set(cluster_dict.values()))
    num_edges = G.number_of_edges()
    cluster_sizes = cluster_df['Cluster'].value_counts().sort_index()
    multi_group_clusters = cluster_sizes[cluster_sizes > 1]
    
    if verbose_closure:
        print(f"\n{'='*60}")
        print(f"ITEM {item_index + 1} - {dif_type.upper()} RESULTS (Threshold: {threshold})")
        print(f"{'='*60}")
        print(f"Total groups: {len(groups)}")
        print(f"Number of clusters: {num_clusters}")
        print(f"Number of edges (connections): {num_edges}")
        
       # Display transitive closure information
        if closure_info['total_changes'] > 0:
            print("\nTransitive Closure Applied:")
        print(f"  Iterations: {closure_info['iterations']}")
        print(f"  Changes made: {closure_info['total_changes']}")
        if not verbose_closure:
            print("  (Use verbose_closure=True to see detailed changes)")
    else:
        print("\nTransitive Closure: No changes needed (already transitive)")
    
    if edges:
        print(f"\nFinal Connections (DIF < {threshold}):")
        for group1, group2, dif_val in sorted(edges, key=lambda x: x[2]):
            print(f"  {group1} - {group2}: {dif_type} = {dif_val:.4f}")
    
    # Generate visualizations
    visualize_connected_components_per_item(G, cluster_dict, groups, threshold, dif_type, item_index)
    
    return cluster_dict, components, G, edges, closure_info

def DIF_Cluster_Components_Per_Item(dif_data, dif_type, test_thresholds=None,
                                   show_matrices=False, items_to_analyze=None,
                                   verbose_closure=False, verbose = False):
    """
    Analyze DIF clustering for each item individually.
    This modified version *does not* generate plots directly.
    It returns all the necessary data for external plotting.

    Parameters:
        dif_data: Dictionary containing DIF data
        dif_type: 'DIF_a' or 'DIF_b'
        test_thresholds: List of thresholds to test
        show_matrices: Whether to show heatmap matrices (still handled here, but could be moved out)
        items_to_analyze: List of item indices to analyze (None for all)
        verbose_closure: Whether to show detailed transitive closure changes
    
    Returns:
        dict: Complete analysis results including clustering data for plotting
    """
    df = dif_data[dif_type]

    if df.empty:
        print(f"\nNo data available for {dif_type}")
        return None

    # Extract groups
    groups = extract_groups_from_columns(df, dif_type)
    
    if verbose:
        print(f"Found {len(groups)} groups: {groups}")
        print(f"Total items to analyze: {len(df)}")

    # Determine which items to analyze
    if items_to_analyze is None:
        items_to_analyze = list(range(len(df)))

    all_item_results = {}

    # Analyze each item
    for item_idx in items_to_analyze:
        if verbose:
            print(f"\n{'-'*50}")
            print(f"ANALYZING ITEM {item_idx + 1}")
            print(f"{'-'*50}")

        # Create DIF matrix for this specific item
        dif_matrix = create_dif_matrix_per_item(df, groups, item_idx, dif_type)

        # Show matrix if requested (this can stay here as it's a specific matrix view)
        if show_matrices:
            visualize_dif_matrix_per_item(dif_matrix, dif_type, item_idx)

        # Find recommended threshold for this item
        summary_df, recommended_threshold, threshold_results = find_recommended_threshold_per_item(
            dif_matrix, groups, dif_type, item_idx, test_thresholds, verbose_closure)

        # Display threshold summary
        if verbose:
            print(f"\nThreshold Summary for Item {item_idx + 1}:")
            print(summary_df.to_string(index=False))

        # Perform the clustering with the recommended threshold
        cluster_dict, components, G, edges, closure_info = connected_components_clustering(
            dif_matrix, groups, recommended_threshold, dif_type, verbose_closure)

        # Store results, including data needed for later plotting
        all_item_results[item_idx] = {
            'dif_matrix': dif_matrix,
            'summary_df': summary_df,
            'recommended_threshold': recommended_threshold,
            'threshold_results': threshold_results,
            'final_clustering': {
                'cluster_dict': cluster_dict,
                'components': components,
                'graph': G,
                'edges': edges,
                'closure_info': closure_info
            },
            'groups': groups, # Add groups for plotting outside
            'dif_type': dif_type, # Add dif_type for plotting outside
            'item_index': item_idx # Add item_index for plotting outside
        }

    return {
        'groups': groups,
        'dif_type': dif_type,
        'items_analyzed': items_to_analyze,
        'item_results': all_item_results
    }

def create_summary_across_items(results, dif_type, show_low_dif=True, show_high_dif=False):
    """
    Create a summary showing clustering patterns across all items
    
    Parameters:
        results: Results from DIF_Cluster_Components_Per_Item
        dif_type: 'DIF_a' or 'DIF_b'
        show_low_dif: Whether to show items with low DIF probabilities (default: True)
        show_high_dif: Whether to show items with high DIF probabilities (default: False)
    """
    if results is None:
        return
    
    groups = results['groups']
    item_results = results['item_results']
    
    # Create summary table
    summary_data = []
    for item_idx, item_result in item_results.items():
        final_clustering = item_result['final_clustering']
        cluster_dict = final_clustering['cluster_dict']
        
        # Count clusters and connections
        num_clusters = len(set(cluster_dict.values()))
        num_connections = final_clustering['graph'].number_of_edges()
        
        # Find multi-group clusters
        cluster_sizes = pd.Series([cluster_dict[g] for g in groups]).value_counts()
        multi_group_clusters = cluster_sizes[cluster_sizes > 1]
        
        summary_data.append({
            'Item': item_idx,
            'Recommended_Threshold': item_result['recommended_threshold'],
            'Num_Clusters': num_clusters,
            'Num_Connections': num_connections,
            'Multi_Group_Clusters': len(multi_group_clusters),
            'Connected_Groups': ', '.join([f"{u}-{v}" for u, v, _ in final_clustering['edges']])
        })
    
    # Analyze connections grouped by item (LOW DIF - below threshold)
    items_with_connections = []
    items_with_high_dif = []  # NEW: for high DIF probabilities
    
    for item_idx, item_result in item_results.items():
        edges = item_result['final_clustering']['edges']
        dif_matrix = item_result['dif_matrix']
        threshold = item_result['recommended_threshold']
        
        # LOW DIF: Items that have connections (existing logic)
        if edges:
            item_connections = []
            for group1, group2, dif_val in edges:
                pair_key = tuple(sorted([group1, group2]))
                item_connections.append((pair_key, dif_val))
            items_with_connections.append((item_idx, item_connections))
        
        # NEW: HIGH DIF - pairs exceeding threshold
        high_dif_pairs = []
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                group1, group2 = groups[i], groups[j]
                dif_val = dif_matrix.iloc[i, j]
                
                # Check if DIF value exists and exceeds threshold
                if not np.isnan(dif_val) and dif_val >= threshold:
                    pair_key = tuple(sorted([group1, group2]))
                    high_dif_pairs.append((pair_key, dif_val))
        
        if high_dif_pairs:
            items_with_high_dif.append((item_idx, high_dif_pairs))
    
    # Print LOW DIF probabilities (controlled by toggle)
    if show_low_dif and items_with_connections:
        if dif_type == "DIF_a": 
            print(f"\n{'#'*60}")
            print("Items With Low DIF Probabilities of DIF On a Across Groups:")
            print(f"{'#'*60}")
        
        elif dif_type == "DIF_b": 
            print(f"\n{'#'*60}")
            print("Items With Low DIF Probabilities of DIF On b Across Groups:")
            print(f"{'#'*60}")
            
        # Sort items by number of connections (descending)
        items_with_connections.sort(key=lambda x: len(x[1]), reverse=True)
        
        for item_idx, connections in items_with_connections:
            # Sort connections by DIF value (ascending - lowest DIF first)
            connections_sorted = sorted(connections, key=lambda x: x[1])
            
            print(f"Item {item_idx + 1}: {len(connections)} connection(s)")
            for (group1, group2), dif_val in connections_sorted:
                print(f"  {group1} - {group2} = {dif_val:.4f}")
    elif show_low_dif:
        print("\nNo low DIF connections found across any items.")
    
    # Print HIGH DIF probabilities (controlled by toggle)
    if show_high_dif and items_with_high_dif:
        if dif_type == "DIF_a": 
            print(f"\n{'#'*60}")
            print("Items With High DIF Probabilities of DIF On a Across Groups:")
            print(f"{'#'*60}")
        
        elif dif_type == "DIF_b": 
            print(f"\n{'#'*60}")
            print("Items With High DIF Probabilities of DIF On b Across Groups:")
            print(f"{'#'*60}")
            
        # Sort items by number of high DIF pairs (descending)
        items_with_high_dif.sort(key=lambda x: len(x[1]), reverse=True)
        
        for item_idx, high_dif_pairs in items_with_high_dif:
            # Sort pairs by DIF value (descending - highest DIF first)
            pairs_sorted = sorted(high_dif_pairs, key=lambda x: x[1], reverse=True)
            
            print(f"Item {item_idx + 1}: {len(high_dif_pairs)} high DIF pair(s)")
            for (group1, group2), dif_val in pairs_sorted:
                print(f"  {group1} - {group2} = {dif_val:.4f}")
    elif show_high_dif:
        print("\nNo high DIF probabilities found across any items.")

def extract_connected_groups_simple(dif_data, items_to_analyze=None, test_thresholds=None):
    """
    Extract connected groups for each item with minimal output.
    
    Parameters:
        dif_data: Dictionary containing DIF data
        items_to_analyze: List of item indices to analyze (None for all)
        test_thresholds: List of thresholds to test (None for default)
    
    Returns:
        dict: Results with connected groups for each item and DIF type
    """
    results = {}
    
    for dif_type in ['DIF_a', 'DIF_b']:
        df = dif_data[dif_type]
        
        if df.empty:
            continue
            
        # Extract groups
        groups = extract_groups_from_columns(df, dif_type)
        
        # Determine which items to analyze
        if items_to_analyze is None:
            items_to_analyze_local = list(range(len(df)))
        else:
            items_to_analyze_local = items_to_analyze
        
        results[dif_type] = {
            'groups': groups,
            'items': {}
        }
        
        # Analyze each item
        for item_idx in items_to_analyze_local:
            # Create DIF matrix for this specific item
            dif_matrix = create_dif_matrix_per_item(df, groups, item_idx, dif_type)
            
            # Find recommended threshold for this item (silently)
            if test_thresholds is None:
                test_thresholds_local = [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
            else:
                test_thresholds_local = test_thresholds
            
            # Test thresholds to find the best one
            best_threshold = None
            for threshold in test_thresholds_local:
                cluster_dict, components, G, edges, closure_info = connected_components_clustering(
                    dif_matrix, groups, threshold, dif_type, verbose_closure=False)
                
                # Check if this threshold creates meaningful clusters
                cluster_sizes = pd.Series([cluster_dict[g] for g in groups]).value_counts()
                multi_group_clusters = cluster_sizes[cluster_sizes > 1]
                
                if len(multi_group_clusters) > 0:
                    best_threshold = threshold
                    break
            
            # If no threshold creates clusters, use middle value
            if best_threshold is None:
                best_threshold = test_thresholds_local[len(test_thresholds_local)//2]
            
            # Get final clustering with best threshold
            cluster_dict, components, G, edges, closure_info = connected_components_clustering(
                dif_matrix, groups, best_threshold, dif_type, verbose_closure=False)
            
            # Store results
            results[dif_type]['items'][item_idx] = {
                'threshold': best_threshold,
                'connected_components': list(components),
                'edges': edges,
                'cluster_dict': cluster_dict
            }
    
    return results


def print_connected_groups_simple(results):
    """
    Print only the connected groups for each item in a clean format.
    
    Parameters:
        results: Results from extract_connected_groups_simple()
    """
    for dif_type in ['DIF_a', 'DIF_b']:
        if dif_type not in results:
            continue
            
        print(f"\n{'='*60}")
        print(f"CONNECTED GROUPS - {dif_type.upper()}")
        print(f"{'='*60}")
        
        groups = results[dif_type]['groups']
        items = results[dif_type]['items']
        
        for item_idx in sorted(items.keys()):
            item_data = items[item_idx]
            components = item_data['connected_components']
            threshold = item_data['threshold']
            
            print(f"\nItem {item_idx + 1} (threshold: {threshold}):")
            
            if len(components) == 1 and len(components[0]) == len(groups):
                print("  All groups connected")
            elif len(components) == len(groups):
                print("  No groups connected (all isolated)")
            else:
                for i, component in enumerate(components):
                    if len(component) > 1:
                        component_list = sorted(list(component), key=lambda x: int(x.replace('Group', '')))
                        print(f"  Connected: {' ~ '.join(component_list)}")


def generate_mirt_constraints(results):
    """
    Generate MIRT constraint syntax for connected groups using INCLUSIVE syntax.
    Connected groups have no DIF, and should have equal parameters and be constrained together.
    
    Parameters:
        results: Results from extract_connected_groups_simple()
    
    Returns:
        dict: MIRT constraint strings organized by item
    """
    mirt_output = {}
    
    # Get all items that were analyzed
    all_items = set()
    for dif_type in ['DIF_a', 'DIF_b']:
        if dif_type in results:
            all_items.update(results[dif_type]['items'].keys())
    
    # Process each item
    for item_idx in sorted(all_items):
        mirt_output[item_idx] = {
            'a1_constraints': [],
            'd_constraints': []
        }
        
        # Process DIF_a (converts to a1 parameter)
        if 'DIF_a' in results and item_idx in results['DIF_a']['items']:
            dif_a_data = results['DIF_a']['items'][item_idx]
            groups = results['DIF_a']['groups']
            components = dif_a_data['connected_components']
            
            # Check if all groups are connected
            if len(components) == 1 and len(components[0]) == len(groups):
                # All groups connected - constrain all groups together
                mirt_output[item_idx]['a1_constraints'].append(f"CONSTRAINB = ({item_idx + 1}, a1)")
            else:
                # Generate constraints for each connected component with more than 1 group
                for component in components:
                    if len(component) > 1:
                        # Use INCLUSIVE syntax - list only the connected groups
                        connected_groups = sorted(list(component), key=lambda x: int(x.replace('Group', '')))
                        connected_str = ', '.join(connected_groups)
                        mirt_output[item_idx]['a1_constraints'].append(f"CONSTRAINB [{connected_str}] = ({item_idx + 1}, a1)")
                # Note: Groups not in any constraint remain freely estimated
        
        # Process DIF_b (converts to d parameter)
        if 'DIF_b' in results and item_idx in results['DIF_b']['items']:
            dif_b_data = results['DIF_b']['items'][item_idx]
            groups = results['DIF_b']['groups']
            components = dif_b_data['connected_components']
            
            # Check if all groups are connected
            if len(components) == 1 and len(components[0]) == len(groups):
                # All groups connected - constrain all groups together
                mirt_output[item_idx]['d_constraints'].append(f"CONSTRAINB = ({item_idx + 1}, d)")
            else:
                # Generate constraints for each connected component with more than 1 group
                for component in components:
                    if len(component) > 1:
                        # Use INCLUSIVE syntax - list only the connected groups
                        connected_groups = sorted(list(component), key=lambda x: int(x.replace('Group', '')))
                        connected_str = ', '.join(connected_groups)
                        mirt_output[item_idx]['d_constraints'].append(f"CONSTRAINB [{connected_str}] = ({item_idx + 1}, d)")
                # Note: Groups not in any constraint remain freely estimated
    
    return mirt_output


def print_mirt_constraints(mirt_output):
    """
    Generate MIRT constraint syntax as a formatted string.
    
    Parameters:
        mirt_output: Output from generate_mirt_constraints()
        
    Returns:
        str: Formatted constraint syntax string
    """
    result = "F = 1-10\n"
    
    for item_idx in sorted(mirt_output.keys()):
        constraints = mirt_output[item_idx]
        item_number = item_idx + 1
        
        # Check if this item has constraints with group specifications
        has_group_constraints = False
        constraint_lines = []
        
        # Process a1 constraints
        if constraints['a1_constraints']:
            for constraint in constraints['a1_constraints']:
                if '[' in constraint and ']' in constraint:
                    # Extract the part between brackets (e.g., "Group1, Group2")
                    start = constraint.find('[') + 1
                    end = constraint.find(']')
                    groups_str = constraint[start:end]
                    
                    # Convert 'GroupX' to 'X' and handle single group or empty string
                    group_ids = []
                    for g_name in groups_str.split(','):
                        g_name = g_name.strip()
                        if g_name.startswith('Group'):
                            group_ids.append(g_name.replace('Group', ''))
                    
                    # Join with comma and handle cases for single group or empty
                    if len(group_ids) > 1:
                        formatted_groups = ', '.join(group_ids)
                    elif len(group_ids) == 1:
                        formatted_groups = group_ids[0]
                    else: # This handles the 'empty' case for your example `[1, ]`
                        formatted_groups = ''
                        
                    constraint_lines.append(f"    CONSTRAINB[{formatted_groups}] = ({item_number}, a1)")
                    has_group_constraints = True
        
        # Process d constraints  
        if constraints['d_constraints']:
            for constraint in constraints['d_constraints']:
                if '[' in constraint and ']' in constraint:
                    # Extract the part between brackets (e.g., "Group1, Group2")
                    start = constraint.find('[') + 1
                    end = constraint.find(']')
                    groups_str = constraint[start:end]
                    
                    # Convert 'GroupX' to 'X' and handle single group or empty string
                    group_ids = []
                    for g_name in groups_str.split(','):
                        g_name = g_name.strip()
                        if g_name.startswith('Group'):
                            group_ids.append(g_name.replace('Group', ''))
                    
                    # Join with comma and handle cases for single group or empty
                    if len(group_ids) > 1:
                        formatted_groups = ', '.join(group_ids)
                    elif len(group_ids) == 1:
                        formatted_groups = group_ids[0]
                    else: # This handles the 'empty' case for your example `[1, ]`
                        formatted_groups = ''
                        
                    constraint_lines.append(f"    CONSTRAINB[{formatted_groups}] = ({item_number}, d)")
                    has_group_constraints = True
        
        # Only add constraints for items that have group-specific constraints
        if has_group_constraints:
            result += f"    # Group specific constraints for item {item_number}\n"
            for line in constraint_lines:
                result += line + "\n"
    
    return result

def save_mirt_constraints_to_file(mirt_constraints_string, filename="mirt_constraints.txt"):
    """
    Saves the generated MIRT constraint string to a specified text file.

    Args:
        mirt_constraints_string (str): The string containing the MIRT constraints.
        filename (str): The name of the file to save the constraints to.
    """
    try:
        with open(filename, 'w') as f:
            f.write(mirt_constraints_string)
    except IOError as e:
        print(f"Error saving MIRT constraints to file: {e}")


#%% Main Function Calls

def DIF_Detection(groups, sizes, 
                  loss = "binary_crossentropy", 
                  feature_selection=None,
                  save_results = False,
                  val_split = False,
                  merged=False):
    """
    Complete InterDIFNet pipeline for training and evaluating DIF detection models.

    This function provides an end-to-end pipeline for differential item functioning (DIF) 
    detection using neural networks. It supports multiple feature selection strategies,
    trains separate or merged models for uniform and non-uniform DIF detection, and evaluates
    performance on test datasets.

    Parameters:
    -----------
    groups : str
        String indicating number of groups for analysis 
        - "Ten"
        - "Three"
        - "Two"
    sizes : list
        List of sample sizes to evaluate on test sets:
        - Ten Group: [1000, 2000, 4000]  
        - Three Group: [250, 500, 1000]
        - Two Group: [125, 250, 500]
    feature_selection : str or None, optional
        Feature selection strategy:
        - "Ablation": Performs ablation study to identify optimal feature combinations
        - "TLP": Uses only TLP (Truncated Lasso Penalty) features
        - None: Uses all available features (default)
    merged : bool
        If True, use merged model architecture

    Returns:
    --------
    tuple
        If feature_selection == "Ablation":
            (ablation_results_df, selected_features, training_results, testing_results)
        If feature_selection == "TLP" or None:
            (training_results, testing_results)
    """

    if feature_selection == "Ablation":
        training_data, training_labels, feature_names, DIF_tests = load_training_data(groups=groups)
        y_set1, y_set2, set1_cols, set2_cols = prepare_label_sets(training_labels)

        ablation_results_df, selected_features, _ = run_ablation_study(groups=groups,
                                                                       training_data=training_data, 
                                                                       training_labels=training_labels, 
                                                                       n_outputs_per_set=training_labels.shape[1],
                                                                       loss=loss,
                                                                       DIF_tests=DIF_tests, 
                                                                       set1_cols=set1_cols, 
                                                                       set2_cols=set2_cols, 
                                                                       val_size=0.2, 
                                                                       n_repeats=3,
                                                                       merged=merged)

        training_results = complete_training_pipeline(groups=groups, loss=loss,
                                                      training_features=selected_features,
                                                      val_split=val_split,
                                                      merged=merged)

        testing_results = evaluate_models_on_test_sets(selected_features=training_results['feature_names'],
                                                       set1_cols=training_results['set1_cols'], 
                                                       set2_cols=training_results['set2_cols'], 
                                                       groups=groups,
                                                       model_dif_a=training_results['merged_model'] if merged else training_results['model_dif_a'], 
                                                       model_dif_b=None if merged else training_results['model_dif_b'], 
                                                       scaler=training_results['scaler'], 
                                                       opt_thr_a=training_results['opt_thr_a'], 
                                                       opt_thr_b=training_results['opt_thr_b'],
                                                       sizes=sizes,
                                                       save_results=save_results,
                                                       merged=merged)

        return ablation_results_df, selected_features, training_results, testing_results

    else:
        training_results = complete_training_pipeline(groups=groups, 
                                                      loss=loss, 
                                                      training_features="TLP" if feature_selection == "TLP" else None,
                                                      val_split=val_split,
                                                      merged=merged)

        testing_results = evaluate_models_on_test_sets(selected_features=training_results['feature_names'],
                                                       set1_cols=training_results['set1_cols'], 
                                                       set2_cols=training_results['set2_cols'], 
                                                       groups=groups,
                                                       model_dif_a=training_results['merged_model'] if merged else training_results['model_dif_a'], 
                                                       model_dif_b=None if merged else training_results['model_dif_b'], 
                                                       scaler=training_results['scaler'], 
                                                       opt_thr_a=training_results['opt_thr_a'], 
                                                       opt_thr_b=training_results['opt_thr_b'],
                                                       sizes=sizes,
                                                       save_results=save_results,
                                                       merged=merged)

        return training_results, testing_results


def Clustered_Components_DIF(groups, 
                             sizes, 
                             percentages=[20, 40],  
                             replications=range(1, 51), 
                             dif_types=['DIF_a', 'DIF_b'],
                             items_to_analyze=None,
                             test_thresholds=None,
                             show_matrices=False,
                             show_low_dif=False,
                             show_high_dif=False,
                             verbose_closure=False,
                             generate_plots=True,
                             print_constraints=True,
                             verbose=False,
                             save_mirt_constraints = True):
    """
    Complete DIF clustering analysis pipeline that loads data, performs clustering,
    creates visualizations, and generates MIRT constraints across various
    sample sizes, percentages of DIF items, and replications.

    Parameters:
        groups (int): The number of groups involved in the DIF analysis.
        sizes (list of int): A list of sample sizes (N) to iterate through for data loading and analysis.
        percentages (list of int, optional): A list of percentages (P) representing the proportion of DIF items to simulate. Defaults to [20, 40].
        replications (range or list of int, optional): A range or list of replication numbers (R) for the simulation. Defaults to range(1, 51).
        dif_types (list of str, optional): A list specifying the types of Differential Item Functioning (DIF) to analyze. Options are 'DIF_a' and 'DIF_b'. Defaults to ['DIF_a', 'DIF_b'].
        items_to_analyze (list of int, optional): A list of specific item indices to include in the analysis. If `None`, all items are analyzed. Defaults to None.
        test_thresholds (list of float, optional): A list of threshold values to test for determining DIF clusters. If `None`, default thresholds will be used within `DIF_Cluster_Components_Per_Item`. Defaults to None.
        show_matrices (bool, optional): If `True`, displays heatmaps of the DIF matrices during the clustering process. Defaults to False.
        show_low_dif (bool, optional): If `True`, includes items with low DIF probabilities in the summary output. Defaults to False.
        show_high_dif (bool, optional): If `True`, includes items with high DIF probabilities in the summary output. Defaults to False.
        verbose_closure (bool, optional): If `True`, displays detailed information about the transitive closure during clustering. Defaults to False.
        generate_plots (bool, optional): If `True`, generates and displays visualization plots for the clustering results. Defaults to True.
        print_constraints (bool, optional): If `True`, prints the generated MIRT (Multidimensional Item Response Theory) constraint syntax to the console. Defaults to True.
        verbose (bool, optional): If `True`, enables verbose output, including connection summaries and MIRT constraint summaries. Defaults to False.
        save_mirt_constraints (bool, optional): If `True`, saves the generated MIRT constraint syntax to a text file for each replication. Defaults to True.

    Returns:
        Saves a txt file with MIRT constraints to current working directory
    """
    for n in sizes: 
        for p in percentages:            
            print(f"\nProcessing - N: {n}, DIF Percent: {p}%")
            for r in replications:
        
                try:
                    dif_data = load_dif_data(groups, n, p, r)
                    
                except Exception:
                    continue  # Skip to next replication
                
                # Perform clustering analysis
                detailed_results = {}
                
                # Perform clustering analysis for each DIF type
                for dif_type in dif_types:
                   if not dif_data[dif_type].empty:
                       result = DIF_Cluster_Components_Per_Item(
                           dif_data, dif_type,
                           test_thresholds=test_thresholds,
                           show_matrices=show_matrices, # This still controls the matrix heatmap
                           items_to_analyze=items_to_analyze,
                           verbose_closure=verbose_closure
                       )
                       detailed_results[dif_type] = result
                       # if verbose:
                       # # Create summary across items
                       #     create_summary_across_items(result, dif_type,
                       #                                 show_low_dif=show_low_dif,
                       #                                 show_high_dif=show_high_dif)

                       if generate_plots and result is not None:
                           print(f"\nGenerating plots for {dif_type}...")
                           for item_idx, item_res in result['item_results'].items():
                               print(f"  Plotting Item {item_idx + 1}...")
                               # Re-extract necessary components for plotting
                               G = item_res['final_clustering']['graph']
                               cluster_dict = item_res['final_clustering']['cluster_dict']
                               groups_for_plot = item_res['groups']
                               threshold_for_plot = item_res['recommended_threshold']
                               dif_type_for_plot = item_res['dif_type']

                               visualize_connected_components_per_item(
                                   G, cluster_dict, groups_for_plot, threshold_for_plot,
                                   dif_type=dif_type_for_plot, item_index=item_idx
                               )
                   else:
                       print(f"Skipping analysis for {dif_type} as data is empty for N={n}, P={p}, R={r}")

                # Extract connected groups (simplified)
              
                connected_results = extract_connected_groups_simple(
                    dif_data, 
                    items_to_analyze=items_to_analyze,
                    test_thresholds=test_thresholds
                )
                
                # Print connected groups summary
                if verbose:
                    print_connected_groups_simple(connected_results)
                
                # Generate MIRT constraints
                mirt_constraints_dict = generate_mirt_constraints(connected_results) # Use a new variable name
                
                if print_constraints:
                    mirt_constraints_string_to_print = print_mirt_constraints(mirt_constraints_dict) 
                    print(mirt_constraints_string_to_print)
                
                if save_mirt_constraints:
                    if not print_constraints:
                         mirt_constraints_string_to_print = print_mirt_constraints(mirt_constraints_dict)
                    
                    filename = f"MIRT_constraints_{groups}_{n}_{p}_Replication{r}.txt"
                    save_mirt_constraints_to_file(mirt_constraints_string_to_print, filename)
                
                # Count items with connections
                connection_summary = {}
                
                for dif_type in dif_types:
                    if dif_type in connected_results:
                        items_data = connected_results[dif_type]['items']
                        items_with_connections = 0
                        total_connections = 0
                        
                        for item_idx, item_data in items_data.items():
                            if item_data['edges']:  # Has connections
                                items_with_connections += 1
                                total_connections += len(item_data['edges'])
                        
                        connection_summary[dif_type] = {
                            'total_items': len(items_data),
                            'items_with_connections': items_with_connections,
                            'total_connections': total_connections
                        }
                
                if verbose:
                    print("Connection Summary:")
                    for dif_type, summary in connection_summary.items():
                        print(f"  {dif_type}:")
                        print(f"    Total items analyzed: {summary['total_items']}")
                        print(f"    Items with connections: {summary['items_with_connections']}")
                        print(f"    Total group connections: {summary['total_connections']}")
                
                # Count MIRT constraints
                total_constraints = 0
                items_with_constraints = 0
                
                for item_idx, constraints in mirt_constraints_dict.items():
                    item_constraint_count = len(constraints['a1_constraints']) + len(constraints['d_constraints'])
                    if item_constraint_count > 0:
                        items_with_constraints += 1
                        total_constraints += item_constraint_count
                
                if verbose:
                    print("\nMIRT Constraint Summary:")
                    print(f"  Items requiring constraints: {items_with_constraints}")
                    print(f"  Total constraints generated: {total_constraints}")
                    
                    print("\n" + "="*80)
                    print("ANALYSIS COMPLETE")
                    print("="*80)
            