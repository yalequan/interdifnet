#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIF Model Utilities
Functions for training, evaluating, and applying DIF detection models.

Created on Mon Dec 23 18:29:29 2024
@author: yalequan
"""

# Auto-install and import required packages
import sys
import subprocess
import importlib.util


def _check_and_install_package(package_name, import_name=None):
    """
    Check if a package is installed, and install it if not.
    
    Parameters:
    -----------
    package_name : str
        The name of the package to install (used with pip)
    import_name : str, optional
        The name to use when importing (if different from package_name)
    """
    if import_name is None:
        import_name = package_name
    
    # Check if package is already installed
    if importlib.util.find_spec(import_name) is None:
        print(f"\n{'='*60}")
        print(f"Package '{package_name}' not found.")
        print(f"Installing '{package_name}'...")
        print(f"{'='*60}\n")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            print(f"  Successfully installed '{package_name}'\n")
        except subprocess.CalledProcessError as e:
            print(f"  Error installing '{package_name}': {e}")
            print(f"Please install manually: pip install {package_name}")
            raise


# Define required packages with their install and import names
_REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("tensorflow", "tensorflow"),
    ("scikit-learn", "sklearn"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("scikit-multilearn", "skmultilearn"),
    ("networkx", "networkx"),
]

# Check and install all required packages
print("Checking InterDIFNet dependencies...")
for package_name, import_name in _REQUIRED_PACKAGES:
    _check_and_install_package(package_name, import_name)
print("All dependencies are ready.\n")

# Now import all required packages (suppressing linter warnings about import order)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import glob  # noqa: E402
from tensorflow.keras import Input  # noqa: E402
from tensorflow.keras.models import Sequential, Model  # noqa: E402
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  # noqa: E402
from tensorflow.keras.optimizers import Adam  # noqa: E402
from tensorflow.keras.callbacks import EarlyStopping  # noqa: E402
from tensorflow.keras.regularizers import l2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from skmultilearn.model_selection import iterative_train_test_split  # noqa: E402
import os  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from tensorflow.keras.metrics import AUC  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402
from sklearn.utils import shuffle  # noqa: E402
from itertools import combinations  # noqa: E402
import seaborn as sns  # noqa: E402
import tensorflow.keras.backend as K  # noqa: E402
import gc  # noqa: E402
import tensorflow as tf  # noqa: E402
from pathlib import Path  # noqa: E402

# Loss Function

def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = K.pow((1 - p_t), gamma)
        return -K.mean(alpha_factor * modulating_factor * K.log(p_t))
    return loss


# Model Creation
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

# Data Pre-Processing
def load_training_data(groups, training_features=None, 
                       replications=500, 
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
    labels_pattern : str
        Glob pattern for training labels files
    merged : bool
        If True, creates a dict for y to be used in the merged models
    
    Returns:
    --------
    tuple
        (training_data, training_labels, feature_names, DIF_tests)
    """
    
    # Create the folder path dynamically based on groups parameter
    folder_name = f"{groups}_Group_Datasets"
    print(f"Loading data from folder: {folder_name}")
    
    # Set path to folder
    data_folder = Path(folder_name)
    
    # Construct file patterns
    data_pattern = f"{groups}_Group_Training_data_ALL_Replication*.csv"
    labels_pattern = f"{groups}_Group_Training_labels_Replication*.csv"
                       
    # Construct full file patterns with the folder path
    data_file_pattern = data_folder / data_pattern
    labels_file_pattern = data_folder / labels_pattern
    
    # Debug Code
    # print(f"Data Folder exists: {data_folder.exists()}")
    # print(f"Files in folder: {list(data_folder.glob('*.csv')) if data_folder.exists() else 'Folder not found'}")
    # print(f"Current working directory: {os.getcwd()}")
    # current_dir = Path('.')
    # folders = [item for item in current_dir.iterdir() if item.is_dir()]
    # print(f"Folders in current directory: {folders}")
    # group_folders = [item for item in current_dir.iterdir() if item.is_dir() and 'Group' in item.name]
    # print(f"Folders containing 'Group': {group_folders}")

    # Load training data
    training_csv_files = sorted(glob.glob(str(data_file_pattern)), 
                               key=lambda x: int(x.split('Replication')[1].split('.')[0]))
    training_csv_files = training_csv_files[:replications]
    training_data = pd.concat((pd.read_csv(file) for file in training_csv_files), ignore_index=True)
    
    # Load training labels
    training_labels_csv_files = sorted(glob.glob(str(labels_file_pattern)), 
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

# InterDIFNet Training

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
    print("Starting Non-Uniform DIF threshold search...")
    opt_thr_a, macro_f1_a, _, _ = macro_f1_vs_threshold(
        y_val_set1, val_pred_a, set_name="Non-Uniform DIF", plotting=True
    )
    print("Finished Non-Uniform DIF threshold search.")
    
    print("Starting Uniform DIF threshold search...")
    opt_thr_b, macro_f1_b, _, _ = macro_f1_vs_threshold(
        y_val_set2, val_pred_b, set_name="Uniform DIF", plotting=True
    )
    print("Finished Uniform DIF threshold search.")
    
    print(f"\nNon-Uniform DIF Threshold: {opt_thr_a:.3f}")
    print(f"\nUniform DIF Threshold: {opt_thr_b:.3f}")

    print("Saving thresholds now...")

    threshold_df = pd.DataFrame([{
        'Threshold_a': opt_thr_a,
        'Threshold_b': opt_thr_b
    }])

    #Save optimal thresholds 
    output_filename = "Optimal_Thresholds.csv"
    threshold_df.to_csv(output_filename, index=False)

    return opt_thr_a, opt_thr_b


def complete_training_pipeline(groups, loss, replications=500, epochs=200, batch_size=32, 
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

# DIF Ablation Functions
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

# DIF Detection on Testing Data for Simulation Study
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
    if save_results:
        print("Saving Probabilities")
        threshold_df = pd.DataFrame([{
            'Threshold_a': opt_thr_a,
            'Threshold_b': opt_thr_b
            }])
        thresholds_filename = f"Optimal_Thresholds_{groups}.csv"
        threshold_df.to_csv(thresholds_filename, index=False)
                
    DIF_summary_results = []
    
    # Create the folder path
    folder_name = f"{groups}_Group_Datasets"
    data_folder = Path(folder_name)

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
                # Construct full file paths with folder
                filename_testing = data_folder / f"{groups}_Group_Testing_data_ALL_{n}_{p}_Replication{r}.csv"
                filename_testing_labels = data_folder / f"{groups}_Group_Testing_data_labels_{n}_{p}_Replication{r}.csv"
            
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

                prob_preds_full = pd.concat([temp_results_dif_a, temp_results_dif_b], axis=1)
                prob_preds_full = prob_preds_full[testing_labels.columns]
                prob_preds_full.columns = testing_labels.columns

                binary_preds_a = (temp_results_dif_a > opt_thr_a).astype(int)
                binary_preds_b = (temp_results_dif_b > opt_thr_b).astype(int)

                predicted_DIF_labels = pd.concat([binary_preds_a, binary_preds_b], axis=1)
                predicted_DIF_labels = predicted_DIF_labels[testing_labels.columns]
                predicted_DIF_labels.columns = testing_labels.columns

                if save_results:
                    output_filename = f"Classification_Results_{groups}_{n}_{p}_Replication{r}.csv"
                    prob_preds_full.to_csv(output_filename, index=False)
  
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

# Model Saving Functions

def save_trained_model(training_results, groups, model_dir="./models/", merged=False, model_name=None):
    """Save trained DIF detection models and metadata to disk.
    
    Args:
        training_results (dict): Dictionary containing trained models, scaler, thresholds, 
            and metadata
        groups (str): Number of groups (e.g., "Ten", "Three", "Two")
        model_dir (str, optional): Directory path where models will be saved. 
            Defaults to "./models/".
        merged (bool, optional): Whether the model is a merged architecture. 
            Defaults to False.
        model_name (str, optional): Custom name for the saved model. If None, uses 
            "{groups}_Group" format. Defaults to None.
    
    Returns:
        list: List of saved file paths
    """
    from pathlib import Path
    import pickle
    from datetime import datetime
    
    # Create directory if it doesn't exist
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Determine base filename
    base_name = model_name if model_name else f"{groups}_Group"
    
    saved_files = []
    
    # Save model(s)
    if merged:
        model_file = model_path / f"{base_name}_merged_model.keras"
        training_results['merged_model'].save(model_file)
        saved_files.append(str(model_file))
        print(f"Saved merged model: {model_file}")
    else:
        model_a_file = model_path / f"{base_name}_model_a.keras"
        model_b_file = model_path / f"{base_name}_model_b.keras"
        training_results['model_dif_a'].save(model_a_file)
        training_results['model_dif_b'].save(model_b_file)
        saved_files.extend([str(model_a_file), str(model_b_file)])
        print(f"Saved DIF_a model: {model_a_file}")
        print(f"Saved DIF_b model: {model_b_file}")
    
    # Prepare metadata bundle
    metadata = {
        'scaler': training_results['scaler'],
        'opt_thr_a': training_results['opt_thr_a'],
        'opt_thr_b': training_results['opt_thr_b'],
        'feature_names': training_results['feature_names'],
        'set1_cols': training_results['set1_cols'],
        'set2_cols': training_results['set2_cols'],
        'loss': training_results['loss'],
        'merged': training_results['merged'],
        'groups': groups,
        'model_name': model_name,
        'keras_version': tf.__version__,
        'save_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'val_ratio': training_results.get('val_ratio', 0.2)
    }
    
    # Save metadata
    metadata_file = model_path / f"{base_name}_metadata.pkl"
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    saved_files.append(str(metadata_file))
    print(f"Saved metadata: {metadata_file}")
    
    print(f"\nAll files saved to: {model_path.absolute()}")
    return saved_files


def load_trained_model(groups, model_dir="./models/", merged=False, model_name=None):
    """Load trained DIF detection models and metadata from disk.
    
    Args:
        groups (str): Number of groups (e.g., "Ten", "Three", "Two")
        model_dir (str, optional): Directory path where models are saved. 
            Defaults to "./models/".
        merged (bool, optional): Whether to load a merged architecture model. 
            Defaults to False.
        model_name (str, optional): Custom name of the saved model. If None, uses 
            "{groups}_Group" format. Defaults to None.
    
    Returns:
        dict: Dictionary with same structure as training_results from training pipeline
    
    Raises:
        FileNotFoundError: If model directory or required files don't exist
    """
    from pathlib import Path
    import pickle
    from tensorflow.keras.models import load_model as keras_load_model
    
    model_path = Path(model_dir)
    
    # Determine base filename
    base_name = model_name if model_name else f"{groups}_Group"
    
    # Check if directory exists
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_path.absolute()}\n"
            f"Please train a model first or check the path."
        )
    
    # Load metadata
    metadata_file = model_path / f"{base_name}_metadata.pkl"
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_file}\n"
            f"Expected file: {base_name}_metadata.pkl"
        )
    
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Loading model from: {model_path.absolute()}")
    print(f"Model saved on: {metadata['save_date']}")
    print(f"Keras version: {metadata['keras_version']}")
    
    # Load model(s)
    training_results = {}
    
    if merged:
        model_file = model_path / f"{base_name}_merged_model.keras"
        if not model_file.exists():
            raise FileNotFoundError(
                f"Merged model file not found: {model_file}\n"
                f"Expected file: {base_name}_merged_model.keras"
            )
        training_results['merged_model'] = keras_load_model(model_file, custom_objects={'loss': focal_loss()})
        training_results['model_dif_a'] = None
        training_results['model_dif_b'] = None
        print(f"Loaded merged model: {model_file.name}")
    else:
        model_a_file = model_path / f"{base_name}_model_a.keras"
        model_b_file = model_path / f"{base_name}_model_b.keras"
        
        if not model_a_file.exists() or not model_b_file.exists():
            raise FileNotFoundError(
                f"Model files not found. Expected:\n"
                f"  - {model_a_file}\n"
                f"  - {model_b_file}"
            )
        
        training_results['model_dif_a'] = keras_load_model(model_a_file, custom_objects={'loss': focal_loss()})
        training_results['model_dif_b'] = keras_load_model(model_b_file, custom_objects={'loss': focal_loss()})
        training_results['merged_model'] = None
        print(f"Loaded DIF_a model: {model_a_file.name}")
        print(f"Loaded DIF_b model: {model_b_file.name}")
    
    # Add metadata to results
    training_results.update(metadata)
    
    print("Model loaded successfully\n")
    return training_results


# Main Function Calls

def train_InterDIFNet(groups,
                      loss = "binary_crossentropy", 
                      feature_selection=None,
                      val_split = False,
                      merged=False,
                      save_model=False,
                      model_dir="./models/",
                      model_name=None):
    """Complete InterDIFNet pipeline for training.

    This function provides an end-to-end pipeline for differential item functioning (DIF) 
    detection using neural networks. It supports multiple feature selection strategies,
    trains separate or merged models for uniform and non-uniform DIF detection.

    Args:
        groups (str): Number of groups for analysis ("Ten", "Three", or "Two")
        loss (str, optional): Loss function ("binary_crossentropy" or "focal_loss"). 
            Defaults to "binary_crossentropy".
        feature_selection (str, optional): Feature selection strategy:
            - "Ablation": Performs ablation study to identify optimal feature combinations
            - "TLP": Uses only TLP (Truncated Lasso Penalty) features
            - None: Uses all available features
            Defaults to None.
        save_results (bool, optional): Whether to save threshold results to CSV. 
            Defaults to False.
        val_split (bool, optional): Whether to use validation split during training. 
            Defaults to False.
        merged (bool, optional): If True, use merged model architecture. Defaults to False.
        save_model (bool, optional): If True, save trained model to disk. Defaults to False.
        model_dir (str, optional): Directory path where models will be saved. 
            Defaults to "./models/".
        model_name (str, optional): Custom name for the saved model. If None, uses 
            "{groups}_Group" format. Defaults to None.

    Returns:
        dict: training_results dictionary containing models, scaler, thresholds, and metadata
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
        
        if save_model:
            save_trained_model(training_results, groups, model_dir=model_dir, merged=merged, model_name=model_name)
    
        return ablation_results_df, selected_features, training_results
    
    else:
        training_results = complete_training_pipeline(groups=groups, 
                                                      loss=loss, 
                                                      training_features="TLP" if feature_selection == "TLP" else None,
                                                      val_split=val_split,
                                                      merged=merged)
        
        if save_model:
            save_trained_model(training_results, groups, model_dir=model_dir, merged=merged, model_name=model_name)
    
        return training_results
    
def DIF_Detection(data_filename, model_name=None,
                  verbose=False, save_results=False,
                  merged=False, output_filename=None,
                  model_dir="./models/"):
    """Apply trained DIF detection models to empirical dataset.
    
    Automatically detects the number of groups from the saved model metadata.
    If model_name is not provided and only one model exists in model_dir, it will
    be loaded automatically. If multiple models exist, model_name must be specified.

    Args:
        data_filename (str): Path to the empirical data CSV file
        model_name (str, optional): Name of the saved model to load. If None and only
            one model exists in model_dir, uses that model automatically. If multiple
            models exist, this parameter is required. Defaults to None.
        verbose (bool, optional): Controls printing output. Defaults to False.
        save_results (bool, optional): Whether to save results to CSV. Defaults to False.
        merged (bool, optional): Whether model is a merged architecture. Defaults to False.
        output_filename (str, optional): Custom filename for saved results. 
            If None, auto-generates name. Defaults to None.
        model_dir (str, optional): Directory to load model from. Defaults to "./models/".

    Returns:
        dict: Dictionary containing:
            - probabilities_a: DataFrame with DIF_a probabilities
            - probabilities_b: DataFrame with DIF_b probabilities  
            - predictions: DataFrame with binary predictions (thresholded)
            - all_probabilities: DataFrame with all probabilities combined
    """
    import pandas as pd
    import numpy as np
    import os
    from pathlib import Path
    import pickle
    
    # Auto-detect model if not specified
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_path.absolute()}\n"
            f"Please train a model first or check the path."
        )
    
    # Find metadata files in directory
    metadata_files = list(model_path.glob("*_metadata.pkl"))
    
    if model_name is None:
        if len(metadata_files) == 0:
            raise FileNotFoundError(
                f"No model metadata files found in {model_path.absolute()}\n"
                f"Please train and save a model first."
            )
        elif len(metadata_files) == 1:
            # Auto-detect the single model
            metadata_file = metadata_files[0]
            model_name = metadata_file.stem.replace("_metadata", "")
            print(f"Auto-detected model: {model_name}")
        else:
            # Multiple models found, require user to specify
            available_models = [f.stem.replace("_metadata", "") for f in metadata_files]
            raise ValueError(
                f"Multiple models found in {model_path.absolute()}\n"
                f"Available models: {available_models}\n"
                f"Please specify model_name parameter."
            )
    
    # Load metadata to extract groups and merged info
    metadata_file = model_path / f"{model_name}_metadata.pkl"
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_file}\n"
            f"Expected file: {model_name}_metadata.pkl"
        )
    
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    
    groups = metadata['groups']
    merged = metadata['merged']
    
    print(f"Loading trained model: {model_name}")
    print(f"Model groups: {groups}")
    print(f"Model architecture: {'Merged' if merged else 'Separate'}")
    
    # Load the full model
    training_results = load_trained_model(groups, model_dir=model_dir, merged=merged, model_name=model_name)
    
    selected_features=training_results['feature_names']
    set1_cols=training_results['set1_cols']
    set2_cols=training_results['set2_cols'] 
    model_dif_a=training_results['merged_model'] if merged else training_results['model_dif_a'] 
    model_dif_b=None if merged else training_results['model_dif_b'] 
    scaler=training_results['scaler']
    opt_thr_a=training_results['opt_thr_a'] 
    opt_thr_b=training_results['opt_thr_b']
    
    # Check if file exists
    if not os.path.exists(data_filename):
        raise FileNotFoundError(f"Data file not found: {data_filename}")
    
    # Load data
    data = pd.read_csv(data_filename)
    
    if verbose:
        print(f"Testing data shape: {data.shape}")
    
    # Filter features if specified
    if selected_features is not None:
        data = data[selected_features]
        if verbose:
            print(f"Filtered data shape: {data.shape}")
    
    # Clean data (remove NaN values)
    data_clean = data.dropna()
    
    # Scale the data
    X_scaled = pd.DataFrame(scaler.transform(data_clean.to_numpy()))
    X_scaled.columns = data_clean.columns
    
    # Get predictions
    if merged:
        predictions = model_dif_a.predict(X_scaled, verbose=0)
        raw_predictions_a = predictions[0]
        raw_predictions_b = predictions[1]
    else:
        raw_predictions_a = model_dif_a.predict(X_scaled, verbose=0)
        raw_predictions_b = model_dif_b.predict(X_scaled, verbose=0)
    
    # Reshape predictions
    n_samples = X_scaled.shape[0]
    n_outputs_a = len(set1_cols)
    n_outputs_b = len(set2_cols)
    
    raw_predictions_a = np.array(raw_predictions_a).reshape(n_samples, n_outputs_a)
    raw_predictions_b = np.array(raw_predictions_b).reshape(n_samples, n_outputs_b)
    
    # Create probability DataFrames
    probabilities_a = pd.DataFrame(raw_predictions_a, columns=set1_cols, index=data_clean.index)
    probabilities_b = pd.DataFrame(raw_predictions_b, columns=set2_cols, index=data_clean.index)
    
    if verbose:
        print(f"DIF_a probabilities shape: {probabilities_a.shape}")
        print(f"DIF_b probabilities shape: {probabilities_b.shape}")
        print(f"DIF_a probability range: [{probabilities_a.values.min():.3f}, {probabilities_a.values.max():.3f}]")
        print(f"DIF_b probability range: [{probabilities_b.values.min():.3f}, {probabilities_b.values.max():.3f}]")
    
    # Apply thresholds to get binary predictions
    binary_predictions_a = (probabilities_a > opt_thr_a).astype(int)
    binary_predictions_b = (probabilities_b > opt_thr_b).astype(int)
    
    # Combine binary predictions
    binary_predictions = pd.concat([binary_predictions_a, binary_predictions_b], axis=1)
    binary_predictions.insert(0, "Item", range(1,len(binary_predictions)+1))
    
    # Combine all probabilities
    all_probabilities = pd.concat([probabilities_a, probabilities_b], axis=1)
    all_probabilities.insert(0, "Item", range(1,len(all_probabilities)+1))
    
    
    # Save results if requested
    if save_results:
        if output_filename is None:
            base_name = Path(data_filename).stem
            prob_filename = f"DIF_Probabilities_{base_name}.csv"
            pred_filename = f"DIF_Predictions_{base_name}.csv"
            print(f"Files saved as: {prob_filename} and {pred_filename}")
        else:
            base_name = Path(output_filename).stem
            prob_filename = f"{base_name}_probabilities.csv"
            pred_filename = f"{base_name}_predictions.csv"
            print(f"Files saved as: {prob_filename} and {pred_filename}")
        
        # Save separate files
        all_probabilities.to_csv(prob_filename, index=False)
        binary_predictions.to_csv(pred_filename, index=False)
          
    return {
        'probabilities_a': probabilities_a,
        'probabilities_b': probabilities_b,
        'predictions': binary_predictions,
        'all_probabilities': all_probabilities
    }

def Simulation_Study(groups, sizes, 
                  loss = "binary_crossentropy", 
                  feature_selection=None,
                  save_results = False,
                  val_split = False,
                  merged=False,
                  save_model=False,
                  model_dir="./models/",
                  load_existing=False,
                  model_name=None):
    """Complete InterDIFNet pipeline for training and evaluating DIF detection models.

    This function provides an end-to-end pipeline for differential item functioning (DIF) 
    detection using neural networks. It supports multiple feature selection strategies,
    trains separate or merged models for uniform and non-uniform DIF detection, and evaluates
    performance on test datasets.

    Args:
        groups (str): Number of groups for analysis ("Ten", "Three", or "Two")
        sizes (list): Sample sizes to evaluate on test sets
            - Ten Group: [1000, 2000, 4000]  
            - Three Group: [250, 500, 1000]
            - Two Group: [125, 250, 500]
        loss (str, optional): Loss function ("binary_crossentropy" or "focal_loss"). 
            Defaults to "binary_crossentropy".
        feature_selection (str, optional): Feature selection strategy:
            - "Ablation": Performs ablation study to identify optimal feature combinations
            - "TLP": Uses only TLP (Truncated Lasso Penalty) features
            - None: Uses all available features
            Defaults to None.
        save_results (bool, optional): Whether to save threshold and test results to CSV. 
            Defaults to False.
        val_split (bool, optional): Whether to use validation split during training. 
            Defaults to False.
        merged (bool, optional): If True, use merged model architecture. Defaults to False.
        save_model (bool, optional): If True, save trained model to disk. Defaults to False.
        model_dir (str, optional): Directory path for saving/loading models. 
            Defaults to "./models/".
        load_existing (bool, optional): If True, load existing model from model_dir instead 
            of training. Defaults to False.
        model_name (str, optional): Custom name for the saved model. If None, uses 
            "{groups}_Group" format. Defaults to None.

    Returns:
        tuple: 
            - If feature_selection == "Ablation": 
              (ablation_results_df, selected_features, training_results, testing_results)
            - If feature_selection == "TLP" or None: 
              (training_results, testing_results)
    """

    # Handle load_existing flag
    if load_existing:
        print(f"Loading existing model for {groups} groups...")
        training_results = load_trained_model(groups, model_dir=model_dir, merged=merged, model_name=model_name)
        
        # Skip to testing
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
    
    # Train model(s)
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
        
        if save_model:
            save_trained_model(training_results, groups, model_dir=model_dir, merged=merged, model_name=model_name)

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
        
        if save_model:
            save_trained_model(training_results, groups, model_dir=model_dir, merged=merged, model_name=model_name)

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