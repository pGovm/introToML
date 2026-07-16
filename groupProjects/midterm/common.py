"""
This is a python file contains common libraries and helper functions that all the classifier notebooks share
"""

# Importing required libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Setting up the local system's base path
BASE_DIR = Path(__file__).parent
common_path = BASE_DIR / "output"
os.makedirs(os.path.dirname(common_path), exist_ok=True)

###################################
##### Common Helper Functions #####
###################################

def load_cancer_ds():
    """
    Load the cancer dataset
    """

    breast_cancer_ds = load_breast_cancer()

    # Feature data
    x = breast_cancer_ds.data
    
    # Whether cancer is Malignant(0) or Benign(1)
    y = 1 - breast_cancer_ds.target

    # Feature names
    class_names = ["Malignant", "Benign"]

    return x, y, class_names


def scale_data(X_train, X_test, scaler=False):
    """
    Scales and standardizes training and testing data
    """

    if not scaler:
        return X_train, X_test
    
    std = StandardScaler()

    X_train_std = std.fit_transform(X_train)
    X_test_std = std.transform(X_test)

    return X_train_std, X_test_std

def data_preprocess(scaler):
    """
    Function to load dataset, split it, and scale it (if desired)

    Ensures all classifiers have the same processed dataset
    """

    x, y, class_names = load_cancer_ds()

    X_train_raw, X_test_raw, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_test = scale_data(X_train_raw, X_test_raw, scaler)

    return X_train, X_test, Y_train, Y_test, class_names 
    
def train_and_eval_model(classifier, X_train, X_test, Y_train, Y_test):
    """
    Trains model based on classifier passed and returns model and its metrics
    """

    model = classifier.fit(X_train, Y_train)
    pred_Y = model.predict(X_test)

    metrics = {
        "Testing Accuracy" :   accuracy_score(Y_test, pred_Y),
        "Model Precision":   precision_score(Y_test, pred_Y, average="binary"),
        "Model Recall"   :   recall_score(Y_test, pred_Y, average="binary"),
        "F1 Score"       :   f1_score(Y_test, pred_Y, average="binary")
    }

    return model, pred_Y, metrics

def _cm(true, pred):
    """
    Getter function for computing the confusion matrix
    """
    return confusion_matrix(true, pred)

def print_metrics(metrics, title):

    """
    Prints resultant model metrics
    """    
    print("\n")
    print("=" * 80)
    print(f'{title}')
    print("=" * 80)

    for k, v in metrics.items():
        print(f'{k:<10}:    {v:.4f}')

####################################
##### Common Plotter Functions #####
####################################

def plot_cm(Y_test, pred_Y, class_names, title, path):
    """
    Plots the confusion matrix

    Title should be entered by assuming "Confusion Matrix of" is prepended to it
    """

    cm = _cm(Y_test, pred_Y)

    plt.figure(figsize=(8,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f'Confusion Matrix of {title}')
    plt.tight_layout()

    full_path = common_path / path
    os.makedirs(os.path.dirname(full_path.parent), exist_ok=True)
    plt.savefig(full_path, dpi=200, bbox_inches="tight")

    plt.show()