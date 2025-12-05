# utils.py
import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Load preprocessed dataset
# -----------------------------
def load_dataset(path="data/processed/preprocessed.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)

# -----------------------------
# Load ML model
# -----------------------------
def load_model(path="models/rf_model.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    return joblib.load(path)

# -----------------------------
# Automata / Rule-based prediction
# -----------------------------
def automata_rule(row):
    if "FlowBytes" in row and row["FlowBytes"] > 1000000:
        return 1  # Attack
    if "DestinationPort" in row and row["DestinationPort"] not in [80, 443, 22, 21, 25]:
        return 1
    return 0

# -----------------------------
# Compute metrics
# -----------------------------
def compute_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, average="weighted", zero_division=0)
    }

# -----------------------------
# Plot confusion matrix
# -----------------------------
def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix", cmap="Blues"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    return fig
