"""Shared evaluation metrics used by training and the Streamlit app."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(model, X, y):
    """Return metrics, confusion matrix, and per-class report for labeled rows."""
    prediction = model.predict(X)
    probabilities = model.predict_proba(X)
    labels = list(model.classes_)
    metrics = {
        "Accuracy": accuracy_score(y, prediction),
        "AUC (OvR macro)": roc_auc_score(
            y, probabilities, labels=labels, multi_class="ovr", average="macro"
        ),
        "Precision (macro)": precision_score(y, prediction, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y, prediction, average="macro", zero_division=0),
        "F1 (macro)": f1_score(y, prediction, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, prediction),
    }
    matrix = confusion_matrix(y, prediction, labels=labels)
    report = classification_report(y, prediction, labels=labels, output_dict=True, zero_division=0)
    return metrics, matrix, report, labels
