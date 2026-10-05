import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix

def compute_classification_metrics(y_true: list, y_pred: list, y_prob: list = None) -> dict:
    """Computes standard evaluation metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    roc_auc = 0.5
    fpr, tpr = [0, 1], [0, 1]

    if y_prob is not None and len(np.unique(y_true)) > 1:
        y_prob = np.array(y_prob)
        fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
        fpr = fpr_arr.tolist()
        tpr = tpr_arr.tolist()
        roc_auc = float(auc(fpr_arr, tpr_arr))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "auc": roc_auc,
        "fpr": fpr,
        "tpr": tpr,
        "confusion_matrix": cm
    }
