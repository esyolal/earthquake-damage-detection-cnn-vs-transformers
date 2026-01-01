from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    y_true: (N,)
    y_prob: (N,) -> positive class probability
    """
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    # AUC için her iki sınıfın da olması lazım; yoksa NaN döndür
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred).tolist()  # [[tn, fp],[fn,tp]]

    return {
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": cm
    }
