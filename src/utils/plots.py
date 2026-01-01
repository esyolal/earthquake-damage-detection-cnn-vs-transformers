from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, ConfusionMatrixDisplay

def plot_curves(history_csv_path: str, out_dir: str) -> None:
    import pandas as pd
    df = pd.read_csv(history_csv_path)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # loss
    if "train_loss" in df.columns and "val_loss" in df.columns:
        plt.figure()
        plt.plot(df["epoch"], df["train_loss"], label="train_loss")
        plt.plot(df["epoch"], df["val_loss"], label="val_loss")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
        plt.tight_layout()
        plt.savefig(out / "loss_curve.png", dpi=200)
        plt.close()

    # acc
    if "train_acc" in df.columns and "val_acc" in df.columns:
        plt.figure()
        plt.plot(df["epoch"], df["train_acc"], label="train_acc")
        plt.plot(df["epoch"], df["val_acc"], label="val_acc")
        plt.xlabel("epoch"); plt.ylabel("acc"); plt.legend()
        plt.tight_layout()
        plt.savefig(out / "acc_curve.png", dpi=200)
        plt.close()

    # f1
    if "val_f1" in df.columns:
        plt.figure()
        plt.plot(df["epoch"], df["val_f1"], label="val_f1")
        plt.xlabel("epoch"); plt.ylabel("f1"); plt.legend()
        plt.tight_layout()
        plt.savefig(out / "f1_curve.png", dpi=200)
        plt.close()

def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, out_path: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_confusion(cm_2x2: list[list[int]], out_path: str) -> None:
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm_2x2), display_labels=["undamaged", "damaged"])
    fig, ax = plt.subplots()
    disp.plot(ax=ax, values_format="d")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
