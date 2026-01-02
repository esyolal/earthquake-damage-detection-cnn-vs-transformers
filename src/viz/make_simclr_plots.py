import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix


def save_loss_curve(df: pd.DataFrame, out_path: Path) -> None:
    if "epoch" not in df.columns:
        raise SystemExit("Expected 'epoch' column in history CSV")
    # prefer train_loss then loss
    train_col = "train_loss" if "train_loss" in df.columns else ("loss" if "loss" in df.columns else None)
    val_col = "val_loss" if "val_loss" in df.columns else None
    
    if train_col is None:
        return
    
    plt.figure()
    plt.plot(df["epoch"], df[train_col], marker="o", label="train_loss")
    if val_col is not None:
        plt.plot(df["epoch"], df[val_col], marker="o", label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_acc_curve(df: pd.DataFrame, out_path: Path) -> None:
    if "epoch" not in df.columns:
        return
    has_train = "train_acc" in df.columns
    has_val = "val_acc" in df.columns
    
    if not (has_train or has_val):
        return
    
    plt.figure()
    if has_train:
        plt.plot(df["epoch"], df["train_acc"], marker="o", label="train_acc")
    if has_val:
        plt.plot(df["epoch"], df["val_acc"], marker="o", label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_f1_curve(df: pd.DataFrame, out_path: Path) -> None:
    if "epoch" not in df.columns or "val_f1" not in df.columns:
        return
    plt.figure()
    plt.plot(df["epoch"], df["val_f1"], marker="o", color="C1", label="val_f1")
    plt.xlabel("epoch")
    plt.ylabel("f1")
    plt.title("F1 Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_roc_from_preds(preds_path: Path, out_path: Path) -> None:
    data = np.load(preds_path)
    y_true = data["y_true"]
    y_prob = data["y_prob"]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_confusion_from_preds(preds_path: Path, out_path: Path) -> None:
    data = np.load(preds_path)
    y_true = data["y_true"]
    y_pred = data.get("y_pred")
    if y_pred is None:
        # fallback: threshold y_prob at 0.5
        y_prob = data["y_prob"]
        y_pred = (y_prob >= 0.5).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["undamaged", "damaged"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Run directory containing simclr outputs")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    # prefer fine-tune or linear histories if available
    candidates = [run_dir / "linear" / "history.csv", run_dir / "finetune" / "history.csv", run_dir / "simclr_history.csv"]
    hist_path = None
    for p in candidates:
        if p.exists():
            hist_path = p
            break

    if hist_path is None:
        raise SystemExit(f"No history CSV found in {run_dir.as_posix()} (looked for linear/finetune/simclr history)")

    df = pd.read_csv(hist_path)

    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_loss_curve(df, out_dir / "loss_curve.png")
    save_acc_curve(df, out_dir / "acc_curve.png")
    save_f1_curve(df, out_dir / "f1_curve.png")

    preds_path = run_dir / "preds_test.npz"
    if not preds_path.exists():
        # try sibling folders (linear/finetune)
        for sub in ["linear", "finetune"]:
            p = run_dir / sub / "preds_test.npz"
            if p.exists():
                preds_path = p
                break

    if preds_path.exists():
        save_roc_from_preds(preds_path, out_dir / "roc_curve.png")
        save_confusion_from_preds(preds_path, out_dir / "confusion_matrix.png")
        print("✅ Plots saved to:", out_dir.as_posix())
    else:
        print("⚠️ preds_test.npz not found. To produce ROC/CM run the export script:")
        print(f"python src/train/export_test_preds.py --run_dir {run_dir.as_posix()} --split_csv outputs/splits/test.csv --img_size 224")
        print("Loss / F1 plots created; run export_test_preds to generate preds_test.npz for ROC/CM")


if __name__ == "__main__":
    main()
