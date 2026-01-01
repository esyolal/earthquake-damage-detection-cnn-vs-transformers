from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from src.data.transforms import get_train_transforms, get_eval_transforms
from src.data.loader import build_loaders
from src.models.factory import create_model
from src.utils.metrics import compute_metrics
from src.utils.plots import plot_curves, plot_roc, plot_confusion

@dataclass
class TrainCfg:
    seed: int
    img_size: int
    train_csv: str
    val_csv: str
    test_csv: str
    epochs: int
    batch_size: int
    num_workers: int
    lr: float
    weight_decay: float
    label_smoothing: float
    early_stopping_patience: int
    monitor: str  # "val_f1" or "val_auc"
    out_dir: str
    model_name: str
    pretrained: bool = True

def _to_device(batch, device):
    x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

@torch.no_grad()
def predict_probs(model: nn.Module, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys = []
    probs = []
    for batch in loader:
        x, y = _to_device(batch, device)
        logits = model(x)                 # (B,2)
        p = torch.softmax(logits, dim=1)[:, 1]  # positive prob
        ys.append(y.detach().cpu().numpy())
        probs.append(p.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(probs)

def train_one_model(cfg: TrainCfg) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg.out_dir)
    plots_dir = out_dir / "plots"
    preds_dir = out_dir / "preds"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    with open(out_dir / "config_resolved.json", "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    train_tfms = get_train_transforms(cfg.img_size)
    eval_tfms  = get_eval_transforms(cfg.img_size)

    dl_train, dl_val, dl_test = build_loaders(
        cfg.train_csv, cfg.val_csv, cfg.test_csv,
        train_tfms, eval_tfms,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers
    )

    model = create_model(cfg.model_name, num_classes=2, pretrained=cfg.pretrained).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_score = -1e9
    best_path = out_dir / "best.pt"
    patience = 0

    rows = []
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0

        pbar = tqdm(dl_train, desc=f"{cfg.model_name} | epoch {epoch}/{cfg.epochs}", leave=False)
        for batch in pbar:
            x, y = _to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            preds = torch.argmax(logits, dim=1)
            train_correct += int((preds == y).sum().item())
            train_total += int(y.numel())

        train_loss = float(np.mean(train_losses))
        train_acc = float(train_correct / max(train_total, 1))

        # Val
        y_true_val, y_prob_val = predict_probs(model, dl_val, device)
        val_metrics = compute_metrics(y_true_val, y_prob_val)
        val_loss = float("nan")  # istersen val loss da hesaplarız, şimdilik sade.

        monitor_value = val_metrics["f1"] if cfg.monitor == "val_f1" else val_metrics["auc"]
        score = float(monitor_value)

        rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_metrics["acc"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
        })

        # Early stop + best
        if score > best_score:
            best_score = score
            patience = 0
            torch.save({"model_name": cfg.model_name, "state_dict": model.state_dict()}, best_path)
        else:
            patience += 1

        # kısa log
        print(f"[{cfg.model_name}] epoch={epoch} train_acc={train_acc:.4f} val_f1={val_metrics['f1']:.4f} val_auc={val_metrics['auc']:.4f}")

        if patience >= cfg.early_stopping_patience:
            print(f"[{cfg.model_name}] Early stopping (patience={cfg.early_stopping_patience}).")
            break

    # save history
    hist_path = out_dir / "history.csv"
    pd.DataFrame(rows).to_csv(hist_path, index=False)

    # load best and evaluate test
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    y_true_test, y_prob_test = predict_probs(model, dl_test, device)
    test_metrics = compute_metrics(y_true_test, y_prob_test)

    # save preds
    pd.DataFrame({
        "y_true": y_true_test.astype(int),
        "y_prob": y_prob_test.astype(float),
        "y_pred": (y_prob_test >= 0.5).astype(int),
    }).to_csv(preds_dir / "test_predictions.csv", index=False)

    # save metrics
    with open(out_dir / "metrics_test.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    with open(out_dir / "metrics_val_last.json", "w", encoding="utf-8") as f:
        json.dump({k: rows[-1][k] for k in rows[-1].keys() if k.startswith("val_")}, f, indent=2)

    # plots
    plot_curves(str(hist_path), str(plots_dir))
    plot_roc(y_true_test, y_prob_test, str(plots_dir / "roc_curve.png"))
    plot_confusion(test_metrics["confusion_matrix"], str(plots_dir / "confusion_matrix.png"))

    elapsed = time.time() - t0
    result = {
        "model_name": cfg.model_name,
        "out_dir": str(out_dir),
        "best_monitor_score": float(best_score),
        "test": test_metrics,
        "elapsed_sec": float(elapsed),
        "device": str(device),
    }
    return result
