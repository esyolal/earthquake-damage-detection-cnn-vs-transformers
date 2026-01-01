from __future__ import annotations

import argparse
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)

from src.models.simclr.encoder import TimmEncoder
from src.utils.seed import seed_everything


# -----------------------------
# Dataset (CSV -> image,label)
# -----------------------------
class CSVDataset(Dataset):
    """
    Expected CSV columns:
      - path: str
      - label: int (0/1)
    """
    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path)
        if "path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError(f"{csv_path} must contain columns: path,label")
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        y = int(row["label"])
        if self.transform:
            img = self.transform(img)
        return img, y


# -----------------------------
# Transforms
# -----------------------------
def get_train_transforms(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def get_eval_transforms(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


# -----------------------------
# Head
# -----------------------------
class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# -----------------------------
# Utils
# -----------------------------
def load_yaml_like(path: str) -> dict:
    # YAML okuyacak minimal yöntem (pyyaml var zaten)
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(encoder: nn.Module, head: nn.Module, dl: DataLoader, device: torch.device):
    encoder.eval()
    head.eval()

    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        feats = encoder(x)
        logits = head(feats)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        y_true_all.append(y.detach().cpu().numpy())
        y_pred_all.append(preds.detach().cpu().numpy())
        y_prob_all.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    y_prob = np.concatenate(y_prob_all)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    # AUC için iki sınıf da olmalı
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    return {
        "acc": float(acc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "auc": float(auc),
    }


def train_one(cfg: dict, model_name: str, ckpt_path: str, out_dir: str, mode: str):
    """
    mode:
      - linear: encoder frozen, only head train
      - finetune: encoder + head train (low lr recommended)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = int(cfg["seed"])
    seed_everything(seed)

    img_size = int(cfg["data"]["img_size"])
    train_csv = cfg["data"]["train_csv"]
    val_csv = cfg["data"]["val_csv"]
    test_csv = cfg["data"]["test_csv"]

    ft = cfg["finetune"]
    epochs = int(ft["epochs"])
    batch_size = int(ft["batch_size"])
    num_workers = int(ft["num_workers"])
    lr = float(ft["lr"])
    wd = float(ft["weight_decay"])
    patience = int(ft["early_stopping_patience"])
    monitor = ft.get("monitor", "val_f1")

    # Data
    ds_tr = CSVDataset(train_csv, transform=get_train_transforms(img_size))
    ds_va = CSVDataset(val_csv, transform=get_eval_transforms(img_size))
    ds_te = CSVDataset(test_csv, transform=get_eval_transforms(img_size))

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Encoder
    encoder = TimmEncoder(model_name, pretrained=False, img_size=img_size).to(device)

    # Load SimCLR pretrained weights
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder_state"], strict=True)

    # Head
    head = LinearHead(encoder.feature_dim, num_classes=2).to(device)

    # Freeze for linear eval
    if mode == "linear":
        for p in encoder.parameters():
            p.requires_grad = False

    # Optimizer
    train_params = list(head.parameters())
    if mode == "finetune":
        train_params += [p for p in encoder.parameters() if p.requires_grad]

    opt = AdamW(train_params, lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "config_resolved.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": mode,
                "model_name": model_name,
                "seed": seed,
                "ckpt_path": ckpt_path,
                "feature_dim": encoder.feature_dim,
                "train_csv": train_csv,
                "val_csv": val_csv,
                "test_csv": test_csv,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": wd,
            },
            f,
            indent=2,
        )

    best_score = -1e9
    best_path = out / "best.pt"
    bad = 0
    hist = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        encoder.train()
        head.train()

        losses = []
        for x, y in tqdm(dl_tr, desc=f"{mode}:{model_name} ep {epoch}/{epochs}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            feats = encoder(x)
            logits = head(feats)
            loss = criterion(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(loss.item())

        val_metrics = evaluate(encoder, head, dl_va, device)
        hist.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )

        print(
            f"[{mode}:{model_name}] epoch={epoch} "
            f"train_loss={hist[-1]['train_loss']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_auc={val_metrics['auc']:.4f}"
        )

        score = val_metrics["f1"] if monitor == "val_f1" else val_metrics["auc"]
        if score > best_score:
            best_score = score
            bad = 0
            torch.save({"encoder": encoder.state_dict(), "head": head.state_dict()}, best_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"[{mode}:{model_name}] Early stopping (patience={patience}).")
                break

    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)

    # Load best and evaluate on test
    best = torch.load(best_path, map_location=device)
    encoder.load_state_dict(best["encoder"], strict=True)
    head.load_state_dict(best["head"], strict=True)

    test_metrics = evaluate(encoder, head, dl_te, device)
    with open(out / "metrics_test.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    out_str = str(out).replace("\\", "/")
    elapsed = time.time() - t0

    print(f"\n=== {mode.upper()} SUMMARY ({model_name}) ===")
    print(f"acc={test_metrics['acc']:.4f} f1={test_metrics['f1']:.4f} auc={test_metrics['auc']:.4f}")
    print(f"-> {out_str}")
    print(f"Elapsed: {elapsed:.1f}s | device={device}")

    return test_metrics, out_str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="src/config/simclr.yaml")
    ap.add_argument("--model", required=True, help="densenet121 or swin_tiny_patch4_window7_224")
    ap.add_argument("--ckpt", required=True, help="outputs/runs/simclr/.../simclr_pretrained.pt")
    ap.add_argument("--mode", choices=["linear", "finetune"], default="finetune")
    args = ap.parse_args()

    cfg = load_yaml_like(args.config)
    seed = int(cfg["seed"])

    out_root = Path(cfg["output"]["root"]) / f"{args.model}_s{seed}" / args.mode
    train_one(cfg, args.model, args.ckpt, str(out_root), args.mode)


if __name__ == "__main__":
    main()
