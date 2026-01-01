from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from src.data.simclr_dataset import SimCLRDataset
from src.data.simclr_transforms import get_simclr_transforms
from src.models.simclr.encoder import TimmEncoder
from src.models.simclr.projector import Projector
from src.models.simclr.loss import nt_xent_loss


@dataclass
class SimCLRCfg:
    seed: int
    img_size: int
    train_csv: str
    epochs: int
    batch_size: int
    num_workers: int
    lr: float
    weight_decay: float
    temperature: float
    projection_dim: int
    model_name: str
    out_dir: str


def train_simclr(cfg: SimCLRCfg) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out / "simclr_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    # Dataset / Loader (SimCLR -> train only)
    tfm = get_simclr_transforms(cfg.img_size)
    ds = SimCLRDataset(cfg.train_csv, transform=tfm)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,   # SimCLR için batch sabit kalsın
    )

    # Models
    encoder = TimmEncoder(cfg.model_name, pretrained=True, img_size=cfg.img_size).to(device)
    projector = Projector(encoder.feature_dim, cfg.projection_dim).to(device)

    # Memory-friendly formats (özellikle CUDA’da faydalı)
    if device.type == "cuda":
        encoder = encoder.to(memory_format=torch.channels_last)
        projector = projector.to(memory_format=torch.channels_last)

    # Optimizer
    params = list(encoder.parameters()) + list(projector.parameters())
    opt = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # AMP scaler
    scaler = GradScaler(enabled=(device.type == "cuda"))

    rows = []
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        encoder.train()
        projector.train()

        losses = []
        pbar = tqdm(dl, desc=f"SimCLR {cfg.model_name} {epoch}/{cfg.epochs}", leave=False)

        for x1, x2 in pbar:
            # Move to device
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            if device.type == "cuda":
                x1 = x1.to(memory_format=torch.channels_last)
                x2 = x2.to(memory_format=torch.channels_last)

            opt.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                f1 = encoder(x1)
                f2 = encoder(x2)
                z1 = projector(f1)
                z2 = projector(f2)
                loss = nt_xent_loss(z1, z2, cfg.temperature)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        ep_loss = float(np.mean(losses)) if losses else float("nan")
        rows.append({"epoch": epoch, "loss": ep_loss})
        print(f"[SimCLR:{cfg.model_name}] epoch={epoch} loss={ep_loss:.4f}")

    # Save history
    pd.DataFrame(rows).to_csv(out / "simclr_history.csv", index=False)

    # Save pretrained encoder weights
    ckpt_path = out / "simclr_pretrained.pt"
    torch.save(
        {
            "model_name": cfg.model_name,
            "encoder_state": encoder.state_dict(),
            "feature_dim": encoder.feature_dim,
        },
        ckpt_path,
    )

    elapsed = time.time() - t0
    print(f"✅ SimCLR pretrain done -> {ckpt_path}")
    print(f"Elapsed: {elapsed:.1f}s | device={device}")

    return str(ckpt_path).replace("\\", "/")
