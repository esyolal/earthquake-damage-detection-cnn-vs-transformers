from __future__ import annotations
import argparse
from pathlib import Path
import yaml

from src.utils.seed import seed_everything
from src.train.train_simclr import SimCLRCfg, train_simclr

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    seed_everything(seed)

    out_root = Path(cfg["output"]["root"])
    out_root.mkdir(parents=True, exist_ok=True)

    img_size = int(cfg["data"]["img_size"])
    train_csv = cfg["data"]["train_csv"]

    sc = cfg["simclr"]
    epochs = int(sc["epochs"])
    batch_size = int(sc["batch_size"])
    num_workers = int(sc["num_workers"])
    lr = float(sc["lr"])
    wd = float(sc["weight_decay"])
    temp = float(sc["temperature"])
    proj_dim = int(sc["projection_dim"])

    for m in cfg["models"]:
        model_name = m["name"]
        out_dir = out_root / f"{model_name}_s{seed}"
        tcfg = SimCLRCfg(
            seed=seed,
            img_size=img_size,
            train_csv=train_csv,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            lr=lr,
            weight_decay=wd,
            temperature=temp,
            projection_dim=proj_dim,
            model_name=model_name,
            out_dir=str(out_dir),
        )
        train_simclr(tcfg)

if __name__ == "__main__":
    main()
