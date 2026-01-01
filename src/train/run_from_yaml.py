from __future__ import annotations
import argparse
from pathlib import Path
import yaml

from src.utils.seed import seed_everything
from src.train.train_cls import TrainCfg, train_one_model

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
    val_csv   = cfg["data"]["val_csv"]
    test_csv  = cfg["data"]["test_csv"]

    epochs = int(cfg["train"]["epochs"])
    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"]["num_workers"])
    lr = float(cfg["train"]["lr"])
    wd = float(cfg["train"]["weight_decay"])
    ls = float(cfg["train"]["label_smoothing"])
    patience = int(cfg["train"]["early_stopping_patience"])
    monitor = str(cfg["train"]["monitor"])

    results = []
    for m in cfg["models"]:
        model_name = m["name"]
        run_dir = out_root / f"{model_name}_s{seed}"

        tcfg = TrainCfg(
            seed=seed,
            img_size=img_size,
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            lr=lr,
            weight_decay=wd,
            label_smoothing=ls,
            early_stopping_patience=patience,
            monitor=monitor,
            out_dir=str(run_dir),
            model_name=model_name,
            pretrained=True
        )

        res = train_one_model(tcfg)
        results.append(res)

    # kısa özet
    print("\n=== RUN SUMMARY ===")
    for r in results:
        t = r["test"]
        print(f"{r['model_name']}: acc={t['acc']:.4f} f1={t['f1']:.4f} auc={t['auc']:.4f} -> {r['out_dir']}")

if __name__ == "__main__":
    main()
