from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

@dataclass
class SplitConfig:
    root: str
    damaged_dir: str
    undamaged_dir: str
    train: float
    val: float
    test: float
    stratify: bool
    save_dir: str

def _list_images(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])

def build_manifest(cfg: SplitConfig) -> pd.DataFrame:
    root = Path(cfg.root)
    damaged = root / cfg.damaged_dir
    undamaged = root / cfg.undamaged_dir

    if not damaged.exists() or not undamaged.exists():
        raise FileNotFoundError(
            f"Dataset klasörü bulunamadı. Beklenen:\n- {damaged}\n- {undamaged}"
        )

    damaged_paths = _list_images(damaged)
    undamaged_paths = _list_images(undamaged)

    rows = []
    for p in damaged_paths:
        rows.append({"path": str(p.as_posix()), "label": 1, "class": "damaged"})
    for p in undamaged_paths:
        rows.append({"path": str(p.as_posix()), "label": 0, "class": "undamaged"})

    df = pd.DataFrame(rows)
    return df

def make_splits(df: pd.DataFrame, cfg: SplitConfig, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(cfg.train + cfg.val + cfg.test - 1.0) < 1e-6, "split oranları 1.0 etmeli"

    y = df["label"] if cfg.stratify else None

    df_train, df_tmp = train_test_split(
        df,
        test_size=(1.0 - cfg.train),
        random_state=seed,
        shuffle=True,
        stratify=y
    )

    # tmp -> val + test
    val_ratio_of_tmp = cfg.val / (cfg.val + cfg.test)
    y_tmp = df_tmp["label"] if cfg.stratify else None

    df_val, df_test = train_test_split(
        df_tmp,
        test_size=(1.0 - val_ratio_of_tmp),
        random_state=seed,
        shuffle=True,
        stratify=y_tmp
    )

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

def save_splits(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, save_dir: str) -> None:
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(out / "train.csv", index=False)
    df_val.to_csv(out / "val.csv", index=False)
    df_test.to_csv(out / "test.csv", index=False)


if __name__ == "__main__":
    # Allow running as: python -m src.data.splits
    import argparse
    from src.utils.seed import seed_everything
    import yaml
    
    def load_yaml(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config/base.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    
    cfg = load_yaml(args.config)
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    seed_everything(seed)
    
    scfg = SplitConfig(
        root=cfg["data"]["root"],
        damaged_dir=cfg["data"]["damaged_dir"],
        undamaged_dir=cfg["data"]["undamaged_dir"],
        train=float(cfg["split"]["train"]),
        val=float(cfg["split"]["val"]),
        test=float(cfg["split"]["test"]),
        stratify=bool(cfg["split"]["stratify"]),
        save_dir=str(cfg["split"]["save_dir"]),
    )
    
    df = build_manifest(scfg)
    
    damaged_count = int((df["label"] == 1).sum())
    undamaged_count = int((df["label"] == 0).sum())
    
    print("✅ Dataset manifest oluşturuldu.")
    print(f"   Damaged:   {damaged_count}")
    print(f"   Undamaged: {undamaged_count}")
    print(f"   Total:     {len(df)}")
    
    df_train, df_val, df_test = make_splits(df, scfg, seed)
    save_splits(df_train, df_val, df_test, scfg.save_dir)
    
    def counts(dfx):
        return int((dfx["label"] == 1).sum()), int((dfx["label"] == 0).sum()), len(dfx)
    
    tr = counts(df_train)
    va = counts(df_val)
    te = counts(df_test)
    
    print("\n✅ Splitler kaydedildi ->", scfg.save_dir)
    print(f"   Train: damaged={tr[0]} undamaged={tr[1]} total={tr[2]}")
    print(f"   Val:   damaged={va[0]} undamaged={va[1]} total={va[2]}")
    print(f"   Test:  damaged={te[0]} undamaged={te[1]} total={te[2]}")
    
    # küçük kontrol: dosyalar var mı
    out = Path(scfg.save_dir)
    for name in ["train.csv", "val.csv", "test.csv"]:
        p = out / name
        if not p.exists():
            raise RuntimeError(f"Split dosyası oluşmadı: {p}")