from __future__ import annotations
from pathlib import Path
import argparse
import yaml

from src.utils.seed import seed_everything
from src.data.splits import SplitConfig, build_manifest, make_splits, save_splits

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
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

if __name__ == "__main__":
    main()
