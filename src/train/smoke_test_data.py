from __future__ import annotations
import argparse

from src.data.transforms import get_train_transforms, get_eval_transforms
from src.data.loader import build_loaders

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    train_tfms = get_train_transforms(args.img_size)
    eval_tfms = get_eval_transforms(args.img_size)

    train_csv = "outputs/splits/train.csv"
    val_csv   = "outputs/splits/val.csv"
    test_csv  = "outputs/splits/test.csv"

    dl_train, dl_val, dl_test = build_loaders(
        train_csv, val_csv, test_csv,
        train_tfms, eval_tfms,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    x, y = next(iter(dl_train))
    print("✅ Train batch OK")
    print("   x shape:", tuple(x.shape))  # (B, 3, 224, 224)
    print("   y shape:", tuple(y.shape))  # (B,)
    print("   y unique:", sorted(set(y.tolist())))

    x2, y2 = next(iter(dl_val))
    print("\n✅ Val batch OK")
    print("   x shape:", tuple(x2.shape))
    print("   y unique:", sorted(set(y2.tolist())))

    x3, y3 = next(iter(dl_test))
    print("\n✅ Test batch OK")
    print("   x shape:", tuple(x3.shape))
    print("   y unique:", sorted(set(y3.tolist())))

if __name__ == "__main__":
    main()
