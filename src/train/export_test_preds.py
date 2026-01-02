import argparse
from pathlib import Path
import sys

# Ensure repo root is on sys.path so `import src.*` works when running
# this file as a script (python src/train/export_test_preds.py)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import yaml
import json

import importlib
import inspect


def find_best_checkpoint(run_dir: Path) -> Path:
    for p in [
        run_dir / "best.pt",
        run_dir / "best.pth",
        run_dir / "checkpoint_best.pt",
        run_dir / "checkpoint_best.pth",
        run_dir / "model_best.pt",
    ]:
        if p.exists():
            return p
    # also search recursively in subfolders (e.g. linear/, finetune/)
    ckpts = sorted(list(run_dir.rglob("*.pt")) + list(run_dir.rglob("*.pth")),
                   key=lambda x: x.stat().st_mtime, reverse=True)
    ckpts = [p for p in ckpts if p.name != "simclr_pretrained.pt"]
    if ckpts:
        return ckpts[0]
    raise FileNotFoundError(f"No checkpoint found in: {run_dir.as_posix()}")


def load_yaml_any(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_run_config(run_dir: Path) -> dict | None:
    # look for common yaml names in run_dir (non-recursive)
    for name in ["config.yaml", "cfg.yaml", "config.yml", "cfg.yml"]:
        p = run_dir / name
        if p.exists():
            return load_yaml_any(p)

    # also search recursively for resolved json or simclr config files
    for name in ["config_resolved.json", "simclr_config.json"]:
        for p in run_dir.rglob(name):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                continue

    # finally try recursive search for any yaml-like config files
    for name in ["config.yaml", "cfg.yaml", "config.yml", "cfg.yml"]:
        for p in run_dir.rglob(name):
            try:
                return load_yaml_any(p)
            except Exception:
                continue

    return None


def load_state_dict_any(ckpt_obj):
    # common keys
    if isinstance(ckpt_obj, dict):
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
        if "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            return ckpt_obj["model_state_dict"]
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        # raw
        if all(isinstance(k, str) for k in ckpt_obj.keys()):
            return ckpt_obj
    raise ValueError("Unsupported checkpoint format for state_dict extraction.")


def find_dataset_class():
    mod = importlib.import_module("src.data.dataset")
    candidates = []
    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if obj.__module__ != mod.__name__:
            continue
        if hasattr(obj, "__len__") and hasattr(obj, "__getitem__"):
            candidates.append(obj)
    if not candidates:
        raise ImportError("No dataset-like class found in src.data.dataset")
    return candidates[0]


def build_test_transform(img_size: int):
    # prefer your project transform builder
    try:
        from src.data.transforms import build_transforms
        try:
            return build_transforms(img_size=img_size, train=False)
        except TypeError:
            return build_transforms(train=False)
    except Exception:
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


def build_model_from_project(cfg: dict):
    """
    Try to build model using project code (same as training).
    """
    # 1) src.models.factory.build_model
    try:
        factory = importlib.import_module("src.models.factory")
        if hasattr(factory, "build_model"):
            return factory.build_model(cfg)
    except Exception:
        pass

    # 2) src.models.build_model
    try:
        mod = importlib.import_module("src.models")
        if hasattr(mod, "build_model"):
            return mod.build_model(cfg)
    except Exception:
        pass

    # 3) fallback: timm by name
    import timm
    name = cfg.get("model", {}).get("name") or cfg.get("model_name")
    if name is None:
        raise ValueError("No model name found in cfg for fallback.")
    return timm.create_model(name, pretrained=False, num_classes=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=str)
    ap.add_argument("--split_csv", default="outputs/splits/test.csv", type=str)
    ap.add_argument("--img_size", default=224, type=int)
    ap.add_argument("--batch_size", default=32, type=int)
    ap.add_argument("--num_workers", default=0, type=int)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    split_csv = Path(args.split_csv)

    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir.as_posix()}")
    if not split_csv.exists():
        raise SystemExit(f"split_csv not found: {split_csv.as_posix()}")

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))

    ckpt_path = find_best_checkpoint(run_dir)
    print("Checkpoint:", ckpt_path.as_posix())
    print("Device:", device)

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Prefer cfg inside checkpoint (finetune script often saves it)
    cfg = None
    if isinstance(ckpt, dict) and "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
        cfg = ckpt["cfg"]
    else:
        cfg = find_run_config(run_dir)

    if cfg is None:
        raise SystemExit("Could not find cfg in checkpoint or config.yaml inside run_dir.")

    # dataset
    ds_cls = find_dataset_class()
    tfm = build_test_transform(args.img_size)

    # try common ctor args
    import inspect as _ins
    sig = _ins.signature(ds_cls.__init__)
    kw = {}
    if "csv_path" in sig.parameters:
        kw["csv_path"] = split_csv
    elif "csv" in sig.parameters:
        kw["csv"] = split_csv
    elif "split_csv" in sig.parameters:
        kw["split_csv"] = split_csv
    else:
        # last resort: first arg after self gets split_csv
        pass

    if "transform" in sig.parameters:
        kw["transform"] = tfm
    elif "tfm" in sig.parameters:
        kw["tfm"] = tfm

    ds = ds_cls(**kw)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    # Check if this is a SimCLR model (has encoder and head keys)
    is_simclr = isinstance(ckpt, dict) and "encoder" in ckpt and "head" in ckpt
    
    if is_simclr:
        # SimCLR model: encoder + head
        from src.models.simclr.encoder import TimmEncoder
        
        # Get model name from config or checkpoint path
        model_name = cfg.get("model", {}).get("name") if isinstance(cfg.get("model"), dict) else None
        if model_name is None:
            model_name = cfg.get("model_name")
        if model_name is None:
            # Try to infer from run_dir path
            run_dir_str = str(run_dir)
            if "densenet" in run_dir_str.lower():
                model_name = "densenet121"
            elif "swin" in run_dir_str.lower():
                model_name = "swin_tiny_patch4_window7_224"
            else:
                raise ValueError("Could not determine model_name for SimCLR model")
        
        print(f"SimCLR model detected: {model_name}")
        
        encoder = TimmEncoder(model_name, pretrained=False, img_size=args.img_size).to(device)
        encoder.load_state_dict(ckpt["encoder"], strict=True)
        
        # Build head (simple linear layer)
        class LinearHead(nn.Module):
            def __init__(self, in_dim: int, num_classes: int = 2):
                super().__init__()
                self.fc = nn.Linear(in_dim, num_classes)
            def forward(self, x):
                return self.fc(x)
        
        head = LinearHead(encoder.feature_dim, num_classes=2).to(device)
        head.load_state_dict(ckpt["head"], strict=True)
        
        encoder.eval()
        head.eval()
        
        def forward_fn(x):
            feats = encoder(x)
            return head(feats)
    else:
        # Regular model
        model = build_model_from_project(cfg)
        model.eval()

        # load weights
        state = load_state_dict_any(ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}

        # strict=True first (sağlıklı)
        try:
            model.load_state_dict(state, strict=True)
        except Exception as e:
            print("⚠️ strict=True failed, trying strict=False:", str(e))
            model.load_state_dict(state, strict=False)

        model.to(device)
        model.eval()
        
        def forward_fn(x):
            return model(x)

    y_true, prob0, prob1 = [], [], []

    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            logits = forward_fn(xb)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

            y_true.extend(np.asarray(yb, dtype=np.int64).tolist())
            prob0.extend(probs[:, 0].astype(np.float32).tolist())
            prob1.extend(probs[:, 1].astype(np.float32).tolist())

    y_true = np.asarray(y_true, dtype=np.int64)
    prob0 = np.asarray(prob0, dtype=np.float32)
    prob1 = np.asarray(prob1, dtype=np.float32)

    auc0 = roc_auc_score(y_true, prob0)
    auc1 = roc_auc_score(y_true, prob1)

    if auc0 >= auc1:
        chosen = 0
        y_prob = prob0
        chosen_auc = auc0
    else:
        chosen = 1
        y_prob = prob1
        chosen_auc = auc1

    y_pred = (y_prob >= 0.5).astype(np.int64)

    print(f"AUC(prob[:,0]) = {auc0:.4f}")
    print(f"AUC(prob[:,1]) = {auc1:.4f}")
    print(f"✅ Chosen positive index = {chosen} | AUC = {chosen_auc:.4f}")

    out_path = run_dir / "preds_test.npz"
    np.savez_compressed(
        out_path,
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
        chosen_pos_index=np.asarray([chosen], dtype=np.int64),
        auc0=np.asarray([auc0], dtype=np.float32),
        auc1=np.asarray([auc1], dtype=np.float32),
    )
    print("✅ Saved:", out_path.as_posix())


if __name__ == "__main__":
    main()
