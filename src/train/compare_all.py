from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd


def read_json(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def parse_run_name(run_dir: Path) -> str:
    name = run_dir.name
    m = re.match(r"(.+)_s\d+$", name)
    return m.group(1) if m else name


def collect_supervised(root: Path, family: str) -> list[dict]:
    rows = []
    for run_dir in sorted(root.glob("*_s*")):
        metrics_path = run_dir / "metrics_test.json"
        if not metrics_path.exists():
            continue

        m = read_json(metrics_path)
        rows.append({
            "group": family,                 # transformers / cnns
            "protocol": "supervised",        # supervised
            "model": parse_run_name(run_dir),
            "run_dir": str(run_dir).replace("\\", "/"),
            "acc": safe_float(m.get("acc")),
            "f1": safe_float(m.get("f1")),
            "auc": safe_float(m.get("auc")),
        })
    return rows


def collect_simclr(root: Path) -> list[dict]:
    """
    Expected structure:
      outputs/runs/simclr/<model>_s42/linear/metrics_test.json
      outputs/runs/simclr/<model>_s42/finetune/metrics_test.json
    """
    rows = []
    for model_seed_dir in sorted(root.glob("*_s*")):
        model_name = parse_run_name(model_seed_dir)

        for protocol in ["linear", "finetune"]:
            metrics_path = model_seed_dir / protocol / "metrics_test.json"
            if not metrics_path.exists():
                continue

            m = read_json(metrics_path)
            rows.append({
                "group": "simclr",
                "protocol": f"simclr_{protocol}",  # simclr_linear / simclr_finetune
                "model": model_name,
                "run_dir": str(model_seed_dir / protocol).replace("\\", "/"),
                "acc": safe_float(m.get("acc")),
                "f1": safe_float(m.get("f1")),
                "auc": safe_float(m.get("auc")),
            })
    return rows


def format_table_md(df: pd.DataFrame) -> str:
    show = df.copy()

    # Nice names
    show["Model"] = show["model"]
    show["Group"] = show["group"]
    show["Protocol"] = show["protocol"]
    show["Accuracy"] = show["acc"].map(lambda x: f"{x:.4f}")
    show["F1"] = show["f1"].map(lambda x: f"{x:.4f}")
    show["AUC"] = show["auc"].map(lambda x: f"{x:.4f}")

    show = show[["Group", "Protocol", "Model", "Accuracy", "F1", "AUC"]]
    # Sort best first by F1 then AUC
    show = show.sort_values(["F1", "AUC"], ascending=False)

    return show.to_markdown(index=False)


def save_table_png(md_path: Path, png_path: Path):
    """
    Markdown tabloyu PNG'ye çevirmek için: pandas + matplotlib.
    """
    import matplotlib.pyplot as plt

    df = pd.read_csv(md_path.parent / "summary_metrics.csv")

    # Sort by f1 then auc
    df = df.sort_values(["f1", "auc"], ascending=False).reset_index(drop=True)

    display_df = df[["group", "protocol", "model", "acc", "f1", "auc"]].copy()
    display_df["acc"] = display_df["acc"].map(lambda x: f"{x:.4f}")
    display_df["f1"] = display_df["f1"].map(lambda x: f"{x:.4f}")
    display_df["auc"] = display_df["auc"].map(lambda x: f"{x:.4f}")

    fig, ax = plt.subplots(figsize=(12, 0.6 + 0.35 * len(display_df)))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=["group", "protocol", "model", "acc", "f1", "auc"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.3)

    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def pick_best(df: pd.DataFrame, group: str | None = None, protocol_prefix: str | None = None):
    x = df.copy()
    if group is not None:
        x = x[x["group"] == group]
    if protocol_prefix is not None:
        x = x[x["protocol"].str.startswith(protocol_prefix)]

    if len(x) == 0:
        return None

    x = x.sort_values(["f1", "auc", "acc"], ascending=False).iloc[0]
    return {
        "model": x["model"],
        "f1": float(x["f1"]),
        "auc": float(x["auc"]),
        "acc": float(x["acc"]),
        "run_dir": x["run_dir"],
        "protocol": x["protocol"],
        "group": x["group"],
    }


def main():
    runs_root = Path("outputs/runs")
    out_dir = Path("outputs/comparisons")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # supervised
    rows += collect_supervised(runs_root / "transformers", "transformers")
    rows += collect_supervised(runs_root / "cnns", "cnns")

    # simclr downstream
    simclr_root = runs_root / "simclr"
    if simclr_root.exists():
        rows += collect_simclr(simclr_root)

    if len(rows) == 0:
        raise SystemExit("No runs found under outputs/runs")

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = out_dir / "summary_metrics.csv"
    df.to_csv(csv_path, index=False)

    # Save MD
    md_path = out_dir / "summary_table.md"
    md_path.write_text(format_table_md(df), encoding="utf-8")

    # Save PNG
    png_path = out_dir / "summary_table.png"
    save_table_png(md_path, png_path)

    print("Summary files created:")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")
    print(f"  - {png_path}")

    # Bests
    best_tr = pick_best(df, group="transformers")
    best_cnn = pick_best(df, group="cnns")
    best_simclr_lin = pick_best(df, group="simclr", protocol_prefix="simclr_linear")
    best_simclr_ft = pick_best(df, group="simclr", protocol_prefix="simclr_finetune")
    best_overall = pick_best(df)

    if best_tr:
        print("\n Best Transformer (supervised):")
        print(f"  {best_tr['model']} | f1={best_tr['f1']:.4f} auc={best_tr['auc']:.4f} acc={best_tr['acc']:.4f}")
        print(f"  run_dir: {best_tr['run_dir']}")

    if best_cnn:
        print("\n Best CNN (supervised):")
        print(f"  {best_cnn['model']} | f1={best_cnn['f1']:.4f} auc={best_cnn['auc']:.4f} acc={best_cnn['acc']:.4f}")
        print(f"  run_dir: {best_cnn['run_dir']}")

    if best_simclr_lin:
        print("\n Best SimCLR Linear:")
        print(f"  {best_simclr_lin['model']} | f1={best_simclr_lin['f1']:.4f} auc={best_simclr_lin['auc']:.4f} acc={best_simclr_lin['acc']:.4f}")
        print(f"  run_dir: {best_simclr_lin['run_dir']}")

    if best_simclr_ft:
        print("\n Best SimCLR Fine-tune:")
        print(f"  {best_simclr_ft['model']} | f1={best_simclr_ft['f1']:.4f} auc={best_simclr_ft['auc']:.4f} acc={best_simclr_ft['acc']:.4f}")
        print(f"  run_dir: {best_simclr_ft['run_dir']}")

    if best_overall:
        print("\n Best Overall (all runs):")
        print(f"  [{best_overall['group']}/{best_overall['protocol']}] {best_overall['model']} | f1={best_overall['f1']:.4f} auc={best_overall['auc']:.4f} acc={best_overall['acc']:.4f}")
        print(f"  run_dir: {best_overall['run_dir']}")


if __name__ == "__main__":
    main()
