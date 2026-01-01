from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def gather_runs(outputs_root: str = "outputs/runs") -> pd.DataFrame:
    root = Path(outputs_root)
    if not root.exists():
        raise FileNotFoundError(f"Bulunamadı: {root}")

    rows = []
    # runs altında her şey: cnns/*, transformers/*, simclr/* vs olabilir
    for run_dir in root.rglob("*"):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics_test.json"
        cfg_path = run_dir / "config_resolved.json"

        if not metrics_path.exists():
            continue

        # metrics
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)

        # config (opsiyonel ama model/seed/family için iyi)
        cfg = {}
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

        model_name = cfg.get("model_name", run_dir.name)
        seed = cfg.get("seed", None)

        # family çıkarımı: path’e göre
        p = str(run_dir).replace("\\", "/").lower()
        if "/transformers/" in p:
            family = "transformer"
        elif "/cnns/" in p:
            family = "cnn"
        elif "/simclr/" in p:
            family = "simclr"
        else:
            family = "unknown"

        rows.append({
            "family": family,
            "model": model_name,
            "seed": seed,
            "acc": _safe_float(m.get("acc")),
            "precision": _safe_float(m.get("precision")),
            "recall": _safe_float(m.get("recall")),
            "f1": _safe_float(m.get("f1")),
            "auc": _safe_float(m.get("auc")),
            "run_dir": str(run_dir).replace("\\", "/"),
        })

    if not rows:
        raise RuntimeError("Hiç metrics_test.json bulunamadı. outputs/runs altında run var mı?")

    df = pd.DataFrame(rows)
    return df

def save_markdown_table(df: pd.DataFrame, out_path: str, top_n: int | None = None) -> None:
    dfx = df.copy()
    dfx = dfx.sort_values(["f1", "auc"], ascending=False)

    if top_n is not None:
        dfx = dfx.head(top_n)

    # Görüntüde de aynı sırayı kullanacağız
    md = dfx.drop(columns=["run_dir"]).to_markdown(index=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(md, encoding="utf-8")

def save_png_table(df: pd.DataFrame, out_path: str, title: str = "Model Comparison (Test)") -> None:
    dfx = df.copy()
    dfx = dfx.sort_values(["f1", "auc"], ascending=False)

    # Sadece gerekli kolonlar
    show = dfx[["family", "model", "acc", "f1", "auc", "precision", "recall"]].copy()

    # Yuvarla
    for c in ["acc", "f1", "auc", "precision", "recall"]:
        show[c] = show[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "NaN")

    # matplotlib table
    nrows = len(show)
    # satır sayısına göre dinamik yükseklik
    fig_h = 1.2 + nrows * 0.35
    fig_w = 14

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=12)

    table = ax.table(
        cellText=show.values,
        colLabels=show.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)

    # Header kalın
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def save_summary(df: pd.DataFrame, out_dir: str = "outputs/comparisons") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # csv
    csv_path = out / "summary_metrics.csv"
    df_sorted = df.sort_values(["f1", "auc"], ascending=False)
    df_sorted.to_csv(csv_path, index=False)

    # md + png
    md_path = out / "summary_table.md"
    png_path = out / "summary_table.png"
    save_markdown_table(df_sorted, str(md_path))
    save_png_table(df_sorted, str(png_path))

    # best per family (cnn/transformer)
    best_transformer = df_sorted[df_sorted["family"] == "transformer"].head(1)
    best_cnn = df_sorted[df_sorted["family"] == "cnn"].head(1)

    result = {
        "csv": str(csv_path).replace("\\", "/"),
        "md": str(md_path).replace("\\", "/"),
        "png": str(png_path).replace("\\", "/"),
        "best_transformer": best_transformer.to_dict(orient="records")[0] if len(best_transformer) else None,
        "best_cnn": best_cnn.to_dict(orient="records")[0] if len(best_cnn) else None,
    }
    return result
