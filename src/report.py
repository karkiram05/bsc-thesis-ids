"""Generate report: tables (metrics, confusion, importance, MITRE) and plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import METRICS_DIR, FIGURES_DIR, ALERTS_DIR, REPORTS_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    ap.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    ap.add_argument("--alerts-dir", type=Path, default=ALERTS_DIR)
    ap.add_argument("--out-dir", type=Path, default=REPORTS_DIR)
    args = ap.parse_args()

    mdir = Path(args.metrics_dir)
    fdir = Path(args.figures_dir)
    adir = Path(args.alerts_dir)
    out = Path(args.out_dir)
    fdir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    if not (mdir / "metrics.json").exists():
        raise SystemExit("Missing metrics.json. Run: make eval")

    m = json.loads((mdir / "metrics.json").read_text())
    summary = m.get("summary", m)
    full = m.get("full", m)

    lines = ["# IDS Pipeline Report\n", "\n## 1. Metrics summary\n\n"]
    lines.append("| Model | Macro P | Macro R | Macro F1 | ROC AUC (OvR) | PR AUC (macro) |\n")
    lines.append("|-------|---------|---------|----------|---------------|----------------|\n")

    for k, v in full.items():
        if not isinstance(v, dict) or "macro_f1" not in v:
            continue
        roc = v.get("roc_auc_ovr") or v.get("roc_auc")
        pr = v.get("pr_auc_macro") or v.get("pr_auc")
        roc_s = f"{roc:.4f}" if roc is not None else "—"
        pr_s = f"{pr:.4f}" if pr is not None else "—"
        lines.append(
            f"| {k} | {v['macro_precision']:.4f} | {v['macro_recall']:.4f} | "
            f"{v['macro_f1']:.4f} | {roc_s} | {pr_s} |\n"
        )

    err = summary.get("error_analysis", {})
    best = err.get("best_model", "—")
    lines.append(f"\n**Best model (macro F1):** {best}\n\n")
    lines.append("## 2. Top confusions (error analysis)\n\n")
    for c in err.get("top_confusions", [])[:15]:
        lines.append(f"- True: **{c['true']}** → Pred: **{c['pred']}** (n={c['count']})\n")

    lines.append("\n## 3. Feature importance\n\n")
    for key in ["random_forest", "xgboost"]:
        p = mdir / f"feature_importance_{key}.csv"
        if not p.exists():
            continue
        imp = pd.read_csv(p).head(15)
        lines.append(f"### {key}\n\n")
        lines.append(imp.to_string(index=False) + "\n\n")

    lines.append("## 4. MITRE ATT&CK mapping (alerts)\n\n")
    apath = adir / "alerts.json"
    if apath.exists():
        al = json.loads(apath.read_text())
        lines.append(f"Model: **{al.get('model', '—')}**\n\n")
        lines.append("| Predicted attack | ATT&CK ID | ATT&CK name | CK phase | Justification |\n")
        lines.append("|------------------|-----------|-------------|----------|---------------|\n")
        for a in al.get("alerts", []):
            tid = a.get("attck_id") or "—"
            tname = a.get("attck_name") or "—"
            ck = a.get("ck_phase") or "—"
            j = (a.get("justification") or "—")[:80]
            lines.append(f"| {a.get('predicted_attack', '—')} | {tid} | {tname} | {ck} | {j} |\n")
    else:
        lines.append("Run `make report` (or `python -m src.mitre_alerts`) to generate alerts.\n")

    report_md = out / "report.md"
    report_md.write_text("".join(lines), encoding="utf-8")
    print(f"[report] wrote {report_md}")

    # Plots: confusion heatmap (best model), feature importance bar
    if best and best in full and "confusion_matrix" in full[best]:
        cm = np.array(full[best]["confusion_matrix"])
        labels = full[best].get("confusion_labels", [str(i) for i in range(cm.shape[0])])
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), max(6, len(labels) * 0.4)))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion matrix ({best})")
        plt.tight_layout()
        plt.savefig(fdir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[report] wrote {fdir / 'confusion_matrix.png'}")

    for key in ["random_forest", "xgboost"]:
        p = mdir / f"feature_importance_{key}.csv"
        if not p.exists():
            continue
        imp = pd.read_csv(p).head(20)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(imp)), imp["importance"].values, color="steelblue")
        ax.set_yticks(range(len(imp)))
        ax.set_yticklabels(imp["feature"].values, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature importance ({key})")
        plt.tight_layout()
        plt.savefig(fdir / f"feature_importance_{key}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[report] wrote {fdir / f'feature_importance_{key}.png'}")


if __name__ == "__main__":
    main()
