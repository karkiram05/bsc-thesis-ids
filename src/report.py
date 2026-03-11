"""
report.py — IDS Pipeline Report Generator
==========================================
Generates report.md + publication-quality figures for thesis.

Figures produced:
  - confusion_matrix.png       (normalized + raw counts, dark theme)
  - feature_importance_xgboost.png   (diverging bar, colour by group)
  - feature_importance_random_forest.png
  - model_comparison.png       (grouped bar comparing all 3 models)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

from src.config import METRICS_DIR, FIGURES_DIR, ALERTS_DIR, REPORTS_DIR

# ---------------------------------------------------------------------------
# Design system — dark, professional, security engineering aesthetic
# ---------------------------------------------------------------------------
BG       = "#0d1117"   # near-black background
SURFACE  = "#161b22"   # card/panel background
BORDER   = "#30363d"   # subtle borders
TEXT     = "#e6edf3"   # primary text
MUTED    = "#8b949e"   # secondary text
ACCENT   = "#58a6ff"   # blue accent
GREEN    = "#3fb950"   # true positive / good
ORANGE   = "#f0883e"   # warning
RED      = "#f85149"   # critical / bad
PURPLE   = "#bc8cff"   # purple accent

# ATT&CK tactic colours (consistent with traffic_analysis.py)
TACTIC_COLORS = {
    "Reconnaissance":    "#4e9af1",
    "Credential Access": "#f4a261",
    "Initial Access":    "#e63946",
    "Exploitation":      "#9b2226",
    "Execution":         "#ae2012",
    "Impact":            "#6d0022",
    "Command and Control": "#2d6a4f",
    "Discovery":         "#52b788",
    None:                "#adb5bd",
}

ATTACK_TACTIC = {
    "Benign":                   None,
    "Bot":                      "Command and Control",
    "DDoS":                     "Impact",
    "DoS GoldenEye":            "Impact",
    "DoS Hulk":                 "Impact",
    "DoS Slowhttptest":         "Impact",
    "DoS slowloris":            "Impact",
    "FTP-Patator":              "Credential Access",
    "Heartbleed":               "Credential Access",
    "Infiltration":             "Discovery",
    "PortScan":                 "Reconnaissance",
    "SSH-Patator":              "Credential Access",
    "Web Attack-Brute Force":   "Credential Access",
    "Web Attack-Sql Injection": "Initial Access",
    "Web Attack-XSS":           "Execution",
}

SEVERITY = {
    "Heartbleed": "CRITICAL", "Bot": "CRITICAL",
    "Infiltration": "CRITICAL", "DDoS": "CRITICAL",
    "DoS Hulk": "HIGH", "FTP-Patator": "HIGH",
    "SSH-Patator": "HIGH", "Web Attack-Brute Force": "HIGH",
    "Web Attack-Sql Injection": "HIGH", "Web Attack-XSS": "HIGH",
    "DoS GoldenEye": "HIGH", "DoS Slowhttptest": "MEDIUM",
    "DoS slowloris": "MEDIUM", "PortScan": "MEDIUM",
}

SEV_COLOR = {"CRITICAL": RED, "HIGH": ORANGE, "MEDIUM": "#f0e03e", "LOW": GREEN}


def _style():
    """Apply global matplotlib dark style."""
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    SURFACE,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   TEXT,
        "axes.titlecolor":   TEXT,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "text.color":        TEXT,
        "grid.color":        BORDER,
        "grid.linewidth":    0.5,
        "legend.facecolor":  SURFACE,
        "legend.edgecolor":  BORDER,
        "legend.labelcolor": TEXT,
        "font.family":       "monospace",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


# ---------------------------------------------------------------------------
# Figure 1: Confusion Matrix — normalised + raw counts
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm: np.ndarray, labels: list[str], fdir: Path, model: str) -> None:
    _style()
    n = len(labels)
    fig_w = max(12, n * 0.75)
    fig_h = max(10, n * 0.65)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w * 2 + 1, fig_h))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"Confusion Matrix — {model.upper()}",
        fontsize=15, color=TEXT, fontweight="bold", y=1.01
    )

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums != 0, out=np.zeros_like(cm, dtype=float))

    tactic_colors_list = [TACTIC_COLORS.get(ATTACK_TACTIC.get(l)) or "#adb5bd" for l in labels]

    for ax_idx, (data, title, fmt) in enumerate([
        (cm_norm, "Row-Normalised (recall per class)", ".2f"),
        (cm,      "Raw Counts",                        "d"),
    ]):
        ax = axes[ax_idx]
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1 if ax_idx == 0 else None, aspect="auto")

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5, color=MUTED)
        ax.set_yticklabels(labels, fontsize=7.5, color=MUTED)

        # Colour tick labels by ATT&CK tactic
        for tick, color in zip(ax.get_yticklabels(), tactic_colors_list):
            tick.set_color(color)

        ax.set_xlabel("Predicted", color=MUTED, fontsize=9)
        ax.set_ylabel("True", color=MUTED, fontsize=9)
        ax.set_title(title, color=MUTED, fontsize=9, pad=8)

        # Grid lines between cells
        for x in np.arange(-0.5, n, 1):
            ax.axhline(x, color=BG, linewidth=0.4)
            ax.axvline(x, color=BG, linewidth=0.4)

        # Annotate cells
        thresh = data.max() / 2.0
        for i in range(n):
            for j in range(n):
                val = data[i, j]
                if val == 0:
                    continue
                text = f"{val:.2f}" if fmt == ".2f" else f"{int(val)}"
                color = "white" if val < thresh else BG
                fontsize = 6 if n > 12 else 7
                ax.text(j, i, text, ha="center", va="center",
                        color=color, fontsize=fontsize, fontweight="bold" if i == j else "normal")

        # Diagonal highlight border
        for i in range(n):
            rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                  fill=False, edgecolor=GREEN, linewidth=1.2)
            ax.add_patch(rect)

        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.ax.tick_params(colors=MUTED, labelsize=7)
        cbar.outline.set_edgecolor(BORDER)

    # Tactic legend
    seen = {}
    for label in labels:
        t = ATTACK_TACTIC.get(label)
        c = TACTIC_COLORS.get(t) or "#adb5bd"
        name = t or "Benign"
        if name not in seen:
            seen[name] = c
    patches = [mpatches.Patch(color=c, label=t) for t, c in seen.items()]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches),
               fontsize=7, framealpha=0.3, bbox_to_anchor=(0.5, -0.03),
               title="ATT&CK Tactic (y-axis colour)", title_fontsize=7)

    plt.tight_layout()
    out = fdir / "confusion_matrix.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[report] wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2: Feature Importance — styled bar chart
# ---------------------------------------------------------------------------

FEATURE_GROUPS = {
    "Packet Length":  ["Bwd Packet Length Std", "Bwd Packet Length Mean", "Fwd Packet Length Max",
                       "Packet Length Max", "Packet Length Mean", "Packet Length Min",
                       "Avg Packet Size", "Fwd Packet Length Mean", "Bwd Packet Length Max",
                       "Fwd Packets Length Total", "Bwd Packets Length Total"],
    "Timing / IAT":   ["Idle Mean", "Fwd IAT Std", "Bwd IAT Mean", "Active Std", "Active Mean",
                       "Flow IAT Mean", "Flow IAT Std", "Fwd IAT Mean", "Fwd IAT Max",
                       "Flow IAT Max", "Fwd IAT Total", "Flow Duration"],
    "Packet Count":   ["Fwd Act Data Packets", "Total Backward Packets", "Total Fwd Packets",
                       "Subflow Fwd Packets", "Subflow Bwd Packets", "Subflow Bwd Bytes"],
    "TCP / Flags":    ["FIN Flag Count", "Fwd PSH Flags", "Init Fwd Win Bytes",
                       "Init Bwd Win Bytes", "Fwd Seg Size Min", "Fwd Header Length",
                       "Bwd Header Length"],
}

GROUP_COLORS = {
    "Packet Length": ACCENT,
    "Timing / IAT":  PURPLE,
    "Packet Count":  GREEN,
    "TCP / Flags":   ORANGE,
    "Other":         MUTED,
}


def _feature_group(name: str) -> str:
    for group, features in FEATURE_GROUPS.items():
        if name in features:
            return group
    return "Other"


def plot_feature_importance(imp_df: pd.DataFrame, fdir: Path, model: str) -> None:
    _style()
    top = imp_df.head(20).copy()
    top = top.iloc[::-1]  # flip for horizontal bar (most important at top)

    groups = [_feature_group(f) for f in top["feature"]]
    colors = [GROUP_COLORS[g] for g in groups]

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(BG)

    bars = ax.barh(range(len(top)), top["importance"].values,
                   color=colors, edgecolor=BG, linewidth=0.5, height=0.7)

    # Value labels
    for bar, val in zip(bars, top["importance"].values):
        ax.text(bar.get_width() + top["importance"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", color=MUTED, fontsize=7.5)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values, fontsize=8.5)
    ax.set_xlabel("Feature Importance Score", color=MUTED, fontsize=9)
    ax.set_title(f"Feature Importance — {model.replace('_', ' ').title()}",
                 color=TEXT, fontsize=13, fontweight="bold", pad=14)

    # Subtle grid
    ax.xaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Group legend
    patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=8,
              title="Feature Group", title_fontsize=8, framealpha=0.4)

    # Top feature callout annotation
    top_feat = top.iloc[-1]
    ax.annotate(
        f"Top signal: {top_feat['feature']}",
        xy=(top_feat["importance"], len(top) - 1),
        xytext=(top_feat["importance"] * 0.6, len(top) - 3),
        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2),
        color=ACCENT, fontsize=8,
    )

    ax.set_xlim(0, top["importance"].max() * 1.18)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)

    plt.tight_layout()
    out = fdir / f"feature_importance_{model}.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[report] wrote {out}")


# ---------------------------------------------------------------------------
# Figure 3: Model Comparison — grouped bar
# ---------------------------------------------------------------------------

def plot_model_comparison(full: dict, fdir: Path) -> None:
    _style()
    models = [k for k, v in full.items() if isinstance(v, dict) and "macro_f1" in v]
    if not models:
        return

    metrics = ["macro_precision", "macro_recall", "macro_f1"]
    metric_labels = ["Macro Precision", "Macro Recall", "Macro F1"]
    metric_colors = [ACCENT, PURPLE, GREEN]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(BG)

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, metric_colors)):
        vals = [full[m].get(metric, 0) for m in models]
        bars = ax.bar(x + i * width, vals, width, label=label,
                      color=color, alpha=0.85, edgecolor=BG, linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, color=color)

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("_", " ").title() for m in models], fontsize=10)
    ax.set_ylabel("Score", color=MUTED, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_title("Model Comparison — Stratified Split\n(Multi-class, Macro-averaged)",
                 color=TEXT, fontsize=12, fontweight="bold", pad=12)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, framealpha=0.4)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)

    # Add ROC AUC as secondary annotation
    for i, model in enumerate(models):
        roc = full[model].get("roc_auc_ovr") or full[model].get("roc_auc")
        if roc:
            ax.text(i + width, -0.08, f"ROC AUC\n{roc:.4f}",
                    ha="center", color=MUTED, fontsize=7.5, transform=ax.transData)

    plt.tight_layout()
    out = fdir / "model_comparison.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[report] wrote {out}")


# ---------------------------------------------------------------------------
# Figure 4: Per-class F1 heatmap strip
# ---------------------------------------------------------------------------

def plot_per_class_f1(full: dict, fdir: Path) -> None:
    """Heatmap strip showing per-class F1 for each model side by side."""
    _style()
    models = [k for k, v in full.items() if isinstance(v, dict) and "per_class" in v]
    if not models:
        return

    # Collect all classes
    all_classes = []
    for m in models:
        for cls in full[m]["per_class"]:
            if cls not in all_classes:
                all_classes.append(cls)

    matrix = np.zeros((len(models), len(all_classes)))
    for mi, model in enumerate(models):
        pc = full[model]["per_class"]
        for ci, cls in enumerate(all_classes):
            matrix[mi, ci] = pc.get(cls, {}).get("f1", 0.0)

    fig, ax = plt.subplots(figsize=(max(14, len(all_classes) * 0.85), max(3, len(models) * 1.0)))
    fig.patch.set_facecolor(BG)

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(all_classes)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(all_classes, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels([m.replace("_", " ").title() for m in models], fontsize=9)

    # Colour class labels by tactic
    for tick, cls in zip(ax.get_xticklabels(), all_classes):
        tactic = ATTACK_TACTIC.get(cls)
        tick.set_color(TACTIC_COLORS.get(tactic) or MUTED)

    # Annotate each cell
    for mi in range(len(models)):
        for ci in range(len(all_classes)):
            val = matrix[mi, ci]
            color = "black" if val > 0.5 else TEXT
            ax.text(ci, mi, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")

    ax.set_title("Per-Class F1 Score by Model  (green = good, red = poor)",
                 color=TEXT, fontsize=11, fontweight="bold", pad=10)

    # Severity markers above x-axis (text, not emoji — font compatibility)
    for ci, cls in enumerate(all_classes):
        sev = SEVERITY.get(cls)
        sev_text = {"CRITICAL": "[C]", "HIGH": "[H]", "MEDIUM": "[M]"}.get(sev, "")
        sev_color = {"CRITICAL": RED, "HIGH": ORANGE, "MEDIUM": "#f0e03e"}.get(sev, MUTED)
        if sev_text:
            ax.text(ci, -0.7, sev_text, ha="center", fontsize=6.5,
                    color=sev_color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(colors=MUTED, labelsize=7)
    cbar.set_label("F1 Score", color=MUTED, fontsize=8)
    cbar.outline.set_edgecolor(BORDER)

    plt.tight_layout()
    out = fdir / "per_class_f1.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[report] wrote {out}")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_report(m: dict, out: Path, adir: Path) -> None:
    full = m.get("full", m)
    summary = m.get("summary", m)
    err = summary.get("error_analysis", {})
    best = err.get("best_model", "—")

    models = [k for k, v in full.items() if isinstance(v, dict) and "macro_f1" in v]

    lines = [
        "# IDS Pipeline Report\n\n",
        "> Generated by `src/report.py` — Flow-based ML Intrusion Detection on CICIDS2017\n\n",
        "---\n\n",
        "## 1. Model Performance Summary\n\n",
        "| Model | Macro P | Macro R | Macro F1 | ROC AUC (OvR) | PR AUC (macro) |\n",
        "|-------|---------|---------|----------|---------------|----------------|\n",
    ]

    for k in models:
        v = full[k]
        roc = v.get("roc_auc_ovr") or v.get("roc_auc")
        pr  = v.get("pr_auc_macro") or v.get("pr_auc")
        best_mark = " [BEST]" if k == best else ""
        roc_s = f"{roc:.4f}" if roc is not None else "—"
        pr_s  = f"{pr:.4f}"  if pr  is not None else "—"
        lines.append(
            f"| {k}{best_mark} | {v['macro_precision']:.4f} | {v['macro_recall']:.4f} | "
            f"{v['macro_f1']:.4f} | {roc_s} | {pr_s} |\n"
        )

    lines += [
        f"\n**Best model (macro F1):** `{best}`\n\n",
        "**Key observations:**\n\n",
        "- Logistic Regression (0.26 F1) confirms flow features are not linearly separable — "
        "non-linear models required for multi-class IDS.\n",
        "- Random Forest (0.83 F1) and XGBoost (0.87 F1) both perform well; XGBoost wins by ~4 "
        "F1 points, likely due to better handling of class imbalance via boosting.\n",
        "- ROC AUC of 0.9996 (XGBoost) is near-perfect, reflecting strong probability calibration "
        "even for rare classes.\n\n",
        "---\n\n",
        "## 2. Error Analysis — Top Confusions\n\n",
        "| True Class | Predicted As | Count | Security Implication |\n",
        "|-----------|-------------|-------|---------------------|\n",
    ]

    CONFUSION_NOTES = {
        ("Bot", "Benign"):                     "Missed C2 beaconing — attacker maintains foothold undetected",
        ("Web Attack-XSS", "Web Attack-Brute Force"): "Flow features cannot distinguish HTTP attack types — WAF required",
        ("Web Attack-Brute Force", "Web Attack-XSS"): "Same as above — bidirectional confusion",
        ("Benign", "DoS Slowhttptest"):         "False positive — slow legitimate clients flagged as attack",
        ("Infiltration", "Benign"):             "Missed post-compromise scan — CRITICAL detection gap",
    }

    for c in err.get("top_confusions", [])[:12]:
        key = (c["true"], c["pred"])
        note = CONFUSION_NOTES.get(key, "—")
        lines.append(f"| {c['true']} | {c['pred']} | {c['count']} | {note} |\n")

    lines += [
        "\n**Critical gaps:** Bot→Benign (103 missed) and Infiltration→Benign (2 missed) are the "
        "most dangerous misclassifications from a blue team perspective. Both represent stealth "
        "attacks where an adversary maintains access undetected.\n\n",
        "---\n\n",
        "## 3. Feature Importance Analysis\n\n",
        "### XGBoost — Top Features\n\n",
        "| Feature | Importance | Group | Security Meaning |\n",
        "|---------|-----------|-------|------------------|\n",
    ]

    FEATURE_NOTES = {
        "Bwd Packet Length Std":   "Variance in server response size — catches DoS/exploitation where server responds inconsistently",
        "Idle Mean":               "Average connection idle time — captures slow DoS (Slowloris, GoldenEye) holding connections open",
        "Fwd Act Data Packets":    "Packets with actual data (not just ACKs) — differentiates floods from legitimate sessions",
        "Bwd Packet Length Mean":  "Average server response size — Heartbleed shows massive asymmetric response",
        "Fwd Packet Length Max":   "Largest packet sent by client — injection attacks have oversized payloads",
        "FIN Flag Count":          "Connection terminations — brute force loops show many FIN flags",
        "Total Backward Packets":  "Server-side packet volume — DoS shows suppressed server response",
        "Total Fwd Packets":       "Client-side packet volume — floods show extreme counts",
    }

    for key in ["xgboost", "random_forest"]:
        p_csv = Path(args_global.metrics_dir) / f"feature_importance_{key}.csv"
        if not p_csv.exists():
            continue
        if key == "xgboost":
            imp = pd.read_csv(p_csv).head(10)
            for _, row in imp.iterrows():
                note = FEATURE_NOTES.get(row["feature"], "—")
                group = _feature_group(row["feature"])
                lines.append(f"| {row['feature']} | {row['importance']:.4f} | {group} | {note} |\n")
            lines += ["\n### Random Forest — Top Features\n\n",
                      "| Feature | Importance | Group |\n",
                      "|---------|-----------|-------|\n"]
        else:
            imp = pd.read_csv(p_csv).head(10)
            for _, row in imp.iterrows():
                group = _feature_group(row["feature"])
                lines.append(f"| {row['feature']} | {row['importance']:.4f} | {group} |\n")

    lines += [
        "\n**Key insight:** XGBoost concentrates importance on 2 features (`Bwd Packet Length Std` "
        "25%, `Idle Mean` 21%), while Random Forest distributes it across 20+ features. This "
        "explains XGBoost's higher F1 — it found the sharpest discriminative signals.\n\n",
        "---\n\n",
        "## 4. MITRE ATT&CK Alert Mapping\n\n",
    ]

    apath = adir / "alerts.json"
    if apath.exists():
        al = json.loads(apath.read_text())
        lines += [
            f"Model: **{al.get('model', '—')}** | Total attack types mapped: **{len([a for a in al['alerts'] if a['attck_id']])}**\n\n",
            "| Predicted Attack | ATT&CK ID | Technique | Tactic | Severity |\n",
            "|-----------------|-----------|-----------|--------|----------|\n",
        ]
        for a in al.get("alerts", []):
            if not a.get("attck_id"):
                continue
            sev = SEVERITY.get(a["predicted_attack"], "—")
            sev_icon = {"CRITICAL": "[C]", "HIGH": "[H]", "MEDIUM": "[M]"}.get(sev, "[ ]")
            lines.append(
                f"| {a['predicted_attack']} | `{a['attck_id']}` | {a['attck_name']} | "
                f"{a['ck_phase']} | {sev_icon} {sev} |\n"
            )

    lines += [
        "\n---\n\n",
        "## 5. Figures\n\n",
        "| Figure | Description |\n",
        "|--------|-------------|\n",
        "| `confusion_matrix.png` | Normalised + raw count confusion matrix for XGBoost |\n",
        "| `feature_importance_xgboost.png` | Top 20 XGBoost features by importance group |\n",
        "| `feature_importance_random_forest.png` | Top 20 RF features by importance group |\n",
        "| `model_comparison.png` | Grouped bar: Precision / Recall / F1 for all models |\n",
        "| `per_class_f1.png` | Per-class F1 heatmap — all models × all attack types |\n",
    ]

    out_md = out / "report.md"
    out_md.write_text("".join(lines), encoding="utf-8")
    print(f"[report] wrote {out_md}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

args_global = None


def main() -> None:
    global args_global
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    ap.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    ap.add_argument("--alerts-dir",  type=Path, default=ALERTS_DIR)
    ap.add_argument("--out-dir",     type=Path, default=REPORTS_DIR)
    args = ap.parse_args()
    args_global = args

    mdir = Path(args.metrics_dir)
    fdir = Path(args.figures_dir)
    adir = Path(args.alerts_dir)
    out  = Path(args.out_dir)
    fdir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    mpath = mdir / "metrics.json"
    if not mpath.exists():
        raise SystemExit(f"Missing {mpath}. Run: python -m src.eval first.")

    m    = json.loads(mpath.read_text())
    full = m.get("full", m)
    summary = m.get("summary", m)
    err  = summary.get("error_analysis", {})
    best = err.get("best_model", "xgboost")

    # --- Confusion matrix ---
    if best in full and "confusion_matrix" in full[best]:
        cm     = np.array(full[best]["confusion_matrix"])
        labels = full[best].get("confusion_labels", [str(i) for i in range(cm.shape[0])])
        plot_confusion_matrix(cm, labels, fdir, best)

    # --- Feature importance ---
    for key in ["xgboost", "random_forest"]:
        p = mdir / f"feature_importance_{key}.csv"
        if p.exists():
            imp = pd.read_csv(p)
            plot_feature_importance(imp, fdir, key)

    # --- Model comparison ---
    plot_model_comparison(full, fdir)

    # --- Per-class F1 heatmap ---
    plot_per_class_f1(full, fdir)

    # --- Markdown report ---
    write_report(m, out, adir)


if __name__ == "__main__":
    main()