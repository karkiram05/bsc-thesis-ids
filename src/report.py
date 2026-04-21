"""Generate report.md and thesis figures from evaluation metrics."""

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

# Dark theme colours
BG       = "#0d1117"
SURFACE  = "#161b22"
BORDER   = "#30363d"
TEXT     = "#e6edf3"
MUTED    = "#8b949e"
ACCENT   = "#58a6ff"
GREEN    = "#3fb950"
ORANGE   = "#f0883e"
RED      = "#f85149"
PURPLE   = "#bc8cff"

TACTIC_COLORS = {
    "Reconnaissance":      "#4e9af1",
    "Discovery":           "#52b788",
    "Credential Access":   "#f4a261",
    "Initial Access":      "#e63946",
    "Execution":           "#ae2012",
    "Impact":              "#6d0022",
    "Command and Control": "#2d6a4f",
    None:                  "#adb5bd",
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
    "Heartbleed":               "Initial Access",
    "Infiltration":             "Discovery",
    "PortScan":                 "Discovery",
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


def _style():
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


# Figure 1: Confusion Matrix

def plot_confusion_matrix(cm: np.ndarray, labels: list[str], fdir: Path, model: str) -> None:
    _style()
    n = len(labels)
    fig_w = max(12, n * 0.75)
    fig_h = max(10, n * 0.65)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w * 2 + 1, fig_h))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Confusion Matrix — {model.upper()}",
                 fontsize=15, color=TEXT, fontweight="bold", y=1.01)

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
        for tick, color in zip(ax.get_yticklabels(), tactic_colors_list):
            tick.set_color(color)
        ax.set_xlabel("Predicted", color=MUTED, fontsize=9)
        ax.set_ylabel("True", color=MUTED, fontsize=9)
        ax.set_title(title, color=MUTED, fontsize=9, pad=8)
        for x in np.arange(-0.5, n, 1):
            ax.axhline(x, color=BG, linewidth=0.4)
            ax.axvline(x, color=BG, linewidth=0.4)
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
                        color=color, fontsize=fontsize,
                        fontweight="bold" if i == j else "normal")
        for i in range(n):
            rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                  fill=False, edgecolor=GREEN, linewidth=1.2)
            ax.add_patch(rect)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.ax.tick_params(colors=MUTED, labelsize=7)
        cbar.outline.set_edgecolor(BORDER)

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


# Figure 2: Feature Importance

def plot_feature_importance(imp_df: pd.DataFrame, fdir: Path, model: str) -> None:
    _style()
    top = imp_df.head(20).copy()
    top = top.iloc[::-1]

    groups = [_feature_group(f) for f in top["feature"]]
    colors = [GROUP_COLORS[g] for g in groups]

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(BG)

    bars = ax.barh(range(len(top)), top["importance"].values,
                   color=colors, edgecolor=BG, linewidth=0.5, height=0.7)

    for bar, val in zip(bars, top["importance"].values):
        ax.text(bar.get_width() + top["importance"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", color=MUTED, fontsize=7.5)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values, fontsize=8.5)
    ax.set_xlabel("Feature Importance Score", color=MUTED, fontsize=9)
    ax.set_title(f"Feature Importance — {model.replace('_', ' ').title()}",
                 color=TEXT, fontsize=13, fontweight="bold", pad=14)

    ax.xaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=8,
              title="Feature Group", title_fontsize=8, framealpha=0.4)


    ax.set_xlim(0, top["importance"].max() * 1.18)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)

    plt.tight_layout()
    out = fdir / f"feature_importance_{model}.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[report] wrote {out}")


# Figure 3: Model Comparison

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
    ax.set_ylim(0, 1.18)
    ax.set_title("Model Comparison — Stratified Split\n(Multi-class, Macro-averaged)",
                 color=TEXT, fontsize=12, fontweight="bold", pad=12)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, framealpha=0.4)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)

    # Accuracy / ROC text above bars
    for i, model in enumerate(models):
        acc = full[model].get("accuracy") or full[model].get("micro_f1")
        roc = full[model].get("roc_auc_ovr") or full[model].get("roc_auc")
        if acc is not None:
            ax.text(i + width, 1.10, f"Acc: {acc:.4f}",
                    ha="center", color=MUTED, fontsize=8, style="italic")
        elif roc is not None:
            ax.text(i + width, 1.10, f"ROC: {roc:.4f}",
                    ha="center", color=MUTED, fontsize=8, style="italic")

    plt.tight_layout()
    out = fdir / "model_comparison.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[report] wrote {out}")


# Figure 4: Per-class F1 heatmap

def plot_per_class_f1(full: dict, fdir: Path) -> None:
    _style()
    models = [k for k, v in full.items() if isinstance(v, dict) and "per_class" in v]
    if not models:
        return

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

    for tick, cls in zip(ax.get_xticklabels(), all_classes):
        tactic = ATTACK_TACTIC.get(cls)
        tick.set_color(TACTIC_COLORS.get(tactic) or MUTED)

    for mi in range(len(models)):
        for ci in range(len(all_classes)):
            val = matrix[mi, ci]
            color = "black" if val > 0.5 else TEXT
            ax.text(ci, mi, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")

    ax.set_title("Per-Class F1 Score by Model  (green = good, red = poor)",
                 color=TEXT, fontsize=11, fontweight="bold", pad=10)

    # Severity markers above x-axis
    for ci, cls in enumerate(all_classes):
        sev = SEVERITY.get(cls)
        sev_text = {"CRITICAL": "[C]", "HIGH": "[H]", "MEDIUM": "[M]"}.get(sev, "")
        sev_color = {"CRITICAL": RED, "HIGH": ORANGE, "MEDIUM": "#f0e03e"}.get(sev, MUTED)
        if sev_text:
            ax.text(ci, -0.7, sev_text, ha="center", fontsize=6.5,
                    color=sev_color, fontweight="bold")

    # Severity legend
    sev_patches = [
        mpatches.Patch(color=RED,      label="[C] = Critical severity"),
        mpatches.Patch(color=ORANGE,   label="[H] = High severity"),
        mpatches.Patch(color="#f0e03e",label="[M] = Medium severity"),
    ]
    ax.legend(handles=sev_patches, loc="upper right", fontsize=7,
              title="Severity label key", title_fontsize=7, framealpha=0.5)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(colors=MUTED, labelsize=7)
    cbar.set_label("F1 Score", color=MUTED, fontsize=8)
    cbar.outline.set_edgecolor(BORDER)

    plt.tight_layout()
    out = fdir / "per_class_f1.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[report] wrote {out}")


# Figure 5: ROC curves per class — XGBoost OvR

def plot_roc_curves(full: dict, fdir: Path) -> None:
    """Per-class ROC curves from roc_curve_data in metrics.json."""
    _style()
    roc_data = full.get("xgboost", {}).get("roc_curve_data")
    if not roc_data:
        print("[report] roc_curve_data not in metrics.json — skipping roc_curves.png")
        print("         (make sure you are running the updated eval.py)")
        return

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor(BG)

    for cls, curves in roc_data.items():
        tactic = ATTACK_TACTIC.get(cls)
        colour = TACTIC_COLORS.get(tactic) or MUTED
        roc_auc = curves["auc"]
        lw = 2.5 if roc_auc < 0.90 else 1.2
        alpha = 1.0 if roc_auc < 0.90 else 0.55
        ax.plot(curves["fpr"], curves["tpr"], color=colour, lw=lw, alpha=alpha,
                label=f"{cls}  (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "w--", lw=0.8, alpha=0.4, label="Random")
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False positive rate", color=MUTED, fontsize=11)
    ax.set_ylabel("True positive rate", color=MUTED, fontsize=11)
    ax.set_title("ROC curves per class — XGBoost (OvR)\nThicker/brighter lines = lower AUC classes",
                 color=TEXT, fontsize=12, fontweight="bold", pad=14)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.5, ncol=2)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)

    plt.tight_layout()
    out = fdir / "roc_curves.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[report] wrote {out}")


# Figure 6: Precision-Recall curves — minority classes only

def plot_pr_curves(full: dict, fdir: Path) -> None:
    """PR curves for minority classes from pr_curve_data in metrics.json."""
    _style()
    pr_data = full.get("xgboost", {}).get("pr_curve_data")
    if not pr_data:
        print("[report] pr_curve_data not in metrics.json — skipping pr_curves_minority.png")
        print("         (make sure you are running the updated eval.py)")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(BG)

    for cls, curves in pr_data.items():
        tactic = ATTACK_TACTIC.get(cls)
        colour = TACTIC_COLORS.get(tactic) or MUTED
        ap = curves["ap"]
        n = curves["n_test"]
        ax.plot(curves["recall"], curves["precision"], color=colour, lw=2,
                label=f"{cls}  AP={ap:.3f}  (n={n})")

    ax.set_xlabel("Recall", color=MUTED, fontsize=11)
    ax.set_ylabel("Precision", color=MUTED, fontsize=11)
    ax.set_title("Precision–Recall — minority classes (n<500)\nXGBoost stratified split",
                 color=TEXT, fontsize=12, fontweight="bold", pad=14)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.5)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)

    plt.tight_layout()
    out = fdir / "pr_curves_minority.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[report] wrote {out}")


# Figure 7: Cross-dataset generalisation delta

def plot_crossdataset_delta(fdir: Path, cross_metrics_path: Path) -> None:
    """Generalisation gap chart: strat vs day split."""
    _style()

    strat_path = cross_metrics_path.parent.parent / "metrics_strat" / "metrics.json"
    day_path   = cross_metrics_path.parent.parent / "metrics_day"   / "metrics.json"

    if not strat_path.exists() or not day_path.exists():
        print(f"[report] crossdataset_delta.png skipped — need both metrics_strat and metrics_day")
        print(f"         Run: python -m src.train --split day && python -m src.eval --split day")
        return

    m_strat = json.loads(strat_path.read_text()).get("full", {})
    m_day   = json.loads(day_path.read_text()).get("full", {})

    # Use XGBoost if available, else best available model
    model_key = "xgboost" if "xgboost" in m_strat else next(iter(m_strat), None)
    if not model_key or model_key not in m_day:
        print("[report] crossdataset_delta.png skipped — model not present in both splits")
        return

    s = m_strat[model_key]
    d = m_day[model_key]

    metric_keys   = ["macro_f1", "macro_precision", "macro_recall"]
    metric_labels = ["Macro F1", "Macro Precision", "Macro Recall"]
    in_vals  = [s.get(k, 0) for k in metric_keys]
    out_vals = [d.get(k, 0) for k in metric_keys]
    deltas   = [i - o for i, o in zip(in_vals, out_vals)]

    x = np.arange(len(metric_keys))
    w = 0.30

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor(BG)

    # Left: grouped bar
    b1 = ax1.bar(x - w/2, in_vals,  w, label="Strat split (in-dist)",  color=ACCENT,  alpha=0.9)
    b2 = ax1.bar(x + w/2, out_vals, w, label="Day split (out-of-dist)", color=ORANGE, alpha=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, color=TEXT, fontsize=10)
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("Score", color=MUTED)
    ax1.set_title(f"Strat vs Day split — {model_key.replace('_',' ').title()}",
                  color=TEXT, fontsize=11, fontweight="bold", pad=10)
    ax1.legend(fontsize=9, framealpha=0.4)
    ax1.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax1.set_axisbelow(True)
    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                 f"{h:.3f}", ha="center", va="bottom", color=ACCENT, fontsize=8)
    for bar in b2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                 f"{h:.3f}", ha="center", va="bottom", color=ORANGE, fontsize=8)

    # Right: delta
    delta_colors = [RED if d > 0.05 else ORANGE if d > 0 else GREEN for d in deltas]
    ax2.bar(x, deltas, color=delta_colors, alpha=0.85)
    ax2.axhline(0, color=TEXT, lw=0.8, alpha=0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_labels, color=TEXT, fontsize=10)
    ax2.set_ylabel("Generalisation gap (strat − day)", color=MUTED)
    ax2.set_title("Performance drop on day split\nred = large gap, green = stable",
                  color=TEXT, fontsize=11, fontweight="bold", pad=10)
    ax2.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax2.set_axisbelow(True)
    for xi, delta in zip(x, deltas):
        ax2.text(xi, delta + (0.005 if delta >= 0 else -0.015),
                 f"{delta:+.3f}", ha="center", va="bottom", color=TEXT, fontsize=9)

    fig.suptitle("Cross-split generalisation — CICIDS2017 stratified vs temporal day split",
                 color=TEXT, fontsize=12, y=1.02)
    plt.tight_layout()
    out = fdir / "crossdataset_delta.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[report] wrote {out}")


# Figure 8: MITRE ATT&CK mapping heatmap table

def plot_attck_mapping_table(full: dict, fdir: Path, alerts_path: Path) -> None:
    """ATT&CK mapping heatmap with per-class F1 and flow detectability."""
    _style()

    # Best available model with per_class data
    model_key = "xgboost" if "xgboost" in full else next(
        (k for k, v in full.items() if isinstance(v, dict) and "per_class" in v), None)
    if not model_key:
        print("[report] attck_mapping_table.png skipped — no per_class data in metrics.json")
        return

    per_class = full[model_key].get("per_class", {})

    # Technique IDs and SOC notes
    TECHNIQUE_MAP = {
        "DoS Hulk":                ("T1498",     "Network DoS",             True,  "High volume — easy"),
        "DDoS":                    ("T1498",     "Network DoS",             True,  "Bandwidth signature"),
        "DoS GoldenEye":           ("T1498",     "Network DoS",             True,  "HTTP keep-alive"),
        "DoS slowloris":           ("T1499",     "Endpoint DoS",            True,  "Low-rate long flows"),
        "DoS Slowhttptest":        ("T1499",     "Endpoint DoS",            True,  "Partial HTTP headers"),
        "FTP-Patator":             ("T1110.001", "Password Guessing",       True,  "High login attempt rate"),
        "SSH-Patator":             ("T1110.001", "Password Guessing",       True,  "Port 22 burst"),
        "PortScan":                ("T1046",     "Network Service Scan",    True,  "Many short flows"),
        "Bot":                     ("T1071",     "App Layer Protocol",      True,  "Periodic beaconing"),
        "Web Attack-Brute Force":  ("T1110",     "Brute Force",             True,  "HTTP POST burst"),
        "Web Attack-XSS":          ("T1059.007", "JavaScript Interpreter",  False, "Payload in HTTP body"),
        "Web Attack-Sql Injection":("T1190",     "Exploit Public App",      False, "Not in flow features"),
        "Infiltration":            ("T1046",     "Network Service Scan",    False, "Too few samples"),
        "Heartbleed":              ("T1190",     "Exploit Public App",      False, "TLS internal, not in flow"),
        "Benign":                  (None,        None,                      True,  "Normal traffic"),
    }

    rows = []
    for cls, tactic in ATTACK_TACTIC.items():
        if cls == "Benign":
            continue
        tech_id, tech_name, flow_vis, soc = TECHNIQUE_MAP.get(cls, ("T1595", "Active Scanning", False, "—"))
        f1 = per_class.get(cls, {}).get("f1", 0.0)
        rows.append((cls, tactic or "—", tech_id or "—", tech_name or "—", f1, flow_vis, soc))

    # Sort by tactic then F1
    rows.sort(key=lambda r: (r[1], -r[4]))
    n = len(rows)

    fig, ax = plt.subplots(figsize=(17, max(6, n * 0.55 + 1.5)))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 7.5)
    ax.set_ylim(-0.5, n + 0.5)
    ax.axis("off")

    col_x      = [0.05, 1.5, 2.75, 3.8,  5.05, 5.75, 6.15]
    col_labels = ["Attack class", "Tactic", "Tech ID", "Technique", "F1", "Flow\nvisible", "SOC note"]

    for cx, cl in zip(col_x, col_labels):
        ax.text(cx, n + 0.1, cl, color=TEXT, fontsize=9, fontweight="bold", va="bottom")

    ax.axhline(n - 0.25, color=BORDER, lw=0.8)

    for i, (cls, tactic, tid, tname, f1, flow, note) in enumerate(reversed(rows)):
        tactic_col = TACTIC_COLORS.get(tactic, MUTED)
        bg_alpha = 0.07 if i % 2 == 0 else 0.03
        ax.barh(i, 7.5, left=0, height=0.82, color="white", alpha=bg_alpha)
        ax.barh(i, 0.06, left=0, height=0.82, color=tactic_col, alpha=0.9)

        ax.text(col_x[0], i, cls,    color=TEXT,       fontsize=8.5, va="center")
        ax.text(col_x[1], i, tactic, color=tactic_col, fontsize=8.5, va="center", fontweight="bold")
        ax.text(col_x[2], i, tid,    color=MUTED,      fontsize=8,   va="center", family="monospace")
        ax.text(col_x[3], i, tname,  color=MUTED,      fontsize=8,   va="center")

        f1_col = GREEN if f1 >= 0.90 else (ORANGE if f1 >= 0.60 else RED)
        ax.barh(i, 0.30, left=col_x[4] - 0.05, height=0.58, color=f1_col, alpha=0.85)
        ax.text(col_x[4] + 0.10, i, f"{f1:.2f}", color=TEXT, fontsize=8.5, va="center", ha="center")

        flow_sym = "YES" if flow else "NO"
        flow_col = GREEN if flow else RED
        ax.text(col_x[5] + 0.10, i, flow_sym, color=flow_col, fontsize=8.5,
                va="center", ha="center", fontweight="bold")

        ax.text(col_x[6], i, note, color=MUTED, fontsize=7.5, va="center")

    # Tactic legend
    unique_tactics = sorted({r[1] for r in rows if r[1] != "—"})
    patches = [mpatches.Patch(color=TACTIC_COLORS.get(t, MUTED), label=t) for t in unique_tactics]
    ax.legend(handles=patches, loc="lower right", facecolor=SURFACE, edgecolor=BORDER,
              labelcolor=TEXT, fontsize=8, title="ATT&CK Tactic",
              title_fontsize=8, framealpha=0.9)

    ax.set_title(
        f"MITRE ATT&CK mapping — CICIDS2017  |  Model F1 = {model_key.replace('_',' ').title()} stratified split\n"
        "F1 colour: green ≥ 0.90  |  orange ≥ 0.60  |  red < 0.60  |  "
        "'Flow visible' = detectable from flow features alone",
        color=TEXT, fontsize=10, pad=14, loc="left")

    plt.tight_layout()
    out = fdir / "attck_mapping_table.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[report] wrote {out}")


# Markdown report

def _feature_group_report(name: str) -> str:
    for group, features in FEATURE_GROUPS.items():
        if name in features:
            return group
    return "Other"


def write_report(m: dict, out: Path, adir: Path) -> None:
    full    = m.get("full", m)
    summary = m.get("summary", m)
    err     = summary.get("error_analysis", {})
    best    = err.get("best_model", "—")
    models  = [k for k, v in full.items() if isinstance(v, dict) and "macro_f1" in v]

    lines = [
        "# IDS Pipeline Report\n\n",
        "> Generated by `src/report.py` — Flow-based ML Intrusion Detection on CICIDS2017\n\n",
        "---\n\n",
        "## 1. Model Performance Summary\n\n",
        "| Model | Macro P | Macro R | Macro F1 | ROC AUC (OvR) | PR AUC (macro) |\n",
        "|-------|---------|---------|----------|---------------|----------------|\n",
    ]

    for k in models:
        v   = full[k]
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
    ]
    # Dynamic observations from actual metrics
    model_f1s = {k: full[k]["macro_f1"] for k in models}
    sorted_models = sorted(model_f1s.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_models) >= 1:
        best_name, best_f1 = sorted_models[0]
        lines.append(f"- Best model: {best_name} with macro F1 = {best_f1:.2f}.\n")
    if "logreg" in model_f1s:
        lines.append(f"- Logistic Regression ({model_f1s['logreg']:.2f} F1) confirms flow features are not linearly separable.\n")
    if len(sorted_models) >= 2:
        gap = sorted_models[0][1] - sorted_models[1][1]
        lines.append(f"- {sorted_models[0][0]} leads {sorted_models[1][0]} by {gap:.2f} F1 points.\n")
    lines += [
        "- High accuracy masks per-class failures on rare classes — see per_class_f1.png.\n\n",
        "---\n\n",
        "## 2. Error Analysis — Top Confusions\n\n",
        "| True Class | Predicted As | Count | Security Implication |\n",
        "|-----------|-------------|-------|---------------------|\n",
    ]

    CONFUSION_NOTES = {
        ("Bot", "Benign"):                         "Missed C2 beaconing — attacker maintains foothold",
        ("Web Attack-XSS", "Web Attack-Brute Force"): "Flow cannot distinguish HTTP attack types — WAF needed",
        ("Benign", "DoS Slowhttptest"):             "False positive — slow legitimate clients flagged",
        ("Infiltration", "Benign"):                "Missed post-compromise scan — CRITICAL gap",
    }

    for c in err.get("top_confusions", [])[:12]:
        key  = (c["true"], c["pred"])
        note = CONFUSION_NOTES.get(key, "—")
        lines.append(f"| {c['true']} | {c['pred']} | {c['count']} | {note} |\n")

    lines += [
        "\n---\n\n",
        "## 3. Feature Importance Analysis\n\n",
        "| Feature | Importance | Group | Security Meaning |\n",
        "|---------|-----------|-------|------------------|\n",
    ]

    FEATURE_NOTES = {
        "Bwd Packet Length Std":  "Variance in server response — catches DoS/exploitation",
        "Idle Mean":              "Avg connection idle time — captures slow DoS (Slowloris)",
        "Fwd Act Data Packets":   "Data packets vs ACKs — differentiates floods",
        "Bwd Packet Length Mean": "Avg server response size — Heartbleed shows massive response",
        "Fwd Packet Length Max":  "Largest client packet — injection has oversized payloads",
        "FIN Flag Count":         "Connection terminations — brute force loops show many FINs",
    }

    for key in ["xgboost", "lightgbm", "random_forest"]:
        p_csv = Path(args_global.metrics_dir) / f"feature_importance_{key}.csv"
        if not p_csv.exists():
            continue
        imp = pd.read_csv(p_csv).head(10)
        lines.append(f"\n### {key.replace('_',' ').title()}\n\n")
        for _, row in imp.iterrows():
            note  = FEATURE_NOTES.get(row["feature"], "—")
            group = _feature_group_report(row["feature"])
            lines.append(f"| {row['feature']} | {row['importance']:.4f} | {group} | {note} |\n")

    lines += [
        "\n---\n\n",
        "## 4. MITRE ATT&CK Alert Mapping\n\n",
    ]

    apath = adir / "alerts.json"
    if apath.exists():
        al = json.loads(apath.read_text())
        lines += [
            f"Model: **{al.get('model', '—')}** | Mapped: **{len([a for a in al['alerts'] if a['attck_id']])}**\n\n",
            "| Predicted Attack | ATT&CK ID | Technique | Tactic | Severity |\n",
            "|-----------------|-----------|-----------|--------|----------|\n",
        ]
        for a in al.get("alerts", []):
            if not a.get("attck_id"):
                continue
            sev     = SEVERITY.get(a["predicted_attack"], "—")
            sev_tag = {"CRITICAL": "[C]", "HIGH": "[H]", "MEDIUM": "[M]"}.get(sev, "[ ]")
            lines.append(
                f"| {a['predicted_attack']} | `{a['attck_id']}` | {a['attck_name']} | "
                f"{a['ck_phase']} | {sev_tag} {sev} |\n"
            )

    lines += [
        "\n---\n\n",
        "## 5. Figures\n\n",
        "| Figure | Description |\n",
        "|--------|-------------|\n",
        "| `confusion_matrix.png` | Normalised + raw count confusion matrix for best model |\n",
        "| `feature_importance_xgboost.png` | Top 20 XGBoost features by importance group |\n",
        "| `feature_importance_random_forest.png` | Top 20 RF features by importance group |\n",
        "| `model_comparison.png` | Grouped bar: Precision / Recall / F1 for all models |\n",
        "| `per_class_f1.png` | Per-class F1 heatmap with severity legend |\n",
        "| `roc_curves.png` | Per-class ROC/AUC — XGBoost OvR |\n",
        "| `pr_curves_minority.png` | PR curves for minority classes (n < 500) |\n",
        "| `crossdataset_delta.png` | Generalisation gap: strat vs day split |\n",
        "| `attck_mapping_table.png` | ATT&CK mapping heatmap with F1 and flow detectability |\n",
    ]

    out_md = out / "report.md"
    out_md.write_text("".join(lines), encoding="utf-8")
    print(f"[report] wrote {out_md}")


# Main

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

    m       = json.loads(mpath.read_text())
    full    = m.get("full", m)
    summary = m.get("summary", m)
    err     = summary.get("error_analysis", {})
    best    = err.get("best_model", "xgboost")

    # Figure 1: Confusion matrix
    if best in full and "confusion_matrix" in full[best]:
        cm     = np.array(full[best]["confusion_matrix"])
        labels = full[best].get("confusion_labels", [str(i) for i in range(cm.shape[0])])
        plot_confusion_matrix(cm, labels, fdir, best)

    # Figure 2: Feature importance (both models, annotation removed)
    for key in ["xgboost", "lightgbm", "random_forest"]:
        p = mdir / f"feature_importance_{key}.csv"
        if p.exists():
            imp = pd.read_csv(p)
            plot_feature_importance(imp, fdir, key)

    # Figure 3: Model comparison (accuracy row fixed)
    plot_model_comparison(full, fdir)

    # Figure 4: Per-class F1 (severity legend added)
    plot_per_class_f1(full, fdir)

    # Figure 5: ROC curves
    plot_roc_curves(full, fdir)

    # Figure 6: PR curves minority classes
    plot_pr_curves(full, fdir)

    # Figure 7: Cross-dataset delta
    plot_crossdataset_delta(fdir, mpath)

    # Figure 8: ATT&CK mapping heatmap table
    plot_attck_mapping_table(full, fdir, adir)

    # Markdown report
    write_report(m, out, adir)


if __name__ == "__main__":
    main()
