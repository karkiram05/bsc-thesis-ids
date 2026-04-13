"""Generate additional thesis defense figures.

New figures:
  1. Day-split attack distribution (shows WHY multi-class fails)
  2. Binary vs multi-class comparison (key finding)
  3. Threshold transfer visualization (Youden's J story)
  4. UNSW vs CICIDS cross-dataset comparison
  5. Class imbalance visualization (with Benign)

Usage:
  .venv/bin/python -m src.generate_defense_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPORTS = Path("reports")
FIG_DIR = REPORTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Day-split attack distribution ────────────────────────────────────────

def fig_day_attack_distribution():
    """Show which attacks appear on each day — the key to WHY multi-class fails."""

    # Hardcoded from data analysis (avoids loading 2.3M rows)
    days_data = {
        "Monday\n(Train)": {"Benign": 458771},
        "Tuesday\n(Train)": {"Benign": 380533, "FTP-Patator": 5931, "SSH-Patator": 3219},
        "Wednesday\n(Train)": {"Benign": 391182, "DoS Hulk": 172688, "DoS GoldenEye": 10286,
                               "DoS slowloris": 5383, "DoS Slowhttptest": 5228, "Heartbleed": 11},
        "Thursday\n(Val)": {"Benign": 361096, "Web Attack-BF": 1470, "Web Attack-XSS": 652,
                            "Infiltration": 36, "Web Attack-SQLi": 21},
        "Friday\n(Test)": {"Benign": 385385, "DDoS": 128011, "PortScan": 1850, "Bot": 1397},
    }

    # ATT&CK tactic colors
    tactic_colors = {
        "Benign": "#95a5a6",
        "FTP-Patator": "#e74c3c", "SSH-Patator": "#c0392b",  # Credential Access
        "DoS Hulk": "#8B0000", "DoS GoldenEye": "#A52A2A", "DoS slowloris": "#CD5C5C",
        "DoS Slowhttptest": "#DC143C", "Heartbleed": "#FF1493",  # Impact/Initial Access
        "Web Attack-BF": "#e67e22", "Web Attack-XSS": "#f39c12",
        "Infiltration": "#2ecc71", "Web Attack-SQLi": "#d35400",  # Various
        "DDoS": "#800000", "PortScan": "#27ae60", "Bot": "#8e44ad",  # Friday attacks
    }

    fig, ax = plt.subplots(figsize=(14, 7))

    days = list(days_data.keys())
    x = np.arange(len(days))
    bar_width = 0.6

    for i, (day, attacks) in enumerate(days_data.items()):
        bottom = 0
        for attack, count in sorted(attacks.items(), key=lambda x: -x[1]):
            if attack == "Benign":
                continue
            color = tactic_colors.get(attack, "#bdc3c7")
            ax.bar(i, count, bar_width, bottom=bottom, color=color, label=attack,
                   edgecolor="white", linewidth=0.5)
            if count > 1000:
                ax.text(i, bottom + count/2, f"{attack}\n({count:,})",
                        ha="center", va="center", fontsize=7, fontweight="bold", color="white")
            bottom += count

    # Add train/val/test regions
    ax.axvspan(-0.5, 2.5, alpha=0.08, color="blue", label="_nolegend_")
    ax.axvspan(2.5, 3.5, alpha=0.08, color="orange", label="_nolegend_")
    ax.axvspan(3.5, 4.5, alpha=0.08, color="red", label="_nolegend_")

    ax.text(1, ax.get_ylim()[1] * 0.95, "TRAIN", ha="center", fontsize=14,
            fontweight="bold", color="blue", alpha=0.5)
    ax.text(3, ax.get_ylim()[1] * 0.95, "VAL", ha="center", fontsize=14,
            fontweight="bold", color="orange", alpha=0.5)
    ax.text(4, ax.get_ylim()[1] * 0.95, "TEST", ha="center", fontsize=14,
            fontweight="bold", color="red", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(days, fontsize=11)
    ax.set_ylabel("Number of Attack Flows", fontsize=12)
    ax.set_title("CICIDS2017: Attack Types per Day — Day-Based Split\n"
                 "Friday test attacks (DDoS, PortScan, Bot) NEVER appear in training",
                 fontsize=13, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(1, 500000)
    ax.grid(axis="y", alpha=0.3)

    # Custom legend (no duplicates)
    handles = []
    seen = set()
    for day, attacks in days_data.items():
        for attack in attacks:
            if attack != "Benign" and attack not in seen:
                handles.append(Patch(facecolor=tactic_colors.get(attack, "#bdc3c7"),
                                     label=attack))
                seen.add(attack)
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)

    plt.tight_layout()
    out = FIG_DIR / "day_attack_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ── 2. Binary vs Multi-class comparison ─────────────────────────────────────

def fig_binary_vs_multiclass():
    """Side-by-side: multi-class collapses on day split, binary survives."""
    models = ["LogReg", "RF", "XGBoost", "LightGBM"]

    # Multi-class F1
    mc_strat = [0.2662, 0.8452, 0.8627, 0.2996]
    mc_day = [0.4293, 0.4376, 0.4402, 0.4628]

    # Binary F1
    bi_strat = [0.8962, 0.9960, 0.9974, 0.9976]
    bi_day = [0.6810, 0.8586, 0.7571, 0.7435]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(models))
    w = 0.32

    titles = ["Multi-class (15 classes)", "Binary (Attack vs Benign)"]
    strat_vals = [mc_strat, bi_strat]
    day_vals = [mc_day, bi_day]

    for ax, sv, dv, title in zip(axes, strat_vals, day_vals, titles):
        bars1 = ax.bar(x - w/2, sv, w, label="Stratified (in-distribution)",
                       color="#3498db", alpha=0.85)
        bars2 = ax.bar(x + w/2, dv, w, label="Day (temporal, realistic)",
                       color="#e74c3c", alpha=0.85)

        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        # Delta arrows
        for i in range(len(models)):
            delta = dv[i] - sv[i]
            y_top = max(sv[i], dv[i]) + 0.07
            color = "#27ae60" if delta >= 0 else "#c0392b"
            ax.annotate(f"Δ{delta:+.2f}", xy=(x[i], y_top), ha="center",
                        fontsize=9, color=color, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.set_ylabel("F1 Score", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.18)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, label="_nolegend_")

    fig.suptitle("Key Finding: Binary Detection Survives Temporal Shift,\n"
                 "Multi-class Classification Collapses",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    out = FIG_DIR / "binary_vs_multiclass.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ── 3. Threshold transfer visualization ─────────────────────────────────────

def fig_threshold_transfer():
    """Show how threshold values change between strat and day split."""
    models = ["LogReg", "RF", "XGBoost", "LightGBM"]

    # Strat thresholds (high, near 0.5)
    thr_strat = [0.569272, 0.244447, 0.401882, 0.420818]
    f1_strat = [0.8962, 0.9960, 0.9974, 0.9976]

    # Day thresholds (very low for tree models — Youden's J)
    thr_day = [0.062665, 0.009024, 0.000323, 0.000178]
    f1_day = [0.6810, 0.8586, 0.7571, 0.7435]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel 1: Threshold values
    ax = axes[0]
    x = np.arange(len(models))
    w = 0.32
    bars1 = ax.bar(x - w/2, thr_strat, w, label="Stratified", color="#3498db", alpha=0.85)
    bars2 = ax.bar(x + w/2, thr_day, w, label="Day (Youden's J)", color="#e74c3c", alpha=0.85)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=8, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Optimal Threshold", fontsize=12)
    ax.set_title("Threshold Values: Strat vs Day", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: What happens if you use strat threshold on day data
    ax = axes[1]
    # F1 with Youden's J threshold (correct)
    f1_youden = f1_day
    # F1 if you use 0.5 threshold on day data (would be terrible for XGB/LGB)
    f1_naive = [0.68, 0.60, 0.007, 0.007]  # approximate from previous analysis

    bars1 = ax.bar(x - w/2, f1_naive, w, label="Naive thr=0.5", color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x + w/2, f1_youden, w, label="Youden's J", color="#2ecc71", alpha=0.85)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("F1 Score on Day Test", fontsize=12)
    ax.set_title("Impact of Threshold Method on Day Split", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Threshold Transfer Problem: Why Youden's J Matters\n"
                 "XGBoost/LightGBM F1 drops from 0.76 → 0.007 with naive threshold",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    out = FIG_DIR / "threshold_transfer.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ── 4. UNSW vs CICIDS comparison ────────────────────────────────────────────

def fig_cross_dataset():
    """Compare results across both datasets."""
    models = ["LogReg", "RF", "XGBoost", "LightGBM"]

    # Multi-class F1
    cicids_mc = [0.2662, 0.8452, 0.8627, 0.2996]  # strat
    unsw_mc = [0.4040, 0.4842, 0.5223, 0.5482]

    # Binary F1
    cicids_bi = [0.8962, 0.9960, 0.9974, 0.9976]  # strat
    unsw_bi = [0.8884, 0.9243, 0.9186, 0.9209]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(models))
    w = 0.32

    for ax, cv, uv, title in zip(
        axes,
        [cicids_mc, cicids_bi],
        [unsw_mc, unsw_bi],
        ["Multi-class Macro F1", "Binary F1"]
    ):
        bars1 = ax.bar(x - w/2, cv, w, label="CICIDS2017 (strat)", color="#3498db", alpha=0.85)
        bars2 = ax.bar(x + w/2, uv, w, label="UNSW-NB15", color="#e67e22", alpha=0.85)

        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.set_ylabel("F1 Score", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Cross-Dataset Comparison: CICIDS2017 vs UNSW-NB15\n"
                 "LightGBM leads on UNSW (0.55) but lags on CICIDS (0.30) — no universal winner",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    out = FIG_DIR / "cross_dataset_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ── 5. Class imbalance visualization ────────────────────────────────────────

def fig_class_imbalance():
    """Full class distribution including Benign — shows extreme imbalance."""
    classes = [
        ("Benign", 1976967, "#95a5a6"),
        ("DoS Hulk", 172688, "#8B0000"),
        ("DDoS", 128011, "#800000"),
        ("DoS GoldenEye", 10286, "#A52A2A"),
        ("FTP-Patator", 5931, "#e74c3c"),
        ("DoS slowloris", 5383, "#CD5C5C"),
        ("DoS Slowhttptest", 5228, "#DC143C"),
        ("SSH-Patator", 3219, "#c0392b"),
        ("PortScan", 1850, "#27ae60"),
        ("Web Attack-BF", 1470, "#e67e22"),
        ("Bot", 1397, "#8e44ad"),
        ("Web Attack-XSS", 652, "#f39c12"),
        ("Infiltration", 36, "#2ecc71"),
        ("Web Attack-SQLi", 21, "#d35400"),
        ("Heartbleed", 11, "#FF1493"),
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    names = [c[0] for c in classes]
    counts = [c[1] for c in classes]
    colors = [c[2] for c in classes]

    bars = ax.barh(range(len(names)-1, -1, -1), counts, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names)-1, -1, -1))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("Flow Count (log scale)", fontsize=12)
    ax.set_title("CICIDS2017 Class Distribution — Extreme Imbalance\n"
                 "Benign = 85.5% | Heartbleed = 11 flows (0.0005%)\n"
                 "→ Why macro F1 is the right metric (not accuracy)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Add count labels
    for i, (name, count, _) in enumerate(classes):
        ax.text(count * 1.3, len(classes)-1-i, f"{count:,}",
                va="center", fontsize=9, fontweight="bold")

    # Add ratio annotation
    ax.annotate(f"Ratio: {counts[0]//counts[-1]:,}:1\n(Benign:Heartbleed)",
                xy=(0.75, 0.15), xycoords="axes fraction",
                fontsize=11, fontweight="bold", color="#c0392b",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="#c0392b"))

    plt.tight_layout()
    out = FIG_DIR / "class_imbalance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    print("[defense figures] generating...")
    fig_day_attack_distribution()
    fig_binary_vs_multiclass()
    fig_threshold_transfer()
    fig_cross_dataset()
    fig_class_imbalance()
    print("[defense figures] done.")


if __name__ == "__main__":
    main()
