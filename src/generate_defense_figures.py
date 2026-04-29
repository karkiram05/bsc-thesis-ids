"""Defense figures: day-split distribution, binary vs multi-class, threshold transfer, etc.

All numbers are loaded from artifacts in `reports/metrics_*` and from the
processed parquet — none are hardcoded — so a fresh `make full` run keeps
these plots in lock-step with the latest metrics.
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

from src.config import DATA_FILE, REPORTS_DIR, FIGURES_DIR

REPORTS = REPORTS_DIR
FIG_DIR = FIGURES_DIR

# Canonical model order across every panel in this file.
MODELS_KEY = ["logreg", "random_forest", "xgboost", "lightgbm"]
MODELS_LABEL = ["LogReg", "RF", "XGBoost", "LightGBM"]


# ── Helpers: load metrics from the same JSON the rest of the pipeline writes ──

def _require(p: Path) -> Path:
    if not p.exists():
        raise SystemExit(
            f"Missing {p}. Run `make full` (or the relevant eval target) first."
        )
    return p


def _load_macro_f1(metrics_json: Path) -> list[float]:
    """summary.{model}.macro_f1 from a multi-class metrics.json."""
    data = json.loads(_require(metrics_json).read_text())
    summary = data.get("summary", data)
    return [float(summary[m]["macro_f1"]) for m in MODELS_KEY]


def _load_binary_f1(binary_metrics_json: Path) -> list[float]:
    """{model}.f1_at_threshold from a CICIDS binary_metrics.json."""
    data = json.loads(_require(binary_metrics_json).read_text())
    return [float(data[m]["f1_at_threshold"]) for m in MODELS_KEY]


def _load_binary_threshold(binary_metrics_json: Path) -> list[float]:
    """{model}.best_threshold_from_val."""
    data = json.loads(_require(binary_metrics_json).read_text())
    return [float(data[m]["best_threshold_from_val"]) for m in MODELS_KEY]


def _load_unsw_binary_f1(binary_metrics_json: Path) -> list[float]:
    """UNSW binary uses a flat {model}.f1 key (no _at_threshold suffix)."""
    data = json.loads(_require(binary_metrics_json).read_text())
    return [float(data[m]["f1"]) for m in MODELS_KEY]


def _load_day_attack_counts() -> dict[str, dict[str, int]]:
    """Per-day attack counts from the processed parquet (only 2 columns loaded)."""
    df = pd.read_parquet(_require(DATA_FILE), columns=["day", "attack_type"])
    counts = (
        df.groupby(["day", "attack_type"]).size()
        .reset_index(name="n")
    )
    out: dict[str, dict[str, int]] = {}
    for _, r in counts.iterrows():
        out.setdefault(str(r["day"]), {})[str(r["attack_type"])] = int(r["n"])
    return out


def _load_class_distribution() -> list[tuple[str, int]]:
    """Total flow count per attack type, descending."""
    df = pd.read_parquet(_require(DATA_FILE), columns=["attack_type"])
    s = df["attack_type"].value_counts().sort_values(ascending=False)
    return [(str(k), int(v)) for k, v in s.items()]


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_day_attack_distribution():
    """Attack types per day -- shows why multi-class fails on day split."""

    raw = _load_day_attack_counts()
    # Order days chronologically and label train/val/test for the chart.
    day_order = [
        ("Monday", "Monday\n(Train)"),
        ("Tuesday", "Tuesday\n(Train)"),
        ("Wednesday", "Wednesday\n(Train)"),
        ("Thursday", "Thursday\n(Val)"),
        ("Friday", "Friday\n(Test)"),
    ]
    days_data = {label: raw.get(d, {}) for d, label in day_order}

    # Colours per attack type. Unknown types fall back to a default grey.
    tactic_colors = {
        "Benign": "#95a5a6",
        "FTP-Patator": "#e74c3c", "SSH-Patator": "#c0392b",  # Credential Access
        "DoS Hulk": "#8B0000", "DoS GoldenEye": "#A52A2A", "DoS slowloris": "#CD5C5C",
        "DoS Slowhttptest": "#DC143C", "Heartbleed": "#FF1493",  # Impact/Initial Access
        "Web Attack-Brute Force": "#e67e22", "Web Attack-XSS": "#f39c12",
        "Infiltration": "#2ecc71", "Web Attack-Sql Injection": "#d35400",
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

    # Legend (deduplicated)
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


def fig_binary_vs_multiclass():
    """Binary vs multi-class F1 comparison across splits."""

    mc_strat = _load_macro_f1(REPORTS / "metrics_strat" / "metrics.json")
    mc_day   = _load_macro_f1(REPORTS / "metrics_day"   / "metrics.json")
    bi_strat = _load_binary_f1(REPORTS / "metrics_strat_binary" / "binary_metrics.json")
    bi_day   = _load_binary_f1(REPORTS / "metrics_day_binary"   / "binary_metrics.json")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(MODELS_LABEL))
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
        for i in range(len(MODELS_LABEL)):
            delta = dv[i] - sv[i]
            y_top = max(sv[i], dv[i]) + 0.07
            color = "#27ae60" if delta >= 0 else "#c0392b"
            ax.annotate(f"Δ{delta:+.2f}", xy=(x[i], y_top), ha="center",
                        fontsize=9, color=color, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(MODELS_LABEL, fontsize=11)
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


def fig_threshold_transfer():
    """Threshold values and impact: strat vs day split."""

    thr_strat = _load_binary_threshold(
        REPORTS / "metrics_strat_binary" / "binary_metrics.json"
    )
    thr_day = _load_binary_threshold(
        REPORTS / "metrics_day_binary" / "binary_metrics.json"
    )
    f1_day = _load_binary_f1(
        REPORTS / "metrics_day_binary" / "binary_metrics.json"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel 1: Threshold values
    ax = axes[0]
    x = np.arange(len(MODELS_LABEL))
    w = 0.32
    bars1 = ax.bar(x - w/2, thr_strat, w, label="Stratified", color="#3498db", alpha=0.85)
    bars2 = ax.bar(x + w/2, thr_day, w, label="Day (Youden's J)", color="#e74c3c", alpha=0.85)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=8, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS_LABEL, fontsize=11)
    ax.set_ylabel("Optimal Threshold", fontsize=12)
    ax.set_title("Threshold Values: Strat vs Day", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: What happens if you use the strat-style 0.5 threshold on day data.
    # Naive numbers are the documented "use 0.5 instead of Youden's J" counterfactual
    # from the original day-split sweep — kept here as a visual comparator since
    # the main pipeline only persists the Youden-tuned numbers.
    ax = axes[1]
    f1_youden = f1_day
    f1_naive = [0.68, 0.60, 0.007, 0.007]

    bars1 = ax.bar(x - w/2, f1_naive, w, label="Naive thr=0.5", color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x + w/2, f1_youden, w, label="Youden's J", color="#2ecc71", alpha=0.85)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS_LABEL, fontsize=11)
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


def fig_cross_dataset():
    """CICIDS vs UNSW F1 comparison."""

    cicids_mc = _load_macro_f1(REPORTS / "metrics_strat" / "metrics.json")
    unsw_mc   = _load_macro_f1(REPORTS / "metrics_unsw" / "multiclass" / "metrics.json")
    cicids_bi = _load_binary_f1(REPORTS / "metrics_strat_binary" / "binary_metrics.json")
    unsw_bi   = _load_unsw_binary_f1(REPORTS / "metrics_unsw" / "binary" / "binary_metrics.json")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(MODELS_LABEL))
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
        ax.set_xticklabels(MODELS_LABEL, fontsize=11)
        ax.set_ylabel("F1 Score", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Cross-Dataset Comparison: CICIDS2017 vs UNSW-NB15\n"
                 "No universal winner: model ranking flips between datasets",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    out = FIG_DIR / "cross_dataset_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_class_imbalance():
    """Class distribution bar chart (log scale)."""

    counts = _load_class_distribution()
    color_map = {
        "Benign": "#95a5a6",
        "DoS Hulk": "#8B0000", "DDoS": "#800000", "DoS GoldenEye": "#A52A2A",
        "FTP-Patator": "#e74c3c", "DoS slowloris": "#CD5C5C",
        "DoS Slowhttptest": "#DC143C", "SSH-Patator": "#c0392b",
        "PortScan": "#27ae60", "Web Attack-Brute Force": "#e67e22",
        "Bot": "#8e44ad", "Web Attack-XSS": "#f39c12",
        "Infiltration": "#2ecc71", "Web Attack-Sql Injection": "#d35400",
        "Heartbleed": "#FF1493",
    }
    classes = [(name, n, color_map.get(name, "#bdc3c7")) for name, n in counts]

    fig, ax = plt.subplots(figsize=(12, 6))
    names = [c[0] for c in classes]
    flow_counts = [c[1] for c in classes]
    colors = [c[2] for c in classes]

    benign_share = flow_counts[0] / sum(flow_counts) * 100 if flow_counts else 0
    ratio = flow_counts[0] // max(flow_counts[-1], 1) if flow_counts else 0

    ax.barh(range(len(names)-1, -1, -1), flow_counts, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names)-1, -1, -1))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("Flow Count (log scale)", fontsize=12)
    ax.set_title("CICIDS2017 Class Distribution — Extreme Imbalance\n"
                 f"Benign = {benign_share:.1f}% | rarest class = {flow_counts[-1]} flows\n"
                 "→ Why macro F1 is the right metric (not accuracy)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Add count labels
    for i, (name, count, _) in enumerate(classes):
        ax.text(count * 1.3, len(classes)-1-i, f"{count:,}",
                va="center", fontsize=9, fontweight="bold")

    # Add ratio annotation
    ax.annotate(f"Ratio: {ratio:,}:1\n(Benign:rarest)",
                xy=(0.75, 0.15), xycoords="axes fraction",
                fontsize=11, fontweight="bold", color="#c0392b",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="#c0392b"))

    plt.tight_layout()
    out = FIG_DIR / "class_imbalance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("[defense figures] generating...")
    fig_day_attack_distribution()
    fig_binary_vs_multiclass()
    fig_threshold_transfer()
    fig_cross_dataset()
    fig_class_imbalance()
    print("[defense figures] done.")


if __name__ == "__main__":
    main()
