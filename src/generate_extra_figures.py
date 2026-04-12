"""Generate additional thesis figures from existing data.

Figures:
  1. LODO fold-level performance (grouped bar per model per fold)
  2. Calibration curves (reliability diagrams) for binary classifiers
  3. Master results table (all experiments, both datasets, both tasks)
  4. Strat-vs-day generalisation gap for ALL models (multi + binary)

Usage:
  python -m src.generate_extra_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORTS = Path("reports")
FIG_DIR = REPORTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. LODO fold-level performance ──────────────────────────────────────────

def fig_lodo_folds():
    """Grouped bar chart: ROC-AUC and F1 per fold per model."""
    data = json.loads((REPORTS / "lodo" / "lodo_results.json").read_text())
    df = pd.DataFrame(data)

    models = ["logreg", "random_forest", "xgboost", "lightgbm"]
    model_labels = ["LogReg", "RF", "XGBoost", "LightGBM"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title in zip(
        axes, ["roc_auc", "f1"],
        ["ROC-AUC per Fold", "F1 per Fold"]
    ):
        x = np.arange(len(days))
        width = 0.18
        for i, (model, label) in enumerate(zip(models, model_labels)):
            vals = []
            for day in days:
                row = df[(df["model"] == model) & (df["held_out_day"] == day)]
                v = row[metric].values[0]
                vals.append(v if v is not None and not pd.isna(v) else 0)
            bars = ax.bar(x + i * width, vals, width, label=label, color=colors[i], alpha=0.85)
            # Mark Monday as benign-only
            if metric in ("roc_auc", "f1"):
                bars[0].set_hatch("//")
                bars[0].set_alpha(0.4)

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(days, fontsize=10)
        ax.set_ylabel(metric.upper().replace("_", "-"), fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(y=0, color="black", linewidth=0.5)

    fig.suptitle("Leave-One-Day-Out Binary Cross-Validation\n(hatched = Monday, benign-only fold)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "lodo_folds.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ── 2. Calibration curves ──────────────────────────────────────────────────

def fig_calibration():
    """Calibration (reliability) diagrams for binary day split."""
    models = ["logreg", "random_forest", "xgboost", "lightgbm"]
    labels = ["LogReg", "RF", "XGBoost", "LightGBM"]
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, split, split_label in zip(
        axes,
        ["metrics_strat_binary", "metrics_day_binary"],
        ["Stratified Split", "Day Split"]
    ):
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
        for model, label, color in zip(models, labels, colors):
            fpath = REPORTS / split / f"binary_calibration_curve_{model}.csv"
            if not fpath.exists():
                continue
            cal = pd.read_csv(fpath)
            ax.plot(cal["mean_predicted_prob"], cal["fraction_positives"],
                    "o-", label=label, color=color, markersize=5, linewidth=1.5)

        ax.set_xlabel("Mean Predicted Probability", fontsize=11)
        ax.set_ylabel("Fraction of Positives", fontsize=11)
        ax.set_title(f"Calibration — {split_label}", fontsize=13)
        ax.legend(fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "calibration_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ── 3. Master results table ─────────────────────────────────────────────────

def fig_master_table():
    """Consolidated CSV + markdown table across all experiments."""
    rows = []
    models = ["logreg", "random_forest", "xgboost", "lightgbm"]

    # CICIDS strat multi-class
    mc_strat = json.loads((REPORTS / "metrics_strat" / "metrics.json").read_text())
    for m in models:
        rows.append({
            "Dataset": "CICIDS2017", "Split": "Stratified", "Task": "Multi-class",
            "Model": m, "Macro F1": round(mc_strat["full"][m]["macro_f1"], 4),
            "ROC-AUC": round(mc_strat["full"][m].get("roc_auc_ovr", 0), 4),
        })

    # CICIDS day multi-class
    mc_day = json.loads((REPORTS / "metrics_day" / "metrics.json").read_text())
    for m in models:
        rows.append({
            "Dataset": "CICIDS2017", "Split": "Day", "Task": "Multi-class",
            "Model": m, "Macro F1": round(mc_day["full"][m]["macro_f1"], 4),
            "ROC-AUC": round(mc_day["full"][m].get("roc_auc_ovr", 0), 4),
        })

    # CICIDS strat binary
    bs = json.loads((REPORTS / "metrics_strat_binary" / "binary_metrics.json").read_text())
    for m in models:
        rows.append({
            "Dataset": "CICIDS2017", "Split": "Stratified", "Task": "Binary",
            "Model": m, "Macro F1": round(bs[m]["f1_at_threshold"], 4),
            "ROC-AUC": round(bs[m]["roc_auc"], 4),
        })

    # CICIDS day binary
    bd = json.loads((REPORTS / "metrics_day_binary" / "binary_metrics.json").read_text())
    for m in models:
        rows.append({
            "Dataset": "CICIDS2017", "Split": "Day", "Task": "Binary",
            "Model": m, "Macro F1": round(bd[m]["f1_at_threshold"], 4),
            "ROC-AUC": round(bd[m]["roc_auc"], 4),
        })

    # UNSW multi-class
    um = json.loads((REPORTS / "metrics_unsw" / "multiclass" / "metrics.json").read_text())
    for m in models:
        rows.append({
            "Dataset": "UNSW-NB15", "Split": "Random", "Task": "Multi-class",
            "Model": m, "Macro F1": round(um["full"][m]["macro_f1"], 4),
            "ROC-AUC": round(um["full"][m].get("roc_auc_ovr", 0), 4),
        })

    # UNSW binary
    ub = json.loads((REPORTS / "metrics_unsw" / "binary" / "binary_metrics.json").read_text())
    for m in models:
        f1_key = "f1_at_threshold" if "f1_at_threshold" in ub[m] else "f1"
        rows.append({
            "Dataset": "UNSW-NB15", "Split": "Random", "Task": "Binary",
            "Model": m, "Macro F1": round(ub[m][f1_key], 4),
            "ROC-AUC": round(ub[m]["roc_auc"], 4),
        })

    # LODO means
    lodo = pd.read_csv(REPORTS / "lodo" / "lodo_summary.csv")
    for _, r in lodo.iterrows():
        rows.append({
            "Dataset": "CICIDS2017", "Split": "LODO (mean)", "Task": "Binary",
            "Model": r["model"],
            "Macro F1": round(r["f1_mean"], 4),
            "ROC-AUC": round(r["roc_auc_mean"], 4),
        })

    df = pd.DataFrame(rows)
    csv_path = REPORTS / "master_results_table.csv"
    df.to_csv(csv_path, index=False)

    md_path = REPORTS / "master_results_table.md"
    df.to_markdown(md_path, index=False)

    print(f"  wrote {csv_path}")
    print(f"  wrote {md_path}")


# ── 4. Generalisation gap — all models, multi + binary ───────────────────────

def fig_generalisation_gap():
    """Strat vs day F1 for all 4 models, multiclass + binary side by side."""
    models = ["logreg", "random_forest", "xgboost", "lightgbm"]
    labels = ["LogReg", "RF", "XGBoost", "LightGBM"]
    colors_strat = "#3498db"
    colors_day = "#e67e22"

    # Multi-class
    mc_s = json.loads((REPORTS / "metrics_strat" / "metrics.json").read_text())
    mc_d = json.loads((REPORTS / "metrics_day" / "metrics.json").read_text())
    mc_strat = [mc_s["full"][m]["macro_f1"] for m in models]
    mc_day = [mc_d["full"][m]["macro_f1"] for m in models]

    # Binary
    bs = json.loads((REPORTS / "metrics_strat_binary" / "binary_metrics.json").read_text())
    bd = json.loads((REPORTS / "metrics_day_binary" / "binary_metrics.json").read_text())
    bi_strat = [bs[m]["f1_at_threshold"] for m in models]
    bi_day = [bd[m]["f1_at_threshold"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(models))
    w = 0.32

    for ax, strat_vals, day_vals, title in zip(
        axes,
        [mc_strat, bi_strat],
        [mc_day, bi_day],
        ["Multi-class Macro F1", "Binary F1"]
    ):
        bars1 = ax.bar(x - w/2, strat_vals, w, label="Stratified", color=colors_strat, alpha=0.85)
        bars2 = ax.bar(x + w/2, day_vals, w, label="Day", color=colors_day, alpha=0.85)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=9)

        # Add delta annotations
        for i in range(len(models)):
            delta = day_vals[i] - strat_vals[i]
            mid_x = x[i]
            mid_y = max(strat_vals[i], day_vals[i]) + 0.06
            color = "#27ae60" if delta >= 0 else "#c0392b"
            ax.text(mid_x, mid_y, f"Δ{delta:+.2f}", ha="center", fontsize=8,
                    color=color, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Generalisation Gap: Stratified vs Day Split — All Models", fontsize=14, y=1.01)
    plt.tight_layout()
    out = FIG_DIR / "generalisation_gap_all_models.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ── 5. Feature importance overlap: CICIDS vs UNSW ───────────────────────────

def fig_feature_overlap():
    """Side-by-side top-10 features for XGBoost on CICIDS vs UNSW."""
    cicids = pd.read_csv(REPORTS / "metrics_strat" / "feature_importance_xgboost.csv")
    unsw = pd.read_csv(REPORTS / "metrics_unsw" / "multiclass" / "feature_importance_xgboost.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, df, title, color in zip(
        axes,
        [cicids.head(15), unsw.head(15)],
        ["CICIDS2017 — XGBoost Top 15", "UNSW-NB15 — XGBoost Top 15"],
        ["#3498db", "#e74c3c"]
    ):
        ax.barh(range(len(df)-1, -1, -1), df["importance"], color=color, alpha=0.8)
        ax.set_yticks(range(len(df)-1, -1, -1))
        ax.set_yticklabels(df["feature"], fontsize=9)
        ax.set_xlabel("Importance", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Feature Importance Comparison Across Datasets", fontsize=14, y=1.01)
    plt.tight_layout()
    out = FIG_DIR / "feature_importance_cross_dataset.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    print("[extra figures] generating...")
    fig_lodo_folds()
    fig_calibration()
    fig_master_table()
    fig_generalisation_gap()
    fig_feature_overlap()
    print("[extra figures] done.")


if __name__ == "__main__":
    main()
