"""
Regenerate two figures with thesis-print-friendly styling.

Fixes applied:
1. confusion_matrix.png — white background, larger fonts, single
   model (random forest), stratified vs day side-by-side. Replaces
   the dark-theme GitHub-style version.
2. evasion_storyboard.png — 2x2 storyboard with larger fonts.
   Panel A: clean attack flow score distribution. Panel B: greedy
   perturbation trajectory. Panel C: top-5 exploited features.
   Panel D: robust-accuracy curve vs epsilon.

Run:
    python -m src.regenerate_thesis_figures

Reads:
    reports/metrics_strat/confusion_matrix_random_forest.csv
    reports/metrics_day/confusion_matrix_random_forest.csv
    reports/adversarial/adversarial_results.json

Writes:
    thesis/overleaf/figures/confusion_matrix.png
    thesis/overleaf/figures/evasion_storyboard.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT_DIR = ROOT / "thesis" / "overleaf" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.grid": False,
})


def _load_cm(path: Path) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(path, index_col=0)
    labels = list(df.index.astype(str))
    cm = df.values.astype(float)
    return cm, labels


def fig_confusion_matrix() -> None:
    """Side-by-side confusion matrices (stratified vs day) for RF.

    Each panel shows only the rows (true classes) actually present in
    the test set for that split, but keeps all training-set classes as
    columns (predicted labels) so collapse onto trained-but-wrong
    classes is visible. Day-split right panel therefore has just 4 rows
    (Benign + Friday's 3 attack classes) instead of 15 empty ones.
    """
    strat_cm, labels_s = _load_cm(REPORTS / "metrics_strat" / "confusion_matrix_random_forest.csv")
    # Day-split CM: use the per-class rebuild (src.rebuild_day_confusion_matrix)
    # if available, since the legacy CSV from eval.py collapses Bot/DDoS/PortScan
    # into a single __unseen__ row.
    perclass = REPORTS / "metrics_day" / "confusion_matrix_random_forest_perclass.csv"
    if perclass.exists():
        day_cm, labels_d = _load_cm(perclass)
    else:
        day_cm, labels_d = _load_cm(REPORTS / "metrics_day" / "confusion_matrix_random_forest.csv")

    def _filter_rows(cm: np.ndarray, labels: list[str]):
        # Drop true-class rows whose support is zero (no test data)
        keep = cm.sum(axis=1) > 0
        return cm[keep, :], [l for l, k in zip(labels, keep) if k]

    strat_cm, labels_s_rows = _filter_rows(strat_cm, labels_s)
    day_cm,   labels_d_rows = _filter_rows(day_cm,   labels_d)

    def _norm(cm):
        row = cm.sum(axis=1, keepdims=True)
        return np.divide(cm, row, where=row != 0, out=np.zeros_like(cm))

    s_norm = _norm(strat_cm)
    d_norm = _norm(day_cm)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8.5),
                             gridspec_kw={"width_ratios": [3, 2.2]})
    fig.suptitle("Random forest confusion matrix (row-normalised recall) — "
                 "rows = test classes present, columns = predicted label",
                 fontsize=15, fontweight="bold", y=1.03)

    panels = [
        (axes[0], s_norm, labels_s_rows, labels_s, "Stratified split (test)"),
        (axes[1], d_norm, labels_d_rows, labels_d, "Day split (Friday test only)"),
    ]
    for ax, mat, row_lbl, col_lbl, title in panels:
        nrow, ncol = mat.shape
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(ncol))
        ax.set_yticks(range(nrow))
        ax.set_xticklabels(col_lbl, rotation=45, ha="right", fontsize=10, color="black")
        ax.set_yticklabels(row_lbl, fontsize=11, color="black")
        ax.set_xlabel("Predicted label", fontsize=12)
        ax.set_ylabel("True label", fontsize=12)
        ax.set_title(title, fontsize=13, pad=10)
        thresh = 0.5
        for i in range(nrow):
            for j in range(ncol):
                v = mat[i, j]
                if v < 1e-3:
                    continue
                # mark diagonal (same class label) bold green
                same_class = (row_lbl[i] == col_lbl[j])
                ax.text(j, i, f"{v:.2f}",
                        ha="center", va="center",
                        color="white" if v > thresh else "black",
                        fontsize=10,
                        fontweight="bold" if same_class else "normal")
                if same_class:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               fill=False, edgecolor="#27ae60",
                                               linewidth=1.6))
        cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    out = OUT_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def fig_evasion_storyboard() -> None:
    """2x2 storyboard summarising the adversarial attack."""
    adv = json.loads((REPORTS / "adversarial" / "adversarial_results.json").read_text())
    rac_u = adv["robust_accuracy_curve_undefended"]
    rac_d = adv["robust_accuracy_curve_defended"]
    top_feats = adv["top_features_exploited"][:6]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Adversarial evasion: $\\varepsilon = 0.25\\sigma$, "
                 "score-query black-box, 19 perturbable features",
                 fontsize=16, fontweight="bold", y=1.00)

    # Panel A: headline numbers as text
    ax = axes[0, 0]
    ax.axis("off")
    ax.set_title("A. Headline numbers", fontsize=15, loc="left", pad=10)
    headline = (
        f"Undefended random forest\n"
        f"  Clean F1 (binary day):       {adv['clean_undefended']['f1']:.3f}\n"
        f"  Evasion @ $\\varepsilon = 0.25\\sigma$:    "
        f"{adv['evasion_rate_vs_rf_undefended']:.1%}\n"
        f"  Median $L^\\infty$ to flip (succ): "
        f"{adv['median_linf_to_flip_undefended_sigma']:.3f}$\\sigma$\n\n"
        f"Naive adversarial training\n"
        f"  Clean F1 (binary day):       {adv['clean_defended']['f1']:.3f}\n"
        f"  Evasion @ $\\varepsilon = 0.25\\sigma$:    "
        f"{adv['evasion_rate_vs_rf_defended']:.1%}\n"
        f"  Median $L^\\infty$ to flip (succ): "
        f"{adv['median_linf_to_flip_defended_sigma']:.3f}$\\sigma$\n\n"
        f"Transfer to XGBoost: "
        f"{adv['transferability'].get('xgb_evasion_on_rf_adversarial', 0):.1%}"
    )
    ax.text(0.02, 0.95, headline, ha="left", va="top",
            fontsize=12.5, family="monospace",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.6",
                      facecolor="#f4f4f4", edgecolor="#888"))

    # Panel B: top exploited features
    ax = axes[0, 1]
    if top_feats:
        # top_features_exploited is a list of [feature_name, share] pairs
        names = [t[0] if isinstance(t, (list, tuple)) else t["feature"]
                 for t in top_feats]
        shares = [t[1] if isinstance(t, (list, tuple))
                  else t.get("share_of_successful_evasions",
                             t.get("share", 0))
                  for t in top_feats]
        y = np.arange(len(names))[::-1]
        bars = ax.barh(y, shares, color="#c0392b", alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=11)
        ax.set_xlim(0, max(shares) * 1.15)
        ax.set_xlabel("Share of successful evasions", fontsize=12)
        ax.set_title("B. Top features exploited by the attack",
                     fontsize=15, loc="left", pad=10)
        for bar, val in zip(bars, shares):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1%}", va="center", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    else:
        ax.axis("off")

    # Panel C: robust accuracy curve
    ax = axes[1, 0]
    def _curve(rac):
        # Keys look like "0.25σ", "0.5σ", "1.0σ", "2.0σ"
        items = sorted(rac.items(),
                       key=lambda kv: float(kv[0].replace("σ", "")))
        eps = np.array([float(k.replace("σ", "")) for k, _ in items])
        acc = np.array([v for _, v in items])
        return eps, acc
    eps_u, acc_u = _curve(rac_u)
    eps_d, acc_d = _curve(rac_d)
    ax.plot(eps_u, acc_u, "-o", color="#c0392b", linewidth=2.4,
            markersize=7, label="Undefended RF")
    ax.plot(eps_d, acc_d, "-s", color="#27ae60", linewidth=2.4,
            markersize=7, label="AT-defended RF")
    ax.axvline(0.25, color="#555", linestyle="--", linewidth=1.5,
               label="Deployment budget $\\varepsilon = 0.25\\sigma$")
    ax.set_xlabel("Perturbation budget $\\varepsilon$ ($\\sigma$ units)",
                  fontsize=12)
    ax.set_ylabel("Robust accuracy on attack flows", fontsize=12)
    ax.set_title("C. Robust accuracy vs perturbation budget",
                 fontsize=15, loc="left", pad=10)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel D: transferability (RF-crafted adv -> XGBoost)
    ax = axes[1, 1]
    transfer = adv["transferability"]
    labels_t = ["XGBoost on\nclean attack flows",
                "XGBoost on\nRF-crafted adv. flows"]
    vals_t = [transfer.get("xgb_evasion_on_clean", 0),
              transfer.get("xgb_evasion_on_rf_adversarial", 0)]
    colors_t = ["#7f8c8d", "#c0392b"]
    bars = ax.bar(labels_t, vals_t, color=colors_t, alpha=0.9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Evasion / misclassification rate", fontsize=12)
    ax.set_title("D. Transferability: RF $\\to$ XGBoost",
                 fontsize=15, loc="left", pad=10)
    for bar, val in zip(bars, vals_t):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.1%}", ha="center", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=11)

    plt.tight_layout()
    out = OUT_DIR / "evasion_storyboard.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def fig_feature_importance_xgb() -> None:
    """White-bg, larger-text replacement for feature_importance_xgboost.png."""
    df = pd.read_csv(REPORTS / "metrics_strat" / "feature_importance_xgboost.csv")
    top = df.head(20).iloc[::-1]

    fig, ax = plt.subplots(figsize=(12, 9))
    bars = ax.barh(range(len(top)), top["importance"], color="#2980b9", alpha=0.88)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=12)
    ax.set_xlabel("Gain-based importance", fontsize=13)
    ax.set_title("XGBoost feature importance (top 20, stratified split)",
                 fontsize=15, pad=12, fontweight="bold")
    for bar, val in zip(bars, top["importance"]):
        ax.text(val * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10, color="#333")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    out = OUT_DIR / "feature_importance_xgboost.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    fig_confusion_matrix()
    fig_evasion_storyboard()
    fig_feature_importance_xgb()


if __name__ == "__main__":
    main()
