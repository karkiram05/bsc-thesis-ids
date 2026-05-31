from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import auc, f1_score

from src.config import DATA_FILE

# ── paths ────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT_DIR = ROOT / "thesis" / "overleaf" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── style ────────────────────────────────────────────────────────────

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.grid": False,
})

MODELS = ["logreg", "random_forest", "xgboost", "lightgbm"]
MODEL_LABELS = {"logreg": "LogReg", "random_forest": "RF",
                "xgboost": "XGBoost", "lightgbm": "LightGBM"}
COLORS = {"logreg": "#e74c3c", "random_forest": "#27ae60",
          "xgboost": "#2980b9", "lightgbm": "#8e44ad"}

DPI = 200

# ── helpers ──────────────────────────────────────────────────────────

def _save(fig, name: str) -> None:
    out = OUT_DIR / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


def _require(p: Path) -> Path:
    if not p.exists():
        raise SystemExit(f"Missing {p}. Run `make full` first.")
    return p


def _load_macro_f1(metrics_json: Path) -> list[float]:
    data = json.loads(_require(metrics_json).read_text())
    summary = data.get("summary", data)
    return [float(summary[m]["macro_f1"]) for m in MODELS]


def _load_binary_f1(binary_metrics_json: Path) -> list[float]:
    data = json.loads(_require(binary_metrics_json).read_text())
    return [float(data[m]["f1_at_threshold"]) for m in MODELS]


def _load_unsw_binary_f1(binary_metrics_json: Path) -> list[float]:
    data = json.loads(_require(binary_metrics_json).read_text())
    return [float(data[m]["f1"]) for m in MODELS]


def _load_preds(split: str, model: str) -> pd.DataFrame:
    return pd.read_csv(REPORTS / f"metrics_{split}" / f"predictions_{model}.csv")


def _load_cm(path: Path) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(path, index_col=0)
    labels = list(df.index.astype(str))
    return df.values.astype(float), labels


# ── confusion matrix ─────────────────────────────────────────────────

def fig_confusion_matrix() -> None:
    """Side-by-side confusion matrices (stratified vs day) for RF."""
    strat_cm, labels_s = _load_cm(
        REPORTS / "metrics_strat" / "confusion_matrix_random_forest.csv")

    perclass = REPORTS / "metrics_day" / "confusion_matrix_random_forest_perclass.csv"
    if perclass.exists():
        day_cm, labels_d = _load_cm(perclass)
    else:
        day_cm, labels_d = _load_cm(
            REPORTS / "metrics_day" / "confusion_matrix_random_forest.csv")

    def _filter_rows(cm, labels):
        keep = cm.sum(axis=1) > 0
        return cm[keep, :], [l for l, k in zip(labels, keep) if k]

    strat_cm, labels_s_rows = _filter_rows(strat_cm, labels_s)
    day_cm, labels_d_rows = _filter_rows(day_cm, labels_d)

    def _norm(cm):
        row = cm.sum(axis=1, keepdims=True)
        return np.divide(cm, row, where=row != 0, out=np.zeros_like(cm))

    s_norm = _norm(strat_cm)
    d_norm = _norm(day_cm)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8.5),
                             gridspec_kw={"width_ratios": [3, 2.2]})
    fig.suptitle(
        "Random forest confusion matrix (row-normalised recall): "
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
        ax.set_xticklabels(col_lbl, rotation=45, ha="right", fontsize=10,
                           color="black")
        ax.set_yticklabels(row_lbl, fontsize=11, color="black")
        ax.set_xlabel("Predicted label", fontsize=12)
        ax.set_ylabel("True label", fontsize=12)
        ax.set_title(title, fontsize=13, pad=10)
        for i in range(nrow):
            for j in range(ncol):
                v = mat[i, j]
                if v < 1e-3:
                    continue
                same_class = (row_lbl[i] == col_lbl[j])
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=10,
                        fontweight="bold" if same_class else "normal")
                if same_class:
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor="#27ae60", linewidth=1.6))
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02).ax.tick_params(
            labelsize=10)

    plt.tight_layout()
    _save(fig, "confusion_matrix.png")


def fig_evasion_storyboard() -> None:
    """2x2 storyboard summarising the adversarial attack."""
    adv = json.loads(
        (REPORTS / "adversarial" / "adversarial_results.json").read_text())
    rac_u = adv["robust_accuracy_curve_undefended"]
    rac_d = adv["robust_accuracy_curve_defended"]
    top_feats = adv["top_features_exploited"][:6]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        r"Adversarial evasion: $\varepsilon = 0.25\sigma$, "
        "score-query black-box, 19 perturbable features",
        fontsize=20, fontweight="bold", y=1.00)

    ax = axes[0, 0]
    ax.axis("off")
    ax.set_title("A. Headline numbers", fontsize=18, loc="left", pad=10)
    bounded_undef = 1.0 - adv["robust_accuracy_curve_undefended"]["0.25σ"]
    bounded_def = 1.0 - adv["robust_accuracy_curve_defended"]["0.25σ"]
    headline = (
        f"Undefended random forest\n"
        f"  Clean F1 (binary day):       {adv['clean_undefended']['f1']:.3f}\n"
        f"  Evasion @ $\\varepsilon = 0.25\\sigma$:    "
        f"{bounded_undef:.1%}\n"
        f"  Median $L^\\infty$ to flip (succ): "
        f"{adv['median_linf_to_flip_undefended_sigma']:.3f}$\\sigma$\n\n"
        f"Naive adversarial training\n"
        f"  Clean F1 (binary day):       {adv['clean_defended']['f1']:.3f}\n"
        f"  Evasion @ $\\varepsilon = 0.25\\sigma$:    "
        f"{bounded_def:.1%}\n"
        f"  Median $L^\\infty$ to flip (succ): "
        f"{adv['median_linf_to_flip_defended_sigma']:.3f}$\\sigma$\n\n"
        f"Transfer to XGBoost: "
        f"{adv['transferability'].get('xgb_evasion_on_rf_adversarial', 0):.1%}"
    )
    ax.text(0.02, 0.95, headline, ha="left", va="top",
            fontsize=15, family="monospace", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.6",
                      facecolor="#f4f4f4", edgecolor="#888"))

    ax = axes[0, 1]
    if top_feats:
        names = [t[0] if isinstance(t, (list, tuple)) else t["feature"]
                 for t in top_feats]
        shares = [t[1] if isinstance(t, (list, tuple))
                  else t.get("share_of_successful_evasions",
                             t.get("share", 0))
                  for t in top_feats]
        y = np.arange(len(names))[::-1]
        bars = ax.barh(y, shares, color="#c0392b", alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=14)
        ax.set_xlim(0, max(shares) * 1.15)
        ax.set_xlabel("Share of successful evasions", fontsize=14)
        ax.set_title("B. Top features exploited by the attack",
                     fontsize=18, loc="left", pad=10)
        for bar, val in zip(bars, shares):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1%}", va="center", fontsize=14)
        ax.tick_params(axis="x", labelsize=12)
        ax.spines[["top", "right"]].set_visible(False)
    else:
        ax.axis("off")

    ax = axes[1, 0]

    def _curve(rac):
        items = sorted(rac.items(),
                       key=lambda kv: float(kv[0].replace("σ", "")))
        eps = np.array([float(k.replace("σ", "")) for k, _ in items])
        acc = np.array([v for _, v in items])
        return eps, acc

    eps_u, acc_u = _curve(rac_u)
    eps_d, acc_d = _curve(rac_d)
    ax.plot(eps_u, acc_u, "-o", color="#c0392b", linewidth=2.8,
            markersize=9, label="Undefended RF")
    ax.plot(eps_d, acc_d, "-s", color="#27ae60", linewidth=2.8,
            markersize=9, label="AT-defended RF")
    ax.axvline(0.25, color="#555", linestyle="--", linewidth=1.8,
               label=r"Deployment budget $\varepsilon = 0.25\sigma$")
    ax.set_xlabel(r"Perturbation budget $\varepsilon$ ($\sigma$ units)",
                  fontsize=14)
    ax.set_ylabel("Robust accuracy on attack flows", fontsize=14)
    ax.set_title("C. Robust accuracy vs perturbation budget",
                 fontsize=18, loc="left", pad=10)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=13)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=12)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 1]
    transfer = adv["transferability"]
    labels_t = ["XGBoost on\nclean attack flows",
                "XGBoost on\nRF-crafted adv. flows"]
    vals_t = [transfer.get("xgb_evasion_on_clean", 0),
              transfer.get("xgb_evasion_on_rf_adversarial", 0)]
    colors_t = ["#7f8c8d", "#c0392b"]
    bars = ax.bar(labels_t, vals_t, color=colors_t, alpha=0.9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Evasion / misclassification rate", fontsize=14)
    ax.set_title(r"D. Transferability: RF $\to$ XGBoost",
                 fontsize=18, loc="left", pad=10)
    for bar, val in zip(bars, vals_t):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.1%}", ha="center", fontsize=15, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=12)

    plt.tight_layout()
    _save(fig, "evasion_storyboard.png")


def fig_feature_importance_xgb() -> None:
    """XGBoost top-20 gain-based importance (stratified split)."""
    df = pd.read_csv(
        REPORTS / "metrics_strat" / "feature_importance_xgboost.csv")
    top = df.head(20).iloc[::-1]

    fig, ax = plt.subplots(figsize=(12, 9))
    bars = ax.barh(range(len(top)), top["importance"],
                   color="#2980b9", alpha=0.88)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=12)
    ax.set_xlabel("Gain-based importance", fontsize=13)
    ax.set_title("XGBoost feature importance (top 20, stratified split)",
                 fontsize=15, pad=12, fontweight="bold")
    for bar, val in zip(bars, top["importance"]):
        ax.text(val * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10, color="#333")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    _save(fig, "feature_importance_xgboost.png")


def fig_feature_importance_rf() -> None:
    """Random forest top-20 Gini importance (stratified split)."""
    df = pd.read_csv(
        REPORTS / "metrics_strat" / "feature_importance_random_forest.csv")
    top = df.head(20).iloc[::-1]

    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(range(len(top)), top["importance"],
                   color="#27ae60", alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=11)
    ax.set_xlabel("Mean decrease in impurity (Gini)", fontsize=12)
    ax.set_title("Random Forest feature importance, top 20, stratified split",
                 fontsize=14, fontweight="bold", pad=12)
    for bar, val in zip(bars, top["importance"]):
        ax.text(val * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, color="#333")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    _save(fig, "feature_importance_random_forest.png")


def fig_feature_importance_lgbm() -> None:
    """LightGBM top-20 gain importance (stratified split)."""
    df = pd.read_csv(
        REPORTS / "metrics_strat" / "feature_importance_lightgbm.csv")
    top = df.head(20).iloc[::-1]

    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(range(len(top)), top["importance"],
                   color="#8e44ad", alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=11)
    ax.set_xlabel("Gain-based importance", fontsize=12)
    ax.set_title("LightGBM feature importance, top 20, stratified split",
                 fontsize=14, fontweight="bold", pad=12)
    for bar, val in zip(bars, top["importance"]):
        ax.text(val * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}", va="center", fontsize=9, color="#333")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    _save(fig, "feature_importance_lightgbm.png")


def fig_roc_curves() -> None:
    """Binary day-split ROC curves, all 4 models."""
    metrics = json.loads(
        (REPORTS / "metrics_day_binary" / "binary_metrics.json").read_text())

    fig, ax = plt.subplots(figsize=(8, 6))
    for model in MODELS:
        csv_path = REPORTS / "metrics_day_binary" / f"binary_roc_curve_{model}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        fpr, tpr = df["fpr"].values, df["tpr"].values
        roc_auc = metrics.get(model, {}).get("roc_auc", auc(fpr, tpr))
        ax.plot(fpr, tpr, color=COLORS[model], linewidth=2.2,
                label=f"{MODEL_LABELS[model]} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves, binary day split (Friday test)",
                 fontweight="bold")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, "roc_curves.png")


def fig_pr_curves_minority() -> None:
    """Binary day-split precision-recall curves, all 4 models."""
    metrics_path = REPORTS / "metrics_day_binary" / "binary_metrics.json"
    pr_aucs = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    fig, ax = plt.subplots(figsize=(9, 6))
    for model in MODELS:
        csv_path = (REPORTS / "metrics_day_binary" /
                    f"binary_pr_curve_{model}.csv")
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        prec, rec = df["precision"].values, df["recall"].values
        # Use sklearn's average_precision_score from JSON for label consistency
        # with thesis tables. Trapezoidal auc(rec, prec) gives a slightly
        # different value (~0.02 higher for RF) due to step-vs-trapezoid.
        ap = pr_aucs.get(model, {}).get("pr_auc")
        if ap is None:
            order = np.argsort(rec)
            ap = auc(rec[order], prec[order])
        ax.plot(rec, prec, color=COLORS[model], linewidth=2,
                label=f"{MODEL_LABELS[model]} (PR-AUC = {ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves, binary day split (Friday test)",
                 fontweight="bold")
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    _save(fig, "pr_curves_minority.png")


def fig_per_class_f1() -> None:
    """Per-class binary F1 heatmap across all 4 models (stratified)."""
    all_classes = None
    scores: dict[str, dict[str, float]] = {}
    for model in MODELS:
        df = _load_preds("strat", model)
        classes = sorted(df["true_label"].unique())
        if all_classes is None:
            all_classes = classes
        per = {}
        for cls in all_classes:
            y_t = (df["true_label"] == cls).astype(int).values
            y_p = (df["pred_label"] == cls).astype(int).values
            per[cls] = f1_score(y_t, y_p, zero_division=0)
        scores[MODEL_LABELS[model]] = per

    mat = pd.DataFrame(scores).T[all_classes]

    fig, ax = plt.subplots(figsize=(max(14, len(all_classes) * 0.9), 4.5))
    im = ax.imshow(mat.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(all_classes)))
    ax.set_xticklabels(all_classes, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index, fontsize=12)
    ax.set_title("Per-class F1 score by model, stratified split\n"
                 "(green = high, red = low)", fontsize=14, fontweight="bold")
    for i in range(len(mat)):
        for j, cls in enumerate(all_classes):
            v = mat.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if (v > 0.7 or v < 0.2) else "black",
                    fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, shrink=0.9, pad=0.01)
    cbar.set_label("F1 score", fontsize=11)
    plt.tight_layout()
    _save(fig, "per_class_f1.png")


def fig_calibration_curves() -> None:
    """Calibration curves for binary classifiers, both splits side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Calibration curves, fraction of positives vs mean predicted probability",
        fontsize=14, fontweight="bold")

    split_dirs = [
        ("metrics_strat_binary", "Stratified split"),
        ("metrics_day_binary", "Day split (Friday test)"),
    ]
    all_models = ["logreg", "random_forest", "xgboost", "lightgbm"]
    for ax, (split_dir, title) in zip(axes, split_dirs):
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect",
                alpha=0.7)
        base = REPORTS / split_dir
        for model in all_models:
            csv_path = base / f"binary_calibration_curve_{model}.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            ax.plot(df["mean_predicted_prob"], df["fraction_positives"],
                    "o-", color=COLORS[model], linewidth=2, markersize=5,
                    label=MODEL_LABELS[model])
        ax.set_xlabel("Mean predicted probability", fontsize=12)
        ax.set_ylabel("Fraction of positives", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(loc="upper left", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, "calibration_curves.png")


def fig_day_attack_distribution() -> None:
    """Attack types per day, shows why multi-class fails on day split."""
    df = pd.read_parquet(_require(DATA_FILE), columns=["day", "attack_type"])
    counts = df.groupby(["day", "attack_type"]).size().reset_index(name="n")
    raw: dict[str, dict[str, int]] = {}
    for _, r in counts.iterrows():
        raw.setdefault(str(r["day"]), {})[str(r["attack_type"])] = int(r["n"])

    day_order = [
        ("Monday", "Monday\n(Train)"),
        ("Tuesday", "Tuesday\n(Train)"),
        ("Wednesday", "Wednesday\n(Train)"),
        ("Thursday", "Thursday\n(Val)"),
        ("Friday", "Friday\n(Test)"),
    ]
    days_data = {label: raw.get(d, {}) for d, label in day_order}

    tactic_colors = {
        "Benign": "#95a5a6",
        "FTP-Patator": "#e74c3c", "SSH-Patator": "#c0392b",
        "DoS Hulk": "#1f77b4", "DoS GoldenEye": "#3498db",
        "DoS slowloris": "#5dade2", "DoS Slowhttptest": "#85c1e9",
        "Heartbleed": "#FF1493",
        "Web Attack-Brute Force": "#e67e22", "Web Attack-XSS": "#f39c12",
        "Infiltration": "#2ecc71", "Web Attack-Sql Injection": "#d35400",
        "DDoS": "#1f3a93", "PortScan": "#27ae60", "Bot": "#8e44ad",
    }

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(days_data))

    for i, (day, attacks) in enumerate(days_data.items()):
        bottom = 0
        for attack, count in sorted(attacks.items(), key=lambda kv: -kv[1]):
            if attack == "Benign":
                continue
            color = tactic_colors.get(attack, "#bdc3c7")
            ax.bar(i, count, 0.6, bottom=bottom, color=color, label=attack,
                   edgecolor="white", linewidth=0.5)
            if count > 1000:
                ax.text(i, bottom + count / 2, f"{attack}\n({count:,})",
                        ha="center", va="center", fontsize=7,
                        fontweight="bold", color="white")
            bottom += count

    ax.axvspan(-0.5, 2.5, alpha=0.04, color="#6fa8dc", zorder=0)
    ax.axvspan(2.5, 3.5, alpha=0.04, color="#f6b26b", zorder=0)
    ax.axvspan(3.5, 4.5, alpha=0.04, color="#e06666", zorder=0)
    ylim = ax.get_ylim()[1]
    ax.text(1, ylim * 0.92, "TRAIN", ha="center", fontsize=13,
            fontweight="bold", color="#1f3a93", alpha=0.45)
    ax.text(3, ylim * 0.92, "VAL", ha="center", fontsize=13,
            fontweight="bold", color="#a0522d", alpha=0.45)
    ax.text(4, ylim * 0.92, "TEST", ha="center", fontsize=13,
            fontweight="bold", color="#8b0000", alpha=0.45)

    ax.set_xticks(x)
    ax.set_xticklabels(list(days_data.keys()), fontsize=11)
    ax.set_ylabel("Number of attack flows (log scale)", fontsize=12)
    ax.set_title(
        "CICIDS2017 attack types per day under the day-based split\n"
        "Friday test attacks (DDoS, PortScan, Bot) never appear in training",
        fontsize=13, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(1, 500000)
    ax.grid(axis="y", alpha=0.3)

    handles, seen = [], set()
    for day, attacks in days_data.items():
        for attack in attacks:
            if attack != "Benign" and attack not in seen:
                handles.append(
                    Patch(facecolor=tactic_colors.get(attack, "#bdc3c7"),
                          label=attack))
                seen.add(attack)
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)

    plt.tight_layout()
    _save(fig, "day_attack_distribution.png")


def fig_class_imbalance() -> None:
    """Class distribution bar chart (log scale)."""
    df = pd.read_parquet(_require(DATA_FILE), columns=["attack_type"])
    s = df["attack_type"].value_counts().sort_values(ascending=False)
    counts = [(str(k), int(v)) for k, v in s.items()]

    color_map = {
        "Benign": "#95a5a6",
        "DoS Hulk": "#8B0000", "DDoS": "#800000",
        "DoS GoldenEye": "#A52A2A", "FTP-Patator": "#e74c3c",
        "DoS slowloris": "#CD5C5C", "DoS Slowhttptest": "#DC143C",
        "SSH-Patator": "#c0392b", "PortScan": "#27ae60",
        "Web Attack-Brute Force": "#e67e22", "Bot": "#8e44ad",
        "Web Attack-XSS": "#f39c12", "Infiltration": "#2ecc71",
        "Web Attack-Sql Injection": "#d35400", "Heartbleed": "#FF1493",
    }

    names = [c[0] for c in counts]
    flow_counts = [c[1] for c in counts]
    colors = [color_map.get(n, "#bdc3c7") for n in names]

    benign_share = flow_counts[0] / sum(flow_counts) * 100
    ratio = flow_counts[0] // max(flow_counts[-1], 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(names) - 1, -1, -1), flow_counts, color=colors,
            edgecolor="white")
    ax.set_yticks(range(len(names) - 1, -1, -1))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("Flow count (log scale)", fontsize=12)
    ax.set_title(
        f"CICIDS2017 class distribution, extreme imbalance\n"
        f"Benign = {benign_share:.1f}% | rarest class = {flow_counts[-1]} flows",
        fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    for i, (name, count) in enumerate(counts):
        ax.text(count * 1.3, len(counts) - 1 - i, f"{count:,}",
                va="center", fontsize=9, fontweight="bold")

    ax.annotate(f"Ratio: {ratio:,}:1\n(Benign:rarest)",
                xy=(0.75, 0.15), xycoords="axes fraction",
                fontsize=11, fontweight="bold", color="#c0392b",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="lightyellow", edgecolor="#c0392b"))
    plt.tight_layout()
    _save(fig, "class_imbalance.png")


def fig_cross_dataset() -> None:
    """CICIDS vs UNSW F1 comparison."""
    cicids_mc = _load_macro_f1(REPORTS / "metrics_strat" / "metrics.json")
    unsw_mc = _load_macro_f1(
        REPORTS / "metrics_unsw" / "multiclass" / "metrics.json")
    cicids_bi = _load_binary_f1(
        REPORTS / "metrics_strat_binary" / "binary_metrics.json")
    unsw_bi = _load_unsw_binary_f1(
        REPORTS / "metrics_unsw" / "binary" / "binary_metrics.json")

    labels = [MODEL_LABELS[m] for m in MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(labels))
    w = 0.32

    for ax, cv, uv, title in zip(
        axes, [cicids_mc, cicids_bi], [unsw_mc, unsw_bi],
        ["Multi-class Macro F1", "Binary F1"],
    ):
        bars1 = ax.bar(x - w / 2, cv, w, label="CICIDS2017 (strat)",
                       color="#3498db", alpha=0.85)
        bars2 = ax.bar(x + w / 2, uv, w, label="UNSW-NB15",
                       color="#e67e22", alpha=0.85)
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=9,
                        fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel("F1 Score", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Cross-dataset comparison: CICIDS2017 vs UNSW-NB15\n"
        "No universal winner, model ranking flips between datasets",
        fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    _save(fig, "cross_dataset_comparison.png")


def fig_lodo_folds() -> None:
    """LODO fold-level ROC-AUC and F1 bar chart."""
    data = json.loads(
        (REPORTS / "lodo" / "lodo_results.json").read_text())
    df = pd.DataFrame(data)

    labels = [MODEL_LABELS[m] for m in MODELS]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    colors_list = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title in zip(
        axes, ["roc_auc", "f1"],
        ["ROC-AUC per Fold", "F1 per Fold"],
    ):
        x = np.arange(len(days))
        width = 0.18
        for i, (model, label) in enumerate(zip(MODELS, labels)):
            vals = []
            for day in days:
                row = df[(df["model"] == model) & (df["held_out_day"] == day)]
                if len(row) == 0:
                    vals.append(0)
                    continue
                v = row[metric].values[0]
                vals.append(v if v is not None and not pd.isna(v) else 0)
            bars = ax.bar(x + i * width, vals, width, label=label,
                          color=colors_list[i], alpha=0.85)
            # Monday is benign-only, mark with hatching
            bars[0].set_hatch("//")
            bars[0].set_alpha(0.4)

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(days, fontsize=10)
        ax.set_ylabel(metric.upper().replace("_", "-"), fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Leave-One-Day-Out binary cross-validation\n"
        "(Monday is benign-only: ROC-AUC and F1 are undefined; bars omitted)",
        fontsize=14, y=1.02)
    plt.tight_layout()
    _save(fig, "lodo_folds.png")


def fig_generalisation_gap() -> None:
    """Strat vs day F1 for all models, multi-class and binary."""
    labels = [MODEL_LABELS[m] for m in MODELS]

    mc_s = json.loads(
        (REPORTS / "metrics_strat" / "metrics.json").read_text())
    mc_d = json.loads(
        (REPORTS / "metrics_day" / "metrics.json").read_text())
    mc_strat = [mc_s["full"][m]["macro_f1"] for m in MODELS]
    mc_day = [mc_d["full"][m]["macro_f1"] for m in MODELS]

    bs = json.loads(
        (REPORTS / "metrics_strat_binary" / "binary_metrics.json").read_text())
    bd = json.loads(
        (REPORTS / "metrics_day_binary" / "binary_metrics.json").read_text())
    bi_strat = [bs[m]["f1_at_threshold"] for m in MODELS]
    bi_day = [bd[m]["f1_at_threshold"] for m in MODELS]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(MODELS))
    w = 0.32

    for ax, strat_vals, day_vals, title in zip(
        axes, [mc_strat, bi_strat], [mc_day, bi_day],
        ["Multi-class Macro F1", "Binary F1"],
    ):
        bars1 = ax.bar(x - w / 2, strat_vals, w, label="Stratified",
                       color="#3498db", alpha=0.85)
        bars2 = ax.bar(x + w / 2, day_vals, w, label="Day",
                       color="#e67e22", alpha=0.85)
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=9)
        for i in range(len(MODELS)):
            delta = day_vals[i] - strat_vals[i]
            mid_y = max(strat_vals[i], day_vals[i]) + 0.06
            color = "#27ae60" if delta >= 0 else "#c0392b"
            ax.text(x[i], mid_y, f"Δ{delta:+.2f}", ha="center",
                    fontsize=8, color=color, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Generalisation gap: stratified vs day split",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, "generalisation_gap_all_models.png")


def master_results_table() -> None:
    """Aggregate results CSV and markdown table across all experiments."""
    rows = []

    mc_strat = json.loads(
        (REPORTS / "metrics_strat" / "metrics.json").read_text())
    mc_day = json.loads(
        (REPORTS / "metrics_day" / "metrics.json").read_text())
    bs = json.loads(
        (REPORTS / "metrics_strat_binary" / "binary_metrics.json").read_text())
    bd = json.loads(
        (REPORTS / "metrics_day_binary" / "binary_metrics.json").read_text())
    um = json.loads(
        (REPORTS / "metrics_unsw" / "multiclass" / "metrics.json").read_text())
    ub = json.loads(
        (REPORTS / "metrics_unsw" / "binary" / "binary_metrics.json").read_text())

    for m in MODELS:
        rows.append({"Dataset": "CICIDS2017", "Split": "Stratified",
                     "Task": "Multi-class", "Model": m,
                     "Macro F1": round(mc_strat["full"][m]["macro_f1"], 4),
                     "ROC-AUC": round(mc_strat["full"][m].get("roc_auc_ovr", 0), 4)})
        rows.append({"Dataset": "CICIDS2017", "Split": "Day",
                     "Task": "Multi-class", "Model": m,
                     "Macro F1": round(mc_day["full"][m]["macro_f1"], 4),
                     "ROC-AUC": round(mc_day["full"][m].get("roc_auc_ovr", 0), 4)})
        rows.append({"Dataset": "CICIDS2017", "Split": "Stratified",
                     "Task": "Binary", "Model": m,
                     "Macro F1": round(bs[m]["f1_at_threshold"], 4),
                     "ROC-AUC": round(bs[m]["roc_auc"], 4)})
        rows.append({"Dataset": "CICIDS2017", "Split": "Day",
                     "Task": "Binary", "Model": m,
                     "Macro F1": round(bd[m]["f1_at_threshold"], 4),
                     "ROC-AUC": round(bd[m]["roc_auc"], 4)})
        rows.append({"Dataset": "UNSW-NB15", "Split": "Random",
                     "Task": "Multi-class", "Model": m,
                     "Macro F1": round(um["full"][m]["macro_f1"], 4),
                     "ROC-AUC": round(um["full"][m].get("roc_auc_ovr", 0), 4)})
        f1_key = "f1_at_threshold" if "f1_at_threshold" in ub[m] else "f1"
        rows.append({"Dataset": "UNSW-NB15", "Split": "Random",
                     "Task": "Binary", "Model": m,
                     "Macro F1": round(ub[m][f1_key], 4),
                     "ROC-AUC": round(ub[m]["roc_auc"], 4)})

    lodo = pd.read_csv(REPORTS / "lodo" / "lodo_summary.csv")
    for _, r in lodo.iterrows():
        rows.append({"Dataset": "CICIDS2017", "Split": "LODO (mean)",
                     "Task": "Binary", "Model": r["model"],
                     "Macro F1": round(r["f1_mean"], 4),
                     "ROC-AUC": round(r["roc_auc_mean"], 4)})

    df = pd.DataFrame(rows)
    csv_path = REPORTS / "master_results_table.csv"
    df.to_csv(csv_path, index=False)
    md_path = REPORTS / "master_results_table.md"
    df.to_markdown(md_path, index=False)
    print(f"  wrote {csv_path}")
    print(f"  wrote {md_path}")


def copy_external_figures() -> None:
    """Copy figures generated by other scripts into the thesis figure dir."""
    search_dirs = [
        ROOT / "reports" / "figures",
        ROOT / "reports" / "traffic_analysis" / "figures",
    ]
    external = [
        "shap_summary_rf_binary.png",
        "operating_points.png",
        "adversarial_robustness.png",
        "tactic_distribution.png",
        "shap_waterfall_attack.png",
        "shap_waterfall_benign.png",
    ]
    for name in external:
        copied = False
        for src_dir in search_dirs:
            src = src_dir / name
            if src.exists():
                shutil.copy2(src, OUT_DIR / name)
                print(f"  copied {name}")
                copied = True
                break
        if not copied:
            print(f"  skip {name} (not found)")

def fig_reality_collapse() -> None:
    """Single-figure headline: stratified XGB macro F1 vs day-split XGB
    macro F1. Numbers come straight from the metrics JSONs so the figure
    stays in sync with the actual run."""
    strat = json.loads(
        (REPORTS / "metrics_strat" / "metrics.json").read_text())
    day = json.loads(
        (REPORTS / "metrics_day" / "metrics.json").read_text())
    f1_strat = float(strat["summary"]["xgboost"]["macro_f1"])
    f1_day = float(day["summary"]["xgboost"]["macro_f1"])
    drop = f1_strat - f1_day

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(
        ["Published-style\nRandom Stratified Split",
         "Realistic\nTemporal Day Split"],
        [f1_strat, f1_day],
        color=["#1f3a93", "#8b0000"], edgecolor="black", linewidth=1.5,
        width=0.55,
    )
    for bar, val in zip(bars, [f1_strat, f1_day]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.2f}", ha="center", fontsize=22, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("XGBoost Macro F1 on CICIDS2017", fontsize=13,
                  fontweight="bold")
    ax.set_title(
        "The Reality Collapse: Published Benchmarks Overstate Deployment Performance",
        fontsize=14, fontweight="bold", pad=15)
    ax.annotate(
        f"{drop:.2f} collapse",
        xy=(1, f1_day + 0.05), xytext=(0.55, f1_strat - 0.15),
        fontsize=14, color="#8b0000", fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#8b0000", lw=2.2),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#8b0000", linewidth=1.8))
    ax.text(0.5, -0.18,
            "Same model. Same data. Same hyperparameters. Only the evaluation protocol changes.",
            ha="center", va="top", fontsize=11, style="italic",
            color="#555", transform=ax.transAxes)
    ax.set_facecolor("#f5f5f5")
    ax.grid(axis="y", alpha=0.3, color="white", linewidth=1.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, "reality_collapse.png")


def fig_deployment_economics() -> None:
    """Single-budget recall bar chart. Pulls recall@0.1%FPR per model
    straight from operating_points.json so the figure cannot drift."""
    ops = json.loads(
        (REPORTS / "operating_points" / "operating_points.json").read_text())
    names = ["logreg", "random_forest", "xgboost", "lightgbm"]
    pretty = ["LogReg", "Random Forest", "XGBoost", "LightGBM"]
    vals = [ops[m]["operating_points"]["fpr_0.001"]["recall"] * 100
            for m in names]
    colors = ["#7f8c8d", "#1f3a93", "#8b0000", "#e67e22"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(pretty, vals, color=colors, edgecolor="black",
                  linewidth=1.2, width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                f"{v:.1f}%", ha="center", fontsize=16, fontweight="bold")
    ax.set_ylabel("Attack recall (%)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 75)
    ax.set_title(
        "Deployment Economics: Attacks Caught Per Day at a 0.1% FPR Budget\n"
        "(~1,500 false alerts/day on 2M flows = ~250 analyst-hours/day at 10 min/alert)",
        fontsize=12, fontweight="bold", pad=12)
    ax.text(0.5, -0.18,
            f"At a fixed 0.1 percent FPR budget, RF catches {vals[1]:.0f} percent of attacks "
            f"while XGBoost catches {vals[2]:.0f} percent.\n"
            "Low-FPR ROC shape, not peak F1, determines this gap.",
            ha="center", va="top", fontsize=10, style="italic",
            color="#555", transform=ax.transAxes)
    ax.set_facecolor("#f5f5f5")
    ax.grid(axis="y", alpha=0.3, color="white", linewidth=1.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, "deployment_economics.png")


def main() -> None:
    print("[thesis figures] generating...")

    fig_reality_collapse()
    fig_confusion_matrix()
    fig_evasion_storyboard()
    fig_feature_importance_xgb()
    fig_feature_importance_rf()
    fig_feature_importance_lgbm()
    fig_roc_curves()
    fig_pr_curves_minority()
    fig_per_class_f1()
    fig_calibration_curves()
    fig_day_attack_distribution()
    fig_class_imbalance()
    fig_cross_dataset()
    fig_lodo_folds()
    fig_generalisation_gap()
    fig_deployment_economics()
    master_results_table()
    copy_external_figures()

    print("[thesis figures] done.")


if __name__ == "__main__":
    main()
