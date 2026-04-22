from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from src.config import DATA_FILE, NON_FEATURE, MODELS_DIR, REPORTS_DIR, FIGURES_DIR

warnings.filterwarnings("ignore")

OUT = REPORTS_DIR / "operating_points"

FPR_TARGETS = [0.0001, 0.001, 0.01]   # 0.01%, 0.1%, 1%
ECE_BINS = 15


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                                n_bins: int = ECE_BINS) -> float:
    """Equal-width-bin ECE (Naeini et al., 2015)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if not mask.any():
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        weight = mask.sum() / n
        ece += weight * abs(bin_conf - bin_acc)
    return float(ece)


def _recall_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float) -> tuple[float, float, float]:
    """Return (recall, actual_fpr, threshold) at the largest threshold with FPR ≤ target."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    mask = fpr <= target_fpr
    if not mask.any():
        return 0.0, 0.0, 1.0
    idx = np.where(mask)[0].max()
    return float(tpr[idx]), float(fpr[idx]), float(thr[idx])


def _load_binary_test():
    df = pd.read_parquet(DATA_FILE)
    test = df[df["split_day"] == "test"].copy()
    feat_cols = [c for c in test.columns if c not in NON_FEATURE]
    return test[feat_cols], test["is_attack"].to_numpy().astype(int)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("[ops] loading data ...")
    X_test, y_test = _load_binary_test()
    print(f"[ops] test n={len(y_test):,}  attack rate={y_test.mean():.4f}")

    model_dir = MODELS_DIR / "binary_day"
    models = ["logreg", "random_forest", "xgboost", "lightgbm"]

    results: dict[str, dict] = {}
    for name in models:
        path = model_dir / f"{name}.joblib"
        if not path.exists():
            print(f"[ops] skip {name}")
            continue
        model = joblib.load(path)
        proba = model.predict_proba(X_test)[:, 1]

        ece = _expected_calibration_error(y_test, proba)
        ops = {}
        for t in FPR_TARGETS:
            rec, fpr_act, thr = _recall_at_fpr(y_test, proba, t)
            ops[f"fpr_{t}"] = {
                "target_fpr": t,
                "actual_fpr": fpr_act,
                "recall": rec,
                "threshold": thr,
            }
        results[name] = {"ece": ece, "operating_points": ops}
        print(f"[ops] {name}: ECE={ece:.4f}  "
              f"recall@0.1%FPR={ops['fpr_0.001']['recall']:.4f}  "
              f"recall@1%FPR={ops['fpr_0.01']['recall']:.4f}")

    (OUT / "operating_points.json").write_text(json.dumps(results, indent=2))

    lines = [
        "# Operating Points + Calibration (Binary Day Split)",
        "",
        f"- Test set: Friday, n = {len(y_test):,} flows, attack rate = {y_test.mean():.4f}",
        "- ECE: Expected Calibration Error with 15 equal-width bins.",
        "- Operating points: largest threshold keeping FPR below the target.",
        "",
        "## Recall at FPR budgets",
        "",
        "A 0.1% FPR on a 2M-flow day = 2,000 false alarms/day. "
        "A 0.01% FPR = 200 false alarms/day (manageable by one analyst).",
        "",
        "| Model | ECE ↓ | Recall @ 0.01% FPR | Recall @ 0.1% FPR | Recall @ 1% FPR |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        op = r["operating_points"]
        lines.append(
            f"| {name} | {r['ece']:.4f} | "
            f"{op['fpr_0.0001']['recall']:.4f} | "
            f"{op['fpr_0.001']['recall']:.4f} | "
            f"{op['fpr_0.01']['recall']:.4f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- **ECE** measures how well predicted probability matches observed",
        "  frequency. < 0.05 is well-calibrated; > 0.15 needs isotonic / Platt.",
        "- **Recall at fixed FPR** is the right metric for comparing IDS models",
        "  under deployment constraints, because it bounds the cost (false alerts)",
        "  and asks what fraction of real attacks are caught. Models with similar",
        "  ROC-AUC can diverge sharply at low FPR, which is the operational regime.",
        "",
    ]
    (OUT / "operating_points.md").write_text("\n".join(lines))

    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(results.keys())
    x = np.arange(len(names))
    w = 0.25
    vals = {t: [results[n]["operating_points"][f"fpr_{t}"]["recall"] for n in names]
            for t in FPR_TARGETS}
    ax.bar(x - w, vals[0.0001], w, label="FPR = 0.01%", color="#C44E52")
    ax.bar(x,     vals[0.001],  w, label="FPR = 0.1%",  color="#DD8452")
    ax.bar(x + w, vals[0.01],   w, label="FPR = 1%",    color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Recall (true positive rate)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Operating points — recall at fixed FPR budgets (Day split, Friday test)")
    ax.legend()
    for i, name in enumerate(names):
        for off, t in zip([-w, 0, w], FPR_TARGETS):
            v = results[name]["operating_points"][f"fpr_{t}"]["recall"]
            ax.text(i + off, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig_path = FIGURES_DIR / "operating_points.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[ops] wrote {OUT}")
    print(f"[ops] wrote {fig_path}")


if __name__ == "__main__":
    main()
