from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, roc_auc_score, roc_curve,
)

from src.config import (
    DATA_FILE, NON_FEATURE, REPORTS_DIR, FIGURES_DIR, MODELS_DIR, RNG,
    feature_cols,
)

warnings.filterwarnings("ignore")

OUT = REPORTS_DIR / "anomaly"
OUT.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# LOF with novelty=True stores the training set, so keep it small
LOF_TRAIN_SAMPLE = 30_000
# IsolationForest is fast but still does not need all benign rows
IFOREST_TRAIN_SAMPLE = 200_000
FPR_TARGETS = [0.001, 0.01]   # 0.1%, 1% — same budgets as operating_points.py


def _load_day_split():
    df = pd.read_parquet(DATA_FILE)
    feat = feature_cols(df)

    train = df[df["split_day"] == "train"]
    test = df[df["split_day"] == "test"]

    # only benign flows are used to fit the anomaly detectors (pure unsupervised)
    train_benign = train[train["attack_type"].astype(str) == "Benign"]
    X_train = train_benign[feat]
    X_test = test[feat]
    y_test = (test["attack_type"].astype(str) != "Benign").astype(int).to_numpy()
    return X_train, X_test, y_test, feat


def _recall_at_fpr(y_true: np.ndarray, score: np.ndarray, target_fpr: float) -> tuple[float, float, float]:
    # larger score = more anomalous = more likely attack
    fpr, tpr, thr = roc_curve(y_true, score)
    mask = fpr <= target_fpr
    if not mask.any():
        return 0.0, 0.0, 1.0
    idx = np.where(mask)[0].max()
    return float(tpr[idx]), float(fpr[idx]), float(thr[idx])


def _metrics(name: str, y_true: np.ndarray, score: np.ndarray) -> dict:
    roc = float(roc_auc_score(y_true, score))
    pr = float(average_precision_score(y_true, score))
    ops = {}
    for t in FPR_TARGETS:
        rec, fpr_act, thr = _recall_at_fpr(y_true, score, t)
        ops[f"fpr_{t}"] = {"target_fpr": t, "actual_fpr": fpr_act,
                           "recall": rec, "threshold": thr}
    print(f"[anomaly] {name:20s}  ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  "
          f"R@1%FPR={ops['fpr_0.01']['recall']:.4f}")
    return {"roc_auc": roc, "pr_auc": pr, "operating_points": ops}


def _supervised_baseline() -> dict:
    """Read the supervised binary-day numbers already on disk, for comparison."""
    path = REPORTS_DIR / "metrics_day_binary" / "binary_metrics.json"
    if not path.exists():
        return {}
    m = json.loads(path.read_text())
    # keep only roc_auc and pr_auc so table lines up
    out = {}
    for name, r in m.items():
        out[name] = {"roc_auc": r.get("roc_auc"), "pr_auc": r.get("pr_auc")}
    return out


def main():
    print("[anomaly] loading data ...")
    X_train, X_test, y_test, feat = _load_day_split()
    print(f"[anomaly] benign train n={len(X_train):,}  test n={len(y_test):,}  "
          f"test attack rate={y_test.mean():.4f}  features={len(feat)}")

    # scale because LOF is distance-based; IForest does not care but no harm
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # subsample for speed / memory
    rng = np.random.default_rng(RNG)
    if len(X_train_s) > IFOREST_TRAIN_SAMPLE:
        sub_if = rng.choice(len(X_train_s), IFOREST_TRAIN_SAMPLE, replace=False)
        X_if = X_train_s[sub_if]
    else:
        X_if = X_train_s
    if len(X_train_s) > LOF_TRAIN_SAMPLE:
        sub_lof = rng.choice(len(X_train_s), LOF_TRAIN_SAMPLE, replace=False)
        X_lof = X_train_s[sub_lof]
    else:
        X_lof = X_train_s

    results: dict[str, dict] = {}

    # Isolation Forest
    print(f"[anomaly] training IsolationForest on {len(X_if):,} benign rows ...")
    iso = IsolationForest(
        n_estimators=200, contamination="auto",
        max_samples=min(256, len(X_if)),
        random_state=RNG, n_jobs=-1,
    )
    iso.fit(X_if)
    # score_samples: higher = more normal, so flip sign for "anomaly score"
    iso_score = -iso.score_samples(X_test_s)
    results["isolation_forest"] = _metrics("IsolationForest", y_test, iso_score)

    # Local Outlier Factor (novelty mode)
    print(f"[anomaly] training LOF (novelty) on {len(X_lof):,} benign rows ...")
    lof = LocalOutlierFactor(
        n_neighbors=20, novelty=True, n_jobs=-1,
    )
    lof.fit(X_lof)
    lof_score = -lof.score_samples(X_test_s)
    results["lof"] = _metrics("LOF (novelty=True)", y_test, lof_score)

    # Supervised comparison (read from disk, trained elsewhere)
    supervised = _supervised_baseline()

    payload = {
        "unsupervised": results,
        "supervised_binary_day": supervised,
        "config": {
            "iforest_train_sample": min(IFOREST_TRAIN_SAMPLE, len(X_train)),
            "lof_train_sample": min(LOF_TRAIN_SAMPLE, len(X_train)),
            "test_n": int(len(y_test)),
            "test_attack_rate": float(y_test.mean()),
            "fpr_targets": FPR_TARGETS,
        },
    }
    (OUT / "anomaly_metrics.json").write_text(json.dumps(payload, indent=2))

    # Save a trained IsolationForest + scaler, in case someone wants to reuse
    models_out = MODELS_DIR / "anomaly"
    models_out.mkdir(parents=True, exist_ok=True)
    joblib.dump(iso, models_out / "isolation_forest.joblib")
    joblib.dump(scaler, models_out / "scaler.joblib")

    # Markdown
    lines = [
        "# Unsupervised Anomaly Detection vs Supervised Binary",
        "",
        "Isolation Forest and Local Outlier Factor are trained on **benign flows only** "
        "(Monday-Wednesday), without seeing any attack labels. They are then asked to "
        "flag Friday anomalies. This is the honest unsupervised baseline for IDS.",
        "",
        f"- Benign training pool: {len(X_train):,} flows "
        f"(subsample: IForest {min(IFOREST_TRAIN_SAMPLE, len(X_train)):,}, "
        f"LOF {min(LOF_TRAIN_SAMPLE, len(X_train)):,})",
        f"- Test: Friday, n = {len(y_test):,}, attack rate = {y_test.mean():.4f}",
        "",
        "## Results",
        "",
        "| Model | Supervision | ROC-AUC | PR-AUC | Recall @ 0.1% FPR | Recall @ 1% FPR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        op = r["operating_points"]
        lines.append(
            f"| {name} | unsupervised | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} | "
            f"{op['fpr_0.001']['recall']:.4f} | {op['fpr_0.01']['recall']:.4f} |"
        )
    if supervised:
        for name, r in supervised.items():
            roc = r.get("roc_auc")
            pr = r.get("pr_auc")
            lines.append(
                f"| {name} | supervised | "
                f"{roc:.4f} | {pr:.4f} | — | — |"
            )

    lines += [
        "",
        "## Interpretation",
        "",
        "- Unsupervised detectors need **zero** attack labels. The price paid is",
        "  usually lower PR-AUC and weaker recall at tight FPR budgets.",
        "- If the unsupervised ROC-AUC is close to the supervised number, it means",
        "  the attack traffic simply looks different from benign in feature space,",
        "  and labels are not strictly necessary to catch it.",
        "- If the gap is large, labels are doing real work — the supervised model",
        "  is learning an attack-specific decision boundary that pure density or",
        "  isolation cannot recover.",
        "- In practice deployments combine both: unsupervised for novelty, supervised",
        "  for known families, MITRE mapping for triage.",
        "",
    ]
    (OUT / "anomaly_metrics.md").write_text("\n".join(lines))
    print(f"[anomaly] wrote {OUT / 'anomaly_metrics.md'}")

    # Figure: ROC-AUC bar chart, unsupervised vs supervised
    fig, ax = plt.subplots(figsize=(10, 5))
    unsup_names = list(results.keys())
    sup_names = list(supervised.keys())
    all_names = unsup_names + sup_names
    all_vals = [results[n]["roc_auc"] for n in unsup_names] + \
               [supervised[n]["roc_auc"] for n in sup_names]
    colors = ["#C44E52"] * len(unsup_names) + ["#4C72B0"] * len(sup_names)
    bars = ax.bar(all_names, all_vals, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ROC-AUC (binary: Attack vs Benign)")
    ax.set_title("Unsupervised vs Supervised IDS — binary day split (Friday test)")
    for b, v in zip(bars, all_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=9)
    # legend proxies
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#C44E52", label="Unsupervised (trained on benign only)"),
        Patch(facecolor="#4C72B0", label="Supervised (trained with attack labels)"),
    ], loc="lower right")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig_path = FIGURES_DIR / "anomaly_vs_supervised.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[anomaly] wrote {fig_path}")


if __name__ == "__main__":
    main()
