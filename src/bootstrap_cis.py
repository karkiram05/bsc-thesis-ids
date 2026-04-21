from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)

from src.config import DATA_FILE, NON_FEATURE, MODELS_DIR, REPORTS_DIR

warnings.filterwarnings("ignore")

OUT = REPORTS_DIR / "bootstrap"

B = 1000
SEED = 42
CI_LOW, CI_HIGH = 2.5, 97.5


def _bootstrap_metric(metric_fn, y_true, y_pred_or_score, rng, b: int = B) -> tuple[float, float, float]:
    """Point value plus 95% CI via resampling."""
    n = len(y_true)
    samples = np.empty(b, dtype=float)
    for i in range(b):
        idx = rng.integers(0, n, n)
        try:
            samples[i] = metric_fn(y_true[idx], y_pred_or_score[idx])
        except ValueError:
            samples[i] = np.nan
    point = metric_fn(y_true, y_pred_or_score)
    lo = float(np.nanpercentile(samples, CI_LOW))
    hi = float(np.nanpercentile(samples, CI_HIGH))
    return float(point), lo, hi


def _load_binary_test() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(DATA_FILE)
    test = df[df["split_day"] == "test"].copy()
    feat_cols = [c for c in test.columns if c not in NON_FEATURE]
    X = test[feat_cols]
    y = test["is_attack"].to_numpy().astype(int)
    return X, y


def _load_threshold(model_name: str) -> float:
    # use the same Youden-J threshold as the main binary eval
    path = REPORTS_DIR / "metrics_day_binary" / "binary_metrics.json"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run: make binary-day first.")
    metrics = json.loads(path.read_text())
    if model_name not in metrics:
        raise SystemExit(f"Model '{model_name}' not found in {path}. Available: {list(metrics)}")
    return float(metrics[model_name]["best_threshold_from_val"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("[bootstrap] loading data ...")
    X_test, y_test = _load_binary_test()
    print(f"[bootstrap] test shape={X_test.shape}  attack rate={y_test.mean():.4f}")

    rng = np.random.default_rng(SEED)
    model_dir = MODELS_DIR / "binary_day"
    models = ["logreg", "random_forest", "xgboost", "lightgbm"]

    results: dict[str, dict] = {}
    for name in models:
        path = model_dir / f"{name}.joblib"
        if not path.exists():
            print(f"[bootstrap] skip {name}")
            continue
        print(f"[bootstrap] {name} ...")
        model = joblib.load(path)
        proba = model.predict_proba(X_test)[:, 1]
        thr = _load_threshold(name)
        pred = (proba >= thr).astype(int)

        rf = {}
        rf["precision"] = _bootstrap_metric(
            lambda yt, yp: precision_score(yt, yp, zero_division=0), y_test, pred, rng)
        rf["recall"] = _bootstrap_metric(
            lambda yt, yp: recall_score(yt, yp, zero_division=0), y_test, pred, rng)
        rf["f1"] = _bootstrap_metric(
            lambda yt, yp: f1_score(yt, yp, zero_division=0), y_test, pred, rng)
        rf["roc_auc"] = _bootstrap_metric(roc_auc_score, y_test, proba, rng)
        rf["pr_auc"] = _bootstrap_metric(average_precision_score, y_test, proba, rng)
        rf["threshold"] = thr

        results[name] = rf
        print(f"  F1={rf['f1'][0]:.4f} [{rf['f1'][1]:.4f}, {rf['f1'][2]:.4f}]  "
              f"ROC={rf['roc_auc'][0]:.4f} [{rf['roc_auc'][1]:.4f}, {rf['roc_auc'][2]:.4f}]")

    (OUT / "binary_day_cis.json").write_text(json.dumps(results, indent=2))

    lines = [
        "# Bootstrap 95% Confidence Intervals — Binary Day Split",
        "",
        f"- Resamples: **B = {B}**",
        f"- Seed: **{SEED}**",
        "- Method: non-parametric percentile bootstrap (2.5–97.5)",
        f"- Test set: Friday, n = {len(y_test):,} flows, attack rate = {y_test.mean():.4f}",
        "",
        "| Model | F1 | Precision | Recall | ROC-AUC | PR-AUC |",
        "|---|---|---|---|---|---|",
    ]

    def _fmt(tup):
        return f"{tup[0]:.4f} [{tup[1]:.4f}, {tup[2]:.4f}]"

    for name, rf in results.items():
        lines.append(
            f"| {name} | {_fmt(rf['f1'])} | {_fmt(rf['precision'])} | "
            f"{_fmt(rf['recall'])} | {_fmt(rf['roc_auc'])} | {_fmt(rf['pr_auc'])} |"
        )
    lines += [
        "",
        "Interpretation: the CIs are narrow because the test set is large "
        f"(n={len(y_test):,}). The informative numbers are differences between "
        "models — overlapping CIs on precision/recall confirm the qualitative ranking.",
        "",
    ]
    (OUT / "binary_day_cis.md").write_text("\n".join(lines))
    print(f"[bootstrap] wrote {OUT}")


if __name__ == "__main__":
    main()
