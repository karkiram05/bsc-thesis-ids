"""
Binary evaluation (Benign vs Attack) with automatic threshold tuning on VAL per model,
then evaluation on TEST.

This prevents your earlier mistake: using one model's threshold for all models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

from src.config import DATA_FILE, REPORTS_DIR, SPLIT_COL_DAY, SPLIT_COL_STRAT


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _model_uses_internal_scaler(model) -> bool:
    return hasattr(model, "steps") and any(n == "scaler" for n, _ in model.steps)


def _p_attack(model, X_raw, X_scaled) -> np.ndarray:
    Xin = X_raw if _model_uses_internal_scaler(model) else X_scaled
    proba = model.predict_proba(Xin)
    return np.asarray(proba)[:, 1]


def _conf_metrics(y_true: np.ndarray, p: np.ndarray, thr: float) -> dict:
    y_pred = (p >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    prec_attack = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_attack = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_attack = (2 * prec_attack * rec_attack / (prec_attack + rec_attack)) if (prec_attack + rec_attack) > 0 else 0.0

    prec_benign = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_benign = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_benign = (2 * prec_benign * rec_benign / (prec_benign + rec_benign)) if (prec_benign + rec_benign) > 0 else 0.0

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "threshold": float(thr),
        "precision_attack": float(prec_attack),
        "recall_attack": float(rec_attack),
        "f1_attack": float(f1_attack),
        "precision_benign": float(prec_benign),
        "recall_benign": float(rec_benign),
        "f1_benign": float(f1_benign),
        "macro_f1": float((f1_attack + f1_benign) / 2.0),
        "accuracy": float((tn + tp) / cm.sum()),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "confusion_matrix": cm.tolist(),
        "confusion_labels": ["Benign", "Attack"],
    }


def _tune_threshold(y_val: np.ndarray, p_val: np.ndarray, max_fpr: float | None) -> dict:
    # scan thresholds
    ths = np.linspace(0.01, 0.99, 99)
    best = None
    for t in ths:
        m = _conf_metrics(y_val, p_val, float(t))
        if max_fpr is not None and m["fpr"] > max_fpr:
            continue
        if best is None or m["f1_attack"] > best["f1_attack"]:
            best = m

    # fallback if constraint too strict
    if best is None:
        best = _conf_metrics(y_val, p_val, 0.5)
        best["note"] = "No threshold met max_fpr constraint; used 0.5 fallback."

    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["day", "strat"], default="day")
    ap.add_argument("--models-dir", type=Path, default=Path("models/binary"))
    ap.add_argument("--out-dir", type=Path, default=Path(REPORTS_DIR) / "metrics" / "binary")
    ap.add_argument("--out-suffix", type=str, default="day_autotune")
    ap.add_argument("--max-fpr", type=float, default=None, help="Example: 0.01 for <=1% FPR tuning on VAL")
    args = ap.parse_args()

    split_col = SPLIT_COL_DAY if args.split == "day" else SPLIT_COL_STRAT
    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    _safe_mkdir(out_dir)

    df = pd.read_parquet(DATA_FILE)

    feat = json.loads((models_dir / "feature_names.json").read_text(encoding="utf-8"))
    scaler = joblib.load(models_dir / "scaler.joblib")

    def part(s: str):
        sub = df[df[split_col] == s].copy()
        X = sub[feat]
        y = (sub["attack_type"].astype(str) != "Benign").astype(int).to_numpy()
        Xs = scaler.transform(X)
        return X, Xs, y

    Xv, Xv_s, yv = part("val")
    Xt, Xt_s, yt = part("test")

    models = {
        "logreg": joblib.load(models_dir / "logreg.joblib"),
        "random_forest": joblib.load(models_dir / "random_forest.joblib"),
        "xgboost": joblib.load(models_dir / "xgboost.joblib"),
    }

    results = {
        "task": "binary",
        "split_col": split_col,
        "n_val": int(len(yv)),
        "n_test": int(len(yt)),
        "tuning": {
            "max_fpr": args.max_fpr,
            "method": "scan 0.01..0.99 by 0.01, maximize F1_attack on VAL (optionally under FPR constraint)",
        },
        "models": {},
    }

    best_model = None
    best_macro = -1.0

    for name, model in models.items():
        p_val = _p_attack(model, Xv, Xv_s)
        tune = _tune_threshold(yv, p_val, args.max_fpr)
        thr = tune["threshold"]

        p_test = _p_attack(model, Xt, Xt_s)
        test_m = _conf_metrics(yt, p_test, thr)

        test_m["roc_auc"] = float(roc_auc_score(yt, p_test))
        test_m["pr_auc"] = float(average_precision_score(yt, p_test))
        test_m["tuned_on_val"] = tune

        results["models"][name] = test_m

        if test_m["macro_f1"] > best_macro:
            best_macro = test_m["macro_f1"]
            best_model = name

    results["best_model"] = best_model
    results["best_macro_f1"] = float(best_macro)

    out_path = out_dir / f"metrics_{args.out_suffix}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"[eval-binary] wrote {out_path}")
    print(f"[eval-binary] best model: {best_model} (macro F1={best_macro:.4f})")


if __name__ == "__main__":
    main()