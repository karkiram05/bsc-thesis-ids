"""Evaluate multi-class models (LogReg, RF, XGBoost)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import DATA_FILE, MODELS_DIR, REPORTS_DIR, SPLIT_COL_DAY, SPLIT_COL_STRAT


def _ensure_int_labels(pred, le):
    arr = np.asarray(pred)
    if arr.dtype.kind in {"U", "S", "O"}:
        return le.transform(arr.astype(str)).astype(int)
    return arr.astype(int)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["day", "strat"], default="day")
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    ap.add_argument("--out-dir", type=Path, default=Path(REPORTS_DIR) / "metrics")
    ap.add_argument("--out-suffix", type=str, default="")
    args = ap.parse_args()

    split_col = SPLIT_COL_DAY if args.split == "day" else SPLIT_COL_STRAT

    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_FILE)
    test = df[df[split_col] == "test"].copy()

    feat = json.loads((models_dir / "feature_names.json").read_text(encoding="utf-8"))
    X = test[feat]

    le = joblib.load(models_dir / "label_encoder.joblib")
    y_true_str = test["attack_type"].astype(str).to_numpy()
    y_true = le.transform(y_true_str).astype(int)

    scaler = joblib.load(models_dir / "scaler.joblib")
    Xs = scaler.transform(X)

    models = {
        "logreg": joblib.load(models_dir / "logreg.joblib"),
        "random_forest": joblib.load(models_dir / "random_forest.joblib"),
        "xgboost": joblib.load(models_dir / "xgboost.joblib") if (models_dir / "xgboost.joblib").exists() else None,
    }

    # For XGB remap
    xgb_classes = None
    xgb_classes_path = models_dir / "xgb_classes.json"
    if xgb_classes_path.exists():
        xgb_classes = np.array(json.loads(xgb_classes_path.read_text(encoding="utf-8")), dtype=int)

    results = {
        "task": "multiclass",
        "split_col": split_col,
        "n_test": int(len(test)),
        "models": {},
    }

    best_name = None
    best_f1 = -1.0

    for name, model in models.items():
        if model is None:
            continue

        if name == "logreg":
            y_pred = _ensure_int_labels(model.predict(X), le)
        elif name == "random_forest":
            y_pred = _ensure_int_labels(model.predict(Xs), le)
        elif name == "xgboost":
            if xgb_classes is None:
                raise SystemExit("Missing xgb_classes.json for xgboost evaluation")
            y_pred_trainspace = model.predict(Xs).astype(int)
            # map back to global label space
            y_pred = xgb_classes[y_pred_trainspace]
        else:
            raise SystemExit(f"Unknown model: {name}")

        pr, rc, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=np.arange(len(le.classes_)), zero_division=0
        )
        macro_f1 = float(np.mean(f1))
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(le.classes_)))

        results["models"][name] = {
            "macro_f1": macro_f1,
            "per_class": {
                le.inverse_transform([i])[0]: {
                    "precision": float(pr[i]),
                    "recall": float(rc[i]),
                    "f1": float(f1[i]),
                    "support": int(sup[i]),
                }
                for i in range(len(le.classes_))
            },
            "confusion_matrix": cm.tolist(),
        }

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_name = name

    results["best_model"] = best_name
    results["best_macro_f1"] = float(best_f1)

    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    out_path = out_dir / f"metrics{suffix}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"[eval] wrote {out_path}")
    print(f"[eval] best model: {best_name} (macro F1={best_f1:.4f})")


if __name__ == "__main__":
    main()