"""Evaluate UNSW-NB15 models: multi-class and binary.

Produces metrics.json with the same structure as eval.py / eval_binary.py
so results can be compared across datasets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from src.config import (
    UNSW_DATA_FILE,
    UNSW_MODELS_DIR,
    UNSW_METRICS_DIR,
    UNSW_NON_FEATURE,
)


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in UNSW_NON_FEATURE]


def _uses_internal_scaler(model):
    return hasattr(model, "steps") and "scaler" in [s[0] for s in model.steps]


def eval_multiclass(df: pd.DataFrame, models_dir: Path, out_dir: Path, feat: list[str]) -> None:
    """Evaluate multi-class models on UNSW-NB15 test set."""
    le = joblib.load(models_dir / "label_encoder.joblib")
    class_names = list(le.classes_)
    n_classes = len(class_names)

    sub = df.loc[df["split"] == "test"]
    X = sub[feat].copy()
    y = le.transform(sub["attack_cat"])

    results = {}

    for key in ["logreg", "random_forest", "xgboost", "lightgbm"]:
        path = models_dir / f"{key}.joblib"
        if not path.exists():
            print(f"[eval] skipping {key} (not found)")
            continue

        print(f"[eval] evaluating {key} ...")
        model = joblib.load(path)

        X_in = X  # all models use raw features (LogReg has internal scaler)
        pred = model.predict(X_in)
        proba = model.predict_proba(X_in) if hasattr(model, "predict_proba") else None

        pr, rc, f1, sup = precision_recall_fscore_support(
            y, pred, labels=np.arange(n_classes), zero_division=0)
        cm = confusion_matrix(y, pred, labels=np.arange(n_classes))

        has_support = sup > 0
        rec = {
            "macro_precision": float(np.mean(pr[has_support])) if has_support.any() else 0.0,
            "macro_recall":    float(np.mean(rc[has_support])) if has_support.any() else 0.0,
            "macro_f1":        float(np.mean(f1[has_support])) if has_support.any() else 0.0,
            "micro_f1":        float(precision_recall_fscore_support(
                                    y, pred, average="micro", zero_division=0)[2]),
            "per_class": {class_names[i]: {
                "precision": float(pr[i]), "recall": float(rc[i]),
                "f1": float(f1[i]), "support": int(sup[i])}
                for i in range(n_classes)},
            "confusion_matrix": cm.tolist(),
            "confusion_labels": class_names,
        }

        # ROC-AUC (OvR)
        if proba is not None and n_classes > 2:
            try:
                rec["roc_auc_ovr"] = float(
                    roc_auc_score(y, proba, multi_class="ovr", average="macro"))
            except Exception as e:
                print(f"[warn] ROC-AUC failed for {key}: {e}")
                rec["roc_auc_ovr"] = None

            # PR-AUC macro
            try:
                ap_scores = []
                for i in range(n_classes):
                    if (y == i).sum() > 0:
                        ap_scores.append(
                            average_precision_score((y == i).astype(int), proba[:, i]))
                rec["pr_auc_macro"] = float(np.mean(ap_scores)) if ap_scores else None
            except Exception:
                rec["pr_auc_macro"] = None

        results[key] = rec

        # Save per-model reports
        report = classification_report(
            y, pred, target_names=class_names, digits=4, zero_division=0)
        (out_dir / f"classification_report_{key}.txt").write_text(report, encoding="utf-8")
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
            out_dir / f"confusion_matrix_{key}.csv")

        # Feature importance
        base = model
        if hasattr(model, "steps"):
            for _, step in model.steps:
                if hasattr(step, "feature_importances_"):
                    base = step
                    break
        if hasattr(base, "feature_importances_"):
            pd.DataFrame({
                "feature": feat, "importance": base.feature_importances_,
            }).sort_values("importance", ascending=False).to_csv(
                out_dir / f"feature_importance_{key}.csv", index=False)

        print(f"[eval] {key}: macro_f1={rec['macro_f1']:.4f}")

    if not results:
        raise SystemExit("No model files found.")

    best_key = max(results, key=lambda k: results[k]["macro_f1"])
    summary = {
        k: {kk: vv for kk, vv in v.items()
            if kk not in ("per_class", "confusion_matrix", "confusion_labels")}
        for k, v in results.items()
    }
    summary["error_analysis"] = {"best_model": best_key}
    summary["dataset"] = "unsw-nb15"
    summary["task"] = "multiclass"
    summary["n_test"] = int(len(y))
    summary["class_names"] = class_names

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"summary": summary, "full": results}, f, indent=2)

    print(f"[eval] wrote {out_dir / 'metrics.json'}")
    print(f"[eval] best: {best_key} (macro F1={results[best_key]['macro_f1']:.4f})")


def eval_binary(df: pd.DataFrame, models_dir: Path, out_dir: Path, feat: list[str]) -> None:
    """Evaluate binary models on UNSW-NB15 test set."""
    sub = df.loc[df["split"] == "test"]
    X = sub[feat].copy()
    ys = sub["label"].values.astype(int)

    # Also load val for threshold tuning
    sub_val = df.loc[df["split"] == "val"]
    Xv = sub_val[feat].copy()
    yv = sub_val["label"].values.astype(int)

    print(f"[binary_eval] test={len(ys)} (attack={ys.sum()}) val={len(yv)} (attack={yv.sum()})")

    results = {}

    for key in ["logreg", "random_forest", "xgboost", "lightgbm"]:
        path = models_dir / f"{key}.joblib"
        if not path.exists():
            continue

        model = joblib.load(path)
        proba = model.predict_proba(X)[:, 1]
        proba_val = model.predict_proba(Xv)[:, 1]

        # Threshold tuning on val — Youden's J statistic (maximize TPR − FPR)
        fpr_val, tpr_val, thr_roc = roc_curve(yv, proba_val)
        j_scores = tpr_val[:-1] - fpr_val[:-1]
        best_j_idx = int(np.argmax(j_scores))
        best_thr = float(thr_roc[best_j_idx])

        pred = (proba >= best_thr).astype(int)

        roc_auc = float(roc_auc_score(ys, proba))
        pr_auc = float(average_precision_score(ys, proba))
        p, r, f1, _ = precision_recall_fscore_support(
            ys, pred, average="binary", zero_division=0)
        n_neg = (ys == 0).sum()
        fp = ((pred == 1) & (ys == 0)).sum()
        fpr = float(fp / n_neg) if n_neg > 0 else 0.0

        # Save ROC/PR curves
        fpr_arr, tpr_arr, _ = roc_curve(ys, proba)
        pd.DataFrame({"fpr": fpr_arr, "tpr": tpr_arr}).to_csv(
            out_dir / f"binary_roc_curve_{key}.csv", index=False)

        prec_arr, rec_arr, _ = precision_recall_curve(ys, proba)
        pd.DataFrame({"precision": prec_arr, "recall": rec_arr}).to_csv(
            out_dir / f"binary_pr_curve_{key}.csv", index=False)

        cm = confusion_matrix(ys, pred)
        pd.DataFrame(cm, index=["Normal", "Attack"], columns=["Pred_Normal", "Pred_Attack"]
                     ).to_csv(out_dir / f"binary_confusion_matrix_{key}.csv")

        report = classification_report(
            ys, pred, target_names=["Normal", "Attack"], digits=4, zero_division=0)
        (out_dir / f"binary_classification_report_{key}.txt").write_text(report, encoding="utf-8")

        results[key] = {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "best_threshold": round(best_thr, 6),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f1), 4),
            "fpr": round(fpr, 4),
        }

        print(f"[{key}] ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}  "
              f"thr={best_thr:.2f}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}  FPR={fpr:.4f}")

    with open(out_dir / "binary_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[binary_eval] wrote {out_dir / 'binary_metrics.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["multiclass", "binary"], default="multiclass")
    ap.add_argument("--models-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    if args.models_dir is None:
        args.models_dir = UNSW_MODELS_DIR / args.task
    if args.out_dir is None:
        args.out_dir = UNSW_METRICS_DIR / args.task

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not UNSW_DATA_FILE.exists():
        raise SystemExit(f"Missing {UNSW_DATA_FILE}. Run: python -m src.prepare_unsw")

    df = pd.read_parquet(UNSW_DATA_FILE)
    feat = _feature_cols(df)

    if args.task == "multiclass":
        eval_multiclass(df, args.models_dir, args.out_dir, feat)
    else:
        eval_binary(df, args.models_dir, args.out_dir, feat)


if __name__ == "__main__":
    main()
