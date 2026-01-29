"""Evaluate models: precision, recall, F1, ROC AUC, per-class CM, error analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    classification_report,
)

from src.config import DATA_FILE, MODELS_DIR, NON_FEATURE, METRICS_DIR, RNG


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE and c in df.columns]


def _uses_internal_scaler(model) -> bool:
    if hasattr(model, "steps"):
        names = [s[0] for s in model.steps]
        return "scaler" in names
    return False


def load_test_data(df: pd.DataFrame, split_col: str, feat: list[str]):
    sub = df.loc[df[split_col] == "test"]
    y_raw = sub["attack_type"].astype(str)
    X = sub[feat].copy()
    return X, y_raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    ap.add_argument("--split", choices=["day", "strat"], default="day")
    ap.add_argument("--out-dir", type=Path, default=METRICS_DIR)
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_col = "split_day" if args.split == "day" else "split_strat"
    if not DATA_FILE.exists():
        raise SystemExit(f"Missing {DATA_FILE}. Run: make data")
    if not (models_dir / "meta.json").exists():
        raise SystemExit(f"Missing {models_dir / 'meta.json'}. Run: make train")

    df = pd.read_parquet(DATA_FILE)
    meta = json.loads((models_dir / "meta.json").read_text())
    feat = meta["feature_names"]
    if meta.get("split_col") != split_col:
        raise SystemExit(
            f"Split mismatch: models trained with split_col={meta.get('split_col')}, "
            f"eval uses --split {args.split} ({split_col}). Re-run with matching --split."
        )
    le = joblib.load(models_dir / "label_encoder.joblib")
    scaler = joblib.load(models_dir / "scaler.joblib")

    X, y_raw = load_test_data(df, split_col, feat)
    y = le.transform(y_raw)
    X_s = scaler.transform(X)
    n_classes = len(le.classes_)
    class_names = list(le.classes_)

    model_files = [
        ("logreg", models_dir / "logreg.joblib"),
        ("random_forest", models_dir / "random_forest.joblib"),
        ("xgboost", models_dir / "xgboost.joblib"),
    ]
    results = {}

    for key, path in model_files:
        if not path.exists():
            continue
        model = joblib.load(path)
        use_raw = _uses_internal_scaler(model)
        X_in = X if use_raw else X_s

        # Predict (special handling for XGBoost when train split is missing some classes)
        if key == 'xgboost' and (models_dir / 'xgb_classes.json').exists():
            xgb_classes = np.array(json.loads((models_dir / 'xgb_classes.json').read_text()), dtype=int)
            pred_local = model.predict(X_in).astype(int)
            pred = xgb_classes[pred_local]
            if hasattr(model, 'predict_proba'):
                proba_local = model.predict_proba(X_in)
                proba = np.zeros((proba_local.shape[0], n_classes), dtype=float)
                proba[:, xgb_classes] = proba_local
            else:
                proba = None
        else:
            pred = model.predict(X_in)
            proba = model.predict_proba(X_in) if hasattr(model, 'predict_proba') else None

        pr, rc, f1, sup = precision_recall_fscore_support(
            y, pred, labels=np.arange(n_classes), zero_division=0
        )
        cm = confusion_matrix(y, pred, labels=np.arange(n_classes))

        rec = {
            "macro_precision": float(np.mean(pr)),
            "macro_recall": float(np.mean(rc)),
            "macro_f1": float(np.mean(f1)),
            "micro_precision": float(
                precision_recall_fscore_support(y, pred, average="micro", zero_division=0)[0]
            ),
            "micro_recall": float(
                precision_recall_fscore_support(y, pred, average="micro", zero_division=0)[1]
            ),
            "micro_f1": float(
                precision_recall_fscore_support(y, pred, average="micro", zero_division=0)[2]
            ),
            "per_class": {
                class_names[i]: {
                    "precision": float(pr[i]),
                    "recall": float(rc[i]),
                    "f1": float(f1[i]),
                    "support": int(sup[i]),
                }
                for i in range(n_classes)
            },
            "confusion_matrix": cm.tolist(),
            "confusion_labels": class_names,
        }

        if proba is not None and n_classes == 2:
            try:
                rec["roc_auc"] = float(roc_auc_score(y, proba[:, 1]))
                rec["pr_auc"] = float(average_precision_score(y, proba[:, 1]))
            except Exception:
                rec["roc_auc"] = None
                rec["pr_auc"] = None
        elif proba is not None and n_classes > 2:
            try:
                rec["roc_auc_ovr"] = float(
                    roc_auc_score(y, proba, multi_class="ovr", average="macro")
                )
                rec["roc_auc_ovo"] = float(
                    roc_auc_score(y, proba, multi_class="ovo", average="macro")
                )
            except Exception:
                rec["roc_auc_ovr"] = None
                rec["roc_auc_ovo"] = None
            try:
                ap_scores = []
                for i in range(n_classes):
                    yi = (y == i).astype(int)
                    if yi.sum() > 0:
                        ap_scores.append(average_precision_score(yi, proba[:, i]))
                rec["pr_auc_macro"] = float(np.mean(ap_scores)) if ap_scores else None
            except Exception:
                rec["pr_auc_macro"] = None
        else:
            rec["roc_auc_ovr"] = None
            rec["roc_auc_ovo"] = None
            rec["pr_auc_macro"] = None

        results[key] = rec

        # Classification report text
        report = classification_report(
            y, pred, target_names=class_names, digits=4, zero_division=0
        )
        (out_dir / f"classification_report_{key}.txt").write_text(report, encoding="utf-8")

        # Confusion matrix CSV
        cm_df = pd.DataFrame(
            cm, index=class_names, columns=class_names
        )
        cm_df.to_csv(out_dir / f"confusion_matrix_{key}.csv")

        # Test predictions for report / MITRE alerts
        pred_labels = le.inverse_transform(pred)
        pred_df = pd.DataFrame({
            "true_label": y_raw.values,
            "pred_label": pred_labels,
            "model": key,
        })
        pred_df.to_csv(out_dir / f"predictions_{key}.csv", index=False)

        # Feature importance (tree-based models)
        if hasattr(model, "feature_importances_"):
            base = model
            if hasattr(model, "steps"):
                for _, s in model.steps:
                    if hasattr(s, "feature_importances_"):
                        base = s
                        break
            imp = base.feature_importances_
            imp_df = pd.DataFrame({"feature": feat, "importance": imp}).sort_values(
                "importance", ascending=False
            )
            imp_df.to_csv(out_dir / f"feature_importance_{key}.csv", index=False)

    # Error analysis: worst confusions (off-diagonal) for best model
    best_key = max(results, key=lambda k: results[k]["macro_f1"])
    cm = np.array(results[best_key]["confusion_matrix"])
    err_pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                err_pairs.append((class_names[i], class_names[j], int(cm[i, j])))
    err_pairs.sort(key=lambda x: -x[2])
    error_analysis = {
        "best_model": best_key,
        "top_confusions": [{"true": a, "pred": b, "count": c} for a, b, c in err_pairs[:20]],
    }

    # Metrics summary
    summary = {k: {kk: vv for kk, vv in v.items() if kk not in ("per_class", "confusion_matrix", "confusion_labels")} for k, v in results.items()}
    summary["error_analysis"] = error_analysis
    summary["split_col"] = split_col
    summary["n_test"] = int(len(y))
    summary["class_names"] = class_names

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"summary": summary, "full": results}, f, indent=2)

    print(f"[eval] wrote {out_dir / 'metrics.json'}")
    print(f"[eval] best model: {best_key} (macro F1={results[best_key]['macro_f1']:.4f})")


if __name__ == "__main__":
    main()