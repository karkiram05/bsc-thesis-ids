"""Evaluate models: precision, recall, F1, ROC AUC, per-class CM, error analysis.

Changes vs original:
  - Saves roc_curve_data per class (FPR/TPR arrays + AUC) into metrics.json
  - Saves pr_curve_data for minority classes (precision/recall arrays + AP) into metrics.json
  - Both are used by report.py to generate roc_curves.png and pr_curves_minority.png
  - Everything else is identical to the original
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.metrics import (precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, average_precision_score, classification_report,
    roc_curve, precision_recall_curve)
from src.config import DATA_FILE, MODELS_DIR, NON_FEATURE, METRICS_DIR

# Minority class threshold: classes with fewer test rows than this get PR curves
MINORITY_THRESHOLD = 500


def auc_from_arrays(fpr, tpr):
    """Trapezoid AUC — avoids sklearn import duplication."""
    return float(np.trapz(tpr, fpr))


def _feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURE]


def _uses_internal_scaler(model):
    return hasattr(model, "steps") and "scaler" in [s[0] for s in model.steps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    ap.add_argument("--split", choices=["day", "strat"], default="day")
    ap.add_argument("--out-dir", type=Path, default=METRICS_DIR)
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_col = "split_day" if args.split == "day" else "split_strat"

    if not DATA_FILE.exists():
        raise SystemExit(f"Missing {DATA_FILE}.")
    if not (models_dir / "meta.json").exists():
        raise SystemExit(f"Missing meta.json in {models_dir}. Run train first.")

    df   = pd.read_parquet(DATA_FILE)
    meta = json.loads((models_dir / "meta.json").read_text())
    feat = meta["feature_names"]

    if meta.get("split_col") != split_col:
        raise SystemExit(f"Split mismatch: trained={meta.get('split_col')}, eval={split_col}")

    le     = joblib.load(models_dir / "label_encoder.joblib")
    scaler = joblib.load(models_dir / "scaler.joblib")

    sub   = df.loc[df[split_col] == "test"]
    y_raw = sub["attack_type"].astype(str)
    X     = sub[feat].copy()
    y     = le.transform(y_raw)
    X_s   = scaler.transform(X)

    n_classes  = len(le.classes_)
    class_names = list(le.classes_)

    remap_path      = models_dir / "xgboost_label_remap.json"
    xgb_train_classes = None
    if remap_path.exists():
        xgb_train_classes = json.loads(remap_path.read_text())["train_classes"]

    results = {}

    for key, path in [("logreg",        models_dir / "logreg.joblib"),
                      ("random_forest", models_dir / "random_forest.joblib"),
                      ("xgboost",       models_dir / "xgboost.joblib")]:

        if not path.exists():
            print(f"[eval] skipping {key} (not found)")
            continue

        print(f"[eval] evaluating {key} ...")
        model    = joblib.load(path)
        X_in     = X if _uses_internal_scaler(model) else X_s
        raw_pred = model.predict(X_in)
        raw_proba = model.predict_proba(X_in) if hasattr(model, "predict_proba") else None

        # XGBoost label remapping (day split: non-contiguous classes)
        if key == "xgboost" and xgb_train_classes is not None:
            pred = np.array([xgb_train_classes[p] for p in raw_pred.tolist()], dtype=np.int64)
            if raw_proba is not None:
                proba = np.zeros((len(raw_proba), n_classes), dtype=np.float64)
                for local_i, global_i in enumerate(xgb_train_classes):
                    proba[:, global_i] = raw_proba[:, local_i]
            else:
                proba = None
        else:
            pred  = raw_pred
            proba = raw_proba

        pr, rc, f1, sup = precision_recall_fscore_support(
            y, pred, labels=np.arange(n_classes), zero_division=0)
        cm = confusion_matrix(y, pred, labels=np.arange(n_classes))

        rec = {
            "macro_precision": float(np.mean(pr)),
            "macro_recall":    float(np.mean(rc)),
            "macro_f1":        float(np.mean(f1)),
            "micro_f1":        float(precision_recall_fscore_support(
                                    y, pred, average="micro", zero_division=0)[2]),
            "per_class":       {class_names[i]: {
                                    "precision": float(pr[i]),
                                    "recall":    float(rc[i]),
                                    "f1":        float(f1[i]),
                                    "support":   int(sup[i])}
                                for i in range(n_classes)},
            "confusion_matrix": cm.tolist(),
            "confusion_labels": class_names,
        }

        # ── ROC AUC + ROC curve data (new) ───────────────────────────────
        if proba is not None and n_classes > 2:
            try:
                rec["roc_auc_ovr"] = float(
                    roc_auc_score(y, proba, multi_class="ovr", average="macro"))
                rec["roc_auc_ovo"] = float(
                    roc_auc_score(y, proba, multi_class="ovo", average="macro"))
            except Exception:
                rec["roc_auc_ovr"] = rec["roc_auc_ovo"] = None

            try:
                rec["pr_auc_macro"] = float(np.mean([
                    average_precision_score((y == i).astype(int), proba[:, i])
                    for i in range(n_classes) if (y == i).sum() > 0
                ]))
            except Exception:
                rec["pr_auc_macro"] = None

            # Per-class ROC curve data — stored compactly (sample every 5th point)
            roc_data = {}
            for i, cls in enumerate(class_names):
                y_bin = (y == i).astype(int)
                if y_bin.sum() == 0:
                    continue
                try:
                    fpr_arr, tpr_arr, _ = roc_curve(y_bin, proba[:, i])
                    auc_val = float(auc_from_arrays(fpr_arr, tpr_arr))
                    # downsample to keep JSON small
                    step = max(1, len(fpr_arr) // 200)
                    roc_data[cls] = {
                        "fpr": fpr_arr[::step].tolist(),
                        "tpr": tpr_arr[::step].tolist(),
                        "auc": auc_val,
                    }
                except Exception:
                    pass
            rec["roc_curve_data"] = roc_data

            # PR curve data — minority classes only
            pr_data = {}
            for i, cls in enumerate(class_names):
                n_test = int((y == i).sum())
                if n_test == 0 or n_test >= MINORITY_THRESHOLD:
                    continue
                y_bin = (y == i).astype(int)
                try:
                    prec_arr, rec_arr, _ = precision_recall_curve(y_bin, proba[:, i])
                    ap_val = float(average_precision_score(y_bin, proba[:, i]))
                    step = max(1, len(prec_arr) // 200)
                    pr_data[cls] = {
                        "precision": prec_arr[::step].tolist(),
                        "recall":    rec_arr[::step].tolist(),
                        "ap":        ap_val,
                        "n_test":    n_test,
                    }
                except Exception:
                    pass
            rec["pr_curve_data"] = pr_data

        results[key] = rec

        # Per-model flat files (same as original)
        present = sorted(set(y.tolist()))
        report  = classification_report(
            y, pred, labels=present,
            target_names=[class_names[i] for i in present],
            digits=4, zero_division=0)
        (out_dir / f"classification_report_{key}.txt").write_text(report, encoding="utf-8")
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
            out_dir / f"confusion_matrix_{key}.csv")

        pred_labels = le.inverse_transform(pred)
        pd.DataFrame({
            "true_label": y_raw.values,
            "pred_label": pred_labels,
            "model":      key,
        }).to_csv(out_dir / f"predictions_{key}.csv", index=False)

        base = model
        if hasattr(model, "steps"):
            for _, step in model.steps:
                if hasattr(step, "feature_importances_"):
                    base = step
                    break
        if hasattr(base, "feature_importances_"):
            pd.DataFrame({
                "feature":    feat,
                "importance": base.feature_importances_,
            }).sort_values("importance", ascending=False).to_csv(
                out_dir / f"feature_importance_{key}.csv", index=False)

        print(f"[eval] {key}: macro_f1={rec['macro_f1']:.4f}")

    if not results:
        raise SystemExit("No model files found.")

    best_key = max(results, key=lambda k: results[k]["macro_f1"])
    cm_best  = np.array(results[best_key]["confusion_matrix"])
    err_pairs = sorted([
        (class_names[i], class_names[j], int(cm_best[i, j]))
        for i in range(cm_best.shape[0])
        for j in range(cm_best.shape[1])
        if i != j and cm_best[i, j] > 0
    ], key=lambda x: -x[2])

    summary = {
        k: {kk: vv for kk, vv in v.items()
            if kk not in ("per_class", "confusion_matrix", "confusion_labels",
                          "roc_curve_data", "pr_curve_data")}
        for k, v in results.items()
    }
    summary["error_analysis"] = {
        "best_model":     best_key,
        "top_confusions": [{"true": a, "pred": b, "count": c} for a, b, c in err_pairs[:20]],
    }
    summary["split_col"]  = split_col
    summary["n_test"]     = int(len(y))
    summary["class_names"] = class_names

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"summary": summary, "full": results}, f, indent=2)

    print(f"[eval] wrote {out_dir / 'metrics.json'}")
    print(f"[eval] best model: {best_key} (macro F1={results[best_key]['macro_f1']:.4f})")
    print(f"[eval] roc_curve_data + pr_curve_data saved in metrics.json (used by report.py)")


if __name__ == "__main__":
    main()
