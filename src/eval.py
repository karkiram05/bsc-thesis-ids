"""Evaluate models: precision, recall, F1, ROC AUC, per-class CM, error analysis."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.metrics import (precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, average_precision_score, classification_report,
    roc_curve, precision_recall_curve)
from src.config import DATA_FILE, MODELS_DIR, METRICS_DIR

# classes with fewer test rows than this get per-class PR curves
MINORITY_THRESHOLD = 500


def auc_from_arrays(fpr, tpr):
    """Trapezoid AUC from FPR/TPR arrays."""
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    if _trapz is None:
        raise RuntimeError("numpy has neither trapezoid nor trapz — upgrade numpy")
    return float(_trapz(tpr, fpr))


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

    sub   = df.loc[df[split_col] == "test"]
    y_raw = sub["attack_type"].astype(str)
    X     = sub[feat].copy()

    known_classes = list(le.classes_)
    known_to_idx = {c: i for i, c in enumerate(known_classes)}
    unseen_mask = ~y_raw.isin(known_classes)
    n_unseen = int(unseen_mask.sum())

    if n_unseen > 0:
        unseen_labels = sorted(y_raw[unseen_mask].unique().tolist())
        print(
            f"[warn] test split contains {n_unseen} rows with labels unseen in training: {unseen_labels}. "
            "These rows are evaluated as an explicit __unseen__ class with expected low recall."
        )
        unknown_idx = len(known_classes)
        y = np.array([known_to_idx.get(lbl, unknown_idx) for lbl in y_raw], dtype=np.int64)
        class_names = known_classes + ["__unseen__"]
    else:
        y = le.transform(y_raw)
        class_names = known_classes

    n_classes  = len(class_names)

    # Load label remaps for boosting models (day split: non-contiguous classes)
    remap_path      = models_dir / "xgboost_label_remap.json"
    xgb_train_classes = None
    if remap_path.exists():
        xgb_train_classes = json.loads(remap_path.read_text())["train_classes"]

    results = {}

    for key, path in [("logreg",        models_dir / "logreg.joblib"),
                      ("random_forest", models_dir / "random_forest.joblib"),
                      ("xgboost",       models_dir / "xgboost.joblib"),
                      ("lightgbm",      models_dir / "lightgbm.joblib")]:

        if not path.exists():
            print(f"[eval] skipping {key} (not found)")
            continue

        print(f"[eval] evaluating {key} ...")
        model    = joblib.load(path)

        # All 4 trained models skip the external scaler:
        # LogReg pipeline has its own scaler, trees are scale-invariant.
        X_in = X

        raw_pred = model.predict(X_in)
        raw_proba = model.predict_proba(X_in) if hasattr(model, "predict_proba") else None

        # XGBoost label remapping (day split has non-contiguous classes)
        remap_classes = None
        if key == "xgboost" and xgb_train_classes is not None:
            remap_classes = xgb_train_classes

        if remap_classes is not None:
            pred = np.array([remap_classes[p] for p in raw_pred.tolist()], dtype=np.int64)
            if raw_proba is not None:
                proba = np.zeros((len(raw_proba), n_classes), dtype=np.float64)
                for local_i, global_i in enumerate(remap_classes):
                    proba[:, global_i] = raw_proba[:, local_i]
            else:
                proba = None
        else:
            pred  = raw_pred
            proba = raw_proba

        # Pad probabilities if we have an unseen class column
        if proba is not None and proba.shape[1] < n_classes:
            padded = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
            padded[:, :proba.shape[1]] = proba
            proba = padded

        pr, rc, f1, sup = precision_recall_fscore_support(
            y, pred, labels=np.arange(n_classes), zero_division=0)
        cm = confusion_matrix(y, pred, labels=np.arange(n_classes))

        # Average only over classes with support
        has_support = sup > 0
        rec = {
            "macro_precision": float(np.mean(pr[has_support])) if has_support.any() else 0.0,
            "macro_recall":    float(np.mean(rc[has_support])) if has_support.any() else 0.0,
            "macro_f1":        float(np.mean(f1[has_support])) if has_support.any() else 0.0,
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

        # ── ROC AUC + ROC curve data ─────────────────────────────────────
        # include n_classes == 2 too: when unseen-class filtering leaves only
        # benign + one attack, we still want an AUC in the JSON.
        if proba is not None and n_classes >= 2:
            # ROC-AUC only on classes with support in test
            classes_with_support = [i for i in range(n_classes) if sup[i] > 0]

            try:
                if len(classes_with_support) == 2:
                    # 2 classes with support -> binary ROC-AUC
                    pos_cls = classes_with_support[1]
                    y_bin = (y == pos_cls).astype(int)
                    rec["roc_auc_ovr"] = float(roc_auc_score(y_bin, proba[:, pos_cls]))
                elif len(classes_with_support) > 2:
                    # Remap to contiguous 0..K-1 for sklearn
                    mask = np.isin(y, classes_with_support)
                    old_to_new = {old: new for new, old in enumerate(classes_with_support)}
                    y_remapped = np.array([old_to_new[v] for v in y[mask]], dtype=np.int64)
                    proba_filtered = proba[mask][:, classes_with_support]
                    rec["roc_auc_ovr"] = float(
                        roc_auc_score(y_remapped, proba_filtered,
                                      multi_class="ovr", average="macro"))
                else:
                    rec["roc_auc_ovr"] = None
            except Exception as e:
                print(f"[warn] ROC-AUC OvR failed for {key}: {e}")
                rec["roc_auc_ovr"] = None

            try:
                if len(classes_with_support) == 2:
                    pos_cls = classes_with_support[1]
                    y_bin = (y == pos_cls).astype(int)
                    rec["roc_auc_ovo"] = float(roc_auc_score(y_bin, proba[:, pos_cls]))
                elif len(classes_with_support) > 2:
                    mask = np.isin(y, classes_with_support)
                    old_to_new = {old: new for new, old in enumerate(classes_with_support)}
                    y_remapped = np.array([old_to_new[v] for v in y[mask]], dtype=np.int64)
                    proba_filtered = proba[mask][:, classes_with_support]
                    rec["roc_auc_ovo"] = float(
                        roc_auc_score(y_remapped, proba_filtered,
                                      multi_class="ovo", average="macro"))
                else:
                    rec["roc_auc_ovo"] = None
            except Exception as e:
                print(f"[warn] ROC-AUC OvO failed for {key}: {e}")
                rec["roc_auc_ovo"] = None

            try:
                ap_scores = []
                for i in range(n_classes):
                    if (y == i).sum() > 0:
                        ap_scores.append(
                            average_precision_score((y == i).astype(int), proba[:, i]))
                rec["pr_auc_macro"] = float(np.mean(ap_scores)) if ap_scores else None
            except Exception as e:
                print(f"[warn] PR-AUC macro failed for {key}: {e}")
                rec["pr_auc_macro"] = None

            # Per-class ROC curve data (downsampled for JSON size)
            roc_data = {}
            for i, cls in enumerate(class_names):
                # __unseen__ column is zero-padded (no training signal), ROC
                # would be a degenerate single point — skip to avoid misleading
                # figures downstream.
                if cls == "__unseen__":
                    continue
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
                except Exception as e:
                    print(f"[warn] ROC curve failed for class '{cls}': {e}")
            rec["roc_curve_data"] = roc_data

            # PR curves for minority classes only
            pr_data = {}
            for i, cls in enumerate(class_names):
                if cls == "__unseen__":
                    continue
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
                except Exception as e:
                    print(f"[warn] PR curve failed for class '{cls}': {e}")
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

        pred_labels = np.array([class_names[i] if 0 <= i < len(class_names) else "__unseen__" for i in pred], dtype=object)
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
    print("[eval] roc_curve_data + pr_curve_data saved in metrics.json (used by report.py)")


if __name__ == "__main__":
    main()
