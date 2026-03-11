"""Binary IDS evaluation: benign vs attack.

Trains a fresh binary classifier (and optionally re-uses the multi-class
model's probabilities) on the same data and split.

Outputs (in --out-dir):
  binary_metrics.json          — AUC, PR-AUC, Brier, threshold table
  binary_classification_report.txt
  binary_confusion_matrix.csv
  binary_roc_curve.csv         — FPR / TPR / threshold (for plotting)
  binary_pr_curve.csv          — precision / recall / threshold
  binary_threshold_report.txt  — table: threshold / precision / recall / F1 / FPR

Usage:
  python -m src.eval_binary --split strat --out-dir reports/metrics_strat_binary
  python -m src.eval_binary --split day   --out-dir reports/metrics_day_binary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.config import (
    DATA_FILE,
    MODELS_DIR,
    NON_FEATURE,
    RNG,
    SPLIT_COL_DAY,
    SPLIT_COL_STRAT,
)


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE]


def _load_binary_splits(df: pd.DataFrame, split_col: str):
    feat = _feature_cols(df)

    def part(s: str):
        sub = df.loc[df[split_col] == s]
        y = (sub["attack_type"].astype(str) != "Benign").astype(int)
        X = sub[feat].copy()
        return X, y.values

    Xt, yt = part("train")
    Xv, yv = part("val")
    Xs, ys = part("test")
    return Xt, yt, Xv, yv, Xs, ys, feat


def _threshold_table(y_true, y_prob, thresholds=None):
    """Return DataFrame with threshold / precision / recall / F1 / FPR rows."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)
    rows = []
    n_neg = (y_true == 0).sum()
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        fp = ((pred == 1) & (y_true == 0)).sum()
        fpr = fp / n_neg if n_neg > 0 else 0.0
        rows.append({"threshold": round(t, 2), "precision": round(p, 4),
                     "recall": round(r, 4), "f1": round(f1, 4),
                     "fpr": round(fpr, 4)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["day", "strat"], default="strat")
    ap.add_argument("--out-dir", type=Path, default=Path("reports/metrics_binary"))
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR,
                    help="Where to save/load binary models")
    args = ap.parse_args()

    split_col = SPLIT_COL_DAY if args.split == "day" else SPLIT_COL_STRAT
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        raise SystemExit(f"Missing {DATA_FILE}. Run: python -m src.prepare_data")

    print(f"[binary_eval] loading data ...")
    df = pd.read_parquet(DATA_FILE)
    Xt, yt, Xv, yv, Xs, ys, feat = _load_binary_splits(df, split_col)

    print(f"[binary_eval] train={len(yt)} (attack={yt.sum()}) | "
          f"val={len(yv)} (attack={yv.sum()}) | "
          f"test={len(ys)} (attack={ys.sum()})")

    scaler = StandardScaler()
    Xt_s = scaler.fit_transform(Xt)
    Xv_s = scaler.transform(Xv)
    Xs_s = scaler.transform(Xs)

    # --- Train binary models ---
    models = {}

    # Logistic Regression
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="saga", max_iter=2000, tol=1e-3,
            class_weight="balanced", random_state=RNG,
        )),
    ])
    lr.fit(Xt, yt)
    models["logreg"] = (lr, Xt, Xv, Xs)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=24, max_features="sqrt",
        class_weight="balanced_subsample", random_state=RNG, n_jobs=-1,
    )
    rf.fit(Xt_s, yt)
    models["random_forest"] = (rf, Xt_s, Xv_s, Xs_s)

    # XGBoost (calibrated on val set)
    xgb_raw = xgb.XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        eval_metric="logloss", verbosity=0,
        scale_pos_weight=float((yt == 0).sum()) / float((yt == 1).sum()),
        random_state=RNG, n_jobs=-1,
    )
    xgb_raw.fit(Xt_s, yt)

    # Calibrate using val set
    xgb_cal = CalibratedClassifierCV(xgb_raw, method="isotonic", cv="prefit")
    xgb_cal.fit(Xv_s, yv)
    models["xgboost"] = (xgb_raw, Xt_s, Xv_s, Xs_s)
    models["xgboost_calibrated"] = (xgb_cal, Xt_s, Xv_s, Xs_s)

    # Save binary scaler and models
    bin_models_dir = Path(args.models_dir) / f"binary_{args.split}"
    bin_models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, bin_models_dir / "scaler.joblib")
    for name, (m, *_) in models.items():
        joblib.dump(m, bin_models_dir / f"{name}.joblib")

    # --- Evaluate ---
    all_results = {}

    for key, (model, X_tr, X_val, X_test) in models.items():
        proba = model.predict_proba(X_test)[:, 1]

        # Optimal threshold chosen on val, never test
        proba_val = model.predict_proba(X_val)[:, 1]
        thr_table_val = _threshold_table(yv, proba_val)
        best_thr = float(thr_table_val.loc[thr_table_val["f1"].idxmax(), "threshold"])

        pred = (proba >= best_thr).astype(int)

        # Core metrics
        roc_auc = float(roc_auc_score(ys, proba))
        pr_auc = float(average_precision_score(ys, proba))
        brier = float(brier_score_loss(ys, proba))
        p, r, f1, _ = precision_recall_fscore_support(
            ys, pred, average="binary", zero_division=0
        )
        n_neg = (ys == 0).sum()
        fp = ((pred == 1) & (ys == 0)).sum()
        fpr_at_thr = float(fp / n_neg) if n_neg > 0 else 0.0

        # ROC curve data
        fpr_arr, tpr_arr, roc_thr = roc_curve(ys, proba)
        pd.DataFrame({
            "fpr": fpr_arr, "tpr": tpr_arr, "threshold": roc_thr
        }).to_csv(out / f"binary_roc_curve_{key}.csv", index=False)

        # PR curve data
        prec_arr, rec_arr, pr_thr = precision_recall_curve(ys, proba)
        pd.DataFrame({
            "precision": prec_arr, "recall": rec_arr,
            "threshold": np.append(pr_thr, np.nan)
        }).to_csv(out / f"binary_pr_curve_{key}.csv", index=False)

        # Threshold table on test
        thr_table_test = _threshold_table(ys, proba)
        thr_table_test.to_csv(out / f"binary_threshold_table_{key}.csv", index=False)

        # Classification report
        report_txt = classification_report(
            ys, pred, target_names=["Benign", "Attack"],
            digits=4, zero_division=0
        )
        (out / f"binary_classification_report_{key}.txt").write_text(report_txt)

        # Confusion matrix
        cm = confusion_matrix(ys, pred)
        pd.DataFrame(
            cm, index=["Benign", "Attack"], columns=["Pred_Benign", "Pred_Attack"]
        ).to_csv(out / f"binary_confusion_matrix_{key}.csv")

        # Calibration curve
        if key in ("xgboost", "xgboost_calibrated", "random_forest"):
            try:
                frac_pos, mean_pred = calibration_curve(ys, proba, n_bins=10)
                pd.DataFrame({
                    "mean_predicted_prob": mean_pred,
                    "fraction_positives": frac_pos
                }).to_csv(out / f"binary_calibration_curve_{key}.csv", index=False)
            except Exception:
                pass

        all_results[key] = {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "best_threshold_from_val": round(best_thr, 2),
            "precision_at_threshold": round(float(p), 4),
            "recall_at_threshold": round(float(r), 4),
            "f1_at_threshold": round(float(f1), 4),
            "fpr_at_threshold": round(fpr_at_thr, 4),
            "split": args.split,
        }

        print(f"[{key}] ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}  "
              f"Brier={brier:.4f}  thr={best_thr:.2f}  "
              f"P={p:.4f}  R={r:.4f}  F1={f1:.4f}  FPR={fpr_at_thr:.4f}")

    # Calibration improvement summary
    if "xgboost" in all_results and "xgboost_calibrated" in all_results:
        before = all_results["xgboost"]["brier_score"]
        after = all_results["xgboost_calibrated"]["brier_score"]
        all_results["calibration_improvement"] = {
            "xgboost_brier_before": before,
            "xgboost_brier_after": after,
            "delta": round(after - before, 4),
            "note": "Negative delta means calibration improved (lower Brier = better).",
        }
        print(f"[calibration] Brier before={before:.4f}  after={after:.4f}  "
              f"delta={after-before:+.4f}")

    with open(out / "binary_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"[binary_eval] wrote {out / 'binary_metrics.json'}")


if __name__ == "__main__":
    main()