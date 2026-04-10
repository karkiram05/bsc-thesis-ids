"""Binary IDS evaluation: benign vs attack.

Evaluation-only script: loads saved binary models and evaluates on test split.
Thresholds are selected on validation split and then applied once on test.

Usage:
  python -m src.train_binary --split strat --out-dir models/binary_strat
  python -m src.eval_binary  --split strat --models-dir models --out-dir reports/metrics_strat_binary
  python -m src.eval_binary  --split strat --models-dir models/binary_strat --out-dir reports/metrics_strat_binary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
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

from src.config import (
    DATA_FILE,
    MODELS_DIR,
    NON_FEATURE,
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


def _uses_internal_scaler(model):
    return hasattr(model, "steps") and "scaler" in [s[0] for s in model.steps]


def _resolve_models_dir(models_dir: Path, split: str) -> Path:
    """Accept either models/ or a direct binary_<split>/ directory."""
    direct_dir = Path(models_dir)
    nested_dir = direct_dir / f"binary_{split}"

    direct_has_meta = (direct_dir / "meta.json").exists()
    nested_has_meta = (nested_dir / "meta.json").exists()

    if direct_has_meta:
        return direct_dir
    if nested_has_meta:
        return nested_dir

    raise SystemExit(
        f"Could not find binary model artifacts in either {direct_dir} or {nested_dir}. "
        f"Run train first: python -m src.train_binary --split {split} --out-dir {nested_dir}"
    )


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

    print("[binary_eval] loading data ...")
    df = pd.read_parquet(DATA_FILE)
    Xt, yt, Xv, yv, Xs, ys, feat = _load_binary_splits(df, split_col)

    print(f"[binary_eval] train={len(yt)} (attack={yt.sum()}) | "
          f"val={len(yv)} (attack={yv.sum()}) | "
          f"test={len(ys)} (attack={ys.sum()})")

    bin_models_dir = _resolve_models_dir(Path(args.models_dir), args.split)
    scaler_path = bin_models_dir / "scaler.joblib"
    if not scaler_path.exists():
        raise SystemExit(f"Missing {scaler_path}. Re-run src.train_binary.")
    scaler = joblib.load(scaler_path)

    # --- Load trained binary models (evaluation only) ---
    models: dict[str, tuple[object, pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray]] = {}
    for key in ["logreg", "random_forest", "xgboost", "xgboost_calibrated"]:
        p = bin_models_dir / f"{key}.joblib"
        if not p.exists():
            continue
        model = joblib.load(p)
        # FIX: LogReg pipeline has internal scaler -> feed raw features.
        # Tree models (RF, XGB) trained on raw features -> feed raw features.
        # All models now get raw features.
        # All models receive raw features: LogReg has internal pipeline scaler,
        # tree models (RF, XGB) are scale-invariant.
        models[key] = (model, Xv, Xs)

    if not models:
        raise SystemExit(
            f"No binary model files found in {bin_models_dir}. "
            "Run src.train_binary before src.eval_binary."
        )

    # --- Evaluate ---
    all_results = {}

    for key, (model, X_val, X_test) in models.items():
        proba = model.predict_proba(X_test)[:, 1]

        # Optimal threshold chosen on val, never test
        proba_val = model.predict_proba(X_val)[:, 1]
        thr_table_val = _threshold_table(yv, proba_val)
        best_thr = float(thr_table_val.loc[thr_table_val["f1"].idxmax(), "threshold"])

        # FIX: Also find FPR-constrained threshold (FPR <= 1%)
        fpr_constrained_rows = thr_table_val[thr_table_val["fpr"] <= 0.01]
        if len(fpr_constrained_rows) > 0:
            fpr_thr = float(fpr_constrained_rows.loc[fpr_constrained_rows["f1"].idxmax(), "threshold"])
        else:
            fpr_thr = best_thr  # fallback

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
            except Exception as e:
                print(f"[warn] Calibration curve failed for model '{key}': {e}")

        all_results[key] = {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "best_threshold_from_val": round(best_thr, 2),
            "fpr_constrained_threshold": round(fpr_thr, 2),
            "precision_at_threshold": round(float(p), 4),
            "recall_at_threshold": round(float(r), 4),
            "f1_at_threshold": round(float(f1), 4),
            "fpr_at_threshold": round(fpr_at_thr, 4),
            "split": args.split,
        }

        print(f"[{key}] ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}  "
              f"Brier={brier:.4f}  thr={best_thr:.2f}  "
              f"P={p:.4f}  R={r:.4f}  F1={f1:.4f}  FPR={fpr_at_thr:.4f}")

    with open(out / "binary_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"[binary_eval] wrote {out / 'binary_metrics.json'}")


if __name__ == "__main__":
    main()
