from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, f1_score

from src.config import DATA_FILE, MODELS_DIR, REPORTS_DIR, feature_cols

warnings.filterwarnings("ignore")

OUT = REPORTS_DIR / "calibration"
FPR_TARGETS = [0.0001, 0.001, 0.01]
ECE_BINS = 15


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                                n_bins: int = ECE_BINS) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if not mask.any():
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        weight = mask.sum() / n
        ece += weight * abs(bin_conf - bin_acc)
    return float(ece)


def recall_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    mask = fpr <= target_fpr
    if not mask.any():
        return 0.0, 0.0, 1.0
    idx = np.where(mask)[0].max()
    return float(tpr[idx]), float(fpr[idx]), float(thr[idx])


def fit_platt(y_val: np.ndarray, p_val: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(solver="lbfgs")
    clf.fit(p_val.reshape(-1, 1), y_val)
    return clf


def fit_isotonic(y_val: np.ndarray, p_val: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val, y_val)
    return iso


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("[calib] loading data ...")
    df = pd.read_parquet(DATA_FILE)
    feat = feature_cols(df)

    val = df[df["split_day"] == "val"]
    test = df[df["split_day"] == "test"]
    X_val = val[feat]
    y_val = val["is_attack"].to_numpy().astype(int)
    X_test = test[feat]
    y_test = test["is_attack"].to_numpy().astype(int)
    print(f"[calib] val={len(y_val):,}  test={len(y_test):,}  attack_rate={y_test.mean():.4f}")

    model_dir = MODELS_DIR / "binary_day"
    models_to_calibrate = ["random_forest", "xgboost", "lightgbm"]

    results = {}

    for name in models_to_calibrate:
        path = model_dir / f"{name}.joblib"
        if not path.exists():
            print(f"[calib] skip {name}")
            continue
        print(f"\n[calib] === {name} ===")
        model = joblib.load(path)
        p_val_raw = model.predict_proba(X_val)[:, 1]
        p_test_raw = model.predict_proba(X_test)[:, 1]

        ece_raw = expected_calibration_error(y_test, p_test_raw)

        platt = fit_platt(y_val, p_val_raw)
        p_test_platt = platt.predict_proba(p_test_raw.reshape(-1, 1))[:, 1]
        ece_platt = expected_calibration_error(y_test, p_test_platt)

        iso = fit_isotonic(y_val, p_val_raw)
        p_test_iso = iso.predict(p_test_raw)
        ece_iso = expected_calibration_error(y_test, p_test_iso)

        choices = [("raw", p_test_raw, ece_raw),
                   ("platt", p_test_platt, ece_platt),
                   ("isotonic", p_test_iso, ece_iso)]
        best_method, p_best, ece_best = min(choices, key=lambda x: x[2])

        ops_raw = {}
        ops_best = {}
        for t in FPR_TARGETS:
            rec_r, fpr_r, thr_r = recall_at_fpr(y_test, p_test_raw, t)
            rec_b, fpr_b, thr_b = recall_at_fpr(y_test, p_best, t)
            ops_raw[f"fpr_{t}"] = {"recall": rec_r, "actual_fpr": fpr_r, "threshold": thr_r}
            ops_best[f"fpr_{t}"] = {"recall": rec_b, "actual_fpr": fpr_b, "threshold": thr_b}

        fpr_val, tpr_val, thr_val = roc_curve(y_val, p_val_raw)
        j = tpr_val - fpr_val
        theta_raw = float(thr_val[int(np.argmax(j))])
        pred_raw = (p_test_raw >= theta_raw).astype(int)
        f1_raw = f1_score(y_test, pred_raw)

        if best_method == "platt":
            p_val_best = platt.predict_proba(p_val_raw.reshape(-1, 1))[:, 1]
        elif best_method == "isotonic":
            p_val_best = iso.predict(p_val_raw)
        else:
            p_val_best = p_val_raw
        fpr_val_b, tpr_val_b, thr_val_b = roc_curve(y_val, p_val_best)
        j_b = tpr_val_b - fpr_val_b
        theta_best = float(thr_val_b[int(np.argmax(j_b))])
        pred_best = (p_best >= theta_best).astype(int)
        f1_best = f1_score(y_test, pred_best)

        print(f"  ECE: raw={ece_raw:.4f}  platt={ece_platt:.4f}  iso={ece_iso:.4f}  "
              f"-> best={best_method} ({ece_best:.4f})")
        print(f"  F1 (Youden) raw={f1_raw:.4f}  calibrated={f1_best:.4f}")
        for t in FPR_TARGETS:
            print(f"  recall @ {t*100:g}% FPR: raw={ops_raw[f'fpr_{t}']['recall']:.4f}  "
                  f"calib={ops_best[f'fpr_{t}']['recall']:.4f}")

        results[name] = {
            "ece_raw": ece_raw,
            "ece_platt": ece_platt,
            "ece_isotonic": ece_iso,
            "ece_best": ece_best,
            "best_method": best_method,
            "f1_youden_raw": f1_raw,
            "f1_youden_calibrated": f1_best,
            "threshold_raw_youden": theta_raw,
            "threshold_calibrated_youden": theta_best,
            "operating_points_raw": ops_raw,
            "operating_points_calibrated": ops_best,
        }

        if best_method != "raw":
            calib_path = model_dir / f"{name}_calibrator_{best_method}.joblib"
            joblib.dump(platt if best_method == "platt" else iso, calib_path)
            print(f"  saved calibrator to {calib_path.name}")

    (OUT / "calibration_results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[calib] wrote {OUT}/calibration_results.json")

    lines = [
        "# Probability Calibration (Binary Day Split)\n",
        "Post-processing calibration using Platt scaling and isotonic regression. "
        "Both calibrators are fit on validation-set probabilities (Thursday) and "
        "applied to test-set probabilities (Friday). The better of the two on test "
        "ECE is reported per model.\n",
        "## Calibration comparison\n",
        "| Model | ECE raw | ECE Platt | ECE isotonic | Best | F1 raw | F1 calibrated |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['ece_raw']:.4f} | {r['ece_platt']:.4f} | {r['ece_isotonic']:.4f} | "
            f"{r['best_method']} | {r['f1_youden_raw']:.4f} | {r['f1_youden_calibrated']:.4f} |"
        )

    lines.append("\n## Operating points (recall at fixed FPR)\n")
    lines.append("| Model | FPR target | Recall raw | Recall calibrated | Delta |")
    lines.append("|---|---|---:|---:|---:|")
    for name, r in results.items():
        for t in FPR_TARGETS:
            rec_r = r["operating_points_raw"][f"fpr_{t}"]["recall"]
            rec_b = r["operating_points_calibrated"][f"fpr_{t}"]["recall"]
            delta = rec_b - rec_r
            lines.append(f"| {name} | {t*100:g}% | {rec_r:.4f} | {rec_b:.4f} | {delta:+.4f} |")

    (OUT / "calibration_summary.md").write_text("\n".join(lines))
    print(f"[calib] wrote {OUT}/calibration_summary.md")


if __name__ == "__main__":
    main()
