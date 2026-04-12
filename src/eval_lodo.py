"""Leave-One-Day-Out (LODO) cross-validation for binary IDS.

Trains binary (Benign vs Attack) models using 4 days and tests on the held-out
day. Repeats for all 5 days. Reports mean ± std of ROC-AUC, PR-AUC, F1, and FPR.

This provides temporal cross-validation evidence that binary detection
generalises across days, even though multi-class detection fails due to
disjoint attack types.

Output:
  reports/lodo/
    - lodo_results.json      — per-fold metrics
    - lodo_summary.md        — formatted table for thesis
    - lodo_summary.csv       — machine-readable summary

Usage:
  python -m src.eval_lodo
  python -m src.eval_lodo --baseline-only   # skip XGBoost
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DATA_FILE, NON_FEATURE, REPORTS_DIR, RNG

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
OUT_DIR = REPORTS_DIR / "lodo"


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE]


def _binary_y(df: pd.DataFrame) -> np.ndarray:
    return (df["attack_type"].astype(str) != "Benign").astype(int).to_numpy()


def _build_models(baseline_only: bool = False):
    """Return list of (name, model_factory) tuples."""
    models = [
        ("logreg", lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                solver="saga", max_iter=2000, tol=1e-3,
                class_weight="balanced", random_state=RNG,
            )),
        ])),
        ("random_forest", lambda: RandomForestClassifier(
            n_estimators=300, max_depth=24, max_features="sqrt",
            class_weight="balanced_subsample", random_state=RNG, n_jobs=-1,
        )),
    ]

    if not baseline_only:
        def _xgb_factory():
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=400, max_depth=8, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                min_child_weight=1, eval_metric="logloss",
                random_state=RNG, n_jobs=-1,
            )
        models.append(("xgboost", _xgb_factory))

        def _lgb_factory():
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                n_estimators=400, max_depth=8, learning_rate=0.05,
                num_leaves=63, min_child_samples=20, is_unbalance=True,
                random_state=RNG, n_jobs=-1, verbose=-1,
            )
        models.append(("lightgbm", _lgb_factory))

    return models


def main() -> None:
    ap = argparse.ArgumentParser(description="LODO binary cross-validation")
    ap.add_argument("--baseline-only", action="store_true")
    args = ap.parse_args()

    if not DATA_FILE.exists():
        raise SystemExit(f"Missing {DATA_FILE}. Run: python -m src.prepare_data")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_FILE)
    feat = _feature_cols(df)
    assert feat, "No feature columns found"

    model_specs = _build_models(args.baseline_only)

    # Per-fold, per-model results
    all_results: list[dict] = []

    for hold_out_day in DAYS:
        train_days = [d for d in DAYS if d != hold_out_day]
        train_df = df[df["day"].isin(train_days)]
        test_df = df[df["day"] == hold_out_day]

        X_test = test_df[feat]
        y_test = _binary_y(test_df)

        # Inner split: use one of the training days as val for threshold tuning.
        # Pick the last training day chronologically (closest to test).
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        train_days_ordered = [d for d in day_order if d in train_days]
        inner_val_day = train_days_ordered[-1]
        inner_train_days = [d for d in train_days if d != inner_val_day]

        inner_train_df = df[df["day"].isin(inner_train_days)]
        inner_val_df = df[df["day"] == inner_val_day]

        X_train = inner_train_df[feat]
        y_train = _binary_y(inner_train_df)
        X_val = inner_val_df[feat]
        y_val = _binary_y(inner_val_df)

        n_attack_test = int(y_test.sum())
        n_attack_val = int(y_val.sum())
        attack_rate = float(y_test.mean())

        print(f"\n[LODO] Hold-out={hold_out_day}  "
              f"train={len(y_train)} val={len(y_val)} (val_day={inner_val_day}) "
              f"test={len(y_test)} "
              f"attack_test={n_attack_test} ({attack_rate:.4f})")

        for model_name, model_factory in model_specs:
            model = model_factory()
            model.fit(X_train, y_train)

            proba = model.predict_proba(X_test)[:, 1]

            # Tune threshold on inner val set — Youden's J statistic
            # (prevalence-invariant, transfers across distribution shifts)
            if n_attack_val > 0:
                proba_val = model.predict_proba(X_val)[:, 1]
                fpr_val, tpr_val, thr_roc = roc_curve(y_val, proba_val)
                j_scores = tpr_val[:-1] - fpr_val[:-1]
                best_j_idx = int(np.argmax(j_scores))
                best_thr = float(thr_roc[best_j_idx])
            else:
                # Val has no attacks (e.g., Monday as val) — use default
                best_thr = 0.5

            pred = (proba >= best_thr).astype(int)

            fold_result = {
                "held_out_day": hold_out_day,
                "model": model_name,
                "n_train": int(len(y_train)),
                "n_val": int(len(y_val)),
                "val_day": inner_val_day,
                "n_test": int(len(y_test)),
                "n_attack_test": n_attack_test,
                "attack_rate_test": round(attack_rate, 4),
                "threshold": round(best_thr, 6),
            }

            # Monday is benign-only — ROC-AUC and PR-AUC are undefined
            if n_attack_test == 0:
                # Only FPR is meaningful
                fp = int(((pred == 1) & (y_test == 0)).sum())
                n_neg = int((y_test == 0).sum())
                fpr = float(fp / n_neg) if n_neg > 0 else 0.0
                fold_result.update({
                    "roc_auc": None,
                    "pr_auc": None,
                    "f1": None,
                    "fpr": round(fpr, 6),
                    "note": "Benign-only fold — only FPR is meaningful",
                })
            else:
                roc_auc = float(roc_auc_score(y_test, proba))
                pr_auc = float(average_precision_score(y_test, proba))
                f1 = float(f1_score(y_test, pred, average="binary", zero_division=0))
                fp = int(((pred == 1) & (y_test == 0)).sum())
                n_neg = int((y_test == 0).sum())
                fpr = float(fp / n_neg) if n_neg > 0 else 0.0
                fold_result.update({
                    "roc_auc": round(roc_auc, 4),
                    "pr_auc": round(pr_auc, 4),
                    "f1": round(f1, 4),
                    "fpr": round(fpr, 6),
                    "note": None,
                })

            all_results.append(fold_result)
            metrics_str = (
                f"  {model_name}: "
                f"ROC={fold_result['roc_auc']}  "
                f"PR={fold_result['pr_auc']}  "
                f"F1={fold_result['f1']}  "
                f"FPR={fold_result['fpr']:.6f}  "
                f"thr={best_thr:.6f}"
            )
            print(metrics_str)

    # Save per-fold results
    results_path = OUT_DIR / "lodo_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[LODO] wrote {results_path}")

    # Compute summary: mean ± std per model (excluding benign-only folds for ROC/PR/F1)
    results_df = pd.DataFrame(all_results)
    summary_rows = []
    for model_name in results_df["model"].unique():
        mdf = results_df[results_df["model"] == model_name]

        # FPR: all folds (including benign-only)
        fpr_vals = mdf["fpr"].values
        fpr_mean = float(np.mean(fpr_vals))
        fpr_std = float(np.std(fpr_vals, ddof=1)) if len(fpr_vals) > 1 else 0.0

        # ROC, PR, F1: only folds with attacks
        attack_folds = mdf[mdf["roc_auc"].notna()]
        n_attack_folds = len(attack_folds)

        if n_attack_folds > 0:
            roc_vals = attack_folds["roc_auc"].values.astype(float)
            pr_vals = attack_folds["pr_auc"].values.astype(float)
            f1_vals = attack_folds["f1"].values.astype(float)
            summary_rows.append({
                "model": model_name,
                "n_folds": len(DAYS),
                "n_attack_folds": n_attack_folds,
                "roc_auc_mean": round(float(np.mean(roc_vals)), 4),
                "roc_auc_std": round(float(np.std(roc_vals, ddof=1)) if len(roc_vals) > 1 else 0.0, 4),
                "pr_auc_mean": round(float(np.mean(pr_vals)), 4),
                "pr_auc_std": round(float(np.std(pr_vals, ddof=1)) if len(pr_vals) > 1 else 0.0, 4),
                "f1_mean": round(float(np.mean(f1_vals)), 4),
                "f1_std": round(float(np.std(f1_vals, ddof=1)) if len(f1_vals) > 1 else 0.0, 4),
                "fpr_mean": round(fpr_mean, 6),
                "fpr_std": round(fpr_std, 6),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "lodo_summary.csv", index=False)

    # Write markdown summary
    lines = [
        "# Leave-One-Day-Out Binary Cross-Validation\n\n",
        "**Protocol**: Train on 3 days, tune threshold on 1 inner val day (Youden's J), test on 1 held-out day. Repeat for all 5 days.\n",
        "**Task**: Binary (Benign vs Attack).\n",
        "**Note**: Monday is benign-only; ROC-AUC/PR-AUC/F1 computed on 4 folds with attacks.\n",
        "FPR is computed on all 5 folds.\n\n",
        "## Summary (mean ± std across folds)\n\n",
        "| Model | ROC-AUC | PR-AUC | F1 | FPR |\n",
        "|-------|---------|--------|----|-----|\n",
    ]

    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['model']} "
            f"| {row['roc_auc_mean']:.4f} ± {row['roc_auc_std']:.4f} "
            f"| {row['pr_auc_mean']:.4f} ± {row['pr_auc_std']:.4f} "
            f"| {row['f1_mean']:.4f} ± {row['f1_std']:.4f} "
            f"| {row['fpr_mean']:.6f} ± {row['fpr_std']:.6f} |\n"
        )

    lines += [
        "\n## Per-fold results\n\n",
        "| Day | Model | Attacks | Threshold | ROC-AUC | PR-AUC | F1 | FPR |\n",
        "|-----|-------|---------|-----------|---------|--------|----|-----|\n",
    ]
    for r in all_results:
        roc = f"{r['roc_auc']:.4f}" if r["roc_auc"] is not None else "—"
        pr = f"{r['pr_auc']:.4f}" if r["pr_auc"] is not None else "—"
        f1 = f"{r['f1']:.4f}" if r["f1"] is not None else "—"
        lines.append(
            f"| {r['held_out_day']} | {r['model']} | {r['n_attack_test']} "
            f"| {r['threshold']:.6f} "
            f"| {roc} | {pr} | {f1} | {r['fpr']:.6f} |\n"
        )

    lines += [
        "\n## Interpretation\n\n",
        "- If ROC-AUC is consistently high across folds (>0.95), binary detection generalises well temporally.\n",
        "- Thresholds are tuned per fold using Youden's J statistic (maximize TPR − FPR) on an inner validation day.\n",
        "- Youden's J is prevalence-invariant, so thresholds transfer better across days with different attack rates.\n",
        "- F1 depends on both discrimination (AUC) and threshold calibration. High AUC with low F1 means the threshold doesn't transfer.\n",
        "- Monday (benign-only) tests the false positive rate in isolation.\n",
        "- This complements the single day-split evaluation by providing variance estimates.\n",
    ]

    md_path = OUT_DIR / "lodo_summary.md"
    md_path.write_text("".join(lines), encoding="utf-8")
    print(f"[LODO] wrote {md_path}")
    print(f"[LODO] wrote {OUT_DIR / 'lodo_summary.csv'}")

    # Print summary to stdout
    print("\n" + "=" * 60)
    print("LODO SUMMARY")
    print("=" * 60)
    for _, row in summary_df.iterrows():
        print(
            f"  {row['model']:20s}  "
            f"ROC={row['roc_auc_mean']:.4f}±{row['roc_auc_std']:.4f}  "
            f"PR={row['pr_auc_mean']:.4f}±{row['pr_auc_std']:.4f}  "
            f"F1={row['f1_mean']:.4f}±{row['f1_std']:.4f}  "
            f"FPR={row['fpr_mean']:.6f}±{row['fpr_std']:.6f}"
        )


if __name__ == "__main__":
    main()