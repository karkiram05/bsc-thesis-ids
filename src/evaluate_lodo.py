"""Leave-One-Day-Out cross-validation for binary and multi-class tasks."""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import DATA_FILE, REPORTS_DIR, RNG, feature_cols, binary_y

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def _build_binary_models(baseline_only: bool = False):
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
        def _xgb():
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=400, max_depth=8, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                min_child_weight=1, eval_metric="logloss",
                random_state=RNG, n_jobs=-1, verbosity=0,
            )
        models.append(("xgboost", _xgb))

        def _lgb():
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                n_estimators=400, max_depth=8, learning_rate=0.05,
                num_leaves=63, min_child_samples=20, is_unbalance=True,
                random_state=RNG, n_jobs=-1, verbose=-1,
            )
        models.append(("lightgbm", _lgb))
    return models


def _build_multiclass_models(baseline_only: bool = False, skip_logreg: bool = False):
    models = []
    if not skip_logreg:
        models.append(("logreg", lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                solver="lbfgs", max_iter=500, tol=1e-3,
                class_weight="balanced", random_state=RNG,
            )),
        ])))
    models.append(("random_forest", lambda: RandomForestClassifier(
        n_estimators=300, max_depth=24, max_features="sqrt",
        class_weight="balanced_subsample", random_state=RNG, n_jobs=-1,
    )))
    if not baseline_only:
        def _xgb():
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                subsample=0.9, colsample_bytree=0.9,
                eval_metric="mlogloss",
                random_state=RNG, n_jobs=-1, verbosity=0,
            )
        models.append(("xgboost", _xgb))

        def _lgb():
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                num_leaves=63, min_child_samples=20,
                random_state=RNG, n_jobs=-1, verbose=-1,
            )
        models.append(("lightgbm", _lgb))
    return models


# ── binary LODO ──────────────────────────────────────────────────────

def _binary_main(args) -> None:
    out_dir = REPORTS_DIR / "lodo"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_FILE)
    feat = feature_cols(df)
    model_specs = _build_binary_models(args.baseline_only)

    all_results: list[dict] = []

    for hold_out_day in DAYS:
        train_days = [d for d in DAYS if d != hold_out_day]
        test_df = df[df["day"] == hold_out_day]
        X_test = test_df[feat]
        y_test = binary_y(test_df)

        train_days_ordered = [d for d in DAYS if d in train_days]
        inner_val_day = train_days_ordered[-1]
        inner_train_days = [d for d in train_days if d != inner_val_day]

        inner_train_df = df[df["day"].isin(inner_train_days)]
        inner_val_df = df[df["day"] == inner_val_day]
        X_train = inner_train_df[feat]
        y_train = binary_y(inner_train_df)
        X_val = inner_val_df[feat]
        y_val = binary_y(inner_val_df)

        n_attack_test = int(y_test.sum())
        n_attack_val = int(y_val.sum())
        attack_rate = float(y_test.mean())

        print(f"\n[LODO] Hold-out={hold_out_day}  "
              f"train={len(y_train)} val={len(y_val)} (val_day={inner_val_day}) "
              f"test={len(y_test)} attack_test={n_attack_test} ({attack_rate:.4f})")

        for model_name, model_factory in model_specs:
            model = model_factory()
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]

            if n_attack_val > 0:
                proba_val = model.predict_proba(X_val)[:, 1]
                fpr_val, tpr_val, thr_roc = roc_curve(y_val, proba_val)
                j_scores = tpr_val[:-1] - fpr_val[:-1]
                best_thr = float(thr_roc[int(np.argmax(j_scores))])
            else:
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

            if n_attack_test == 0:
                fp = int(((pred == 1) & (y_test == 0)).sum())
                n_neg = int((y_test == 0).sum())
                fpr = float(fp / n_neg) if n_neg > 0 else 0.0
                fold_result.update({
                    "roc_auc": None, "pr_auc": None, "f1": None,
                    "fpr": round(fpr, 6),
                    "note": "Benign-only fold",
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
            print(f"  {model_name}: ROC={fold_result['roc_auc']}  "
                  f"F1={fold_result['f1']}  FPR={fold_result['fpr']:.6f}")

    with open(out_dir / "lodo_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[LODO] wrote {out_dir / 'lodo_results.json'}")

    results_df = pd.DataFrame(all_results)
    summary_rows = []
    for model_name in results_df["model"].unique():
        mdf = results_df[results_df["model"] == model_name]
        fpr_vals = mdf["fpr"].values
        fpr_mean = float(np.mean(fpr_vals))
        fpr_std = float(np.std(fpr_vals, ddof=1)) if len(fpr_vals) > 1 else 0.0

        attack_folds = mdf[mdf["roc_auc"].notna()]
        if len(attack_folds) > 0:
            roc_vals = attack_folds["roc_auc"].values.astype(float)
            pr_vals = attack_folds["pr_auc"].values.astype(float)
            f1_vals = attack_folds["f1"].values.astype(float)
            summary_rows.append({
                "model": model_name,
                "n_folds": len(DAYS),
                "n_attack_folds": len(attack_folds),
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
    summary_df.to_csv(out_dir / "lodo_summary.csv", index=False)

    lines = [
        "# Leave-One-Day-Out Binary Cross-Validation\n\n",
        "**Protocol**: Train on 3 days, tune threshold on 1 inner val day "
        "(Youden's J), test on 1 held-out day.\n",
        "**Note**: Monday is benign-only; ROC-AUC/PR-AUC/F1 on 4 folds with attacks.\n\n",
        "## Summary\n\n",
        "| Model | ROC-AUC | PR-AUC | F1 | FPR |\n",
        "|-------|---------|--------|----|-----|\n",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['model']} "
            f"| {row['roc_auc_mean']:.4f} +/- {row['roc_auc_std']:.4f} "
            f"| {row['pr_auc_mean']:.4f} +/- {row['pr_auc_std']:.4f} "
            f"| {row['f1_mean']:.4f} +/- {row['f1_std']:.4f} "
            f"| {row['fpr_mean']:.6f} +/- {row['fpr_std']:.6f} |\n"
        )
    (out_dir / "lodo_summary.md").write_text("".join(lines), encoding="utf-8")
    print(f"[LODO] wrote {out_dir / 'lodo_summary.md'}")

    print("\n" + "=" * 60 + "\nLODO SUMMARY\n" + "=" * 60)
    for _, row in summary_df.iterrows():
        print(f"  {row['model']:20s}  "
              f"ROC={row['roc_auc_mean']:.4f}+/-{row['roc_auc_std']:.4f}  "
              f"F1={row['f1_mean']:.4f}+/-{row['f1_std']:.4f}")


# ── multi-class LODO ─────────────────────────────────────────────────

def _multiclass_main(args) -> None:
    out_dir = REPORTS_DIR / "lodo_multiclass"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_FILE)
    feat = feature_cols(df)

    le = LabelEncoder().fit(df["attack_type"].astype(str))
    y_full = le.transform(df["attack_type"].astype(str))
    df = df.copy()
    df["_y"] = y_full

    model_specs = _build_multiclass_models(args.baseline_only, args.skip_logreg)
    all_results: list[dict] = []

    for hold_out_day in DAYS:
        train_df = df[df["day"] != hold_out_day]
        test_df = df[df["day"] == hold_out_day]

        if args.sample_frac < 1.0:
            train_df = (
                train_df.groupby("attack_type", group_keys=False)
                .apply(lambda g: g.sample(
                    frac=args.sample_frac, random_state=RNG)
                       if len(g) > 1 else g)
            )

        X_train = train_df[feat]
        y_train = train_df["_y"].to_numpy()
        X_test = test_df[feat]
        y_test = test_df["_y"].to_numpy()

        train_classes = sorted(set(y_train.tolist()))
        test_classes = sorted(set(y_test.tolist()))
        unseen = sorted(set(test_classes) - set(train_classes))
        unseen_names = [le.classes_[c] for c in unseen]

        print(f"\n[LODO-MC] Hold-out={hold_out_day}  "
              f"train={len(y_train)} test={len(y_test)} "
              f"train_classes={len(train_classes)} test_classes={len(test_classes)} "
              f"unseen={unseen_names}")

        unique_train = sorted(set(y_train.tolist()))
        global_to_local = {g: i for i, g in enumerate(unique_train)}
        local_to_global = {i: g for g, i in global_to_local.items()}
        y_train_local = np.array([global_to_local[v] for v in y_train])

        for model_name, factory in model_specs:
            model = factory()
            model.fit(X_train, y_train_local)
            pred_local = model.predict(X_test)
            pred = np.array([local_to_global[int(p)] for p in pred_local])

            labels_eval = test_classes
            macro_f1 = float(f1_score(
                y_test, pred, labels=labels_eval,
                average="macro", zero_division=0))
            weighted_f1 = float(f1_score(
                y_test, pred, labels=labels_eval,
                average="weighted", zero_division=0))
            all_labels = sorted(set(train_classes) | set(test_classes))
            per_class = f1_score(y_test, pred, labels=all_labels,
                                 average=None, zero_division=0)
            per_class_map = {
                le.classes_[c]: round(float(s), 4)
                for c, s in zip(all_labels, per_class)
            }

            all_results.append({
                "held_out_day": hold_out_day,
                "model": model_name,
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "n_train_classes": len(train_classes),
                "n_test_classes": len(test_classes),
                "unseen_classes": unseen_names,
                "macro_f1": round(macro_f1, 4),
                "weighted_f1": round(weighted_f1, 4),
                "per_class_f1": per_class_map,
            })
            print(f"  {model_name:14s}  macro_F1={macro_f1:.4f}  "
                  f"weighted_F1={weighted_f1:.4f}  unseen={len(unseen)}")

    (out_dir / "lodo_multiclass_results.json").write_text(
        json.dumps(all_results, indent=2))

    results_df = pd.DataFrame(all_results)
    attack_df = results_df[results_df["held_out_day"] != "Monday"]

    summary_rows = []
    for model_name in results_df["model"].unique():
        mdf = attack_df[attack_df["model"] == model_name]
        m = mdf["macro_f1"].values.astype(float)
        w = mdf["weighted_f1"].values.astype(float)
        summary_rows.append({
            "model": model_name,
            "n_attack_folds": len(mdf),
            "macro_f1_mean": round(float(np.mean(m)), 4),
            "macro_f1_std": round(float(np.std(m, ddof=1)), 4),
            "macro_f1_min": round(float(np.min(m)), 4),
            "macro_f1_max": round(float(np.max(m)), 4),
            "weighted_f1_mean": round(float(np.mean(w)), 4),
            "weighted_f1_std": round(float(np.std(w, ddof=1)), 4),
        })
    pd.DataFrame(summary_rows).to_csv(
        out_dir / "lodo_multiclass_summary.csv", index=False)

    lines = [
        "# Leave-One-Day-Out Multi-Class Cross-Validation\n\n",
        "Tests whether the multi-class collapse observed on the Friday "
        "hold-out generalises to the other days.\n\n",
        "Monday is benign-only and excluded from summary stats.\n\n",
        "## Summary across the four attack-bearing folds\n\n",
        "| Model | Macro F1 (mean +/- std) | Range | Weighted F1 |\n",
        "|-------|-----------------------|-------|-------------|\n",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model']} "
            f"| {row['macro_f1_mean']:.3f} +/- {row['macro_f1_std']:.3f} "
            f"| [{row['macro_f1_min']:.3f}, {row['macro_f1_max']:.3f}] "
            f"| {row['weighted_f1_mean']:.3f} +/- {row['weighted_f1_std']:.3f} |\n"
        )

    md_path = out_dir / "lodo_multiclass_summary.md"
    md_path.write_text("".join(lines), encoding="utf-8")
    print(f"\n[LODO-MC] wrote {md_path}")


# ── entry point ──────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="LODO cross-validation")
    ap.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--skip-logreg", action="store_true",
                    help="Skip LogReg (only for multiclass)")
    ap.add_argument("--sample-frac", type=float, default=1.0,
                    help="Subsample training data (only for multiclass)")
    args = ap.parse_args()

    if not DATA_FILE.exists():
        raise SystemExit(f"Missing {DATA_FILE}. Run: python -m src.prepare_cicids")

    if args.task == "binary":
        _binary_main(args)
    else:
        _multiclass_main(args)


if __name__ == "__main__":
    main()
