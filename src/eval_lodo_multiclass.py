from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import DATA_FILE, REPORTS_DIR, RNG, feature_cols

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
OUT_DIR = REPORTS_DIR / "lodo_multiclass"


def _build_models(baseline_only: bool = False, skip_logreg: bool = False):
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
        def _xgb_factory():
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                subsample=0.9, colsample_bytree=0.9,
                eval_metric="mlogloss",
                random_state=RNG, n_jobs=-1, verbosity=0,
            )
        models.append(("xgboost", _xgb_factory))

        def _lgb_factory():
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                num_leaves=63, min_child_samples=20,
                random_state=RNG, n_jobs=-1, verbose=-1,
            )
        models.append(("lightgbm", _lgb_factory))
    return models


def main() -> None:
    ap = argparse.ArgumentParser(description="LODO multi-class cross-validation")
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--skip-logreg", action="store_true",
                    help="Skip LogReg (slow on full multiclass)")
    ap.add_argument("--sample-frac", type=float, default=1.0,
                    help="Subsample training data to this fraction "
                         "(stratified per attack_type). Test data always full.")
    args = ap.parse_args()

    if not DATA_FILE.exists():
        raise SystemExit(f"Missing {DATA_FILE}. Run: python -m src.prepare_data")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_FILE)
    feat = feature_cols(df)

    le = LabelEncoder().fit(df["attack_type"].astype(str))
    y_full = le.transform(df["attack_type"].astype(str))
    df = df.copy()
    df["_y"] = y_full

    model_specs = _build_models(args.baseline_only, args.skip_logreg)
    all_results: list[dict] = []

    for hold_out_day in DAYS:
        train_df = df[df["day"] != hold_out_day]
        test_df = df[df["day"] == hold_out_day]

        # Optional subsampling of training data (stratified by class)
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

        print(
            f"\n[LODO-MC] Hold-out={hold_out_day}  "
            f"train={len(y_train)} test={len(y_test)} "
            f"train_classes={len(train_classes)} test_classes={len(test_classes)} "
            f"unseen_in_test={unseen_names}"
        )

        # Per-fold remap to contiguous label space for models (XGBoost
        # in particular) that require classes 0..K-1. Predictions are
        # mapped back to the global label space for cross-fold comparison.
        unique_train = sorted(set(y_train.tolist()))
        global_to_local = {g: i for i, g in enumerate(unique_train)}
        local_to_global = {i: g for g, i in global_to_local.items()}
        y_train_local = np.array([global_to_local[v] for v in y_train])

        for model_name, factory in model_specs:
            model = factory()
            model.fit(X_train, y_train_local)
            pred_local = model.predict(X_test)
            pred = np.array([local_to_global[int(p)] for p in pred_local])

            # Macro F1 over the union of classes present in test set.
            labels_eval = test_classes
            macro_f1 = float(
                f1_score(y_test, pred, labels=labels_eval,
                         average="macro", zero_division=0)
            )
            weighted_f1 = float(
                f1_score(y_test, pred, labels=labels_eval,
                         average="weighted", zero_division=0)
            )
            all_labels = sorted(set(train_classes) | set(test_classes))
            per_class = f1_score(y_test, pred, labels=all_labels,
                                 average=None, zero_division=0)
            per_class_map = {
                le.classes_[c]: round(float(s), 4)
                for c, s in zip(all_labels, per_class)
            }

            row = {
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
            }
            all_results.append(row)
            print(
                f"  {model_name:14s}  macro_F1={macro_f1:.4f}  "
                f"weighted_F1={weighted_f1:.4f}  "
                f"unseen_classes={len(unseen)}"
            )

    (OUT_DIR / "lodo_multiclass_results.json").write_text(
        json.dumps(all_results, indent=2)
    )

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
        OUT_DIR / "lodo_multiclass_summary.csv", index=False
    )

    # Markdown summary
    lines = [
        "# Leave-One-Day-Out Multi-Class Cross-Validation\n\n",
        "Tests whether the multi-class collapse observed on the Friday "
        "hold-out generalises to the other days.\n\n",
        "Monday is benign-only and is therefore not reported in the summary "
        "(macro F1 is undefined for a single-class test set).\n\n",
        "## Per-fold macro F1 (rows = test day, columns = model)\n\n",
        "| Held-out day | LogReg | RF | XGBoost | LightGBM | Unseen classes in test |\n",
        "|--------------|--------|------|---------|----------|-------------------------|\n",
    ]
    for day in DAYS:
        if day == "Monday":
            continue
        row = {r["model"]: r for r in all_results if r["held_out_day"] == day}
        unseen = row.get("random_forest", {}).get("unseen_classes", [])
        lines.append(
            f"| {day} "
            f"| {row.get('logreg', {}).get('macro_f1', '—')} "
            f"| {row.get('random_forest', {}).get('macro_f1', '—')} "
            f"| {row.get('xgboost', {}).get('macro_f1', '—')} "
            f"| {row.get('lightgbm', {}).get('macro_f1', '—')} "
            f"| {', '.join(unseen) if unseen else 'none'} |\n"
        )

    lines += [
        "\n## Summary across the four attack-bearing folds\n\n",
        "| Model | Macro F1 (mean ± std) | Macro F1 range | Weighted F1 (mean ± std) |\n",
        "|-------|-----------------------|----------------|----------------------------|\n",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model']} "
            f"| {row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f} "
            f"| [{row['macro_f1_min']:.3f}, {row['macro_f1_max']:.3f}] "
            f"| {row['weighted_f1_mean']:.3f} ± {row['weighted_f1_std']:.3f} |\n"
        )

    lines += [
        "\n## Per-class F1 for the random forest (collapse evidence)\n\n",
        "Rows = test day (held out), columns = predicted attack class. "
        "Zeros mark classes where the model never produces the correct label.\n\n",
    ]
    rf_rows = [r for r in all_results if r["model"] == "random_forest"
               and r["held_out_day"] != "Monday"]
    
    all_class_names = sorted({c for r in rf_rows for c in r["per_class_f1"]})
    header = "| Held-out day | " + " | ".join(all_class_names) + " |\n"
    sep = "|" + "|".join("---" for _ in range(len(all_class_names) + 1)) + "|\n"
    lines.append(header)
    lines.append(sep)
    for r in rf_rows:
        cells = [f"{r['per_class_f1'].get(c, 0):.2f}" for c in all_class_names]
        lines.append(f"| {r['held_out_day']} | " + " | ".join(cells) + " |\n")

    md_path = OUT_DIR / "lodo_multiclass_summary.md"
    md_path.write_text("".join(lines), encoding="utf-8")
    print(f"\n[LODO-MC] wrote {md_path}")


if __name__ == "__main__":
    main()
