"""Train models on UNSW-NB15: multi-class and binary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import (
    UNSW_DATA_FILE,
    UNSW_MODELS_DIR,
    UNSW_NON_FEATURE,
    RNG,
)


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in UNSW_NON_FEATURE]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["multiclass", "binary"], default="multiclass")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--baseline-only", action="store_true", help="Skip XGBoost")
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = UNSW_MODELS_DIR / args.task

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not UNSW_DATA_FILE.exists():
        raise SystemExit(f"Missing {UNSW_DATA_FILE}. Run: python -m src.prepare_unsw")

    df = pd.read_parquet(UNSW_DATA_FILE)
    feat = _feature_cols(df)

    for s in ["train", "val", "test"]:
        if (df["split"] == s).sum() == 0:
            raise SystemExit(f"No '{s}' rows in split column.")

    def part(s: str):
        sub = df.loc[df["split"] == s]
        X = sub[feat].copy()
        return X, sub

    Xt, train_sub = part("train")
    Xv, val_sub = part("val")
    Xs, test_sub = part("test")

    if args.task == "binary":
        yt = (train_sub["label"].values).astype(int)
        yv = (val_sub["label"].values).astype(int)
        ys = (test_sub["label"].values).astype(int)
        class_names = ["Normal", "Attack"]
        le = None
        print(f"[load] Binary: train={len(yt)} val={len(yv)} test={len(ys)} "
              f"features={len(feat)} pos_rate={yt.mean():.4f}")
    else:
        le = LabelEncoder()
        le.fit(train_sub["attack_cat"])
        yt = le.transform(train_sub["attack_cat"])
        yv = le.transform(val_sub["attack_cat"])
        ys = le.transform(test_sub["attack_cat"])
        class_names = list(le.classes_)
        print(f"[load] Multi-class: train={len(yt)} val={len(yv)} test={len(ys)} "
              f"features={len(feat)} classes={len(class_names)}")

    # Scaler (only LogReg needs it)
    scaler = StandardScaler()
    scaler.fit(Xt)

    models: list[tuple[str, object]] = []

    # LogReg
    print("[train] LogReg ...")
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="saga", max_iter=2000, tol=1e-3,
            class_weight="balanced", random_state=RNG,
        )),
    ])
    lr.fit(Xt, yt)
    models.append(("logreg", lr))
    print("[train] LogReg done.")

    # RandomForest
    print("[train] RandomForest ...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=24, max_features="sqrt",
        class_weight="balanced_subsample", random_state=RNG, n_jobs=-1,
    )
    rf.fit(Xt, yt)
    models.append(("random_forest", rf))
    print("[train] RandomForest done.")

    # XGBoost
    if not args.baseline_only:
        import xgboost as xgb
        print("[train] XGBoost ...")

        xgb_params = {
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.1,
            "random_state": RNG,
            "n_jobs": -1,
            "verbosity": 0,
        }

        if args.task == "binary":
            n_pos = max(int(yt.sum()), 1)
            n_neg = max(int((yt == 0).sum()), 1)
            xgb_params["scale_pos_weight"] = n_neg / n_pos
            xgb_params["eval_metric"] = "logloss"
            xgb_clf = xgb.XGBClassifier(**xgb_params)
        else:
            xgb_params["eval_metric"] = "mlogloss"
            xgb_clf = xgb.XGBClassifier(**xgb_params)

        xgb_clf.fit(Xt, yt, verbose=False)
        models.append(("xgboost", xgb_clf))
        print("[train] XGBoost done.")

        # LightGBM
        import lightgbm as lgb
        print("[train] LightGBM ...")
        lgb_params = {
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.1,
            "num_leaves": 63,
            "min_child_samples": 20,
            "random_state": RNG,
            "n_jobs": -1,
            "verbose": -1,
        }
        if args.task == "binary":
            lgb_params["is_unbalance"] = True
        else:
            lgb_params["class_weight"] = "balanced"
        lgb_clf = lgb.LGBMClassifier(**lgb_params)
        lgb_clf.fit(Xt, yt)
        models.append(("lightgbm", lgb_clf))
        print("[train] LightGBM done.")

    # Save artifacts
    joblib.dump(scaler, out / "scaler.joblib")
    if le is not None:
        joblib.dump(le, out / "label_encoder.joblib")

    meta = {
        "dataset": "unsw-nb15",
        "task": args.task,
        "feature_names": feat,
        "classes": class_names,
        "n_classes": len(class_names),
    }
    (out / "feature_names.json").write_text(json.dumps(feat, indent=2), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for name, m in models:
        path = out / f"{name}.joblib"
        joblib.dump(m, path)
        print(f"[save] {path}")

    print(f"[train-unsw] done ({args.task}).")


if __name__ == "__main__":
    main()
