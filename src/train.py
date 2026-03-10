"""Train baseline (LogReg) and stronger models (RF, XGBoost). Multi-class on attack_type."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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


def load_splits(df: pd.DataFrame, split_col: str):
    """Return (X_train, y_train, X_val, y_val, X_test, y_test, le, feat).

    The LabelEncoder is fitted on ALL attack_type labels in the full dataframe,
    not just the three splits. This ensures labels are always contiguous 0..N-1
    even when some classes are absent from the training split (e.g. day split
    where Friday-only attacks never appear in Mon-Wed training data).
    XGBoost requires contiguous labels starting from 0 -- this guarantees that.
    """
    feat = _feature_cols(df)
    assert feat, "No feature columns found"

    le = LabelEncoder()
    le.fit(df["attack_type"].astype(str))

    def part(s: str):
        sub = df.loc[df[split_col] == s]
        y_raw = sub["attack_type"].astype(str)
        X = sub[feat].copy()
        y_enc = le.transform(y_raw)
        return X, np.asarray(y_enc)

    Xt, yt = part("train")
    Xv, yv = part("val")
    Xs, ys = part("test")

    return Xt, yt, Xv, yv, Xs, ys, le, feat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["day", "strat"], default="day")
    ap.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--baseline-only", action="store_true")
    args = ap.parse_args()

    split_col = SPLIT_COL_DAY if args.split == "day" else SPLIT_COL_STRAT
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        raise SystemExit(f"Missing {DATA_FILE}. Run: python -m src.prepare_data")

    df = pd.read_parquet(DATA_FILE)
    for s in ["train", "val", "test"]:
        if (df[split_col] == s).sum() == 0:
            raise SystemExit(f"No '{s}' split in {split_col}.")

    if args.sample > 0:
        parts = []
        for s in ["train", "val", "test"]:
            sub = df[df[split_col] == s]
            parts.append(sub.sample(n=min(args.sample, len(sub)), random_state=RNG))
        df = pd.concat(parts, ignore_index=True)
        print(f"[sample] {args.sample} per split -> {len(df)} rows")

    Xt, yt, Xv, yv, Xs, ys, le, feat = load_splits(df, split_col)
    n_classes = len(le.classes_)
    print(f"[load] train={len(yt)} val={len(yv)} test={len(ys)} features={len(feat)} classes={n_classes}")

    scaler = StandardScaler()
    Xt_s = scaler.fit_transform(Xt)

    models = []

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

    print("[train] RandomForest ...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=24, max_features="sqrt",
        class_weight="balanced_subsample", random_state=RNG, n_jobs=-1,
    )
    rf.fit(Xt_s, yt)
    models.append(("random_forest", rf))
    print("[train] RandomForest done.")

    if not args.baseline_only:
        print("[train] XGBoost ...")
        xgb_clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            eval_metric="mlogloss",
            random_state=RNG, n_jobs=-1,
            verbosity=0,
        )
        xgb_clf.fit(Xt_s, yt)
        models.append(("xgboost", xgb_clf))
        print("[train] XGBoost done.")

    joblib.dump(le, out / "label_encoder.joblib")
    joblib.dump(scaler, out / "scaler.joblib")
    with open(out / "feature_names.json", "w") as f:
        json.dump(feat, f, indent=2)
    with open(out / "meta.json", "w") as f:
        json.dump({"split_col": split_col, "n_classes": n_classes,
                   "feature_names": feat, "classes": list(le.classes_)}, f, indent=2)

    for name, m in models:
        path = out / f"{name}.joblib"
        joblib.dump(m, path)
        print(f"[save] {path}")

    print("[train] done.")


if __name__ == "__main__":
    main()
