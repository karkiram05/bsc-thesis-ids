"""Train baseline (LogReg) and stronger models (RF, XGBoost). Multi-class on attack_type."""

from __future__ import annotations

import argparse
import json
import time
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
    return [c for c in df.columns if c not in NON_FEATURE and c in df.columns]


def load_splits(df: pd.DataFrame, split_col: str):
    """Return (X_train, y_train, X_val, y_val, X_test, y_test, le, feat)."""
    feat = _feature_cols(df)
    assert feat, "No feature columns found"

    def part(s: str):
        sub = df.loc[df[split_col] == s]
        y = sub["attack_type"].astype(str)
        X = sub[feat].copy()
        return X, y

    Xt, yt = part("train")
    Xv, yv = part("val")
    Xs, ys = part("test")

    le = LabelEncoder()
    all_labels = pd.concat([yt, yv, ys], ignore_index=True)
    le.fit(all_labels)
    yt_ = le.transform(yt)
    yv_ = le.transform(yv)
    ys_ = le.transform(ys)

    return Xt, np.asarray(yt_), Xv, np.asarray(yv_), Xs, np.asarray(ys_), le, feat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["day", "strat"], default="day", help="split_day or split_strat")
    ap.add_argument("--out-dir", type=Path, default=MODELS_DIR, help="Models output dir")
    ap.add_argument("--sample", type=int, default=0, help="If >0, sample N per split for quick runs")
    ap.add_argument("--baseline-only", action="store_true", help="Train only LogReg + RF (no XGBoost)")
    args = ap.parse_args()

    split_col = SPLIT_COL_DAY if args.split == "day" else SPLIT_COL_STRAT
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        raise SystemExit(f"Missing {DATA_FILE}. Run: make data")

    df = pd.read_parquet(DATA_FILE)
    for s in ["train", "val", "test"]:
        cnt = (df[split_col] == s).sum()
        if cnt == 0:
            raise SystemExit(f"No '{s}' split in {split_col}. Check prepare_data.")

    if args.sample > 0:
        parts = []
        for s in ["train", "val", "test"]:
            sub = df[df[split_col] == s]
            n = min(args.sample, len(sub))
            parts.append(sub.sample(n=n, random_state=RNG))
        df = pd.concat(parts, ignore_index=True)
        print(f"[sample] {args.sample} per split -> {len(df)} rows")

    Xt, yt, Xv, yv, Xs, ys, le, feat = load_splits(df, split_col)
    n_classes = len(le.classes_)
    print(f"[load] train={len(yt)} val={len(yv)} test={len(ys)} features={len(feat)} classes={n_classes}")

    scaler = StandardScaler()
    Xt_s = scaler.fit_transform(Xt)

    models = []
    # Baseline
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="saga", max_iter=2000, tol=1e-3,
            class_weight="balanced", random_state=RNG,
        )),
    ])
    lr.fit(Xt, yt_)
    models.append(("LogReg", lr))

    # Stronger 1: RF
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=24, max_features="sqrt",
        class_weight="balanced_subsample", random_state=RNG, n_jobs=-1,
    )
    rf.fit(Xt_s, yt_)
    models.append(("RandomForest", rf))

    # Stronger 2: XGBoost (skip with --baseline-only)
    if not args.baseline_only:
        xgb_clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=RNG, n_jobs=-1,
        )
        # XGBoost sklearn wrapper requires classes to be 0..K-1 contiguous.
        # With day-based split, some global classes may be missing in train, so yt_ can have gaps.
        xgb_classes = np.sort(np.unique(yt_))
        yt_xgb = np.searchsorted(xgb_classes, yt_)
        xgb_clf.fit(Xt_s, yt_xgb, verbose=False)
        (out / 'xgb_classes.json').write_text(json.dumps(xgb_classes.tolist()), encoding='utf-8')
        models.append(("XGBoost", xgb_clf))

    # Save artifacts
    joblib.dump(le, out / "label_encoder.joblib")
    joblib.dump(scaler, out / "scaler.joblib")
    with open(out / "feature_names.json", "w") as f:
        json.dump(feat, f, indent=2)
    meta = {"split_col": split_col, "n_classes": n_classes, "feature_names": feat}
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    for name, m in models:
        path = out / f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(m, path)
        print(f"[save] {path}")

    print("[train] done.")


if __name__ == "__main__":
    main()