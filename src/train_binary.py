"""Train binary IDS models: Benign vs Attack.

This exists because CICIDS2017 "day split" has non-overlapping attack labels across days,
so multiclass temporal evaluation is not meaningful. Binary remains meaningful.

Outputs:
  <out-dir>/
    - meta.json
    - feature_names.json
    - scaler.joblib
    - logreg.joblib
    - random_forest.joblib
    - xgboost.joblib   (unless --baseline-only)
"""

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
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

from src.config import DATA_FILE, MODELS_DIR, NON_FEATURE, RNG, SPLIT_COL_DAY, SPLIT_COL_STRAT


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE and c in df.columns]


def _make_binary_y(attack_type: pd.Series) -> np.ndarray:
    # 0 = Benign, 1 = Attack
    return (attack_type.astype(str) != "Benign").astype(int).to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["day", "strat"], default="day")
    ap.add_argument("--out-dir", type=Path, default=Path(MODELS_DIR) / "binary")
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

    feat = _feature_cols(df)
    assert feat, "No feature columns found"

    def part(s: str):
        sub = df.loc[df[split_col] == s]
        X = sub[feat].copy()
        y = _make_binary_y(sub["attack_type"])
        return X, y

    Xt, yt = part("train")
    Xv, yv = part("val")
    Xs, ys = part("test")

    print(
        f"[load] train={len(yt)} val={len(yv)} test={len(ys)} features={len(feat)} "
        f"pos_rate_train={yt.mean():.4f}"
    )

    # For RF/XGB we feed scaled features to be consistent with your multiclass pipeline.
    scaler = StandardScaler()
    Xt_s = scaler.fit_transform(Xt)

    models: list[tuple[str, object]] = []

    # LogReg (pipeline with its own scaler)
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="saga",
            max_iter=2000,
            tol=1e-3,
            class_weight="balanced",
            random_state=RNG,
        )),
    ])
    lr.fit(Xt, yt)
    models.append(("logreg", lr))

    # RandomForest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=24,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RNG,
        n_jobs=-1,
    )
    rf.fit(Xt_s, yt)
    models.append(("random_forest", rf))

    # XGBoost
    if not args.baseline_only:
        # Helpful for imbalance
        n_pos = max(int(yt.sum()), 1)
        n_neg = max(int((yt == 0).sum()), 1)
        scale_pos_weight = n_neg / n_pos

        xgb_clf = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=1,
            eval_metric="logloss",
            random_state=RNG,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
        )
        xgb_clf.fit(Xt_s, yt, verbose=False)
        models.append(("xgboost", xgb_clf))

    # Save artifacts
    joblib.dump(scaler, out / "scaler.joblib")
    (out / "feature_names.json").write_text(json.dumps(feat, indent=2), encoding="utf-8")

    meta = {
        "task": "binary",
        "split_col": split_col,
        "feature_names": feat,
        "classes": ["Benign", "Attack"],  # 0, 1
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for name, m in models:
        path = out / f"{name}.joblib"
        joblib.dump(m, path)
        print(f"[save] {path}")

    print("[train-binary] done.")


if __name__ == "__main__":
    main()