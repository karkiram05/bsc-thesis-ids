"""Train baseline (LogReg) and stronger models (RF, XGBoost). Multi-class on attack_type."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.config import (
    DATA_FILE,
    LEAKAGE_COLUMNS,
    MODELS_DIR,
    NON_FEATURE,
    RNG,
    SPLIT_COL_DAY,
    SPLIT_COL_STRAT,
)

def _feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURE]

def _safe_transform(le: LabelEncoder, labels: pd.Series) -> np.ndarray:
    vals = labels.astype(str)
    known = set(le.classes_.tolist())
    y = np.full(len(vals), -1, dtype=np.int64)
    mask = vals.isin(known)
    if mask.any():
        y[mask.to_numpy()] = le.transform(vals[mask])
    return y


def load_splits(df, split_col):
    feat = _feature_cols(df)
    assert feat, "No feature columns found"

    train_labels = df.loc[df[split_col] == "train", "attack_type"].astype(str)
    if len(train_labels) == 0:
        raise SystemExit(f"No training rows found in {split_col}.")

    le = LabelEncoder()
    le.fit(train_labels)

    def part(split_name):
        sub = df.loc[df[split_col] == split_name]
        y = _safe_transform(le, sub["attack_type"])
        n_unseen = int((y == -1).sum())
        if n_unseen > 0:
            unseen = sorted(sub.loc[y == -1, "attack_type"].astype(str).unique().tolist())
            print(
                f"[warn] {split_name}: {n_unseen} rows have labels unseen in training; "
                f"labels={unseen}"
            )
        return sub[feat].copy(), y

    Xt, yt = part("train")
    Xv, yv = part("val")
    Xs, ys = part("test")

    if (yt == -1).any():
        raise SystemExit("Training labels contain unseen-class marker; aborting.")

    return Xt, yt, Xv, yv, Xs, ys, le, feat

def main():
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
    present_leak = [c for c in df.columns if c in LEAKAGE_COLUMNS]
    if present_leak:
        raise SystemExit(
            "Leakage columns are present in processed data. "
            f"Rebuild dataset with src.prepare_data. Found: {present_leak}"
        )
    for s in ["train", "val", "test"]:
        if (df[split_col] == s).sum() == 0:
            raise SystemExit(f"No '{s}' split in {split_col}.")
    if args.sample > 0:
        parts = [df[df[split_col] == s].sample(n=min(args.sample, (df[split_col] == s).sum()), random_state=RNG) for s in ["train","val","test"]]
        df = pd.concat(parts, ignore_index=True)
    Xt, yt, Xv, yv, Xs, ys, le, feat = load_splits(df, split_col)
    n_classes = len(le.classes_)
    print(f"[load] train={len(yt)} val={len(yv)} test={len(ys)} features={len(feat)} classes={n_classes}")

    # Scaler: saved for models that need it.
    # NOTE: RF and XGBoost are tree-based and invariant to monotonic feature
    # transforms.  We still save a scaler so eval.py has a consistent interface,
    # but only LogReg actually requires it (via its internal Pipeline scaler).
    scaler = StandardScaler()
    scaler.fit(Xt)  # Fit only; tree models use raw features, LogReg has internal scaler

    models = []

    # LogReg — uses its own internal scaler pipeline
    print("[train] LogReg ...")
    lr = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(solver="saga", max_iter=2000, tol=1e-3, class_weight="balanced", random_state=RNG))])
    lr.fit(Xt, yt)
    models.append(("logreg", lr))
    print("[train] LogReg done.")

    # RandomForest — tree-based, does NOT need scaling.
    # We feed raw features so feature importances are directly interpretable.
    print("[train] RandomForest ...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=24, max_features="sqrt", class_weight="balanced_subsample", random_state=RNG, n_jobs=-1)
    rf.fit(Xt, yt)  # FIX: use Xt (raw) instead of Xt_s (scaled)
    models.append(("random_forest", rf))
    print("[train] RandomForest done.")

    if not args.baseline_only:
        import xgboost as xgb
        print("[train] XGBoost ...")
        # For day split, training labels may not be contiguous 0..N-1.
        # Remap to contiguous for XGBoost, save mapping for eval to undo.
        train_classes = sorted(set(yt.tolist()))
        remap = {c: i for i, c in enumerate(train_classes)}
        reverse_remap = {i: c for c, i in remap.items()}
        yt_xgb = np.array([remap[v] for v in yt.tolist()], dtype=np.int32)
        # XGBoost is tree-based — scaling is unnecessary, feed raw features.
        xgb_clf = xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
            eval_metric="mlogloss", random_state=RNG, n_jobs=-1, verbosity=0)
        xgb_clf.fit(Xt, yt_xgb)  # FIX: use Xt (raw) instead of Xt_s
        models.append(("xgboost", xgb_clf))
        with open(out / "xgboost_label_remap.json", "w") as f:
            json.dump({"train_classes": train_classes, "n_global_classes": n_classes}, f, indent=2)
        print("[train] XGBoost done.")

        # LightGBM — histogram-based boosting, fast and competitive with XGBoost.
        # Uses the same contiguous label remap as XGBoost.
        import lightgbm as lgb
        print("[train] LightGBM ...")
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            num_leaves=63, min_child_samples=20,
            class_weight="balanced", random_state=RNG, n_jobs=-1, verbose=-1,
        )
        lgb_clf.fit(Xt, yt_xgb)
        models.append(("lightgbm", lgb_clf))
        # LightGBM uses the same label remap as XGBoost
        with open(out / "lightgbm_label_remap.json", "w") as f:
            json.dump({"train_classes": train_classes, "n_global_classes": n_classes}, f, indent=2)
        print("[train] LightGBM done.")

    joblib.dump(le, out / "label_encoder.joblib")
    joblib.dump(scaler, out / "scaler.joblib")
    with open(out / "feature_names.json", "w") as f:
        json.dump(feat, f, indent=2)
    with open(out / "meta.json", "w") as f:
        json.dump({"split_col": split_col, "n_classes": n_classes, "feature_names": feat, "classes": list(le.classes_)}, f, indent=2)
    for name, m in models:
        path = out / f"{name}.joblib"
        joblib.dump(m, path)
        print(f"[save] {path}")
    print("[train] done.")

if __name__ == "__main__":
    main()