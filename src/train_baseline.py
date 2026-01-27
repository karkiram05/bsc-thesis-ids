from __future__ import annotations

from pathlib import Path
import argparse
import time

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
)

DATA = Path("data/processed/cicids2017/all_clean.parquet")
OUT = Path("reports/baselines/baseline_report.md")

# Recommended:
SPLIT_COL = "split_strat"


def load_split(df: pd.DataFrame, split_name: str):
    sub = df[df[SPLIT_COL] == split_name].copy()
    y = sub["is_attack"].astype(int).to_numpy()

    X = sub.drop(
        columns=[
            "Label",
            "is_attack",
            "day",
            "source_file",
            "split",
            "split_day",
            "split_strat",
        ],
        errors="ignore",
    )
    return X, y


def pick_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    best_t = 0.5
    best_f1 = -1.0

    for t in np.linspace(0.05, 0.95, 91):
        pred = (proba >= t).astype(int)
        f1 = f1_score(y_true, pred, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return best_t


def eval_binary(
    name: str,
    y_true: np.ndarray,
    pred: np.ndarray,
    proba: np.ndarray | None,
) -> dict:
    cm = confusion_matrix(y_true, pred)
    macro_f1 = f1_score(y_true, pred, average="macro")

    pr, rc, f1s, sup = precision_recall_fscore_support(
        y_true, pred, labels=[0, 1], zero_division=0
    )

    out = {
        "name": name,
        "macro_f1": float(macro_f1),
        "cm": cm,
        "per_class": {
            "benign(0)": {
                "precision": float(pr[0]),
                "recall": float(rc[0]),
                "f1": float(f1s[0]),
                "support": int(sup[0]),
            },
            "attack(1)": {
                "precision": float(pr[1]),
                "recall": float(rc[1]),
                "f1": float(f1s[1]),
                "support": int(sup[1]),
            },
        },
        "report": classification_report(y_true, pred, digits=4, zero_division=0),
    }

    if proba is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, proba))
        except Exception:
            out["roc_auc"] = float("nan")

        try:
            out["pr_auc"] = float(average_precision_score(y_true, proba))
        except Exception:
            out["pr_auc"] = float("nan")

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        default=str(OUT),
        help="Output markdown report path (default: reports/baselines/baseline_report.md)",
    )
    p.add_argument(
        "--split-col",
        default=SPLIT_COL,
        help="Split column to use (default: split_strat)",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=0,
        help="If >0, sample this many rows per split (train/val/test) for quick runs.",
    )
    p.add_argument(
        "--models",
        default="logreg,rf",
        help="Comma list: logreg,rf (default: logreg,rf)",
    )
    p.add_argument(
        "--rf-trees",
        type=int,
        default=300,
        help="RandomForest n_estimators (default: 300)",
    )
    p.add_argument(
        "--rf-max-depth",
        type=int,
        default=0,
        help="RandomForest max_depth (0 means None)",
    )
    p.add_argument(
        "--rf-max-features",
        default="sqrt",
        help="RandomForest max_features (default: sqrt)",
    )
    return p.parse_args()


def maybe_sample(df: pd.DataFrame, split_name: str, n: int, rng: int = 42) -> pd.DataFrame:
    if n <= 0:
        return df
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=rng)


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)

    if not DATA.exists():
        raise SystemExit(f"Missing processed dataset: {DATA}. Run build_dataset.py first.")

    df = pd.read_parquet(DATA)
    print("[load]", df.shape)

    split_col = args.split_col
    if split_col not in df.columns:
        raise SystemExit(f"Missing split column {split_col}. Re-run build_dataset.py.")

    print("[split counts]\n", df[split_col].value_counts())
    print("[attack ratio by split]")
    for s in ["train", "val", "test"]:
        sub = df[df[split_col] == s]
        if len(sub) == 0:
            continue
        ratio = float(sub["is_attack"].mean())
        print(f"  {s}: n={len(sub)} attack_ratio={ratio:.4f}")

    # optional sampling per split
    if args.sample and args.sample > 0:
        parts = []
        for s in ["train", "val", "test"]:
            sub = df[df[split_col] == s]
            sub = maybe_sample(sub, s, args.sample)
            parts.append(sub)
        df = pd.concat(parts, ignore_index=True)
        print(f"[sample] per split={args.sample} -> df shape={df.shape}")

    # enforce the chosen split col name for helper
    global SPLIT_COL
    SPLIT_COL = split_col

    X_train, y_train = load_split(df, "train")
    X_val, y_val = load_split(df, "val")
    X_test, y_test = load_split(df, "test")

    # Models
    models_requested = {m.strip().lower() for m in args.models.split(",") if m.strip()}

    models: list[tuple[str, object]] = []

    if "logreg" in models_requested:
        lr = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True)),
                (
                    "clf",
                    LogisticRegression(
                        solver="saga",
                        max_iter=1000,
                        tol=1e-3,
                        class_weight="balanced",
                        verbose=1,
                        random_state=42,
                    ),
                ),
            ]
        )
        models.append(("LogReg", lr))

    if "rf" in models_requested:
        rf = RandomForestClassifier(
            n_estimators=int(args.rf_trees),
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
            max_depth=None if int(args.rf_max_depth) == 0 else int(args.rf_max_depth),
            max_features=args.rf_max_features,
        )
        models.append(("RandomForest", rf))

    if not models:
        raise SystemExit("No models selected. Use --models logreg,rf")

    lines: list[str] = []
    lines.append("# Baseline Results (Binary: Attack vs Benign)\n\n")
    lines.append(f"- split column: `{split_col}`\n")
    lines.append(f"- train: {len(y_train)} rows\n")
    lines.append(f"- val:   {len(y_val)} rows\n")
    lines.append(f"- test:  {len(y_test)} rows\n")
    if args.sample and args.sample > 0:
        lines.append(f"- sampled per split: {args.sample}\n")
    lines.append("\n")

    for name, model in models:
        t0 = time.time()
        print(f"[train] {name} ...")
        model.fit(X_train, y_train)
        t_train = time.time() - t0
        print(f"[train] {name} done in {t_train:.1f}s")

        val_proba = None
        test_proba = None
        if hasattr(model, "predict_proba"):
            val_proba = model.predict_proba(X_val)[:, 1]
            test_proba = model.predict_proba(X_test)[:, 1]

        thr = pick_threshold(y_val, val_proba) if val_proba is not None else 0.5

        val_pred = (val_proba >= thr).astype(int) if val_proba is not None else model.predict(X_val)
        test_pred = (test_proba >= thr).astype(int) if test_proba is not None else model.predict(X_test)

        val_res = eval_binary(name, y_val, val_pred, val_proba)
        test_res = eval_binary(name, y_test, test_pred, test_proba)

        lines.append(f"## {name}\n\n")
        lines.append(f"- threshold (picked on val): {thr:.2f}\n")
        lines.append(f"- train time: {t_train:.1f}s\n\n")

        lines.append("### Validation\n")
        lines.append(f"- macro F1: {val_res['macro_f1']:.4f}\n")
        lines.append(f"- attack recall: {val_res['per_class']['attack(1)']['recall']:.4f}\n")
        lines.append(f"- ROC-AUC: {val_res.get('roc_auc', float('nan')):.4f}\n")
        lines.append(f"- PR-AUC:  {val_res.get('pr_auc', float('nan')):.4f}\n")
        lines.append("Confusion matrix (rows=true, cols=pred):\n")
        lines.append(f"```\n{val_res['cm']}\n```\n")
        lines.append("Classification report:\n")
        lines.append(f"```\n{val_res['report']}\n```\n")

        lines.append("### Test\n")
        lines.append(f"- macro F1: {test_res['macro_f1']:.4f}\n")
        lines.append(f"- attack recall: {test_res['per_class']['attack(1)']['recall']:.4f}\n")
        lines.append(f"- ROC-AUC: {test_res.get('roc_auc', float('nan')):.4f}\n")
        lines.append(f"- PR-AUC:  {test_res.get('pr_auc', float('nan')):.4f}\n")
        lines.append("Confusion matrix (rows=true, cols=pred):\n")
        lines.append(f"```\n{test_res['cm']}\n```\n")
        lines.append("Classification report:\n")
        lines.append(f"```\n{test_res['report']}\n```\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()