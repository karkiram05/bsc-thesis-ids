from __future__ import annotations

from pathlib import Path
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

SPLIT_COL = "split_strat"


def load_split(df: pd.DataFrame, split_name: str):
    sub = df[df[SPLIT_COL] == split_name].copy()
    y = sub["is_attack"].astype(int).to_numpy()

    # DO NOT let the model see row_hash (it can cheat)
    drop_cols = [
        "Label",
        "is_attack",
        "day",
        "source_file",
        "split",
        "split_day",
        "split_strat",
        "row_hash",
    ]
    X = sub.drop(columns=drop_cols, errors="ignore")
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


def eval_binary(name: str, y_true: np.ndarray, pred: np.ndarray, proba: np.ndarray | None) -> dict:
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


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"Missing processed dataset: {DATA}. Run build_dataset.py first.")

    df = pd.read_parquet(DATA)
    print("[load]", df.shape)

    if SPLIT_COL not in df.columns:
        raise SystemExit(f"Missing split column {SPLIT_COL}. Re-run build_dataset.py.")

    print("[split counts]\n", df[SPLIT_COL].value_counts())
    print("[attack ratio by split]")
    for s in ["train", "val", "test"]:
        sub = df[df[SPLIT_COL] == s]
        if len(sub) == 0:
            continue
        ratio = float(sub["is_attack"].mean())
        print(f"  {s}: n={len(sub)} attack_ratio={ratio:.4f}")

    X_train, y_train = load_split(df, "train")
    X_val, y_val = load_split(df, "val")
    X_test, y_test = load_split(df, "test")

    # Logistic Regression baseline
    lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="saga",
                    max_iter=2000,
                    class_weight="balanced",
                    verbose=0,
                ),
            ),
        ]
    )

    # Random Forest baseline
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    models = [
        ("LogReg", lr),
        ("RandomForest", rf),
    ]

    lines: list[str] = []
    lines.append("# Baseline Results (Binary: Attack vs Benign)\n\n")
    lines.append(f"- split column: `{SPLIT_COL}`\n")
    lines.append(f"- train: {len(y_train)} rows\n")
    lines.append(f"- val:   {len(y_val)} rows\n")
    lines.append(f"- test:  {len(y_test)} rows\n\n")

    for name, model in models:
        print(f"[train] {name} ...")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_s = time.time() - t0
        print(f"[train] {name} done in {train_s:.1f}s")

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
        lines.append(f"- train time: {train_s:.1f}s\n\n")

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

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(lines), encoding="utf-8")
    print(f"[save] {OUT}")


if __name__ == "__main__":
    main()