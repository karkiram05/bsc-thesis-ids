from __future__ import annotations

from pathlib import Path
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
)


DATA = Path("data/processed/cicids2017/all_clean.parquet")
OUT = Path("reports/baselines/baseline_report.md")


def load_split(df: pd.DataFrame, split_name: str):
    sub = df[df["split"] == split_name].copy()
    y = sub["is_attack"].astype(int).values
    X = sub.drop(columns=["Label", "is_attack", "day", "source_file", "split"])
    return X, y


def eval_binary(name: str, model, X, y) -> dict:
    pred = model.predict(X)
    f1 = f1_score(y, pred, average="macro")
    cm = confusion_matrix(y, pred)

    pr, rc, f1s, sup = precision_recall_fscore_support(
        y, pred, labels=[0, 1], zero_division=0
    )

    # class 1 recall is the important one (missed attacks)
    return {
        "name": name,
        "macro_f1": float(f1),
        "cm": cm,
        "per_class": {
            "benign(0)": {"precision": float(pr[0]), "recall": float(rc[0]), "f1": float(f1s[0]), "support": int(sup[0])},
            "attack(1)": {"precision": float(pr[1]), "recall": float(rc[1]), "f1": float(f1s[1]), "support": int(sup[1])},
        },
        "report": classification_report(y, pred, digits=4, zero_division=0),
    }


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"Missing processed dataset: {DATA}. Run build_dataset.py first.")

    df = pd.read_parquet(DATA)
    print("[load]", df.shape)

    X_train, y_train = load_split(df, "train")
    X_val, y_val = load_split(df, "val")
    X_test, y_test = load_split(df, "test")

    # Baseline 1: Logistic Regression (scaled)
    lr = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),  # safe for large sparse-like; still ok
            ("clf", LogisticRegression(max_iter=200, n_jobs=None)),
        ]
    )

    # Baseline 2: Random Forest (no scaling needed)
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    models = [
        ("LogReg", lr),
        ("RandomForest", rf),
    ]

    lines = []
    lines.append("# Baseline Results (Binary: Attack vs Benign)\n\n")
    lines.append(f"- train: {len(y_train)} rows\n")
    lines.append(f"- val:   {len(y_val)} rows\n")
    lines.append(f"- test:  {len(y_test)} rows\n\n")

    for name, model in models:
        print(f"[train] {name}")
        model.fit(X_train, y_train)

        val_res = eval_binary(name, model, X_val, y_val)
        test_res = eval_binary(name, model, X_test, y_test)

        lines.append(f"## {name}\n\n")

        lines.append("### Validation\n")
        lines.append(f"- macro F1: {val_res['macro_f1']:.4f}\n")
        lines.append(f"- attack recall: {val_res['per_class']['attack(1)']['recall']:.4f}\n")
        lines.append("Confusion matrix (rows=true, cols=pred):\n")
        lines.append(f"```\n{val_res['cm']}\n```\n")
        lines.append("Classification report:\n")
        lines.append(f"```\n{val_res['report']}\n```\n")

        lines.append("### Test\n")
        lines.append(f"- macro F1: {test_res['macro_f1']:.4f}\n")
        lines.append(f"- attack recall: {test_res['per_class']['attack(1)']['recall']:.4f}\n")
        lines.append("Confusion matrix (rows=true, cols=pred):\n")
        lines.append(f"```\n{test_res['cm']}\n```\n")
        lines.append("Classification report:\n")
        lines.append(f"```\n{test_res['report']}\n```\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(lines), encoding="utf-8")
    print(f"[save] {OUT}")


if __name__ == "__main__":
    main()
