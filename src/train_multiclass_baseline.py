"""DEPRECATED: Multi-class baseline: LogReg + RF only.

Use src.train instead (which handles LabelEncoder correctly, supports XGBoost,
and properly handles unseen classes in the day split).

FIX applied: LabelEncoder now fitted on train labels only (was train+val+test).
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from src.config import DATA_FILE, NON_FEATURE, RNG, BASELINES_DIR


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE and c in df.columns]


def main() -> None:
    warnings.warn(
        "train_multiclass_baseline.py is DEPRECATED. Use src.train instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    ap = argparse.ArgumentParser(description="Train multi-class baseline (LogReg + RF), write multiclass_report.md.")
    ap.add_argument("--split", choices=["day", "strat"], default="strat", help="split_strat (default) or split_day")
    ap.add_argument("--out", type=Path, default=BASELINES_DIR / "multiclass_report.md")
    args = ap.parse_args()

    split_col = "split_strat" if args.split == "strat" else "split_day"
    if not DATA_FILE.exists():
        raise SystemExit(f"Missing {DATA_FILE}. Run: make data")

    df = pd.read_parquet(DATA_FILE)
    feat = _feature_cols(df)
    assert feat, "No feature columns"

    def part(s: str):
        sub = df.loc[df[split_col] == s]
        y = sub["attack_type"].astype(str)
        X = sub[feat].copy()
        return X, y

    Xt, yt = part("train")
    Xv, yv = part("val")
    Xs, ys = part("test")

    # FIX: Fit LabelEncoder on TRAIN labels only.
    # Previously fitted on train+val+test, which leaks test label information.
    le = LabelEncoder()
    le.fit(yt)

    # Handle unseen labels in val/test gracefully
    known = set(le.classes_)
    def safe_encode(labels):
        mapped = []
        for lbl in labels:
            if lbl in known:
                mapped.append(le.transform([lbl])[0])
            else:
                mapped.append(-1)  # unseen class
        return np.array(mapped, dtype=np.int64)

    yt_ = le.transform(yt)
    yv_ = safe_encode(yv)
    ys_ = safe_encode(ys)

    n_classes = len(le.classes_)
    class_names = list(le.classes_)

    # Report unseen classes
    unseen_val = set(yv.unique()) - known
    unseen_test = set(ys.unique()) - known
    if unseen_val:
        print(f"[warn] val has unseen classes: {sorted(unseen_val)}")
    if unseen_test:
        print(f"[warn] test has unseen classes: {sorted(unseen_test)}")

    scaler = StandardScaler()
    Xt_s = scaler.fit_transform(Xt)
    Xs_s = scaler.transform(Xs)

    models = [
        (
            "LogReg",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    solver="saga", max_iter=2000, tol=1e-3,
                    class_weight="balanced", random_state=RNG,
                )),
            ]),
            "raw",
        ),
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=200, max_depth=24, max_features="sqrt",
                class_weight="balanced_subsample", random_state=RNG, n_jobs=-1,
            ),
            "raw",  # FIX: RF doesn't need scaling
        ),
    ]

    lines = [
        "# Multi-class baseline report\n\n",
        f"- **Split**: `{split_col}`\n",
        f"- **Train** {len(yt)}, **Val** {len(yv)}, **Test** {len(ys)}\n",
        f"- **Classes** {n_classes}: " + ", ".join(class_names) + "\n\n",
        "---\n\n",
    ]

    # Filter out unseen-class rows for evaluation
    test_mask = ys_ >= 0
    ys_eval = ys_[test_mask]
    Xs_eval = Xs.iloc[test_mask]

    for name, model, inp in models:
        model.fit(Xt, yt_)
        pred = model.predict(Xs_eval)
        pr, rc, f1, sup = precision_recall_fscore_support(
            ys_eval, pred, labels=np.arange(n_classes), zero_division=0
        )
        cm = confusion_matrix(ys_eval, pred, labels=np.arange(n_classes))
        macro_f1 = float(np.mean(f1))

        lines.append(f"## {name}\n\n")
        lines.append(f"- **Macro F1** (test): {macro_f1:.4f}\n\n")
        lines.append("### Per-class recall (test)\n\n")
        lines.append("| Class | Recall | Support |\n")
        lines.append("|-------|--------|--------|\n")
        for i in range(n_classes):
            lines.append(f"| {class_names[i]} | {rc[i]:.4f} | {int(sup[i])} |\n")
        lines.append("\n")
        lines.append("### Confusion matrix (rows = true, cols = pred)\n\n")
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        lines.append("```\n")
        lines.append(cm_df.to_string())
        lines.append("\n```\n\n")
        lines.append("### Classification report\n\n")
        lines.append("```\n")
        lines.append(
            classification_report(
                ys_eval, pred, target_names=class_names, digits=4, zero_division=0
            )
        )
        lines.append("\n```\n\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(lines), encoding="utf-8")
    print(f"[multiclass baseline] wrote {args.out}")


if __name__ == "__main__":
    main()