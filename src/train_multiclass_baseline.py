"""Multi-class baseline: LogReg + RF only. Writes reports/baselines/multiclass_report.md."""

from __future__ import annotations

import argparse
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

    le = LabelEncoder()
    all_labels = pd.concat([yt, yv, ys], ignore_index=True)
    le.fit(all_labels)
    yt_ = le.transform(yt)
    yv_ = le.transform(yv)
    ys_ = le.transform(ys)
    n_classes = len(le.classes_)
    class_names = list(le.classes_)

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
            "scaled",
        ),
    ]

    lines = [
        "# Multi-class baseline report\n\n",
        f"- **Split**: `{split_col}`\n",
        f"- **Train** {len(yt)}, **Val** {len(yv)}, **Test** {len(ys)}\n",
        f"- **Classes** {n_classes}: " + ", ".join(class_names) + "\n\n",
        "---\n\n",
    ]

    for name, model, inp in models:
        if inp == "raw":
            model.fit(Xt, yt_)
            X_eval = Xs
        else:
            model.fit(Xt_s, yt_)
            X_eval = Xs_s

        pred = model.predict(X_eval)
        pr, rc, f1, sup = precision_recall_fscore_support(
            ys_, pred, labels=np.arange(n_classes), zero_division=0
        )
        cm = confusion_matrix(ys_, pred, labels=np.arange(n_classes))
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
                ys_, pred, target_names=class_names, digits=4, zero_division=0
            )
        )
        lines.append("\n```\n\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(lines), encoding="utf-8")
    print(f"[multiclass baseline] wrote {args.out}")


if __name__ == "__main__":
    main()
