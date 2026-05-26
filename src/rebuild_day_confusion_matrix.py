"""Rebuild the day-split confusion matrix WITHOUT the eval.py
__unseen__ aggregation.

The existing reports/metrics_day/confusion_matrix_random_forest.csv
aggregates Bot, DDoS, and PortScan into a single __unseen__ row,
which obscures the per-class collapse pattern. This script retrains
a random forest on Mon-Wed (with the global label encoder so
unseen classes keep their identity), predicts on Friday, and
writes a per-class confusion matrix to:
    reports/metrics_day/confusion_matrix_random_forest_perclass.csv

Used by src.regenerate_thesis_figures to produce a readable
right-panel of the confusion-matrix figure.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

from src.config import DATA_FILE, REPORTS_DIR, RNG, feature_cols

OUT = REPORTS_DIR / "metrics_day"


def main() -> None:
    if not DATA_FILE.exists():
        raise SystemExit(f"Missing {DATA_FILE}")
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_FILE)
    feat = feature_cols(df)

    le = LabelEncoder().fit(df["attack_type"].astype(str))
    df = df.copy()
    df["_y"] = le.transform(df["attack_type"].astype(str))

    train_df = df[df["day"].isin(["Monday", "Tuesday", "Wednesday"])]
    test_df = df[df["day"] == "Friday"]

    # Subsample training for speed; full prediction on Friday
    train_df = (
        train_df.groupby("attack_type", group_keys=False)
        .apply(lambda g: g.sample(frac=0.25, random_state=RNG)
               if len(g) > 1 else g)
    )

    X_train = train_df[feat]
    y_train = train_df["_y"].to_numpy()
    X_test = test_df[feat]
    y_test = test_df["_y"].to_numpy()

    train_classes = sorted(set(y_train.tolist()))
    g2l = {g: i for i, g in enumerate(train_classes)}
    l2g = {i: g for g, i in g2l.items()}
    y_train_local = np.array([g2l[v] for v in y_train])

    print(f"Train rows={len(y_train)}, test rows={len(y_test)}, "
          f"train_classes={len(train_classes)}")

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=24, max_features="sqrt",
        class_weight="balanced_subsample", random_state=RNG, n_jobs=-1,
    )
    rf.fit(X_train, y_train_local)
    pred_local = rf.predict(X_test)
    pred = np.array([l2g[int(p)] for p in pred_local])

    # Confusion matrix over union of (test classes ∪ train classes)
    all_classes_int = sorted(set(y_test.tolist()) | set(train_classes))
    cm = confusion_matrix(y_test, pred, labels=all_classes_int)
    class_names = [le.classes_[c] for c in all_classes_int]
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    out = OUT / "confusion_matrix_random_forest_perclass.csv"
    cm_df.to_csv(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
