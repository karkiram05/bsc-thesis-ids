from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    UNSW_RAW_DIR,
    UNSW_PROCESSED_DIR,
    UNSW_DATA_FILE,
    UNSW_META_FILE,
    UNSW_NON_FEATURE,
    UNSW_CATEGORICAL,
    RNG,
)


def main() -> None:
    if not UNSW_RAW_DIR.exists():
        raise SystemExit(f"Missing raw dir: {UNSW_RAW_DIR}. Place UNSW-NB15 CSV files there.")

    UNSW_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load pre-split CSV files
    train_csv = UNSW_RAW_DIR / "UNSW_NB15_training-set.csv"
    test_csv = UNSW_RAW_DIR / "UNSW_NB15_testing-set.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise SystemExit(
            f"Missing UNSW_NB15_training-set.csv or UNSW_NB15_testing-set.csv in {UNSW_RAW_DIR}."
        )

    df_train = pd.read_csv(train_csv, encoding="utf-8")
    df_test = pd.read_csv(test_csv, encoding="utf-8")

    print(f"[read] training-set: {df_train.shape}, testing-set: {df_test.shape}")

    # Clean column names
    df_train.columns = [c.strip() for c in df_train.columns]
    df_test.columns = [c.strip() for c in df_test.columns]

    # Clean attack_cat: strip whitespace, normalise empty/NaN to "Normal"
    for sub in (df_train, df_test):
        sub["attack_cat"] = sub["attack_cat"].fillna("Normal").astype(str).str.strip()
        sub.loc[sub["attack_cat"] == "", "attack_cat"] = "Normal"
        sub["attack_cat"] = sub["attack_cat"].str.strip()

    # Show label distribution
    print("[labels] training-set:")
    print(df_train["attack_cat"].value_counts().to_string())
    print("[labels] testing-set:")
    print(df_test["attack_cat"].value_counts().to_string())

    # Splits: carve val from training, keep original test
    df_train["split"] = "train"
    df_test["split"] = "test"

    # Stratified val split from training set
    train_idx = df_train.index.to_numpy()
    y_strat = df_train["attack_cat"]
    _, val_idx = train_test_split(
        train_idx, test_size=0.2, random_state=RNG, stratify=y_strat
    )
    df_train.loc[val_idx, "split"] = "val"

    print(f"[split] train={int((df_train['split'] == 'train').sum())} "
          f"val={int((df_train['split'] == 'val').sum())} "
          f"test={len(df_test)}")

    # Combine
    df = pd.concat([df_train, df_test], ignore_index=True)

    # Drop id column
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Clean Inf/NaN
    numeric = df.select_dtypes(include=[np.number]).columns
    before = len(df)
    if len(numeric):
        df.loc[:, numeric] = df.loc[:, numeric].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0)
    dropped_nan = int(before - len(df))
    print(f"[clean] dropped {dropped_nan} NaN/inf rows, shape={df.shape}")

    # One-hot encode categorical columns using TRAIN vocabulary only.
    # If test (or val) has a category not seen in train, reindex drops it:
    # that row ends up with all-zero dummies for the unseen value, which is
    # the safe fallback (the model never saw positive signal for it anyway).
    # Fitting on train+test together would otherwise give the model a feature
    # column that is always 0 in training - test-dependent and undesirable.
    train_mask = df["split"] == "train"
    for col in UNSW_CATEGORICAL:
        if col in df.columns:
            train_vocab = pd.get_dummies(df.loc[train_mask, col], prefix=col, dtype=int).columns
            all_dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
            all_dummies = all_dummies.reindex(columns=train_vocab, fill_value=0)
            df = pd.concat([df.drop(columns=[col]), all_dummies], axis=1)
    print(f"[encode] after one-hot encoding (train-only vocab): {df.shape}")

    # Save
    df.to_parquet(UNSW_DATA_FILE, index=False)
    print(f"[save] {UNSW_DATA_FILE}")

    # Feature list
    feat = [c for c in df.columns if c not in UNSW_NON_FEATURE]
    meta_lines = [
        "# Processed UNSW-NB15\n\n",
        f"- rows: {len(df)}\n",
        f"- dropped NaN/inf: {dropped_nan}\n",
        f"- features: {len(feat)}\n",
        f"- categorical encoded: {UNSW_CATEGORICAL}\n\n",
        "## Split\n\n",
        df["split"].value_counts().to_string() + "\n\n",
        "## Attack categories\n\n",
        df["attack_cat"].value_counts().to_string() + "\n",
    ]
    UNSW_META_FILE.write_text("".join(meta_lines), encoding="utf-8")
    print(f"[save] {UNSW_META_FILE}")


if __name__ == "__main__":
    main()
