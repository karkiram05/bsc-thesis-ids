from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw/cicids2017")
OUT_DIR = Path("data/processed/cicids2017")
OUT_DATA = OUT_DIR / "all_clean.parquet"
OUT_META = OUT_DIR / "meta.md"

RNG = 42


def infer_day_from_filename(name: str) -> str:
    # Example: Benign-Monday-no-metadata.parquet
    for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        if re.search(rf"\b{d}\b", name):
            return d
    raise ValueError(f"Could not infer day from filename: {name}")


def main() -> None:
    if not RAW_DIR.exists():
        raise SystemExit(f"Missing raw dir: {RAW_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files found in {RAW_DIR}")

    dfs: list[pd.DataFrame] = []
    for f in files:
        print(f"[read] {f.name}")
        df = pd.read_parquet(f)

        # standardize column names
        df.columns = [c.strip() for c in df.columns]

        if "Label" not in df.columns:
            raise SystemExit(f"Missing Label column in {f.name}")

        df["Label"] = df["Label"].astype(str).str.strip()
        df["source_file"] = f.name
        df["day"] = infer_day_from_filename(f.name)

        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"[concat] shape={df_all.shape}")

    # Binary target
    df_all["is_attack"] = (df_all["Label"].str.lower() != "benign").astype(int)

    # Clean numeric columns: replace inf with NaN, then drop rows with any NaN
    numeric_cols = df_all.select_dtypes(include=[np.number]).columns
    before = len(df_all)

    if len(numeric_cols) > 0:
        df_all.loc[:, numeric_cols] = df_all.loc[:, numeric_cols].replace(
            [np.inf, -np.inf], np.nan
        )

    df_all = df_all.dropna(axis=0)
    dropped = int(before - len(df_all))
    print(f"[clean] shape={df_all.shape} dropped_rows={dropped}")

    # -----------------------------
    # Splits
    # -----------------------------
    # 1) Day-based split (time/generalization stress-test)
    day_to_split = {
        "Monday": "train",
        "Tuesday": "train",
        "Wednesday": "train",
        "Thursday": "val",
        "Friday": "test",
    }

    df_all["split_day"] = df_all["day"].map(day_to_split)
    if df_all["split_day"].isna().any():
        bad = sorted(
            df_all.loc[df_all["split_day"].isna(), "day"].astype(str).unique().tolist()
        )
        raise SystemExit(f"Unknown day values found: {bad}")

    # Keep legacy `split` as day split (your old meta uses this)
    df_all["split"] = df_all["split_day"]

    # 2) Stratified split (recommended for tuning)
    y = df_all["is_attack"].astype(int)
    idx_all = df_all.index.to_numpy()

    # ratios close to your current run: train~0.62, val~0.16, test~0.22
    test_size = 0.2233
    idx_trainval, idx_test = train_test_split(
        idx_all,
        test_size=test_size,
        random_state=RNG,
        stratify=y,
    )

    y_trainval = y.loc[idx_trainval]
    val_size_of_rest = 0.2023  # 0.1571/(1-0.2233)

    _, idx_val = train_test_split(
        idx_trainval,
        test_size=val_size_of_rest,
        random_state=RNG,
        stratify=y_trainval,
    )

    df_all["split_strat"] = "train"
    df_all.loc[idx_val, "split_strat"] = "val"
    df_all.loc[idx_test, "split_strat"] = "test"

    # Save
    df_all.to_parquet(OUT_DATA, index=False)
    print(f"[save] {OUT_DATA}")

    # Meta
    non_feature = {
        "Label",
        "is_attack",
        "day",
        "source_file",
        "split",
        "split_day",
        "split_strat",
    }
    feat_cols = [c for c in df_all.columns if c not in non_feature]

    meta_lines: list[str] = []
    meta_lines.append("# Processed CICIDS2017 Dataset\n\n")
    meta_lines.append(f"\n- raw files: {len(files)}\n\n")
    meta_lines.append(f"- cleaned rows: {len(df_all)}\n\n")
    meta_lines.append(f"- dropped rows (NaN/inf): {dropped}\n\n")
    meta_lines.append(f"- features: {len(feat_cols)}\n\n")

    meta_lines.append("\n## Split counts\n\n")

    meta_lines.append("### split_day (time split)\n\n")
    meta_lines.append(df_all["split_day"].value_counts().to_string())
    meta_lines.append("\n\n")

    meta_lines.append("### split_strat (stratified)\n\n")
    meta_lines.append(df_all["split_strat"].value_counts().to_string())
    meta_lines.append("\n\n")

    meta_lines.append("\n## Attack vs benign counts\n\n")
    meta_lines.append(df_all["is_attack"].value_counts().to_string())
    meta_lines.append("\n\n")

    meta_lines.append("\n## Top labels\n\n")
    meta_lines.append(df_all["Label"].value_counts().head(20).to_string())
    meta_lines.append("\n")

    OUT_META.write_text("".join(meta_lines), encoding="utf-8")
    print(f"[save] {OUT_META}")


if __name__ == "__main__":
    main()