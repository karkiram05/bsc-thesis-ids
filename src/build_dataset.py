from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


RAW_DIR = Path("data/raw/cicids2017")
OUT_DIR = Path("data/processed/cicids2017")

OUT_ALL = OUT_DIR / "all_clean.parquet"
OUT_META = OUT_DIR / "meta.md"

# day-based split:
# train: Mon+Tue+Wed, val: Thu, test: Fri
DAY_SPLIT = {
    "train": ["Monday", "Tuesday", "Wednesday"],
    "val": ["Thursday"],
    "test": ["Friday"],
}


def infer_day_from_filename(name: str) -> str:
    # your file names look like: Benign-Monday-no-metadata.parquet
    # we find the weekday token
    for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        if d in name:
            return d
    return "Unknown"


def read_one(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p)
    df.columns = [c.strip() for c in df.columns]

    if "Label" not in df.columns:
        raise ValueError(f"Missing Label column in {p.name}")

    df["source_file"] = p.name
    df["day"] = infer_day_from_filename(p.name)
    return df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # replace inf -> NaN, then drop NaN rows
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(axis=0)
    after = len(df)

    # enforce numeric features only (Label/day/source_file kept separate)
    y = df["Label"].astype(str)
    day = df["day"].astype(str)
    src = df["source_file"].astype(str)

    # all columns except Label/day/source_file
    feature_cols = [c for c in df.columns if c not in ["Label", "day", "source_file"]]
    X = df[feature_cols]

    # keep only numeric columns (safety)
    X = X.select_dtypes(include=[np.number])

    out = X.copy()
    out["Label"] = y.values
    out["is_attack"] = (y != "Benign").astype(np.int8).values
    out["day"] = day.values
    out["source_file"] = src.values

    dropped = before - after
    return out, dropped


def assign_split(day: str) -> str:
    for split, days in DAY_SPLIT.items():
        if day in days:
            return split
    return "unknown"


def main() -> None:
    files = sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files found in {RAW_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    parts = []
    for p in files:
        print(f"[read] {p.name}")
        parts.append(read_one(p))

    df = pd.concat(parts, ignore_index=True)
    print(f"[concat] shape={df.shape}")

    cleaned, dropped = clean_df(df)
    print(f"[clean] shape={cleaned.shape} dropped_rows={dropped}")

    # create split column
    cleaned["split"] = cleaned["day"].apply(assign_split)

    # sanity check: no unknown days
    if (cleaned["split"] == "unknown").any():
        unknown_days = cleaned.loc[cleaned["split"] == "unknown", "day"].value_counts()
        raise SystemExit(f"Unknown day(s) found:\n{unknown_days}")

    # save
    cleaned.to_parquet(OUT_ALL, index=False)
    print(f"[save] {OUT_ALL}")

    # write meta report
    lines = []
    lines.append("# Processed CICIDS2017 Dataset\n\n")
    lines.append(f"- raw files: {len(files)}\n")
    lines.append(f"- cleaned rows: {len(cleaned)}\n")
    lines.append(f"- dropped rows (NaN/inf): {dropped}\n")
    lines.append(f"- features: {cleaned.drop(columns=['Label','is_attack','day','source_file','split']).shape[1]}\n\n")

    lines.append("## Split counts\n")
    lines.append(cleaned["split"].value_counts().to_string() + "\n\n")

    lines.append("## Attack vs benign counts\n")
    lines.append(cleaned["is_attack"].value_counts().to_string() + "\n\n")

    lines.append("## Top labels\n")
    lines.append(cleaned["Label"].value_counts().head(20).to_string() + "\n\n")

    OUT_META.write_text("\n".join(lines), encoding="utf-8")
    print(f"[save] {OUT_META}")


if __name__ == "__main__":
    main()
