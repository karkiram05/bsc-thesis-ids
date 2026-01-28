from __future__ import annotations

from pathlib import Path
import re
import time

import numpy as np
import pandas as pd


RAW_DIR = Path("data/raw/cicids2017")
OUT_DIR = Path("data/processed/cicids2017")
OUT_DATA = OUT_DIR / "all_clean.parquet"
OUT_META = OUT_DIR / "meta.md"

SEED = 42

# columns that are NOT features
META_COLS = {
    "Label",
    "is_attack",
    "day",
    "source_file",
    "split",
    "split_day",
    "split_strat",
    "row_hash",
}

DAY_WORDS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def infer_day_from_name(name: str) -> str:
    for d in DAY_WORDS:
        if d.lower() in name.lower():
            return d
    return "Unknown"


def normalize_label(s: pd.Series) -> pd.Series:
    # fix the weird replacement char (�) often seen in CICIDS2017 exports
    # example: "Web Attack � Brute Force"
    s = s.astype(str)
    s = s.str.replace("\uFFFD", "-", regex=False)  # unicode replacement char
    s = s.str.replace("�", "-", regex=False)       # common display replacement
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.str.strip()
    return s


def read_all_parquets(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files found in {raw_dir}")

    dfs: list[pd.DataFrame] = []
    for p in files:
        print(f"[read] {p.name}")
        df = pd.read_parquet(p)

        # standardize column names
        df.columns = [c.strip() for c in df.columns]

        # attach provenance
        df["source_file"] = p.name
        df["day"] = infer_day_from_name(p.stem)

        if "Label" not in df.columns:
            raise SystemExit(f"Missing Label column in {p.name}")

        df["Label"] = normalize_label(df["Label"])

        # binary target
        df["is_attack"] = (df["Label"].str.lower() != "benign").astype(int)

        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    print(f"[concat] shape={out.shape}")
    return out


def drop_nan_inf(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)

    # drop any NaN anywhere
    df2 = df.dropna(axis=0, how="any").copy()

    # drop rows containing inf in numeric columns only
    num = df2.select_dtypes(include=[np.number])
    if num.shape[1] > 0:
        inf_mask = np.isinf(num.to_numpy()).any(axis=1)
        if inf_mask.any():
            df2 = df2.loc[~inf_mask].copy()

    dropped = before - len(df2)
    return df2, int(dropped)


def add_row_hash(df: pd.DataFrame) -> pd.DataFrame:
    # hash ONLY feature columns, exclude meta columns if they exist
    drop_cols = [c for c in META_COLS if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")

    # important: stable hashing over the feature dataframe
    h = pd.util.hash_pandas_object(X, index=False).astype("uint64")

    df2 = df.copy()
    df2["row_hash"] = h
    return df2


def remove_conflicting_hashes(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    # conflicting = same row_hash has both labels 0 and 1
    nunq = df.groupby("row_hash")["is_attack"].nunique()
    conflict_hashes = set(nunq[nunq > 1].index)

    rows_before = len(df)
    df2 = df.loc[~df["row_hash"].isin(conflict_hashes)].copy()
    rows_removed = rows_before - len(df2)

    return df2, int(len(conflict_hashes)), int(rows_removed)


def split_by_hash_stratified(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Group split by row_hash, stratified by is_attack at the hash level.
    Output in df['split_strat'] with train/val/test.

    target ratio:
      train ~ 62%
      val   ~ 16%
      test  ~ 22%
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    g = df.groupby("row_hash")["is_attack"].max().reset_index()
    hashes = g["row_hash"].to_numpy()
    y = g["is_attack"].to_numpy()

    # first split: train vs temp
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.38, random_state=seed)
    train_idx, temp_idx = next(sss1.split(hashes, y))

    train_hash = set(hashes[train_idx])
    temp_hashes = hashes[temp_idx]
    temp_y = y[temp_idx]

    # second split: temp -> val/test
    # choose test bigger than val, similar to your earlier numbers
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.57, random_state=seed)
    val_idx, test_idx = next(sss2.split(temp_hashes, temp_y))

    val_hash = set(temp_hashes[val_idx])
    test_hash = set(temp_hashes[test_idx])

    def assign(h: int) -> str:
        if h in train_hash:
            return "train"
        if h in val_hash:
            return "val"
        return "test"

    df2 = df.copy()
    df2["split_strat"] = df2["row_hash"].map(assign)
    return df2


def split_random_rowwise(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    # simple row-wise split (NOT safe for leakage, but keep for reference)
    rng = np.random.default_rng(seed)
    r = rng.random(len(df))

    df2 = df.copy()
    df2["split"] = np.where(r < 0.62, "train", np.where(r < 0.78, "val", "test"))
    return df2


def split_by_day(df: pd.DataFrame) -> pd.DataFrame:
    # simple day-based split (you can change later)
    # train: Monday-Thursday, test: Friday, val: Unknown/others
    train_days = {"Monday", "Tuesday", "Wednesday", "Thursday"}
    test_days = {"Friday"}

    def assign(d: str) -> str:
        if d in train_days:
            return "train"
        if d in test_days:
            return "test"
        return "val"

    df2 = df.copy()
    df2["split_day"] = df2["day"].map(assign)
    return df2


def count_features(df: pd.DataFrame) -> int:
    cols = [c for c in df.columns if c not in META_COLS]
    return int(len(cols))


def leakage_check(df: pd.DataFrame) -> tuple[int, int]:
    # check if same feature row exists in multiple split_strat splits
    drop_cols = [c for c in META_COLS if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")

    h = pd.util.hash_pandas_object(X, index=False).astype("uint64")
    tmp = pd.DataFrame({"h": h, "split": df["split_strat"]})

    g = tmp.groupby("h")["split"].nunique()
    leaky = int((g > 1).sum())
    total = int(g.shape[0])
    return leaky, total


def write_meta(
    df: pd.DataFrame,
    raw_files: int,
    dropped_nan_inf: int,
    conflict_hashes: int,
    conflict_rows_removed: int,
    leak_in_splits: int,
    leak_total_unique: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Processed CICIDS2017 Dataset\n\n")

    lines.append(f"- raw files: {raw_files}\n")
    lines.append(f"- cleaned rows: {len(df)}\n")
    lines.append(f"- dropped rows (NaN/inf): {dropped_nan_inf}\n")
    lines.append(f"- conflicting feature-hashes removed: {conflict_hashes}\n")
    lines.append(f"- rows removed due to conflicts: {conflict_rows_removed}\n")
    lines.append(f"- features: {count_features(df)}\n\n")

    lines.append("## Split counts (split_strat)\n\n")
    lines.append(df["split_strat"].value_counts().to_string())
    lines.append("\n\n")

    lines.append("## Attack vs benign counts\n\n")
    lines.append(df["is_attack"].value_counts().to_string())
    lines.append("\n\n")

    lines.append("## Top labels\n\n")
    lines.append(df["Label"].value_counts().head(25).to_string())
    lines.append("\n\n")

    lines.append("## Leakage check (split_strat)\n\n")
    lines.append(f"- unique feature rows: {leak_total_unique}\n")
    lines.append(f"- rows appearing in >1 split: {leak_in_splits}\n")
    if leak_total_unique > 0:
        lines.append(f"- leak ratio: {leak_in_splits / leak_total_unique:.6f}\n")
    lines.append("\n")

    OUT_META.write_text("".join(lines), encoding="utf-8")
    print(f"[save] {OUT_META}")


def main() -> None:
    t0 = time.time()

    if not RAW_DIR.exists():
        raise SystemExit(f"Missing raw dir: {RAW_DIR}")

    raw_files = len(list(RAW_DIR.glob("*.parquet")))
    df = read_all_parquets(RAW_DIR)

    df, dropped_nan_inf = drop_nan_inf(df)
    # after this, you have Label, is_attack, day, source_file

    # add row hash and remove conflicting feature rows
    df = add_row_hash(df)
    df, n_conflict_hashes, n_conflict_rows = remove_conflicting_hashes(df)
    print(f"[conflicts] hashes={n_conflict_hashes} removed_rows={n_conflict_rows}")

    # splits
    df = split_random_rowwise(df, seed=SEED)
    df = split_by_day(df)
    df = split_by_hash_stratified(df, seed=SEED)

    # leakage check on final dataset
    leak_in_splits, leak_total_unique = leakage_check(df)
    print(f"[leak-check] unique_rows={leak_total_unique} rows_in_>1_split={leak_in_splits}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DATA, index=False)
    print(f"[save] {OUT_DATA}")

    write_meta(
        df=df,
        raw_files=raw_files,
        dropped_nan_inf=dropped_nan_inf,
        conflict_hashes=n_conflict_hashes,
        conflict_rows_removed=n_conflict_rows,
        leak_in_splits=leak_in_splits,
        leak_total_unique=leak_total_unique,
    )

    print(f"[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()