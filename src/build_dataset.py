from __future__ import annotations

from pathlib import Path
import json
import time
import pandas as pd
import numpy as np

RAW_DIR = Path("data/raw/cicids2017")
OUT_PARQUET = Path("data/processed/cicids2017/all_clean.parquet")
OUT_META = Path("data/processed/cicids2017/meta.md")

# Adjust if your raw filenames differ
RAW_FILES = [
    "Benign-Monday-no-metadata.parquet",
    "Bruteforce-Tuesday-no-metadata.parquet",
    "DoS-Wednesday-no-metadata.parquet",
    "WebAttacks-Thursday-no-metadata.parquet",
    "Infiltration-Thursday-no-metadata.parquet",
    "Botnet-Friday-no-metadata.parquet",
    "Portscan-Friday-no-metadata.parquet",
    "DDoS-Friday-no-metadata.parquet",
]

NON_FEATURE_COLS = {
    "Label",
    "is_attack",
    "attack_type",
    "day",
    "source_file",
    "split",
    "split_day",
    "split_strat",
}

DAY_FROM_FILE = {
    "Benign-Monday": "Monday",
    "Bruteforce-Tuesday": "Tuesday",
    "DoS-Wednesday": "Wednesday",
    "WebAttacks-Thursday": "Thursday",
    "Infiltration-Thursday": "Thursday",
    "Botnet-Friday": "Friday",
    "Portscan-Friday": "Friday",
    "DDoS-Friday": "Friday",
}


def infer_day_from_source(source_file: str) -> str:
    for k, v in DAY_FROM_FILE.items():
        if k in source_file:
            return v
    return "Unknown"


def infer_attack_type(label: str) -> str:
    # Keep simple and transparent. You can refine later.
    if label is None:
        return "Unknown"
    s = str(label).strip()
    if s.lower() == "benign":
        return "Benign"

    # Many CICIDS labels contain hyphens, spaces, etc.
    # We just return the raw label as attack type for now.
    return s


def make_is_attack(label: str) -> int:
    return 0 if str(label).strip().lower() == "benign" else 1


def feature_hash(df: pd.DataFrame) -> pd.Series:
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feat_cols]
    # stable uint64 hash per row
    return pd.util.hash_pandas_object(X, index=False).astype("uint64")


def stratified_split_by_hash(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    """
    Split at HASH GROUP level, not row level.
    This prevents duplicates (same features) leaking across splits.
    Keeps label stratification approximately correct.
    """
    rng = np.random.default_rng(seed)

    df = df.copy()
    df["_h"] = feature_hash(df)

    # group label at hash level (should be single label after conflicts removed)
    grp = df.groupby("_h")["is_attack"].first().reset_index()
    h_attack = grp[grp["is_attack"] == 1]["_h"].to_numpy()
    h_benign = grp[grp["is_attack"] == 0]["_h"].to_numpy()

    rng.shuffle(h_attack)
    rng.shuffle(h_benign)

    def split_hashes(h_arr: np.ndarray) -> dict[str, set[int]]:
        n = len(h_arr)
        n_train = int(round(n * 0.62))
        n_val = int(round(n * 0.16))
        # rest is test
        train = set(h_arr[:n_train].tolist())
        val = set(h_arr[n_train : n_train + n_val].tolist())
        test = set(h_arr[n_train + n_val :].tolist())
        return {"train": train, "val": val, "test": test}

    sa = split_hashes(h_attack)
    sb = split_hashes(h_benign)

    h2split: dict[int, str] = {}
    for s in ["train", "val", "test"]:
        for h in sa[s]:
            h2split[int(h)] = s
        for h in sb[s]:
            h2split[int(h)] = s

    split = df["_h"].map(lambda x: h2split[int(x)]).astype("category")
    df.drop(columns=["_h"], inplace=True)
    return split


def day_split(day: str) -> str:
    # classic protocol: Mon-Wed train, Thu val, Fri test
    if day in {"Monday", "Tuesday", "Wednesday"}:
        return "train"
    if day == "Thursday":
        return "val"
    if day == "Friday":
        return "test"
    return "train"


def main() -> None:
    t0 = time.time()

    parts = []
    for f in RAW_FILES:
        p = RAW_DIR / f
        if not p.exists():
            raise SystemExit(f"Missing raw file: {p}")
        df = pd.read_parquet(p)
        print("[read]", p.name)

        df["source_file"] = p.name
        df["day"] = df["source_file"].map(infer_day_from_source)
        df["is_attack"] = df["Label"].map(make_is_attack).astype(int)
        df["attack_type"] = df["Label"].map(infer_attack_type)

        parts.append(df)

    df = pd.concat(parts, ignore_index=True)
    print("[concat] shape=", df.shape)

    # Basic cleanup
    # Replace inf and keep numeric stable
    df = df.replace([np.inf, -np.inf], np.nan)

    # Remove rows with all features missing (rare, but safe)
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    before = len(df)
    df = df.dropna(subset=feat_cols, how="all")
    dropped = before - len(df)
    print("[clean] dropped_rows=", dropped)

    # 1) Remove conflicting-label duplicates by feature hash
    h = feature_hash(df)
    tmp = pd.DataFrame({"h": h, "label": df["is_attack"].astype(int)})
    conflicts = tmp.groupby("h")["label"].nunique()
    bad_hashes = set(conflicts[conflicts > 1].index.astype("uint64").tolist())
    if bad_hashes:
        before = len(df)
        keep_mask = ~h.isin(bad_hashes)
        df = df.loc[keep_mask].copy()
        removed = before - len(df)
        print(f"[conflicts] hashes={len(bad_hashes)} removed_rows={removed}")
    else:
        print("[conflicts] hashes=0 removed_rows=0")

    # 2) Create splits
    df["split_day"] = df["day"].map(day_split).astype("category")
    df["split_strat"] = stratified_split_by_hash(df).astype("category")

    # 3) Leak-check: ensure same hash not across splits (for split_strat)
    h2 = feature_hash(df)
    chk = pd.DataFrame({"h": h2, "split": df["split_strat"].astype(str)})
    g = chk.groupby("h")["split"].nunique()
    leaky = int((g > 1).sum())
    total = int(g.shape[0])
    print(f"[leak-check] unique_rows={total} rows_in_>1_split={leaky}")
    if leaky != 0:
        raise SystemExit("Leak-check failed: same feature rows appear in multiple splits.")

    # Save
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)

    meta = []
    meta.append("# Dataset Meta\n\n")
    meta.append(f"- rows: {len(df)}\n")
    meta.append(f"- cols: {df.shape[1]}\n\n")
    meta.append("## Splits\n\n")
    meta.append(str(df["split_strat"].value_counts()) + "\n\n")
    meta.append("## Attack ratio by split_strat\n\n")
    for s in ["train", "val", "test"]:
        sub = df[df["split_strat"] == s]
        if len(sub) == 0:
            continue
        meta.append(f"- {s}: n={len(sub)} attack_ratio={float(sub['is_attack'].mean()):.4f}\n")

    OUT_META.write_text("".join(meta), encoding="utf-8")

    print("[save]", OUT_PARQUET)
    print("[save]", OUT_META)
    print(f"[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()