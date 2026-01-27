"""Prepare CICIDS2017: load flows, remove leakage, handle duplicates and conflicting labels."""

from __future__ import annotations

import re
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DIR,
    PROCESSED_DIR,
    DATA_FILE,
    META_FILE,
    LEAKAGE_COLUMNS,
    NON_FEATURE,
    DAY_TO_SPLIT,
    RNG,
)


def infer_day_from_filename(name: str) -> str:
    for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        if re.search(rf"\b{d}\b", name, re.IGNORECASE):
            return d
    raise ValueError(f"Cannot infer day from filename: {name}")


# Common encoding junk (e.g. "Web Attack XSS") -> replace with canonical
LABEL_FIXES = (
    ("\ufffd", ""),           # Unicode replacement char
    ("  ", " "),
    (" – ", "-"),             # en-dash
    (" - ", "-"),             # normalize "Web Attack - XSS" -> "Web Attack-XSS"
)
LABEL_ALIASES = {
    "web attack - xss": "Web Attack-XSS",
    "web attack - sql injection": "Web Attack-Sql Injection",
    "web attack - brute force": "Web Attack-Brute Force",
    "web attack xss": "Web Attack-XSS",
    "web attack sql injection": "Web Attack-Sql Injection",
    "web attack brute force": "Web Attack-Brute Force",
    "dos attacks-hulk": "DoS attacks-Hulk",
    "dos attacks-goldeneye": "DoS attacks-GoldenEye",
    "dos attacks-slowhttptest": "DoS attacks-SlowHTTPTest",
    "dos attacks-slowloris": "DoS attacks-Slowloris",
    "ddos attacks-hoic": "DDoS attacks-HOIC",
    "ddos attacks-loic-udp": "DDoS attacks-LOIC-UDP",
    "ftp-bruteforce": "FTP-BruteForce",
    "ssh-bruteforce": "SSH-BruteForce",
    "portscan": "PortScan",
    "port scan": "Port Scan",
    "benign": "Benign",
}


def normalize_label(s: str) -> str:
    """Clean label: fix encoding, collapse whitespace, apply canonical aliases."""
    t = str(s).strip()
    for a, b in LABEL_FIXES:
        t = t.replace(a, b)
    t = " ".join(t.split()) if t else ""
    key = t.lower()
    return LABEL_ALIASES.get(key, t) if key else t


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare CICIDS2017: clean, dedup, split.")
    p.add_argument("--no-leakage-drop", action="store_true", help="Do not drop leakage columns.")
    p.add_argument("--no-dedup", action="store_true", help="Do not drop duplicate rows.")
    p.add_argument("--no-conflict-drop", action="store_true", help="Do not drop conflicting-label rows.")
    args = p.parse_args()

    if not RAW_DIR.exists():
        raise SystemExit(f"Missing raw dir: {RAW_DIR}. Place CICIDS2017 parquet files there.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files in {RAW_DIR}.")

    dfs: list[pd.DataFrame] = []
    for f in files:
        print(f"[read] {f.name}")
        df = pd.read_parquet(f)
        df.columns = [c.strip() for c in df.columns]
        if "Label" not in df.columns:
            raise SystemExit(f"Missing Label in {f.name}")
        df["Label"] = df["Label"].astype(str).map(normalize_label)
        df["source_file"] = f.name
        df["day"] = infer_day_from_filename(f.name)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"[concat] shape={df.shape}")

    # Remove leakage columns (match by exact or case-insensitive)
    if not args.no_leakage_drop:
        leak_set = {x.lower() for x in LEAKAGE_COLUMNS}
        drop = [c for c in df.columns if c in LEAKAGE_COLUMNS or c.lower() in leak_set]
        df = df.drop(columns=drop, errors="ignore")
        if drop:
            print(f"[leakage] dropped {len(drop)} columns: {drop}")

    df["is_attack"] = (df["Label"].str.lower() != "benign").astype(int)
    df["attack_type"] = df["Label"]

    # Inf/NaN
    numeric = df.select_dtypes(include=[np.number]).columns
    before = len(df)
    if len(numeric):
        df.loc[:, numeric] = df.loc[:, numeric].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0)
    dropped_nan = int(before - len(df))
    print(f"[clean] shape={df.shape} dropped NaN/inf rows={dropped_nan}")

    # Duplicates
    dup_dropped = 0
    if not args.no_dedup:
        n_before = len(df)
        df = df.drop_duplicates(keep="first")
        dup_dropped = int(n_before - len(df))
        print(f"[dedup] dropped {dup_dropped} duplicate rows, shape={df.shape}")

    # Conflicting labels: same feature vector, different Label -> drop all such rows
    meta_cols = {"Label", "is_attack", "attack_type", "day", "source_file"}
    feature_cols = [c for c in df.columns if c not in meta_cols and c not in LEAKAGE_COLUMNS]
    conflict_dropped = 0
    if not args.no_conflict_drop and feature_cols:
        bad = (
            df.groupby(feature_cols, dropna=False)["Label"]
            .transform(lambda s: s.nunique() > 1)
        )
        conflict_idx = df.index[bad]
        df = df.drop(index=conflict_idx)
        conflict_dropped = len(conflict_idx)
        print(f"[conflicts] dropped {conflict_dropped} rows with conflicting labels, shape={df.shape}")

    # Splits
    df["split_day"] = df["day"].map(DAY_TO_SPLIT)
    missing = df["split_day"].isna()
    if missing.any():
        bad = sorted(df.loc[missing, "day"].astype(str).unique().tolist())
        raise SystemExit(f"Unknown day values: {bad}")
    df["split"] = df["split_day"]

    # Stratified split (by binary label)
    y = df["is_attack"].astype(int)
    idx = df.index.to_numpy()
    test_size = 0.2233
    idx_tv, idx_te = train_test_split(idx, test_size=test_size, random_state=RNG, stratify=y)
    y_tv = y.loc[idx_tv]
    val_size = 0.2023
    _, idx_val = train_test_split(idx_tv, test_size=val_size, random_state=RNG, stratify=y_tv)
    df["split_strat"] = "train"
    df.loc[idx_val, "split_strat"] = "val"
    df.loc[idx_te, "split_strat"] = "test"

    df.to_parquet(DATA_FILE, index=False)
    print(f"[save] {DATA_FILE}")

    feat = [c for c in df.columns if c not in NON_FEATURE and c in df.columns]
    meta_lines = [
        "# Processed CICIDS2017\n\n",
        f"- raw files: {len(files)}\n",
        f"- rows: {len(df)}\n",
        f"- dropped NaN/inf: {dropped_nan}\n",
        f"- dropped duplicates: {dup_dropped}\n",
        f"- dropped conflicting labels: {conflict_dropped}\n",
        f"- features: {len(feat)}\n\n",
        "## split_day\n\n",
        df["split_day"].value_counts().to_string() + "\n\n",
        "## split_strat\n\n",
        df["split_strat"].value_counts().to_string() + "\n\n",
        "## is_attack\n\n",
        df["is_attack"].value_counts().to_string() + "\n\n",
        "## Top labels\n\n",
        df["Label"].value_counts().head(25).to_string() + "\n",
    ]
    META_FILE.write_text("".join(meta_lines), encoding="utf-8")
    print(f"[save] {META_FILE}")


if __name__ == "__main__":
    main()
