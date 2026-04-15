"""Sanity check: dataset shape, class distribution, split coverage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DATA_FILE,
    LEAKAGE_COLUMNS,
    NON_FEATURE,
    PROCESSED_DIR,
    REPORTS_DIR,
    SPLIT_COL_DAY,
    SPLIT_COL_STRAT,
)


def _class_dist(series: pd.Series) -> pd.DataFrame:
    counts = series.value_counts().sort_index()
    pct = (counts / counts.sum() * 100).round(2)
    return pd.DataFrame({"count": counts, "pct": pct})


def main() -> None:
    out_dir = REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = ["# Sanity Check Report\n"]

    # --- 1. File exists ---
    if not DATA_FILE.exists():
        print(f"[ERROR] Missing {DATA_FILE}. Run: python -m src.prepare_data")
        return

    print(f"[sanity] Loading {DATA_FILE} ...")
    df = pd.read_parquet(DATA_FILE)

    # --- 2. Shape ---
    lines.append(f"## Dataset Shape\n\n- Rows: {len(df):,}\n- Columns: {len(df.columns)}\n")
    print(f"[shape] {df.shape[0]:,} rows x {df.shape[1]} columns")

    # --- 3. Feature count ---
    feat = [c for c in df.columns if c not in NON_FEATURE]
    lines.append(f"- Feature columns (excluding metadata): {len(feat)}\n")
    print(f"[features] {len(feat)} feature columns")

    # --- 4. Missing / inf ---
    n_missing = int(df[feat].isnull().sum().sum())
    n_inf = int(np.isinf(df[feat].select_dtypes("number")).sum().sum())
    lines.append(f"\n## Data Quality\n\n- Missing values in features: {n_missing}\n"
                 f"- Inf values in features: {n_inf}\n")
    if n_missing > 0:
        print(f"[WARN] {n_missing} missing values in feature columns")
    if n_inf > 0:
        print(f"[WARN] {n_inf} inf values in feature columns")
    else:
        print("[quality] No missing or inf values in features")

    # --- 5. Leakage columns ---
    present_leak = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    dropped_leak = [c for c in LEAKAGE_COLUMNS if c not in df.columns]
    lines.append(f"\n## Leakage Columns\n\n"
                 f"- Expected to be dropped: {sorted(LEAKAGE_COLUMNS)}\n"
                 f"- Still present in data: {present_leak}\n"
                 f"- Confirmed dropped: {dropped_leak}\n")
    if present_leak:
        print(f"[WARN] Leakage columns still present: {present_leak}")
    else:
        print(f"[leakage] All {len(LEAKAGE_COLUMNS)} leakage columns confirmed absent")

    # --- 6. Class distribution (overall) ---
    lines.append(f"\n## Attack Type Distribution (Overall)\n\n")
    dist = _class_dist(df["attack_type"])
    lines.append(dist.to_markdown() + "\n")
    print(f"[classes] {df['attack_type'].nunique()} unique attack types")

    # --- 7. Split coverage per split strategy ---
    for split_col in [SPLIT_COL_STRAT, SPLIT_COL_DAY]:
        if split_col not in df.columns:
            lines.append(f"\n## Split: {split_col}\n\nColumn not found — run prepare_data.\n")
            continue

        lines.append(f"\n## Split: `{split_col}`\n")
        for part in ["train", "val", "test"]:
            sub = df[df[split_col] == part]
            n = len(sub)
            classes = set(sub["attack_type"].unique())
            pct = n / len(df) * 100
            lines.append(f"\n### {part} ({n:,} rows, {pct:.1f}%)\n\n")
            dist_part = _class_dist(sub["attack_type"])
            lines.append(dist_part.to_markdown() + "\n")

        # Missing classes by split
        all_classes = set(df["attack_type"].unique())
        for part in ["train", "val", "test"]:
            sub = df[df[split_col] == part]
            present = set(sub["attack_type"].unique())
            missing = sorted(all_classes - present)
            if missing:
                lines.append(f"\n> **WARNING** `{split_col}` {part} missing classes: "
                             f"{missing}\n")
                print(f"[WARN] {split_col} {part} missing classes: {missing}")
            else:
                print(f"[split] {split_col} {part}: all {len(all_classes)} classes present")

    # --- 8. Duplicates summary ---
    n_exact = int(df.duplicated().sum())
    lines.append(f"\n## Duplicate Rows\n\n- Exact duplicates: {n_exact:,}\n")
    print(f"[dupes] {n_exact:,} exact duplicate rows")

    # --- 9. Day distribution ---
    if "day" in df.columns:
        lines.append(f"\n## Rows per Capture Day\n\n")
        day_dist = df.groupby("day")["attack_type"].value_counts().unstack(fill_value=0)
        lines.append(day_dist.to_markdown() + "\n")

    # --- 10. Column list ---
    meta_cols = [c for c in df.columns if c in NON_FEATURE or c == "row_hash"]
    lines.append(f"\n## Metadata Columns\n\n```\n{chr(10).join(sorted(meta_cols))}\n```\n")
    lines.append(f"\n## Feature Columns ({len(feat)} total)\n\n"
                 f"```\n{chr(10).join(sorted(feat))}\n```\n")

    # --- Write report ---
    report_path = out_dir / "sanity_check.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[sanity] wrote {report_path}")


if __name__ == "__main__":
    main()