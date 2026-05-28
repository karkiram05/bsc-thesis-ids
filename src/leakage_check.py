from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_FILE, REPORTS_DIR, feature_cols

# 5-tuple columns (may or may not be present)
TUPLE5_CANDIDATES = [
    ["Source IP", "Destination IP", "Source Port", "Destination Port", "Protocol"],
    ["SourceIP", "DestinationIP", "SourcePort", "DestinationPort", "Protocol"],
]


def _has_5tuple(df: pd.DataFrame) -> list[str] | None:
    for cand in TUPLE5_CANDIDATES:
        if all(c in df.columns for c in cand):
            return cand
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Run leakage checks (exact + near duplicates), write report.")
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "leakage_check.md")
    ap.add_argument("--near-dup-decimals", type=int, default=2, help="Round numeric features to this many decimals for near-dup check")
    args = ap.parse_args()

    if not DATA_FILE.exists():
        raise SystemExit(f"Missing {DATA_FILE}. Run: make data")

    df = pd.read_parquet(DATA_FILE)
    feat = feature_cols(df)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Leakage check report\n\n"]

    # --- Available keys ---
    lines.append("## 1. Available grouping keys\n\n")
    has_day = "day" in df.columns
    has_file = "source_file" in df.columns
    tuple5 = _has_5tuple(df)
    lines.append(f"- **day**: {has_day}\n")
    lines.append(f"- **source_file**: {has_file}\n")
    if tuple5:
        lines.append("- **5-tuple** (Source IP, Destination IP, Source Port, Destination Port, Protocol): **yes**\n")
    else:
        lines.append(
            "- **5-tuple** (Source IP, Destination IP, Source Port, Destination Port, Protocol): **no** "
            "(not in this dataset). Near-duplicate check uses **rounded numeric features** only.\n"
        )
    lines.append("\n")

    # --- Exact duplicates ---
    lines.append("## 2. Exact duplicate rows\n\n")
    dup = df.duplicated(keep=False)
    n_dup = int(dup.sum())
    n_unique_dup = int(dup.sum() - df.duplicated(keep="first").sum()) if n_dup else 0
    lines.append(f"- Total rows: {len(df)}\n")
    lines.append(f"- Duplicate rows (including first of group): {n_dup}\n")
    lines.append(f"- Extra copies (duplicates beyond first): {n_unique_dup}\n")
    if n_dup > 0:
        dup_df = df[dup]
        for sc in ["split_day", "split_strat"]:
            if sc not in df.columns:
                continue
            dup_sub = dup_df[[sc] + feat].drop_duplicates()
            grp = dup_sub.groupby(feat)[sc].apply(set)
            cross = grp.apply(lambda s: "train" in s and "test" in s).sum()
            lines.append(f"- **{sc}**: duplicate groups that span both train and test: {int(cross)}\n")
    else:
        lines.append("- No exact duplicates found (good).\n")
    lines.append("\n")

    # --- Near duplicates (rounded numeric features) ---
    lines.append("## 3. Near-duplicate rows (rounded numeric features)\n\n")
    lines.append(
        f"Flows with identical **rounded** numeric features (to {args.near_dup_decimals} decimals) "
        "can leak between train and test if both splits contain them.\n\n"
    )
    numeric = [c for c in feat if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        lines.append("No numeric feature columns found. Skipping near-dup check.\n\n")
    else:
        X = df[numeric].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        r = 10 ** (-args.near_dup_decimals)
        for c in numeric:
            X[c] = (np.round(X[c].to_numpy() / r) * r).astype(np.float64)

        for sc in ["split_day", "split_strat"]:
            if sc not in df.columns:
                continue
            lines.append(f"### {sc}\n\n")
            d = pd.DataFrame({sc: df[sc].values, **{c: X[c].values for c in numeric}})
            g = d.groupby(numeric)[sc].apply(lambda s: set(s.unique()))
            cross = g.apply(lambda s: "train" in s and "test" in s)
            n_common = int(cross.sum())
            cross_keys = set(g[cross].index)
            d["_key"] = d[numeric].apply(lambda r: tuple(r), axis=1)
            d["_cross"] = d["_key"].map(lambda t: t in cross_keys)
            train_in = int(d.loc[d[sc] == "train", "_cross"].sum())
            test_in = int(d.loc[d[sc] == "test", "_cross"].sum())
            lines.append(f"- Near-dup groups (rounded features) that appear in **both** train and test: {n_common}\n")
            lines.append(f"- Train rows in such groups: {train_in}\n")
            lines.append(f"- Test rows in such groups: {test_in}\n")
            if n_common > 0:
                lines.append(
                    "  **Warning:** These rows are potential leakage. Consider removing or using a stricter key.\n"
                )
            lines.append("\n")

    # --- Limitation ---
    lines.append("## 4. Limitation\n\n")
    if not tuple5:
        lines.append(
            "This dataset does **not** include 5-tuple (Source/Dest IP, Source/Dest Port, Protocol). "
            "Near-duplicate detection therefore uses **rounded flow features** only. "
            "Flows that differ only in endpoints but have similar statistics can be grouped together. "
            "For stronger leakage checks, use data that includes 5-tuple or flow IDs.\n"
        )
    else:
        lines.append("5-tuple is available. Consider adding a separate check that groups by 5-tuple and reports cross-split overlap.\n")

    out.write_text("".join(lines), encoding="utf-8")
    print(f"[leakage_check] wrote {out}")


if __name__ == "__main__":
    main()
