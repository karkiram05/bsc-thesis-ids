from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path("data/raw/cicids2017")
OUT_MD = Path("reports/dataset_audit.md")


def audit_one(p: Path) -> dict:
    df = pd.read_parquet(p)

    # standardize column names
    df.columns = [c.strip() for c in df.columns]

    label_col = "Label" if "Label" in df.columns else None

    # missing values
    missing_by_col = df.isna().sum().to_dict()
    missing_total = int(df.isna().sum().sum())

    # numeric-only inf values
    numeric_df = df.select_dtypes(include=[np.number])
    inf_by_col: dict[str, int] = {}
    inf_total = 0
    if numeric_df.shape[1] > 0:
        inf_matrix = np.isinf(numeric_df.to_numpy())
        inf_total = int(inf_matrix.sum())
        inf_counts = inf_matrix.sum(axis=0)
        inf_by_col = dict(zip(numeric_df.columns, inf_counts.astype(int).tolist()))

    dup_rows = int(df.duplicated().sum())

    # labels
    label_counts = df[label_col].value_counts().head(15).to_dict() if label_col else {}
    label_unique = int(df[label_col].nunique()) if label_col else 0

    # only show columns with problems (>0)
    top_missing = [
        (k, int(v))
        for k, v in sorted(missing_by_col.items(), key=lambda x: x[1], reverse=True)
        if v > 0
    ][:10]

    top_inf = [
        (k, int(v))
        for k, v in sorted(inf_by_col.items(), key=lambda x: x[1], reverse=True)
        if v > 0
    ][:10]

    return {
        "file": p.name,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "has_label": bool(label_col),
        "label_unique": label_unique,
        "dup_rows": dup_rows,
        "missing_total": missing_total,
        "inf_total": inf_total,
        "top_labels": label_counts,
        "top_missing": top_missing,
        "top_inf": top_inf,
    }


def main() -> None:
    files = sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files found in {RAW_DIR}")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# CICIDS2017 Parquet Dataset Audit\n\n",
        f"Files audited: {len(files)}\n\n",
    ]

    for f in files:
        r = audit_one(f)
        lines.append(f"## {r['file']}\n")
        lines.append(f"- rows: {r['rows']}\n")
        lines.append(f"- cols: {r['cols']}\n")
        lines.append(f"- has Label column: {r['has_label']}\n")
        if r["has_label"]:
            lines.append(f"- unique labels: {r['label_unique']}\n")
        lines.append(f"- duplicate rows: {r['dup_rows']}\n")
        lines.append(f"- missing_total: {r['missing_total']}\n")
        lines.append(f"- inf_total: {r['inf_total']}\n")

        lines.append("\nTop labels (first 15):\n")
        if r["top_labels"]:
            for k, v in r["top_labels"].items():
                lines.append(f"- {k}: {v}\n")
        else:
            lines.append("- (no Label column found)\n")

        lines.append("\nTop missing columns (>0):\n")
        if r["top_missing"]:
            for k, v in r["top_missing"]:
                lines.append(f"- {k}: {v}\n")
        else:
            lines.append("- (no missing values found)\n")

        lines.append("\nTop inf columns (>0):\n")
        if r["top_inf"]:
            for k, v in r["top_inf"]:
                lines.append(f"- {k}: {v}\n")
        else:
            lines.append("- (no inf values found)\n")

        lines.append("\n")

    OUT_MD.write_text("".join(lines), encoding="utf-8")
    print(f"Saved: {OUT_MD}")


if __name__ == "__main__":
    main()