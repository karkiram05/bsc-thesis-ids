"""Produce alert output: predicted attack type -> MITRE ATT&CK technique + justification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import METRICS_DIR, ALERTS_DIR

MAPPING_PATH = Path(__file__).resolve().parent / "mitre_mapping.json"


def _lookup(mapping: dict, attack_type: str) -> dict:
    at = str(attack_type).strip()
    for key in [at, at.replace(" - ", "-"), at.replace(" ", ""), at.replace(" ", "-")]:
        if key in mapping:
            return {**mapping[key], "mapped_from": key}
    for k, v in mapping.items():
        if k.startswith("_") or k == "Benign":
            continue
        if k.lower() == at.lower():
            return {**v, "mapped_from": k}
    return {**mapping["_default"], "mapped_from": "_default"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    ap.add_argument("--out-dir", type=Path, default=ALERTS_DIR)
    ap.add_argument("--model", default=None, help="Use this model's predictions; default: best by macro F1")
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = json.loads(MAPPING_PATH.read_text())
    m = json.loads((metrics_dir / "metrics.json").read_text())
    summary = m["summary"] if "summary" in m else m
    full = m.get("full", m)

    if args.model:
        best = args.model
    else:
        best = summary.get("error_analysis", {}).get("best_model")
        if not best:
            best = max(
                (k for k in full if isinstance(full[k], dict) and "macro_f1" in full[k]),
                key=lambda k: full[k]["macro_f1"],
                default=None,
            )
    if not best:
        raise SystemExit("No model found in metrics. Run: make eval")

    pred_path = metrics_dir / f"predictions_{best}.csv"
    if not pred_path.exists():
        raise SystemExit(f"Missing {pred_path}. Run: make eval")

    pred_df = pd.read_csv(pred_path)
    unique_pred = pred_df["pred_label"].unique().tolist()

    alerts = []
    for at in unique_pred:
        row = _lookup(mapping, at)
        alerts.append({
            "predicted_attack": at,
            "attck_id": row.get("attck_id"),
            "attck_name": row.get("attck_name"),
            "ck_phase": row.get("ck_phase"),
            "justification": row.get("justification"),
            "mapped_from": row.get("mapped_from"),
        })

    out_json = out_dir / "alerts.json"
    with open(out_json, "w") as f:
        json.dump({"model": best, "alerts": alerts}, f, indent=2)
    print(f"[mitre] wrote {out_json}")

    out_csv = out_dir / "alerts.csv"
    pd.DataFrame(alerts).to_csv(out_csv, index=False)
    print(f"[mitre] wrote {out_csv}")


if __name__ == "__main__":
    main()
