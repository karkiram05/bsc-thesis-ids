from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import METRICS_DIR, ALERTS_DIR

MAPPING_PATH = Path(__file__).resolve().parent / "mitre_mapping.json"


def _lookup(mapping: dict, attack_type: str) -> dict:
    """Find MITRE entry for attack_type, trying fuzzy matches before fallback."""
    at = str(attack_type).strip()

    if at in mapping:
        return {**mapping[at], "mapped_from": at}

    # different dash characters appear across datasets
    norm = at.replace(" - ", "-").replace(" – ", "-")
    if norm in mapping:
        return {**mapping[norm], "mapped_from": norm}

    at_lower = at.lower()
    norm_lower = norm.lower()
    for k, v in mapping.items():
        if k.startswith("_"):
            continue
        if k.lower() == at_lower or k.lower() == norm_lower:
            return {**v, "mapped_from": k}

    for k, v in mapping.items():
        if k.startswith("_"):
            continue
        if at_lower in k.lower() or k.lower() in at_lower:
            return {**v, "mapped_from": k}

    # if mapping file got edited and _default is missing, still return something usable
    default = mapping.get("_default", {
        "attck_id": None,
        "attck_name": None,
        "ck_phase": None,
        "justification": "No MITRE mapping available for this attack type.",
    })
    return {**default, "mapped_from": "_default"}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Map model predictions to MITRE ATT&CK techniques."
    )
    ap.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    ap.add_argument("--out-dir", type=Path, default=ALERTS_DIR)
    ap.add_argument(
        "--model",
        default=None,
        help="Use this model's predictions (e.g. random_forest). Default: best by macro F1.",
    )
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not MAPPING_PATH.exists():
        raise SystemExit(f"Missing MITRE mapping file: {MAPPING_PATH}")
    if not (metrics_dir / "metrics.json").exists():
        raise SystemExit(f"Missing metrics.json in {metrics_dir}. Run: python -m src.evaluate first.")

    mapping = json.loads(MAPPING_PATH.read_text())
    m = json.loads((metrics_dir / "metrics.json").read_text())
    summary = m.get("summary", m)
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
        raise SystemExit("No model found in metrics.json. Run: python -m src.evaluate")

    pred_path = metrics_dir / f"predictions_{best}.csv"
    if not pred_path.exists():
        raise SystemExit(f"Missing {pred_path}. Run: python -m src.evaluate")

    pred_df = pd.read_csv(pred_path)
    unique_pred = sorted(p for p in pred_df["pred_label"].unique().tolist()
                         if p != "__unseen__")

    alerts = []
    unmapped = []
    for at in unique_pred:
        row = _lookup(mapping, at)
        entry = {
            "predicted_attack": at,
            "attck_id": row.get("attck_id"),
            "attck_name": row.get("attck_name"),
            "ck_phase": row.get("ck_phase"),
            "justification": row.get("justification"),
            "mapped_from": row.get("mapped_from"),
        }
        alerts.append(entry)
        if row.get("mapped_from") == "_default":
            unmapped.append(at)

    if unmapped:
        print(f"[mitre] WARNING: {len(unmapped)} attack types used _default fallback mapping:")
        for u in unmapped:
            print(f"         - '{u}'")
        print("         Add these to src/mitre_mapping.json for precise mappings.")

    out_json = out_dir / "alerts.json"
    with open(out_json, "w") as f:
        json.dump({"model": best, "alerts": alerts}, f, indent=2)
    print(f"[mitre] wrote {out_json}")

    out_csv = out_dir / "alerts.csv"
    pd.DataFrame(alerts).to_csv(out_csv, index=False)
    print(f"[mitre] wrote {out_csv}")

    print(f"[mitre] {len(alerts)} unique predicted attack types mapped.")
    print(f"[mitre] {len(alerts) - len(unmapped)} had precise mappings, {len(unmapped)} used fallback.")


if __name__ == "__main__":
    main()