#!/usr/bin/env python3
"""run_pipeline.py — Run the full thesis pipeline end to end.

Runs every step in order with proper dependency checking.
Saves a run manifest (run_manifest.json) with timestamps, git hash, and all output paths.

Usage:
  python run_pipeline.py                    # full run, both splits
  python run_pipeline.py --split strat      # stratified split only
  python run_pipeline.py --skip-data        # skip prepare_data (data already built)
  python run_pipeline.py --dry-run          # print steps without running
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _run(cmd: list[str], dry_run: bool = False) -> int:
    label = " ".join(cmd)
    print(f"\n{'='*60}")
    print(f"STEP: {label}")
    print(f"{'='*60}")
    if dry_run:
        print("[dry-run] skipping")
        return 0
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[ERROR] Step failed (exit {result.returncode}): {label}")
        sys.exit(result.returncode)
    print(f"[ok] {elapsed:.1f}s")
    return result.returncode


def _git_hash() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=ROOT
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run full thesis pipeline")
    ap.add_argument("--split", choices=["strat", "day", "both"], default="both")
    ap.add_argument("--skip-data", action="store_true",
                    help="Skip prepare_data (data already built)")
    ap.add_argument("--skip-leakage", action="store_true",
                    help="Skip leakage check")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print steps without executing")
    ap.add_argument("--baseline-only", action="store_true",
                    help="Train only LogReg + RF, skip XGBoost")
    args = ap.parse_args()

    dry = args.dry_run
    splits = ["strat", "day"] if args.split == "both" else [args.split]
    bl = ["--baseline-only"] if args.baseline_only else []

    manifest = {
        "run_start": datetime.now().isoformat(),
        "git_hash": _git_hash(),
        "splits": splits,
        "baseline_only": args.baseline_only,
        "steps": [],
    }

    def step(cmd, label):
        t0 = time.time()
        _run(cmd, dry_run=dry)
        manifest["steps"].append({
            "label": label,
            "cmd": " ".join(cmd),
            "elapsed_s": round(time.time() - t0, 1),
        })

    py = [sys.executable]

    # Step 1: Sanity check (always first)
    step(py + ["-m", "src.sanity_check"], "sanity_check")

    # Step 2: Prepare data
    if not args.skip_data:
        step(py + ["-m", "src.prepare_data"], "prepare_data")
    else:
        print("[skip] prepare_data (--skip-data)")

    # Step 3: Leakage check
    if not args.skip_leakage:
        step(py + ["-m", "src.leakage_check"], "leakage_check")
    else:
        print("[skip] leakage_check (--skip-leakage)")

    # Steps 4-8: Per split
    for split in splits:
        models_dir = f"models/baseline_{split}"
        metrics_dir = f"reports/metrics_{split}"
        binary_dir = f"reports/metrics_{split}_binary"
        alerts_dir = f"reports/alerts_{split}"

        step(
            py + ["-m", "src.train",
                  "--split", split,
                  "--out-dir", models_dir] + bl,
            f"train_{split}"
        )

        step(
            py + ["-m", "src.eval",
                  "--split", split,
                  "--models-dir", models_dir,
                  "--out-dir", metrics_dir],
            f"eval_{split}"
        )

        step(
            py + ["-m", "src.eval_binary",
                  "--split", split,
                  "--models-dir", models_dir,
                  "--out-dir", binary_dir],
            f"eval_binary_{split}"
        )

        step(
            py + ["-m", "src.mitre_alerts",
                  "--metrics-dir", metrics_dir,
                  "--out-dir", alerts_dir],
            f"mitre_alerts_{split}"
        )

    # Step: Traffic analysis
    step(py + ["-m", "src.traffic_analysis"], "traffic_analysis")

    # Step: Report (use strat as primary)
    primary_metrics = "reports/metrics_strat"
    primary_alerts = "reports/alerts_strat"
    if "day" in splits and "strat" not in splits:
        primary_metrics = "reports/metrics_day"
        primary_alerts = "reports/alerts_day"

    step(
        py + ["-m", "src.report",
              "--metrics-dir", primary_metrics,
              "--alerts-dir", primary_alerts],
        "report"
    )

    # Step: Split comparison (only if both splits ran)
    if len(splits) == 2:
        step(
            py + ["-m", "src.eval_split_compare",
                  "--skip-train",
                  "--models-strat", "models/baseline_strat",
                  "--models-day", "models/baseline_day",
                  "--metrics-strat", "reports/metrics_strat",
                  "--metrics-day", "reports/metrics_day"],
            "eval_split_compare"
        )

    # Write manifest
    manifest["run_end"] = datetime.now().isoformat()
    manifest["status"] = "success"
    manifest_path = ROOT / "reports" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"  Git hash : {manifest['git_hash']}")
    print(f"  Finished : {manifest['run_end']}")
    print(f"  Manifest : {manifest_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()