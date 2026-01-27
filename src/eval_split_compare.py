"""Run strat vs day train+eval, leakage check; write reports/eval_split_compare.md."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.config import REPORTS_DIR, MODELS_DIR


def _run(cmd: list[str]) -> None:
    r = subprocess.run([sys.executable, "-m"] + cmd, check=False)
    if r.returncode != 0:
        raise SystemExit(f"Command failed: python -m {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare stratified vs day split; write eval_split_compare.md.")
    ap.add_argument("--models-strat", type=Path, default=MODELS_DIR / "baseline_strat")
    ap.add_argument("--models-day", type=Path, default=MODELS_DIR / "baseline_day")
    ap.add_argument("--metrics-strat", type=Path, default=REPORTS_DIR / "metrics_strat")
    ap.add_argument("--metrics-day", type=Path, default=REPORTS_DIR / "metrics_day")
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "eval_split_compare.md")
    args = ap.parse_args()

    models_strat = Path(args.models_strat)
    models_day = Path(args.models_day)
    metrics_strat = Path(args.metrics_strat)
    metrics_day = Path(args.metrics_day)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Train + eval stratified (baseline only: LogReg + RF)
    _run(["src.train", "--split", "strat", "--out-dir", str(models_strat), "--baseline-only"])
    _run(["src.eval", "--split", "strat", "--models-dir", str(models_strat), "--out-dir", str(metrics_strat)])

    # Train + eval day (baseline only)
    _run(["src.train", "--split", "day", "--out-dir", str(models_day), "--baseline-only"])
    _run(["src.eval", "--split", "day", "--models-dir", str(models_day), "--out-dir", str(metrics_day)])

    # Leakage check
    leakage_md = REPORTS_DIR / "leakage_check.md"
    _run(["src.leakage_check", "--out", str(leakage_md)])

    # Load metrics
    def load_metrics(p: Path) -> dict:
        j = p / "metrics.json"
        if not j.exists():
            return {}
        return json.loads(j.read_text())

    m_strat = load_metrics(metrics_strat)
    m_day = load_metrics(metrics_day)

    full_s = m_strat.get("full", m_strat)
    full_d = m_day.get("full", m_day)

    lines = [
        "# Eval: Stratified vs day split\n\n",
        "## 0. Macro F1 comparison\n\n",
        "| Model | Stratified | Day-based | Δ (day − strat) |\n",
        "|-------|------------|-----------|------------------|\n",
    ]
    for key in ["logreg", "random_forest", "xgboost"]:
        fs = full_s.get(key, {})
        fd = full_d.get(key, {})
        if not fs or not fd:
            continue
        ms = fs.get("macro_f1") or float("nan")
        md = fd.get("macro_f1") or float("nan")
        delta = (md - ms) if (isinstance(md, (int, float)) and isinstance(ms, (int, float))) else "—"
        d_str = f"{delta:.4f}" if isinstance(delta, (int, float)) else str(delta)
        lines.append(f"| {key} | {ms:.4f} | {md:.4f} | {d_str} |\n")
    lines.append("\n")

    lines.append("## 1. Stratified split results\n\n")
    lines.append(
        "Train/val/test are random splits with **same attack ratio** (stratified by binary attack vs benign). "
        "No temporal separation.\n\n",
    )

    for key in ["logreg", "random_forest", "xgboost"]:
        v = full_s.get(key)
        if not v:
            continue
        lines.append(f"### {key}\n\n")
        lines.append(f"- Macro F1: {v.get('macro_f1', float('nan')):.4f}\n")
        lines.append(f"- Macro precision: {v.get('macro_precision', float('nan')):.4f}\n")
        lines.append(f"- Macro recall: {v.get('macro_recall', float('nan')):.4f}\n")
        if "roc_auc_ovr" in v and v["roc_auc_ovr"] is not None:
            lines.append(f"- ROC AUC (OvR): {v['roc_auc_ovr']:.4f}\n")
        lines.append("\n")

    lines.append("## 2. Day-based split results\n\n")
    lines.append(
        "Train = Mon–Wed, Val = Thu, Test = Fri. **No overlap by day.** "
        "Test is a completely different day than train.\n\n",
    )

    for key in ["logreg", "random_forest", "xgboost"]:
        v = full_d.get(key)
        if not v:
            continue
        lines.append(f"### {key}\n\n")
        lines.append(f"- Macro F1: {v.get('macro_f1', float('nan')):.4f}\n")
        lines.append(f"- Macro precision: {v.get('macro_precision', float('nan')):.4f}\n")
        lines.append(f"- Macro recall: {v.get('macro_recall', float('nan')):.4f}\n")
        if "roc_auc_ovr" in v and v["roc_auc_ovr"] is not None:
            lines.append(f"- ROC AUC (OvR): {v['roc_auc_ovr']:.4f}\n")
        lines.append("\n")

    lines.append("## 3. Why day-based is more realistic\n\n")
    lines.append(
        "1. **Temporal generalization**: In production, the IDS sees **future** traffic. "
        "Random split lets the model see flows from the same days in both train and test, "
        "including similar background traffic and attack timing. That inflates metrics.\n\n"
    )
    lines.append(
        "2. **Day split = strict unseen day**: Test (Friday) is never seen during training. "
        "Attack mix and benign behavior can differ across days. Lower performance on day split "
        "reflects **realistic** generalization, not a broken model.\n\n"
    )
    lines.append(
        "3. **Credibility**: Reporting only stratified results is misleading. "
        "We report both; **day-based numbers are the ones that matter** for deployment.\n\n"
    )

    lines.append("## 4. Leakage check summary\n\n")
    lines.append(
        "See **`reports/leakage_check.md`** for exact duplicates, near-duplicates (rounded features), "
        "and 5-tuple limitation. Run `python -m src.leakage_check` to regenerate.\n\n"
    )

    out.write_text("".join(lines), encoding="utf-8")
    print(f"[eval_split_compare] wrote {out}")


if __name__ == "__main__":
    main()
