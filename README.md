# BSc Thesis IDS

Reproducible pipeline for CICIDS2017 flow-based intrusion detection: data prep, training, evaluation, explanation, MITRE ATT&CK mapping, and reporting.

## Pipeline

1. **Prepare data**: Load CICIDS2017 flows, remove leakage columns, handle duplicates and conflicting labels.
2. **Split**: Stratified split by **day** (no overlap) or by **stratified** random; train/val/test.
3. **Train**: Baseline (LogReg) + two stronger models (Random Forest, XGBoost). Multi-class on `attack_type`.
4. **Evaluate**: Precision, recall, F1 (macro/micro), ROC AUC, PR AUC, per-class confusion matrix, error analysis.
5. **Explain**: Feature importance (RF, XGBoost) and confusion breakdown.
6. **MITRE**: Map predicted attack types to ATT&CK techniques and Cyber Kill Chain phase; produce alert output with justification.
7. **Report**: Tables (metrics, confusions, importance, alerts) and plots (confusion heatmap, importance bars).

## Commands

```bash
make data     # Preprocess (prepare_data)
make train    # Train models (depends on data)
make eval     # Evaluate (depends on train)
make report   # MITRE alerts + report tables/plots (depends on eval)
make all      # Same as make report
make clean    # Remove data/processed, reports, models
```

**Credibility-focused (run in this order):**

```bash
make multiclass-baseline   # LogReg + RF multi-class -> reports/baselines/multiclass_report.md
make leakage-check         # Exact + near-dup leakage -> reports/leakage_check.md
make eval-split-compare    # Strat vs day split -> reports/eval_split_compare.md
```

Override split or Python:

```bash
make train SPLIT=strat    # Use stratified split instead of day
make data PY=python3.11
```

## Setup

1. Create a virtualenv and install deps:

   ```bash
   python -m venv .venv && source .venv/bin/activate  # or equivalent
   pip install -r requirements.txt
   ```

2. Place CICIDS2017 parquet files in `data/raw/cicids2017/`.  
   Each file must have a `Label` column and a day in the filename (e.g. `Benign-Monday-....parquet`, `...-Friday-....parquet`).  
   Generate parquets from the official CSVs (e.g. with a small script) if needed.

3. Run the pipeline:

   ```bash
   make data && make train && make eval && make report
   ```

## Outputs

| Command   | Outputs |
|----------|---------|
| `make data` | `data/processed/cicids2017/all_clean.parquet`, `meta.md` |
| `make train` | `models/*.joblib`, `label_encoder.joblib`, `scaler.joblib`, `meta.json`, `feature_names.json` |
| `make eval` | `reports/metrics/metrics.json`, `confusion_matrix_*.csv`, `predictions_*.csv`, `feature_importance_*.csv`, `classification_report_*.txt` |
| `make report` | `reports/alerts/alerts.json`, `alerts.csv`; `reports/report.md`; `reports/figures/confusion_matrix.png`, `feature_importance_*.png` |

## Leakage columns

The following CICFlowMeter-derived rate features are dropped by default to avoid leakage:

- `Flow Bytes/s`, `Flow Packets/s`, `Fwd Packets/s`, `Bwd Packets/s`

Use `--no-leakage-drop` when running `prepare_data` to keep them (not recommended).

## Scripts

- `src/prepare_data.py` — Preprocess, dedup, conflict resolution, label cleaning, splits.
- `src/train.py` — Train LogReg, RF, XGBoost (`--baseline-only` for LogReg+RF only).
- `src/train_multiclass_baseline.py` — Multi-class baseline (LogReg + RF) → `multiclass_report.md`.
- `src/leakage_check.py` — Exact + near-duplicate leakage checks; documents 5-tuple limitation.
- `src/eval_split_compare.py` — Stratified vs day split comparison → `eval_split_compare.md`.
- `src/eval.py` — Metrics, confusion matrices, feature importance, predictions.
- `src/mitre_alerts.py` — Map predictions to ATT&CK + CK, write alerts.
- `src/report.py` — Build `report.md` and figures.
- `src/config.py` — Paths, leakage list, split config.
- `src/mitre_mapping.json` — Attack type → ATT&CK technique + justification.

## Legacy

- `src/build_dataset.py`, `src/train_baseline.py`, `src/audit_parquet.py` — Older scripts; the Makefile uses the new pipeline above.
