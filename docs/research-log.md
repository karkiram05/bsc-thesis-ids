# Research log

## Pipeline (reproducible)

- **Data**: `make data` → `prepare_data`: load CICIDS2017 parquets, drop leakage cols, **clean labels** (encoding fixes, aliases), handle duplicates and conflicting labels, split by day + stratified.
- **Train**: `make train` → LogReg, RF, XGBoost; multi-class on `attack_type`. `--baseline-only` for LogReg+RF only.
- **Eval**: `make eval` → metrics, per-class confusion, feature importance, predictions.
- **Report**: `make report` → MITRE alerts, `report.md`, figures.

## Credibility (order of operations)

1. **Multi-class baseline**  
   `make multiclass-baseline` → LogReg + RF, macro F1, per-class recall, confusion matrix → `reports/baselines/multiclass_report.md`.

2. **Leakage check**  
   `make leakage-check` → exact duplicates, **near-duplicates** (rounded numeric features), 5-tuple limitation → `reports/leakage_check.md`.

3. **Strat vs day split**  
   `make eval-split-compare` → train+eval on stratified and on day split, leakage check, comparison + “why day-based is more realistic” → `reports/eval_split_compare.md`.

4. **Real model** (after above)  
   Add XGBoost/LightGBM/MLP in `train.py`, document in `reports/models/model_v1.md`.

5. **MITRE alerts**  
   `make report` uses `mitre_alerts` → predicted class, confidence, technique id+name, severity.
