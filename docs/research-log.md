# Research log

## Pipeline (reproducible)

- **Data**: `make data` -- `prepare_cicids`: load CICIDS2017 parquets, drop leakage cols, clean labels (encoding fixes, aliases), handle duplicates and conflicting labels, split by day + stratified.
- **Train**: `make train` -- LogReg, RF, XGBoost, LightGBM; multi-class on `attack_type`. `--baseline-only` for LogReg+RF only.
- **Eval**: `make eval` -- metrics, per-class confusion, feature importance, predictions.
- **Report**: `make report` -- MITRE alerts, `report.md`, figures.

## Credibility (order of operations)

1. **Multi-class baseline**
   LogReg + RF, macro F1, per-class recall, confusion matrix.

2. **Leakage check**
   `make leakage-check` -- exact duplicates, near-duplicates (rounded numeric features), 5-tuple limitation.

3. **Strat vs day split**
   Train+eval on stratified and on day split. Day-based is more realistic because Friday attack types never appear in training.

4. **Full model suite**
   XGBoost + LightGBM added. Evaluated under both splits plus LODO cross-validation.

5. **MITRE alerts**
   `make report` uses `mitre_alerts` -- predicted class, confidence, technique id+name, severity.
