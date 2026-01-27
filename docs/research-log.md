# Research log

## Pipeline (reproducible)

- **Data**: `make data` → `prepare_data`: load CICIDS2017 parquets, drop leakage cols (Flow Bytes/s, Flow Packets/s, Fwd/Bwd Packets/s), handle duplicates and conflicting labels, split by day (train/val/test) and stratified.
- **Train**: `make train` → LogReg (baseline), RF, XGBoost; multi-class on `attack_type`. Models and scaler/encoder saved under `models/`.
- **Eval**: `make eval` → metrics (P/R/F1, ROC AUC, PR AUC), per-class confusion matrices, feature importance, predictions.
- **Report**: `make report` → MITRE alerts (attack → ATT&CK + CK + justification), `report.md`, confusion and importance plots.

Commands: `make data`, `make train`, `make eval`, `make report` (or `make all`).
