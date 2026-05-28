# Flow-Based Machine Learning for Network Intrusion Detection with MITRE ATT&CK Mapping

**Bachelor Thesis**: Ram Karki (S236176), General Engineering, DTU, 15 ECTS
**Supervisor**: Gaurav Choudhary, Department of Applied Mathematics and Computer Science

---

## Overview

A reproducible flow-based intrusion detection pipeline on two public datasets: CICIDS2017 (primary) and UNSW-NB15 (cross-dataset check). Four supervised models (Logistic Regression, Random Forest, XGBoost, LightGBM) and two unsupervised baselines (Isolation Forest, Local Outlier Factor) trained for multi-class and binary attack detection, evaluated under three protocols (stratified random split, day-based temporal split, leave-one-day-out cross-validation), with a score-query black-box adversarial study, a Flow IAT Min ablation, a post-processing probability calibration experiment, MITRE ATT&CK mapping for every predicted class, and a live Flask plus Wireshark demo that scores real packets from the host network interface.

The contribution is not a new accuracy record. It is an honest end-to-end evaluation that exposes the gap between random-split benchmark numbers and the deployment-relevant numbers a Security Operations Centre would see.

---

## Headline Findings

1. **Multi-class collapses under temporal evaluation.** XGBoost macro F1 falls from 0.86 (stratified) to 0.44 (day-based), a 49 percent relative drop on the same data and hyperparameters, because Friday's DDoS, PortScan, and Bot classes never appear in Monday to Wednesday training.
2. **Binary detection survives.** Random Forest reaches F1 = 0.86 on the day split (vs 0.996 stratified) and ROC-AUC = 0.926 plus or minus 0.053 across the four attack-bearing LODO folds, the lowest variance of the four models.
3. **The model is brittle under low-effort adversarial pressure.** A score-query black-box greedy attack flips 20.8 percent of correctly classified attacks under a hard 0.25 sigma budget, 25.6 percent unbounded. 84.4 percent of successful evasions move a single feature (Flow IAT Min); a controlled ablation shows the attack re-concentrates on Forward IAT Min when Flow IAT Min is removed, so the vulnerability is structural to inter-arrival timing rather than feature-specific.
4. **Model secrecy is not a defence.** 86.8 percent of adversarial flows crafted against Random Forest also evade XGBoost.
5. **Naive single-shot adversarial training helps modestly.** Evasion drops from 25.6 to 22.2 percent at a 3.2-point clean-F1 cost (0.859 to 0.827). Iterative PGD-AT is identified as future work.
6. **Post-processing calibration does not transfer across day boundaries.** Platt scaling reduces RF ECE from 0.178 to 0.161, but recall at fixed FPR is unchanged or slightly worse, because the val-to-test mapping does not survive the temporal shift.

---

## Project Structure

```
bsc-thesis-ids/
├── data/
│   ├── raw/cicids2017/             # raw CICFlowMeter parquet files (gitignored)
│   ├── unsw-nb15/raw/              # UNSW-NB15 CSV files (gitignored)
│   ├── processed/cicids2017/       # cleaned CICIDS2017 (built by prepare_cicids)
│   └── processed/unsw-nb15/        # cleaned UNSW-NB15 (built by prepare_unsw)
├── models/                         # trained model artifacts (gitignored)
├── reports/                        # metrics, figures, tables (gitignored)
├── src/                            # all pipeline code (see below)
├── demo/                           # Flask SOC console + Wireshark bridge
├── thesis/                         # generated thesis figures
└── docs/                           # research log
```

### Source modules (`src/`)

**Data preparation**
- `config.py` -- shared paths, leakage columns, split definitions
- `prepare_cicids.py` -- clean, split, and save CICIDS2017
- `prepare_unsw.py` -- clean, split, and save UNSW-NB15
- `leakage_check.py` -- verify rate-derived features are removed
- `sanity_check.py` -- post-prepare data quality report

**Training**
- `train.py` -- multi-class models on CICIDS2017
- `train_binary.py` -- binary (benign vs attack) models on CICIDS2017
- `train_unsw.py` -- multi-class and binary models on UNSW-NB15

**Evaluation**
- `evaluate.py` -- multi-class evaluation (metrics, confusion matrices, ROC/PR curves)
- `evaluate_binary.py` -- binary evaluation (threshold tuning, calibration curves)
- `evaluate_lodo.py` -- leave-one-day-out CV, binary and multi-class (`--task`)
- `evaluate_unsw.py` -- UNSW-NB15 evaluation

**Analysis and experiments**
- `adversarial_eval.py` -- score-query evasion, transferability, naive AT
- `ablation_flow_iat_min.py` -- drop Flow IAT Min, retrain, re-attack
- `calibrate_binary_day.py` -- Platt scaling and isotonic regression
- `operating_points.py` -- recall at fixed FPR operating points
- `anomaly_detection.py` -- unsupervised baselines (IForest, LOF)
- `bootstrap_cis.py` -- bootstrap 95% confidence intervals
- `benchmark_inference.py` -- inference latency and throughput
- `validation.py` -- near-duplicate and split-policy sensitivity checks
**Figures and reporting**
- `thesis_figures.py` -- all thesis figures (white background, print-ready)
- `shap_mcnemar.py` -- SHAP beeswarm/waterfall, McNemar tests, summary tables
- `report.py` -- dark-theme report figures and markdown for GitHub
- `traffic_analysis.py` -- per-day traffic analysis and MITRE tactic distribution
- `mitre_alerts.py` -- MITRE ATT&CK alert generation

---

## Setup

```bash
git clone <repo>
cd bsc-thesis-ids
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the raw datasets:

- CICIDS2017 parquet files in `data/raw/cicids2017/`
- UNSW-NB15 CSV files in `data/unsw-nb15/raw/`

---

## Reproducing All Results

### Full pipeline

```bash
make full
```

Runtime is roughly 2 to 5 hours on a modern laptop. The two slowest stages are `lodo` (20 model fits across 5 folds) and `adversarial` (greedy evasion plus adversarial training set generation). On macOS, prevent lid sleeping with `caffeinate -i make full`.

### Per-stage targets

```bash
make data                # build CICIDS2017 parquet
make sanity              # post-prepare sanity report
make leakage-check       # leakage audit
make train SPLIT=strat   # multi-class training, stratified split
make train SPLIT=day     # multi-class training, day split
make eval  SPLIT=strat   # multi-class evaluation
make eval  SPLIT=day
make binary              # binary models, both splits
make lodo                # LODO binary cross-validation
make lodo-multiclass     # LODO multi-class cross-validation
make unsw-all            # UNSW-NB15 training and evaluation
make figures             # render all figures
make benchmark           # inference latency and throughput
make bootstrap           # 95% CIs for binary metrics
make operating-points    # recall at fixed FPR
make anomaly             # unsupervised baselines
make adversarial         # evasion, transferability, naive AT
make ablation            # Flow IAT Min ablation
make calibrate           # post-processing calibration
make validation          # near-duplicate and split-policy checks
```

---

## Day-Split Configuration

```
Monday    -> train   (Benign only)
Tuesday   -> train   (FTP and SSH brute force)
Wednesday -> train   (DoS Slowloris, Slowhttptest, Heartbleed)
Thursday  -> val     (Web Attacks, Infiltration; threshold tuning only)
Friday    -> test    (DDoS, PortScan, Bot; held-out evaluation)
```

Friday's classes do not appear in training, so the model has to actually generalise.

---

## Key Results

All numbers from `reports/master_results_table.md`.

### Multi-class: the generalisation collapse

| Model | Stratified macro F1 | Day macro F1 | Delta |
|---|---:|---:|---:|
| Logistic Regression | 0.266 | 0.429 | +0.163 |
| Random Forest | 0.845 | 0.438 | -0.408 |
| **XGBoost** | **0.863** | **0.440** | **-0.423** |
| LightGBM | 0.300 | 0.463 | +0.163 |

### Binary: the deployable task

| Model | F1 (stratified) | F1 (day) | ROC-AUC (day) |
|---|---:|---:|---:|
| Logistic Regression | 0.896 | 0.681 | 0.981 |
| **Random Forest** | **0.996** | **0.859** | **0.940** |
| XGBoost | 0.997 | 0.757 | 0.972 |
| LightGBM | 0.998 | 0.744 | 0.957 |

### Leave-one-day-out cross-validation (binary, 4 attack-bearing folds)

| Model | F1 | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.293 | 0.630 |
| **Random Forest** | **0.549** | **0.926 +/- 0.053** |
| XGBoost | 0.437 | 0.921 |
| LightGBM | 0.340 | 0.838 |

Random Forest has the highest mean and the lowest variance, the most stable detector across days with very different attack mixes.

### UNSW-NB15 cross-dataset

| Model | Multi-class F1 | Binary F1 |
|---|---:|---:|
| Logistic Regression | 0.404 | 0.888 |
| Random Forest | 0.479 | 0.921 |
| XGBoost | 0.522 | 0.919 |
| **LightGBM** | **0.548** | 0.921 |

The model ranking flips between datasets. There is no universal best.

### Adversarial robustness (binary RF, day split, 500 sampled attack flows)

| Setting | Evasion rate (unbounded) | Bounded at 0.25 sigma | Clean F1 |
|---|---:|---:|---:|
| Undefended RF | 0.256 | 0.208 | 0.859 |
| Naive adversarial-trained RF | 0.222 | 0.220 | 0.827 |

Transferability: 86.8 percent of RF-adversarial flows also evade XGBoost.

Top exploited features in successful evasions:

| Feature | Percent of successful evasions moving this feature |
|---|---:|
| Flow IAT Min | 84.4 |
| Flow IAT Mean | 21.9 |
| Flow IAT Max | 14.8 |
| Flow Duration | 12.5 |
| Forward IAT Min | 10.2 |

Flow IAT Min ablation (drop the most-exploited feature, retrain, re-attack):

| Quantity | Original RF | Ablated RF |
|---|---:|---:|
| Clean F1 | 0.859 | 0.853 |
| Unbounded evasion | 0.256 | 0.224 |
| Top exploited feature | Flow IAT Min (84.4%) | Forward IAT Min (87.5%) |

Removing the most-exploited feature does not stop the attack; it re-concentrates on the next timing feature. The vulnerability is structural to inter-arrival timing.

### Post-processing calibration

Platt scaling reduces RF expected calibration error from 0.178 to 0.161, but recall at 0.1 percent FPR stays at 0.621. For XGBoost and LightGBM the operating-point recall drops slightly after calibration because the Thursday-to-Friday probability mapping does not transfer.

---

## Live Demo

A working Flask SOC console plus a Wireshark live-capture bridge are in the `demo/` directory. Quick start:

```bash
cd demo
python app.py                       # browser dashboard at http://127.0.0.1:5050
DEFENCE_SCAN=1 ./live_capture.sh    # capture, convert, score live packets
```

See `demo/README.md` for the dashboard documentation and `demo/WIRESHARK.md` for the live-capture playbook including the one-time install steps for tshark, nmap, and the hieulw cicflowmeter fork.

---

## MITRE ATT&CK Coverage

All 14 CICIDS attack classes mapped to ATT&CK techniques across 6 tactics.

| Technique | Attack types | Coverage |
|---|---|---|
| T1110 Brute Force | FTP/SSH/Web Brute Force | Good |
| T1499 Endpoint DoS | DoS Hulk, GoldenEye, Slowhttptest, slowloris | Excellent |
| T1498 Network DoS | DDoS | Excellent |
| T1046 Network Service Scanning | PortScan, Infiltration | Excellent |
| T1071 Application Layer Protocol | Bot (C2) | Partial |
| T1190 Exploit Public-Facing App | XSS, SQL Injection, Heartbleed | Flow-blind (payload-bound) |

---

## Datasets

**CICIDS2017** (primary): 2.31 million flows, 73 features after leakage filtering, 14 attack classes plus Benign, 74.6 percent benign. Four rate-derived features removed before modelling following Engelen et al. 2021.

**UNSW-NB15** (cross-dataset check): 257 thousand flows, 196 features after one-hot encoding, 9 attacks plus Normal, 36.3 percent normal.

---

## Methodology

- Three split protocols: stratified random (60/16/22), day-based temporal (Mon-Wed train / Thu val / Fri test), and leave-one-day-out cross-validation.
- Youden's J threshold tuning on validation, prevalence-invariant.
- Bootstrap 95 percent confidence intervals everywhere, 1000 resamples, percentile method.
- McNemar's test with Yates continuity correction for pairwise model comparisons.
- `random_state=42` throughout.
- Test set is touched exactly once. No hyperparameter selection uses test feedback.

---

## Dependencies

Core runtime: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `shap`, `joblib`, `matplotlib`, `pyarrow`, `scipy`, `flask`. Install with `pip install -r requirements.txt`. Notebook tooling lives in `requirements-dev.txt` and is not needed for `make full`.

For the live capture demo: Wireshark (provides `tshark`), nmap, and the hieulw CICFlowMeter fork (see `demo/WIRESHARK.md`).

---

## Reproducibility

- Python 3.12, tested with 3.12.4
- macOS and Linux (Windows not tested)
- Roughly 4 GB disk and 8 GB RAM
- All experiments use `random_state=42`
- Re-running with a different seed reproduces every headline number within plus or minus 0.5 percent macro F1
