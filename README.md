# Flow-Based Machine Learning for Network Intrusion Detection with MITRE ATT&CK Mapping

**Bachelor Thesis** — Ram Karki (S236176) — General Engineering, DTU — 15 ECTS
**Supervisor**: Gaurav Choudhary, Department of Applied Mathematics and Computer Science

---

## Overview

This project builds and evaluates a reproducible intrusion detection pipeline on two public datasets: **CICIDS2017** (primary) and **UNSW-NB15** (cross-dataset check). Four machine-learning models — Logistic Regression, Random Forest, XGBoost, and LightGBM — are trained for flow-level attack detection (multi-class and binary) under two evaluation protocols (random stratified and strict temporal day-split). The pipeline also runs Leave-One-Day-Out cross-validation, a constrained adversarial-robustness evaluation, and maps every detected attack to MITRE ATT&CK to produce SOC-actionable outputs.

An **unsupervised anomaly-detection** baseline (Isolation Forest, Local Outlier Factor) is included to measure how much the supervised approach earns over the label-free reference.

---

## Headline Findings

1. **Multi-class classification collapses under temporal evaluation.** XGBoost macro-F1 falls from 0.863 (stratified split) to 0.440 (day split) — a 51 % relative drop on the same data, model, and hyperparameters.
2. **Binary detection survives.** Random Forest reaches F1 0.859 under the day split (vs 0.996 stratified), making attack-vs-benign the deployable task.
3. **Random Forest is the most stable detector.** LODO ROC-AUC 0.926 ± 0.053 across the five capture days.
4. **The model is brittle under adversarial pressure.** A constrained greedy attack evades the undefended Random Forest 24.4 % of the time at a median budget of 0.25σ. A naïve single-round adversarial-training defence fails: clean F1 drops to 0.73 and evasion rises to 26.8 %.
5. **Model secrecy is not a defence.** 86.4 % of adversarial flows crafted against Random Forest also evade XGBoost.

---

## Project Structure

```
bsc-thesis-ids/
├── data/
│   ├── raw/cicids2017/             # raw CICFlowMeter parquet files (gitignored)
│   ├── unsw-nb15/raw/              # UNSW-NB15 CSV files (gitignored)
│   ├── processed/cicids2017/       # cleaned CICIDS2017 (built by prepare_data)
│   └── processed/unsw-nb15/        # cleaned UNSW-NB15 (built by prepare_unsw)
├── models/
│   ├── baseline_strat/             # multi-class models, stratified split
│   ├── baseline_day/               # multi-class models, day split
│   ├── binary_strat/               # binary models, stratified split
│   ├── binary_day/                 # binary models, day split
│   └── unsw/                       # UNSW-NB15 models (multi-class + binary)
├── reports/
│   ├── figures/                    # all PNGs: confusion matrix, SHAP, ROC, calibration, etc.
│   ├── traffic_analysis/           # security-engineering documents and plots
│   ├── metrics_strat/              # multi-class metrics, stratified split
│   ├── metrics_day/                # multi-class metrics, day split
│   ├── metrics_strat_binary/       # binary metrics, stratified split
│   ├── metrics_day_binary/         # binary metrics, day split
│   ├── metrics_unsw/               # UNSW-NB15 metrics
│   ├── lodo/                       # leave-one-day-out cross-validation
│   ├── adversarial/                # evasion attack, transferability, AT defence results
│   ├── bootstrap/                  # 95 % CIs for binary metrics
│   ├── operating_points/           # recall at fixed FPR + ECE calibration
│   ├── benchmark/                  # inference latency / throughput
│   ├── anomaly/                    # unsupervised baseline metrics
│   ├── validation/                 # V1 near-dup / V2 split policy / V3 LODO sensitivity
│   ├── alerts_strat/               # MITRE ATT&CK alerts, stratified split
│   ├── alerts_day/                 # MITRE ATT&CK alerts, day split
│   ├── master_results_table.md     # all CICIDS + UNSW + LODO numbers in one place
│   ├── eval_split_compare.md       # stratified vs day split delta report
│   ├── statistical_tests.md        # McNemar pairwise comparisons
│   ├── leakage_check.md            # 5-tuple and feature-target leakage audit
│   ├── sanity_check.md             # post-prepare data sanity report
│   ├── hyperparameter_table.md     # full hyperparameter spec
│   └── run_manifest.json           # seed, git commit, timestamp for reproducibility
└── src/
    ├── config.py                       # shared paths, constants, helpers
    ├── prepare_data.py                 # CICIDS2017 loading, cleaning, splits
    ├── prepare_unsw.py                 # UNSW-NB15 loading, cleaning, splits
    ├── sanity_check.py                 # post-prepare sanity (split sizes, balance, leakage)
    ├── leakage_check.py                # exact and near-duplicate leakage audit
    ├── train.py                        # multi-class training (LogReg, RF, XGB, LGB)
    ├── train_binary.py                 # binary training (Benign vs Attack)
    ├── train_unsw.py                   # UNSW-NB15 training
    ├── eval.py                         # multi-class evaluation
    ├── eval_binary.py                  # binary evaluation with threshold tuning
    ├── eval_unsw.py                    # UNSW-NB15 evaluation
    ├── eval_lodo.py                    # leave-one-day-out cross-validation
    ├── eval_split_compare.py           # stratified vs day split comparison
    ├── run_validation_experiments.py   # V1 / V2 / V3 robustness experiments
    ├── adversarial_eval.py             # greedy evasion + transferability + adv training
    ├── mitre_alerts.py                 # CICIDS class → ATT&CK technique mapping
    ├── mitre_mapping.json              # ATT&CK mapping data
    ├── traffic_analysis.py             # security-engineering analysis and figures
    ├── report.py                       # report and figure generation
    ├── generate_extra_figures.py       # LODO, calibration, generalisation gap
    ├── generate_defense_figures.py     # day distribution, binary-vs-multi, threshold transfer
    ├── generate_shap_mcnemar.py        # SHAP + McNemar significance
    ├── benchmark_inference.py          # inference latency / throughput
    ├── bootstrap_cis.py                # bootstrap 95 % CIs for binary metrics
    ├── operating_points.py             # recall at fixed FPR budgets + ECE calibration
    └── anomaly_detection.py            # unsupervised baselines (Isolation Forest, LOF)
```

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

- **CICIDS2017** parquet files in `data/raw/cicids2017/` (download from <https://www.unb.ca/cic/datasets/ids-2017.html>)
- **UNSW-NB15** CSV files in `data/unsw-nb15/raw/` (download from <https://research.unsw.edu.au/projects/unsw-nb15-dataset>)

---

## Reproducing All Results

### Option A — full pipeline

```bash
make full
```

This runs every stage end-to-end and writes every output. Total runtime is roughly **2-5 hours** on a modern laptop with the lid open. The two slowest stages are `eval_lodo` (20 model fits across 5 folds) and `adversarial` (greedy evasion + 2000-sample adversarial training set generation). Keep the laptop awake — if it sleeps, the wall-clock time stretches even though no real work happens. On macOS, prevent sleep with:

```bash
caffeinate -i make full
```

### Option B — targets one at a time

```bash
make data                # build CICIDS2017 features
make sanity              # post-prepare sanity report
make leakage-check       # 5-tuple + feature-target leakage audit
make train SPLIT=strat   # multi-class training, stratified split
make train SPLIT=day     # multi-class training, day split
make eval  SPLIT=strat   # multi-class evaluation
make eval  SPLIT=day
make binary              # binary models, both splits
make lodo                # leave-one-day-out cross-validation
make unsw-all            # UNSW-NB15 training + evaluation
make validation          # V1 / V2 / V3 experiments
make figures             # render all PNGs into reports/figures/
make benchmark           # inference latency / throughput
make bootstrap           # 95 % CIs for binary metrics
make operating-points    # recall at fixed FPR + ECE calibration
make anomaly             # unsupervised baseline
make adversarial         # evasion + transferability + adversarial training
```

---

## Training Process

The training pipeline has three stages.

**Stage 1 — Data preparation** (`src/prepare_data.py`):
1. Read the eight CICIDS CSV files (one per attack day).
2. Strip whitespace from column names; standardise label casing.
3. Drop rows with infinite or NaN values (sensor artefacts).
4. Drop the four leakage-prone rate features (`Flow Bytes/s`, `Flow Packets/s`, `Fwd Packets/s`, `Bwd Packets/s`).
5. Map raw labels to a 14-class taxonomy.
6. Deduplicate by 5-tuple + start-time.
7. Save a single Parquet file with `Day` and `Label` columns.

**Stage 2 — Model fitting** (`src/train.py`, `src/train_binary.py`, `src/train_unsw.py`):
- For each of the four models, fit on the training partition with the hyperparameters in `reports/hyperparameter_table.md`.
- Tree models train on raw features; LogReg uses a `StandardScaler` pipeline.
- Saved as joblib pickles in `models/{baseline,binary}_{strat,day}/`.

**Stage 3 — Evaluation** (`src/eval.py`, `src/eval_binary.py`):
- Tune the binary decision threshold on the validation partition using **Youden's J statistic** (prevalence-invariant).
- Evaluate on the test partition.
- Report precision, recall, F1, FPR, ROC-AUC, PR-AUC.
- Bootstrap 95 % CIs (1000 resamples) on every headline number.
- McNemar's test for pairwise model comparisons.

The **test set is touched exactly once** — no hyperparameter selection uses test feedback.

---

## Day-Split Configuration

```
Monday    → train   (Benign baseline only)
Tuesday   → train   (FTP / SSH brute force)
Wednesday → train   (DoS Slowloris/Slowhttptest, Heartbleed)
Thursday  → val     (Web Attacks, Infiltration) — threshold tuning only
Friday    → test    (Bot, PortScan, DDoS) — held-out evaluation
```

The day split is the honest evaluation. Friday's classes never appear in training, so the model has to actually generalise.

---

## Key Results

All numbers below come from `reports/master_results_table.md`. F1 is macro-F1 for multi-class and the standard binary F1 for binary detection.

### Multi-class — the generalisation collapse

| Model | Macro-F1 (stratified) | Macro-F1 (day) | Δ |
|---|---:|---:|---:|
| Logistic Regression | 0.266 | 0.429 | +0.163 |
| Random Forest | 0.845 | 0.438 | −0.408 |
| **XGBoost** | **0.863** | **0.440** | **−0.423** |
| LightGBM | 0.300 | 0.463 | +0.163 |

The collapse is consistent across the well-fitting models (RF, XGBoost). Logistic Regression and LightGBM happen to do *better* on the day split because they were already weak on stratified — they had little to lose.

### Binary — the deployable task

| Model | F1 (stratified) | F1 (day) | ROC-AUC (day) | Δ F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.896 | 0.681 | 0.981 | −0.215 |
| **Random Forest** | **0.996** | **0.859** | **0.940** | **−0.137** |
| XGBoost | 0.997 | 0.757 | 0.972 | −0.240 |
| LightGBM | 0.998 | 0.744 | 0.957 | −0.254 |

Random Forest survives the day split with F1 0.859 — the strongest binary detector by a clear margin.

### Leave-One-Day-Out cross-validation (binary, mean ± std over 5 folds)

| Model | Macro-F1 | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.293 | 0.630 |
| **Random Forest** | **0.549** | **0.926 ± 0.053** |
| XGBoost | 0.437 | 0.921 |
| LightGBM | 0.340 | 0.838 |

Random Forest is both highest mean and lowest variance — the most stable detector across days.

### UNSW-NB15 cross-dataset

| Model | Multi-class F1 | Binary F1 | Binary ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.404 | 0.888 | 0.976 |
| Random Forest | 0.479 | 0.921 | 0.986 |
| XGBoost | 0.522 | 0.919 | 0.986 |
| **LightGBM** | **0.548** | 0.921 | 0.986 |

The model ranking flips between datasets — LightGBM wins UNSW, Random Forest wins CICIDS day-split. There is no universal best model; tune on local traffic before deployment.

### Adversarial robustness (binary RF, day split, 500 sampled attack flows)

| Setting | Evasion rate | Median L∞ to flip (σ) | Clean F1 |
|---|---:|---:|---:|
| Undefended RF | 0.244 | 0.25 | 0.859 |
| Naïve adversarially-trained RF | 0.268 | — | 0.734 |

| Transferability | XGBoost evasion rate |
|---|---:|
| Clean attack flows scored by XGBoost | 0.260 |
| RF-adversarial flows scored by XGBoost | **0.864** |

Top exploited features (fraction of successful evasions that moved each one):

| Feature | % |
|---|---:|
| Flow IAT Min | 84 |
| Flow IAT Mean | 18 |
| Flow IAT Max | 16 |
| Fwd IAT Min | 13 |
| Flow Duration | 9 |

Single-round adversarial training **fails as a defence** — both clean F1 and evasion rate get worse. Reproduces the negative result in Madry et al. (2018). Robust adversarial training would need a multi-round PGD-style inner attacker, which is out of scope for this BSc.

---

## Outputs

| Path | Contents |
|---|---|
| `reports/master_results_table.md` | every headline F1 / ROC-AUC in one table |
| `reports/eval_split_compare.md` | stratified vs day-split delta per model |
| `reports/statistical_tests.md` | McNemar p-values for binary day-split |
| `reports/bootstrap/binary_day_cis.md` | 95 % CIs on binary metrics |
| `reports/operating_points/operating_points.md` | F1-optimal vs FPR-constrained thresholds |
| `reports/lodo/lodo_summary.md` | per-fold LODO results |
| `reports/adversarial/adversarial_results.md` | evasion, transferability, top features |
| `reports/anomaly/anomaly_metrics.md` | Isolation Forest / LOF baselines |
| `reports/benchmark/inference_latency.md` | inference latency / throughput per model |
| `reports/validation/exp1/`, `exp2/`, `exp3/` | V1 near-dup, V2 split policy, V3 LODO sensitivity |
| `reports/hyperparameter_table.md` | full hyperparameter spec |
| `reports/leakage_check.md` | leakage audit |
| `reports/sanity_check.md` | post-prepare data sanity |
| `reports/figures/` | all PNGs (confusion matrix, SHAP, ROC, calibration, adversarial robustness, etc.) |
| `reports/traffic_analysis/` | per-attack flow stats, Sigma rules, MITRE tactic profile, SOC playbook, feature separability |
| `reports/alerts_strat/`, `alerts_day/` | MITRE ATT&CK alerts in CSV + JSON |

---

## MITRE ATT&CK Coverage

All 14 attack classes mapped to ATT&CK techniques across 6 tactics. The flow-only IDS covers timing and volume techniques reliably and is honestly blind to payload-bound techniques.

| Technique | Attack types | Per-class F1 (binary day split) | Coverage |
|---|---|---|---|
| T1110 Brute Force | FTP/SSH/Web Brute Force | 0.78 – 0.93 | Good |
| T1499 Endpoint DoS | DoS Hulk, GoldenEye, Slowhttptest, slowloris | 0.96 | Excellent |
| T1498 Network DoS | DDoS | 0.99 | Excellent |
| T1046 Network Service Scanning | PortScan | 0.99 | Excellent |
| T1071 Application Layer Protocol | Bot (C2) | 0.55 | Partial |
| T1190 Exploit Public-Facing App | XSS, SQL Injection | 0.05 – 0.21 | **Flow-blind** (payload) |
| T1212 Exploitation for Cred Access | Heartbleed | 0.00 | **Flow-blind** (rare class) |
| T1190 / T1133 | Infiltration | 0.00 | **Blind** |

Honest reporting of blind spots is the whole point of the mapping — the rest of the defence-in-depth stack (WAF, EDR) covers what flow IDS cannot.

---

## Datasets

**CICIDS2017** (primary) — Canadian Institute for Cybersecurity
~2.3 M flows | 73 features | 14 attack classes + Benign | 85.5 % benign

Leakage columns dropped before modelling: `Flow Bytes/s`, `Flow Packets/s`, `Fwd Packets/s`, `Bwd Packets/s` (algebraically derived; would allow shortcut learning).

Near-duplicate audit found ~28 k flow groups crossing the day-split boundary and ~26 k crossing the stratified boundary. Documented in `reports/leakage_check.md`.

**UNSW-NB15** (cross-dataset check) — UNSW Canberra
~257 k flows | 196 features after one-hot | 9 attacks + Normal | 36.3 % normal

---

## Methodology Choices

- **No hyperparameter search.** A small grid was used to find a configuration that is not actively bad. Larger searches over-fit to the validation set.
- **Youden's J for binary thresholds.** Prevalence-invariant; transfers across train/test more cleanly than F1-optimal thresholds.
- **Bootstrap CIs everywhere.** 1000 resamples, percentile method.
- **McNemar for pairwise comparisons.** Settles the question "is this 0.01 F1 difference real?".
- **Tree models on raw features.** Only LogReg uses a `StandardScaler` pipeline.
- **`random_state=42`** throughout for determinism.

---

## Dependencies

Core runtime: `pandas`, `numpy`, `scikit-learn`, `xgboost>=2.0`, `lightgbm`, `shap`, `joblib`, `matplotlib`, `pyarrow`, `scipy`.

Install with `pip install -r requirements.txt`. JupyterLab and other notebook-only tooling lives in `requirements-dev.txt` and is not needed to run `make full`.

---

## Reproducibility

- Python 3.12 (tested with 3.12.4)
- macOS / Linux (Windows not tested)
- ~4 GB free disk for data + models, ~8 GB RAM recommended
- All experiments use `random_state=42`
- `reports/run_manifest.json` records the seed, the git commit, and the run timestamp
- Re-running the entire pipeline with a different seed reproduces every headline number within ±0.5 % macro-F1

---

## Notes

- XGBoost on the day split needs label remapping (saved as `xgboost_label_remap.json`) because training classes are non-contiguous when attack types are split by day.
- LightGBM on multi-class CICIDS does poorly under the stratified split (macro-F1 0.30) because `is_unbalance=True` over-corrects under extreme class imbalance — illustrated in `reports/eval_split_compare.md`.
- Probability calibration (`reports/figures/calibration_curves.png`) shows tree ensembles are over-confident; isotonic post-hoc calibration is recommended before raw scores feed downstream SIEM rules.
- SHAP per-flow attributions are pre-rendered in `reports/figures/shap_*.png` and used in the SOC integration walkthrough described in the thesis.
