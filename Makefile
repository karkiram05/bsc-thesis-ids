.PHONY: data train eval report all clean leakage-check eval-split-compare sanity
.PHONY: unsw-data unsw-train unsw-eval unsw-all
.PHONY: binary binary-strat binary-day lodo validation
.PHONY: figures shap-mcnemar defense-figures extra-figures traffic
.PHONY: benchmark bootstrap operating-points anomaly adversarial full

PY ?= python
SPLIT ?= day

# ── CICIDS2017 ────────────────────────────────────────────────────────
data:
	$(PY) -m src.prepare_data

sanity: data
	$(PY) -m src.sanity_check

train: data
	$(PY) -m src.train --split $(SPLIT)

eval: train
	$(PY) -m src.eval --split $(SPLIT)

report: eval
	$(PY) -m src.mitre_alerts
	$(PY) -m src.report

all: report

leakage-check: data
	$(PY) -m src.leakage_check

eval-split-compare: data
	$(PY) -m src.eval_split_compare

# ── Binary models (both splits) ──────────────────────────────────────
binary-strat: data
	$(PY) -m src.train_binary --split strat --out-dir models/binary_strat
	$(PY) -m src.eval_binary  --split strat --models-dir models/binary_strat --out-dir reports/metrics_strat_binary

binary-day: data
	$(PY) -m src.train_binary --split day --out-dir models/binary_day
	$(PY) -m src.eval_binary  --split day --models-dir models/binary_day --out-dir reports/metrics_day_binary

binary: binary-strat binary-day

# ── LODO cross-validation ────────────────────────────────────────────
lodo: data
	$(PY) -m src.eval_lodo

# ── Validation experiments (V1, V2, V3) ──────────────────────────────
validation: data
	$(PY) -m src.run_validation_experiments

# ── UNSW-NB15 ─────────────────────────────────────────────────────────
unsw-data:
	$(PY) -m src.prepare_unsw

unsw-train: unsw-data
	$(PY) -m src.train_unsw --task multiclass
	$(PY) -m src.train_unsw --task binary

unsw-eval: unsw-train
	$(PY) -m src.eval_unsw --task multiclass
	$(PY) -m src.eval_unsw --task binary

unsw-all: unsw-eval

# ── Figures + analysis ───────────────────────────────────────────────
extra-figures:
	$(PY) -m src.generate_extra_figures

defense-figures:
	$(PY) -m src.generate_defense_figures

shap-mcnemar:
	$(PY) -m src.generate_shap_mcnemar

traffic:
	$(PY) -m src.traffic_analysis

figures: extra-figures defense-figures shap-mcnemar traffic

# ── Deployment / operational metrics (cybersec engineer view) ───────
benchmark:
	$(PY) -m src.benchmark_inference

bootstrap:
	$(PY) -m src.bootstrap_cis

operating-points:
	$(PY) -m src.operating_points

anomaly:
	$(PY) -m src.anomaly_detection

adversarial:
	$(PY) -m src.adversarial_eval

# ── Full reproducible pipeline ──────────────────────────────────────
full: data sanity leakage-check
	$(PY) -m src.train --split strat --out-dir models/baseline_strat
	$(PY) -m src.train --split day   --out-dir models/baseline_day
	$(PY) -m src.eval  --split strat --models-dir models/baseline_strat --out-dir reports/metrics_strat
	$(PY) -m src.eval  --split day   --models-dir models/baseline_day   --out-dir reports/metrics_day
	$(MAKE) binary
	$(MAKE) lodo
	$(MAKE) unsw-all
	$(PY) -m src.mitre_alerts --metrics-dir reports/metrics_strat --out-dir reports/alerts_strat
	$(PY) -m src.mitre_alerts --metrics-dir reports/metrics_day   --out-dir reports/alerts_day
	$(PY) -m src.report --metrics-dir reports/metrics_strat --alerts-dir reports/alerts_strat
	$(PY) -m src.eval_split_compare --skip-train
	$(MAKE) validation
	$(MAKE) figures   # this target already runs src.traffic_analysis as a sub-step
	$(MAKE) benchmark
	$(MAKE) bootstrap
	$(MAKE) operating-points
	$(MAKE) anomaly
	$(MAKE) adversarial

# ── Cleanup ──────────────────────────────────────────────────────────
clean:
	rm -rf data/processed reports models
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
