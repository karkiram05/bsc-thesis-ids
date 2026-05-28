.PHONY: data train eval report all clean leakage-check sanity
.PHONY: unsw-data unsw-train unsw-eval unsw-all
.PHONY: binary binary-strat binary-day lodo lodo-multiclass validation
.PHONY: figures thesis-figures shap-mcnemar traffic
.PHONY: benchmark bootstrap operating-points anomaly adversarial ablation calibrate full

PY ?= python
SPLIT ?= day

data:
	$(PY) -m src.prepare_cicids

sanity: data
	$(PY) -m src.sanity_check

train: data
	$(PY) -m src.train --split $(SPLIT)

eval: train
	$(PY) -m src.evaluate --split $(SPLIT)

report: eval
	$(PY) -m src.mitre_alerts
	$(PY) -m src.report

all: report

leakage-check: data
	$(PY) -m src.leakage_check

binary-strat: data
	$(PY) -m src.train_binary --split strat --out-dir models/binary_strat
	$(PY) -m src.evaluate_binary --split strat --models-dir models/binary_strat --out-dir reports/metrics_strat_binary

binary-day: data
	$(PY) -m src.train_binary --split day --out-dir models/binary_day
	$(PY) -m src.evaluate_binary --split day --models-dir models/binary_day --out-dir reports/metrics_day_binary

binary: binary-strat binary-day

lodo: data
	$(PY) -m src.evaluate_lodo --task binary

lodo-multiclass: data
	$(PY) -m src.evaluate_lodo --task multiclass

validation: data
	$(PY) -m src.validation

unsw-data:
	$(PY) -m src.prepare_unsw

unsw-train: unsw-data
	$(PY) -m src.train_unsw --task multiclass
	$(PY) -m src.train_unsw --task binary

unsw-eval: unsw-train
	$(PY) -m src.evaluate_unsw --task multiclass
	$(PY) -m src.evaluate_unsw --task binary

unsw-all: unsw-eval

shap-mcnemar:
	$(PY) -m src.shap_mcnemar

traffic:
	$(PY) -m src.traffic_analysis

thesis-figures:
	$(PY) -m src.thesis_figures

figures: shap-mcnemar traffic thesis-figures

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

ablation:
	$(PY) -m src.ablation_flow_iat_min

calibrate:
	$(PY) -m src.calibrate_binary_day

full: data sanity leakage-check
	$(PY) -m src.train --split strat --out-dir models/baseline_strat
	$(PY) -m src.train --split day   --out-dir models/baseline_day
	$(PY) -m src.evaluate --split strat --models-dir models/baseline_strat --out-dir reports/metrics_strat
	$(PY) -m src.evaluate --split day   --models-dir models/baseline_day   --out-dir reports/metrics_day
	$(MAKE) binary
	$(MAKE) lodo
	$(MAKE) lodo-multiclass
	$(MAKE) unsw-all
	$(PY) -m src.mitre_alerts --metrics-dir reports/metrics_strat --out-dir reports/alerts_strat
	$(PY) -m src.mitre_alerts --metrics-dir reports/metrics_day   --out-dir reports/alerts_day
	$(PY) -m src.report --metrics-dir reports/metrics_strat --alerts-dir reports/alerts_strat
	$(MAKE) validation
	$(MAKE) figures
	$(MAKE) benchmark
	$(MAKE) bootstrap
	$(MAKE) operating-points
	$(MAKE) anomaly
	$(MAKE) adversarial
	$(MAKE) ablation
	$(MAKE) calibrate

clean:
	rm -rf data/processed reports models
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
