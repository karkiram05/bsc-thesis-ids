.PHONY: data train eval report all clean multiclass-baseline leakage-check eval-split-compare

PY ?= python
SPLIT ?= day

data:
	$(PY) -m src.prepare_data

train: data
	$(PY) -m src.train --split $(SPLIT)

eval: train
	$(PY) -m src.eval --split $(SPLIT)

report: eval
	$(PY) -m src.mitre_alerts
	$(PY) -m src.report

all: report

# Multi-class baseline (LogReg + RF), stratified split -> reports/baselines/multiclass_report.md
multiclass-baseline: data
	$(PY) -m src.train_multiclass_baseline --split strat

# Leakage check (exact + near duplicates) -> reports/leakage_check.md
leakage-check: data
	$(PY) -m src.leakage_check

# Strat vs day split compare (train+eval both, leakage check) -> reports/eval_split_compare.md
eval-split-compare: data
	$(PY) -m src.eval_split_compare

clean:
	rm -rf data/processed reports models
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
