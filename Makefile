.PHONY: data train eval report all clean leakage-check eval-split-compare
.PHONY: unsw-data unsw-train unsw-eval unsw-all

PY ?= python
SPLIT ?= day

# ── CICIDS2017 ────────────────────────────────────────────────────────
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

# Leakage check (exact + near duplicates) -> reports/leakage_check.md
leakage-check: data
	$(PY) -m src.leakage_check

# Strat vs day split compare (train+eval both, leakage check) -> reports/eval_split_compare.md
eval-split-compare: data
	$(PY) -m src.eval_split_compare

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

# ── Validation experiments ─────────────────────────────────────────────
validation: data
	$(PY) -m src.run_validation_experiments

# ── Cleanup ────────────────────────────────────────────────────────────
clean:
	rm -rf data/processed reports models
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
