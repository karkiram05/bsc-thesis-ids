.PHONY: data train eval report all clean

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

clean:
	rm -rf data/processed reports models
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
