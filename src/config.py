"""Shared config: paths, leakage columns, split and model defaults."""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Data
RAW_DIR = ROOT / "data" / "raw" / "cicids2017"
PROCESSED_DIR = ROOT / "data" / "processed" / "cicids2017"
DATA_FILE = PROCESSED_DIR / "all_clean.parquet"
META_FILE = PROCESSED_DIR / "meta.md"

# Leakage columns — CICFlowMeter-derived rates, exclude to avoid shortcut learning
LEAKAGE_COLUMNS = frozenset({
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packets/s",
    "Bwd Packets/s",
})

# Non-feature columns — never feed these to a model
NON_FEATURE = frozenset({
    "Label",
    "is_attack",
    "attack_type",
    "day",
    "source_file",
    "split",
    "split_day",
    "split_strat",
    "row_hash",
})

# Day split — temporal ordering preserved:
#   Train = Mon-Wed  (Benign, DoS variants, FTP/SSH brute force, Heartbleed)
#   Val   = Thu      (Benign, Web Attacks, Infiltration) — threshold tuning
#   Test  = Fri      (Benign, Bot, DDoS, PortScan) — final held-out evaluation
#
# RATIONALE: In deployment, the model is trained on past data and evaluated on
# *future* data.  Thursday (day 4) comes before Friday (day 5), so we tune
# thresholds on Thursday and report final metrics on Friday.  This is the
# standard temporal-validation protocol.
#
# NOTE: attack types are disjoint across days.  Multi-class evaluation under
# this split is expected to fail on unseen classes.  Binary (benign vs attack)
# remains meaningful.
DAY_TO_SPLIT = {
    "Monday": "train",
    "Tuesday": "train",
    "Wednesday": "train",
    "Thursday": "val",
    "Friday": "test",
}

RNG = 42

# Output dirs
REPORTS_DIR = ROOT / "reports"
BASELINES_DIR = REPORTS_DIR / "baselines"
MODELS_DIR = ROOT / "models"
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = ROOT / "reports" / "figures"
ALERTS_DIR = REPORTS_DIR / "alerts"

# Split column names
SPLIT_COL_DAY = "split_day"
SPLIT_COL_STRAT = "split_strat"
DEFAULT_SPLIT_COL = SPLIT_COL_DAY
