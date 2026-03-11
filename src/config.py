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

# Day split:
# Train = Mon-Wed: Benign, DoS variants, FTP/SSH brute force, Heartbleed
# Test  = Thu:     Benign, Web Attacks (Brute Force, SQLi, XSS), Infiltration
# Val   = Fri:     Benign, Bot, DDoS, PortScan  (OOD probe — very different)
# This means test shares Benign with train but has unseen attack types.
# The performance drop from strat->day split is the generalisation finding.
DAY_TO_SPLIT = {
    "Monday": "train",
    "Tuesday": "train",
    "Wednesday": "train",
    "Thursday": "test",
    "Friday": "val",
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
