"""Shared config: paths, leakage columns, split and model defaults."""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ── CICIDS2017 ────────────────────────────────────────────────────────
RAW_DIR = ROOT / "data" / "raw" / "cicids2017"
PROCESSED_DIR = ROOT / "data" / "processed" / "cicids2017"
DATA_FILE = PROCESSED_DIR / "all_clean.parquet"
META_FILE = PROCESSED_DIR / "meta.md"

# ── UNSW-NB15 ─────────────────────────────────────────────────────────
UNSW_RAW_DIR = ROOT / "data" / "unsw-nb15" / "raw"
UNSW_PROCESSED_DIR = ROOT / "data" / "processed" / "unsw-nb15"
UNSW_DATA_FILE = UNSW_PROCESSED_DIR / "all_clean.parquet"
UNSW_META_FILE = UNSW_PROCESSED_DIR / "meta.md"

# Non-feature columns in UNSW-NB15
UNSW_NON_FEATURE = frozenset({
    "id", "attack_cat", "label", "split",
})

# UNSW-NB15 categorical columns
UNSW_CATEGORICAL = ["proto", "service", "state"]

# Leakage columns — derived rates that leak label info
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

# Day split — temporal ordering: train Mon-Wed, val Thu, test Fri.
# Attack types are disjoint across days, so multi-class fails on unseen classes.
# Binary (benign vs attack) still works.
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

# UNSW-NB15 output dirs
UNSW_MODELS_DIR = ROOT / "models" / "unsw"
UNSW_METRICS_DIR = ROOT / "reports" / "metrics_unsw"

# Split column names
SPLIT_COL_DAY = "split_day"
SPLIT_COL_STRAT = "split_strat"
DEFAULT_SPLIT_COL = SPLIT_COL_DAY
