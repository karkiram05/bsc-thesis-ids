"""Shared config: paths, leakage columns, split and model defaults."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Data
RAW_DIR = ROOT / "data" / "raw" / "cicids2017"
PROCESSED_DIR = ROOT / "data" / "processed" / "cicids2017"
DATA_FILE = PROCESSED_DIR / "all_clean.parquet"
META_FILE = PROCESSED_DIR / "meta.md"

# Leakage / redundant columns (CICFlowMeter-derived rates from duration + totals)
# Flow Bytes/s = total bytes / duration; Flow Packets/s = total packets / duration.
# Excluding these avoids shortcut learning and leakage.
LEAKAGE_COLUMNS = frozenset({
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packets/s",
    "Bwd Packets/s",
})

# Non-feature columns (metadata, targets, splits)
NON_FEATURE = frozenset({
    "Label",
    "is_attack",
    "attack_type",  # normalized multi-class label
    "day",
    "source_file",
    "split",
    "split_day",
    "split_strat",
})

# Splits: by day (no overlap), then optional stratified within train/val/test
DAY_TO_SPLIT = {
    "Monday": "train",
    "Tuesday": "train",
    "Wednesday": "train",
    "Thursday": "val",
    "Friday": "test",
}

RNG = 42

# Outputs
REPORTS_DIR = ROOT / "reports"
BASELINES_DIR = REPORTS_DIR / "baselines"
MODELS_DIR = ROOT / "models"
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"
ALERTS_DIR = REPORTS_DIR / "alerts"

# Train/eval
SPLIT_COL_DAY = "split_day"
SPLIT_COL_STRAT = "split_strat"
DEFAULT_SPLIT_COL = SPLIT_COL_DAY  # by day, no overlap
