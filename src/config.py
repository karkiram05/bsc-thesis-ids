"""Shared paths, leakage columns, split definitions, and small helpers."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw" / "cicids2017"
PROCESSED_DIR = ROOT / "data" / "processed" / "cicids2017"
DATA_FILE = PROCESSED_DIR / "all_clean.parquet"
META_FILE = PROCESSED_DIR / "meta.md"

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

# Leakage columns - derived rates that leak label info
LEAKAGE_COLUMNS = frozenset({
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packets/s",
    "Bwd Packets/s",
})

# Non-feature columns - never feed these to a model
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

# Day split - temporal ordering: train Mon-Wed, val Thu, test Fri.
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


def feature_cols(df: pd.DataFrame, non_feature: Iterable[str] = NON_FEATURE) -> list[str]:
    return [c for c in df.columns if c not in non_feature]


def uses_internal_scaler(model) -> bool:
    # true when the model is a sklearn Pipeline that already has a "scaler" step
    return hasattr(model, "steps") and "scaler" in [s[0] for s in model.steps]


def safe_transform(le, labels: pd.Series) -> np.ndarray:
    # map unseen labels to -1 so callers can drop them (avoid LabelEncoder crash)
    vals = labels.astype(str)
    known = set(le.classes_.tolist())
    y = np.full(len(vals), -1, dtype=np.int64)
    mask = vals.isin(known)
    if mask.any():
        y[mask.values] = le.transform(vals[mask])
    return y


def binary_y(df: pd.DataFrame) -> np.ndarray:
    return (df["attack_type"].astype(str) != "Benign").astype(int).to_numpy()
