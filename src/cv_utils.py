"""Cross-Validation & Time-Series Leakage Prevention Utilities.

Course: Machine Learning (25SC2107E) - Foundation for M4 & M5
Enforces strict chronological boundary validation to prevent look-ahead
and autocorrelation leakage across 15-minute sensor intervals.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, KFold

import config


def get_time_series_splits(df, n_splits=5, date_col="DATE_TIME"):
    """Generate chronological TimeSeriesSplit folds (expanding window).
    
    Guarantees that training data strictly precedes validation data in time,
    eliminating future information leakage in time-dependent solar sensor feeds.
    
    Yields:
        fold_idx (int), train_indices (np.ndarray), val_indices (np.ndarray)
    """
    data = df.sort_values(by=date_col).reset_index(drop=True)
    n = len(data)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(data)):
        # Verify strict temporal ordering
        assert_temporal_boundary(data[date_col].iloc[train_idx], data[date_col].iloc[val_idx])
        yield fold + 1, train_idx, val_idx


def get_blocked_time_splits(df, n_splits=5, date_col="DATE_TIME"):
    """Generate contiguous, non-overlapping chronological block folds.
    
    Useful for assessing performance across distinct calendar regimes without
    growing window effects.
    """
    data = df.sort_values(by=date_col).reset_index(drop=True)
    n = len(data)
    block_size = n // n_splits
    
    for fold in range(n_splits):
        val_start = fold * block_size
        val_end = n if fold == n_splits - 1 else (fold + 1) * block_size
        val_idx = np.arange(val_start, val_end)
        train_idx = np.setdiff1d(np.arange(n), val_idx)
        yield fold + 1, train_idx, val_idx


def assert_temporal_boundary(train_dates, val_dates):
    """Assert that the latest timestamp in training strictly precedes the earliest validation timestamp."""
    max_train = pd.to_datetime(train_dates).max()
    min_val = pd.to_datetime(val_dates).min()
    if max_train >= min_val:
        raise ValueError(
            f"Temporal leakage detected! Training max date ({max_train}) "
            f"overlaps or exceeds Validation min date ({min_val})."
        )
    return True


def audit_cross_validation_leakage(train_df, n_splits=5):
    """Audit and compare Standard K-Fold (with shuffle) vs TimeSeriesSplit.
    
    Returns an empirical report quantifying timestamp overlap in standard random splits.
    """
    df = train_df.sort_values(by="DATE_TIME").reset_index(drop=True)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    
    shuffled_overlaps = 0
    for fold, (tr_idx, va_idx) in enumerate(kf.split(df)):
        tr_dates = pd.to_datetime(df["DATE_TIME"].iloc[tr_idx])
        va_dates = pd.to_datetime(df["DATE_TIME"].iloc[va_idx])
        
        # Check how many validation timestamps fall between min and max training timestamps
        interleaved = va_dates[(va_dates >= tr_dates.min()) & (va_dates <= tr_dates.max())]
        shuffled_overlaps += len(interleaved)
        
    overlap_pct = round(shuffled_overlaps / (len(df) * n_splits) * 100, 2)
    
    return {
        "n_splits": n_splits,
        "shuffled_kfold_overlap_count": shuffled_overlaps,
        "shuffled_kfold_overlap_pct": overlap_pct,
        "time_series_split_overlap_pct": 0.0,
        "conclusion": (
            f"Randomly shuffled K-Fold leaks temporal future context for {overlap_pct}% of validation instances. "
            "TimeSeriesSplit enforces 0.0% overlap."
        )
    }
