"""Model evaluation utilities for summaries and recovery metrics.

This package exposes:
- dataframe-first helpers for aggregating simulated or posterior samples
- scalar metrics for parameter recovery checks
"""

from .metrics import (
    aggregate_data,
    estimate_recovery,
    compute_ccc,
    compute_corr,
    compute_rmse,
)

__all__ = [
    "aggregate_data",
    "estimate_recovery",
    "compute_ccc",
    "compute_corr",
    "compute_rmse",
]
