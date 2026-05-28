"""Metric-computation entrypoints for supported inference backends."""

from .scalars import compute_ccc, compute_corr, compute_rmse
from .bayesflow import (
    bf_pop_recovery,
    bf_ind_recovery,
    bf_flex_ind_recovery,
    sample_posterior,
    estimate_population_recovery,
    estimate_fixed_individual_recovery,
    estimate_flex_individual_recovery,
    bf_calibration,
    bf_coverage,
    bf_zscore,
)
from .brms import (
    brms_pop_recovery,
    brms_ind_recovery,
    brms_calibration,
    brms_coverage,
    brms_zscore,
)

__all__ = [
    "compute_ccc",
    "compute_corr",
    "compute_rmse",
    "bf_pop_recovery",
    "bf_ind_recovery",
    "bf_flex_ind_recovery",
    "sample_posterior",
    "estimate_population_recovery",
    "estimate_fixed_individual_recovery",
    "estimate_flex_individual_recovery",
    "bf_calibration",
    "bf_coverage",
    "bf_zscore",
    "brms_pop_recovery",
    "brms_ind_recovery",
    "brms_calibration",
    "brms_coverage",
    "brms_zscore",
]
