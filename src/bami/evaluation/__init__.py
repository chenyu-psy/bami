"""Model evaluation utilities for recovery, summaries, and plots.

This package exposes:
- contract validators for table schemas used in evaluation plots
- metric helpers for recovery and scalar summaries
- backend-agnostic plotting functions that consume contract tables
"""

from .contracts import (
    REQUIRED_DIAGNOSTIC_COLUMNS,
    REQUIRED_RECOVERY_COLUMNS,
    OPTIONAL_RECOVERY_COLUMNS,
    VALID_LEVELS,
    validate_recovery_contract,
    validate_diagnostic_contract,
)
from .metrics import (
    aggregate_data,
    estimate_recovery,
    compute_ccc,
    compute_corr,
    compute_rmse,
)
from .plots import (
    plot_population_recovery,
    plot_individual_recovery,
    plot_trial_sensitivity_recovery,
    plot_calibration_ecdf,
    plot_coverage,
    plot_zscore_contraction,
)

__all__ = [
    "REQUIRED_DIAGNOSTIC_COLUMNS",
    "REQUIRED_RECOVERY_COLUMNS",
    "OPTIONAL_RECOVERY_COLUMNS",
    "VALID_LEVELS",
    "validate_recovery_contract",
    "validate_diagnostic_contract",
    "aggregate_data",
    "estimate_recovery",
    "compute_ccc",
    "compute_corr",
    "compute_rmse",
    "plot_population_recovery",
    "plot_individual_recovery",
    "plot_trial_sensitivity_recovery",
    "plot_calibration_ecdf",
    "plot_coverage",
    "plot_zscore_contraction",
]
