"""Model evaluation utilities shared across BayesFlow and brms workflows.

This package exposes:
- contract validators for table schemas used in evaluation plots
- backend-specific metric builders (BayesFlow implemented, brms placeholders)
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
    brms_pop_recovery,
    brms_ind_recovery,
    brms_calibration,
    brms_coverage,
    brms_zscore,
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
    "plot_population_recovery",
    "plot_individual_recovery",
    "plot_trial_sensitivity_recovery",
    "plot_calibration_ecdf",
    "plot_coverage",
    "plot_zscore_contraction",
]
