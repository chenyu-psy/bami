# Plots and table contracts

Plot helpers consume the long-format recovery and diagnostic tables produced by
the metric helpers. Contract validators check that those tables have the
required columns.

## Recovery plots

::: bami.evaluation.plots.recovery
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - plot_population_recovery
        - plot_individual_recovery
        - plot_trial_sensitivity_recovery

## Diagnostic plots

::: bami.evaluation.plots.diagnostics
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - plot_calibration_ecdf
        - plot_coverage
        - plot_zscore_contraction

## Table contracts

::: bami.evaluation.contracts
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - REQUIRED_DIAGNOSTIC_COLUMNS
        - REQUIRED_RECOVERY_COLUMNS
        - OPTIONAL_RECOVERY_COLUMNS
        - VALID_LEVELS
        - validate_recovery_contract
        - validate_diagnostic_contract

## Examples

```python
from bami.evaluation.contracts import (
    OPTIONAL_RECOVERY_COLUMNS,
    REQUIRED_DIAGNOSTIC_COLUMNS,
    REQUIRED_RECOVERY_COLUMNS,
    VALID_LEVELS,
    validate_diagnostic_contract,
    validate_recovery_contract,
)
from bami.evaluation.plots.diagnostics import (
    plot_calibration_ecdf,
    plot_coverage,
    plot_zscore_contraction,
)
from bami.evaluation.plots.recovery import (
    plot_individual_recovery,
    plot_population_recovery,
    plot_trial_sensitivity_recovery,
)


validate_recovery_contract(recovery_rows)
validate_diagnostic_contract(diagnostic_rows)

print(REQUIRED_RECOVERY_COLUMNS)
print(OPTIONAL_RECOVERY_COLUMNS)
print(REQUIRED_DIAGNOSTIC_COLUMNS)
print(VALID_LEVELS)

pop_fig = plot_population_recovery(recovery_rows, pars=["c_mu", "kappa_mu"])
ind_fig = plot_individual_recovery(recovery_rows, pars=["c", "kappa"])
trial_fig = plot_trial_sensitivity_recovery(recovery_rows, pars=["c", "kappa"])

cal_fig = plot_calibration_ecdf(diagnostic_rows)
cov_fig = plot_coverage(diagnostic_rows)
z_fig = plot_zscore_contraction(diagnostic_rows)
```
