# Metrics

Metric helpers return long-format tables for recovery and diagnostics.

## Scalar metrics

::: bami.evaluation.metrics.scalars
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - compute_corr
        - compute_ccc
        - compute_rmse

## BayesFlow metrics

::: bami.evaluation.metrics.bayesflow
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - sample_posterior
        - estimate_population_recovery
        - estimate_fixed_individual_recovery
        - estimate_flex_individual_recovery
        - bf_pop_recovery
        - bf_ind_recovery
        - bf_flex_ind_recovery
        - bf_calibration
        - bf_coverage
        - bf_zscore

## brms metrics

::: bami.evaluation.metrics.brms
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - brms_pop_recovery
        - brms_ind_recovery
        - brms_calibration
        - brms_coverage
        - brms_zscore

## Examples

```python
from bami.evaluation.metrics import compute_ccc, compute_corr, compute_rmse
from bami.evaluation.metrics.bayesflow import (
    bf_calibration,
    bf_coverage,
    bf_flex_ind_recovery,
    bf_ind_recovery,
    bf_pop_recovery,
    bf_zscore,
    estimate_fixed_individual_recovery,
    estimate_flex_individual_recovery,
    estimate_population_recovery,
    sample_posterior,
)


# `model` is a trained SimpleWorkflow or HierarchicalWorkflow object.
# `test_data` is usually produced by model.workflow.simulate(n_datasets).
test_data = model.workflow.simulate(20)

samples = sample_posterior(
    workflow=model.workflow,
    test_data=test_data,
    num_samples=500,
)

population_rows = estimate_population_recovery(
    test_data=test_data,
    samples=samples,
)

truth = population_rows["true_value"]
estimate = population_rows["est_value"]

r = compute_corr(truth, estimate)
ccc = compute_ccc(truth, estimate)
rmse = compute_rmse(truth, estimate)

fixed_individual_rows = estimate_fixed_individual_recovery(
    test_data=test_data,
    samples=samples,
    base_params=["theta"],
)

flex_individual_rows = estimate_flex_individual_recovery(
    model=model,
    test_data=test_data,
    samples=samples,
    base_params=["theta"],
    n_jobs=1,
)

pop_rows = bf_pop_recovery(model.workflow, test_data=20, num_samples=500)
ind_rows = bf_ind_recovery(model=model, test_data=20, num_samples=500)
flex_rows = bf_flex_ind_recovery(model=model, test_data=20, num_group_samples=500)

calibration_rows = bf_calibration(model.workflow, test_data=20, num_samples=500)
coverage_rows = bf_coverage(model.workflow, test_data=20, num_samples=500)
zscore_rows = bf_zscore(model.workflow, test_data=20, num_samples=500)
```

The brms functions define the future shared interface. They currently raise
`NotImplementedError`.

```python
from bami.evaluation.metrics.brms import (
    brms_calibration,
    brms_coverage,
    brms_ind_recovery,
    brms_pop_recovery,
    brms_zscore,
)


# Planned future usage once brms extraction is implemented:
# population_rows = brms_pop_recovery(fit)
# individual_rows = brms_ind_recovery(fit)
# calibration_rows = brms_calibration(fit)
# coverage_rows = brms_coverage(fit)
# zscore_rows = brms_zscore(fit)
```
