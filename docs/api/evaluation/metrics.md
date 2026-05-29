# Metrics

Metric helpers return long-format tables for summaries and recovery checks.

## Data aggregation

::: bami.evaluation.metrics.aggregate
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - aggregate_data

## Recovery metrics

::: bami.evaluation.metrics.recovery
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - estimate_recovery

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

## Examples

```python
import pandas as pd

from bami.evaluation.metrics import (
    aggregate_data,
    compute_ccc,
    compute_corr,
    compute_rmse,
    estimate_recovery,
)


# Simulated and estimated values are long-format tables.
simulated_values = pd.DataFrame(
    {
        "dataset_id": [0, 1, 0, 1],
        "param": ["c", "c", "kappa", "kappa"],
        "simulated_value": [0.8, 1.2, 2.0, 3.0],
    }
)
estimated_values = pd.DataFrame(
    {
        "dataset_id": [0, 1, 0, 1],
        "param": ["c", "c", "kappa", "kappa"],
        "estimated_value": [0.9, 1.1, 2.2, 2.8],
    }
)
recovery_metrics = estimate_recovery(
    simulated_values,
    estimated_values,
    group_by="param",
    metrics=["ccc", "rmse"],
)

value_summary = aggregate_data(
    estimated_values,
    variables="estimated_value",
    group_by="param",
    stats=["mean", "se", "lower_ci", "upper_ci"],
)

truth = simulated_values["simulated_value"]
estimate = estimated_values["estimated_value"]

r = compute_corr(truth, estimate)
ccc = compute_ccc(truth, estimate)
rmse = compute_rmse(truth, estimate)
```

For advanced posterior diagnostics, call BayesFlow directly through the model
workflow:

```python
diagnostics = model.workflow.compute_default_diagnostics(
    test_data=test_data,
    num_samples=500,
    variable_keys=None,
    as_data_frame=True,
)
```

These diagnostics are BayesFlow-native outputs rather than bami evaluation
tables.
