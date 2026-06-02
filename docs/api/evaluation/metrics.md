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

### Example

```python
import numpy as np
import pandas as pd

from bami.evaluation.metrics import aggregate_data


rng = np.random.default_rng(2026)
n_datasets = 40
estimated_values = pd.DataFrame(
    {
        "dataset_id": np.arange(n_datasets),
        "param": "c",
        "estimated_value": rng.normal(loc=1.0, scale=0.15, size=n_datasets),
    }
)

value_summary = aggregate_data(
    estimated_values,
    variables="estimated_value",
    group_by="param",
    stats=["mean", "se", "lower_ci", "upper_ci"],
)
```

## Recovery metrics

::: bami.evaluation.metrics.recovery
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - estimate_recovery

### Example

```python
import numpy as np
import pandas as pd

from bami.evaluation.metrics import estimate_recovery


rng = np.random.default_rng(2026)
n_datasets = 40
simulated_c = rng.normal(loc=1.0, scale=0.2, size=n_datasets)
simulated_values = pd.DataFrame(
    {
        "dataset_id": np.arange(n_datasets),
        "param": "c",
        "simulated_value": simulated_c,
    }
)
estimated_values = pd.DataFrame(
    {
        "dataset_id": np.arange(n_datasets),
        "param": "c",
        "estimated_value": simulated_c
        + rng.normal(loc=0.0, scale=0.08, size=n_datasets),
    }
)

recovery_metrics = estimate_recovery(
    simulated_values,
    estimated_values,
    group_by="param",
    metrics=["ccc", "rmse"],
)
```

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

### Example

```python
import numpy as np
import pandas as pd

from bami.evaluation.metrics import (
    compute_ccc,
    compute_corr,
    compute_rmse,
)


rng = np.random.default_rng(2026)
n_datasets = 40
simulated_c = rng.normal(loc=1.0, scale=0.2, size=n_datasets)
simulated_values = pd.DataFrame(
    {
        "dataset_id": np.arange(n_datasets),
        "param": "c",
        "simulated_value": simulated_c,
    }
)
estimated_values = pd.DataFrame(
    {
        "dataset_id": np.arange(n_datasets),
        "param": "c",
        "estimated_value": simulated_c
        + rng.normal(loc=0.0, scale=0.08, size=n_datasets),
    }
)

truth = simulated_values["simulated_value"]
estimate = estimated_values["estimated_value"]

r = compute_corr(truth, estimate)
ccc = compute_ccc(truth, estimate)
rmse = compute_rmse(truth, estimate)
```
