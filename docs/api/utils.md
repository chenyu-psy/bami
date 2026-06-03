# Utils

Use utility helpers when you want to reshape workflow outputs for analysis or
reporting without changing the workflow sampling contract.

## Posterior dataframes

The `sample_*` methods return dictionaries because later workflow steps may
need raw keys. Use `posterior_to_dataframe(...)` when you want a tidy table for
pandas-based summaries, plotting, or export.

::: bami.utils.posterior_to_dataframe
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 3

### Example

```python
from bami.utils import posterior_to_dataframe

group_samples = model.sample_group_posterior(
    test_data=test_data,
    num_samples=500,
)

posterior_df = posterior_to_dataframe(
    group_samples,
    model.priors,
    level="group",
)
```

## Data aggregation

Use `aggregate_data(...)` when you have an analysis table and want grouped
summaries such as means, standard errors, or confidence intervals.

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
