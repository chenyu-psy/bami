# Utilities

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
