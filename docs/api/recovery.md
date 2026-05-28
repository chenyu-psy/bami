# Recovery

Use these functions for group-generated parameter recovery checks.

## Simulation

::: bami.recovery.simulate
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 3

## Recovery estimation

::: bami.recovery.recover
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 3

## Summary

::: bami.recovery.summarize
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 3

## Example

```python
import pandas as pd

from bami import recovery


base_params = ["theta"]
population_params = ["theta_mu"]

payload = recovery.simulate(
    generator=generator,
    n_reps=100,
)

population_rows, individual_rows = recovery.recover(
    fit_name="fixed_hierarchy",
    fit_model=model,
    sim_data=payload["sim_data"],
    base_params=base_params,
    posterior_samples=500,
)

outputs = recovery.summarize(
    population_rows=population_rows,
    individual_rows=individual_rows,
    base_params=base_params,
    quantile_bias_params=population_params,
)

# Saving is handled with standard pandas/matplotlib methods.
population_rows.to_csv("population_rows.csv", index=False)
individual_rows.to_csv("individual_rows.csv", index=False)
outputs["population_figure"].savefig("population_recovery.png", dpi=200)
```
