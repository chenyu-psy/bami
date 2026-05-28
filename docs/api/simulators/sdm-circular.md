# SDM simulator

Use `simulate_sdm_simple()` as the workflow simulator for trial-level circular
errors. The remaining functions on this page are SDM and circular-error helpers
for inspecting probabilities, binning errors, and computing summary features.

## SDM simulation

::: bami.simulators.sdm
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - sdm_probs
        - simulate_sdm_simple

## Circular error helpers

::: bami.simulators.circular
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - GRID_SIZE
        - degree_grid
        - errors_to_indices
        - indices_to_errors
        - circular_moments_from_errors
        - check_n_trials

## Examples

```python
import numpy as np

from bami.simulators.circular import (
    GRID_SIZE,
    check_n_trials,
    circular_moments_from_errors,
    degree_grid,
    errors_to_indices,
    indices_to_errors,
)
from bami.simulators.sdm import sdm_probs, simulate_sdm_simple


grid = degree_grid(grid_size=GRID_SIZE)
probs = sdm_probs(c=1.0, kappa=1.2, grid_size=GRID_SIZE)

n_trials = check_n_trials(25)
errors = simulate_sdm_simple(c=1.0, kappa=1.2, n_trials=n_trials)

indices = errors_to_indices(np.array([0, 1, -1, 180]))
round_trip_errors = indices_to_errors(indices)
moments = circular_moments_from_errors(errors)
```
