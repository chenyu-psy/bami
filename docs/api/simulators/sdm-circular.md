# SDM simulator

Use `simulate_sdm_simple()` as the workflow simulator for trial-level circular
errors. It returns one continuous signed error in radians per trial.

User-provided observed SDM data should use the same convention: one trial row
contains one error in `[-pi, pi]`.

## Function

::: bami.simulators.sdm
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - simulate_sdm_simple

## Example

```python
from bami.simulators import simulate_sdm_simple


errors = simulate_sdm_simple(
    c=1.0,
    kappa=1.2,
    n_trials=25,
)
```
