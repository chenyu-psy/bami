# ezDM simulator

Use `simulate_ezdm_simple()` as the built-in simulator for summary-level ezDM
workflows. It returns one aggregate row with `pc`, `mrt`, and `vrt`.

For alternative EZ equations, trial-level diffusion simulations, or additional
response features, define a custom simulator function and pass it to a workflow.

## Function

::: bami.simulators.ezdm
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - simulate_ezdm_simple

## Example

```python
from bami.simulators import simulate_ezdm_simple


summary_row = simulate_ezdm_simple(
    v=0.1,
    a=0.14,
    t0=0.3,
    n_trials=50,
    s=0.1,
)
```
