# SimpleWorkflow

Use `SimpleWorkflow` when one prior draw generates one simulated dataset. This
is the usual entry point for non-hierarchical models.

::: bami.workflows.simple.SimpleWorkflow
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 3
      members: false

## Example

```python
from bami.simulators import GRID_SIZE, simulate_sdm_simple
from bami.workflows import SimpleWorkflow


model = SimpleWorkflow(
    name="SDM",
    param_names=["c", "kappa"],
    priors={
        "c": {"mean": "normal(1, 0.35)", "sd": 0.15, "link": "log"},
        "kappa": {"mean": "normal(1.2, 0.35)", "sd": 0.15, "link": "log"},
    },
    simulator=simulate_sdm_simple,
    observation="trial",
    simulator_kwargs={"grid_size": GRID_SIZE, "error_scale": 180.0},
    obs_names=["error"],
    n_trials=25,
)

sim = model.workflow.simulate(4)
print(sim["data"].shape)
```
