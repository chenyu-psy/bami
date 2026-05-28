# HierarchicalWorkflow

Use `HierarchicalWorkflow` when one simulated dataset contains multiple
subjects and the workflow should infer group-level parameters.

::: bami.workflows.hierarchical.HierarchicalWorkflow
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 3
      members: false

## Example

```python
from bami.inference import transform_hierarchical_samples
from bami.simulators import simulate_ezdm_simple
from bami.workflows import HierarchicalWorkflow


model = HierarchicalWorkflow(
    name="ezDM",
    priors={
        "v": {"mean": "normal(0, 0.6)", "sd": 0.15, "link": "log"},
        "a": {"mean": "normal(0.2, 0.4)", "sd": 0.15, "link": "log"},
        "t0": {"mean": "logistic(-2, 0.5)", "sd": 0.15, "link": "logit"},
    },
    simulator=simulate_ezdm_simple,
    observation="aggregate",
    simulator_kwargs={"s": 1},
    obs_names=["pc", "mrt", "vrt"],
    n_subjects=3,
    n_trials=20,
    transform_samples=transform_hierarchical_samples,
)

sim = model.workflow.simulate(4)
print(sim["data"].shape)
```
