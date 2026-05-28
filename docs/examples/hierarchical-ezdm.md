# Hierarchical ezDM workflow

This example builds a hierarchical ezDM workflow. Each simulated dataset
contains multiple subjects, and each subject contributes one aggregate summary
row with `pc`, `mrt`, and `vrt`.

```python
import numpy as np

from bami.inference import transform_hierarchical_samples
from bami.simulators import simulate_ezdm_simple
from bami.workflows import HierarchicalWorkflow


np.random.seed(2026)

priors = {
    "v": {"mean": "normal(0, 0.6)", "sd": 0.15, "link": "log"},
    "a": {"mean": "normal(0.2, 0.4)", "sd": 0.15, "link": "log"},
    "t0": {"mean": "logistic(-2, 0.5)", "sd": 0.15, "link": "logit"},
}

model = HierarchicalWorkflow(
    name="ezDM",
    priors=priors,
    simulator=simulate_ezdm_simple,
    observation="aggregate",
    simulator_kwargs={"s": 1},
    obs_names=["pc", "mrt", "vrt"],
    n_subjects=3,
    n_trials=20,
    summary_dim=4,
    n_coupling_layers=2,
    transform_samples=transform_hierarchical_samples,
)

sim = model.workflow.simulate(5)
print(sim["data"].shape)
```

Expected shape:

```text
(5, 3, 3)
```

The dimensions mean:

- `5` simulated datasets
- `3` subjects per dataset
- `3` aggregate features per subject: `pc`, `mrt`, and `vrt`

## Flexible subjects and trials

Use `n_subjects_range` and `n_trials_range` when group size and trial count
should vary across simulated datasets.

```python
from bami.inputs import aggregate_summary


model = HierarchicalWorkflow(
    name="ezDM",
    priors=priors,
    simulator=simulate_ezdm_simple,
    observation="aggregate",
    simulator_kwargs={"s": 1},
    obs_names=["pc", "mrt", "vrt"],
    n_subjects_range=(2, 5),
    n_trials_range=(10, 15),
    input_format=aggregate_summary(n_range=(10, 14)),
    summary_dim=4,
    n_coupling_layers=2,
    transform_samples=transform_hierarchical_samples,
)

sim = model.workflow.simulate(6)
print(sim["data"].shape)
```

Expected shape:

```text
(6, 4, 5)
```

The output includes the three ezDM summary features, a scaled trial-count
feature, and an active-subject mask. The mask lets the workflow represent
groups with different subject counts in one padded tensor.
