# Simple SDM workflow

This example builds a simple, non-hierarchical SDM workflow. Each prior draw
generates one dataset with a fixed number of trial-level circular errors in
radians.

```python
import numpy as np

from bami.simulators import simulate_sdm_simple
from bami.workflows import SimpleWorkflow


np.random.seed(2026)

model = SimpleWorkflow(
    name="SDM",
    param_names=["c", "kappa"],
    priors={
        "c": {"mean": "normal(1, 0.35)", "sd": 0.15, "link": "log"},
        "kappa": {"mean": "normal(1.2, 0.35)", "sd": 0.15, "link": "log"},
    },
    simulator=simulate_sdm_simple,
    observation="trial",
    obs_names=["error_rad"],
    n_trials=25,
    summary_dim=4,
    n_coupling_layers=2,
)

sim = model.workflow.simulate(5)
print(sim["data"].shape)
```

Expected shape:

```text
(5, 25, 1)
```

The dimensions mean:

- `5` simulated datasets
- `25` trial rows per dataset
- `1` observation feature per trial: signed circular error in radians

The SDM simulator returns continuous signed circular errors in `[-pi, pi]`.
Observed SDM data should use the same radian convention before being passed to
a workflow.

## Flexible trial counts

Use `n_trials_range` when simulated datasets should have different trial counts:

```python
model = SimpleWorkflow(
    name="SDM",
    param_names=["c", "kappa"],
    priors={
        "c": {"mean": "normal(1, 0.35)", "sd": 0.15, "link": "log"},
        "kappa": {"mean": "normal(1.2, 0.35)", "sd": 0.15, "link": "log"},
    },
    simulator=simulate_sdm_simple,
    observation="trial",
    obs_names=["error_rad"],
    n_trials=None,
    n_trials_range=(20, 31),
    summary_dim=4,
    n_coupling_layers=2,
)

sim = model.workflow.simulate(5)
print(sim["data"].shape)
```

Expected shape:

```text
(5, 30, 2)
```

The second feature is an active-trial mask. It marks real trials with `1` and
padded rows with `0`, so BayesFlow can receive a fixed tensor shape while the
research design still varies in trial count.
