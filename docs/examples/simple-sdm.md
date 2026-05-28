# Simple SDM workflow

This example builds a simple, non-hierarchical SDM workflow. Each prior draw
generates one dataset with a fixed number of trial-level circular errors.

```python
import numpy as np

from bami.simulators import GRID_SIZE, simulate_sdm_simple
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
    simulator_kwargs={"grid_size": GRID_SIZE, "error_scale": 180.0},
    obs_names=["error"],
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
- `1` observation feature per trial: signed circular error

The SDM simulator returns signed circular errors. In this example,
`error_scale=180.0` stores those errors on an approximately unit-scaled range,
which is easier for the neural network than raw degrees.

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
    simulator_kwargs={"grid_size": GRID_SIZE, "error_scale": 180.0},
    obs_names=["error"],
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
