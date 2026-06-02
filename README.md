# bami

`bami` provides reusable [BayesFlow](https://bayesflow.org/) workflows for
simulator-based Bayesian amortized model inference.

Use `bami` when you already have a simulator and want a consistent workflow for
simulation, training, posterior sampling, and diagnostic checks. The package
keeps the main modeling choices explicit: parameters, priors, observation
format, trial-count design, and whether the workflow is simple or hierarchical.

`bami` is not a full modeling language. You provide the scientific simulator;
`bami` provides the workflow around it.

## Installation

Install from GitHub with `pip`:

```bash
pip install "bami @ git+https://github.com/chenyu-psy/bami.git"
```

If your project uses `uv`, add it with:

```bash
uv add "bami @ git+https://github.com/chenyu-psy/bami.git"
```

## Quick Start

This minimal example builds a simple SDM workflow and simulates four datasets.
The SDM simulator is included as an example model; the same workflow structure
can be used with your own simulator.

```python
from bami.simulators import simulate_sdm_simple
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
    obs_names=["error"],
    n_trials=25,
)

sim = model.simulate(4)
print(sim["data"].shape)
```

Expected output:

```text
(4, 25, 1)
```

## Full Documentation

Detailed guides and API references are available at
[chenyu-psy.github.io/bami](https://chenyu-psy.github.io/bami/).

Useful starting points:

- [Simple SDM workflow](https://chenyu-psy.github.io/bami/examples/simple-sdm/)
- [Hierarchical ezDM workflow](https://chenyu-psy.github.io/bami/examples/hierarchical-ezdm/)
- [Workflow reference](https://chenyu-psy.github.io/bami/api/workflows/)
- [Evaluation metrics](https://chenyu-psy.github.io/bami/api/evaluation/metrics/)
- [Diagnostic plots](https://chenyu-psy.github.io/bami/api/evaluation/diagnostics/)
