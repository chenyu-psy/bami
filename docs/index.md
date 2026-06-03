<div class="bami-home-title">
  <div class="bami-home-logo">b</div>
  <h1>bami</h1>
</div>

<p class="bami-lead">
Bayesian amortized model inference workflows for cognitive and behavioral
models.
</p>

## Overview

`bami` helps researchers build Bayesian amortized model inference workflows with
[BayesFlow](https://bayesflow.org/). It is designed for projects where the
scientific model is already available as a simulator, and the repeated work is
setting up priors, simulation, training, posterior sampling, and recovery
checks.

Use `bami` when you want to fit cognitive or behavioral models with simulated
data and want a reusable workflow around BayesFlow. The public API keeps the
main research decisions visible:

- which parameters are inferred
- what prior each parameter uses
- whether the data are aggregate summaries or trial-level rows
- whether the design has fixed or varying trial counts
- whether the model is simple or hierarchical

`bami` is not a full modeling language. You still provide the scientific model
through a simulator function.

## Installation

With `uv`:

```bash
uv add "bami @ git+https://github.com/chenyu-psy/bami.git"
```

With `pip`:

```bash
pip install "bami @ git+https://github.com/chenyu-psy/bami.git"
```

## A minimal workflow

This small SDM example shows the usual package flow: build a workflow, simulate
validation data, train or load the workflow, sample from the posterior, and run
a recovery diagnostic.

```python
import numpy as np

from bami.simulators import simulate_sdm_simple
from bami.workflows import SimpleWorkflow


np.random.seed(2026)

model = SimpleWorkflow(
    name="SDM",
    param_names=["c", "kappa"],
    priors={
        "c": {"mean": "normal(4, 1.2)", "sd": 0.5, "link": "softplus"},
        "kappa": {"mean": "normal(4, 1.2)", "sd": 0.5, "link": "softplus"},
    },
    simulator=simulate_sdm_simple,
    observation="trial",
    obs_names=["error"],
    n_trials=40,
)

validation_data = model.simulate(64)

model.train_workflow(
    max_epochs=20,
    initial_epochs=5,
    n_batch=50,
    batch_size=64,
    validation_data=validation_data,
    file="saved_workflows/sdm_simple.keras",
)

test_data = model.simulate(20)
samples = model.sample_posterior(test_data, num_samples=200)

fig = model.plot_parameter_recovery(
    n_datasets=20,
    num_samples=200,
    metrics=["corr", "ccc"],
)
```

These settings are intentionally small so the workflow shape is easy to check.
For a real analysis, increase the training and recovery settings after
confirming that the model simulates, trains, samples, and produces diagnostics
on your machine.

## Core ideas

### Workflows

A workflow is the analysis setup for one simulator-based model. It tells
`bami` which parameters to estimate, how to simulate training data, how to train
the model, and how to check the fitted model.

Use `SimpleWorkflow` when each simulated dataset has one set of model
parameters to estimate. Use `HierarchicalWorkflow` when the data come from a
group or study and you want population-level parameter estimates, with optional
subject-level estimates after the group model is trained.

The same simulator can often be used in both workflows. The workflow changes
the parameter structure and the inference target, not necessarily the
simulator. See [Choose the right workflow structure](articles/advanced-workflows.md)
for guidance on simple, flexible, and hierarchical workflows, and the
[workflow reference](api/workflows.md) for exact arguments.

### Simulators

A simulator is the bridge between a psychological model and `bami`. It is a
regular Python function that receives model parameter values and `n_trials`,
then returns simulated behavioral data.

The built-in simulator pages document examples shipped with `bami`; they are
not the only simulators you can use. You can write your own simulator or wrap a
simulator from another package. See
[Write a simulator for your own model](articles/simulators.md) for a
step-by-step guide.

### Priors

Priors define the range of parameter values that `bami` learns from during
simulation and training. They also determine the range of behavioral patterns
the workflow sees before it is used on real or held-out data.

In the minimal workflow above, `c` and `kappa` are estimated parameters because
their prior entries are dictionaries. Scalar entries are treated as fixed
simulator settings and passed to the simulator without being estimated. See
[Choose priors and links for model parameters](articles/priors.md) for prior
dictionaries, distribution strings, and link functions; the
[workflow reference](api/workflows.md) gives the exact constructor details.

### Observation shape

The `observation` argument tells `bami` what kind of behavioral data your
simulator returns.

- `observation="aggregate"` means the data have already been summarized. For
  example, each simulated subject might contribute one row with summary values
  such as a mean response, a response standard deviation, or an accuracy/rate.
- `observation="trial"` means the data are still at the trial level. For
  example, each row might be one trial's response, RT, error, or other measured
  outcome.

Use the form that matches the data produced by your simulator and the data you
plan to analyze. The [workflow reference](api/workflows.md) gives the exact
array shapes and trial-count options.
