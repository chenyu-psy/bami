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

## Usage

A typical `bami` analysis follows the same data flow:

1. Write or choose a simulator that turns model parameters into data.
2. Describe priors for the parameters.
3. Build a `SimpleWorkflow` or `HierarchicalWorkflow`.
4. Train the workflow or load saved weights.
5. Sample from the posterior.
6. Check recovery and diagnostics.

The examples in this site focus on the first three steps so the workflow shape
is easy to inspect before starting longer training runs.

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
    obs_names=["error_rad"],
    n_trials=25,
)

sim = model.workflow.simulate(4)
print(sim["data"].shape)
```

## Main concepts

### Simulators

A simulator is a regular Python function. For simple workflows, it receives
model parameters, `n_trials`, and `rng`, then returns simulated data. For
hierarchical workflows, `bami` calls the same kind of simulator once per
subject.

### Priors

Priors use a compact dictionary format:

```python
priors = {
    "a": {"mean": 0.0, "sd": 1.0, "link": "identity"},
    "c": {"mean": "normal(0, 1)", "sd": "exponential(1)", "link": "identity"},
}
```

Dictionary-valued parameters are inferred. Scalar values are treated as fixed
constants and passed to the simulator.

### Observation contracts

The `observation` argument makes the data shape explicit:

- `observation="aggregate"` means one fixed-width summary row per dataset or subject.
- `observation="trial"` means one row per trial.

This distinction matters because BayesFlow uses different summary-network
behavior for fixed rows and exchangeable trial sets.

### Fixed and flexible designs

Use `n_trials` for fixed trial counts:

```python
model = SimpleWorkflow(..., n_trials=100)
```

Use `n_trials_range` when simulated datasets should vary in trial count:

```python
model = SimpleWorkflow(..., n_trials=None, n_trials_range=(50, 201))
```

The upper bound is exclusive, so this example draws 50 to 200 trials.

## Main modules

- `bami.workflows`: `SimpleWorkflow` and `HierarchicalWorkflow`.
- `bami.inputs`: input-format helpers for aggregate rows that encode trial count.
- `bami.simulators`: reusable simulator helpers for SDM, ezDM, M3, and circular data.
- `bami.inference`: priors, transforms, checkpoints, runtime settings, and posterior helpers.
- `bami.recovery`: training and recovery-analysis helpers.
- `bami.evaluation`: metric tables, validation contracts, and diagnostic plots.

## Learning bami

Start with one of the short examples, then use the API reference when you need
the exact arguments for a function or workflow.

<div class="bami-link-list">
  <a href="examples/simple-sdm/">Simple SDM workflow</a>
  <a href="examples/hierarchical-ezdm/">Hierarchical ezDM workflow</a>
  <a href="api/workflows/">Workflow reference</a>
</div>
