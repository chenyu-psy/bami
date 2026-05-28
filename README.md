# bami

`bami` helps researchers build Bayesian amortized model inference workflows
with [BayesFlow](https://bayesflow.org/). It is meant for research projects
where you already have a model simulator and want a reusable workflow for
simulation, training, posterior sampling, recovery checks, and diagnostic plots.

The package is organized around a simple idea:

1. Write or choose a simulator that turns model parameters into data.
2. Describe the priors for those parameters.
3. Build a `SimpleWorkflow` or `HierarchicalWorkflow`.
4. Train the workflow, sample from the posterior, and check recovery.

## Who this is for

Use `bami` if you are fitting cognitive or behavioral models with simulated
data and want to avoid rewriting the same BayesFlow setup code for each model.
The package is written for psychology researchers, so the public API keeps the
main modeling decisions visible:

- which parameters are inferred
- what prior each parameter uses
- whether observations are aggregate summaries or trial-level rows
- whether datasets have a fixed or varying number of trials
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

## Quick start: a simple SDM workflow

The example below builds a standard SDM workflow for trial-level circular
errors. The simulator returns one signed radian error per trial. The workflow
then learns to infer the SDM parameters `c` and `kappa` from those trial rows.

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

simulated = model.workflow.simulate(4)
print(simulated["data"].shape)
```

Expected shape:

```text
(4, 25, 1)
```

This means:

- `4` simulated datasets
- `25` trial rows per dataset
- `1` observation feature per trial: the signed circular error in radians

## Core concepts

### Simulators

A simulator is a regular Python function. For simple workflows, it should accept
model parameters, `n_trials`, and `rng`.

```python
def simulator(param_a, param_b, n_trials, rng):
    ...
    return data_row
```

For `observation="aggregate"`, return one fixed-width row such as summary
statistics, counts, or proportions.

For `observation="trial"`, return one row per trial. This is useful when the
trial sequence itself should be summarized by the neural network.

For SDM trial-level workflows, `simulate_sdm_simple()` returns signed circular
errors in radians. User-provided SDM data should use the same convention, with
one trial row containing one error in `[-pi, pi]`.

### Priors

Priors use a compact dictionary format:

```python
priors = {
    "a": {"mean": 0.0, "sd": 1.0, "link": "identity"},
    "c": {"mean": "normal(0, 1)", "sd": "exponential(1)", "link": "identity"},
}
```

Each dictionary-valued parameter becomes an inferred parameter. Scalar values
are treated as fixed constants and are passed to the simulator but are not
inferred.

Common links are handled by `bami.inference.priors`. Use links to keep
parameters on their valid scale while training in an unconstrained raw space.

### Observation contracts

The `observation` argument is required because it changes the shape of the data
given to BayesFlow:

- `observation="aggregate"`: one row per simulated dataset or subject
- `observation="trial"`: one row per trial

When trial count carries reliability information, use an input format to make
that encoding explicit:

```python
from bami.inputs import aggregate_summary

input_format = aggregate_summary(n_range=(50, 200))
```

This appends a scaled trial-count feature to aggregate rows.

### Fixed and flexible trial counts

Use `n_trials` for a fixed design:

```python
model = SimpleWorkflow(..., n_trials=100)
```

Use `n_trials_range` when the number of trials varies across simulated
datasets:

```python
model = SimpleWorkflow(..., n_trials=None, n_trials_range=(50, 201))
```

The upper bound is exclusive. In this example, simulated trial counts can range
from 50 to 200.

## Main modules

`bami.workflows`
: `SimpleWorkflow` and `HierarchicalWorkflow`.

`bami.inputs`
: Input-format helpers for aggregate rows that encode trial count.

`bami.simulators`
: Reusable simulator helpers for existing model families, including EZ-diffusion
and circular-response summary utilities.

`bami.inference`
: Prior drawing, parameter transforms, saved-workflow helpers, runtime settings,
and posthoc subject-level sampling.

`bami.recovery`
: Recovery workflows for comparing true and estimated parameters.

`bami.evaluation`
: Recovery tables, diagnostic metric tables, and plotting functions.

`bami.data_ops` and `bami.data_shapes`
: Small utilities for tabular data handling, validation, and padding.

## Training and saved workflows

After building a workflow, train through the model object:

```python
history = model.train_workflow(
    max_epochs=100,
    initial_epochs=20,
    n_batch=200,
    batch_size=64,
    validation_data=64,
    patience=10,
    min_delta=0.001,
    file="saved_workflows/sdm_workflow.keras",
)
```

If the saved workflow file already exists, `train_workflow()` loads it by
default instead of retraining. Use `overwrite=True` to force a new fit.

## Evaluation workflow

A typical analysis script uses this order:

1. Build the workflow.
2. Simulate validation or recovery datasets.
3. Train or load the workflow.
4. Sample posterior draws.
5. Compute recovery or diagnostic tables.
6. Plot the results.

The evaluation helpers return ordinary pandas data frames so they can be
inspected, saved, or plotted with project-specific code.

```python
from bami.evaluation import (
    estimate_population_recovery,
    plot_population_recovery,
    sample_posterior,
)

test_data = model.workflow.simulate(100)
samples = sample_posterior(model.workflow, test_data, num_samples=500)
recovery_df = estimate_population_recovery(model, test_data, samples)
fig = plot_population_recovery(recovery_df)
```

## Development

Run the test suite:

```bash
uv run pytest
```

Format code:

```bash
uv run black src tests
```

Lint code:

```bash
uv run ruff check src tests
```

## Design notes for contributors

The package favors explicit, readable code over compact abstractions. Public
functions should make modeling assumptions visible, especially data shape,
parameter scale, trial-count handling, and whether values are on the raw or
public scale.

When adding a new model family, prefer this structure:

1. Put reusable simulator code in `src/bami/simulators/`.
2. Build workflows through `SimpleWorkflow` or `HierarchicalWorkflow`.
3. Put recovery or analysis-specific orchestration in project scripts unless it
is reusable across projects.
4. Add focused tests that check data shapes, parameter names, and posterior
transforms.
