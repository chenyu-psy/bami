# Simple SDM workflow

This example shows a complete simple-model workflow:

1. build the model
2. simulate validation data
3. train or load the saved workflow
4. sample posterior parameters
5. build long-format recovery tables
6. summarize and estimate recovery
7. draw a model diagnostic plot

The example uses the SDM simulator, where each trial is a signed circular error
in radians.

## Build the model

```python
import numpy as np
import pandas as pd

from bami.evaluation.metrics import aggregate_data, estimate_recovery
from bami.simulators import simulate_sdm_simple
from bami.utils import posterior_to_dataframe
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
    obs_names=["error"],
    n_trials=25,
    summary_dim=4,
    n_coupling_layers=2,
)
```

## Simulate validation data

```python
validation_data = model.simulate(64)
print(validation_data["data"].shape)
```

Expected shape:

```text
(64, 25, 1)
```

The dimensions mean:

- `64` simulated datasets
- `25` trial rows per dataset
- `1` observation feature per trial: signed circular error in radians

The SDM simulator returns continuous signed circular errors in `[-pi, pi]`.
Observed SDM data should use the same radian convention before being passed to
a workflow.

## Train or load the workflow

`train_workflow()` loads the saved file when it already exists. Use
`overwrite=True` when you intentionally want to retrain.

```python
history = model.train_workflow(
    max_epochs=100,
    initial_epochs=20,
    n_batch=200,
    batch_size=64,
    validation_data=validation_data,
    patience=10,
    min_delta=0.001,
    file="saved_workflows/sdm_simple.keras",
)
```

For a real project, increase `n_batch`, `batch_size`, or `max_epochs` only after
checking that the small workflow runs correctly on your machine.

## Sample posterior parameters

Simulate a fresh recovery set, then draw posterior samples for each dataset.

```python
test_data = model.simulate(50)
samples = model.sample_posterior(
    test_data=test_data,
    num_samples=500,
)

print(samples["c"].shape)
posterior_df = posterior_to_dataframe(samples, model.priors, level="simple")
print(posterior_df.head())
```

Typical shape:

```text
(50, 500)
```

The first dimension is the simulated dataset. The second dimension is the
posterior-sample axis. The sampling method returns a dictionary so workflow
steps can still use raw keys when needed. Use `posterior_to_dataframe(...)`
when you want a tidy table for reporting or plotting.

## Build long-format recovery tables

The evaluation helpers work on ordinary pandas data frames. The small helper
below converts simulated truth and posterior means into long-format tables.

```python
def simple_recovery_tables(test_data, samples, params):
    """Return simulated and estimated long tables for simple recovery checks."""

    simulated_rows = []
    estimated_rows = []

    for param in params:
        posterior_mean = np.mean(samples[param], axis=1)
        for dataset_id in range(len(test_data[param])):
            simulated_rows.append(
                {
                    "dataset_id": dataset_id,
                    "param": param,
                    "simulated_value": test_data[param][dataset_id],
                }
            )
            estimated_rows.append(
                {
                    "dataset_id": dataset_id,
                    "param": param,
                    "estimated_value": posterior_mean[dataset_id],
                }
            )

    return pd.DataFrame(simulated_rows), pd.DataFrame(estimated_rows)


simulated_values, estimated_values = simple_recovery_tables(
    test_data,
    samples,
    params=["c", "kappa"],
)
```

## Summarize and estimate recovery

Use `aggregate_data(...)` when you want descriptive summaries of estimated
values. Use `estimate_recovery(...)` when you want paired recovery metrics
between simulated truth and estimates.

```python
estimate_summary = aggregate_data(
    estimated_values,
    variables="estimated_value",
    group_by="param",
    stats=["mean", "se", "lower_ci", "upper_ci", "median", "lower_q", "upper_q"],
    ci=0.95,
    q=0.95,
)

recovery_metrics = estimate_recovery(
    simulated_data=simulated_values,
    estimated_data=estimated_values,
    group_by="param",
    metrics=["corr", "ccc", "rmse"],
)

print(estimate_summary)
print(recovery_metrics)
```

## Draw a diagnostic plot

For one fitted model, the model-level diagnostic plot can run the same
simulate-sample-compare workflow automatically.

```python
fig = model.plot_parameter_recovery(
    n_datasets=50,
    num_samples=500,
    metrics=["corr", "ccc"],
)
```

This diagnostic is for one model. If you are comparing several fitted models,
build explicit long-format tables and use `estimate_recovery(...)` instead.

## Flexible trial counts

Use `n_trials_range` when simulated datasets should have different trial counts:

```python
flex_model = SimpleWorkflow(
    name="SDM",
    param_names=["c", "kappa"],
    priors={
        "c": {"mean": "normal(1, 0.35)", "sd": 0.15, "link": "log"},
        "kappa": {"mean": "normal(1.2, 0.35)", "sd": 0.15, "link": "log"},
    },
    simulator=simulate_sdm_simple,
    observation="trial",
    obs_names=["error"],
    n_trials=None,
    n_trials_range=(20, 31),
    summary_dim=4,
    n_coupling_layers=2,
)

flex_sim = flex_model.simulate(5)
print(flex_sim["data"].shape)
```

Expected shape:

```text
(5, 30, 2)
```

The second feature is an active-trial mask. It marks real trials with `1` and
padded rows with `0`, so BayesFlow can receive a fixed tensor shape while the
research design still varies in trial count.
