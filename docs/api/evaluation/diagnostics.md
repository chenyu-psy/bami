# Diagnostics

Diagnostic plots are model-level checks for one fitted workflow. They simulate
fresh data, sample from the fitted posterior, and compare simulated truth with
posterior means. They are not model-comparison tools; for comparing models,
build explicit long-format tables and use `estimate_recovery(...)`.

These high-level diagnostics use a conservative internal sampling batch size.
For advanced BayesFlow sampling controls, call `simulate(...)` and the relevant
posterior sampling method on your fitted workflow object.

## Simple parameter recovery

Use `plot_parameter_recovery(...)` to check whether a simple workflow can
recover each simulated parameter from newly simulated datasets. Requested
metrics are shown in each scatter panel title.

::: bami.evaluation.diagnostics.plot_parameter_recovery
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 3

### Example

```python
from bami.evaluation.diagnostics import plot_parameter_recovery


fig = model.plot_parameter_recovery(
    n_datasets=50,
    num_samples=500,
    metrics="corr",
)

# Equivalent explicit evaluation call:
fig = plot_parameter_recovery(model, n_datasets=50, num_samples=500)

# Multiple metrics are shown in each panel title.
fig = model.plot_parameter_recovery(
    n_datasets=50,
    num_samples=500,
    metrics=["corr", "ccc", "rmse"],
)
```

The model method is a shallow alias for the evaluation function.

## Population parameter recovery

Use `plot_population_recovery(...)` to diagnose group-level parameters in a
hierarchical workflow, such as `c_mu` and `c_sigma`. Requested metrics are
shown in each scatter panel title.

::: bami.evaluation.diagnostics.plot_population_recovery
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 3

### Example

```python
from bami.evaluation.diagnostics import plot_population_recovery


fig = model.plot_population_recovery(
    n_datasets=50,
    num_samples=500,
    params=["c_mu", "c_sigma"],
    metrics="corr",
)

# Equivalent explicit evaluation call:
fig = plot_population_recovery(
    model,
    n_datasets=50,
    num_samples=500,
    params=["c_mu", "c_sigma"],
    metrics="corr",
)

# Multiple metrics are shown in each panel title.
fig = model.plot_population_recovery(
    n_datasets=50,
    num_samples=500,
    params=["c_mu", "c_sigma"],
    metrics=["corr", "ccc", "rmse"],
)
```

The model method is a shallow alias for the evaluation function.

## Random parameter recovery

Use `plot_random_recovery(...)` to diagnose subject-level random parameters in
a hierarchical workflow. The workflow must save subject truth with
`keep_subject_truth`. Each point is one simulated dataset's subject-level
recovery metric for one parameter.

::: bami.evaluation.diagnostics.plot_random_recovery
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 3

### Example

```python
from bami.evaluation.diagnostics import plot_random_recovery


fig = model.plot_random_recovery(
    n_datasets=50,
    num_samples=500,
    params=["c", "kappa"],
    metrics="corr",
)

# Equivalent explicit evaluation call:
fig = plot_random_recovery(
    model,
    n_datasets=50,
    num_samples=500,
    params=["c", "kappa"],
    metrics="corr",
)

# Multiple metrics are shown as separate panels.
fig = model.plot_random_recovery(
    n_datasets=50,
    num_samples=500,
    params=["c", "kappa"],
    metrics=["corr", "ccc", "rmse"],
)
```

The model method is a shallow alias for the evaluation function.

## Advanced BayesFlow diagnostics

Advanced posterior diagnostics can be computed directly through the underlying
BayesFlow workflow:

```python
diagnostics = model.workflow.compute_default_diagnostics(
    test_data=test_data,
    num_samples=500,
    variable_keys=None,
    as_data_frame=True,
)
```
