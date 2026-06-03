# Choose the right workflow structure

Start with the simplest workflow that matches the scientific design. Move to a
more flexible workflow only when the data structure requires it.

The simulator contract usually stays the same. What changes is how often the
simulator is called, how simulated rows are arranged, and which parameter level
the workflow estimates.

## Fixed simple workflow

Use a fixed simple workflow when each dataset or subject has one parameter
vector and one fixed trial count.

```python
model = SimpleWorkflow(
    name="SDM",
    param_names=["c", "kappa"],
    priors=priors,
    simulator=simulate_sdm_simple,
    observation="trial",
    obs_names=["error"],
    n_trials=100,
)
```

This is the best first workflow for most new models. It keeps the simulation
shape easy to inspect:

```text
(n_datasets, n_trials, n_features)
```

For aggregate simulators, the middle dimension is `1` because each dataset has
one summary row.

## Flexible simple workflow

Use a flexible simple workflow when each dataset still has one parameter vector
but trial counts vary.

```python
model = SimpleWorkflow(
    name="SDM",
    param_names=["c", "kappa"],
    priors=priors,
    simulator=simulate_sdm_simple,
    observation="trial",
    obs_names=["error"],
    n_trials=None,
    n_trials_range=(50, 201),
)
```

For trial-level data, `bami` pads datasets to a common maximum trial count and
adds an active-trial mask. Real trials are marked with `1`; padded rows are
marked with `0`.

Use this workflow when variable trial count is part of the study design or
expected missingness pattern. If all subjects have the same planned number of
trials, start with fixed simple.

## Fixed hierarchical workflow

Use a fixed hierarchical workflow when each simulated dataset is a group or
study with several subjects, and the number of subjects and trials is fixed.

After sampling, hierarchical workflows return public-scale group parameters
such as `c_mu`, `c_sigma`, `kappa_mu`, and `kappa_sigma` for summaries,
recovery plots, and reports.

```python
from bami.workflows import HierarchicalWorkflow


model = HierarchicalWorkflow(
    name="SDM",
    priors=priors,
    simulator=simulate_sdm_simple,
    observation="trial",
    obs_names=["error"],
    n_subjects=40,
    n_trials=100,
)
```

The group workflow estimates population parameters such as `c_mu`,
`c_sigma`, `kappa_mu`, and `kappa_sigma`. The simulator still receives
subject-level public parameters such as `c` and `kappa`.

Use hierarchical workflows when population-level inference matters or when
subject-level estimates should borrow strength from the group.

## Flexible hierarchical workflow

Use a flexible hierarchical workflow when group size, subject trial count, or
both can vary.

Flexible hierarchical workflows return the same public-scale group parameters
after sampling.

```python
model = HierarchicalWorkflow(
    name="SDM",
    priors=priors,
    simulator=simulate_sdm_simple,
    observation="trial",
    obs_names=["error"],
    n_subjects_range=(20, 61),
    n_trials_range=(50, 201),
)
```

This is the most flexible shape, but it is also the hardest to debug. Build and
check a fixed simple workflow first whenever possible.

## Subject-level recovery

Hierarchical workflows can save subject-level truth values for recovery checks.
The default `keep_subject_truth=None` keeps all stochastic subject parameters.
Use a list to keep selected parameters:

```python
model = HierarchicalWorkflow(
    ...,
    keep_subject_truth=["c", "kappa"],
)
```

Subject-level recovery uses the random estimator path:

```python
model.train_random_estimator(file="saved_workflows/sdm_random_estimator.pt")

group_samples = model.sample_group_posterior(
    test_data=test_data,
    num_samples=500,
)

subject_estimates = model.estimate_random_parameter(
    observed_data=test_data,
    group_samples=group_samples,
)
```

Use this output for point-estimate recovery of subject parameters. It is not a
posterior sample and should not be used for posterior intervals or coverage
checks.

## Model-comparison workflows

Model-comparison projects can build on the same pieces:

1. Define model-specific priors and simulator contracts.
2. Train fixed simple, flexible simple, and hierarchical workflows as needed.
3. Generate recovery datasets from a chosen generation prior.
4. Fit each trained workflow to the same generated datasets.
5. Store long-format recovery rows with columns such as `fit_model`, `param`,
   `simulated_value`, and `estimated_value`.
6. Compare recovery metrics with `estimate_recovery(...)`.

The important design rule is to keep the simulator contract stable. New
workflow types should change the data arrangement and inference target, not the
meaning of the simulator's public parameters.
