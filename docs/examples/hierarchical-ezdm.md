# Fit a hierarchical ezDM workflow

This example shows a complete hierarchical workflow:

1. build the group model
2. keep subject-level truth for random-effect diagnostics
3. train or load the group workflow
4. train or load the random-effect workflow
5. sample group and subject parameters
6. build long-format recovery tables
7. summarize and estimate recovery
8. draw population and random diagnostic plots

Each simulated dataset contains multiple subjects. Each subject contributes one
aggregate summary row with `pc`, `mrt`, and `vrt`.

## Build the model

```python
import numpy as np
import pandas as pd

from bami.evaluation.metrics import aggregate_data, estimate_recovery
from bami.simulators import simulate_ezdm_simple
from bami.utils import posterior_to_dataframe
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
    n_subjects=8,
    n_trials=40,
    summary_dim=4,
    n_coupling_layers=2,
)
```

By default, hierarchical workflows save all stochastic subject-level simulated
values as keys such as `v_subj`, `a_subj`, and `t0_subj`. These truth arrays
are needed for random parameter recovery checks.

## Simulate validation data

```python
validation_data = model.simulate(64)
print(validation_data["data"].shape)
```

Expected shape:

```text
(64, 8, 3)
```

The dimensions mean:

- `64` simulated datasets
- `8` subjects per dataset
- `3` aggregate features per subject: `pc`, `mrt`, and `vrt`

## Train or load the workflows

The group workflow estimates population parameters such as `v_mu` and
`v_sigma`. The random workflow estimates subject-level parameters such as `v`,
`a`, and `t0`.

```python
group_history = model.train_workflow(
    max_epochs=100,
    initial_epochs=20,
    n_batch=200,
    batch_size=64,
    validation_data=validation_data,
    patience=10,
    min_delta=0.001,
    file="saved_workflows/ezdm_group.keras",
)

random_history = model.train_random_workflow(
    max_epochs=100,
    initial_epochs=20,
    n_batch=200,
    batch_size=64,
    validation_data=64,
    patience=10,
    min_delta=0.001,
    file="saved_workflows/ezdm_random.keras",
)
```

Both methods load existing saved workflow files by default. Use
`overwrite=True` only when you intentionally want to retrain. The random
workflow receives `validation_data=64` so it can simulate validation data from
its own subject-level training simulator.

## Sample group and subject parameters

```python
test_data = model.simulate(50)

group_samples = model.sample_group_posterior(
    test_data=test_data,
    num_samples=500,
)

subject_samples = model.sample_random_posterior(
    observed_data=test_data,
    group_samples=group_samples,
)

print(group_samples["v_mu"].shape)
print(subject_samples["v"].shape)
group_posterior_df = posterior_to_dataframe(
    group_samples,
    model.priors,
    level="group",
)
subject_posterior_df = posterior_to_dataframe(
    subject_samples,
    model.priors,
    level="subject",
)
print(group_posterior_df.head())
print(subject_posterior_df.head())
```

Typical shapes:

```text
(50, 500)
(50, 500, 8)
```

The subject-level sample arrays keep shape
`(n_datasets, n_samples, n_subjects)`. The sampling methods return
dictionaries because the random-effect workflow needs raw group keys for
shrinkage. Use `posterior_to_dataframe(...)` when you want tidy analysis tables.

## Build group-level recovery tables

The helper below converts group truth and posterior means into long-format
tables. Group recovery pairs one simulated population value with one posterior
mean per dataset and parameter.

```python
def group_recovery_tables(test_data, samples, params):
    """Return simulated and estimated long tables for group recovery checks."""

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


group_simulated, group_estimated = group_recovery_tables(
    test_data,
    group_samples,
    params=["v_mu", "v_sigma", "a_mu", "a_sigma", "t0_mu", "t0_sigma"],
)
```

## Build subject-level recovery tables

Subject recovery uses the saved subject truth keys such as `v_subj` and the
posterior mean of subject-level samples such as `subject_samples["v"]`.

```python
def subject_recovery_tables(test_data, samples, params):
    """Return long tables for subject-level random recovery checks."""

    simulated_rows = []
    estimated_rows = []

    for param in params:
        truth = test_data[f"{param}_subj"]
        posterior_mean = np.mean(samples[param], axis=1)

        for dataset_id in range(truth.shape[0]):
            for subject_id in range(truth.shape[1]):
                if not np.isfinite(truth[dataset_id, subject_id]):
                    continue
                simulated_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "subject_id": subject_id,
                        "param": param,
                        "simulated_value": truth[dataset_id, subject_id],
                    }
                )
                estimated_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "subject_id": subject_id,
                        "param": param,
                        "estimated_value": posterior_mean[dataset_id, subject_id],
                    }
                )

    return pd.DataFrame(simulated_rows), pd.DataFrame(estimated_rows)


subject_simulated, subject_estimated = subject_recovery_tables(
    test_data,
    subject_samples,
    params=["v", "a", "t0"],
)
```

The `np.isfinite(...)` check skips padded subjects in flexible-subject designs.
For the fixed design above, all subjects are real.

## Summarize and estimate recovery

```python
group_summary = aggregate_data(
    group_estimated,
    variables="estimated_value",
    group_by="param",
    stats=["mean", "se", "lower_ci", "upper_ci"],
)

group_recovery = estimate_recovery(
    simulated_data=group_simulated,
    estimated_data=group_estimated,
    group_by="param",
    metrics=["corr", "ccc", "rmse"],
)

subject_recovery = estimate_recovery(
    simulated_data=subject_simulated,
    estimated_data=subject_estimated,
    id_cols=["dataset_id", "subject_id", "param"],
    group_by="param",
    metrics=["corr", "ccc", "rmse"],
)

print(group_summary)
print(group_recovery)
print(subject_recovery)
```

Use `id_cols` when subject-level rows need to match on both dataset and
subject. Model comparison can add a column such as `fit_model` and include it
in `group_by`.

## Draw diagnostic plots

Population recovery draws simulated population truth against posterior means.
Random recovery computes one subject-level metric per simulated dataset and
shows the metric distribution across datasets.

```python
fig_pop = model.plot_population_recovery(
    n_datasets=50,
    num_samples=500,
    params=["v_mu", "v_sigma", "a_mu", "a_sigma", "t0_mu", "t0_sigma"],
    metrics=["corr", "ccc"],
)

fig_random = model.plot_random_recovery(
    n_datasets=50,
    num_samples=500,
    params=["v", "a", "t0"],
    metrics=["corr", "ccc", "rmse"],
)
```

The diagnostic plots are for checking one fitted model. For comparing models,
keep the explicit long-format tables and use `estimate_recovery(...)`.

## Flexible subjects and trials

Use two-value `n_subjects` and `n_trials` ranges when group size and trial
count should vary across simulated datasets.

```python
from bami.inputs import aggregate_summary


flex_model = HierarchicalWorkflow(
    name="ezDM",
    priors=priors,
    simulator=simulate_ezdm_simple,
    observation="aggregate",
    simulator_kwargs={"s": 1},
    obs_names=["pc", "mrt", "vrt"],
    n_subjects=(2, 5),
    n_trials=(10, 15),
    input_format=aggregate_summary(n_range=(10, 14)),
    summary_dim=4,
    n_coupling_layers=2,
)

flex_sim = flex_model.simulate(6)
print(flex_sim["data"].shape)
```

Expected shape:

```text
(6, 4, 5)
```

The output includes the three ezDM summary features, a scaled trial-count
feature, and an active-subject mask. The mask lets the workflow represent
groups with different subject counts in one padded tensor.
