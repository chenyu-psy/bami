# Parameters

Use this page after training a workflow when you want parameter estimates.
Most analyses need one or both levels:

- population or group-level parameters, such as group means and group standard
  deviations
- individual-level parameters for observed subjects in a hierarchical workflow

## Population parameters

Use model-level sampling methods to draw posterior samples after training or
loading a saved workflow. These methods keep the model object as the first
thing researchers work with, instead of asking them to reach through
`model.workflow`.

For simple workflows, call `model.sample_posterior(...)`:

::: bami.workflows.simple.SimpleWorkflow.sample_posterior
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 3

```python
test_data = model.simulate(20)
samples = model.sample_posterior(
    test_data=test_data,
    num_samples=500,
)
```

For hierarchical workflows, call `model.sample_group_posterior(...)` when you
want group-level posterior draws:

::: bami.workflows.hierarchical.HierarchicalWorkflow.sample_group_posterior
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 3

```python
test_data = model.simulate(20)
group_samples = model.sample_group_posterior(
    test_data=test_data,
    num_samples=500,
)
```

The workflow applies its public-scale posterior conversion through
`model.convert_posterior(...)` when one is configured, so returned samples use
the parameter names and scales researchers normally interpret.

## Random-effect subject parameters

Hierarchical workflows can also train a separate random-effect workflow for
individual subjects. This workflow estimates standardized subject deviations
and combines them with group posterior draws, so group shrinkage remains part
of the estimate.

Train it separately from the group workflow:

::: bami.workflows.hierarchical.HierarchicalWorkflow.train_random_workflow
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 3

```python
model.train_workflow(file="saved_workflows/ezdm_group.keras")
model.train_random_workflow(file="saved_workflows/ezdm_random.keras")
```

Then sample group parameters and subject parameters in two explicit steps:

::: bami.workflows.hierarchical.HierarchicalWorkflow.sample_random_posterior
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 3

```python
group_samples = model.sample_group_posterior(
    test_data=group_data,
    num_samples=500,
)

subject_samples = model.sample_random_posterior(
    observed_data=subject_data,
    group_samples=group_samples,
)
```

`observed_data` may contain one subject or many subjects. The returned subject
sample arrays keep shape `(n_datasets, n_samples, n_subjects)`.

For trial-level workflows, trial-count handling comes from the model. Fixed
trial models issue a warning if observed subjects use a different trial count
than the training `n_trials`, because the resulting subject posteriors may be
less reliable. Flexible trial models accept raw variable-length subject trial
arrays and add the padding and `active_trial` mask internally, as long as the
observed trial counts fit inside `n_trials_range`.

## Legacy posthoc subject parameters

Use `summarize_subject_posterior()` only when you specifically need the current
posthoc helper for observed subject counts in a hierarchical workflow. This
approximate route remains available while the random-effect posterior sampling
design is evaluated.

::: bami.inference.posthoc.summarize_subject_posterior
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 3

```python
from bami.inference import summarize_subject_posterior


subject_params = summarize_subject_posterior(
    model,
    counts,
    group_samples=group_samples,
    n_candidates=4000,
)
```

The returned table contains one row per subject and parameter, including
posterior medians, intervals, and diagnostic columns such as effective sample
size.
