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
the underlying BayesFlow workflow. They are instance methods on fitted workflow
objects; your analysis script can name the object however you like. They return
dictionaries so downstream workflow steps can reuse raw posterior keys when
needed.

For simple workflows, call `SimpleWorkflow.sample_posterior(...)`:

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

For hierarchical workflows, call
`HierarchicalWorkflow.sample_group_posterior(...)` when you want group-level
posterior draws:

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

The workflow applies public-scale posterior conversion automatically, so
returned samples use the parameter names and scales researchers normally
interpret. Raw keys remain available in the same dictionary for workflow steps
such as random-effect sampling.

To make a tidy dataframe for reporting, convert the sample dictionary
explicitly:

```python
from bami.utils import posterior_to_dataframe

posterior_df = posterior_to_dataframe(
    group_samples,
    model.priors,
    level="group",
)
```

The dataframe columns are `dataset`, `draw`, `level`, `param`, `basis`,
`quantity`, and `value`.

## Random-effect subject parameters

Hierarchical workflows can also train a deterministic random-effect estimator
for individual subjects. This estimator combines observed subject data with
group posterior draws, so group shrinkage remains part of the subject-level
estimate. It returns point estimates, not posterior draws or calibrated
uncertainty intervals.

Train it separately from the group workflow:

::: bami.workflows.hierarchical.HierarchicalWorkflow.train_random_estimator
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 3

```python
model.train_workflow(file="saved_workflows/ezdm_group.keras")
model.train_random_estimator(file="saved_workflows/ezdm_random_estimator.pt")
```

Then sample group parameters and estimate subject parameters in two explicit
steps:

::: bami.workflows.hierarchical.HierarchicalWorkflow.estimate_random_parameter
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 3

```python
group_samples = model.sample_group_posterior(
    test_data=group_data,
    num_samples=500,
)

subject_estimates = model.estimate_random_parameter(
    observed_data=subject_data,
    group_samples=group_samples,
)
```

`observed_data` may contain one subject or many subjects. The returned
`pandas.DataFrame` has one row per active subject. By default, it contains
`dataset_id`, `subject_id`, and public parameter estimates. Use
`include_scales=True` to add raw, group-centered deviation, standardized `z`,
and group-scale diagnostic columns.

For trial-level workflows, trial-count handling comes from the workflow object.
Fixed trial models issue a warning if observed subjects use a different trial
count than the training `n_trials`, because the resulting subject posteriors may
be less reliable. Flexible trial models accept raw variable-length subject trial
arrays and add the padding and `active_trial` mask internally, as long as the
observed trial counts fit inside the range used for `n_trials`.
