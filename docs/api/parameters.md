# Parameters

Use this page after training a workflow when you want parameter estimates.
Most analyses need one or both levels:

- population or group-level parameters, such as group means and group standard
  deviations
- individual-level parameters for observed subjects in a hierarchical workflow

## Population parameters

Use `sample_posterior()` to draw posterior samples for the parameters inferred
by a trained BayesFlow workflow.

::: bami.evaluation.metrics.bayesflow.sample_posterior
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 3

```python
from bami.evaluation import sample_posterior


test_data = model.workflow.simulate(20)
samples = sample_posterior(
    workflow=model.workflow,
    test_data=test_data,
    num_samples=500,
)
```

The workflow applies its public-scale posterior conversion through
`model.convert_posterior(...)`, so returned samples use the parameter names and
scales researchers normally interpret.

## Individual parameters

Use `summarize_subject_posterior()` when you have observed subject counts and
want individual-level parameter summaries from a hierarchical workflow. This is
for estimating parameters in observed data; recovery checks with known
simulated truth are documented separately under evaluation and recovery.

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
