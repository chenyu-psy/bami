# Workflows

Use workflows to connect a cognitive-model simulator to BayesFlow. A workflow
keeps the main modeling choices in one place: priors, simulator, observation
shape, trial or subject design, and training behavior.

## Which workflow should I use?

| Workflow | Use when | Simulator call |
| --- | --- | --- |
| `SimpleWorkflow` | One prior draw generates one dataset. | Once per simulated dataset. |
| `HierarchicalWorkflow` | One group draw generates several subjects. | Once per subject in each simulated dataset. |

Most arguments are shared. The main difference is that `HierarchicalWorkflow`
also needs a subject-count design and can keep subject-level truth values for
recovery checks.

## Basic recipe

A typical workflow setup follows this order:

1. Write `priors` for the parameters you want to infer.
2. Choose or write a `simulator` that receives public parameter values,
   `n_trials`, and `rng`.
3. Set `observation` to match the simulator output shape.
4. Set trial counts with `n_trials` or `n_trials_range`.
5. For hierarchical workflows, set subject counts with `n_subjects` or
   `n_subjects_range`.
6. Build the workflow object.
7. Simulate, train, and sample from the posterior.

## Setting priors

`priors` is a compact dictionary. Each dictionary-valued parameter is inferred
by the workflow:

```python
priors = {
    "a": {"mean": 0.0, "sd": 1.0, "link": "identity"},
    "c": {"mean": "normal(0, 1)", "sd": "exponential(1)", "link": "log"},
    "p": {"mean": "logistic(0, 0.75)", "sd": 0.2, "link": "logit"},
}
```

Each inferred parameter has three fields:

| Field | Meaning |
| --- | --- |
| `mean` | Raw-space center for the parameter. This can be a number or a distribution string. |
| `sd` | Raw-space standard deviation. This can be a positive number or a positive distribution string. |
| `link` | Transformation from raw values to the public values passed to the simulator. |

The supported links are:

| Link | Use when | Public scale |
| --- | --- | --- |
| `identity` | The parameter can be any real value. | `raw` |
| `log` | The parameter must be positive. | `exp(raw)` |
| `logit` | The parameter must be between 0 and 1. | inverse-logit of `raw` |

Distribution strings use a short function-like format, such as
`"normal(0, 1)"`, `"logistic(0, 0.75)"`, or `"exponential(1)"`. Use
distribution strings when the center or spread should vary across simulated
datasets instead of staying fixed.

`sd` must always be positive. If `sd` is a distribution string, use a
positive-support distribution such as `"exponential(1)"`, `"gamma(2, 0.1)"`,
`"beta(2, 8)"`, or a positive uniform range such as `"uniform(0.1, 0.3)"`.

Scalar values in `priors` are fixed simulator constants. They are passed to the
simulator but are not inferred:

```python
priors = {
    "a": {"mean": 0.0, "sd": 1.0, "link": "log"},
    "s": 1.0,
}
```

In this example, `a` is inferred and `s` is fixed.

## Simple example

Use `SimpleWorkflow` when one prior draw should generate one dataset.

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

## Hierarchical example

Use `HierarchicalWorkflow` when one group draw should generate several
subjects.

```python
from bami.inference import transform_hierarchical_samples
from bami.simulators import simulate_ezdm_simple
from bami.workflows import HierarchicalWorkflow


model = HierarchicalWorkflow(
    name="ezDM",
    priors={
        "v": {"mean": "normal(0, 0.6)", "sd": 0.15, "link": "log"},
        "a": {"mean": "normal(0.2, 0.4)", "sd": 0.15, "link": "log"},
        "t0": {"mean": "logistic(-2, 0.5)", "sd": 0.15, "link": "logit"},
    },
    simulator=simulate_ezdm_simple,
    observation="aggregate",
    simulator_kwargs={"s": 1},
    obs_names=["pc", "mrt", "vrt"],
    n_subjects=3,
    n_trials=20,
    transform_samples=transform_hierarchical_samples,
)

sim = model.workflow.simulate(4)
print(sim["data"].shape)
```

## Choosing key arguments

### Observation shape

Use `observation="aggregate"` when the simulator returns one fixed-width row of
summary statistics, counts, or proportions for each dataset or subject.

Use `observation="trial"` when the simulator returns one row per trial. This is
useful when trial-level responses should be summarized by the neural network.

### Trial counts

Use `n_trials` for a fixed trial count:

```python
model = SimpleWorkflow(..., n_trials=100)
```

Use `n_trials_range=(low, high)` when simulated datasets should vary in trial
count:

```python
model = SimpleWorkflow(..., n_trials=None, n_trials_range=(50, 201))
```

The upper bound is exclusive, so this example draws 50 to 200 trials.

### Subject counts

Hierarchical workflows also need a subject-count design. Use `n_subjects` for a
fixed number of subjects:

```python
model = HierarchicalWorkflow(..., n_subjects=30)
```

Use `n_subjects_range=(low, high)` when simulated datasets should vary in
subject count:

```python
model = HierarchicalWorkflow(..., n_subjects=None, n_subjects_range=(20, 41))
```

### Observation names

`obs_names` labels the columns returned by the simulator. For aggregate ezDM
summaries, this might be:

```python
obs_names = ["pc", "mrt", "vrt"]
```

For trial-level SDM errors, this might be:

```python
obs_names = ["error_rad"]
```

### Input format

Use `input_format` when an aggregate row needs to encode trial count explicitly.
This is useful when reliability changes with trial count and the workflow
should receive that information as part of the data row.

For trial-level workflows with flexible trial counts, `bami` pads rows and adds
an active-trial mask automatically.

### Subject truth

Use `keep_subject_truth=True` in hierarchical workflows when recovery checks
need subject-level true values. This stores subject values for later evaluation
without changing the simulator interface.

## Common workflow actions

After constructing a workflow object, the usual actions are:

```python
sim = model.workflow.simulate(4)
history = model.train_workflow(max_epochs=50, ...)
posterior = model.convert_posterior(raw_samples)
```

`model.workflow.simulate(...)` is the BayesFlow workflow method created by
`bami`. The `train_workflow(...)` and `convert_posterior(...)` methods live on
the `bami` workflow object.

::: bami.workflows.simple.SimpleWorkflow.train_workflow
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 3

::: bami.workflows.simple.SimpleWorkflow.convert_posterior
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 3

## Constructor reference

Use this section when you need the complete argument list.

### SimpleWorkflow

::: bami.workflows.simple.SimpleWorkflow
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 4
      members: false

### HierarchicalWorkflow

::: bami.workflows.hierarchical.HierarchicalWorkflow
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 4
      members: false
