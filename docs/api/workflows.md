# Workflows

Use workflows to connect a cognitive-model simulator to BayesFlow. Both
workflow classes use the same basic ingredients: priors, a simulator,
an observation contract, trial-count settings, and optional input formatting.

## Which workflow should I use?

| Workflow | Use when | Simulator call |
| --- | --- | --- |
| `SimpleWorkflow` | One prior draw generates one dataset. | Once per simulated dataset. |
| `HierarchicalWorkflow` | One group draw generates several subjects. | Once per subject in each simulated dataset. |

Most arguments are shared. The main difference is that `HierarchicalWorkflow`
also needs a subject-count design and can keep subject-level truth values for
recovery checks.

## Shared concepts

- `priors` describe which parameters are inferred and how raw samples map to
  public parameter scales.
- `simulator` receives public parameters, `n_trials`, and `rng`.
- `observation="aggregate"` means one fixed-width summary row per dataset or
  subject.
- `observation="trial"` means one row per trial.
- `n_trials` gives a fixed design; `n_trials_range=(low, high)` draws trial
  counts from `[low, high)`.
- `input_format` is useful when aggregate rows need an explicit encoded trial
  count.

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

## Parameter differences

| Parameter group | `SimpleWorkflow` | `HierarchicalWorkflow` |
| --- | --- | --- |
| Parameter names | Uses `param_names` for public parameters. | Infers group-level raw variables from `priors`. |
| Subject counts | Not used. | Uses `n_subjects` or `n_subjects_range`. |
| Trial counts | Uses `n_trials` or `n_trials_range`. | Uses `n_trials` or `n_trials_range` per subject. |
| Subject truth | Not used. | `keep_subject_truth` stores subject values for recovery checks. |
| Posthoc setup | Not used. | `posthoc_estimator` and `posthoc_kwargs` configure subject posterior summaries. |

## Simple example

```python
from bami.simulators import GRID_SIZE, simulate_sdm_simple
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
    simulator_kwargs={"grid_size": GRID_SIZE, "error_scale": 180.0},
    obs_names=["error"],
    n_trials=25,
)

sim = model.workflow.simulate(4)
print(sim["data"].shape)
```

## Hierarchical example

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
