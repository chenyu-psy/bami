# Workflows

Use this page to look up workflow constructors and methods. For step-by-step
teaching examples, start with the [SDM workflow article](../articles/sdm-fixed-simple.md),
the [hierarchical ezDM example](../examples/hierarchical-ezdm.md), or
[workflow structure guide](../articles/advanced-workflows.md).

## Constructor reference

Use constructors to define the simulator, priors, observation shape, trial
design, and runtime device for a workflow.

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

## Method reference

Examples assume that `model` is already a configured workflow object. The
examples are intentionally short; full workflows live in the articles linked
above.

### Common methods

These methods are available on both `SimpleWorkflow` and
`HierarchicalWorkflow`. The reference blocks below use `SimpleWorkflow` as the
representative signature source; hierarchical workflows expose the same method
names, with data shapes following the hierarchical workflow contract.

#### simulate

::: bami.workflows.simple.SimpleWorkflow.simulate
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 5

Example:

```python
sim = model.simulate(4)
print(sim["data"].shape)
```

#### train_workflow

::: bami.workflows.simple.SimpleWorkflow.train_workflow
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 5

Example:

```python
validation_data = model.simulate(32)
history = model.train_workflow(
    max_epochs=5,
    n_batch=10,
    validation_data=validation_data,
    file="saved_workflows/simple.keras",
)
```

### Simple workflow methods

#### SimpleWorkflow.sample_posterior

::: bami.workflows.simple.SimpleWorkflow.sample_posterior
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 5

Example:

```python
test_data = model.simulate(4)
samples = model.sample_posterior(test_data, num_samples=200)
print(samples["c"].shape)
```

### Hierarchical workflow methods

Use these methods for group posterior sampling and the current subject-level
point-estimate path.

#### HierarchicalWorkflow.sample_group_posterior

::: bami.workflows.hierarchical.HierarchicalWorkflow.sample_group_posterior
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 5

Example:

```python
test_data = model.simulate(4)
group_samples = model.sample_group_posterior(
    test_data=test_data,
    num_samples=200,
)
print(group_samples["c_mu"].shape)
```

#### HierarchicalWorkflow.train_random_estimator

::: bami.workflows.hierarchical.HierarchicalWorkflow.train_random_estimator
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 5

Example:

```python
model.train_random_estimator(
    max_epochs=20,
    file="saved_workflows/random_estimator.pt",
)
```

#### HierarchicalWorkflow.estimate_random_parameter

::: bami.workflows.hierarchical.HierarchicalWorkflow.estimate_random_parameter
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 5

Example:

```python
subject_estimates = model.estimate_random_parameter(
    observed_data=test_data,
    group_samples=group_samples,
)
print(subject_estimates.head())
```

### Legacy subject posterior workflow

Use these methods only when you need subject-level posterior draws. For
ordinary subject-level recovery, prefer `train_random_estimator(...)` and
`estimate_random_parameter(...)`.

#### HierarchicalWorkflow.train_random_workflow

::: bami.workflows.hierarchical.HierarchicalWorkflow.train_random_workflow
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 5

Example:

```python
model.train_random_workflow(
    max_epochs=20,
    validation_data=32,
    file="saved_workflows/random_workflow.keras",
)
```

#### HierarchicalWorkflow.sample_random_posterior

::: bami.workflows.hierarchical.HierarchicalWorkflow.sample_random_posterior
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 5

Example:

```python
subject_samples = model.sample_random_posterior(
    observed_data=test_data,
    group_samples=group_samples,
)
print(subject_samples["c"].shape)
```

## Argument lookup

| Argument | Used in | Quick meaning | More detail |
| --- | --- | --- | --- |
| `priors` | both workflows | Parameters to estimate plus fixed simulator constants. | [Priors](../articles/priors.md) |
| `simulator` | both workflows | Function that receives public parameter values and `n_trials`. | [Simulators](../articles/simulators.md) |
| `observation` | both workflows | `"aggregate"` for one summary row; `"trial"` for one row per trial. | [Simulators](../articles/simulators.md) |
| `obs_names` | both workflows | Names for simulator output columns, in returned order. | [Simulators](../articles/simulators.md) |
| `simulator_kwargs` | both workflows | Fixed technical options passed to the simulator. | [Simulators](../articles/simulators.md) |
| `n_trials` | both workflows | Fixed trial count, or a variable range `(low, high)` with an exclusive upper bound. | [Workflow structure](../articles/advanced-workflows.md) |
| `n_subjects` | hierarchical only | Fixed subject count, or a variable range `(low, high)` with an exclusive upper bound. | [Workflow structure](../articles/advanced-workflows.md) |
| `input_format` | both workflows | Optional encoder for aggregate rows, often used to include trial count. | [Input formats](inputs/formats.md) |
| `device` | both workflows | Runtime device: `"cpu"`, `"mps"`, or `"cuda"`. | [Runtime](inference/checkpoints-runtime.md) |

## Related guides

- [Build and check a simple SDM workflow](../articles/sdm-fixed-simple.md)
- [Fit a hierarchical ezDM workflow](../examples/hierarchical-ezdm.md)
- [Choose the right workflow structure](../articles/advanced-workflows.md)
- [Write a simulator for your own model](../articles/simulators.md)
- [Choose priors and links for model parameters](../articles/priors.md)
