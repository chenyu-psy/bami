# Advanced saved workflows and runtime

These helpers are mainly for long-running scripts that need explicit saved
workflow files. Most examples call workflow methods directly and do not need
these functions.

## Checkpoints

::: bami.inference.checkpoints
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - save_workflow_weights
        - load_workflow_weights
        - load_model

## Runtime

`bami` currently uses the Torch backend for BayesFlow/Keras workflows. Device
placement is a workflow-level setting, not a training option. Construct
workflows with `device="cpu"`, `device="mps"`, or `device="cuda"`. The default
is `device="cpu"` because it is the most stable option across macOS, Linux,
and Windows.

TensorFlow and JAX backends are not part of the current tested `bami` workflow
contract. Supporting those backends would require a separate compatibility
review.

::: bami.inference.runtime
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - validate_device
        - runtime_device

## Examples

```python
from bami.inference.checkpoints import (
    load_model,
    load_workflow_weights,
    save_workflow_weights,
)


# `model` is a configured and trained SimpleWorkflow or HierarchicalWorkflow.
saved_file = save_workflow_weights(model, "saved_workflows/sdm.keras")
model = load_workflow_weights(model, saved_file)
model = load_model(model, saved_file)
```
