# Advanced saved workflows and runtime

These helpers are mainly for long-running scripts that need explicit saved
workflow files or device selection. Most examples call workflow methods
directly and do not need these functions.

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

`bami` currently uses the Torch backend for BayesFlow/Keras workflows.
The `torch_device` training argument and `configure_torch_device(...)` helper
only control Torch device selection. Use `None` to leave the current Torch
default unchanged, or pass `"cpu"`, `"mps"`, or `"cuda"` when a script should
request a specific Torch device.

TensorFlow and JAX backends are not part of the current tested `bami` workflow
contract. Supporting those backends would require a separate compatibility
review rather than just changing `torch_device`.

::: bami.inference.runtime
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - configure_torch_device

## Examples

```python
from bami.inference.checkpoints import (
    load_model,
    load_workflow_weights,
    save_workflow_weights,
)
from bami.inference.runtime import configure_torch_device


device = configure_torch_device("cpu")

# `model` is a configured and trained SimpleWorkflow or HierarchicalWorkflow.
saved_file = save_workflow_weights(model, "saved_workflows/sdm.keras")
model = load_workflow_weights(model, saved_file)
model = load_model(model, saved_file)
```
