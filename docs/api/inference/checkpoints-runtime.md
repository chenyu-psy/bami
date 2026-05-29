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
