# fit_workflow

Use `fit_workflow()` when a configured workflow should be trained with shared
early-stopping settings and optional checkpoint loading or saving.

::: bami.training.fit_workflow
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 3

## Examples

```python
from bami.training import fit_workflow


# `model` is a configured SimpleWorkflow or HierarchicalWorkflow object.
history = fit_workflow(
    model,
    max_epochs=50,
    initial_epochs=10,
    n_batch=100,
    batch_size=64,
    validation_data=64,
    patience=8,
    min_delta=0.001,
    file="checkpoints/model.keras",
)
```
