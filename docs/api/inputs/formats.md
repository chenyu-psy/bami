# Input formats

Input-format functions tell a workflow how to encode simulator output rows
before they are passed to BayesFlow. They are most useful when aggregate rows
need to carry trial-count information.

## Input format functions

::: bami.inputs.formats
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - aggregate_summary
        - proportions
        - counts

## Examples

```python
from bami.inputs import aggregate_summary, counts, proportions


summary_format = aggregate_summary(n_range=(10, 100))
encoded = summary_format.encode([0.72, 0.41, 0.08], n_trials=40)
print(summary_format.output_width(data_width=3))
print(summary_format.transform_n(40))
print(summary_format.to_dict())

prop_format = proportions(n_range=(20, 200))
count_format = counts(add_n=True, n_range=(20, 200))
```
