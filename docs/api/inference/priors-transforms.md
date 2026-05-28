# Advanced prior utilities

Most users pass prior dictionaries to `SimpleWorkflow` or
`HierarchicalWorkflow` and let the workflow handle raw-space sampling and
posterior transforms. Use these utilities only when checking prior draws or
writing custom workflow code.

## Prior utilities

::: bami.inference.priors
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - draw_prior
        - draw_prior_with_raw
        - raw_key
        - mu_raw_key
        - log_sigma_key
        - validate_positive_sd_spec
        - transform_simple_samples
        - transform_hierarchical_samples
        - apply_link
        - invert_link

## Sample transforms

::: bami.inference.transforms
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - transform_simple_samples
        - transform_hierarchical_samples

## Examples

```python
import numpy as np

from bami.inference.priors import (
    apply_link,
    draw_prior,
    draw_prior_with_raw,
    invert_link,
    log_sigma_key,
    mu_raw_key,
    raw_key,
    transform_hierarchical_samples,
    transform_simple_samples,
    validate_positive_sd_spec,
)


priors = {
    "c": {"mean": "normal(1, 0.35)", "sd": 0.15, "link": "log"},
    "kappa": {"mean": "normal(1.2, 0.35)", "sd": 0.15, "link": "log"},
}

draw = draw_prior(**priors)
draw_with_raw = draw_prior_with_raw(**priors)

c_raw_name = raw_key("c")
c_mu_raw_name = mu_raw_key("c")
c_log_sigma_name = log_sigma_key("c")

validate_positive_sd_spec("c", priors["c"]["sd"])
public_value = apply_link(0.0, link="log")
raw_value = invert_link(public_value, link="log")

simple_samples = transform_simple_samples(
    {"c_raw": np.array([[0.0, 0.1]])},
    {"c": {"mean": 0.0, "sd": 1.0, "link": "log"}},
)
hier_samples = transform_hierarchical_samples(
    {"c_mu_raw": np.array([[0.0]]), "c_log_sigma": np.array([[0.1]])},
    {"c": {"mean": 0.0, "sd": 1.0, "link": "log"}},
)
```
