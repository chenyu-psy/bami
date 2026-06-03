# Write a simulator for your own model

`bami` is simulator-first. A simulator is the piece you customize: it turns
model parameters into simulated behavioral performance or observation rows. The
workflow handles priors, repeated simulation, training, posterior sampling, and
diagnostics around that function.

The built-in SDM, ezDM, and M3 simulators are examples shipped with the package.
They are not the only simulators you can use. If a model can be written as a
function that receives parameter values and returns data, it can usually be
connected to a `bami` workflow.

For prior and link-function details, see [Priors](priors.md). This article
focuses on writing the simulator function itself.

## The simulator contract

For a simple workflow, `bami` calls your simulator like this:

```python
simulator(**params, n_trials=n_trials, **simulator_kwargs)
```

`params` are the public-scale values drawn from your priors. For example, if
your workflow estimates `v`, `a`, and `t0`, then those values are passed into
your simulator as function arguments. `n_trials` is the trial count requested by
the workflow. `simulator_kwargs` are fixed settings you pass when constructing
the workflow.

The simulator should return one of two shapes:

| Observation | Return shape | Meaning |
| --- | --- | --- |
| `aggregate` | `(features,)` | One summary row for the whole simulated dataset. |
| `trial` | `(n_trials, features)` | One row per simulated trial. |

Use `np.random.seed(...)` before simulation when you want reproducible examples.
The workflow does not pass a random-generator object into the simulator.

## A hand-written ezDM simulator

This example mirrors the built-in `simulate_ezdm_simple()` function. It returns
one aggregate row with proportion correct (`pc`), mean response time (`mrt`),
and response-time variance (`vrt`).

The same pattern works for your own model: receive public model parameters,
simulate behavior, and return the summary row or trial rows that the workflow
should learn from.

```python
import numpy as np


def inv_logit(x):
    """Convert a real-valued logit to a probability."""

    if x >= 0:
        return 1 / (1 + np.exp(-x))
    exp_x = np.exp(x)
    return exp_x / (1 + exp_x)


def edge_correct_pc(n_correct, n_trials):
    """Avoid impossible observed accuracies of exactly 0 or 1."""

    if n_correct == 0:
        return 0.5 / n_trials
    if n_correct == n_trials:
        return 1 - 0.5 / n_trials
    return n_correct / n_trials


def simulate_my_ezdm(v, a, t0, n_trials=100, s=0.1):
    """Return one ezDM summary row: [pc, mrt, vrt]."""

    s2 = s**2
    logit_pc = v * a / s2
    pc_true = inv_logit(logit_pc)
    mdt = (a / (2 * v)) * np.tanh(logit_pc / 2)
    mrt = mdt + t0

    numerator = logit_pc * (
        logit_pc * pc_true**2 - logit_pc * pc_true + pc_true - 0.5
    )
    vrt = numerator / ((abs(v) / s) ** 4)

    n_correct = np.random.binomial(n_trials, pc_true)
    pc = edge_correct_pc(n_correct, n_trials)
    return np.array([pc, mrt, vrt], dtype=np.float32)
```

The important point is not the ezDM equations themselves. The important point is
the data flow:

- `v`, `a`, and `t0` enter as model parameters.
- `n_trials` controls the simulated sample size.
- `s` is a fixed model setting.
- the return value is one aggregate row in a fixed order: `[pc, mrt, vrt]`.

## Use the simulator in a workflow

Pass the simulator function directly to `SimpleWorkflow`. The parameter names
in `param_names` and `priors` should match the simulator arguments you want to
estimate.

```python
import numpy as np

from bami.workflows import SimpleWorkflow


np.random.seed(2026)

model = SimpleWorkflow(
    name="custom-ezDM",
    param_names=["v", "a", "t0"],
    priors={
        "v": {"mean": "normal(0.2, 0.1)", "sd": 0.05, "link": "identity"},
        "a": {"mean": "normal(1.4, 0.2)", "sd": 0.1, "link": "log"},
        "t0": {"mean": "normal(0.3, 0.05)", "sd": 0.02, "link": "log"},
        "s": 0.1,
    },
    simulator=simulate_my_ezdm,
    observation="aggregate",
    obs_names=["pc", "mrt", "vrt"],
    n_trials=50,
)

sim = model.simulate(4)
print(sim["data"].shape)
```

The expected shape is:

```text
(4, 1, 3)
```

The middle dimension is `1` because an aggregate simulator returns one summary
row per simulated dataset. The last dimension has three features:
`pc`, `mrt`, and `vrt`.

## Fixed settings

Use scalar entries in `priors` for fixed model settings that should be visible
beside the estimated parameters. In the ezDM example, `s` is fixed and passed to
the simulator, but it is not estimated:

```python
priors = {
    "v": {"mean": "normal(0.2, 0.1)", "sd": 0.05, "link": "identity"},
    "a": {"mean": "normal(1.4, 0.2)", "sd": 0.1, "link": "log"},
    "t0": {"mean": "normal(0.3, 0.05)", "sd": 0.02, "link": "log"},
    "s": 0.1,
}
```

Use `simulator_kwargs` for technical options that are not part of the model
parameter table:

```python
model = SimpleWorkflow(
    ...,
    priors=priors,
    simulator=simulate_my_model,
    simulator_kwargs={"return_summary": True},
)
```

Both approaches pass fixed values to the simulator. Scalar prior entries are
usually clearer for model constants; `simulator_kwargs` are useful for
implementation options.

## Checklist for your own simulator

When adapting this pattern to your own model, check four things before
training:

1. The simulator argument names match the estimated parameter names.
2. The simulator accepts `n_trials`.
3. The return shape matches `observation`.
4. `obs_names` uses the same order as the returned row or trial columns.

A quick `model.simulate(4)` call should be the first test. Inspect the shape
before starting a longer training run.
