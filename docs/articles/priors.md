# Choose priors and links for model parameters

Priors describe which parameter values the workflow should learn from during
simulation and training. They define the range of simulated behavior the
workflow sees before it is applied to real or held-out data.

This article explains the prior dictionary format, supported distribution
strings, and link functions. See [Simulators](simulators.md) for the simulator
contract that receives public-scale parameter values.

## Prior dictionary format

A prior dictionary has two kinds of entries:

- dictionary entries for parameters the workflow estimates
- scalar entries for fixed simulator constants

```python
priors = {
    "c": {"mean": "normal(4, 1.2)", "sd": 0.5, "link": "softplus"},
    "kappa": {"mean": "normal(4, 1.2)", "sd": 0.5, "link": "softplus"},
    "noise": 1.0,
}
```

Here, `c` and `kappa` are estimated. `noise` is passed to the simulator but is
not estimated.

Each estimated parameter has three fields:

| Field | Meaning |
| --- | --- |
| `mean` | Raw-space center for the parameter. |
| `sd` | Raw-space standard deviation around that center. |
| `link` | Transform from raw values to public simulator values. |

The key point is that `mean` and `sd` live on the raw scale. The simulator
receives the value after the link is applied.

## Distribution strings

`mean` and `sd` can be numbers or compact distribution strings. Distribution
strings are useful when a setting should vary across simulated datasets.

Supported distribution strings are:

| Distribution | Arguments | Example |
| --- | --- | --- |
| `normal` | mean, sd | `"normal(0, 1)"` |
| `logistic` | location, scale | `"logistic(0, 0.75)"` |
| `uniform` | low, high | `"uniform(-1, 1)"` |
| `truncnorm` | lower z, upper z, mean, sd | `"truncnorm(-2, 2, 0, 1)"` |
| `beta` | alpha, beta | `"beta(2, 8)"` |
| `gamma` | shape, scale | `"gamma(2, 0.1)"` |
| `exponential` | scale | `"exponential(1)"` |
| `binomial` | n, p | `"binomial(10, 0.5)"` |

For `sd`, the value must always be positive. Practical `sd` choices are:

- a positive number, such as `0.5`
- `"uniform(low, high)"` with `0 < low < high`
- `"exponential(scale)"` with `scale > 0`
- `"gamma(shape, scale)"` with positive arguments
- `"beta(alpha, beta)"` with positive arguments

Use broad distributions carefully. Very broad priors can simulate behavior that
is not relevant to the study and make training harder.

## Link functions

The link tells `bami` how to convert raw values into public values before
calling the simulator.

| Link | Public scale | Use when |
| --- | --- | --- |
| `identity` | any real value | The parameter can be negative, zero, or positive. |
| `log` | positive value | The parameter must be positive and multiplicative changes are natural. |
| `softplus` | positive value | The parameter must be positive, with a smoother lower boundary than `log`. |
| `logit` | value between 0 and 1 | The parameter is a probability or proportion. |
| `probit` | value between 0 and 1 | The parameter is probability-scale with a normal-CDF transform. |
| `cloglog` | value between 0 and 1 | The parameter is probability-scale with an asymmetric transform. |

### Unconstrained parameter

Use `identity` when the simulator can accept any real value.

```python
priors = {
    "bias": {"mean": "normal(0, 1)", "sd": 0.2, "link": "identity"},
}
```

Raw values and public values are the same.

### Positive parameter

Use `log` or `softplus` when the simulator needs a positive value. SDM `c` and
`kappa` are examples.

```python
priors = {
    "c": {"mean": "normal(4, 1.2)", "sd": 0.5, "link": "softplus"},
    "kappa": {"mean": "normal(4, 1.2)", "sd": 0.5, "link": "softplus"},
}
```

Both fields are centered on the raw scale. The simulator receives positive
public values after the `softplus` transform.

### Probability parameter

Use `logit`, `probit`, or `cloglog` when the simulator needs a value between
0 and 1.

```python
priors = {
    "guess_rate": {"mean": "logistic(0, 0.75)", "sd": 0.2, "link": "logit"},
}
```

The raw values can range over the real line, but the simulator receives a
probability-scale value.

## Fixed constants

Use scalar prior entries when a value is part of the model definition but not a
parameter to estimate.

```python
priors = {
    "v": {"mean": "normal(1.5, 1)", "sd": 0.25, "link": "softplus"},
    "a": {"mean": "normal(1.5, 1)", "sd": 0.25, "link": "softplus"},
    "t0": {"mean": "logistic(-0.5, 0.25)", "sd": 0.15, "link": "logit"},
    "s": 1.0,
}
```

Scalar constants are passed to the simulator, but posterior samples and
recovery diagnostics are only created for dictionary-valued parameters.
