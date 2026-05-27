"""Shared prior and link-function helpers for model simulation.

These functions implement the project-wide ``mean`` / ``sd`` / ``link`` prior
format used by current model families.
"""

import re

import numpy as np
from scipy.stats import truncnorm


MIN_POSITIVE_SCALE = 1e-8
DIST_PATTERN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$")
POSITIVE_SD_DISTS = {"uniform", "exponential", "gamma", "beta"}


def draw_prior(rng=None, **param_specs):
    """Draw public-scale parameter values from group prior specifications.

    Parameters
    ----------
    rng : numpy.random.Generator, optional
        Random number generator. Defaults to ``np.random``.
    **param_specs : dict
        Parameter specifications. A parameter can be a scalar constant or a
        dictionary with ``mean``, ``sd``, and ``link``.

    Returns
    -------
    dict
        Public-scale parameter dictionary.
    """

    if rng is None:
        rng = np.random

    prior_draws = {}
    for name, spec in param_specs.items():
        if np.isscalar(spec):
            prior_draws[name] = spec
            continue

        if isinstance(spec, dict):
            group_mean = draw_group_mean_raw(name, spec, rng)
            group_sd = draw_group_sd_raw(name, spec, rng)
            raw_value = rng.normal(loc=group_mean, scale=group_sd)
            prior_draws[name] = apply_link(raw_value, spec.get("link", "identity"))
            continue

        raise TypeError(f"Invalid specification for parameter '{name}': {spec}")

    return prior_draws


def raw_key(name: str) -> str:
    """Return the raw-space inference key for a simple parameter."""

    return f"{name}_raw"


def mu_raw_key(name: str) -> str:
    """Return the raw-space group-mean key for a hierarchical parameter."""

    return f"{name}_mu_raw"


def log_sigma_key(name: str) -> str:
    """Return the raw-space log-sigma key for a hierarchical parameter."""

    return f"{name}_log_sigma"


def subj_raw_key(name: str, subj: int) -> str:
    """Return the raw-space subject key for a fixed subject slot."""

    return f"{name}_subj_raw_{subj}"


def subj_key(name: str, subj: int) -> str:
    """Return the public subject key for a fixed subject slot."""

    return f"{name}_subj_{subj}"


def draw_prior_with_raw(rng=None, **param_specs):
    """Draw simple-model parameters on raw and public scales.

    Parameters
    ----------
    rng : numpy.random.Generator, optional
        Random number generator.
    **param_specs
        Parameter prior specifications.

    Returns
    -------
    dict
        Public parameter values plus raw inference values. New-format specs
        draw a group mean first and then one actual raw parameter from
        ``Normal(group_mean, sd)``.
    """

    if rng is None:
        rng = np.random

    prior_draws = {}
    for name, spec in param_specs.items():
        if np.isscalar(spec):
            prior_draws[name] = spec
            continue

        if isinstance(spec, dict):
            link = spec.get("link", "identity")
            group_mean = draw_group_mean_raw(name, spec, rng)
            group_sd = draw_group_sd_raw(name, spec, rng)
            raw_value = rng.normal(loc=group_mean, scale=group_sd)
            prior_draws[raw_key(name)] = raw_value
            prior_draws[name] = apply_link(raw_value, link)
            continue

        raise TypeError(f"Invalid specification for parameter '{name}': {spec}")

    return prior_draws


def is_group_prior_spec(spec: dict) -> bool:
    """Return whether a parameter uses the group-mean plus SD prior format."""

    return "mean" in spec and "sd" in spec


def is_distribution_value(value) -> bool:
    """Return whether a spec value is a distribution string."""

    return isinstance(value, str) and DIST_PATTERN.match(value) is not None


def is_stochastic_mean(spec: dict) -> bool:
    """Return whether the group mean should be an inferred variable."""

    _check_group_prior_spec("", spec)
    return is_distribution_value(spec.get("mean"))


def is_stochastic_sd(spec: dict) -> bool:
    """Return whether group SD should be an inferred variable."""

    _check_group_prior_spec("", spec)
    return is_distribution_value(spec.get("sd"))


def parse_distribution_expr(expr: str) -> tuple[str, list[float]]:
    """Parse a compact distribution expression."""

    match = DIST_PATTERN.match(expr)
    if match is None:
        raise ValueError(f"Invalid distribution expression: {expr!r}.")

    dist_name = match.group(1).lower()
    arg_text = match.group(2).strip()
    if not arg_text:
        args = []
    else:
        try:
            args = [float(part.strip()) for part in arg_text.split(",")]
        except ValueError as exc:
            raise ValueError(
                f"Distribution arguments must be numeric in {expr!r}."
            ) from exc
    return dist_name, args


def draw_distribution_expr(expr: str, rng=None) -> float:
    """Draw one value from a compact distribution expression."""

    if rng is None:
        rng = np.random

    dist_name, args = parse_distribution_expr(expr)
    match dist_name:
        case "normal":
            _require_arg_count(expr, args, 2)
            return float(rng.normal(loc=args[0], scale=args[1]))
        case "logistic":
            _require_arg_count(expr, args, 2)
            return float(rng.logistic(loc=args[0], scale=args[1]))
        case "uniform":
            _require_arg_count(expr, args, 2)
            return float(rng.uniform(low=args[0], high=args[1]))
        case "truncnorm":
            _require_arg_count(expr, args, 4)
            return float(
                truncnorm.rvs(
                    args[0], args[1], loc=args[2], scale=args[3], random_state=rng
                )
            )
        case "beta":
            _require_arg_count(expr, args, 2)
            return float(rng.beta(args[0], args[1]))
        case "gamma":
            _require_arg_count(expr, args, 2)
            return float(rng.gamma(shape=args[0], scale=args[1]))
        case "exponential":
            _require_arg_count(expr, args, 1)
            return float(rng.exponential(scale=args[0]))
        case "binomial":
            _require_arg_count(expr, args, 2)
            return float(rng.binomial(n=int(args[0]), p=args[1]))
        case _:
            raise ValueError(f"Invalid distribution name '{dist_name}'.")


def _require_arg_count(expr: str, args: list[float], expected: int) -> None:
    """Check the number of parsed distribution arguments."""

    if len(args) != expected:
        raise ValueError(f"{expr!r} requires {expected} numeric arguments.")


def validate_positive_sd_spec(name: str, value) -> None:
    """Validate a raw-space SD specification."""

    if np.isscalar(value) and not isinstance(value, str):
        if float(value) <= 0:
            raise ValueError(f"Parameter '{name}' sd must be positive.")
        return

    if not is_distribution_value(value):
        raise ValueError(
            f"Parameter '{name}' sd must be a positive number or distribution string."
        )

    dist_name, args = parse_distribution_expr(value)
    if dist_name not in POSITIVE_SD_DISTS:
        raise ValueError(
            f"Parameter '{name}' sd distribution must be one of "
            f"{sorted(POSITIVE_SD_DISTS)}."
        )
    if dist_name == "uniform":
        _require_arg_count(value, args, 2)
        if not (0 < args[0] < args[1]):
            raise ValueError(
                f"Parameter '{name}' uniform sd must satisfy 0 < low < high."
            )
    if dist_name == "exponential":
        _require_arg_count(value, args, 1)
        if args[0] <= 0:
            raise ValueError(
                f"Parameter '{name}' exponential sd scale must be positive."
            )
    if dist_name in {"gamma", "beta"}:
        _require_arg_count(value, args, 2)
        if args[0] <= 0 or args[1] <= 0:
            raise ValueError(
                f"Parameter '{name}' {dist_name} sd arguments must be positive."
            )


def draw_group_mean_raw(name: str, spec: dict, rng=None) -> float:
    """Draw or return the raw-space group mean for a parameter."""

    if rng is None:
        rng = np.random

    _check_group_prior_spec(name, spec)
    mean_spec = spec["mean"]
    if is_distribution_value(mean_spec):
        return draw_distribution_expr(mean_spec, rng)
    if np.isscalar(mean_spec):
        return float(mean_spec)
    raise ValueError(
        f"Parameter '{name}' mean must be numeric or a distribution string."
    )


def draw_group_sd_raw(name: str, spec: dict, rng=None) -> float:
    """Draw or return the positive raw-space SD for a parameter."""

    if rng is None:
        rng = np.random

    _check_group_prior_spec(name, spec)
    sd_spec = spec["sd"]
    validate_positive_sd_spec(name, sd_spec)
    if is_distribution_value(sd_spec):
        return max(draw_distribution_expr(sd_spec, rng), MIN_POSITIVE_SCALE)
    return float(sd_spec)


def _check_group_prior_spec(name: str, spec: dict) -> None:
    """Check that a parameter uses the current group prior format."""

    if not is_group_prior_spec(spec):
        prefix = f"Parameter '{name}'" if name else "Parameter"
        raise ValueError(f"{prefix} must define both 'mean' and 'sd'.")


def transform_simple_samples(samples: dict, priors: dict) -> dict:
    """Add public-scale simple parameter samples from raw posterior samples.

    Parameters
    ----------
    samples
        Posterior sample dictionary from BayesFlow.
    priors
        Prior specification containing link functions.

    Returns
    -------
    dict
        Copy of ``samples`` with public keys such as ``a`` and ``ra`` added.
    """

    out = dict(samples)
    for name, spec in priors.items():
        if not isinstance(spec, dict):
            continue
        key = raw_key(name)
        if key in out:
            out[name] = apply_link(out[key], spec.get("link", "identity"))
    return out


def transform_hierarchical_samples(samples: dict, priors: dict) -> dict:
    """Add public-scale hierarchical samples from raw posterior samples.

    Parameters
    ----------
    samples
        Posterior sample dictionary from BayesFlow.
    priors
        Prior specification containing link functions.

    Returns
    -------
    dict
        Copy of ``samples`` with public keys such as ``a_mu``, ``a_sigma``,
        and ``a_subj_0`` added when raw keys are present.
    """

    out = dict(samples)
    for name, spec in priors.items():
        if not isinstance(spec, dict):
            continue
        link = spec.get("link", "identity")

        mu_key = mu_raw_key(name)
        if mu_key in out:
            out[f"{name}_mu"] = apply_link(out[mu_key], link)

        sigma_key = log_sigma_key(name)
        if sigma_key in out:
            out[f"{name}_sigma"] = np.exp(out[sigma_key])

        prefix = f"{name}_subj_raw_"
        for key, value in list(out.items()):
            if key.startswith(prefix):
                subj = key.removeprefix(prefix)
                out[f"{name}_subj_{subj}"] = apply_link(value, link)
    return out


def apply_link(value, link="identity"):
    """Apply link transformation to raw values.

    Parameters
    ----------
    value : float | np.ndarray
        Raw value.
    link : str, optional
        Link function name.

    Returns
    -------
    float | np.ndarray
        Transformed value.
    """

    match link:
        case "identity":
            pass
        case "log":
            value = np.exp(value)
        case "logit":
            value = 1 / (1 + np.exp(-value))
        case _:
            raise ValueError(f"Invalid link function '{link}'.")
    return value


def invert_link(val, link):
    """Map linked values back to raw parameter space.

    Parameters
    ----------
    val : float | np.ndarray
        Linked-space value.
    link : str | None
        Link function name.

    Returns
    -------
    float | np.ndarray
        Raw-space value.
    """

    if link is None:
        return val
    if link == "log":
        return np.log(np.maximum(val, 1e-8))
    if link == "logit":
        val = np.clip(val, 1e-6, 1 - 1e-6)
        return np.log(val / (1.0 - val))
    return val
