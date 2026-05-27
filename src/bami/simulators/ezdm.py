"""Preset ezDM simulation helpers.

The functions in this module implement the Wagenmakers et al. (2007)
EZ-diffusion relationships between model parameters and observed summaries.
They are independent of legacy model package so they can support the future
model-agnostic workflow package.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


GROUP_PARAM_NAMES = ["mu_v", "sd_v", "mu_a", "sd_a", "mu_t0", "sd_t0"]
SIMPLE_COLUMNS = ["n_trials", "pc", "mrt", "vrt", "v", "a", "t0"]
HIERARCHICAL_COLUMNS = [
    "subj",
    "n_trials",
    "pc",
    "mrt",
    "vrt",
    "v",
    "a",
    "t0",
    "mu_v",
    "sd_v",
    "mu_a",
    "sd_a",
    "mu_t0",
    "sd_t0",
]


def ezdm_moments(v: float, a: float, t0: float, s: float = 0.1) -> dict[str, float]:
    """Return theoretical ezDM summary moments.

    Parameters
    ----------
    v, a, t0
        Public-scale EZ-diffusion parameters.
    s
        Diffusion scaling parameter from Wagenmakers et al. (2007).

    Returns
    -------
    dict[str, float]
        Theoretical ``pc``, ``mrt``, and ``vrt`` values.
    """

    v = _check_finite(v, "v")
    a = _check_positive(a, "a")
    t0 = _check_positive(t0, "t0")
    s = _check_positive(s, "s")

    if v == 0:
        raise ValueError("v must not equal 0 for the EZ moment equations.")

    s2 = s**2
    logit_pc = v * a / s2
    pc = _inv_logit(logit_pc)
    # The tanh form is algebraically equivalent to the published expression
    # but avoids overflow when simulated prior draws imply near-perfect accuracy.
    mdt = (a / (2 * v)) * np.tanh(logit_pc / 2)
    mrt = mdt + t0

    # This is the forward form implied by the appendix inverse equation.
    # Keeping the same algebra makes the code easy to compare with the paper.
    numerator = logit_pc * (logit_pc * pc**2 - logit_pc * pc + pc - 0.5)
    vrt = numerator / ((abs(v) / s) ** 4)

    return {"pc": float(pc), "mrt": float(mrt), "vrt": float(vrt)}


def params_to_moments(
    v: float,
    a: float,
    t0: float,
    s: float = 0.1,
) -> dict[str, float]:
    """Convert EZ parameters to theoretical response summaries.

    Parameters
    ----------
    v
        Drift rate. Positive values imply accuracy above chance.
    a
        Boundary separation. Must be positive.
    t0
        Non-decision time. Must be positive.
    s
        Diffusion scaling parameter from Wagenmakers et al. (2007).

    Returns
    -------
    dict[str, float]
        Theoretical ``pc``, ``mrt``, and ``vrt`` implied by the EZ equations.
    """

    return ezdm_moments(v=v, a=a, t0=t0, s=s)


def moments_to_params(
    pc: float,
    vrt: float,
    mrt: float,
    s: float = 0.1,
) -> dict[str, float]:
    """Convert observed EZ summaries to ``v``, ``a``, and ``t0``.

    Parameters
    ----------
    pc
        Proportion correct. Must be between 0 and 1 and cannot equal 0.5.
    vrt
        Variance of response times.
    mrt
        Mean response time.
    s
        Diffusion scaling parameter from Wagenmakers et al. (2007).

    Returns
    -------
    dict[str, float]
        Estimated ``v``, ``a``, and ``t0``.
    """

    pc = _check_pc(pc)
    vrt = _check_positive(vrt, "vrt")
    mrt = _check_positive(mrt, "mrt")
    s = _check_positive(s, "s")

    s2 = s**2
    logit_pc = _logit(pc)
    x = logit_pc * (logit_pc * pc**2 - logit_pc * pc + pc - 0.5) / vrt
    if x <= 0:
        raise ValueError("pc and vrt imply an invalid drift-rate term.")

    v = np.sign(pc - 0.5) * s * x**0.25
    a = s2 * logit_pc / v
    y = -v * a / s2
    mdt = (a / (2 * v)) * ((1 - np.exp(y)) / (1 + np.exp(y)))
    t0 = mrt - mdt

    return {"v": float(v), "a": float(a), "t0": float(t0)}


def simulate_ezdm_summary(
    v: float,
    a: float,
    t0: float,
    n_trials: int = 100,
    s: float = 0.1,
    rng=None,
) -> np.ndarray:
    """Simulate one ezDM summary row.

    Parameters
    ----------
    v, a, t0
        Public-scale EZ-diffusion parameters.
    n_trials
        Number of trials used to sample observed accuracy.
    s
        Diffusion scaling parameter.
    rng
        Optional NumPy random generator. Defaults to ``np.random`` so existing
        project-level seeding remains effective.

    Returns
    -------
    numpy.ndarray
        Summary vector ``[pc, mrt, vrt]``.
    """

    checked_trials = _check_n_trials(n_trials)
    rng = np.random if rng is None else rng
    moments = ezdm_moments(v=v, a=a, t0=t0, s=s)
    n_correct = rng.binomial(checked_trials, moments["pc"])
    pc = _edge_correct_pc(n_correct, checked_trials)
    return np.array([pc, moments["mrt"], moments["vrt"]], dtype=np.float32)


def simulate_simple(
    v: float,
    a: float,
    t0: float,
    n_trials: int,
    seed: int | None = None,
    s: float = 0.1,
) -> pd.DataFrame:
    """Generate one summary-level EZ dataset from a parameter set.

    Parameters
    ----------
    v, a, t0
        Public-scale EZ-diffusion parameters.
    n_trials
        Number of trials used to draw observed accuracy.
    seed
        Optional random seed for reproducible accuracy sampling.
    s
        Diffusion scaling parameter from Wagenmakers et al. (2007).

    Returns
    -------
    pandas.DataFrame
        One-row summary data with observed summaries and true parameters.
    """

    v = _check_finite(v, "v")
    a = _check_positive(a, "a")
    t0 = _check_positive(t0, "t0")
    n_trials = _check_n_trials(n_trials)
    s = _check_positive(s, "s")

    rng = np.random.default_rng(seed)
    summary = simulate_ezdm_summary(
        v=v,
        a=a,
        t0=t0,
        n_trials=n_trials,
        s=s,
        rng=rng,
    )
    data = {
        "n_trials": n_trials,
        "pc": float(summary[0]),
        "mrt": float(summary[1]),
        "vrt": float(summary[2]),
        "v": v,
        "a": a,
        "t0": t0,
    }
    return pd.DataFrame([data], columns=SIMPLE_COLUMNS)


def simulate_hierarchical(
    group_params: Mapping[str, float],
    n_subjects: int,
    n_trials: int,
    seed: int | None = None,
    s: float = 0.1,
) -> pd.DataFrame:
    """Generate subject-level EZ summaries from group parameters.

    Parameters
    ----------
    group_params
        Mapping with ``mu_v``, ``sd_v``, ``mu_a``, ``sd_a``, ``mu_t0``, and
        ``sd_t0``.
    n_subjects
        Number of subjects to simulate.
    n_trials
        Number of trials per subject.
    seed
        Optional random seed controlling subject parameters and accuracy draws.
    s
        Diffusion scaling parameter from Wagenmakers et al. (2007).

    Returns
    -------
    pandas.DataFrame
        One row per subject with observed summaries, true subject parameters,
        and group-level parameters.
    """

    params = _check_group_params(group_params)
    n_subjects = _check_n_subjects(n_subjects)
    n_trials = _check_n_trials(n_trials)
    s = _check_positive(s, "s")

    rng = np.random.default_rng(seed)
    frames = []

    for subj in range(1, n_subjects + 1):
        subj_v = _draw_nonzero_normal(rng, params["mu_v"], params["sd_v"], "v")
        subj_a = _draw_positive_normal(rng, params["mu_a"], params["sd_a"], "a")
        subj_t0 = _draw_positive_normal(rng, params["mu_t0"], params["sd_t0"], "t0")
        subj_seed = int(rng.integers(0, np.iinfo(np.int32).max))

        subj_df = simulate_simple(
            v=subj_v,
            a=subj_a,
            t0=subj_t0,
            n_trials=n_trials,
            seed=subj_seed,
            s=s,
        )
        subj_df.insert(0, "subj", subj)
        for name in GROUP_PARAM_NAMES:
            subj_df[name] = params[name]
        frames.append(subj_df)

    data = pd.concat(frames, ignore_index=True)
    return data[HIERARCHICAL_COLUMNS]


def _logit(value: float) -> float:
    """Return the logit of a probability.

    Parameters
    ----------
    value
        Probability strictly between 0 and 1.

    Returns
    -------
    float
        Logit-transformed probability.
    """

    return float(np.log(value / (1.0 - value)))


def _inv_logit(value: float) -> float:
    """Return a numerically stable inverse-logit value.

    Parameters
    ----------
    value
        Real-valued logit.

    Returns
    -------
    float
        Probability between 0 and 1, allowing floating-point edge values for
        very large finite logits.
    """

    if value >= 0:
        return float(1 / (1 + np.exp(-value)))
    exp_value = np.exp(value)
    return float(exp_value / (1 + exp_value))


def _edge_correct_pc(n_correct: int, n_trials: int) -> float:
    """Move observed accuracy away from exactly zero or one.

    Parameters
    ----------
    n_correct
        Number of correct responses.
    n_trials
        Number of simulated trials.

    Returns
    -------
    float
        Edge-corrected proportion correct.
    """

    if n_correct == 0:
        return 0.5 / n_trials
    if n_correct == n_trials:
        return 1.0 - 0.5 / n_trials
    return n_correct / n_trials


def _check_pc(pc: float) -> float:
    """Validate a proportion correct value for EZ inversion.

    Parameters
    ----------
    pc
        Candidate proportion correct.

    Returns
    -------
    float
        Validated proportion correct.
    """

    checked = _check_finite(pc, "pc")
    if checked <= 0 or checked >= 1:
        raise ValueError("pc must be strictly between 0 and 1.")
    if checked == 0.5:
        raise ValueError("pc must not equal 0.5 for the EZ equations.")
    return checked


def _check_finite(value: float, name: str) -> float:
    """Validate a finite scalar value.

    Parameters
    ----------
    value
        Candidate scalar value.
    name
        Parameter name used in error messages.

    Returns
    -------
    float
        Finite scalar value.
    """

    checked = float(value)
    if not np.isfinite(checked):
        raise ValueError(f"{name} must be finite.")
    return checked


def _check_positive(value: float, name: str) -> float:
    """Validate a positive scalar value.

    Parameters
    ----------
    value
        Candidate scalar value.
    name
        Parameter name used in error messages.

    Returns
    -------
    float
        Positive scalar value.
    """

    checked = _check_finite(value, name)
    if checked <= 0:
        raise ValueError(f"{name} must be positive.")
    return checked


def _check_nonnegative(value: float, name: str) -> float:
    """Validate a nonnegative scalar value.

    Parameters
    ----------
    value
        Candidate scalar value.
    name
        Parameter name used in error messages.

    Returns
    -------
    float
        Nonnegative scalar value.
    """

    checked = _check_finite(value, name)
    if checked < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return checked


def _check_n_trials(n_trials: int) -> int:
    """Validate the number of simulated trials.

    Parameters
    ----------
    n_trials
        Candidate trial count.

    Returns
    -------
    int
        Positive integer trial count.
    """

    checked = int(n_trials)
    if checked < 1:
        raise ValueError("n_trials must be at least 1.")
    return checked


def _check_n_subjects(n_subjects: int) -> int:
    """Validate the number of subjects for hierarchical simulation.

    Parameters
    ----------
    n_subjects
        Candidate subject count.

    Returns
    -------
    int
        Positive integer subject count.
    """

    checked = int(n_subjects)
    if checked < 1:
        raise ValueError("n_subjects must be at least 1.")
    return checked


def _check_group_params(group_params: Mapping[str, float]) -> dict[str, float]:
    """Validate group-level means and standard deviations.

    Parameters
    ----------
    group_params
        Mapping with all names in ``GROUP_PARAM_NAMES``.

    Returns
    -------
    dict[str, float]
        Clean finite group parameters.
    """

    missing = [name for name in GROUP_PARAM_NAMES if name not in group_params]
    if missing:
        raise ValueError(f"Missing group parameter(s): {missing}")

    params = {
        name: _check_finite(group_params[name], name) for name in GROUP_PARAM_NAMES
    }
    for name in ["sd_v", "sd_a", "sd_t0"]:
        params[name] = _check_nonnegative(params[name], name)
    for name in ["mu_a", "mu_t0"]:
        params[name] = _check_positive(params[name], name)
    return params


def _draw_nonzero_normal(
    rng: np.random.Generator,
    mean: float,
    sd: float,
    name: str,
) -> float:
    """Draw a nonzero subject parameter from a normal distribution.

    Parameters
    ----------
    rng
        NumPy random generator used for reproducible subject parameters.
    mean
        Normal mean.
    sd
        Normal standard deviation.
    name
        Parameter name used in the error message.

    Returns
    -------
    float
        Nonzero draw.
    """

    if sd == 0:
        checked = _check_finite(mean, name)
        if checked == 0:
            raise ValueError(f"{name} must not equal 0.")
        return checked

    for _ in range(1000):
        draw = float(rng.normal(mean, sd))
        if draw != 0:
            return draw

    raise ValueError(
        f"Could not draw a nonzero {name} value after 1000 attempts. "
        "Use a nonzero mean or larger standard deviation."
    )


def _draw_positive_normal(
    rng: np.random.Generator,
    mean: float,
    sd: float,
    name: str,
) -> float:
    """Draw a positive subject parameter from a normal distribution.

    Parameters
    ----------
    rng
        NumPy random generator used for reproducible subject parameters.
    mean
        Normal mean.
    sd
        Normal standard deviation.
    name
        Parameter name used in the error message.

    Returns
    -------
    float
        Positive draw.
    """

    if sd == 0:
        return _check_positive(mean, name)

    for _ in range(1000):
        draw = float(rng.normal(mean, sd))
        if draw > 0:
            return draw

    raise ValueError(
        f"Could not draw a positive {name} value after 1000 attempts. "
        "Use a larger mean or smaller standard deviation."
    )
