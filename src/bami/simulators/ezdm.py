"""Built-in ezDM simulator for summary-level bami workflows.

The public entry point is ``simulate_ezdm_simple``. It implements the
Wagenmakers et al. (2007) EZ-diffusion equations needed to turn public-scale
parameters into one aggregate summary row: proportion correct, mean response
time, and response-time variance.
"""

from __future__ import annotations

import numpy as np


def simulate_ezdm_simple(
    v: float,
    a: float,
    t0: float,
    n_trials: int = 100,
    s: float = 0.1,
    rng=None,
) -> np.ndarray:
    """Simulate one ezDM aggregate summary row for bami workflows.

    This simulator uses EZ moment equations for the mean and variance of
    response time. It samples only the observed accuracy from ``n_trials`` so it
    stays a compact summary-level simulator rather than a trial-level
    diffusion-process simulator.

    Args:
        v: Drift rate. Positive values imply accuracy above chance.
        a: Boundary separation. Must be positive.
        t0: Non-decision time. Must be positive.
        n_trials: Number of trials used to sample observed accuracy.
        s: Diffusion scaling parameter from Wagenmakers et al. (2007).
        rng (numpy.random.Generator | None): Optional NumPy random generator. Defaults to ``np.random`` so
            existing project-level seeding remains effective.

    Returns:
        numpy.ndarray: Summary vector ``[pc, mrt, vrt]``.
    """

    v = _check_finite(v, "v")
    a = _check_positive(a, "a")
    t0 = _check_positive(t0, "t0")
    n_trials = _check_n_trials(n_trials)
    s = _check_positive(s, "s")
    rng = np.random if rng is None else rng

    if v == 0:
        raise ValueError("v must not equal 0 for the EZ equations.")

    s2 = s**2
    logit_pc = v * a / s2
    pc_true = _inv_logit(logit_pc)
    # This tanh form is algebraically equivalent to the published expression
    # and avoids overflow for prior draws that imply near-perfect accuracy.
    mdt = (a / (2 * v)) * np.tanh(logit_pc / 2)
    mrt = mdt + t0

    # This forward variance formula matches the appendix inverse equation,
    # making the implementation easy to compare with Wagenmakers et al. (2007).
    numerator = logit_pc * (logit_pc * pc_true**2 - logit_pc * pc_true + pc_true - 0.5)
    vrt = numerator / ((abs(v) / s) ** 4)

    n_correct = rng.binomial(n_trials, pc_true)
    pc = _edge_correct_pc(n_correct, n_trials)
    return np.array([pc, mrt, vrt], dtype=np.float32)


def _inv_logit(value: float) -> float:
    """Return a numerically stable inverse-logit value.

    Args:
        value:
            Real-valued logit.

    Returns:
        float: Probability between 0 and 1, allowing floating-point edge values for
            very large finite logits.
    """

    if value >= 0:
        return float(1 / (1 + np.exp(-value)))
    exp_value = np.exp(value)
    return float(exp_value / (1 + exp_value))


def _edge_correct_pc(n_correct: int, n_trials: int) -> float:
    """Move observed accuracy away from exactly zero or one.

    Args:
        n_correct:
            Number of correct responses.
        n_trials:
            Number of simulated trials.

    Returns:
        float: Edge-corrected proportion correct.
    """

    if n_correct == 0:
        return 0.5 / n_trials
    if n_correct == n_trials:
        return 1.0 - 0.5 / n_trials
    return n_correct / n_trials


def _check_finite(value: float, name: str) -> float:
    """Validate a finite scalar value.

    Args:
        value:
            Candidate scalar value.
        name:
            Parameter name used in error messages.

    Returns:
        float: Finite scalar value.
    """

    checked = float(value)
    if not np.isfinite(checked):
        raise ValueError(f"{name} must be finite.")
    return checked


def _check_positive(value: float, name: str) -> float:
    """Validate a positive scalar value.

    Args:
        value:
            Candidate scalar value.
        name:
            Parameter name used in error messages.

    Returns:
        float: Positive scalar value.
    """

    checked = _check_finite(value, name)
    if checked <= 0:
        raise ValueError(f"{name} must be positive.")
    return checked


def _check_n_trials(n_trials: int) -> int:
    """Validate the number of simulated trials.

    Args:
        n_trials:
            Candidate trial count.

    Returns:
        int: Positive integer trial count.
    """

    checked = int(n_trials)
    if checked < 1:
        raise ValueError("n_trials must be at least 1.")
    return checked
