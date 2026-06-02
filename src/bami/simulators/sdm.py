"""Trial-level SDM simulation helpers.

Use this module when an SDM workflow needs simulated signed circular errors in
radians. The public simulator keeps the researcher-facing model simple: pass
``c`` and ``kappa``, receive one continuous radian error per trial.
"""

from __future__ import annotations

import numpy as np
from scipy.special import i0

from bami.simulators.circular import check_n_trials

_SUPPORT_SIZE = 4096


def simulate_sdm_simple(
    c: float,
    kappa: float,
    n_trials: int = 100,
    rng=None,
) -> np.ndarray:
    """Simulate continuous trial-level SDM errors in radians.

    Args:
        c: Public-scale SDM activation strength.
        kappa: Public-scale concentration of the circular similarity kernel.
        n_trials: Number of trial-level errors to simulate.
        rng (numpy.random.Generator | None): Optional NumPy random generator. Defaults to ``np.random`` so
            existing project-level seeding remains effective.

    Returns:
        numpy.ndarray: Trial-level signed circular errors in radians with
            shape ``(n_trials, 1)``. Values lie in ``[-pi, pi]``.
    """

    checked_c = _check_positive_float(c, "c")
    checked_kappa = _check_positive_float(kappa, "kappa")
    checked_trials = check_n_trials(n_trials)
    rng = np.random if rng is None else rng

    errors = _sample_sdm_errors(
        c=checked_c,
        kappa=checked_kappa,
        n_trials=checked_trials,
        rng=rng,
    )
    return np.asarray(errors, dtype=np.float32).reshape(-1, 1)


def _sample_sdm_errors(c: float, kappa: float, n_trials: int, rng) -> np.ndarray:
    """Draw signed radian errors from the continuous SDM density.

    Args:
        c: Validated SDM activation strength.
        kappa: Validated SDM concentration parameter.
        n_trials: Number of errors to draw.
        rng: NumPy-compatible random generator.

    Returns:
        numpy.ndarray: One-dimensional array of signed radian errors.
    """

    support = np.linspace(-np.pi, np.pi, _SUPPORT_SIZE + 1)
    density = _sdm_density_unnormalized(support, c=c, kappa=kappa)
    cdf = _trapezoid_cdf(support, density)
    draws = rng.uniform(0.0, cdf[-1], size=n_trials)
    errors = np.interp(draws, cdf, support)
    return _wrap_radians(errors)


def _sdm_density_unnormalized(
    error_rad: np.ndarray,
    c: float,
    kappa: float,
) -> np.ndarray:
    """Evaluate the unnormalized continuous SDM density.

    Args:
        error_rad: Signed circular errors in radians.
        c: Validated SDM activation strength.
        kappa: Validated SDM concentration parameter.

    Returns:
        numpy.ndarray: Positive density values proportional to
            ``exp(c * similarity(error_rad; kappa))``.
    """

    activation = c * _von_mises_similarity(error_rad, kappa=kappa)
    return np.exp(activation - np.max(activation))


def _von_mises_similarity(error_rad: np.ndarray, kappa: float) -> np.ndarray:
    """Return SDM circular similarity for radian errors.

    Args:
        error_rad:
            Signed circular errors in radians.
        kappa:
            Concentration of the circular similarity kernel.

    Returns:
        numpy.ndarray: Similarity values at each error.
    """

    return np.exp(kappa * np.cos(error_rad)) / (2.0 * np.pi * i0(kappa))


def _trapezoid_cdf(support: np.ndarray, density: np.ndarray) -> np.ndarray:
    """Return a cumulative distribution from sampled density values.

    Args:
        support:
            Increasing radian support values.
        density:
            Positive density values evaluated at ``support``.

    Returns:
        numpy.ndarray: Cumulative area values with the same length as ``support``.
    """

    widths = np.diff(support)
    areas = 0.5 * (density[:-1] + density[1:]) * widths
    return np.concatenate(([0.0], np.cumsum(areas)))


def _wrap_radians(errors: np.ndarray) -> np.ndarray:
    """Wrap signed circular errors to ``[-pi, pi]``.

    Args:
        errors:
            Radian error values.

    Returns:
        numpy.ndarray: Wrapped radian errors.
    """

    return ((errors + np.pi) % (2.0 * np.pi)) - np.pi


def _check_positive_float(value: float, name: str) -> float:
    """Return a positive finite floating-point value.

    Args:
        value:
            Candidate numeric value.
        name:
            Parameter name used in error messages.

    Returns:
        float: Positive finite value.
    """

    checked = float(value)
    if not np.isfinite(checked) or checked <= 0:
        raise ValueError(f"{name} must be positive.")
    return checked
