"""Trial-level SDM simulation helpers.

The standard SDM represents observations as trial-level signed circular errors.
"""

from __future__ import annotations

import numpy as np
from scipy.special import i0, logsumexp

from bami.simulators.circular import GRID_SIZE, check_n_trials, indices_to_errors


def von_mises_kernel(kappa: float, grid_size: int = GRID_SIZE) -> np.ndarray:
    """Evaluate the SDM von Mises kernel on the circular grid.

    Parameters
    ----------
    kappa
        Concentration of the von Mises kernel.
    grid_size
        Number of grid bins.

    Returns
    -------
    numpy.ndarray
        Kernel values for each SDM error bin.
    """

    if kappa <= 0:
        raise ValueError("kappa must be positive.")

    theta = np.arange(grid_size) * 2.0 * np.pi / grid_size
    return np.exp(kappa * np.cos(theta)) / (2.0 * np.pi * i0(kappa))


def sdm_log_probs(c: float, kappa: float, grid_size: int = GRID_SIZE) -> np.ndarray:
    """Compute SDM log probabilities for all error bins.

    Parameters
    ----------
    c
        Strength applied to the von Mises activation profile.
    kappa
        Concentration of the von Mises kernel.
    grid_size
        Number of grid bins.

    Returns
    -------
    numpy.ndarray
        Log probabilities with length ``grid_size``.
    """

    if c <= 0:
        raise ValueError("c must be positive.")

    activation = c * von_mises_kernel(kappa, grid_size=grid_size)
    return activation - logsumexp(activation)


def sdm_probs(c: float, kappa: float, grid_size: int = GRID_SIZE) -> np.ndarray:
    """Compute SDM probabilities for all error bins.

    Parameters
    ----------
    c
        Strength applied to the von Mises activation profile.
    kappa
        Concentration of the von Mises kernel.
    grid_size
        Number of grid bins.

    Returns
    -------
    numpy.ndarray
        Probabilities that sum to one.
    """

    return np.exp(sdm_log_probs(c=c, kappa=kappa, grid_size=grid_size))


def simulate_sdm_simple(
    c: float,
    kappa: float,
    n_trials: int = 100,
    grid_size: int = GRID_SIZE,
    error_scale: float | None = None,
    jitter: bool = True,
    rng=None,
) -> np.ndarray:
    """Simulate trial-level SDM circular errors.

    Parameters
    ----------
    c, kappa
        Public-scale SDM parameters.
    n_trials
        Number of trial-level errors to simulate.
    grid_size
        Number of circular-error bins used by the SDM probability grid.
    error_scale
        Optional divisor applied to signed degree errors. Use ``180`` to pass
        approximately unit-scaled errors to a neural network while preserving
        the circular ordering.
    jitter
        Whether to add uniform within-bin jitter before circular wrapping.
    rng
        Optional NumPy random generator. Defaults to ``np.random`` so existing
        project-level seeding remains effective.

    Returns
    -------
    numpy.ndarray
        Trial-level signed errors with shape ``(n_trials, 1)``.
    """

    checked_trials = check_n_trials(n_trials)
    rng = np.random if rng is None else rng
    probs = sdm_probs(c=c, kappa=kappa, grid_size=grid_size)
    indices = rng.choice(grid_size, size=checked_trials, replace=True, p=probs)
    errors = indices_to_errors(indices, grid_size=grid_size)
    if jitter:
        errors = errors + rng.uniform(-0.5, 0.5, size=checked_trials)
        errors = ((errors + 180.0) % 360.0) - 180.0
    if error_scale is not None:
        checked_scale = float(error_scale)
        if checked_scale <= 0:
            raise ValueError("error_scale must be positive.")
        errors = errors / checked_scale
    return np.asarray(errors, dtype=np.float32).reshape(-1, 1)
