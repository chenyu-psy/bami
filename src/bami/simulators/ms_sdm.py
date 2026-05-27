"""Moment-summary SDM simulator helpers.

msSDM summarizes trial-level response errors with the first two circular
moments. The default observation row is ``[C1, S1, C2, S2, n_trials]``.
"""

from __future__ import annotations

import numpy as np

from bami.simulators.circular import (
    GRID_SIZE,
    check_n_trials,
    circular_moments_from_errors,
)
from bami.simulators.sdm import simulate_sdm_errors


def errors_to_ms_sdm_summary(
    errors_deg: np.ndarray,
    error_scale: float | None = None,
) -> np.ndarray:
    """Convert trial-level response errors to an msSDM summary row.

    Parameters
    ----------
    errors_deg
        Trial-level response errors. Values are interpreted as degrees unless
        ``error_scale`` is provided.
    error_scale
        Optional multiplier for unit-scaled errors. For example, pass ``180``
        when errors were stored as ``error_deg / 180``.

    Returns
    -------
    numpy.ndarray
        Float32 vector ``[C1, S1, C2, S2, n_trials]``.
    """

    moments = circular_moments_from_errors(errors_deg, error_scale=error_scale)
    n_trials = np.asarray(errors_deg).reshape(-1).size
    return np.concatenate([moments, np.array([n_trials], dtype=np.float32)]).astype(
        np.float32
    )


def simulate_ms_sdm_summary(
    c: float,
    kappa: float,
    n_trials: int = 100,
    grid_size: int = GRID_SIZE,
    error_scale: float | None = None,
    jitter: bool = True,
    rng=None,
) -> np.ndarray:
    """Simulate one msSDM circular-moment summary row.

    Parameters
    ----------
    c, kappa
        Public-scale SDM parameters used to generate response errors.
    n_trials
        Number of trial-level errors to simulate and summarize.
    grid_size
        Number of circular-error bins used by the SDM probability grid.
    error_scale
        Optional divisor applied during trial simulation. If supplied, the same
        value is used to convert simulated errors back to degrees before
        computing circular moments.
    jitter
        Whether trial-level simulation should add uniform within-bin jitter.
    rng
        Optional NumPy random generator. Defaults to ``np.random`` so existing
        project-level seeding remains effective.

    Returns
    -------
    numpy.ndarray
        Float32 vector ``[C1, S1, C2, S2, n_trials]``.
    """

    checked_trials = check_n_trials(n_trials)
    errors = simulate_sdm_errors(
        c=c,
        kappa=kappa,
        n_trials=checked_trials,
        grid_size=grid_size,
        error_scale=error_scale,
        jitter=jitter,
        rng=rng,
    )
    return errors_to_ms_sdm_summary(errors, error_scale=error_scale)
