"""Shared circular-error helpers.

These helpers use radians so simulated SDM data and observed SDM data follow
the same convention.
"""

from __future__ import annotations

import numpy as np


def circular_moments_from_errors(errors_rad: np.ndarray) -> np.ndarray:
    """Compute the first two circular moments for radian errors.

    Args:
        errors_rad: Trial-level signed circular errors in radians.

    Returns:
        numpy.ndarray: Float32 vector ``[C1, S1, C2, S2]``.
    """

    errors = _as_error_radians(errors_rad)
    c1 = np.mean(np.cos(errors))
    s1 = np.mean(np.sin(errors))
    c2 = np.mean(np.cos(2.0 * errors))
    s2 = np.mean(np.sin(2.0 * errors))
    return np.array([c1, s1, c2, s2], dtype=np.float32)


def _as_error_radians(errors_rad: np.ndarray) -> np.ndarray:
    """Return finite, non-empty response errors in radians.

    Args:
        errors_rad:
            Trial-level signed circular errors in radians.

    Returns:
        numpy.ndarray: One-dimensional float array in radians.
    """

    errors = np.asarray(errors_rad, dtype=float).reshape(-1)
    if errors.size == 0:
        raise ValueError("errors must contain at least one trial.")
    if np.any(~np.isfinite(errors)):
        raise ValueError("errors must not contain missing or infinite values.")
    return errors


def check_n_trials(n_trials: int) -> int:
    """Validate a positive trial count.

    Args:
        n_trials: Candidate trial count.

    Returns:
        int: Positive integer trial count.
    """

    checked = int(n_trials)
    if checked < 1:
        raise ValueError("n_trials must be at least 1.")
    return checked
