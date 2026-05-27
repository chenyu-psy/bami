"""Shared circular-error helpers for SDM-family simulators.

This module contains model-neutral circular utilities used by SDM and msSDM.
The helpers keep degree conventions in one place so each model module can
focus on its own observation format.
"""

from __future__ import annotations

import numpy as np

GRID_SIZE = 360


def degree_grid(grid_size: int = GRID_SIZE) -> np.ndarray:
    """Return signed degree labels for a circular response-error grid.

    Parameters
    ----------
    grid_size
        Number of equally spaced bins around the response circle.

    Returns
    -------
    numpy.ndarray
        Degree labels in the order ``0, 1, ..., 180, -179, ..., -1``.
    """

    degrees = np.arange(grid_size, dtype=float)
    degrees[degrees > 180.0] -= 360.0
    return degrees


def errors_to_indices(
    errors_deg: np.ndarray,
    grid_size: int = GRID_SIZE,
) -> np.ndarray:
    """Convert trial-level errors in degrees to circular grid indices.

    Parameters
    ----------
    errors_deg
        Error values in degrees, rounded before circular wrapping.
    grid_size
        Number of grid bins.

    Returns
    -------
    numpy.ndarray
        Integer bin indices in ``[0, grid_size)``.
    """

    errors_deg = np.asarray(errors_deg)
    return np.rint(errors_deg).astype(int) % grid_size


def indices_to_errors(
    indices: np.ndarray,
    grid_size: int = GRID_SIZE,
) -> np.ndarray:
    """Convert circular grid indices back to signed degree errors.

    Parameters
    ----------
    indices
        Circular grid indices.
    grid_size
        Number of grid bins.

    Returns
    -------
    numpy.ndarray
        Degree labels using the project convention.
    """

    errors = np.asarray(indices, dtype=float) % grid_size
    errors[errors > 180.0] -= 360.0
    return errors


def circular_moments_from_errors(
    errors_deg: np.ndarray,
    error_scale: float | None = None,
) -> np.ndarray:
    """Compute the first two circular moments for response errors.

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
        Float32 vector ``[C1, S1, C2, S2]``.
    """

    errors = _as_error_degrees(errors_deg, error_scale=error_scale)
    theta = np.deg2rad(errors)
    c1 = np.mean(np.cos(theta))
    s1 = np.mean(np.sin(theta))
    c2 = np.mean(np.cos(2.0 * theta))
    s2 = np.mean(np.sin(2.0 * theta))
    return np.array([c1, s1, c2, s2], dtype=np.float32)


def _as_error_degrees(
    errors_deg: np.ndarray,
    error_scale: float | None = None,
) -> np.ndarray:
    """Return finite, non-empty response errors in degree units.

    Parameters
    ----------
    errors_deg
        Trial-level response errors.
    error_scale
        Optional multiplier used to recover degrees from scaled errors.

    Returns
    -------
    numpy.ndarray
        One-dimensional float array in degree units.
    """

    errors = np.asarray(errors_deg, dtype=float).reshape(-1)
    if errors.size == 0:
        raise ValueError("errors must contain at least one trial.")
    if np.any(~np.isfinite(errors)):
        raise ValueError("errors must not contain missing or infinite values.")
    if error_scale is None:
        return errors

    checked_scale = float(error_scale)
    if checked_scale <= 0:
        raise ValueError("error_scale must be positive.")
    return errors * checked_scale


def check_n_trials(n_trials: int) -> int:
    """Validate a positive trial count.

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
