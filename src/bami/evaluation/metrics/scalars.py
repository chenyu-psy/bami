"""Scalar metrics for evaluating recovered parameter values.

Use this module when you need one summary number for agreement between true
parameters and estimated parameters.
"""

import numpy as np


def compute_corr(truth: np.ndarray, estimate: np.ndarray) -> float:
    """Compute Pearson correlation with simple safety checks.

    Args:
        truth: Ground-truth values.
        estimate: Point estimates aligned with ``truth``.

    Returns:
        float: Pearson r, or NaN when the vectors are too short or either
            vector has zero variance.
    """

    x = np.asarray(truth, dtype=float).reshape(-1)
    y = np.asarray(estimate, dtype=float).reshape(-1)

    if x.shape[0] != y.shape[0]:
        raise ValueError("truth and estimate must have the same number of elements.")
    if x.shape[0] < 2:
        return float("nan")
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compute_ccc(truth: np.ndarray, estimate: np.ndarray) -> float:
    """Compute Lin's concordance correlation coefficient.

    CCC is useful for parameter recovery because high Pearson correlation can
    still hide estimates that are biased or compressed toward the mean.

    Args:
        truth: Ground-truth values.
        estimate: Point estimates aligned with ``truth``.

    Returns:
        float: Concordance correlation coefficient, or NaN when it is
            undefined.
    """

    x = np.asarray(truth, dtype=float).reshape(-1)
    y = np.asarray(estimate, dtype=float).reshape(-1)

    if x.shape[0] != y.shape[0]:
        raise ValueError("truth and estimate must have the same number of elements.")
    if x.shape[0] < 2:
        return float("nan")

    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    var_x = float(np.mean((x - mean_x) ** 2))
    var_y = float(np.mean((y - mean_y) ** 2))
    cov_xy = float(np.mean((x - mean_x) * (y - mean_y)))
    denom = var_x + var_y + (mean_x - mean_y) ** 2
    if denom == 0.0:
        return float("nan")
    return float((2.0 * cov_xy) / denom)


def compute_rmse(truth: np.ndarray, estimate: np.ndarray) -> float:
    """Compute root mean squared error between true and estimated values.

    Args:
        truth: Ground-truth values.
        estimate: Point estimates aligned with ``truth``.

    Returns:
        float: Root mean squared error, or NaN when no paired values are
            supplied.
    """

    x = np.asarray(truth, dtype=float).reshape(-1)
    y = np.asarray(estimate, dtype=float).reshape(-1)

    if x.shape[0] != y.shape[0]:
        raise ValueError("truth and estimate must have the same number of elements.")
    if x.shape[0] == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y - x) ** 2)))
