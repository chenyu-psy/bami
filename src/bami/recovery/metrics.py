"""Metric helpers for parameter-recovery analysis."""

from collections.abc import Iterable

import numpy as np
import pandas as pd


def compute_pearson_r(truth: np.ndarray, estimate: np.ndarray) -> float:
    """Compute Pearson correlation with simple safety checks.

    Parameters
    ----------
    truth : np.ndarray
        Ground-truth values.
    estimate : np.ndarray
        Point estimates aligned with truth.

    Returns
    -------
    float
        Pearson r, or NaN when variance is zero/insufficient.
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

    Parameters
    ----------
    truth : np.ndarray
        Ground-truth values.
    estimate : np.ndarray
        Point estimates aligned with truth.

    Returns
    -------
    float
        Concordance correlation coefficient, or NaN when it is undefined.

    Notes
    -----
    CCC penalizes both weak association and poor agreement with the identity
    line. It is useful for population recovery because high Pearson r can still
    hide biased or compressed estimates.
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


def validate_result_schema(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Validate expected recovery result columns.

    Parameters
    ----------
    df : pd.DataFrame
        Result table.
    required : Iterable[str]
        Required column names.

    Returns
    -------
    None
        Raises ValueError if schema is incomplete.
    """

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required result columns: {missing}")
