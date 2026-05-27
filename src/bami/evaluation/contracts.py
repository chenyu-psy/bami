"""Shared contracts for evaluation tables.

This module defines the tabular schema used by plotting functions so that
BayesFlow and brms outputs can be visualized with the same plotting API.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

REQUIRED_RECOVERY_COLUMNS: tuple[str, ...] = (
    "level",
    "param",
    "true_value",
    "est_value",
)

OPTIONAL_RECOVERY_COLUMNS: tuple[str, ...] = (
    "dataset_id",
    "subject_id",
    "model",
)

VALID_LEVELS: tuple[str, ...] = ("population", "individual")

REQUIRED_DIAGNOSTIC_COLUMNS: tuple[str, ...] = ("param", "metric", "value")


def _missing_columns(df: pd.DataFrame, required: Iterable[str]) -> list[str]:
    """Return missing columns from a DataFrame.

    Parameters
    ----------
    df
        Input table to validate.
    required
        Required column names.

    Returns
    -------
    list[str]
        Column names that are missing from ``df``.
    """

    return [col for col in required if col not in df.columns]


def validate_recovery_contract(df: pd.DataFrame) -> None:
    """Validate the recovery-plot contract.

    Parameters
    ----------
    df
        Recovery table in long format.

    Returns
    -------
    None
        Raises ``ValueError`` when required columns are missing or
        ``level`` contains unsupported labels.
    """

    missing = _missing_columns(df, REQUIRED_RECOVERY_COLUMNS)
    if missing:
        raise ValueError(f"Recovery table missing required columns: {missing}")

    levels = set(df["level"].dropna().astype(str).unique().tolist())
    invalid_levels = sorted(levels - set(VALID_LEVELS))
    if invalid_levels:
        raise ValueError(
            "Recovery table has unsupported level values: "
            f"{invalid_levels}. Allowed: {list(VALID_LEVELS)}"
        )


def validate_diagnostic_contract(df: pd.DataFrame) -> None:
    """Validate the diagnostic-metric plotting contract.

    Parameters
    ----------
    df
        Diagnostic metric table in long format.

    Returns
    -------
    None
        Raises ``ValueError`` when required columns are missing.
    """

    missing = _missing_columns(df, REQUIRED_DIAGNOSTIC_COLUMNS)
    if missing:
        raise ValueError(f"Diagnostic table missing required columns: {missing}")
