"""General table aggregation helpers for evaluation outputs.

Use this module when a data table needs readable summary statistics by one or
more variables, optionally within grouping columns such as parameter or
condition.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import t

DEFAULT_STATS: tuple[str, ...] = ("mean", "se", "lower_ci", "upper_ci")
VALID_STATS: tuple[str, ...] = (
    "n",
    "mean",
    "median",
    "sd",
    "se",
    "lower_ci",
    "upper_ci",
    "lower_q",
    "upper_q",
)


def _as_name_list(value, name: str) -> list[str]:
    """Normalize one column name or many column names to a list.

    Parameters
    ----------
    value
        A string column name or a sequence of string column names.
    name
        Argument name used in error messages.

    Returns
    -------
    list[str]
        Non-empty column names in their requested order.
    """

    if value is None:
        return []
    if isinstance(value, str):
        names = [value]
    else:
        try:
            names = list(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be a column name or a list of names.") from exc

    if not names:
        raise ValueError(f"{name} must contain at least one column name.")
    if any(not isinstance(col, str) or not col.strip() for col in names):
        raise ValueError(f"{name} must contain non-empty string column names.")
    return names


def _check_columns(data: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    """Raise a clear error when requested columns are absent.

    Parameters
    ----------
    data
        Input table.
    cols
        Column names that must be present.
    name
        Argument name used in error messages.

    Returns
    -------
    None
        Raises ``ValueError`` when any column is missing.
    """

    missing = [col for col in cols if col not in data.columns]
    if missing:
        raise ValueError(f"{name} contains columns not found in data: {missing}")


def _check_stats(stats: Sequence[str] | None) -> list[str]:
    """Validate requested summary statistic names.

    Parameters
    ----------
    stats
        Requested statistic names. ``None`` uses ``DEFAULT_STATS``.

    Returns
    -------
    list[str]
        Statistic names in the requested order.
    """

    requested = list(DEFAULT_STATS if stats is None else stats)
    if not requested:
        raise ValueError("stats must contain at least one statistic name.")
    invalid = sorted(set(requested).difference(VALID_STATS))
    if invalid:
        raise ValueError(
            f"stats contains unsupported values: {invalid}. "
            f"Supported values: {list(VALID_STATS)}"
        )
    return requested


def _check_interval_mass(value: float, name: str) -> float:
    """Validate a central interval mass such as 0.95.

    Parameters
    ----------
    value
        Requested central interval mass.
    name
        Argument name used in error messages.

    Returns
    -------
    float
        Validated interval mass between zero and one.
    """

    checked = float(value)
    if checked <= 0.0 or checked >= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return checked


def _summarize_values(values: pd.Series, stats: Sequence[str], ci: float, q: float) -> dict:
    """Compute requested statistics for one numeric vector.

    Parameters
    ----------
    values
        Numeric values for one variable within one group.
    stats
        Requested statistic names.
    ci
        Central confidence interval mass for the mean.
    q
        Central quantile interval mass for observed values.

    Returns
    -------
    dict
        Requested summary statistics.
    """

    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = int(arr.size)
    mean = float(np.mean(arr)) if n else np.nan
    sd = float(np.std(arr, ddof=1)) if n > 1 else np.nan
    se = float(sd / np.sqrt(n)) if n > 1 else np.nan

    alpha_ci = (1.0 - ci) / 2.0
    ci_width = float(t.ppf(1.0 - alpha_ci, df=n - 1) * se) if n > 1 else np.nan

    alpha_q = (1.0 - q) / 2.0
    out = {}
    for stat in stats:
        if stat == "n":
            out[stat] = n
        elif stat == "mean":
            out[stat] = mean
        elif stat == "median":
            out[stat] = float(np.median(arr)) if n else np.nan
        elif stat == "sd":
            out[stat] = sd
        elif stat == "se":
            out[stat] = se
        elif stat == "lower_ci":
            out[stat] = mean - ci_width if n > 1 else np.nan
        elif stat == "upper_ci":
            out[stat] = mean + ci_width if n > 1 else np.nan
        elif stat == "lower_q":
            out[stat] = float(np.quantile(arr, alpha_q)) if n else np.nan
        elif stat == "upper_q":
            out[stat] = float(np.quantile(arr, 1.0 - alpha_q)) if n else np.nan
    return out


def aggregate_data(
    data: pd.DataFrame,
    variables: str | Sequence[str],
    group_by: str | Sequence[str] | None = None,
    stats: Sequence[str] | None = None,
    ci: float = 0.95,
    q: float = 0.95,
) -> pd.DataFrame:
    """Aggregate numeric columns into a researcher-friendly long table.

    Parameters
    ----------
    data
        Input table containing variables to summarize.
    variables
        One variable column name, or a list of variable column names.
    group_by
        Optional grouping column name or grouping column names.
    stats
        Statistic names to return. By default returns ``mean``, ``se``,
        ``lower_ci``, and ``upper_ci``.
    ci
        Central confidence interval mass for ``lower_ci`` and ``upper_ci``.
        These columns summarize uncertainty around the mean.
    q
        Central quantile interval mass for ``lower_q`` and ``upper_q``. These
        columns summarize the observed data distribution.

    Returns
    -------
    pandas.DataFrame
        Long-format table with grouping columns, ``variable``, and requested
        statistics.
    """

    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame.")

    variable_cols = _as_name_list(variables, "variables")
    group_cols = _as_name_list(group_by, "group_by") if group_by is not None else []
    requested_stats = _check_stats(stats)
    ci = _check_interval_mass(ci, "ci")
    q = _check_interval_mass(q, "q")

    _check_columns(data, variable_cols, "variables")
    _check_columns(data, group_cols, "group_by")

    non_numeric = [
        col for col in variable_cols if not pd.api.types.is_numeric_dtype(data[col])
    ]
    if non_numeric:
        raise ValueError(f"variables must be numeric columns: {non_numeric}")

    rows = []
    if group_cols:
        grouped = data.groupby(group_cols, sort=True, dropna=False)
        for group_values, group_df in grouped:
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            base_row = dict(zip(group_cols, group_values, strict=True))
            for variable in variable_cols:
                row = dict(base_row)
                row["variable"] = variable
                row.update(_summarize_values(group_df[variable], requested_stats, ci, q))
                rows.append(row)
    else:
        for variable in variable_cols:
            row = {"variable": variable}
            row.update(_summarize_values(data[variable], requested_stats, ci, q))
            rows.append(row)

    return pd.DataFrame(rows)
