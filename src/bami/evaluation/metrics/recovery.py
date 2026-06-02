"""Recovery metric helpers for simulated and estimated values.

This module compares simulated parameter values with estimated parameter values
after both have been arranged as long-format tables.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .scalars import compute_ccc, compute_corr, compute_rmse

VALID_RECOVERY_METRICS: tuple[str, ...] = ("ccc", "corr", "rmse")
SIM_VALUE_COL = "_simulated_value"
EST_VALUE_COL = "_estimated_value"


def _as_name_list(value, name: str) -> list[str]:
    """Normalize one column name or many column names to a list.

    Args:
        value:
            A string column name or a sequence of string column names.
        name:
            Argument name used in error messages.

    Returns:
        list[str]: Non-empty column names in their requested order.
    """

    if isinstance(value, str):
        names = [value]
    else:
        try:
            names = list(value)
        except TypeError as exc:
            raise ValueError(
                f"{name} must be a column name or a list of names."
            ) from exc

    if not names:
        raise ValueError(f"{name} must contain at least one column name.")
    if any(not isinstance(col, str) or not col.strip() for col in names):
        raise ValueError(f"{name} must contain non-empty string column names.")
    return names


def _check_columns(data: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    """Raise a clear error when required columns are missing.

    Args:
        data:
            Input table.
        cols:
            Column names that must be present.
        name:
            Argument name used in error messages.

    Returns:
        None: Raises ``ValueError`` when any column is missing.
    """

    missing = [col for col in cols if col not in data.columns]
    if missing:
        raise ValueError(f"{name} contains columns not found in data: {missing}")


def _check_metrics(metrics: Sequence[str]) -> list[str]:
    """Validate recovery metric names.

    Args:
        metrics:
            Requested metric names.

    Returns:
        list[str]: Metric names in the requested order.
    """

    requested = _as_name_list(metrics, "metrics")
    invalid = sorted(set(requested).difference(VALID_RECOVERY_METRICS))
    if invalid:
        raise ValueError(
            f"metrics contains unsupported values: {invalid}. "
            f"Supported values: {list(VALID_RECOVERY_METRICS)}"
        )
    return requested


def _infer_id_cols(
    simulated_data: pd.DataFrame,
    estimated_data: pd.DataFrame,
    *,
    simulated_col: str,
    estimated_col: str,
) -> list[str]:
    """Infer pairing columns shared by simulated and estimated tables.

    Args:
        simulated_data: Long-format table containing simulated values.
        estimated_data: Long-format table containing estimated values.
        simulated_col: Simulated-value column excluded from automatic pairing.
        estimated_col: Estimated-value column excluded from automatic pairing.

    Returns:
        list[str]: Shared columns used to merge simulated and estimated values.
    """

    excluded = {simulated_col, estimated_col}
    estimated_cols = set(estimated_data.columns)
    id_cols = [col for col in simulated_data.columns if col in estimated_cols]
    id_cols = [col for col in id_cols if col not in excluded]
    if not id_cols:
        raise ValueError(
            "id_cols could not be inferred. Provide id_cols explicitly so "
            "simulated and estimated rows can be paired."
        )
    return id_cols


def _compute_metric(metric: str, simulated: pd.Series, estimated: pd.Series) -> float:
    """Compute one requested recovery metric.

    Args:
        metric:
            Metric name, one of ``ccc``, ``corr``, or ``rmse``.
        simulated:
            Simulated values for one group.
        estimated:
            Estimated values paired with ``simulated``.

    Returns:
        float: Requested recovery metric.
    """

    x = simulated.to_numpy(dtype=float)
    y = estimated.to_numpy(dtype=float)
    if metric == "ccc":
        return compute_ccc(x, y)
    if metric == "corr":
        return compute_corr(x, y)
    return compute_rmse(x, y)


def _select_recovery_cols(
    data: pd.DataFrame,
    *,
    pair_cols: Sequence[str],
    group_cols: Sequence[str],
    value_col: str,
    internal_value_col: str,
) -> pd.DataFrame:
    """Select merge columns and rename the value column for internal use.

    Args:
        data:
            Long-format table used in recovery estimation.
        pair_cols:
            Columns used to pair simulated and estimated rows.
        group_cols:
            Columns requested for grouped metrics. Columns are kept only when they
            exist in this table; the final merged table is checked later.
        value_col:
            User-facing value column name in ``data``.
        internal_value_col:
            Internal column name used after renaming.

    Returns:
        pandas.DataFrame: A copy of the selected columns with a stable internal value name.
    """

    cols = list(pair_cols)
    for col in group_cols:
        if col in data.columns and col not in cols:
            cols.append(col)
    if value_col not in cols:
        cols.append(value_col)

    return data.loc[:, cols].rename(columns={value_col: internal_value_col})


def estimate_recovery(
    simulated_data: pd.DataFrame,
    estimated_data: pd.DataFrame,
    group_by: str | Sequence[str],
    metrics: Sequence[str],
    id_cols: str | Sequence[str] | None = None,
    simulated_col: str = "simulated_value",
    estimated_col: str = "estimated_value",
) -> pd.DataFrame:
    """Estimate recovery metrics from paired simulated and estimated values.

    Use this after simulation truth and model estimates have been converted to
    long tables. Rows are paired by ``id_cols`` before metrics are computed, so
    the output is only meaningful when those columns uniquely identify the same
    simulated item in both tables.

    Args:
        simulated_data: Long-format table containing simulated values.
        estimated_data: Long-format table containing estimated values.
        group_by: Column name or column names defining each metric group.
        metrics: Recovery metrics to compute. Supported values are ``ccc``,
            ``corr``, and ``rmse``.
        id_cols: Column name or column names used to pair simulated and
            estimated rows. When ``None``, shared columns are used after excluding
            value columns.
        simulated_col: Name of the simulated-value column in
            ``simulated_data``.
        estimated_col: Name of the estimated-value column in
            ``estimated_data``.

    Returns:
        pandas.DataFrame: One row per group with requested recovery metrics
            and ``n`` paired rows.
    """

    if not isinstance(simulated_data, pd.DataFrame):
        raise ValueError("simulated_data must be a pandas DataFrame.")
    if not isinstance(estimated_data, pd.DataFrame):
        raise ValueError("estimated_data must be a pandas DataFrame.")

    group_cols = _as_name_list(group_by, "group_by")
    requested_metrics = _check_metrics(metrics)
    pair_cols = (
        _infer_id_cols(
            simulated_data,
            estimated_data,
            simulated_col=simulated_col,
            estimated_col=estimated_col,
        )
        if id_cols is None
        else _as_name_list(id_cols, "id_cols")
    )

    _check_columns(simulated_data, pair_cols + [simulated_col], "simulated_data")
    _check_columns(estimated_data, pair_cols + [estimated_col], "estimated_data")

    sim_values = _select_recovery_cols(
        simulated_data,
        pair_cols=pair_cols,
        group_cols=group_cols,
        value_col=simulated_col,
        internal_value_col=SIM_VALUE_COL,
    )
    est_values = _select_recovery_cols(
        estimated_data,
        pair_cols=pair_cols,
        group_cols=group_cols,
        value_col=estimated_col,
        internal_value_col=EST_VALUE_COL,
    )

    merged = sim_values.merge(
        est_values,
        on=pair_cols,
        how="inner",
        validate="one_to_many",
    )
    if merged.empty:
        raise ValueError("No paired recovery rows were found after merging data.")

    _check_columns(merged, group_cols, "group_by")

    rows = []
    for group_values, group_df in merged.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = dict(zip(group_cols, group_values, strict=True))
        for metric in requested_metrics:
            row[metric] = _compute_metric(
                metric,
                group_df[SIM_VALUE_COL],
                group_df[EST_VALUE_COL],
            )
        row["n"] = int(group_df.shape[0])
        rows.append(row)
    return pd.DataFrame(rows)
