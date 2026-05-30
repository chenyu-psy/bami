"""Tabular data transformation helpers.

These utilities are intended for future real-data integration and keep data
manipulation logic out of entry scripts.
"""

import polars as pl


def drop_non_parameter_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Drop non-parameter columns used in simulated long-format exports.

    Parameters
    ----------
    df : pl.DataFrame
        Input table.

    Returns
    -------
    pl.DataFrame
        Filtered table without `task` and `mean*` columns.
    """

    keep_cols = [
        col for col in df.columns if not col.startswith("mean") and col != "task"
    ]
    return df.select(keep_cols)


def add_participant_rank(df: pl.DataFrame) -> pl.DataFrame:
    """Add participant index by ranked `nPart` within simulation groups.

    Parameters
    ----------
    df : pl.DataFrame
        Long-format table containing `nPart`, `rep`, `nResp`, and `parameter`.

    Returns
    -------
    pl.DataFrame
        Table with a new `participant` column.
    """

    return df.with_columns(
        pl.col("nPart")
        .rank("ordinal")
        .over(["rep", "nResp", "parameter"])
        .alias("participant")
    )


def pivot_parameter_values(df: pl.DataFrame) -> pl.DataFrame:
    """Pivot long parameter values to wide format by parameter name.

    Parameters
    ----------
    df : pl.DataFrame
        Long-format input with `parameter` and `value` columns.

    Returns
    -------
    pl.DataFrame
        Wide-format table.
    """

    return df.pivot(
        index=["rep", "nResp", "participant", "data_type"],
        on="parameter",
        values="value",
        aggregate_function="first",
    )
