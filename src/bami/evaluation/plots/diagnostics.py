"""Diagnostic-metric plotting helpers shared across inference backends.

Functions in this module consume the unified diagnostic metric table and render
single-purpose figures for calibration, coverage, and z-score contraction.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from bami.evaluation.contracts import validate_diagnostic_contract


def _plot_metric_bars(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
) -> plt.Figure:
    """Plot a single diagnostic metric as bars by parameter.

    Parameters
    ----------
    df
        Diagnostic table with columns ``param``, ``metric``, and ``value``.
    metric
        Metric label to select.
    title
        Figure title.
    ylabel
        Y-axis label.

    Returns
    -------
    matplotlib.figure.Figure
        Bar plot for the selected metric.
    """

    validate_diagnostic_contract(df)
    sub = df[df["metric"] == metric].copy()
    if sub.empty:
        raise ValueError(f"No rows found for metric '{metric}'.")

    sub = sub.sort_values("param")
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar(sub["param"], sub["value"].astype(float))
    ax.set_title(title)
    ax.set_xlabel("Parameter")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_calibration_ecdf(df: pd.DataFrame) -> plt.Figure:
    """Plot calibration quality (Log Gamma) by parameter.

    Parameters
    ----------
    df
        Diagnostic metric table from compute functions.

    Returns
    -------
    matplotlib.figure.Figure
        Figure for calibration diagnostics.
    """

    return _plot_metric_bars(
        df=df,
        metric="Log Gamma",
        title="Calibration (Log Gamma)",
        ylabel="Log Gamma",
    )


def plot_coverage(df: pd.DataFrame) -> plt.Figure:
    """Plot coverage error by parameter.

    Parameters
    ----------
    df
        Diagnostic metric table from compute functions.

    Returns
    -------
    matplotlib.figure.Figure
        Figure for coverage diagnostics.
    """

    return _plot_metric_bars(
        df=df,
        metric="Calibration Error",
        title="Coverage (Calibration Error)",
        ylabel="Calibration Error",
    )


def plot_zscore_contraction(df: pd.DataFrame) -> plt.Figure:
    """Plot posterior contraction by parameter.

    Parameters
    ----------
    df
        Diagnostic metric table from compute functions.

    Returns
    -------
    matplotlib.figure.Figure
        Figure for z-score contraction diagnostics.
    """

    return _plot_metric_bars(
        df=df,
        metric="Posterior Contraction",
        title="Z-score Contraction (Posterior Contraction)",
        ylabel="Posterior Contraction",
    )
