"""Recovery plotting helpers shared across inference backends.

Functions in this module consume the unified recovery contract table and return
matplotlib figures for population-level and individual-level recovery checks.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bami.evaluation.contracts import validate_recovery_contract
from bami.evaluation.metrics import compute_corr


def _select_plot_pars(df: pd.DataFrame, pars: list[str] | None) -> list[str]:
    """Select parameter names to include in a recovery figure.

    Parameters
    ----------
    df
        Recovery rows for one level after filtering by ``level``.
    pars
        Optional parameter names to plot. When ``None``, all available
        parameters are plotted.

    Returns
    -------
    list[str]
        Parameter names to plot, in plotting order.

    Raises
    ------
    ValueError
        If any requested parameter is not present in ``df``.
    """

    available_pars = df["param"].astype(str).unique().tolist()
    if pars is None:
        return sorted(available_pars)

    missing_pars = [par for par in pars if par not in available_pars]
    if missing_pars:
        raise ValueError(
            f"Requested parameters not found in recovery table: {missing_pars}"
        )

    return pars


def _plot_recovery_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    point_alpha: float,
    point_size: float,
) -> None:
    """Draw one true-vs-estimated recovery panel on an existing axis.

    Parameters
    ----------
    ax
        Axis to draw on.
    df
        Table containing ``true_value`` and ``est_value`` for one parameter.
    title
        Panel title.
    xlabel
        X-axis label.
    ylabel
        Y-axis label.
    point_alpha
        Marker alpha.
    point_size
        Marker size.

    Returns
    -------
    None
        The axis is updated in place.
    """

    x = df["true_value"].to_numpy(dtype=float)
    y = df["est_value"].to_numpy(dtype=float)

    ax.scatter(x, y, alpha=point_alpha, s=point_size)
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)

    corr = compute_corr(x, y)
    ax.set_title(f"{title} (r={corr:.3f})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)


def _plot_recovery_grid(
    df: pd.DataFrame,
    level: str,
    figure_title: str,
    xlabel: str,
    ylabel: str,
    pars: list[str] | None,
    point_alpha: float,
    point_size: float,
) -> plt.Figure:
    """Plot one multi-panel recovery figure for a single recovery level.

    Parameters
    ----------
    df
        Recovery contract table with one or more recovery levels.
    level
        Recovery level to plot. Must match values in the ``level`` column.
    figure_title
        Figure-level title shown above all panels.
    xlabel
        X-axis label used in each panel.
    ylabel
        Y-axis label used in each panel.
    pars
        Optional parameter names to plot. When provided, only these parameters
        are shown and their order is preserved.
    point_alpha
        Marker alpha.
    point_size
        Marker size.

    Returns
    -------
    matplotlib.figure.Figure
        One figure with one panel per plotted parameter.

    Raises
    ------
    ValueError
        If the requested recovery level has no rows or requested parameters are
        missing.
    """

    level_df = df[df["level"] == level].copy()
    if level_df.empty:
        raise ValueError(f"No {level} rows found in recovery table.")

    selected_pars = _select_plot_pars(level_df, pars)
    n_cols = min(3, max(1, len(selected_pars)))
    n_rows = (len(selected_pars) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 4.2 * n_rows),
        squeeze=False,
    )

    for i, par in enumerate(selected_pars):
        ax = axes[i // n_cols][i % n_cols]
        sub = level_df[level_df["param"].astype(str) == par]
        _plot_recovery_panel(
            ax=ax,
            df=sub,
            title=par,
            xlabel=xlabel,
            ylabel=ylabel,
            point_alpha=point_alpha,
            point_size=point_size,
        )

    for j in range(len(selected_pars), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle(figure_title, y=0.995)
    fig.tight_layout()
    return fig


def _dataset_recovery_correlations(
    df: pd.DataFrame, pars: list[str] | None
) -> pd.DataFrame:
    """Compute one subject-level recovery correlation per dataset and parameter.

    Parameters
    ----------
    df
        Recovery contract table with individual rows.
    pars
        Optional parameter order. When ``None``, all parameters are used.

    Returns
    -------
    pd.DataFrame
        Long table with columns ``param``, ``dataset_id``, and ``r``.
    """

    level_df = df[df["level"] == "individual"].copy()
    if level_df.empty:
        raise ValueError("No individual rows found in recovery table.")
    if "dataset_id" not in level_df.columns:
        raise ValueError(
            "plot_individual_recovery requires dataset_id to compute "
            "simulation-level correlations."
        )

    selected_pars = _select_plot_pars(level_df, pars)
    rows = []
    for param in selected_pars:
        param_df = level_df[level_df["param"].astype(str) == param]
        for dataset_id, dataset_df in param_df.groupby("dataset_id", sort=True):
            r_val = compute_corr(
                dataset_df["true_value"].to_numpy(dtype=float),
                dataset_df["est_value"].to_numpy(dtype=float),
            )
            if np.isfinite(r_val):
                rows.append(
                    {
                        "param": param,
                        "dataset_id": dataset_id,
                        "r": float(r_val),
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(
            "No finite simulation-level individual recovery correlations found."
        )
    return out


def _trial_recovery_correlations(
    df: pd.DataFrame, pars: list[str] | None
) -> pd.DataFrame:
    """Compute recovery correlations for each trial count and parameter.

    Parameters
    ----------
    df
        Individual recovery table with ``trial_count`` and ``dataset_id``.
    pars
        Optional parameter order. When ``None``, all parameters are used.

    Returns
    -------
    pd.DataFrame
        Long table with ``trial_count``, ``dataset_id``, ``param``, and ``r``.
    """

    level_df = df[df["level"] == "individual"].copy()
    if level_df.empty:
        raise ValueError("No individual rows found in recovery table.")
    if "dataset_id" not in level_df.columns:
        raise ValueError(
            "plot_trial_sensitivity_recovery requires dataset_id to compute "
            "simulation-level correlations."
        )
    if "trial_count" not in level_df.columns:
        raise ValueError("plot_trial_sensitivity_recovery requires trial_count.")

    selected_pars = _select_plot_pars(level_df, pars)
    rows = []
    for trial_count in sorted(level_df["trial_count"].dropna().unique().tolist()):
        trial_df = level_df[level_df["trial_count"] == trial_count]
        for param in selected_pars:
            param_df = trial_df[trial_df["param"].astype(str) == param]
            for dataset_id, dataset_df in param_df.groupby("dataset_id", sort=True):
                r_val = compute_corr(
                    dataset_df["true_value"].to_numpy(dtype=float),
                    dataset_df["est_value"].to_numpy(dtype=float),
                )
                if np.isfinite(r_val):
                    rows.append(
                        {
                            "trial_count": int(trial_count),
                            "dataset_id": int(dataset_id),
                            "param": param,
                            "r": float(r_val),
                        }
                    )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No finite trial-sensitivity recovery correlations found.")
    return out


def plot_population_recovery(
    df: pd.DataFrame,
    pars: list[str] | None = None,
    point_alpha: float = 0.35,
    point_size: float = 16.0,
) -> plt.Figure:
    """Plot population-level recovery in one multi-panel figure.

    Parameters
    ----------
    df
        Recovery contract table. Only rows with ``level='population'`` are used.
    pars
        Optional parameter names to plot. When provided, only these parameters
        are shown and their order is preserved.
    point_alpha
        Marker alpha.
    point_size
        Marker size.

    Returns
    -------
    matplotlib.figure.Figure
        One figure with one panel per plotted population parameter.

    Raises
    ------
    ValueError
        If the recovery table has no population rows or any requested
        parameters are missing.
    """

    validate_recovery_contract(df)
    return _plot_recovery_grid(
        df=df,
        level="population",
        figure_title="Population Recovery",
        xlabel="True",
        ylabel="Estimated",
        pars=pars,
        point_alpha=point_alpha,
        point_size=point_size,
    )


def plot_individual_recovery(
    df: pd.DataFrame,
    pars: list[str] | None = None,
    point_alpha: float = 0.35,
    point_size: float = 18.0,
) -> plt.Figure:
    """Plot simulation-level individual recovery correlations.

    Parameters
    ----------
    df
        Recovery contract table. Individual rows are summarized into one
        subject-level Pearson correlation per dataset and parameter.
    pars
        Optional parameter names to plot. When provided, only these parameters
        are shown and their order is preserved.
    point_alpha
        Marker alpha.
    point_size
        Marker size.

    Returns
    -------
    matplotlib.figure.Figure
        One figure with one boxplot plus jittered simulation correlations per
        plotted individual parameter.

    Raises
    ------
    ValueError
        If the recovery table has no individual rows, lacks ``dataset_id``, or
        any requested parameters are missing.
    """

    validate_recovery_contract(df)
    corr_df = _dataset_recovery_correlations(df, pars)
    selected_pars = _select_plot_pars(corr_df, pars)

    fig, ax = plt.subplots(figsize=(1.7 * len(selected_pars) + 2.0, 4.6))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    rng = np.random.default_rng(2026)

    positions = np.arange(1, len(selected_pars) + 1)
    data_by_param = [
        corr_df.loc[corr_df["param"] == param, "r"].to_numpy(dtype=float)
        for param in selected_pars
    ]

    box = ax.boxplot(
        data_by_param,
        positions=positions,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
    )
    for i, patch in enumerate(box["boxes"]):
        color = colors[i % len(colors)]
        patch.set(facecolor="white", edgecolor=color, linewidth=1.5)
        box["medians"][i].set(color=color, linewidth=1.5)
        box["whiskers"][2 * i].set(color=color, linewidth=1.2)
        box["whiskers"][2 * i + 1].set(color=color, linewidth=1.2)
        box["caps"][2 * i].set(color=color, linewidth=1.2)
        box["caps"][2 * i + 1].set(color=color, linewidth=1.2)

    for i, values in enumerate(data_by_param):
        color = colors[i % len(colors)]
        jitter = rng.uniform(-0.16, 0.16, size=values.shape[0])
        ax.scatter(
            positions[i] + jitter,
            values,
            s=point_size,
            alpha=point_alpha,
            color=color,
            edgecolors="none",
        )

    tick_labels = []
    for param, values in zip(selected_pars, data_by_param, strict=True):
        median_r = np.nanmedian(values)
        tick_labels.append(f"{param}\nmedian={median_r:.2f}")

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_title("Individual Recovery by Simulation")
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Correlation")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    return fig


def plot_trial_sensitivity_recovery(
    df: pd.DataFrame,
    pars: list[str] | None = None,
    point_alpha: float = 0.35,
    point_size: float = 18.0,
) -> plt.Figure:
    """Plot individual recovery correlations across trial-count conditions.

    Parameters
    ----------
    df
        Individual recovery table with ``trial_count``. Rows are summarized
        into one subject-level Pearson correlation per
        ``trial_count x dataset_id x parameter``.
    pars
        Optional parameter names. When provided, facet order follows this list.
    point_alpha
        Marker alpha for jittered simulation points.
    point_size
        Marker size for jittered simulation points.

    Returns
    -------
    matplotlib.figure.Figure
        Faceted figure with one panel per parameter and trial count on the
        x-axis.
    """

    validate_recovery_contract(df)
    corr_df = _trial_recovery_correlations(df, pars)
    selected_pars = _select_plot_pars(corr_df, pars)
    trial_counts = sorted(corr_df["trial_count"].unique().tolist())

    fig, axes = plt.subplots(
        1,
        len(selected_pars),
        figsize=(4.1 * len(selected_pars), 4.4),
        sharey=True,
        squeeze=False,
    )
    axes_row = axes[0]
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    rng = np.random.default_rng(2026)
    positions = np.arange(1, len(trial_counts) + 1)

    for i, param in enumerate(selected_pars):
        ax = axes_row[i]
        color = colors[i % len(colors)]
        param_df = corr_df[corr_df["param"] == param]
        data_by_trial = [
            param_df.loc[param_df["trial_count"] == trial_count, "r"].to_numpy(
                dtype=float
            )
            for trial_count in trial_counts
        ]

        box = ax.boxplot(
            data_by_trial,
            positions=positions,
            widths=0.48,
            patch_artist=True,
            showfliers=False,
        )
        for patch in box["boxes"]:
            patch.set(facecolor="white", edgecolor=color, linewidth=1.5)
        for line in box["medians"] + box["whiskers"] + box["caps"]:
            line.set(color=color, linewidth=1.2)

        for pos, values in zip(positions, data_by_trial, strict=True):
            jitter = rng.uniform(-0.14, 0.14, size=values.shape[0])
            ax.scatter(
                pos + jitter,
                values,
                s=point_size,
                alpha=point_alpha,
                color=color,
                edgecolors="none",
            )

        ax.set_title(param)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(x) for x in trial_counts])
        ax.set_xlabel("Trials per subject")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
        ax.grid(axis="y", alpha=0.25)
        if i == 0:
            ax.set_ylabel("Correlation")

    fig.suptitle("Individual Recovery by Trial Count", y=1.02)
    fig.tight_layout()
    return fig
