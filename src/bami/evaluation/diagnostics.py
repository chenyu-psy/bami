"""Model-level diagnostic plots for fitted workflow objects.

These helpers live in evaluation because they compute diagnostic summaries and
figures. Public users should call the shallow workflow methods
``model.plot_parameter_recovery(...)``, ``model.plot_population_recovery(...)``,
or ``model.plot_random_recovery(...)`` on a workflow object.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import logging
import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from bami.evaluation.metrics import compute_ccc, compute_corr, compute_rmse
from bami.inference.runtime import runtime_device

VALID_RECOVERY_METRICS: tuple[str, ...] = ("corr", "ccc", "rmse")
SCATTER_POINT_ALPHA = 0.6
SCATTER_POINT_SIZE = 24.0
RANDOM_POINT_ALPHA = 0.65
RANDOM_POINT_SIZE = 24.0


@contextmanager
def _suppress_bayesflow_output():
    """Temporarily hide BayesFlow sampling progress and info logs.

    Args:
        None

    Returns:
        None: Standard output, standard error, and the BayesFlow logger are
        restored after the wrapped block.
    """

    logger = logging.getLogger("bayesflow")
    old_disabled = logger.disabled
    logger.disabled = True
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            yield
    finally:
        logger.disabled = old_disabled


def _as_param_list(params: str | Sequence[str] | None) -> list[str] | None:
    """Normalize optional parameter names for plotting.

    Args:
        params:
            ``None`` to use all available parameters, one parameter name, or a
            sequence of parameter names.

    Returns:
        list[str] | None: Normalized parameter list, or ``None`` when all available parameters
            should be used.
    """

    if params is None:
        return None
    if isinstance(params, str):
        names = [params]
    else:
        try:
            names = list(params)
        except TypeError as exc:
            raise ValueError(
                "params must be None, a parameter name, or a list."
            ) from exc

    if not names:
        raise ValueError("params must contain at least one parameter name.")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("params must contain non-empty string names.")
    return names


def _as_metric_list(metrics: str | Sequence[str]) -> list[str]:
    """Normalize and validate recovery metric names.

    Args:
        metrics:
            One metric name or a sequence of metric names. Supported metrics are
            ``corr``, ``ccc``, and ``rmse``.

    Returns:
        list[str]: Validated metric names in the requested order.
    """

    if isinstance(metrics, str):
        names = [metrics]
    else:
        try:
            names = list(metrics)
        except TypeError as exc:
            raise ValueError(
                "metrics must be a metric name or a list of names."
            ) from exc

    if not names:
        raise ValueError("metrics must contain at least one metric name.")
    invalid = sorted(set(names).difference(VALID_RECOVERY_METRICS))
    if invalid:
        raise ValueError(
            f"metrics contains unsupported values: {invalid}. "
            f"Supported values: {list(VALID_RECOVERY_METRICS)}"
        )
    return names


def _posterior_mean(samples: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    """Return posterior means using axis 1 as the posterior-sample axis.

    Args:
        samples:
            Posterior sample dictionary returned by a workflow sampling method.
        key:
            Parameter key to summarize.

    Returns:
        numpy.ndarray: Posterior mean array with the sample axis removed.
    """

    values = np.asarray(samples[key], dtype=float)
    if values.ndim < 2:
        raise ValueError(f"Posterior samples for '{key}' must include a sample axis.")
    with warnings.catch_warnings():
        # Padded subjects can be all-NaN across posterior samples. They are
        # removed later when finite truth/estimate pairs are selected.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=1)


def _finite_pairs(truth, estimate) -> tuple[np.ndarray, np.ndarray]:
    """Flatten paired truth and estimate arrays, dropping non-finite rows.

    Args:
        truth: Simulated truth values.
        estimate: Posterior point estimates aligned with ``truth``.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: One-dimensional finite truth and estimate vectors.
    """

    truth_arr = np.asarray(truth, dtype=float)
    estimate_arr = np.asarray(estimate, dtype=float)
    if truth_arr.shape != estimate_arr.shape:
        raise ValueError(
            "Simulated truth and posterior estimates must have matching shapes."
        )
    x = truth_arr.reshape(-1)
    y = estimate_arr.reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def _combine_pair_parts(
    pair_parts: Mapping[str, tuple[list[np.ndarray], list[np.ndarray]]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Combine chunked recovery pairs into one plot-ready mapping.

    Args:
        pair_parts:
            Mapping from parameter name to chunked truth and estimate vectors.

    Returns:
        dict[str, tuple[numpy.ndarray, numpy.ndarray]]: Concatenated recovery
            pairs for plotting.
    """

    pairs = {}
    for name, (truth_parts, estimate_parts) in pair_parts.items():
        pairs[name] = (
            np.concatenate(truth_parts) if truth_parts else np.array([]),
            np.concatenate(estimate_parts) if estimate_parts else np.array([]),
        )
    return pairs


def _compute_metric(metric: str, truth: np.ndarray, estimate: np.ndarray) -> float:
    """Compute one requested recovery metric.

    Args:
        metric:
            Metric name, one of ``corr``, ``ccc``, or ``rmse``.
        truth: Simulated truth values.
        estimate: Posterior point estimates aligned with ``truth``.

    Returns:
        float: Requested metric value.
    """

    if metric == "corr":
        return compute_corr(truth, estimate)
    if metric == "ccc":
        return compute_ccc(truth, estimate)
    return compute_rmse(truth, estimate)


def _compute_random_metric(
    metric: str, truth: np.ndarray, estimate: np.ndarray
) -> float:
    """Compute one random-recovery metric for one simulated dataset.

    Args:
        metric:
            Metric name, one of ``corr``, ``ccc``, or ``rmse``.
        truth: Subject-level simulated truth values for one dataset.
        estimate: Subject-level posterior point estimates for one dataset.

    Returns:
        float: Requested metric value.
    """

    if truth.size < 2:
        raise ValueError(
            "plot_random_recovery requires at least two finite subject pairs "
            "per dataset and parameter."
        )
    return _compute_metric(metric, truth, estimate)


def _format_title(
    param: str,
    truth: np.ndarray,
    estimate: np.ndarray,
    *,
    metrics: Sequence[str],
) -> str:
    """Build a panel title with requested recovery metrics.

    Args:
        param:
            Parameter name shown in the panel title.
        truth: Simulated truth values.
        estimate: Posterior point estimates aligned with ``truth``.
        metrics:
            Recovery metrics appended to the title.

    Returns:
        str: Human-readable panel title.
    """

    labels = {"corr": "corr", "ccc": "CCC", "rmse": "RMSE"}
    parts = [param]
    for metric in metrics:
        parts.append(f"{labels[metric]}={_compute_metric(metric, truth, estimate):.2f}")
    return " | ".join(parts)


def _plot_recovery_grid(
    pairs: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    metrics: Sequence[str],
    n_cols: int,
) -> plt.Figure:
    """Draw one simulated-vs-estimated scatter panel per parameter.

    Args:
        pairs:
            Mapping from parameter name to paired truth and posterior mean values.
        metrics:
            Recovery metrics shown in each panel title.
        n_cols:
            Maximum number of columns in the panel grid.
    Returns:
        matplotlib.figure.Figure: Recovery plot figure.
    """

    if not pairs:
        raise ValueError("No parameters are available for recovery plotting.")
    if n_cols < 1:
        raise ValueError("n_cols must be at least 1.")

    n_panels = len(pairs)
    n_cols = min(int(n_cols), n_panels)
    n_rows = int(math.ceil(n_panels / n_cols))
    figsize = (4.0 * n_cols, 3.6 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    flat_axes = axes.reshape(-1)

    for ax, (param, (truth, estimate)) in zip(flat_axes, pairs.items(), strict=False):
        if truth.size == 0:
            raise ValueError(f"No finite recovery pairs are available for '{param}'.")

        ax.scatter(truth, estimate, alpha=SCATTER_POINT_ALPHA, s=SCATTER_POINT_SIZE)
        low = float(np.nanmin([np.min(truth), np.min(estimate)]))
        high = float(np.nanmax([np.max(truth), np.max(estimate)]))
        if np.isclose(low, high):
            pad = 0.5 if np.isclose(low, 0.0) else abs(low) * 0.05
            low -= pad
            high += pad
        ax.plot([low, high], [low, high], color="black", linewidth=1.0, alpha=0.7)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xlabel("Simulated value")
        ax.set_ylabel("Posterior mean")
        ax.set_title(
            _format_title(
                param,
                truth,
                estimate,
                metrics=metrics,
            )
        )

    for ax in flat_axes[n_panels:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig


def _plot_random_metric_distribution(
    rows: Sequence[dict],
    *,
    params: Sequence[str],
    metrics: Sequence[str],
    n_cols: int,
) -> plt.Figure:
    """Draw dataset-level random-recovery metric distributions.

    Args:
        rows:
            Long rows with ``dataset_id``, ``param``, ``metric``, and ``value``.
        params:
            Parameter names defining x-axis order.
        metrics:
            Metric names defining facet order.
        n_cols:
            Maximum number of columns in the facet grid.
    Returns:
        matplotlib.figure.Figure: Random recovery metric distribution figure.
    """

    if n_cols < 1:
        raise ValueError("n_cols must be at least 1.")

    n_panels = len(metrics)
    n_cols = min(int(n_cols), n_panels)
    n_rows = int(math.ceil(n_panels / n_cols))
    figsize = (4.0 * n_cols, 3.6 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    flat_axes = axes.reshape(-1)
    x_positions = np.arange(1, len(params) + 1, dtype=float)

    for ax, metric in zip(flat_axes, metrics, strict=False):
        grouped_values = []
        grouped_x = []
        for pos, param in zip(x_positions, params, strict=False):
            values = np.asarray(
                [
                    row["value"]
                    for row in rows
                    if row["metric"] == metric and row["param"] == param
                ],
                dtype=float,
            )
            values = values[np.isfinite(values)]
            grouped_values.append(values)
            if values.size:
                offsets = np.linspace(-0.08, 0.08, values.size)
                grouped_x.append(np.full(values.size, pos) + offsets)
            else:
                grouped_x.append(np.array([], dtype=float))

        ax.boxplot(
            grouped_values,
            positions=x_positions,
            widths=0.45,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#D8E2F0", "edgecolor": "#334155"},
            medianprops={"color": "#111827", "linewidth": 1.2},
            whiskerprops={"color": "#334155"},
            capprops={"color": "#334155"},
        )
        for x_values, y_values in zip(grouped_x, grouped_values, strict=False):
            if y_values.size:
                ax.scatter(
                    x_values,
                    y_values,
                    alpha=RANDOM_POINT_ALPHA,
                    s=RANDOM_POINT_SIZE,
                    color="#2563EB",
                    edgecolors="none",
                )
        ax.set_xticks(x_positions, params)
        ax.set_xlabel("Parameter")
        ax.set_ylabel(metric.upper() if metric == "rmse" else metric)
        ax.set_title(metric.upper() if metric == "rmse" else metric)
        ax.grid(axis="y", alpha=0.2)

    for ax in flat_axes[n_panels:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig


def _select_available_params(
    available: Sequence[str],
    requested: str | Sequence[str] | None,
    *,
    label: str,
) -> list[str]:
    """Choose parameters to plot and validate requested names.

    Args:
        available:
            Parameter names that have both simulated truth and posterior estimates.
        requested:
            Optional user-requested parameter names.
        label:
            Label used in error messages.

    Returns:
        list[str]: Parameter names to plot.
    """

    available_names = list(available)
    requested_names = _as_param_list(requested)
    if requested_names is None:
        if not available_names:
            raise ValueError(f"No {label} parameters are available for plotting.")
        return available_names

    missing = [name for name in requested_names if name not in available_names]
    if missing:
        raise ValueError(
            f"params contains unavailable {label} parameters: {missing}. "
            f"Available parameters: {available_names}"
        )
    return requested_names


def _simple_recovery_params(model, simulated_data, samples, params) -> list[str]:
    """Return simple-model parameter names available for recovery plotting.

    Args:
        model:
            ``SimpleWorkflow`` instance.
        simulated_data: Simulated truth dictionary.
        samples: Posterior sample dictionary.
        params:
            Optional requested parameter names.

    Returns:
        list[str]: Public parameter keys to plot.
    """

    available = []
    for name, spec in model.priors.items():
        if isinstance(spec, dict) and name in simulated_data and name in samples:
            available.append(name)
    return _select_available_params(available, params, label="simple")


def _population_recovery_params(model, simulated_data, samples, params) -> list[str]:
    """Return hierarchical population keys available for plotting.

    Args:
        model:
            ``HierarchicalWorkflow`` instance.
        simulated_data: Simulated truth dictionary.
        samples: Group posterior sample dictionary.
        params:
            Optional requested parameter names.

    Returns:
        list[str]: Public group-level parameter keys to plot.
    """

    available = []
    for name, spec in model.priors.items():
        if not isinstance(spec, dict):
            continue
        for key in (f"{name}_mu", f"{name}_sigma"):
            if key in simulated_data and key in samples:
                available.append(key)
    return _select_available_params(available, params, label="population")


def _random_recovery_params(model, simulated_data, estimates, params) -> list[str]:
    """Return subject-level parameter names available for recovery plotting.

    Args:
        model:
            ``HierarchicalWorkflow`` instance.
        simulated_data: Simulated truth dictionary.
        estimates: Random estimator DataFrame.
        params:
            Optional requested parameter names.

    Returns:
        list[str]: Subject-level public parameter names to plot.
    """

    subject_truth_names = [
        key.removesuffix("_subj")
        for key in simulated_data
        if key.endswith("_subj") and key.removesuffix("_subj") in estimates.columns
    ]
    if not subject_truth_names:
        raise ValueError(
            "plot_random_recovery requires simulated subject truth arrays named "
            "<param>_subj and matching estimate columns."
        )

    return _select_available_params(subject_truth_names, params, label="random")


def plot_parameter_recovery(
    model,
    *,
    n_datasets: int,
    num_samples: int,
    params: str | Sequence[str] | None = None,
    metrics: str | Sequence[str] = "corr",
    n_cols: int = 3,
    sample_batch_size: int = 10,
    recovery_batch_size: int = 10,
    show_progress: bool = True,
) -> plt.Figure:
    """Simulate, sample, and plot simple-model parameter recovery.

    This diagnostic draws fresh datasets from a fitted ``SimpleWorkflow``,
    samples the posterior for each dataset, and compares simulated truth with
    posterior means. It is a single-model recovery check, not a model
    comparison routine.

    Args:
        model (SimpleWorkflow): ``SimpleWorkflow`` instance used to simulate datasets and
            sample the fitted posterior.
        n_datasets: Number of simulated datasets used for the diagnostic
            plot.
        num_samples: Number of posterior draws per simulated dataset.
        params: Optional parameter name or names to plot. By default all
            public inferred parameters with both simulated truth and posterior
            samples are shown.
        metrics: Metric name or names shown in each panel title. Supported
            values are ``corr``, ``ccc``, and ``rmse``.
        n_cols: Maximum number of columns in the plot grid.
        sample_batch_size: BayesFlow posterior sampling mini-batch size. Larger
            values usually reduce sampling overhead; lower this value if a
            diagnostic run exceeds available memory.
        recovery_batch_size: Number of recovery datasets simulated and sampled
            in one chunk. Smaller values reduce peak memory use.
        show_progress: Whether to show one BAMI progress bar while scoring
            recovery datasets. BayesFlow's internal sampling output is hidden.

    Returns:
        matplotlib.figure.Figure: Recovery plot figure.
    """

    n_datasets = _check_positive_int(n_datasets, "n_datasets")
    num_samples = _check_positive_int(num_samples, "num_samples")
    sample_batch_size = _check_positive_int(sample_batch_size, "sample_batch_size")
    recovery_batch_size = _check_positive_int(
        recovery_batch_size,
        "recovery_batch_size",
    )
    metric_names = _as_metric_list(metrics)
    pair_parts = {}
    names = None
    progress = tqdm(
        total=n_datasets,
        desc="Parameter recovery",
        unit="dataset",
        disable=not show_progress,
    )
    try:
        dataset_offset = 0
        with runtime_device(getattr(model, "device", "cpu")):
            while dataset_offset < n_datasets:
                batch_n = min(recovery_batch_size, n_datasets - dataset_offset)
                simulated_data = model.simulate(batch_n)
                with _suppress_bayesflow_output():
                    samples = model.sample_posterior(
                        test_data=simulated_data,
                        num_samples=num_samples,
                        sample_batch_size=sample_batch_size,
                    )
                batch_names = _simple_recovery_params(
                    model, simulated_data, samples, params
                )
                if names is None:
                    names = batch_names
                    pair_parts = {name: ([], []) for name in names}
                elif names != batch_names:
                    raise ValueError(
                        "Parameter recovery batches must expose the same parameter names."
                    )
                for name in names:
                    x, y = _finite_pairs(
                        simulated_data[name], _posterior_mean(samples, name)
                    )
                    pair_parts[name][0].append(x)
                    pair_parts[name][1].append(y)
                dataset_offset += batch_n
                progress.update(batch_n)
    finally:
        progress.close()
    pairs = _combine_pair_parts(pair_parts)
    return _plot_recovery_grid(
        pairs,
        metrics=metric_names,
        n_cols=n_cols,
    )


def plot_population_recovery(
    model,
    *,
    n_datasets: int,
    num_samples: int,
    params: str | Sequence[str] | None = None,
    metrics: str | Sequence[str] = "corr",
    n_cols: int = 3,
    sample_batch_size: int = 10,
    recovery_batch_size: int = 10,
    show_progress: bool = True,
) -> plt.Figure:
    """Simulate, sample, and plot hierarchical population recovery.

    This diagnostic simulates group datasets, samples group-level posteriors,
    and compares group truth such as ``c_mu`` or ``c_sigma`` with posterior
    means.

    Args:
        model (HierarchicalWorkflow): ``HierarchicalWorkflow`` instance used to simulate group
            datasets and sample group-level posteriors.
        n_datasets: Number of simulated group datasets used for the
            diagnostic plot.
        num_samples: Number of group posterior draws per simulated dataset.
        params: Optional population parameter key or keys to plot, such as
            ``"c_mu"`` or ``"c_sigma"``. By default all available public group keys
            are shown.
        metrics: Metric name or names shown in each panel title. Supported
            values are ``corr``, ``ccc``, and ``rmse``.
        n_cols: Maximum number of columns in the plot grid.
        sample_batch_size: BayesFlow posterior sampling mini-batch size. Larger
            values usually reduce sampling overhead; lower this value if a
            diagnostic run exceeds available memory.
        recovery_batch_size: Number of recovery datasets simulated and sampled
            in one chunk. Smaller values reduce peak memory use.
        show_progress: Whether to show one BAMI progress bar while scoring
            recovery datasets. BayesFlow's internal sampling output is hidden.

    Returns:
        matplotlib.figure.Figure: Recovery plot figure.
    """

    n_datasets = _check_positive_int(n_datasets, "n_datasets")
    num_samples = _check_positive_int(num_samples, "num_samples")
    sample_batch_size = _check_positive_int(sample_batch_size, "sample_batch_size")
    recovery_batch_size = _check_positive_int(
        recovery_batch_size,
        "recovery_batch_size",
    )
    metric_names = _as_metric_list(metrics)
    pair_parts = {}
    names = None
    progress = tqdm(
        total=n_datasets,
        desc="Population recovery",
        unit="dataset",
        disable=not show_progress,
    )
    try:
        dataset_offset = 0
        with runtime_device(getattr(model, "device", "cpu")):
            while dataset_offset < n_datasets:
                batch_n = min(recovery_batch_size, n_datasets - dataset_offset)
                simulated_data = model.simulate(batch_n)
                with _suppress_bayesflow_output():
                    samples = model.sample_group_posterior(
                        test_data=simulated_data,
                        num_samples=num_samples,
                        sample_batch_size=sample_batch_size,
                    )
                batch_names = _population_recovery_params(
                    model,
                    simulated_data,
                    samples,
                    params,
                )
                if names is None:
                    names = batch_names
                    pair_parts = {name: ([], []) for name in names}
                elif names != batch_names:
                    raise ValueError(
                        "Population recovery batches must expose the same parameter names."
                    )
                for name in names:
                    x, y = _finite_pairs(
                        simulated_data[name], _posterior_mean(samples, name)
                    )
                    pair_parts[name][0].append(x)
                    pair_parts[name][1].append(y)
                dataset_offset += batch_n
                progress.update(batch_n)
    finally:
        progress.close()
    pairs = _combine_pair_parts(pair_parts)
    return _plot_recovery_grid(
        pairs,
        metrics=metric_names,
        n_cols=n_cols,
    )


def plot_random_recovery(
    model,
    *,
    n_datasets: int,
    num_samples: int = 100,
    params: str | Sequence[str] | None = None,
    metrics: str | Sequence[str] = "corr",
    n_cols: int = 3,
    sample_batch_size: int = 10,
    recovery_batch_size: int = 10,
    show_progress: bool = True,
) -> plt.Figure:
    """Plot dataset-level random parameter recovery metrics.

    This diagnostic simulates group datasets, samples group posteriors, estimates
    subject-level random effects, and computes one recovery metric per simulated
    dataset and parameter. The random estimator is deterministic and does not
    return posterior draws. The simulated data must contain subject truth arrays
    named ``<param>_subj`` for scoring.

    Args:
        model (HierarchicalWorkflow): ``HierarchicalWorkflow`` instance used to simulate group
            datasets, sample group posteriors, and estimate random effects.
        n_datasets: Number of simulated group datasets used for the
            diagnostic plot.
        num_samples: Number of group posterior draws per simulated dataset.
            Subject-level random effects are estimated deterministically from the
            group posterior mean.
        params: Optional subject-level parameter name or names to plot, such
            as ``"c"`` or ``"kappa"``. By default all saved subject-truth parameters
            with posterior samples are shown.
        metrics: Metric name or names to plot. Supported values are
            ``corr``, ``ccc``, and ``rmse``. Each metric is shown in a separate
            panel.
        n_cols: Maximum number of columns in the metric panel grid.
        sample_batch_size: BayesFlow group posterior sampling mini-batch size.
        recovery_batch_size: Number of recovery datasets simulated, sampled,
            and estimated in one chunk. Smaller values reduce peak memory use.
        show_progress: Whether to show one BAMI progress bar while scoring
            recovery datasets. BayesFlow's internal sampling output is hidden.

    Returns:
        matplotlib.figure.Figure: Random recovery metric distribution figure.
    """

    n_datasets = _check_positive_int(n_datasets, "n_datasets")
    num_samples = _check_positive_int(num_samples, "num_samples")
    sample_batch_size = _check_positive_int(sample_batch_size, "sample_batch_size")
    recovery_batch_size = _check_positive_int(
        recovery_batch_size,
        "recovery_batch_size",
    )
    metric_names = _as_metric_list(metrics)
    rows = []
    names = None
    progress = tqdm(
        total=n_datasets,
        desc="Random recovery",
        unit="dataset",
        disable=not show_progress,
    )
    try:
        dataset_offset = 0
        with runtime_device(getattr(model, "device", "cpu")):
            while dataset_offset < n_datasets:
                batch_n = min(recovery_batch_size, n_datasets - dataset_offset)
                simulated_data = model.simulate(batch_n)
                with _suppress_bayesflow_output():
                    group_samples = model.sample_group_posterior(
                        test_data=simulated_data,
                        num_samples=num_samples,
                        sample_batch_size=sample_batch_size,
                    )
                estimates = model.estimate_random_parameter(
                    observed_data=simulated_data,
                    group_samples=group_samples,
                )
                batch_names = _random_recovery_params(
                    model,
                    simulated_data,
                    estimates,
                    params,
                )
                if names is None:
                    names = batch_names
                elif names != batch_names:
                    raise ValueError(
                        "Random recovery batches must expose the same parameter names."
                    )
                _add_random_recovery_rows(
                    rows=rows,
                    simulated_data=simulated_data,
                    estimates=estimates,
                    names=names,
                    metrics=metric_names,
                    dataset_offset=dataset_offset,
                )
                dataset_offset += batch_n
                progress.update(batch_n)
    finally:
        progress.close()
    if names is None:
        names = []
    return _plot_random_metric_distribution(
        rows,
        params=names,
        metrics=metric_names,
        n_cols=n_cols,
    )


def _check_positive_int(value: int, name: str) -> int:
    """Return ``value`` as a positive integer.

    Args:
        value:
            User-supplied integer-like value.
        name:
            Argument name used in error messages.

    Returns:
        int: Validated positive integer.
    """

    checked = int(value)
    if checked < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return checked


def _add_random_recovery_rows(
    *,
    rows: list[dict],
    simulated_data: Mapping[str, np.ndarray],
    estimates,
    names: Sequence[str],
    metrics: Sequence[str],
    dataset_offset: int,
) -> None:
    """Append dataset-level random recovery metric rows for one batch.

    Args:
        rows:
            Mutable row list used by the final plotting helper.
        simulated_data:
            Simulated batch containing subject-level truth arrays.
        estimates:
            DataFrame returned by ``estimate_random_parameter`` for the same
            batch.
        names:
            Subject-level parameter names to score.
        metrics:
            Recovery metric names to compute.
        dataset_offset:
            Number of datasets processed before this batch. It keeps dataset IDs
            unique across diagnostic batches.

    Returns:
        None: ``rows`` is updated in place.
    """

    for name in names:
        truth_key = f"{name}_subj"
        truth = np.asarray(simulated_data[truth_key], dtype=float)
        if truth.ndim != 2:
            raise ValueError(
                "plot_random_recovery expects subject truth with shape "
                "(n_datasets, n_subjects)."
            )
        for dataset_id in range(truth.shape[0]):
            dataset_estimates = estimates[estimates["dataset_id"] == dataset_id]
            dataset_estimates = dataset_estimates.sort_values("subject_id")
            estimate_values = dataset_estimates[name].to_numpy(dtype=float)
            truth_estimates = truth[dataset_id, : len(estimate_values)]
            truth_values, estimate_values = _finite_pairs(
                truth_estimates,
                estimate_values,
            )
            for metric in metrics:
                rows.append(
                    {
                        "dataset_id": dataset_offset + dataset_id,
                        "param": name,
                        "metric": metric,
                        "value": _compute_random_metric(
                            metric,
                            truth_values,
                            estimate_values,
                        ),
                    }
                )
