"""Internal helpers for group-generated recovery analysis.

This module supports the group-generated three-step recovery workflow:
1) generate group-level datasets,
2) fit trained models to generated datasets,
3) examine recovery metrics and plots.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bami.evaluation import (
    compute_ccc,
    compute_corr,
    validate_recovery_contract,
)
from bami.evaluation.metrics.bayesflow import (
    _sample_posterior,
    estimate_fixed_individual_recovery,
    estimate_flex_individual_recovery,
    estimate_population_recovery,
)


def _population_param_key(base_param: str) -> str:
    """Return the displayed population parameter name.

    Parameters
    ----------
    base_param : str
        Base parameter name, such as ``theta``.

    Returns
    -------
    str
        Population-level parameter name used in result tables.
    """

    return f"{base_param}_mu"


def _check_param_list(params: list[str] | tuple[str, ...], name: str) -> list[str]:
    """Validate a user-supplied list of parameter names.

    Parameters
    ----------
    params
        Parameter names supplied by the caller.
    name
        Argument name used in error messages.

    Returns
    -------
    list[str]
        Parameter names as a plain list.
    """

    if params is None:
        raise ValueError(f"{name} must be provided explicitly.")
    checked = list(params)
    if not checked:
        raise ValueError(f"{name} must contain at least one parameter name.")
    if any(not str(param).strip() for param in checked):
        raise ValueError(f"{name} must contain non-empty parameter names.")
    return checked


def _flex_conditions_from_counts(
    flex_model: object,
    count_data: np.ndarray,
    *,
    n_trials: int | None = None,
) -> dict[str, np.ndarray]:
    """Convert batched count data to flex-hierarchy BayesFlow conditions.

    Parameters
    ----------
    flex_model : object
        Fit model that defines the padding width.
    count_data : np.ndarray
        Batch of subject-by-category count arrays with trailing five response
        categories.

    Returns
    -------
    dict[str, np.ndarray]
        Conditions dictionary with flex-hierarchy summary data.
    """

    data = np.asarray(count_data, dtype=np.float32)
    data_width = int(getattr(flex_model, "data_width", 5))
    if data.ndim != 3 or data.shape[-1] != data_width:
        raise ValueError(
            "count_data must have shape " f"(n_datasets, n_subjects, {data_width})."
        )

    flex_batches = []
    for dataset_counts in data:
        input_rows = _append_n_trials_if_needed(flex_model, dataset_counts, n_trials)
        flex_data, _ = flex_model._prepare_observed_counts(input_rows)
        flex_batches.append(flex_data[0])
    return {"data": np.stack(flex_batches, axis=0)}


def _flex_simple_conditions_from_counts(
    flex_model: object,
    count_data: np.ndarray,
    *,
    n_trials: int | None = None,
) -> dict[str, np.ndarray]:
    """Convert count data to flex-simple BayesFlow conditions.

    Parameters
    ----------
    flex_model : object
        Fit model that defines the flex-simple input format.
    count_data : np.ndarray
        Batch of count arrays with trailing five response categories.

    Returns
    -------
    dict[str, np.ndarray]
        Conditions dictionary with count totals added.
    """

    data = np.asarray(count_data, dtype=np.float32)
    data_width = int(getattr(flex_model, "data_width", 5))
    if data.ndim != 3 or data.shape[-1] != data_width:
        raise ValueError(
            f"count_data must have shape (n_datasets, n_rows, {data_width})."
        )

    flex_batches = []
    for dataset_counts in data:
        input_rows = _append_n_trials_if_needed(flex_model, dataset_counts, n_trials)
        flex_data, _ = flex_model._prepare_observed_counts(input_rows)
        flex_batches.append(flex_data[0])
    return {"data": np.stack(flex_batches, axis=0)}


def _append_n_trials_if_needed(
    model: object,
    rows: np.ndarray,
    n_trials: int | None,
) -> np.ndarray:
    """Append explicit trial counts for workflows whose input format encodes n.

    Parameters
    ----------
    model
        Workflow model whose input format determines whether an ``n`` column
        is required.
    rows
        Subject rows before workflow-level formatting.
    n_trials
        Fixed trial count represented by each row.

    Returns
    -------
    numpy.ndarray
        Original rows, or rows with one appended trial-count column.
    """

    input_format = getattr(model, "input_format", None)
    if input_format is None or not getattr(input_format, "add_n", False):
        return rows
    if n_trials is None:
        raise ValueError(
            "n_trials is required when converting generated summary rows for "
            "a flex workflow whose input_format encodes trial count."
        )
    n_col = np.full((*rows.shape[:-1], 1), int(n_trials), dtype=np.float32)
    return np.concatenate([rows, n_col], axis=-1)


def _summarize_recovery_r(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
) -> pd.DataFrame:
    """Summarize recovery rows with Pearson correlations.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format rows with ``true_value`` and ``est_value``.
    group_cols : list[str]
        Columns defining each correlation group.

    Returns
    -------
    pd.DataFrame
        One row per group with Pearson ``r`` and row count ``n``.
    """

    rows = []
    for group_values, group_df in df.groupby(group_cols, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = dict(zip(group_cols, group_values, strict=True))
        truth = group_df["true_value"].to_numpy(dtype=float)
        estimate = group_df["est_value"].to_numpy(dtype=float)
        row["r"] = compute_corr(truth, estimate)
        row["n"] = int(group_df.shape[0])
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_recovery_ccc(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
) -> pd.DataFrame:
    """Summarize recovery rows with Lin's concordance correlation coefficient.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format rows with ``true_value`` and ``est_value``.
    group_cols : list[str]
        Columns defining each population recovery group.

    Returns
    -------
    pd.DataFrame
        One row per group with ``ccc`` and row count ``n``.
    """

    rows = []
    for group_values, group_df in df.groupby(group_cols, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = dict(zip(group_cols, group_values, strict=True))
        row["ccc"] = compute_ccc(
            group_df["true_value"].to_numpy(dtype=float),
            group_df["est_value"].to_numpy(dtype=float),
        )
        row["n"] = int(group_df.shape[0])
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_population_recovery_diagnostics(
    rows: pd.DataFrame,
    *,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize population recovery with fit, bias, and scale diagnostics.

    Parameters
    ----------
    rows : pd.DataFrame
        Long-format population recovery rows with ``true_value`` and
        ``est_value`` columns.
    group_cols : list[str], optional
        Columns defining each diagnostic group. By default diagnostics are
        computed for every ``fit_model`` and ``param`` combination.

    Returns
    -------
    pd.DataFrame
        One row per group with CCC, Pearson ``r``, linear slope/intercept,
        mean bias, MAE, RMSE, and row count. A slope below one together with
        positive low-end bias and negative high-end bias indicates center
        compression.
    """

    if group_cols is None:
        group_cols = ["fit_model", "param"]

    required_cols = set(group_cols + ["true_value", "est_value"])
    missing = sorted(required_cols.difference(rows.columns))
    if missing:
        raise ValueError(f"Missing required recovery columns: {missing}")

    diagnostic_rows = []
    for group_values, group_df in rows.groupby(group_cols, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        truth = group_df["true_value"].to_numpy(dtype=float)
        estimate = group_df["est_value"].to_numpy(dtype=float)
        error = estimate - truth

        slope = np.nan
        intercept = np.nan
        if truth.shape[0] >= 2 and np.nanstd(truth) > 0.0:
            slope, intercept = np.polyfit(truth, estimate, deg=1)

        out_row = dict(zip(group_cols, group_values, strict=True))
        out_row.update(
            {
                "ccc": compute_ccc(truth, estimate),
                "r": compute_corr(truth, estimate),
                "slope": float(slope),
                "intercept": float(intercept),
                "bias": float(np.nanmean(error)),
                "mae": float(np.nanmean(np.abs(error))),
                "rmse": float(np.sqrt(np.nanmean(error**2))),
                "n": int(group_df.shape[0]),
            }
        )
        diagnostic_rows.append(out_row)

    return pd.DataFrame(diagnostic_rows)


def _summarize_param_quantile_bias(
    rows: pd.DataFrame,
    *,
    params: list[str] | tuple[str, ...],
    n_quantiles: int = 4,
) -> pd.DataFrame:
    """Summarize bias across true-value quantiles for selected parameters.

    Parameters
    ----------
    rows : pd.DataFrame
        Long-format population recovery rows.
    params : list[str]
        Population parameters to diagnose.
    n_quantiles : int, optional
        Number of true-value bins within each fit-model and parameter group.

    Returns
    -------
    pd.DataFrame
        Quantile-level bias table. Low true-value bins with positive bias and
        high true-value bins with negative bias indicate shrinkage toward the
        middle of the parameter range.
    """

    selected_params = _check_param_list(params, "params")
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be at least 2.")

    required_cols = {"fit_model", "param", "true_value", "est_value"}
    missing = sorted(required_cols.difference(rows.columns))
    if missing:
        raise ValueError(f"Missing required recovery columns: {missing}")

    diagnostic_rows = []
    selected_rows = rows[rows["param"].isin(selected_params)].copy()
    for (fit_name, param), group_df in selected_rows.groupby(
        ["fit_model", "param"],
        sort=True,
    ):
        group_df = group_df.sort_values("true_value").reset_index(drop=True)
        if group_df.empty:
            continue

        if group_df["true_value"].nunique() < 2:
            bin_ids = pd.Series(np.zeros(group_df.shape[0], dtype=int))
        else:
            bin_ids = pd.qcut(
                group_df["true_value"],
                q=min(n_quantiles, group_df.shape[0]),
                labels=False,
                duplicates="drop",
            )
        group_df = group_df.assign(true_quantile=bin_ids)
        for quantile_id, bin_df in group_df.groupby("true_quantile", sort=True):
            error = bin_df["est_value"].to_numpy(dtype=float) - bin_df[
                "true_value"
            ].to_numpy(dtype=float)
            diagnostic_rows.append(
                {
                    "fit_model": fit_name,
                    "param": param,
                    "true_quantile": int(quantile_id),
                    "true_min": float(bin_df["true_value"].min()),
                    "true_max": float(bin_df["true_value"].max()),
                    "true_mean": float(bin_df["true_value"].mean()),
                    "est_mean": float(bin_df["est_value"].mean()),
                    "bias": float(np.nanmean(error)),
                    "mae": float(np.nanmean(np.abs(error))),
                    "n": int(bin_df.shape[0]),
                }
            )

    return pd.DataFrame(diagnostic_rows)


def _plot_group_population_recovery(
    rows: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    base_params: list[str] | tuple[str, ...],
    fit_models: list[str] | None = None,
) -> plt.Figure:
    """Plot population recovery scatters for all fit models.

    Each panel shows the identity line and a simple linear fit ``y ~ x``. The
    fitted line helps show bias or compression that CCC penalizes.

    Parameters
    ----------
    rows : pd.DataFrame
        Population recovery rows with one row per dataset.
    summary : pd.DataFrame
        CCC values by fit model and parameter.
    Returns
    -------
    matplotlib.figure.Figure
        Population recovery figure.
    """

    point_color = "#8ecae6"
    fit_color = "#1f4e79"

    selected_fit_models = fit_models or sorted(rows["fit_model"].unique())
    shown_fit_models = [
        name for name in selected_fit_models if name in rows["fit_model"].unique()
    ]
    selected_base_params = _check_param_list(base_params, "base_params")
    params = [_population_param_key(param) for param in selected_base_params]
    fig, axes = plt.subplots(
        len(shown_fit_models),
        len(params),
        figsize=(4.0 * len(params), 3.2 * len(shown_fit_models)),
        squeeze=False,
    )

    for row_idx, fit_name in enumerate(shown_fit_models):
        for col_idx, param in enumerate(params):
            ax = axes[row_idx][col_idx]
            sub = rows[(rows["fit_model"] == fit_name) & (rows["param"] == param)]
            ax.scatter(
                sub["true_value"],
                sub["est_value"],
                color=point_color,
                alpha=0.45,
                s=18,
            )
            if not sub.empty:
                lo = min(sub["true_value"].min(), sub["est_value"].min())
                hi = max(sub["true_value"].max(), sub["est_value"].max())
                ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="gray")
                x = sub["true_value"].to_numpy(dtype=float)
                y = sub["est_value"].to_numpy(dtype=float)
                if x.shape[0] >= 2 and np.std(x) > 0.0:
                    slope, intercept = np.polyfit(x, y, deg=1)
                    fit_x = np.linspace(lo, hi, 100, dtype=float)
                    fit_y = intercept + slope * fit_x
                    if x.shape[0] > 2:
                        residual = y - (intercept + slope * x)
                        residual_se = np.sqrt(np.sum(residual**2) / (x.shape[0] - 2))
                        x_mean = float(np.mean(x))
                        x_ss = float(np.sum((x - x_mean) ** 2))
                        if x_ss > 0.0:
                            mean_se = residual_se * np.sqrt(
                                (1.0 / x.shape[0]) + ((fit_x - x_mean) ** 2 / x_ss)
                            )
                            ci = 1.96 * mean_se
                            ax.fill_between(
                                fit_x,
                                fit_y - ci,
                                fit_y + ci,
                                color=fit_color,
                                alpha=0.18,
                                linewidth=0,
                            )
                    ax.plot(fit_x, fit_y, linestyle="-", linewidth=1.4, color=fit_color)

            metric_rows = summary[
                (summary["fit_model"] == fit_name) & (summary["param"] == param)
            ]
            ccc_val = (
                float(metric_rows["ccc"].iloc[0]) if not metric_rows.empty else np.nan
            )
            ax.set_title(f"{fit_name}\n{param} (ccc={ccc_val:.3f})")
            ax.set_xlabel("True value")
            ax.set_ylabel("Estimated value")
            ax.grid(alpha=0.2)

    fig.tight_layout()
    return fig


def _plot_group_individual_recovery(
    correlations: pd.DataFrame,
    *,
    base_params: list[str] | tuple[str, ...],
    fit_models: list[str] | None = None,
) -> plt.Figure:
    """Plot per-dataset individual recovery correlations.

    Parameters
    ----------
    correlations : pd.DataFrame
        One correlation per fit model, dataset, and parameter.
    Returns
    -------
    matplotlib.figure.Figure
        Individual recovery figure.
    """

    selected_fit_models = fit_models or sorted(correlations["fit_model"].unique())
    shown_fit_models = [
        fit_name
        for fit_name in selected_fit_models
        if fit_name in correlations["fit_model"].unique()
    ]
    fig, axes = plt.subplots(
        len(shown_fit_models),
        1,
        figsize=(8.0, 3.2 * len(shown_fit_models)),
        squeeze=False,
    )
    rng = np.random.default_rng(2026)
    params = _check_param_list(base_params, "base_params")
    positions = np.arange(1, len(params) + 1)

    for row_idx, fit_name in enumerate(shown_fit_models):
        ax = axes[row_idx][0]
        sub = correlations[correlations["fit_model"] == fit_name]
        data_by_param = [
            sub.loc[sub["param"] == param, "r"].to_numpy(dtype=float)
            for param in params
        ]
        box = ax.boxplot(
            data_by_param,
            positions=positions,
            widths=0.48,
            patch_artist=True,
            showfliers=False,
        )
        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
        for i, patch in enumerate(box["boxes"]):
            color = colors[i % len(colors)]
            patch.set(facecolor="white", edgecolor=color, linewidth=1.5)
            box["medians"][i].set(color=color, linewidth=1.5)
            box["whiskers"][2 * i].set(color=color, linewidth=1.2)
            box["whiskers"][2 * i + 1].set(color=color, linewidth=1.2)
            box["caps"][2 * i].set(color=color, linewidth=1.2)
            box["caps"][2 * i + 1].set(color=color, linewidth=1.2)

        tick_labels = []
        for i, values in enumerate(data_by_param):
            color = colors[i % len(colors)]
            jitter = rng.uniform(-0.16, 0.16, size=values.shape[0])
            ax.scatter(
                positions[i] + jitter,
                values,
                s=18,
                alpha=0.35,
                color=color,
                edgecolors="none",
            )
            median_r = np.nanmedian(values)
            tick_labels.append(f"{params[i]}\nmedian={median_r:.2f}")

        ax.set_title(f"{fit_name} Individual Recovery by Dataset")
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Parameter")
        ax.set_ylabel("Correlation")
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    return fig


def simulate(
    *,
    generator: object,
    n_reps: int,
) -> dict[str, object]:
    """Generate group-level datasets for recovery checks.

    Parameters
    ----------
    generator : object
        Generator that draws group-level parameters, subject-level parameters,
        and subject count data.
    n_reps : int
        Number of generated datasets. This is the user-facing knob that can be
        increased from 100 to larger comparison runs.

    Returns
    -------
    dict
        Payload with ``n_reps``, ``n_subjects``, and ``sim_data``.
    """

    sim_data = generator.workflow.simulate(n_reps)
    return {
        "n_reps": int(n_reps),
        "n_subjects": int(np.asarray(sim_data["data"]).shape[1]),
        "sim_data": sim_data,
    }


def _model_conditions(
    *,
    fit_name: str,
    fit_model,
    sim_data: dict[str, np.ndarray],
    n_trials: int | None = None,
) -> dict[str, np.ndarray]:
    """Build BayesFlow conditions for one model-comparison fit.

    Parameters
    ----------
    fit_name : str
        Fit model label: ``fixed_simple``, ``flex_simple``,
        ``fixed_hierarchy``, or ``flex_hierarchy``.
    fit_model
        Model object used when flexible data formatting is needed.
    sim_data : dict[str, np.ndarray]
        Group-generated simulation payload.
    n_trials : int, optional
        Fixed trial count to append when a flex fit model uses an input format
        with encoded trial counts.

    Returns
    -------
    dict[str, np.ndarray]
        Conditions dictionary suitable for the internal BayesFlow posterior
        sampling helper.
    """

    if fit_name == "flex_simple":
        return _flex_simple_conditions_from_counts(
            fit_model,
            sim_data["data"],
            n_trials=n_trials,
        )
    if fit_name == "flex_hierarchy":
        return _flex_conditions_from_counts(
            fit_model,
            sim_data["data"],
            n_trials=n_trials,
        )
    return {"data": sim_data["data"]}


def _estimate_population_recovery(
    *,
    fit_name: str,
    sim_data: dict[str, np.ndarray],
    samples: dict[str, np.ndarray],
    base_params: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Build population recovery rows for one model-comparison fit.

    Parameters
    ----------
    fit_name : str
        Fit model label.
    sim_data : dict[str, np.ndarray]
        Generated group-level simulation payload.
    samples : dict[str, np.ndarray]
        Posterior samples for one fitted model.
    base_params : list[str]
        Public subject parameter names to evaluate.

    Returns
    -------
    pd.DataFrame
        Population recovery rows with ``fit_model`` added.
    """

    truth_data = {}
    aligned_samples = {}
    selected_base_params = _check_param_list(base_params, "base_params")
    for base_param in selected_base_params:
        truth_key = _population_param_key(base_param)
        sample_key = (
            base_param if fit_name in {"fixed_simple", "flex_simple"} else truth_key
        )
        if truth_key in sim_data and sample_key in samples:
            truth_data[truth_key] = sim_data[truth_key]
            aligned_samples[truth_key] = samples[sample_key]

    rows = estimate_population_recovery(
        test_data=truth_data,
        samples=aligned_samples,
        variable_keys=[_population_param_key(param) for param in selected_base_params],
        show_progress=False,
    )
    return _label_fit_model(rows, fit_name)


def _estimate_simple_subject_recovery(
    *,
    fit_name: str,
    fit_model: object,
    sim_data: dict[str, np.ndarray],
    posterior_samples: int,
    base_params: list[str] | tuple[str, ...],
    n_trials: int | None = None,
    approximator_kwargs: dict | None = None,
    sample_batch_size: int | None = None,
) -> pd.DataFrame:
    """Estimate subject-level recovery rows with a simple model.

    Each subject's count vector is treated as one simple-model dataset. This is
    specific to group-generated model comparison because the simple models do
    not have subject-level latent variables.

    Parameters
    ----------
    fit_name : str
        Either ``fixed_simple`` or ``flex_simple``.
    fit_model : fixed-simple or flex-simple workflow
        Trained simple model.
    sim_data : dict[str, np.ndarray]
        Generated group-level simulation payload.
    posterior_samples : int
        Number of posterior draws per subject-level dataset.
    base_params : list[str]
        Public subject parameter names to evaluate.
    n_trials : int, optional
        Fixed trial count to append for flex summary workflows.

    Returns
    -------
    pd.DataFrame
        Individual recovery rows with ``fit_model`` added.
    """

    selected_base_params = _check_param_list(base_params, "base_params")
    data = np.asarray(sim_data["data"], dtype=np.float32)
    data_width = int(getattr(fit_model, "data_width", data.shape[-1]))
    if data.ndim != 3 or data.shape[-1] != data_width:
        raise ValueError(
            "sim_data['data'] must have shape "
            f"(n_datasets, n_subjects, {data_width})."
        )

    n_datasets, n_subjects, _ = data.shape
    flat_counts = data.reshape(n_datasets * n_subjects, 1, data.shape[-1])
    if fit_name == "fixed_simple":
        conditions = {"data": flat_counts}
    elif fit_name == "flex_simple":
        conditions = _flex_simple_conditions_from_counts(
            fit_model,
            flat_counts,
            n_trials=n_trials,
        )
    else:
        raise ValueError("fit_name must be 'fixed_simple' or 'flex_simple'.")

    samples = _sample_posterior(
        workflow=fit_model.workflow,
        test_data=conditions,
        num_samples=posterior_samples,
        approximator_kwargs=approximator_kwargs,
        sample_batch_size=sample_batch_size,
    )

    rows = []
    for base_param in selected_base_params:
        if base_param not in samples:
            continue
        est = np.median(np.asarray(samples[base_param], dtype=float), axis=1)
        est = est.reshape(n_datasets, n_subjects)
        for dataset_id in range(n_datasets):
            for subject_id in range(n_subjects):
                array_truth_key = f"{base_param}_subj"
                indexed_truth_key = f"{base_param}_subj_{subject_id}"
                if array_truth_key in sim_data:
                    true_value = np.asarray(sim_data[array_truth_key])[
                        dataset_id, subject_id
                    ]
                elif indexed_truth_key in sim_data:
                    true_value = sim_data[indexed_truth_key][dataset_id]
                else:
                    continue
                rows.append(
                    {
                        "level": "individual",
                        "param": base_param,
                        "true_value": float(true_value),
                        "est_value": float(est[dataset_id, subject_id]),
                        "dataset_id": int(dataset_id),
                        "subject_id": int(subject_id),
                        "model": "bayesflow",
                    }
                )

    out = pd.DataFrame(rows)
    validate_recovery_contract(out)
    return _label_fit_model(out, fit_name)


def _prepare_flex_subject_recovery_data(
    flex_model: object,
    sim_data: dict[str, np.ndarray],
    *,
    base_params: list[str] | tuple[str, ...],
    n_trials: int | None = None,
) -> dict[str, np.ndarray]:
    """Convert fixed group-generated data to flex-hierarchy recovery data.

    Parameters
    ----------
    flex_model : object
        Flex model that defines padded data width.
    sim_data : dict[str, np.ndarray]
        Generated fixed-subject group data.
    base_params : list[str]
        Public subject parameter names to copy into the flex-formatted payload.

    Returns
    -------
    dict[str, np.ndarray]
        Flex-formatted data and subject truth arrays retained for the future
        exchangeable hierarchy recovery table.
    """

    conditions = _flex_conditions_from_counts(
        flex_model,
        sim_data["data"],
        n_trials=n_trials,
    )
    data = conditions["data"]
    n_datasets, max_subjects = data.shape[:2]
    n_subjects = np.asarray(sim_data["data"]).shape[1]
    raw_counts = np.zeros((n_datasets, max_subjects, 5), dtype=np.float32)
    raw_counts[:, :n_subjects, :] = np.asarray(sim_data["data"], dtype=np.float32)
    out = {"data": data, "raw_counts": raw_counts}

    for base_param in _check_param_list(base_params, "base_params"):
        truth = np.full((n_datasets, max_subjects), np.nan, dtype=np.float32)
        array_key = f"{base_param}_subj"
        if array_key in sim_data:
            values = np.asarray(sim_data[array_key], dtype=np.float32)
            if values.shape != (n_datasets, n_subjects):
                raise ValueError(
                    f"Subject truth key '{array_key}' must have shape "
                    f"({n_datasets}, {n_subjects})."
                )
            truth[:, :n_subjects] = values
        else:
            for subject_id in range(n_subjects):
                key = f"{base_param}_subj_{subject_id}"
                if key in sim_data:
                    values = np.asarray(sim_data[key], dtype=np.float32).reshape(-1)
                    if values.shape[0] != n_datasets:
                        raise ValueError(
                            f"Subject truth key '{key}' must have one value per dataset."
                        )
                    truth[:, subject_id] = values
        out[f"{base_param}_subj"] = truth
    return out


def _label_fit_model(df: pd.DataFrame, fit_name: str) -> pd.DataFrame:
    """Add the model-comparison fit label to recovery rows.

    Parameters
    ----------
    df : pd.DataFrame
        Recovery rows.
    fit_name : str
        Fit model label.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with ``fit_model`` added.
    """

    out = df.copy()
    out["fit_model"] = fit_name
    return out


def recover(
    *,
    fit_name: str,
    fit_model: object,
    sim_data: dict[str, np.ndarray],
    base_params: list[str] | tuple[str, ...],
    posterior_samples: int,
    n_trials: int | None = None,
    approximator_kwargs: dict | None = None,
    sample_batch_size: int | None = None,
    n_candidates: int = 4000,
    min_ess: float = 200.0,
    max_candidates: int = 20000,
    batch_candidates: int | None = None,
    adaptive: bool = True,
    show_progress: bool = True,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate population and individual recovery rows for one fitted model.

    Parameters
    ----------
    fit_name : str
        Fit model label: ``fixed_simple``, ``flex_simple``,
        ``fixed_hierarchy``, or ``flex_hierarchy``.
    fit_model : object
        Trained workflow object exposing ``workflow``.
    sim_data : dict[str, np.ndarray]
        Group-generated simulation payload.
    base_params : list[str]
        Public subject parameter names to evaluate.
    posterior_samples : int
        Number of posterior samples to draw.
    n_trials : int, optional
        Fixed trial count to append for flexible summary workflows.
    approximator_kwargs : dict, optional
        Extra keyword arguments forwarded to posterior sampling.
    sample_batch_size : int, optional
        Optional sampling batch size for BayesFlow.
    n_candidates, min_ess, max_candidates, batch_candidates, adaptive
        Retained for the future flexible hierarchical subject recovery route.
    show_progress : bool, optional
        Whether to print progress from slower recovery routines.
    n_jobs : int, optional
        Retained for the future flexible hierarchical subject recovery route.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Population recovery rows and individual recovery rows, both labeled
        with ``fit_model``.
    """

    selected_base_params = _check_param_list(base_params, "base_params")
    conditions = _model_conditions(
        fit_name=fit_name,
        fit_model=fit_model,
        sim_data=sim_data,
        n_trials=n_trials,
    )
    samples = _sample_posterior(
        workflow=fit_model.workflow,
        test_data=conditions,
        num_samples=posterior_samples,
        approximator_kwargs=approximator_kwargs,
        sample_batch_size=sample_batch_size,
    )
    population_rows = _estimate_population_recovery(
        fit_name=fit_name,
        sim_data=sim_data,
        samples=samples,
        base_params=selected_base_params,
    )

    if fit_name in {"fixed_simple", "flex_simple"}:
        individual_rows = _estimate_simple_subject_recovery(
            fit_name=fit_name,
            fit_model=fit_model,
            sim_data=sim_data,
            posterior_samples=posterior_samples,
            base_params=selected_base_params,
            n_trials=n_trials,
            approximator_kwargs=approximator_kwargs,
            sample_batch_size=sample_batch_size,
        )
    elif fit_name == "fixed_hierarchy":
        individual_rows = estimate_fixed_individual_recovery(
            test_data=sim_data,
            samples=samples,
            base_params=selected_base_params,
        )
        individual_rows = _label_fit_model(individual_rows, fit_name)
    elif fit_name == "flex_hierarchy":
        flex_data = _prepare_flex_subject_recovery_data(
            fit_model,
            sim_data,
            base_params=selected_base_params,
            n_trials=n_trials,
        )
        individual_rows = estimate_flex_individual_recovery(
            model=fit_model,
            test_data=flex_data,
            samples=samples,
            n_candidates=n_candidates,
            min_ess=min_ess,
            max_candidates=max_candidates,
            batch_candidates=batch_candidates,
            adaptive=adaptive,
            base_params=selected_base_params,
            n_trials=n_trials,
            show_progress=show_progress,
            n_jobs=n_jobs,
        )
        individual_rows = _label_fit_model(individual_rows, fit_name)
    else:
        raise ValueError(
            "fit_name must be one of fixed_simple, flex_simple, "
            "fixed_hierarchy, or flex_hierarchy."
        )

    return population_rows, individual_rows


def summarize(
    *,
    population_rows: pd.DataFrame,
    individual_rows: pd.DataFrame,
    base_params: list[str] | tuple[str, ...],
    quantile_bias_params: list[str] | tuple[str, ...] | None = None,
    fit_models: list[str] | None = None,
) -> dict[str, object]:
    """Summarize population and individual recovery rows.

    Parameters
    ----------
    population_rows : pd.DataFrame
        Population recovery rows from one or more fitted models.
    individual_rows : pd.DataFrame
        Individual recovery rows from one or more fitted models.
    base_params : list[str]
        Public subject parameter names shown in plots.
    quantile_bias_params : list[str], optional
        Population parameters used for quantile-bias diagnostics.
    fit_models : list[str], optional
        Display order for fit models in plots.

    Returns
    -------
    dict
        Summary tables and matplotlib figures.
    """

    selected_base_params = _check_param_list(base_params, "base_params")
    validate_recovery_contract(population_rows)
    validate_recovery_contract(individual_rows)

    population_summary = _summarize_recovery_ccc(
        population_rows,
        group_cols=["fit_model", "param"],
    )
    population_diagnostics = _summarize_population_recovery_diagnostics(
        population_rows,
    )
    individual_correlations = _summarize_recovery_r(
        individual_rows,
        group_cols=["fit_model", "dataset_id", "param"],
    )

    output = {
        "population_summary": population_summary,
        "population_diagnostics": population_diagnostics,
        "individual_correlations": individual_correlations,
    }
    if quantile_bias_params is not None:
        output["param_quantile_bias"] = _summarize_param_quantile_bias(
            population_rows,
            params=quantile_bias_params,
        )

    output["population_figure"] = _plot_group_population_recovery(
        population_rows,
        population_summary,
        base_params=selected_base_params,
        fit_models=fit_models,
    )
    output["individual_figure"] = _plot_group_individual_recovery(
        individual_correlations,
        base_params=selected_base_params,
        fit_models=fit_models,
    )

    return output
