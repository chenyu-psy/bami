"""Model-comparison recovery analysis helpers.

This module supports the group-generated three-step recovery workflow:
1) generate model-comparison datasets,
2) fit trained models to generated datasets,
3) examine recovery metrics and plots.
"""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bami.evaluation import (
    estimate_population_recovery,
    sample_posterior,
    validate_recovery_contract,
)
from bami.inference.runtime import configure_torch_device

from .metrics import compute_ccc, compute_pearson_r

BASE_PARAMS = ["a", "c", "ra", "rc"]
GROUP_FIT_MODELS = [
    "fixed_simple",
    "flex_simple",
    "fixed_hierarchy",
    "flex_hierarchy",
]
EDGE_PARAMS = ["ra_mu", "rc_mu"]


def set_global_seed(seed: int) -> None:
    """Set deterministic random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value.

    Returns
    -------
    None
        Updates global random states.
    """

    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def train_model(
    model,
    *,
    max_epochs: int,
    initial_epochs: int,
    n_batch: int,
    batch_size: int,
    validation_data: int,
    patience: int,
    min_delta: float,
    workers: int = 4,
    max_queue_size: int = 16,
    torch_device: str | None = None,
    verbose: int = 1,
    file: str | Path | None = None,
    overwrite: bool = False,
):
    """Train one model using shared training settings.

    Parameters
    ----------
    model
        Model object to train. The object must expose ``dynamic_fit`` and,
        when ``file`` is supplied, ``workflow.approximator``.
    max_epochs, initial_epochs, n_batch, batch_size, validation_data, patience, min_delta :
        Training control values passed to `dynamic_fit`.
    workers : int, optional
        Number of Keras data-loading workers for online simulation batches.
    max_queue_size : int, optional
        Maximum queue length for prefetched simulation batches.
    torch_device : str, optional
        Torch default device to use during training, such as ``"mps"`` or
        ``"cpu"``. Unavailable accelerators fall back to CPU.
    verbose : int, optional
        Training log verbosity level passed to Keras.
    file : str or pathlib.Path, optional
        Checkpoint path. When supplied, existing weights are loaded by default
        and new weights are saved after training.
    overwrite : bool, optional
        Whether to retrain and overwrite ``file`` when the checkpoint already
        exists.

    Returns
    -------
    dict
        Training history object returned by model workflow, or a small
        ``{"loaded": True, "file": path}`` dictionary when an existing
        checkpoint is reused.
    """

    selected_device = configure_torch_device(torch_device)
    if torch_device is not None:
        print(f"Using Torch device for training: {selected_device}", flush=True)

    checkpoint_path = None
    if file is not None:
        checkpoint_path = _check_checkpoint_path(file)
        if checkpoint_path.exists() and not overwrite:
            from bami.inference.checkpoints import load_workflow_weights

            load_workflow_weights(model, checkpoint_path)
            print(f"Loaded workflow weights: {checkpoint_path}", flush=True)
            return {"loaded": True, "file": checkpoint_path}

    history = model.dynamic_fit(
        max_epochs=max_epochs,
        initial_epochs=initial_epochs,
        n_batch=n_batch,
        batch_size=batch_size,
        validation_data=validation_data,
        patience=patience,
        min_delta=min_delta,
        workers=workers,
        use_multiprocessing=False,
        max_queue_size=max_queue_size,
        verbose=verbose,
        keep_optimizer=True,
    )
    if checkpoint_path is not None:
        from bami.inference.checkpoints import save_workflow_weights

        saved_path = save_workflow_weights(model, checkpoint_path)
        print(f"Saved workflow weights: {saved_path}", flush=True)
    return history


def _check_checkpoint_path(file: str | Path) -> Path:
    """Validate a user-supplied checkpoint path.

    Parameters
    ----------
    file
        String or ``Path`` pointing to the desired checkpoint artifact.

    Returns
    -------
    pathlib.Path
        Normalized non-empty checkpoint path.
    """

    checkpoint_path = Path(file)
    if str(checkpoint_path).strip() == "":
        raise ValueError("file must be a non-empty checkpoint path.")
    if checkpoint_path.exists() and checkpoint_path.is_dir():
        raise ValueError("file must point to a checkpoint file, not a directory.")
    if checkpoint_path.suffix != ".keras":
        raise ValueError("file must use the .keras checkpoint extension.")
    return checkpoint_path


def _population_param_key(base_param: str) -> str:
    """Return the displayed population parameter name.

    Parameters
    ----------
    base_param : str
        Base M3 parameter name, such as ``a`` or ``ra``.

    Returns
    -------
    str
        Population-level parameter name used in result tables.
    """

    return f"{base_param}_mu"


def _save_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Save a dataframe to CSV with parent-folder creation.

    Parameters
    ----------
    df : pd.DataFrame
        Table to save.
    path : str | Path
        Output file path.

    Returns
    -------
    Path
        Saved path.
    """

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def _save_pickle(obj, path: str | Path) -> Path:
    """Save a Python object as pickle.

    Parameters
    ----------
    obj : Any
        Object to store.
    path : str | Path
        Destination file path.

    Returns
    -------
    Path
        Saved path.
    """

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(obj, f)
    return out


def _load_pickle(path: str | Path):
    """Load a Python object from pickle.

    Parameters
    ----------
    path : str | Path
        Pickle file path.

    Returns
    -------
    Any
        Loaded object.
    """

    with Path(path).open("rb") as f:
        return pickle.load(f)


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
            "count_data must have shape "
            f"(n_datasets, n_subjects, {data_width})."
        )

    flex_batches = []
    for dataset_counts in data:
        input_rows = _append_n_trials_if_needed(flex_model, dataset_counts, n_trials)
        flex_data, _ = flex_model.counts_to_data(input_rows)
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
        flex_data, _ = flex_model.counts_to_data(input_rows)
        flex_batches.append(flex_data[0])
    return {"data": np.stack(flex_batches, axis=0)}


def _append_n_trials_if_needed(
    model: object,
    rows: np.ndarray,
    n_trials: int | None,
) -> np.ndarray:
    """Append explicit trial counts for workflows whose ``ObsSpec`` encodes n.

    Parameters
    ----------
    model
        Workflow model whose observation spec determines whether an ``n`` column
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

    obs_spec = getattr(model, "obs_spec", None)
    if obs_spec is None or not getattr(obs_spec, "add_n", False):
        return rows
    if n_trials is None:
        raise ValueError(
            "n_trials is required when converting generated summary rows for "
            "a flex workflow whose ObsSpec encodes trial count."
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
        row["r"] = compute_pearson_r(truth, estimate)
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


def summarize_population_recovery_diagnostics(
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
                "r": compute_pearson_r(truth, estimate),
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


def summarize_edge_param_quantile_bias(
    rows: pd.DataFrame,
    *,
    params: list[str] | None = None,
    n_quantiles: int = 4,
) -> pd.DataFrame:
    """Summarize bias across true-value quantiles for boundary parameters.

    Parameters
    ----------
    rows : pd.DataFrame
        Long-format population recovery rows.
    params : list[str], optional
        Population parameters to diagnose. Defaults to ``ra_mu`` and
        ``rc_mu`` because these parameters are bounded and visually showed
        possible center compression.
    n_quantiles : int, optional
        Number of true-value bins within each fit-model and parameter group.

    Returns
    -------
    pd.DataFrame
        Quantile-level bias table. Low true-value bins with positive bias and
        high true-value bins with negative bias indicate shrinkage toward the
        middle of the parameter range.
    """

    if params is None:
        params = EDGE_PARAMS
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be at least 2.")

    required_cols = {"fit_model", "param", "true_value", "est_value"}
    missing = sorted(required_cols.difference(rows.columns))
    if missing:
        raise ValueError(f"Missing required recovery columns: {missing}")

    diagnostic_rows = []
    edge_rows = rows[rows["param"].isin(params)].copy()
    for (fit_name, param), group_df in edge_rows.groupby(
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
    path: str | Path,
    *,
    base_params: list[str] | None = None,
    fit_models: list[str] | None = None,
) -> Path:
    """Plot population recovery scatters for all fit models.

    Each panel shows the identity line and a simple linear fit ``y ~ x``. The
    fitted line helps show bias or compression that CCC penalizes.

    Parameters
    ----------
    rows : pd.DataFrame
        Population recovery rows with one row per dataset.
    summary : pd.DataFrame
        CCC values by fit model and parameter.
    path : str or Path
        Destination image path.

    Returns
    -------
    Path
        Saved plot path.
    """

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    point_color = "#8ecae6"
    fit_color = "#1f4e79"

    selected_fit_models = fit_models or GROUP_FIT_MODELS
    shown_fit_models = [
        name for name in selected_fit_models if name in rows["fit_model"].unique()
    ]
    selected_base_params = base_params or BASE_PARAMS
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
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def _plot_group_individual_recovery(
    correlations: pd.DataFrame,
    path: str | Path,
    *,
    base_params: list[str] | None = None,
    fit_models: list[str] | None = None,
) -> Path:
    """Plot per-dataset individual recovery correlations.

    Parameters
    ----------
    correlations : pd.DataFrame
        One correlation per fit model, dataset, and parameter.
    path : str or Path
        Destination image path.

    Returns
    -------
    Path
        Saved plot path.
    """

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    selected_fit_models = fit_models or GROUP_FIT_MODELS
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
    params = base_params or BASE_PARAMS
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
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def generate_group_datasets(
    *,
    group_generator: object,
    n_reps: int,
    seed: int,
    artifact_dir: str | Path,
    artifact_prefix: str = "11_M3",
) -> Path:
    """Generate group-level datasets for model comparison.

    Parameters
    ----------
    group_generator : object
        Generator that draws group-level parameters, subject-level parameters,
        and subject count data.
    n_reps : int
        Number of generated datasets. This is the user-facing knob that can be
        increased from 100 to larger comparison runs.
    seed : int
        Random seed.
    artifact_dir : str or Path
        Directory to save the generated artifact.
    artifact_prefix : str, optional
        File prefix for the saved artifact.

    Returns
    -------
    Path
        Path to the generated-data artifact file.
    """

    set_global_seed(seed)
    sim_data = group_generator.workflow.simulate(n_reps)
    payload = {
        "seed": seed,
        "n_reps": int(n_reps),
        "n_subjects": int(np.asarray(sim_data["data"]).shape[1]),
        "sim_data": sim_data,
    }
    return _save_pickle(
        payload,
        Path(artifact_dir) / f"{artifact_prefix}_group_generated_data.pkl",
    )


def load_group_generated_data(path: str | Path) -> dict:
    """Load one group-generated data artifact.

    Parameters
    ----------
    path : str or Path
        Path produced by ``generate_group_datasets``.

    Returns
    -------
    dict
        Generated payload with ``sim_data`` and metadata.
    """

    return _load_pickle(path)


def group_model_conditions(
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
        Fixed trial count to append when a flex fit model uses an ``ObsSpec``
        with encoded trial counts.

    Returns
    -------
    dict[str, np.ndarray]
        Conditions dictionary suitable for ``sample_posterior``.
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


def estimate_group_population_recovery(
    *,
    fit_name: str,
    sim_data: dict[str, np.ndarray],
    samples: dict[str, np.ndarray],
    base_params: list[str] | None = None,
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
    base_params : list[str], optional
        Public subject parameter names to evaluate.

    Returns
    -------
    pd.DataFrame
        Population recovery rows with ``fit_model`` added.
    """

    truth_data = {}
    aligned_samples = {}
    selected_base_params = base_params or BASE_PARAMS
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
    return label_fit_model(rows, fit_name)


def estimate_simple_subject_recovery(
    *,
    fit_name: str,
    fit_model: object,
    sim_data: dict[str, np.ndarray],
    posterior_samples: int,
    base_params: list[str] | None = None,
    n_trials: int | None = None,
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
    base_params : list[str], optional
        Public subject parameter names to evaluate.
    n_trials : int, optional
        Fixed trial count to append for flex summary workflows.

    Returns
    -------
    pd.DataFrame
        Individual recovery rows with ``fit_model`` added.
    """

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

    samples = sample_posterior(
        workflow=fit_model.workflow,
        test_data=conditions,
        num_samples=posterior_samples,
    )

    rows = []
    selected_base_params = base_params or BASE_PARAMS
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
    return label_fit_model(out, fit_name)


def prepare_flex_subject_recovery_data(
    flex_model: object,
    sim_data: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Convert fixed group-generated data to flex-hierarchy recovery data.

    Parameters
    ----------
    flex_model : object
        Flex model that defines padded data width.
    sim_data : dict[str, np.ndarray]
        Generated fixed-subject group data.

    Returns
    -------
    dict[str, np.ndarray]
        Flex-formatted data and subject truth arrays used by
        ``estimate_flex_individual_recovery``.
    """

    conditions = _flex_conditions_from_counts(flex_model, sim_data["data"])
    data = conditions["data"]
    n_datasets, max_subjects = data.shape[:2]
    n_subjects = np.asarray(sim_data["data"]).shape[1]
    raw_counts = np.zeros((n_datasets, max_subjects, 5), dtype=np.float32)
    raw_counts[:, :n_subjects, :] = np.asarray(sim_data["data"], dtype=np.float32)
    out = {"data": data, "raw_counts": raw_counts}

    for base_param in BASE_PARAMS:
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


def label_fit_model(df: pd.DataFrame, fit_name: str) -> pd.DataFrame:
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


def save_group_recovery_rows(
    *,
    population_rows: pd.DataFrame,
    individual_rows: pd.DataFrame,
    artifact_dir: str | Path,
    artifact_prefix: str = "11_M3",
) -> dict[str, Path]:
    """Save Step 22 recovery rows as pickle and CSV artifacts.

    Parameters
    ----------
    population_rows : pd.DataFrame
        Population recovery rows for all fit models.
    individual_rows : pd.DataFrame
        Individual recovery rows for all fit models.
    artifact_dir : str or Path
        Artifact directory.
    artifact_prefix : str, optional
        File prefix for saved artifacts.

    Returns
    -------
    dict[str, Path]
        Saved artifact paths.
    """

    out_dir = Path(artifact_dir)
    return {
        "population_rows_pkl": _save_pickle(
            population_rows,
            out_dir / f"{artifact_prefix}_group_population_recovery_rows.pkl",
        ),
        "population_rows_csv": _save_table(
            population_rows,
            out_dir / f"{artifact_prefix}_group_population_recovery_rows.csv",
        ),
        "individual_rows_pkl": _save_pickle(
            individual_rows,
            out_dir / f"{artifact_prefix}_group_individual_recovery_rows.pkl",
        ),
        "individual_rows_csv": _save_table(
            individual_rows,
            out_dir / f"{artifact_prefix}_group_individual_recovery_rows.csv",
        ),
    }


def load_group_recovery_rows(
    *,
    population_rows_path: str | Path,
    individual_rows_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Step 22 recovery-row artifacts.

    Parameters
    ----------
    population_rows_path : str or Path
        Pickle path for population recovery rows.
    individual_rows_path : str or Path
        Pickle path for individual recovery rows.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Population rows and individual rows.
    """

    return _load_pickle(population_rows_path), _load_pickle(individual_rows_path)


def examine_group_recovery(
    *,
    population_rows_path: str | Path,
    individual_rows_path: str | Path,
    result_dir: str | Path,
    artifact_prefix: str = "11_M3",
    base_params: list[str] | None = None,
    edge_params: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    """Compute group-generated population and individual recovery outputs.

    Parameters
    ----------
    population_rows_path : str or Path
        Pickle path for Step 22 population recovery rows.
    individual_rows_path : str or Path
        Pickle path for Step 22 individual recovery rows.
    result_dir : str or Path
        Directory to save CSV and plot outputs.
    artifact_prefix : str, optional
        File prefix for saved comparison outputs.
    base_params : list[str], optional
        Public subject parameter names shown in plots.
    edge_params : list[str], optional
        Population parameters used for quantile-bias diagnostics.

    Returns
    -------
    tuple
        Population rows, population summary, individual rows, individual
        correlations, and a dictionary of output paths.
    """

    population_rows, individual_rows = load_group_recovery_rows(
        population_rows_path=population_rows_path,
        individual_rows_path=individual_rows_path,
    )
    validate_recovery_contract(population_rows)
    validate_recovery_contract(individual_rows)

    population_summary = _summarize_recovery_ccc(
        population_rows,
        group_cols=["fit_model", "param"],
    )
    population_diagnostics = summarize_population_recovery_diagnostics(
        population_rows,
    )
    selected_edge_params = EDGE_PARAMS if edge_params is None else edge_params
    edge_param_quantile_bias = summarize_edge_param_quantile_bias(
        population_rows,
        params=selected_edge_params,
    )

    individual_correlations = _summarize_recovery_r(
        individual_rows,
        group_cols=["fit_model", "dataset_id", "param"],
    )

    out_dir = Path(result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "population_rows": _save_table(
            population_rows,
            out_dir / f"{artifact_prefix}_group_population_recovery_rows.csv",
        ),
        "population_summary": _save_table(
            population_summary,
            out_dir / f"{artifact_prefix}_group_population_recovery_summary.csv",
        ),
        "population_diagnostics": _save_table(
            population_diagnostics,
            out_dir / f"{artifact_prefix}_group_population_recovery_diagnostics.csv",
        ),
        "edge_param_quantile_bias": _save_table(
            edge_param_quantile_bias,
            out_dir / f"{artifact_prefix}_group_edge_param_quantile_bias.csv",
        ),
        "individual_rows": _save_table(
            individual_rows,
            out_dir / f"{artifact_prefix}_group_individual_recovery_rows.csv",
        ),
        "individual_correlations": _save_table(
            individual_correlations,
            out_dir / f"{artifact_prefix}_group_individual_recovery_correlations.csv",
        ),
    }

    paths["population_plot"] = _plot_group_population_recovery(
        population_rows,
        population_summary,
        out_dir / f"{artifact_prefix}_group_population_recovery_plot.png",
        base_params=base_params,
    )
    paths["individual_plot"] = _plot_group_individual_recovery(
        individual_correlations,
        out_dir / f"{artifact_prefix}_group_individual_recovery_plot.png",
        base_params=base_params,
    )

    return (
        population_rows,
        population_summary,
        individual_rows,
        individual_correlations,
        paths,
    )
