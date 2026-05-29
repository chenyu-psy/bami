"""BayesFlow-specific evaluation metric builders.

This module computes long-format tables for recovery and diagnostics, designed
to feed backend-agnostic plotting functions in ``utils.evaluation.plots``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from bami.evaluation.contracts import (
    validate_diagnostic_contract,
    validate_recovery_contract,
)
from bami.inference import summarize_subject_posterior

_M3_POSTHOC_WORKER_STATE: dict[str, object] = {}


def _progress(iterable, *, total: int | None = None, desc: str, show: bool):
    """Return a tqdm iterator that is visible in console-style notebooks.

    Parameters
    ----------
    iterable
        Values to iterate over.
    total
        Optional total number of steps for tqdm.
    desc
        Short label shown before the progress bar.
    show
        Whether to display the progress bar.

    Returns
    -------
    iterable
        ``iterable`` wrapped by tqdm when ``show`` is true.

    Notes
    -----
    Positron reliably displays stdout from Python sessions. Writing tqdm to
    stdout makes these recovery progress bars visible alongside BayesFlow's
    own sampling progress.
    """

    return tqdm(
        iterable,
        total=total,
        desc=desc,
        disable=not show,
        file=sys.stdout,
        dynamic_ncols=True,
    )


def _progress_message(message: str, show: bool) -> None:
    """Print a visible progress message when progress reporting is enabled.

    Parameters
    ----------
    message
        Message to print.
    show
        Whether to print the message.

    Returns
    -------
    None
        Writes to stdout with flushing so notebook consoles show the update
        immediately.
    """

    if show:
        print(message, flush=True)


def _resolve_n_jobs(n_jobs: int) -> int:
    """Convert a user-facing worker count into a positive integer.

    Parameters
    ----------
    n_jobs
        Number of parallel workers. Use ``1`` for serial execution, a positive
        integer for that many workers, or ``-1`` for all detected CPU cores.

    Returns
    -------
    int
        Positive worker count used by the recovery loop.
    """

    if n_jobs == -1:
        return os.cpu_count() or 1
    if n_jobs < 1:
        raise ValueError("n_jobs must be 1 or greater, or -1 for all CPU cores.")
    return int(n_jobs)


def _build_m3_posthoc_config(model) -> dict | None:
    """Extract a lightweight M3 config for process-safe posthoc workers.

    Parameters
    ----------
    model
        Candidate model passed to hierarchy posthoc recovery.

    Returns
    -------
    dict or None
        Pickle-friendly M3 settings for real M3 hierarchy models. Returns
        ``None`` for dummy test models or unsupported model classes so the
        caller can use a simpler fallback path.
    """

    if getattr(model, "posthoc_kind", None) != "m3":
        return None

    config = getattr(model, "posthoc_kwargs", None)
    if config is None:
        raise ValueError(
            "Cannot build process-safe M3 posthoc worker config. "
            "Missing workflow posthoc_kwargs."
        )
    return dict(config)


def _estimate_one_flex_dataset(
    *,
    model,
    test: Mapping[str, np.ndarray],
    samples: Mapping[str, np.ndarray],
    data: np.ndarray,
    dataset_id: int,
    n_candidates: int,
    min_ess: float,
    max_candidates: int,
    batch_candidates: int | None,
    adaptive: bool,
    base_params: Sequence[str],
    n_trials: int | None,
) -> list[dict]:
    """Estimate and tabulate posthoc recovery rows for one simulated dataset.

    Parameters
    ----------
    model
        Hierarchy model configured for posthoc subject posterior summaries.
    test
        Simulated data dictionary containing subject-level truth arrays.
    samples
        Group posterior samples for all simulated datasets.
    data
        Response-count array from ``test["data"]``. Fixed hierarchy uses five
        response columns; flex hierarchy uses five response columns plus trial
        count and active-subject mask.
    dataset_id
        Index of the simulated dataset to estimate.
    n_candidates, min_ess, max_candidates, batch_candidates, adaptive
        Posthoc candidate controls passed to ``summarize_subject_posterior``.
    base_params
        Individual parameter names to include in the recovery table.
    n_trials
        Optional fixed trial count added to each output row.

    Returns
    -------
    list[dict]
        Recovery rows for the requested dataset.
    """

    if data.shape[-1] == 5:
        active = np.ones(data.shape[1], dtype=bool)
        counts = data[dataset_id, :, :5]
    else:
        active = data[dataset_id, :, -1] > 0.5
        if "raw_counts" in test:
            raw_counts = np.asarray(test["raw_counts"])
            counts = raw_counts[dataset_id, active, :5]
        else:
            counts = data[dataset_id, active, :5]
    one_dataset_samples = {
        key: np.asarray(val)[dataset_id : dataset_id + 1]
        for key, val in samples.items()
    }
    estimates = summarize_subject_posterior(
        model,
        counts,
        group_samples=one_dataset_samples,
        n_candidates=n_candidates,
        min_ess=min_ess,
        max_candidates=max_candidates,
        batch_candidates=batch_candidates,
        adaptive=adaptive,
        random_seed=dataset_id,
    )

    rows: list[dict] = []
    for param in base_params:
        truth_key = f"{param}_subj"
        if truth_key in test:
            truth = np.asarray(test[truth_key][dataset_id], dtype=float)[active]
        else:
            truth_values = []
            for subject_id in range(int(np.sum(active))):
                indexed_key = f"{param}_subj_{subject_id}"
                if indexed_key not in test:
                    break
                truth_values.append(float(np.asarray(test[indexed_key][dataset_id])))
            if not truth_values:
                continue
            truth = np.asarray(truth_values, dtype=float)
        est_param = estimates[estimates["param"] == param].reset_index(drop=True)
        if truth.shape[0] != est_param.shape[0]:
            raise ValueError(
                f"Mismatched posthoc individual recovery rows for {param}."
            )

        for subject_id, true_val in enumerate(truth):
            estimate_row = est_param.loc[subject_id]
            row = {
                "level": "individual",
                "param": param,
                "true_value": float(true_val),
                "est_value": float(estimate_row["median"]),
                "dataset_id": int(dataset_id),
                "subject_id": int(subject_id),
                "model": "bayesflow_hierarchy_posthoc",
                "ess": float(estimate_row["ess"]),
                "ess_ratio": float(estimate_row["ess_ratio"]),
                "max_weight": float(estimate_row["max_weight"]),
                "n_candidates_used": int(estimate_row["n_candidates_used"]),
                "posthoc_status": str(estimate_row["posthoc_status"]),
            }
            if n_trials is not None:
                row["trial_count"] = int(n_trials)
            rows.append(row)

    return rows


def _init_m3_posthoc_worker(
    config: Mapping,
    test: Mapping[str, np.ndarray],
    samples: Mapping[str, np.ndarray],
    data: np.ndarray,
    n_candidates: int,
    min_ess: float,
    max_candidates: int,
    batch_candidates: int | None,
    adaptive: bool,
    base_params: Sequence[str],
    n_trials: int | None,
) -> None:
    """Initialize one process-safe M3 posthoc worker.

    Parameters
    ----------
    config
        Lightweight M3 model settings.
    test, samples, data
        Recovery inputs shared by all dataset tasks in the worker process.
    n_candidates, min_ess, max_candidates, batch_candidates, adaptive
        Posthoc controls passed to the worker-local estimator.
    base_params
        Individual parameter names to include in the recovery table.
    n_trials
        Optional fixed trial count added to each output row.

    Returns
    -------
    None
        Stores worker-local state for dataset tasks.
    """

    from bami.inference.posthoc import M3PosthocEstimator

    _M3_POSTHOC_WORKER_STATE.clear()
    _M3_POSTHOC_WORKER_STATE.update(
        {
            "model": M3PosthocEstimator(
                n_options=config["n_options"],
                rule=config["rule"],
                priors=config["priors"],
                const_params=config["const_params"],
                hier_params=config["hier_params"],
            ),
            "test": test,
            "samples": samples,
            "data": data,
            "n_candidates": n_candidates,
            "min_ess": min_ess,
            "max_candidates": max_candidates,
            "batch_candidates": batch_candidates,
            "adaptive": adaptive,
            "base_params": tuple(base_params),
            "n_trials": n_trials,
        }
    )


def _estimate_m3_posthoc_dataset_from_worker(dataset_id: int) -> list[dict]:
    """Estimate one dataset using process-local M3 posthoc state.

    Parameters
    ----------
    dataset_id
        Simulated dataset index assigned to this worker.

    Returns
    -------
    list[dict]
        Individual recovery rows for one dataset.
    """

    return _estimate_one_flex_dataset(
        model=_M3_POSTHOC_WORKER_STATE["model"],
        test=_M3_POSTHOC_WORKER_STATE["test"],
        samples=_M3_POSTHOC_WORKER_STATE["samples"],
        data=_M3_POSTHOC_WORKER_STATE["data"],
        dataset_id=dataset_id,
        n_candidates=_M3_POSTHOC_WORKER_STATE["n_candidates"],
        min_ess=_M3_POSTHOC_WORKER_STATE["min_ess"],
        max_candidates=_M3_POSTHOC_WORKER_STATE["max_candidates"],
        batch_candidates=_M3_POSTHOC_WORKER_STATE["batch_candidates"],
        adaptive=_M3_POSTHOC_WORKER_STATE["adaptive"],
        base_params=_M3_POSTHOC_WORKER_STATE["base_params"],
        n_trials=_M3_POSTHOC_WORKER_STATE["n_trials"],
    )


def _estimate_m3_datasets_process_safe(
    *,
    config: Mapping,
    test: Mapping[str, np.ndarray],
    samples: Mapping[str, np.ndarray],
    data: np.ndarray,
    dataset_ids: range,
    n_jobs: int,
    n_candidates: int,
    min_ess: float,
    max_candidates: int,
    batch_candidates: int | None,
    adaptive: bool,
    base_params: Sequence[str],
    n_trials: int | None,
    show_progress: bool,
) -> list[dict]:
    """Estimate M3 posthoc recovery with lightweight spawn workers.

    Parameters
    ----------
    config
        Lightweight M3 model settings used to build worker-local posthoc
        estimators.
    test, samples, data
        Recovery inputs for posthoc estimation.
    dataset_ids
        Dataset indices to estimate.
    n_jobs
        Positive number of workers.
    n_candidates, min_ess, max_candidates, batch_candidates, adaptive
        Posthoc candidate controls passed to ``summarize_subject_posterior``.
    base_params
        Individual parameter names to include in the recovery table.
    n_trials
        Optional fixed trial count added to each output row.
    show_progress
        Whether to display progress across datasets.

    Returns
    -------
    list[dict]
        Combined recovery rows in dataset order.

    Notes
    -----
    Workers use ``spawn`` and rebuild a workflow-free M3 posthoc object. This
    avoids moving loaded BayesFlow/Keras/Torch objects across process
    boundaries while still using true process-level parallelism.
    """

    rows: list[dict] = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=n_jobs,
        mp_context=context,
        initializer=_init_m3_posthoc_worker,
        initargs=(
            config,
            test,
            samples,
            data,
            n_candidates,
            min_ess,
            max_candidates,
            batch_candidates,
            adaptive,
            tuple(base_params),
            n_trials,
        ),
    ) as executor:
        results = executor.map(_estimate_m3_posthoc_dataset_from_worker, dataset_ids)
        progress = _progress(
            results,
            total=len(dataset_ids),
            desc="Hierarchy individual recovery",
            show=show_progress,
        )
        for dataset_rows in progress:
            rows.extend(dataset_rows)

    return rows


def _estimate_flex_datasets_parallel(
    *,
    model,
    test: Mapping[str, np.ndarray],
    samples: Mapping[str, np.ndarray],
    data: np.ndarray,
    dataset_ids: range,
    n_jobs: int,
    n_candidates: int,
    min_ess: float,
    max_candidates: int,
    batch_candidates: int | None,
    adaptive: bool,
    base_params: Sequence[str],
    n_trials: int | None,
    show_progress: bool,
) -> list[dict]:
    """Estimate M3 hierarchy posthoc recovery with process-safe workers.

    Parameters
    ----------
    model, test, samples, data
        Shared inputs for posthoc recovery.
    dataset_ids
        Dataset indices to estimate.
    n_jobs
        Positive number of workers.
    n_candidates, min_ess, max_candidates, batch_candidates, adaptive
        Posthoc controls passed to ``summarize_subject_posterior``.
    base_params
        Individual parameter names to include in the recovery table.
    n_trials
        Optional fixed trial count added to each output row.
    show_progress
        Whether to display progress across datasets.

    Returns
    -------
    list[dict]
        Combined recovery rows in dataset order.
    """

    config = _build_m3_posthoc_config(model)
    if config is None:
        raise ValueError(
            "Parallel hierarchy posthoc recovery supports M3 fixed/flex hierarchy "
            "models only. Custom models must use n_jobs=1."
        )

    return _estimate_m3_datasets_process_safe(
        config=config,
        test=test,
        samples=samples,
        data=data,
        dataset_ids=dataset_ids,
        n_jobs=n_jobs,
        n_candidates=n_candidates,
        min_ess=min_ess,
        max_candidates=max_candidates,
        batch_candidates=batch_candidates,
        adaptive=adaptive,
        base_params=base_params,
        n_trials=n_trials,
        show_progress=show_progress,
    )


def _as_1d(arr: np.ndarray) -> np.ndarray:
    """Flatten arrays into a 1D vector for row-wise tabulation.

    Parameters
    ----------
    arr
        Array with trailing singleton dimensions.

    Returns
    -------
    np.ndarray
        Flattened 1D array.
    """

    return np.asarray(arr, dtype=float).reshape(-1)


def _posterior_median(samples: np.ndarray) -> np.ndarray:
    """Compute posterior median across posterior-sample axis.

    Parameters
    ----------
    samples
        Posterior draws with shape ``(n_datasets, n_samples, ...)``.

    Returns
    -------
    np.ndarray
        Median along axis 1, keeping dataset axis first.
    """

    return np.median(np.asarray(samples, dtype=float), axis=1)


def _parse_fixed_subject_key(key: str) -> tuple[str, int] | None:
    """Parse public fixed-subject keys and ignore raw-space subject keys.

    Parameters
    ----------
    key
        Candidate key such as ``a_subj_0``.

    Returns
    -------
    tuple[str, int] or None
        Base parameter and subject index for public subject keys. Returns
        ``None`` for raw keys such as ``a_subj_raw_0``.
    """

    if "_subj_" not in key:
        return None

    param, subj_str = key.rsplit("_subj_", maxsplit=1)
    if not subj_str.isdigit():
        return None
    return param, int(subj_str)


def _resolve_test_data(
    workflow, test_data: Mapping[str, np.ndarray] | int
) -> Mapping[str, np.ndarray]:
    """Resolve test data from dict input or by simulation size.

    Parameters
    ----------
    workflow
        BayesFlow workflow instance exposing ``simulate``.
    test_data
        Existing test dictionary or simulation size.

    Returns
    -------
    Mapping[str, np.ndarray]
        Test data dictionary used for diagnostics.
    """

    if isinstance(test_data, int):
        return workflow.simulate(test_data)
    return test_data


def _sample_posterior(
    workflow,
    test_data: Mapping[str, np.ndarray],
    num_samples: int,
    approximator_kwargs: Mapping | None = None,
    sample_batch_size: int | None = None,
) -> Mapping[str, np.ndarray]:
    """Sample posterior draws from BayesFlow workflow.

    Parameters
    ----------
    workflow
        BayesFlow workflow instance exposing ``sample``.
    test_data
        Conditions passed to ``workflow.sample``.
    num_samples
        Number of posterior draws per dataset.
    approximator_kwargs
        Optional keyword arguments forwarded to ``workflow.sample``.
    sample_batch_size
        Optional number of datasets per posterior sampling batch. This controls
        memory during diagnostics and is independent of training batch size.

    Returns
    -------
    Mapping[str, np.ndarray]
        Posterior samples dictionary keyed by parameter name.
    """

    sample_kwargs = dict(approximator_kwargs or {})
    if sample_batch_size is not None:
        sample_kwargs["batch_size"] = int(sample_batch_size)
    samples = workflow.sample(
        num_samples=num_samples, conditions=test_data, **sample_kwargs
    )
    transform = getattr(workflow, "transform_posterior_samples", None)
    if transform is not None:
        return transform(samples)
    return samples


def _check_ci(ci: float) -> float:
    """Return a validated credible-interval width.

    Parameters
    ----------
    ci
        Credible-interval mass, such as ``0.95`` for a central 95% interval.

    Returns
    -------
    float
        Validated interval mass.
    """

    ci = float(ci)
    if ci <= 0 or ci >= 1:
        raise ValueError("ci must be between 0 and 1.")
    return ci


def _squeeze_trailing_singletons(
    arr: np.ndarray,
    min_ndim: int = 1,
) -> np.ndarray:
    """Remove trailing singleton dimensions that do not name parameters.

    Parameters
    ----------
    arr
        Posterior draw array from BayesFlow or a transform helper.
    min_ndim
        Number of dimensions that should be preserved even when the final
        dimension has length one.

    Returns
    -------
    numpy.ndarray
        Array with trailing length-one dimensions removed.
    """

    out = arr
    while out.ndim > min_ndim and out.shape[-1] == 1:
        out = np.squeeze(out, axis=-1)
    return out


def _summarize_draws(draws: np.ndarray, ci: float) -> dict[str, float]:
    """Summarize one vector of posterior draws.

    Parameters
    ----------
    draws
        One-dimensional posterior draws for one parameter and dataset.
    ci
        Credible-interval mass.

    Returns
    -------
    dict[str, float]
        Posterior mean, standard deviation, and central interval bounds.
    """

    draw_arr = np.asarray(draws, dtype=float).reshape(-1)
    alpha = (1.0 - ci) / 2.0
    sd = np.std(draw_arr, ddof=1) if draw_arr.size > 1 else 0.0
    return {
        "estimate": float(np.mean(draw_arr)),
        "sd": float(sd),
        "ci_lower": float(np.quantile(draw_arr, alpha)),
        "ci_upper": float(np.quantile(draw_arr, 1.0 - alpha)),
    }


def _lookup_ess(
    ess: Mapping[str, object] | None,
    param: str,
    index: tuple[int, ...] = (),
) -> float:
    """Return an ESS value for one summary row.

    Parameters
    ----------
    ess
        Optional mapping from parameter name to scalar or indexed ESS values.
    param
        Parameter name being summarized.
    index
        Dataset and subject indexes for this row, if present.

    Returns
    -------
    float
        Matching ESS value, or ``nan`` when none is available.
    """

    if ess is None or param not in ess:
        return np.nan
    value = np.asarray(ess[param])
    if value.ndim == 0:
        return float(value)
    if len(index) == value.ndim and all(i < s for i, s in zip(index, value.shape)):
        return float(value[index])
    if value.ndim == 1 and index and index[-1] < value.shape[0]:
        return float(value[index[-1]])
    return np.nan


def summarize_group_parameters(
    samples: Mapping[str, np.ndarray],
    ci: float = 0.95,
    ess: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Summarize group-level posterior draws in a researcher-friendly table.

    Parameters
    ----------
    samples
        Mapping from group-level parameter name to posterior draws. Arrays
        should be shaped ``(n_datasets, n_samples)`` for batched data, or
        ``(n_samples,)`` for one dataset without an explicit dataset axis.
    ci
        Central credible-interval width.
    ess
        Optional mapping from parameter name to effective sample size. Values
        may be scalars or one value per dataset.

    Returns
    -------
    pandas.DataFrame
        Table with ``param``, ``estimate``, ``sd``, ``ci_lower``, ``ci_upper``,
        and ``ess``. A ``dataset_id`` column is included when the samples have
        an explicit dataset axis.
    """

    ci = _check_ci(ci)
    rows = []
    for param, values in samples.items():
        arr = _squeeze_trailing_singletons(np.asarray(values, dtype=float))
        if arr.ndim == 1:
            row = {"param": param, **_summarize_draws(arr, ci)}
            row["ess"] = _lookup_ess(ess, param)
            rows.append(row)
        elif arr.ndim == 2:
            for dataset_id in range(arr.shape[0]):
                row = {
                    "dataset_id": int(dataset_id),
                    "param": param,
                    **_summarize_draws(arr[dataset_id, :], ci),
                }
                row["ess"] = _lookup_ess(ess, param, (dataset_id,))
                rows.append(row)
        else:
            raise ValueError(
                "group parameter samples must have shape (n_samples,) or "
                "(n_datasets, n_samples)."
            )
    return pd.DataFrame(rows)


def summarize_random_parameters(
    samples: Mapping[str, np.ndarray],
    ci: float = 0.95,
    ess: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Summarize subject-level posterior draws in a long table.

    Parameters
    ----------
    samples
        Mapping from subject-level parameter name to posterior draws. Arrays
        should be shaped ``(n_samples, n_subjects)`` for one dataset or
        ``(n_datasets, n_samples, n_subjects)`` for batched data.
    ci
        Central credible-interval width.
    ess
        Optional mapping from parameter name to effective sample size. Values
        may be scalars, one value per subject, or one value per dataset and
        subject.

    Returns
    -------
    pandas.DataFrame
        Table with ``param``, ``subject_id``, ``estimate``, ``sd``,
        ``ci_lower``, ``ci_upper``, and ``ess``. A ``dataset_id`` column is
        included when the samples have an explicit dataset axis.
    """

    ci = _check_ci(ci)
    rows = []
    for param, values in samples.items():
        arr = _squeeze_trailing_singletons(
            np.asarray(values, dtype=float),
            min_ndim=3,
        )
        if arr.ndim == 2:
            for subject_id in range(arr.shape[1]):
                row = {
                    "subject_id": int(subject_id),
                    "param": param,
                    **_summarize_draws(arr[:, subject_id], ci),
                }
                row["ess"] = _lookup_ess(ess, param, (subject_id,))
                rows.append(row)
        elif arr.ndim == 3:
            for dataset_id in range(arr.shape[0]):
                for subject_id in range(arr.shape[2]):
                    row = {
                        "dataset_id": int(dataset_id),
                        "subject_id": int(subject_id),
                        "param": param,
                        **_summarize_draws(arr[dataset_id, :, subject_id], ci),
                    }
                    row["ess"] = _lookup_ess(ess, param, (dataset_id, subject_id))
                    rows.append(row)
        else:
            raise ValueError(
                "random parameter samples must have shape "
                "(n_samples, n_subjects) or "
                "(n_datasets, n_samples, n_subjects)."
            )
    return pd.DataFrame(rows)


def _check_indexed_subject_recovery_allowed(workflow) -> None:
    """Fail clearly when indexed subject outputs are not subject-aligned.

    Parameters
    ----------
    workflow
        BayesFlow workflow passed to legacy ``bf_ind_recovery`` calls. Only
        workflows that explicitly mark indexed subject recovery as aligned are
        allowed to use this path.

    Returns
    -------
    None
        Raises ``ValueError`` when direct indexed individual recovery is unsafe.
    """

    aligned = getattr(workflow, "indexed_subject_recovery_aligned", None)
    if aligned is True:
        return

    raise ValueError(
        "Legacy workflow-only individual recovery is unsafe unless the workflow "
        "explicitly marks indexed subject recovery as aligned. Pass the model "
        "to bf_ind_recovery(model=...) so the correct model-specific recovery "
        "method can be selected."
    )


def _simple_subject_conditions(model, test_data: Mapping[str, np.ndarray]) -> dict:
    """Build simple-model conditions from group-generated subject counts.

    Parameters
    ----------
    model
        Fixed-simple or flex-simple M3 model.
    test_data
        Group-style recovery data with ``data`` shaped
        ``(n_datasets, n_subjects, 5)``.

    Returns
    -------
    dict
        BayesFlow conditions for fitting each subject count vector as one
        simple-model dataset.
    """

    data = np.asarray(test_data["data"], dtype=np.float32)
    if data.ndim != 3 or data.shape[-1] != 5:
        raise ValueError(
            "Simple-model individual recovery requires test_data['data'] with "
            "shape (n_datasets, n_subjects, 5)."
        )

    n_datasets, n_subjects, n_categories = data.shape
    flat_counts = data.reshape(n_datasets * n_subjects, 1, n_categories)
    if hasattr(model, "_prepare_observed_counts"):
        conditions, _ = model._prepare_observed_counts(flat_counts)
        return {"data": conditions}
    return {"data": flat_counts}


def _estimate_simple_model_subject_recovery(
    *,
    model,
    test_data: Mapping[str, np.ndarray],
    num_samples: int,
    base_params: Sequence[str],
    approximator_kwargs: Mapping | None,
    sample_batch_size: int | None = None,
    samples: Mapping[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Estimate individual recovery by fitting each subject with a simple model.

    Parameters
    ----------
    model
        Fixed-simple or flex-simple M3 model.
    test_data
        Group-style simulated data containing subject count rows and true
        ``<param>_subj`` arrays or legacy ``<param>_subj_<id>`` values.
    num_samples
        Posterior draws per subject.
    base_params
        Individual parameter names to include.
    approximator_kwargs
        Optional keyword arguments forwarded to BayesFlow sampling.
    sample_batch_size
        Optional number of datasets per posterior sampling batch.
    samples
        Optional precomputed posterior samples for the flattened subject
        conditions.

    Returns
    -------
    pd.DataFrame
        Individual recovery rows aligned by dataset and subject ID.
    """

    count_data = np.asarray(test_data["data"], dtype=np.float32)
    conditions = _simple_subject_conditions(model, test_data)
    if samples is None:
        samples = sample_posterior(
            workflow=model.workflow,
            test_data=conditions,
            num_samples=num_samples,
            approximator_kwargs=approximator_kwargs,
            sample_batch_size=sample_batch_size,
        )

    n_datasets, n_subjects = count_data.shape[:2]
    rows: list[dict] = []
    for base_param in base_params:
        if base_param not in samples:
            continue
        est = np.median(np.asarray(samples[base_param], dtype=float), axis=1)
        est = est.reshape(n_datasets, n_subjects)
        for dataset_id in range(n_datasets):
            for subject_id in range(n_subjects):
                array_truth_key = f"{base_param}_subj"
                indexed_truth_key = f"{base_param}_subj_{subject_id}"
                if array_truth_key in test_data:
                    true_value = np.asarray(test_data[array_truth_key])[
                        dataset_id, subject_id
                    ]
                elif indexed_truth_key in test_data:
                    true_value = test_data[indexed_truth_key][dataset_id]
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
    if out.empty:
        raise ValueError("No simple-model individual recovery rows were created.")
    validate_recovery_contract(out)
    return out


def sample_posterior(
    workflow,
    test_data: Mapping[str, np.ndarray],
    num_samples: int,
    approximator_kwargs: Mapping | None = None,
    sample_batch_size: int | None = None,
) -> Mapping[str, np.ndarray]:
    """Sample posterior draws from a BayesFlow workflow.

    This is a lower-level compatibility helper for code that already works
    directly with ``model.workflow``. New user-facing scripts should usually
    call ``model.sample_posterior(...)`` for simple workflows or
    ``model.sample_group_posterior(...)`` for hierarchical group parameters.

    Parameters
    ----------
    workflow
        BayesFlow workflow exposing ``sample``.
    test_data
        Conditions passed to ``workflow.sample``.
    num_samples
        Number of posterior draws per dataset.
    approximator_kwargs
        Optional keyword arguments forwarded to ``workflow.sample``.
    sample_batch_size
        Optional number of datasets per posterior sampling batch. Use this for
        large nested diagnostics where sending all conditions at once would
        exceed accelerator memory. ``None`` keeps BayesFlow's default behavior.

    Returns
    -------
    Mapping[str, np.ndarray]
        Posterior samples keyed by public parameter names when the workflow
        exposes ``transform_posterior_samples``.
    """

    return _sample_posterior(
        workflow=workflow,
        test_data=test_data,
        num_samples=num_samples,
        approximator_kwargs=approximator_kwargs,
        sample_batch_size=sample_batch_size,
    )


def estimate_population_recovery(
    test_data: Mapping[str, np.ndarray],
    samples: Mapping[str, np.ndarray],
    variable_keys: Sequence[str] | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Convert posterior samples to population recovery rows.

    Parameters
    ----------
    test_data
        Truth dictionary used as fitting conditions.
    samples
        Posterior sample dictionary after any public-space transform.
    variable_keys
        Optional explicit list of population keys. If ``None``, infer
        population-level keys from ``test_data`` and ``samples``.
    show_progress
        Whether to display progress while tabulating rows.

    Returns
    -------
    pd.DataFrame
        Recovery contract table with ``level='population'``.
    """

    if variable_keys is None:
        hierarchical_keys = [
            k
            for k in test_data.keys()
            if (
                k.endswith("_mu")
                or (k.endswith("_sigma") and not k.endswith("_log_sigma"))
            )
            and k in samples
        ]
        if hierarchical_keys:
            keys = hierarchical_keys
        else:
            keys = [
                k
                for k in test_data.keys()
                if ("_subj_" not in k)
                and (not k.endswith("_log_sigma"))
                and k in samples
            ]
    else:
        keys = [
            k
            for k in variable_keys
            if (not k.endswith("_log_sigma")) and k in test_data and k in samples
        ]

    if not keys:
        raise ValueError("No population-level variable keys found for recovery.")

    rows: list[dict] = []
    selected_keys = sorted(keys)
    total_rows = 0
    for key in selected_keys:
        total_rows += _as_1d(test_data[key]).shape[0]

    _progress_message(
        f"Population recovery: writing {total_rows} rows for {len(selected_keys)} parameters.",
        show_progress,
    )
    progress = _progress(
        range(total_rows),
        total=total_rows,
        desc="Population recovery",
        show=show_progress,
    )
    progress_iter = iter(progress)

    for key in selected_keys:
        truth = _as_1d(test_data[key])
        est = _as_1d(_posterior_median(samples[key]))
        if truth.shape[0] != est.shape[0]:
            raise ValueError(
                f"Mismatched shape for key '{key}' in population recovery."
            )

        for dataset_id, (true_val, est_val) in enumerate(zip(truth, est, strict=True)):
            next(progress_iter)
            rows.append(
                {
                    "level": "population",
                    "param": key,
                    "true_value": float(true_val),
                    "est_value": float(est_val),
                    "dataset_id": int(dataset_id),
                    "model": "bayesflow",
                }
            )
    progress.close()
    _progress_message(f"Population recovery: done ({len(rows)} rows).", show_progress)

    out = pd.DataFrame(rows)
    validate_recovery_contract(out)
    return out


def estimate_fixed_individual_recovery(
    test_data: Mapping[str, np.ndarray],
    samples: Mapping[str, np.ndarray],
    base_params: Sequence[str] = ("a", "c", "ra", "rc"),
) -> pd.DataFrame:
    """Convert fixed-subject posterior samples to individual recovery rows.

    Parameters
    ----------
    test_data
        Truth dictionary containing keys like ``a_subj_0``.
    samples
        Posterior sample dictionary containing matching subject-level keys.
    base_params
        Base parameter names to include.

    Returns
    -------
    pd.DataFrame
        Recovery contract table with ``level='individual'``.
    """

    subj_keys = []
    for key in test_data.keys():
        if key in samples and _parse_fixed_subject_key(key) is not None:
            subj_keys.append(key)
    if not subj_keys:
        raise ValueError(
            "No subject-level keys were found. Individual recovery requires infer_subj_level=True."
        )

    rows: list[dict] = []
    allowed = set(base_params)

    for key in sorted(subj_keys):
        parsed = _parse_fixed_subject_key(key)
        if parsed is None:
            continue
        param, subject_id = parsed
        if param not in allowed:
            continue

        truth = _as_1d(test_data[key])
        est = _as_1d(_posterior_median(samples[key]))
        if truth.shape[0] != est.shape[0]:
            raise ValueError(
                f"Mismatched shape for key '{key}' in individual recovery."
            )

        for dataset_id, (true_val, est_val) in enumerate(zip(truth, est, strict=True)):
            rows.append(
                {
                    "level": "individual",
                    "param": param,
                    "true_value": float(true_val),
                    "est_value": float(est_val),
                    "dataset_id": int(dataset_id),
                    "subject_id": subject_id,
                    "model": "bayesflow",
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(
            "No matching subject-level rows created for requested base_params."
        )
    validate_recovery_contract(out)
    return out


def estimate_flex_individual_recovery(
    model,
    test_data: Mapping[str, np.ndarray],
    samples: Mapping[str, np.ndarray],
    n_candidates: int = 4000,
    min_ess: float = 200.0,
    max_candidates: int = 20000,
    batch_candidates: int | None = None,
    adaptive: bool = True,
    base_params: Sequence[str] = ("a", "c", "ra", "rc"),
    n_trials: int | None = None,
    show_progress: bool = True,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Convert flex-hierarchy posthoc estimates to individual recovery rows.

    Parameters
    ----------
    model
        Hierarchy model configured for posthoc subject posterior summaries.
    test_data
        Flex-formatted data and subject-level truth arrays.
    samples
        Group posterior samples for all datasets.
    n_candidates, min_ess, max_candidates, batch_candidates, adaptive
        Posthoc candidate controls passed to ``summarize_subject_posterior``.
    base_params
        Base individual parameters to include.
    n_trials
        Optional fixed trial count added to output rows.
    show_progress
        Whether to display progress across datasets.
    n_jobs
        Worker count for dataset-level parallelization.

    Returns
    -------
    pd.DataFrame
        Recovery contract table comparing subject truth and posthoc medians.
    """

    data = np.asarray(test_data["data"])
    n_datasets = data.shape[0]
    worker_count = _resolve_n_jobs(n_jobs)

    _progress_message(
        (
            "Hierarchy individual recovery: estimating "
            f"{n_datasets} simulated datasets with n_jobs={worker_count}."
        ),
        show_progress,
    )

    dataset_ids = range(n_datasets)
    if worker_count == 1:
        rows: list[dict] = []
        for dataset_id in _progress(
            dataset_ids,
            total=n_datasets,
            desc="Hierarchy individual recovery",
            show=show_progress,
        ):
            rows.extend(
                _estimate_one_flex_dataset(
                    model=model,
                    test=test_data,
                    samples=samples,
                    data=data,
                    dataset_id=dataset_id,
                    n_candidates=n_candidates,
                    min_ess=min_ess,
                    max_candidates=max_candidates,
                    batch_candidates=batch_candidates,
                    adaptive=adaptive,
                    base_params=base_params,
                    n_trials=n_trials,
                )
            )
    else:
        rows = _estimate_flex_datasets_parallel(
            model=model,
            test=test_data,
            samples=samples,
            data=data,
            dataset_ids=dataset_ids,
            n_jobs=worker_count,
            n_candidates=n_candidates,
            min_ess=min_ess,
            max_candidates=max_candidates,
            batch_candidates=batch_candidates,
            adaptive=adaptive,
            base_params=base_params,
            n_trials=n_trials,
            show_progress=show_progress,
        )

    _progress_message(
        f"Hierarchy individual recovery: done ({len(rows)} rows).", show_progress
    )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No flex individual recovery rows were created.")
    validate_recovery_contract(out)
    return out


def bf_pop_recovery(
    workflow,
    test_data: Mapping[str, np.ndarray] | int,
    num_samples: int = 500,
    variable_keys: Sequence[str] | None = None,
    approximator_kwargs: Mapping | None = None,
    sample_batch_size: int | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Compute population-level recovery rows from BayesFlow outputs.

    Parameters
    ----------
    workflow
        Trained BayesFlow workflow.
    test_data
        Test data dictionary or number of simulated datasets.
    num_samples
        Posterior sample count per dataset.
    variable_keys
        Optional explicit list of population keys. If ``None``, infer keys
        ending with public ``_mu`` or ``_sigma``. Raw ``*_log_sigma`` keys are
        internal unconstrained variables and are not reported.
    approximator_kwargs
        Optional sampling kwargs.
    sample_batch_size
        Optional number of datasets per posterior sampling batch. This is useful
        for padded nested trial data where diagnostic data are much larger than
        training batches.
    show_progress
        Whether to show a progress bar while tabulating recovery rows.

    Returns
    -------
    pd.DataFrame
        Recovery contract table with ``level='population'``.
    """

    test = _resolve_test_data(workflow, test_data)
    samples = sample_posterior(
        workflow=workflow,
        test_data=test,
        num_samples=num_samples,
        approximator_kwargs=approximator_kwargs,
        sample_batch_size=sample_batch_size,
    )
    return estimate_population_recovery(
        test_data=test,
        samples=samples,
        variable_keys=variable_keys,
        show_progress=show_progress,
    )


def bf_ind_recovery(
    workflow=None,
    test_data: Mapping[str, np.ndarray] | int | None = None,
    num_samples: int = 500,
    base_params: Sequence[str] = ("a", "c", "ra", "rc"),
    approximator_kwargs: Mapping | None = None,
    sample_batch_size: int | None = None,
    samples: Mapping[str, np.ndarray] | None = None,
    *,
    model=None,
    n_candidates: int = 4000,
    min_ess: float = 200.0,
    max_candidates: int = 20000,
    batch_candidates: int | None = None,
    adaptive: bool = True,
    n_trials: int | None = None,
    show_progress: bool = True,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Compute model-aware individual-level recovery rows.

    Parameters
    ----------
    workflow
        Legacy BayesFlow workflow. This path is allowed only when the workflow
        explicitly marks indexed subject recovery as aligned. Prefer ``model``.
    test_data
        Test data dictionary or number of simulated datasets.
    num_samples
        Posterior sample count per dataset.
    base_params
        Base parameter names expected for ``*_subj_<id>`` keys.
    approximator_kwargs
        Optional sampling kwargs.
    sample_batch_size
        Optional number of datasets per posterior sampling batch.
    samples
        Optional precomputed posterior samples. This is useful when the same
        samples are already used for population recovery.
    model
        Optional model object. When provided, recovery dispatches by model
        capability: hierarchy models use posthoc recovery and simple models fit
        each subject count vector as one simple-model dataset.
    n_candidates, min_ess, max_candidates, batch_candidates, adaptive
        Posthoc controls used for exchangeable hierarchy models.
    n_trials
        Optional fixed trial count used when simulating flex-hierarchy recovery
        data from an integer ``test_data``.
    show_progress
        Whether to show progress for posthoc recovery.
    n_jobs
        Worker count for posthoc recovery.

    Returns
    -------
    pd.DataFrame
        Recovery contract table with ``level='individual'``.
    """

    if test_data is None:
        raise ValueError("bf_ind_recovery requires test_data.")

    if model is None and workflow is not None and hasattr(workflow, "workflow"):
        model = workflow
        workflow = None

    if model is not None:
        workflow_level = getattr(model.workflow, "workflow_level", None)
        workflow_family = getattr(model.workflow, "workflow_family", None)

        if workflow_level == "hierarchical" and workflow_family == "fixed_hierarchical":
            test = _resolve_test_data(model.workflow, test_data)
            if samples is None:
                samples = sample_posterior(
                    workflow=model.workflow,
                    test_data=test,
                    num_samples=num_samples,
                    approximator_kwargs=approximator_kwargs,
                    sample_batch_size=sample_batch_size,
                )
            return estimate_flex_individual_recovery(
                model=model,
                test_data=test,
                samples=samples,
                n_candidates=n_candidates,
                min_ess=min_ess,
                max_candidates=max_candidates,
                batch_candidates=batch_candidates,
                adaptive=adaptive,
                base_params=base_params,
                n_trials=n_trials,
                show_progress=show_progress,
                n_jobs=n_jobs,
            )

        if workflow_level == "hierarchical" and workflow_family == "flex_hierarchical":
            if samples is not None:
                if isinstance(test_data, int):
                    raise ValueError(
                        "Precomputed samples for flex-hierarchy recovery require "
                        "explicit test_data, not an integer simulation count."
                    )
                return estimate_flex_individual_recovery(
                    model=model,
                    test_data=test_data,
                    samples=samples,
                    n_candidates=n_candidates,
                    min_ess=min_ess,
                    max_candidates=max_candidates,
                    batch_candidates=batch_candidates,
                    adaptive=adaptive,
                    base_params=base_params,
                    n_trials=n_trials,
                    show_progress=show_progress,
                    n_jobs=n_jobs,
                )
            return bf_flex_ind_recovery(
                model=model,
                test_data=test_data,
                num_group_samples=num_samples,
                sample_batch_size=sample_batch_size,
                n_candidates=n_candidates,
                min_ess=min_ess,
                max_candidates=max_candidates,
                batch_candidates=batch_candidates,
                adaptive=adaptive,
                base_params=base_params,
                n_trials=n_trials,
                show_progress=show_progress,
                n_jobs=n_jobs,
            )

        if workflow_level == "simple":
            test = _resolve_test_data(model.workflow, test_data)
            return _estimate_simple_model_subject_recovery(
                model=model,
                test_data=test,
                num_samples=num_samples,
                base_params=base_params,
                approximator_kwargs=approximator_kwargs,
                sample_batch_size=sample_batch_size,
                samples=samples,
            )

        raise ValueError(
            "Unsupported model for bf_ind_recovery. Pass a simple or "
            "hierarchical workflow model."
        )

    if workflow is None:
        raise ValueError("bf_ind_recovery requires either model or workflow.")

    _check_indexed_subject_recovery_allowed(workflow)
    test = _resolve_test_data(workflow, test_data)
    if samples is None:
        samples = sample_posterior(
            workflow=workflow,
            test_data=test,
            num_samples=num_samples,
            approximator_kwargs=approximator_kwargs,
            sample_batch_size=sample_batch_size,
        )
    return estimate_fixed_individual_recovery(
        test_data=test,
        samples=samples,
        base_params=base_params,
    )


def bf_flex_ind_recovery(
    model,
    test_data: Mapping[str, np.ndarray] | int,
    num_group_samples: int = 500,
    sample_batch_size: int | None = None,
    n_candidates: int = 4000,
    min_ess: float = 200.0,
    max_candidates: int = 20000,
    batch_candidates: int | None = None,
    adaptive: bool = True,
    base_params: Sequence[str] = ("a", "c", "ra", "rc"),
    n_trials: int | None = None,
    show_progress: bool = True,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Compute posthoc individual recovery for ``m3_flex_hierarchy``.

    Parameters
    ----------
    model
        Flexible hierarchy model exposing ``workflow`` and posthoc estimator
        settings.
    test_data
        Simulated flex-hierarchy data dictionary or number of datasets to
        simulate from ``model.workflow``.
    num_group_samples
        Number of group posterior draws from BayesFlow.
    sample_batch_size
        Optional number of group datasets per posterior sampling batch.
    n_candidates
        Number of likelihood-weighted posthoc candidates per subject.
    min_ess
        Target effective sample size for adaptive posthoc sampling.
    max_candidates
        Maximum candidates allowed per subject.
    batch_candidates
        Extra candidates drawn per adaptive step. If ``None``, use
        ``n_candidates``.
    adaptive
        Whether to add candidates when ESS is low.
    base_params
        Base individual parameter names to evaluate.
    n_trials
        Optional fixed trial count for simulated recovery data. This is used
        only when ``test_data`` is an integer. The model's original
        ``n_trials_range`` is restored before returning. When provided, the
        output includes ``trial_count`` for plotting sensitivity analyses.
    show_progress
        Whether to show progress across datasets. This is useful because
        posthoc individual estimation is usually the slowest diagnostic step.
    n_jobs
        Number of datasets to estimate at the same time. Use ``1`` for the
        original serial behavior, a positive integer for that many workers, or
        ``-1`` for all detected CPU cores.

    Returns
    -------
    pd.DataFrame
        Recovery contract table comparing simulated subject truth with posthoc
        subject estimates.
    """

    original_range = getattr(model, "n_trials_range", None)
    if n_trials is not None and int(n_trials) < 1:
        raise ValueError("n_trials must be positive when provided.")

    try:
        if n_trials is not None and isinstance(test_data, int):
            model.n_trials_range = (int(n_trials), int(n_trials) + 1)
        test = _resolve_test_data(model.workflow, test_data)
    finally:
        if n_trials is not None and original_range is not None:
            model.n_trials_range = original_range

    samples = sample_posterior(
        workflow=model.workflow,
        test_data=test,
        num_samples=num_group_samples,
        sample_batch_size=sample_batch_size,
    )
    return estimate_flex_individual_recovery(
        model=model,
        test_data=test,
        samples=samples,
        n_candidates=n_candidates,
        min_ess=min_ess,
        max_candidates=max_candidates,
        batch_candidates=batch_candidates,
        adaptive=adaptive,
        base_params=base_params,
        n_trials=n_trials,
        show_progress=show_progress,
        n_jobs=n_jobs,
    )


def _extract_metric_long(
    metrics_df: pd.DataFrame,
    metric_name: str,
) -> pd.DataFrame:
    """Extract one metric row from BayesFlow diagnostics into long format.

    Parameters
    ----------
    metrics_df
        Output from ``workflow.compute_default_diagnostics(..., as_data_frame=True)``.
    metric_name
        Exact row label to extract.

    Returns
    -------
    pd.DataFrame
        Long table with columns ``param``, ``metric``, ``value``, and ``model``.
    """

    if metric_name not in metrics_df.index:
        raise ValueError(f"Metric '{metric_name}' not found in diagnostics table.")

    row = metrics_df.loc[metric_name]
    out = pd.DataFrame(
        {
            "param": row.index.astype(str),
            "metric": metric_name,
            "value": row.to_numpy(dtype=float),
            "model": "bayesflow",
        }
    )
    validate_diagnostic_contract(out)
    return out


def bf_calibration(
    workflow,
    test_data: Mapping[str, np.ndarray] | int,
    num_samples: int = 500,
    variable_keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute calibration metrics (Log Gamma) in long format.

    Parameters
    ----------
    workflow
        Trained BayesFlow workflow.
    test_data
        Test data dictionary or number of simulated datasets.
    num_samples
        Posterior sample count per dataset.
    variable_keys
        Optional subset of keys passed to BayesFlow diagnostics.

    Returns
    -------
    pd.DataFrame
        Diagnostic table with ``metric='Log Gamma'``.
    """

    metrics = workflow.compute_default_diagnostics(
        test_data=test_data,
        num_samples=num_samples,
        variable_keys=variable_keys,
        as_data_frame=True,
    )
    return _extract_metric_long(metrics, "Log Gamma")


def bf_coverage(
    workflow,
    test_data: Mapping[str, np.ndarray] | int,
    num_samples: int = 500,
    variable_keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute coverage-error metrics in long format.

    Parameters
    ----------
    workflow
        Trained BayesFlow workflow.
    test_data
        Test data dictionary or number of simulated datasets.
    num_samples
        Posterior sample count per dataset.
    variable_keys
        Optional subset of keys passed to BayesFlow diagnostics.

    Returns
    -------
    pd.DataFrame
        Diagnostic table with ``metric='Calibration Error'``.
    """

    metrics = workflow.compute_default_diagnostics(
        test_data=test_data,
        num_samples=num_samples,
        variable_keys=variable_keys,
        as_data_frame=True,
    )
    return _extract_metric_long(metrics, "Calibration Error")


def bf_zscore(
    workflow,
    test_data: Mapping[str, np.ndarray] | int,
    num_samples: int = 500,
    variable_keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute z-score contraction metrics in long format.

    Parameters
    ----------
    workflow
        Trained BayesFlow workflow.
    test_data
        Test data dictionary or number of simulated datasets.
    num_samples
        Posterior sample count per dataset.
    variable_keys
        Optional subset of keys passed to BayesFlow diagnostics.

    Returns
    -------
    pd.DataFrame
        Diagnostic table with ``metric='Posterior Contraction'``.
    """

    metrics = workflow.compute_default_diagnostics(
        test_data=test_data,
        num_samples=num_samples,
        variable_keys=variable_keys,
        as_data_frame=True,
    )
    return _extract_metric_long(metrics, "Posterior Contraction")
