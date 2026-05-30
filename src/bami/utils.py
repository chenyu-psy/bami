"""General user-facing utility helpers.

This module contains small helpers that are useful across workflows. The first
public helper converts posterior sample dictionaries into tidy dataframes for
analysis and reporting, while leaving workflow sampling methods dictionary-based
for model plumbing.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

POSTERIOR_DF_COLUMNS = [
    "dataset",
    "draw",
    "level",
    "param",
    "basis",
    "quantity",
    "value",
]

__all__ = ["posterior_to_dataframe"]


def posterior_to_dataframe(
    samples: Mapping[str, np.ndarray],
    priors: Mapping,
    level: str,
    include_raw: bool = False,
) -> pd.DataFrame:
    """Convert posterior sample dictionaries into a tidy dataframe.

    Use this when you want posterior draws in a table for plotting, summaries,
    or export. The input stays dictionary-based in workflow methods because
    that is the shape BayesFlow uses internally.

    Args:
        samples: Posterior sample dictionary returned by a ``sample_*``
            workflow method.
        priors: Workflow prior specification. Only stochastic parameters
            represented by dictionaries are included, so constants do not become
            posterior rows.
        level: Posterior level to convert. Use ``"simple"``, ``"group"``, or
            ``"subject"``. ``"random"`` is accepted as an alias for ``"subject"``.
        include_raw: Whether to include raw-space values such as
            ``theta_raw``, ``theta_mu_raw``, ``theta_log_sigma``,
            ``theta_subj_raw``, and standardized ``theta_z`` values.

    Returns:
        pandas.DataFrame: Tidy posterior table with columns ``dataset``,
            ``draw``, ``level``, ``param``, ``basis``, ``quantity``, and ``value``.

    Example:
        ```python
        samples = {"theta": np.array([[0.1, 0.2]])}
        priors = {"theta": {"mean": 0, "sd": 1, "link": "identity"}}
        posterior_to_dataframe(samples, priors, level="simple")
        ```
    """

    checked_level = _check_level(level)
    rows = []
    if checked_level == "simple":
        _add_simple_rows(rows, samples, priors, include_raw=include_raw)
    elif checked_level == "group":
        _add_group_rows(rows, samples, priors, include_raw=include_raw)
    else:
        _add_subject_rows(rows, samples, priors, include_raw=include_raw)
    return pd.DataFrame(rows, columns=POSTERIOR_DF_COLUMNS)


def _check_level(level: str) -> str:
    """Return a normalized posterior level name.

    Args:
        level:
            User-supplied level label.

    Returns:
        str: One of ``"simple"``, ``"group"``, or ``"subject"``.
    """

    if level == "random":
        return "subject"
    if level in {"simple", "group", "subject"}:
        return level
    raise ValueError("level must be 'simple', 'group', 'subject', or 'random'.")


def _posterior_params(priors: Mapping) -> list[str]:
    """Return stochastic parameter names from a workflow prior specification.

    Args:
        priors:
            Workflow prior specification.

    Returns:
        list[str]: Parameters represented by prior dictionaries.
    """

    return [name for name, spec in priors.items() if isinstance(spec, dict)]


def _as_dataset_draw_array(values, key: str) -> np.ndarray:
    """Return values with explicit dataset and draw axes.

    Args:
        values:
            Posterior sample values for one global parameter.
        key:
            Sample dictionary key used in error messages.

    Returns:
        numpy.ndarray: Array with shape ``(n_datasets, n_draws)``.
    """

    arr = np.asarray(values, dtype=float)
    while arr.ndim > 2 and arr.shape[-1] == 1:
        arr = np.squeeze(arr, axis=-1)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError(f"samples['{key}'] must have dataset and draw axes.")
    return arr


def _add_global_rows(
    rows: list[dict],
    *,
    arr: np.ndarray,
    level: str,
    param: str,
    quantity: str,
) -> None:
    """Append global posterior rows to a row list.

    Args:
        rows: Mutable row list being assembled for the output dataframe.
        arr: Posterior array with shape ``(n_datasets, n_draws)``.
        level: Posterior level label written to the tidy dataframe.
        param: Parameter label written to the tidy dataframe.
        quantity: Quantity label written to the tidy dataframe.

    Returns:
        None: Rows are appended in place.
    """

    for dataset_id in range(arr.shape[0]):
        for draw_id in range(arr.shape[1]):
            rows.append(
                {
                    "dataset": dataset_id,
                    "draw": draw_id,
                    "level": level,
                    "param": param,
                    "basis": "global",
                    "quantity": quantity,
                    "value": float(arr[dataset_id, draw_id]),
                }
            )


def _add_simple_rows(
    rows: list[dict],
    samples: Mapping[str, np.ndarray],
    priors: Mapping,
    *,
    include_raw: bool,
) -> None:
    """Append simple-workflow posterior rows.

    Args:
        rows:
            Mutable row list being assembled for the output dataframe.
        samples:
            Posterior sample dictionary from ``SimpleWorkflow.sample_posterior``.
        priors:
            Simple workflow priors.
        include_raw:
            Whether to include raw-space parameter rows.

    Returns:
        None: Rows are appended in place.
    """

    from bami.inference.priors import raw_key

    for param in _posterior_params(priors):
        if param in samples:
            arr = _as_dataset_draw_array(samples[param], param)
            _add_global_rows(
                rows,
                arr=arr,
                level="simple",
                param=param,
                quantity="value",
            )
        raw = raw_key(param)
        if include_raw and raw in samples:
            arr = _as_dataset_draw_array(samples[raw], raw)
            _add_global_rows(
                rows,
                arr=arr,
                level="simple",
                param=param,
                quantity="raw",
            )


def _add_group_rows(
    rows: list[dict],
    samples: Mapping[str, np.ndarray],
    priors: Mapping,
    *,
    include_raw: bool,
) -> None:
    """Append hierarchical group posterior rows.

    Args:
        rows:
            Mutable row list being assembled for the output dataframe.
        samples:
            Posterior sample dictionary from
            ``HierarchicalWorkflow.sample_group_posterior``.
        priors:
            Hierarchical workflow priors.
        include_raw:
            Whether to include raw group mean and log-sigma rows.

    Returns:
        None: Rows are appended in place.
    """

    from bami.inference.priors import log_sigma_key, mu_raw_key

    for param in _posterior_params(priors):
        public_keys = [(f"{param}_mu", "mu"), (f"{param}_sigma", "sigma")]
        for key, quantity in public_keys:
            if key in samples:
                arr = _as_dataset_draw_array(samples[key], key)
                _add_global_rows(
                    rows,
                    arr=arr,
                    level="group",
                    param=param,
                    quantity=quantity,
                )
        if include_raw:
            raw_keys = [
                (mu_raw_key(param), "mu_raw"),
                (log_sigma_key(param), "log_sigma"),
            ]
            for key, quantity in raw_keys:
                if key in samples:
                    arr = _as_dataset_draw_array(samples[key], key)
                    _add_global_rows(
                        rows,
                        arr=arr,
                        level="group",
                        param=param,
                        quantity=quantity,
                    )


def _as_subject_array(values, key: str) -> np.ndarray:
    """Return subject posterior values with dataset, draw, and subject axes.

    Args:
        values:
            Posterior sample values for one subject-level parameter.
        key:
            Sample dictionary key used in error messages.

    Returns:
        numpy.ndarray: Array with shape ``(n_datasets, n_draws, n_subjects)``.
    """

    arr = np.asarray(values, dtype=float)
    if arr.ndim != 3:
        raise ValueError(
            f"samples['{key}'] must have shape (n_datasets, n_draws, n_subjects)."
        )
    return arr


def _add_subject_rows(
    rows: list[dict],
    samples: Mapping[str, np.ndarray],
    priors: Mapping,
    *,
    include_raw: bool,
) -> None:
    """Append random-effect subject posterior rows.

    Args:
        rows:
            Mutable row list being assembled for the output dataframe.
        samples:
            Posterior sample dictionary from
            ``HierarchicalWorkflow.sample_random_posterior``.
        priors:
            Hierarchical workflow priors.
        include_raw:
            Whether to include raw subject and standardized z rows.

    Returns:
        None: Rows are appended in place.
    """

    for param in _posterior_params(priors):
        if param in samples:
            arr = _as_subject_array(samples[param], param)
            _add_subject_param_rows(
                rows,
                arr=arr,
                param=param,
                quantity="value",
            )
        if include_raw:
            raw_keys = [(f"{param}_subj_raw", "raw"), (f"{param}_z", "z")]
            for key, quantity in raw_keys:
                if key in samples:
                    arr = _as_subject_array(samples[key], key)
                    _add_subject_param_rows(
                        rows,
                        arr=arr,
                        param=param,
                        quantity=quantity,
                    )


def _add_subject_param_rows(
    rows: list[dict],
    *,
    arr: np.ndarray,
    param: str,
    quantity: str,
) -> None:
    """Append rows for one subject-level parameter array.

    Args:
        rows: Mutable row list being assembled for the output dataframe.
        arr: Posterior array with shape ``(n_datasets, n_draws, n_subjects)``.
        param: Parameter label written to the tidy dataframe.
        quantity: Quantity label written to the tidy dataframe.

    Returns:
        None: Rows are appended in place.
    """

    for dataset_id in range(arr.shape[0]):
        for draw_id in range(arr.shape[1]):
            for subject_id in range(arr.shape[2]):
                rows.append(
                    {
                        "dataset": dataset_id,
                        "draw": draw_id,
                        "level": "subject",
                        "param": param,
                        "basis": f"subject:{subject_id}",
                        "quantity": quantity,
                        "value": float(arr[dataset_id, draw_id, subject_id]),
                    }
                )
