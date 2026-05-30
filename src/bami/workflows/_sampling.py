"""Internal posterior sampling helpers for workflow methods.

This module is intentionally private. User-facing code should call model-level
methods such as ``model.sample_posterior(...)`` or
``model.sample_group_posterior(...)`` instead of importing helpers from here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _sample_posterior(
    workflow,
    test_data: Mapping[str, np.ndarray],
    num_samples: int,
    approximator_kwargs: Mapping | None = None,
    sample_batch_size: int | None = None,
) -> Mapping[str, np.ndarray]:
    """Sample posterior draws from a trained workflow.

    Args:
        workflow:
            Trained workflow object exposing ``sample``.
        test_data:
            Conditions passed to ``workflow.sample``.
        num_samples:
            Number of posterior draws per dataset.
        approximator_kwargs:
            Optional keyword arguments forwarded to ``workflow.sample``.
        sample_batch_size:
            Optional number of datasets per posterior sampling batch. This is a
            memory-control option, not a training batch-size setting.

    Returns:
        Mapping[str, numpy.ndarray]: Posterior samples keyed by parameter name. If the workflow provides
            ``transform_posterior_samples``, returned samples are transformed to
            the public parameter scale.
    """

    sample_kwargs = dict(approximator_kwargs or {})
    if sample_batch_size is not None:
        sample_kwargs["batch_size"] = int(sample_batch_size)

    samples = workflow.sample(
        num_samples=num_samples,
        conditions=test_data,
        **sample_kwargs,
    )
    transform = getattr(workflow, "transform_posterior_samples", None)
    if transform is not None:
        return transform(samples)
    return samples


def _sample_random_posterior(
    model,
    observed_data,
    group_samples: Mapping[str, np.ndarray],
    approximator_kwargs: Mapping | None = None,
    sample_batch_size: int | None = None,
) -> Mapping[str, np.ndarray]:
    """Sample subject-level random effects using BayesFlow ancestral sampling.

    Args:
        model:
            Hierarchical workflow that owns ``random_workflow`` and the
            model-specific data normalization helpers.
        observed_data:
            One or more observed subjects using the model's observation contract.
        group_samples:
            Group posterior samples returned by ``model.sample_group_posterior``.
        approximator_kwargs:
            Optional keyword arguments forwarded to BayesFlow's
            ``ancestral_sample`` method.
        sample_batch_size:
            Optional BayesFlow sampling batch size. This is a memory-control
            option, not a training batch-size setting.

    Returns:
        Mapping[str, numpy.ndarray]: Subject posterior samples with shape
            ``(n_datasets, n_samples, n_subjects)`` for each random-effect key.
    """

    if model.random_workflow is None:
        raise ValueError("Call train_random_workflow(...) before sampling.")
    observed_arr, active_counts = model._normalize_random_observed_data(observed_data)
    n_datasets, n_subject_slots = observed_arr.shape[:2]
    group_n_datasets, n_samples = model._group_sample_shape(group_samples)
    if group_n_datasets != n_datasets:
        raise ValueError(
            "observed_data and group_samples must have the same dataset count."
        )
    group_arrays = {
        key: model._group_condition_array(
            group_samples,
            key,
            n_datasets,
            n_samples,
        )
        for key in model._random_group_condition_keys()
    }

    sample_kwargs = dict(approximator_kwargs or {})
    if sample_batch_size is not None:
        sample_kwargs["batch_size"] = int(sample_batch_size)
    z_samples = model.random_workflow.ancestral_sample(
        conditions=_random_ancestral_conditions(model, observed_arr),
        ancestral_conditions=_bayesflow_ancestral_group_conditions(group_arrays),
        **sample_kwargs,
    )
    return _format_random_samples(
        model=model,
        z_samples=z_samples,
        group_arrays=group_arrays,
        n_datasets=n_datasets,
        n_samples=n_samples,
        n_subjects=n_subject_slots,
        active_counts=active_counts,
    )


def _random_ancestral_conditions(model, observed_arr: np.ndarray) -> dict:
    """Return child conditions for BayesFlow ancestral sampling.

    Args:
        model:
            Hierarchical workflow with an ``observation`` contract.
        observed_arr:
            Normalized observed subject data. Aggregate workflows use shape
            ``datasets x subjects x features``. Trial workflows use shape
            ``datasets x subjects x trials x features``.

    Returns:
        dict: BayesFlow child-condition dictionary. Aggregate subjects receive a
            singleton row axis because the random workflow was trained with one
            subject row per condition.
    """

    if model.observation == "aggregate":
        return {"data": observed_arr[:, :, np.newaxis, :]}
    return {"data": observed_arr}


def _bayesflow_ancestral_group_conditions(
    group_arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return group conditions with a scalar feature axis for BayesFlow.

    Args:
        group_arrays:
            Raw group posterior arrays with shape ``datasets x draws``.

    Returns:
        dict[str, numpy.ndarray]: Group condition arrays with shape
            ``datasets x draws x 1``. BayesFlow ancestral sampling concatenates
            these with learned summaries, so scalar group conditions need an
            explicit final feature dimension.
    """

    return {
        key: np.asarray(values, dtype=np.float32)[..., np.newaxis]
        for key, values in group_arrays.items()
    }


def _format_random_samples(
    *,
    model,
    z_samples: Mapping[str, np.ndarray],
    group_arrays: Mapping[str, np.ndarray],
    n_datasets: int,
    n_samples: int,
    n_subjects: int,
    active_counts: Sequence[int],
) -> dict[str, np.ndarray]:
    """Convert sampled ``z`` values into subject posterior samples.

    Args:
        model:
            Hierarchical workflow that defines priors and random-effect key names.
        z_samples:
            Random workflow ancestral output for all subject/group-draw pairs.
        group_arrays:
            Raw group condition arrays used for paired conversion.
        n_datasets:
            Number of observed datasets.
        n_samples:
            Number of paired group posterior draws.
        n_subjects:
            Number of padded subject slots.
        active_counts:
            Number of real subjects in each dataset. Padded subject slots are set
            to ``NaN`` in the returned arrays.

    Returns:
        dict[str, numpy.ndarray]: Public, raw, and standardized subject posterior arrays.
    """

    from bami.inference.priors import apply_link, log_sigma_key, mu_raw_key

    out = {}
    active_mask = _random_active_subject_mask(
        n_datasets=n_datasets,
        n_subjects=n_subjects,
        active_counts=active_counts,
    )
    for param_name, spec in model.priors.items():
        if not isinstance(spec, dict):
            continue
        z_key = model._random_z_key(param_name)
        z_arr = _reshape_ancestral_z_samples(
            z_samples[z_key],
            key=z_key,
            n_datasets=n_datasets,
            n_samples=n_samples,
            n_subjects=n_subjects,
        )
        z_arr = np.where(active_mask[:, np.newaxis, :], z_arr, np.nan)

        mu = group_arrays[mu_raw_key(param_name)]
        log_sigma = group_arrays[log_sigma_key(param_name)]
        raw_arr = mu[:, :, np.newaxis] + np.exp(log_sigma[:, :, np.newaxis]) * z_arr
        public_arr = apply_link(raw_arr, spec.get("link", "identity"))

        out[z_key] = z_arr
        out[f"{param_name}_subj_raw"] = raw_arr.astype(np.float32)
        out[param_name] = np.asarray(public_arr, dtype=np.float32)
    return out


def _random_active_subject_mask(
    *,
    n_datasets: int,
    n_subjects: int,
    active_counts: Sequence[int],
) -> np.ndarray:
    """Return a boolean mask for real subject slots.

    Args:
        n_datasets:
            Number of observed datasets.
        n_subjects:
            Number of padded subject slots.
        active_counts:
            Number of non-padded subjects in each dataset.

    Returns:
        numpy.ndarray: Boolean mask with shape ``(n_datasets, n_subjects)``.
    """

    mask = np.zeros((n_datasets, n_subjects), dtype=bool)
    for dataset_id, count in enumerate(active_counts):
        mask[dataset_id, : int(count)] = True
    return mask


def _reshape_ancestral_z_samples(
    values,
    *,
    key: str,
    n_datasets: int,
    n_samples: int,
    n_subjects: int,
) -> np.ndarray:
    """Return ancestral ``z`` samples in the public subject-sample shape.

    Args:
        values:
            BayesFlow ancestral output for one standardized random-effect key.
            Expected axes are datasets, subjects, and group posterior draws, with
            optional singleton sample or feature axes.
        key:
            Parameter key used in validation messages.
        n_datasets:
            Expected number of datasets.
        n_samples:
            Expected number of paired posterior draws.
        n_subjects:
            Expected number of subject slots.

    Returns:
        numpy.ndarray: Array shaped ``(n_datasets, n_samples, n_subjects)``.
    """

    arr = np.asarray(values, dtype=np.float32)
    while arr.ndim > 3 and arr.shape[-1] == 1:
        arr = np.squeeze(arr, axis=-1)
    if arr.shape != (n_datasets, n_subjects, n_samples):
        raise ValueError(
            f"z_samples['{key}'] must have shape "
            f"({n_datasets}, {n_subjects}, {n_samples}) after removing "
            "singleton axes."
        )
    return np.transpose(arr, (0, 2, 1)).astype(np.float32)
