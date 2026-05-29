"""Internal posterior sampling helpers for workflow methods.

This module is intentionally private. User-facing code should call model-level
methods such as ``model.sample_posterior(...)`` or
``model.sample_group_posterior(...)`` instead of importing helpers from here.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _sample_posterior(
    workflow,
    test_data: Mapping[str, np.ndarray],
    num_samples: int,
    approximator_kwargs: Mapping | None = None,
    sample_batch_size: int | None = None,
) -> Mapping[str, np.ndarray]:
    """Sample posterior draws from a trained workflow.

    Parameters
    ----------
    workflow
        Trained workflow object exposing ``sample``.
    test_data
        Conditions passed to ``workflow.sample``.
    num_samples
        Number of posterior draws per dataset.
    approximator_kwargs
        Optional keyword arguments forwarded to ``workflow.sample``.
    sample_batch_size
        Optional number of datasets per posterior sampling batch. This is a
        memory-control option, not a training batch-size setting.

    Returns
    -------
    Mapping[str, numpy.ndarray]
        Posterior samples keyed by parameter name. If the workflow provides
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
