"""Tests for the flexible-subject hierarchy model."""

import numpy as np

from fixtures_model_specs import M3_SPEC, m3_activation
from bami.inference import transform_hierarchical_samples
from bami.simulators.m3 import prop_m3, simulate_m3_custom
from bami.workflows import HierarchicalWorkflow


def _build_small_model(
    *,
    priors=None,
    normalize_counts: bool = False,
    n_subjects_range=(1, 5),
    n_trials_range=(5, 9),
) -> HierarchicalWorkflow:
    """Build a small flex model for fast unit tests.

    Parameters
    ----------
    priors
        Optional M3 prior specification.
    normalize_counts
        Whether to use response proportions and scaled trial counts.
    n_subjects_range, n_trials_range
        Flexible design ranges passed to ``HierarchicalWorkflow``.

    Returns
    -------
    HierarchicalWorkflow
        Model with small subject and trial ranges.
    """

    priors = M3_SPEC["priors"] if priors is None else priors
    row_transform = prop_m3 if normalize_counts else None
    return HierarchicalWorkflow(
        name=M3_SPEC["model_name"],
        priors=priors,
        simulator=simulate_m3_custom,
        observation="aggregate",
        simulator_kwargs={
            "activation_fn": m3_activation,
            "n_options": M3_SPEC["n_options"],
            "rule": M3_SPEC["rule"],
        },
        data_width=len(M3_SPEC["activation_contract"]["order"]),
        n_subjects_range=n_subjects_range,
        n_trials_range=n_trials_range,
        include_trial_feature=True,
        keep_subject_truth=["a", "c", "ra", "rc"],
        raw_data_key="raw_counts" if normalize_counts else None,
        row_transform=row_transform,
        trial_feature_scale=n_trials_range[1] - 1 if normalize_counts else None,
        transform_samples=transform_hierarchical_samples,
    )


def test_flex_simulator_uses_variable_subject_counts_and_trials():
    """Simulated datasets should vary in subject count and subject trial totals."""

    np.random.seed(2026)
    model = _build_small_model()
    sim = model.workflow.simulate(20)

    n_subjects = sim["n_subjects"].reshape(-1)
    assert set(np.unique(n_subjects)).issubset({1, 2, 3, 4})
    assert len(np.unique(n_subjects)) > 1

    data = sim["data"]
    for dataset_id, n_subj in enumerate(n_subjects):
        active = data[dataset_id, :, -1] > 0.5
        totals = data[dataset_id, active, 5]
        assert int(active.sum()) == int(n_subj)
        assert np.all(totals >= 5)
        assert np.all(totals < 9)


def test_flex_workflow_infers_group_variables_only():
    """Flex hierarchy should infer stochastic group keys, not fixed subject slots."""

    model = _build_small_model()
    adapter_text = str(model.workflow.adapter)

    assert "a_mu_raw" in adapter_text
    assert "a_log_sigma" not in adapter_text
    assert "_subj_" not in adapter_text
    assert model.subject_id_mode == "exchangeable"
    assert model.workflow.indexed_subject_recovery_aligned is False


def test_flex_normalized_input_keeps_shape_and_scales_counts():
    """Normalized diagnostic input should keep flex data shape stable.

    Returns
    -------
    None
        Asserts response counts become proportions and trial count is scaled.
    """

    model = _build_small_model(normalize_counts=True)
    counts = np.array([[2, 1, 1, 0, 1]], dtype=np.float32)

    data, subject_ids = model._prepare_observed_counts(counts)

    assert data.shape == (1, 4, 7)
    assert subject_ids == [0]
    assert np.allclose(data[0, 0, :5], counts[0] / counts.sum())
    assert np.isclose(data[0, 0, 5], counts.sum() / 8.0)
    assert data[0, 0, 6] == 1.0


def test_flex_normalized_simulator_keeps_raw_counts():
    """Normalized simulations should retain integer counts for diagnostics."""

    np.random.seed(2026)
    model = _build_small_model(
        normalize_counts=True,
        n_subjects_range=(1, 3),
        n_trials_range=(5, 9),
    )

    sim = model.workflow.simulate(3)

    assert "raw_counts" in sim
    assert sim["raw_counts"].shape == (3, 2, 5)
    assert np.allclose(sim["raw_counts"], np.round(sim["raw_counts"]))


def test_flex_workflow_infers_stochastic_sd_when_configured():
    """Distribution-valued SD should add log-sigma inference variables."""

    priors = {
        "a": {"mean": "normal(0, 1)", "sd": "gamma(2, 0.1)", "link": "log"},
        "c": {"mean": 0.0, "sd": 0.2, "link": "log"},
        "ra": {"mean": 0.0, "sd": 0.2, "link": "logit"},
        "rc": {"mean": 0.0, "sd": 0.2, "link": "logit"},
        "b": 0,
    }
    model = _build_small_model(priors=priors)
    adapter_text = str(model.workflow.adapter)

    assert "a_mu_raw" in adapter_text
    assert "a_log_sigma" in adapter_text
    assert "c_mu_raw" not in adapter_text
    assert "c_log_sigma" not in adapter_text
