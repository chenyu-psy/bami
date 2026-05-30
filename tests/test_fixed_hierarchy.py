"""Tests for the fixed-design hierarchy model."""

from fixtures_model_specs import M3_SPEC, m3_activation
from bami.inference import transform_hierarchical_samples
from bami.simulators.m3 import simulate_m3_custom
from bami.workflows import HierarchicalWorkflow


def _build_fixed_hierarchy(**kwargs) -> HierarchicalWorkflow:
    """Build a small fixed M3 hierarchy with generic workflow settings.

    Parameters
    ----------
    **kwargs
        Fixed subject and trial settings for ``HierarchicalWorkflow``.

    Returns
    -------
    HierarchicalWorkflow
        Generic hierarchy workflow with M3 simulation settings.
    """

    keep_subject_truth = kwargs.pop("keep_subject_truth", ["a", "c", "ra", "rc"])
    return HierarchicalWorkflow(
        name=M3_SPEC["model_name"],
        priors=M3_SPEC["priors"],
        simulator=simulate_m3_custom,
        observation="aggregate",
        simulator_kwargs={
            "activation_fn": m3_activation,
            "n_options": M3_SPEC["n_options"],
            "rule": M3_SPEC["rule"],
        },
        data_width=len(M3_SPEC["activation_contract"]["order"]),
        keep_subject_truth=keep_subject_truth,
        transform_samples=transform_hierarchical_samples,
        **kwargs,
    )


def test_fixed_hierarchy_uses_exchangeable_group_level_workflow():
    """Fixed hierarchy should infer group parameters, not indexed subjects."""

    model = _build_fixed_hierarchy(
        n_subjects=2,
        n_trials=5,
    )
    adapter_text = str(model.workflow.adapter)

    assert "a_mu_raw" in adapter_text
    assert "a_log_sigma" not in adapter_text
    assert "a_subj_raw_0" not in adapter_text
    assert "a_subj_raw_1" not in adapter_text
    assert model.subject_id_mode == "exchangeable"
    assert model.workflow.indexed_subject_recovery_aligned is False
    assert "DeepSet" in type(model.summary_network).__name__


def test_fixed_hierarchy_simulates_subject_truth():
    """Simulated fixed datasets should keep subject truth outside inference."""

    model = _build_fixed_hierarchy(
        n_subjects=2,
        n_trials=5,
    )
    sim_data = model.workflow.simulate(3)

    assert sim_data["data"].shape == (3, 2, 5)
    assert sim_data["a_subj"].shape == (3, 2)
    assert "a_subj_0" not in sim_data


def test_hierarchy_keeps_all_subject_truth_by_default():
    """Default subject truth should include all stochastic hierarchy parameters."""

    model = HierarchicalWorkflow(
        name=M3_SPEC["model_name"],
        priors=M3_SPEC["priors"],
        simulator=simulate_m3_custom,
        observation="aggregate",
        simulator_kwargs={
            "activation_fn": m3_activation,
            "n_options": M3_SPEC["n_options"],
            "rule": M3_SPEC["rule"],
        },
        data_width=len(M3_SPEC["activation_contract"]["order"]),
        n_subjects=2,
        n_trials=5,
        transform_samples=transform_hierarchical_samples,
    )

    assert model.keep_subject_truth == ["a", "c", "ra", "rc"]


def test_hierarchy_can_disable_subject_truth():
    """An empty keep_subject_truth list should save no subject truth arrays."""

    model = _build_fixed_hierarchy(
        n_subjects=2,
        n_trials=5,
        keep_subject_truth=[],
    )
    sim_data = model.workflow.simulate(2)

    assert model.keep_subject_truth == []
    assert "a_subj" not in sim_data


def test_hierarchy_rejects_unknown_subject_truth_name():
    """Subject truth names should match stochastic hierarchy parameters."""

    try:
        _build_fixed_hierarchy(
            n_subjects=2,
            n_trials=5,
            keep_subject_truth=["missing"],
        )
    except ValueError as exc:
        assert "keep_subject_truth" in str(exc)
        assert "missing" in str(exc)
    else:
        raise AssertionError("Expected keep_subject_truth validation to fail.")
