"""Tests for the fixed-design hierarchy model."""

import numpy as np

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


def test_random_workflow_standardizes_data_conditions_and_z_targets():
    """Random workflow should standardize all random-stage BayesFlow inputs."""

    model = _build_fixed_hierarchy(
        n_subjects=2,
        n_trials=5,
    )

    model._build_random_workflow()
    standardizer = model.random_workflow.approximator.standardizer

    expected = [
        "inference_variables",
        "summary_variables",
        "inference_conditions",
    ]
    assert standardizer.standardize == expected
    assert list(standardizer.standardize_layers) == expected


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


def test_random_estimator_trains_and_estimates_public_parameters(tmp_path):
    """Route C estimator should return public point estimates by default."""

    model = _build_fixed_hierarchy(n_subjects=2, n_trials=5)
    file = tmp_path / "random_estimator.pt"

    out = model.train_random_estimator(
        file=file,
        n_groups_per_sigma=2,
        subjects_per_group=2,
        sigma_values=(0.1,),
        max_epochs=2,
        batch_size=4,
        patience=2,
        show_progress=False,
        seed=11,
    )
    sim = model.simulate(1)
    group_samples = {
        key: np.asarray(sim[key], dtype=np.float32)[:, np.newaxis]
        for key in model.param_names
    }

    estimates = model.estimate_random_parameter(
        observed_data=sim,
        group_samples=group_samples,
    )

    assert file.exists()
    assert out is None
    assert model.random_estimator is not None
    assert list(estimates.columns) == ["dataset_id", "subject_id", "a", "c", "ra", "rc"]
    assert estimates.shape[0] == 2
    assert np.isfinite(estimates[["a", "c", "ra", "rc"]].to_numpy()).all()


def test_random_estimator_auto_sigma_values_are_parameter_specific(tmp_path):
    """Default sigma bins should come from each parameter's group sigma prior."""

    model = _build_fixed_hierarchy(n_subjects=2, n_trials=5)

    model.train_random_estimator(
        file=tmp_path / "random_estimator.pt",
        n_groups_per_sigma=2,
        subjects_per_group=2,
        max_epochs=2,
        batch_size=4,
        patience=2,
        show_progress=False,
        seed=21,
    )

    resolved = model.random_estimator.metadata["config"]["resolved_sigma_values"]
    assert resolved["source"] == "auto"
    assert resolved["n_sigma_bins"] == 3
    assert set(resolved["sigma_by_param"]) == {"a", "c", "ra", "rc"}
    for values in resolved["sigma_by_param"].values():
        assert len(values) == 3
        assert np.isfinite(values).all()
        assert np.all(np.asarray(values) > 0.0)


def test_random_estimator_shared_sigma_values_apply_to_all_parameters(tmp_path):
    """A sequence sigma grid should preserve the legacy shared-sigma behavior."""

    model = _build_fixed_hierarchy(n_subjects=2, n_trials=5)

    model.train_random_estimator(
        file=tmp_path / "random_estimator.pt",
        n_groups_per_sigma=2,
        subjects_per_group=2,
        sigma_values=(0.1, 0.2),
        max_epochs=2,
        batch_size=4,
        patience=2,
        show_progress=False,
        seed=22,
    )

    resolved = model.random_estimator.metadata["config"]["resolved_sigma_values"]
    assert resolved["source"] == "shared"
    assert resolved["n_sigma_bins"] == 2
    for values in resolved["sigma_by_param"].values():
        assert values == [0.1, 0.2]


def test_random_estimator_dict_sigma_values_can_vary_by_parameter(tmp_path):
    """A dict sigma grid should allow manual parameter-specific bins."""

    model = _build_fixed_hierarchy(n_subjects=2, n_trials=5)
    sigma_values = {
        "a": (0.1, 0.2),
        "c": (0.2, 0.3),
        "ra": (0.3, 0.4),
        "rc": (0.4, 0.5),
    }

    model.train_random_estimator(
        file=tmp_path / "random_estimator.pt",
        n_groups_per_sigma=2,
        subjects_per_group=2,
        sigma_values=sigma_values,
        max_epochs=2,
        batch_size=4,
        patience=2,
        show_progress=False,
        seed=23,
    )

    resolved = model.random_estimator.metadata["config"]["resolved_sigma_values"]
    assert resolved["source"] == "dict"
    assert resolved["sigma_by_param"]["a"] == [0.1, 0.2]
    assert resolved["sigma_by_param"]["rc"] == [0.4, 0.5]


def test_random_estimator_rejects_invalid_sigma_values(tmp_path):
    """Sigma values should be positive and cover every hierarchical parameter."""

    cases = [
        ({"a": (0.1,), "c": (0.1,), "ra": (0.1,)}, "missing"),
        (
            {"a": (0.1,), "c": (0.1,), "ra": (0.1,), "rc": (0.1,), "bad": (0.1,)},
            "unknown",
        ),
        ({"a": (0.1,), "c": (0.1, 0.2), "ra": (0.1,), "rc": (0.1,)}, "same number"),
        ({"a": (0.1,), "c": (0.1,), "ra": (0.1,), "rc": (0.0,)}, "positive"),
        ((), "at least one"),
    ]

    for idx, (sigma_values, message) in enumerate(cases):
        model = _build_fixed_hierarchy(n_subjects=2, n_trials=5)
        try:
            model.train_random_estimator(
                file=tmp_path / f"random_estimator_{idx}.pt",
                n_groups_per_sigma=2,
                subjects_per_group=2,
                sigma_values=sigma_values,
                max_epochs=2,
                batch_size=4,
                patience=2,
                show_progress=False,
                seed=30 + idx,
            )
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError("Expected invalid sigma_values to fail.")


def test_random_estimator_show_progress_false_is_quiet(tmp_path, capsys):
    """Estimator progress bars should be optional for quiet scripted runs."""

    model = _build_fixed_hierarchy(n_subjects=2, n_trials=5)

    model.train_random_estimator(
        file=tmp_path / "random_estimator.pt",
        n_groups_per_sigma=2,
        subjects_per_group=2,
        sigma_values=(0.1,),
        max_epochs=2,
        batch_size=4,
        patience=2,
        show_progress=False,
        seed=41,
    )

    captured = capsys.readouterr()
    assert "random estimator" not in captured.out.lower()
    assert "random estimator" not in captured.err.lower()


def test_random_estimator_include_scales_adds_diagnostic_columns(tmp_path):
    """Diagnostic estimator output should expose raw, delta, z, and group scales."""

    model = _build_fixed_hierarchy(n_subjects=2, n_trials=5)
    model.train_random_estimator(
        file=tmp_path / "random_estimator.pt",
        n_groups_per_sigma=2,
        subjects_per_group=2,
        sigma_values=(0.1,),
        max_epochs=2,
        batch_size=4,
        patience=2,
        show_progress=False,
        seed=12,
    )
    sim = model.simulate(1)
    group_samples = {
        key: np.asarray(sim[key], dtype=np.float32)[:, np.newaxis]
        for key in model.param_names
    }

    estimates = model.estimate_random_parameter(
        observed_data=sim,
        group_samples=group_samples,
        include_scales=True,
    )

    for param in ["a", "c", "ra", "rc"]:
        assert param in estimates
        assert f"{param}_raw" in estimates
        assert f"{param}_delta_raw" in estimates
        assert f"{param}_z" in estimates
        assert f"{param}_mu_raw" in estimates
        assert f"{param}_log_sigma" in estimates


def test_random_estimator_loads_existing_pt_checkpoint(tmp_path):
    """Existing estimator checkpoints should load without recomputing auto sigma."""

    file = tmp_path / "random_estimator.pt"
    model = _build_fixed_hierarchy(n_subjects=2, n_trials=5)
    model.train_random_estimator(
        file=file,
        n_groups_per_sigma=2,
        subjects_per_group=2,
        max_epochs=2,
        batch_size=4,
        patience=2,
        show_progress=False,
        seed=13,
    )
    loaded_model = _build_fixed_hierarchy(n_subjects=2, n_trials=5)

    out = loaded_model.train_random_estimator(
        file=file,
        sigma_values={"not_a_parameter": (0.0,)},
    )

    assert out is None
    assert loaded_model.random_estimator is not None
    resolved = loaded_model.random_estimator.metadata["config"]["resolved_sigma_values"]
    assert resolved["source"] == "auto"


def test_random_estimator_rejects_keras_checkpoint_suffix(tmp_path):
    """Estimator checkpoints should not reuse the legacy Keras suffix."""

    model = _build_fixed_hierarchy(n_subjects=2, n_trials=5)

    try:
        model.train_random_estimator(file=tmp_path / "random_estimator.keras")
    except ValueError as exc:
        assert ".pt" in str(exc)
        assert ".keras" in str(exc)
    else:
        raise AssertionError("Expected .keras estimator checkpoint to fail.")
