"""Tests for SDM workflows."""

import numpy as np
from fixtures_model_specs import SDM_SPEC
from bami.simulators.sdm import (
    simulate_sdm_simple,
)
from bami.workflows import (
    HierarchicalWorkflow,
    SimpleWorkflow,
)


def test_sdm_fixed_trial_workflow_outputs_trial_rows():
    """Fixed standard SDM workflow should use trials as the set dimension."""

    np.random.seed(2026)
    model = SimpleWorkflow(
        name=SDM_SPEC["model_name"],
        param_names=["c", "kappa"],
        priors=SDM_SPEC["priors"],
        simulator=simulate_sdm_simple,
        observation="trial",
        obs_names=SDM_SPEC["obs_names"],
        n_trials=25,
        summary_dim=4,
        n_coupling_layers=2,
    )

    sim = model.workflow.simulate(5)

    assert sim["data"].shape == (5, 25, 1)
    assert np.all(sim["data"] >= -np.pi)
    assert np.all(sim["data"] <= np.pi)


def test_sdm_flex_trial_workflow_pads_with_active_mask():
    """Flex standard SDM workflow should pad trials and append a mask."""

    np.random.seed(2026)
    model = SimpleWorkflow(
        name=SDM_SPEC["model_name"],
        param_names=["c", "kappa"],
        priors=SDM_SPEC["priors"],
        simulator=simulate_sdm_simple,
        observation="trial",
        obs_names=SDM_SPEC["obs_names"],
        n_trials=None,
        n_trials_range=(20, 31),
        summary_dim=4,
        n_coupling_layers=2,
    )

    sim = model.workflow.simulate(5)
    data = sim["data"]
    active_counts = data[:, :, 1].sum(axis=1)

    assert data.shape == (5, 30, 2)
    assert np.all(data[:, :, 0] >= -np.pi)
    assert np.all(data[:, :, 0] <= np.pi)
    assert np.all((data[:, :, 1] == 0) | (data[:, :, 1] == 1))
    assert np.all(active_counts >= 20)
    assert np.all(active_counts <= 30)


def test_sdm_fixed_hierarchy_outputs_nested_continuous_errors():
    """Fixed SDM hierarchy should keep subject and trial dimensions visible."""

    np.random.seed(2026)
    model = HierarchicalWorkflow(
        name=SDM_SPEC["model_name"],
        priors=SDM_SPEC["priors"],
        simulator=simulate_sdm_simple,
        observation="trial",
        obs_names=SDM_SPEC["obs_names"],
        n_subjects=4,
        n_trials=25,
        keep_subject_truth=["c", "kappa"],
        summary_dim=4,
        n_coupling_layers=2,
    )

    sim = model.workflow.simulate(3)

    assert sim["data"].shape == (3, 4, 25, 1)
    assert np.all(sim["data"] >= -np.pi)
    assert np.all(sim["data"] <= np.pi)
    assert sim["c_subj"].shape == (3, 4)
    assert sim["kappa_subj"].shape == (3, 4)


def test_sdm_flex_hierarchy_pads_nested_continuous_errors():
    """Flex SDM hierarchy should pad subjects and trials with trial masks."""

    np.random.seed(2026)
    model = HierarchicalWorkflow(
        name=SDM_SPEC["model_name"],
        priors=SDM_SPEC["priors"],
        simulator=simulate_sdm_simple,
        observation="trial",
        obs_names=SDM_SPEC["obs_names"],
        n_subjects_range=(2, 5),
        n_trials_range=(20, 31),
        keep_subject_truth=["c", "kappa"],
        summary_dim=4,
        n_coupling_layers=2,
    )

    sim = model.workflow.simulate(3)
    data = sim["data"]
    mask = data[:, :, :, 1]
    n_subjects = sim["n_subjects"].reshape(-1)

    assert data.shape == (3, 4, 30, 2)
    assert np.all(data[:, :, :, 0] >= -np.pi)
    assert np.all(data[:, :, :, 0] <= np.pi)
    assert np.all((mask == 0) | (mask == 1))
    assert np.all(n_subjects >= 2)
    assert np.all(n_subjects < 5)


def test_sdm_fixed_hierarchy_tiny_training_accepts_stage_metrics():
    """Tiny SDM hierarchy training should not fail on BayesFlow stage metrics."""

    np.random.seed(2026)
    model = HierarchicalWorkflow(
        name=SDM_SPEC["model_name"],
        priors=SDM_SPEC["priors"],
        simulator=simulate_sdm_simple,
        observation="trial",
        obs_names=SDM_SPEC["obs_names"],
        n_subjects=2,
        n_trials=3,
        summary_dim=4,
        n_coupling_layers=2,
    )

    history = model.train_workflow(
        max_epochs=1,
        initial_epochs=1,
        n_batch=1,
        batch_size=2,
        validation_data=2,
        patience=1,
        min_delta=0.0,
        workers=1,
        max_queue_size=1,
        verbose=0,
        file=None,
    )

    hist = history.history if hasattr(history, "history") else history
    assert "loss" in hist


def test_sdm_trial_random_estimator_accepts_fixed_and_flex_trials(tmp_path):
    """Random estimator should support fixed and flexible trial contracts."""

    for case_id, trial_kwargs in enumerate(
        [
            {"n_subjects": 2, "n_trials": 3},
            {"n_subjects_range": (2, 4), "n_trials_range": (2, 4)},
        ]
    ):
        model = HierarchicalWorkflow(
            name=SDM_SPEC["model_name"],
            priors=SDM_SPEC["priors"],
            simulator=simulate_sdm_simple,
            observation="trial",
            obs_names=SDM_SPEC["obs_names"],
            keep_subject_truth=["c", "kappa"],
            summary_dim=4,
            n_coupling_layers=2,
            **trial_kwargs,
        )
        model.train_random_estimator(
            file=tmp_path / f"sdm_estimator_{case_id}.pt",
            n_groups_per_sigma=2,
            subjects_per_group=1,
            sigma_values=(0.1,),
            max_epochs=1,
            batch_size=2,
            patience=1,
            show_progress=False,
            seed=2026,
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

        assert {"dataset_id", "subject_id", "c", "kappa"} <= set(estimates.columns)
        assert len(estimates) >= 2
        assert np.isfinite(estimates[["c", "kappa"]].to_numpy()).all()
