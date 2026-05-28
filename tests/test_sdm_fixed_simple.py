"""Tests for SDM workflows."""

import numpy as np
from fixtures_model_specs import SDM_SPEC
from bami.inference import transform_hierarchical_samples
from bami.simulators.circular import (
    GRID_SIZE,
    degree_grid,
    errors_to_indices,
    indices_to_errors,
)
from bami.simulators.sdm import (
    simulate_sdm_simple,
)
from bami.training import fit_workflow
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
        simulator_kwargs={"grid_size": GRID_SIZE, "error_scale": 180.0},
        obs_names=SDM_SPEC["trial_contract"]["order"],
        n_trials=25,
        summary_dim=4,
        n_coupling_layers=2,
    )

    sim = model.workflow.simulate(5)

    assert sim["data"].shape == (5, 25, 1)
    assert np.all(sim["data"] >= -1)
    assert np.all(sim["data"] < 1)


def test_sdm_flex_trial_workflow_pads_with_active_mask():
    """Flex standard SDM workflow should pad trials and append a mask."""

    np.random.seed(2026)
    model = SimpleWorkflow(
        name=SDM_SPEC["model_name"],
        param_names=["c", "kappa"],
        priors=SDM_SPEC["priors"],
        simulator=simulate_sdm_simple,
        observation="trial",
        simulator_kwargs={"grid_size": GRID_SIZE, "error_scale": 180.0},
        obs_names=SDM_SPEC["trial_contract"]["order"],
        n_trials=None,
        n_trials_range=(20, 31),
        summary_dim=4,
        n_coupling_layers=2,
    )

    sim = model.workflow.simulate(5)
    data = sim["data"]
    active_counts = data[:, :, 1].sum(axis=1)

    assert data.shape == (5, 30, 2)
    assert np.all(data[:, :, 0] >= -1)
    assert np.all(data[:, :, 0] < 1)
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
        simulator_kwargs={"grid_size": GRID_SIZE, "error_scale": 180.0},
        obs_names=SDM_SPEC["trial_contract"]["order"],
        n_subjects=4,
        n_trials=25,
        keep_subject_truth=["c", "kappa"],
        summary_dim=4,
        n_coupling_layers=2,
        transform_samples=transform_hierarchical_samples,
    )

    sim = model.workflow.simulate(3)

    assert sim["data"].shape == (3, 4, 25, 1)
    assert np.all(sim["data"] >= -1)
    assert np.all(sim["data"] < 1)
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
        simulator_kwargs={"grid_size": GRID_SIZE, "error_scale": 180.0},
        obs_names=SDM_SPEC["trial_contract"]["order"],
        n_subjects_range=(2, 5),
        n_trials_range=(20, 31),
        keep_subject_truth=["c", "kappa"],
        summary_dim=4,
        n_coupling_layers=2,
        transform_samples=transform_hierarchical_samples,
    )

    sim = model.workflow.simulate(3)
    data = sim["data"]
    mask = data[:, :, :, 1]
    n_subjects = sim["n_subjects"].reshape(-1)

    assert data.shape == (3, 4, 30, 2)
    assert np.all(data[:, :, :, 0] >= -1)
    assert np.all(data[:, :, :, 0] < 1)
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
        simulator_kwargs={"grid_size": GRID_SIZE, "error_scale": 180.0},
        obs_names=SDM_SPEC["trial_contract"]["order"],
        n_subjects=2,
        n_trials=3,
        summary_dim=4,
        n_coupling_layers=2,
        transform_samples=transform_hierarchical_samples,
    )

    history = fit_workflow(
        model,
        max_epochs=1,
        initial_epochs=1,
        n_batch=1,
        batch_size=2,
        validation_data=2,
        patience=1,
        min_delta=0.0,
        workers=1,
        max_queue_size=1,
        torch_device="cpu",
        verbose=0,
        file=None,
    )

    hist = history.history if hasattr(history, "history") else history
    assert "loss" in hist


def test_sdm_degree_index_convention_matches_demo():
    """Degree labels and grid indices should follow the demo convention."""

    errors = np.array([0, 1, 180, -179, -1, -180])
    indices = errors_to_indices(errors)
    round_trip = indices_to_errors(indices)
    grid = degree_grid()

    assert np.array_equal(indices, np.array([0, 1, 180, 181, 359, 180]))
    assert np.array_equal(round_trip, np.array([0, 1, 180, -179, -1, 180]))
    assert grid[0] == 0
    assert grid[180] == 180
    assert grid[181] == -179
    assert grid[359] == -1
