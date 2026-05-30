"""Tests for workflow posterior sampling helpers."""

from pathlib import Path
import subprocess
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

from bami.evaluation import diagnostics
from bami.workflows import HierarchicalWorkflow, SimpleWorkflow
from bami.workflows._sampling import _sample_posterior
from bami.workflows import training

ROOT = Path(__file__).resolve().parents[1]


class DummyWorkflow:
    """Minimal workflow stub exposing BayesFlow-like APIs for tests."""

    def __init__(self):
        self.indexed_subject_recovery_aligned = True
        self.last_sample_kwargs = None
        self._test = {
            "a_mu": np.array([[0.1], [0.2]]),
            "a_sigma": np.array([[0.3], [0.4]]),
            "a_log_sigma": np.array([[-1.2], [-0.9]]),
            "a_subj_0": np.array([[0.11], [0.21]]),
            "a_subj_1": np.array([[0.09], [0.19]]),
            "c_subj_0": np.array([[0.6], [0.7]]),
            "c_subj_1": np.array([[0.8], [0.9]]),
        }

    def simulate(self, batch_size: int):
        """Return deterministic test data regardless of requested batch size."""

        return self._test

    def sample(self, *, num_samples: int, conditions, **kwargs):
        """Return deterministic posterior draws with sample axis in position 1."""

        self.last_sample_kwargs = dict(kwargs)
        out = {}
        for key, val in conditions.items():
            base = val.astype(float)
            stacked = np.stack([base + 0.01 * i for i in range(num_samples)], axis=1)
            out[key] = stacked
        return out


class DummyTransformWorkflow:
    """Tiny workflow that records sampling calls and transforms raw samples."""

    def __init__(self):
        self.last_sample_kwargs = None

    def sample(self, *, num_samples: int, conditions, **kwargs):
        """Return raw draws with the same sample axis used by BayesFlow."""

        self.last_sample_kwargs = {
            "num_samples": num_samples,
            "conditions": conditions,
            **kwargs,
        }
        raw = np.asarray(conditions["raw_theta"], dtype=float)
        return {"theta_raw": np.stack([raw + i for i in range(num_samples)], axis=1)}

    def transform_posterior_samples(self, samples: dict) -> dict:
        """Convert raw draws to a public-scale toy parameter."""

        return {"theta": samples["theta_raw"] + 10.0}


class DummyRandomWorkflow:
    """Minimal random workflow that returns z values from conditions."""

    def __init__(self):
        self.last_sample_kwargs = None
        self.last_conditions = None

    def sample(self, *, num_samples: int, conditions, **kwargs):
        """Return deterministic z draws for paired random-effect tests."""

        self.last_sample_kwargs = {"num_samples": num_samples, **kwargs}
        self.last_conditions = conditions
        return {"theta_z": conditions["theta_mu_raw"] + conditions["data"][:, 0, 0]}


class DummyTrainWorkflow:
    """Tiny workflow stub used to test training config storage."""

    def __init__(self):
        self.history = {"loss": [], "val_loss": []}


def test_internal_sample_posterior_forwards_sample_batch_size():
    """Internal sampling helper should map sample_batch_size to BayesFlow."""

    wf = DummyWorkflow()
    test = wf.simulate(2)

    _sample_posterior(wf, test, num_samples=3, sample_batch_size=8)

    assert wf.last_sample_kwargs["batch_size"] == 8


def test_internal_sample_posterior_keeps_default_batching():
    """Internal helper should preserve BayesFlow's default batching."""

    wf = DummyWorkflow()
    test = wf.simulate(2)

    _sample_posterior(wf, test, num_samples=3)

    assert "batch_size" not in wf.last_sample_kwargs


def test_simple_workflow_sample_posterior_delegates_and_transforms():
    """SimpleWorkflow should expose researcher-facing posterior sampling."""

    wf = DummyTransformWorkflow()
    model = SimpleWorkflow.__new__(SimpleWorkflow)
    model.workflow = wf
    test_data = {"raw_theta": np.array([0.5, 1.5])}

    samples = model.sample_posterior(
        test_data=test_data,
        num_samples=3,
        approximator_kwargs={"temperature": 0.8},
        sample_batch_size=4,
    )

    assert wf.last_sample_kwargs["num_samples"] == 3
    assert wf.last_sample_kwargs["conditions"] is test_data
    assert wf.last_sample_kwargs["temperature"] == 0.8
    assert wf.last_sample_kwargs["batch_size"] == 4
    np.testing.assert_allclose(samples["theta"][0, :], [10.5, 11.5, 12.5])


def test_simple_workflow_simulate_delegates_to_bayesflow_workflow():
    """SimpleWorkflow should expose simulation without workflow reach-through."""

    wf = DummyWorkflow()
    model = SimpleWorkflow.__new__(SimpleWorkflow)
    model.workflow = wf

    out = model.simulate(2)

    assert out is wf._test


def test_hierarchical_workflow_sample_group_posterior_is_explicit():
    """HierarchicalWorkflow should name group-level posterior sampling clearly."""

    wf = DummyTransformWorkflow()
    model = HierarchicalWorkflow.__new__(HierarchicalWorkflow)
    model.workflow = wf
    test_data = {"raw_theta": np.array([0.2])}

    samples = model.sample_group_posterior(test_data=test_data, num_samples=2)

    assert not hasattr(HierarchicalWorkflow, "sample_posterior")
    assert wf.last_sample_kwargs["num_samples"] == 2
    np.testing.assert_allclose(samples["theta"][0, :], [10.2, 11.2])


def test_hierarchical_workflow_simulate_delegates_to_bayesflow_workflow():
    """HierarchicalWorkflow should expose simulation without workflow reach-through."""

    wf = DummyWorkflow()
    model = HierarchicalWorkflow.__new__(HierarchicalWorkflow)
    model.workflow = wf

    out = model.simulate(2)

    assert out is wf._test


def test_simple_workflow_plot_parameter_recovery_returns_figure():
    """Simple model diagnostics should plot one panel per recovered parameter."""

    model = SimpleWorkflow.__new__(SimpleWorkflow)
    model.priors = {
        "theta": {"mean": 0.0, "sd": 1.0, "link": "identity"},
        "scale": {"mean": 0.0, "sd": 1.0, "link": "log"},
    }
    simulated = {
        "theta": np.array([0.0, 1.0, 2.0]),
        "scale": np.array([1.0, 2.0, 3.0]),
    }

    def fake_simulate(n_datasets):
        """Return deterministic simulated truth for plot diagnostics."""

        assert n_datasets == 3
        return simulated

    def fake_sample_posterior(**kwargs):
        """Return posterior draws with axis 1 as the sample dimension."""

        assert kwargs["num_samples"] == 2
        assert kwargs["sample_batch_size"] == 16
        theta = np.array([[0.0, 0.2], [1.0, 1.2], [2.0, 2.2]])
        scale = np.array([[1.1, 1.3], [2.1, 2.3], [3.1, 3.3]])
        return {"theta": theta, "scale": scale}

    model.simulate = fake_simulate
    model.sample_posterior = fake_sample_posterior

    fig = model.plot_parameter_recovery(n_datasets=3, num_samples=2)

    assert len(fig.axes) == 2
    assert "theta" in fig.axes[0].get_title()
    assert "corr=" in fig.axes[0].get_title()
    assert "CCC=" not in fig.axes[0].get_title()
    plt.close(fig)


def test_evaluation_diagnostics_plot_parameter_recovery_direct_call():
    """Diagnostics functions should live in evaluation and accept a workflow."""

    model = SimpleWorkflow.__new__(SimpleWorkflow)
    model.priors = {"theta": {"mean": 0.0, "sd": 1.0, "link": "identity"}}
    model.simulate = lambda n_datasets: {"theta": np.array([0.0, 1.0, 2.0])}
    model.sample_posterior = lambda **kwargs: {
        "theta": np.array([[0.0, 0.2], [1.0, 1.2], [2.0, 2.2]])
    }

    fig = diagnostics.plot_parameter_recovery(
        model,
        n_datasets=3,
        num_samples=2,
        params=None,
        metrics=["corr", "ccc", "rmse"],
        n_cols=3,
    )

    title = fig.axes[0].get_title()
    assert "corr=" in title
    assert "CCC=" in title
    assert "RMSE=" in title
    plt.close(fig)


def test_simple_workflow_plot_parameter_recovery_rejects_bad_metrics():
    """Parameter recovery should reject empty or unsupported metrics."""

    model = SimpleWorkflow.__new__(SimpleWorkflow)
    model.priors = {"theta": {"mean": 0.0, "sd": 1.0, "link": "identity"}}
    model.simulate = lambda n_datasets: {"theta": np.array([0.0, 1.0, 2.0])}
    model.sample_posterior = lambda **kwargs: {
        "theta": np.array([[0.0, 0.2], [1.0, 1.2], [2.0, 2.2]])
    }

    with pytest.raises(ValueError, match="unsupported"):
        model.plot_parameter_recovery(n_datasets=3, num_samples=2, metrics="bad")
    with pytest.raises(ValueError, match="at least one"):
        model.plot_parameter_recovery(n_datasets=3, num_samples=2, metrics=[])


def test_hierarchical_workflow_plot_population_recovery_limits_params():
    """Population recovery plots should honor requested group parameter keys."""

    model = HierarchicalWorkflow.__new__(HierarchicalWorkflow)
    model.priors = {
        "theta": {"mean": 0.0, "sd": 1.0, "link": "identity"},
        "scale": {"mean": 0.0, "sd": 1.0, "link": "log"},
    }
    simulated = {
        "theta_mu": np.array([0.0, 1.0, 2.0]),
        "theta_sigma": np.array([0.5, 0.6, 0.7]),
        "scale_mu": np.array([1.0, 2.0, 3.0]),
    }
    samples = {
        "theta_mu": np.array([[0.0, 0.2], [1.0, 1.2], [2.0, 2.2]]),
        "theta_sigma": np.array([[0.4, 0.6], [0.5, 0.7], [0.6, 0.8]]),
        "scale_mu": np.array([[1.0, 1.2], [2.0, 2.2], [3.0, 3.2]]),
    }
    model.simulate = lambda n_datasets: simulated
    model.sample_group_posterior = lambda **kwargs: samples

    fig = model.plot_population_recovery(
        n_datasets=3,
        num_samples=2,
        params=["theta_mu"],
    )

    assert len(fig.axes) == 1
    assert fig.axes[0].get_title().startswith("theta_mu")
    assert "corr=" in fig.axes[0].get_title()
    plt.close(fig)


def test_evaluation_diagnostics_plot_population_recovery_direct_call():
    """Population recovery plotting should be directly available in evaluation."""

    model = HierarchicalWorkflow.__new__(HierarchicalWorkflow)
    model.priors = {"theta": {"mean": 0.0, "sd": 1.0, "link": "identity"}}
    model.simulate = lambda n_datasets: {"theta_mu": np.array([0.0, 1.0, 2.0])}
    model.sample_group_posterior = lambda **kwargs: {
        "theta_mu": np.array([[0.0, 0.2], [1.0, 1.2], [2.0, 2.2]])
    }

    fig = diagnostics.plot_population_recovery(
        model,
        n_datasets=3,
        num_samples=2,
        params=["theta_mu"],
        metrics=["corr", "ccc", "rmse"],
        n_cols=3,
    )

    title = fig.axes[0].get_title()
    assert "corr=" in title
    assert "CCC=" in title
    assert "RMSE=" in title
    plt.close(fig)


def test_hierarchical_workflow_plot_population_recovery_rejects_bad_metrics():
    """Population recovery should reject empty or unsupported metrics."""

    model = HierarchicalWorkflow.__new__(HierarchicalWorkflow)
    model.priors = {"theta": {"mean": 0.0, "sd": 1.0, "link": "identity"}}
    model.simulate = lambda n_datasets: {"theta_mu": np.array([0.0, 1.0, 2.0])}
    model.sample_group_posterior = lambda **kwargs: {
        "theta_mu": np.array([[0.0, 0.2], [1.0, 1.2], [2.0, 2.2]])
    }

    with pytest.raises(ValueError, match="unsupported"):
        model.plot_population_recovery(n_datasets=3, num_samples=2, metrics="bad")
    with pytest.raises(ValueError, match="at least one"):
        model.plot_population_recovery(n_datasets=3, num_samples=2, metrics=[])


def test_hierarchical_workflow_plot_random_recovery_ignores_padded_truth():
    """Random recovery should plot one dataset-level metric per parameter."""

    model = HierarchicalWorkflow.__new__(HierarchicalWorkflow)
    model.keep_subject_truth = ["theta"]
    simulated = {
        "theta_subj": np.array([[0.1, 0.2, np.nan], [0.3, 0.4, np.nan]]),
        "data": np.zeros((2, 3, 1)),
    }
    random_samples = {
        "theta": np.array(
            [
                [[0.11, 0.21, np.nan], [0.12, 0.22, np.nan]],
                [[0.31, 0.41, np.nan], [0.32, 0.42, np.nan]],
            ]
        )
    }
    model.simulate = lambda n_datasets: simulated
    model.sample_group_posterior = lambda **kwargs: {"theta_mu_raw": np.zeros((2, 2))}
    model.sample_random_posterior = lambda **kwargs: random_samples

    fig = model.plot_random_recovery(n_datasets=2, num_samples=2)

    offsets = fig.axes[0].collections[0].get_offsets()
    assert offsets.shape[0] == 2
    assert fig.axes[0].get_ylabel() == "corr"
    plt.close(fig)


def test_evaluation_diagnostics_plot_random_recovery_direct_call():
    """Random recovery should facet when multiple metrics are requested."""

    model = HierarchicalWorkflow.__new__(HierarchicalWorkflow)
    model.keep_subject_truth = ["theta"]
    simulated = {
        "theta_subj": np.array([[0.1, 0.2], [0.3, 0.4]]),
        "data": np.zeros((2, 2, 1)),
    }
    model.simulate = lambda n_datasets: simulated
    model.sample_group_posterior = lambda **kwargs: {"theta_mu_raw": np.zeros((2, 2))}
    model.sample_random_posterior = lambda **kwargs: {
        "theta": np.array(
            [
                [[0.11, 0.21], [0.12, 0.22]],
                [[0.31, 0.41], [0.32, 0.42]],
            ]
        )
    }

    fig = diagnostics.plot_random_recovery(
        model,
        n_datasets=2,
        num_samples=2,
        params=None,
        metrics=["corr", "ccc", "rmse"],
        n_cols=3,
    )

    assert [ax.get_title() for ax in fig.axes] == ["corr", "ccc", "RMSE"]
    plt.close(fig)


def test_hierarchical_workflow_plot_random_recovery_rejects_bad_metrics():
    """Random recovery should reject empty or unsupported metric requests."""

    model = HierarchicalWorkflow.__new__(HierarchicalWorkflow)
    model.keep_subject_truth = ["theta"]
    model.simulate = lambda n_datasets: {
        "theta_subj": np.array([[0.1, 0.2]]),
        "data": np.zeros((1, 2, 1)),
    }
    model.sample_group_posterior = lambda **kwargs: {"theta_mu_raw": np.zeros((1, 2))}
    model.sample_random_posterior = lambda **kwargs: {
        "theta": np.array([[[0.11, 0.21], [0.12, 0.22]]])
    }

    with pytest.raises(ValueError, match="unsupported"):
        model.plot_random_recovery(n_datasets=1, num_samples=2, metrics="bad")
    with pytest.raises(ValueError, match="at least one"):
        model.plot_random_recovery(n_datasets=1, num_samples=2, metrics=[])


def test_hierarchical_workflow_plot_random_recovery_requires_subject_truth():
    """Random recovery needs saved subject truth from keep_subject_truth."""

    model = HierarchicalWorkflow.__new__(HierarchicalWorkflow)
    model.keep_subject_truth = []
    model.simulate = lambda n_datasets: {"data": np.zeros((1, 1, 1))}
    model.sample_group_posterior = lambda **kwargs: {"theta_mu_raw": np.zeros((1, 1))}
    model.sample_random_posterior = lambda **kwargs: {"theta": np.zeros((1, 1, 1))}

    with pytest.raises(ValueError, match="keep_subject_truth"):
        model.plot_random_recovery(n_datasets=1, num_samples=1)


def test_train_workflow_stores_effective_training_config():
    """Group workflow training should save reusable training settings."""

    class DummyTrainModel:
        """Minimal model exposing the training helper contract."""

        def __init__(self):
            self.workflow = DummyTrainWorkflow()

        def _resolve_validation_data(self, validation_data):
            """Return an empty validation dictionary for zero-epoch tests."""

            return {"data": np.zeros((int(validation_data), 1), dtype=np.float32)}

    model = DummyTrainModel()

    training.train_workflow(
        model,
        max_epochs=0,
        n_batch=11,
        batch_size=7,
        validation_data=3,
        patience=4,
        custom_flag=True,
    )

    assert model._train_config["n_batch"] == 11
    assert model._train_config["batch_size"] == 7
    assert model._train_config["validation_data"] == 3
    assert model._train_config["patience"] == 4
    assert model._train_config["fit_kwargs"] == {"custom_flag": True}


def test_train_random_workflow_inherits_saved_group_config(monkeypatch):
    """Random workflow training should inherit group config and local overrides."""

    captured = {}

    def fake_train_workflow(model, **kwargs):
        """Capture random-training arguments without fitting BayesFlow."""

        captured["workflow"] = model.workflow
        captured["kwargs"] = kwargs
        return {"trained": True}

    monkeypatch.setattr(training, "train_workflow", fake_train_workflow)
    model = HierarchicalWorkflow.__new__(HierarchicalWorkflow)
    model.random_workflow = DummyTrainWorkflow()
    model._train_config = training.make_train_config(
        max_epochs=9,
        initial_epochs=2,
        n_batch=13,
        batch_size=5,
        validation_data=4,
        patience=3,
        min_delta=0.2,
        workers=2,
        max_queue_size=6,
        torch_device="cpu",
        verbose=0,
        fit_kwargs={"keep_best": True},
    )

    out = model.train_random_workflow(
        batch_size=8,
        file="random.keras",
        overwrite=True,
        extra_arg="yes",
    )

    assert out == {"trained": True}
    assert captured["workflow"] is model.random_workflow
    assert captured["kwargs"]["max_epochs"] == 9
    assert captured["kwargs"]["n_batch"] == 13
    assert captured["kwargs"]["batch_size"] == 8
    assert captured["kwargs"]["file"] == "random.keras"
    assert captured["kwargs"]["overwrite"] is True
    assert captured["kwargs"]["keep_best"] is True
    assert captured["kwargs"]["extra_arg"] == "yes"
    assert model._random_train_config["batch_size"] == 8
    assert "file" not in model._random_train_config


def test_random_workflow_uses_standardized_z_variables():
    """Random-effect priors should encode subject effects as z deviations."""

    model = HierarchicalWorkflow.__new__(HierarchicalWorkflow)
    model.priors = {
        "theta": {"mean": 1.0, "sd": 0.5, "link": "identity"},
        "fixed": 2.0,
    }

    draw = model._draw_random_subject_prior(np.random.default_rng(2026))

    assert model._random_inference_variables() == ["theta_z"]
    assert "theta_subj_raw" not in draw
    expected_raw = (
        draw["theta_mu_raw"] + np.exp(draw["theta_log_sigma"]) * draw["theta_z"]
    )
    assert np.isclose(draw["theta"], expected_raw)


def test_sample_random_posterior_uses_paired_group_draws():
    """Random posterior should combine z draws with paired group samples."""

    model = HierarchicalWorkflow.__new__(HierarchicalWorkflow)
    model.priors = {"theta": {"mean": 0.0, "sd": 1.0, "link": "log"}}
    model.observation = "aggregate"
    model.input_format = None
    model.include_trial_feature = False
    model.include_mask = False
    model.data_width = 1
    model.random_workflow = DummyRandomWorkflow()
    observed = {"data": np.array([[1.0], [2.0]], dtype=np.float32)}
    group_samples = {
        "theta_mu_raw": np.array([[0.0, 0.5]], dtype=np.float32),
        "theta_log_sigma": np.log(np.array([[1.0, 2.0]], dtype=np.float32)),
    }

    out = model.sample_random_posterior(
        observed_data=observed,
        group_samples=group_samples,
        sample_batch_size=3,
    )

    assert model.random_workflow.last_sample_kwargs["num_samples"] == 1
    assert model.random_workflow.last_sample_kwargs["batch_size"] == 3
    assert out["theta"].shape == (1, 2, 2)
    expected_z = np.array([[[1.0, 2.0], [1.5, 2.5]]], dtype=np.float32)
    expected_raw = np.array([[[1.0, 2.0], [3.5, 5.5]]], dtype=np.float32)
    np.testing.assert_allclose(out["theta_z"], expected_z)
    np.testing.assert_allclose(out["theta_subj_raw"], expected_raw)
    np.testing.assert_allclose(out["theta"], np.exp(expected_raw))


def test_sample_random_posterior_uses_fixed_group_components():
    """Fixed group components should come from priors when absent from samples."""

    model = HierarchicalWorkflow.__new__(HierarchicalWorkflow)
    model.priors = {"theta": {"mean": "normal(0, 1)", "sd": 2.0, "link": "identity"}}
    model.observation = "aggregate"
    model.input_format = None
    model.include_trial_feature = False
    model.include_mask = False
    model.data_width = 1
    model.random_workflow = DummyRandomWorkflow()
    observed = {"data": np.array([[1.0]], dtype=np.float32)}
    group_samples = {
        "theta_mu_raw": np.array([[0.5, 1.0]], dtype=np.float32),
    }

    out = model.sample_random_posterior(
        observed_data=observed,
        group_samples=group_samples,
    )

    expected_z = np.array([[[1.5], [2.0]]], dtype=np.float32)
    expected_raw = np.array([[[3.5], [5.0]]], dtype=np.float32)
    np.testing.assert_allclose(out["theta_z"], expected_z)
    np.testing.assert_allclose(out["theta_subj_raw"], expected_raw)
    np.testing.assert_allclose(out["theta"], expected_raw)


def _build_trial_random_model(*, trial_design: str = "fixed") -> HierarchicalWorkflow:
    """Build a minimal trial-level random-effect workflow stub.

    Parameters
    ----------
    trial_design
        Either ``"fixed"`` or ``"flex"``.

    Returns
    -------
    HierarchicalWorkflow
        Partially initialized model with a dummy random workflow for sampling
        tests.
    """

    model = HierarchicalWorkflow.__new__(HierarchicalWorkflow)
    model.priors = {"theta": {"mean": 0.0, "sd": 1.0, "link": "identity"}}
    model.observation = "trial"
    model.input_format = None
    model.include_trial_feature = False
    model.include_mask = False
    model.data_width = 1
    model.random_workflow = DummyRandomWorkflow()
    model.trial_design = trial_design
    if trial_design == "fixed":
        model.n_trials = 3
        model.n_trials_range = None
        model.max_trials = 3
    else:
        model.n_trials = None
        model.n_trials_range = (2, 5)
        model.max_trials = 4
    return model


def _trial_group_samples(n_samples: int = 1) -> dict[str, np.ndarray]:
    """Return simple group posterior draws for trial-level random tests.

    Parameters
    ----------
    n_samples
        Number of paired group posterior draws.

    Returns
    -------
    dict[str, numpy.ndarray]
        Raw group posterior samples for one dataset.
    """

    return {
        "theta_mu_raw": np.zeros((1, n_samples), dtype=np.float32),
        "theta_log_sigma": np.zeros((1, n_samples), dtype=np.float32),
    }


def test_sample_random_posterior_fixed_trial_count_warns_on_mismatch():
    """Fixed trial random sampling should warn, not fail, on trial mismatch."""

    model = _build_trial_random_model(trial_design="fixed")
    observed = np.array([[1.0], [2.0]], dtype=np.float32)

    with pytest.warns(UserWarning, match="fixed n_trials=3"):
        out = model.sample_random_posterior(
            observed_data=observed,
            group_samples=_trial_group_samples(),
        )

    assert out["theta"].shape == (1, 1, 1)


def test_sample_random_posterior_fixed_trial_count_accepts_match():
    """Matching fixed trial data should not emit reliability warnings."""

    model = _build_trial_random_model(trial_design="fixed")
    observed = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = model.sample_random_posterior(
            observed_data=observed,
            group_samples=_trial_group_samples(),
        )

    assert caught == []
    assert out["theta"].shape == (1, 1, 1)


def test_sample_random_posterior_flex_trial_pads_raw_single_subject():
    """Flex trial sampling should pad raw subject trials and add a mask."""

    model = _build_trial_random_model(trial_design="flex")
    observed = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)

    out = model.sample_random_posterior(
        observed_data=observed,
        group_samples=_trial_group_samples(),
    )

    condition_data = model.random_workflow.last_conditions["data"]
    assert condition_data.shape == (1, 4, 2)
    np.testing.assert_allclose(condition_data[0, :3, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(condition_data[0, :3, 1], 1.0)
    np.testing.assert_allclose(condition_data[0, 3, :], 0.0)
    assert out["theta"].shape == (1, 1, 1)


def test_sample_random_posterior_flex_trial_pads_ragged_subjects():
    """Flex trial sampling should accept subjects with different trial counts."""

    model = _build_trial_random_model(trial_design="flex")
    observed = [
        np.array([[1.0], [2.0]], dtype=np.float32),
        np.array([[3.0], [4.0], [5.0]], dtype=np.float32),
    ]

    out = model.sample_random_posterior(
        observed_data=observed,
        group_samples=_trial_group_samples(),
    )

    condition_data = model.random_workflow.last_conditions["data"]
    assert condition_data.shape == (2, 4, 2)
    np.testing.assert_allclose(condition_data[0, :, 1], [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(condition_data[1, :, 1], [1.0, 1.0, 1.0, 0.0])
    assert out["theta"].shape == (1, 1, 2)


def test_sample_random_posterior_flex_trial_accepts_padded_masked_data():
    """Pre-padded flex trial data should pass through when masks are valid."""

    model = _build_trial_random_model(trial_design="flex")
    observed = np.zeros((2, 4, 2), dtype=np.float32)
    observed[0, :2, 0] = [1.0, 2.0]
    observed[0, :2, 1] = 1.0
    observed[1, :3, 0] = [3.0, 4.0, 5.0]
    observed[1, :3, 1] = 1.0

    out = model.sample_random_posterior(
        observed_data=observed,
        group_samples=_trial_group_samples(),
    )

    condition_data = model.random_workflow.last_conditions["data"]
    assert condition_data.shape == (2, 4, 2)
    np.testing.assert_allclose(condition_data, observed)
    assert out["theta"].shape == (1, 1, 2)


def test_sample_random_posterior_flex_trial_rejects_too_many_trials():
    """Flex trial data should stay within the model's trained trial range."""

    model = _build_trial_random_model(trial_design="flex")
    observed = np.ones((5, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="more trials"):
        model.sample_random_posterior(
            observed_data=observed,
            group_samples=_trial_group_samples(),
        )


def test_inference_package_init_does_not_eager_import_model_classes():
    """Inference init should not import model classes during helper imports."""

    inference_init = (ROOT / "src/bami/inference/__init__.py").read_text(
        encoding="utf-8"
    )

    forbidden_lines = [
        "from .fixed_simple import",
        "from .flex_simple import",
        "from .fixed_hierarchy import",
        "from .flex_hierarchy import",
    ]
    for line in forbidden_lines:
        assert line not in inference_init


def test_priors_import_does_not_initialize_bayesflow_backend():
    """Importing shared priors should not initialize BayesFlow/Torch."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import bami.inference.priors; print('ok')",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    combined = result.stdout + result.stderr

    assert "ok" in result.stdout
    assert "INFO:bayesflow" not in combined
    assert "Using backend 'torch'" not in combined
    assert "disable autograd" not in combined
