"""Tests for BayesFlow metric table builders."""

from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from bami.evaluation.metrics.bayesflow import (
    _build_m3_posthoc_config,
    bf_calibration,
    bf_coverage,
    bf_flex_ind_recovery,
    bf_ind_recovery,
    bf_pop_recovery,
    bf_zscore,
    estimate_flex_individual_recovery,
    estimate_fixed_individual_recovery,
    estimate_population_recovery,
    sample_posterior,
    summarize_group_parameters,
    summarize_random_parameters,
)
from fixtures_model_specs import M3_SPEC, m3_activation
from bami.inference import transform_hierarchical_samples
from bami.inference.posthoc import M3PosthocEstimator
from bami.inference.priors import (
    log_sigma_key,
    mu_raw_key,
    raw_key,
    transform_simple_samples,
)
from bami.simulators.m3 import prop_m3, simulate_m3_custom
from bami.workflows import HierarchicalWorkflow, SimpleWorkflow
from bami.workflows import training

ROOT = Path(__file__).resolve().parents[1]


def _build_m3_hierarchy(
    *,
    n_subjects=None,
    n_subjects_range=None,
    n_trials=None,
    n_trials_range=None,
    normalize_counts: bool = False,
) -> HierarchicalWorkflow:
    """Build a generic M3 hierarchy for recovery tests.

    Parameters
    ----------
    n_subjects, n_subjects_range, n_trials, n_trials_range
        Fixed or flexible design settings.
    normalize_counts
        Whether to use response proportions and scaled trial counts.

    Returns
    -------
    HierarchicalWorkflow
        Generic workflow with M3 posthoc settings.
    """

    row_transform = prop_m3 if normalize_counts else None
    trial_feature_scale = None
    if normalize_counts and n_trials_range is not None:
        trial_feature_scale = n_trials_range[1] - 1
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
        n_subjects=n_subjects,
        n_subjects_range=n_subjects_range,
        n_trials=n_trials,
        n_trials_range=n_trials_range,
        include_trial_feature=n_trials_range is not None,
        keep_subject_truth=["a", "c", "ra", "rc"],
        raw_data_key="raw_counts" if normalize_counts else None,
        row_transform=row_transform,
        trial_feature_scale=trial_feature_scale,
        transform_samples=transform_hierarchical_samples,
        posthoc_estimator=M3PosthocEstimator,
        posthoc_kwargs={
            "n_options": M3_SPEC["n_options"],
            "rule": M3_SPEC["rule"],
            "priors": M3_SPEC["priors"],
            "const_params": {"b": M3_SPEC["priors"]["b"]},
            "hier_params": {
                name: M3_SPEC["priors"][name] for name in ["a", "c", "ra", "rc"]
            },
        },
        posthoc_kind="m3",
    )


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

    def compute_default_diagnostics(self, **kwargs):
        """Return a tiny diagnostics table with BayesFlow metric row names."""

        return pd.DataFrame(
            {
                "a": [0.1, 1.5, 0.2, 0.6],
                "c": [0.2, 1.1, 0.1, 0.4],
            },
            index=["NRMSE", "Log Gamma", "Calibration Error", "Posterior Contraction"],
        )


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


def test_population_recovery_schema_and_rows():
    """Population recovery should emit expected rows and columns."""

    wf = DummyWorkflow()
    df = bf_pop_recovery(workflow=wf, test_data=2, num_samples=3)

    assert set(["level", "param", "true_value", "est_value"]).issubset(df.columns)
    assert set(df["level"].unique().tolist()) == {"population"}
    assert set(df["param"].unique().tolist()) == {"a_mu", "a_sigma"}
    assert "a_log_sigma" not in set(df["param"].unique().tolist())
    assert len(df) == 4

    explicit_df = bf_pop_recovery(
        workflow=wf,
        test_data=2,
        num_samples=3,
        variable_keys=("a_mu", "a_sigma", "a_log_sigma"),
    )
    assert set(explicit_df["param"].unique().tolist()) == {"a_mu", "a_sigma"}


def test_sample_posterior_forwards_sample_batch_size():
    """Diagnostic sampling batch size should map to BayesFlow sample batch_size."""

    wf = DummyWorkflow()
    test = wf.simulate(2)

    sample_posterior(wf, test, num_samples=3, sample_batch_size=8)

    assert wf.last_sample_kwargs["batch_size"] == 8


def test_sample_posterior_keeps_default_batching():
    """Omitting sample_batch_size should preserve BayesFlow's default behavior."""

    wf = DummyWorkflow()
    test = wf.simulate(2)

    sample_posterior(wf, test, num_samples=3)

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


def test_summarize_group_parameters_keeps_dataset_ids_and_ess():
    """Group summaries should report one row per dataset and parameter."""

    samples = {
        "theta_mu": np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 6.0],
            ]
        )
    }
    ess = {"theta_mu": np.array([100.0, 80.0])}

    out = summarize_group_parameters(samples, ci=0.5, ess=ess)

    assert list(out["dataset_id"]) == [0, 1]
    assert list(out["param"]) == ["theta_mu", "theta_mu"]
    np.testing.assert_allclose(out["estimate"], [2.0, 4.0])
    np.testing.assert_allclose(out["sd"], [1.0, 2.0])
    np.testing.assert_allclose(out["ci_lower"], [1.5, 3.0])
    np.testing.assert_allclose(out["ci_upper"], [2.5, 5.0])
    np.testing.assert_allclose(out["ess"], [100.0, 80.0])


def test_summarize_random_parameters_keeps_dataset_and_subject_ids():
    """Random summaries should report one row per dataset, subject, and param."""

    samples = {
        "theta": np.array(
            [
                [[1.0, 10.0], [3.0, 12.0], [5.0, 14.0]],
                [[2.0, 20.0], [4.0, 22.0], [6.0, 24.0]],
            ]
        )
    }
    ess = {"theta": np.array([[90.0, 91.0], [70.0, 71.0]])}

    out = summarize_random_parameters(samples, ci=0.5, ess=ess)

    assert list(out["dataset_id"]) == [0, 0, 1, 1]
    assert list(out["subject_id"]) == [0, 1, 0, 1]
    np.testing.assert_allclose(out["estimate"], [3.0, 12.0, 4.0, 22.0])
    np.testing.assert_allclose(out["ess"], [90.0, 91.0, 70.0, 71.0])


def test_population_recovery_forwards_sample_batch_size():
    """Population recovery should expose memory-safe posterior batching."""

    wf = DummyWorkflow()

    bf_pop_recovery(
        workflow=wf,
        test_data=2,
        num_samples=3,
        sample_batch_size=4,
        show_progress=False,
    )

    assert wf.last_sample_kwargs["batch_size"] == 4


def test_individual_recovery_forwards_sample_batch_size():
    """Individual recovery should expose memory-safe posterior batching."""

    wf = DummyWorkflow()

    bf_ind_recovery(
        workflow=wf,
        test_data=2,
        num_samples=3,
        sample_batch_size=4,
        show_progress=False,
    )

    assert wf.last_sample_kwargs["batch_size"] == 4


def test_population_recovery_prints_progress_messages(capsys):
    """Population recovery should print visible progress messages when requested."""

    wf = DummyWorkflow()
    bf_pop_recovery(workflow=wf, test_data=2, num_samples=3, show_progress=True)

    captured = capsys.readouterr()
    assert "Population recovery: writing" in captured.out
    assert "Population recovery: done" in captured.out


def test_individual_recovery_uses_subject_keys_and_median():
    """Individual recovery should parse subject keys into param + subject_id."""

    wf = DummyWorkflow()
    df = bf_ind_recovery(
        workflow=wf,
        test_data=2,
        num_samples=5,
        base_params=("a", "c"),
    )

    assert set(df["level"].unique().tolist()) == {"individual"}
    assert set(df["param"].unique().tolist()) == {"a", "c"}
    assert set(df["subject_id"].unique().tolist()) == {0, 1}

    # Median offset for num_samples=5 and +0.01*i draws is +0.02
    first_row = df.iloc[0]
    assert np.isclose(first_row["est_value"] - first_row["true_value"], 0.02)


def test_population_helper_matches_public_recovery():
    """Shared population helper should match the public recovery wrapper."""

    wf = DummyWorkflow()
    test = wf.simulate(2)
    samples = sample_posterior(wf, test, num_samples=3)

    helper_df = estimate_population_recovery(
        test_data=test,
        samples=samples,
        show_progress=False,
    )
    public_df = bf_pop_recovery(
        workflow=wf,
        test_data=test,
        num_samples=3,
        show_progress=False,
    )

    pd.testing.assert_frame_equal(helper_df, public_df)


def test_fixed_individual_helper_matches_public_recovery():
    """Shared fixed-individual helper should match the public recovery wrapper."""

    wf = DummyWorkflow()
    test = wf.simulate(2)
    samples = sample_posterior(wf, test, num_samples=5)

    helper_df = estimate_fixed_individual_recovery(
        test_data=test,
        samples=samples,
        base_params=("a", "c"),
    )
    public_df = bf_ind_recovery(
        workflow=wf,
        test_data=test,
        num_samples=5,
        base_params=("a", "c"),
    )

    pd.testing.assert_frame_equal(helper_df, public_df)


def test_workflow_only_individual_recovery_requires_alignment_marker():
    """Legacy workflow-only recovery should fail without an explicit marker."""

    wf = DummyWorkflow()
    del wf.indexed_subject_recovery_aligned

    with pytest.raises(ValueError, match="workflow-only individual recovery is unsafe"):
        bf_ind_recovery(
            workflow=wf,
            test_data=wf.simulate(2),
            num_samples=5,
            base_params=("a",),
        )


def test_model_aware_fixed_hierarchy_recovery_uses_posthoc_samples():
    """Exchangeable fixed hierarchy should use posthoc subject recovery."""

    model = _build_m3_hierarchy(
        n_subjects=2,
        n_trials=5,
    )
    data = np.zeros((1, 2, 5), dtype=np.float32)
    data[0, 0, :5] = np.array([3, 1, 1, 0, 0], dtype=np.float32)
    data[0, 1, :5] = np.array([1, 2, 1, 0, 1], dtype=np.float32)
    test = {
        "data": data,
        "a_subj": np.array([[0.5, 0.7]], dtype=np.float32),
    }
    samples = {}
    for base_param in ["a", "c", "ra", "rc"]:
        samples[mu_raw_key(base_param)] = np.zeros((1, 8), dtype=np.float32)
        samples[log_sigma_key(base_param)] = np.full(
            (1, 8),
            np.log(0.2),
            dtype=np.float32,
        )

    df = bf_ind_recovery(
        model=model,
        test_data=test,
        samples=samples,
        n_candidates=12,
        min_ess=1,
        max_candidates=12,
        base_params=("a",),
        show_progress=False,
    )

    assert df.shape[0] == 2
    assert set(df["subject_id"]) == {0, 1}
    assert df["param"].unique().tolist() == ["a"]
    assert "ess" in df.columns


def test_model_aware_flex_hierarchy_recovery_uses_posthoc_samples():
    """Exchangeable flex hierarchy should use posthoc subject recovery."""

    model = _build_m3_hierarchy(
        n_subjects_range=(1, 2),
        n_trials_range=(5, 6),
    )
    data = np.zeros((1, 1, 7), dtype=np.float32)
    data[0, 0, :5] = np.array([3, 1, 1, 0, 0], dtype=np.float32)
    data[0, 0, 5] = 5
    data[0, 0, 6] = 1
    test = {"data": data}
    samples = {}
    for base_param in ["a", "c", "ra", "rc"]:
        test[f"{base_param}_subj"] = np.array([[0.5]], dtype=np.float32)
        samples[mu_raw_key(base_param)] = np.zeros((1, 8), dtype=np.float32)
        samples[log_sigma_key(base_param)] = np.full(
            (1, 8),
            np.log(0.2),
            dtype=np.float32,
        )

    df = bf_ind_recovery(
        model=model,
        test_data=test,
        samples=samples,
        n_candidates=12,
        min_ess=1,
        max_candidates=12,
        base_params=("a",),
        show_progress=False,
    )

    assert df.shape[0] == 1
    assert df.loc[0, "param"] == "a"
    assert "ess" in df.columns


def test_m3_posthoc_module_avoids_bayesflow_runtime_imports():
    """Worker-only posthoc code should not initialize training backends."""

    text = (ROOT / "src/bami/inference/posthoc.py").read_text(encoding="utf-8")
    forbidden_tokens = [
        "import bayesflow",
        "import torch",
        "scripts.model_spec",
        "m3_flex_hierarchy",
        "m3_fixed_hierarchy",
    ]
    for token in forbidden_tokens:
        assert token not in text


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


def test_posthoc_import_does_not_initialize_bayesflow_backend():
    """Importing worker-only posthoc code should not print BayesFlow backend logs."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import bami.inference.posthoc; print('ok')",
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


def test_m3_posthoc_config_is_plain_worker_data():
    """Process worker config should contain only pickle-friendly M3 settings."""

    model = _build_m3_hierarchy(
        n_subjects_range=(1, 2),
        n_trials_range=(5, 6),
    )
    config = _build_m3_posthoc_config(model)

    assert set(config) == {
        "n_options",
        "rule",
        "priors",
        "const_params",
        "hier_params",
    }
    assert "workflow" not in config
    assert "activations" not in config


def test_fixed_hierarchy_posthoc_can_run_with_process_safe_workers():
    """Fixed hierarchy posthoc should parallelize without loaded workflow objects."""

    model = _build_m3_hierarchy(
        n_subjects=2,
        n_trials=5,
    )
    data = np.zeros((2, 2, 5), dtype=np.float32)
    data[0, 0, :5] = np.array([3, 1, 1, 0, 0], dtype=np.float32)
    data[0, 1, :5] = np.array([1, 2, 1, 0, 1], dtype=np.float32)
    data[1, 0, :5] = np.array([2, 1, 1, 1, 0], dtype=np.float32)
    data[1, 1, :5] = np.array([1, 1, 2, 0, 1], dtype=np.float32)
    test = {
        "data": data,
        "a_subj": np.array([[0.5, 0.7], [0.6, 0.8]], dtype=np.float32),
    }
    samples = {}
    for base_param in ["a", "c", "ra", "rc"]:
        samples[mu_raw_key(base_param)] = np.zeros((2, 8), dtype=np.float32)
        samples[log_sigma_key(base_param)] = np.full(
            (2, 8),
            np.log(0.2),
            dtype=np.float32,
        )

    out = estimate_flex_individual_recovery(
        model=model,
        test_data=test,
        samples=samples,
        n_candidates=12,
        min_ess=1,
        max_candidates=12,
        base_params=("a",),
        show_progress=False,
        n_jobs=2,
    )

    assert out["dataset_id"].tolist() == [0, 0, 1, 1]
    assert out["subject_id"].tolist() == [0, 1, 0, 1]


def test_fixed_individual_recovery_ignores_raw_subject_keys():
    """Fixed individual recovery should not treat raw-space keys as subjects."""

    wf = DummyWorkflow()
    test = dict(wf.simulate(2))
    test["a_subj_raw_0"] = np.array([[0.01], [0.02]])
    samples = sample_posterior(wf, test, num_samples=5)

    df = estimate_fixed_individual_recovery(
        test_data=test,
        samples=samples,
        base_params=("a",),
    )

    assert "a_subj_raw" not in set(df["param"])
    assert set(df["subject_id"]) == {0, 1}


def test_diagnostic_metric_extractors_return_long_format():
    """Calibration, coverage, and z-score tables should be long and labeled."""

    wf = DummyWorkflow()

    calibration = bf_calibration(workflow=wf, test_data=2)
    coverage = bf_coverage(workflow=wf, test_data=2)
    zscore = bf_zscore(workflow=wf, test_data=2)

    assert set(calibration.columns) == {"param", "metric", "value", "model"}
    assert set(coverage.columns) == {"param", "metric", "value", "model"}
    assert set(zscore.columns) == {"param", "metric", "value", "model"}

    assert calibration["metric"].unique().tolist() == ["Log Gamma"]
    assert coverage["metric"].unique().tolist() == ["Calibration Error"]
    assert zscore["metric"].unique().tolist() == ["Posterior Contraction"]


def test_population_recovery_falls_back_to_simple_parameter_keys():
    """Population recovery should support simple models without _mu/_sigma keys."""

    class DummySimpleWorkflow(DummyWorkflow):
        def __init__(self):
            self._test = {
                "a": np.array([[0.1], [0.2]]),
                "c": np.array([[0.3], [0.4]]),
                "ra": np.array([[0.5], [0.6]]),
                "rc": np.array([[0.7], [0.8]]),
            }

    wf = DummySimpleWorkflow()
    df = bf_pop_recovery(workflow=wf, test_data=2, num_samples=3)
    assert set(df["param"].unique().tolist()) == {"a", "c", "ra", "rc"}


def test_population_recovery_uses_transformed_public_samples():
    """Population recovery should compare public truth to transformed raw draws."""

    priors = {
        "a": {"mean": 0.0, "sd": 1.0, "link": "log"},
        "c": {"mean": 0.0, "sd": 1.0, "link": "log"},
        "ra": {"mean": 0.0, "sd": 1.0, "link": "logit"},
        "rc": {"mean": 0.0, "sd": 1.0, "link": "logit"},
    }

    class DummyRawSimpleWorkflow:
        """Minimal simple workflow with raw posterior output."""

        def __init__(self):
            self._test = {
                "a": np.array([[1.0], [2.0]]),
                "c": np.array([[1.5], [2.5]]),
                "ra": np.array([[0.2], [0.3]]),
                "rc": np.array([[0.4], [0.5]]),
            }

        def simulate(self, batch_size: int):
            """Return public-scale truth for simple recovery."""

            return self._test

        def sample(self, *, num_samples: int, conditions, **kwargs):
            """Return raw posterior draws requiring public transformation."""

            return {
                raw_key("a"): np.zeros((2, num_samples, 1)),
                raw_key("c"): np.zeros((2, num_samples, 1)),
                raw_key("ra"): np.zeros((2, num_samples, 1)),
                raw_key("rc"): np.zeros((2, num_samples, 1)),
            }

        def transform_posterior_samples(self, samples: dict) -> dict:
            """Transform raw samples to public simple parameter keys."""

            return transform_simple_samples(samples, priors)

    wf = DummyRawSimpleWorkflow()
    df = bf_pop_recovery(workflow=wf, test_data=2, num_samples=4)

    assert set(df["param"].unique().tolist()) == {"a", "c", "ra", "rc"}
    assert np.all(df["est_value"] > 0)
    assert np.all(df.loc[df["param"].isin(["ra", "rc"]), "est_value"] < 1)


def test_flex_individual_recovery_keeps_posthoc_diagnostics():
    """Flex individual recovery should retain posthoc ESS diagnostics."""

    class DummyFlexWorkflow:
        """Tiny workflow with one dataset and one active subject."""

        def simulate(self, batch_size: int):
            """Return one padded flex dataset with subject-level truth."""

            data = np.zeros((1, 1, 7), dtype=np.float32)
            data[0, 0, :5] = np.array([4, 1, 1, 0, 1], dtype=np.float32)
            data[0, 0, 5] = 7.0
            data[0, 0, 6] = 1.0
            return {
                "data": data,
                "a_subj": np.array([[1.0]], dtype=np.float32),
            }

        def sample(self, *, num_samples: int, conditions, **kwargs):
            """Return placeholder group samples consumed by the dummy model."""

            return {"dummy_group": np.zeros((1, num_samples), dtype=np.float32)}

    class DummyPosthocEstimator:
        """Return one posthoc estimate with diagnostics."""

        def estimate_subjects(self, counts, **kwargs):
            """Return one posthoc estimate with diagnostics."""

            return pd.DataFrame(
                {
                    "subject_id": [0],
                    "param": ["a"],
                    "median": [1.2],
                    "lower": [0.9],
                    "upper": [1.5],
                    "ci": [0.95],
                    "ess": [42.0],
                    "ess_ratio": [0.21],
                    "max_weight": [0.08],
                    "n_candidates_used": [200],
                    "posthoc_status": ["low_ess"],
                }
            )

    class DummyFlexModel:
        """Minimal model exposing workflow and posthoc estimator settings."""

        def __init__(self):
            self.workflow = DummyFlexWorkflow()
            self.posthoc_estimator = DummyPosthocEstimator
            self.posthoc_kwargs = {}

    df = bf_flex_ind_recovery(
        model=DummyFlexModel(),
        test_data=1,
        num_group_samples=3,
        n_candidates=10,
        base_params=("a",),
    )

    assert df.loc[0, "est_value"] == 1.2
    assert df.loc[0, "ess"] == 42.0
    assert df.loc[0, "ess_ratio"] == 0.21
    assert df.loc[0, "max_weight"] == 0.08
    assert df.loc[0, "n_candidates_used"] == 200
    assert df.loc[0, "posthoc_status"] == "low_ess"


def test_flex_posthoc_helper_matches_public_recovery():
    """Shared flex helper should match the public posthoc recovery wrapper."""

    class DummyFlexWorkflow:
        """Tiny workflow with one dataset and one active subject."""

        def simulate(self, batch_size: int):
            """Return one padded flex dataset with subject-level truth."""

            data = np.zeros((1, 1, 7), dtype=np.float32)
            data[0, 0, :5] = np.array([4, 1, 1, 0, 1], dtype=np.float32)
            data[0, 0, 6] = 1.0
            return {
                "data": data,
                "a_subj": np.array([[1.0]], dtype=np.float32),
            }

        def sample(self, *, num_samples: int, conditions, **kwargs):
            """Return placeholder group samples consumed by the dummy model."""

            return {"dummy_group": np.zeros((1, num_samples), dtype=np.float32)}

    class DummyPosthocEstimator:
        """Return one posthoc estimate with diagnostics."""

        def estimate_subjects(self, counts, **kwargs):
            """Return one posthoc estimate with diagnostics."""

            return pd.DataFrame(
                {
                    "subject_id": [0],
                    "param": ["a"],
                    "median": [1.2],
                    "lower": [0.9],
                    "upper": [1.5],
                    "ci": [0.95],
                    "ess": [42.0],
                    "ess_ratio": [0.21],
                    "max_weight": [0.08],
                    "n_candidates_used": [200],
                    "posthoc_status": ["ok"],
                }
            )

    class DummyFlexModel:
        """Minimal model exposing workflow and posthoc estimator settings."""

        def __init__(self):
            self.workflow = DummyFlexWorkflow()
            self.posthoc_estimator = DummyPosthocEstimator
            self.posthoc_kwargs = {}

    model = DummyFlexModel()
    test = model.workflow.simulate(1)
    samples = sample_posterior(model.workflow, test, num_samples=3)

    helper_df = estimate_flex_individual_recovery(
        model=model,
        test_data=test,
        samples=samples,
        n_candidates=10,
        base_params=("a",),
        show_progress=False,
    )
    public_df = bf_flex_ind_recovery(
        model=model,
        test_data=test,
        num_group_samples=3,
        n_candidates=10,
        base_params=("a",),
        show_progress=False,
    )

    pd.testing.assert_frame_equal(helper_df, public_df)


def test_flex_posthoc_uses_raw_counts_when_data_is_normalized():
    """Posthoc recovery should accept normalized summary data plus raw counts."""

    class DummyPosthocEstimator:
        """Record counts and return one parameter estimate."""

        def __init__(self, parent):
            """Store the test model so counts can be inspected."""

            self.parent = parent

        def estimate_subjects(self, counts, **kwargs):
            """Record counts and return one parameter estimate."""

            self.parent.seen_counts = np.asarray(counts)
            return pd.DataFrame(
                {
                    "subject_id": [0],
                    "param": ["a"],
                    "median": [1.1],
                    "lower": [0.9],
                    "upper": [1.3],
                    "ci": [0.95],
                    "ess": [30.0],
                    "ess_ratio": [0.3],
                    "max_weight": [0.05],
                    "n_candidates_used": [100],
                    "posthoc_status": ["ok"],
                }
            )

    class DummyFlexModel:
        """Minimal model exposing posthoc estimator settings."""

        def __init__(self):
            self.seen_counts = None
            self.posthoc_estimator = DummyPosthocEstimator
            self.posthoc_kwargs = {"parent": self}

    data = np.zeros((1, 1, 7), dtype=np.float32)
    data[0, 0, :5] = np.array([0.4, 0.2, 0.2, 0.0, 0.2], dtype=np.float32)
    data[0, 0, 5] = 1.0
    data[0, 0, 6] = 1.0
    raw_counts = np.array([[[4, 2, 2, 0, 2]]], dtype=np.float32)
    test = {
        "data": data,
        "raw_counts": raw_counts,
        "a_subj": np.array([[1.0]], dtype=np.float32),
    }
    samples = {"dummy_group": np.zeros((1, 3), dtype=np.float32)}
    model = DummyFlexModel()

    out = estimate_flex_individual_recovery(
        model=model,
        test_data=test,
        samples=samples,
        n_candidates=10,
        base_params=("a",),
        show_progress=False,
    )

    assert np.array_equal(model.seen_counts, raw_counts[0])
    assert out.loc[0, "est_value"] == 1.1


def test_m3_flex_posthoc_process_workers_accept_normalized_data_with_raw_counts():
    """Real M3 process workers should use raw counts for normalized flex data."""

    model = _build_m3_hierarchy(
        n_subjects_range=(1, 2),
        n_trials_range=(5, 6),
        normalize_counts=True,
    )
    data = np.zeros((2, 1, 7), dtype=np.float32)
    data[:, 0, :5] = np.array([0.4, 0.2, 0.2, 0.0, 0.2], dtype=np.float32)
    data[:, 0, 5] = 1.0
    data[:, 0, 6] = 1.0
    raw_counts = np.array(
        [
            [[4, 2, 2, 0, 2]],
            [[8, 4, 4, 0, 4]],
        ],
        dtype=np.float32,
    )
    test = {
        "data": data,
        "raw_counts": raw_counts,
        "a_subj": np.array([[1.0], [2.0]], dtype=np.float32),
    }
    samples = {}
    for base_param in ["a", "c", "ra", "rc"]:
        samples[mu_raw_key(base_param)] = np.zeros((2, 8), dtype=np.float32)
        samples[log_sigma_key(base_param)] = np.full(
            (2, 8),
            np.log(0.2),
            dtype=np.float32,
        )

    out = estimate_flex_individual_recovery(
        model=model,
        test_data=test,
        samples=samples,
        n_candidates=12,
        min_ess=1,
        max_candidates=12,
        base_params=("a",),
        show_progress=False,
        n_jobs=2,
    )

    assert out["dataset_id"].tolist() == [0, 1]
    assert out["subject_id"].tolist() == [0, 0]
    assert out["param"].tolist() == ["a", "a"]


def test_flex_individual_recovery_prints_progress_messages(capsys):
    """Flex individual recovery should print visible progress messages."""

    class DummyFlexWorkflow:
        """Tiny workflow with one active subject for progress tests."""

        def simulate(self, batch_size: int):
            """Return one padded flex dataset with subject truth."""

            data = np.zeros((1, 1, 7), dtype=np.float32)
            data[0, 0, :5] = np.array([4, 1, 1, 0, 1], dtype=np.float32)
            data[0, 0, 6] = 1.0
            return {
                "data": data,
                "a_subj": np.array([[1.0]], dtype=np.float32),
            }

        def sample(self, *, num_samples: int, conditions, **kwargs):
            """Return placeholder group samples."""

            return {"dummy_group": np.zeros((1, num_samples), dtype=np.float32)}

    class DummyPosthocEstimator:
        """Return one posthoc estimate with diagnostics."""

        def estimate_subjects(self, counts, **kwargs):
            """Return one posthoc estimate with diagnostics."""

            return pd.DataFrame(
                {
                    "subject_id": [0],
                    "param": ["a"],
                    "median": [1.2],
                    "lower": [0.9],
                    "upper": [1.5],
                    "ci": [0.95],
                    "ess": [42.0],
                    "ess_ratio": [0.21],
                    "max_weight": [0.08],
                    "n_candidates_used": [200],
                    "posthoc_status": ["ok"],
                }
            )

    class DummyFlexModel:
        """Minimal flex model for progress tests."""

        def __init__(self):
            self.workflow = DummyFlexWorkflow()
            self.posthoc_estimator = DummyPosthocEstimator
            self.posthoc_kwargs = {}

    bf_flex_ind_recovery(
        model=DummyFlexModel(),
        test_data=1,
        num_group_samples=3,
        n_candidates=10,
        base_params=("a",),
        show_progress=True,
    )

    captured = capsys.readouterr()
    assert "Hierarchy individual recovery: estimating" in captured.out
    assert "Hierarchy individual recovery: done" in captured.out


def test_flex_individual_recovery_can_override_simulated_trial_count():
    """Flex recovery should temporarily override n_trials when simulating test data."""

    class TrialAwareWorkflow:
        """Workflow that checks the parent model's trial range during simulation."""

        def __init__(self, parent):
            self.parent = parent

        def simulate(self, batch_size: int):
            """Return one dataset and record that n_trials was overridden."""

            assert self.parent.n_trials_range == (25, 26)
            data = np.zeros((1, 1, 7), dtype=np.float32)
            data[0, 0, :5] = np.array([4, 1, 1, 0, 1], dtype=np.float32)
            data[0, 0, 5] = 25.0
            data[0, 0, 6] = 1.0
            return {
                "data": data,
                "a_subj": np.array([[1.0]], dtype=np.float32),
            }

        def sample(self, *, num_samples: int, conditions, **kwargs):
            """Return placeholder group samples."""

            return {"dummy_group": np.zeros((1, num_samples), dtype=np.float32)}

    class DummyPosthocEstimator:
        """Return one posthoc estimate with diagnostics."""

        def estimate_subjects(self, counts, **kwargs):
            """Return one posthoc estimate with diagnostics."""

            return pd.DataFrame(
                {
                    "subject_id": [0],
                    "param": ["a"],
                    "median": [1.2],
                    "lower": [0.9],
                    "upper": [1.5],
                    "ci": [0.95],
                    "ess": [42.0],
                    "ess_ratio": [0.21],
                    "max_weight": [0.08],
                    "n_candidates_used": [200],
                    "posthoc_status": ["ok"],
                }
            )

    class TrialAwareModel:
        """Minimal model with a mutable trial range."""

        def __init__(self):
            self.n_trials_range = (5, 9)
            self.workflow = TrialAwareWorkflow(self)
            self.posthoc_estimator = DummyPosthocEstimator
            self.posthoc_kwargs = {}

    model = TrialAwareModel()
    df = bf_flex_ind_recovery(
        model=model,
        test_data=1,
        num_group_samples=3,
        n_candidates=10,
        base_params=("a",),
        n_trials=25,
        show_progress=False,
    )

    assert model.n_trials_range == (5, 9)
    assert df.loc[0, "trial_count"] == 25
