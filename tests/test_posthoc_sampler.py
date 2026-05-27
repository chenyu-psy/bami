"""Tests for the generic PosthocSampler interface."""

import numpy as np

from bami.inference.distributions import Binomial
from bami.inference.posthoc import PosthocSampler
from bami.workflows import HierarchicalWorkflow


PRIORS = {
    "p": {"mean": "normal(0, 0.1)", "sd": 0.2, "link": "logit"},
}


def _sim_binary_summary(p: float, n_trials: int, rng=None):
    """Simulate one proportion-correct summary row.

    Parameters
    ----------
    p
        Public-scale success probability.
    n_trials
        Number of trials behind the summary.
    rng
        Optional NumPy random generator.

    Returns
    -------
    numpy.ndarray
        One-element row containing proportion correct.
    """

    rng = np.random if rng is None else rng
    return np.array([rng.binomial(n_trials, p) / n_trials], dtype=np.float32)


def _theta_to_binomial(theta, context):
    """Map candidate probabilities to binomial likelihood parameters."""

    return {"n": context["n_trials"], "p": theta["p"]}


def _build_model() -> HierarchicalWorkflow:
    """Build a tiny hierarchy with named observation columns."""

    return HierarchicalWorkflow(
        name="toy",
        priors=PRIORS,
        simulator=_sim_binary_summary,
        observation="aggregate",
        obs_names=["pc"],
        n_subjects=2,
        n_trials=20,
        summary_dim=4,
        n_coupling_layers=2,
    )


def test_workflow_accepts_obs_names_without_data_width():
    """obs_names should define data_width for researcher-facing setup."""

    model = _build_model()

    assert model.obs_names == ["pc"]
    assert model.data_width == 1
    assert model.workflow.obs_names == ["pc"]


def test_posthoc_sampler_returns_mean_median_and_distribution():
    """Sampler should weight candidates and return requested outputs."""

    model = _build_model()
    data = {
        "data": np.array(
            [
                [[0.8], [0.2]],
            ],
            dtype=np.float32,
        )
    }
    group_samples = {
        "p_mu_raw": np.array([[0.0, 0.0, 0.0, 0.0]], dtype=float),
        "p_log_sigma": np.log(np.array([[1.0, 1.0, 1.0, 1.0]], dtype=float)),
    }
    sampler = PosthocSampler.from_model(
        model,
        observation=Binomial(),
        theta_to_params=_theta_to_binomial,
    )

    result = sampler.sample_subjects(
        data=data,
        group_samples=group_samples,
        n_candidates=300,
        statistic=["mean", "median", "distribution"],
        n_draws=25,
        random_seed=2026,
    )

    assert set(result.summary["statistic"]) == {"mean", "median"}
    assert result.summary.shape[0] == 4
    assert result.diagnostics.shape[0] == 2
    assert {"ess", "max_weight", "n_candidates"}.issubset(result.diagnostics.columns)
    assert result.draws is not None
    assert result.draws.shape[0] == 50
    high_subject = result.summary[
        (result.summary["subject_id"] == 0) & (result.summary["statistic"] == "mean")
    ]["est_value"].iloc[0]
    low_subject = result.summary[
        (result.summary["subject_id"] == 1) & (result.summary["statistic"] == "mean")
    ]["est_value"].iloc[0]
    assert high_subject > low_subject
