"""Tests for public utility helpers."""

import numpy as np

from bami.utils import posterior_to_dataframe


def test_posterior_to_dataframe_converts_simple_public_samples():
    """Simple posterior samples should become public-scale tidy rows."""

    samples = {
        "theta_raw": np.array([[0.0, 1.0]], dtype=np.float32),
        "theta": np.array([[1.0, 2.0]], dtype=np.float32),
    }
    priors = {"theta": {"mean": 0.0, "sd": 1.0, "link": "identity"}}

    out = posterior_to_dataframe(samples, priors, level="simple")

    assert list(out.columns) == [
        "dataset",
        "draw",
        "level",
        "param",
        "basis",
        "quantity",
        "value",
    ]
    assert out.to_dict("records") == [
        {
            "dataset": 0,
            "draw": 0,
            "level": "simple",
            "param": "theta",
            "basis": "global",
            "quantity": "value",
            "value": 1.0,
        },
        {
            "dataset": 0,
            "draw": 1,
            "level": "simple",
            "param": "theta",
            "basis": "global",
            "quantity": "value",
            "value": 2.0,
        },
    ]


def test_posterior_to_dataframe_converts_group_public_samples():
    """Group posterior rows should use parameter and quantity columns."""

    samples = {
        "theta_mu_raw": np.array([[0.0, 1.0]], dtype=np.float32),
        "theta_log_sigma": np.array([[0.1, 0.2]], dtype=np.float32),
        "theta_mu": np.array([[1.0, 2.0]], dtype=np.float32),
        "theta_sigma": np.array([[3.0, 4.0]], dtype=np.float32),
    }
    priors = {"theta": {"mean": 0.0, "sd": 1.0, "link": "log"}}

    out = posterior_to_dataframe(samples, priors, level="group")

    assert set(out["quantity"]) == {"mu", "sigma"}
    assert set(out["param"]) == {"theta"}
    assert set(out["basis"]) == {"global"}
    assert len(out) == 4


def test_posterior_to_dataframe_converts_subject_samples():
    """Subject posterior rows should identify subject slots in basis."""

    samples = {
        "theta": np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
        "theta_subj_raw": np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32),
        "theta_z": np.array([[[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32),
    }
    priors = {"theta": {"mean": 0.0, "sd": 1.0, "link": "identity"}}

    out = posterior_to_dataframe(samples, priors, level="subject")

    assert set(out["level"]) == {"subject"}
    assert set(out["basis"]) == {"subject:0", "subject:1"}
    assert set(out["quantity"]) == {"value"}
    assert len(out) == 4


def test_posterior_to_dataframe_can_include_raw_rows():
    """Raw rows should be opt-in so default tables stay researcher-facing."""

    samples = {
        "theta_mu_raw": np.array([[0.0]], dtype=np.float32),
        "theta_log_sigma": np.array([[0.5]], dtype=np.float32),
        "theta_mu": np.array([[1.0]], dtype=np.float32),
        "theta_sigma": np.array([[2.0]], dtype=np.float32),
    }
    priors = {"theta": {"mean": 0.0, "sd": 1.0, "link": "identity"}}

    public_only = posterior_to_dataframe(samples, priors, level="group")
    with_raw = posterior_to_dataframe(
        samples,
        priors,
        level="group",
        include_raw=True,
    )

    assert set(public_only["quantity"]) == {"mu", "sigma"}
    assert set(with_raw["quantity"]) == {"mu", "sigma", "mu_raw", "log_sigma"}
