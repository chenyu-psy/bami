"""Tests for raw-space posterior transformations.

The models let BayesFlow infer unconstrained raw variables. These tests check
that the shared transformations recover public M3 parameters on valid support.
"""

import numpy as np
import pytest

from fixtures_model_specs import M3_SPEC
from bami.inference.priors import (
    draw_group_mean_raw,
    draw_group_sd_raw,
    log_sigma_key,
    mu_raw_key,
    parse_distribution_expr,
    raw_key,
    subj_raw_key,
    transform_hierarchical_samples,
    transform_simple_samples,
    validate_positive_sd_spec,
)


def test_distribution_parser_accepts_compact_specs():
    """Distribution strings should parse into names and numeric arguments."""

    assert parse_distribution_expr("normal(0, 1)") == ("normal", [0.0, 1.0])
    assert parse_distribution_expr("gamma(2, 0.1)") == ("gamma", [2.0, 0.1])
    assert parse_distribution_expr("exponential(0.2)") == ("exponential", [0.2])


def test_group_prior_draws_mean_and_positive_sd():
    """New prior specs should draw raw group means and positive SD values."""

    rng = np.random.default_rng(2026)
    spec = {"mean": "uniform(-1, 1)", "sd": "beta(2, 8)", "link": "logit"}

    mean = draw_group_mean_raw("ra", spec, rng)
    sd = draw_group_sd_raw("ra", spec, rng)

    assert -1 <= mean <= 1
    assert 0 < sd < 1


def test_sd_validation_rejects_nonpositive_specs():
    """SD specs must be positive constants or positive-only distributions."""

    with pytest.raises(ValueError, match="sd must be positive"):
        validate_positive_sd_spec("a", 0)
    with pytest.raises(ValueError, match="0 < low < high"):
        validate_positive_sd_spec("a", "uniform(0, 1)")
    with pytest.raises(ValueError, match="must be one of"):
        validate_positive_sd_spec("a", "normal(0, 1)")


def test_simple_transform_returns_valid_public_support():
    """Simple raw samples should transform to valid public M3 parameters."""

    samples = {
        raw_key("a"): np.array([[-10.0, 0.0, 10.0]]),
        raw_key("c"): np.array([[-2.0, 0.0, 2.0]]),
        raw_key("ra"): np.array([[-20.0, 0.0, 20.0]]),
        raw_key("rc"): np.array([[-5.0, 0.0, 5.0]]),
    }

    out = transform_simple_samples(samples, M3_SPEC["priors"])

    assert np.all(out["a"] > 0)
    assert np.all(out["c"] > 0)
    assert np.all(out["ra"] > 0)
    assert np.all(out["ra"] < 1)
    assert np.all(out["rc"] > 0)
    assert np.all(out["rc"] < 1)


def test_hierarchical_transform_returns_valid_public_support():
    """Hierarchy raw samples should produce valid means, sigmas, and subjects."""

    samples = {}
    for base_param in ["a", "c", "ra", "rc"]:
        samples[mu_raw_key(base_param)] = np.array([[-5.0, 0.0, 5.0]])
        samples[log_sigma_key(base_param)] = np.array([[-20.0, 0.0, 20.0]])
        samples[subj_raw_key(base_param, 0)] = np.array([[-4.0, 0.0, 4.0]])

    out = transform_hierarchical_samples(samples, M3_SPEC["priors"])

    assert np.all(out["a_mu"] > 0)
    assert np.all(out["c_mu"] > 0)
    assert np.all(out["a_subj_0"] > 0)
    assert np.all(out["c_subj_0"] > 0)

    for base_param in ["a", "c", "ra", "rc"]:
        assert np.all(out[f"{base_param}_sigma"] > 0)

    for key in ["ra_mu", "rc_mu", "ra_subj_0", "rc_subj_0"]:
        assert np.all(out[key] > 0)
        assert np.all(out[key] < 1)
