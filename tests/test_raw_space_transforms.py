"""Tests for raw-space posterior transformations.

The models let BayesFlow infer unconstrained raw variables. These tests check
that the shared transformations recover public M3 parameters on valid support.
"""

import numpy as np
import pytest

from fixtures_model_specs import M3_SPEC
from bami.inference.priors import (
    apply_link,
    draw_group_mean_raw,
    draw_group_sd_raw,
    invert_link,
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


def test_apply_link_supports_standard_positive_and_probability_links():
    """Standard links should map raw values onto their expected public support."""

    raw = np.array([-2.0, 0.0, 2.0, 20.0])

    softplus = apply_link(raw, "softplus")
    probit = apply_link(raw, "probit")
    cloglog = apply_link(raw, "cloglog")

    assert np.all(softplus > 0)
    assert softplus[-1] < np.exp(raw[-1])
    assert np.isclose(softplus[1], np.log(2))

    assert np.all(probit > 0)
    assert np.all(probit < 1)
    assert np.isclose(probit[1], 0.5)

    assert np.all(cloglog > 0)
    assert np.all(cloglog < 1)
    assert np.isclose(cloglog[1], 1 - np.exp(-1))


@pytest.mark.parametrize("link", ["softplus", "probit", "cloglog"])
def test_new_links_invert_back_to_raw_values(link):
    """New links should round-trip between raw and public parameter scales."""

    raw = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])

    public = apply_link(raw, link)
    back_to_raw = invert_link(public, link)

    np.testing.assert_allclose(back_to_raw, raw, atol=1e-8)


def test_simple_transform_uses_new_links_from_prior_specs():
    """Simple posterior transforms should use new link names from prior specs."""

    samples = {
        raw_key("scale"): np.array([[-1.0, 0.0, 1.0]]),
        raw_key("prob"): np.array([[-1.0, 0.0, 1.0]]),
        raw_key("hazard"): np.array([[-1.0, 0.0, 1.0]]),
    }
    priors = {
        "scale": {"mean": 0.0, "sd": 1.0, "link": "softplus"},
        "prob": {"mean": 0.0, "sd": 1.0, "link": "probit"},
        "hazard": {"mean": 0.0, "sd": 1.0, "link": "cloglog"},
    }

    out = transform_simple_samples(samples, priors)

    np.testing.assert_allclose(
        out["scale"], apply_link(samples["scale_raw"], "softplus")
    )
    np.testing.assert_allclose(out["prob"], apply_link(samples["prob_raw"], "probit"))
    np.testing.assert_allclose(
        out["hazard"], apply_link(samples["hazard_raw"], "cloglog")
    )


def test_hierarchical_transform_uses_new_links_from_prior_specs():
    """Hierarchy transforms should apply new links to group means and subjects."""

    samples = {
        mu_raw_key("scale"): np.array([[-1.0, 0.0, 1.0]]),
        subj_raw_key("scale", 0): np.array([[-0.5, 0.0, 0.5]]),
        mu_raw_key("prob"): np.array([[-1.0, 0.0, 1.0]]),
        subj_raw_key("prob", 0): np.array([[-0.5, 0.0, 0.5]]),
        mu_raw_key("hazard"): np.array([[-1.0, 0.0, 1.0]]),
        subj_raw_key("hazard", 0): np.array([[-0.5, 0.0, 0.5]]),
    }
    priors = {
        "scale": {"mean": 0.0, "sd": 1.0, "link": "softplus"},
        "prob": {"mean": 0.0, "sd": 1.0, "link": "probit"},
        "hazard": {"mean": 0.0, "sd": 1.0, "link": "cloglog"},
    }

    out = transform_hierarchical_samples(samples, priors)

    np.testing.assert_allclose(
        out["scale_mu"], apply_link(samples["scale_mu_raw"], "softplus")
    )
    np.testing.assert_allclose(
        out["scale_subj_0"], apply_link(samples["scale_subj_raw_0"], "softplus")
    )
    np.testing.assert_allclose(
        out["prob_mu"], apply_link(samples["prob_mu_raw"], "probit")
    )
    np.testing.assert_allclose(
        out["prob_subj_0"], apply_link(samples["prob_subj_raw_0"], "probit")
    )
    np.testing.assert_allclose(
        out["hazard_mu"], apply_link(samples["hazard_mu_raw"], "cloglog")
    )
    np.testing.assert_allclose(
        out["hazard_subj_0"],
        apply_link(samples["hazard_subj_raw_0"], "cloglog"),
    )


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
