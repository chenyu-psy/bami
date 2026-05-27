"""Tests for generic posthoc observation distributions."""

import numpy as np
from scipy.special import gammaln, i0

from bami.inference.distributions import (
    Bernoulli,
    Binomial,
    Gaussian,
    Joint,
    Multinomial,
    Normal,
    Poisson,
    VonMises,
)


def test_binomial_matches_reference_formula():
    """Binomial log_prob should match the closed-form likelihood."""

    dist = Binomial()
    p = np.array([0.2, 0.5, 0.8])
    got = dist.log_prob(0.7, n=10, p=p)
    k = 7
    expected = (
        gammaln(11)
        - gammaln(k + 1)
        - gammaln(10 - k + 1)
        + k * np.log(p)
        + (10 - k) * np.log1p(-p)
    )

    np.testing.assert_allclose(got, expected)


def test_multinomial_matches_reference_formula():
    """Multinomial log_prob should score one count vector per candidate."""

    dist = Multinomial()
    counts = np.array([2, 1, 3])
    p = np.array([[0.2, 0.3, 0.5], [0.4, 0.2, 0.4]])
    got = dist.log_prob(counts, p=p)
    log_const = gammaln(7) - np.sum(gammaln(counts + 1))
    expected = log_const + np.sum(counts * np.log(p), axis=1)

    np.testing.assert_allclose(got, expected)


def test_gaussian_matches_normal():
    """Gaussian is a researcher-facing alias with Normal behavior."""

    x = np.array([0.1, 0.2])
    mu = np.array([[0.0, 0.2], [0.2, 0.2]])
    sigma = 0.5

    np.testing.assert_allclose(
        Gaussian().log_prob(x, mu=mu, sigma=sigma),
        Normal().log_prob(x, mu=mu, sigma=sigma),
    )


def test_joint_sums_named_component_log_probs():
    """Joint should add log probabilities from named observations."""

    dist = Joint(correct=Binomial(), rating=Gaussian())
    obs = {"correct": 0.8, "rating": 1.0}
    got = dist.log_prob(
        obs,
        correct={"n": 10, "p": np.array([0.7, 0.8])},
        rating={"mu": np.array([0.9, 1.2]), "sigma": 0.5},
    )
    expected = Binomial().log_prob(0.8, n=10, p=np.array([0.7, 0.8]))
    expected = expected + Gaussian().log_prob(
        1.0,
        mu=np.array([0.9, 1.2]),
        sigma=0.5,
    )

    np.testing.assert_allclose(got, expected)


def test_scalar_distributions_return_candidate_vectors():
    """Scalar observation distributions should return one value per candidate."""

    assert Bernoulli().log_prob(1, p=np.array([0.2, 0.8])).shape == (2,)
    assert Poisson().log_prob(3, rate=np.array([2.0, 4.0])).shape == (2,)
    got = VonMises().log_prob(0.2, mu=np.array([0.1, 0.3]), kappa=2.0)
    expected = 2.0 * np.cos(0.2 - np.array([0.1, 0.3]))
    expected = expected - np.log(2.0 * np.pi) - np.log(i0(2.0))
    np.testing.assert_allclose(got, expected)
