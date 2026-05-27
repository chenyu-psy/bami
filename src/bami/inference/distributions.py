"""Analytic observation distributions for posthoc candidate weighting.

Each distribution exposes ``log_prob`` and is vectorized over candidate
parameters. These small classes let model-specific posthoc code describe the
observation model by choosing a distribution instead of rewriting likelihood
math for every model.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.special import gammaln, i0e, logsumexp


LOG_2PI = float(np.log(2.0 * np.pi))


class Bernoulli:
    """Bernoulli likelihood for binary observations.

    Parameters
    ----------
    eps
        Small clipping value used to avoid ``log(0)``.

    Returns
    -------
    None
        The initialized distribution exposes ``log_prob``.
    """

    def __init__(self, eps: float = 1e-12):
        self.eps = float(eps)

    def log_prob(self, x, *, p) -> np.ndarray:
        """Return candidate log probabilities for binary observations.

        Parameters
        ----------
        x
            Binary observation or array of binary observations.
        p
            Candidate success probabilities. The first dimension indexes
            candidates.

        Returns
        -------
        numpy.ndarray
            One log probability per candidate.
        """

        x_arr = np.asarray(x, dtype=float)
        p_arr = _clip_prob(p, self.eps)
        logp = x_arr * np.log(p_arr) + (1.0 - x_arr) * np.log1p(-p_arr)
        return _sum_non_candidate_axes(logp)


class Binomial:
    """Binomial likelihood for correct counts out of ``n`` trials."""

    def __init__(self, eps: float = 1e-12):
        """Create a binomial distribution.

        Parameters
        ----------
        eps
            Small clipping value used to avoid ``log(0)``.

        Returns
        -------
        None
            The initialized distribution exposes ``log_prob``.
        """

        self.eps = float(eps)

    def log_prob(self, x=None, *, n, p, k=None) -> np.ndarray:
        """Return candidate log probabilities for binomial observations.

        Parameters
        ----------
        x
            Proportion correct or count correct. Values between 0 and 1 are
            treated as proportions and multiplied by ``n``.
        n
            Trial count.
        p
            Candidate success probabilities.
        k
            Optional count correct. When supplied, ``x`` is ignored.

        Returns
        -------
        numpy.ndarray
            One log probability per candidate.
        """

        n_arr = np.asarray(n, dtype=float)
        p_arr = _clip_prob(p, self.eps)
        if k is None:
            x_arr = np.asarray(x, dtype=float)
            if np.all((0.0 <= x_arr) & (x_arr <= 1.0)):
                k_arr = np.rint(x_arr * n_arr)
            else:
                k_arr = x_arr
        else:
            k_arr = np.asarray(k, dtype=float)
        log_const = gammaln(n_arr + 1.0) - gammaln(k_arr + 1.0) - gammaln(
            n_arr - k_arr + 1.0
        )
        return log_const + k_arr * np.log(p_arr) + (n_arr - k_arr) * np.log1p(-p_arr)


class Categorical:
    """Categorical likelihood for one or more K-way choices."""

    def __init__(self, eps: float = 1e-12):
        """Create a categorical distribution."""

        self.eps = float(eps)

    def log_prob(self, x, *, p) -> np.ndarray:
        """Return candidate log probabilities for categorical observations.

        Parameters
        ----------
        x
            Integer category index or vector of indices.
        p
            Candidate category probabilities with shape ``(n_candidates, K)``
            or ``(n_candidates, n_trials, K)``.

        Returns
        -------
        numpy.ndarray
            One log probability per candidate.
        """

        x_arr = np.asarray(x, dtype=int)
        p_arr = _clip_prob(p, self.eps)
        if x_arr.ndim == 0:
            return np.log(p_arr[:, int(x_arr)])
        trial_idx = np.arange(x_arr.shape[0])
        return np.sum(np.log(p_arr[:, trial_idx, x_arr]), axis=1)


class Multinomial:
    """Multinomial likelihood for K-category count observations."""

    def __init__(self, eps: float = 1e-12):
        """Create a multinomial distribution."""

        self.eps = float(eps)

    def log_prob(self, x, *, p, n=None) -> np.ndarray:
        """Return candidate log probabilities for count observations.

        Parameters
        ----------
        x
            Count vector with one value per category.
        p
            Candidate category probabilities with shape ``(n_candidates, K)``.
        n
            Optional total count. Defaults to ``sum(x)``.

        Returns
        -------
        numpy.ndarray
            One log probability per candidate.
        """

        counts = np.asarray(x, dtype=float)
        total = float(np.sum(counts)) if n is None else float(n)
        p_arr = _clip_prob(p, self.eps)
        log_const = gammaln(total + 1.0) - np.sum(gammaln(counts + 1.0))
        return log_const + np.sum(counts * np.log(p_arr), axis=-1)


class Normal:
    """Normal likelihood for continuous observations."""

    def log_prob(self, x, *, mu, sigma) -> np.ndarray:
        """Return candidate log probabilities for normal observations.

        Parameters
        ----------
        x
            Observed scalar or vector.
        mu
            Candidate means.
        sigma
            Candidate or fixed standard deviations.

        Returns
        -------
        numpy.ndarray
            One log probability per candidate.
        """

        x_arr = np.asarray(x, dtype=float)
        mu_arr = np.asarray(mu, dtype=float)
        sigma_arr = _check_positive(sigma, "sigma")
        z = (x_arr - mu_arr) / sigma_arr
        logp = -0.5 * (z**2 + LOG_2PI) - np.log(sigma_arr)
        return _sum_non_candidate_axes(logp)


class Gaussian(Normal):
    """Alias for ``Normal`` using common Gaussian terminology."""


class LogNormal:
    """Log-normal likelihood for positive continuous observations."""

    def log_prob(self, x, *, mu, sigma) -> np.ndarray:
        """Return candidate log probabilities for log-normal observations."""

        x_arr = _check_positive(x, "x")
        log_x = np.log(x_arr)
        mu_arr = np.asarray(mu, dtype=float)
        sigma_arr = _check_positive(sigma, "sigma")
        z = (log_x - mu_arr) / sigma_arr
        logp = -0.5 * (z**2 + LOG_2PI) - np.log(sigma_arr) - log_x
        return _sum_non_candidate_axes(logp)


class Poisson:
    """Poisson likelihood for nonnegative count observations."""

    def log_prob(self, x, *, rate=None, lam=None) -> np.ndarray:
        """Return candidate log probabilities for Poisson observations."""

        rate_arr = lam if rate is None else rate
        rate_arr = _check_positive(rate_arr, "rate")
        x_arr = np.asarray(x, dtype=float)
        logp = x_arr * np.log(rate_arr) - rate_arr - gammaln(x_arr + 1.0)
        return _sum_non_candidate_axes(logp)


class VonMises:
    """Von Mises likelihood for circular observations in radians."""

    def log_prob(self, x, *, mu, kappa) -> np.ndarray:
        """Return candidate log probabilities for circular observations."""

        x_arr = np.asarray(x, dtype=float)
        mu_arr = np.asarray(mu, dtype=float)
        kappa_arr = _check_positive(kappa, "kappa")
        # log(I0(kappa)) = log(i0e(kappa)) + abs(kappa), which is stable for
        # larger concentration values.
        log_i0 = np.log(i0e(kappa_arr)) + np.abs(kappa_arr)
        logp = kappa_arr * np.cos(x_arr - mu_arr) - np.log(2.0 * np.pi) - log_i0
        return _sum_non_candidate_axes(logp)


class IID:
    """Product likelihood for independent repeated observations."""

    def __init__(self, base):
        """Create an IID wrapper around another distribution.

        Parameters
        ----------
        base
            Distribution exposing ``log_prob``.

        Returns
        -------
        None
            The initialized wrapper exposes ``log_prob``.
        """

        self.base = base

    def log_prob(self, x, **params) -> np.ndarray:
        """Return summed log probabilities over repeated observations."""

        return self.base.log_prob(x, **params)


class Joint:
    """Sum log probabilities from named observation components."""

    def __init__(self, **parts):
        """Create a joint distribution from named component distributions."""

        if not parts:
            raise ValueError("Joint requires at least one named distribution.")
        self.parts = dict(parts)

    def log_prob(self, x, **params) -> np.ndarray:
        """Return summed log probabilities over named components.

        Parameters
        ----------
        x
            Mapping from component names to observations.
        **params
            Mapping from component names to distribution-parameter mappings.

        Returns
        -------
        numpy.ndarray
            One joint log probability per candidate.
        """

        if not isinstance(x, Mapping):
            raise TypeError("Joint observations must be a mapping.")
        total = None
        for name, dist in self.parts.items():
            if name not in x:
                raise ValueError(f"Missing observation component '{name}'.")
            if name not in params:
                raise ValueError(f"Missing parameter component '{name}'.")
            part_logp = dist.log_prob(x[name], **params[name])
            total = part_logp if total is None else total + part_logp
        return total


class Mixture:
    """Weighted mixture of component distributions."""

    def __init__(self, weights, components):
        """Create a mixture distribution.

        Parameters
        ----------
        weights
            Nonnegative component weights. They are normalized internally.
        components
            Sequence of component distributions exposing ``log_prob``.

        Returns
        -------
        None
            The initialized mixture exposes ``log_prob``.
        """

        weights_arr = np.asarray(weights, dtype=float)
        if weights_arr.ndim != 1 or np.any(weights_arr < 0) or np.sum(weights_arr) <= 0:
            raise ValueError("Mixture weights must be one-dimensional and nonnegative.")
        if len(weights_arr) != len(components):
            raise ValueError("Mixture weights and components must have the same length.")
        self.weights = weights_arr / np.sum(weights_arr)
        self.components = list(components)

    def log_prob(self, x, **params) -> np.ndarray:
        """Return candidate log probabilities for a mixture observation."""

        if "components" in params:
            component_params = params["components"]
        else:
            component_params = [params for _ in self.components]
        pieces = []
        for weight, dist, one_params in zip(
            self.weights, self.components, component_params, strict=True
        ):
            pieces.append(np.log(weight) + dist.log_prob(x, **one_params))
        return logsumexp(np.stack(pieces, axis=0), axis=0)


def _clip_prob(value, eps: float) -> np.ndarray:
    """Clip probabilities to a numerically safe open interval."""

    return np.clip(np.asarray(value, dtype=float), eps, 1.0 - eps)


def _check_positive(value, name: str) -> np.ndarray:
    """Return a positive numeric array or raise a clear error."""

    arr = np.asarray(value, dtype=float)
    if np.any(arr <= 0) or np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} must contain positive finite values.")
    return arr


def _sum_non_candidate_axes(logp: np.ndarray) -> np.ndarray:
    """Sum all axes after the candidate axis.

    Distribution parameters are expected to carry candidates on the first axis.
    Scalar outputs are promoted to one-dimensional arrays for a consistent
    sampler interface.
    """

    arr = np.asarray(logp, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    return np.sum(arr, axis=tuple(range(1, arr.ndim)))
