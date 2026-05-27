"""BayesFlow-free posthoc subject estimation for M3 hierarchy models.

This module is used inside multiprocessing workers. It intentionally avoids
importing the trained BayesFlow workflow or M3 model classes, so workers only
run NumPy likelihood-weighted estimation from already sampled group posteriors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

from bami.inference.posthoc_sampler import PosthocResult, PosthocSampler
from bami.inference.priors import (
    apply_link,
    draw_group_mean_raw,
    draw_group_sd_raw,
    log_sigma_key,
    mu_raw_key,
)


RESPONSE_COLUMNS = ["correct", "other", "dist", "other_dist", "new"]
BASE_PARAMS = ["a", "c", "ra", "rc"]

__all__ = ["M3PosthocEstimator", "PosthocResult", "PosthocSampler"]


class M3PosthocEstimator:
    """Estimate M3 subject parameters without a BayesFlow workflow.

    Parameters
    ----------
    n_options
        Number of response options represented by each response category.
    rule
        Response rule. Supported values match the M3 model: ``"softmax"`` and
        ``"simple"``.
    priors
        Prior specification used to interpret group posterior samples.
    const_params
        Constant M3 parameters, such as ``b``.
    hier_params
        Hierarchical M3 parameters estimated posthoc for each subject.

    Returns
    -------
    None
        Initializes a lightweight estimator for ``estimate_subjects``.
    """

    def __init__(
        self,
        *,
        n_options,
        rule: str,
        priors: Mapping,
        const_params: Mapping,
        hier_params: Mapping,
    ):
        """Create a workflow-free M3 posthoc estimator.

        Parameters
        ----------
        n_options, rule, priors, const_params, hier_params
            Settings copied from a trained M3 hierarchy model.

        Returns
        -------
        None
            Stores only the state needed for likelihood-weighted posthoc
            estimation.
        """

        self.n_options = n_options
        self.rule = rule
        self.priors = dict(priors)
        self._const_params = dict(const_params)
        self._hier_params = dict(hier_params)

    def validate_counts(
        self, counts: pd.DataFrame | np.ndarray
    ) -> tuple[np.ndarray, list]:
        """Validate subject-by-category response counts.

        Parameters
        ----------
        counts
            DataFrame or 2D array with columns ordered as ``correct, other,
            dist, other_dist, new``.

        Returns
        -------
        tuple[np.ndarray, list]
            Integer count matrix and subject identifiers.
        """

        if isinstance(counts, pd.DataFrame):
            missing = [col for col in RESPONSE_COLUMNS if col not in counts.columns]
            if missing:
                raise ValueError(f"Missing response count columns: {missing}")
            arr = counts[RESPONSE_COLUMNS].to_numpy()
            if "subject_id" in counts.columns:
                subject_ids = counts["subject_id"].tolist()
            else:
                subject_ids = counts.index.tolist()
        else:
            arr = np.asarray(counts)
            subject_ids = list(range(arr.shape[0])) if arr.ndim == 2 else []

        if arr.ndim != 2 or arr.shape[1] != len(RESPONSE_COLUMNS):
            raise ValueError("counts must have shape (n_subjects, 5).")
        if np.any(~np.isfinite(arr)):
            raise ValueError("counts must not contain missing or infinite values.")
        if np.any(arr < 0):
            raise ValueError("counts must be nonnegative.")
        if not np.allclose(arr, np.round(arr)):
            raise ValueError("counts must contain integer response counts.")

        return arr.astype(np.float32), subject_ids

    def m3_trial(self, a, c, ra, rc, b) -> np.ndarray:
        """Compute M3 response probabilities for one candidate parameter set.

        Parameters
        ----------
        a, c, ra, rc, b
            Public-scale M3 parameters.

        Returns
        -------
        np.ndarray
            Five response-category probabilities in the shared M3 order.
        """

        activations = np.array(
            [
                a + c + b,
                a + b,
                ra * a + rc * c + b,
                ra * a + b,
                b,
            ],
            dtype=float,
        )
        n_options = self._format_n_options(len(activations))
        return self._compute_probs(activations, n_options)

    def _format_n_options(self, n_categories: int) -> np.ndarray:
        """Return response-option weights aligned to the category vector.

        Parameters
        ----------
        n_categories
            Number of response categories.

        Returns
        -------
        np.ndarray
            One weight per response category.
        """

        if self.n_options is None:
            return np.full(n_categories, 1.0, dtype=float)
        if isinstance(self.n_options, int):
            return np.full(n_categories, self.n_options, dtype=float)
        if hasattr(self.n_options, "__len__") and len(self.n_options) == n_categories:
            return np.asarray(self.n_options, dtype=float)
        raise ValueError(
            "n_options must be None, an int, or an array-like with the same "
            f"length as activations ({n_categories}), but got {self.n_options}"
        )

    def _compute_probs(self, activations: np.ndarray, n_options: np.ndarray):
        """Apply the M3 response rule to activation scores.

        Parameters
        ----------
        activations
            Five M3 activation scores.
        n_options
            Response-option weights for the five categories.

        Returns
        -------
        np.ndarray
            Normalized category probabilities.
        """

        if self.rule == "softmax":
            max_logit = np.max(activations)
            temp = 2.0
            exp_vals = np.exp((activations - max_logit) / temp)
            weighted = exp_vals * n_options
            return weighted / np.sum(weighted)

        if self.rule == "simple":
            shifted = activations - np.min(activations) + 1e-8
            weighted = shifted * n_options
            return weighted / np.sum(weighted)

        raise ValueError(f"Unsupported rule: {self.rule}")

    @staticmethod
    def _weighted_quantile(
        values: np.ndarray, weights: np.ndarray, probs: Sequence[float]
    ) -> np.ndarray:
        """Compute weighted quantiles for one parameter.

        Parameters
        ----------
        values
            Candidate parameter values.
        weights
            Normalized candidate weights.
        probs
            Quantile probabilities.

        Returns
        -------
        np.ndarray
            Weighted quantile values.
        """

        order = np.argsort(values)
        sorted_values = values[order]
        sorted_weights = weights[order]
        cdf = np.cumsum(sorted_weights)
        cdf = cdf / cdf[-1]
        return np.interp(probs, cdf, sorted_values)

    def _draw_subject_candidates(
        self,
        group_samples: Mapping[str, np.ndarray],
        n_candidates: int,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        """Draw subject candidates from group posterior samples.

        Parameters
        ----------
        group_samples
            Group posterior samples for one simulated dataset.
        n_candidates
            Number of candidate subject parameter sets.
        rng
            Random generator for reproducible candidate draws.

        Returns
        -------
        dict[str, np.ndarray]
            Candidate arrays keyed by parameter name.
        """

        candidates = {}
        for p_name, spec in self._hier_params.items():
            link = spec.get("link", None)
            mu_key = mu_raw_key(p_name)
            sigma_key = log_sigma_key(p_name)
            if mu_key in group_samples:
                mu = np.asarray(group_samples[mu_key], dtype=float).reshape(-1)
            else:
                mu = np.full(1, draw_group_mean_raw(p_name, spec), dtype=float)
            if sigma_key in group_samples:
                sigma = np.exp(
                    np.asarray(group_samples[sigma_key], dtype=float).reshape(-1)
                )
            else:
                sigma = np.full(1, draw_group_sd_raw(p_name, spec), dtype=float)
            draw_idx = rng.integers(0, mu.size, size=n_candidates)
            if sigma.size == mu.size:
                sigma_idx = draw_idx
            else:
                sigma_idx = rng.integers(0, sigma.size, size=n_candidates)
            raw_subj = rng.normal(loc=mu[draw_idx], scale=sigma[sigma_idx])
            candidates[p_name] = np.asarray(apply_link(raw_subj, link), dtype=float)
        return candidates

    def _candidate_loglik(
        self, counts: np.ndarray, candidates: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        """Evaluate candidates against one subject's response counts.

        Parameters
        ----------
        counts
            Five response-category counts for one subject.
        candidates
            Candidate subject parameter arrays.

        Returns
        -------
        np.ndarray
            Multinomial log likelihood for each candidate.
        """

        n_candidates = len(next(iter(candidates.values())))
        loglik = np.zeros(n_candidates, dtype=float)
        total = float(np.sum(counts))
        log_const = gammaln(total + 1.0) - np.sum(gammaln(counts + 1.0))

        for i in range(n_candidates):
            params = {p_name: values[i] for p_name, values in candidates.items()}
            for p_name, val in self._const_params.items():
                params[p_name] = val
            probs = np.clip(self.m3_trial(**params), 1e-12, 1.0)
            loglik[i] = log_const + np.sum(counts * np.log(probs))
        return loglik

    @staticmethod
    def _normalize_log_weights(loglik: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Normalize likelihood weights and compute ESS diagnostics.

        Parameters
        ----------
        loglik
            Candidate log likelihood values.

        Returns
        -------
        tuple[np.ndarray, float, float]
            Normalized weights, effective sample size, and largest weight.
        """

        log_weights = loglik - logsumexp(loglik)
        weights = np.exp(log_weights)
        ess = 1.0 / np.sum(weights**2)
        max_weight = float(np.max(weights))
        return weights, float(ess), max_weight

    @staticmethod
    def _append_candidates(
        current: Mapping[str, np.ndarray],
        new: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Append newly drawn candidate arrays to existing arrays.

        Parameters
        ----------
        current
            Existing candidate arrays.
        new
            New candidate arrays with the same keys.

        Returns
        -------
        dict[str, np.ndarray]
            Combined candidate arrays.
        """

        return {
            p_name: np.concatenate(
                [np.asarray(current[p_name]), np.asarray(new[p_name])]
            )
            for p_name in current
        }

    def _draw_adaptive_subject_candidates(
        self,
        counts: np.ndarray,
        group_samples: Mapping[str, np.ndarray],
        n_candidates: int,
        min_ess: float,
        max_candidates: int,
        batch_candidates: int,
        adaptive: bool,
        rng: np.random.Generator,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, float, float, str]:
        """Draw posthoc candidates until ESS is adequate or capped.

        Parameters
        ----------
        counts
            Five response-category counts for one subject.
        group_samples
            Group posterior samples for one simulated dataset.
        n_candidates, min_ess, max_candidates, batch_candidates, adaptive
            Candidate drawing controls.
        rng
            Random generator for reproducible posthoc sampling.

        Returns
        -------
        tuple
            Candidate values, normalized weights, ESS, max weight, and status.
        """

        candidates = self._draw_subject_candidates(group_samples, n_candidates, rng)
        loglik = self._candidate_loglik(counts, candidates)
        weights, ess, max_weight = self._normalize_log_weights(loglik)

        while adaptive and ess < min_ess and len(loglik) < max_candidates:
            remaining = max_candidates - len(loglik)
            draw_count = min(batch_candidates, remaining)
            new_candidates = self._draw_subject_candidates(
                group_samples, draw_count, rng
            )
            new_loglik = self._candidate_loglik(counts, new_candidates)
            candidates = self._append_candidates(candidates, new_candidates)
            loglik = np.concatenate([loglik, new_loglik])
            weights, ess, max_weight = self._normalize_log_weights(loglik)

        status = "ok" if ess >= min_ess else "low_ess"
        return candidates, weights, ess, max_weight, status

    def estimate_subjects(
        self,
        counts: pd.DataFrame | np.ndarray,
        group_samples: Mapping[str, np.ndarray],
        n_candidates: int = 4000,
        ci: float = 0.95,
        random_seed: int | None = None,
        *,
        adaptive: bool = True,
        min_ess: float = 200.0,
        max_candidates: int = 20000,
        batch_candidates: int | None = None,
    ) -> pd.DataFrame:
        """Estimate subject-level M3 parameters from group posterior samples.

        Parameters
        ----------
        counts
            Subject-by-category integer count matrix.
        group_samples
            Precomputed group posterior samples for one dataset.
        n_candidates
            Number of initial posthoc candidates per subject.
        ci
            Credible interval width.
        random_seed
            Optional seed for reproducible candidate draws.
        adaptive, min_ess, max_candidates, batch_candidates
            Candidate drawing controls.

        Returns
        -------
        pd.DataFrame
            Long table with one row per subject and parameter.
        """

        count_arr, subject_ids = self.validate_counts(counts)
        if n_candidates < 1:
            raise ValueError("n_candidates must be at least 1.")
        if max_candidates < n_candidates:
            raise ValueError("max_candidates must be at least n_candidates.")
        if min_ess < 1:
            raise ValueError("min_ess must be at least 1.")
        if batch_candidates is None:
            batch_candidates = n_candidates
        if batch_candidates < 1:
            raise ValueError("batch_candidates must be at least 1.")

        rng = np.random.default_rng(random_seed)
        alpha = (1.0 - ci) / 2.0
        probs = [alpha, 0.5, 1.0 - alpha]

        rows = []
        for subj_idx, subject_id in enumerate(subject_ids):
            candidates, weights, ess, max_weight, status = (
                self._draw_adaptive_subject_candidates(
                    count_arr[subj_idx],
                    group_samples,
                    n_candidates,
                    min_ess,
                    max_candidates,
                    batch_candidates,
                    adaptive,
                    rng,
                )
            )
            n_candidates_used = len(weights)
            ess_ratio = ess / n_candidates_used

            for p_name in BASE_PARAMS:
                if p_name not in candidates:
                    continue
                lower, median, upper = self._weighted_quantile(
                    candidates[p_name],
                    weights,
                    probs,
                )
                rows.append(
                    {
                        "subject_id": subject_id,
                        "param": p_name,
                        "median": float(median),
                        "lower": float(lower),
                        "upper": float(upper),
                        "ci": float(ci),
                        "ess": float(ess),
                        "ess_ratio": float(ess_ratio),
                        "max_weight": float(max_weight),
                        "n_candidates_used": int(n_candidates_used),
                        "posthoc_status": status,
                    }
                )

        return pd.DataFrame(rows)
