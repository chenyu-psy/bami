"""Generic posthoc subject sampler for hierarchical workflows.

The sampler draws subject-level candidate parameters from group posterior
samples, scores those candidates with an analytic observation distribution, and
summarizes the weighted candidates for each subject.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from bami.inference.priors import (
    apply_link,
    invert_link,
    log_sigma_key,
    mu_raw_key,
)


@dataclass
class PosthocResult:
    """Container returned by ``PosthocSampler.sample_subjects``.

    Parameters
    ----------
    summary
        Long-format point-estimate rows. Empty when only distribution draws are
        requested.
    diagnostics
        Per-subject ESS and maximum-weight diagnostics.
    draws
        Optional posterior-like draws resampled from weighted candidates.

    Returns
    -------
    None
        Dataclass container with analysis-facing tables.
    """

    summary: pd.DataFrame
    diagnostics: pd.DataFrame
    draws: pd.DataFrame | None = None


class PosthocSampler:
    """Draw and weight subject-level candidates for one hierarchy model.

    Parameters
    ----------
    priors
        Project prior specification for subject-level parameters.
    obs_names
        Names of observation columns produced by the simulator before design
        features such as trial count and mask columns.
    observation
        Distribution object exposing ``log_prob``.
    theta_to_params
        Function called as ``theta_to_params(theta, context)``. It maps
        candidate subject parameters to distribution parameters.
    context
        Optional fixed values used by ``theta_to_params``.
    model
        Optional workflow model. When created with ``from_model``, design
        metadata such as fixed trial count and mask behavior are read from it.

    Returns
    -------
    None
        The initialized sampler exposes ``sample_subjects``.
    """

    def __init__(
        self,
        *,
        priors: Mapping,
        obs_names: Sequence[str],
        observation,
        theta_to_params: Callable,
        context: Mapping | None = None,
        model=None,
    ):
        self.priors = dict(priors)
        self.param_names = [
            name for name, spec in self.priors.items() if isinstance(spec, dict)
        ]
        if not self.param_names:
            raise ValueError("PosthocSampler requires at least one stochastic prior.")
        self.obs_names = self._check_obs_names(obs_names)
        self.observation = observation
        self.theta_to_params = self._check_callable(theta_to_params, "theta_to_params")
        self.context = dict(context or {})
        self.model = model

    @classmethod
    def from_model(
        cls,
        model,
        *,
        observation,
        theta_to_params: Callable,
        context: Mapping | None = None,
    ) -> "PosthocSampler":
        """Build a sampler from a configured hierarchy model.

        Parameters
        ----------
        model
            Workflow object exposing ``priors`` and ``obs_names``.
        observation
            Distribution object exposing ``log_prob``.
        theta_to_params
            Function mapping candidate parameters to observation parameters.
        context
            Optional extra values used during weighting.

        Returns
        -------
        PosthocSampler
            Configured sampler.
        """

        if not hasattr(model, "priors"):
            raise ValueError("model must expose priors.")
        obs_names = getattr(model, "obs_names", None)
        if obs_names is None:
            raise ValueError(
                "model must expose obs_names. Pass obs_names when building the workflow."
            )
        return cls(
            priors=model.priors,
            obs_names=obs_names,
            observation=observation,
            theta_to_params=theta_to_params,
            context=context,
            model=model,
        )

    def sample_subjects(
        self,
        data,
        group_samples: Mapping[str, np.ndarray],
        *,
        n_candidates: int = 4000,
        statistic: str | Sequence[str] = "mean",
        n_draws: int = 500,
        random_seed: int | None = None,
    ) -> PosthocResult:
        """Estimate subject parameters with weighted posthoc candidates.

        Parameters
        ----------
        data
            Dataset dictionary containing ``data`` or a data array with shape
            ``(n_datasets, n_subjects, n_features)``.
        group_samples
            Group posterior sample dictionary for all datasets.
        n_candidates
            Number of subject candidates drawn per dataset.
        statistic
            ``"mean"``, ``"median"``, ``"distribution"``, or a list combining
            these values.
        n_draws
            Draw count used when ``statistic`` includes ``"distribution"``.
        random_seed
            Optional seed for reproducible candidate draws and resampling.

        Returns
        -------
        PosthocResult
            Summary rows, diagnostics, and optional resampled draws.
        """

        stats = self._check_statistics(statistic)
        checked_candidates = self._check_positive_int(n_candidates, "n_candidates")
        checked_draws = self._check_positive_int(n_draws, "n_draws")
        rng = np.random.default_rng(random_seed)
        payload = self._prepare_data(data)
        n_datasets = payload["data"].shape[0]

        summary_rows: list[dict] = []
        diagnostic_rows: list[dict] = []
        draw_rows: list[dict] = []

        for dataset_id in range(n_datasets):
            candidates = self._draw_candidates(
                group_samples=group_samples,
                dataset_id=dataset_id,
                n_candidates=checked_candidates,
                rng=rng,
            )
            active_subjects = payload["active"][dataset_id]
            for subject_id in np.where(active_subjects)[0]:
                obs = self._subject_observation(payload, dataset_id, int(subject_id))
                context = self._subject_context(payload, dataset_id, int(subject_id))
                params = self.theta_to_params(candidates, context)
                log_weights = np.asarray(
                    self.observation.log_prob(obs, **params),
                    dtype=float,
                ).reshape(-1)
                if log_weights.shape[0] != checked_candidates:
                    raise ValueError(
                        "observation.log_prob must return one value per candidate."
                    )
                weights = self._normalize_log_weights(log_weights)
                diagnostics = self._diagnostics(weights)
                diagnostic_rows.append(
                    {
                        "dataset_id": int(dataset_id),
                        "subject_id": int(subject_id),
                        "ess": diagnostics["ess"],
                        "max_weight": diagnostics["max_weight"],
                        "n_candidates": checked_candidates,
                    }
                )
                for one_stat in stats:
                    if one_stat == "distribution":
                        draw_rows.extend(
                            self._resample_draw_rows(
                                candidates=candidates,
                                weights=weights,
                                dataset_id=dataset_id,
                                subject_id=int(subject_id),
                                n_draws=checked_draws,
                                rng=rng,
                            )
                        )
                    else:
                        summary_rows.extend(
                            self._summary_rows(
                                candidates=candidates,
                                weights=weights,
                                statistic=one_stat,
                                dataset_id=dataset_id,
                                subject_id=int(subject_id),
                            )
                        )

        return PosthocResult(
            summary=pd.DataFrame(summary_rows),
            diagnostics=pd.DataFrame(diagnostic_rows),
            draws=pd.DataFrame(draw_rows) if "distribution" in stats else None,
        )

    def _draw_candidates(
        self,
        *,
        group_samples: Mapping[str, np.ndarray],
        dataset_id: int,
        n_candidates: int,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        """Draw subject candidates from one dataset's group posterior samples."""

        candidates = {}
        for param_name in self.param_names:
            spec = self.priors[param_name]
            mu = self._group_sample_array(
                group_samples,
                dataset_id,
                raw_key=mu_raw_key(param_name),
                public_key=f"{param_name}_mu",
                link=spec.get("link", "identity"),
                raw_from_public=True,
            )
            sigma = self._group_sample_array(
                group_samples,
                dataset_id,
                raw_key=log_sigma_key(param_name),
                public_key=f"{param_name}_sigma",
                link=None,
                raw_from_public=False,
            )
            if log_sigma_key(param_name) in group_samples:
                sigma = np.exp(sigma)
            draw_idx = rng.integers(0, mu.shape[0], size=n_candidates)
            sigma_idx = (
                draw_idx
                if sigma.shape[0] == mu.shape[0]
                else rng.integers(0, sigma.shape[0], size=n_candidates)
            )
            raw_values = rng.normal(loc=mu[draw_idx], scale=sigma[sigma_idx])
            candidates[param_name] = np.asarray(
                apply_link(raw_values, spec.get("link", "identity")),
                dtype=float,
            )

        for name, spec in self.priors.items():
            if not isinstance(spec, dict):
                candidates[name] = np.full(n_candidates, float(spec), dtype=float)
        return candidates

    def _group_sample_array(
        self,
        group_samples: Mapping[str, np.ndarray],
        dataset_id: int,
        *,
        raw_key: str,
        public_key: str,
        link: str | None,
        raw_from_public: bool,
    ) -> np.ndarray:
        """Return one dataset's group posterior array for candidate drawing."""

        if raw_key in group_samples:
            values = np.asarray(group_samples[raw_key], dtype=float)[dataset_id]
            return values.reshape(-1)
        if public_key in group_samples:
            values = np.asarray(group_samples[public_key], dtype=float)[dataset_id]
            if raw_from_public:
                values = invert_link(values, link)
            return values.reshape(-1)
        raise ValueError(
            f"group_samples must contain '{raw_key}' or '{public_key}' for posthoc."
        )

    def _prepare_data(self, data) -> dict[str, np.ndarray]:
        """Return data, observations, active masks, and trial counts."""

        if isinstance(data, Mapping):
            arr = np.asarray(data["data"], dtype=float)
        else:
            arr = np.asarray(data, dtype=float)
        if arr.ndim != 3:
            raise ValueError(
                "data must have shape (n_datasets, n_subjects, n_features)."
            )
        if arr.shape[-1] < len(self.obs_names):
            raise ValueError("data has fewer columns than obs_names.")

        obs = arr[:, :, : len(self.obs_names)]
        active = np.ones(arr.shape[:2], dtype=bool)
        if self._model_bool("include_mask"):
            active = arr[:, :, -1] > 0.5

        trial_counts = self._extract_trial_counts(arr)
        return {"data": arr, "obs": obs, "active": active, "n_trials": trial_counts}

    def _extract_trial_counts(self, arr: np.ndarray) -> np.ndarray:
        """Return per-subject trial counts for fixed or explicit-trial data."""

        n_datasets, n_subjects = arr.shape[:2]
        if "n_trials" in self.context:
            n_trials = int(self.context["n_trials"])
            return np.full((n_datasets, n_subjects), n_trials, dtype=int)

        fixed_trials = getattr(self.model, "n_trials", None)
        if fixed_trials is not None:
            return np.full((n_datasets, n_subjects), int(fixed_trials), dtype=int)

        obs_spec = getattr(self.model, "obs_spec", None)
        if obs_spec is not None and getattr(obs_spec, "add_n", False):
            raise ValueError(
                "Cannot recover raw n_trials from encoded ObsSpec data. "
                "Pass fixed n_trials or include explicit trial counts in the data."
            )

        if self._model_bool("include_trial_feature"):
            col = len(self.obs_names)
            trial_values = arr[:, :, col]
            scale = getattr(self.model, "trial_feature_scale", None)
            if scale is not None:
                trial_values = trial_values * float(scale)
            return np.rint(trial_values).astype(int)

        # Count-like observations can infer n from the observed row. Summary
        # observations should pass fixed n_trials or explicit trial features.
        return np.rint(np.sum(arr[:, :, : len(self.obs_names)], axis=-1)).astype(int)

    def _subject_observation(
        self,
        payload: Mapping[str, np.ndarray],
        dataset_id: int,
        subject_id: int,
    ):
        """Return one subject observation for the configured distribution."""

        values = payload["obs"][dataset_id, subject_id]
        if hasattr(self.observation, "parts"):
            return {name: float(values[i]) for i, name in enumerate(self.obs_names)}
        if len(self.obs_names) == 1:
            return float(values[0])
        return values

    def _subject_context(
        self,
        payload: Mapping[str, np.ndarray],
        dataset_id: int,
        subject_id: int,
    ) -> dict:
        """Return context values for one subject likelihood evaluation."""

        context = dict(self.context)
        context.setdefault("n_trials", int(payload["n_trials"][dataset_id, subject_id]))
        context.setdefault("dataset_id", int(dataset_id))
        context.setdefault("subject_id", int(subject_id))
        return context

    @staticmethod
    def _normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
        """Normalize log weights with a stable softmax."""

        log_weights = np.asarray(log_weights, dtype=float)
        if np.any(~np.isfinite(log_weights)):
            raise ValueError("log weights must be finite.")
        return np.exp(log_weights - logsumexp(log_weights))

    @staticmethod
    def _diagnostics(weights: np.ndarray) -> dict[str, float]:
        """Return ESS and maximum-weight diagnostics."""

        return {
            "ess": float(1.0 / np.sum(weights**2)),
            "max_weight": float(np.max(weights)),
        }

    def _summary_rows(
        self,
        *,
        candidates: Mapping[str, np.ndarray],
        weights: np.ndarray,
        statistic: str,
        dataset_id: int,
        subject_id: int,
    ) -> list[dict]:
        """Return long rows for one point-estimate statistic."""

        rows = []
        for param_name in self.param_names:
            values = np.asarray(candidates[param_name], dtype=float)
            if statistic == "mean":
                est = float(np.sum(weights * values))
            elif statistic == "median":
                est = float(self._weighted_quantile(values, weights, 0.5))
            else:
                raise ValueError(f"Unsupported statistic: {statistic}")
            rows.append(
                {
                    "dataset_id": int(dataset_id),
                    "subject_id": int(subject_id),
                    "param": param_name,
                    "statistic": statistic,
                    "est_value": est,
                }
            )
        return rows

    def _resample_draw_rows(
        self,
        *,
        candidates: Mapping[str, np.ndarray],
        weights: np.ndarray,
        dataset_id: int,
        subject_id: int,
        n_draws: int,
        rng: np.random.Generator,
    ) -> list[dict]:
        """Return posterior-like draws resampled from weighted candidates."""

        draw_idx = rng.choice(weights.shape[0], size=n_draws, replace=True, p=weights)
        rows = []
        for draw_id, idx in enumerate(draw_idx):
            row = {
                "dataset_id": int(dataset_id),
                "subject_id": int(subject_id),
                "draw_id": int(draw_id),
            }
            for param_name in self.param_names:
                row[param_name] = float(candidates[param_name][idx])
            rows.append(row)
        return rows

    @staticmethod
    def _weighted_quantile(
        values: np.ndarray,
        weights: np.ndarray,
        prob: float,
    ) -> float:
        """Return one weighted quantile."""

        order = np.argsort(values)
        sorted_values = values[order]
        sorted_weights = weights[order]
        cdf = np.cumsum(sorted_weights)
        cdf = cdf / cdf[-1]
        return float(np.interp(prob, cdf, sorted_values))

    @staticmethod
    def _check_obs_names(obs_names: Sequence[str]) -> list[str]:
        """Validate observation column names."""

        names = list(obs_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("obs_names must contain at least one non-empty string.")
        if len(set(names)) != len(names):
            raise ValueError("obs_names must be unique.")
        return names

    @staticmethod
    def _check_callable(value, name: str):
        """Validate a required callable."""

        if not callable(value):
            raise TypeError(f"{name} must be callable.")
        return value

    @staticmethod
    def _check_positive_int(value: int, name: str) -> int:
        """Validate a positive integer setting."""

        checked = int(value)
        if checked < 1:
            raise ValueError(f"{name} must be at least 1.")
        return checked

    @staticmethod
    def _check_statistics(statistic: str | Sequence[str]) -> list[str]:
        """Validate requested output statistics."""

        if isinstance(statistic, str):
            stats = [statistic]
        else:
            stats = list(statistic)
        allowed = {"mean", "median", "distribution"}
        invalid = [name for name in stats if name not in allowed]
        if invalid:
            raise ValueError(f"Unsupported statistic(s): {invalid}")
        return stats

    def _model_bool(self, name: str) -> bool:
        """Return a boolean model attribute, defaulting to ``False``."""

        return bool(getattr(self.model, name, False))
