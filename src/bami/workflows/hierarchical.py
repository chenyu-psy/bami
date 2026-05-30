"""Generic hierarchical BayesFlow workflow.

This module builds group-level workflows where one simulated dataset contains
multiple subjects. Aggregate observations use one fixed-width row per subject.
Trial observations use nested subject-by-trial rows for continuous data.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import warnings

import numpy as np
import bayesflow as bf
import keras

from bami.inputs import InputFormat
from bami.workflows import training
from bami.workflows.contracts import validate_observation
from bami.workflows.simple import SimpleWorkflow


class _RandomWorkflowTrainingAdapter:
    """Expose a random workflow through the shared training helper interface.

    Parameters
    ----------
    model
        Hierarchical workflow that owns ``random_workflow`` and random
        validation-data handling.

    Returns
    -------
    None
        The adapter presents ``workflow`` and ``_resolve_validation_data`` to
        ``bami.workflows.training.train_workflow``.
    """

    def __init__(self, model):
        """Store the parent model and random BayesFlow workflow."""

        self.model = model
        self.workflow = model.random_workflow

    def _resolve_validation_data(self, validation_data: int | dict) -> dict:
        """Return validation data from the parent random workflow.

        Parameters
        ----------
        validation_data
            Integer validation-set size or pre-simulated validation data.

        Returns
        -------
        dict
            Validation data dictionary for ``random_workflow``.
        """

        return self.model._resolve_random_validation_data(validation_data)


@keras.saving.register_keras_serializable(package="bayesflow_ind")
class NestedDeepSetSummary(bf.networks.SummaryNetwork):
    """Summarize nested subject-by-trial data with two DeepSet stages.

    Parameters
    ----------
    summary_dim
        Width of the learned summary at both the trial and subject levels.
    **kwargs
        Extra layer settings passed to the BayesFlow ``SummaryNetwork`` base.

    Returns
    -------
    None
        The initialized layer expects data shaped
        ``(batch, subjects, trials, features)`` and returns one group summary
        row per batch item.
    """

    def __init__(self, summary_dim: int = 64, **kwargs):
        """Create trial-level and subject-level DeepSet summarizers."""

        super().__init__(**kwargs)
        self.summary_dim = int(summary_dim)
        self.trial_summary = bf.networks.DeepSet(summary_dim=self.summary_dim)
        self.subject_summary = bf.networks.DeepSet(summary_dim=self.summary_dim)

    def call(self, x, training: bool = False, **kwargs):
        """Compress trial rows within subjects, then subjects within groups.

        Parameters
        ----------
        x
            Nested trial data with shape ``batch x subjects x trials x
            features``.
        training
            Whether the layer is being called during training.
        **kwargs
            Accepted for compatibility with BayesFlow summary-network calls.

        Returns
        -------
        Tensor
            Group-level summary with shape ``batch x summary_dim``.
        """

        del kwargs

        subject_summaries = self.trial_summary(x, training=training)
        return self.subject_summary(subject_summaries, training=training)

    def get_config(self) -> dict:
        """Return Keras-serializable layer settings."""

        config = super().get_config()
        config.update({"summary_dim": self.summary_dim})
        return config


@keras.saving.register_keras_serializable(package="bayesflow_ind")
class MaskedEquivariantSetEncoder(keras.Layer):
    """Encode one padded exchangeable set with mask-aware context blocks.

    Parameters
    ----------
    summary_dim
        Width of the returned set summary.
    element_widths
        Layer widths for the element embedding network.
    n_context_blocks
        Number of masked context-injection blocks. Each block computes a
        masked mean and variance over active elements, broadcasts that context
        back to each active element, and applies a residual update.
    post_widths
        Layer widths for the post-pooling network.
    activation
        Activation function used in the hidden layers.
    include_count_features
        Whether to append active set-size features after masked pooling.
    **kwargs
        Extra layer settings passed to the Keras layer base.

    Returns
    -------
    None
        The initialized layer expects values shaped ``... x set_size x
        features`` and a mask shaped ``... x set_size``.
    """

    def __init__(
        self,
        summary_dim: int = 64,
        element_widths: Sequence[int] = (64, 64),
        n_context_blocks: int = 2,
        post_widths: Sequence[int] = (64, 64),
        activation: str = "silu",
        include_count_features: bool = False,
        **kwargs,
    ):
        """Create one generic masked set encoder.

        Parameters
        ----------
        summary_dim
            Width of the final set summary.
        element_widths
            Hidden widths used before masked set context is computed.
        n_context_blocks
            Number of equivariant context-injection updates.
        post_widths
            Hidden widths after final masked pooling.
        activation
            Keras activation name used in hidden layers.
        include_count_features
            Whether to append active-fraction and log-count-fraction features.
        **kwargs
            Extra layer settings passed to ``keras.Layer``.

        Returns
        -------
        None
            The layer is initialized for later calls.
        """

        super().__init__(**kwargs)
        self.summary_dim = int(summary_dim)
        self.element_widths = tuple(int(width) for width in element_widths)
        self.n_context_blocks = int(n_context_blocks)
        self.post_widths = tuple(int(width) for width in post_widths)
        self.activation = activation
        self.include_count_features = bool(include_count_features)
        self.element_layers = [
            keras.layers.Dense(
                width,
                activation=self.activation,
                kernel_initializer="he_normal",
            )
            for width in self.element_widths
        ]
        context_width = self.element_widths[-1]
        self.context_layers = [
            keras.layers.Dense(
                context_width,
                activation=self.activation,
                kernel_initializer="he_normal",
            )
            for _ in range(self.n_context_blocks)
        ]
        self.context_norms = [
            keras.layers.LayerNormalization() for _ in range(self.n_context_blocks)
        ]
        self.post_layers = [
            keras.layers.Dense(
                width,
                activation=self.activation,
                kernel_initializer="he_normal",
            )
            for width in self.post_widths
        ]
        self.output_projector = keras.layers.Dense(
            self.summary_dim,
            activation="linear",
            kernel_initializer="he_normal",
        )

    def call(self, values, mask, training: bool = False, **kwargs):
        """Encode active set entries while ignoring padded entries.

        Parameters
        ----------
        values
            Tensor shaped ``... x set_size x features``.
        mask
            Binary active-entry mask shaped ``... x set_size``.
        training
            Whether the layer is being called during training.
        **kwargs
            Accepted for compatibility with Keras layer calls.

        Returns
        -------
        Tensor
            Masked set summary with the set dimension removed.
        """

        del kwargs

        encoded = values
        for layer in self.element_layers:
            encoded = layer(encoded, training=training)

        active_mask = keras.ops.cast(mask, encoded.dtype)
        for layer, norm in zip(self.context_layers, self.context_norms):
            encoded = self._context_update(
                encoded,
                active_mask,
                layer,
                norm,
                training=training,
            )

        pooled = self._masked_mean(encoded, active_mask, axis=-2)
        if self.include_count_features:
            pooled = self._append_count_features(pooled, active_mask)

        summary = pooled
        for layer in self.post_layers:
            summary = layer(summary, training=training)
        return self.output_projector(summary, training=training)

    def _context_update(self, values, mask, layer, norm, training: bool = False):
        """Inject masked set context into active element representations.

        Parameters
        ----------
        values
            Current element representations shaped ``... x set_size x
            features``.
        mask
            Binary active-entry mask shaped ``... x set_size``.
        layer
            Dense layer used to propose the residual element update.
        norm
            Layer normalization applied after the residual update.
        training
            Whether the layer is being called during training.

        Returns
        -------
        Tensor
            Updated element representations with the same shape as ``values``.
        """

        mean = self._masked_mean(values, mask, axis=-2, keepdims=True)
        var = self._masked_variance(values, mask, mean, axis=-2, keepdims=True)
        context = keras.ops.concatenate([mean, var], axis=-1)
        context = context * keras.ops.ones_like(values[..., :1])
        update_input = keras.ops.concatenate([values, context], axis=-1)
        update = layer(update_input, training=training)
        updated = norm(values + update, training=training)
        active_mask = keras.ops.expand_dims(mask, axis=-1)
        return updated * active_mask + values * (1.0 - active_mask)

    @staticmethod
    def _masked_mean(values, mask, axis: int, keepdims: bool = False):
        """Average values over active set entries only.

        Parameters
        ----------
        values
            Encoded values to average.
        mask
            Binary mask matching ``values`` up to the feature dimension.
        axis
            Set dimension to average over.
        keepdims
            Whether to keep the averaged set dimension.

        Returns
        -------
        Tensor
            Masked mean with the selected set dimension removed.
        """

        expanded_mask = keras.ops.expand_dims(mask, axis=-1)
        weighted_values = values * expanded_mask
        numerator = keras.ops.sum(weighted_values, axis=axis, keepdims=keepdims)
        denominator = keras.ops.sum(expanded_mask, axis=axis, keepdims=keepdims)
        denominator = keras.ops.maximum(denominator, keras.ops.ones_like(denominator))
        return numerator / denominator

    @classmethod
    def _masked_variance(cls, values, mask, mean, axis: int, keepdims: bool = False):
        """Return the variance over active set entries only.

        Parameters
        ----------
        values
            Encoded values to summarize.
        mask
            Binary mask matching ``values`` up to the feature dimension.
        mean
            Masked mean of ``values`` with ``keepdims=True`` for broadcasting.
        axis
            Set dimension to summarize.
        keepdims
            Whether to keep the summarized set dimension.

        Returns
        -------
        Tensor
            Masked variance over the selected set dimension.
        """

        centered = values - mean
        return cls._masked_mean(centered * centered, mask, axis=axis, keepdims=keepdims)

    @staticmethod
    def _append_count_features(summary, mask):
        """Append active set-size features to a pooled summary.

        Parameters
        ----------
        summary
            Pooled set summary with shape ``... x features``.
        mask
            Binary active-entry mask with shape ``... x set_size``.

        Returns
        -------
        Tensor
            Summary with ``active_fraction`` and ``log_count_fraction`` added.
        """

        active_count = keras.ops.sum(mask, axis=-1, keepdims=True)
        padded_count = keras.ops.sum(keras.ops.ones_like(mask), axis=-1, keepdims=True)
        active_count = keras.ops.maximum(
            active_count, keras.ops.ones_like(active_count)
        )
        padded_count = keras.ops.maximum(
            padded_count, keras.ops.ones_like(padded_count)
        )
        active_fraction = active_count / padded_count
        log_count_fraction = keras.ops.log(
            active_count + keras.ops.ones_like(active_count)
        ) / keras.ops.log(padded_count + keras.ops.ones_like(padded_count))
        return keras.ops.concatenate(
            [summary, active_fraction, log_count_fraction],
            axis=-1,
        )

    def get_config(self) -> dict:
        """Return Keras-serializable layer settings."""

        config = super().get_config()
        config.update(
            {
                "summary_dim": self.summary_dim,
                "element_widths": self.element_widths,
                "n_context_blocks": self.n_context_blocks,
                "post_widths": self.post_widths,
                "activation": self.activation,
                "include_count_features": self.include_count_features,
            }
        )
        return config


MaskedSetEncoder = MaskedEquivariantSetEncoder


@keras.saving.register_keras_serializable(package="bayesflow_ind")
class MaskedNestedSummary(bf.networks.SummaryNetwork):
    """Summarize nested trial data with explicit padding masks.

    Parameters
    ----------
    summary_dim
        Width of the learned group summary.
    has_trial_mask
        Whether the final input feature is an ``active_trial`` mask. Flexible
        trial hierarchies use this to ignore padded trials and subjects.
    include_count_features
        Whether each nested set encoder should retain active set-size features.
    **kwargs
        Extra layer settings passed to the BayesFlow ``SummaryNetwork`` base.

    Returns
    -------
    None
        The initialized layer expects data shaped
        ``(batch, subjects, trials, features)`` and returns one group summary
        row per batch item.
    """

    def __init__(
        self,
        summary_dim: int = 64,
        has_trial_mask: bool = False,
        include_count_features: bool = False,
        **kwargs,
    ):
        """Create a nested summary network with trial and subject encoders."""

        super().__init__(**kwargs)
        self.summary_dim = int(summary_dim)
        self.has_trial_mask = bool(has_trial_mask)
        self.include_count_features = bool(include_count_features)
        self.trial_encoder = MaskedEquivariantSetEncoder(
            summary_dim=self.summary_dim,
            include_count_features=self.include_count_features,
        )
        self.subject_encoder = MaskedEquivariantSetEncoder(
            summary_dim=self.summary_dim,
            include_count_features=self.include_count_features,
        )

    def call(self, x, training: bool = False, **kwargs):
        """Compress trials within subjects and subjects within groups.

        Parameters
        ----------
        x
            Nested trial data with shape ``batch x subjects x trials x
            features``. When ``has_trial_mask`` is true, the final feature is
            the active-trial mask and is not treated as an observed response.
        training
            Whether the layer is being called during training.
        **kwargs
            Accepted for compatibility with BayesFlow summary-network calls.

        Returns
        -------
        Tensor
            Group-level summary with shape ``batch x summary_dim``.
        """

        del kwargs

        trial_features, trial_mask = self._split_trial_mask(x)
        subject_summaries = self.trial_encoder(
            trial_features,
            trial_mask,
            training=training,
        )
        subject_mask = self._subject_mask(trial_mask)
        return self.subject_encoder(
            subject_summaries,
            subject_mask,
            training=training,
        )

    def _split_trial_mask(self, x):
        """Return substantive trial features and the active-trial mask.

        Parameters
        ----------
        x
            Nested trial tensor passed to ``call``.

        Returns
        -------
        tuple
            ``(trial_features, trial_mask)`` where ``trial_mask`` has shape
            ``batch x subjects x trials``.
        """

        if self.has_trial_mask:
            return x[..., :-1], x[..., -1]
        return x, keras.ops.ones_like(x[..., 0])

    @staticmethod
    def _subject_mask(trial_mask):
        """Return an active-subject mask inferred from active trials.

        Parameters
        ----------
        trial_mask
            Active-trial mask with shape ``batch x subjects x trials``.

        Returns
        -------
        Tensor
            Active-subject mask with shape ``batch x subjects``.
        """

        return keras.ops.max(trial_mask, axis=-1)

    def get_config(self) -> dict:
        """Return Keras-serializable layer settings."""

        config = super().get_config()
        config.update(
            {
                "summary_dim": self.summary_dim,
                "has_trial_mask": self.has_trial_mask,
                "include_count_features": self.include_count_features,
            }
        )
        return config


class HierarchicalWorkflow:
    """Build a group-level BayesFlow workflow from a prior and simulator.

    Parameters
    ----------
    name
        Short model name attached to the BayesFlow workflow.
    priors
        Project prior specification using ``mean``, ``sd``, and ``link`` for
        stochastic parameters. Scalar entries are treated as subject-level
        constants.
    simulator
        Function called as ``simulator(**params, n_trials=..., rng=...,
        **simulator_kwargs)``. For aggregate workflows, it should return one
        subject data row. For trial workflows, it should return one row per
        trial for that subject.
    observation
        Observation contract. Use ``"aggregate"`` when the simulator returns
        one fixed-width row per subject, or ``"trial"`` when the simulator
        returns one row per trial for each subject.
    simulator_kwargs
        Constant keyword arguments passed to ``simulator`` on every simulation.
    data_width
        Number of features returned by ``simulator`` before optional design
        columns are appended. For new code, prefer ``obs_names`` so the column
        meanings are visible.
    obs_names
        Names of simulator output columns. When supplied, ``data_width`` is
        inferred from ``len(obs_names)``.
    n_subjects, n_subjects_range
        Fixed subject count or range ``(low, high)``. Provide exactly one.
    n_trials, n_trials_range
        Fixed trial count or range ``(low, high)``. Provide exactly one.
    include_trial_feature
        Whether to append the subject trial count to each data row.
    include_mask
        Whether to append an active-subject mask. If omitted, the mask is added
        for flexible aggregate designs and omitted for fixed designs and nested
        trial designs.
    keep_subject_truth
        Subject-level parameter names to save as ``<param>_subj`` truth arrays.
        Use ``None`` to save all stochastic subject-level parameters, or an
        empty list to save none.
    raw_data_key
        Optional output key used to save untransformed simulator rows.
    row_transform
        Optional function that formats one simulator row before design columns
        are appended.
    trial_feature_scale
        Optional divisor used when writing the trial-count feature.
    input_format
        Optional input-format helper. If supplied, it formats
        simulator rows and trial counts before padding and masking.
    summary_dim, n_coupling_layers
        BayesFlow network settings.
    transform_samples
        Optional posterior transform function.

    Returns
    -------
    None
        The initialized object exposes ``workflow`` and ``train_workflow``.
    """

    workflow_level = "hierarchical"

    def __init__(
        self,
        name: str,
        priors: Mapping,
        simulator: Callable,
        observation: str | None,
        data_width: int | None = None,
        *,
        simulator_kwargs: Mapping | None = None,
        obs_names: Sequence[str] | None = None,
        n_subjects: int | None = None,
        n_subjects_range: Sequence[int] | None = None,
        n_trials: int | None = None,
        n_trials_range: Sequence[int] | None = None,
        include_trial_feature: bool = False,
        include_mask: bool | None = None,
        keep_subject_truth: Sequence[str] | None = None,
        raw_data_key: str | None = None,
        row_transform: Callable | None = None,
        trial_feature_scale: float | None = None,
        input_format: InputFormat | None = None,
        summary_dim: int = 64,
        n_coupling_layers: int = 10,
        transform_samples: Callable | None = None,
    ):
        self.model_name = self._check_name(name)
        self.priors = self._check_priors(priors)
        self.observation = validate_observation(observation, "HierarchicalWorkflow")
        self.param_names = self._check_param_names(
            self._hierarchical_inference_variables(self.priors)
        )
        self._simulator_fn = self._check_callable(
            simulator,
            "simulator",
        )
        self.simulator_kwargs = dict(simulator_kwargs or {})
        self.obs_names = SimpleWorkflow._resolve_obs_names(obs_names, data_width)
        if self.obs_names is None:
            self.data_width = SimpleWorkflow._check_positive_int(
                data_width, "data_width"
            )
            self.obs_names = [f"x{i}" for i in range(self.data_width)]
        else:
            self.data_width = len(self.obs_names)
        (
            self.subject_design,
            self.n_subjects,
            self.n_subjects_range,
        ) = self._resolve_design_count(
            fixed_value=n_subjects,
            range_value=n_subjects_range,
            fixed_name="n_subjects",
            range_name="n_subjects_range",
        )
        self.trial_design, self.n_trials, self.n_trials_range = (
            self._resolve_design_count(
                fixed_value=n_trials,
                range_value=n_trials_range,
                fixed_name="n_trials",
                range_name="n_trials_range",
            )
        )
        self.max_subjects = self._resolve_max_subjects()
        self.max_trials = self._resolve_max_trials()
        self.include_trial_feature = bool(include_trial_feature)
        self.include_mask = (
            self.subject_design == "flex" and self.observation == "aggregate"
            if include_mask is None
            else bool(include_mask)
        )
        self.keep_subject_truth = self._resolve_keep_subject_truth(keep_subject_truth)
        self.raw_data_key = raw_data_key
        self.row_transform = row_transform
        self.trial_feature_scale = trial_feature_scale
        self.input_format = SimpleWorkflow._check_input_format(input_format)
        self.workflow_family = f"{self.subject_design}_hierarchical"
        self.subject_id_mode = "exchangeable"
        self.summary_dim = int(summary_dim)
        self.n_coupling_layers = int(n_coupling_layers)
        self._transform_samples = transform_samples

        self._build_workflow()
        self.random_workflow = None
        self.random_validation_data = None
        self.validation_data = None

    def _draw_independent_group_prior(self, rng=np.random) -> dict[str, float]:
        """Draw one independent hierarchical group prior.

        Parameters
        ----------
        rng
            NumPy-compatible random generator.

        Returns
        -------
        dict[str, float]
            Raw group means, raw log sigmas, and public group parameters.
        """

        return self._draw_group_prior_from_spec(self.priors, rng)

    def _resolve_keep_subject_truth(
        self,
        keep_subject_truth: Sequence[str] | None,
    ) -> list[str]:
        """Return validated subject-level truth names to store.

        Parameters
        ----------
        keep_subject_truth
            ``None`` to store all stochastic subject-level parameters, an empty
            sequence to store none, or a sequence of parameter names to store.

        Returns
        -------
        list[str]
            Parameter names that simulation should save as ``<param>_subj``.
        """

        stochastic_names = [
            name for name, spec in self.priors.items() if isinstance(spec, dict)
        ]
        if keep_subject_truth is None:
            return stochastic_names

        names = list(keep_subject_truth)
        missing = [name for name in names if name not in stochastic_names]
        if missing:
            raise ValueError(
                "keep_subject_truth contains parameters that are not stochastic "
                f"subject-level prior keys: {missing}. Available parameters: "
                f"{stochastic_names}"
            )
        return names

    def _draw_independent_subject_params(
        self,
        group_params: Mapping[str, float],
        rng=np.random,
    ) -> dict[str, float]:
        """Draw one subject from this workflow's independent group prior.

        Parameters
        ----------
        group_params
            Raw group means and log sigmas from
            ``_draw_independent_group_prior``.
        rng
            NumPy-compatible random generator.

        Returns
        -------
        dict[str, float]
            Public subject-level parameters, plus scalar constants.
        """

        return self._draw_subject_params_from_spec(self.priors, group_params, rng)

    @staticmethod
    def _hierarchical_inference_variables(priors: Mapping) -> list[str]:
        """Return raw group-level variables inferred by BayesFlow.

        Parameters
        ----------
        priors
            Project prior specification using ``mean``, ``sd``, and ``link``
            for stochastic parameters.

        Returns
        -------
        list[str]
            Raw group mean keys, plus raw log-sigma keys when the prior makes
            the group standard deviation stochastic.
        """

        from bami.inference.priors import (
            is_stochastic_mean,
            is_stochastic_sd,
            log_sigma_key,
            mu_raw_key,
        )

        inference_variables = []
        for param_name, spec in priors.items():
            if not isinstance(spec, dict):
                continue
            if is_stochastic_mean(spec):
                inference_variables.append(mu_raw_key(param_name))
            if is_stochastic_sd(spec):
                inference_variables.append(log_sigma_key(param_name))
        return inference_variables

    @staticmethod
    def _draw_group_prior_from_spec(
        priors: Mapping,
        rng=np.random,
    ) -> dict[str, float]:
        """Draw one independent hierarchical group prior from a spec.

        Parameters
        ----------
        priors
            Project prior specification using ``mean``, ``sd``, and ``link``.
        rng
            NumPy-compatible random generator.

        Returns
        -------
        dict[str, float]
            Raw group means, raw log sigmas, and public group parameters.
        """

        from bami.inference.priors import (
            apply_link,
            draw_group_mean_raw,
            draw_group_sd_raw,
            log_sigma_key,
            mu_raw_key,
        )

        group_params = {}
        for param_name, spec in priors.items():
            if not isinstance(spec, dict):
                continue
            link = spec.get("link", "identity")
            mu_raw = draw_group_mean_raw(param_name, spec, rng)
            sigma = draw_group_sd_raw(param_name, spec, rng)

            group_params[mu_raw_key(param_name)] = mu_raw
            group_params[log_sigma_key(param_name)] = np.log(sigma)
            group_params[f"{param_name}_mu"] = apply_link(mu_raw, link)
            group_params[f"{param_name}_sigma"] = sigma
        return group_params

    @staticmethod
    def _draw_subject_params_from_spec(
        priors: Mapping,
        group_params: Mapping[str, float],
        rng=np.random,
    ) -> dict[str, float]:
        """Draw one subject from independent group-level priors.

        Parameters
        ----------
        priors
            Project prior specification using ``mean``, ``sd``, and ``link``.
        group_params
            Raw group means and log sigmas from
            ``_draw_group_prior_from_spec``.
        rng
            NumPy-compatible random generator.

        Returns
        -------
        dict[str, float]
            Public subject-level parameters, plus scalar constants from
            ``priors``.
        """

        from bami.inference.priors import apply_link, log_sigma_key, mu_raw_key

        subject_params = {}
        for param_name, spec in priors.items():
            if isinstance(spec, dict):
                link = spec.get("link", "identity")
                mu_raw = group_params[mu_raw_key(param_name)]
                sigma = np.exp(group_params[log_sigma_key(param_name)])
                subject_raw = rng.normal(loc=mu_raw, scale=sigma)
                subject_params[param_name] = apply_link(subject_raw, link)
            else:
                subject_params[param_name] = float(spec)
        return subject_params

    @classmethod
    def _resolve_design_count(
        cls,
        fixed_value: int | None,
        range_value: Sequence[int] | None,
        fixed_name: str,
        range_name: str,
    ) -> tuple[str, int | None, tuple[int, int] | None]:
        """Validate one fixed-or-flexible design count.

        Parameters
        ----------
        fixed_value
            Fixed count value, or ``None`` when using a range.
        range_value
            Two-value range, or ``None`` when using a fixed count.
        fixed_name, range_name
            Setting names used in error messages.

        Returns
        -------
        tuple[str, int | None, tuple[int, int] | None]
            Design label and validated count settings.
        """

        if fixed_value is not None and range_value is not None:
            raise ValueError(f"Provide either {fixed_name} or {range_name}, not both.")
        if fixed_value is None and range_value is None:
            raise ValueError(f"Provide one of {fixed_name} or {range_name}.")
        if range_value is not None:
            return "flex", None, SimpleWorkflow._check_range(range_value, range_name)
        checked = SimpleWorkflow._check_positive_int(fixed_value, fixed_name)
        return "fixed", checked, None

    def _resolve_max_subjects(self) -> int:
        """Return the padded subject-row count for simulated datasets.

        Returns
        -------
        int
            Fixed subject count or the largest possible flexible subject count.
        """

        if self.subject_design == "fixed":
            return int(self.n_subjects)
        return int(self.n_subjects_range[1] - 1)

    def _resolve_max_trials(self) -> int:
        """Return the padded trial count for nested trial observations.

        Returns
        -------
        int
            Fixed trial count or the largest possible flexible trial count.
        """

        if self.trial_design == "fixed":
            return int(self.n_trials)
        return int(self.n_trials_range[1] - 1)

    def _draw_count(
        self,
        design: str,
        fixed_value: int | None,
        range_value: tuple[int, int] | None,
        rng=np.random,
    ) -> int:
        """Draw one fixed or flexible count.

        Parameters
        ----------
        design
            Either ``"fixed"`` or ``"flex"``.
        fixed_value
            Fixed count used when ``design`` is ``"fixed"``.
        range_value
            Range used when ``design`` is ``"flex"``.
        rng
            NumPy-compatible random module or generator.

        Returns
        -------
        int
            Fixed or sampled count.
        """

        if design == "fixed":
            return int(fixed_value)
        low, high = range_value
        return int(rng.randint(low, high))

    def _feature_width(self) -> int:
        """Return the final subject-row feature width.

        Returns
        -------
        int
            Base simulator width plus optional design columns.
        """

        width = self._encoded_data_width()
        if self.input_format is None and self.include_trial_feature:
            width += 1
        if self.include_mask:
            width += 1
        return width

    def _trial_feature_width(self) -> int:
        """Return the feature width for one nested trial row.

        Returns
        -------
        int
            Base trial-feature count plus an active-trial mask for flexible
            trial designs.
        """

        width = self.data_width
        if self.trial_design == "flex":
            width += 1
        return width

    def _encoded_data_width(self) -> int:
        """Return the row width after observation-level encoding.

        Returns
        -------
        int
            Base simulator width or ``input_format`` output width.
        """

        if self.input_format is None:
            return self.data_width
        return self.input_format.output_width(self.data_width)

    def _format_trial_feature(self, n_trials: int) -> float:
        """Return the trial-count feature for one subject row.

        Parameters
        ----------
        n_trials
            Number of responses for the subject.

        Returns
        -------
        float
            Raw or scaled trial count depending on workflow settings.
        """

        if self.trial_feature_scale is None:
            return float(n_trials)
        return float(n_trials) / float(self.trial_feature_scale)

    def _simulate_dataset(self, **group_params) -> dict[str, np.ndarray]:
        """Simulate one group-level dataset.

        Parameters
        ----------
        **group_params
            Group-level parameter dictionary from ``draw_group_prior``.

        Returns
        -------
        dict[str, numpy.ndarray]
            Data array and flexible-design metadata.
        """

        if self.observation == "trial":
            return self._simulate_trial_dataset(**group_params)
        return self._simulate_aggregate_dataset(**group_params)

    def _simulate_aggregate_dataset(self, **group_params) -> dict[str, np.ndarray]:
        """Simulate one aggregate group-level dataset.

        Parameters
        ----------
        **group_params
            Group-level parameter dictionary from ``draw_group_prior``.

        Returns
        -------
        dict[str, numpy.ndarray]
            Subject-by-feature data array and flexible-design metadata.
        """

        n_subjects = self._draw_count(
            self.subject_design,
            self.n_subjects,
            self.n_subjects_range,
            np.random,
        )
        data = np.zeros((self.max_subjects, self._feature_width()), dtype=np.float32)
        raw_data = None
        if self.raw_data_key is not None:
            raw_data = np.zeros(
                (self.max_subjects, self.data_width),
                dtype=np.float32,
            )
        subj_truth = {
            name: np.full(self.max_subjects, np.nan, dtype=np.float32)
            for name in self.keep_subject_truth
        }

        for subject_id in range(n_subjects):
            params = self._draw_independent_subject_params(group_params, np.random)
            n_trials = self._draw_count(
                self.trial_design,
                self.n_trials,
                self.n_trials_range,
                np.random,
            )
            row = self._simulator_fn(
                **params,
                n_trials=n_trials,
                rng=np.random,
                **self.simulator_kwargs,
            )
            row_arr = np.asarray(row, dtype=np.float32)
            if row_arr.shape != (self.data_width,):
                raise ValueError(
                    "simulator must return a row with shape " f"({self.data_width},)."
                )
            if raw_data is not None:
                raw_data[subject_id] = row_arr
            if self.input_format is not None:
                row_arr = self.input_format.encode(row_arr, n_trials)
                if row_arr.shape != (self._encoded_data_width(),):
                    raise ValueError(
                        "input_format must return a row with shape "
                        f"({self._encoded_data_width()},)."
                    )
            elif self.row_transform is not None:
                row_arr = np.asarray(
                    self.row_transform(row_arr, n_trials=n_trials, model=self),
                    dtype=np.float32,
                )
                if row_arr.shape != (self.data_width,):
                    raise ValueError(
                        "row_transform must return a row with shape "
                        f"({self.data_width},)."
                    )

            col = 0
            encoded_width = self._encoded_data_width()
            data[subject_id, col : col + encoded_width] = row_arr
            col += encoded_width
            if self.input_format is None and self.include_trial_feature:
                data[subject_id, col] = self._format_trial_feature(n_trials)
                col += 1
            if self.include_mask:
                data[subject_id, col] = 1.0
            for param_name in self.keep_subject_truth:
                if param_name in params:
                    subj_truth[param_name][subject_id] = float(params[param_name])

        out = dict(group_params)
        out["data"] = data
        if self.raw_data_key is not None:
            out[self.raw_data_key] = raw_data
        for param_name, values in subj_truth.items():
            out[f"{param_name}_subj"] = values
        if self.subject_design == "flex":
            out["n_subjects"] = np.array(n_subjects, dtype=np.int32)
        return out

    def _simulate_trial_dataset(self, **group_params) -> dict[str, np.ndarray]:
        """Simulate one nested subject-by-trial group-level dataset.

        Parameters
        ----------
        **group_params
            Group-level parameter dictionary from ``draw_group_prior``.

        Returns
        -------
        dict[str, numpy.ndarray]
            Trial-level data with shape ``subjects x trials x features`` plus
            flexible-design metadata.
        """

        n_subjects = self._draw_count(
            self.subject_design,
            self.n_subjects,
            self.n_subjects_range,
            np.random,
        )
        data = np.zeros(
            (self.max_subjects, self.max_trials, self._trial_feature_width()),
            dtype=np.float32,
        )
        subj_truth = {
            name: np.full(self.max_subjects, np.nan, dtype=np.float32)
            for name in self.keep_subject_truth
        }

        for subject_id in range(n_subjects):
            params = self._draw_independent_subject_params(group_params, np.random)
            n_trials = self._draw_count(
                self.trial_design,
                self.n_trials,
                self.n_trials_range,
                np.random,
            )
            rows = self._simulator_fn(
                **params,
                n_trials=n_trials,
                rng=np.random,
                **self.simulator_kwargs,
            )
            rows_arr = np.asarray(rows, dtype=np.float32)
            if rows_arr.ndim == 1:
                rows_arr = rows_arr[:, np.newaxis]
            if rows_arr.shape != (int(n_trials), self.data_width):
                raise ValueError(
                    "trial simulator must return shape "
                    f"({n_trials}, {self.data_width}) when "
                    "observation='trial'."
                )

            data[subject_id, :n_trials, : self.data_width] = rows_arr
            if self.trial_design == "flex":
                data[subject_id, :n_trials, self.data_width] = 1.0
            for param_name in self.keep_subject_truth:
                if param_name in params:
                    subj_truth[param_name][subject_id] = float(params[param_name])

        out = dict(group_params)
        out["data"] = data
        for param_name, values in subj_truth.items():
            out[f"{param_name}_subj"] = values
        if self.subject_design == "flex":
            out["n_subjects"] = np.array(n_subjects, dtype=np.int32)
        return out

    def _random_group_condition_keys(self) -> list[str]:
        """Return group parameters used to condition random-effect sampling.

        Returns
        -------
        list[str]
            Raw group mean and log-sigma keys for each hierarchical parameter.
        """

        from bami.inference.priors import log_sigma_key, mu_raw_key

        keys = []
        for param_name, spec in self.priors.items():
            if not isinstance(spec, dict):
                continue
            keys.append(mu_raw_key(param_name))
            keys.append(log_sigma_key(param_name))
        return keys

    def _random_z_key(self, param_name: str) -> str:
        """Return the standardized random-effect key for one parameter.

        Parameters
        ----------
        param_name
            Public parameter name from the hierarchical prior.

        Returns
        -------
        str
            Key used for the standard-normal subject deviation.
        """

        return f"{param_name}_z"

    def _random_inference_variables(self) -> list[str]:
        """Return standardized variables inferred by the random workflow.

        Returns
        -------
        list[str]
            Standardized subject-deviation keys, one for each hierarchical
            parameter.
        """

        return [
            self._random_z_key(param_name)
            for param_name, spec in self.priors.items()
            if isinstance(spec, dict)
        ]

    def _draw_random_subject_prior(self, rng=np.random) -> dict[str, float]:
        """Draw one group and one subject for random-effect training.

        Parameters
        ----------
        rng
            NumPy-compatible random generator.

        Returns
        -------
        dict[str, float]
            Group raw parameters, standardized deviations, and public subject
            parameters for simulator calls.
        """

        from bami.inference.priors import apply_link, log_sigma_key, mu_raw_key

        group_params = self._draw_independent_group_prior(rng)
        out = dict(group_params)
        for param_name, spec in self.priors.items():
            if isinstance(spec, dict):
                mu_raw = group_params[mu_raw_key(param_name)]
                sigma = np.exp(group_params[log_sigma_key(param_name)])
                z_value = rng.normal(loc=0.0, scale=1.0)
                subj_raw = mu_raw + sigma * z_value
                out[self._random_z_key(param_name)] = z_value
                out[param_name] = apply_link(subj_raw, spec.get("link", "identity"))
            else:
                out[param_name] = float(spec)
        return out

    def _random_aggregate_feature_width(self) -> int:
        """Return aggregate feature width for one random-effect subject.

        Returns
        -------
        int
            Encoded subject-row width without the parent workflow's
            active-subject mask.
        """

        width = self._encoded_data_width()
        if self.input_format is None and self.include_trial_feature:
            width += 1
        return width

    def _simulate_random_subject(self, **params) -> dict[str, np.ndarray]:
        """Simulate one subject for the random-effect workflow.

        Parameters
        ----------
        **params
            Group raw parameters, subject raw parameters, public subject
            parameters, and scalar constants from the random prior draw.

        Returns
        -------
        dict[str, numpy.ndarray]
            Single-subject data formatted for BayesFlow.
        """

        if self.observation == "trial":
            return self._simulate_random_trial_subject(**params)
        return self._simulate_random_aggregate_subject(**params)

    def _simulate_random_aggregate_subject(self, **params) -> dict[str, np.ndarray]:
        """Simulate one aggregate subject for random-effect training.

        Parameters
        ----------
        **params
            Public subject parameters and scalar constants for the simulator.

        Returns
        -------
        dict[str, numpy.ndarray]
            Data array with shape ``(1, random_feature_width)``.
        """

        subject_params = self._public_subject_params(params)
        n_trials = self._draw_count(
            self.trial_design,
            self.n_trials,
            self.n_trials_range,
            np.random,
        )
        row = self._simulator_fn(
            **subject_params,
            n_trials=n_trials,
            rng=np.random,
            **self.simulator_kwargs,
        )
        row_arr = np.asarray(row, dtype=np.float32)
        if row_arr.shape != (self.data_width,):
            raise ValueError(
                "simulator must return a row with shape " f"({self.data_width},)."
            )
        if self.input_format is not None:
            row_arr = self.input_format.encode(row_arr, n_trials)
        elif self.row_transform is not None:
            row_arr = np.asarray(
                self.row_transform(row_arr, n_trials=n_trials, model=self),
                dtype=np.float32,
            )
        if self.input_format is None and self.include_trial_feature:
            row_arr = np.concatenate(
                [
                    row_arr,
                    np.array([self._format_trial_feature(n_trials)], dtype=np.float32),
                ]
            )
        expected_width = self._random_aggregate_feature_width()
        if row_arr.shape != (expected_width,):
            raise ValueError(
                "random aggregate subject data must have shape " f"({expected_width},)."
            )
        return {"data": row_arr[np.newaxis, :]}

    def _simulate_random_trial_subject(self, **params) -> dict[str, np.ndarray]:
        """Simulate one trial-level subject for random-effect training.

        Parameters
        ----------
        **params
            Public subject parameters and scalar constants for the simulator.

        Returns
        -------
        dict[str, numpy.ndarray]
            Trial data with shape ``(max_trials, trial_feature_width)``.
        """

        subject_params = self._public_subject_params(params)
        n_trials = self._draw_count(
            self.trial_design,
            self.n_trials,
            self.n_trials_range,
            np.random,
        )
        rows = self._simulator_fn(
            **subject_params,
            n_trials=n_trials,
            rng=np.random,
            **self.simulator_kwargs,
        )
        rows_arr = np.asarray(rows, dtype=np.float32)
        if rows_arr.ndim == 1:
            rows_arr = rows_arr[:, np.newaxis]
        if rows_arr.shape != (int(n_trials), self.data_width):
            raise ValueError(
                "trial simulator must return shape "
                f"({n_trials}, {self.data_width}) when observation='trial'."
            )
        if self.trial_design == "fixed":
            return {"data": rows_arr}

        data = np.zeros(
            (self.max_trials, self._trial_feature_width()),
            dtype=np.float32,
        )
        data[:n_trials, : self.data_width] = rows_arr
        data[:n_trials, self.data_width] = 1.0
        return {"data": data}

    def _public_subject_params(self, params: Mapping) -> dict[str, float]:
        """Return simulator parameters from a random-effect prior draw.

        Parameters
        ----------
        params
            Prior draw containing public subject parameters, scalar constants,
            and raw inference variables.

        Returns
        -------
        dict[str, float]
            Public parameters suitable for ``simulator(**params)``.
        """

        out = {}
        for param_name, spec in self.priors.items():
            if isinstance(spec, dict) or np.isscalar(spec):
                out[param_name] = float(params[param_name])
        return out

    def _build_random_workflow(self) -> None:
        """Build the BayesFlow workflow for subject-level random effects.

        Returns
        -------
        None
            Sets ``random_workflow`` on this object.
        """

        def _draw_random_prior():
            """Draw one group and one subject-level prior sample."""

            return self._draw_random_subject_prior(np.random)

        def _simulate_random_subject(**params):
            """Simulate one subject from a random prior draw."""

            return self._simulate_random_subject(**params)

        self.random_simulator = bf.make_simulator(
            [_draw_random_prior, _simulate_random_subject]
        )
        if self.observation == "aggregate":
            random_summary_network = bf.networks.DeepSet(summary_dim=self.summary_dim)
        else:
            random_summary_network = bf.networks.DeepSet(summary_dim=self.summary_dim)
        random_inference_network = bf.networks.CouplingFlow(
            n_coupling_layers=self.n_coupling_layers
        )
        self.random_workflow = bf.BasicWorkflow(
            simulator=self.random_simulator,
            inference_network=random_inference_network,
            summary_network=random_summary_network,
            inference_variables=self._random_inference_variables(),
            inference_conditions=self._random_group_condition_keys(),
            summary_variables=["data"],
        )
        self.random_workflow.transform_posterior_samples = self.convert_random_posterior
        self.random_workflow.workflow_level = "random"
        self.random_workflow.workflow_family = f"{self.workflow_family}_random"
        self.random_workflow.model_name = self.model_name
        self.random_workflow.observation = self.observation
        self.random_workflow.input_format = self.input_format
        self.random_workflow.input_format_metadata = self._input_format_metadata()
        self.random_workflow.obs_names = self._workflow_obs_names()

    def _build_workflow(self) -> None:
        """Build the simulator, networks, and BayesFlow workflow.

        Returns
        -------
        None
            Sets ``simulator``, ``summary_network``, ``inference_network``, and
            ``workflow`` on this object.
        """

        def _draw_group_prior():
            """Draw one group-level prior sample."""

            return self._draw_independent_group_prior(np.random)

        def _simulate_dataset(**group_params):
            """Simulate one group-level dataset."""

            return self._simulate_dataset(**group_params)

        self.simulator = bf.make_simulator([_draw_group_prior, _simulate_dataset])
        self.summary_network = self._build_summary_network()
        self.inference_network = bf.networks.CouplingFlow(
            n_coupling_layers=self.n_coupling_layers
        )
        self.workflow = bf.BasicWorkflow(
            simulator=self.simulator,
            inference_network=self.inference_network,
            summary_network=self.summary_network,
            inference_variables=self.param_names,
            inference_conditions=None,
            summary_variables=["data"],
        )
        self.workflow.transform_posterior_samples = self.convert_posterior
        self.workflow.workflow_level = self.workflow_level
        self.workflow.workflow_family = self.workflow_family
        self.workflow.subject_design = self.subject_design
        self.workflow.trial_design = self.trial_design
        self.workflow.model_name = self.model_name
        self.workflow.observation = self.observation
        self.workflow.indexed_subject_recovery_aligned = False
        self.workflow.subject_id_mode = "exchangeable"
        self.workflow.input_format = self.input_format
        self.workflow.input_format_metadata = self._input_format_metadata()
        self.workflow.obs_names = self._workflow_obs_names()

    def _workflow_obs_names(self) -> list[str]:
        """Return feature names exposed on the BayesFlow workflow.

        Returns
        -------
        list[str]
            Base observation names, plus the active-trial mask for flexible
            nested trial data.
        """

        names = list(self.obs_names)
        if self.observation == "trial" and self.trial_design == "flex":
            names.append("active_trial")
        return names

    def _build_summary_network(self):
        """Build the BayesFlow summary network for this observation contract.

        Returns
        -------
        object
            One DeepSet for aggregate subject rows, or a mask-aware nested
            summary network for subject-by-trial rows.
        """

        if self.observation == "aggregate":
            return bf.networks.DeepSet(summary_dim=self.summary_dim)
        return MaskedNestedSummary(
            summary_dim=self.summary_dim,
            has_trial_mask=self.trial_design == "flex",
            include_count_features=(
                self.trial_design == "flex" or self.subject_design == "flex"
            ),
        )

    def convert_posterior(self, samples: dict) -> dict:
        """Transform posterior samples when a transform function is supplied.

        Parameters
        ----------
        samples
            Raw posterior sample dictionary from BayesFlow.

        Returns
        -------
        dict
            Transformed posterior samples, or the original samples when no
            transform was supplied.
        """

        if self._transform_samples is None:
            return samples
        return self._transform_samples(samples, self.priors)

    def convert_random_posterior(self, samples: dict) -> dict:
        """Return random-workflow samples without changing the ``z`` scale.

        Parameters
        ----------
        samples
            Posterior sample dictionary from the random workflow.

        Returns
        -------
        dict
            Copy of ``samples``. Public subject parameters are created later
            because they require paired group posterior draws.
        """

        return dict(samples)

    def simulate(self, n_datasets: int) -> Mapping[str, np.ndarray]:
        """Simulate datasets from this hierarchical workflow.

        Parameters
        ----------
        n_datasets
            Number of group-level datasets to draw from the workflow prior and
            simulator.

        Returns
        -------
        Mapping[str, numpy.ndarray]
            Simulated group data, group parameter truth arrays, and any saved
            subject-level truth arrays using the workflow's data-shape
            contract.
        """

        return self.workflow.simulate(n_datasets)

    def sample_group_posterior(
        self,
        test_data: Mapping[str, np.ndarray],
        num_samples: int,
        approximator_kwargs: Mapping | None = None,
        sample_batch_size: int | None = None,
    ) -> Mapping[str, np.ndarray]:
        """Draw group-level posterior samples for this hierarchy.

        Parameters
        ----------
        test_data
            Observed or simulated hierarchy data dictionary passed to the
            trained BayesFlow workflow. In examples this is often created with
            ``model.simulate(n_datasets)``.
        num_samples
            Number of group-level posterior draws to request for each dataset.
        approximator_kwargs
            Optional keyword arguments forwarded to BayesFlow's
            ``workflow.sample`` method.
        sample_batch_size
            Optional number of datasets to sample at once. Use this when many
            datasets would otherwise exceed accelerator memory.

        Returns
        -------
        Mapping[str, numpy.ndarray]
            Group-level posterior draws. When this workflow defines a posterior
            transform, returned values use researcher-facing parameter names
            and scales.
        """

        from bami.workflows._sampling import _sample_posterior

        return _sample_posterior(
            workflow=self.workflow,
            test_data=test_data,
            num_samples=num_samples,
            approximator_kwargs=approximator_kwargs,
            sample_batch_size=sample_batch_size,
        )

    def plot_population_recovery(
        self,
        n_datasets: int,
        num_samples: int,
        params: str | Sequence[str] | None = None,
        metrics: str | Sequence[str] = "corr",
        n_cols: int = 3,
    ):
        """Plot group-level parameter recovery for this hierarchy.

        The method simulates group datasets from the workflow prior, samples
        group-level posteriors, and plots simulated group parameters against
        posterior means. It is intended for diagnosing one fitted hierarchical
        model, not for comparing multiple models.

        Parameters
        ----------
        n_datasets
            Number of simulated group datasets used for the diagnostic plot.
        num_samples
            Number of group posterior draws per simulated dataset.
        params
            Optional population parameter key or keys to plot, such as
            ``"c_mu"`` or ``"c_sigma"``. By default all available public group
            keys are shown.
        metrics
            Metric name or names shown in each panel title. Supported values
            are ``corr``, ``ccc``, and ``rmse``.
        n_cols
            Maximum number of columns in the plot grid.

        Returns
        -------
        matplotlib.figure.Figure
            Population parameter recovery figure.
        """

        from bami.evaluation.diagnostics import plot_population_recovery

        return plot_population_recovery(
            self,
            n_datasets=n_datasets,
            num_samples=num_samples,
            params=params,
            metrics=metrics,
            n_cols=n_cols,
        )

    def _group_sample_array(
        self,
        group_samples: Mapping[str, np.ndarray],
        key: str,
        n_datasets: int | None = None,
    ) -> np.ndarray:
        """Return a validated group posterior sample array.

        Parameters
        ----------
        group_samples
            Posterior samples returned by ``sample_group_posterior``.
        key
            Required raw group parameter key.
        n_datasets
            Optional expected dataset count.

        Returns
        -------
        numpy.ndarray
            Array with shape ``(n_datasets, n_samples)``.
        """

        if key not in group_samples:
            raise ValueError(
                "group_samples must contain raw group keys such as "
                f"'{key}'. Use model.sample_group_posterior(...) to create them."
            )
        arr = np.asarray(group_samples[key], dtype=np.float32)
        while arr.ndim > 2 and arr.shape[-1] == 1:
            arr = np.squeeze(arr, axis=-1)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.ndim != 2:
            raise ValueError(
                f"group_samples['{key}'] must have shape " "(n_datasets, n_samples)."
            )
        if n_datasets is not None and arr.shape[0] != n_datasets:
            raise ValueError(
                "observed_data and group_samples must contain the same number "
                f"of datasets. Got {n_datasets} observed datasets and "
                f"{arr.shape[0]} group-sample datasets."
            )
        return arr

    def _fixed_group_condition_value(self, key: str) -> float | None:
        """Return a fixed group condition value from the prior.

        Parameters
        ----------
        key
            Raw group condition key such as ``theta_mu_raw`` or
            ``theta_log_sigma``.

        Returns
        -------
        float or None
            Fixed raw group value, or ``None`` when posterior samples are
            required.
        """

        from bami.inference.priors import is_distribution_value

        if key.endswith("_mu_raw"):
            param_name = key.removesuffix("_mu_raw")
            spec = self.priors.get(param_name)
            if not isinstance(spec, dict):
                return None
            mean_spec = spec["mean"]
            if is_distribution_value(mean_spec):
                return None
            return float(mean_spec)
        if key.endswith("_log_sigma"):
            param_name = key.removesuffix("_log_sigma")
            spec = self.priors.get(param_name)
            if not isinstance(spec, dict):
                return None
            sd_spec = spec["sd"]
            if is_distribution_value(sd_spec):
                return None
            return float(np.log(float(sd_spec)))
        return None

    def _group_condition_array(
        self,
        group_samples: Mapping[str, np.ndarray],
        key: str,
        n_datasets: int,
        n_samples: int,
    ) -> np.ndarray:
        """Return group conditions from posterior samples or fixed priors.

        Parameters
        ----------
        group_samples
            Posterior samples returned by ``sample_group_posterior``.
        key
            Raw group condition key.
        n_datasets, n_samples
            Expected output dimensions.

        Returns
        -------
        numpy.ndarray
            Group condition array with shape ``(n_datasets, n_samples)``.
        """

        if key in group_samples:
            return self._group_sample_array(group_samples, key, n_datasets)
        fixed_value = self._fixed_group_condition_value(key)
        if fixed_value is None:
            raise ValueError(
                "group_samples must contain raw group keys such as "
                f"'{key}'. Use model.sample_group_posterior(...) to create them."
            )
        return np.full((n_datasets, n_samples), fixed_value, dtype=np.float32)

    def _group_sample_shape(
        self,
        group_samples: Mapping[str, np.ndarray],
    ) -> tuple[int, int]:
        """Return dataset and sample counts from raw group posterior samples.

        Parameters
        ----------
        group_samples
            Posterior samples returned by ``sample_group_posterior``.

        Returns
        -------
        tuple[int, int]
            Number of datasets and paired posterior draws.
        """

        keys = self._random_group_condition_keys()
        if not keys:
            raise ValueError("No hierarchical parameters are available to sample.")
        sample_key = next((key for key in keys if key in group_samples), None)
        if sample_key is None:
            raise ValueError(
                "group_samples must contain at least one sampled raw group key."
            )
        arr = self._group_sample_array(group_samples, sample_key)
        return int(arr.shape[0]), int(arr.shape[1])

    def _normalize_random_observed_data(
        self, observed_data
    ) -> tuple[np.ndarray, list[int]]:
        """Return observed subjects in a rectangular data array.

        Parameters
        ----------
        observed_data
            Observed subject data. Accepts a data dictionary or a raw data
            array using the workflow's observation contract.

        Returns
        -------
        tuple[numpy.ndarray, list[int]]
            Data array and active subject counts per dataset.
        """

        if self.observation == "trial":
            return self._normalize_random_trial_data(observed_data)
        return self._normalize_random_aggregate_data(observed_data)

    def _normalize_random_aggregate_data(
        self, observed_data
    ) -> tuple[np.ndarray, list[int]]:
        """Return aggregate observed subjects for random-effect sampling.

        Parameters
        ----------
        observed_data
            Mapping with ``data`` or an aggregate data array.

        Returns
        -------
        tuple[numpy.ndarray, list[int]]
            Data with shape ``(datasets, subjects, random_feature_width)`` and
            active subject counts.
        """

        raw = (
            observed_data["data"]
            if isinstance(observed_data, Mapping)
            else observed_data
        )
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, np.newaxis, :]
        elif arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        elif arr.ndim != 3:
            raise ValueError(
                "aggregate observed_data must have shape (features,), "
                "(subjects, features), or (datasets, subjects, features)."
            )

        random_width = self._random_aggregate_feature_width()
        if arr.shape[-1] == random_width:
            active_counts = self._active_subject_counts(
                observed_data,
                arr,
                has_subject_mask=False,
            )
            return arr, active_counts
        if self.include_mask and arr.shape[-1] == random_width + 1:
            active_counts = self._active_subject_counts(
                observed_data,
                arr,
                has_subject_mask=True,
            )
            return arr[..., :random_width], active_counts
        raise ValueError(
            "aggregate observed_data has incompatible feature width. Expected "
            f"{random_width} features for random sampling."
        )

    def _normalize_random_trial_data(
        self, observed_data
    ) -> tuple[np.ndarray, list[int]]:
        """Return trial-level observed subjects for random-effect sampling.

        Parameters
        ----------
        observed_data
            Mapping with ``data`` or a trial-level data array.

        Returns
        -------
        tuple[numpy.ndarray, list[int]]
            Data with shape ``(datasets, subjects, trials, features)`` and
            active subject counts.
        """

        raw = (
            observed_data["data"]
            if isinstance(observed_data, Mapping)
            else observed_data
        )
        if self.trial_design == "flex":
            return self._normalize_random_flex_trial_data(raw)
        return self._normalize_random_fixed_trial_data(observed_data, raw)

    def _normalize_random_fixed_trial_data(
        self,
        observed_data,
        raw,
    ) -> tuple[np.ndarray, list[int]]:
        """Return fixed-trial data and warn when trial counts differ.

        Parameters
        ----------
        observed_data
            Original user input, used only to preserve subject-count handling.
        raw
            Trial data array with fixed trial rows.

        Returns
        -------
        tuple[numpy.ndarray, list[int]]
            Data with shape ``(datasets, subjects, trials, features)`` and
            active subject counts.
        """

        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, np.newaxis, :, :]
        elif arr.ndim == 3:
            arr = arr[np.newaxis, :, :, :]
        elif arr.ndim != 4:
            raise ValueError(
                "trial observed_data must have shape (trials, features), "
                "(subjects, trials, features), or "
                "(datasets, subjects, trials, features)."
            )
        if arr.shape[-1] != self.data_width:
            raise ValueError(
                "trial observed_data has incompatible feature width. Expected "
                f"{self.data_width} features."
            )
        if arr.shape[-2] != int(self.n_trials):
            warnings.warn(
                "sample_random_posterior received trial data with "
                f"{arr.shape[-2]} trials, but this random workflow was trained "
                f"with fixed n_trials={self.n_trials}. Posterior samples may "
                "be unreliable.",
                UserWarning,
                stacklevel=3,
            )
        active_counts = self._active_subject_counts(
            observed_data,
            arr,
            has_subject_mask=False,
        )
        return arr, active_counts

    def _normalize_random_flex_trial_data(self, raw) -> tuple[np.ndarray, list[int]]:
        """Return flex-trial data with automatic padding and trial masks.

        Parameters
        ----------
        raw
            Trial data supplied by the user. It may already contain the
            ``active_trial`` mask or may contain raw variable-length trial rows.

        Returns
        -------
        tuple[numpy.ndarray, list[int]]
            Padded data with shape ``(datasets, subjects, max_trials,
            data_width + 1)`` and active subject counts.
        """

        arr = self._try_rectangular_trial_array(raw)
        if arr is not None and arr.shape[-1] == self._trial_feature_width():
            normalized = self._ensure_random_trial_axes(arr)
            self._validate_flex_trial_mask(normalized)
            return normalized, [int(normalized.shape[1])] * int(normalized.shape[0])
        if arr is not None and arr.shape[-1] == self.data_width:
            normalized = self._ensure_random_trial_axes(arr)
            return self._pad_random_flex_trial_data(normalized)
        return self._normalize_ragged_random_flex_trial_data(raw)

    @staticmethod
    def _try_rectangular_trial_array(raw) -> np.ndarray | None:
        """Return ``raw`` as a float array when it is rectangular.

        Parameters
        ----------
        raw
            User-supplied trial data.

        Returns
        -------
        numpy.ndarray or None
            Rectangular array, or ``None`` when ragged subject lengths prevent
            direct conversion.
        """

        try:
            return np.asarray(raw, dtype=np.float32)
        except ValueError:
            return None

    @staticmethod
    def _ensure_random_trial_axes(arr: np.ndarray) -> np.ndarray:
        """Ensure trial data has dataset and subject axes.

        Parameters
        ----------
        arr
            Trial data shaped as one subject, one dataset, or many datasets.

        Returns
        -------
        numpy.ndarray
            Data with shape ``(datasets, subjects, trials, features)``.
        """

        if arr.ndim == 2:
            return arr[np.newaxis, np.newaxis, :, :]
        if arr.ndim == 3:
            return arr[np.newaxis, :, :, :]
        if arr.ndim == 4:
            return arr
        raise ValueError(
            "trial observed_data must have shape (trials, features), "
            "(subjects, trials, features), or "
            "(datasets, subjects, trials, features)."
        )

    def _validate_flex_trial_mask(self, arr: np.ndarray) -> None:
        """Validate a pre-padded flexible trial array.

        Parameters
        ----------
        arr
            Data with a final ``active_trial`` mask column.

        Returns
        -------
        None
            Raises an error when the shape or mask is incompatible.
        """

        if arr.shape[-2] != self.max_trials:
            raise ValueError(
                "pre-padded flex trial observed_data must use "
                f"max_trials={self.max_trials} rows."
            )
        mask = arr[..., self.data_width]
        if not np.all((mask == 0.0) | (mask == 1.0)):
            raise ValueError("active_trial mask values must be 0 or 1.")

    def _pad_random_flex_trial_data(
        self,
        arr: np.ndarray,
    ) -> tuple[np.ndarray, list[int]]:
        """Pad rectangular raw flexible trial data and append a mask.

        Parameters
        ----------
        arr
            Raw trial data with shape ``(datasets, subjects, trials,
            data_width)``.

        Returns
        -------
        tuple[numpy.ndarray, list[int]]
            Padded masked data and active subject counts.
        """

        if arr.shape[-2] > self.max_trials:
            raise ValueError(
                "trial observed_data has more trials than this model supports. "
                f"Expected at most {self.max_trials} trials from n_trials_range."
            )
        out = np.zeros(
            (arr.shape[0], arr.shape[1], self.max_trials, self._trial_feature_width()),
            dtype=np.float32,
        )
        n_trials = arr.shape[-2]
        out[:, :, :n_trials, : self.data_width] = arr
        out[:, :, :n_trials, self.data_width] = 1.0
        return out, [int(arr.shape[1])] * int(arr.shape[0])

    def _normalize_ragged_random_flex_trial_data(
        self,
        raw,
    ) -> tuple[np.ndarray, list[int]]:
        """Pad ragged flexible trial data and append active-trial masks.

        Parameters
        ----------
        raw
            A list of subject arrays, or a list of datasets where each dataset
            is a list of subject arrays.

        Returns
        -------
        tuple[numpy.ndarray, list[int]]
            Padded masked data and active subject counts.
        """

        datasets = self._as_trial_datasets(raw)
        n_datasets = len(datasets)
        n_subjects = max(len(dataset) for dataset in datasets)
        out = np.zeros(
            (n_datasets, n_subjects, self.max_trials, self._trial_feature_width()),
            dtype=np.float32,
        )
        active_counts = []
        for dataset_id, dataset in enumerate(datasets):
            active_counts.append(len(dataset))
            for subject_id, subject in enumerate(dataset):
                subject_arr = np.asarray(subject, dtype=np.float32)
                if subject_arr.ndim == 1:
                    subject_arr = subject_arr[:, np.newaxis]
                if subject_arr.ndim != 2 or subject_arr.shape[-1] != self.data_width:
                    raise ValueError(
                        "raw flex trial subjects must have shape "
                        f"(trials, {self.data_width})."
                    )
                n_trials = subject_arr.shape[0]
                if n_trials > self.max_trials:
                    raise ValueError(
                        "trial observed_data has more trials than this model "
                        f"supports. Expected at most {self.max_trials} trials "
                        "from n_trials_range."
                    )
                out[
                    dataset_id,
                    subject_id,
                    :n_trials,
                    : self.data_width,
                ] = subject_arr
                out[dataset_id, subject_id, :n_trials, self.data_width] = 1.0
        return out, active_counts

    @staticmethod
    def _as_trial_datasets(raw) -> list[list]:
        """Return ragged trial input as ``datasets -> subjects``.

        Parameters
        ----------
        raw
            Ragged trial data supplied by the user.

        Returns
        -------
        list[list]
            Nested datasets, each containing subject trial arrays.
        """

        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError("ragged flex trial observed_data must be a sequence.")
        if len(raw) == 0:
            raise ValueError("ragged flex trial observed_data cannot be empty.")
        first = raw[0]
        first_arr = np.asarray(first, dtype=object)
        if first_arr.ndim == 2:
            return [list(raw)]
        return [list(dataset) for dataset in raw]

    def _active_subject_counts(
        self,
        observed_data,
        arr: np.ndarray,
        has_subject_mask: bool,
    ) -> list[int]:
        """Return active subject counts for observed random-effect data.

        Parameters
        ----------
        observed_data
            Original observed data input.
        arr
            Normalized data array with dataset and subject axes.
        has_subject_mask
            Whether the final aggregate feature is an active-subject mask.

        Returns
        -------
        list[int]
            Active subject count for each dataset.
        """

        n_datasets = arr.shape[0]
        n_subjects = arr.shape[1]
        if isinstance(observed_data, Mapping) and "n_subjects" in observed_data:
            counts = np.asarray(observed_data["n_subjects"], dtype=int).reshape(-1)
            if counts.size == 1 and n_datasets > 1:
                counts = np.repeat(counts, n_datasets)
            if counts.size != n_datasets:
                raise ValueError("n_subjects must have one value per dataset.")
            if np.any(counts < 1) or np.any(counts > n_subjects):
                raise ValueError("n_subjects contains values outside the data shape.")
            return [int(value) for value in counts]
        if has_subject_mask:
            mask = arr[..., -1]
            return [
                int(np.sum(mask[dataset_id] > 0.5)) for dataset_id in range(n_datasets)
            ]
        return [int(n_subjects)] * n_datasets

    def sample_random_posterior(
        self,
        observed_data,
        group_samples: Mapping[str, np.ndarray],
        approximator_kwargs: Mapping | None = None,
        sample_batch_size: int | None = None,
    ) -> Mapping[str, np.ndarray]:
        """Draw subject-level random-effect posterior samples.

        Parameters
        ----------
        observed_data
            One or more observed subjects using this workflow's observation
            contract. A full group data dictionary from ``model.simulate`` is
            also accepted. For trial observations, fixed-trial models warn
            when the observed trial count differs from the model's ``n_trials``.
            Flexible-trial models infer the required padding and
            ``active_trial`` mask from ``n_trials_range`` when raw
            variable-length subject trial arrays are supplied.
        group_samples
            Group posterior samples from ``model.sample_group_posterior``. Raw
            group keys such as ``theta_mu_raw`` and ``theta_log_sigma`` are
            required because they define the shrinkage transform.
        approximator_kwargs
            Optional keyword arguments forwarded to the random workflow's
            ``sample`` method.
        sample_batch_size
            Optional number of subject/draw condition rows to sample at once.
            Larger values can be faster but use more memory.

        Returns
        -------
        Mapping[str, numpy.ndarray]
            Subject posterior samples with shape
            ``(n_datasets, n_samples, n_subjects)`` for public parameter keys,
            raw subject keys, and standardized ``z`` keys.
        """

        if self.random_workflow is None:
            raise ValueError("Call train_random_workflow(...) before sampling.")
        observed_arr, active_counts = self._normalize_random_observed_data(
            observed_data
        )
        n_datasets, n_subject_slots = observed_arr.shape[:2]
        group_n_datasets, n_samples = self._group_sample_shape(group_samples)
        if group_n_datasets != n_datasets:
            raise ValueError(
                "observed_data and group_samples must have the same dataset count."
            )
        group_arrays = {
            key: self._group_condition_array(
                group_samples,
                key,
                n_datasets,
                n_samples,
            )
            for key in self._random_group_condition_keys()
        }

        condition_rows = []
        row_index = []
        group_conditions = {key: [] for key in self._random_group_condition_keys()}
        for dataset_id in range(n_datasets):
            for subject_id in range(active_counts[dataset_id]):
                subject_data = observed_arr[dataset_id, subject_id]
                for sample_id in range(n_samples):
                    if self.observation == "aggregate":
                        condition_rows.append(subject_data[np.newaxis, :])
                    else:
                        condition_rows.append(subject_data)
                    row_index.append((dataset_id, sample_id, subject_id))
                    for key in group_conditions:
                        group_conditions[key].append(
                            group_arrays[key][dataset_id, sample_id]
                        )

        conditions = {"data": np.asarray(condition_rows, dtype=np.float32)}
        for key, values in group_conditions.items():
            conditions[key] = np.asarray(values, dtype=np.float32)

        sample_kwargs = dict(approximator_kwargs or {})
        if sample_batch_size is not None:
            sample_kwargs["batch_size"] = int(sample_batch_size)
        z_samples = self.random_workflow.sample(
            num_samples=1,
            conditions=conditions,
            **sample_kwargs,
        )
        return self._format_random_samples(
            z_samples=z_samples,
            group_arrays=group_arrays,
            row_index=row_index,
            n_datasets=n_datasets,
            n_samples=n_samples,
            n_subjects=n_subject_slots,
        )

    def plot_random_recovery(
        self,
        n_datasets: int,
        num_samples: int,
        params: str | Sequence[str] | None = None,
        metrics: str | Sequence[str] = "corr",
        n_cols: int = 3,
    ):
        """Plot dataset-level random parameter recovery metrics.

        The method simulates group datasets, samples group posteriors, samples
        subject-level random effects, and plots one recovery metric per
        simulated dataset and parameter. It requires ``keep_subject_truth`` so
        the simulation contains subject-level true values.

        Parameters
        ----------
        n_datasets
            Number of simulated group datasets used for the diagnostic plot.
        num_samples
            Number of group posterior draws per simulated dataset. These draws
            define the paired shrinkage conditions for random-effect sampling.
        params
            Optional subject-level parameter name or names to plot, such as
            ``"c"`` or ``"kappa"``. By default all saved subject-truth
            parameters with posterior samples are shown.
        metrics
            Metric name or names to plot. Supported values are ``corr``,
            ``ccc``, and ``rmse``.
        n_cols
            Maximum number of columns in the metric panel grid.

        Returns
        -------
        matplotlib.figure.Figure
            Subject-level random parameter recovery figure.
        """

        from bami.evaluation.diagnostics import plot_random_recovery

        return plot_random_recovery(
            self,
            n_datasets=n_datasets,
            num_samples=num_samples,
            params=params,
            metrics=metrics,
            n_cols=n_cols,
        )

    def _format_random_samples(
        self,
        *,
        z_samples: Mapping[str, np.ndarray],
        group_arrays: Mapping[str, np.ndarray],
        row_index: list[tuple[int, int, int]],
        n_datasets: int,
        n_samples: int,
        n_subjects: int,
    ) -> dict[str, np.ndarray]:
        """Convert sampled ``z`` values into subject posterior samples.

        Parameters
        ----------
        z_samples
            Random workflow output for all subject/draw condition rows.
        group_arrays
            Raw group condition arrays used for paired conversion.
        row_index
            Mapping from flattened condition rows back to dataset, sample, and
            subject positions.
        n_datasets, n_samples, n_subjects
            Output dimensions.

        Returns
        -------
        dict[str, numpy.ndarray]
            Public, raw, and standardized subject posterior arrays.
        """

        from bami.inference.priors import apply_link, log_sigma_key, mu_raw_key

        out = {}
        for param_name, spec in self.priors.items():
            if not isinstance(spec, dict):
                continue
            z_key = self._random_z_key(param_name)
            z_values = np.asarray(z_samples[z_key], dtype=np.float32).reshape(-1)
            z_arr = np.full(
                (n_datasets, n_samples, n_subjects),
                np.nan,
                dtype=np.float32,
            )
            for row_id, (dataset_id, sample_id, subject_id) in enumerate(row_index):
                z_arr[dataset_id, sample_id, subject_id] = z_values[row_id]

            mu = group_arrays[mu_raw_key(param_name)]
            log_sigma = group_arrays[log_sigma_key(param_name)]
            raw_arr = mu[:, :, np.newaxis] + np.exp(log_sigma[:, :, np.newaxis]) * z_arr
            public_arr = apply_link(raw_arr, spec.get("link", "identity"))

            out[z_key] = z_arr
            out[f"{param_name}_subj_raw"] = raw_arr.astype(np.float32)
            out[param_name] = np.asarray(public_arr, dtype=np.float32)
        return out

    def _prepare_observed_counts(self, counts) -> tuple[np.ndarray, list]:
        """Convert subject count rows to padded hierarchy summary data.

        Parameters
        ----------
        counts
            Subject-by-feature array ending in ``data_width`` base features.

        Returns
        -------
        tuple[numpy.ndarray, list]
            One-dataset BayesFlow data array and subject identifiers.
        """

        arr = np.asarray(counts, dtype=np.float32)
        subject_ids = list(range(arr.shape[0])) if arr.ndim == 2 else []
        if self.input_format is not None and self.input_format.add_n:
            expected_width = self.data_width + 1
            if arr.ndim != 2 or arr.shape[1] != expected_width:
                raise ValueError(
                    "counts must include base features plus n_trials when "
                    "input_format encodes n."
                )
            n_trials_values = arr[:, -1]
            arr = arr[:, : self.data_width]
        else:
            if arr.ndim != 2 or arr.shape[1] != self.data_width:
                raise ValueError(
                    f"counts must have shape (n_subjects, {self.data_width})."
                )
            n_trials_values = arr.sum(axis=1)
        if np.any(~np.isfinite(arr)):
            raise ValueError("counts must not contain missing or infinite values.")
        if np.any(arr < 0):
            raise ValueError("counts must be nonnegative.")

        n_subjects = arr.shape[0]
        if n_subjects > self.max_subjects:
            raise ValueError(
                f"counts contains {n_subjects} subjects, but this workflow pads "
                f"to {self.max_subjects}."
            )
        data = np.zeros((1, self.max_subjects, self._feature_width()), dtype=np.float32)
        for subject_id, row in enumerate(arr):
            n_trials = int(n_trials_values[subject_id])
            row_arr = row
            if self.input_format is not None:
                row_arr = self.input_format.encode(row, n_trials)
            elif self.row_transform is not None:
                row_arr = np.asarray(
                    self.row_transform(row, n_trials=n_trials, model=self),
                    dtype=np.float32,
                )
            col = 0
            encoded_width = self._encoded_data_width()
            data[0, subject_id, col : col + encoded_width] = row_arr
            col += encoded_width
            if self.input_format is None and self.include_trial_feature:
                data[0, subject_id, col] = self._format_trial_feature(n_trials)
                col += 1
            if self.include_mask:
                data[0, subject_id, col] = 1.0
        return data, subject_ids

    def _input_format_metadata(self) -> dict | None:
        """Return JSON-safe input-format metadata for this workflow.

        Returns
        -------
        dict or None
            Metadata from ``input_format`` when one is configured.
        """

        if self.input_format is None:
            return None
        return self.input_format.to_dict()

    def _resolve_random_validation_data(self, validation_data: int | dict) -> dict:
        """Return validation data for random-workflow training.

        Parameters
        ----------
        validation_data
            Integer validation-set size or a pre-simulated random-workflow
            validation dictionary.

        Returns
        -------
        dict
            BayesFlow validation data dictionary for ``random_workflow``.
        """

        if self.random_workflow is None:
            self._build_random_workflow()
        if isinstance(validation_data, int):
            if self.random_validation_data is None:
                self.random_validation_data = self.random_workflow.simulate(
                    validation_data
                )
            elif len(next(iter(self.random_validation_data.values()))) != int(
                validation_data
            ):
                self.random_validation_data = self.random_workflow.simulate(
                    validation_data
                )
            return self.random_validation_data
        return validation_data

    def _resolve_random_train_config(
        self,
        *,
        inherit_config: bool,
        overrides: Mapping,
        fit_kwargs: Mapping,
    ) -> dict:
        """Return the effective training config for the random workflow.

        Parameters
        ----------
        inherit_config
            Whether to start from this model's saved group-workflow training
            config.
        overrides
            Training config values that should replace inherited values.
        fit_kwargs
            Additional training options inherited from or shared with
            ``train_workflow``.

        Returns
        -------
        dict
            Effective random-workflow training config.
        """

        if inherit_config:
            base = getattr(self, "_train_config", None)
            if base is None:
                base = training.default_train_config()
        else:
            base = training.default_train_config()

        inherited_fit_kwargs = dict(base.get("fit_kwargs", {}))
        inherited_fit_kwargs.update(fit_kwargs)
        config_values = dict(base)
        for key, value in overrides.items():
            if value is not None:
                config_values[key] = value
        return training.make_train_config(
            max_epochs=config_values["max_epochs"],
            initial_epochs=config_values["initial_epochs"],
            n_batch=config_values["n_batch"],
            batch_size=config_values["batch_size"],
            validation_data=config_values["validation_data"],
            patience=config_values["patience"],
            min_delta=config_values["min_delta"],
            workers=config_values["workers"],
            max_queue_size=config_values["max_queue_size"],
            torch_device=config_values["torch_device"],
            verbose=config_values["verbose"],
            fit_kwargs=inherited_fit_kwargs,
        )

    def train_random_workflow(
        self,
        *,
        inherit_config: bool = True,
        file=None,
        overwrite=False,
        **kwargs,
    ):
        """Train the subject-level random-effect workflow.

        Parameters
        ----------
        inherit_config
            Whether to inherit this model's saved ``train_workflow`` settings.
        file
            Optional saved workflow file for the random workflow. This is never
            inherited from group-workflow training.
        overwrite
            Whether to refit and overwrite ``file`` when it already exists.
        **kwargs
            Optional training config overrides shared with ``train_workflow``
            such as ``max_epochs``, ``n_batch``, and ``batch_size``.

        Returns
        -------
        object or dict
            BayesFlow training history, or ``{"loaded": True, "file": path}``
            when an existing saved random workflow is reused.
        """

        if self.random_workflow is None:
            self._build_random_workflow()
        config_keys = {
            key for key in training.TRAIN_CONFIG_DEFAULTS.keys() if key != "fit_kwargs"
        }
        overrides = {}
        fit_kwargs = {}
        for key, value in kwargs.items():
            if key in config_keys:
                overrides[key] = value
            else:
                fit_kwargs[key] = value
        config = self._resolve_random_train_config(
            inherit_config=inherit_config,
            overrides=overrides,
            fit_kwargs=fit_kwargs,
        )
        self._random_train_config = config
        trainer = _RandomWorkflowTrainingAdapter(self)
        return training.train_workflow(
            trainer,
            max_epochs=config["max_epochs"],
            initial_epochs=config["initial_epochs"],
            n_batch=config["n_batch"],
            batch_size=config["batch_size"],
            validation_data=config["validation_data"],
            patience=config["patience"],
            min_delta=config["min_delta"],
            workers=config["workers"],
            max_queue_size=config["max_queue_size"],
            torch_device=config["torch_device"],
            verbose=config["verbose"],
            file=file,
            overwrite=overwrite,
            **config["fit_kwargs"],
        )

    def train_workflow(
        self,
        max_epochs=100,
        initial_epochs=5,
        n_batch=2000,
        batch_size=32,
        validation_data=200,
        patience=5,
        min_delta=0.1,
        workers=1,
        max_queue_size=4,
        torch_device=None,
        verbose=1,
        file=None,
        overwrite=False,
        **kwargs,
    ):
        """Train the workflow with optional saved-workflow handling.

        Parameters
        ----------
        max_epochs, initial_epochs
            Maximum and initial training epochs.
        n_batch, batch_size
            Online simulation batches per epoch and datasets per batch.
        validation_data
            Integer validation-set size or a pre-simulated validation dict.
        patience, min_delta
            Early-stopping controls based on validation loss.
        workers
            Number of Keras data-loading workers for online simulation batches.
        max_queue_size
            Maximum queue length for prefetched simulation batches.
        torch_device
            Torch default device to use during training, such as ``"mps"`` or
            ``"cpu"``. Unavailable accelerators fall back to CPU.
        verbose
            Training log verbosity level passed to Keras.
        file
            Optional saved workflow file. When supplied, existing weights are
            loaded by default and new weights are saved after fitting.
        overwrite
            Whether to refit and overwrite ``file`` when the saved workflow
            file already exists.
        **kwargs
            Additional keyword arguments passed to ``workflow.fit_online``.

        Returns
        -------
        object or dict
            BayesFlow training history, or ``{"loaded": True, "file": path}``
            when an existing saved workflow file is reused.
        """

        return training.train_workflow(
            self,
            max_epochs=max_epochs,
            initial_epochs=initial_epochs,
            n_batch=n_batch,
            batch_size=batch_size,
            validation_data=validation_data,
            patience=patience,
            min_delta=min_delta,
            workers=workers,
            max_queue_size=max_queue_size,
            torch_device=torch_device,
            verbose=verbose,
            file=file,
            overwrite=overwrite,
            **kwargs,
        )

    def _resolve_validation_data(self, validation_data: int | dict) -> dict:
        """Return validation data for training.

        Parameters
        ----------
        validation_data
            Integer validation-set size or a pre-simulated validation dict.

        Returns
        -------
        dict
            BayesFlow validation data dictionary.
        """

        return SimpleWorkflow._resolve_validation_data(self, validation_data)

    @staticmethod
    def _check_name(name) -> str:
        """Validate a model name.

        Parameters
        ----------
        name
            Candidate model name.

        Returns
        -------
        str
            Non-empty model name.
        """

        checked = str(name).strip()
        if checked == "":
            raise ValueError("name must be non-empty.")
        return checked

    @staticmethod
    def _check_param_names(param_names) -> list[str]:
        """Validate group-level inference variable names.

        Parameters
        ----------
        param_names
            Candidate parameter-name sequence.

        Returns
        -------
        list[str]
            Non-empty list of parameter names.
        """

        if isinstance(param_names, str):
            raise ValueError("param_names must be a sequence, not one string.")
        checked = [str(name).strip() for name in param_names]
        if not checked or any(name == "" for name in checked):
            raise ValueError("param_names must contain non-empty names.")
        return checked

    @staticmethod
    def _check_priors(priors) -> Mapping:
        """Validate group prior metadata.

        Parameters
        ----------
        priors
            Prior specification used by optional transforms.

        Returns
        -------
        Mapping
            Validated prior mapping.
        """

        if not isinstance(priors, Mapping):
            raise ValueError("priors must be a mapping.")
        return priors

    @staticmethod
    def _check_callable(value, name: str) -> Callable:
        """Validate a callable simulator component.

        Parameters
        ----------
        value
            Candidate callable.
        name
            Field name used in error messages.

        Returns
        -------
        Callable
            The original callable.
        """

        if not callable(value):
            raise ValueError(f"{name} must be callable.")
        return value
