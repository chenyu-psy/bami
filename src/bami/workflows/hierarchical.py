"""Generic hierarchical BayesFlow workflow.

This module builds group-level workflows where one simulated dataset contains
multiple subjects. Aggregate observations use one fixed-width row per subject.
Trial observations use nested subject-by-trial rows for continuous data.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import bayesflow as bf
import keras

from bami.inputs import InputFormat
from bami.workflows import training
from bami.workflows.contracts import validate_observation
from bami.workflows.simple import SimpleWorkflow


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
    posthoc_estimator
        Optional estimator class for posthoc subject-level posterior summaries.
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
        posthoc_estimator: Callable | None = None,
        posthoc_kwargs: Mapping | None = None,
        posthoc_kind: str | None = None,
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
        self.keep_subject_truth = list(keep_subject_truth or [])
        self.raw_data_key = raw_data_key
        self.row_transform = row_transform
        self.trial_feature_scale = trial_feature_scale
        self.input_format = SimpleWorkflow._check_input_format(input_format)
        self.posthoc_estimator = posthoc_estimator
        self.posthoc_kwargs = dict(posthoc_kwargs or {})
        self.posthoc_kind = posthoc_kind
        self.workflow_family = f"{self.subject_design}_hierarchical"
        self.subject_id_mode = "exchangeable"
        self.summary_dim = int(summary_dim)
        self.n_coupling_layers = int(n_coupling_layers)
        self._transform_samples = transform_samples

        self._build_workflow()
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
        self.workflow.posthoc_kind = self.posthoc_kind
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

    def build_posthoc_estimator(self):
        """Build the configured posthoc estimator for this workflow.

        Returns
        -------
        object
            Estimator exposing ``estimate_subjects``.
        """

        if self.posthoc_estimator is None:
            raise ValueError("This workflow does not define a posthoc estimator.")
        return self.posthoc_estimator(**self.posthoc_kwargs)

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

    def train_workflow(
        self,
        max_epochs=100,
        initial_epochs=10,
        n_batch=5000,
        batch_size=32,
        validation_data=200,
        patience=5,
        min_delta=0.1,
        workers=4,
        max_queue_size=16,
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
