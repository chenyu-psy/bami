"""Generic simple BayesFlow workflow.

This module builds non-hierarchical workflows where one prior draw generates
one dataset. The observation contract is explicit: aggregate observations are
one fixed-width summary row, while trial observations are one row per trial.
The model remains simple in the project sense because it has no group-level
hierarchy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import bayesflow as bf

from bami.inputs import InputFormat
from bami.workflows import training
from bami.workflows.contracts import validate_observation, validate_workflow_contract


class SimpleWorkflow:
    """Build a non-hierarchical BayesFlow workflow from a simulator.

    Use this when one prior draw produces one independent dataset. Aggregate
    observations are one fixed-width summary row; trial observations are one row
    per trial.

    Args:
        name: Short model name attached to the BayesFlow workflow.
        param_names: Public parameter names used by the model.
        priors: Prior specification passed to ``draw_prior_with_raw``.
        simulator: Function called as
            ``simulator(**params, n_trials=..., rng=..., **simulator_kwargs)``.
        observation: Use ``"aggregate"`` when the simulator returns one
            fixed-width summary row, or ``"trial"`` when it returns one row per
            trial.
        simulator_kwargs: Constant keyword arguments passed to ``simulator``
            on every simulation.
        data_width: Number of features returned by ``simulator``. For new
            code, prefer ``obs_names`` so column meanings are visible.
        obs_names: Names of simulator output columns. When supplied,
            ``data_width`` is inferred from ``len(obs_names)``.
        contract: Optional compatibility mapping with ``name``,
            ``param_names``, ``priors``, ``simulator``, and ``data_width``.
        n_trials: Fixed number of trials passed to the simulator. Provide
            this or ``n_trials_range``, but not both.
        n_trials_range: Range ``(low, high)``. Trial counts are drawn from
            ``[low, high)`` for each simulated dataset.
        include_trial_feature: Whether to append the simulated trial count to
            each data row.
        input_format: Optional helper that formats simulator rows and trial
            counts instead of ``include_trial_feature``.
        summary_dim: Width of the DeepSet summary network.
        n_coupling_layers: Number of coupling layers in the inference
            network.
        transform_samples: Optional posterior transform function. If omitted,
            raw samples are transformed with ``transform_simple_samples``.
    """

    workflow_level = "simple"

    def __init__(
        self,
        name: str | None = None,
        param_names: list[str] | None = None,
        priors: Mapping | None = None,
        simulator: Callable | None = None,
        observation: str | None = None,
        simulator_kwargs: Mapping | None = None,
        data_width: int | None = None,
        obs_names: Sequence[str] | None = None,
        contract: Mapping | None = None,
        n_trials: int | None = 100,
        n_trials_range: Sequence[int] | None = None,
        include_trial_feature: bool = False,
        input_format: InputFormat | None = None,
        summary_dim: int = 64,
        n_coupling_layers: int = 6,
        transform_samples: Callable | None = None,
    ):
        self.contract = self._resolve_contract(
            name=name,
            param_names=param_names,
            priors=priors,
            simulator=simulator,
            data_width=data_width,
            obs_names=obs_names,
            contract=contract,
        )
        self.model_name = self.contract["name"]
        self.param_names = self.contract["param_names"]
        self.priors = self.contract["priors"]
        self._simulator_fn = self.contract["simulator"]
        self.observation = validate_observation(observation, "SimpleWorkflow")
        self.simulator_kwargs = dict(
            simulator_kwargs or self.contract.get("simulator_kwargs", {})
        )
        self.obs_names = self.contract["obs_names"]
        self.data_width = self.contract["data_width"]
        self.trial_design, self.n_trials, self.n_trials_range = self._resolve_trials(
            n_trials=n_trials,
            n_trials_range=n_trials_range,
        )
        self.include_trial_feature = bool(include_trial_feature)
        self.input_format = self._check_input_format(input_format)
        self.max_trials = self._resolve_max_trials()
        self.include_mask = self.observation == "trial" and self.trial_design == "flex"
        self.workflow_family = f"{self.trial_design}_simple"
        self.summary_dim = int(summary_dim)
        self.n_coupling_layers = int(n_coupling_layers)
        self._transform_samples = transform_samples

        self._build_workflow()
        self.validation_data = None

    @staticmethod
    def _resolve_contract(
        name: str | None,
        param_names: list[str] | None,
        priors: Mapping | None,
        simulator: Callable | None,
        data_width: int | None,
        obs_names: Sequence[str] | None,
        contract: Mapping | None,
    ) -> dict:
        """Return a validated workflow contract.

        Args:
            name: Optional workflow name from simulator-first setup.
            param_names: Optional public parameter names.
            priors: Optional prior specification.
            simulator: Optional simulator callable.
            data_width: Optional simulator output width.
            obs_names: Optional simulator output column names.
            contract: Optional legacy contract mapping. It is accepted so existing model
                code can migrate one workflow at a time.

        Returns:
            dict: Validated workflow contract used internally by BayesFlow setup.
        """

        if contract is not None:
            out = validate_workflow_contract(contract)
            out["obs_names"] = list(
                contract.get("obs_names", [f"x{i}" for i in range(out["data_width"])])
            )
            return out

        checked_obs_names = SimpleWorkflow._resolve_obs_names(obs_names, data_width)
        explicit_contract = {
            "name": name,
            "param_names": param_names,
            "priors": priors,
            "simulator": simulator,
            "data_width": len(checked_obs_names) if checked_obs_names else data_width,
        }
        missing = [key for key, value in explicit_contract.items() if value is None]
        if missing:
            raise ValueError(
                "SimpleWorkflow requires explicit simulator settings: " f"{missing}"
            )
        out = validate_workflow_contract(explicit_contract)
        out["obs_names"] = checked_obs_names or [
            f"x{i}" for i in range(out["data_width"])
        ]
        return out

    @staticmethod
    def _resolve_obs_names(
        obs_names: Sequence[str] | None,
        data_width: int | None,
    ) -> list[str] | None:
        """Return validated observation names or ``None``.

        Args:
            obs_names:
                Optional simulator-output column names.
            data_width:
                Optional expected simulator-output width.

        Returns:
            list[str] or None: Validated observation names when supplied.
        """

        if obs_names is None:
            return None
        names = list(obs_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("obs_names must contain non-empty strings.")
        if len(set(names)) != len(names):
            raise ValueError("obs_names must be unique.")
        if data_width is not None and int(data_width) != len(names):
            raise ValueError("data_width must match len(obs_names).")
        return names

    @staticmethod
    def _public_params(param: Mapping) -> dict:
        """Return simulator parameters after dropping raw inference keys.

        Args:
            param:
                Prior draw containing public parameters and raw-space inference
                variables.

        Returns:
            dict: Public-scale parameters suitable for ``simulator(**params)``.
        """

        public_params = {}
        for key, value in param.items():
            if key.endswith("_raw"):
                continue
            if key.endswith("_mu_raw"):
                continue
            if key.endswith("_log_sigma"):
                continue
            public_params[key] = value
        return public_params

    @classmethod
    def _resolve_trials(
        cls,
        n_trials: int | None,
        n_trials_range: Sequence[int] | None,
    ) -> tuple[str, int | None, tuple[int, int] | None]:
        """Validate fixed or flexible trial-count settings.

        Args:
            n_trials:
                Fixed trial count, or ``None`` when using a range.
            n_trials_range:
                Two-value range with lower and exclusive upper bounds, or ``None``
                when using a fixed trial count.

        Returns:
            tuple[str, int | None, tuple[int, int] | None]: Trial design label and validated trial settings.
        """

        if n_trials is not None and n_trials_range is not None:
            raise ValueError("Provide either n_trials or n_trials_range, not both.")
        if n_trials is None and n_trials_range is None:
            raise ValueError("Provide one of n_trials or n_trials_range.")
        if n_trials_range is not None:
            return "flex", None, cls._check_range(n_trials_range, "n_trials_range")
        return "fixed", cls._check_positive_int(n_trials, "n_trials"), None

    def _draw_n_trials(self, rng=np.random) -> int:
        """Draw the trial count for one simulated dataset.

        Args:
            rng:
                NumPy-compatible random module or generator.

        Returns:
            int: Fixed or sampled trial count.
        """

        if self.trial_design == "fixed":
            return int(self.n_trials)
        low, high = self.n_trials_range
        return int(rng.randint(low, high))

    def _resolve_max_trials(self) -> int | None:
        """Return padded trial rows for trial observations.

        Returns:
            int or None: Fixed or largest flexible trial count for ``observation="trial"``;
                ``None`` for aggregate workflows.
        """

        if self.observation == "aggregate":
            return None
        if self.trial_design == "fixed":
            return int(self.n_trials)
        return int(self.n_trials_range[1] - 1)

    def _build_workflow(self) -> None:
        """Build the simulator, networks, and BayesFlow workflow.

        Returns:
            None: Sets ``simulator``, ``summary_network``, ``inference_network``, and
                ``workflow`` on this object.
        """

        def _draw_prior():
            """Draw raw and public model parameters for one simulation.

            Returns:
                dict: Prior draw containing raw inference keys and public truth keys.
            """

            from bami.inference.priors import draw_prior_with_raw

            return draw_prior_with_raw(**self.priors)

        def _simulate_data(**param):
            """Simulate one simple dataset.

            Args:
                    **param
                    Prior draw with public parameters and raw keys.

            Returns:
                dict[str, numpy.ndarray]: BayesFlow summary data matching the explicit observation
                    contract.
            """

            n_trials = self._draw_n_trials(np.random)
            public_params = self._public_params(param)
            simulated = self._simulator_fn(
                **public_params,
                n_trials=n_trials,
                rng=np.random,
                **self.simulator_kwargs,
            )
            if self.observation == "aggregate":
                data = self._as_aggregate_data(simulated, n_trials)
            else:
                data = self._as_trial_data(simulated, n_trials)
            return {"data": data}

        from bami.inference.priors import raw_key

        inference_variables = [
            raw_key(name)
            for name, spec in self.priors.items()
            if isinstance(spec, dict)
        ]

        self.simulator = bf.make_simulator([_draw_prior, _simulate_data])
        self.summary_network = bf.networks.DeepSet(summary_dim=self.summary_dim)
        self.inference_network = bf.networks.CouplingFlow(
            n_coupling_layers=self.n_coupling_layers
        )
        self.workflow = bf.BasicWorkflow(
            simulator=self.simulator,
            inference_network=self.inference_network,
            summary_network=self.summary_network,
            inference_variables=inference_variables,
            inference_conditions=None,
            summary_variables=["data"],
        )
        self.workflow.transform_posterior_samples = self.convert_posterior
        self.workflow.workflow_level = self.workflow_level
        self.workflow.workflow_family = self.workflow_family
        self.workflow.trial_design = self.trial_design
        self.workflow.model_name = self.model_name
        self.workflow.observation = self.observation
        self.workflow.include_trial_feature = self.include_trial_feature
        self.workflow.include_mask = self.include_mask
        self.workflow.input_format = self.input_format
        self.workflow.input_format_metadata = self._input_format_metadata()
        self.workflow.obs_names = self._workflow_obs_names()

    def _feature_width(self) -> int:
        """Return the final BayesFlow feature width for one simple row.

        Returns:
            int: Simulator row width plus optional trial-count feature.
        """

        if self.input_format is not None:
            return self.input_format.output_width(self.data_width)

        width = self.data_width
        if self.observation == "aggregate" and self.include_trial_feature:
            width += 1
        if self.observation == "trial" and self.include_mask:
            width += 1
        return width

    def _workflow_obs_names(self) -> list[str]:
        """Return feature names exposed on the BayesFlow workflow.

        Returns:
            list[str]: Observation feature names, plus design columns when the workflow
                appends them.
        """

        names = list(self.obs_names)
        if self.observation == "aggregate" and self.include_trial_feature:
            names.append("n_trials")
        if self.observation == "trial" and self.include_mask:
            names.append("active_trial")
        return names

    @staticmethod
    def _check_input_format(
        input_format: InputFormat | None,
    ) -> InputFormat | None:
        """Validate an optional input format.

        Args:
            input_format:
                Candidate input format or ``None``.

        Returns:
            InputFormat or None: The validated input format.
        """

        if input_format is None:
            return None
        if not isinstance(input_format, InputFormat):
            raise TypeError("input_format must be an InputFormat or None.")
        return input_format

    def _input_format_metadata(self) -> dict | None:
        """Return JSON-safe input-format metadata for this workflow.

        Returns:
            dict or None: Metadata from ``input_format`` when one is configured.
        """

        if self.input_format is None:
            return None
        return self.input_format.to_dict()

    def _append_trial_feature(self, row, n_trials: int) -> np.ndarray:
        """Append the trial count to one simulator row.

        Args:
            row:
                Simulator output before workflow-level design features.
            n_trials:
                Trial count used for this simulated dataset.

        Returns:
            numpy.ndarray: One row with the trial-count feature appended.
        """

        row_arr = np.asarray(row, dtype=np.float32).reshape(-1)
        return np.concatenate([row_arr, np.array([n_trials], dtype=np.float32)])

    def _as_data_row(self, row) -> np.ndarray:
        """Convert simulator output to BayesFlow simple data shape.

        Args:
            row:
                Subject-level simulator output with ``data_width`` values.

        Returns:
            numpy.ndarray: Float array with shape ``(1, data_width)``.
        """

        feature_width = self._feature_width()
        data = np.asarray(row, dtype=np.float32)
        if data.shape == (feature_width,):
            return data[np.newaxis, :]
        if data.shape == (1, feature_width):
            return data
        raise ValueError(
            "aggregate simulator must return a row with shape "
            f"({feature_width},) or (1, {feature_width}) when "
            "observation='aggregate'."
        )

    def _as_aggregate_data(self, row, n_trials: int) -> np.ndarray:
        """Format one aggregate simulator row for BayesFlow.

        Args:
            row:
                Simulator output representing one summarized dataset.
            n_trials:
                Trial count used to produce ``row``.

        Returns:
            numpy.ndarray: Aggregate data with shape ``(1, feature_width)``.
        """

        if self.input_format is not None:
            row = self.input_format.encode(row, n_trials)
        elif self.include_trial_feature:
            row = self._append_trial_feature(row, n_trials)
        return self._as_data_row(row)

    def _as_trial_data(self, rows, n_trials: int) -> np.ndarray:
        """Format trial simulator rows for BayesFlow.

        Args:
            rows:
                Trial observation rows returned by the simulator.
            n_trials:
                Number of active trials represented by ``rows``.

        Returns:
            numpy.ndarray: Data with shape ``(n_trials, data_width)`` for fixed designs or
                ``(max_trials, data_width + 1)`` for flexible designs.
        """

        arr = np.asarray(rows, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        if arr.shape != (int(n_trials), self.data_width):
            raise ValueError(
                "trial simulator must return shape "
                f"({n_trials}, {self.data_width}) when observation='trial'."
            )
        if not self.include_mask:
            return arr

        data = np.zeros((self.max_trials, self._feature_width()), dtype=np.float32)
        data[:n_trials, : self.data_width] = arr
        data[:n_trials, self.data_width] = 1.0
        return data

    def _prepare_observed_counts(self, counts) -> tuple[np.ndarray, list]:
        """Convert simple count rows to BayesFlow summary data.

        Args:
            counts:
                Array ending in ``data_width`` count features. A final trial-count
                feature is appended when ``include_trial_feature`` is enabled.

        Returns:
            tuple[numpy.ndarray, list]: Batched data array and row identifiers.
        """

        if self.observation != "aggregate":
            raise ValueError(
                "_prepare_observed_counts is only available for aggregate data."
            )
        arr = np.asarray(counts, dtype=np.float32)
        row_ids = list(range(arr.shape[0])) if arr.ndim >= 2 else [0]
        if self.input_format is not None and self.input_format.add_n:
            if arr.shape[-1] != self.data_width + 1:
                raise ValueError(
                    "counts must include base features plus n_trials when "
                    "input_format encodes n."
                )
            n_trials = arr[..., -1]
            arr = arr[..., : self.data_width]
        else:
            if arr.shape[-1] != self.data_width:
                raise ValueError(f"counts must end with {self.data_width} features.")
            n_trials = arr.sum(axis=-1)
        if np.any(~np.isfinite(arr)):
            raise ValueError("counts must not contain missing or infinite values.")
        if np.any(arr < 0):
            raise ValueError("counts must be nonnegative.")
        if self.input_format is not None:
            encoded_rows = []
            flat_arr = arr.reshape(-1, self.data_width)
            flat_n = np.asarray(n_trials).reshape(-1)
            for row, n_value in zip(flat_arr, flat_n, strict=True):
                encoded_rows.append(self.input_format.encode(row, int(n_value)))
            arr = np.asarray(encoded_rows, dtype=np.float32).reshape(
                *arr.shape[:-1],
                self._feature_width(),
            )
        elif self.include_trial_feature:
            totals = arr.sum(axis=-1, keepdims=True)
            arr = np.concatenate([arr, totals], axis=-1)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        return arr.astype(np.float32), row_ids

    def convert_posterior(self, samples: dict) -> dict:
        """Transform raw posterior samples to public parameter keys.

        Args:
            samples: Raw posterior sample dictionary from BayesFlow.

        Returns:
            dict: Posterior samples with public-scale parameter keys added.
        """

        if self._transform_samples is not None:
            return self._transform_samples(samples, self.priors)
        from bami.inference.priors import transform_simple_samples

        return transform_simple_samples(samples, self.priors)

    def simulate(self, n_datasets: int) -> Mapping[str, np.ndarray]:
        """Simulate datasets from this simple workflow.

        Args:
            n_datasets: Number of simulated datasets to draw from the
                workflow prior and simulator.

        Returns:
            Mapping[str, numpy.ndarray]: Simulated data and parameter truth
                arrays using the workflow's data-shape contract.
        """

        return self.workflow.simulate(n_datasets)

    def sample_posterior(
        self,
        test_data: Mapping[str, np.ndarray],
        num_samples: int,
        approximator_kwargs: Mapping | None = None,
        sample_batch_size: int | None = None,
    ) -> Mapping[str, np.ndarray]:
        """Draw posterior samples for this simple workflow.

        Args:
            test_data: Observed or simulated data dictionary passed to the
                trained BayesFlow workflow. In examples this is often created with
                ``model.simulate(n_datasets)``.
            num_samples: Number of posterior draws to request for each
                dataset.
            approximator_kwargs: Optional keyword arguments forwarded to
                BayesFlow's ``workflow.sample`` method.
            sample_batch_size: Optional number of datasets to sample at once.
                Use this when many datasets would otherwise exceed accelerator
                memory.

        Returns:
            Mapping[str, numpy.ndarray]: Posterior draws on the public
                parameter scale. The sample axis is usually axis 1 for batched data.
        """

        from bami.workflows._sampling import _sample_posterior

        return _sample_posterior(
            workflow=self.workflow,
            test_data=test_data,
            num_samples=num_samples,
            approximator_kwargs=approximator_kwargs,
            sample_batch_size=sample_batch_size,
        )

    def plot_parameter_recovery(
        self,
        n_datasets: int,
        num_samples: int,
        params: str | Sequence[str] | None = None,
        metrics: str | Sequence[str] = "corr",
        n_cols: int = 3,
        sample_batch_size: int = 100,
    ):
        """Plot parameter recovery for this simple workflow.

        The method simulates datasets from the workflow prior, samples the
        posterior for each dataset, and plots simulated parameter values
        against posterior means. It is intended for diagnosing one fitted model,
        not for comparing multiple models.

        Args:
            n_datasets: Number of simulated datasets used for the diagnostic
                plot.
            num_samples: Number of posterior draws per simulated dataset.
            params: Optional parameter name or names to plot. By default all
                public inferred parameters with both simulated truth and posterior
                samples are shown.
            metrics: Metric name or names shown in each panel title.
                Supported values are ``corr``, ``ccc``, and ``rmse``.
            n_cols: Maximum number of columns in the plot grid.
            sample_batch_size: BayesFlow posterior sampling mini-batch size.
                Larger values usually reduce sampling overhead; lower this
                value if a diagnostic run exceeds available memory.

        Returns:
            matplotlib.figure.Figure: Parameter recovery figure.
        """

        from bami.evaluation.diagnostics import plot_parameter_recovery

        return plot_parameter_recovery(
            self,
            n_datasets=n_datasets,
            num_samples=num_samples,
            params=params,
            metrics=metrics,
            n_cols=n_cols,
            sample_batch_size=sample_batch_size,
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
    ) -> object | dict:
        """Train the workflow with optional saved-workflow handling.

        Args:
            max_epochs (int): Maximum number of training epochs.
            initial_epochs (int): Number of epochs in the first training block.
            n_batch (int): Online simulation batches per epoch.
            batch_size (int): Simulated datasets per online batch.
            validation_data (int | Mapping | None): Integer validation-set size or a pre-simulated
                validation dictionary.
            patience (int): Early-stopping patience based on validation loss.
            min_delta (float): Minimum validation-loss improvement counted as
                progress.
            workers (int): Number of Keras data-loading workers for online
                simulation batches.
            max_queue_size (int): Maximum queue length for prefetched simulation
                batches.
            torch_device (str | None): Torch default device to use during training, such
                as ``"mps"`` or ``"cpu"``. Unavailable accelerators fall back to CPU.
            verbose (int): Training log verbosity level passed to Keras.
            file (str | pathlib.Path | None): Optional saved workflow file. Existing weights are loaded
                by default, and new weights are saved after fitting.
            overwrite (bool): Whether to refit and overwrite ``file`` when the saved
                workflow file already exists.
            **kwargs (Any): Additional keyword arguments passed to
                ``workflow.fit_online``.

        Returns:
            object | dict: BayesFlow training history, or
                ``{"loaded": True, "file": path}`` when an existing saved workflow
                file is reused.
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

        Args:
            validation_data (int | Mapping | None):
                Integer validation-set size or a pre-simulated validation dict.

        Returns:
            dict: BayesFlow validation data dictionary.
        """

        if isinstance(validation_data, dict):
            self.validation_data = validation_data
            return self.validation_data

        if isinstance(validation_data, int):
            if self.validation_data is None:
                self.validation_data = self.workflow.simulate(validation_data)
            elif len(self.validation_data["data"]) != validation_data:
                self.validation_data = self.workflow.simulate(validation_data)
            return self.validation_data

        raise ValueError("validation_data must be an integer size or a data dict.")

    @staticmethod
    def _check_positive_int(value, name: str) -> int:
        """Validate a positive integer setting.

        Args:
            value:
                Candidate integer.
            name:
                Setting name used in error messages.

        Returns:
            int: Positive integer value.
        """

        checked = int(value)
        if checked < 1:
            raise ValueError(f"{name} must be at least 1.")
        return checked

    @classmethod
    def _check_range(cls, value: Sequence[int], name: str) -> tuple[int, int]:
        """Validate a two-value integer range.

        Args:
            value:
                Candidate range with lower and exclusive upper bounds.
            name:
                Setting name used in error messages.

        Returns:
            tuple[int, int]: Validated range.
        """

        try:
            if len(value) != 2:
                raise ValueError(f"{name} must have exactly two values.")
            low = cls._check_positive_int(value[0], f"{name}[0]")
            high = int(value[1])
        except TypeError as exc:
            raise ValueError(f"{name} must be a two-value sequence.") from exc

        if high <= low:
            raise ValueError(f"{name} must satisfy 1 <= low < high.")
        return low, high
