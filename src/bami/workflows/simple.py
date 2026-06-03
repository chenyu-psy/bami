"""Generic simple BayesFlow workflow.

This module builds non-hierarchical workflows where one prior draw generates
one dataset. The observation contract is explicit: aggregate observations are
one fixed-width summary row, while trial observations are one row per trial.
The model remains simple in the project sense because it has no group-level
hierarchy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
import warnings

import numpy as np
import bayesflow as bf

from bami.inputs import InputFormat
from bami.inference.runtime import runtime_device, validate_device
from bami.workflows import training
from bami.workflows.contracts import validate_observation

_SINGLETON_SOFTMAX_MESSAGE = (
    r"You are using a softmax over axis .* of a tensor of shape .*"
    r"This axis has size 1.*"
)


@contextmanager
def _suppress_singleton_softmax_warning(enabled: bool):
    """Hide the harmless DeepSet warning for one-row aggregate summaries.

    Simple aggregate workflows pass one summary row per simulated dataset into
    BayesFlow's DeepSet. Its attention pooling therefore applies softmax over a
    set axis of size one. The operation is harmless but otherwise clutters
    researcher-facing training and sampling logs.

    Args:
        enabled: Whether to suppress the singleton softmax warning inside this
            context. Trial-level workflows pass ``False`` so their warnings stay
            visible.

    Yields:
        None. Code inside the context runs with only this specific warning
        filtered.
    """

    if not enabled:
        yield
        return

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=_SINGLETON_SOFTMAX_MESSAGE,
            category=UserWarning,
            module=r"keras\.src\.ops\.nn",
        )
        yield


class SimpleWorkflow:
    """Build a non-hierarchical BayesFlow workflow from a simulator.

    Use this when one prior draw produces one independent dataset. Aggregate
    observations are one fixed-width summary row; trial observations are one row
    per trial.

    Args:
        name: Short model name attached to the BayesFlow workflow.
        param_names: Public parameter names used by this workflow.
        priors: Prior specification passed to ``draw_prior_with_raw``.
        simulator: Function called as
            ``simulator(**params, n_trials=..., **simulator_kwargs)``.
        observation: Use ``"aggregate"`` when the simulator returns one
            fixed-width summary row, or ``"trial"`` when it returns one row per
            trial.
        simulator_kwargs: Constant keyword arguments passed to ``simulator``
            on every simulation.
        obs_names: Names of simulator output columns. The workflow infers the
            simulator row width from ``len(obs_names)``.
        n_trials: Fixed number of trials passed to the simulator, or a
            two-value range ``(low, high)``. Range values draw trial counts
            from ``[low, high)`` for each simulated dataset.
        input_format: Optional helper that formats simulator rows and trial
            counts, for example by appending an encoded trial-count feature.
        summary_dim: Width of the DeepSet summary network.
        n_coupling_layers: Number of coupling layers in the inference
            network.
        device: Workflow runtime device. CPU is the stable default; use
            ``"mps"`` or ``"cuda"`` only when accelerator training is desired.
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
        obs_names: Sequence[str] | None = None,
        n_trials: int | Sequence[int] = 100,
        input_format: InputFormat | None = None,
        summary_dim: int = 64,
        n_coupling_layers: int = 6,
        device: str | None = "cpu",
    ):
        self.device = validate_device(device)
        self.model_name = self._check_required(name, "name")
        self.param_names = self._check_param_names(param_names)
        self.priors = self._check_required(priors, "priors")
        self._simulator_fn = self._check_callable(simulator, "simulator")
        self.observation = validate_observation(observation, "SimpleWorkflow")
        self.simulator_kwargs = dict(simulator_kwargs or {})
        self.obs_names = self._resolve_obs_names(obs_names)
        self.data_width = len(self.obs_names)
        self.trial_design, self.n_trials, self.n_trials_range = self._resolve_trials(
            n_trials
        )
        self.input_format = self._check_input_format(input_format)
        self.max_trials = self._resolve_max_trials()
        self.include_mask = self.observation == "trial" and self.trial_design == "flex"
        self.workflow_family = f"{self.trial_design}_simple"
        self.summary_dim = int(summary_dim)
        self.n_coupling_layers = int(n_coupling_layers)

        with runtime_device(self.device):
            self._build_workflow()
        self.validation_data = None

    @staticmethod
    def _check_required(value, name: str):
        """Return a required constructor value.

        Args:
            value: Candidate value.
            name: Parameter name used in the error message.

        Returns:
            object: The supplied value when it is not ``None``.
        """

        if value is None:
            raise ValueError(f"SimpleWorkflow requires {name}.")
        return value

    @staticmethod
    def _check_param_names(param_names: Sequence[str] | None) -> list[str]:
        """Return validated public parameter names.

        Args:
            param_names: Names inferred by the workflow.

        Returns:
            list[str]: Non-empty unique parameter names.
        """

        if param_names is None:
            raise ValueError("SimpleWorkflow requires param_names.")
        names = list(param_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("param_names must contain non-empty strings.")
        if len(set(names)) != len(names):
            raise ValueError("param_names must be unique.")
        return names

    @staticmethod
    def _check_callable(value, name: str) -> Callable:
        """Return a required callable constructor value.

        Args:
            value: Candidate callable.
            name: Parameter name used in the error message.

        Returns:
            Callable: The supplied callable.
        """

        if not callable(value):
            raise ValueError(f"{name} must be callable.")
        return value

    @staticmethod
    def _resolve_obs_names(
        obs_names: Sequence[str] | None,
    ) -> list[str]:
        """Return validated observation names.

        Args:
            obs_names:
                Simulator-output column names.

        Returns:
            list[str]: Validated observation names.
        """

        if obs_names is None:
            raise ValueError("obs_names is required.")
        names = list(obs_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("obs_names must contain non-empty strings.")
        if len(set(names)) != len(names):
            raise ValueError("obs_names must be unique.")
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
        n_trials: int | Sequence[int],
    ) -> tuple[str, int | None, tuple[int, int] | None]:
        """Validate fixed or flexible trial-count settings from ``n_trials``.

        Args:
            n_trials:
                Positive fixed trial count, or a two-value range with lower
                and exclusive upper bounds.

        Returns:
            tuple[str, int | None, tuple[int, int] | None]: Trial design label and validated trial settings.
        """

        return cls._resolve_count(n_trials, "n_trials")

    @classmethod
    def _resolve_count(
        cls,
        value: int | Sequence[int],
        name: str,
    ) -> tuple[str, int | None, tuple[int, int] | None]:
        """Validate one fixed or flexible positive count setting.

        Args:
            value:
                Positive fixed count, or a two-value range with lower and
                exclusive upper bounds.
            name:
                Setting name used in error messages.

        Returns:
            tuple[str, int | None, tuple[int, int] | None]: Design label and validated count settings.
        """

        if value is None:
            raise ValueError(
                f"{name} must be a positive integer or a two-value range "
                "(low, high)."
            )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return "flex", None, cls._check_range(value, name)
        return "fixed", cls._check_positive_int(value, name), None

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
                **self.simulator_kwargs,
            )
            if self.observation == "aggregate":
                raw_row = np.asarray(simulated, dtype=np.float32)
                data = self._as_aggregate_data(simulated, n_trials)
            else:
                raw_row = None
                data = self._as_trial_data(simulated, n_trials)
            out = {"data": data}
            raw_key = self._raw_output_key()
            if raw_key is not None and raw_row is not None:
                out[raw_key] = raw_row
            return out

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
        if (
            self.observation == "aggregate"
            and self.input_format is not None
            and self.input_format.add_n
        ):
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

    def _raw_output_key(self) -> str | None:
        """Return the optional key for preserving raw aggregate rows.

        Returns:
            str or None: Output key from ``input_format.keep_raw_as`` when the
            configured input format requests raw diagnostic rows.
        """

        if self.input_format is None:
            return None
        return self.input_format.keep_raw_as

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
                Array ending in the base count features. When ``input_format``
                encodes trial count, the final column must contain ``n_trials``.

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
            can_infer_n = self.input_format.kind in {"counts", "counts_as_proportions"}
            if arr.shape[-1] == self.data_width + 1:
                n_trials = arr[..., -1]
                arr = arr[..., : self.data_width]
            elif can_infer_n and arr.shape[-1] == self.data_width:
                n_trials = arr.sum(axis=-1)
            else:
                raise ValueError(
                    "counts must include base features plus n_trials when "
                    "input_format encodes n."
                )
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

        with runtime_device(getattr(self, "device", "cpu")):
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
                the workflow object's ``simulate(n_datasets)`` method.
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

        with runtime_device(getattr(self, "device", "cpu")):
            with _suppress_singleton_softmax_warning(
                getattr(self, "observation", None) == "aggregate"
            ):
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
        sample_batch_size: int = 10,
        recovery_batch_size: int = 10,
        show_progress: bool = True,
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
            recovery_batch_size: Number of recovery datasets simulated and
                sampled per chunk. Smaller values reduce peak memory use.
            show_progress: Whether to show one BAMI progress bar while scoring
                recovery datasets. BayesFlow's internal sampling output is hidden.

        Returns:
            matplotlib.figure.Figure: Parameter recovery figure.
        """

        from bami.evaluation.diagnostics import plot_parameter_recovery

        with runtime_device(getattr(self, "device", "cpu")):
            with _suppress_singleton_softmax_warning(
                getattr(self, "observation", None) == "aggregate"
            ):
                return plot_parameter_recovery(
                    self,
                    n_datasets=n_datasets,
                    num_samples=num_samples,
                    params=params,
                    metrics=metrics,
                    n_cols=n_cols,
                    sample_batch_size=sample_batch_size,
                    recovery_batch_size=recovery_batch_size,
                    show_progress=show_progress,
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

        with runtime_device(getattr(self, "device", "cpu")):
            with _suppress_singleton_softmax_warning(
                getattr(self, "observation", None) == "aggregate"
            ):
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
            low = int(value[0])
            high = int(value[1])
        except TypeError as exc:
            raise ValueError(f"{name} must be a two-value sequence.") from exc
        except ValueError as exc:
            if str(exc).startswith(f"{name} must"):
                raise
            raise ValueError(f"{name} must contain integer values.") from exc

        if low < 1 or high <= low:
            raise ValueError(f"{name} must satisfy 1 <= low < high.")
        return low, high
