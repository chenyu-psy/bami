"""Deterministic random-effect estimator for hierarchical workflows.

This module implements the Route C recovery head. It is intentionally separate
from the legacy BayesFlow random workflow: the estimator returns point estimates
for subject-level parameters, not posterior samples or calibrated intervals.
"""

from __future__ import annotations

from pathlib import Path
from contextlib import contextmanager
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from bami.inference.priors import apply_link, log_sigma_key, mu_raw_key

RANDOM_ESTIMATOR_BACKEND = "supervised_estimator"


class RandomEstimatorNet(nn.Module):
    """Small DeepSet-style network for subject-level point estimates.

    Parameters
    ----------
    n_row_features
        Number of features in each subject row or trial row.
    n_conditions
        Number of group-condition features.
    n_outputs
        Number of supervised target outputs.
    hidden_width
        Width of the MLP layers.

    Returns
    -------
    None
        The module predicts standardized ``delta_raw`` and ``z`` targets.
    """

    def __init__(
        self,
        n_row_features: int,
        n_conditions: int,
        n_outputs: int,
        hidden_width: int = 64,
    ) -> None:
        super().__init__()
        self.row_encoder = nn.Sequential(
            nn.Linear(n_row_features, 32),
            nn.SiLU(),
            nn.Linear(32, hidden_width),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_width + n_conditions, 96),
            nn.SiLU(),
            nn.Linear(96, 96),
            nn.SiLU(),
            nn.Linear(96, n_outputs),
        )

    def forward(self, rows: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """Predict standardized random-effect targets.

        Parameters
        ----------
        rows
            Tensor shaped ``batch x rows x features``. Aggregate observations use
            one row; trial observations use one row per trial.
        conditions
            Tensor shaped ``batch x condition_features`` with raw group values.

        Returns
        -------
        torch.Tensor
            Standardized predictions shaped ``batch x n_outputs``.
        """

        row_summary = self.row_encoder(rows).mean(dim=1)
        features = torch.cat([row_summary, conditions], dim=1)
        return self.head(features)


class RandomEstimator:
    """Loaded deterministic estimator plus its standardization metadata.

    Parameters
    ----------
    network
        Trained PyTorch network.
    metadata
        Checkpoint metadata containing target names, group-condition names, and
        standardization arrays.

    Returns
    -------
    None
        The object is used by ``HierarchicalWorkflow.estimate_random_parameter``.
    """

    def __init__(self, network: RandomEstimatorNet, metadata: Mapping) -> None:
        self.network = network
        self.metadata = dict(metadata)


def train_random_estimator(model, **kwargs) -> None:
    """Train or load a deterministic random-effect estimator.

    Parameters
    ----------
    model
        ``HierarchicalWorkflow`` that supplies simulation and prior helpers.
    **kwargs
        Training settings from ``HierarchicalWorkflow.train_random_estimator``.

    Returns
    -------
    None
        The trained or loaded estimator is stored on ``model.random_estimator``.
    """

    with _torch_cpu_default():
        file = kwargs.pop("file", None)
        overwrite = bool(kwargs.pop("overwrite", False))
        if file is not None:
            checkpoint_path = _check_estimator_path(file)
            if checkpoint_path.exists() and not overwrite:
                model.random_estimator = load_random_estimator(checkpoint_path)
                return None
        else:
            checkpoint_path = None

        cfg = _estimator_config(model, **kwargs)
        rows, conditions, targets, group_ids = _simulate_training_arrays(model, cfg)
        split = _split_groups(group_ids, cfg["validation_fraction"], cfg["seed"])
        arrays, stats = _standardize_arrays(rows, conditions, targets, split["train"])

        network, _history = _fit_network(arrays, group_ids, split, cfg)
        metadata = {
            "random_backend": RANDOM_ESTIMATOR_BACKEND,
            "model_name": model.model_name,
            "param_names": cfg["param_names"],
            "condition_names": cfg["condition_names"],
            "target_names": cfg["target_names"],
            "observation": model.observation,
            "input_format_metadata": model._input_format_metadata(),
            "config": cfg,
            "standardization": stats,
        }
        model.random_estimator = RandomEstimator(network, metadata)
        if checkpoint_path is not None:
            save_random_estimator(model.random_estimator, checkpoint_path)
        return None


def estimate_random_parameter(
    model,
    *,
    observed_data,
    group_samples: Mapping[str, np.ndarray],
    include_scales: bool = False,
) -> pd.DataFrame:
    """Estimate subject-level random parameters with the deterministic head.

    Parameters
    ----------
    model
        ``HierarchicalWorkflow`` with a trained or loaded ``random_estimator``.
    observed_data
        Observed subjects using the model's aggregate or trial contract.
    group_samples
        Group posterior samples. Raw group keys are averaged over posterior
        draws before deterministic subject estimation.
    include_scales
        Whether to include raw, deviation, standardized ``z``, and group columns.

    Returns
    -------
    pandas.DataFrame
        One row per active dataset-subject. By default only public parameter
        estimates are included.
    """

    with _torch_cpu_default():
        estimator = getattr(model, "random_estimator", None)
        if estimator is None:
            raise ValueError("Call train_random_estimator(...) before estimating.")

        observed_arr, active_counts = model._normalize_random_observed_data(
            observed_data
        )
        group_values = _group_condition_means(model, group_samples)
        pred_targets = _predict_targets(
            estimator, observed_arr, active_counts, group_values
        )
        return _estimates_to_frame(
            model=model,
            estimator=estimator,
            pred_targets=pred_targets,
            active_counts=active_counts,
            group_values=group_values,
            include_scales=include_scales,
        )


def save_random_estimator(estimator: RandomEstimator, file: str | Path) -> Path:
    """Save a random estimator to one PyTorch checkpoint file.

    Parameters
    ----------
    estimator
        Trained estimator and metadata.
    file
        Path ending in ``.pt``.

    Returns
    -------
    pathlib.Path
        Saved checkpoint path.
    """

    path = _check_estimator_path(file)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": estimator.network.state_dict(),
            "metadata": estimator.metadata,
        },
        path,
    )
    return path


def load_random_estimator(file: str | Path) -> RandomEstimator:
    """Load a random estimator from one PyTorch checkpoint file.

    Parameters
    ----------
    file
        Path ending in ``.pt``.

    Returns
    -------
    RandomEstimator
        Loaded estimator ready for deterministic subject estimation.
    """

    with _torch_cpu_default():
        path = _check_estimator_path(file)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        metadata = checkpoint["metadata"]
        cfg = metadata["config"]
        network = RandomEstimatorNet(
            n_row_features=int(cfg["n_row_features"]),
            n_conditions=len(metadata["condition_names"]),
            n_outputs=len(metadata["target_names"]),
            hidden_width=int(cfg["hidden_width"]),
        )
        network.load_state_dict(checkpoint["state_dict"])
        network.eval()
        return RandomEstimator(network, metadata)


@contextmanager
def _torch_cpu_default():
    """Run estimator code on CPU regardless of BayesFlow's global device.

    BayesFlow training may set Torch's default device to ``"mps"``. The
    estimator uses CPU DataLoaders and CPU checkpoints, so it temporarily resets
    the default device to avoid MPS/CPU generator mismatches.
    """

    try:
        previous_device = torch.get_default_device()
    except AttributeError:
        previous_device = "cpu"
    torch.set_default_device("cpu")
    try:
        yield
    finally:
        torch.set_default_device(previous_device)


def _estimator_config(model, **kwargs) -> dict:
    """Return concrete training settings for the estimator."""

    seed = kwargs.get("seed")
    if seed is None:
        seed = 2026
    param_names = [
        name for name, spec in model.priors.items() if isinstance(spec, dict)
    ]
    target_names = []
    for name in param_names:
        target_names.append(f"{name}_delta_raw")
        target_names.append(f"{name}_z")
    n_row_features = (
        model._random_aggregate_feature_width()
        if model.observation == "aggregate"
        else model._trial_feature_width()
    )
    cfg = {
        "param_names": param_names,
        "condition_names": model._random_group_condition_keys(),
        "target_names": target_names,
        "n_row_features": int(n_row_features),
        "hidden_width": int(kwargs.get("hidden_width", 64)),
        "n_groups_per_sigma": int(kwargs.get("n_groups_per_sigma", 30)),
        "subjects_per_group": kwargs.get("subjects_per_group"),
        "sigma_values": kwargs.get("sigma_values"),
        "sigma_quantiles": [0.2, 0.5, 0.8],
        "n_sigma_prior_draws": 2000,
        "validation_fraction": float(kwargs.get("validation_fraction", 0.2)),
        "max_epochs": int(kwargs.get("max_epochs", 100)),
        "batch_size": int(kwargs.get("batch_size", 128)),
        "patience": int(kwargs.get("patience", 10)),
        "learning_rate": float(kwargs.get("learning_rate", 1e-3)),
        "loss": kwargs.get("loss", "mse"),
        "corr_weight": float(kwargs.get("corr_weight", 0.1)),
        "show_progress": bool(kwargs.get("show_progress", True)),
        "seed": int(seed),
    }
    cfg["resolved_sigma_values"] = _resolve_sigma_values(model, cfg)
    return cfg


def _resolve_sigma_values(model, cfg: Mapping) -> dict:
    """Return parameter-specific sigma bins for estimator training.

    ``None`` means automatic bins from the group sigma prior. A list or tuple
    shares one sigma grid across parameters. A dict gives each parameter its
    own grid. Bins are synchronized during simulation to avoid a Cartesian
    product of all parameter-specific sigma values.
    """

    param_names = list(cfg["param_names"])
    sigma_values = cfg["sigma_values"]
    if sigma_values is None:
        return _auto_sigma_values(model, cfg)
    if isinstance(sigma_values, Mapping):
        sigma_by_param = _manual_sigma_dict(param_names, sigma_values)
        source = "dict"
    else:
        shared = _sigma_list(sigma_values, label="sigma_values")
        sigma_by_param = {name: list(shared) for name in param_names}
        source = "shared"

    n_sigma_bins = _validate_sigma_by_param(param_names, sigma_by_param)
    return {
        "param_names": param_names,
        "sigma_by_param": sigma_by_param,
        "n_sigma_bins": n_sigma_bins,
        "source": source,
    }


def _auto_sigma_values(model, cfg: Mapping) -> dict:
    """Choose low/mid/high sigma bins from each parameter's group prior."""

    param_names = list(cfg["param_names"])
    rng = np.random.default_rng(int(cfg["seed"]) + 17)
    draws = {name: [] for name in param_names}
    for _ in range(int(cfg["n_sigma_prior_draws"])):
        group_params = model._draw_independent_group_prior(rng)
        for name in param_names:
            draws[name].append(float(np.exp(group_params[log_sigma_key(name)])))

    quantiles = np.asarray(cfg["sigma_quantiles"], dtype=float)
    sigma_by_param = {}
    for name in param_names:
        sigma_by_param[name] = [
            float(value) for value in np.quantile(np.asarray(draws[name]), quantiles)
        ]
    n_sigma_bins = _validate_sigma_by_param(param_names, sigma_by_param)
    return {
        "param_names": param_names,
        "sigma_by_param": sigma_by_param,
        "n_sigma_bins": n_sigma_bins,
        "source": "auto",
        "quantiles": [float(value) for value in quantiles],
        "n_prior_draws": int(cfg["n_sigma_prior_draws"]),
    }


def _manual_sigma_dict(
    param_names: Sequence[str], sigma_values: Mapping
) -> dict[str, list[float]]:
    """Validate and normalize a parameter-specific sigma grid."""

    missing = [name for name in param_names if name not in sigma_values]
    if missing:
        raise ValueError(
            "sigma_values dict must include every hierarchical parameter; "
            f"missing {missing}."
        )
    unknown = [name for name in sigma_values if name not in param_names]
    if unknown:
        raise ValueError(
            "sigma_values dict contains unknown hierarchical parameters: " f"{unknown}."
        )
    return {
        name: _sigma_list(sigma_values[name], label=f"sigma_values[{name!r}]")
        for name in param_names
    }


def _sigma_list(values, *, label: str) -> list[float]:
    """Return one positive finite sigma grid from a user value."""

    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of positive sigma values.")
    try:
        out = [float(value) for value in values]
    except TypeError as exc:
        raise ValueError(
            f"{label} must be a sequence of positive sigma values."
        ) from exc
    if not out:
        raise ValueError(f"{label} must contain at least one sigma value.")
    bad = [value for value in out if not np.isfinite(value) or value <= 0.0]
    if bad:
        raise ValueError(f"{label} must contain only positive finite sigma values.")
    return out


def _validate_sigma_by_param(
    param_names: Sequence[str], sigma_by_param: Mapping[str, Sequence[float]]
) -> int:
    """Ensure every parameter has a positive sigma grid with equal length."""

    lengths = []
    for name in param_names:
        values = _sigma_list(sigma_by_param[name], label=f"sigma_values[{name!r}]")
        sigma_by_param[name] = values
        lengths.append(len(values))
    if len(set(lengths)) != 1:
        raise ValueError(
            "Each parameter in sigma_values must have the same number of sigma bins."
        )
    return lengths[0]


def _simulate_training_arrays(
    model, cfg: Mapping
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate sigma-varying subject rows and supervised targets."""

    rng = np.random.default_rng(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    subjects_per_group = cfg["subjects_per_group"]
    if subjects_per_group is None:
        subjects_per_group = model.max_subjects
    subjects_per_group = int(subjects_per_group)

    row_arrays = []
    condition_rows = []
    target_rows = []
    group_ids = []
    group_id = 0
    resolved_sigma = cfg["resolved_sigma_values"]
    sigma_by_param = resolved_sigma["sigma_by_param"]
    total_groups = int(resolved_sigma["n_sigma_bins"]) * int(cfg["n_groups_per_sigma"])
    progress = tqdm(
        total=total_groups,
        desc="Simulating random estimator groups",
        unit="group",
        disable=not bool(cfg["show_progress"]),
    )
    for sigma_idx in range(int(resolved_sigma["n_sigma_bins"])):
        for _ in range(int(cfg["n_groups_per_sigma"])):
            group_params = model._draw_independent_group_prior(rng)
            for name in cfg["param_names"]:
                sigma = float(sigma_by_param[name][sigma_idx])
                group_params[log_sigma_key(name)] = float(np.log(sigma))
            for _ in range(subjects_per_group):
                params, target = _draw_subject_for_estimator(model, group_params, rng)
                simulated = model._simulate_random_subject(**params)
                row_arrays.append(np.asarray(simulated["data"], dtype=np.float32))
                condition_rows.append(
                    [group_params[key] for key in cfg["condition_names"]]
                )
                target_rows.append([target[key] for key in cfg["target_names"]])
                group_ids.append(group_id)
            group_id += 1
            progress.update(1)
    progress.close()
    return (
        np.asarray(row_arrays, dtype=np.float32),
        np.asarray(condition_rows, dtype=np.float32),
        np.asarray(target_rows, dtype=np.float32),
        np.asarray(group_ids, dtype=np.int64),
    )


def _draw_subject_for_estimator(
    model, group_params: Mapping[str, float], rng
) -> tuple[dict, dict]:
    """Draw one subject and its supervised ``delta_raw`` and ``z`` targets."""

    params = dict(group_params)
    target = {}
    for name, spec in model.priors.items():
        if isinstance(spec, dict):
            z_value = float(rng.normal(0.0, 1.0))
            sigma = float(np.exp(group_params[log_sigma_key(name)]))
            delta_raw = sigma * z_value
            raw_value = float(group_params[mu_raw_key(name)] + delta_raw)
            params[name] = apply_link(raw_value, spec.get("link", "identity"))
            target[f"{name}_delta_raw"] = delta_raw
            target[f"{name}_z"] = z_value
        else:
            params[name] = float(spec)
    return params, target


def _split_groups(
    group_ids: np.ndarray, validation_fraction: float, seed: int
) -> dict[str, np.ndarray]:
    """Split row indices by group id so subjects from one group do not leak."""

    rng = np.random.default_rng(seed + 1)
    groups = np.unique(group_ids)
    if len(groups) < 2:
        raise ValueError(
            "random estimator training needs at least two simulated groups."
        )
    rng.shuffle(groups)
    n_valid = max(1, int(round(len(groups) * validation_fraction)))
    valid_groups = set(groups[:n_valid])
    train_idx = []
    valid_idx = []
    for idx, group_id in enumerate(group_ids):
        if group_id in valid_groups:
            valid_idx.append(idx)
        else:
            train_idx.append(idx)
    return {
        "train": np.asarray(train_idx, dtype=np.int64),
        "validation": np.asarray(valid_idx, dtype=np.int64),
    }


def _standardize_arrays(
    rows: np.ndarray,
    conditions: np.ndarray,
    targets: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    """Standardize rows, conditions, and targets using training rows only."""

    rows_std, row_mean, row_sd = _standardize(train=rows[train_idx], values=rows)
    cond_std, cond_mean, cond_sd = _standardize(
        train=conditions[train_idx], values=conditions
    )
    target_std, target_mean, target_sd = _standardize(
        train=targets[train_idx], values=targets
    )
    arrays = {
        "rows": rows_std.astype(np.float32),
        "conditions": cond_std.astype(np.float32),
        "targets": target_std.astype(np.float32),
    }
    stats = {
        "row_mean": row_mean.tolist(),
        "row_sd": row_sd.tolist(),
        "condition_mean": cond_mean.tolist(),
        "condition_sd": cond_sd.tolist(),
        "target_mean": target_mean.tolist(),
        "target_sd": target_sd.tolist(),
    }
    return arrays, stats


def _standardize(
    train: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return standardized values and training-set mean/SD arrays."""

    axes = tuple(range(train.ndim - 1))
    mean = train.mean(axis=axes, keepdims=True)
    sd = train.std(axis=axes, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (values - mean) / sd, np.squeeze(mean), np.squeeze(sd)


def _fit_network(
    arrays: Mapping[str, np.ndarray],
    group_ids: np.ndarray,
    split: Mapping[str, np.ndarray],
    cfg: Mapping,
) -> tuple[RandomEstimatorNet, list[dict[str, float]]]:
    """Fit the supervised estimator network with early stopping."""

    torch.manual_seed(int(cfg["seed"]))
    network = RandomEstimatorNet(
        n_row_features=arrays["rows"].shape[-1],
        n_conditions=arrays["conditions"].shape[-1],
        n_outputs=arrays["targets"].shape[-1],
        hidden_width=int(cfg["hidden_width"]),
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=float(cfg["learning_rate"]))
    train_loader = _make_loader(
        arrays, group_ids, split["train"], int(cfg["batch_size"]), shuffle=True
    )
    valid_loader = _make_loader(
        arrays, group_ids, split["validation"], int(cfg["batch_size"]), shuffle=False
    )

    best_state = None
    best_valid = np.inf
    stale = 0
    history = []
    progress = tqdm(
        range(1, int(cfg["max_epochs"]) + 1),
        desc="Training random estimator",
        unit="epoch",
        disable=not bool(cfg["show_progress"]),
    )
    for epoch in progress:
        network.train()
        train_losses = []
        for rows, conditions, targets, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad():
                pred = network(rows, conditions)
                loss = _batch_loss(pred, targets, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        network.eval()
        valid_losses = []
        with torch.no_grad():
            for rows, conditions, targets, _ in valid_loader:
                pred = network(rows, conditions)
                valid_losses.append(
                    float(_batch_loss(pred, targets, cfg).detach().cpu())
                )
        train_loss = float(np.mean(train_losses))
        valid_loss = float(np.mean(valid_losses))
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": valid_loss}
        )
        progress.set_postfix(
            train_loss=f"{train_loss:.4g}",
            validation_loss=f"{valid_loss:.4g}",
        )
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in network.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= int(cfg["patience"]):
            break
    progress.close()

    if best_state is not None:
        network.load_state_dict(best_state)
    network.eval()
    return network, history


def _make_loader(
    arrays: Mapping[str, np.ndarray],
    group_ids: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
) -> DataLoader:
    """Build a simple PyTorch loader for estimator training."""

    dataset = TensorDataset(
        torch.as_tensor(arrays["rows"][indices], dtype=torch.float32),
        torch.as_tensor(arrays["conditions"][indices], dtype=torch.float32),
        torch.as_tensor(arrays["targets"][indices], dtype=torch.float32),
        torch.as_tensor(group_ids[indices], dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _batch_loss(pred: torch.Tensor, target: torch.Tensor, cfg: Mapping) -> torch.Tensor:
    """Compute MSE plus optional small correlation loss."""

    loss = torch.nn.functional.mse_loss(pred, target)
    if cfg["loss"] == "mse_corr":
        loss = loss + float(cfg["corr_weight"]) * _corr_loss(pred, target)
    elif cfg["loss"] != "mse":
        raise ValueError("random estimator loss must be 'mse' or 'mse_corr'.")
    return loss


def _corr_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return mean one-minus-correlation over target columns."""

    losses = []
    for col in range(target.shape[1]):
        x = pred[:, col] - pred[:, col].mean()
        y = target[:, col] - target[:, col].mean()
        denom = torch.sqrt(torch.sum(x * x) * torch.sum(y * y) + 1e-8)
        losses.append(1.0 - torch.sum(x * y) / denom)
    return torch.stack(losses).mean()


def _group_condition_means(
    model, group_samples: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Return posterior-mean group conditions for each dataset."""

    n_datasets, n_samples = model._group_sample_shape(group_samples)
    out = {}
    for key in model._random_group_condition_keys():
        values = model._group_condition_array(group_samples, key, n_datasets, n_samples)
        out[key] = values.mean(axis=1).astype(np.float32)
    return out


def _predict_targets(
    estimator: RandomEstimator,
    observed_arr: np.ndarray,
    active_counts: Sequence[int],
    group_values: Mapping[str, np.ndarray],
) -> list[dict[str, float]]:
    """Predict unstandardized targets for active subjects."""

    metadata = estimator.metadata
    rows = []
    conditions = []
    index_rows = []
    for dataset_id, count in enumerate(active_counts):
        condition = [
            group_values[key][dataset_id] for key in metadata["condition_names"]
        ]
        for subject_id in range(int(count)):
            subject_rows = np.asarray(
                observed_arr[dataset_id, subject_id], dtype=np.float32
            )
            if subject_rows.ndim == 1:
                subject_rows = subject_rows[np.newaxis, :]
            rows.append(subject_rows)
            conditions.append(condition)
            index_rows.append({"dataset_id": dataset_id, "subject_id": subject_id})
    if not rows:
        return []

    row_arr = np.asarray(rows, dtype=np.float32)
    cond_arr = np.asarray(conditions, dtype=np.float32)
    std = metadata["standardization"]
    row_std = (row_arr - np.asarray(std["row_mean"], dtype=np.float32)) / np.asarray(
        std["row_sd"], dtype=np.float32
    )
    cond_std = (
        cond_arr - np.asarray(std["condition_mean"], dtype=np.float32)
    ) / np.asarray(std["condition_sd"], dtype=np.float32)
    with torch.no_grad():
        pred_std = estimator.network(
            torch.as_tensor(row_std, dtype=torch.float32),
            torch.as_tensor(cond_std, dtype=torch.float32),
        ).numpy()
    pred = pred_std * np.asarray(std["target_sd"], dtype=np.float32) + np.asarray(
        std["target_mean"], dtype=np.float32
    )
    out = []
    for idx, index_row in enumerate(index_rows):
        row = dict(index_row)
        for col, target_name in enumerate(metadata["target_names"]):
            row[target_name] = float(pred[idx, col])
        out.append(row)
    return out


def _estimates_to_frame(
    *,
    model,
    estimator: RandomEstimator,
    pred_targets: list[dict[str, float]],
    active_counts: Sequence[int],
    group_values: Mapping[str, np.ndarray],
    include_scales: bool,
) -> pd.DataFrame:
    """Convert target predictions to a researcher-facing DataFrame."""

    rows = []
    for pred_row in pred_targets:
        dataset_id = int(pred_row["dataset_id"])
        out = {
            "dataset_id": dataset_id,
            "subject_id": int(pred_row["subject_id"]),
        }
        for name in estimator.metadata["param_names"]:
            mu_raw = float(group_values[mu_raw_key(name)][dataset_id])
            log_sigma = float(group_values[log_sigma_key(name)][dataset_id])
            delta_raw = float(pred_row[f"{name}_delta_raw"])
            z_value = float(pred_row[f"{name}_z"])
            raw_value = mu_raw + delta_raw
            public_value = apply_link(
                raw_value, model.priors[name].get("link", "identity")
            )
            out[name] = float(public_value)
            if include_scales:
                out[f"{name}_raw"] = raw_value
                out[f"{name}_delta_raw"] = delta_raw
                out[f"{name}_z"] = z_value
                out[mu_raw_key(name)] = mu_raw
                out[log_sigma_key(name)] = log_sigma
        rows.append(out)
    return pd.DataFrame(rows)


def _check_estimator_path(file: str | Path) -> Path:
    """Validate that estimator checkpoints use the ``.pt`` suffix."""

    path = Path(file)
    if path.suffix == ".keras":
        raise ValueError(
            "random estimator checkpoints use '.pt'. The '.keras' suffix is "
            "reserved for legacy BayesFlow workflows."
        )
    if path.suffix != ".pt":
        raise ValueError("random estimator checkpoints must use a '.pt' file.")
    return path
