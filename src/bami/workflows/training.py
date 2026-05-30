"""Shared training helpers for workflow classes.

This module keeps the training loop and saved-workflow file handling in one
place so simple and hierarchical workflows expose the same public
``train_workflow`` behavior.
"""

from pathlib import Path

import numpy as np

from bami.inference.runtime import configure_torch_device

TRAIN_CONFIG_DEFAULTS = {
    "max_epochs": 100,
    "initial_epochs": 5,
    "n_batch": 2000,
    "batch_size": 32,
    "validation_data": 200,
    "patience": 5,
    "min_delta": 0.1,
    "workers": 1,
    "max_queue_size": 4,
    "torch_device": None,
    "verbose": 1,
    "fit_kwargs": {},
}


def make_train_config(
    *,
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
    fit_kwargs=None,
) -> dict:
    """Return a reusable workflow training configuration.

    Args:
        max_epochs: Maximum number of training epochs.
        initial_epochs: Number of epochs in the first training block.
        n_batch: Online simulation batches per epoch.
        batch_size: Simulated datasets per online batch.
        validation_data: Integer validation-set size or a pre-simulated
            validation dictionary.
        patience: Early-stopping patience based on validation loss.
        min_delta: Minimum validation-loss improvement counted as progress.
        workers: Number of Keras data-loading workers.
        max_queue_size: Maximum queue length for prefetched simulation
            batches.
        torch_device: Torch default device, such as ``"mps"`` or ``"cpu"``.
        verbose: Training log verbosity level passed to Keras.
        fit_kwargs: Optional keyword arguments passed to BayesFlow's
            ``fit_online``.

    Returns:
        dict: Training configuration without saved-file settings.
    """

    return {
        "max_epochs": max_epochs,
        "initial_epochs": initial_epochs,
        "n_batch": n_batch,
        "batch_size": batch_size,
        "validation_data": validation_data,
        "patience": patience,
        "min_delta": min_delta,
        "workers": workers,
        "max_queue_size": max_queue_size,
        "torch_device": torch_device,
        "verbose": verbose,
        "fit_kwargs": dict(fit_kwargs or {}),
    }


def default_train_config() -> dict:
    """Return a fresh copy of the default training configuration.

    Returns:
        dict: Default training settings used by ``train_workflow``.
    """

    return make_train_config(**TRAIN_CONFIG_DEFAULTS)


def train_workflow(
    model,
    *,
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
    """Train a workflow model with optional saved-workflow handling.

    Args:
        model: Workflow object exposing ``workflow`` and
            ``_resolve_validation_data``.
        max_epochs: Maximum number of training epochs.
        initial_epochs: Number of epochs in the first training block.
        n_batch: Online simulation batches per epoch.
        batch_size: Simulated datasets per online batch.
        validation_data: Integer validation-set size or a pre-simulated
            validation dictionary.
        patience: Early-stopping patience based on validation loss.
        min_delta: Minimum validation-loss improvement counted as progress.
        workers: Number of Keras data-loading workers.
        max_queue_size: Maximum queue length for prefetched simulation
            batches.
        torch_device: Torch default device, such as ``"mps"`` or ``"cpu"``.
        verbose: Training log verbosity level passed to Keras.
        file: Optional saved workflow file. Existing weights are loaded by
            default, and new weights are saved after fitting.
        overwrite: Whether to refit and overwrite ``file`` when the saved
            workflow file already exists.
        **kwargs: Additional keyword arguments passed to
            ``workflow.fit_online``.

    Returns:
        object | dict: BayesFlow training history, or
            ``{"loaded": True, "file": path}`` when an existing saved workflow file
            is reused.
    """

    config = make_train_config(
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
        fit_kwargs=kwargs,
    )
    model._train_config = config

    selected_device = configure_torch_device(torch_device)
    if torch_device is not None:
        print(f"Using Torch device for training: {selected_device}", flush=True)

    saved_workflow_path = None
    if file is not None:
        saved_workflow_path = _check_saved_workflow_path(file)
        if saved_workflow_path.exists() and not overwrite:
            from bami.inference.checkpoints import load_workflow_weights

            load_workflow_weights(model, saved_workflow_path)
            print(f"Loaded saved workflow: {saved_workflow_path}", flush=True)
            return {"loaded": True, "file": saved_workflow_path}

    fit_kwargs = {
        "workers": workers,
        "use_multiprocessing": False,
        "max_queue_size": max_queue_size,
        "verbose": verbose,
        "keep_optimizer": True,
    }
    fit_kwargs.update(kwargs)
    history = _run_training_loop(
        model,
        max_epochs=max_epochs,
        initial_epochs=initial_epochs,
        n_batch=n_batch,
        batch_size=batch_size,
        validation_data=validation_data,
        patience=patience,
        min_delta=min_delta,
        **fit_kwargs,
    )

    if saved_workflow_path is not None:
        from bami.inference.checkpoints import save_workflow_weights

        saved_path = save_workflow_weights(model, saved_workflow_path)
        print(f"Saved workflow: {saved_path}", flush=True)
    return history


def _run_training_loop(
    model,
    *,
    max_epochs=100,
    initial_epochs=5,
    n_batch=2000,
    batch_size=32,
    validation_data=200,
    patience=5,
    min_delta=0.1,
    **fit_kwargs,
):
    """Run online training with reusable validation data.

    Args:
        model: Workflow object exposing ``workflow`` and
            ``_resolve_validation_data``.
        max_epochs: Maximum number of training epochs.
        initial_epochs: Number of epochs in the first training block.
        n_batch: Online simulation batches per epoch.
        batch_size: Simulated datasets per online batch.
        validation_data: Integer validation-set size or a pre-simulated
            validation dict.
        patience: Early-stopping patience based on validation loss.
        min_delta: Minimum validation-loss improvement counted as progress.
        **fit_kwargs: Additional keyword arguments passed to
            ``workflow.fit_online``.

    Returns:
        object: BayesFlow training history.
    """

    fixed_validation_data = model._resolve_validation_data(validation_data)
    total_epochs = 0
    best_val = np.inf
    no_improve_epochs = 0
    history_all = {"loss": [], "val_loss": []}

    while total_epochs < max_epochs:
        if total_epochs == 0 and initial_epochs > 0:
            current_epochs = initial_epochs
        else:
            current_epochs = max(1, patience - (no_improve_epochs % patience))

        hist = model.workflow.fit_online(
            epochs=current_epochs,
            num_batches_per_epoch=n_batch,
            batch_size=batch_size,
            validation_data=fixed_validation_data,
            **fit_kwargs,
        )
        stage_hist = hist.history if hasattr(hist, "history") else hist
        stage_loss = stage_hist.get("loss", [])
        stage_val = stage_hist.get("val_loss", stage_loss)

        history_all["loss"].extend(stage_loss)
        history_all["val_loss"].extend(stage_val)
        total_epochs += len(stage_val)
        model.workflow.history = hist
        hist.history = history_all

        for val_loss in stage_val:
            if best_val - val_loss > min_delta:
                best_val = val_loss
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    return model.workflow.history

    return model.workflow.history


def _check_saved_workflow_path(file) -> Path:
    """Validate a user-supplied saved workflow path.

    Args:
        file:
            String or ``Path`` pointing to the desired saved workflow file.

    Returns:
        pathlib.Path: Normalized non-empty saved workflow path.
    """

    saved_path = Path(file)
    if str(saved_path).strip() == "":
        raise ValueError("file must be a non-empty saved workflow path.")
    if saved_path.exists() and saved_path.is_dir():
        raise ValueError("file must point to a saved workflow file, not a directory.")
    if saved_path.suffix != ".keras":
        raise ValueError("file must use the .keras saved workflow extension.")
    return saved_path
