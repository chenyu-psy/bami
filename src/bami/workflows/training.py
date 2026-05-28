"""Shared training helpers for workflow classes.

This module keeps the training loop and saved-workflow file handling in one
place so simple and hierarchical workflows expose the same public
``train_workflow`` behavior.
"""

from pathlib import Path

import numpy as np

from bami.inference.runtime import configure_torch_device


def train_workflow(
    model,
    *,
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
    """Train a workflow model with optional saved-workflow handling.

    Parameters
    ----------
    model
        Workflow object exposing ``workflow`` and ``_resolve_validation_data``.
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
        Whether to refit and overwrite ``file`` when the saved workflow file
        already exists.
    **kwargs
        Additional keyword arguments passed to ``workflow.fit_online``.

    Returns
    -------
    object or dict
        BayesFlow training history, or ``{"loaded": True, "file": path}`` when
        an existing saved workflow file is reused.
    """

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
    initial_epochs=10,
    n_batch=5000,
    batch_size=32,
    validation_data=200,
    patience=5,
    min_delta=0.1,
    **fit_kwargs,
):
    """Run online training with reusable validation data.

    Parameters
    ----------
    model
        Workflow object exposing ``workflow`` and ``_resolve_validation_data``.
    max_epochs, initial_epochs
        Maximum and initial training epochs.
    n_batch, batch_size
        Online simulation batches per epoch and datasets per batch.
    validation_data
        Integer validation-set size or a pre-simulated validation dict.
    patience, min_delta
        Early-stopping controls based on validation loss.
    **fit_kwargs
        Additional keyword arguments passed to ``workflow.fit_online``.

    Returns
    -------
    object
        BayesFlow training history.
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

    Parameters
    ----------
    file
        String or ``Path`` pointing to the desired saved workflow file.

    Returns
    -------
    pathlib.Path
        Normalized non-empty saved workflow path.
    """

    saved_path = Path(file)
    if str(saved_path).strip() == "":
        raise ValueError("file must be a non-empty saved workflow path.")
    if saved_path.exists() and saved_path.is_dir():
        raise ValueError("file must point to a saved workflow file, not a directory.")
    if saved_path.suffix != ".keras":
        raise ValueError("file must use the .keras saved workflow extension.")
    return saved_path
