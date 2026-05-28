"""Workflow training helpers.

Use this module when a configured workflow should be fitted with shared
training settings and optional checkpoint loading or saving.
"""

from pathlib import Path

from bami.inference.runtime import configure_torch_device

__all__ = ["fit_workflow"]


def fit_workflow(
    model,
    *,
    max_epochs: int,
    initial_epochs: int,
    n_batch: int,
    batch_size: int,
    validation_data: int,
    patience: int,
    min_delta: float,
    workers: int = 4,
    max_queue_size: int = 16,
    torch_device: str | None = None,
    verbose: int = 1,
    file: str | Path | None = None,
    overwrite: bool = False,
):
    """Fit one workflow with shared training and checkpoint settings.

    Parameters
    ----------
    model
        Configured workflow object. The object must expose ``train_workflow``
        and, when ``file`` is supplied, ``workflow.approximator``.
    max_epochs, initial_epochs, n_batch, batch_size, validation_data, patience, min_delta
        Training control values passed to ``train_workflow``.
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
        Checkpoint path. When supplied, existing weights are loaded by default
        and new weights are saved after fitting.
    overwrite
        Whether to refit and overwrite ``file`` when the checkpoint already
        exists.

    Returns
    -------
    dict
        Training history object returned by the workflow, or
        ``{"loaded": True, "file": path}`` when an existing checkpoint is
        reused.
    """

    selected_device = configure_torch_device(torch_device)
    if torch_device is not None:
        print(f"Using Torch device for training: {selected_device}", flush=True)

    checkpoint_path = None
    if file is not None:
        checkpoint_path = _check_checkpoint_path(file)
        if checkpoint_path.exists() and not overwrite:
            from bami.inference.checkpoints import load_workflow_weights

            load_workflow_weights(model, checkpoint_path)
            print(f"Loaded workflow weights: {checkpoint_path}", flush=True)
            return {"loaded": True, "file": checkpoint_path}

    history = model.train_workflow(
        max_epochs=max_epochs,
        initial_epochs=initial_epochs,
        n_batch=n_batch,
        batch_size=batch_size,
        validation_data=validation_data,
        patience=patience,
        min_delta=min_delta,
        workers=workers,
        use_multiprocessing=False,
        max_queue_size=max_queue_size,
        verbose=verbose,
        keep_optimizer=True,
    )
    if checkpoint_path is not None:
        from bami.inference.checkpoints import save_workflow_weights

        saved_path = save_workflow_weights(model, checkpoint_path)
        print(f"Saved workflow weights: {saved_path}", flush=True)
    return history


def _check_checkpoint_path(file: str | Path) -> Path:
    """Validate a user-supplied checkpoint path.

    Parameters
    ----------
    file
        String or ``Path`` pointing to the desired checkpoint artifact.

    Returns
    -------
    pathlib.Path
        Normalized non-empty checkpoint path.
    """

    checkpoint_path = Path(file)
    if str(checkpoint_path).strip() == "":
        raise ValueError("file must be a non-empty checkpoint path.")
    if checkpoint_path.exists() and checkpoint_path.is_dir():
        raise ValueError("file must point to a checkpoint file, not a directory.")
    if checkpoint_path.suffix != ".keras":
        raise ValueError("file must use the .keras checkpoint extension.")
    return checkpoint_path
