"""Runtime helpers for BayesFlow analysis workflows.

The functions here keep training setup code outside model-family modules so
generic workflows can be used directly from researcher-facing notebooks.
"""


def configure_torch_device(device: str | None = None) -> str:
    """Set the Torch default device for BayesFlow/Keras training.

    Parameters
    ----------
    device
        Requested Torch device. Use ``"mps"`` on Apple Silicon, ``"cpu"`` for
        CPU training, or ``None`` to leave the current default unchanged.

    Returns
    -------
    str
        Device actually selected. Falls back to ``"cpu"`` when the requested
        accelerator is unavailable.
    """

    if device is None:
        return "unchanged"

    requested = str(device).lower()
    supported_devices = {"cpu", "mps", "cuda"}
    if requested not in supported_devices:
        raise ValueError(
            "torch_device must be one of None, 'cpu', 'mps', or 'cuda'. "
            "It controls the Torch device only; TensorFlow and JAX backends "
            "are not part of the current bami workflow contract."
        )

    try:
        import torch
    except Exception:
        return "unavailable"

    selected = requested
    if requested == "mps":
        if not torch.backends.mps.is_available():
            selected = "cpu"
    elif requested == "cuda":
        if not torch.cuda.is_available():
            selected = "cpu"

    torch.set_default_device(selected)
    return selected
