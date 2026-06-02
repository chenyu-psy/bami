"""Runtime helpers for BayesFlow analysis workflows.

The functions here keep backend device selection explicit. BayesFlow uses the
Keras torch backend, so setting only PyTorch's default device is not enough:
Keras also needs its own device context when workflows build, train, load, and
sample.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import os

SUPPORTED_DEVICES = {"cpu", "mps", "cuda"}


def validate_device(device: str | None) -> str:
    """Return a supported workflow runtime device name.

    Args:
        device:
            User-facing workflow device. ``None`` means the stable package
            default, ``"cpu"``.

    Returns:
        str: Normalized device name: ``"cpu"``, ``"mps"``, or ``"cuda"``.

    Raises:
        ValueError: If the device name is unsupported or the requested
            accelerator is unavailable.
    """

    if device is None:
        return "cpu"

    selected = str(device).lower()
    if selected not in SUPPORTED_DEVICES:
        raise ValueError("device must be one of 'cpu', 'mps', or 'cuda'.")

    try:
        import torch
    except Exception as exc:
        if selected == "cpu":
            return selected
        raise ValueError(f"device='{selected}' requires PyTorch.") from exc

    if selected == "mps" and not torch.backends.mps.is_available():
        raise ValueError("device='mps' was requested, but MPS is unavailable.")
    if selected == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' was requested, but CUDA is unavailable.")

    return selected


@contextmanager
def runtime_device(device: str | None) -> Iterator[str]:
    """Run Keras torch backend work on one explicit workflow device.

    Args:
        device:
            Workflow runtime device. ``None`` uses ``"cpu"``. The context sets
            both Keras' backend device and PyTorch's default device for the
            duration of the block, then restores the prior PyTorch default.

    Yields:
        str: Normalized selected device.
    """

    selected = validate_device(device)
    if selected == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    try:
        import keras
        import torch
    except Exception:
        yield selected
        return

    try:
        previous_torch_device = torch.get_default_device()
    except Exception:
        previous_torch_device = "cpu"

    torch.set_default_device(selected)
    try:
        with keras.device(selected):
            yield selected
    finally:
        torch.set_default_device(previous_torch_device)
