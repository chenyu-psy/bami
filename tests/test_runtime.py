"""Tests for runtime helpers that configure supported training devices."""

import sys
from types import SimpleNamespace

import pytest

from bami.inference.runtime import configure_torch_device


class _FakeTorch:
    """Small Torch stand-in used to test device selection without hardware."""

    def __init__(self, *, mps_available: bool, cuda_available: bool):
        """Create backend availability flags and an empty selected-device list."""

        self.selected_devices = []
        self.backends = SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps_available)
        )
        self.cuda = SimpleNamespace(is_available=lambda: cuda_available)

    def set_default_device(self, device: str) -> None:
        """Record the requested default device."""

        self.selected_devices.append(device)


def test_configure_torch_device_none_leaves_default_unchanged():
    """None should not import Torch or change the current default device."""

    assert configure_torch_device(None) == "unchanged"


def test_configure_torch_device_rejects_unsupported_names():
    """Invalid device names should fail with a bami-level explanation."""

    with pytest.raises(ValueError, match="None, 'cpu', 'mps', or 'cuda'"):
        configure_torch_device("gpu")


def test_configure_torch_device_uses_cpu_when_requested(monkeypatch):
    """Explicit CPU training should set the Torch default device to CPU."""

    fake_torch = _FakeTorch(mps_available=False, cuda_available=False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    selected = configure_torch_device("cpu")

    assert selected == "cpu"
    assert fake_torch.selected_devices == ["cpu"]


def test_configure_torch_device_falls_back_when_mps_is_unavailable(monkeypatch):
    """Unavailable Apple Silicon MPS should fall back to CPU."""

    fake_torch = _FakeTorch(mps_available=False, cuda_available=True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    selected = configure_torch_device("mps")

    assert selected == "cpu"
    assert fake_torch.selected_devices == ["cpu"]


def test_configure_torch_device_falls_back_when_cuda_is_unavailable(monkeypatch):
    """Unavailable CUDA should fall back to CPU."""

    fake_torch = _FakeTorch(mps_available=True, cuda_available=False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    selected = configure_torch_device("cuda")

    assert selected == "cpu"
    assert fake_torch.selected_devices == ["cpu"]
