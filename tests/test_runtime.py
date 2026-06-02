"""Tests for runtime helpers that configure workflow devices."""

import os
import subprocess
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from bami.inference.runtime import runtime_device, validate_device


class _FakeTorch:
    """Small Torch stand-in used to test device selection without hardware."""

    def __init__(self, *, mps_available: bool, cuda_available: bool):
        """Create backend availability flags and selected-device tracking."""

        self.selected_devices = []
        self.backends = SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps_available)
        )
        self.cuda = SimpleNamespace(is_available=lambda: cuda_available)
        self._default_device = "cpu"

    def get_default_device(self):
        """Return the current fake default device."""

        return self._default_device

    def set_default_device(self, device: str) -> None:
        """Record and store the requested default device."""

        self.selected_devices.append(device)
        self._default_device = device


class _FakeKeras:
    """Small Keras stand-in exposing the public device context."""

    def __init__(self):
        """Create an empty list of requested device contexts."""

        self.devices = []

    @contextmanager
    def device(self, device: str):
        """Record the requested device for the duration of the context."""

        self.devices.append(device)
        yield


def test_validate_device_defaults_to_cpu():
    """Missing device should use the stable CPU default."""

    assert validate_device(None) == "cpu"


def test_package_import_sets_mps_fallback_before_torch_backend():
    """Importing bami should set the MPS fallback before BayesFlow loads Torch."""

    code = (
        "import os\n"
        "os.environ.pop('PYTORCH_ENABLE_MPS_FALLBACK', None)\n"
        "import bami\n"
        "print(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "1"


def test_validate_device_rejects_unsupported_names():
    """Invalid device names should fail with a bami-level explanation."""

    with pytest.raises(ValueError, match="'cpu', 'mps', or 'cuda'"):
        validate_device("gpu")


def test_validate_device_accepts_cpu_with_fake_torch(monkeypatch):
    """CPU should not require accelerator availability."""

    fake_torch = _FakeTorch(mps_available=False, cuda_available=False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert validate_device("cpu") == "cpu"


def test_validate_device_rejects_unavailable_mps(monkeypatch):
    """Unavailable Apple Silicon MPS should fail instead of silently falling back."""

    fake_torch = _FakeTorch(mps_available=False, cuda_available=True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(ValueError, match="MPS is unavailable"):
        validate_device("mps")


def test_validate_device_rejects_unavailable_cuda(monkeypatch):
    """Unavailable CUDA should fail instead of silently falling back."""

    fake_torch = _FakeTorch(mps_available=True, cuda_available=False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(ValueError, match="CUDA is unavailable"):
        validate_device("cuda")


def test_runtime_device_sets_keras_and_torch_devices(monkeypatch):
    """Runtime context should configure Keras and PyTorch consistently."""

    fake_torch = _FakeTorch(mps_available=False, cuda_available=False)
    fake_keras = _FakeKeras()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "keras", fake_keras)

    with runtime_device("cpu") as selected:
        assert selected == "cpu"

    assert fake_keras.devices == ["cpu"]
    assert fake_torch.selected_devices == ["cpu", "cpu"]


def test_runtime_device_enables_mps_fallback(monkeypatch):
    """Explicit MPS should enable PyTorch's CPU fallback for unsupported ops."""

    fake_torch = _FakeTorch(mps_available=True, cuda_available=False)
    fake_keras = _FakeKeras()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "keras", fake_keras)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)

    with runtime_device("mps") as selected:
        assert selected == "mps"

    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
    assert fake_keras.devices == ["mps"]
