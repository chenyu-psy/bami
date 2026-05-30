"""Tests for runtime helpers that configure supported training devices."""

import os
import subprocess
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


def test_configure_torch_device_rejects_unsupported_names():
    """Invalid device names should fail with a bami-level explanation."""

    with pytest.raises(ValueError, match="None, 'cpu', 'mps', or 'cuda'"):
        configure_torch_device("gpu")


def test_configure_torch_device_uses_cpu_when_requested(monkeypatch):
    """Explicit CPU training should set the Torch default device to CPU."""

    fake_torch = _FakeTorch(mps_available=False, cuda_available=False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)

    selected = configure_torch_device("cpu")

    assert selected == "cpu"
    assert fake_torch.selected_devices == ["cpu"]
    assert "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ


def test_configure_torch_device_enables_mps_cpu_fallback(monkeypatch, capsys):
    """Available MPS should use MPS with CPU fallback for missing ops."""

    fake_torch = _FakeTorch(mps_available=True, cuda_available=True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)

    selected = configure_torch_device("mps")

    captured = capsys.readouterr()
    assert selected == "mps"
    assert fake_torch.selected_devices == ["mps"]
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
    assert "CPU fallback for unsupported MPS operations" in captured.out


def test_configure_torch_device_falls_back_when_mps_is_unavailable(monkeypatch, capsys):
    """Unavailable Apple Silicon MPS should fall back to CPU."""

    fake_torch = _FakeTorch(mps_available=False, cuda_available=True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)

    selected = configure_torch_device("mps")

    captured = capsys.readouterr()
    assert selected == "cpu"
    assert fake_torch.selected_devices == ["cpu"]
    assert "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ
    assert "Requested Torch device 'mps' is unavailable; using 'cpu'." in captured.out


def test_configure_torch_device_falls_back_when_cuda_is_unavailable(
    monkeypatch, capsys
):
    """Unavailable CUDA should fall back to CPU."""

    fake_torch = _FakeTorch(mps_available=True, cuda_available=False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    selected = configure_torch_device("cuda")

    captured = capsys.readouterr()
    assert selected == "cpu"
    assert fake_torch.selected_devices == ["cpu"]
    assert "Requested Torch device 'cuda' is unavailable; using 'cpu'." in captured.out
