"""Tests for fit_workflow checkpoint loading and saving."""

import pytest

from bami.training import fit_workflow


class _ToyApproximator:
    """Small save target used to test checkpoint writing."""

    def save(self, path) -> None:
        """Write a tiny marker file to the requested checkpoint path."""

        path.write_text("saved", encoding="utf-8")


class _ToyTrainWorkflow:
    """Minimal workflow object exposing an approximator."""

    def __init__(self):
        """Create a workflow with a fake approximator."""

        self.approximator = _ToyApproximator()


class _ToyTrainModel:
    """Minimal model object exposing the fit_workflow interface."""

    def __init__(self):
        """Create a fake model and record train_workflow calls."""

        self.workflow = _ToyTrainWorkflow()
        self.fit_kwargs = None
        self.fit_count = 0

    def train_workflow(self, **kwargs):
        """Record training settings and return a fake history."""

        self.fit_count += 1
        self.fit_kwargs = kwargs
        return {"loss": [1.0]}


def _train_kwargs() -> dict:
    """Return the smallest valid training settings for fit_workflow tests."""

    return {
        "max_epochs": 1,
        "initial_epochs": 1,
        "n_batch": 1,
        "batch_size": 1,
        "validation_data": 1,
        "patience": 1,
        "min_delta": 0.0,
        "workers": 1,
        "max_queue_size": 1,
        "torch_device": None,
        "verbose": 0,
    }


def test_fit_workflow_saves_checkpoint_when_file_is_supplied(tmp_path):
    """fit_workflow should save workflow weights after a successful fit."""

    model = _ToyTrainModel()
    checkpoint_path = tmp_path / "toy_workflow.keras"

    history = fit_workflow(model, file=checkpoint_path, **_train_kwargs())

    assert history == {"loss": [1.0]}
    assert checkpoint_path.read_text(encoding="utf-8") == "saved"
    assert model.fit_kwargs["keep_optimizer"] is True


def test_fit_workflow_loads_existing_checkpoint_by_default(tmp_path, monkeypatch):
    """Existing checkpoints should be loaded unless overwrite is requested."""

    model = _ToyTrainModel()
    checkpoint_path = tmp_path / "toy_workflow.keras"
    checkpoint_path.write_text("existing", encoding="utf-8")
    loaded_paths = []

    def fake_load(model_arg, path_arg):
        """Record the requested load path and return the model."""

        loaded_paths.append(path_arg)
        return model_arg

    monkeypatch.setattr("bami.inference.checkpoints.load_workflow_weights", fake_load)

    result = fit_workflow(model, file=checkpoint_path, **_train_kwargs())

    assert result == {"loaded": True, "file": checkpoint_path}
    assert loaded_paths == [checkpoint_path]
    assert model.fit_count == 0


def test_fit_workflow_overwrites_existing_checkpoint_when_requested(tmp_path):
    """overwrite=True should force training and save over an existing file."""

    model = _ToyTrainModel()
    checkpoint_path = tmp_path / "toy_workflow.keras"
    checkpoint_path.write_text("existing", encoding="utf-8")

    history = fit_workflow(
        model,
        file=checkpoint_path,
        overwrite=True,
        **_train_kwargs(),
    )

    assert history == {"loss": [1.0]}
    assert checkpoint_path.read_text(encoding="utf-8") == "saved"
    assert model.fit_count == 1


def test_fit_workflow_rejects_invalid_checkpoint_paths(tmp_path):
    """Checkpoint files must be non-directory .keras paths."""

    model = _ToyTrainModel()

    with pytest.raises(ValueError, match="not a directory"):
        fit_workflow(model, file=tmp_path, **_train_kwargs())

    with pytest.raises(ValueError, match=".keras"):
        fit_workflow(
            model,
            file=tmp_path / "toy_workflow.pt",
            **_train_kwargs(),
        )
