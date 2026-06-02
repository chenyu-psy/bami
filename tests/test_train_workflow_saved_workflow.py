"""Tests for train_workflow saved-workflow loading and saving."""

import pytest

from bami.workflows.simple import SimpleWorkflow


class _ToyApproximator:
    """Small save target used to test saved workflow writing."""

    def save(self, path) -> None:
        """Write a tiny marker file to the requested saved workflow path."""

        path.write_text("saved", encoding="utf-8")


class _ToyTrainWorkflow:
    """Minimal workflow object exposing an approximator and training method."""

    def __init__(self):
        """Create a workflow with a fake approximator."""

        self.approximator = _ToyApproximator()
        self.history = None

    def fit_online(self, **kwargs):
        """Record training settings and return a fake history object."""

        self.fit_kwargs = kwargs

        class _History:
            """Tiny Keras-like history container."""

            history = {"loss": [1.0], "val_loss": [1.0]}

        return _History()

    def simulate(self, n_data):
        """Return fake validation data with the requested size."""

        return {"data": [0] * n_data}


class _ToyModel(SimpleWorkflow):
    """Small SimpleWorkflow shell used to test train_workflow file behavior."""

    def __init__(self):
        """Create a fake workflow model without running the full constructor."""

        self.workflow = _ToyTrainWorkflow()
        self.validation_data = None


def _train_kwargs() -> dict:
    """Return the smallest valid training settings for train_workflow tests."""

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
        "verbose": 0,
    }


def test_train_workflow_saves_saved_workflow_when_file_is_supplied(tmp_path):
    """train_workflow should save workflow weights after a successful fit."""

    model = _ToyModel()
    saved_path = tmp_path / "toy_workflow.keras"

    history = model.train_workflow(file=saved_path, **_train_kwargs())

    assert history.history == {"loss": [1.0], "val_loss": [1.0]}
    assert saved_path.read_text(encoding="utf-8") == "saved"
    assert model.workflow.fit_kwargs["keep_optimizer"] is True


def test_train_workflow_loads_existing_saved_workflow_by_default(tmp_path, monkeypatch):
    """Existing saved workflows should be loaded unless overwrite is requested."""

    model = _ToyModel()
    saved_path = tmp_path / "toy_workflow.keras"
    saved_path.write_text("existing", encoding="utf-8")
    loaded_paths = []

    def fake_load(model_arg, path_arg):
        """Record the requested load path and return the model."""

        loaded_paths.append(path_arg)
        return model_arg

    monkeypatch.setattr("bami.inference.checkpoints.load_workflow_weights", fake_load)

    result = model.train_workflow(file=saved_path, **_train_kwargs())

    assert result == {"loaded": True, "file": saved_path}
    assert loaded_paths == [saved_path]
    assert not hasattr(model.workflow, "fit_kwargs")


def test_train_workflow_overwrites_existing_saved_workflow_when_requested(tmp_path):
    """overwrite=True should force training and save over an existing file."""

    model = _ToyModel()
    saved_path = tmp_path / "toy_workflow.keras"
    saved_path.write_text("existing", encoding="utf-8")

    history = model.train_workflow(
        file=saved_path,
        overwrite=True,
        **_train_kwargs(),
    )

    assert history.history == {"loss": [1.0], "val_loss": [1.0]}
    assert saved_path.read_text(encoding="utf-8") == "saved"
    assert model.workflow.fit_kwargs["keep_optimizer"] is True


def test_train_workflow_rejects_invalid_saved_workflow_paths(tmp_path):
    """Saved workflow files must be non-directory .keras paths."""

    model = _ToyModel()

    with pytest.raises(ValueError, match="not a directory"):
        model.train_workflow(file=tmp_path, **_train_kwargs())

    with pytest.raises(ValueError, match=".keras"):
        model.train_workflow(
            file=tmp_path / "toy_workflow.pt",
            **_train_kwargs(),
        )
