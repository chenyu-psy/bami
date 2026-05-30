"""Model checkpoint helpers for BayesFlow approximators.

These helpers use BayesFlow/Keras-native save/load for the workflow
approximator, which is more robust than ad-hoc object pickling.
"""

from pathlib import Path
import os

os.environ.setdefault("KERAS_BACKEND", "torch")
import keras


def save_workflow_weights(model, checkpoint_path: str | Path) -> Path:
    """Save trained workflow approximator to a `.keras` artifact.

    Args:
        model (object): Trained model with a BayesFlow ``workflow.approximator``.
        checkpoint_path: Path to the ``.keras`` checkpoint file.

    Returns:
        pathlib.Path: Saved checkpoint path.
    """

    ckpt_path = Path(checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    model.workflow.approximator.save(ckpt_path)
    return ckpt_path


def load_workflow_weights(model, checkpoint_path: str | Path) -> object:
    """Load a `.keras` approximator artifact into a model workflow.

    Args:
        model (object): Model instance with initialized BayesFlow workflow.
        checkpoint_path: Path to the ``.keras`` checkpoint file.

    Returns:
        object: The same model with loaded network weights.
    """

    ckpt_path = Path(checkpoint_path)
    model.workflow.approximator = keras.saving.load_model(ckpt_path)
    return model


def load_model(model, checkpoint_path: str | Path) -> object:
    """Load a trained checkpoint into a configured model shell.

    Args:
        model (object): Model instance with initialized BayesFlow workflow.
        checkpoint_path: Path to the saved ``.keras`` checkpoint.

    Returns:
        object: The same model with trained network weights loaded.
    """

    return load_workflow_weights(model, checkpoint_path)
