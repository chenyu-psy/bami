"""Tests for researcher-facing checkpoint-loading aliases."""

from bami.inference.checkpoints import load_model, load_workflow_weights


def test_load_model_alias_is_available():
    """load_model should remain a simple alias for workflow checkpoint loading."""

    assert load_model is not None
    assert load_workflow_weights is not None
