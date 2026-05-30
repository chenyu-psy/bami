"""Tests for researcher-facing workflow training defaults."""

import inspect

from bami.workflows import training
from bami.workflows.hierarchical import HierarchicalWorkflow
from bami.workflows.simple import SimpleWorkflow

EXPECTED_TRAIN_DEFAULTS = {
    "max_epochs": 100,
    "initial_epochs": 5,
    "n_batch": 2000,
    "batch_size": 32,
    "validation_data": 200,
    "patience": 5,
    "min_delta": 0.1,
    "workers": 1,
    "max_queue_size": 4,
    "torch_device": None,
    "verbose": 1,
}


def test_default_train_config_uses_general_research_defaults():
    """Shared training defaults should balance fit quality and machine load."""

    config = training.default_train_config()

    for key, value in EXPECTED_TRAIN_DEFAULTS.items():
        assert config[key] == value
    assert config["fit_kwargs"] == {}


def test_workflow_train_method_signatures_match_shared_defaults():
    """Public workflow methods should expose the same default training values."""

    methods = [
        training.train_workflow,
        SimpleWorkflow.train_workflow,
        HierarchicalWorkflow.train_workflow,
    ]

    for method in methods:
        signature = inspect.signature(method)
        for key, value in EXPECTED_TRAIN_DEFAULTS.items():
            assert signature.parameters[key].default == value


def test_random_workflow_without_group_config_uses_shared_defaults(monkeypatch):
    """Random-effect workflow training should inherit shared defaults by default."""

    captured = {}

    def fake_train_workflow(model, **kwargs):
        """Capture random-workflow training settings without fitting."""

        captured["workflow"] = model.workflow
        captured["kwargs"] = kwargs
        return {"trained": True}

    monkeypatch.setattr(training, "train_workflow", fake_train_workflow)
    model = HierarchicalWorkflow.__new__(HierarchicalWorkflow)
    model.random_workflow = object()

    out = model.train_random_workflow()

    assert out == {"trained": True}
    assert captured["workflow"] is model.random_workflow
    for key, value in EXPECTED_TRAIN_DEFAULTS.items():
        assert captured["kwargs"][key] == value
        assert model._random_train_config[key] == value
