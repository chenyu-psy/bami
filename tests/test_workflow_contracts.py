"""Tests for model-agnostic workflow observation validation."""

from pathlib import Path

import pytest

from bami.workflows import validate_observation

ROOT = Path(__file__).resolve().parents[1]


def test_observation_validation_accepts_public_contracts():
    """Workflow observation contracts should be explicit and reusable."""

    assert validate_observation("aggregate", "DemoWorkflow") == "aggregate"
    assert validate_observation("trial", "DemoWorkflow") == "trial"


def test_observation_validation_rejects_missing_or_unknown_values():
    """Observation errors should tell users the supported options."""

    with pytest.raises(ValueError, match="observation='aggregate'"):
        validate_observation(None, "DemoWorkflow")
    with pytest.raises(ValueError, match="aggregate.*trial"):
        validate_observation("summary", "DemoWorkflow")


def test_workflow_modules_do_not_define_model_family_presets():
    """Generic workflow modules should not contain model-specific presets."""

    workflow_files = {path.name for path in (ROOT / "src/bami/workflows").glob("*.py")}

    assert "ezdm.py" not in workflow_files
    assert "sdm.py" not in workflow_files
    assert "m3.py" not in workflow_files
