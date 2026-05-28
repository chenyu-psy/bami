"""Tests for model-agnostic workflow contract validation."""

from pathlib import Path

import pytest

from bami.workflows import validate_observation, validate_workflow_contract

ROOT = Path(__file__).resolve().parents[1]


def _toy_simulator(theta, n_trials, rng):
    """Return a tiny subject row for contract tests.

    Parameters
    ----------
    theta
        Public toy parameter.
    n_trials
        Number of simulated trials.
    rng
        Random generator accepted for API compatibility.

    Returns
    -------
    list[int]
        One simple data row.
    """

    return [theta, n_trials]


def _subject_loglik(data_row, candidates):
    """Return a dummy subject log likelihood for contract tests.

    Parameters
    ----------
    data_row
        Observed subject data row.
    candidates
        Candidate parameter table.

    Returns
    -------
    float
        Dummy log likelihood value.
    """

    return 0.0


def test_valid_training_contract_passes_without_loglik():
    """Workflow fitting and group recovery should not require subject_loglik."""

    contract = {
        "name": "demo",
        "param_names": ["theta"],
        "priors": {"theta": {"mean": 0.0, "sd": 1.0, "link": "identity"}},
        "simulator": _toy_simulator,
        "data_width": 1,
    }

    out = validate_workflow_contract(contract)

    assert out["name"] == "demo"
    assert out["param_names"] == ["theta"]
    assert out["data_width"] == 1


def test_individual_recovery_contract_requires_loglik():
    """Posthoc individual recovery should require a subject scoring function."""

    contract = {
        "name": "demo",
        "param_names": ["theta"],
        "priors": {},
        "simulator": _toy_simulator,
        "data_width": 1,
    }

    with pytest.raises(ValueError, match="subject_loglik"):
        validate_workflow_contract(contract, require_loglik=True)

    contract["subject_loglik"] = _subject_loglik
    out = validate_workflow_contract(contract, require_loglik=True)

    assert out["subject_loglik"] is _subject_loglik


def test_contract_validation_reports_missing_fields():
    """Missing required fields should produce a readable error."""

    with pytest.raises(ValueError, match="missing required fields"):
        validate_workflow_contract({"name": "demo"})


def test_contract_validation_rejects_bad_callables():
    """Simulator and likelihood fields should be real callables."""

    contract = {
        "name": "demo",
        "param_names": ["theta"],
        "priors": {},
        "simulator": "not a function",
        "data_width": 1,
    }

    with pytest.raises(ValueError, match="simulator"):
        validate_workflow_contract(contract)


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
