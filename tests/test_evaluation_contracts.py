"""Tests for evaluation table contracts."""

import pandas as pd
import pytest

from bami.evaluation.contracts import (
    validate_diagnostic_contract,
    validate_recovery_contract,
)


def test_validate_recovery_contract_accepts_minimum_schema():
    """Recovery validator should accept required columns and valid levels."""

    df = pd.DataFrame(
        {
            "level": ["population", "individual"],
            "param": ["a_mu", "a"],
            "true_value": [0.5, 0.2],
            "est_value": [0.6, 0.3],
        }
    )
    validate_recovery_contract(df)


def test_validate_recovery_contract_rejects_invalid_level():
    """Recovery validator should reject unsupported level values."""

    df = pd.DataFrame(
        {
            "level": ["group"],
            "param": ["a_mu"],
            "true_value": [0.5],
            "est_value": [0.6],
        }
    )
    with pytest.raises(ValueError, match="unsupported level"):
        validate_recovery_contract(df)


def test_validate_diagnostic_contract_requires_metric_columns():
    """Diagnostic validator should reject tables with missing required columns."""

    df = pd.DataFrame({"param": ["a"], "value": [0.1]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_diagnostic_contract(df)
