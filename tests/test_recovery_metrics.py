"""Tests for recovery metric helpers."""

import numpy as np
import pandas as pd
import pytest

from bami.recovery.metrics import (
    compute_ccc,
    compute_pearson_r,
    validate_result_schema,
)


def test_compute_pearson_r_near_one():
    """Correlation should be high for nearly aligned arrays."""

    truth = np.array([1.0, 2.0, 3.0, 4.0])
    estimate = np.array([1.1, 1.9, 3.2, 3.9])
    r = compute_pearson_r(truth, estimate)
    assert 0.95 < r <= 1.0


def test_compute_ccc_penalizes_scale_mismatch():
    """CCC should be lower than Pearson r when estimates are compressed."""

    truth = np.array([1.0, 2.0, 3.0, 4.0])
    estimate = np.array([1.5, 2.0, 2.5, 3.0])

    assert np.isclose(compute_pearson_r(truth, estimate), 1.0)
    assert compute_ccc(truth, estimate) < 1.0


def test_validate_result_schema_missing_column_raises():
    """Schema validator should raise when required columns are missing."""

    df = pd.DataFrame({"data_source": ["simple"], "fit_model": ["simple"]})
    with pytest.raises(ValueError):
        validate_result_schema(df, ["data_source", "fit_model", "r"])
