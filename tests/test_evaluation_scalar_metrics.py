"""Tests for scalar evaluation metrics."""

import numpy as np

from bami.evaluation.metrics import (
    compute_ccc,
    compute_corr,
    compute_rmse,
)


def test_compute_corr_near_one():
    """Correlation should be high for nearly aligned arrays."""

    truth = np.array([1.0, 2.0, 3.0, 4.0])
    estimate = np.array([1.1, 1.9, 3.2, 3.9])
    r = compute_corr(truth, estimate)
    assert 0.95 < r <= 1.0


def test_compute_ccc_penalizes_scale_mismatch():
    """CCC should be lower than Pearson r when estimates are compressed."""

    truth = np.array([1.0, 2.0, 3.0, 4.0])
    estimate = np.array([1.5, 2.0, 2.5, 3.0])

    assert np.isclose(compute_corr(truth, estimate), 1.0)
    assert compute_ccc(truth, estimate) < 1.0


def test_compute_rmse_matches_manual_value():
    """RMSE should match the direct squared-error calculation."""

    truth = np.array([1.0, 2.0, 3.0])
    estimate = np.array([1.0, 4.0, 5.0])

    assert np.isclose(compute_rmse(truth, estimate), np.sqrt(8.0 / 3.0))
