"""Tests for scalar evaluation metrics."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

from bami.evaluation.metrics import (
    aggregate_data,
    compute_ccc,
    compute_corr,
    compute_rmse,
    estimate_recovery,
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


def test_aggregate_data_default_stats_for_one_variable():
    """Default aggregation should return mean, SE, and mean confidence limits."""

    df = pd.DataFrame({"rt": [1.0, 2.0, 3.0, 4.0]})
    out = aggregate_data(df, variables="rt")

    se = np.std(df["rt"], ddof=1) / np.sqrt(4)
    ci_width = t.ppf(0.975, df=3) * se

    assert out.columns.tolist() == ["variable", "mean", "se", "lower_ci", "upper_ci"]
    assert out.loc[0, "variable"] == "rt"
    assert np.isclose(out.loc[0, "mean"], 2.5)
    assert np.isclose(out.loc[0, "se"], se)
    assert np.isclose(out.loc[0, "lower_ci"], 2.5 - ci_width)
    assert np.isclose(out.loc[0, "upper_ci"], 2.5 + ci_width)


def test_aggregate_data_supports_multiple_variables_and_custom_stats():
    """Aggregation should summarize each requested variable in long format."""

    df = pd.DataFrame(
        {
            "rt": [1.0, 2.0, 3.0, 4.0],
            "accuracy": [0.0, 1.0, 1.0, 1.0],
        }
    )

    out = aggregate_data(
        df,
        variables=["rt", "accuracy"],
        stats=["n", "mean", "sd", "median", "lower_q", "upper_q"],
        q=0.5,
    )

    assert out["variable"].tolist() == ["rt", "accuracy"]
    assert out["n"].tolist() == [4, 4]
    np.testing.assert_allclose(out["mean"], [2.5, 0.75])
    np.testing.assert_allclose(out["median"], [2.5, 1.0])
    np.testing.assert_allclose(out["lower_q"], [1.75, 0.75])
    np.testing.assert_allclose(out["upper_q"], [3.25, 1.0])


def test_aggregate_data_groups_by_one_or_more_columns():
    """Group columns should be preserved before the variable column."""

    df = pd.DataFrame(
        {
            "condition": ["A", "A", "B", "B"],
            "param": ["x", "x", "x", "x"],
            "rt": [1.0, 3.0, 2.0, 6.0],
        }
    )

    one_group = aggregate_data(df, variables="rt", group_by="condition", stats=["mean"])
    two_groups = aggregate_data(
        df,
        variables="rt",
        group_by=["condition", "param"],
        stats=["mean"],
    )

    assert one_group.columns.tolist() == ["condition", "variable", "mean"]
    assert one_group["condition"].tolist() == ["A", "B"]
    np.testing.assert_allclose(one_group["mean"], [2.0, 4.0])
    assert two_groups.columns.tolist() == ["condition", "param", "variable", "mean"]
    np.testing.assert_allclose(two_groups["mean"], [2.0, 4.0])


def test_aggregate_data_rejects_invalid_inputs():
    """Invalid aggregation requests should fail with clear messages."""

    df = pd.DataFrame({"rt": [1.0, 2.0], "label": ["a", "b"]})

    with pytest.raises(ValueError, match="variables must contain"):
        aggregate_data(df, variables=[])
    with pytest.raises(ValueError, match="not found"):
        aggregate_data(df, variables="missing")
    with pytest.raises(ValueError, match="numeric"):
        aggregate_data(df, variables="label")
    with pytest.raises(ValueError, match="unsupported"):
        aggregate_data(df, variables="rt", stats=["mean", "bogus"])
    with pytest.raises(ValueError, match="ci must be between"):
        aggregate_data(df, variables="rt", ci=1.0)
    with pytest.raises(ValueError, match="q must be between"):
        aggregate_data(df, variables="rt", q=0.0)


def test_estimate_recovery_uses_explicit_id_cols():
    """Recovery estimation should pair rows with explicit identifiers."""

    simulated = pd.DataFrame(
        {
            "dataset_id": [0, 1, 0, 1],
            "param": ["a", "a", "b", "b"],
            "simulated_value": [1.0, 2.0, 2.0, 4.0],
        }
    )
    estimated = pd.DataFrame(
        {
            "dataset_id": [0, 1, 0, 1],
            "param": ["a", "a", "b", "b"],
            "estimated_value": [1.1, 1.9, 2.2, 3.8],
        }
    )

    out = estimate_recovery(
        simulated,
        estimated,
        id_cols=["dataset_id", "param"],
        group_by="param",
        metrics=["ccc", "corr", "rmse"],
    )

    assert out["param"].tolist() == ["a", "b"]
    assert out["n"].tolist() == [2, 2]
    np.testing.assert_allclose(out["corr"], [1.0, 1.0])
    np.testing.assert_allclose(
        out["rmse"],
        [
            compute_rmse([1.0, 2.0], [1.1, 1.9]),
            compute_rmse([2.0, 4.0], [2.2, 3.8]),
        ],
    )


def test_estimate_recovery_infers_id_cols_and_groups_by_fit_model():
    """Shared non-value columns should be used as pairing IDs by default."""

    simulated = pd.DataFrame(
        {
            "dataset_id": [0, 1],
            "param": ["a", "a"],
            "simulated_value": [1.0, 2.0],
        }
    )
    estimated = pd.DataFrame(
        {
            "dataset_id": [0, 1, 0, 1],
            "param": ["a", "a", "a", "a"],
            "fit_model": ["m1", "m1", "m2", "m2"],
            "estimated_value": [1.0, 2.0, 1.5, 2.5],
        }
    )

    out = estimate_recovery(
        simulated,
        estimated,
        group_by=["fit_model", "param"],
        metrics=["rmse"],
    )

    assert out[["fit_model", "param"]].to_dict("records") == [
        {"fit_model": "m1", "param": "a"},
        {"fit_model": "m2", "param": "a"},
    ]
    np.testing.assert_allclose(out["rmse"], [0.0, 0.5])


def test_estimate_recovery_handles_matching_value_column_names():
    """Matching value column names should not be used as pairing IDs."""

    simulated = pd.DataFrame(
        {
            "dataset_id": [0, 1],
            "param": ["a", "a"],
            "value": [1.0, 2.0],
        }
    )
    estimated = pd.DataFrame(
        {
            "dataset_id": [0, 1, 0, 1],
            "param": ["a", "a", "a", "a"],
            "fit_model": ["m1", "m1", "m2", "m2"],
            "value": [1.0, 2.0, 1.5, 2.5],
        }
    )

    out = estimate_recovery(
        simulated,
        estimated,
        group_by=["fit_model", "param"],
        metrics=["corr", "rmse"],
        simulated_col="value",
        estimated_col="value",
    )

    assert out[["fit_model", "param"]].to_dict("records") == [
        {"fit_model": "m1", "param": "a"},
        {"fit_model": "m2", "param": "a"},
    ]
    np.testing.assert_allclose(out["corr"], [1.0, 1.0])
    np.testing.assert_allclose(out["rmse"], [0.0, 0.5])


def test_estimate_recovery_rejects_invalid_inputs():
    """Invalid recovery requests should fail before returning misleading metrics."""

    simulated = pd.DataFrame({"dataset_id": [0], "simulated_value": [1.0]})
    estimated = pd.DataFrame({"dataset_id": [0], "estimated_value": [1.0]})

    with pytest.raises(ValueError, match="unsupported"):
        estimate_recovery(simulated, estimated, group_by="dataset_id", metrics=["mae"])
    with pytest.raises(ValueError, match="metrics must contain"):
        estimate_recovery(simulated, estimated, group_by="dataset_id", metrics=[])
    with pytest.raises(ValueError, match="not found"):
        estimate_recovery(
            simulated.drop(columns=["simulated_value"]),
            estimated,
            group_by="dataset_id",
            metrics=["rmse"],
        )
    with pytest.raises(ValueError, match="not found"):
        estimate_recovery(
            simulated,
            estimated,
            group_by="missing",
            metrics=["rmse"],
        )
    with pytest.raises(ValueError, match="No paired"):
        estimate_recovery(
            simulated,
            pd.DataFrame({"dataset_id": [99], "estimated_value": [1.0]}),
            group_by="dataset_id",
            metrics=["rmse"],
        )
