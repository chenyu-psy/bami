"""Tests for evaluation plotting helpers."""

import matplotlib
import pandas as pd
import pytest

from bami.evaluation.plots import (
    plot_calibration_ecdf,
    plot_coverage,
    plot_individual_recovery,
    plot_population_recovery,
    plot_trial_sensitivity_recovery,
    plot_zscore_contraction,
)

matplotlib.use("Agg")


def test_recovery_plots_run_with_minimal_input():
    """Population and individual recovery plotting should return figure objects."""

    recovery_df = pd.DataFrame(
        {
            "level": ["population", "population"] + ["individual"] * 4,
            "param": ["a_mu", "c_mu", "a", "a", "a", "a"],
            "true_value": [0.4, 0.8, 0.2, 0.4, 0.3, 0.5],
            "est_value": [0.5, 0.75, 0.25, 0.45, 0.35, 0.55],
            "dataset_id": [0, 0, 0, 0, 1, 1],
            "subject_id": [None, None, 0, 1, 0, 1],
        }
    )

    pop_fig = plot_population_recovery(recovery_df)
    ind_fig = plot_individual_recovery(recovery_df)

    assert pop_fig is not None
    assert ind_fig is not None
    assert pop_fig.axes[0].get_title() == "a_mu (r=nan)"
    assert ind_fig.axes[0].get_title() == "Individual Recovery by Simulation"


def test_population_recovery_respects_pars_order():
    """Population recovery should plot only requested parameters in the given order."""

    recovery_df = pd.DataFrame(
        {
            "level": ["population", "population", "population"],
            "param": ["a_mu", "c_mu", "tau_mu"],
            "true_value": [0.4, 0.8, 0.6],
            "est_value": [0.5, 0.75, 0.65],
        }
    )

    fig = plot_population_recovery(recovery_df, pars=["tau_mu", "a_mu"])
    titles = [ax.get_title() for ax in fig.axes if ax.axison]

    assert titles == ["tau_mu (r=nan)", "a_mu (r=nan)"]


def test_population_recovery_raises_for_missing_pars():
    """Population recovery should fail clearly when requested parameters are absent."""

    recovery_df = pd.DataFrame(
        {
            "level": ["population", "population"],
            "param": ["a_mu", "c_mu"],
            "true_value": [0.4, 0.8],
            "est_value": [0.5, 0.75],
        }
    )

    with pytest.raises(ValueError, match="missing_param"):
        plot_population_recovery(recovery_df, pars=["a_mu", "missing_param"])


def test_individual_recovery_respects_pars_order():
    """Individual recovery should plot only requested parameters in the given order."""

    recovery_df = pd.DataFrame(
        {
            "level": ["individual"] * 12,
            "param": [
                "a",
                "a",
                "a",
                "a",
                "c",
                "c",
                "c",
                "c",
                "tau",
                "tau",
                "tau",
                "tau",
            ],
            "true_value": [0.2, 0.4, 0.3, 0.5, 0.7, 0.9, 0.8, 1.0, 0.5, 0.7, 0.6, 0.8],
            "est_value": [
                0.25,
                0.45,
                0.35,
                0.55,
                0.72,
                0.92,
                0.82,
                1.02,
                0.48,
                0.68,
                0.58,
                0.78,
            ],
            "dataset_id": [0, 0, 1, 1] * 3,
            "subject_id": [0, 1, 0, 1] * 3,
        }
    )

    fig = plot_individual_recovery(recovery_df, pars=["tau", "a"])
    visible_axes = [ax for ax in fig.axes if ax.axison]
    tick_labels = [
        tick.get_text().split("\n")[0] for tick in visible_axes[0].get_xticklabels()
    ]
    jitter_count = sum(
        len(collection.get_offsets()) for collection in visible_axes[0].collections
    )

    assert len(visible_axes) == 1
    assert tick_labels == ["tau", "a"]
    assert jitter_count == 4
    assert visible_axes[0].get_ylim() == (-0.05, 1.05)


def test_individual_recovery_raises_for_missing_pars():
    """Individual recovery should fail clearly when requested parameters are absent."""

    recovery_df = pd.DataFrame(
        {
            "level": ["individual"] * 4,
            "param": ["a", "a", "c", "c"],
            "true_value": [0.2, 0.7, 0.3, 0.8],
            "est_value": [0.25, 0.72, 0.35, 0.85],
            "dataset_id": [0, 0, 0, 0],
            "subject_id": [0, 1, 0, 1],
        }
    )

    with pytest.raises(ValueError, match="missing_param"):
        plot_individual_recovery(recovery_df, pars=["a", "missing_param"])


def test_individual_recovery_requires_dataset_id():
    """Individual recovery correlation plots need simulation identifiers."""

    recovery_df = pd.DataFrame(
        {
            "level": ["individual", "individual"],
            "param": ["a", "a"],
            "true_value": [0.2, 0.7],
            "est_value": [0.25, 0.72],
            "subject_id": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="requires dataset_id"):
        plot_individual_recovery(recovery_df)


def test_trial_sensitivity_recovery_facets_by_parameter():
    """Trial sensitivity plots should use trial count on x and parameter facets."""

    recovery_df = pd.DataFrame(
        {
            "level": ["individual"] * 16,
            "param": ["a", "a", "a", "a", "c", "c", "c", "c"] * 2,
            "true_value": [0.2, 0.4, 0.3, 0.5, 0.7, 0.9, 0.8, 1.0] * 2,
            "est_value": [0.25, 0.45, 0.35, 0.55, 0.72, 0.92, 0.82, 1.02] * 2,
            "dataset_id": [0, 0, 1, 1, 0, 0, 1, 1] * 2,
            "subject_id": [0, 1, 0, 1, 0, 1, 0, 1] * 2,
            "trial_count": [50] * 8 + [200] * 8,
        }
    )

    fig = plot_trial_sensitivity_recovery(recovery_df, pars=["c", "a"])
    visible_axes = [ax for ax in fig.axes if ax.axison]
    titles = [ax.get_title() for ax in visible_axes]
    tick_labels = [tick.get_text() for tick in visible_axes[0].get_xticklabels()]

    assert titles == ["c", "a"]
    assert tick_labels == ["50", "200"]
    assert visible_axes[0].get_ylabel() == "Correlation"


def test_trial_sensitivity_recovery_requires_trial_count():
    """Trial sensitivity plots should fail clearly without trial_count."""

    recovery_df = pd.DataFrame(
        {
            "level": ["individual", "individual"],
            "param": ["a", "a"],
            "true_value": [0.2, 0.7],
            "est_value": [0.25, 0.72],
            "dataset_id": [0, 0],
            "subject_id": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="trial_count"):
        plot_trial_sensitivity_recovery(recovery_df)


def test_diagnostic_plots_run_with_minimal_input():
    """Diagnostic plotting functions should run for each metric class."""

    diag_df = pd.DataFrame(
        {
            "param": ["a", "c", "a", "c", "a", "c"],
            "metric": [
                "Log Gamma",
                "Log Gamma",
                "Calibration Error",
                "Calibration Error",
                "Posterior Contraction",
                "Posterior Contraction",
            ],
            "value": [1.2, 1.1, 0.1, 0.2, 0.5, 0.4],
        }
    )

    fig1 = plot_calibration_ecdf(diag_df)
    fig2 = plot_coverage(diag_df)
    fig3 = plot_zscore_contraction(diag_df)

    assert fig1 is not None
    assert fig2 is not None
    assert fig3 is not None
