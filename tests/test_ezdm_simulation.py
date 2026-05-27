"""Tests for formula-based ezDM data generation."""

import numpy as np
import pandas as pd
import pytest

from bami.simulators.ezdm import (
    moments_to_params,
    params_to_moments,
    simulate_ezdm_summary,
    simulate_hierarchical,
    simulate_simple,
)


def test_params_to_moments_matches_paper_example():
    """The formula should reproduce the Wagenmakers et al. appendix example."""

    moments = params_to_moments(v=0.1, a=0.14, t0=0.3, s=0.1)

    assert moments["pc"] == pytest.approx(0.802, abs=0.001)
    assert moments["mrt"] == pytest.approx(0.723, abs=0.001)
    assert moments["vrt"] == pytest.approx(0.112, abs=0.001)


def test_moments_to_params_matches_paper_example():
    """The inverse formula should recover the appendix parameter values."""

    params = moments_to_params(pc=0.802, vrt=0.112, mrt=0.723, s=0.1)

    assert params["v"] == pytest.approx(0.09993853)
    assert params["a"] == pytest.approx(0.1399702)
    assert params["t0"] == pytest.approx(0.30003)


def test_simulate_simple_is_reproducible_and_has_expected_columns():
    """Simple simulation should return stable summary-level data for a seed."""

    first = simulate_simple(v=0.1, a=0.14, t0=0.3, n_trials=50, seed=2026)
    second = simulate_simple(v=0.1, a=0.14, t0=0.3, n_trials=50, seed=2026)

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == ["n_trials", "pc", "mrt", "vrt", "v", "a", "t0"]
    assert len(first) == 1
    assert first.loc[0, "n_trials"] == 50
    assert 0 < first.loc[0, "pc"] < 1
    assert first.loc[0, "pc"] != 0.5
    assert first.loc[0, "mrt"] > 0
    assert first.loc[0, "vrt"] > 0
    assert first.loc[0, "v"] == 0.1
    assert first.loc[0, "a"] == 0.14
    assert first.loc[0, "t0"] == 0.3


def test_ezdm_summary_stays_finite_for_extreme_finite_parameters():
    """Extreme finite parameters should not create NaN summary values."""

    rng = np.random.default_rng(2026)
    summary = simulate_ezdm_summary(
        v=10,
        a=10,
        t0=0.2,
        n_trials=100,
        rng=rng,
    )

    assert np.all(np.isfinite(summary))
    assert 0 < summary[0] < 1
    assert summary[1] > 0
    assert summary[2] > 0


def test_simulate_hierarchical_draws_subject_parameters_and_group_columns():
    """Hierarchical simulation should return one summary row per subject."""

    group_params = {
        "mu_v": 0.1,
        "sd_v": 0.02,
        "mu_a": 0.14,
        "sd_a": 0.01,
        "mu_t0": 0.3,
        "sd_t0": 0.02,
    }

    data = simulate_hierarchical(
        group_params=group_params,
        n_subjects=4,
        n_trials=50,
        seed=2026,
    )

    assert list(data.columns) == [
        "subj",
        "n_trials",
        "pc",
        "mrt",
        "vrt",
        "v",
        "a",
        "t0",
        "mu_v",
        "sd_v",
        "mu_a",
        "sd_a",
        "mu_t0",
        "sd_t0",
    ]
    assert len(data) == 4
    assert data["subj"].nunique() == 4
    assert np.all(data["n_trials"] == 50)
    assert np.all((data["pc"] > 0) & (data["pc"] < 1))
    assert np.all(data["mrt"] > 0)
    assert np.all(data["vrt"] > 0)
    assert np.all(data["a"] > 0)
    assert np.all(data["t0"] > 0)

    for name, value in group_params.items():
        assert np.all(data[name] == value)


def test_simulate_hierarchical_rejects_missing_group_parameters():
    """Missing group-level values should produce a readable error."""

    with pytest.raises(ValueError, match="Missing group parameter"):
        simulate_hierarchical(
            group_params={"mu_v": 0.1},
            n_subjects=2,
            n_trials=5,
            seed=2026,
        )
