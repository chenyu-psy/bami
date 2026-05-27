"""Tests for preset model simulator functions."""

import numpy as np
import pytest

from bami.simulators import (
    choice_probs,
    errors_to_ms_sdm_summary,
    ezdm_moments,
    m3_activation,
    simulate_ezdm_summary,
    simulate_m3_counts,
    simulate_ms_sdm_summary,
    simulate_sdm_errors,
)


def test_m3_activation_and_choice_probabilities_are_valid():
    """M3 helpers should produce valid probabilities under supported rules."""

    activations = m3_activation(a=0.8, c=0.6, ra=0.4, rc=0.3, b=0)
    softmax_probs = choice_probs(
        activations,
        n_options=[1, 3, 1, 3, 4],
        rule="softmax",
    )
    luce_probs = choice_probs(
        activations,
        n_options=[1, 3, 1, 3, 4],
        rule="luce",
    )

    assert activations.shape == (5,)
    assert softmax_probs.shape == (5,)
    assert luce_probs.shape == (5,)
    assert np.all(softmax_probs > 0)
    assert np.all(luce_probs >= 0)
    assert np.isclose(softmax_probs.sum(), 1.0)
    assert np.isclose(luce_probs.sum(), 1.0)


def test_m3_simulator_returns_reproducible_counts():
    """M3 count simulation should be reproducible with an explicit rng."""

    first = simulate_m3_counts(
        a=0.8,
        c=0.6,
        ra=0.4,
        rc=0.3,
        n_trials=50,
        rng=np.random.default_rng(2026),
    )
    second = simulate_m3_counts(
        a=0.8,
        c=0.6,
        ra=0.4,
        rc=0.3,
        n_trials=50,
        rng=np.random.default_rng(2026),
    )

    assert np.array_equal(first, second)
    assert first.shape == (5,)
    assert np.all(first >= 0)
    assert np.allclose(first, np.round(first))
    assert first.sum() == 50


def test_m3_luce_rejects_negative_strengths():
    """Luce choice should not silently accept negative strengths."""

    with pytest.raises(ValueError, match="nonnegative"):
        choice_probs([1.0, -0.1, 0.2], rule="luce")


def test_sdm_error_simulator_returns_trial_level_errors():
    """SDM error simulation should return reproducible continuous trial rows."""

    first = simulate_sdm_errors(
        c=3.0,
        kappa=4.0,
        n_trials=40,
        rng=np.random.default_rng(2026),
    )
    second = simulate_sdm_errors(
        c=3.0,
        kappa=4.0,
        n_trials=40,
        rng=np.random.default_rng(2026),
    )

    assert np.array_equal(first, second)
    assert first.shape == (40, 1)
    assert np.all(first >= -180)
    assert np.all(first < 180)
    assert not np.allclose(first, np.round(first))


def test_sdm_error_simulator_can_return_scaled_errors():
    """SDM error simulation should support unit-scaled neural-network inputs."""

    out = simulate_sdm_errors(
        c=3.0,
        kappa=4.0,
        n_trials=40,
        error_scale=180,
        rng=np.random.default_rng(2026),
    )

    assert out.shape == (40, 1)
    assert np.all(out >= -1)
    assert np.all(out < 1)


def test_ms_sdm_summary_matches_known_circular_moments():
    """msSDM summaries should store first- and second-order circular moments."""

    out = errors_to_ms_sdm_summary(np.array([0.0, 90.0]))

    assert out.shape == (5,)
    assert out[0] == pytest.approx(0.5)
    assert out[1] == pytest.approx(0.5)
    assert out[2] == pytest.approx(0.0)
    assert out[3] == pytest.approx(0.0, abs=1e-7)
    assert out[4] == 2


def test_ms_sdm_summary_converts_scaled_errors_to_degrees():
    """msSDM summaries should accept unit-scaled errors when scale is provided."""

    out = errors_to_ms_sdm_summary(np.array([0.0, 0.5]), error_scale=180.0)

    assert out[0] == pytest.approx(0.5)
    assert out[1] == pytest.approx(0.5)
    assert out[2] == pytest.approx(0.0)
    assert out[3] == pytest.approx(0.0, abs=1e-7)
    assert out[4] == 2


def test_ms_sdm_summary_rejects_invalid_errors():
    """msSDM summaries should fail clearly for missing or empty trial data."""

    with pytest.raises(ValueError, match="at least one trial"):
        errors_to_ms_sdm_summary(np.array([]))
    with pytest.raises(ValueError, match="missing or infinite"):
        errors_to_ms_sdm_summary(np.array([0.0, np.nan]))


def test_ms_sdm_simulator_returns_reproducible_summary():
    """msSDM simulation should return reproducible circular-moment summaries."""

    first = simulate_ms_sdm_summary(
        c=3.0,
        kappa=4.0,
        n_trials=40,
        rng=np.random.default_rng(2026),
    )
    second = simulate_ms_sdm_summary(
        c=3.0,
        kappa=4.0,
        n_trials=40,
        rng=np.random.default_rng(2026),
    )

    assert np.array_equal(first, second)
    assert first.shape == (5,)
    assert np.all(first[:4] >= -1)
    assert np.all(first[:4] <= 1)
    assert first[4] == 40


def test_ezdm_simulator_returns_reproducible_summary():
    """ezDM summary simulation should return pc, mrt, and vrt."""

    moments = ezdm_moments(v=0.1, a=0.14, t0=0.3, s=0.1)
    first = simulate_ezdm_summary(
        v=0.1,
        a=0.14,
        t0=0.3,
        n_trials=50,
        rng=np.random.default_rng(2026),
    )
    second = simulate_ezdm_summary(
        v=0.1,
        a=0.14,
        t0=0.3,
        n_trials=50,
        rng=np.random.default_rng(2026),
    )

    assert moments["pc"] == pytest.approx(0.802, abs=0.001)
    assert np.array_equal(first, second)
    assert first.shape == (3,)
    assert 0 < first[0] < 1
    assert first[1] > 0
    assert first[2] > 0
