"""Tests for the built-in summary-level ezDM simulator."""

import numpy as np
import pytest

from bami.simulators.ezdm import simulate_ezdm_simple


def test_ezdm_simulator_matches_paper_example_moments():
    """The simulator should reproduce the EZ appendix moments before pc sampling."""

    np.random.seed(2026)
    summary = simulate_ezdm_simple(
        v=0.1,
        a=0.14,
        t0=0.3,
        n_trials=100_000,
        s=0.1,
    )

    assert summary.shape == (3,)
    assert summary[0] == pytest.approx(0.802, abs=0.003)
    assert summary[1] == pytest.approx(0.723, abs=0.001)
    assert summary[2] == pytest.approx(0.112, abs=0.001)


def test_ezdm_simulator_is_reproducible_with_global_seed():
    """Repeated simulations with the same global seed should return the same row."""

    np.random.seed(2026)
    first = simulate_ezdm_simple(
        v=0.1,
        a=0.14,
        t0=0.3,
        n_trials=50,
        s=0.1,
    )
    np.random.seed(2026)
    second = simulate_ezdm_simple(
        v=0.1,
        a=0.14,
        t0=0.3,
        n_trials=50,
        s=0.1,
    )

    assert np.array_equal(first, second)
    assert 0 < first[0] < 1
    assert first[1] > 0
    assert first[2] > 0


def test_ezdm_simulator_stays_finite_for_extreme_finite_parameters():
    """Extreme finite parameters should not create NaN summary values."""

    np.random.seed(2026)
    summary = simulate_ezdm_simple(
        v=10,
        a=10,
        t0=0.2,
        n_trials=100,
    )

    assert np.all(np.isfinite(summary))
    assert 0 < summary[0] < 1
    assert summary[1] > 0
    assert summary[2] > 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"v": 0.0}, "v must not equal 0"),
        ({"a": 0.0}, "a must be positive"),
        ({"t0": 0.0}, "t0 must be positive"),
        ({"n_trials": 0}, "n_trials must be at least 1"),
        ({"s": 0.0}, "s must be positive"),
        ({"v": np.inf}, "v must be finite"),
    ],
)
def test_ezdm_simulator_rejects_invalid_inputs(kwargs, message):
    """Invalid simulator inputs should fail with readable error messages."""

    params = {
        "v": 0.1,
        "a": 0.14,
        "t0": 0.3,
        "n_trials": 50,
        "s": 0.1,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=message):
        simulate_ezdm_simple(**params)
