"""Tests for preset model simulator functions."""

import numpy as np
import pytest

from bami.simulators import (
    prop_m3,
    simulate_ezdm_simple,
    simulate_m3_custom,
    simulate_sdm_simple,
)


def _custom_activation(alpha, beta, mix):
    """Return a compact custom activation vector for simulator tests.

    Parameters
    ----------
    alpha, beta, mix
        Public-scale model parameters passed by ``simulate_m3_custom``.

    Returns
    -------
    list[float]
        Three activation scores used to test non-default category widths.
    """

    return [alpha + beta, mix * alpha, beta]


def _bad_2d_activation(alpha, beta, mix):
    """Return a two-dimensional activation array for validation tests.

    Parameters
    ----------
    alpha, beta, mix
        Accepted for compatibility with ``simulate_m3_custom``.

    Returns
    -------
    list[list[float]]
        Invalid nested activation values.
    """

    return [[1.0, 2.0]]


def _bad_nan_activation(alpha, beta, mix):
    """Return a non-finite activation vector for validation tests.

    Parameters
    ----------
    alpha, beta, mix
        Accepted for compatibility with ``simulate_m3_custom``.

    Returns
    -------
    list[float]
        Invalid activation values with one NaN.
    """

    return [1.0, np.nan, 0.2]


def test_m3_custom_simulator_returns_reproducible_counts():
    """Custom M3 count simulation should be reproducible with a global seed."""

    np.random.seed(2026)
    first = simulate_m3_custom(
        alpha=0.8,
        beta=0.6,
        mix=0.4,
        n_trials=50,
        activation_fn=_custom_activation,
        n_options=[1, 2, 1],
    )
    np.random.seed(2026)
    second = simulate_m3_custom(
        alpha=0.8,
        beta=0.6,
        mix=0.4,
        n_trials=50,
        activation_fn=_custom_activation,
        n_options=[1, 2, 1],
    )

    assert np.array_equal(first, second)
    assert first.shape == (3,)
    assert np.all(first >= 0)
    assert np.allclose(first, np.round(first))
    assert first.sum() == 50


def test_m3_custom_simulator_uses_custom_activation_width():
    """Custom M3 simulation should follow the activation function width."""

    np.random.seed(2026)
    out = simulate_m3_custom(
        alpha=0.8,
        beta=0.6,
        mix=0.4,
        n_trials=40,
        activation_fn=_custom_activation,
        n_options=[1, 2, 1],
    )

    assert out.shape == (3,)
    assert np.all(out >= 0)
    assert out.sum() == 40


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"activation_fn": "not callable"}, "activation_fn must be callable"),
        ({"activation_fn": _bad_2d_activation}, "one-dimensional"),
        (
            {"activation_fn": _custom_activation, "n_options": None},
            "n_options is required",
        ),
        (
            {"activation_fn": _custom_activation, "n_options": [1, 2]},
            "n_options must be an integer or a vector matching activations",
        ),
        ({"activation_fn": _custom_activation, "rule": "bad"}, "rule must be"),
        ({"activation_fn": _bad_nan_activation}, "activations must be finite"),
        ({"activation_fn": _custom_activation, "n_trials": 0}, "n_trials"),
    ],
)
def test_m3_custom_simulator_rejects_invalid_inputs(kwargs, message):
    """Custom M3 simulation should fail clearly for invalid settings."""

    params = {
        "alpha": 0.8,
        "beta": 0.6,
        "mix": 0.4,
        "n_trials": 40,
        "n_options": [1, 2, 1],
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=message):
        simulate_m3_custom(**params)


def test_m3_custom_luce_rejects_negative_strengths():
    """Luce choice should not silently accept negative custom strengths."""

    def negative_activation(alpha, beta, mix):
        """Return one negative strength for Luce-rule validation.

        Parameters
        ----------
        alpha, beta, mix
            Accepted for compatibility with ``simulate_m3_custom``.

        Returns
        -------
        list[float]
            Strengths containing one invalid negative value.
        """

        return [1.0, -0.1, 0.2]

    with pytest.raises(ValueError, match="nonnegative"):
        simulate_m3_custom(
            alpha=0.8,
            beta=0.6,
            mix=0.4,
            n_trials=40,
            activation_fn=negative_activation,
            n_options=[1, 2, 1],
            rule="luce",
        )


def test_prop_m3_returns_proportions():
    """M3 count normalization should convert counts to response proportions."""

    out = prop_m3(np.array([2, 1, 1, 0, 1]), n_trials=5)

    assert np.allclose(out, np.array([0.4, 0.2, 0.2, 0.0, 0.2]))


def test_prop_m3_rejects_invalid_trial_count():
    """M3 count normalization should reject impossible trial totals."""

    with pytest.raises(ValueError, match="n_trials"):
        prop_m3(np.array([2, 1, 1, 0, 1]), n_trials=0)


def test_sdm_error_simulator_returns_trial_level_errors():
    """SDM error simulation should return reproducible continuous trial rows."""

    np.random.seed(2026)
    first = simulate_sdm_simple(
        c=3.0,
        kappa=4.0,
        n_trials=40,
    )
    np.random.seed(2026)
    second = simulate_sdm_simple(
        c=3.0,
        kappa=4.0,
        n_trials=40,
    )

    assert np.array_equal(first, second)
    assert first.shape == (40, 1)
    assert np.all(first >= -np.pi)
    assert np.all(first <= np.pi)
    assert not np.allclose(first, np.round(first))


def test_sdm_error_simulator_rejects_invalid_parameters():
    """SDM error simulation should reject impossible public parameters."""

    with pytest.raises(ValueError, match="c"):
        simulate_sdm_simple(c=0.0, kappa=4.0, n_trials=10)
    with pytest.raises(ValueError, match="kappa"):
        simulate_sdm_simple(c=3.0, kappa=0.0, n_trials=10)


def test_ezdm_simulator_returns_reproducible_summary():
    """ezDM summary simulation should return pc, mrt, and vrt."""

    np.random.seed(2026)
    first = simulate_ezdm_simple(
        v=0.1,
        a=0.14,
        t0=0.3,
        n_trials=50,
    )
    np.random.seed(2026)
    second = simulate_ezdm_simple(
        v=0.1,
        a=0.14,
        t0=0.3,
        n_trials=50,
    )

    assert np.array_equal(first, second)
    assert first.shape == (3,)
    assert 0 < first[0] < 1
    assert first[1] == pytest.approx(0.723, abs=0.001)
    assert first[2] == pytest.approx(0.112, abs=0.001)
