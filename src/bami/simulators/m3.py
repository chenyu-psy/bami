"""Preset M3 simulation helpers.

The functions in this module are small BayesFlow-free generators. They take
public M3 parameter values and return simulated response counts that can later
be wrapped by fixed, flexible, or hierarchical workflows.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

DEFAULT_M3_OPTIONS = (1, 3, 1, 3, 4)


def m3_activation(a: float, c: float, ra: float, rc: float, b: float = 0) -> np.ndarray:
    """Return M3 response-category activation scores.

    Parameters
    ----------
    a, c, ra, rc, b
        Public-scale M3 parameters. The returned order is ``correct``,
        ``other``, ``dist``, ``other_dist``, and ``new``.

    Returns
    -------
    numpy.ndarray
        Five activation scores in the fixed M3 response-category order.
    """

    correct = a + c + b
    other = a + b
    dist = ra * a + rc * c + b
    other_dist = ra * a + b
    new = b
    return np.array([correct, other, dist, other_dist, new], dtype=float)


def choice_probs(
    activations: Sequence[float],
    n_options: int | Sequence[int] | None = None,
    rule: str = "softmax",
) -> np.ndarray:
    """Convert response activations to category probabilities.

    Parameters
    ----------
    activations
        Response-category activation scores.
    n_options
        Number of response options represented by each category. ``None`` uses
        one option per category; a single integer is broadcast to all
        categories.
    rule
        Choice rule. ``"softmax"`` uses the same temperature-2 softmax as the
        current M3 workflow. ``"luce"`` treats activations as nonnegative Luce
        strengths and normalizes them after option weighting.

    Returns
    -------
    numpy.ndarray
        Probability vector that sums to one.
    """

    logits = np.asarray(activations, dtype=float)
    if logits.ndim != 1:
        raise ValueError("activations must be a one-dimensional vector.")

    option_counts = _resolve_n_options(n_options, logits.size)
    weighted = _weighted_strengths(logits, option_counts, rule)
    total = float(np.sum(weighted))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("choice probabilities require positive total strength.")
    return weighted / total


def simulate_m3_counts(
    a: float,
    c: float,
    ra: float,
    rc: float,
    b: float = 0,
    n_trials: int = 100,
    n_options: int | Sequence[int] | None = DEFAULT_M3_OPTIONS,
    rule: str = "softmax",
    rng=None,
) -> np.ndarray:
    """Simulate one M3 response-count vector.

    Parameters
    ----------
    a, c, ra, rc, b
        Public-scale M3 parameters.
    n_trials
        Number of responses to draw.
    n_options
        Number of response options represented by each M3 category.
    rule
        Choice rule passed to ``choice_probs``.
    rng
        Optional NumPy random generator. Defaults to ``np.random`` so existing
        project-level seeding remains effective.

    Returns
    -------
    numpy.ndarray
        Five integer response counts in M3 category order.
    """

    n_trials = _check_n_trials(n_trials)
    rng = np.random if rng is None else rng
    probs = choice_probs(
        m3_activation(a=a, c=c, ra=ra, rc=rc, b=b),
        n_options=n_options,
        rule=rule,
    )
    return rng.multinomial(n_trials, probs).astype(np.int64)


def normalize_m3_count_row(row, n_trials: int, model=None) -> np.ndarray:
    """Scale one M3 count row for normalized hierarchy inputs.

    Parameters
    ----------
    row
        Five response-category counts in M3 order.
    n_trials
        Number of responses represented by ``row``.
    model
        Workflow object with ``n_trials_range`` when available. The upper end
        of that range is used to put trial counts on a stable scale.

    Returns
    -------
    numpy.ndarray
        Five response proportions. The workflow appends the scaled trial count
        separately when ``include_trial_feature`` is enabled.
    """

    row_arr = np.asarray(row, dtype=np.float32)
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")
    return row_arr / float(n_trials)


def _resolve_n_options(
    n_options: int | Sequence[int] | None,
    n_categories: int,
) -> np.ndarray:
    """Return one positive option count per response category.

    Parameters
    ----------
    n_options
        User-provided option count setting.
    n_categories
        Number of response categories implied by the activation vector.

    Returns
    -------
    numpy.ndarray
        Positive option counts with length ``n_categories``.
    """

    if n_options is None:
        option_counts = np.ones(n_categories, dtype=float)
    elif isinstance(n_options, int):
        option_counts = np.full(n_categories, n_options, dtype=float)
    else:
        option_counts = np.asarray(n_options, dtype=float)
        if option_counts.shape != (n_categories,):
            raise ValueError(
                "n_options must be None, an integer, or a vector matching activations."
            )

    if np.any(~np.isfinite(option_counts)) or np.any(option_counts <= 0):
        raise ValueError("n_options must contain positive finite values.")
    return option_counts


def _weighted_strengths(
    activations: np.ndarray,
    n_options: np.ndarray,
    rule: str,
) -> np.ndarray:
    """Return option-weighted strengths for one choice rule.

    Parameters
    ----------
    activations
        One-dimensional activation vector.
    n_options
        Positive option counts with the same length as ``activations``.
    rule
        Choice rule name.

    Returns
    -------
    numpy.ndarray
        Positive or nonnegative weighted strengths.
    """

    if np.any(~np.isfinite(activations)):
        raise ValueError("activations must be finite.")

    if rule == "softmax":
        centered = activations - np.max(activations)
        temperature = 2.0
        return np.exp(centered / temperature) * n_options

    if rule in {"luce", "simple"}:
        if np.any(activations < 0):
            raise ValueError(f"{rule} rule requires nonnegative activations.")
        return activations * n_options

    raise ValueError("rule must be 'softmax' or 'luce'.")


def _check_n_trials(n_trials: int) -> int:
    """Validate the number of simulated responses.

    Parameters
    ----------
    n_trials
        Candidate response count.

    Returns
    -------
    int
        Positive integer response count.
    """

    checked = int(n_trials)
    if checked < 1:
        raise ValueError("n_trials must be at least 1.")
    return checked
