"""Custom M3 count simulator helpers.

The public simulator accepts a user-provided activation function, converts
activations to choice probabilities, and samples response counts. This keeps
the model-specific activation logic visible in user code while reusing the
shared count-sampling mechanics.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def simulate_m3_custom(
    n_trials: int,
    activation_fn,
    n_options: int | Sequence[int],
    rule: str = "softmax",
    **parms,
) -> np.ndarray:
    """Simulate response counts from a user-defined M3 activation function.

    Args:
        n_trials: Number of responses to draw.
        activation_fn (Callable): Function called as ``activation_fn(**parms)``. It must
            return a one-dimensional vector of response-category activations.
        n_options: Number of response options represented by each activation
            category.
        rule: Choice rule used to convert activations to probabilities.
        **parms (Any): Public-scale model parameters passed to ``activation_fn``.

    Returns:
        numpy.ndarray: Integer response counts with one count per activation
            category.
    """

    n_trials = _check_n_trials(n_trials)
    if activation_fn is None:
        raise ValueError("activation_fn is required for simulate_m3_custom.")
    if not callable(activation_fn):
        raise ValueError("activation_fn must be callable.")

    activations = activation_fn(**parms)
    probs = _choice_probs(
        activations,
        n_options=n_options,
        rule=rule,
    )
    return np.random.multinomial(n_trials, probs).astype(np.int64)


def prop_m3(row, n_trials: int, model=None) -> np.ndarray:
    """Convert one M3 count row to response proportions.

    Workflow row transforms may receive the model object, but this
    normalization only needs ``row`` and ``n_trials``.

    Args:
        row: Five response-category counts in M3 order.
        n_trials: Number of responses represented by ``row``.
        model: Unused workflow-compatible argument.

    Returns:
        numpy.ndarray: Response proportions with the same width as ``row``.
    """

    row_arr = np.asarray(row, dtype=np.float32)
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")
    return row_arr / float(n_trials)


def _choice_probs(
    activations: Sequence[float],
    n_options: int | Sequence[int] | None = None,
    rule: str = "softmax",
) -> np.ndarray:
    """Convert response activations to category probabilities.

    Args:
        activations:
            Response-category activation scores.
        n_options:
            Number of response options represented by each category. A single
            integer is broadcast to all categories.
        rule:
            Choice rule. ``"softmax"`` uses temperature-2 softmax. ``"luce"`` and
            ``"simple"`` treat activations as nonnegative strengths.

    Returns:
        numpy.ndarray: Probability vector that sums to one.
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


def _resolve_n_options(
    n_options: int | Sequence[int] | None,
    n_categories: int,
) -> np.ndarray:
    """Return one positive option count per response category.

    Args:
        n_options:
            User-provided option count setting.
        n_categories:
            Number of response categories implied by the activation vector.

    Returns:
        numpy.ndarray: Positive option counts with length ``n_categories``.
    """

    if n_options is None:
        raise ValueError("n_options is required.")
    if isinstance(n_options, int):
        option_counts = np.full(n_categories, n_options, dtype=float)
    else:
        option_counts = np.asarray(n_options, dtype=float)
        if option_counts.shape != (n_categories,):
            raise ValueError(
                "n_options must be an integer or a vector matching activations."
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

    Args:
        activations:
            One-dimensional activation vector.
        n_options:
            Positive option counts with the same length as ``activations``.
        rule:
            Choice rule name.

    Returns:
        numpy.ndarray: Positive or nonnegative weighted strengths.
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

    raise ValueError("rule must be 'softmax', 'luce', or 'simple'.")


def _check_n_trials(n_trials: int) -> int:
    """Validate the number of simulated responses.

    Args:
        n_trials:
            Candidate response count.

    Returns:
        int: Positive integer response count.
    """

    checked = int(n_trials)
    if checked < 1:
        raise ValueError("n_trials must be at least 1.")
    return checked
