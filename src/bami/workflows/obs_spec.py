"""Observation formatting contracts for workflow inputs.

Observation specs make trial-count encoding explicit for workflows whose data
rows lose reliability information, such as aggregate summaries or proportions.
The default workflow behavior remains available by leaving ``obs_spec=None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


ObsKind = Literal["aggregate_summary", "proportions", "counts"]
NTransform = Literal["log_range", "linear_range"]


@dataclass(frozen=True)
class ObsSpec:
    """Describe how one observation row should encode trial count.

    Parameters
    ----------
    kind
        Named observation preset. Supported values are ``"aggregate_summary"``,
        ``"proportions"``, and ``"counts"``.
    add_n
        Whether encoded rows append a trial-count feature.
    n_range
        Two-value range used to scale trial counts when ``add_n`` is true.
    n_transform
        Named transform used for the appended trial-count feature.

    Returns
    -------
    None
        The initialized spec exposes ``encode`` and ``to_dict``.
    """

    kind: ObsKind
    add_n: bool
    n_range: tuple[int, int] | None = None
    n_transform: NTransform = "log_range"

    @classmethod
    def aggregate_summary(
        cls,
        n_range: tuple[int, int],
        n_transform: NTransform = "log_range",
    ) -> "ObsSpec":
        """Create a spec for fixed-width aggregate summaries.

        Parameters
        ----------
        n_range
            Trial-count range used to scale the appended trial-count feature.
        n_transform
            Named transform for the appended trial-count feature.

        Returns
        -------
        ObsSpec
            Spec that preserves summary features and appends encoded
            ``n_trials``.
        """

        return cls(
            kind="aggregate_summary",
            add_n=True,
            n_range=n_range,
            n_transform=n_transform,
        )

    @classmethod
    def proportions(
        cls,
        n_range: tuple[int, int],
        n_transform: NTransform = "log_range",
    ) -> "ObsSpec":
        """Create a spec for proportion rows that need explicit trial count.

        Parameters
        ----------
        n_range
            Trial-count range used to scale the appended trial-count feature.
        n_transform
            Named transform for the appended trial-count feature.

        Returns
        -------
        ObsSpec
            Spec that preserves proportion features and appends encoded
            ``n_trials``.
        """

        return cls(
            kind="proportions",
            add_n=True,
            n_range=n_range,
            n_transform=n_transform,
        )

    @classmethod
    def counts(
        cls,
        add_n: bool = False,
        n_range: tuple[int, int] | None = None,
        n_transform: NTransform = "log_range",
    ) -> "ObsSpec":
        """Create a spec for count rows.

        Parameters
        ----------
        add_n
            Whether to append encoded ``n_trials`` in addition to count sums.
        n_range
            Trial-count range required when ``add_n`` is true.
        n_transform
            Named transform for the appended trial-count feature.

        Returns
        -------
        ObsSpec
            Spec that preserves count rows and optionally appends encoded
            ``n_trials``.
        """

        return cls(
            kind="counts",
            add_n=add_n,
            n_range=n_range,
            n_transform=n_transform,
        )

    def __post_init__(self) -> None:
        """Validate spec fields after dataclass initialization.

        Returns
        -------
        None
            Raises ``ValueError`` when the spec is internally inconsistent.
        """

        if self.kind not in {"aggregate_summary", "proportions", "counts"}:
            raise ValueError(f"Unsupported observation kind: {self.kind!r}.")
        if self.n_transform not in {"log_range", "linear_range"}:
            raise ValueError(f"Unsupported n_transform: {self.n_transform!r}.")
        if self.add_n:
            self._check_n_range(self.n_range)
        if not self.add_n and self.n_range is not None:
            self._check_n_range(self.n_range)

    def output_width(self, data_width: int) -> int:
        """Return encoded row width for a base simulator row width.

        Parameters
        ----------
        data_width
            Number of features returned by the simulator.

        Returns
        -------
        int
            Encoded observation width.
        """

        width = int(data_width)
        if self.add_n:
            width += 1
        return width

    def encode(self, row, n_trials: int) -> np.ndarray:
        """Encode one observation row.

        Parameters
        ----------
        row
            Base simulator row before observation-level trial-count encoding.
        n_trials
            Number of trials represented by ``row``.

        Returns
        -------
        numpy.ndarray
            Encoded float row.
        """

        row_arr = np.asarray(row, dtype=np.float32).reshape(-1)
        if not self.add_n:
            return row_arr
        n_value = np.array([self.transform_n(n_trials)], dtype=np.float32)
        return np.concatenate([row_arr, n_value])

    def transform_n(self, n_trials: int) -> float:
        """Scale a trial count with the configured named transform.

        Parameters
        ----------
        n_trials
            Trial count to encode.

        Returns
        -------
        float
            Scaled trial-count feature.
        """

        if not self.add_n:
            raise ValueError("This obs spec does not encode n_trials.")
        low, high = self._check_n_range(self.n_range)
        n_value = float(n_trials)
        if n_value < low or n_value > high:
            raise ValueError(
                f"n_trials={n_trials} is outside the observation range "
                f"({low}, {high})."
            )
        if self.n_transform == "linear_range":
            return self._scale_to_unit_interval(n_value, low, high)

        log_low = np.log(low)
        log_high = np.log(high)
        log_value = np.log(n_value)
        return self._scale_to_unit_interval(log_value, log_low, log_high)

    def to_dict(self) -> dict:
        """Return JSON-safe observation metadata.

        Returns
        -------
        dict
            Metadata containing only strings, booleans, and numbers.
        """

        return {
            "kind": self.kind,
            "add_n": self.add_n,
            "n_range": None if self.n_range is None else list(self.n_range),
            "n_transform": self.n_transform,
        }

    @staticmethod
    def _scale_to_unit_interval(value: float, low: float, high: float) -> float:
        """Scale a value from ``[low, high]`` to ``[-1, 1]``.

        Parameters
        ----------
        value
            Value to scale.
        low, high
            Inclusive lower and upper bounds.

        Returns
        -------
        float
            Scaled value.
        """

        return float(2.0 * (value - low) / (high - low) - 1.0)

    @staticmethod
    def _check_n_range(n_range) -> tuple[int, int]:
        """Validate a two-value positive trial-count range.

        Parameters
        ----------
        n_range
            Candidate range.

        Returns
        -------
        tuple[int, int]
            Validated range.
        """

        if n_range is None:
            raise ValueError("n_range is required when add_n=True.")
        if len(n_range) != 2:
            raise ValueError("n_range must contain exactly two values.")
        low = int(n_range[0])
        high = int(n_range[1])
        if low < 1 or high <= low:
            raise ValueError("n_range must satisfy 1 <= low < high.")
        return low, high
