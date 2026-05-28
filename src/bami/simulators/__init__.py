"""Preset model simulators for reusable workflow construction.

The functions exported here are convenience generators. Future workflows can
also accept user-supplied simulator functions with the same simple
parameter-to-data role.
"""

from .ezdm import simulate_ezdm_simple
from .m3 import prop_m3, simulate_m3_custom
from .circular import (
    GRID_SIZE,
    circular_moments_from_errors,
    degree_grid,
    errors_to_indices,
    indices_to_errors,
)
from .sdm import (
    sdm_probs,
    simulate_sdm_simple,
)

__all__ = [
    "GRID_SIZE",
    "circular_moments_from_errors",
    "degree_grid",
    "errors_to_indices",
    "indices_to_errors",
    "prop_m3",
    "sdm_probs",
    "simulate_ezdm_simple",
    "simulate_m3_custom",
    "simulate_sdm_simple",
]
