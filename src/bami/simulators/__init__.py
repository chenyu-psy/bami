"""Preset model simulators for reusable workflow construction.

The functions exported here are convenience generators. Future workflows can
also accept user-supplied simulator functions with the same simple
parameter-to-data role.
"""

from .ezdm import (
    ezdm_moments,
    moments_to_params,
    params_to_moments,
    simulate_ezdm_summary,
    simulate_hierarchical,
    simulate_simple,
)
from .m3 import DEFAULT_M3_OPTIONS, choice_probs, m3_activation, simulate_m3_counts
from .circular import (
    GRID_SIZE,
    circular_moments_from_errors,
    degree_grid,
    errors_to_indices,
    indices_to_errors,
)
from .ms_sdm import errors_to_ms_sdm_summary, simulate_ms_sdm_summary
from .sdm import (
    sdm_probs,
    simulate_sdm_errors,
)

__all__ = [
    "DEFAULT_M3_OPTIONS",
    "GRID_SIZE",
    "choice_probs",
    "circular_moments_from_errors",
    "degree_grid",
    "errors_to_indices",
    "errors_to_ms_sdm_summary",
    "ezdm_moments",
    "indices_to_errors",
    "m3_activation",
    "moments_to_params",
    "params_to_moments",
    "sdm_probs",
    "simulate_ezdm_summary",
    "simulate_hierarchical",
    "simulate_m3_counts",
    "simulate_ms_sdm_summary",
    "simulate_simple",
    "simulate_sdm_errors",
]
